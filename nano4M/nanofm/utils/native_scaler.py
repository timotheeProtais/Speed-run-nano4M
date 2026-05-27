# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on timm and 4M code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/apple/ml-4m/
# --------------------------------------------------------
import torch


class NativeScalerWithGradNormCount:
    """Wrapper around PyTorch's GradScaler that supports multiple optimizers.

    The key modification for Muon is the extra_optimizers argument in __call__: since Muon and
    AdamW manage disjoint parameter sets, both must be unscaled and stepped independently by the
    same scaler. Without this, Muon's parameters would receive scaled gradients, corrupting the
    Newton-Schulz orthogonalization. The scaler itself is shared and updated only once per step.
    """
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.amp.GradScaler('cuda', enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, skip_grad=None, parameters=None, create_graph=False, update_grad=True, compute_grad_norm=True, extra_optimizers=None):
        """Backward pass with optional gradient clipping, unscaling, and stepping of all optimizers.

        extra_optimizers (e.g. [optimizer_muon]) are unscaled and stepped alongside the main
        optimizer. All optimizers in the list must have their gradients unscaled before any step
        is taken, to ensure consistent gradient norms across both parameter groups.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            all_optimizers = [optimizer] + (extra_optimizers or [])
            if clip_grad is not None:
                assert parameters is not None
                for opt in all_optimizers:
                    self._scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            elif skip_grad is not None:
                for opt in all_optimizers:
                    self._scaler.unscale_(opt)
                norm = get_grad_norm_(parameters)
                if norm >= skip_grad:
                    self._scaler.update()
                    return norm
            else:
                for opt in all_optimizers:
                    self._scaler.unscale_(opt)
                norm = get_grad_norm_(parameters) if compute_grad_norm else None
            for opt in all_optimizers:
                self._scaler.step(opt)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Computes the total gradient norm across all parameters with a gradient."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm