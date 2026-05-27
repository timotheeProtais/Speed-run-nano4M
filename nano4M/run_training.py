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
import argparse
import datetime
import time
import json
import math
import os
import sys
import omegaconf
import yaml
from hydra.utils import instantiate
from pathlib import Path
from typing import Iterable, Sequence, List, Dict, Optional

import numpy as np
import torch

import nanofm.utils as utils
from nanofm.utils import NativeScalerWithGradNormCount as NativeScaler
from nanofm.utils.optim_factory import create_adamw_optimizer
from nanofm.utils.muon import Muon
from nanofm.utils.scheduler import cosine_scheduler
from nanofm.utils.checkpoint import unwrap_model
from nanofm.utils.native_scaler import get_grad_norm_


def get_args():
    """Parses training arguments from the command line and an optional YAML config file.

    If a config file is provided via -c, it is loaded with OmegaConf (supporting interpolations)
    and used as default values, which can still be overridden by explicit command-line arguments.
    """
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('Pre-training script', add_help=True)
    parser.add_argument('--run_name', type=str, default='auto')


    # Model and data configs, instantiated with hydra
    parser.add_argument("--model_config", default={}, type=Dict,
                        help="Model config (default: %(default)s)")
    parser.add_argument("--train_loader_config", default={}, type=Dict,
                        help="Training dataloader config (default: %(default)s)")
    parser.add_argument("--eval_loader_config", default={}, type=Dict,
                        help="Validation dataloader config (default: %(default)s)")


    # Training parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (default: %(default)s). '
                             'Effective batch size is batch_size * # gpus')
    parser.add_argument('--total_tokens', type=int,
                        help='Number of total training tokens (in millions).')
    parser.add_argument('--warmup_tokens', type=int, default=0,
                        help='Total tokens (in millions) to warmup LR linearly (default: %(default)s)')
    parser.add_argument('--num_tokens_per_sample', type=int,
                        help='Defines how we count the "tokens seen" for a model, here per single sample.')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Data type (default: %(default)s')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Optimization parameters
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=3e-4, 
                        help='Peak learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Min learning rate at the end of training (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: %(default)s)')

    # Eval, checkpointing and resuming
    parser.add_argument('--eval_freq', default=100, type=int, 
                        help="Frequency of evaluation in millions of tokens (default: %(default)s)")
    parser.add_argument('--save_ckpt_freq', default=100, type=int,
                        help='Checkpoint saving frequency in millions of tokens (default: %(default)s)')
    parser.add_argument('--output_dir', default='./outputs/auto',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_iteration', default=0, type=int, help='Start iteration')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)    

    # Distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=False)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

     # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default='auto', type=str,
                        help='Run name on wandb')

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            # Use OmegaConf to parse interpolations
            omegaconf.OmegaConf.register_new_resolver("eval", eval)
            cfg = omegaconf.OmegaConf.create(cfg)
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file is specified.
    args = parser.parse_args(remaining)

    # Add the config path as a final args if given
    args.config_path = args_config.config

    return args


def main(args):
    """Sets up the distributed training environment, model, and dual optimizer, then launches the training loop.

    Parameters are split into two groups with different optimizers. Hidden weight matrices (ndim >= 2,
    excluding embeddings and norms) are optimized with Muon, which applies Newton-Schulz orthogonalization
    to each gradient update. All other parameters (embeddings, norms, biases, output head) use standard
    AdamW. Both optimizers share the same cosine LR schedule, with Muon's LR scaled proportionally
    (lr_muon = 0.02 * lr / peak_lr) to keep the two update magnitudes consistent throughout training.
    """
    # Distributed init
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    args.world_size = utils.get_world_size()
    global_rank = utils.get_rank()
    
    # Seeding
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # CUDNN and data type setup
    torch.backends.cudnn.benchmark = True
    if args.dtype in ['float16', 'fp16']:
        dtype = torch.float16
    elif args.dtype in ['bfloat16', 'bf16']:
        dtype = torch.bfloat16
    elif args.dtype in ['float32', 'fp32']:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    
    # Logger setup
    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
        print(f"Logging to wandb project {args.wandb_project}, entity {args.wandb_entity}, run name {args.wandb_run_name}")
    else:
        log_writer = None
        print("Not logging to wandb")

    # Log args
    print(args)

    # Train and val loader setup
    data_loader_train = instantiate(args.train_loader_config)
    data_loader_eval = instantiate(args.eval_loader_config)   
    
    # Model setup
    model = instantiate(args.model_config).to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module
    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    # Training phases
    args.total_batch_size = args.batch_size * args.world_size
    num_tokens_per_iter = args.num_tokens_per_sample * args.total_batch_size
    args.total_iters = math.ceil(args.total_tokens * 1e6 / num_tokens_per_iter)
    args.warmup_iters = math.ceil(args.warmup_tokens * 1e6 / num_tokens_per_iter)
    args.eval_freq_iters = math.ceil(args.eval_freq * 1e6 / num_tokens_per_iter)
    args.save_ckpt_freq_iters = math.ceil(args.save_ckpt_freq * 1e6 / num_tokens_per_iter)
    print(f"Total tokens: {args.total_tokens}M")
    print(f"Total iters: {args.total_iters}")
    print(f"Warmup tokens: {args.warmup_tokens}M")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Eval freq: every {args.eval_freq_iters} iterations")
    print(f"Save ckpt freq: every {args.save_ckpt_freq_iters} iterations")
    print("Batch size per GPU = %d" % args.batch_size)
    print("Total (effective) batch size = %d" % args.total_batch_size)

    # Optimizer: Muon for hidden weight matrices, AdamW for everything else.
    # Embeddings and norms are excluded from Muon as they are not 2D weight matrices in the
    # traditional sense and do not benefit from orthogonalization.
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    muon_params, adamw_params = [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and 'embed' not in name and 'norm' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    lr_muon = 0.025
    optimizer_muon = Muon(muon_params, lr=lr_muon, momentum=0.95, weight_decay=0.02)
    optimizer_adamw = torch.optim.AdamW(adamw_params, lr=args.lr, betas=tuple(args.opt_betas), weight_decay=args.weight_decay, eps=args.opt_eps)
    loss_scaler = NativeScaler(enabled=dtype == torch.float16)

    # LR scheduler
    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.total_iters, args.warmup_iters)

    # Auto-load from checkpoint
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer_adamw, loss_scaler=loss_scaler, optimizer_muon=optimizer_muon)
    if log_writer is not None:
        log_writer.set_step(args.start_iteration)

    # Training loop, with evaluations at regular intervals
    print(f"Start training for {args.total_tokens}M tokens = {args.total_iters} iterations")
    start_time = time.time()
    
    train_stats = train_loop(
        args,
        model=model,
        data_loader_train=data_loader_train,
        data_loader_eval=data_loader_eval,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        loss_scaler=loss_scaler,
        lr_schedule_values=lr_schedule_values,
        log_writer=log_writer,
        device=device,
        dtype=dtype,
        n_parameters=n_parameters,
    )
    if args.output_dir:
        utils.save_model(
            args=args, iteration=args.total_iters, model=model, model_without_ddp=model_without_ddp, 
            optimizer=optimizer_adamw, optimizer_muon=optimizer_muon, loss_scaler=loss_scaler, ckpt_name='final', 
            save_as_safetensors=True, model_args=args.model_config,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def to_device(data_dict, device, non_blocking=True):
    """Moves all tensor values in a dict to the target device, leaving non-tensors unchanged."""
    return {k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v for k, v in data_dict.items()}


def train_loop(
        args,
        model: torch.nn.Module,
        data_loader_train: Iterable,
        data_loader_eval: Iterable,
        optimizer_muon, optimizer_adamw: torch.optim.Optimizer,
        loss_scaler: NativeScaler,
        lr_schedule_values: Sequence[float],
        log_writer: Optional[utils.WandbLogger] = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float16,
        n_parameters: int = 0,
    ):
    """Runs the main training loop over the full token budget.

    At each step, the LR schedule is applied to both optimizers: AdamW receives the cosine-scheduled
    lr directly, while Muon's lr is scaled as 0.02 * (lr / peak_lr) to keep the ratio between the
    two optimizers constant throughout training. Both optimizers are stepped separately after a shared
    backward pass and gradient unscaling, with gradient clipping applied before either step.
    Evaluation and checkpointing are triggered at the frequencies specified in args.
    """
    model.train()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Training'
    print_freq = 10

    for step, data_dict in enumerate(metric_logger.log_every(data_loader_train, print_freq, iter_len=args.total_iters, header=header, start_iter=args.start_iteration)):
        it = args.start_iteration + step  # global training iteration
        total_tokens_seen = it * args.total_batch_size * args.num_tokens_per_sample
        _step_start = time.time()
        # Move tensors to GPU
        data_dict = to_device(data_dict, device)
        
        # Forward pass and loss computation
        with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            loss, metrics = model(data_dict)

        loss_value = loss.item()
        metrics_values = {f'{metric}': l.item() for metric, l in metrics.items()}

        # Stop training if loss is not finite, and save debug info
        if not math.isfinite(loss_value):
            torch.save(data_dict, os.path.join(args.output_dir, "debug.pt"))
            print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
            print(f"Saved debug info to {args.output_dir}/debug.pt", file=sys.stderr)
            sys.exit(1)

        # Apply cosine LR schedule to both optimizers.
        # Muon's LR is kept proportional to AdamW's to maintain a consistent ratio throughout training.
        lr = lr_schedule_values[it]
        for param_group in optimizer_adamw.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer_muon.param_groups:
            param_group["lr"] = 0.02 * (lr / args.lr)

        loss_scaler._scaler.scale(loss).backward()
        loss_scaler._scaler.unscale_(optimizer_muon)
        loss_scaler._scaler.unscale_(optimizer_adamw)
        if args.clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        else:
            grad_norm = utils.get_grad_norm_(model.parameters())
        loss_scaler._scaler.step(optimizer_muon)
        loss_scaler._scaler.step(optimizer_adamw)
        loss_scaler._scaler.update()
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        torch.cuda.synchronize()
        
        tokens_per_sec = args.total_batch_size * args.num_tokens_per_sample / (time.time() - _step_start)
        mfu = tokens_per_sec * 2 * n_parameters / (2 * 90e12)
        metric_logger.update(tokens_per_sec=tokens_per_sec)
        metric_logger.update(mfu=mfu)

        tokens_per_sec = args.total_batch_size * args.num_tokens_per_sample / (time.time() - _step_start)
        
        mfu = tokens_per_sec * 2 * n_parameters / (2 * 90e12) 
        metric_logger.update(tokens_per_sec=tokens_per_sec)
        metric_logger.update(mfu=mfu)

        # Logging
        metric_logger.update(loss=loss_value)
        metric_logger.update(**metrics_values)
        if dtype == torch.float16:
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value) 
        metric_logger.update(lr=lr)
        metric_logger.update(grad_norm=grad_norm)

        # Launch evaluation at regular intervals and last iteration
        if (
            data_loader_eval is not None
            and args.eval_freq_iters > 0
            and (it != 0)
            and (it % args.eval_freq_iters == 0 or it + 1 == args.total_iters)
        ):
            eval_stats = evaluate(model, data_loader_eval, device, dtype=dtype, prefix='[Eval]/')
            if log_writer is not None:
                log_writer.update(eval_stats)

        # Log loss and metrics to wandb
        if log_writer is not None:
            log_writer.update({
                'loss': loss_value,
                'lr': lr,
                'grad_norm': grad_norm,
                'total_tokens_seen_m': total_tokens_seen / 1e6,
            })
            log_writer.update(metrics_values)

            log_writer.set_step()

        # Save checkpoint at regular intervals
        if (it + 1) % args.save_ckpt_freq_iters == 0:
            utils.save_model(
                args=args, iteration=it, model=model, model_without_ddp=unwrap_model(model),
                optimizer=optimizer_adamw,  optimizer_muon=optimizer_muon, loss_scaler=loss_scaler, save_as_safetensors=True, model_args=args.model_config,
            )

        # Exit training loop manually in case iterator is infinite
        if it + 1 == args.total_iters:
            print("Training done")
            break

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(
        model: torch.nn.Module, 
        data_loader_eval: Iterable, 
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float16, 
        prefix='[Eval]/',
    ):
    """Runs one full pass over the validation set and returns averaged loss and metrics.

    The model is temporarily set to eval mode and restored to its previous state afterward.
    All metrics are synchronized across distributed processes before being returned.
    """
    model_state = model.training # Save the model state
    model.eval()

    print_freq = 10
    iter_len = len(data_loader_eval) if hasattr(data_loader_eval, '__len__') else -1 # Dealing with iterable datasets

    metric_logger = utils.MetricLogger(delimiter='  ')
    for data_dict in metric_logger.log_every(data_loader_eval, print_freq, iter_len=iter_len, header=prefix):

        # Move tensors to GPU
        data_dict = to_device(data_dict, device)

        with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            loss, metrics = model(data_dict)

            loss_value = loss.item()
            metrics_values = {f'{metric}': l.item() for metric, l in metrics.items()}

        metric_logger.update(loss=loss_value)
        metric_logger.update(**metrics_values)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)
    averaged_stats = {prefix + k: meter.global_avg for k, meter in metric_logger.meters.items()}

    torch.cuda.empty_cache()
    model.train(model_state) # Restore the model state before returning
    
    return averaged_stats


if __name__ == '__main__':
    args = get_args()

    utils.setup_run_name(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)