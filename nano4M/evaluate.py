# evaluate.py
# Computes task-specific metrics to compare two nano4M models across all 4 modalities
#
# Chain 1 : RGB -> depth -> normals -> scene_desc:
#   - Depth:           Standardized L1 Error
#   - Surface Normals: Mean Angle Error
#   - Scene desc:      BLEU score
#
# Chain 2 : scene_desc -> RGB:
#   - RGB:             RGB L1

import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from cosmos_tokenizer.image_lib import ImageTokenizer
from nanofm.utils.checkpoint import load_model_from_safetensors
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset

# Number of val samples to evaluate on (None to use the full set)
NUM_SAMPLES = 50
NUM_STEPS = 8
TEMP = 0.001
TOP_P = 0.0
TOP_K = 0.0
MODALITIES = ['tok_rgb@256', 'tok_depth@256', 'tok_normal@256', 'scene_desc']
COSMOS_ENC = '/home/tprotais/cosmos_tokenizer/encoder.jit'
COSMOS_DEC = '/home/tprotais/cosmos_tokenizer/decoder.jit'
DATASET_ROOT = '/work/com-304/datasets/clevr_com_304/'
CKPT_ORIGINAL = './outputs/nano4M/multiclevr_d6-6w512/checkpoint-final.safetensors'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Path of the checkpoint of the optimized model
CKPT_MODIFIED = './outputs/nano4M/multiclevr_d6-6w512/checkpoint-7629.safetensors'


def token_ids_to_image(token_ids: torch.Tensor, image_tokenizer) -> torch.Tensor:
    """Decodes a flat sequence of discrete token IDs into a pixel image using the Cosmos tokenizer.

    The token sequence is assumed to represent a square spatial grid, so its length must be a
    perfect square. The tokens are reshaped into (1, H, W) before being passed to the decoder,
    and the resulting image is rescaled from [-1, 1] to [0, 1].

    Args:
        token_ids: 1D integer tensor of length H*W containing the discrete visual tokens.
        image_tokenizer: A Cosmos ImageTokenizer instance with a loaded decoder.

    Returns:
        A float tensor of shape (C, H, W) with pixel values in [0, 1].
    """
    side = int(math.sqrt(token_ids.numel()))
    token_ids = token_ids.reshape(1, side, side).to(device)
    with torch.no_grad():
        reconst = image_tokenizer.decode(token_ids)
    return (reconst[0].clamp(-1, 1).float().cpu() + 1) / 2


def compute_depth_l1(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, image_tokenizer) -> float:
    """Computes the standardized L1 error between a predicted and a ground-truth depth map.

    Both token sequences are decoded into images and converted to grayscale by averaging over
    channels. The ground-truth map is then standardized (zero mean, unit std), and the same
    normalization is applied to the prediction before computing the mean absolute difference.
    Standardization makes the metric invariant to the global scale and shift of the depth values,
    focusing on structural accuracy rather than absolute range.

    Args:
        pred_tokens: 1D integer tensor of predicted depth tokens.
        gt_tokens: 1D integer tensor of ground-truth depth tokens.
        image_tokenizer: A Cosmos ImageTokenizer instance with a loaded decoder.

    Returns:
        Scalar float representing the mean standardized L1 depth error (lower is better).
    """
    pred_depth = token_ids_to_image(pred_tokens, image_tokenizer).mean(dim=0)
    gt_depth   = token_ids_to_image(gt_tokens,   image_tokenizer).mean(dim=0)

    std = gt_depth.std()
    if std > 1e-6:
        pred_depth = (pred_depth - pred_depth.mean()) / std
        gt_depth = (gt_depth   - gt_depth.mean())   / std

    return (pred_depth - gt_depth).abs().mean().item()


def compute_mean_angle_error(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, image_tokenizer) -> float:
    """Computes the mean angular error in degrees between predicted and ground-truth surface normals.

    Both token sequences are decoded into images, rescaled to [-1, 1] to recover the XYZ normal
    vectors stored in RGB channels, and L2-normalized per pixel. The per-pixel angle between
    predicted and ground-truth normals is computed via the dot product and arccos, then averaged
    over the image. This metric directly reflects how accurately the model recovers the 3D surface
    orientation.

    Args:
        pred_tokens: 1D integer tensor of predicted normal tokens.
        gt_tokens: 1D integer tensor of ground-truth normal tokens.
        image_tokenizer: A Cosmos ImageTokenizer instance with a loaded decoder.

    Returns:
        Scalar float representing the mean angular error in degrees (lower is better).
    """
    pred_normals = token_ids_to_image(pred_tokens, image_tokenizer) * 2 - 1
    gt_normals = token_ids_to_image(gt_tokens,   image_tokenizer)  * 2 - 1

    pred_normals = F.normalize(pred_normals, dim=0)
    gt_normals = F.normalize(gt_normals,   dim=0)

    cos_sim = (pred_normals * gt_normals).sum(dim=0).clamp(-1, 1)
    angle_error = torch.acos(cos_sim) * (180.0 / torch.pi)

    return angle_error.mean().item()


def compute_rgb_l1(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, image_tokenizer) -> float:
    """Computes the mean L1 pixel error between a predicted and a ground-truth RGB image.

    Both token sequences are decoded into [0, 1] images and compared pixel-wise. This provides
    a straightforward measure of low-level visual fidelity for the text-to-image generation chain.

    Args:
        pred_tokens: 1D integer tensor of predicted RGB tokens.
        gt_tokens: 1D integer tensor of ground-truth RGB tokens.
        image_tokenizer: A Cosmos ImageTokenizer instance with a loaded decoder.

    Returns:
        Scalar float representing the mean L1 pixel error (lower is better).
    """
    pred_img = token_ids_to_image(pred_tokens, image_tokenizer)
    gt_img   = token_ids_to_image(gt_tokens,   image_tokenizer)
    return (pred_img - gt_img).abs().mean().item()


def compute_bleu_score(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, text_tokenizer) -> float:
    """Computes the BLEU-4 score between a predicted and a ground-truth scene description.

    Both token sequences are decoded using the GPT-2 tokenizer (with padding tokens filtered out),
    lowercased, and split into word lists. BLEU-4 with smoothing (method1) is then computed to
    handle the short sentences typical of CLEVR scene descriptions. A score of 0 is returned if
    either the prediction or the reference is empty after decoding.

    Args:
        pred_tokens: 1D integer tensor of predicted text tokens.
        gt_tokens: 1D integer tensor of ground-truth text tokens.
        text_tokenizer: A HuggingFace GPT-2 tokenizer with added special tokens ([PAD], [SOS], [EOS]).

    Returns:
        Scalar float in [0, 1] representing the BLEU score (higher is better).
    """
    pad_id = text_tokenizer.pad_token_id
    pred_ids = pred_tokens[pred_tokens != pad_id].tolist()
    gt_ids = gt_tokens[gt_tokens != pad_id].tolist()

    pred_words = text_tokenizer.decode(pred_ids, skip_special_tokens=True).lower().split()
    gt_words = text_tokenizer.decode(gt_ids,   skip_special_tokens=True).lower().split()

    if len(pred_words) == 0 or len(gt_words) == 0:
        return 0.0

    return float(sentence_bleu([gt_words], pred_words, smoothing_function=SmoothingFunction().method1))


@torch.no_grad()
def evaluate_model(model, dataset, image_tokenizer, text_tokenizer, num_samples: int):
    """Evaluates a nano4M model on two multimodal generation chains and returns aggregated metrics.

    The evaluation runs two sequential generation chains on each validation sample:

    - Chain 1 (RGB → depth → normals → scene_desc): starting from a ground-truth RGB image,
      the model auto-regressively generates depth, surface normals, and finally a text description.
      Each modality is generated using ROAR (Random Order Auto-Regressive) decoding with a fixed
      number of steps. Depth and normals use a near-greedy temperature (0.001) for deterministic
      predictions, while scene description uses sampling (temp=0.7, top_p=0.9).

    - Chain 2 (scene_desc → RGB): starting from the ground-truth text description, the model
      generates the corresponding RGB image using 64 ROAR steps with sampling.

    Metrics are averaged over the first `num_samples` samples of the dataset and returned as a
    dictionary with keys: depth_l1, mean_angle_error, bleu_score, rgb_l1.

    Args:
        model: A trained FourM model with a `generate_one_modality_roar` method.
        dataset: A SimpleMultimodalDataset instance (validation split).
        image_tokenizer: A Cosmos ImageTokenizer instance for decoding visual tokens.
        text_tokenizer: A HuggingFace GPT-2 tokenizer for decoding text tokens.
        num_samples: Number of samples to evaluate (capped at dataset length).

    Returns:
        A dict with keys 'depth_l1', 'mean_angle_error', 'bleu_score', 'rgb_l1',
        each rounded to 3 decimal places.
    """
    model.eval()
    indices = list(range(min(num_samples, len(dataset))))

    depth_l1_scores = []
    angle_error_scores = []
    bleu_scores = []
    rgb_l1_scores = []

    for i in tqdm(indices, desc='Evaluating'):

        gt_depth = dataset[i]['tok_depth@256']
        gt_normal = dataset[i]['tok_normal@256']
        gt_rgb = dataset[i]['tok_rgb@256']
        gt_scene = dataset[i]['scene_desc']
        n = gt_depth.shape[0]

        enc_tokens = gt_rgb.unsqueeze(0).to(device)
        enc_positions = torch.arange(n, device=device).unsqueeze(0)
        enc_modalities = MODALITIES.index('tok_rgb@256') * torch.ones(1, n, device=device, dtype=torch.long)

        # RGB -> depth
        pred_depth, x_tok, x_pos, x_mod = model.generate_one_modality_roar(enc_tokens, enc_positions, enc_modalities,
            target_mod='tok_depth@256', num_steps=NUM_STEPS, temp=TEMP, top_p=TOP_P, top_k=TOP_K)

        # RGB + depth -> normals
        pred_normal, x_tok, x_pos, x_mod = model.generate_one_modality_roar(x_tok, x_pos, x_mod,
            target_mod='tok_normal@256', num_steps=NUM_STEPS, temp=TEMP, top_p=TOP_P, top_k=TOP_K,)

        # RGB + depth + normals -> scene_desc
        pred_scene, _, _, _ = model.generate_one_modality_roar(x_tok, x_pos, x_mod,
            target_mod='scene_desc', num_steps=128, temp=0.7, top_p=0.9, top_k=0.0)

        depth_l1_scores.append(compute_depth_l1(pred_depth[0].cpu(), gt_depth, image_tokenizer))
        angle_error_scores.append(compute_mean_angle_error(pred_normal[0].cpu(), gt_normal, image_tokenizer))
        bleu_scores.append(compute_bleu_score(pred_scene[0].cpu(), gt_scene, text_tokenizer))

        # scene_desc -> RGB
        n_text         = gt_scene.shape[0]
        enc_tokens     = gt_scene.unsqueeze(0).to(device)
        enc_positions  = torch.arange(n_text, device=device).unsqueeze(0)
        enc_modalities = MODALITIES.index('scene_desc') * torch.ones(1, n_text, device=device, dtype=torch.long)

        pred_rgb, _, _, _ = model.generate_one_modality_roar(enc_tokens, enc_positions, enc_modalities,
            target_mod='tok_rgb@256', num_steps=64, temp=0.7, top_p=0.9, top_k=0.0)

        rgb_l1_scores.append(compute_rgb_l1(pred_rgb[0].cpu(), gt_rgb, image_tokenizer))

    return {
        'depth_l1': round(np.mean(depth_l1_scores), 3),
        'mean_angle_error': round(np.mean(angle_error_scores), 3),
        'bleu_score': round(np.mean(bleu_scores), 3),
        'rgb_l1': round(np.mean(rgb_l1_scores), 3),
    }


def main():
    """Entry point for the evaluation script.

    Loads the Cosmos image tokenizer, the GPT-2 text tokenizer, and the validation dataset,
    then evaluates both the original and the modified nano4M checkpoints sequentially. Each model
    is deleted from GPU memory after evaluation to avoid OOM errors when loading the second one.
    Results are printed as a formatted comparison table. Set CKPT_MODIFIED to None to evaluate
    only the original model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    print("Loading Cosmos tokenizer...")
    image_tokenizer = ImageTokenizer(checkpoint_enc=COSMOS_ENC, checkpoint_dec=COSMOS_DEC).to(device)

    print("Loading text tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[SOS]', 'eos_token': '[EOS]'})
    text_tokenizer._tokenizer.post_processor = TemplateProcessing(single="[SOS] $A [EOS]",
        special_tokens=[('[EOS]', text_tokenizer.eos_token_id), ('[SOS]', text_tokenizer.bos_token_id)])

    print("Loading validation dataset...")
    dataset = SimpleMultimodalDataset(root_dir=DATASET_ROOT, split='val', modalities=MODALITIES, 
        sample_from_k_augmentations=1, text_tokenizer_path='gpt2', text_max_length=256, transforms=None)

    print(f"\nLoading original model from {CKPT_ORIGINAL} ...")
    model_original = load_model_from_safetensors(CKPT_ORIGINAL, device=device)
    metrics_original = evaluate_model(model_original, dataset, image_tokenizer, text_tokenizer, args.num_samples)
    del model_original
    torch.cuda.empty_cache()

    if CKPT_MODIFIED == None:
        metrics_modified = None
    else :
        print(f"\nLoading modified model from {CKPT_MODIFIED} ...")
        model_modified = load_model_from_safetensors(CKPT_MODIFIED, device=device)
        metrics_modified = evaluate_model(model_modified, dataset, image_tokenizer, text_tokenizer, args.num_samples)
        del model_modified
        torch.cuda.empty_cache()

    print("\n" + "-" * 53)
    print("Metric                   Original        Modified")
    print("-" * 53)
    metrics_info = [
        ('depth_l1',         'Depth    (low better)'),
        ('mean_angle_error', 'Normals  (low better)'),
        ('bleu_score',       'Text     (high better)'),
        ('rgb_l1',           'RGB      (low better)'),
    ]
    
    for key, label in metrics_info:
        orig = metrics_original[key]
        if metrics_modified:
            mod  = metrics_modified[key]
            print(f"{label}      {orig}         {mod}")
        else:
            print(f"{label}      {orig}")
    print("-" * 53)


if __name__ == '__main__':
    main()