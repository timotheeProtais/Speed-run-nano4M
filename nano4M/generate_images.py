# generate_images.py
# Génère des outputs visuels du modèle nano4M final
# Chain 1: RGB -> depth -> normals -> scene_desc
# Chain 2: scene_desc -> RGB

import math
import torch
import numpy as np
from PIL import Image
import os

from cosmos_tokenizer.image_lib import ImageTokenizer
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from nanofm.utils.checkpoint import load_model_from_safetensors
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset

CKPT         = './outputs/nano4M/multiclevr_d6-6w512/checkpoint-final.safetensors'
COSMOS_ENC   = '/work/com-304/snoupy/nvidia/Cosmos-0.1-Tokenizer-DI16x16/encoder.jit'
COSMOS_DEC   = '/work/com-304/snoupy/nvidia/Cosmos-0.1-Tokenizer-DI16x16/decoder.jit'
DATASET_ROOT = '/work/com-304/datasets/clevr_com_304/'
MODALITIES   = ['tok_rgb@256', 'tok_depth@256', 'tok_normal@256', 'scene_desc']
OUTPUT_DIR   = './outputs/generated_images'
NUM_SAMPLES  = 5
device       = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def tokens_to_pil(token_ids, image_tokenizer):
    side = int(math.sqrt(token_ids.numel()))
    token_ids = token_ids.reshape(1, side, side).to(device)
    with torch.no_grad():
        img = image_tokenizer.decode(token_ids)
    img = (img[0].clamp(-1, 1).float().cpu() + 1) / 2
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)

print("Loading models...")
image_tokenizer = ImageTokenizer(checkpoint_enc=COSMOS_ENC, checkpoint_dec=COSMOS_DEC).to(device)
text_tokenizer  = AutoTokenizer.from_pretrained('gpt2')
text_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[SOS]', 'eos_token': '[EOS]'})
text_tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="[SOS] $A [EOS]",
    special_tokens=[('[EOS]', text_tokenizer.eos_token_id), ('[SOS]', text_tokenizer.bos_token_id)])

dataset = SimpleMultimodalDataset(root_dir=DATASET_ROOT, split='val', modalities=MODALITIES,
    sample_from_k_augmentations=1, text_tokenizer_path='gpt2', text_max_length=256, transforms=None)

print(f"Loading model from {CKPT}...")
model = load_model_from_safetensors(CKPT, device=device)
model.eval()

for i in range(NUM_SAMPLES):
    print(f"\nSample {i+1}/{NUM_SAMPLES}...")

    gt_rgb    = dataset[i]['tok_rgb@256']
    gt_depth  = dataset[i]['tok_depth@256']
    gt_normal = dataset[i]['tok_normal@256']
    gt_scene  = dataset[i]['scene_desc']
    n = gt_rgb.shape[0]

    # Save GT images
    tokens_to_pil(gt_rgb,    image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_gt_rgb.png')
    tokens_to_pil(gt_depth,  image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_gt_depth.png')
    tokens_to_pil(gt_normal, image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_gt_normal.png')

    # Chain 1: RGB -> depth -> normals -> scene_desc
    with torch.no_grad():
        enc_tok = gt_rgb.unsqueeze(0).to(device)
        enc_pos = torch.arange(n, device=device).unsqueeze(0)
        enc_mod = MODALITIES.index('tok_rgb@256') * torch.ones(1, n, device=device, dtype=torch.long)

        pred_depth,  x_tok, x_pos, x_mod = model.generate_one_modality_roar(enc_tok, enc_pos, enc_mod,
            target_mod='tok_depth@256', num_steps=8, temp=0.001, top_p=0.0, top_k=0.0)
        pred_normal, x_tok, x_pos, x_mod = model.generate_one_modality_roar(x_tok, x_pos, x_mod,
            target_mod='tok_normal@256', num_steps=8, temp=0.001, top_p=0.0, top_k=0.0)
        pred_scene, _, _, _ = model.generate_one_modality_roar(x_tok, x_pos, x_mod,
            target_mod='scene_desc', num_steps=128, temp=0.7, top_p=0.9, top_k=0.0)

    tokens_to_pil(pred_depth[0].cpu(),  image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_pred_depth.png')
    tokens_to_pil(pred_normal[0].cpu(), image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_pred_normal.png')

    pad_id = text_tokenizer.pad_token_id
    pred_ids = pred_scene[0].cpu()
    pred_text = text_tokenizer.decode(pred_ids[pred_ids != pad_id].tolist(), skip_special_tokens=True)
    gt_text   = text_tokenizer.decode(gt_scene[gt_scene != pad_id].tolist(), skip_special_tokens=True)
    print(f"  GT text:   {gt_text}")
    print(f"  Pred text: {pred_text}")

    # Chain 2: scene_desc -> RGB
    with torch.no_grad():
        n_text  = gt_scene.shape[0]
        enc_tok = gt_scene.unsqueeze(0).to(device)
        enc_pos = torch.arange(n_text, device=device).unsqueeze(0)
        enc_mod = MODALITIES.index('scene_desc') * torch.ones(1, n_text, device=device, dtype=torch.long)

        pred_rgb, _, _, _ = model.generate_one_modality_roar(enc_tok, enc_pos, enc_mod,
            target_mod='tok_rgb@256', num_steps=64, temp=0.7, top_p=0.9, top_k=0.0)

    tokens_to_pil(pred_rgb[0].cpu(), image_tokenizer).save(f'{OUTPUT_DIR}/sample{i+1}_pred_rgb_from_text.png')

print(f"\nDone! Images saved in {OUTPUT_DIR}/")
