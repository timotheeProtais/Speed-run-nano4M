# nano4M

Nano4M with two things added : Muon and FlexAttention (our best model).
generated_images.py and evaluate.py also added to check quality.

---

## File Hierarchy

```
nano4M/
├── cfgs/                          # Hydra configuration files
│   ├── nano4M/                    # Configs for the nano4M model
│
├── data/                          # Local datasets (CIFAR-10, MNIST)
│   ├── cifar-10-batches-py/
│   ├── cifar-10-python.tar.gz
│   └── MNIST/
│
├── nanofm/                        # Core library
│   ├── data/                      # Data loading and preprocessing
│   │   └── multimodal.py          # Multimodal masked dataloader
│   ├── modeling/                  # Training utilities (losses, etc.)
│   │   └── transformer_layers.py  # Transformer layers with FlexAttention
│   ├── models/                    # Model architectures
│   │   └── fourm.py               # FourM encoder-decoder transformer
│   └── utils/                     # Optimizers, helpers
│       └── muon.py                # Muon optimizer implementation
│
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── COM304_FM_part3_nano4M.ipynb
│
├── outputs/                       # Training outputs
│   ├── generated_images/          # Images generated after training
│   └── nano4M/                    # Model checkpoints and logs
│
├── logs/                          # SLURM job output logs
│
├── run_training.py                # Main training entry point
├── evaluate.py                    # Evaluation script
├── generate_images.py             # Image generation
│
├── nano4m.run                     # SLURM job script for nano4M training
│
├── submit_job.sh                  # Helper script to submit SLURM jobs
├── setup_env.sh                   # Environment setup script
├── pyproject.toml                 # Python project metadata and dependencies
└── wandb/                         # Weights & Biases run artifacts
```

---

## Requirements

### Hardware

Training is designed to run on 2× L40-48GB GPUs (distributed, global batch size 512).

### Software

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | |
| PyTorch | ≥ 2.5.0 | |
| hydra-core | ≥ 1.3.0 | Config management |
| wandb | ≥ 0.17.0 | Experiment tracking |
| transformers | ≥ 4.40.0 | GPT-2 tokenizer |
| numpy | ≥ 1.26.0 | |
| Pillow | ≥ 10.0.0 | |
| ||||
| cosmos-tokenizer | ≥ 0.1.0+ | For evaluate.py and generate_images.py |
| pip install cosmos-tokenizer |||
| tokenizers | ≥ 0.19.0 | For evaluate.py and generate_images.py |
| pip install tokenizers |||
| nltk | ≥ 3.8.0 | For evaluate.py |
| pip install nltk |||

> **Note on FlexAttention**: David

> **Note on Muon**: The Muon optimizer is implemented locally in `nanofm/utils/muon.py` — no extra pip install needed.

---

## Reproducing Results

### Configuration

The main config for nano4M is under `cfgs/nano4M/`.

Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 256 (per GPU) | Global batch size: 512 |
| `total_tokens` | 2000M | Total training tokens |
| `warmup_tokens` | 50M | Linear warmup duration |
| `lr` | 1e-3 | Peak learning rate |
| `min_lr` | 6e-5 | Minimum LR after cosine decay |
| `weight_decay` | 0.05 | AdamW weight decay |
| `clip_grad` | 2.0 | Gradient clipping norm |
| `dtype` | bf16 | Mixed precision (bf16 on L40/A100) |
| `dim` | 512 | Transformer model dimension |
| `enc_depth` | 6 | Number of encoder layers |
| `dec_depth` | 6 | Number of decoder layers |
| `head_dim` | 64 | Per-head attention dimension |
| `mlp_ratio` | 2.67 | MLP expansion ratio |
| `eval_freq` | 100M tokens | Evaluation frequency |
| `save_ckpt_freq` | 1000M tokens | Checkpoint save frequency |


**Modalities**: `tok_rgb@256`, `tok_depth@256`, `tok_normal@256`, `scene_desc`

---

### Training

Submit the nano4M training job with:

```bash
sbatch nano4m.run
```

### Evaluation


In evaluate.py, change the path to your checkpoint and do:

```bash
python evaluate.py 
```

### Image Generation

Same as evaluate.py

```bash
python generate_images.py
```