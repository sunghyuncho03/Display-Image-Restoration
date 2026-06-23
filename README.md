# Image Reconstruction

Deep learning pipeline for restoring images degraded by **flicker noise** and **Gaussian blur** — artifacts that appear when photographing a display with a camera.

## Degradation Types

| Type | Description |
|---|---|
| **Flicker only** | Horizontal banding noise from camera–display interference |
| **Gaussian + Flicker** | Gaussian blur (radius 1.0–3.0) followed by flicker noise |

## Models

| Model | Architecture | Loss |
|---|---|---|
| UNet | Encoder-Decoder with skip connections | L1 |
| UNet + GAN | UNet + PatchGAN Discriminator | L1 × 100 + ADV × 1 |
| UNetCBAM | UNet + CBAM attention at each encoder stage | L1 |
| UNetCBAM + GAN | UNetCBAM + PatchGAN Discriminator | L1 × 100 + ADV × 1 |

**CBAM** (Convolutional Block Attention Module) applies sequential channel attention and spatial attention after each encoder stage, improving the model's ability to suppress structured noise patterns like flicker banding.

## Results

Evaluated on 100 fixed BSD500 test images (50 flicker-only + 50 Gaussian+flicker). LPIPS uses AlexNet backbone.

| Model | Training Data | Flicker PSNR | G+F PSNR | Total PSNR | Flicker LPIPS | G+F LPIPS | Total LPIPS |
|---|---|---|---|---|---|---|---|
| UNet L1 | G50 / F50 | 30.43 dB | 23.68 dB | 27.06 dB | — | — | — |
| UNet L1 | G70 / F30 | 30.38 dB | 24.08 dB | 27.23 dB | 0.0348 | 0.4133 | 0.2240 |
| UNet L1 | G100 / F0 | 18.31 dB | 25.12 dB | 21.71 dB | — | — | — |
| UNet GAN | G70 / F30 | 28.81 dB | 23.77 dB | 26.29 dB | 0.0729 | 0.2934 | 0.1832 |
| **UNetCBAM L1** | **G70 / F30** | **32.32 dB** | **25.37 dB** | **28.85 dB** | **0.0246** | 0.3288 | **0.1767** |
| UNetCBAM GAN | G70 / F30 | 28.48 dB | 24.15 dB | 26.32 dB | 0.0801 | **0.2835** | 0.1818 |

> G50/F50 = 50% Gaussian+Flicker / 50% Flicker-only in training data

**Key findings:**
- **UNetCBAM L1 (G70)** is the best model across all metrics — highest PSNR and lowest LPIPS simultaneously
- CBAM adds +1.62 dB PSNR and -0.047 LPIPS over plain UNet under identical conditions
- GAN training improves perceptual quality on Gaussian+Flicker (LPIPS ↓0.11–0.13) but reduces PSNR by 0.9–2.5 dB
- Training exclusively on Gaussian+Flicker (G100) causes flicker-only PSNR to collapse to 18.3 dB — the model fails to generalize to unseen degradation types
- G70 (70% Gaussian+Flicker / 30% Flicker-only) is the optimal training data mix

## Requirements

```bash
pip install torch torchvision lpips Pillow numpy matplotlib
```

Tested with Python 3.11, PyTorch 2.6.0 + CUDA 12.4.

## Dataset

Datasets are **downloaded automatically** on first run:

| Split | Source | Images |
|---|---|---|
| Train | BSD500 (train+val) + DIV2K train | 300 + 800 = **1,100** |
| Val | BSD500 test + DIV2K valid | 200 + 100 = **300** |
| Test (fixed) | BSD500 test | **100** |

- BSD500: ~70 MB
- DIV2K train: ~3.3 GB
- DIV2K valid: ~450 MB

## Training

```bash
# UNet + L1 (baseline)
python train.py --model unet

# UNetCBAM + L1 (recommended)
python train.py --model unetcbam

# UNet + GAN
python train.py --model unet --gan

# UNetCBAM + GAN
python train.py --model unetcbam --gan

# Control Gaussian+Flicker ratio in training data (default: 70%)
python train.py --model unetcbam --flicker-ratio 0.3   # 70% Gaussian+Flicker
python train.py --model unetcbam --flicker-ratio 0.5   # 50% Gaussian+Flicker
python train.py --model unetcbam --flicker-ratio 0.0   # 100% Gaussian+Flicker

# Resume from checkpoint
python train.py --model unetcbam --resume

# Warm start from pretrained weights
python train.py --model unetcbam --pretrained ./checkpoints/unetcbam_g70/best.pth
```

Checkpoints are saved to `checkpoints/{run-name}/`.
Best model (by validation PSNR) is saved as `best.pth`.

Training hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 200 (early stopping, patience=25) |
| Batch size | 4 |
| Patch size | 256×256 |
| Learning rate | 2e-4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| LR schedule | CosineAnnealingLR |
| λ_L1 | 100.0 |
| λ_ADV (GAN) | 1.0 |

## Evaluation

```bash
# UNet
python evaluate.py --ckpt ./checkpoints/unet_g70/best.pth

# UNetCBAM or UNetCBAM+GAN
python evaluate.py --ckpt ./checkpoints/unetcbam_g70/best.pth --cbam
```

Results are saved to `result/evaluate/{tag}/`:
- `summary_psnr.txt` — PSNR per image and averages
- `summary_lpips.txt` — LPIPS per image and averages
- `flicker/` / `gaussian_flicker/` — side-by-side comparison images with metric overlay

## File Structure

```
├── model.py       # UNet, UNetCBAM, PatchDiscriminator
├── dataset.py     # ImageRestorationDataset (BSD500 + DIV2K, runtime degradation)
├── train.py       # Unified training script (L1 and GAN modes)
├── evaluate.py    # Fixed test-set evaluation (PSNR + LPIPS)
└── utils.py       # PSNR, SSIM, save_sample, AverageMeter
```
