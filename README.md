# Image Restoration

Restoring images degraded by **flicker noise** and **Gaussian blur** (simulating screen photography artifacts).

## Models

| Model | Architecture | Loss |
|---|---|---|
| UNet | Encoder-Decoder with skip connections | L1 |
| UNet + GAN | UNet + PatchGAN Discriminator | L1 × 100 + ADV × 1 |
| UNetCBAM | UNet + CBAM attention | L1 |
| UNetCBAM + GAN | UNet + CBAM + PatchGAN | L1 × 100 + ADV × 1 |

**CBAM** (Convolutional Block Attention Module) applies channel and spatial attention after each encoder stage.

## Requirements

```bash
pip install torch torchvision lpips Pillow numpy matplotlib
```

Tested with Python 3.11, PyTorch 2.6.0 + CUDA 12.4.

## Dataset

Training uses **BSD500** and **DIV2K** (downloaded automatically on first run):

- Train: BSD500 train+val (300 images) + DIV2K train (800 images) = 1,100 images
- Val: BSD500 test (200 images) + DIV2K valid (100 images) = 300 images

Degradation applied at runtime:
- **Flicker only**: horizontal banding noise
- **Gaussian + Flicker**: Gaussian blur (radius 1.0–3.0) + flicker noise

## Training

```bash
# UNet + L1
python train.py --model unet

# UNet + GAN
python train.py --model unet --gan

# UNetCBAM + L1
python train.py --model unetcbam

# UNetCBAM + GAN
python train.py --model unetcbam --gan

# Control gaussian+flicker ratio (default: 70%)
python train.py --model unetcbam --flicker-ratio 0.3   # gaussian 70%
python train.py --model unetcbam --flicker-ratio 0.5   # gaussian 50%
python train.py --model unetcbam --flicker-ratio 0.0   # gaussian 100%

# Resume training
python train.py --model unetcbam --gan --resume

# Warm start from pretrained weights
python train.py --model unetcbam --gan --pretrained ./checkpoints/unetcbam_g70/best.pth
```

Checkpoints are saved to `checkpoints/{run-name}/`.  
Best model (by validation PSNR) is saved as `best.pth`.

## Evaluation

Fixed test set: **50 flicker-only** + **50 Gaussian+flicker** images from BSD500 test set.  
Metrics: **PSNR** (higher is better), **LPIPS** (lower is better).

```bash
# UNet
python evaluate.py --ckpt ./checkpoints/unet_g70/best.pth

# UNetCBAM or UNetCBAM + GAN
python evaluate.py --ckpt ./checkpoints/unetcbam_g70/best.pth --cbam
```

Results are saved to `result/evaluate/{tag}/`:
- `summary_psnr.txt` — PSNR per image + averages
- `summary_lpips.txt` — LPIPS per image + averages
- `flicker/` / `gaussian_flicker/` — comparison images with metric overlay

## File Structure

```
├── model.py       # UNet, UNetCBAM, PatchDiscriminator
├── dataset.py     # Dataset (BSD500 + DIV2K, degradation augmentation)
├── utils.py       # PSNR, SSIM, helpers
├── train.py       # Training (all model/dataset combinations)
└── evaluate.py    # Evaluation (PSNR + LPIPS)
```
