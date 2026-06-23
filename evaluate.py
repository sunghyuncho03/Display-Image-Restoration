"""
Fixed test set evaluation (50 flicker-only + 50 gaussian+flicker)

- 100 BSD500 test images, seed fixed per image (reproducible noise)
- Images 0~49  : flicker-only
- Images 50~99 : gaussian+flicker  (radius 1.0~3.0, seed fixed)
- Saves: blur | restored (PSNR/LPIPS overlay) | ground truth comparison
- Saves: summary_psnr.txt, summary_lpips.txt

Usage:
    python evaluate.py --ckpt ./checkpoints/unet_g70/best.pth
    python evaluate.py --ckpt ./checkpoints/unetcbam_g70/best.pth --cbam
"""
import argparse
import math
import os
import random
from pathlib import Path

import torch
import lpips
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms

from model import UNet, UNetCBAM
from dataset import _flickering_noise

_HERE           = os.path.dirname(os.path.abspath(__file__))
BSD500_TEST_DIR = Path(_HERE) / 'data' / 'BSR' / 'BSDS500' / 'data' / 'images' / 'test'
N_FLICKER  = 50
N_GAUSSIAN = 50
PATCH_SIZE = 256
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -- Degradation --

def apply_flicker_only(img: Image.Image) -> Image.Image:
    return _flickering_noise(img)


def apply_gaussian_flicker(img: Image.Image) -> Image.Image:
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    return _flickering_noise(img)


# -- Pre/Post processing --

def center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < PATCH_SIZE or h < PATCH_SIZE:
        scale = PATCH_SIZE / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = img.size
    x, y = (w - PATCH_SIZE) // 2, (h - PATCH_SIZE) // 2
    return img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))


def to_tensor(img: Image.Image) -> torch.Tensor:
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(t)
    return t.unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.squeeze(0).clamp(-1, 1)
    return transforms.ToPILImage()((t + 1) / 2)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred   = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2
    mse = ((pred - target) ** 2).mean().item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')


# -- Model loading --

def load_model(ckpt_path: str, use_cbam: bool = False) -> UNet:
    model = (UNetCBAM() if use_cbam else UNet()).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'G_state' in state:
        state = state['G_state']
    model.load_state_dict(state)
    model.eval()
    print(f'Model: {"UNetCBAM" if use_cbam else "UNet"}')
    return model


# -- Single image inference --

@torch.no_grad()
def process_one(model, lpips_fn, sharp_img, blur_img, noise_label, out_dir, stem):
    inp  = to_tensor(blur_img).to(DEVICE)
    gt   = to_tensor(sharp_img).to(DEVICE)
    pred = model(inp)

    psnr_val     = calc_psnr(pred, gt)
    lpips_val    = lpips_fn(pred, gt).item()
    restored_img = to_pil(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, label in zip(
        axes,
        [blur_img, restored_img, sharp_img],
        ['Blur (Input)', 'Restored', 'Ground Truth'],
    ):
        ax.imshow(img); ax.set_title(label, fontsize=12); ax.axis('off')

    axes[1].text(
        4, PATCH_SIZE - 4, f'PSNR: {psnr_val:.2f} dB\nLPIPS: {lpips_val:.4f}',
        fontsize=10, color='white', fontweight='bold',
        verticalalignment='bottom',
        bbox=dict(facecolor='black', alpha=0.55, pad=3, edgecolor='none'),
    )
    fig.suptitle(f'{stem}  [{noise_label}]', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_compare.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    return psnr_val, lpips_val


# -- Evaluation --

def evaluate(ckpt_path: str, tag: str, use_cbam: bool = False):
    assert BSD500_TEST_DIR.exists(), f'BSD500 test dir not found: {BSD500_TEST_DIR}'
    all_paths = sorted(BSD500_TEST_DIR.glob('*.jpg'))
    assert len(all_paths) >= N_FLICKER + N_GAUSSIAN, 'Not enough images'

    flicker_paths  = all_paths[:N_FLICKER]
    gaussian_paths = all_paths[N_FLICKER: N_FLICKER + N_GAUSSIAN]

    model    = load_model(ckpt_path, use_cbam)
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    print(f'Checkpoint : {ckpt_path}')
    print(f'Eval       : {N_FLICKER} flicker + {N_GAUSSIAN} gaussian+flicker images')

    base_dir     = Path(_HERE) / 'result' / 'evaluate' / tag
    flicker_dir  = base_dir / 'flicker'
    gaussian_dir = base_dir / 'gaussian_flicker'
    flicker_dir.mkdir(parents=True, exist_ok=True)
    gaussian_dir.mkdir(parents=True, exist_ok=True)

    # -- Flicker-only --
    print('\n[1/2] Flicker-only...')
    psnr_flicker, lpips_flicker = [], []
    for idx, p in enumerate(flicker_paths):
        random.seed(idx)
        sharp_img = center_crop(Image.open(p).convert('RGB'))
        blur_img  = apply_flicker_only(sharp_img)
        psnr_val, lpips_val = process_one(
            model, lpips_fn, sharp_img, blur_img, 'flicker', flicker_dir, p.stem)
        psnr_flicker.append(psnr_val)
        lpips_flicker.append(lpips_val)
        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{N_FLICKER}]'
                  f'  PSNR {sum(psnr_flicker)/len(psnr_flicker):.2f} dB'
                  f'  LPIPS {sum(lpips_flicker)/len(lpips_flicker):.4f}')

    # -- Gaussian+Flicker --
    print('\n[2/2] Gaussian+Flicker...')
    psnr_gaussian, lpips_gaussian = [], []
    for idx, p in enumerate(gaussian_paths):
        random.seed(1000 + idx)
        sharp_img = center_crop(Image.open(p).convert('RGB'))
        blur_img  = apply_gaussian_flicker(sharp_img)
        psnr_val, lpips_val = process_one(
            model, lpips_fn, sharp_img, blur_img, 'gaussian+flicker', gaussian_dir, p.stem)
        psnr_gaussian.append(psnr_val)
        lpips_gaussian.append(lpips_val)
        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{N_GAUSSIAN}]'
                  f'  PSNR {sum(psnr_gaussian)/len(psnr_gaussian):.2f} dB'
                  f'  LPIPS {sum(lpips_gaussian)/len(lpips_gaussian):.4f}')

    # -- Summary (PSNR) --
    avg_psnr_f = sum(psnr_flicker)  / len(psnr_flicker)
    avg_psnr_g = sum(psnr_gaussian) / len(psnr_gaussian)
    avg_psnr_t = (sum(psnr_flicker) + sum(psnr_gaussian)) / (len(psnr_flicker) + len(psnr_gaussian))

    with open(base_dir / 'summary_psnr.txt', 'w') as f:
        f.write(f'Checkpoint      : {ckpt_path}\n')
        f.write(f'Gaussian radius : uniform(1.0, 3.0)  [seed fixed per image]\n')
        f.write(f'\n[Flicker-only]       {N_FLICKER} images\n')
        f.write(f'  Avg PSNR : {avg_psnr_f:.4f} dB\n')
        f.write(f'  Min PSNR : {min(psnr_flicker):.4f} dB\n')
        f.write(f'  Max PSNR : {max(psnr_flicker):.4f} dB\n')
        f.write(f'\n[Gaussian+Flicker]   {N_GAUSSIAN} images\n')
        f.write(f'  Avg PSNR : {avg_psnr_g:.4f} dB\n')
        f.write(f'  Min PSNR : {min(psnr_gaussian):.4f} dB\n')
        f.write(f'  Max PSNR : {max(psnr_gaussian):.4f} dB\n')
        f.write(f'\n[Total]              {N_FLICKER + N_GAUSSIAN} images\n')
        f.write(f'  Avg PSNR : {avg_psnr_t:.4f} dB\n')
        f.write('\n--- Per-image (Flicker) ---\n')
        for p, v in zip(flicker_paths, psnr_flicker):
            f.write(f'  {p.name:30s}  {v:.4f} dB\n')
        f.write('\n--- Per-image (Gaussian+Flicker) ---\n')
        for p, v in zip(gaussian_paths, psnr_gaussian):
            f.write(f'  {p.name:30s}  {v:.4f} dB\n')

    # -- Summary (LPIPS) --
    avg_lpips_f = sum(lpips_flicker)  / len(lpips_flicker)
    avg_lpips_g = sum(lpips_gaussian) / len(lpips_gaussian)
    avg_lpips_t = (sum(lpips_flicker) + sum(lpips_gaussian)) / (len(lpips_flicker) + len(lpips_gaussian))

    with open(base_dir / 'summary_lpips.txt', 'w') as f:
        f.write(f'Checkpoint      : {ckpt_path}\n')
        f.write(f'LPIPS network   : AlexNet  (lower is better)\n')
        f.write(f'Gaussian radius : uniform(1.0, 3.0)  [seed fixed per image]\n')
        f.write(f'\n[Flicker-only]       {N_FLICKER} images\n')
        f.write(f'  Avg LPIPS : {avg_lpips_f:.4f}\n')
        f.write(f'  Min LPIPS : {min(lpips_flicker):.4f}\n')
        f.write(f'  Max LPIPS : {max(lpips_flicker):.4f}\n')
        f.write(f'\n[Gaussian+Flicker]   {N_GAUSSIAN} images\n')
        f.write(f'  Avg LPIPS : {avg_lpips_g:.4f}\n')
        f.write(f'  Min LPIPS : {min(lpips_gaussian):.4f}\n')
        f.write(f'  Max LPIPS : {max(lpips_gaussian):.4f}\n')
        f.write(f'\n[Total]              {N_FLICKER + N_GAUSSIAN} images\n')
        f.write(f'  Avg LPIPS : {avg_lpips_t:.4f}\n')
        f.write('\n--- Per-image (Flicker) ---\n')
        for p, v in zip(flicker_paths, lpips_flicker):
            f.write(f'  {p.name:30s}  {v:.4f}\n')
        f.write('\n--- Per-image (Gaussian+Flicker) ---\n')
        for p, v in zip(gaussian_paths, lpips_gaussian):
            f.write(f'  {p.name:30s}  {v:.4f}\n')

    print(f'\n{"="*55}')
    print(f'  Flicker-only     PSNR: {avg_psnr_f:.4f} dB   LPIPS: {avg_lpips_f:.4f}')
    print(f'  Gaussian+Flicker PSNR: {avg_psnr_g:.4f} dB   LPIPS: {avg_lpips_g:.4f}')
    print(f'  Total            PSNR: {avg_psnr_t:.4f} dB   LPIPS: {avg_lpips_t:.4f}')
    print(f'{"="*55}')
    print(f'Saved to: {base_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='checkpoint path')
    parser.add_argument('--tag',  default=None,  help='result folder name (default: checkpoint folder name)')
    parser.add_argument('--cbam', action='store_true', help='load as UNetCBAM')
    args = parser.parse_args()

    tag = args.tag or Path(args.ckpt).parent.name
    evaluate(args.ckpt, tag, args.cbam)


if __name__ == '__main__':
    main()
