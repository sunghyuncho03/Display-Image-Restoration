import os
import math

import torch
import torch.nn.functional as F
import torchvision.utils as vutils


# -- PSNR (Peak Signal-to-Noise Ratio) -- higher is better --
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """pred, target: tensors normalized to [-1, 1] with shape (B, C, H, W)"""
    pred   = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)


# -- SSIM (Structural Similarity Index) -- higher is better --
def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """pred, target: tensors normalized to [-1, 1] with shape (B, C, H, W)"""
    pred   = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2

    kernel = _gaussian_kernel(window_size, sigma=1.5).to(pred.device)
    kernel = kernel.expand(pred.size(1), 1, window_size, window_size)

    pad = window_size // 2
    mu1    = F.conv2d(pred,   kernel, padding=pad, groups=pred.size(1))
    mu2    = F.conv2d(target, kernel, padding=pad, groups=pred.size(1))
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=pred.size(1)) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


# -- Save sample comparison image during training --
def save_sample(blur: torch.Tensor, pred: torch.Tensor, sharp: torch.Tensor,
                save_dir: str, epoch: int, n: int = 4):
    """Save side-by-side comparison: blur / pred / sharp (first n samples from batch)"""
    os.makedirs(save_dir, exist_ok=True)

    def denorm(t):
        return (t.clamp(-1, 1) + 1) / 2

    imgs = torch.cat([denorm(blur[:n]), denorm(pred[:n]), denorm(sharp[:n])], dim=0)
    grid = vutils.make_grid(imgs, nrow=n, padding=2, pad_value=1.0)
    vutils.save_image(grid, os.path.join(save_dir, f'epoch_{epoch:04d}.png'))


# -- Running average tracker --
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
