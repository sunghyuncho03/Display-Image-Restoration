"""
Unified training script

Usage:
    python train.py --model unet                        # UNet + L1
    python train.py --model unet --gan                  # UNet + GAN
    python train.py --model unetcbam                    # UNetCBAM + L1
    python train.py --model unetcbam --gan              # UNetCBAM + GAN

    python train.py --model unetcbam --resume
    python train.py --model unetcbam --gan --run-name unetcbam_gan_v2
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model   import UNet, UNetCBAM, PatchDiscriminator
from utils   import psnr, ssim, save_sample, AverageMeter


# -- Hyperparameters --
_HERE         = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT     = os.path.join(_HERE, 'data')

EPOCHS        = 200
BATCH_SIZE    = 4
PATCH_SIZE    = 256
LR            = 2e-4
NUM_WORKERS   = 4
LAMBDA_L1     = 100.0
LAMBDA_ADV    = 1.0
FLICKER_RATIO = 0.30
ES_PATIENCE   = 25
ES_MIN_DELTA  = 0.01

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
l1_loss  = nn.L1Loss()
adv_loss = nn.BCEWithLogitsLoss()


# -- Early Stopping --
class EarlyStopping:
    def __init__(self):
        self.best    = -float('inf')
        self.counter = 0

    def step(self, val_psnr: float) -> bool:
        if val_psnr >= self.best + ES_MIN_DELTA:
            self.best    = val_psnr
            self.counter = 0
        else:
            self.counter += 1
            print(f'  [EarlyStopping] no improvement {self.counter}/{ES_PATIENCE}'
                  f'  (best {self.best:.2f} dB)')
            if self.counter >= ES_PATIENCE:
                print(f'  [EarlyStopping] stopping training')
                return True
        return False


# -- Live Plot --
class LivePlot:
    def __init__(self, save_dir: str, use_gan: bool):
        self.save_dir = save_dir
        self.use_gan  = use_gan
        os.makedirs(save_dir, exist_ok=True)
        self.h = {'ep': [], 'g_loss': [], 'd_loss': [], 'tr_psnr': [], 'val_psnr': [], 'val_ssim': []}
        plt.ion()
        self.fig_m, self.axes_m = plt.subplots(1, 3 if use_gan else 2, figsize=(15 if use_gan else 10, 4))
        self.fig_s, self.ax_s   = plt.subplots(1, 3, figsize=(15, 4))
        plt.pause(0.001)

    def update(self, epoch, g_loss, val_psnr, val_ssim, train_psnr, sample=None, d_loss=None):
        h = self.h
        h['ep'].append(epoch); h['g_loss'].append(g_loss)
        h['d_loss'].append(d_loss or 0)
        h['tr_psnr'].append(train_psnr); h['val_psnr'].append(val_psnr); h['val_ssim'].append(val_ssim)

        if self.use_gan:
            ax_g, ax_d, ax_p = self.axes_m
        else:
            ax_g, ax_p = self.axes_m

        ax_g.cla()
        ax_g.plot(h['ep'], h['g_loss'], 'b-o', ms=3)
        ax_g.set(title='G Loss'); ax_g.grid(alpha=0.3)

        if self.use_gan:
            ax_d.cla()
            ax_d.plot(h['ep'], h['d_loss'], 'r-o', ms=3)
            ax_d.set(title='D Loss'); ax_d.grid(alpha=0.3)

        ax_p.cla()
        ax_p.plot(h['ep'], h['tr_psnr'], 'b-o', ms=3, label='Train')
        ax_p.plot(h['ep'], h['val_psnr'], 'r-o', ms=3, label='Val')
        ax_p.set(title='PSNR (dB)'); ax_p.legend(); ax_p.grid(alpha=0.3)

        self.fig_m.suptitle(f'Epoch {epoch}  |  Val PSNR {val_psnr:.2f} dB  |  SSIM {val_ssim:.4f}')
        self.fig_m.tight_layout()
        self.fig_m.savefig(os.path.join(self.save_dir, 'metrics.png'), dpi=120)

        if sample is not None:
            blur, pred, sharp = sample
            def t2np(t):
                return ((t[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
            for ax, img, title in zip(self.ax_s,
                [t2np(blur), t2np(pred), t2np(sharp)],
                ['blur', 'restored', 'sharp']):
                ax.cla(); ax.imshow(img); ax.set_title(title, fontsize=11); ax.axis('off')
            self.fig_s.suptitle(f'Epoch {epoch}')
            self.fig_s.tight_layout()
            self.fig_s.savefig(os.path.join(self.save_dir, f'sample_{epoch:04d}.png'), dpi=120)

        plt.pause(0.001)


# -- Checkpoint --
def save_checkpoint(state: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, 'latest.pth'))


def load_checkpoint(save_dir: str) -> dict:
    path = os.path.join(save_dir, 'latest.pth')
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location=DEVICE)


# -- Validation --
@torch.no_grad()
def validate(G, loader, epoch, sample_dir):
    G.eval()
    meter_psnr = AverageMeter()
    meter_ssim = AverageMeter()
    sample = None
    for i, (blur, sharp) in enumerate(loader):
        blur, sharp = blur.to(DEVICE), sharp.to(DEVICE)
        pred = G(blur)
        meter_psnr.update(psnr(pred, sharp))
        meter_ssim.update(ssim(pred, sharp))
        if i == 0:
            save_sample(blur, pred, sharp, sample_dir, epoch)
            sample = (blur[:1].detach(), pred[:1].detach(), sharp[:1].detach())
    return meter_psnr.avg, meter_ssim.avg, sample


# -- Train one epoch (without GAN) --
def train_epoch(G, loader, opt):
    G.train()
    meter_loss = AverageMeter()
    meter_psnr = AverageMeter()
    for i, (blur, sharp) in enumerate(loader):
        blur, sharp = blur.to(DEVICE), sharp.to(DEVICE)
        pred = G(blur)
        loss = LAMBDA_L1 * l1_loss(pred, sharp)
        opt.zero_grad(); loss.backward(); opt.step()
        meter_loss.update(loss.item(), blur.size(0))
        meter_psnr.update(psnr(pred.detach(), sharp), blur.size(0))
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(loader)}]  L1: {meter_loss.avg:.4f}  PSNR: {meter_psnr.avg:.2f} dB')
    return meter_loss.avg, meter_psnr.avg


# -- Train one epoch (with GAN) --
def train_epoch_gan(G, D, loader, opt_g, opt_d):
    G.train(); D.train()
    meter_g    = AverageMeter()
    meter_d    = AverageMeter()
    meter_psnr = AverageMeter()
    for i, (blur, sharp) in enumerate(loader):
        blur, sharp = blur.to(DEVICE), sharp.to(DEVICE)
        bs = blur.size(0)

        with torch.no_grad():
            pred_d = G(blur)
        loss_d = 0.5 * (adv_loss(D(blur, sharp),  torch.ones_like(D(blur, sharp))) +
                        adv_loss(D(blur, pred_d), torch.zeros_like(D(blur, pred_d))))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        pred_g      = G(blur)
        fake_logits = D(blur, pred_g)
        loss_g = (LAMBDA_ADV * adv_loss(fake_logits, torch.ones_like(fake_logits))
                  + LAMBDA_L1 * l1_loss(pred_g, sharp))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        meter_g.update(loss_g.item(), bs)
        meter_d.update(loss_d.item(), bs)
        meter_psnr.update(psnr(pred_g.detach(), sharp), bs)

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(loader)}]  G: {meter_g.avg:.4f}  D: {meter_d.avg:.4f}'
                  f'  PSNR: {meter_psnr.avg:.2f} dB')

    return meter_g.avg, meter_d.avg, meter_psnr.avg


# -- Main --
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         default='unetcbam', choices=['unet', 'unetcbam'])
    parser.add_argument('--gan',           action='store_true')
    parser.add_argument('--flicker-ratio', type=float, default=FLICKER_RATIO,
                        help='flicker-only ratio (default 0.30 = 30%% flicker / 70%% gaussian+flicker)')
    parser.add_argument('--run-name',      default=None,
                        help='default: {model}[_gan]_g{gaussian%%}')
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--pretrained',    default=None)
    args = parser.parse_args()

    gaussian_pct = int((1 - args.flicker_ratio) * 100)
    run_name   = args.run_name or (args.model + ('_gan' if args.gan else '') + f'_g{gaussian_pct}')
    save_dir   = os.path.join(_HERE, 'checkpoints', run_name)
    sample_dir = os.path.join(_HERE, 'samples',     run_name)

    print(f'Device    : {DEVICE}')
    print(f'Model     : {args.model}{"  + GAN" if args.gan else ""}')
    print(f'Run name  : {run_name}  ->  {save_dir}')
    print(f'Loss      : L1x{LAMBDA_L1}' + (f' + ADVx{LAMBDA_ADV}' if args.gan else ''))
    print(f'Data      : flicker {args.flicker_ratio*100:.0f}% / gaussian+flicker {(1-args.flicker_ratio)*100:.0f}%')

    train_loader, val_loader = get_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, patch_size=PATCH_SIZE,
        flicker_ratio=args.flicker_ratio,
    )
    print(f'train: {len(train_loader)} batches  val: {len(val_loader)} batches')

    G = (UNetCBAM() if args.model == 'unetcbam' else UNet()).to(DEVICE)
    opt_g = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sch_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=1e-6)

    if args.gan:
        D     = PatchDiscriminator().to(DEVICE)
        opt_d = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
        sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS, eta_min=1e-6)

    start_epoch, best_psnr = 0, 0.0

    if args.resume:
        ckpt = load_checkpoint(save_dir)
        if ckpt:
            G.load_state_dict(ckpt['G_state'])
            opt_g.load_state_dict(ckpt['opt_g'])
            sch_g.load_state_dict(ckpt['sch_g'])
            if args.gan:
                D.load_state_dict(ckpt['D_state'])
                opt_d.load_state_dict(ckpt['opt_d'])
                sch_d.load_state_dict(ckpt['sch_d'])
            start_epoch = ckpt['epoch']
            best_psnr   = ckpt['best_psnr']
            print(f'[Checkpoint loaded] epoch {start_epoch}, best PSNR {best_psnr:.2f}')
    elif args.pretrained:
        state = torch.load(args.pretrained, map_location=DEVICE)
        if 'G_state' in state: state = state['G_state']
        G.load_state_dict(state)
        print(f'[Warm start] {args.pretrained}')
    else:
        print('Training from scratch')

    monitor = LivePlot(sample_dir, args.gan)
    stopper = EarlyStopping()

    for epoch in range(start_epoch, EPOCHS):
        print(f'\n-- Epoch [{epoch+1}/{EPOCHS}]  (lr: {sch_g.get_last_lr()[0]:.2e}) --')

        if args.gan:
            g_loss, d_loss, train_psnr = train_epoch_gan(G, D, train_loader, opt_g, opt_d)
            sch_d.step()
        else:
            g_loss, train_psnr = train_epoch(G, train_loader, opt_g)
            d_loss = None

        val_psnr, val_ssim, sample = validate(G, val_loader, epoch + 1, sample_dir)
        sch_g.step()

        print(f'G loss: {g_loss:.4f}' + (f'  D loss: {d_loss:.4f}' if args.gan else ''))
        print(f'Train PSNR: {train_psnr:.2f} dB  |  Val PSNR: {val_psnr:.2f} dB  SSIM: {val_ssim:.4f}')

        monitor.update(epoch + 1, g_loss, val_psnr, val_ssim, train_psnr, sample, d_loss)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            os.makedirs(save_dir, exist_ok=True)
            torch.save(G.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f'  * best model saved (PSNR: {best_psnr:.2f} dB)')

        ckpt_state = {
            'epoch': epoch + 1, 'best_psnr': best_psnr,
            'G_state': G.state_dict(),
            'opt_g': opt_g.state_dict(), 'sch_g': sch_g.state_dict(),
        }
        if args.gan:
            ckpt_state.update({
                'D_state': D.state_dict(),
                'opt_d': opt_d.state_dict(), 'sch_d': sch_d.state_dict(),
            })
        save_checkpoint(ckpt_state, save_dir)

        if stopper.step(val_psnr):
            break

    print(f'\nTraining complete. Best PSNR: {best_psnr:.2f} dB')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
