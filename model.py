import torch
import torch.nn as nn


# -- Basic block: Conv -> BN -> ReLU x2 --
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -- Encoder block: MaxPool -> DoubleConv (resolution /2) --
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# -- Decoder block: upsample -> concat skip -> DoubleConv (resolution x2) --
class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -- U-Net
#
# Input : (B, 3, H, W)  blurred image, normalized to [-1, 1]
# Output: (B, 3, H, W)  restored image, Tanh -> [-1, 1]
#
# Encoder : 3 -> 64 -> 128 -> 256 -> 512
# Bottleneck:          512 -> 1024
# Decoder : 1024 -> 512 -> 256 -> 128 -> 64
# --
class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)         # 256 -> 256
        self.enc2 = Down(base_ch,     base_ch * 2)     # 256 -> 128
        self.enc3 = Down(base_ch * 2, base_ch * 4)     # 128 -> 64
        self.enc4 = Down(base_ch * 4, base_ch * 8)     # 64  -> 32

        # Bottleneck
        self.bottleneck = Down(base_ch * 8, base_ch * 16)  # 32 -> 16

        # Decoder
        self.dec4 = Up(base_ch * 16, base_ch * 8)     # 16  -> 32
        self.dec3 = Up(base_ch * 8,  base_ch * 4)     # 32  -> 64
        self.dec2 = Up(base_ch * 4,  base_ch * 2)     # 64  -> 128
        self.dec1 = Up(base_ch * 2,  base_ch)         # 128 -> 256

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder (save feature maps for skip connections)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)


# -- CBAM (Convolutional Block Attention Module) --

class _ChannelAttn(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = x.flatten(2).mean(-1)
        mx  = x.flatten(2).max(-1).values
        return x * self.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)


class _SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.ca = _ChannelAttn(ch, reduction)
        self.sa = _SpatialAttn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# -- UNet + CBAM
# CBAM applied after each encoder stage output,
# used for both skip connections and next encoder input
# --
class UNetCBAM(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 64):
        super().__init__()

        self.enc1       = DoubleConv(in_ch,    base_ch)
        self.enc2       = Down(base_ch,        base_ch * 2)
        self.enc3       = Down(base_ch * 2,    base_ch * 4)
        self.enc4       = Down(base_ch * 4,    base_ch * 8)
        self.bottleneck = Down(base_ch * 8,    base_ch * 16)

        self.dec4 = Up(base_ch * 16, base_ch * 8)
        self.dec3 = Up(base_ch * 8,  base_ch * 4)
        self.dec2 = Up(base_ch * 4,  base_ch * 2)
        self.dec1 = Up(base_ch * 2,  base_ch)

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, kernel_size=1),
            nn.Tanh(),
        )

        self.cbam1 = CBAM(base_ch)
        self.cbam2 = CBAM(base_ch * 2)
        self.cbam3 = CBAM(base_ch * 4)
        self.cbam4 = CBAM(base_ch * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.cbam1(self.enc1(x))
        s2 = self.cbam2(self.enc2(s1))
        s3 = self.cbam3(self.enc3(s2))
        s4 = self.cbam4(self.enc4(s3))

        b  = self.bottleneck(s4)

        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)


# -- PatchGAN Discriminator
#
# Input : concat(blur, sharp/generated) -> (B, 6, H, W)
# Output: (B, 1, H', W') patch-wise real/fake logits
# --
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 6, base_ch: int = 64):
        super().__init__()

        def block(ic, oc, stride, norm=True):
            layers = [nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch,         base_ch,     stride=2, norm=False),  # 256 -> 128
            *block(base_ch,       base_ch * 2, stride=2),              # 128 -> 64
            *block(base_ch * 2,   base_ch * 4, stride=2),              # 64  -> 32
            *block(base_ch * 4,   base_ch * 8, stride=1),              # 32  -> 31
            nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1),  # 31 -> 30
        )

    def forward(self, blur: torch.Tensor, sharp: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([blur, sharp], dim=1))


if __name__ == '__main__':
    model = UNetCBAM()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'UNetCBAM parameters: {total:.2f}M')

    dummy = torch.randn(1, 3, 256, 256)
    out   = model(dummy)
    print(f'Input: {dummy.shape} -> Output: {out.shape}')
