import random
import tarfile
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATCH_SIZE = 256

# ── Dataset URLs ─────────────────────────────────────────────

# BSD500: ~70 MB, 500 natural images (~480×320)
_BSD500_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
    "BSR/BSR_bsds500.tgz"
)
_BSD500_IMAGE_DIR = Path("BSR") / "BSDS500" / "data" / "images"

# DIV2K: 2K high-resolution natural images (train 800 images ~3.3 GB / valid 100 images ~450 MB)
_DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
_DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


# ── Download ──────────────────────────────────────────────────

def _progress(count, block, total):
    print(f"\r  {min(count * block / total * 100, 100):.1f}%", end="", flush=True)


def download_bsd500(root: str = "./data") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    image_dir = root / _BSD500_IMAGE_DIR
    if image_dir.exists():
        return image_dir

    archive = root / "BSR_bsds500.tgz"
    if not archive.exists():
        print("Downloading BSD500 (~70 MB)...")
        urllib.request.urlretrieve(_BSD500_URL, archive, _progress)
        print()

    print("Extracting...")
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    print(f"Done: {image_dir}")
    return image_dir


def download_div2k(root: str = "./data", split: str = "train") -> Path:
    """Download DIV2K HR images and return the folder path.

    split='train' -> 800 images (~3.3 GB)
    split='valid' -> 100 images (~450 MB)
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    url     = _DIV2K_TRAIN_URL if split == "train" else _DIV2K_VALID_URL
    dirname = f"DIV2K_{split}_HR"
    size    = "~3.3 GB" if split == "train" else "~450 MB"

    image_dir = root / dirname
    if image_dir.exists() and any(image_dir.glob("*.png")):
        return image_dir

    archive = root / f"{dirname}.zip"
    if not archive.exists():
        print(f"Downloading DIV2K {split} HR ({size})...")
        urllib.request.urlretrieve(url, archive, _progress)
        print()

    print("Extracting...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    print(f"Done: {image_dir}")
    return image_dir


# ── Blur / noise generation ───────────────────────────────────

def _flickering_noise(img: Image.Image) -> Image.Image:
    """Horizontal banding noise that appears when a camera photographs a display."""
    arr = np.array(img, dtype=np.float32)
    h = arr.shape[0]

    n_bands  = random.randint(2, 8)
    freq     = n_bands / h
    phase    = random.uniform(0, 2 * np.pi)
    strength = random.uniform(20, 60)

    y    = np.arange(h, dtype=np.float32)
    band = strength * np.sin(2 * np.pi * freq * y + phase)  # (H,)
    arr += band[:, np.newaxis, np.newaxis]                   # broadcast → (H, W, C)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_blur_labeled(img: Image.Image, flicker_ratio: float = 0.25) -> tuple:
    """Apply degradation and return (degraded image, noise type string)."""
    if random.random() < flicker_ratio:
        img = _flickering_noise(img)
        label = "flicker"
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
        img = _flickering_noise(img)
        label = "flicker_gaussian"
    return img, label


def _apply_blur(img: Image.Image, flicker_ratio: float = 0.25) -> Image.Image:
    """Apply degradation and return only the result image (for internal Dataset use)."""
    img, _ = _apply_blur_labeled(img, flicker_ratio)
    return img


# ── Dataset ───────────────────────────────────────────────────

class ImageRestorationDataset(Dataset):
    """Uses BSD500 + DIV2K images as sharp originals and applies blur/noise
    at runtime to return (blur, sharp) pairs.

    split='train' -> BSD500 train+val (300 images) + DIV2K train (800 images) = 1,100 images
    split='val'   -> BSD500 test (200 images) + DIV2K valid (100 images) = 300 images
    """

    def __init__(
        self,
        data_root: str = "./data",
        split: str = "train",
        patch_size: int = PATCH_SIZE,
        use_div2k: bool = True,
        flicker_ratio: float = 0.25,
    ):
        self.split = split
        self.patch_size = patch_size
        self.flicker_ratio = flicker_ratio

        paths = []

        # BSD500
        bsd_dir = download_bsd500(data_root)
        bsd_folders = ["train", "val"] if split == "train" else ["test"]
        paths.extend(
            p for folder in bsd_folders
            for p in (bsd_dir / folder).glob("*.jpg")
        )

        # DIV2K
        if use_div2k:
            div2k_split = "train" if split == "train" else "valid"
            div2k_dir = download_div2k(data_root, div2k_split)
            paths.extend(div2k_dir.glob("*.png"))

        self.paths = sorted(paths)
        assert self.paths, "No images found"
        print(f"[{split}] {len(self.paths)} images loaded")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sharp = Image.open(self.paths[idx]).convert("RGB")

        if self.split == "train":
            sharp = self._random_crop(sharp)
            sharp = self._random_augment(sharp)
        else:
            sharp = self._center_crop(sharp)

        blur = _apply_blur(sharp, self.flicker_ratio)

        return (
            self.normalize(self.to_tensor(blur)),
            self.normalize(self.to_tensor(sharp)),
        )

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = img.size
        x = (w - self.patch_size) // 2
        y = (h - self.patch_size) // 2
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _random_augment(self, img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # square patch: 90° rotation does not change size
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k)
        return img


# ── DataLoader helper ────────────────────────────────────────

def get_dataloaders(
    data_root: str = "./data",
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = PATCH_SIZE,
    use_div2k: bool = True,
    flicker_ratio: float = 0.25,
):
    train_ds = ImageRestorationDataset(data_root, split="train", patch_size=patch_size, use_div2k=use_div2k, flicker_ratio=flicker_ratio)
    val_ds   = ImageRestorationDataset(data_root, split="val",   patch_size=patch_size, use_div2k=use_div2k, flicker_ratio=flicker_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("./data", batch_size=2, num_workers=0)

    blur, sharp = next(iter(train_loader))
    print(f"blur  shape : {blur.shape}")
    print(f"sharp shape : {sharp.shape}")
    print(f"blur  range : {blur.min():.2f} ~ {blur.max():.2f}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")
