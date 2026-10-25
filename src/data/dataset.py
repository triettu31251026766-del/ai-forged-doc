from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])


def freq_channels(raw):
    """raw: HxWx3 float [0,1] -> 3 kenh tan so/nhieu (z-score)."""
    import cv2

    gray = raw.mean(axis=2).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec)).astype(np.float32)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    return np.stack([z(hp), z(lap), z(mag)], axis=0)


def letterbox(img, mask, size):
    """Resize giu ti le roi pad thanh size x size (anh + mask cung phep bien doi)."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    mask_r = mask.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(img_r, (left, top))
    mask_canvas.paste(mask_r, (left, top))
    return canvas, mask_canvas


class _BaseForgeryDataset(Dataset):
    """items = [(image_path, mask_path_or_None, label)]. label: 0=that, 1=gia."""

    def __init__(self, items, size=512, return_mask=True, augment=None, freq=False):
        self.items = list(items)
        self.size = size
        self.return_mask = return_mask
        self.augment = augment
        self.freq = freq

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", img.size, 0)

        if self.augment is not None:
            out = self.augment(image=np.asarray(img), mask=np.asarray(mask))
            img = Image.fromarray(out["image"])
            mask = Image.fromarray(out["mask"])

        img, mask = letterbox(img, mask, self.size)

        raw = np.asarray(img, dtype=np.float32) / 255.0
        rgb = (raw - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1))
        if self.freq:
            f = freq_channels(raw)
            image = torch.from_numpy(np.concatenate([rgb, f], axis=0)).contiguous()
        else:
            image = torch.from_numpy(rgb).contiguous()

        binary_mask = (np.asarray(mask) > 127).astype(np.float32)
        mask_t = torch.from_numpy(binary_mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask_t if self.return_mask else torch.zeros(1, 1, 1),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
        }


class DocumentForgeryDataset(_BaseForgeryDataset):
    def __init__(self, forged_dir, mask_dir, authentic_dir=None, size=512, return_mask=True, augment=None, freq=False):
        items = []
        for img_path in list_images(forged_dir):
            msk = Path(mask_dir) / (img_path.stem + ".png")
            if not msk.exists():
                msk = Path(mask_dir) / img_path.name
            if msk.exists():
                items.append((img_path, msk, 1))
        if authentic_dir:
            for img_path in list_images(authentic_dir):
                items.append((img_path, None, 0))
        super().__init__(items, size=size, return_mask=return_mask, augment=augment, freq=freq)


class ItemsDataset(_BaseForgeryDataset):
    """Dung khi da co danh sach item (vd: chia theo generator)."""

    def __init__(self, items, size=512, return_mask=True, augment=None, freq=False):
        super().__init__(items, size=size, return_mask=return_mask, augment=augment, freq=freq)
