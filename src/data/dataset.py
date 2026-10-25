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

    def __init__(self, items, size=512, return_mask=True):
        self.items = list(items)
        self.size = size
        self.return_mask = return_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", img.size, 0)

        img, mask = letterbox(img, mask, self.size)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        binary_mask = (np.asarray(mask) > 127).astype(np.float32)
        mask_t = torch.from_numpy(binary_mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask_t if self.return_mask else torch.zeros(1, 1, 1),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
        }


class DocumentForgeryDataset(_BaseForgeryDataset):
    def __init__(self, forged_dir, mask_dir, authentic_dir=None, size=512, return_mask=True):
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
        super().__init__(items, size=size, return_mask=return_mask)


class ItemsDataset(_BaseForgeryDataset):
    """Dung khi da co danh sach item (vd: chia theo generator)."""

    def __init__(self, items, size=512, return_mask=True):
        super().__init__(items, size=size, return_mask=return_mask)
