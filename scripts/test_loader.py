import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import DocumentForgeryDataset  # noqa: E402

V1 = ROOT / "data" / "raw" / "AIForge-Doc-v1"
V2 = ROOT / "data" / "raw" / "AIForge-Doc-v2"


def build(name, base, split, size=512):
    forged = base / split / "images"
    masks = base / split / "masks"
    # v2 khong co authentic -> dung authentic cua v1 (cung anh goc)
    authentic = base / split / "authentic"
    if not authentic.exists():
        authentic = V1 / split / "authentic"
    ds = DocumentForgeryDataset(forged, masks, authentic, size=size)
    print(f"{name}/{split}: {len(ds)} samples "
          f"(forged={sum(1 for _, _, l in ds.items if l == 1)}, "
          f"authentic={sum(1 for _, _, l in ds.items if l == 0)})")
    return ds


def main():
    for name, base in [("v1", V1), ("v2", V2)]:
        for split in ["TrainingSet", "TestingSet"]:
            ds = build(name, base, split)
            batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
            print("   image:", tuple(batch["image"].shape),
                  "mask:", tuple(batch["mask"].shape),
                  "label:", batch["label"].tolist())
            forged = batch["label"] == 1
            if forged.any():
                frac = batch["mask"][forged].mean().item()
                print(f"   mask frac (forged): {frac*100:.2f}%")
            print()


if __name__ == "__main__":
    main()
