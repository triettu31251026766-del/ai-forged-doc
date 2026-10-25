import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def list_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/AIForge-Doc-v1")
    ap.add_argument("--split", default="TestingSet")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/qc")
    args = ap.parse_args()

    root = Path(args.root) / args.split
    images = list_files(root / "images")
    masks = root / "masks"

    random.seed(args.seed)
    sample = random.sample(images, min(args.n, len(images)))

    Path(args.out).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(sample), 3, figsize=(11, 3.4 * len(sample)))
    if len(sample) == 1:
        axes = np.array([axes])

    for i, img_path in enumerate(sample):
        msk_path = masks / (img_path.stem + ".png")
        if not msk_path.exists():
            msk_path = masks / img_path.name

        im = Image.open(img_path).convert("RGB")
        mk = Image.open(msk_path).convert("L").resize(im.size, Image.NEAREST)

        # resize for display (max height 400)
        scale = min(1.0, 400 / im.height)
        if scale < 1.0:
            new_size = (int(im.width * scale), int(im.height * scale))
            im_disp = im.resize(new_size, Image.LANCZOS)
            mk_disp = mk.resize(new_size, Image.NEAREST)
        else:
            im_disp, mk_disp = im, mk

        arr = np.array(im_disp).astype(np.float32)
        m = (np.array(mk_disp) > 127)
        overlay = arr.copy()
        overlay[m] = overlay[m] * 0.4 + np.array([255, 0, 0]) * 0.6
        overlay = overlay.astype(np.uint8)

        frac = m.mean() * 100
        axes[i, 0].imshow(im_disp); axes[i, 0].set_title(f"Forged {img_path.stem}")
        axes[i, 1].imshow(mk_disp, cmap="gray"); axes[i, 1].set_title("Mask (white=edited)")
        axes[i, 2].imshow(overlay); axes[i, 2].set_title(f"Overlay {frac:.2f}%")
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    out_path = Path(args.out) / f"{args.split}_qc_grid.png"
    plt.savefig(out_path, dpi=130)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
