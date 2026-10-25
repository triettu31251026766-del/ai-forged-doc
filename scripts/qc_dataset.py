import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def list_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description="QC nhanh anh + mask cua AIForge-Doc")
    ap.add_argument("--root", default="data/raw/AIForge-Doc-v1")
    ap.add_argument("--split", default="TrainingSet", choices=["TrainingSet", "TestingSet"])
    ap.add_argument("--n", type=int, default=8, help="so mau kiem tra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/qc")
    args = ap.parse_args()

    root = Path(args.root) / args.split
    img_dir = root / "images"
    msk_dir = root / "masks"
    aut_dir = root / "authentic"

    images = list_files(img_dir)
    masks = list_files(msk_dir)
    auth = list_files(aut_dir) if aut_dir.exists() else []

    print("=" * 60)
    print(f"Split       : {args.split}")
    print(f"Forged imgs : {len(images)}")
    print(f"Masks       : {len(masks)}")
    print(f"Authentic   : {len(auth)}")

    img_stems = {p.stem for p in images}
    msk_stems = {p.stem for p in masks}
    missing_masks = img_stems - msk_stems
    extra_masks = msk_stems - img_stems
    print(f"Img without mask: {len(missing_masks)}")
    print(f"Mask without img: {len(extra_masks)}")
    print("=" * 60)

    random.seed(args.seed)
    sample = random.sample(images, min(args.n, len(images)))

    os.makedirs(args.out, exist_ok=True)
    fractions = []
    problems = 0

    for img_path in sample:
        msk_path = msk_dir / (img_path.stem + ".png")
        if not msk_path.exists():
            # try same name
            msk_path = msk_dir / img_path.name
        if not msk_path.exists():
            print(f"[MISS] mask for {img_path.name}")
            problems += 1
            continue

        im = Image.open(img_path)
        mk = Image.open(msk_path).convert("L")
        if im.size != mk.size:
            print(f"[SIZE MISMATCH] {img_path.name}: img={im.size} mask={mk.size}")
            problems += 1
            continue

        arr = np.array(mk)
        vals = np.unique(arr)
        frac = float((arr > 127).mean())
        fractions.append(frac)
        flag = ""
        if frac == 0:
            flag = "  <-- mask rong (khong co vung sua?)"
            problems += 1
        if not set(vals.tolist()).issubset({0, 255}):
            flag += f"  <-- gia tri la: {vals[:6]}"
        print(f"{img_path.name:>16}  size={im.size}  tampered={frac*100:.2f}%{flag}")

    if fractions:
        print("-" * 60)
        print(f"tampered %: min={min(fractions)*100:.2f}  "
              f"median={np.median(fractions)*100:.2f}  max={max(fractions)*100:.2f}")
    print(f"Problems in sample: {problems}/{len(sample)}")


if __name__ == "__main__":
    main()
