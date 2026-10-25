import argparse
import sys
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def files(folder):
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/AIForge-Doc-v1")
    args = ap.parse_args()

    total = 0
    bad = 0
    for split in ["TrainingSet", "TestingSet"]:
        for sub in ["images", "masks", "authentic"]:
            folder = Path(args.root) / split / sub
            if not folder.exists():
                continue
            fs = files(folder)
            errs = 0
            for p in fs:
                total += 1
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception as e:
                    errs += 1
                    bad += 1
                    if errs <= 3:
                        print(f"BAD: {p} -> {e}")
            print(f"{split}/{sub}: {len(fs)} files, {errs} bad")
    print("=" * 50)
    print(f"TOTAL: {total} files, {bad} bad")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
