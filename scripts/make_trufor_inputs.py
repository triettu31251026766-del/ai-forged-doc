import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "AIForge-Doc-v2" / "TestingSet" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="results/eval_lists/gpt150.txt")
    ap.add_argument("--out", default="data/processed/trufor_inputs")
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    stems = [s.strip() for s in (ROOT / args.list).read_text(encoding="utf-8").splitlines() if s.strip()]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    for s in stems:
        img = cv2.imread(str(SRC / f"{s}.png"))
        if img is None:
            print("MISS", s)
            continue
        h, w = img.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{s}.png"), img)
        meta[s] = {"orig_h": h, "orig_w": w, "new_h": img.shape[0], "new_w": img.shape[1]}
    meta_path = out.parent / f"{out.name}_sizes.json"
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {len(meta)} images to {out} (max side {args.max_side})")


if __name__ == "__main__":
    main()
