"""Package the 613 GPT-Image-2 forged test documents into a DocTamper-style LMDB.

The official DocTamper evaluator (models/eval_dtd.py) expects, in its working directory:
  <name>/data.mdb, <name>/lock.mdb   keys: num-samples, image-%09d, label-%09d
  qt_table.pk                        dict {quality: LongTensor(64)}
  pks/<name>_<minq>.pk               list of per-image quality records, e.g. [[75], ...]

Usage (on Kaggle, after cloning DocTamper):
  python scripts/make_doctamper_lmdb.py --out-dir DocTamper/models \
      --name AIForgeGPT613 --quality 75 --max-side 1024

Then:
  cd DocTamper/models
  CUDA_VISIBLE_DEVICES=0 python eval_dtd.py --data_root ./ \
      --lmdb_name AIForgeGPT613 --pth pths/dtd_doctamper.pth --minq 75
"""

import argparse
import io
import json
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402


def make_qt_table(qualities):
    import jpegio

    table = {}
    dummy = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8"))
    for q in qualities:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            dummy.save(tmp.name, "JPEG", quality=q)
            jpg = jpegio.read(tmp.name)
        qt = np.asarray(jpg.quant_tables[0], dtype=np.int64).flatten()
        table[int(q)] = torch.LongTensor(qt.tolist())
    return table


def load_pair(image_path, mask_path, max_side, multiple=32):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new, Image.BILINEAR)
        mask = mask.resize(new, Image.NEAREST)
    if multiple > 1:
        w, h = img.size
        pw = (-w) % multiple
        ph = (-h) % multiple
        if pw or ph:
            canvas = Image.new("RGB", (w + pw, h + ph), (255, 255, 255))
            canvas.paste(img, (0, 0))
            mcanvas = Image.new("L", (w + pw, h + ph), 0)
            mcanvas.paste(mask, (0, 0))
            img, mask = canvas, mcanvas
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="DocTamper/models")
    ap.add_argument("--name", default="AIForgeGPT613")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--max-side", type=int, default=1024, help="0 keeps native resolution")
    ap.add_argument("--multiple", type=int, default=32, help="pad H,W to this multiple")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pks").mkdir(exist_ok=True)

    items = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
    if args.max_images:
        items = items[: args.max_images]
    print(f"packing {len(items)} forged GPT-Image-2 documents")

    import lmdb

    env = lmdb.open(str(out / args.name), map_size=8 * 1024**3)
    manifest = {}
    with env.begin(write=True) as txn:
        txn.put(b"num-samples", str(len(items)).encode())
        for i, (img_path, mask_path, _) in enumerate(items):
            img, mask = load_pair(img_path, mask_path, args.max_side, args.multiple)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=args.quality, optimize=True)
            ok, mbuf = cv2.imencode(".png", np.asarray(mask, dtype=np.uint8))
            if not ok:
                raise RuntimeError(f"mask encode failed: {mask_path}")
            txn.put(("image-%09d" % i).encode(), buf.getvalue())
            txn.put(("label-%09d" % i).encode(), mbuf.tobytes())
            manifest[str(i)] = {"image": str(img_path), "mask": str(mask_path)}
            if (i + 1) % 50 == 0 or i + 1 == len(items):
                print(f"  {i + 1}/{len(items)}")
    env.close()

    qualities = list(range(50, 101))
    with open(out / "qt_table.pk", "wb") as f:
        pickle.dump(make_qt_table(qualities), f)
    with open(out / "pks" / f"{args.name}_{args.quality}.pk", "wb") as f:
        pickle.dump([[args.quality] for _ in items], f)
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {out / args.name}, qt_table.pk, pks/{args.name}_{args.quality}.pk")


if __name__ == "__main__":
    main()
