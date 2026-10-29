import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import V1, V2, build_protocol_splits, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

BUCKETS = [(0.0, 0.005, "<0.5%"), (0.005, 0.01, "0.5-1%"),
           (0.01, 0.02, "1-2%"), (0.02, 1.01, ">2%")]


def split_folder(path):
    return "TrainingSet" if "TrainingSet" in str(path) else "TestingSet"


def meta_key(rec):
    sd = "TrainingSet" if rec.get("split") == "training" else "TestingSet"
    return sd, str(rec["new_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1.0
    if args.json:
        T = float(json.loads((ROOT / args.json).read_text(encoding="utf-8")).get("T", 1.0))

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = [it for it in raw[args.split] if it[2] == 1]

    meta_by_key = {}
    for folder in [V1, V2]:
        p = folder / "metadata.jsonl"
        if p.exists():
            for rec in load_meta(p):
                if "new_id" in rec:
                    meta_by_key[meta_key(rec)] = rec

    loader = DataLoader(ItemsDataset(items, size=args.size), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="error analysis"):
            x = batch["image"].to(device)
            prob = torch.sigmoid(model(x) / T)
            tgt = batch["mask"].to(device).bool()
            for i in range(prob.shape[0]):
                path = batch["path"][i]
                rec = meta_by_key.get((split_folder(path), Path(path).stem), {})
                src = rec.get("source_dataset", "?")
                field = rec.get("field_name", "?")
                t = tgt[i, 0]
                p = prob[i, 0] > args.thr
                inter = (p & t).sum().item()
                union = (p | t).sum().item()
                rows.append({"path": path, "source": src, "field": field,
                             "area": float(t.float().mean().item()),
                             "iou": inter / max(union, 1)})

    def aggregate(key, sort=True):
        d = defaultdict(list)
        for r in rows:
            d[r[key]].append(r["iou"])
        return {k: {"n": len(v), "iou": float(np.mean(v))} for k, v in
                (sorted(d.items()) if sort else d.items())}

    by_source = aggregate("source")
    by_field = aggregate("field")

    by_bucket = {}
    for lo, hi, name in BUCKETS:
        vals = [r["iou"] for r in rows if lo <= r["area"] < hi]
        if vals:
            by_bucket[name] = {"n": len(vals), "iou": float(np.mean(vals))}

    print(f"ckpt={args.ckpt} | split={args.split} | T={T:.2f} | n={len(rows)}")
    print("\n-- By source --")
    for k, v in by_source.items():
        print(f"{k:>14} n={v['n']:>4}  mean IoU={v['iou']:.4f}")
    print("\n-- By tampered area --")
    for k, v in by_bucket.items():
        print(f"{k:>14} n={v['n']:>4}  mean IoU={v['iou']:.4f}")
    print("\n-- By field --")
    for k, v in list(by_field.items())[:12]:
        print(f"{k:>28} n={v['n']:>4}  mean IoU={v['iou']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt).stem + "__" + args.split + "__error.json"
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "T": T, "rows": rows,
                   "by_source": by_source, "by_area": by_bucket,
                   "by_field": by_field}, f, indent=2)
    print("\nSaved:", out_dir / name)


if __name__ == "__main__":
    main()
