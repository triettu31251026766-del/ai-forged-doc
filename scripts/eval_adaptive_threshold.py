import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402


def new_stats():
    return {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0, "dice": 0.0, "n": 0}


def update(st, pred, tgt):
    inter = (pred & tgt).sum().item()
    st["inter"] += inter
    st["union"] += (pred | tgt).sum().item()
    st["tp"] += inter
    st["fp"] += (pred & ~tgt).sum().item()
    st["fn"] += (~pred & tgt).sum().item()
    st["dice"] += (2 * inter) / (pred.sum().item() + tgt.sum().item() + 1e-6)
    st["n"] += 1


def summarize(st):
    return {
        "iou": st["inter"] / max(st["union"], 1),
        "f1": 2 * st["tp"] / max(2 * st["tp"] + st["fp"] + st["fn"], 1),
        "dice": st["dice"] / max(st["n"], 1),
        "n": st["n"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None, help="json co T")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--percentiles", default="0.5,1,2,5")
    ap.add_argument("--out", default="results/adaptive")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1.0
    if args.json:
        d = json.loads((ROOT / args.json).read_text(encoding="utf-8"))
        T = float(d.get("T", 1.0))
    pcts = [float(x) for x in args.percentiles.split(",")]

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = [it for it in raw[args.split] if it[2] == 1]
    if args.max_test:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(items), size=min(args.max_test, len(items)), replace=False)
        items = [items[i] for i in idx]
    loader = DataLoader(ItemsDataset(items, size=args.size), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    keys = ["fixed"] + [f"p{p}" for p in pcts]
    stats = {k: new_stats() for k in keys}

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            prob = torch.sigmoid(model(x) / T)
            tgt = batch["mask"].to(device).bool()

            for i in range(prob.shape[0]):
                p_i = prob[i, 0]
                t_i = tgt[i, 0]
                update(stats["fixed"], p_i > 0.5, t_i)
                for p in pcts:
                    q = torch.quantile(p_i.flatten(), 1.0 - p / 100.0)
                    update(stats[f"p{p}"], p_i > q, t_i)

    print(f"ckpt={args.ckpt} | split={args.split} | T={T:.2f} | n={stats['fixed']['n']}")
    print(f"{'method':>8} {'IoU':>8} {'F1':>8} {'Dice':>8}")
    out = {"ckpt": args.ckpt, "split": args.split, "T": T, "results": {}}
    for k in keys:
        s = summarize(stats[k])
        out["results"][k] = s
        print(f"{k:>8} {s['iou']:8.4f} {s['f1']:8.4f} {s['dice']:8.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt).stem + "__" + args.split + ".json"
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
