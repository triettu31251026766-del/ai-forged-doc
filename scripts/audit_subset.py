import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import (  # noqa: E402
    V1, V2, GEMINI, IDEOGRAM, build_protocol_splits, forged_items, load_meta,
)
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_seg import eval_pixel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["results/fewshot/k100.pth"])
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--split", default="gpt", choices=["gpt", "gemini", "ideogram"])
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    raw = build_protocol_splits("p1")
    if args.split == "gpt":
        items = raw["test_gptimage"]
    elif args.split == "gemini":
        items = raw["test_in_domain"]
    else:
        meta1 = load_meta(V1 / "metadata.jsonl")
        items = [(img, msk, 1) for img, msk, _ in forged_items(meta1, V1, tool=IDEOGRAM, split="testing")]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(items), size=60, replace=False)
    sub = [items[i] for i in idx]
    sub_f = [it for it in sub if it[2] == 1]
    full_f = [it for it in items if it[2] == 1]

    meta = load_meta(V1 / "metadata.jsonl")
    src = {str(r["new_id"]): r.get("source_dataset", "?") for r in meta}

    def dist(lst):
        return Counter(src.get(Path(it[0]).stem, "?") for it in lst)

    print(f"subset total={len(sub)} forged={len(sub_f)} | full forged={len(full_f)}")
    print("subset sources:", dict(dist(sub_f)))
    print("full   sources:", dict(dist(full_f)))

    model = build_segmentation("resnet18").to(device)
    for ck in args.ckpts:
        model.load_state_dict(torch.load(ROOT / ck, map_location=device))
        model.eval()
        sets = [("FULL", full_f)] if args.full_only else [("SUBSET", sub_f), ("FULL", full_f)]
        for name, lst in sets:
            ds = ItemsDataset(lst, size=args.size)
            px = eval_pixel(model, DataLoader(ds, batch_size=8), device)
            print(f"{ck:34s} {name:7s}: IoU={px['iou']:.4f} F1={px['f1']:.4f} "
                  f"PixAUC={px['pixel_auc']:.4f} n={px['n']}")


if __name__ == "__main__":
    main()
