import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_fewshot_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_seg import dice_loss, eval_image_auc, eval_pixel  # noqa: E402


def maybe_subset(ds, max_n, seed=0):
    if max_n is None or max_n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=max_n, replace=False)
    return Subset(ds, idx.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="so mau moi generator (GPT) de fine-tune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--init", default="results/seg/unet_resnet18.pth")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--out", default="results/fewshot")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | k={args.k} | scratch={args.scratch}")

    raw = build_fewshot_splits(args.k)
    train_ds = ItemsDataset(raw["train"], size=args.size)
    test_names = [k for k in raw if k != "train"]

    bs = min(args.batch_size, max(1, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    pixel_loaders, auc_loaders = {}, {}
    for name in test_names:
        items = raw[name]
        forged = [it for it in items if it[2] == 1]
        pixel_loaders[name] = DataLoader(
            maybe_subset(ItemsDataset(forged, size=args.size), args.max_val),
            batch_size=args.batch_size, shuffle=False, num_workers=0)
        auc_loaders[name] = DataLoader(
            maybe_subset(ItemsDataset(items, size=args.size), args.max_val),
            batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Fine-tune samples: {len(train_ds)} (k={args.k} forged + k authentic)")

    model = build_segmentation("resnet18").to(device)
    if not args.scratch:
        ckpt = ROOT / args.init
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Loaded base:", ckpt)
    else:
        print("Train from scratch")

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"ft {epoch}/{args.epochs}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            tgt = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, tgt) + dice_loss(logits, tgt)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        print(f"  ep {epoch}: loss={running/len(train_ds):.4f}")

    results = {"k": args.k, "scratch": args.scratch, "epochs": args.epochs,
               "lr": args.lr, "sec": round(time.time() - t0, 1), "tests": {}}
    for name in test_names:
        px = eval_pixel(model, pixel_loaders[name], device)
        auc = eval_image_auc(model, auc_loaders[name], device)
        results["tests"][name] = {"pixel": px, "image_auc": auc}
        print(f"{name}: IoU={px['iou']:.4f} F1={px['f1']:.4f} Dice={px['dice']:.4f} "
              f"PixAUC={px['pixel_auc']:.4f} ImgAUC={auc['auc']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"k{args.k}" + ("_scratch" if args.scratch else "")
    torch.save(model.state_dict(), out_dir / f"{tag}.pth")
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir / f"{tag}.json", "and", out_dir / f"{tag}.pth")


if __name__ == "__main__":
    main()
