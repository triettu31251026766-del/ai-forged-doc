import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402


@torch.no_grad()
def collect(model, loader, device, max_n=None):
    probs, masks = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).cpu()
        probs.append(prob)
        masks.append(batch["mask"].bool())
        if max_n and sum(p.shape[0] for p in probs) >= max_n:
            break
    return torch.cat(probs, dim=0), torch.cat(masks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--ckpt", default="results/seg/unet_resnet18.pth")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = [it for it in raw[args.split] if it[2] == 1]
    if args.max_n:
        items = items[:args.max_n]
    loader = DataLoader(ItemsDataset(items, size=args.size),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    probs, masks = collect(model, loader, device)
    n = probs.shape[0]
    print(f"Split: {args.split} | forged images: {n}")

    # pixel AUC trung binh theo anh
    paucs = []
    for i in range(n):
        t = masks[i].flatten().numpy()
        if t.max() > 0:
            paucs.append(roc_auc_score(t, probs[i].flatten().numpy()))
    print(f"Pixel AUC (mean): {np.mean(paucs):.4f}")

    print(f"{'thr':>5} {'IoU':>8} {'F1':>8} {'Dice':>8}")
    best = (-1, 0)
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = probs > thr
        inter = (pred & masks).sum().item()
        union = (pred | masks).sum().item()
        tp = inter
        fp = (pred & ~masks).sum().item()
        fn = (~pred & masks).sum().item()
        iou = inter / max(union, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        dice_num = 2.0 * inter
        dice_den = pred.sum().item() + masks.sum().item()
        dice = dice_num / max(dice_den, 1)
        print(f"{thr:5.2f} {iou:8.4f} {f1:8.4f} {dice:8.4f}")
        if iou > best[1]:
            best = (thr, iou)
    print(f"BEST: thr={best[0]:.2f} IoU={best[1]:.4f}")


if __name__ == "__main__":
    main()
