import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augment import build_train_augment  # noqa: E402
from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

TESTS = ["test_in_domain", "test_ideogram", "test_gptimage"]


def maybe_subset(ds, max_n, seed=0):
    if max_n is None or max_n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=max_n, replace=False)
    return Subset(ds, idx.tolist())


def forged_dataset(items, size, freq=False):
    return ItemsDataset([it for it in items if it[2] == 1], size=size, freq=freq)


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return 1 - (num / den).mean()


@torch.no_grad()
def eval_pixel(model, loader, device, thr=0.5):
    model.eval()
    inter = union = tp = fp = fn = 0
    dice_sum = 0.0
    n = 0
    paucs = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        tgt = batch["mask"].to(device).bool()
        prob = torch.sigmoid(model(x))
        pred = prob > thr
        inter += (pred & tgt).sum().item()
        union += (pred | tgt).sum().item()
        tp += (pred & tgt).sum().item()
        fp += (pred & ~tgt).sum().item()
        fn += (~pred & tgt).sum().item()
        for i in range(pred.shape[0]):
            p, t = pred[i], tgt[i]
            dice_sum += (2 * (p & t).sum().item()) / (p.sum().item() + t.sum().item() + 1e-6)
            n += 1
            tn = t.flatten().cpu().numpy()
            if tn.max() > 0:
                pn = prob[i].flatten().cpu().numpy()
                paucs.append(roc_auc_score(tn, pn))
    return {
        "iou": inter / max(union, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "dice": dice_sum / max(n, 1),
        "pixel_auc": float(np.mean(paucs)) if paucs else 0.0,
        "n": n,
    }


@torch.no_grad()
def eval_image_auc(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x))
        score = prob.flatten(1).max(dim=1).values
        scores.append(score.cpu().numpy())
        labels.append(batch["label"].numpy())
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    return {"auc": float(roc_auc_score(y, s)), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="p1", choices=["p1", "p2", "p3"])
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--freq", action="store_true", help="dung 6 kenh: RGB + tan so/nhieu")
    ap.add_argument("--out", default="results/seg")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    raw = build_protocol_splits(args.protocol)
    test_names = [k for k in raw if k != "train"]

    train_aug = build_train_augment() if args.augment else None
    train_ds = maybe_subset(ItemsDataset(raw["train"], size=args.size, augment=train_aug, freq=args.freq), args.max_train)
    pixel_loaders = {
        name: DataLoader(maybe_subset(forged_dataset(raw[name], args.size, args.freq), args.max_val),
                         batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name in test_names
    }
    auc_loaders = {
        name: DataLoader(maybe_subset(ItemsDataset(raw[name], args.size, freq=args.freq), args.max_val),
                         batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name in test_names
    }

    print(f"Train: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    in_channels = 6 if args.freq else 3
    model = build_segmentation(args.encoder, in_channels=in_channels).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = batch["image"].to(device, non_blocking=True)
            tgt = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, tgt) + dice_loss(logits, tgt)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)

        row = {"epoch": epoch, "loss": running / len(train_ds), "sec": round(time.time() - t0, 1)}
        msg = f"[ep {epoch}] loss={row['loss']:.4f}"
        for name in test_names:
            px = eval_pixel(model, pixel_loaders[name], device)
            auc = eval_image_auc(model, auc_loaders[name], device)
            row[name] = {"pixel": px, "image_auc": auc}
            msg += (f"  |  {name}: IoU={px['iou']:.4f} F1={px['f1']:.4f} "
                    f"Dice={px['dice']:.4f} ImgAUC={auc['auc']:.4f}")
        history.append(row)
        print(msg)

    ckpt = out_dir / f"unet_{args.encoder}.pth"
    torch.save(model.state_dict(), ckpt)
    with open(out_dir / f"unet_{args.encoder}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
