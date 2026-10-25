import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_generator_splits, build_v1_splits  # noqa: E402
from src.models.classifier import build_classifier  # noqa: E402


def maybe_subset(ds, max_n, seed=0):
    if max_n is None or max_n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=max_n, replace=False)
    return Subset(ds, idx.tolist())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["label"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def counts(items):
    forged = sum(1 for _, _, l in items if l == 1)
    return f"{len(items)} ({forged} forged / {len(items) - forged} authentic)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="generator", choices=["v1", "generator"])
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--out", default="results/baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    raw = build_generator_splits() if args.protocol == "generator" else build_v1_splits()

    train_ds = maybe_subset(ItemsDataset(raw["train"], size=args.size), args.max_train)
    test_sets = {}
    for name in ["test_in_domain", "test_ideogram", "test_gptimage"]:
        if name in raw:
            test_sets[name] = maybe_subset(ItemsDataset(raw[name], size=args.size), args.max_val)

    print("Train:", counts(raw["train"]))
    for name, items in raw.items():
        if name != "train":
            print(f"  {name}: {counts(items)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name, ds in test_sets.items()
    }

    model = build_classifier(args.model, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
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
            y = batch["label"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)

        train_loss = running / len(train_ds)
        row = {"epoch": epoch, "loss": train_loss, "sec": round(time.time() - t0, 1)}
        msg = f"[ep {epoch}] loss={train_loss:.4f}"
        for name, loader in test_loaders.items():
            res = evaluate(model, loader, device)
            row[name] = res
            msg += f"  |  {name}: AUC={res['auc']:.4f} AP={res['ap']:.4f}"
        history.append(row)
        print(msg)

    ckpt = out_dir / f"{args.model}_{args.protocol}.pth"
    torch.save(model.state_dict(), ckpt)
    with open(out_dir / f"{args.model}_{args.protocol}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
