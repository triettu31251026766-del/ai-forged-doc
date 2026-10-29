import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_fewshot_splits, build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_seg import dice_loss  # noqa: E402


def split_cal(items, frac, seed):
    rng = random.Random(seed)
    forged = [it for it in items if it[2] == 1]
    auth = [it for it in items if it[2] == 0]
    rng.shuffle(forged)
    rng.shuffle(auth)
    nf = max(1, int(round(len(forged) * frac)))
    na = max(1, int(round(len(auth) * frac)))
    cal = forged[:nf] + auth[:na]
    train = forged[nf:] + auth[na:]
    return train, cal


@torch.no_grad()
def score_items(model, items, device, size, batch_size, mode):
    loader = DataLoader(ItemsDataset(items, size=size), batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        tgt = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        if mode == "hard":
            s = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(dim=(1, 2, 3))
        else:
            p = torch.sigmoid(logits)
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
            s = ent.mean(dim=(1, 2, 3))
        scores.append(s.cpu())
    return torch.cat(scores, dim=0)


def select_replay(model, n, method, device, size, batch_size, seed):
    old = build_protocol_splits("p1")["train"]
    forged = [it for it in old if it[2] == 1]
    auth = [it for it in old if it[2] == 0]
    rng = random.Random(seed)
    n = min(n, len(forged), len(auth))

    if method == "random":
        return rng.sample(forged, n) + rng.sample(auth, n)

    sf = score_items(model, forged, device, size, batch_size, method)
    idx_f = torch.argsort(sf, descending=True)[:n].tolist()
    sa = score_items(model, auth, device, size, batch_size, method)
    idx_a = torch.argsort(sa, descending=True)[:n].tolist()
    return [forged[i] for i in idx_f] + [auth[i] for i in idx_a]


@torch.no_grad()
def collect_calibration(model, loader, device):
    logits_all, targets_all = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        targets_all.append(batch["mask"])
    return torch.cat(logits_all, dim=0), torch.cat(targets_all, dim=0)


def fit_temperature(logits, targets):
    best_T, best_loss = 1.0, float("inf")
    for T in np.linspace(0.5, 5.0, 46):
        loss = F.binary_cross_entropy_with_logits(logits / float(T), targets).item()
        if loss < best_loss:
            best_loss, best_T = loss, float(T)
    return best_T


def fit_threshold(probs, targets):
    best_thr, best_dice = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = probs > float(thr)
        inter = (pred & targets).sum().item()
        denom = pred.sum().item() + targets.sum().item()
        dice = (2.0 * inter / denom) if denom > 0 else 0.0
        if dice > best_dice:
            best_dice, best_thr = dice, float(thr)
    return best_thr, best_dice


@torch.no_grad()
def eval_params(model, loader, device, T, thr):
    model.eval()
    inter = union = tp = fp = fn = 0
    dice_sum = 0.0
    n = 0
    paucs = []
    paps = []
    scores, labels = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        tgt = batch["mask"].to(device).bool()
        prob = torch.sigmoid(model(x) / T)
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
                paps.append(average_precision_score(tn, pn))

        score = prob.flatten(1).max(dim=1).values.cpu().numpy()
        scores.append(score)
        labels.append(batch["label"].numpy())

    y = np.concatenate(labels)
    s = np.concatenate(scores)
    try:
        img_auc = float(roc_auc_score(y, s))
        img_ap = float(average_precision_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        recall_at_1fpr = float(np.interp(0.01, fpr, tpr))
    except ValueError:
        img_auc = img_ap = recall_at_1fpr = float("nan")
    return {
        "iou": inter / max(union, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "dice": dice_sum / max(n, 1),
        "pixel_auc": float(np.mean(paucs)) if paucs else 0.0,
        "pixel_ap": float(np.mean(paps)) if paps else 0.0,
        "image_auc": img_auc,
        "image_ap": img_ap,
        "recall_at_1fpr": recall_at_1fpr,
        "n": int(len(y)),
    }


def maybe_subset_items(items, max_n, seed):
    if not max_n or max_n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=max_n, replace=False)
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--init", default="results/seg/unet_resnet18.pth")
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--cal-method", default="both", choices=["temp", "threshold", "both"])
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--replay-select", default="random", choices=["random", "hard", "uncertainty"])
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/calib")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    raw = build_fewshot_splits(args.k, seed=args.seed)
    train_items, cal_items = split_cal(raw["train"], args.cal_frac, args.seed)

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.init, map_location=device))

    if args.replay > 0:
        rep = select_replay(model, args.replay, args.replay_select, device, args.size, args.batch_size, args.seed)
        train_items = train_items + rep
        print(f"Replay {args.replay} ({args.replay_select}) -> total train {len(train_items)}")

    train_ds = ItemsDataset(train_items, size=args.size)
    cal_ds = ItemsDataset(cal_items, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_items = {}
    for name in ["test_in_domain", "test_ideogram", "test_gptimage"]:
        test_items[name] = maybe_subset_items(raw[name], args.max_test, args.seed)
    test_loaders = {n: DataLoader(ItemsDataset(it, size=args.size), batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
                    for n, it in test_items.items()}

    print(f"k={args.k} train={len(train_ds)} cal={len(cal_ds)} | cal-method={args.cal_method}")

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

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
        print(f"  ep {epoch}: loss={running / len(train_ds):.4f}")

    logits_cal, targets_cal = collect_calibration(model, cal_loader, device)
    T = 1.0
    if args.cal_method in ("temp", "both"):
        T = fit_temperature(logits_cal, targets_cal)
    probs_cal = torch.sigmoid(logits_cal / T)
    thr = 0.5
    if args.cal_method in ("threshold", "both"):
        thr, _ = fit_threshold(probs_cal, targets_cal.bool())

    fixed = {name: eval_params(model, loader, device, 1.0, 0.5) for name, loader in test_loaders.items()}
    cal = {name: eval_params(model, loader, device, T, thr) for name, loader in test_loaders.items()}

    tag = args.tag or (f"k{args.k}_s{args.seed}_{args.cal_method}"
                       + (f"_rp{args.replay}{args.replay_select}" if args.replay else ""))

    line = (f"RESULT tag={tag} T={T:.2f} thr={thr:.2f} | "
            f"GPT fixed={fixed['test_gptimage']['iou']:.4f} cal={cal['test_gptimage']['iou']:.4f} | "
            f"Gemini fixed={fixed['test_in_domain']['iou']:.4f} cal={cal['test_in_domain']['iou']:.4f} | "
            f"Ideogram fixed={fixed['test_ideogram']['iou']:.4f} cal={cal['test_ideogram']['iou']:.4f}")
    print(line)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "T": T, "threshold": thr,
               "fixed": fixed, "calibrated": cal, "sec": round(time.time() - t0, 1)}
    torch.save(model.state_dict(), out_dir / f"{tag}.pth")
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("Saved:", out_dir / f"{tag}.json")


if __name__ == "__main__":
    main()
