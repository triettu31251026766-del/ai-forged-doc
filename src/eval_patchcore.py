import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import authentic_items, build_protocol_splits, V1  # noqa: E402
from src.models.patchcore import PatchFeatureExtractor, features_to_patches, patch_distances  # noqa: E402


def build_bank(model, loader, device, patches_per_batch, bank_max, seed=0):
    bank = []
    for batch in tqdm(loader, desc="memory bank"):
        x = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            feats = model(x)
        patches, _ = features_to_patches(feats)
        n = min(patches_per_batch, patches.shape[0])
        idx = torch.randperm(patches.shape[0], device=patches.device)[:n]
        bank.append(patches[idx].cpu())
    bank = torch.cat(bank, dim=0)
    if bank_max and bank.shape[0] > bank_max:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(bank.shape[0], generator=g)[:bank_max]
        bank = bank[idx]
    return bank.to(device)


@torch.no_grad()
def calibrate_threshold(model, loader, bank, device, out_size, percentile):
    scores = []
    for batch in tqdm(loader, desc="calibrate"):
        x = batch["image"].to(device, non_blocking=True)
        smap = score_map_grid(model, x, bank)
        scores.append(smap.flatten().cpu())
    s = torch.cat(scores).numpy()
    thr = float(np.percentile(s, percentile))
    return thr, s


@torch.no_grad()
def score_map_grid(model, x, bank):
    feats = model(x)
    patches, (h, w) = features_to_patches(feats)
    dist = patch_distances(patches, bank).reshape(x.shape[0], 1, h, w)
    return dist


@torch.no_grad()
def score_map_image(model, x, bank, out_size):
    dist = score_map_grid(model, x, bank)
    return F.interpolate(dist, size=(out_size, out_size), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, bank, device, thr, out_size):
    model.eval()
    scores, labels = [], []
    inter = union = tp = fp = fn = 0
    dice_sum = 0.0
    n_forged = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        smap = score_map_image(model, x, bank, out_size)  # [B,1,S,S]
        score = smap.flatten(1).max(dim=1).values
        scores.append(score.cpu().numpy())
        labels.append(batch["label"].numpy())

        forged = batch["label"] == 1
        if forged.any():
            pred = smap[forged] > thr
            tgt = batch["mask"][forged].to(device) > 0.5
            inter += (pred & tgt).sum().item()
            union += (pred | tgt).sum().item()
            tp += (pred & tgt).sum().item()
            fp += (pred & ~tgt).sum().item()
            fn += (~pred & tgt).sum().item()
            for i in range(pred.shape[0]):
                p, t = pred[i], tgt[i]
                dice_sum += (2 * (p & t).sum().item()) / (p.sum().item() + t.sum().item() + 1e-6)
                n_forged += 1

    y = np.concatenate(labels)
    s = np.concatenate(scores)
    return {
        "image_auc": float(roc_auc_score(y, s)),
        "iou": inter / max(union, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "dice": dice_sum / max(n_forged, 1),
        "n": int(len(y)),
        "n_forged": n_forged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patches-per-image", type=int, default=16)
    ap.add_argument("--bank-max", type=int, default=60000)
    ap.add_argument("--threshold-percentile", type=float, default=99.5)
    ap.add_argument("--max-bank", type=int, default=None, help="gioi han so anh that de build bank")
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--out", default="results/patchcore")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| size:", args.size)

    model = PatchFeatureExtractor(pretrained=True).to(device).eval()

    auth_train = authentic_items(V1, "training")
    if args.max_bank:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(auth_train), size=min(args.max_bank, len(auth_train)), replace=False)
        auth_train = [auth_train[i] for i in idx]
    bank_ds = ItemsDataset(auth_train, size=args.size)
    bank_loader = DataLoader(bank_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    patches_per_batch = args.patches_per_image * args.batch_size
    t0 = time.time()
    bank = build_bank(model, bank_loader, device, patches_per_batch, args.bank_max)
    print(f"Bank: {bank.shape[0]} patches x {bank.shape[1]} dims ({time.time()-t0:.1f}s)")

    # calibrate tren anh that cua test
    auth_test = authentic_items(V1, "testing")
    if args.max_test:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(auth_test), size=min(args.max_test, len(auth_test)), replace=False)
        auth_test = [auth_test[i] for i in idx]
    cal_ds = ItemsDataset(auth_test, size=args.size)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    thr, _ = calibrate_threshold(model, cal_loader, bank, device, args.size, args.threshold_percentile)
    print(f"Threshold (p{args.threshold_percentile}): {thr:.4f}")

    raw = build_protocol_splits("p1")
    results = {}
    for name in ["test_in_domain", "test_ideogram", "test_gptimage"]:
        items = raw[name]
        if args.max_test:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(items), size=min(args.max_test, len(items)), replace=False)
            items = [items[i] for i in idx]
        ds = ItemsDataset(items, size=args.size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        res = evaluate(model, loader, bank, device, thr, args.size)
        results[name] = res
        print(f"{name}: IoU={res['iou']:.4f} F1={res['f1']:.4f} "
              f"Dice={res['dice']:.4f} ImgAUC={res['image_auc']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "patchcore_results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "threshold": thr, "results": results}, f, indent=2)
    print("Saved:", out_dir / "patchcore_results.json")


if __name__ == "__main__":
    main()
