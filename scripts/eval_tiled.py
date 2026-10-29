import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def positions(total, tile, stride):
    if total <= tile:
        return [0]
    xs = list(range(0, total - tile + 1, stride))
    if xs[-1] != total - tile:
        xs.append(total - tile)
    return xs


@torch.no_grad()
def tiled_prob(model, img, tile, stride, device, T):
    H, W = img.shape[:2]
    sump = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for y in positions(H, tile, stride):
        for x in positions(W, tile, stride):
            h = min(tile, H - y)
            w = min(tile, W - x)
            crop = img[y:y + h, x:x + w]
            pad = np.zeros((tile, tile, 3), dtype=np.float32)
            pad[:h, :w] = crop
            xin = (pad - MEAN) / STD
            xin = torch.from_numpy(np.transpose(xin, (2, 0, 1))).unsqueeze(0).to(device)
            p = torch.sigmoid(model(xin) / T)[0, 0].cpu().numpy()
            sump[y:y + h, x:x + w] += p[:h, :w]
            cnt[y:y + h, x:x + w] += 1.0
    return sump / np.maximum(cnt, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--out", default="results/tiled")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1.0
    if args.json:
        T = float(json.loads((ROOT / args.json).read_text(encoding="utf-8")).get("T", 1.0))

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = raw[args.split]
    if args.max_test:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(items), size=min(args.max_test, len(items)), replace=False)
        items = [items[i] for i in idx]

    inter = union = tp = fp = fn = 0
    dice_sum = 0.0
    n_forged = 0
    paucs = []
    scores, labels = [], []

    for img_path, mask_path, label in items:
        img = load_rgb(img_path)
        prob = tiled_prob(model, img, args.tile, args.stride, device, T)
        sc = float(prob.max())
        scores.append(sc)
        labels.append(label)

        if label == 1 and mask_path is not None:
            t = load_mask(mask_path)
            if t.shape != prob.shape:
                t = np.asarray(Image.fromarray(t.astype(np.uint8) * 255).resize(
                    (prob.shape[1], prob.shape[0]), Image.NEAREST)) > 127
            p = prob > args.thr
            inter += int((p & t).sum())
            union += int((p | t).sum())
            tp += int((p & t).sum())
            fp += int((p & ~t).sum())
            fn += int((~p & t).sum())
            dice_sum += (2 * (p & t).sum()) / ((p.sum() + t.sum()) + 1e-6)
            n_forged += 1
            if t.any():
                paucs.append(roc_auc_score(t.flatten(), prob.flatten()))

    y = np.asarray(labels)
    s = np.asarray(scores)
    img_auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    img_ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, s)
        rec1 = float(np.interp(0.01, fpr, tpr))
    else:
        rec1 = float("nan")

    print(f"ckpt={args.ckpt} | split={args.split} | tile={args.tile} stride={args.stride} | n={len(items)}")
    print(f"tiled: IoU={inter / max(union, 1):.4f} F1={2 * tp / max(2 * tp + fp + fn, 1):.4f} "
          f"Dice={dice_sum / max(n_forged, 1):.4f} PixAUC={np.mean(paucs) if paucs else 0:.4f} "
          f"ImgAUC={img_auc:.4f} AP={img_ap:.4f} R@1%={rec1:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt).stem + "__" + args.split + f"__tile{args.tile}_s{args.stride}.json"
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "T": T,
                   "iou": inter / max(union, 1),
                   "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                   "dice": dice_sum / max(n_forged, 1),
                   "pixel_auc": float(np.mean(paucs)) if paucs else 0.0,
                   "image_auc": img_auc, "image_ap": img_ap, "recall_at_1fpr": rec1,
                   "n": len(items), "n_forged": n_forged}, f, indent=2)
    print("Saved:", out_dir / name)


if __name__ == "__main__":
    main()
