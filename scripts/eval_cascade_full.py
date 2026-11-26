import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402


def sc(prob):
    v = np.sort(prob.ravel())
    return float(v[-max(1, int(0.01 * v.size)):].mean())


def run(model, img, P, s1, dev):
    H, W = img.shape[:2]
    cand = []
    with torch.no_grad():
        for cy in range(P // 2, H, s1):
            for cx in range(P // 2, W, s1):
                crop = crop_window(img, cx, cy, P)
                x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                cand.append((sc(prob), cx, cy))
    cand.sort(key=lambda z: -z[0])
    if not cand:
        cx, cy = W // 2, H // 2
        crop = crop_window(img, cx, cy, P)
        x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        return sc(prob), prob, cx, cy
    best = (-1, None, 0, 0)
    seen = set()
    for _, cx, cy in cand[:3]:
        for dy in (-96, 0, 96):
            for dx in (-96, 0, 96):
                nx, ny = cx + dx, cy + dy
                key = (nx // 8, ny // 8)
                if key in seen:
                    continue
                seen.add(key)
                crop = crop_window(img, nx, ny, P)
                x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
                with torch.no_grad():
                    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                s = sc(prob)
                if s > best[0]:
                    best = (s, prob, nx, ny)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--splits", default="test_gptimage")
    ap.add_argument("--P", type=int, default=384)
    ap.add_argument("--stride1", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="results/cascade_full.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    raw = build_protocol_splits("p1")
    res = {}
    for split in args.splits.split(","):
        items = raw[split]
        scores, labels = [], []
        inter = union = tp = fp = fn = 0
        dice = []
        for ip, mp, lab in items:
            img = load_rgb(ip)
            s, prob, cx, cy = run(model, img, args.P, args.stride1, dev)
            scores.append(s)
            labels.append(int(lab))
            if lab != 1 or mp is None:
                continue
            mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            area = int(mask.sum())
            x0, y0 = cx - args.P // 2, cy - args.P // 2
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(W, x0 + args.P), min(H, y0 + args.P)
            pc = prob[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] > args.thr
            mc = mask[gy0:gy1, gx0:gx1]
            ii = int((pc & mc).sum())
            uu = int(pc.sum()) + area - ii
            inter += ii; union += uu
            tp += ii; fp += int((pc & ~mc).sum()); fn += int((~pc & mc).sum())
            dice.append(2 * ii / max(int(pc.sum()) + int(mc.sum()), 1))
        y = np.asarray(labels)
        s = np.asarray(scores)
        fpr, tpr, _ = roc_curve(y, s)
        i = max(np.searchsorted(fpr, 0.01, side="right") - 1, 0)
        res[split] = {
            "n": int(len(y)),
            "iou_pooled": inter / max(union, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "dice": float(np.mean(dice)) if dice else 0.0,
            "image_auc": float(roc_auc_score(y, s)),
            "image_ap": float(average_precision_score(y, s)),
            "recall_at_1fpr": float(tpr[i]),
        }
        print(split, json.dumps(res[split]))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
