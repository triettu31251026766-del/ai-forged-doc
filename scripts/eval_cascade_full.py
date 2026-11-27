import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402

LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.005), 3)  # 201 marks


def sc(prob):
    v = np.sort(prob.ravel())
    return float(v[-max(1, int(0.01 * v.size)):].mean())


def run(model, img, P, s1, dev):
    H, W = img.shape[:2]
    best = (-1, None, 0, 0)
    seen = set()
    cand = []
    with torch.no_grad():
        for cy in range(P // 2, H, s1):
            for cx in range(P // 2, W, s1):
                crop = crop_window(img, cx, cy, P)
                x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                cand.append((sc(prob), cx, cy, prob))
    cand.sort(key=lambda z: -z[0])
    if not cand:
        cx, cy = W // 2, H // 2
        crop = crop_window(img, cx, cy, P)
        x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        return sc(prob), prob, cx, cy
    for _, cx, cy, _ in cand[:3]:
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


def hist_counts(values, mask_vals):
    idx = np.searchsorted(LAMBDAS, values, side="right") - 1
    idx = np.clip(idx, 0, len(LAMBDAS) - 1)
    return np.bincount(idx[mask_vals], minlength=len(LAMBDAS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--splits", default="test_gptimage")
    ap.add_argument("--P", type=int, default=384)
    ap.add_argument("--stride1", type=int, default=256)
    ap.add_argument("--out", default="results/cascade_full")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation(args.encoder, arch=args.arch).to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    raw = build_protocol_splits("p1")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits.split(","):
        items = raw[split]
        scores, labels = [], []
        hm, hb, msz, ntot, stems = [], [], [], [], []
        ious, f1s, dices = [], [], []
        t0 = time.time()
        for ip, mp, lab in items:
            img = load_rgb(ip)
            s, prob, cx, cy = run(model, img, args.P, args.stride1, dev)
            scores.append(s); labels.append(int(lab))
            if lab != 1 or mp is None:
                continue
            mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            area = int(mask.sum())
            x0, y0 = cx - args.P // 2, cy - args.P // 2
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(W, x0 + args.P), min(H, y0 + args.P)
            pc = prob[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0].ravel()
            mc = mask[gy0:gy1, gx0:gx1].ravel()
            hm.append(hist_counts(pc, mc))
            hb.append(hist_counts(pc, ~mc))
            msz.append(area); ntot.append(H * W); stems.append(Path(ip).stem)
            # per-image at 0.5
            pred = pc > 0.5
            tp = int((pred & mc).sum())
            fp = int((pred & ~mc).sum())
            fn = area - tp  # whole-page FN
            ious.append(tp / max(tp + fp + fn, 1))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
            dices.append(2 * tp / max(int(pred.sum()) + area, 1))
        dt = (time.time() - t0) / max(len(items), 1)
        y = np.asarray(labels); s = np.asarray(scores)
        fpr, tpr, _ = roc_curve(y, s)
        i = max(np.searchsorted(fpr, 0.01, side="right") - 1, 0)
        HM = np.stack(hm); HB = np.stack(hb)
        MS = np.asarray(msz, float)
        # pooled over thresholds
        cum_m = np.cumsum(HM[:, ::-1], axis=1)[:, ::-1]
        cum_b = np.cumsum(HB[:, ::-1], axis=1)[:, ::-1]
        TP = cum_m.sum(0); FP = cum_b.sum(0); FN = MS.sum() - TP
        iou_t = TP / np.maximum(TP + FP + FN, 1)
        f1_t = 2 * TP / np.maximum(2 * TP + FP + FN, 1)
        np.savez_compressed(out / f"{split}.npz", lambdas=LAMBDAS, hist_mask=HM, hist_bg=HB,
                            mask_total=MS, n_pixels=np.asarray(ntot, float), stems=np.asarray(stems),
                            scores=np.asarray(scores), labels=np.asarray(labels))
        summary[split] = {
            "n": int(len(y)),
            "sec_per_image": dt,
            "iou_pooled": float(iou_t[100]),
            "iou_mean": float(np.mean(ious)),
            "f1_pooled": float(f1_t[100]),
            "f1_mean": float(np.mean(f1s)),
            "dice_mean": float(np.mean(dices)),
            "image_auc": float(roc_auc_score(y, s)),
            "image_ap": float(average_precision_score(y, s)),
            "recall_at_1fpr": float(tpr[i]),
        }
        print(split, json.dumps(summary[split], indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
