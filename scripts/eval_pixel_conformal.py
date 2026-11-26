import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402


def score(prob):
    v = np.sort(prob.ravel())
    return float(v[-max(1, int(0.01 * v.size)):].mean())


def best_window(model, img, P, s1, dev):
    H, W = img.shape[:2]
    coarse = []
    with torch.no_grad():
        for cy in range(P // 2, H, s1):
            for cx in range(P // 2, W, s1):
                crop = crop_window(img, cx, cy, P)
                x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                coarse.append((score(prob), cx, cy, prob))
    coarse.sort(key=lambda z: -z[0])
    if not coarse:
        cx, cy = W // 2, H // 2
        crop = crop_window(img, cx, cy, P)
        x = torch.from_numpy(np.transpose((crop - MEAN) / STD, (2, 0, 1))).unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        return (score(prob), prob, cx, cy)
    best = (-1, None, 0, 0)
    seen = set()
    for _, cx, cy, _ in coarse[:3]:
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
                s = score(prob)
                if s > best[0]:
                    best = (s, prob, nx, ny)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--stride1", type=int, default=256)
    ap.add_argument("--P", type=int, default=384)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--out", default="results/pixel_conformal.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    items = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(items))
    cal = [items[i] for i in idx[: args.n_cal]]
    test = [items[i] for i in idx[args.n_cal:]]

    def forged_probs(seq):
        out = []
        for ip, mp, _ in seq:
            img = load_rgb(ip)
            mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            s, prob, cx, cy = best_window(model, img, args.P, args.stride1, dev)
            if prob is None:
                continue
            x0, y0 = cx - args.P // 2, cy - args.P // 2
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(W, x0 + args.P), min(H, y0 + args.P)
            pc = prob[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
            mc = mask[gy0:gy1, gx0:gx1]
            if mc.any():
                out.append(pc[mc].astype(np.float32))
        return np.concatenate(out) if out else np.array([0.0])

    cal_p = forged_probs(cal)
    thr = float(np.quantile(cal_p, args.alpha))

    fn = fp = pos = neg = 0
    for ip, mp, _ in test:
        img = load_rgb(ip)
        mask = load_mask(mp) > 0
        H, W = img.shape[:2]
        s, prob, cx, cy = best_window(model, img, args.P, args.stride1, dev)
        if prob is None:
            continue
        x0, y0 = cx - args.P // 2, cy - args.P // 2
        gx0, gy0 = max(0, x0), max(0, y0)
        gx1, gy1 = min(W, x0 + args.P), min(H, y0 + args.P)
        pc = prob[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] > thr
        mc = mask[gy0:gy1, gx0:gx1]
        fn += int((~pc & mc).sum()); pos += int(mc.sum())
        fp += int((pc & ~mc).sum()); neg += int((~mc).sum())

    res = {"threshold": thr, "alpha": args.alpha,
           "calib_FNR": float((cal_p <= thr).mean()),
           "test_FNR": fn / max(pos, 1), "test_FPR": fp / max(neg, 1),
           "n_cal": len(cal), "n_test": len(test)}
    print(json.dumps(res, indent=2))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
