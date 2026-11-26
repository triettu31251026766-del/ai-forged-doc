import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402

THRS = [0.3, 0.4, 0.5, 0.6, 0.7]


def score(prob):
    v = np.sort(prob.ravel())
    return float(v[-max(1, int(0.01 * v.size)):].mean())


def infer(model, img, cx, cy, P, dev):
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def iou(prob, mask, area, cx, cy, P, H, W, thr):
    pred = prob > thr
    x0, y0 = cx - P // 2, cy - P // 2
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
    pc = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    mc = mask[gy0:gy1, gx0:gx1]
    inter = int(np.count_nonzero(pc & mc))
    union = int(np.count_nonzero(pc)) + area - inter
    return inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--strides", default="128,192,256")
    ap.add_argument("--out", default="results/sensitivity")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=dev))
    model.eval()

    items = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
    rng = np.random.default_rng(0)
    items = [items[i] for i in rng.choice(len(items), size=min(args.max_test, len(items)), replace=False)]

    P = args.patch_size
    strides = [int(s) for s in args.strides.split(",")]
    res = {}
    for s1 in strides:
        per = {t: [] for t in THRS}
        nwin = []
        t0 = time.time()
        for ip, mp, _ in items:
            img = load_rgb(ip)
            mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            area = int(mask.sum())
            coarse = []
            for cy in range(P // 2, H, s1):
                for cx in range(P // 2, W, s1):
                    pr = infer(model, img, cx, cy, P, dev)
                    coarse.append((score(pr), cx, cy))
            coarse.sort(key=lambda z: -z[0])
            best = (-1, None, 0, 0)
            seen = set()
            for _, cx, cy in coarse[:3]:
                for dy in (-96, 0, 96):
                    for dx in (-96, 0, 96):
                        nx, ny = cx + dx, cy + dy
                        key = (nx // 8, ny // 8)
                        if key in seen:
                            continue
                        seen.add(key)
                        pr = infer(model, img, nx, ny, P, dev)
                        s = score(pr)
                        if s > best[0]:
                            best = (s, pr, nx, ny)
            _, pr, cx, cy = best
            nwin.append(len(coarse) + len(seen))
            for t in THRS:
                per[t].append(iou(pr, mask, area, cx, cy, P, H, W, t))
        dt = time.time() - t0
        res[str(s1)] = {
            "windows": float(np.mean(nwin)),
            "sec_per_image": dt / len(items),
            "iou": {str(t): float(np.mean(per[t])) for t in THRS},
        }
        print(s1, res[str(s1)])

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", out / "results.json")


if __name__ == "__main__":
    main()
