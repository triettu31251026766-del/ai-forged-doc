import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402


def win_iou(pred, mask, mask_area, cx, cy, P, H, W):
    x0 = cx - P // 2; y0 = cy - P // 2
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
    pc = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    mc = mask[gy0:gy1, gx0:gx1]
    inter = int(np.count_nonzero(pc & mc))
    union = int(np.count_nonzero(pc)) + mask_area - inter
    return inter, union, inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--out", default="results/slide")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    seg = build_segmentation("resnet18").to(dev)
    seg.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=dev)); seg.eval()

    items = [it for it in build_protocol_splits("p1")[args.split] if it[2] == 1]
    rng = np.random.default_rng(args.seed)
    n_take = len(items) if args.max_test <= 0 else min(args.max_test, len(items))
    idx = rng.choice(len(items), size=n_take, replace=False)
    items = [items[i] for i in idx]

    keys = ["top1", "meanpred", "max", "mass", "hybrid"]
    agg = {k: {"inter": 0, "union": 0, "per": []} for k in keys + ["oracle"]}
    nwin = []

    P = args.patch_size
    with torch.no_grad():
        for ip, mp, _ in tqdm(items, desc="slide"):
            img = load_rgb(ip); mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            mask_area = int(mask.sum())
            best = {k: (-1e9, 0, 0, 0.0) for k in keys}
            best_iou = 0.0
            cnt = 0
            for cy in range(P // 2, H, args.stride):
                for cx in range(P // 2, W, args.stride):
                    crop = crop_window(img, cx, cy, P)
                    arr = (crop - MEAN) / STD
                    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
                    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
                    pred = prob > 0.5
                    inter, union, iou = win_iou(pred, mask, mask_area, cx, cy, P, H, W)
                    v = np.sort(prob.ravel())
                    k1 = max(1, int(0.01 * v.size))
                    area = int(np.count_nonzero(pred))
                    sc = {"top1": float(v[-k1:].mean()),
                          "max": float(v[-1]),
                          "meanpred": float(prob[pred].mean()) if area else 0.0,
                          "mass": float(prob.sum()) / prob.size,
                          "hybrid": float(v[-k1:].mean()) * np.sqrt(max(area, 1))}
                    for k in keys:
                        if sc[k] > best[k][0]:
                            best[k] = (sc[k], inter, union, iou)
                    best_iou = max(best_iou, iou)
                    cnt += 1
            nwin.append(cnt)
            if cnt == 0:
                continue
            for k in keys:
                _, inter, union, iou = best[k]
                agg[k]["inter"] += inter
                agg[k]["union"] += union
                agg[k]["per"].append(iou)
            agg["oracle"]["per"].append(best_iou)

    print(f"\n=== Sliding window ({len(items)} imgs, {np.mean(nwin):.0f} windows/img, stride {args.stride}) ===")
    rows = [(agg[k]["inter"] / max(agg[k]["union"], 1), float(np.mean(agg[k]["per"])), k) for k in keys]
    for pooled, mean, k in sorted(rows, reverse=True):
        print(f"{k:9s}: IoU_pooled={pooled:.4f}  IoU_mean={mean:.4f}")
    print(f"{'oracle':9s}: IoU_mean={np.mean(agg['oracle']['per']):.4f}  (best window per image)")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "avg_windows": float(np.mean(nwin)),
                   "methods": {k: {"iou_pooled": agg[k]["inter"] / max(agg[k]["union"], 1),
                                   "iou_mean": float(np.mean(agg[k]["per"]))} for k in keys},
                   "oracle_window_iou_mean": float(np.mean(agg["oracle"]["per"]))}, f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()
