import argparse
import json
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window  # noqa: E402


def decode(buf, flag):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--patch-ckpt", required=True)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--out", default="results/doctamper_eval.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(args.patch_ckpt, map_location=dev))
    model.eval()

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        ids = [k.decode().split("-")[1] for k, _ in txn.cursor() if k.startswith(b"image-")]
    ids = sorted(ids)[: args.max_test]
    print(f"evaluating {len(ids)} images on {dev}")

    P = args.patch_size
    inter = union = tp = fp = fn = 0
    per = []
    n = 0
    with torch.no_grad(), env.begin() as txn:
        for i in ids:
            ib = txn.get(f"image-{i}".encode())
            lb = txn.get(f"label-{i}".encode())
            if ib is None or lb is None:
                continue
            img = decode(ib, cv2.IMREAD_COLOR)
            mask = decode(lb, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            H, W = img.shape[:2]
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask > 127
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask_area = int(mask.sum())
            if mask_area == 0:
                continue

            best = (-1.0, None, 0, 0)
            for cy in range(P // 2, H, args.stride):
                for cx in range(P // 2, W, args.stride):
                    crop = crop_window(img, cx, cy, P)
                    arr = (crop - MEAN) / STD
                    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
                    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                    v = np.sort(prob.ravel())
                    k1 = max(1, int(0.01 * v.size))
                    s = float(v[-k1:].mean())
                    if s > best[0]:
                        best = (s, prob > 0.5, cx, cy)
            _, pred, cx, cy = best
            x0 = cx - P // 2; y0 = cy - P // 2
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
            pc = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
            mc = mask[gy0:gy1, gx0:gx1]
            ii = int(np.count_nonzero(pc & mc))
            uu = int(np.count_nonzero(pc)) + mask_area - ii
            inter += ii; union += uu
            tp += ii
            fp += int(np.count_nonzero(pc & ~mc))
            fn += int(np.count_nonzero(~pc & mc))
            per.append(ii / max(uu, 1))
            n += 1
    env.close()

    res = {"n": n, "iou_pooled": inter / max(union, 1),
           "iou_mean": float(np.mean(per)) if per else 0.0,
           "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    print(json.dumps(res, indent=2))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
