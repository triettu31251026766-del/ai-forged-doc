import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import load_mask, load_rgb  # noqa: E402
from eval_cascade_full import run  # noqa: E402


def corrupt(img, kind):
    if kind == "none":
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    if kind.startswith("jpeg"):
        q = int(kind.split("_")[1])
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif kind == "resize":
        h, w = u8.shape[:2]
        small = cv2.resize(u8, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "blur":
        out = cv2.GaussianBlur(u8, (5, 5), 1.0)
    else:
        return img
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--P", type=int, default=384)
    ap.add_argument("--stride1", type=int, default=256)
    ap.add_argument("--max-forged", type=int, default=200)
    ap.add_argument("--max-auth", type=int, default=100)
    ap.add_argument("--kinds", default="none,jpeg_90,jpeg_75,resize,blur")
    ap.add_argument("--out", default="results/robustness.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    raw = build_protocol_splits("p1")
    forged = [it for it in raw["test_gptimage"] if it[2] == 1][: args.max_forged]
    auth = [it for it in raw["test_gptimage"] if it[2] == 0][: args.max_auth]

    res = {}
    for kind in args.kinds.split(","):
        inter = union = tp = fp = fn = 0
        ious, f1s = [], []
        scores, labels = [], []
        for ip, mp, lab in forged + auth:
            img = corrupt(load_rgb(ip), kind)
            s, prob, cx, cy = run(model, img, args.P, args.stride1, dev)
            scores.append(s); labels.append(int(lab))
            if lab != 1:
                continue
            mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            area = int(mask.sum())
            x0, y0 = cx - args.P // 2, cy - args.P // 2
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(W, x0 + args.P), min(H, y0 + args.P)
            pc = prob[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] > 0.5
            mc = mask[gy0:gy1, gx0:gx1]
            i2 = int((pc & mc).sum())
            u2 = int(pc.sum()) + area - i2
            inter += i2; union += u2
            tp += i2; fp += int((pc & ~mc).sum()); fn += area - i2
            ious.append(i2 / max(u2, 1))
            f1s.append(2 * i2 / max(int(pc.sum()) + area, 1))
        y = np.asarray(labels); s = np.asarray(scores)
        fpr, tpr, _ = roc_curve(y, s)
        i = max(np.searchsorted(fpr, 0.01, side="right") - 1, 0)
        res[kind] = {
            "iou_pooled": inter / max(union, 1),
            "iou_mean": float(np.mean(ious)),
            "f1_pooled": 2 * tp / max(2 * tp + fp + fn, 1),
            "image_auc": float(roc_auc_score(y, s)),
            "image_ap": float(average_precision_score(y, s)),
            "recall_at_1fpr": float(tpr[i]),
        }
        print(kind, json.dumps(res[kind]))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
