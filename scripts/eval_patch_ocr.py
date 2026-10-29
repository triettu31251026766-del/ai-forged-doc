import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

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


def crop_window(img, mask, cx, cy, P):
    H, W = img.shape[:2]
    x0 = cx - P // 2
    y0 = cy - P // 2
    crop_img = np.zeros((P, P, 3), dtype=np.float32)
    crop_mask = np.zeros((P, P), dtype=bool)
    gx0 = max(0, x0); gy0 = max(0, y0)
    gx1 = min(W, x0 + P); gy1 = min(H, y0 + P)
    if gx1 > gx0 and gy1 > gy0:
        dx = gx0 - x0; dy = gy0 - y0
        crop_img[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)] = img[gy0:gy1, gx0:gx1]
        crop_mask[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)] = mask[gy0:gy1, gx0:gx1]
    return crop_img, crop_mask


def place_max(full, prob, cx, cy, P):
    H, W = full.shape
    x0 = cx - P // 2; y0 = cy - P // 2
    gx0 = max(0, x0); gy0 = max(0, y0)
    gx1 = min(W, x0 + P); gy1 = min(H, y0 + P)
    if gx1 <= gx0 or gy1 <= gy0:
        return
    dx = gx0 - x0; dy = gy0 - y0
    region = prob[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)]
    np.maximum(full[gy0:gy1, gx0:gx1], region, out=full[gy0:gy1, gx0:gx1])


def detect_boxes(reader, img_rgb, conf, max_boxes, min_size=8):
    bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    res = reader.readtext(bgr, detail=1, paragraph=False)
    boxes = []
    for bbox, text, c in res:
        if c < conf:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:max_boxes]


def region_scores(prob, box, cx, cy, P):
    x1, y1, x2, y2 = box
    x0 = cx - P // 2; y0 = cy - P // 2
    rx1 = max(0, min(P, x1 - x0)); ry1 = max(0, min(P, y1 - y0))
    rx2 = max(0, min(P, x2 - x0)); ry2 = max(0, min(P, y2 - y0))
    fullmax = float(prob.max())  # dung cho image-level detection
    region = prob[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return {"max": fullmax, "mean": float(prob.mean()), "top5": fullmax, "fullmax": fullmax}
    flat = np.sort(region.flatten())
    k5 = max(1, int(0.05 * flat.size))
    return {"max": float(region.max()), "mean": float(region.mean()),
            "top5": float(flat[-k5:].mean()), "fullmax": fullmax}


def new_stats():
    return {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0, "dice": 0.0, "n": 0}


def update(st, pred, mask):
    inter = int((pred & mask).sum())
    st["inter"] += inter
    st["union"] += int((pred | mask).sum())
    st["tp"] += inter
    st["fp"] += int((pred & ~mask).sum())
    st["fn"] += int((~pred & mask).sum())
    st["dice"] += (2 * inter) / (pred.sum() + mask.sum() + 1e-6)
    st["n"] += 1


def summarize(st):
    return {"iou": st["inter"] / max(st["union"], 1),
            "f1": 2 * st["tp"] / max(2 * st["tp"] + st["fp"] + st["fn"], 1),
            "dice": st["dice"] / max(st["n"], 1)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-boxes", type=int, default=30)
    ap.add_argument("--score-modes", default="max,mean,top5")
    ap.add_argument("--topk", default="1,2,3,5")
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--out", default="results/patch_ocr")
    args = ap.parse_args()

    import easyocr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = raw[args.split]
    rng = np.random.default_rng(0)
    if args.max_test and args.max_test < len(items):
        idx = rng.choice(len(items), size=args.max_test, replace=False)
        items = [items[i] for i in idx]

    MODES = [m.strip() for m in args.score_modes.split(",")]
    KS = [int(x) for x in args.topk.split(",")]
    stats = {(m, k): new_stats() for m in MODES for k in KS}
    paucs = {(m, k): [] for m in MODES for k in KS}
    paps = {(m, k): [] for m in MODES for k in KS}
    scores, labels = [], []
    n_boxes = []

    for img_path, mask_path, label in tqdm(items, desc="ocr+patch"):
        img = load_rgb(img_path)
        H, W = img.shape[:2]
        boxes = detect_boxes(reader, img, args.conf, args.max_boxes)
        n_boxes.append(len(boxes))

        preds = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            crop_img, _ = crop_window(img, np.zeros((H, W), bool), cx, cy, args.patch_size)
            arr = (crop_img - MEAN) / STD
            xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(xin))[0, 0].cpu().numpy()
            preds.append((cx, cy, prob, region_scores(prob, box, cx, cy, args.patch_size)))

        if preds:
            scores.append(max(p[3]["fullmax"] for p in preds))
        else:
            scores.append(0.0)
        labels.append(label)

        if label == 1 and mask_path is not None:
            mask = load_mask(mask_path)
            for m in MODES:
                order = sorted(preds, key=lambda p: p[3][m], reverse=True)
                for k in KS:
                    full = np.zeros((H, W), dtype=np.float32)
                    for (cx, cy, prob, _) in order[:k]:
                        place_max(full, prob, cx, cy, args.patch_size)
                    update(stats[(m, k)], full > args.thr, mask)
                    if mask.any():
                        paucs[(m, k)].append(roc_auc_score(mask.flatten(), full.flatten()))
                        paps[(m, k)].append(average_precision_score(mask.flatten(), full.flatten()))

    y = np.asarray(labels); s = np.asarray(scores)
    if len(np.unique(y)) > 1:
        img_ap = average_precision_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        rec1 = float(np.interp(0.01, fpr, tpr))
    else:
        img_ap = rec1 = float("nan")

    print(f"split={args.split} | n={len(items)} | avg_boxes={np.mean(n_boxes):.1f} "
          f"| ImgAP={img_ap:.4f} R@1%={rec1:.4f}")
    results = {"args": vars(args), "image_ap": float(img_ap), "recall_at_1fpr": float(rec1),
               "avg_boxes": float(np.mean(n_boxes)), "by_mode_k": {}}
    for m in MODES:
        for k in KS:
            s2 = summarize(stats[(m, k)])
            s2["pixel_auc"] = float(np.mean(paucs[(m, k)])) if paucs[(m, k)] else 0.0
            s2["pixel_ap"] = float(np.mean(paps[(m, k)])) if paps[(m, k)] else 0.0
            results["by_mode_k"][f"{m}_{k}"] = s2
            print(f"{m:>5} top-{k}: IoU={s2['iou']:.4f} F1={s2['f1']:.4f} "
                  f"Dice={s2['dice']:.4f} PixAUC={s2['pixel_auc']:.4f} PixAP={s2['pixel_ap']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.split}_ocr_modes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir / f"{args.split}_ocr_modes.json")


if __name__ == "__main__":
    main()
