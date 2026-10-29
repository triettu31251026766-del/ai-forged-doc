import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    V1, V2, GEMINI, GPTIMAGE, authentic_items, build_protocol_splits, forged_items, load_meta,
)
from src.models.classifier import build_classifier  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def bbox_map(meta, split):
    return {str(r["new_id"]): [int(v) for v in r["bbox_xyxy"]]
            for r in meta if r.get("split") == split and "bbox_xyxy" in r}


def crop_window(img, cx, cy, P):
    H, W = img.shape[:2]
    x0 = cx - P // 2; y0 = cy - P // 2
    crop = np.zeros((P, P, 3), dtype=np.float32)
    gx0 = max(0, x0); gy0 = max(0, y0)
    gx1 = min(W, x0 + P); gy1 = min(H, y0 + P)
    if gx1 > gx0 and gy1 > gy0:
        dx = gx0 - x0; dy = gy0 - y0
        crop[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)] = img[gy0:gy1, gx0:gx1]
    return crop


def no_overlap(cx, cy, P, gt, margin=4):
    x0, y0 = cx - P // 2, cy - P // 2
    x1, y1 = x0 + P, y0 + P
    gx1, gy1, gx2, gy2 = gt
    return (x1 <= gx1 + margin or x0 >= gx2 - margin or
            y1 <= gy1 + margin or y0 >= gy2 - margin)


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


def merge_line(boxes, best, gap_factor=1.5):
    bx1, by1, bx2, by2 = best
    bh = max(1, by2 - by1)
    line = []
    for b in boxes:
        x1, y1, x2, y2 = b
        ov = min(by2, y2) - max(by1, y1)
        if ov <= 0.5 * min(by2 - by1, y2 - y1):
            continue
        line.append(b)
    line.sort(key=lambda b: b[0])
    idx = next((i for i, b in enumerate(line) if b == best), None)
    if idx is None:
        return best
    x1, y1, x2, y2 = best
    for i in range(idx + 1, len(line)):
        if line[i][0] - x2 <= gap_factor * bh:
            x1 = min(x1, line[i][0]); x2 = max(x2, line[i][2])
            y1 = min(y1, line[i][1]); y2 = max(y2, line[i][3])
        else:
            break
    for i in range(idx - 1, -1, -1):
        if x1 - line[i][2] <= gap_factor * bh:
            x1 = min(x1, line[i][0]); x2 = max(x2, line[i][2])
            y1 = min(y1, line[i][1]); y2 = max(y2, line[i][3])
        else:
            break
    return (x1, y1, x2, y2)


def place_grid(img, seg, cx, cy, P, grid, full, device, mode="union"):
    if grid <= 1:
        crop = crop_window(img, cx, cy, P)
        arr = (crop - MEAN) / STD
        xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
        prob = torch.sigmoid(seg(xin))[0, 0].cpu().numpy()
        place_max(full, prob, cx, cy, P)
        return
    step = P // 2
    half = grid // 2
    best = None
    for oy in range(-half, half + 1):
        for ox in range(-half, half + 1):
            wx, wy = cx + ox * step, cy + oy * step
            crop = crop_window(img, wx, wy, P)
            arr = (crop - MEAN) / STD
            xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
            prob = torch.sigmoid(seg(xin))[0, 0].cpu().numpy()
            if mode == "best":
                k = max(1, int(0.01 * prob.size))
                score = float(np.sort(prob.ravel())[-k:].mean())
                if best is None or score > best[0]:
                    best = (score, prob, wx, wy)
            else:
                place_max(full, prob, wx, wy, P)
    if mode == "best" and best is not None:
        place_max(full, best[1], best[2], best[3], P)


class BoxDataset(Dataset):
    def __init__(self, items, P=192, jitter=0.15, seed=0):
        self.items = items; self.P = P; self.jitter = jitter
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = load_rgb(it["img"])
        H, W = img.shape[:2]
        if it["bbox"] is not None:
            x1, y1, x2, y2 = it["bbox"]
            gw, gh = max(1, x2 - x1), max(1, y2 - y1)
            gcx, gcy = (x1 + x2) // 2, (y1 + y2) // 2
            if it["pos"]:
                cx, cy = gcx, gcy
                if self.jitter > 0:
                    cx += int(self.rng.uniform(-self.jitter, self.jitter) * gw)
                    cy += int(self.rng.uniform(-self.jitter, self.jitter) * gh)
                cx = max(0, min(W - 1, cx)); cy = max(0, min(H - 1, cy))
            else:
                cx, cy = gcx, gcy
                for _ in range(50):
                    cx = self.rng.randint(0, max(1, W - 1))
                    cy = max(0, min(H - 1, gcy + self.rng.randint(-gh, gh)))
                    if no_overlap(cx, cy, self.P, it["bbox"]):
                        break
        else:
            cx = self.rng.randint(0, max(1, W - 1))
            cy = self.rng.randint(0, max(1, H - 1))

        crop = crop_window(img, cx, cy, self.P)
        arr = (crop - MEAN) / STD
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        y = torch.tensor(1.0 if it["pos"] else 0.0)
        return x, y


def build_train_items(seed=0):
    rng = random.Random(seed)
    meta = load_meta(V1 / "metadata.jsonl")
    bm = bbox_map(meta, "training")
    pos = []
    for img, _, _ in forged_items(meta, V1, tool=GEMINI, split="training"):
        b = bm.get(Path(img).stem)
        if b is not None:
            pos.append({"img": img, "bbox": b, "pos": True})
    neg = [{"img": it["img"], "bbox": it["bbox"], "pos": False}
           for it in pos for _ in range(2)]
    auth = [{"img": img, "bbox": None, "pos": False}
            for img, _, _ in authentic_items(V1, "training")]
    rng.shuffle(auth)
    neg += auth[:max(1, len(pos) // 2)]
    rng.shuffle(pos); rng.shuffle(neg)
    items = pos + neg
    rng.shuffle(items)
    return items


def build_adapt_items(k=100, seed=0):
    rng = random.Random(seed)
    meta = load_meta(V2 / "metadata.jsonl")
    bm = {str(r["new_id"]): [int(v) for v in r["bbox_xyxy"]]
          for r in meta if r.get("split") == "training" and "bbox_xyxy" in r}
    forged = forged_items(meta, V2, tool=GPTIMAGE, split="training")
    rng.shuffle(forged)
    for_pos = forged[:k]
    pos = []
    for img, _, _ in for_pos:
        b = bm.get(Path(img).stem)
        if b is not None:
            pos.append({"img": img, "bbox": b, "pos": True})
    neg = [{"img": it["img"], "bbox": it["bbox"], "pos": False}
           for it in pos for _ in range(2)]
    auth = [{"img": img, "bbox": None, "pos": False}
            for img, _, _ in authentic_items(V2, "training")]
    rng.shuffle(auth)
    neg += auth[:max(1, len(pos) // 2)]
    items = pos + neg
    rng.shuffle(items)
    return items


def detect_boxes(reader, img_rgb, conf, max_boxes, min_size=8):
    bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    res = reader.readtext(bgr, detail=1, paragraph=False)
    out = []
    for bbox, text, c in res:
        if c < conf:
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        out.append(((x1, y1, x2, y2), text))
    out.sort(key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)
    return out[:max_boxes]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1)


def new_stats():
    return {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0, "dice": 0.0, "n": 0}


def update(st, pred, mask):
    inter = int((pred & mask).sum())
    st["inter"] += inter; st["union"] += int((pred | mask).sum()); st["tp"] += inter
    st["fp"] += int((pred & ~mask).sum()); st["fn"] += int((~pred & mask).sum())
    st["dice"] += (2 * inter) / (pred.sum() + mask.sum() + 1e-6); st["n"] += 1


def summarize(st):
    return {"iou": st["inter"] / max(st["union"], 1),
            "f1": 2 * st["tp"] / max(2 * st["tp"] + st["fp"] + st["fn"], 1),
            "dice": st["dice"] / max(st["n"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--cls-size", type=int, default=192)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--seg-ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--digit-bonus", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--grid-mode", choices=["union", "best"], default="union")
    ap.add_argument("--line-merge", action="store_true")
    ap.add_argument("--merge-gap", type=float, default=1.5)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--adapt-k", type=int, default=0)
    ap.add_argument("--adapt-epochs", type=int, default=5)
    ap.add_argument("--adapt-lr", type=float, default=1e-5)
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="results/box_cls")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    items = build_train_items(0)
    if args.max_train:
        items = items[:args.max_train]
    ds = BoxDataset(items, P=args.cls_size, jitter=args.jitter)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Box classifier train: {len(ds)}")

    cls = build_classifier("resnet18", pretrained=True, num_classes=1).to(device)
    if args.eval_only or args.init_ckpt:
        ck = Path(args.init_ckpt) if args.init_ckpt else ROOT / args.out / "box_cls.pth"
        cls.load_state_dict(torch.load(ck, map_location=device))
        print("Loaded:", ck)
    else:
        opt = torch.optim.AdamW(cls.parameters(), lr=args.lr)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(1, args.epochs + 1):
            cls.train(); run = 0.0
            for x, y in tqdm(loader, desc=f"cls {ep}/{args.epochs}", leave=False):
                x = x.to(device); y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logit = cls(x).squeeze(1)
                loss = lossf(logit, y)
                loss.backward(); opt.step()
                run += loss.item() * len(x)
            print(f"  ep {ep}: loss={run/len(ds):.4f}")

    if args.adapt_k > 0:
        adapt_items = build_adapt_items(args.adapt_k)
        ads = BoxDataset(adapt_items, P=args.cls_size, jitter=args.jitter)
        aloader = DataLoader(ads, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"Adapt (few-shot): {len(ads)} samples, k={args.adapt_k}")
        aopt = torch.optim.AdamW(cls.parameters(), lr=args.adapt_lr)
        aloss = nn.BCEWithLogitsLoss()
        for ep in range(1, args.adapt_epochs + 1):
            cls.train(); run = 0.0
            for x, y in aloader:
                x = x.to(device); y = y.to(device)
                aopt.zero_grad(set_to_none=True)
                logit = cls(x).squeeze(1)
                loss = aloss(logit, y)
                loss.backward(); aopt.step()
                run += loss.item() * len(x)
            print(f"  adapt ep {ep}: loss={run/len(ads):.4f}")

    # eval
    import easyocr
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
    seg = build_segmentation("resnet18").to(device)
    seg.load_state_dict(torch.load(ROOT / args.seg_ckpt, map_location=device))
    seg.eval(); cls.eval()

    raw = build_protocol_splits("p1")
    test_items = raw[args.split]
    rng = np.random.default_rng(0)
    if args.max_test and args.max_test < len(test_items):
        idx = rng.choice(len(test_items), size=args.max_test, replace=False)
        test_items = [test_items[i] for i in idx]

    meta = load_meta(V2 / "metadata.jsonl")
    gtb = bbox_map(meta, "testing") if args.split == "test_gptimage" else {}

    stats = new_stats(); paucs = []; paps = []
    scores = []; labels = []
    sel_iou = []; sel_center = 0; n_sel = 0
    sel_dist = []; crop_hit = 0

    with torch.no_grad():
        for img_path, mask_path, label in tqdm(test_items, desc="eval sel"):
            img = load_rgb(img_path); H, W = img.shape[:2]
            gt = gtb.get(Path(img_path).stem)

            if args.oracle and label == 1 and gt is not None:
                cx = (gt[0] + gt[2]) // 2; cy = (gt[1] + gt[3]) // 2
                full = np.zeros((H, W), dtype=np.float32)
                place_grid(img, seg, cx, cy, args.patch_size, args.grid, full, device, args.grid_mode)
                scores.append(float(full.max())); labels.append(label)
                if mask_path is not None:
                    mask = load_mask(mask_path)
                    update(stats, full > 0.5, mask)
                    if mask.any():
                        paucs.append(roc_auc_score(mask.flatten(), full.flatten()))
                        paps.append(average_precision_score(mask.flatten(), full.flatten()))
                sel_iou.append(1.0); sel_center += 1; n_sel += 1
                continue

            boxes = detect_boxes(reader, img, 0.3, 30)
            if not boxes:
                scores.append(0.0); labels.append(label); continue
            cand = []
            for box, text in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                crop = crop_window(img, cx, cy, args.cls_size)
                arr = (crop - MEAN) / STD
                xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
                cscore = float(torch.sigmoid(cls(xin)).item())
                segp = None
                if args.grid <= 1 or args.topk > 1:
                    crop2 = crop_window(img, cx, cy, args.patch_size)
                    arr2 = (crop2 - MEAN) / STD
                    xin2 = torch.from_numpy(np.transpose(arr2, (2, 0, 1))).unsqueeze(0).to(device)
                    segp = torch.sigmoid(seg(xin2))[0, 0].cpu().numpy()
                has_digit = 1.0 if any(ch.isdigit() for ch in text) else 0.0
                sel = cscore + args.digit_bonus * has_digit
                cand.append((box, sel, segp, cx, cy, cscore))
            cand.sort(key=lambda z: z[1], reverse=True)
            best = cand[0]
            sel_box = best[0]
            if args.line_merge:
                sel_box = merge_line([b for b, _ in boxes], best[0], args.merge_gap)
            scx = (sel_box[0] + sel_box[2]) // 2
            scy = (sel_box[1] + sel_box[3]) // 2
            scores.append(best[5]); labels.append(label)
            full = np.zeros((H, W), dtype=np.float32)
            if args.topk > 1:
                for b in cand[:args.topk]:
                    place_max(full, b[2], b[3], b[4], args.patch_size)
            else:
                place_grid(img, seg, scx, scy, args.patch_size, args.grid, full, device, args.grid_mode)

            if label == 1 and mask_path is not None:
                mask = load_mask(mask_path)
                update(stats, full > 0.5, mask)
                if mask.any():
                    paucs.append(roc_auc_score(mask.flatten(), full.flatten()))
                    paps.append(average_precision_score(mask.flatten(), full.flatten()))
                if gt is not None:
                    sel_iou.append(iou(sel_box, gt))
                    gx = (gt[0] + gt[2]) // 2; gy = (gt[1] + gt[3]) // 2
                    if sel_box[0] <= gx <= sel_box[2] and sel_box[1] <= gy <= sel_box[3]:
                        sel_center += 1
                    sel_dist.append(float(np.hypot(scx - gx, scy - gy)))
                    if abs(scx - gx) <= args.patch_size // 2 and abs(scy - gy) <= args.patch_size // 2:
                        crop_hit += 1
                    n_sel += 1

    y = np.asarray(labels); s = np.asarray(scores)
    img_ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, s); rec1 = float(np.interp(0.01, fpr, tpr))
    else:
        rec1 = float("nan")
    sm = summarize(stats)
    print(f"{args.split}: IoU={sm['iou']:.4f} F1={sm['f1']:.4f} Dice={sm['dice']:.4f} "
          f"PixAUC={np.mean(paucs) if paucs else 0:.4f} PixAP={np.mean(paps) if paps else 0:.4f} "
          f"ImgAP={img_ap:.4f} R@1%={rec1:.4f}")
    print(f"selection: mean IoU(selected,GT)={np.mean(sel_iou) if sel_iou else 0:.4f} "
          f"center_in_selected={100*sel_center/max(n_sel,1):.1f}%")
    if not args.oracle:
        print(f"diagnostics: mean_dist={np.mean(sel_dist) if sel_dist else 0:.1f}px "
              f"gt_center_in_crop={100*crop_hit/max(n_sel,1):.1f}% topk={args.topk} grid={args.grid}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(cls.state_dict(), out / "box_cls.pth")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": sm,
                   "pixel_auc": float(np.mean(paucs)) if paucs else 0.0,
                   "pixel_ap": float(np.mean(paps)) if paps else 0.0,
                   "image_ap": float(img_ap), "recall_at_1fpr": float(rec1),
                   "sel_mean_iou": float(np.mean(sel_iou)) if sel_iou else 0.0,
                   "sel_center_in": float(sel_center / max(n_sel, 1)),
                   "sel_mean_dist": float(np.mean(sel_dist)) if sel_dist else 0.0,
                   "gt_center_in_crop": float(crop_hit / max(n_sel, 1)), "topk": args.topk},
                  f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()
