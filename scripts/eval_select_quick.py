import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import build_protocol_splits, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import (  # noqa: E402
    MEAN, STD, bbox_map, crop_window, detect_boxes, iou, load_mask, load_rgb,
    place_max, update, summarize, new_stats,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_params(W, H, size):
    scale = min(size / W, size / H)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    return scale, new_w, new_h, left, top


def global_forward(gmodel, img, size, device):
    H, W = img.shape[:2]
    scale, new_w, new_h, left, top = letterbox_params(W, H, size)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img_r = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img_r, (left, top))
    raw = np.asarray(canvas, dtype=np.float32) / 255.0
    rgb = (raw - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(gmodel(x))[0, 0].cpu().numpy()
    return prob, scale, new_w, new_h, left, top


def win_score(prob):
    k = max(1, int(0.01 * prob.size))
    return float(np.sort(prob.ravel())[-k:].mean())


def local_peaks(prob, scale, nw, nh, left, top, k=3, min_dist=48):
    p = prob.copy()
    p[:top, :] = 0.0
    p[top + nh:, :] = 0.0
    p[:, :left] = 0.0
    p[:, left + nw:] = 0.0
    p = cv2.GaussianBlur(p, (7, 7), 0)
    if p.max() <= 0:
        return []
    mx = cv2.dilate(p, np.ones((2 * min_dist + 1, 2 * min_dist + 1), np.uint8))
    ys, xs = np.where((p == mx) & (p > 0.15 * p.max()))
    vals = p[ys, xs]
    order = np.argsort(vals)[::-1][:k]
    out = []
    for i in order:
        u, v = xs[i], ys[i]
        cx = (u + 0.5 - left) / scale
        cy = (v + 0.5 - top) / scale
        out.append((float(cx), float(cy)))
    return out


def seg_place(img, seg, cx, cy, P, device):
    H, W = img.shape[:2]
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, prob, int(cx), int(cy), P)
    return full, win_score(prob)


def seg_place_scale(img, seg, cx, cy, S, P, device):
    H, W = img.shape[:2]
    crop = crop_window(img, int(cx), int(cy), S)
    rsz = cv2.resize(crop, (P, P), interpolation=cv2.INTER_LINEAR)
    arr = (rsz - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    probS = cv2.resize(prob, (S, S), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, probS, int(cx), int(cy), S)
    return full, win_score(prob)


def field_features(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(0.25 * w), max(2, int(0.4 * h))
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = img[y1:y2, x1:x2].astype(np.float32)
    gray = patch.mean(axis=2)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hp = gray - blur
    med = float(np.median(hp))
    mad = float(np.median(np.abs(hp - med))) + 1e-8
    noise = mad / 0.6745
    hp_std = float(hp.std())
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = float(np.mean(np.abs(gx) + np.abs(gy)))
    bright = float(gray.mean())
    color = float(patch.std(axis=2).mean())
    edge = float((np.abs(gx) > 0.05).mean())
    return np.array([noise, hp_std, lap_var, grad, bright, color, edge], dtype=np.float32)


def handcraft_select(img, boxes):
    feats = []
    for box, _ in boxes:
        f = field_features(img, box)
        if f is None:
            feats.append(None)
        else:
            feats.append(f)
    valid = [i for i, f in enumerate(feats) if f is not None]
    if not valid:
        return None
    F = np.stack([feats[i] for i in valid])
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) + 1e-8
    z = np.abs((F - med) / (1.4826 * mad))
    anom = z.mean(axis=1)
    bi = valid[int(anom.argmax())]
    bx = boxes[bi][0]
    cx = (bx[0] + bx[2]) // 2; cy = (bx[1] + bx[3]) // 2
    return cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--split", default="test_gptimage")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    gmodel = build_segmentation("resnet18").to(device)
    gmodel.load_state_dict(torch.load(ROOT / args.global_ckpt, map_location=device))
    gmodel.eval()
    seg = build_segmentation("resnet18").to(device)
    seg.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=device))
    seg.eval()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)

    raw = build_protocol_splits("p1")
    test_items = raw[args.split]
    rng = np.random.default_rng(0)
    if args.max_test and args.max_test < len(test_items):
        idx = rng.choice(len(test_items), size=args.max_test, replace=False)
        test_items = [test_items[i] for i in idx]

    meta = load_meta(ROOT / "data" / "raw" / "AIForge-Doc-v2" / "metadata.jsonl")
    gtb = bbox_map(meta, "testing") if args.split == "test_gptimage" else {}

    names = ["single_peak", "multipeak", "peak_ms", "multi_ms", "handcraft", "oracle_ms"]
    stats = {n: new_stats() for n in names}
    paucs = {n: [] for n in names}
    paps = {n: [] for n in names}

    with torch.no_grad():
        for img_path, mask_path, label in tqdm(test_items, desc="quick"):
            img = load_rgb(img_path); H, W = img.shape[:2]
            prob384, scale, nw, nh, left, top = global_forward(gmodel, img, args.size, device)

            boxes = detect_boxes(reader, img, 0.3, 30)
            gt = gtb.get(Path(img_path).stem)

            preds = {}

            # single peak
            peaks = local_peaks(prob384, scale, nw, nh, left, top, k=1)
            p0 = peaks[0] if peaks else (0.0, 0.0)
            full0, _ = seg_place(img, seg, p0[0], p0[1], args.size, device)
            preds["single_peak"] = full0

            # multi-peak: top-3 peaks -> patch each -> best by patch response
            mpeaks = local_peaks(prob384, scale, nw, nh, left, top, k=3)
            if mpeaks:
                best = None
                for cx, cy in mpeaks:
                    f, sc = seg_place(img, seg, cx, cy, args.size, device)
                    if best is None or sc > best[0]:
                        best = (sc, f)
                preds["multipeak"] = best[1]

            # multi-scale around single peak
            best_ms = None
            for S in (384, 512, 768):
                f, sc = seg_place_scale(img, seg, p0[0], p0[1], S, 384, device)
                if best_ms is None or sc > best_ms[0]:
                    best_ms = (sc, f)
            preds["peak_ms"] = best_ms[1]

            # multi-scale x multi-peak
            if mpeaks:
                best_mm = None
                for cx, cy in mpeaks:
                    for S in (384, 512, 768):
                        f, sc = seg_place_scale(img, seg, cx, cy, S, 384, device)
                        if best_mm is None or sc > best_mm[0]:
                            best_mm = (sc, f)
                preds["multi_ms"] = best_mm[1]

            # handcrafted field-consistency
            if boxes:
                hc = handcraft_select(img, boxes)
                if hc is not None:
                    f, _ = seg_place(img, seg, hc[0], hc[1], args.size, device)
                    preds["handcraft"] = f

            if label != 1 or mask_path is None:
                continue
            mask = load_mask(mask_path)
            for n, full in preds.items():
                update(stats[n], full > 0.5, mask)
                if mask.any():
                    paucs[n].append(roc_auc_score(mask.flatten(), full.flatten()))
                    paps[n].append(average_precision_score(mask.flatten(), full.flatten()))

            # oracle multi-scale: best scale by IoU with GT
            if gt is not None:
                gc_x = (gt[0] + gt[2]) // 2; gc_y = (gt[1] + gt[3]) // 2
                best_io = -1.0; best_f = None
                for S in (384, 512, 768):
                    f, _ = seg_place_scale(img, seg, gc_x, gc_y, S, 384, device)
                    st = new_stats(); update(st, f > 0.5, mask)
                    io = summarize(st)["iou"]
                    if io > best_io:
                        best_io = io; best_f = f
                if best_f is not None:
                    update(stats["oracle_ms"], best_f > 0.5, mask)
                    if mask.any():
                        paucs["oracle_ms"].append(roc_auc_score(mask.flatten(), best_f.flatten()))
                        paps["oracle_ms"].append(average_precision_score(mask.flatten(), best_f.flatten()))

    print(f"\n=== Quick selection bundle ({args.split}, {args.max_test} imgs) ===")
    for n in names:
        sm = summarize(stats[n])
        print(f"{n:12s}: IoU={sm['iou']:.4f} F1={sm['f1']:.4f} Dice={sm['dice']:.4f} "
              f"PixAUC={np.mean(paucs[n]) if paucs[n] else 0:.4f} "
              f"PixAP={np.mean(paps[n]) if paps[n] else 0:.4f}")


if __name__ == "__main__":
    main()
