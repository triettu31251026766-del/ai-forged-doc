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


def box_region_mean(prob, box, scale, left, top):
    x1, y1, x2, y2 = box
    u1 = int(x1 * scale + left); v1 = int(y1 * scale + top)
    u2 = int(x2 * scale + left); v2 = int(y2 * scale + top)
    u1 = max(0, min(prob.shape[1], u1)); u2 = max(u1 + 1, min(prob.shape[1], u2))
    v1 = max(0, min(prob.shape[0], v1)); v2 = max(v1 + 1, min(prob.shape[0], v2))
    return float(prob[v1:v2, u1:u2].mean())


def peak_center(prob, scale, new_w, new_h, left, top):
    content = prob[top:top + new_h, left:left + new_w]
    if content.size == 0:
        return 0.0, 0.0
    vy, vx = np.unravel_index(int(content.argmax()), content.shape)
    u = vx + left; v = vy + top
    cx = (u + 0.5 - left) / scale
    cy = (v + 0.5 - top) / scale
    return float(cx), float(cy)


def seg_place(img, seg, cx, cy, P, device):
    H, W = img.shape[:2]
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, prob, int(cx), int(cy), P)
    return full


def soft_centroid(full):
    mx = float(full.max())
    if mx <= 0.05:
        return None
    thr = max(0.3, 0.5 * mx)
    ys, xs = np.where(full >= thr)
    if len(xs) < 4:
        iy, ix = np.unravel_index(int(full.argmax()), full.shape)
        return float(ix), float(iy)
    return float(xs.mean()), float(ys.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384, help="do phan giai input cua global model")
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

    names = ["peak", "peak_refine", "peak_refine2", "ocr_global", "ocr_zscore", "fusion"]
    stats = {n: new_stats() for n in names}
    paucs = {n: [] for n in names}
    paps = {n: [] for n in names}

    with torch.no_grad():
        for img_path, mask_path, label in tqdm(test_items, desc="c2f"):
            img = load_rgb(img_path); H, W = img.shape[:2]
            prob384, scale, nw, nh, left, top = global_forward(gmodel, img, args.gsize, device)

            boxes = detect_boxes(reader, img, 0.3, 30)

            preds = {}

            # A: peak center (no OCR)
            pcx, pcy = peak_center(prob384, scale, nw, nh, left, top)
            preds["peak"] = seg_place(img, seg, pcx, pcy, args.size, device)

            # B/C: OCR scored by global heatmap
            if boxes:
                gs = [box_region_mean(prob384, b, scale, left, top) for b, _ in boxes]
                gs = np.asarray(gs, dtype=np.float32)
                bi = int(gs.argmax())
                x1, y1, x2, y2 = boxes[bi][0]
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                preds["ocr_global"] = seg_place(img, seg, cx, cy, args.size, device)

                med = float(np.median(gs))
                mad = float(np.median(np.abs(gs - med))) + 1e-6
                z = (gs - med) / mad
                zi = int(z.argmax())
                zx1, zy1, zx2, zy2 = boxes[zi][0]
                zcx = (zx1 + zx2) // 2; zcy = (zy1 + zy2) // 2
                preds["ocr_zscore"] = seg_place(img, seg, zcx, zcy, args.size, device)

            # D: fusion = global upscale * patch(peak)
            p_content = prob384[top:top + nh, left:left + nw]
            p_native = cv2.resize(p_content, (W, H), interpolation=cv2.INTER_LINEAR)
            preds["fusion"] = p_native * preds["peak"]

            c1 = soft_centroid(preds["peak"])
            if c1 is not None:
                preds["peak_refine"] = seg_place(img, seg, c1[0], c1[1], args.size, device)
                c2 = soft_centroid(preds["peak_refine"])
                if c2 is not None:
                    preds["peak_refine2"] = seg_place(img, seg, c2[0], c2[1], args.size, device)

            if label != 1 or mask_path is None:
                continue
            mask = load_mask(mask_path)
            for n, full in preds.items():
                update(stats[n], full > 0.5, mask)
                if mask.any():
                    paucs[n].append(roc_auc_score(mask.flatten(), full.flatten()))
                    paps[n].append(average_precision_score(mask.flatten(), full.flatten()))

    print(f"\n=== Coarse-to-fine / context-consistency ({args.split}, {args.max_test} imgs) ===")
    for n in names:
        sm = summarize(stats[n])
        print(f"{n:12s}: IoU={sm['iou']:.4f} F1={sm['f1']:.4f} Dice={sm['dice']:.4f} "
              f"PixAUC={np.mean(paucs[n]) if paucs[n] else 0:.4f} "
              f"PixAP={np.mean(paps[n]) if paps[n] else 0:.4f}")


if __name__ == "__main__":
    main()
