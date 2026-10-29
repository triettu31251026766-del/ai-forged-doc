import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits, load_meta  # noqa: E402
from src.models.classifier import build_classifier  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import (  # noqa: E402
    MEAN, STD, bbox_map, crop_window, detect_boxes, iou, load_mask, load_rgb,
    new_stats, place_max, summarize, update,
)
from eval_coarse2fine import box_region_mean, global_forward  # noqa: E402


def rank(v):
    order = np.argsort(-v)
    r = np.empty(len(v), dtype=np.float32)
    r[order] = np.arange(len(v), dtype=np.float32)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-ckpt", default="results/box_cls/box_cls.pth")
    ap.add_argument("--cls-size", type=int, default=384)
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--forged-only", action="store_true")
    ap.add_argument("--split", default="test_gptimage")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    cls = build_classifier("resnet18", pretrained=True, num_classes=1).to(device)
    cls.load_state_dict(torch.load(ROOT / args.cls_ckpt, map_location=device))
    cls.eval()
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

    combos = ["c", "g", "p", "cg", "cp", "gp", "cgp"]
    stats = {n: new_stats() for n in combos}
    paucs = {n: [] for n in combos}
    paps = {n: [] for n in combos}
    sel_center = {n: 0 for n in combos}
    n_sel = {n: 0 for n in combos}

    with torch.no_grad():
        for img_path, mask_path, label in tqdm(test_items, desc="ens"):
            if args.forged_only and label != 1:
                continue
            img = load_rgb(img_path); H, W = img.shape[:2]
            boxes = detect_boxes(reader, img, 0.3, 30)
            if not boxes:
                continue
            prob384, scale, nw, nh, left, top = global_forward(gmodel, img, args.gsize, device)

            c_s, g_s, p_s = [], [], []
            for box, _ in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                crop = crop_window(img, cx, cy, args.cls_size)
                arr = (crop - MEAN) / STD
                xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
                c_s.append(float(torch.sigmoid(cls(xin)).item()))
                g_s.append(box_region_mean(prob384, box, scale, left, top))
                crop2 = crop_window(img, cx, cy, args.patch_size)
                arr2 = (crop2 - MEAN) / STD
                xin2 = torch.from_numpy(np.transpose(arr2, (2, 0, 1))).unsqueeze(0).to(device)
                pr = torch.sigmoid(seg(xin2))[0, 0].cpu().numpy()
                k = max(1, int(0.01 * pr.size))
                p_s.append(float(np.sort(pr.ravel())[-k:].mean()))

            c_s = np.asarray(c_s, dtype=np.float32)
            g_s = np.asarray(g_s, dtype=np.float32)
            p_s = np.asarray(p_s, dtype=np.float32)
            R = {"c": rank(c_s), "g": rank(g_s), "p": rank(p_s)}

            mask = load_mask(mask_path) if (label == 1 and mask_path is not None) else None
            gt = gtb.get(Path(img_path).stem)

            for n in combos:
                rr = np.mean([R[ch] for ch in n], axis=0)
                bi = int(np.argmin(rr))
                bx = boxes[bi][0]
                cx = (bx[0] + bx[2]) // 2; cy = (bx[1] + bx[3]) // 2
                crop = crop_window(img, cx, cy, args.patch_size)
                arr = (crop - MEAN) / STD
                xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
                segp = torch.sigmoid(seg(xin))[0, 0].cpu().numpy()
                full = np.zeros((H, W), dtype=np.float32)
                place_max(full, segp, cx, cy, args.patch_size)
                if mask is not None:
                    update(stats[n], full > 0.5, mask)
                    if mask.any():
                        paucs[n].append(roc_auc_score(mask.flatten(), full.flatten()))
                        paps[n].append(average_precision_score(mask.flatten(), full.flatten()))
                    if gt is not None:
                        gx = (gt[0] + gt[2]) // 2; gy = (gt[1] + gt[3]) // 2
                        if bx[0] <= gx <= bx[2] and bx[1] <= gy <= bx[3]:
                            sel_center[n] += 1
                        n_sel[n] += 1

    print(f"\n=== Ensemble selector (rank fusion, {args.split}, {args.max_test} imgs) ===")
    for n in combos:
        sm = summarize(stats[n])
        print(f"{n:4s}: IoU={sm['iou']:.4f} F1={sm['f1']:.4f} Dice={sm['dice']:.4f} "
              f"PixAUC={np.mean(paucs[n]) if paucs[n] else 0:.4f} "
              f"PixAP={np.mean(paps[n]) if paps[n] else 0:.4f} "
              f"center={100*sel_center[n]/max(n_sel[n],1):.1f}%")


if __name__ == "__main__":
    main()
