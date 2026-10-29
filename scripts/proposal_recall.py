import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import V2, build_protocol_splits, load_meta  # noqa: E402


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-boxes", type=int, default=30)
    ap.add_argument("--max-test", type=int, default=100)
    ap.add_argument("--out", default="results/proposal")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    if args.split == "test_gptimage":
        meta = load_meta(V2 / "metadata.jsonl")
        bm = {str(r["new_id"]): [int(v) for v in r["bbox_xyxy"]]
              for r in meta if r.get("split") == "testing" and "bbox_xyxy" in r}
        srcmap = {str(r["new_id"]): r.get("source_dataset", "?")
                  for r in meta if r.get("split") == "testing"}
    else:
        raise SystemExit("Script nay ho tro test_gptimage")

    raw = build_protocol_splits("p1")
    items = [it for it in raw[args.split] if it[2] == 1]
    rng = np.random.default_rng(0)
    if args.max_test and args.max_test < len(items):
        idx = rng.choice(len(items), size=args.max_test, replace=False)
        items = [items[i] for i in idx]

    rows = []
    for img_path, mask_path, _ in tqdm(items, desc="proposal recall"):
        stem = Path(img_path).stem
        gt = bm.get(stem)
        if gt is None:
            continue
        img = load_rgb(img_path)
        H, W = img.shape[:2]
        boxes = detect_boxes(reader, img, args.conf, args.max_boxes)

        gx = (gt[0] + gt[2]) // 2
        gy = (gt[1] + gt[3]) // 2

        best_iou = 0.0
        center_in = False
        best_cdist = 1e9
        for (x1, y1, x2, y2) in boxes:
            best_iou = max(best_iou, iou(gt, (x1, y1, x2, y2)))
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                center_in = True
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 / max(1.0, (H * W) ** 0.5)
            best_cdist = min(best_cdist, d)

        # coverage of GT bbox by union of 384-crops centered at boxes
        cov = np.zeros((H, W), dtype=bool)
        P = args.patch_size
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cov[max(0, cy - P // 2):min(H, cy + P // 2), max(0, cx - P // 2):min(W, cx + P // 2)] = True
        gx1, gy1, gx2, gy2 = gt
        gt_area = max(1, (gx2 - gx1) * (gy2 - gy1))
        cov_pixels = cov[gy1:gy2, gx1:gx2].sum()
        gt_cov = cov_pixels / gt_area
        center_cov = bool(cov[min(H - 1, gy), min(W - 1, gx)])

        rows.append({"stem": stem, "source": srcmap.get(stem, "?"),
                     "n_boxes": len(boxes), "best_iou": best_iou,
                     "center_in_box": center_in, "center_covered": center_cov,
                     "gt_pixel_coverage": float(gt_cov), "best_center_dist": float(best_cdist),
                     "gt_area": gt_area})

    if not rows:
        print("No rows.")
        return

    best = np.array([r["best_iou"] for r in rows])
    cov = np.array([r["gt_pixel_coverage"] for r in rows])
    cdist = np.array([r["best_center_dist"] for r in rows])
    print(f"n={len(rows)} | avg_boxes={np.mean([r['n_boxes'] for r in rows]):.1f}")
    print(f"center_in_box: {np.mean([r['center_in_box'] for r in rows]) * 100:.1f}%")
    print(f"center_covered_by_384crop: {np.mean([r['center_covered'] for r in rows]) * 100:.1f}%")
    print(f"GT pixel coverage (mean): {cov.mean() * 100:.1f}%")
    print(f"best IoU (mean): {best.mean():.4f}")
    for t in [0.1, 0.3, 0.5]:
        print(f"recall@IoU>={t}: {np.mean(best >= t) * 100:.1f}%")
    print(f"center distance (mean, normalized): {cdist.mean():.4f}")

    src = {}
    for r in rows:
        src.setdefault(r["source"], []).append(r)
    print("\n-- By source --")
    for k, v in sorted(src.items()):
        b = np.array([x["best_iou"] for x in v])
        c = np.array([x["gt_pixel_coverage"] for x in v])
        print(f"{k:>14} n={len(v):>3} bestIoU={b.mean():.3f} coverage={c.mean()*100:5.1f}%")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "proposal_recall.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print("\nSaved:", out / "proposal_recall.json")


if __name__ == "__main__":
    main()
