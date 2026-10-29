import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, bbox_map, crop_window, load_mask, load_rgb, place_max  # noqa: E402
from eval_coarse2fine import global_forward  # noqa: E402
from eval_select_quick import local_peaks, win_score  # noqa: E402


def patch_map(seg, img, cx, cy, P, dev):
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    H, W = img.shape[:2]
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, prob, int(cx), int(cy), P)
    return full, win_score(prob)


def forward_square(model, crop, size, dev):
    c = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = (c - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def zoom_center(model, img, cx, cy, win, size, dev, smooth=5):
    crop = crop_window(img, int(cx), int(cy), win)
    p = forward_square(model, crop, size, dev)
    p = cv2.GaussianBlur(p, (smooth, smooth), 0)
    py, px = np.unravel_index(int(p.argmax()), p.shape)
    s = win / size
    return cx - win // 2 + px * s, cy - win // 2 + py * s


def tile_peaks(model, img, size, dev, frac=0.6, smooth=7):
    H, W = img.shape[:2]
    tw, th = int(W * frac), int(H * frac)
    xs = sorted({0, W - tw})
    ys = sorted({0, H - th})
    cand = []
    for y0 in ys:
        for x0 in xs:
            crop = img[y0:y0 + th, x0:x0 + tw]
            p, scale, nw, nh, left, top = global_forward(model, crop, size, dev)
            p = cv2.GaussianBlur(p, (smooth, smooth), 0)
            p[:top, :] = 0.0
            p[top + nh:, :] = 0.0
            p[:, :left] = 0.0
            p[:, left + nw:] = 0.0
            py, px = np.unravel_index(int(p.argmax()), p.shape)
            cx = x0 + (px + 0.5 - left) / scale
            cy = y0 + (py + 0.5 - top) / scale
            cand.append((float(p.max()), cx, cy))
    cand.sort(reverse=True)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/localize")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    g = build_segmentation("resnet18").to(dev)
    g.load_state_dict(torch.load(ROOT / args.global_ckpt, map_location=dev)); g.eval()
    seg = build_segmentation("resnet18").to(dev)
    seg.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=dev)); seg.eval()

    meta = load_meta(ROOT / "data" / "raw" / "AIForge-Doc-v2" / "metadata.jsonl")
    gtb = bbox_map(meta, "testing")
    items = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
    if args.max_test and args.max_test < len(items):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(items), size=args.max_test, replace=False)
        items = [items[i] for i in idx]

    names = ["peak", "zoom2", "zoom3", "k3sel", "grid3", "tile2", "tile2g3", "grid5", "gt"]
    st = {n: {"inter": 0, "union": 0, "per": [], "err": []} for n in names}

    def add(n, pred, mask, cx, cy, gx, gy):
        d = st[n]
        d["inter"] += int((pred & mask).sum())
        d["union"] += int((pred | mask).sum())
        d["per"].append(float((pred & mask).sum()) / max(int((pred | mask).sum()), 1))
        d["err"].append(float(np.hypot(cx - gx, cy - gy)))

    P = args.patch_size
    with torch.no_grad():
        for ip, mp, _ in tqdm(items, desc="localize"):
            img = load_rgb(ip); mask = load_mask(mp)
            b = gtb.get(Path(ip).stem)
            if b is None:
                continue
            gx, gy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

            pr, scale, nw, nh, left, top = global_forward(g, img, args.gsize, dev)
            peaks = local_peaks(pr, scale, nw, nh, left, top, k=3)
            if not peaks:
                continue

            cx, cy = peaks[0]
            full, _ = patch_map(seg, img, cx, cy, P, dev)
            add("peak", full > 0.5, mask, cx, cy, gx, gy)

            zx, zy = zoom_center(g, img, cx, cy, 768, args.gsize, dev)
            full, _ = patch_map(seg, img, zx, zy, P, dev)
            add("zoom2", full > 0.5, mask, zx, zy, gx, gy)

            z2x, z2y = zoom_center(g, img, zx, zy, 768, args.gsize, dev)
            full, _ = patch_map(seg, img, z2x, z2y, P, dev)
            add("zoom3", full > 0.5, mask, z2x, z2y, gx, gy)

            best = None
            for (px, py) in peaks:
                full, s = patch_map(seg, img, px, py, P, dev)
                if best is None or s > best[0]:
                    best = (s, full, px, py)
            add("k3sel", best[1] > 0.5, mask, best[2], best[3], gx, gy)

            step = 96
            bm, bs = None, -1e9
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    fm, s = patch_map(seg, img, cx + dx, cy + dy, P, dev)
                    if s > bs:
                        bm, bs = fm, s
            add("grid3", bm > 0.5, mask, cx, cy, gx, gy)

            tpk = tile_peaks(g, img, args.gsize, dev)
            tx, ty = tpk[0][1], tpk[0][2]
            full, _ = patch_map(seg, img, tx, ty, P, dev)
            add("tile2", full > 0.5, mask, tx, ty, gx, gy)

            bm, bs = None, -1e9
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    fm, s = patch_map(seg, img, tx + dx, ty + dy, P, dev)
                    if s > bs:
                        bm, bs = fm, s
            add("tile2g3", bm > 0.5, mask, tx, ty, gx, gy)

            bm, bs = None, -1e9
            for dy in (-2 * step, -step, 0, step, 2 * step):
                for dx in (-2 * step, -step, 0, step, 2 * step):
                    fm, s = patch_map(seg, img, cx + dx, cy + dy, P, dev)
                    if s > bs:
                        bm, bs = fm, s
            add("grid5", bm > 0.5, mask, cx, cy, gx, gy)

            full, _ = patch_map(seg, img, gx, gy, P, dev)
            add("gt", full > 0.5, mask, gx, gy, gx, gy)

    print(f"\n=== Localization study ({len(items)} images) ===")
    rows = []
    for n in names:
        d = st[n]
        rows.append((d["inter"] / max(d["union"], 1), float(np.mean(d["per"])),
                     float(np.median(d["err"])), n))
    for pooled, mean, err, n in sorted(rows, reverse=True):
        print(f"{n:8s}: IoU_pooled={pooled:.4f}  IoU_mean={mean:.4f}  peak_err_med={err:.0f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "methods": {n: {"iou_pooled": st[n]["inter"] / max(st[n]["union"], 1),
                                   "iou_mean": float(np.mean(st[n]["per"])),
                                   "peak_err_med": float(np.median(st[n]["err"])),
                                   "per_image": st[n]["per"]} for n in names}}, f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()
