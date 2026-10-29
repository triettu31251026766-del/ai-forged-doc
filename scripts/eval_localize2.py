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


def coarse_tta(g, img, size, dev):
    p1, scale, nw, nh, left, top = global_forward(g, img, size, dev)
    p2, *_ = global_forward(g, np.ascontiguousarray(img[:, ::-1]), size, dev)
    p2 = p2[:, ::-1]
    if p1.shape != p2.shape:
        return p1, scale, nw, nh, left, top
    return 0.5 * (p1 + p2), scale, nw, nh, left, top


def grid_best(seg, img, cx, cy, P, dev, r=1, step=96):
    bm, bs = None, -1e9
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            fm, s = patch_map(seg, img, cx + dx * step, cy + dy * step, P, dev)
            if s > bs:
                bm, bs = fm, s
    return bm


def pcentroid(prob, thr=0.3):
    H, W = prob.shape
    w = np.clip(prob - thr, 0, None)
    if w.sum() <= 1e-6:
        return W / 2, H / 2
    ys, xs = np.mgrid[0:H, 0:W]
    return float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/localize2")
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
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(items), size=(len(items) if args.max_test <= 0 else min(args.max_test, len(items))), replace=False)
    items = [items[i] for i in idx]

    names = ["peak", "peak_tta", "gcent", "grid7", "tta_g7", "gcent_g5"]
    st = {n: {"inter": 0, "union": 0, "per": [], "err": []} for n in names}

    def add(n, pred, mask, cx, cy, gx, gy):
        d = st[n]
        d["inter"] += int((pred & mask).sum())
        d["union"] += int((pred | mask).sum())
        d["per"].append(float((pred & mask).sum()) / max(int((pred | mask).sum()), 1))
        d["err"].append(float(np.hypot(cx - gx, cy - gy)))

    P = args.patch_size
    with torch.no_grad():
        for ip, mp, _ in tqdm(items, desc="loc2"):
            img = load_rgb(ip); mask = load_mask(mp)
            b = gtb.get(Path(ip).stem)
            if b is None:
                continue
            gx, gy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            H, W = img.shape[:2]

            pr, scale, nw, nh, left, top = global_forward(g, img, args.gsize, dev)
            pk = local_peaks(pr, scale, nw, nh, left, top, k=1)
            if not pk:
                continue
            cx, cy = pk[0]
            fm, _ = patch_map(seg, img, cx, cy, P, dev)
            add("peak", fm > 0.5, mask, cx, cy, gx, gy)

            pt, sc, nwt, nht, lft, tpt = coarse_tta(g, img, args.gsize, dev)
            pkt = local_peaks(pt, sc, nwt, nht, lft, tpt, k=1)
            tx, ty = pkt[0]
            fm, _ = patch_map(seg, img, tx, ty, P, dev)
            add("peak_tta", fm > 0.5, mask, tx, ty, gx, gy)

            Pg = cv2.resize(pr[top:top + nh, left:left + nw], (W, H), interpolation=cv2.INTER_LINEAR)
            ux, uy = pcentroid(Pg)
            fm, _ = patch_map(seg, img, ux, uy, P, dev)
            add("gcent", fm > 0.5, mask, ux, uy, gx, gy)

            fm = grid_best(seg, img, cx, cy, P, dev, r=3)
            add("grid7", fm > 0.5, mask, cx, cy, gx, gy)

            fm = grid_best(seg, img, tx, ty, P, dev, r=3)
            add("tta_g7", fm > 0.5, mask, tx, ty, gx, gy)

            fm = grid_best(seg, img, ux, uy, P, dev, r=2)
            add("gcent_g5", fm > 0.5, mask, ux, uy, gx, gy)

    print(f"\n=== Localize v2 ({len(items)} images) ===")
    rows = [(st[n]["inter"] / max(st[n]["union"], 1), float(np.mean(st[n]["per"])),
             float(np.median(st[n]["err"])), n) for n in names]
    for pooled, mean, err, n in sorted(rows, reverse=True):
        print(f"{n:10s}: IoU_pooled={pooled:.4f}  IoU_mean={mean:.4f}  err_med={err:.0f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "methods": {n: {"iou_pooled": st[n]["inter"] / max(st[n]["union"], 1),
                                   "iou_mean": float(np.mean(st[n]["per"])),
                                   "err_med": float(np.median(st[n]["err"])),
                                   "per_image": st[n]["per"]} for n in names}}, f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()

