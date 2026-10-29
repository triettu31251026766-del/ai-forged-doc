import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, bbox_map, crop_window, load_mask, load_rgb, place_max  # noqa: E402
from eval_coarse2fine import global_forward  # noqa: E402
from eval_select_quick import local_peaks, win_score  # noqa: E402
from src.data.splits import load_meta, V2  # noqa: E402


def place_prob(img, seg, cx, cy, P, device):
    H, W = img.shape[:2]
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, prob, int(cx), int(cy), P)
    return full


def win_box(cx, cy, P, H, W):
    x0 = max(0, int(cx) - P // 2); y0 = max(0, int(cy) - P // 2)
    x1 = min(W, int(cx) + P // 2); y1 = min(H, int(cy) + P // 2)
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
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

    raw = build_protocol_splits("p1")
    items = [it for it in raw[args.split] if it[2] == 1]
    if args.max_test and args.max_test < len(items):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(items), size=args.max_test, replace=False)
        items = [items[i] for i in idx]

    configs = []
    for K in (1, 2, 3):
        for a in (0.0, 0.3, 0.5, 0.7):
            configs.append((K, a, 0.5))
    for a in (0.3, 0.5, 0.7):
        for t in (0.4, 0.6):
            configs.append((3, a, t))
    names = [f"K{K}_a{a}_t{t}" for K, a, t in configs] + ["global"]

    st = {n: {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0, "dice": 0.0} for n in names}
    n_img = 0

    with torch.no_grad():
        for img_path, mask_path, _ in tqdm(items, desc="softgate"):
            img = load_rgb(img_path); H, W = img.shape[:2]
            mask = load_mask(mask_path)
            prob384, scale, nw, nh, left, top = global_forward(gmodel, img, args.gsize, device)
            Pg = cv2.resize(prob384[top:top + nh, left:left + nw], (W, H), interpolation=cv2.INTER_LINEAR)

            peaks = local_peaks(prob384, scale, nw, nh, left, top, k=3)
            if not peaks:
                continue
            maps = []
            for cx, cy in peaks:
                maps.append(place_prob(img, seg, cx, cy, args.patch_size, device))

            def add(name, pred, m):
                d = st[name]
                d["inter"] += int((pred & m).sum())
                d["union"] += int((pred | m).sum())
                d["tp"] += int((pred & m).sum())
                d["fp"] += int((pred & ~m).sum())
                d["fn"] += int((~pred & m).sum())
                d["dice"] += (2 * (pred & m).sum()) / (pred.sum() + m.sum() + 1e-6)

            for K, a, t in configs:
                Pp = maps[0].copy()
                for i in range(1, min(K, len(maps))):
                    np.maximum(Pp, maps[i], out=Pp)
                x0, y0, x1, y1 = W, H, 0, 0
                for cx, cy in peaks[:min(K, len(maps))]:
                    bx0, by0, bx1, by1 = win_box(cx, cy, args.patch_size, H, W)
                    x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
                pg = Pg[y0:y1, x0:x1]
                pp = Pp[y0:y1, x0:x1]
                mm = mask[y0:y1, x0:x1]
                with np.errstate(over="ignore", invalid="ignore"):
                    pf = pp if a == 0.0 else np.power(pg, a) * np.power(pp, 1.0 - a)
                add(f"K{K}_a{a}_t{t}", pf > t, mm)

            add("global", Pg > 0.5, mask)
            n_img += 1

    print(f"\n=== Soft-gate TL-HR ({len(items)} images, n={n_img}) ===")
    rows = []
    for n in names:
        d = st[n]
        iou = d["inter"] / max(d["union"], 1)
        f1 = 2 * d["tp"] / max(2 * d["tp"] + d["fp"] + d["fn"], 1)
        dice = d["dice"] / max(n_img, 1)
        rows.append((iou, n, f1, dice))
    for iou, n, f1, dice in sorted(rows, reverse=True):
        print(f"{n:14s}: IoU={iou:.4f} F1={f1:.4f} Dice={dice:.4f}")


if __name__ == "__main__":
    main()
