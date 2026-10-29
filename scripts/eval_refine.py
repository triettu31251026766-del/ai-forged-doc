import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb, place_max  # noqa: E402
from eval_coarse2fine import global_forward, soft_centroid  # noqa: E402
from eval_select_quick import local_peaks, win_score  # noqa: E402


def place_prob(img, seg, cx, cy, P, device):
    H, W = img.shape[:2]
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    full = np.zeros((H, W), dtype=np.float32)
    place_max(full, prob, int(cx), int(cy), P)
    return full


def bootstrap(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    n_ = len(a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_, n_)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.asarray(diffs)
    return {"mean_diff": float(a.mean() - b.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_diff_le_0": float((diffs <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--grid-stride", type=int, default=96)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/refine")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    gmodel = build_segmentation("resnet18").to(device)
    gmodel.load_state_dict(torch.load(ROOT / args.global_ckpt, map_location=device))
    gmodel.eval()
    seg = build_segmentation("resnet18").to(device)
    seg.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=device))
    seg.eval()

    items = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
    if args.max_test and args.max_test < len(items):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(items), size=args.max_test, replace=False)
        items = [items[i] for i in idx]

    names = ["peak", "centroid", "grid3"]
    st = {n: {"inter": 0, "union": 0, "per": []} for n in names}

    def add(n, pred, mask):
        inter = int((pred & mask).sum())
        union = int((pred | mask).sum())
        st[n]["inter"] += inter
        st[n]["union"] += union
        st[n]["per"].append(inter / max(union, 1))

    with torch.no_grad():
        for img_path, mask_path, _ in tqdm(items, desc="refine"):
            img = load_rgb(img_path)
            mask = load_mask(mask_path)
            prob384, scale, nw, nh, left, top = global_forward(gmodel, img, args.gsize, device)
            peaks = local_peaks(prob384, scale, nw, nh, left, top, k=1)
            if not peaks:
                continue
            cx, cy = peaks[0]

            base = place_prob(img, seg, cx, cy, args.patch_size, device)
            add("peak", base > 0.5, mask)

            c = soft_centroid(base)
            if c is None:
                add("centroid", base > 0.5, mask)
            else:
                ref = place_prob(img, seg, c[0], c[1], args.patch_size, device)
                add("centroid", ref > 0.5, mask)

            s = args.grid_stride
            best_map, best_score = base, win_score(base)
            for dy in (-s, 0, s):
                for dx in (-s, 0, s):
                    if dx == 0 and dy == 0:
                        continue
                    m = place_prob(img, seg, cx + dx, cy + dy, args.patch_size, device)
                    sc = win_score(m)
                    if sc > best_score:
                        best_map, best_score = m, sc
            add("grid3", best_map > 0.5, mask)

    print(f"\n=== Refinement ({len(items)} images) ===")
    for n in names:
        pooled = st[n]["inter"] / max(st[n]["union"], 1)
        mean = float(np.mean(st[n]["per"]))
        print(f"{n:10s}: IoU_pooled={pooled:.4f}  IoU_mean={mean:.4f}")
    for n in ["centroid", "grid3"]:
        print(f"{n:10s} vs peak (per-image): {json.dumps(bootstrap(st[n]['per'], st['peak']['per']))}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "iou_pooled": {n: st[n]["inter"] / max(st[n]["union"], 1) for n in names},
                   "iou_mean": {n: float(np.mean(st[n]["per"])) for n in names},
                   "per_image": {n: st[n]["per"] for n in names},
                   "bootstrap": {n: bootstrap(st[n]["per"], st["peak"]["per"])
                                 for n in ["centroid", "grid3"]}}, f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()
