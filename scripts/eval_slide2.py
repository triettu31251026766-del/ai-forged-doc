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
from src.train_box_cls import MEAN, STD, bbox_map, crop_window, load_mask, load_rgb  # noqa: E402
from eval_coarse2fine import global_forward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/slide2")
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

    thrs = [5, 10, 20, 40, 80, 999999]
    agg = {t: {"inter": 0, "union": 0, "per": [], "nwin": []} for t in thrs}
    orc = {t: [] for t in thrs}

    P = args.patch_size; S = args.stride
    with torch.no_grad():
        for ip, mp, _ in tqdm(items, desc="slide2"):
            img = load_rgb(ip); mask = load_mask(mp)
            H, W = img.shape[:2]
            pr, scale, nw, nh, left, top = global_forward(g, img, args.gsize, dev)
            Pg = cv2.resize(pr[top:top + nh, left:left + nw], (W, H), interpolation=cv2.INTER_LINEAR)

            cand = []
            for cy in range(P // 2, H, S):
                for cx in range(P // 2, W, S):
                    cand.append((cx, cy, float(Pg[cy, cx])))
            cand.sort(key=lambda z: -z[2])

            cands = []
            for cx, cy, pc in cand:
                crop = crop_window(img, cx, cy, P)
                arr = (crop - MEAN) / STD
                x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
                prob = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
                v = np.sort(prob.ravel())
                sc = float(v[-max(1, int(0.01 * v.size)):].mean())
                pred = prob > 0.5
                x0 = cx - P // 2; y0 = cy - P // 2
                gx0, gy0 = max(0, x0), max(0, y0)
                gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
                full = np.zeros((H, W), dtype=bool)
                full[gy0:gy1, gx0:gx1] = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
                cands.append((sc, pc, full))
            if not cands:
                continue

            for t in thrs:
                sel = cands[:int(t)]
                if not sel:
                    agg[t]["per"].append(0.0); orc[t].append(0.0); agg[t]["nwin"].append(0)
                    continue
                best = max(sel, key=lambda z: z[0])
                inter = int((best[2] & mask).sum()); union = int((best[2] | mask).sum())
                agg[t]["inter"] += inter; agg[t]["union"] += union
                agg[t]["per"].append(inter / max(union, 1))
                agg[t]["nwin"].append(len(sel))
                oi = 0.0
                for c in sel:
                    oi = max(oi, (c[2] & mask).sum() / max((c[2] | mask).sum(), 1))
                orc[t].append(float(oi))

    print(f"\n=== Coarse-guided sliding ({len(items)} imgs, stride {S}) ===")
    for t in thrs:
        pooled = agg[t]["inter"] / max(agg[t]["union"], 1)
        print(f"K={t}: IoU_pooled={pooled:.4f}  IoU_mean={np.mean(agg[t]['per']):.4f}"
              f"  oracle_win={np.mean(orc[t]):.4f}  win/img={np.mean(agg[t]['nwin']):.1f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "rows": {str(t): {"iou_pooled": agg[t]["inter"] / max(agg[t]["union"], 1),
                                     "iou_mean": float(np.mean(agg[t]["per"])),
                                     "oracle_win": float(np.mean(orc[t])),
                                     "avg_windows": float(np.mean(agg[t]["nwin"]))} for t in thrs}},
                  f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()


