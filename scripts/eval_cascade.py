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
from src.train_box_cls import MEAN, STD, crop_window, load_mask, load_rgb  # noqa: E402


def top1_score(prob):
    v = np.sort(prob.ravel())
    k = max(1, int(0.01 * v.size))
    return float(v[-k:].mean())


def run_window(model, img, cx, cy, P, dev):
    crop = crop_window(img, int(cx), int(cy), P)
    arr = (crop - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob, top1_score(prob)


def iou_at(prob, mask, mask_area, cx, cy, P, H, W):
    pred = prob > 0.5
    x0 = cx - P // 2; y0 = cy - P // 2
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
    pc = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    mc = mask[gy0:gy1, gx0:gx1]
    inter = int(np.count_nonzero(pc & mc))
    union = int(np.count_nonzero(pc)) + mask_area - inter
    return inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--stride1", type=int, default=192)
    ap.add_argument("--stride2", type=int, default=96)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cascade")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=dev)); model.eval()

    items = [it for it in build_protocol_splits("p1")[args.split] if it[2] == 1]
    rng = np.random.default_rng(args.seed)
    n_take = len(items) if args.max_test <= 0 else min(args.max_test, len(items))
    idx = rng.choice(len(items), size=n_take, replace=False)
    items = [items[i] for i in idx]

    K = [1, 2, 3]
    st = {k: {"inter": 0, "union": 0, "per": [], "nwin": []} for k in K}
    st["oracle"] = {"per": [], "nwin": []}

    P = args.patch_size
    with torch.no_grad():
        for ip, mp, _ in tqdm(items, desc="cascade"):
            img = load_rgb(ip); mask = load_mask(mp) > 0
            H, W = img.shape[:2]
            mask_area = int(mask.sum())

            coarse = []
            for cy in range(P // 2, H, args.stride1):
                for cx in range(P // 2, W, args.stride1):
                    prob, s = run_window(model, img, cx, cy, P, dev)
                    coarse.append((s, cx, cy))
            coarse.sort(key=lambda z: -z[0])
            n_stage1 = len(coarse)

            groups = []
            seen = set()
            for s, cx, cy in coarse[:3]:
                group = []
                for dy in (-args.stride2, 0, args.stride2):
                    for dx in (-args.stride2, 0, args.stride2):
                        nx, ny = cx + dx, cy + dy
                        key = (nx // 8, ny // 8)
                        if key in seen:
                            continue
                        seen.add(key)
                        prob, s2 = run_window(model, img, nx, ny, P, dev)
                        group.append((s2, nx, ny, iou_at(prob, mask, mask_area, nx, ny, P, H, W)))
                groups.append(group)
            refined = [w for g in groups for w in g]
            n_used = n_stage1 + len(refined)
            best_all = max(r[3] for r in refined) if refined else 0.0

            for k in K:
                subset = [w for g in groups[:k] for w in g]
                if not subset:
                    continue
                best = max(subset, key=lambda z: z[0])
                st[k]["per"].append(best[3])
                st[k]["nwin"].append(n_stage1 + len(subset))
            st["oracle"]["per"].append(best_all)
            st["oracle"]["nwin"].append(n_used)

    print(f"\n=== Cascade ({len(items)} imgs, stride1={args.stride1}, stride2={args.stride2}) ===")
    print(f"  full dense reference (B44, same 150 imgs): 0.583")
    for k in K:
        print(f"  top-{k}: IoU_mean={np.mean(st[k]['per']):.4f}  windows/img={np.mean(st[k]['nwin']):.0f}")
    print(f"  oracle(visited): IoU_mean={np.mean(st['oracle']['per']):.4f}  windows/img={np.mean(st['oracle']['nwin']):.0f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(args),
           "rows": {f"top{k}": {"iou_mean": float(np.mean(st[k]["per"])),
                                "windows": float(np.mean(st[k]["nwin"]))} for k in K},
           "oracle": {"iou_mean": float(np.mean(st["oracle"]["per"])),
                      "windows": float(np.mean(st["oracle"]["nwin"]))}}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()

