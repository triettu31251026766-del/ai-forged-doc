import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import V2, build_fewshot_splits, build_protocol_splits, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import MEAN, STD, bbox_map, crop_window, load_mask, load_rgb  # noqa: E402

FEATS = ["top1", "top5", "max", "meanpred", "area", "mass", "border", "centoff", "compact"]


def window_features(prob):
    P = prob.shape[0]
    v = np.sort(prob.ravel())
    n = v.size
    k1 = max(1, int(0.01 * n))
    k5 = max(1, int(0.05 * n))
    pred = prob > 0.5
    area = int(pred.sum())
    ys, xs = np.nonzero(pred)
    if area > 0:
        cy, cx = ys.mean(), xs.mean()
        centoff = float(np.hypot(cx - P / 2, cy - P / 2) / (P / 2))
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        compact = area / float(h * w)
        b = 32
        border = (pred[:b, :].sum() + pred[-b:, :].sum() + pred[:, :b].sum() + pred[:, -b:].sum())
        border = float(border / area)
        meanpred = float(prob[pred].mean())
    else:
        centoff, compact, border, meanpred = 1.0, 0.0, 0.0, 0.0
    return [
        float(v[-k1:].mean()),
        float(v[-k5:].mean()),
        float(v[-1]),
        meanpred,
        area / float(n),
        float(prob.mean()),
        border,
        centoff,
        compact,
    ]


def win_iou(pred, mask, mask_area, cx, cy, P, H, W):
    x0 = cx - P // 2; y0 = cy - P // 2
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(W, x0 + P), min(H, y0 + P)
    pc = pred[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    mc = mask[gy0:gy1, gx0:gx1]
    inter = int(np.count_nonzero(pc & mc))
    union = int(np.count_nonzero(pc)) + mask_area - inter
    return inter / max(union, 1)


def image_features(model, img, mask, P, stride, dev):
    H, W = img.shape[:2]
    mask = mask > 0
    mask_area = int(mask.sum())
    feats, ious = [], []
    with torch.no_grad():
        for cy in range(P // 2, H, stride):
            for cx in range(P // 2, W, stride):
                crop = crop_window(img, cx, cy, P)
                arr = (crop - MEAN) / STD
                x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(dev)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                feats.append(window_features(prob))
                ious.append(win_iou(prob > 0.5, mask, mask_area, cx, cy, P, H, W))
    return np.asarray(feats, np.float32), np.asarray(ious, np.float32)


def rank_norm(f):
    r = np.argsort(np.argsort(f, axis=0), axis=0).astype(np.float32)
    return r / max(len(f) - 1, 1)


def collect(model, items, P, stride, dev, desc):
    X, Y, names = [], [], []
    for ip, mp, _ in tqdm(items, desc=desc):
        f, y = image_features(model, load_rgb(ip), load_mask(mp), P, stride, dev)
        X.append(f); Y.append(y); names.append(Path(ip).stem)
    return X, Y, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-ckpt", default="results/patch_seg_gpt/patch_seg.pth")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-gpt", type=int, default=150)
    ap.add_argument("--max-ideo", type=int, default=120)
    ap.add_argument("--out", default="results/ranker")
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=dev)); model.eval()

    cache = out / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Xtr, Ytr = list(z["Xtr"]), list(z["Ytr"])
        Xg, Yg = list(z["Xg"]), list(z["Yg"])
        Xi, Yi = list(z["Xi"]), list(z["Yi"])
        print("loaded cache")
    else:
        raw = build_fewshot_splits(args.k, seed=args.seed)
        v2 = load_meta(V2 / "metadata.jsonl")
        gpt_box = bbox_map(v2, "training")
        items = []
        for img, msk, lab in raw["train"]:
            if lab == 1 and gpt_box.get(Path(img).stem) is not None:
                items.append((img, msk, 1))
        rng = np.random.default_rng(args.seed)
        if args.max_train and args.max_train < len(items):
            idx = rng.choice(len(items), size=args.max_train, replace=False)
            items = [items[i] for i in idx]
        Xtr, Ytr, _ = collect(model, items, args.patch_size, args.stride, dev, "train-feats")

        gpt = [it for it in build_protocol_splits("p1")["test_gptimage"] if it[2] == 1]
        if args.max_gpt and args.max_gpt < len(gpt):
            idx = rng.choice(len(gpt), size=args.max_gpt, replace=False)
            gpt = [gpt[i] for i in idx]
        Xg, Yg, _ = collect(model, gpt, args.patch_size, args.stride, dev, "gpt-feats")

        ideo = [it for it in build_protocol_splits("p1")["test_ideogram"] if it[2] == 1]
        if args.max_ideo and args.max_ideo < len(ideo):
            idx = rng.choice(len(ideo), size=args.max_ideo, replace=False)
            ideo = [ideo[i] for i in idx]
        Xi, Yi, _ = collect(model, ideo, args.patch_size, args.stride, dev, "ideo-feats")

        np.savez_compressed(cache, Xtr=np.array(Xtr, dtype=object), Ytr=np.array(Ytr, dtype=object),
                            Xg=np.array(Xg, dtype=object), Yg=np.array(Yg, dtype=object),
                            Xi=np.array(Xi, dtype=object), Yi=np.array(Yi, dtype=object))
        print("saved cache", cache)

    if args.cache_only:
        return

    Xtr_r = np.concatenate([rank_norm(f) for f in Xtr], 0)
    Ytr_f = np.concatenate(Ytr, 0)

    models = {
        "ridge": Ridge(alpha=1.0, random_state=0),
        "gbm": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=0),
    }
    for m in models.values():
        m.fit(Xtr_r, Ytr_f)
    with open(out / "ranker.pkl", "wb") as f:
        pickle.dump({"models": models, "feats": FEATS, "norm": "percentile"}, f)

    def evaluate(X, Y, tag):
        sel_top1, sel_oracle = [], []
        preds = {n: [] for n in models}
        for f, y in zip(X, Y):
            fr = rank_norm(f)
            j = int(np.argmax(f[:, 0]))
            sel_top1.append(float(y[j]))
            sel_oracle.append(float(y.max()))
            for n, m in models.items():
                k = int(np.argmax(m.predict(fr)))
                preds[n].append(float(y[k]))
        print(f"\n[{tag}] n={len(X)}")
        print(f"  top1-score     IoU_mean={np.mean(sel_top1):.4f}")
        for n in models:
            print(f"  {n:12s}   IoU_mean={np.mean(preds[n]):.4f}")
        print(f"  oracle         IoU_mean={np.mean(sel_oracle):.4f}")
        return {n: float(np.mean(v)) for n, v in preds.items()} | {
            "top1": float(np.mean(sel_top1)), "oracle": float(np.mean(sel_oracle))}

    r_gpt = evaluate(Xg, Yg, "GPT-150")
    r_ideo = evaluate(Xi, Yi, "Ideogram-120")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"gpt": r_gpt, "ideogram": r_ideo}, f, indent=2)
    print("saved", out / "results.json")


if __name__ == "__main__":
    main()
