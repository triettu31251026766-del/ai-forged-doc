import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASK = ROOT / "data" / "raw" / "AIForge-Doc-v2" / "TestingSet" / "masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--list", default="results/eval_lists/gpt150.txt")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="results/trufor_eval.json")
    args = ap.parse_args()

    stems = [s.strip() for s in (ROOT / args.list).read_text(encoding="utf-8").splitlines() if s.strip()]
    npz_dir = Path(args.npz_dir)
    inter = union = tp = fp = fn = 0
    dice_sum = 0.0
    per = []
    scores = []
    used = 0
    maps = []
    for s in stems:
        f = npz_dir / f"{s}.png.npz"
        if not f.exists():
            f = npz_dir / f"{s}.npz"
        mpath = MASK / f"{s}.png"
        if not f.exists() or not mpath.exists():
            continue
        z = np.load(f)
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 0
        p = z["map"].astype(np.float32)
        if p.shape != m.shape:
            p = cv2.resize(p, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
        maps.append((p, m))
        pred = p > args.thr
        i = int(np.count_nonzero(pred & m))
        u = int(np.count_nonzero(pred | m))
        inter += i; union += u
        tp += i
        fp += int(np.count_nonzero(pred & ~m))
        fn += int(np.count_nonzero(~pred & m))
        dice_sum += 2 * i / max(int(pred.sum()) + int(m.sum()), 1)
        per.append(i / max(u, 1))
        if "score" in z:
            scores.append(float(np.asarray(z["score"]).ravel()[0]))
        used += 1

    best = {"thr": None, "f1": -1.0, "iou": 0.0}
    for t in np.arange(0.05, 0.96, 0.05):
        it = un = tpv = fpv = fnv = 0
        for p, m in maps:
            pr = p > float(t)
            i = int(np.count_nonzero(pr & m)); u = int(np.count_nonzero(pr | m))
            it += i; un += u
            tpv += i; fpv += int(np.count_nonzero(pr & ~m)); fnv += int(np.count_nonzero(~pr & m))
        f1 = 2 * tpv / max(2 * tpv + fpv + fnv, 1)
        if f1 > best["f1"]:
            best = {"thr": float(t), "f1": float(f1), "iou": float(it / max(un, 1))}

    res = {
        "n": used,
        "iou_pooled": inter / max(union, 1),
        "iou_mean": float(np.mean(per)) if per else 0.0,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "dice_mean": dice_sum / max(used, 1),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "best_threshold": best,
    }
    print(json.dumps(res, indent=2))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
