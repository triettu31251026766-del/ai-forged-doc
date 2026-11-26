import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def crc_threshold(losses, alpha):
    n = len(losses)
    cands = np.unique(np.concatenate([losses, [0.0, 1.0]]))
    for lam in cands:
        risk = (n / (n + 1)) * np.mean(losses <= lam) + 1 / (n + 1)
        if risk <= alpha:
            return float(lam), float(risk)
    return float(cands[-1]), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/cascade_gpt/test_gptimage.npz")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--out", default="results/crc_summary.json")
    args = ap.parse_args()

    z = np.load(ROOT / args.npz, allow_pickle=True)
    scores = z["scores"].astype(float)
    labels = z["labels"].astype(int)
    lambdas = z["lambdas"].astype(float)
    HM, HB = z["hist_mask"].astype(float), z["hist_bg"].astype(float)
    MS, NP = z["mask_total"].astype(float), z["n_pixels"].astype(float)

    forged = np.where(labels == 1)[0]
    rng = np.random.default_rng(0)
    img_fnr, pix_fnr, pix_fpr = [], [], []
    for _ in range(args.n_splits):
        perm = rng.permutation(len(forged))
        cal, tst = forged[perm[: args.n_cal]], forged[perm[args.n_cal:]]

        # ---- image level: loss = 1 if score < lambda (missed) ----
        losses = scores[cal]
        lam_img, _ = crc_threshold(losses, args.alpha)
        img_fnr.append(float((scores[tst] < lam_img).mean()))

        # ---- pixel (document) level: loss_i(lambda) = FN_i(lambda)/mask_i ----
        cum_m = np.cumsum(HM[:, ::-1], axis=1)[:, ::-1]
        tp_cal = cum_m[cal]
        fn_cal = MS[cal][:, None] - tp_cal
        L = fn_cal / MS[cal][:, None]
        mean_risk = L.mean(axis=0)
        n = len(cal)
        adj = (n / (n + 1)) * mean_risk + 1 / (n + 1)
        idx = np.where(adj <= args.alpha)[0]
        j = int(idx[0]) if len(idx) else int(np.argmin(adj))
        lam_pix = float(lambdas[j])

        tp_t = cum_m[tst, j]
        fn_t = MS[tst] - tp_t
        pix_fnr.append(float(fn_t.sum() / MS[tst].sum()))
        cum_b = np.cumsum(HB[:, ::-1], axis=1)[:, ::-1]
        fp_t = cum_b[tst, j]
        bg = NP[tst] - MS[tst]
        pix_fpr.append(float(fp_t.sum() / max(bg.sum(), 1)))

    res = {
        "alpha": args.alpha, "n_cal": args.n_cal, "n_splits": args.n_splits,
        "image_FNR_mean": float(np.mean(img_fnr)), "image_FNR_std": float(np.std(img_fnr)),
        "pixel_FNR_mean": float(np.mean(pix_fnr)), "pixel_FNR_std": float(np.std(pix_fnr)),
        "pixel_FPR_mean": float(np.mean(pix_fpr)), "pixel_FPR_std": float(np.std(pix_fpr)),
    }
    print(json.dumps(res, indent=2))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
