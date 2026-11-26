import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def crc_pick(risk_mean, n, alpha):
    """CRC: largest lambda whose corrected risk is <= alpha (risk increases with lambda)."""
    adj = (n / (n + 1)) * risk_mean + 1 / (n + 1)
    ok = np.where(adj <= alpha)[0]
    return int(ok[-1]) if len(ok) else int(np.argmin(adj))


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
    cum_m = np.cumsum(HM[:, ::-1], axis=1)[:, ::-1]   # TP(lambda)
    cum_b = np.cumsum(HB[:, ::-1], axis=1)[:, ::-1]   # FP(lambda)
    L = (MS[:, None] - cum_m) / np.maximum(MS[:, None], 1)  # per-doc FNR over lambdas

    rng = np.random.default_rng(0)
    img_fnr, pix_fnr, pix_fpr, pix_doc = [], [], [], []

    for _ in range(args.n_splits):
        perm = rng.permutation(len(forged))
        cal, tst = forged[perm[: args.n_cal]], forged[perm[args.n_cal:]]

        # image level: loss_i(lambda) = 1[score_i < lambda]
        sc = scores[cal]
        risk_img = (sc[None, :] < lambdas[:, None]).mean(axis=1)
        j_img = crc_pick(risk_img, len(cal), args.alpha)
        img_fnr.append(float((scores[tst] < lambdas[j_img]).mean()))

        # document level: loss_i(lambda) = FN_i(lambda) / mask_i
        j_pix = crc_pick(L[cal].mean(axis=0), len(cal), args.alpha)
        fn_t = MS[tst] - cum_m[tst, j_pix]
        pix_doc.append(float(np.mean(fn_t / np.maximum(MS[tst], 1))))
        pix_fnr.append(float(fn_t.sum() / np.maximum(MS[tst].sum(), 1)))
        bg = NP[tst] - MS[tst]
        pix_fpr.append(float(cum_b[tst, j_pix].sum() / max(bg.sum(), 1)))

    res = {
        "alpha": args.alpha, "n_cal": args.n_cal, "n_splits": args.n_splits,
        "image_FNR_mean": float(np.mean(img_fnr)), "image_FNR_std": float(np.std(img_fnr)),
        "pixel_FNR_pooled_mean": float(np.mean(pix_fnr)), "pixel_FNR_pooled_std": float(np.std(pix_fnr)),
        "pixel_FNR_perdoc_mean": float(np.mean(pix_doc)), "pixel_FNR_perdoc_std": float(np.std(pix_doc)),
        "pixel_FPR_mean": float(np.mean(pix_fpr)), "pixel_FPR_std": float(np.std(pix_fpr)),
    }
    print(json.dumps(res, indent=2))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
