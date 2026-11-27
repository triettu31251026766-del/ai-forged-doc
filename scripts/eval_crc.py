import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def crc_pick(risk_mean, n, alpha):
    adj = (n / (n + 1)) * risk_mean + 1 / (n + 1)
    ok = np.where(adj <= alpha)[0]
    return int(ok[-1]) if len(ok) else int(np.argmin(adj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/cascade_gpt/test_gptimage.npz")
    ap.add_argument("--n-cals", default="25,50,100,300")
    ap.add_argument("--alphas", default="0.10,0.15")
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--out", default="results/crc_sweep.json")
    args = ap.parse_args()

    z = np.load(ROOT / args.npz, allow_pickle=True)
    scores = z["scores"].astype(float)
    labels = z["labels"].astype(int)
    lambdas = z["lambdas"].astype(float)
    HM, HB = z["hist_mask"].astype(float), z["hist_bg"].astype(float)
    MS, NP = z["mask_total"].astype(float), z["n_pixels"].astype(float)

    forged_all = np.where(labels == 1)[0]
    auth = np.where(labels == 0)[0]
    cum_m = np.cumsum(HM[:, ::-1], axis=1)[:, ::-1]
    cum_b = np.cumsum(HB[:, ::-1], axis=1)[:, ::-1]
    L = (MS[:, None] - cum_m) / np.maximum(MS[:, None], 1)

    rng = np.random.default_rng(0)
    rows = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        for n_cal in [int(n) for n in args.n_cals.split(",")]:
            img_fnr, img_fpr, pix_fnr, pix_fpr = [], [], [], []
            for _ in range(args.n_splits):
                perm = rng.permutation(len(forged_all))
                cal = forged_all[perm[:n_cal]]
                tst = forged_all[perm[n_cal:]]
                if len(tst) == 0:
                    continue

                risk_img = (scores[cal][None, :] < lambdas[:, None]).mean(axis=1)
                j_img = crc_pick(risk_img, len(cal), alpha)
                img_fnr.append(float((scores[tst] < lambdas[j_img]).mean()))
                img_fpr.append(float((scores[auth] > lambdas[j_img]).mean()))

                j_pix = crc_pick(L[cal].mean(axis=0), len(cal), alpha)
                fn_t = MS[tst] - cum_m[tst, j_pix]
                pix_fnr.append(float(np.mean(fn_t / np.maximum(MS[tst], 1))))
                bg = NP[tst] - MS[tst]
                pix_fpr.append(float(cum_b[tst, j_pix].sum() / max(bg.sum(), 1)))

            rows.append({
                "alpha": alpha, "n_cal": n_cal,
                "image_FNR": float(np.mean(img_fnr)), "image_FNR_std": float(np.std(img_fnr)),
                "image_FPR": float(np.mean(img_fpr)), "image_FPR_std": float(np.std(img_fpr)),
                "pixel_FNR": float(np.mean(pix_fnr)), "pixel_FNR_std": float(np.std(pix_fnr)),
                "pixel_FPR": float(np.mean(pix_fpr)),
            })
            print(json.dumps(rows[-1]))

    (ROOT / args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
