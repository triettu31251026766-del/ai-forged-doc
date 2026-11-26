import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import load_rgb  # noqa: E402
from eval_coarse2fine import global_forward  # noqa: E402


def img_score(prob, scale, nw, nh, left, top):
    p = prob[top:top + nh, left:left + nw]
    v = np.sort(p.ravel())
    return float(v[-max(1, int(0.01 * v.size)):].mean())


def scores(model, items, dev):
    out = []
    with torch.no_grad():
        for ip, _, lab in items:
            prob, scale, nw, nh, left, top = global_forward(model, load_rgb(ip), 384, dev)
            out.append((img_score(prob, scale, nw, nh, left, top), int(lab)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/fewshot_multi/k100_s0.pth")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="results/risk_coverage.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    raw = build_protocol_splits("p1")
    cal = scores(model, raw["test_in_domain"], dev)      # Gemini calibration
    tgt = scores(model, raw["test_gptimage"], dev)       # GPT deployment

    f_cal = np.array([s for s, y in cal if y == 1])
    a_cal = np.array([s for s, y in cal if y == 0])
    thr = float(np.quantile(f_cal, args.alpha))          # conformal: FNR <= alpha

    f_tgt = np.array([s for s, y in tgt if y == 1])
    a_tgt = np.array([s for s, y in tgt if y == 0])

    fnr = float((f_tgt <= thr).mean())
    fpr = float((a_tgt > thr).mean())
    cov = float((f_tgt > thr).mean())
    res = {"threshold": thr, "alpha_cal": args.alpha,
           "target_FNR_on_GPT": fnr, "FPR_on_GPT_authentic": fpr,
           "coverage_of_forged": cov,
           "calib_FPR": float((a_cal > thr).mean()),
           "n_cal": len(f_cal), "n_target": len(f_tgt)}
    print(json.dumps(res, indent=2))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
