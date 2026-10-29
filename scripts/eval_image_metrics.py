import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import load_rgb  # noqa: E402
from eval_coarse2fine import global_forward  # noqa: E402


def score_image(prob, scale, nw, nh, left, top):
    p = prob[top:top + nh, left:left + nw]
    if p.size == 0:
        return 0.0
    v = np.sort(p.ravel())
    k = max(1, int(0.01 * v.size))
    return float(v[-k:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="test_in_domain,test_ideogram,test_gptimage")
    ap.add_argument("--gsize", type=int, default=384)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))
    model.eval()

    raw = build_protocol_splits("p1")
    res = {}
    with torch.no_grad():
        for split in args.splits.split(","):
            items = raw[split]
            scores, labels = [], []
            for ip, mp, lab in tqdm(items, desc=split, leave=False):
                img = load_rgb(ip)
                prob, scale, nw, nh, left, top = global_forward(model, img, args.gsize, dev)
                scores.append(score_image(prob, scale, nw, nh, left, top))
                labels.append(int(lab))
            y = np.asarray(labels)
            s = np.asarray(scores)
            auc = roc_auc_score(y, s)
            apv = average_precision_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            idx = np.searchsorted(fpr, 0.01, side="right") - 1
            r1 = float(tpr[max(idx, 0)])
            res[split] = {"n": int(len(y)), "pos": int(y.sum()),
                          "image_auc": float(auc), "image_ap": float(apv), "recall_at_1fpr": r1}
            print(f"{split}: n={len(y)} pos={int(y.sum())} AUC={auc:.4f} AP={apv:.4f} R@1%={r1:.4f}")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ckpt": args.ckpt, "results": res}, indent=2), encoding="utf-8")
        print("Saved:", out)


if __name__ == "__main__":
    main()
