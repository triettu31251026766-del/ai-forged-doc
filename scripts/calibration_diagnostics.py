import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402


def ece(probs, labels, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(probs)) * abs(labels[m].mean() - probs[m].mean())
    return float(e)


def reliability(probs, labels, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf.append(float(probs[m].mean()))
        acc.append(float(labels[m].mean()))
    return conf, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None, help="file json co T (temperature)")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-pixels", type=int, default=2000)
    ap.add_argument("--out", default="results/calib_diag")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1.0
    if args.json:
        d = json.loads((ROOT / args.json).read_text(encoding="utf-8"))
        T = float(d.get("T", 1.0))
    print(f"Checkpoint: {args.ckpt} | T={T:.3f} | split={args.split}")

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    items = raw[args.split]
    if args.max_test:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(items), size=min(args.max_test, len(items)), replace=False)
        items = [items[i] for i in idx]
    loader = DataLoader(ItemsDataset(items, size=args.size), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    rng = np.random.default_rng(0)
    img_probs = {1.0: [], T: []}
    img_labels = []
    pix_probs = {1.0: [], T: []}
    pix_labels = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        y = batch["label"].numpy()
        mask = batch["mask"]

        prob1 = torch.sigmoid(logits)
        prob2 = torch.sigmoid(logits / T) if T != 1.0 else prob1

        img_probs[1.0].append(prob1.flatten(1).max(dim=1).values.cpu().numpy())
        img_probs[T].append(prob2.flatten(1).max(dim=1).values.cpu().numpy())
        img_labels.append(y)

        for i in range(logits.shape[0]):
            if y[i] != 1:
                continue
            m = (mask[i, 0].flatten().numpy() > 0.5)
            n = min(args.n_pixels, m.size)
            idx = rng.choice(m.size, n, replace=False)
            pix_labels.append(m[idx])
            pix_probs[1.0].append(prob1[i, 0].flatten().cpu().numpy()[idx])
            pix_probs[T].append(prob2[i, 0].flatten().cpu().numpy()[idx])

    y_img = np.concatenate(img_labels)
    y_pix = np.concatenate(pix_labels)

    s_img1 = np.concatenate(img_probs[1.0])
    s_img2 = np.concatenate(img_probs[T])
    s_pix1 = np.concatenate(pix_probs[1.0])
    s_pix2 = np.concatenate(pix_probs[T])

    ece_img = {1.0: ece(s_img1, y_img, args.n_bins), T: ece(s_img2, y_img, args.n_bins)}
    ece_pix = {1.0: ece(s_pix1, y_pix, args.n_bins), T: ece(s_pix2, y_pix, args.n_bins)}

    print(f"ECE image-level: before={ece_img[1.0]:.4f}  after={ece_img[T]:.4f}")
    print(f"ECE pixel-level: before={ece_pix[1.0]:.4f}  after={ece_pix[T]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t, title in [(axes[0], 1.0, "Before calibration (T=1)"),
                         (axes[1], T, f"After calibration (T={T:.2f})")]:
        conf, acc = reliability(s_img1 if t == 1.0 else s_img2, y_img, args.n_bins)
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.plot(conf, acc, "o-", label="model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE={ece_img[t]:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_reliability.png"
    plt.savefig(path, dpi=150)
    print("Saved:", path)


if __name__ == "__main__":
    main()
