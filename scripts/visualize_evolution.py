import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_protocol_splits  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm(t):
    arr = t.permute(1, 2, 0).cpu().numpy()
    arr = arr * STD + MEAN
    return np.clip(arr, 0, 1)


def load_model(path, device):
    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / path, map_location=device))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero", required=True, help="checkpoint zero-shot (base)")
    ap.add_argument("--naive", required=True, help="checkpoint naive fine-tune")
    ap.add_argument("--ours", required=True, help="checkpoint calibration+replay")
    ap.add_argument("--ours-json", default=None, help="json co T cua ours")
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/xai/evolution")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1.0
    if args.ours_json:
        d = json.loads((ROOT / args.ours_json).read_text(encoding="utf-8"))
        T = float(d.get("T", 1.0))

    models = [("Zero-shot", load_model(args.zero, device), 1.0),
              ("Naive fine-tune", load_model(args.naive, device), 1.0),
              (f"Ours (T={T:.2f})", load_model(args.ours, device), T)]

    raw = build_protocol_splits("p1")
    forged = [it for it in raw[args.split] if it[2] == 1]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(forged), size=min(args.n, len(forged)), replace=False)
    items = [forged[i] for i in idx]
    loader = DataLoader(ItemsDataset(items, size=args.size), batch_size=1, shuffle=False, num_workers=0)

    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n))
    if n == 1:
        axes = np.array([axes])

    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = denorm(batch["image"][0])
            gt = batch["mask"][0, 0].numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
            axes[i, 0].set_title("Document + GT")

            for j, (name, model, temp) in enumerate(models, start=1):
                x = batch["image"].to(device)
                prob = torch.sigmoid(model(x) / temp)[0, 0].cpu().numpy()
                axes[i, j].imshow(prob, cmap="jet", vmin=0, vmax=1)
                axes[i, j].set_title(name)

            for j in range(4):
                axes[i, j].axis("off")

    plt.tight_layout()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_evolution.png"
    plt.savefig(path, dpi=140)
    print("Saved:", path)


if __name__ == "__main__":
    main()
