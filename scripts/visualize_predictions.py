import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    raw = build_protocol_splits("p1")
    forged = [it for it in raw[args.split] if it[2] == 1]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(forged), size=min(args.n, len(forged)), replace=False)
    items = [forged[i] for i in idx]
    ds = ItemsDataset(items, size=args.size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    fig, axes = plt.subplots(len(items), 3, figsize=(11, 3.6 * len(items)))
    if len(items) == 1:
        axes = np.array([axes])

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            gt = batch["mask"][0, 0].numpy()
            img = denorm(batch["image"][0])

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Document")
            axes[i, 1].imshow(gt, cmap="gray")
            axes[i, 1].set_title("Ground-truth mask")
            im = axes[i, 2].imshow(prob, cmap="jet", vmin=0, vmax=1)
            axes[i, 2].set_title("Predicted score map")
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            for j in range(3):
                axes[i, j].axis("off")

    plt.tight_layout()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_pred.png"
    plt.savefig(path, dpi=130)
    print("Saved:", path)


if __name__ == "__main__":
    main()
