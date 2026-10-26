import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import V1, V2, GPTIMAGE, authentic_items, forged_items, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402


def select_random(n, k, rng):
    return rng.sample(range(n), min(k, n))


def select_kcenter(emb, k, rng):
    emb = F.normalize(emb, dim=1)
    n = emb.shape[0]
    k = min(k, n)
    first = rng.randrange(n)
    sel = [first]
    min_dist = torch.cdist(emb, emb[first].unsqueeze(0)).squeeze(1)
    for _ in range(k - 1):
        idx = int(torch.argmax(min_dist).item())
        sel.append(idx)
        d = torch.cdist(emb, emb[idx].unsqueeze(0)).squeeze(1)
        min_dist = torch.minimum(min_dist, d)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--method", default="random", choices=["random", "kcenter", "uncertainty"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt", default="results/seg/unet_resnet18.pth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng_auth = random.Random(args.seed)  # authentic giu nguyen giua cac method

    v2_meta = load_meta(V2 / "metadata.jsonl")
    gpt = forged_items(v2_meta, V2, tool=GPTIMAGE, split="training")
    auth = authentic_items(V1, "training")
    print(f"Pool: {len(gpt)} GPT forged, {len(auth)} authentic")

    if args.method == "random":
        sel = select_random(len(gpt), args.k, rng)
    else:
        model = build_segmentation("resnet18").to(device)
        model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
        model.eval()
        loader = DataLoader(ItemsDataset(gpt, size=args.size),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

        embs, ents = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"extract ({args.method})"):
                x = batch["image"].to(device)
                if args.method == "kcenter":
                    feats = model.encoder(x)[-1]
                    e = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                    embs.append(e.cpu())
                else:
                    p = torch.sigmoid(model(x))
                    ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
                    ents.append(ent.mean(dim=(1, 2, 3)).cpu())

        if args.method == "kcenter":
            emb = torch.cat(embs, dim=0)
            sel = select_kcenter(emb, args.k, rng)
        else:
            ent = torch.cat(ents, dim=0).numpy()
            sel = np.argsort(-ent)[:args.k].tolist()

    auth_sel = rng_auth.sample(range(len(auth)), min(args.k, len(auth)))

    items = []
    for i in sel:
        img, msk, _ = gpt[i]
        items.append([str(img), str(msk), 1])
    for j in auth_sel:
        img, _, _ = auth[j]
        items.append([str(img), None, 0])

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"method": args.method, "k": args.k, "seed": args.seed, "items": items}, f)
    print(f"Saved {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
