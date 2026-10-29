import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    V1, V2, IDEOGRAM, build_fewshot_splits, build_protocol_splits, forged_items, load_meta,
)
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_patch_seg import (  # noqa: E402
    PatchDataset, bbox_map, build_test_items, eval_bbox,
)


def build_ideo_test_items(max_n=None):
    meta = load_meta(V1 / "metadata.jsonl")
    bm = bbox_map(meta, "testing")
    out = []
    for img, msk, _ in forged_items(meta, V1, tool=IDEOGRAM, split="testing"):
        b = bm.get(Path(img).stem)
        if b is not None:
            out.append({"img": img, "mask": msk, "bbox": b, "pos": True})
    return out[:max_n] if max_n else out


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return 1 - (num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--replay", type=int, default=100)
    ap.add_argument("--init", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--out", default="results/patch_seg_gpt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    raw = build_fewshot_splits(args.k, seed=args.seed)
    v2 = load_meta(V2 / "metadata.jsonl")
    gpt_box = bbox_map(v2, "training")

    pos, neg = [], []
    auth_neg = []
    for img, msk, lab in raw["train"]:
        if lab == 1:
            b = gpt_box.get(Path(img).stem)
            if b is not None:
                pos.append({"img": img, "mask": msk, "bbox": b, "pos": True})
        else:
            auth_neg.append({"img": img, "mask": None, "bbox": None, "pos": False})
    for it in pos:
        neg.append({"img": it["img"], "mask": None, "bbox": it["bbox"], "pos": False})

    rep_pos, rep_neg = [], []
    if args.replay > 0:
        rng = random.Random(args.seed)
        old = build_protocol_splits("p1")["train"]
        old_f = [it for it in old if it[2] == 1]
        old_a = [it for it in old if it[2] == 0]
        v1 = load_meta(V1 / "metadata.jsonl")
        gem_box = bbox_map(v1, "training")
        for img, msk, _ in rng.sample(old_f, min(args.replay, len(old_f))):
            b = gem_box.get(Path(img).stem)
            if b is not None:
                rep_pos.append({"img": img, "mask": msk, "bbox": b, "pos": True})
        for img, _, _ in rng.sample(old_a, min(args.replay, len(old_a))):
            rep_neg.append({"img": img, "mask": None, "bbox": None, "pos": False})

    items = pos + neg + auth_neg + rep_pos + rep_neg
    random.Random(args.seed).shuffle(items)
    ds = PatchDataset(items, P=args.patch_size, jitter=args.jitter, rng=random.Random(args.seed))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Fine-tune patches: {len(ds)} (gpt_pos={len(pos)}, gpt_neg={len(neg)}, "
          f"auth={len(auth_neg)}, replay_pos={len(rep_pos)}, replay_neg={len(rep_neg)})")

    model = build_segmentation("resnet18").to(device)
    model.load_state_dict(torch.load(ROOT / args.init, map_location=device))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0
        for batch in tqdm(loader, desc=f"ft-patch {ep}/{args.epochs}", leave=False):
            x = batch["image"].to(device)
            t = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, t) + dice_loss(logits, t)
            loss.backward(); opt.step()
            run += loss.item() * len(x)
        print(f"  ep {ep}: loss={run/len(ds):.4f}")

    model.eval()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "patch_seg.pth")

    results = {"args": vars(args), "tests": {}}
    for split in ["test_gptimage", "gemini", "ideogram"]:
        if split == "ideogram":
            items_t = build_ideo_test_items(args.max_test or None)
        else:
            items_t = build_test_items(split, args.seed, args.max_test or None)
        res = eval_bbox(model, items_t, args.patch_size, device)
        results["tests"][split] = res
        print(f"{split}: IoU={res['full_iou']:.4f} F1={res['full_f1']:.4f} "
              f"Dice={res['full_dice']:.4f} PixAUC={res['pixel_auc']:.4f} "
              f"PixAP={res['pixel_ap']:.4f} n={res['n']}")

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out / "patch_seg.pth")


if __name__ == "__main__":
    main()
