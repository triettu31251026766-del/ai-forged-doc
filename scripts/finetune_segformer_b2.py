"""Fine-tune SegFormer-B2 for AI-forged document localization (Kaggle/Linux).

Two-stage protocol P1 (same as the paper):
  stage 1 (optional): train on the Gemini training split
      python scripts/finetune_segformer_b2.py --mode gemini --out results/segformer_b2_gemini
  stage 2: fine-tune with exactly 100 GPT-Image-2 labels + 100 authentic
      python scripts/finetune_segformer_b2.py --mode k100 --k 100 \
          --init results/segformer_b2_gemini/model.pth --out results/segformer_b2_k100

Without a Gemini checkpoint, stage 2 starts from the ADE-512 SegFormer-B2 weights
and is a label-efficiency baseline, not a P1 adaptation run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ItemsDataset  # noqa: E402
from src.data.splits import build_fewshot_splits, build_protocol_splits  # noqa: E402

HF_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def build_model(init, device):
    from transformers import SegformerForSemanticSegmentation

    if init and Path(init).is_file():
        model = SegformerForSemanticSegmentation.from_pretrained(
            HF_NAME, num_labels=1, ignore_mismatched_sizes=True
        )
        state = torch.load(init, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded init weights: {init}")
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            init or HF_NAME, num_labels=1, ignore_mismatched_sizes=True
        )
    return model.to(device)


def forward_logits(model, images):
    out = model(pixel_values=images)
    logits = out.logits
    return F.interpolate(
        logits, size=images.shape[-2:], mode="bilinear", align_corners=False
    )


@torch.no_grad()
def evaluate(model, items, size, batch_size, device, max_n=None):
    if max_n:
        items = items[:max_n]
    ds = ItemsDataset(items, size=size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()

    inter = union = tp = fp = fn = 0
    ious, scores, labels = [], [], []
    for batch in tqdm(dl, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = forward_logits(model, images)
        probs = torch.sigmoid(logits)
        pred = probs > 0.5

        # image-level score: mean of the top 1% pixels
        flat = probs.flatten(1)
        k = max(1, int(0.01 * flat.shape[1]))
        scores.extend(flat.topk(k, dim=1).values.mean(dim=1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())

        for i in range(images.shape[0]):
            if int(batch["label"][i]) != 1:
                continue
            p = pred[i, 0]
            m = masks[i, 0] > 0.5
            ii = int((p & m).sum())
            uu = int((p | m).sum())
            inter += ii
            union += uu
            tp += ii
            fp += int((p & ~m).sum())
            fn += int((~p & m).sum())
            ious.append(ii / max(uu, 1))

    y = np.asarray(labels)
    s = np.asarray(scores)
    fpr, tpr, _ = roc_curve(y, s)
    j = max(np.searchsorted(fpr, 0.01, side="right") - 1, 0)
    return {
        "n_forged": len(ious),
        "iou_pooled": inter / max(union, 1),
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "image_auc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else 0.0,
        "image_ap": float(average_precision_score(y, s)) if len(set(y)) > 1 else 0.0,
        "recall_at_1fpr": float(tpr[j]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gemini", "k100"], required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--init", default=None, help="HF name or model.pth from stage 1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--replay", type=int, default=0, help="old-generator samples (forged and authentic each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--out", default="results/segformer_b2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device={device} mode={args.mode} k={args.k} size={args.size}")

    if args.mode == "gemini":
        raw = build_protocol_splits("p1", seed=args.seed)
    else:
        raw = build_fewshot_splits(args.k, seed=args.seed)
    train_items = list(raw["train"])
    if args.replay and args.mode == "k100":
        import random

        old = build_protocol_splits("p1", seed=args.seed)["train"]
        old_f = [it for it in old if it[2] == 1]
        old_a = [it for it in old if it[2] == 0]
        rng = random.Random(args.seed)
        train_items += rng.sample(old_f, min(args.replay, len(old_f)))
        train_items += rng.sample(old_a, min(args.replay, len(old_a)))
    if args.max_train:
        train_items = train_items[: args.max_train]

    train_ds = ItemsDataset(train_items, size=args.size)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    print(f"train samples: {len(train_ds)} (forged={sum(it[2] == 1 for it in train_items)})")

    model = build_model(args.init, device)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                logits = forward_logits(model, images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach()) * images.shape[0]
        print(f"epoch {epoch}: loss={running / max(len(train_ds), 1):.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "model.pth")

    results = {"args": vars(args), "tests": {}}
    for name in ("test_gptimage", "test_in_domain", "test_ideogram"):
        if name not in raw:
            continue
        print(f"evaluating {name}")
        results["tests"][name] = evaluate(
            model, raw[name], args.size, args.batch_size, device, args.max_val
        )
        print(name, json.dumps(results["tests"][name]))
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("saved", out_dir / "results.json")


if __name__ == "__main__":
    main()
