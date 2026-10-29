import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import V1, V2, GEMINI, GPTIMAGE, authentic_items, forged_items, load_meta  # noqa: E402
from src.models.segmentation import build_segmentation  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- helpers ----------

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def bbox_map(meta, split):
    out = {}
    for r in meta:
        if r.get("split") == split and "bbox_xyxy" in r:
            out[str(r["new_id"])] = [int(v) for v in r["bbox_xyxy"]]
    return out


def crop_window(img, mask, cx, cy, P):
    """Crop PxP quanh (cx,cy), zero-pad chi vung crop (khong pad ca anh)."""
    H, W = img.shape[:2]
    x0 = cx - P // 2
    y0 = cy - P // 2
    crop_img = np.zeros((P, P, 3), dtype=np.float32)
    crop_mask = np.zeros((P, P), dtype=bool)

    gx0 = max(0, x0)
    gy0 = max(0, y0)
    gx1 = min(W, x0 + P)
    gy1 = min(H, y0 + P)
    if gx1 > gx0 and gy1 > gy0:
        dx = gx0 - x0
        dy = gy0 - y0
        crop_img[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)] = img[gy0:gy1, gx0:gx1]
        crop_mask[dy:dy + (gy1 - gy0), dx:dx + (gx1 - gx0)] = mask[gy0:gy1, gx0:gx1]
    return crop_img, crop_mask


# ---------- dataset ----------

class PatchDataset(Dataset):
    """items: list of dict {img, mask, bbox, pos}.
    pos=True: window quanh bbox (jitter khi train). pos=False: window random."""

    def __init__(self, items, P=384, jitter=0.0, rng=None):
        self.items = items
        self.P = P
        self.jitter = jitter
        self.rng = rng or random.Random(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = load_rgb(it["img"])
        mask = load_mask(it["mask"]) if it["mask"] is not None else np.zeros(img.shape[:2], dtype=bool)
        H, W = img.shape[:2]

        if it["pos"]:
            x1, y1, x2, y2 = it["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if self.jitter > 0:
                jx = int(self.rng.uniform(-self.jitter, self.jitter) * self.P)
                jy = int(self.rng.uniform(-self.jitter, self.jitter) * self.P)
                cx = max(0, min(W - 1, cx + jx))
                cy = max(0, min(H - 1, cy + jy))
        else:
            # negative: random window (co trang hoac chu)
            if it.get("bbox") is not None:
                x1, y1, x2, y2 = it["bbox"]
                cy = (y1 + y2) // 2
                # random x xa tampered
                w = max(2, x2 - x1)
                for _ in range(30):
                    cx = self.rng.randint(0, max(1, W - 1))
                    if abs(cx - (x1 + x2) // 2) > 2 * w:
                        break
            else:
                cx = self.rng.randint(0, max(1, W - 1))
                cy = self.rng.randint(0, max(1, H - 1))

        crop_img, crop_mask = crop_window(img, mask, cx, cy, self.P)
        arr = (crop_img - MEAN) / STD
        image = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        m = torch.from_numpy(crop_mask.astype(np.float32)).unsqueeze(0)
        return {"image": image, "mask": m}


def build_train_items(seed, n_neg_per_pos=1):
    rng = random.Random(seed)
    v1_meta = load_meta(V1 / "metadata.jsonl")
    gb = bbox_map(v1_meta, "training")

    pos = []
    for img, msk, _ in forged_items(v1_meta, V1, tool=GEMINI, split="training"):
        b = gb.get(Path(img).stem)
        if b is not None:
            pos.append({"img": img, "mask": msk, "bbox": b, "pos": True})

    # negatives: cung dong (tu chinh forged) + authentic random
    neg = []
    for it in pos:
        neg.append({"img": it["img"], "mask": None, "bbox": it["bbox"], "pos": False})
    auth = authentic_items(V1, "training")
    for img, _, _ in auth:
        neg.append({"img": img, "mask": None, "bbox": None, "pos": False})

    rng.shuffle(pos)
    rng.shuffle(neg)
    neg = neg[: max(len(pos), n_neg_per_pos * len(pos))]
    items = pos + neg
    rng.shuffle(items)
    return items


def build_test_items(split, seed, max_n=None):
    rng = random.Random(seed)
    if split == "test_gptimage":
        meta = load_meta(V2 / "metadata.jsonl")
        bm = bbox_map(meta, "testing")
        forged = forged_items(meta, V2, tool=GPTIMAGE, split="testing")
    else:
        meta = load_meta(V1 / "metadata.jsonl")
        bm = bbox_map(meta, "testing")
        forged = forged_items(meta, V1, tool=GEMINI, split="testing")

    items = []
    for img, msk, _ in forged:
        b = bm.get(Path(img).stem)
        if b is not None:
            items.append({"img": img, "mask": msk, "bbox": b, "pos": True})
    if max_n:
        items = items[:max_n]
    return items


# ---------- eval ----------

@torch.no_grad()
def eval_bbox(model, items, P, device):
    inter_f = union_f = tp_f = fp_f = fn_f = 0
    inter_p = union_p = 0
    dice_sum = 0.0
    n = 0
    paucs, paps = [], []

    for it in tqdm(items, desc="eval bbox"):
        img = load_rgb(it["img"])
        mask = load_mask(it["mask"])
        x1, y1, x2, y2 = it["bbox"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        crop_img, crop_mask = crop_window(img, mask, cx, cy, P)
        arr = (crop_img - MEAN) / STD
        xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(xin))[0, 0].cpu().numpy()

        pred = prob > 0.5
        # patch-level
        inter_p += int((pred & crop_mask).sum())
        union_p += int((pred | crop_mask).sum())

        # full-image: place patch at window
        H, W = mask.shape
        pad = P // 2 + 1
        x0 = cx - P // 2
        y0 = cy - P // 2
        full_pred = np.zeros((H, W), dtype=bool)
        # place crop_mask/pred into full coords
        gy0, gx0 = max(0, y0), max(0, x0)
        gy1, gx1 = min(H, y0 + P), min(W, x0 + P)
        py0, px0 = gy0 - y0, gx0 - x0
        full_pred[gy0:gy1, gx0:gx1] = pred[py0:py0 + (gy1 - gy0), px0:px0 + (gx1 - gx0)]

        inter_f += int((full_pred & mask).sum())
        union_f += int((full_pred | mask).sum())
        tp_f += int((full_pred & mask).sum())
        fp_f += int((full_pred & ~mask).sum())
        fn_f += int((~full_pred & mask).sum())
        dice_sum += (2 * (full_pred & mask).sum()) / (full_pred.sum() + mask.sum() + 1e-6)
        n += 1

        if mask.any():
            pflat = full_pred.flatten()
            # pixel-level scores use full stitched prob map (recompute needed) -> use patch prob placed
            prob_full = np.zeros((H, W), dtype=np.float32)
            prob_full[gy0:gy1, gx0:gx1] = prob[py0:py0 + (gy1 - gy0), px0:px0 + (gx1 - gx0)]
            mflat = mask.flatten()
            paucs.append(roc_auc_score(mflat, prob_full.flatten()))
            paps.append(average_precision_score(mflat, prob_full.flatten()))

    return {
        "full_iou": inter_f / max(union_f, 1),
        "full_f1": 2 * tp_f / max(2 * tp_f + fp_f + fn_f, 1),
        "full_dice": dice_sum / max(n, 1),
        "patch_iou": inter_p / max(union_p, 1),
        "pixel_auc": float(np.mean(paucs)) if paucs else 0.0,
        "pixel_ap": float(np.mean(paps)) if paps else 0.0,
        "n": n,
    }


@torch.no_grad()
def tile_score(model, img, P, stride, device):
    H, W = img.shape[:2]
    sump = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    xs = list(range(0, max(1, W - P + 1), stride))
    ys = list(range(0, max(1, H - P + 1), stride))
    if W > P and xs[-1] != W - P:
        xs.append(W - P)
    if H > P and ys[-1] != H - P:
        ys.append(H - P)
    for y in ys:
        for x in xs:
            h = min(P, H - y)
            w = min(P, W - x)
            crop = np.zeros((P, P, 3), dtype=np.float32)
            crop[:h, :w] = img[y:y + h, x:x + w]
            arr = (crop - MEAN) / STD
            xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
            p = torch.sigmoid(model(xin))[0, 0].cpu().numpy()
            sump[y:y + h, x:x + w] += p[:h, :w]
            cnt[y:y + h, x:x + w] += 1.0
    return sump / np.maximum(cnt, 1.0)


def eval_image_level(model, split, P, stride, device, max_n):
    from src.data.splits import build_protocol_splits
    raw = build_protocol_splits("p1")
    items = raw[split]
    rng = random.Random(0)
    if max_n:
        items = rng.sample(items, min(max_n, len(items)))
    scores, labels = [], []
    for img_path, mask_path, label in tqdm(items, desc="image-level"):
        img = load_rgb(img_path)
        prob = tile_score(model, img, P, stride, device)
        scores.append(float(prob.max()))
        labels.append(label)
    y = np.asarray(labels)
    s = np.asarray(scores)
    if len(np.unique(y)) < 2:
        return {"image_ap": float("nan"), "recall_at_1fpr": float("nan")}
    ap = average_precision_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    rec1 = float(np.interp(0.01, fpr, tpr))
    return {"image_ap": ap, "recall_at_1fpr": rec1}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip-image-level", action="store_true")
    ap.add_argument("--out", default="results/patch_seg")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    train_items = build_train_items(seed=0)
    if args.max_train:
        train_items = train_items[:args.max_train]
    train_ds = PatchDataset(train_items, P=args.patch_size, jitter=args.jitter)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(f"Train patches: {len(train_ds)}")

    model = build_segmentation("resnet18").to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def dice_loss(logits, target):
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3))
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        return 1 - (num / den).mean()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for batch in tqdm(train_loader, desc=f"ep {epoch}/{args.epochs}", leave=False):
            x = batch["image"].to(device)
            t = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, t) + dice_loss(logits, t)
            loss.backward()
            optimizer.step()
            run += loss.item() * len(x)
        print(f"  ep {epoch}: loss={run / len(train_ds):.4f}")

    results = {"args": vars(args), "sec": round(time.time() - t0, 1), "tests": {}}
    for split in ["test_gptimage", "test_in_domain"]:
        test_items = build_test_items(split, seed=0, max_n=args.max_test)
        m = eval_bbox(model, test_items, args.patch_size, device)
        if args.skip_image_level:
            il = {"image_ap": float("nan"), "recall_at_1fpr": float("nan")}
        else:
            il = eval_image_level(model, split, args.patch_size, args.stride, device,
                                  max_n=args.max_test)
        results["tests"][split] = {"bbox": m, "image_level": il}
        print(f"{split}: full_iou={m['full_iou']:.4f} full_f1={m['full_f1']:.4f} "
              f"patch_iou={m['patch_iou']:.4f} pixAUC={m['pixel_auc']:.4f} "
              f"pixAP={m['pixel_ap']:.4f} ImgAP={il['image_ap']:.4f} R@1%={il['recall_at_1fpr']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "patch_seg.pth")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir / "results.json")


if __name__ == "__main__":
    main()
