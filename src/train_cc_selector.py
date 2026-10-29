import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.splits import (  # noqa: E402
    V1, GEMINI, build_protocol_splits, forged_items, load_meta,
)
from src.models.segmentation import build_segmentation  # noqa: E402
from src.train_box_cls import (  # noqa: E402
    MEAN, STD, bbox_map, crop_window, detect_boxes, iou, load_mask, load_rgb,
    new_stats, no_overlap, place_max, summarize, update,
)
from eval_coarse2fine import box_region_mean, global_forward  # noqa: E402

D = 384


class DinoEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.m = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                   pretrained=True, num_classes=0, dynamic_img_size=True)

    def forward(self, x):
        return self.m(x)


def to_tensor(img_rgb, size, device):
    rsz = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = (rsz - MEAN) / STD
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def embed(model, x):
    return model(x)[0].cpu()


def doc_embed(model, img, device):
    return embed(model, to_tensor(img, 224, device))


def crop_embed(model, img, cx, cy, P, device):
    crop = crop_window(img, int(cx), int(cy), P)
    return embed(model, to_tensor(crop, 224, device))


class CCFieldDataset(Dataset):
    def __init__(self, items, dino, device, P=192, jitter=0.15, seed=0):
        self.items = items; self.dino = dino; self.device = device
        self.P = P; self.jitter = jitter
        self.rng = random.Random(seed)
        print("Precompute doc embeddings ...")
        self.doc = {}
        imgs = sorted({it["img"] for it in items})
        for img in tqdm(imgs, desc="doc-embed", leave=False):
            self.doc[img] = doc_embed(dino, load_rgb(img), device)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = load_rgb(it["img"])
        H, W = img.shape[:2]
        x1, y1, x2, y2 = it["bbox"]
        gw, gh = max(1, x2 - x1), max(1, y2 - y1)
        gcx, gcy = (x1 + x2) // 2, (y1 + y2) // 2
        if it["pos"]:
            cx, cy = gcx, gcy
            if self.jitter > 0:
                cx += int(self.rng.uniform(-self.jitter, self.jitter) * gw)
                cy += int(self.rng.uniform(-self.jitter, self.jitter) * gh)
            cx = max(0, min(W - 1, cx)); cy = max(0, min(H - 1, cy))
        else:
            cx, cy = gcx, gcy
            for _ in range(50):
                cx = self.rng.randint(0, max(1, W - 1))
                cy = max(0, min(H - 1, gcy + self.rng.randint(-gh, gh)))
                if no_overlap(cx, cy, self.P, it["bbox"]):
                    break
        v = crop_embed(self.dino, img, cx, cy, self.P, self.device)
        mu = self.doc[it["img"]]
        x = torch.cat([v, mu, v - mu])
        y = torch.tensor(1.0 if it["pos"] else 0.0)
        return x, y


def build_train_items(num=1200, seed=0):
    rng = random.Random(seed)
    meta = load_meta(V1 / "metadata.jsonl")
    bm = bbox_map(meta, "training")
    pos = []
    for img, _, _ in forged_items(meta, V1, tool=GEMINI, split="training"):
        b = bm.get(Path(img).stem)
        if b is not None:
            pos.append({"img": img, "bbox": b, "pos": True})
    rng.shuffle(pos)
    pos = pos[:num]
    neg = [{"img": it["img"], "bbox": it["bbox"], "pos": False}
           for it in pos for _ in range(2)]
    items = pos + neg
    rng.shuffle(items)
    return items


def build_head():
    return nn.Sequential(
        nn.Linear(3 * D, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-test", type=int, default=60)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--patch-ckpt", default="results/patch_seg/patch_seg.pth")
    ap.add_argument("--global-ckpt", default="results/fewshot/k100.pth")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="results/cc_sel")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    dino = DinoEmbed().to(device).eval()

    head = build_head().to(device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        items = build_train_items(args.num_train)
        ds = CCFieldDataset(items, dino, device)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"CC selector train samples: {len(ds)}")
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
        for ep in range(1, args.epochs + 1):
            head.train(); run = 0.0
            for x, y in tqdm(loader, desc=f"cc {ep}/{args.epochs}", leave=False):
                x = x.to(device); y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logit = head(x).squeeze(1)
                loss = lossf(logit, y)
                loss.backward(); opt.step()
                run += loss.item() * len(x)
            print(f"  ep {ep}: loss={run/len(ds):.4f}")
        torch.save(head.state_dict(), out / "cc_head.pth")
    else:
        head.load_state_dict(torch.load(out / "cc_head.pth", map_location=device))
        print("Loaded:", out / "cc_head.pth")
    head.eval()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
    seg = build_segmentation("resnet18").to(device)
    seg.load_state_dict(torch.load(ROOT / args.patch_ckpt, map_location=device))
    seg.eval()
    gmodel = build_segmentation("resnet18").to(device)
    gmodel.load_state_dict(torch.load(ROOT / args.global_ckpt, map_location=device))
    gmodel.eval()

    raw = build_protocol_splits("p1")
    test_items = raw[args.split]
    rng = np.random.default_rng(0)
    if args.max_test and args.max_test < len(test_items):
        idx = rng.choice(len(test_items), size=args.max_test, replace=False)
        test_items = [test_items[i] for i in idx]

    meta = load_meta(ROOT / "data" / "raw" / "AIForge-Doc-v2" / "metadata.jsonl")
    gtb = bbox_map(meta, "testing") if args.split == "test_gptimage" else {}

    names = ["cc_sel", "dino_zero", "cc_fuse"]
    stats = {n: new_stats() for n in names}
    sel_iou = {n: [] for n in names}
    sel_center = {n: 0 for n in names}
    n_sel = {n: 0 for n in names}

    with torch.no_grad():
        for img_path, mask_path, label in tqdm(test_items, desc="eval cc"):
            img = load_rgb(img_path); H, W = img.shape[:2]
            boxes = detect_boxes(reader, img, 0.3, 30)
            mu = doc_embed(dino, img, device)

            # global heatmap scores for fusion
            gs = None
            if boxes:
                prob384, scale, nw2, nh2, left2, top2 = global_forward(gmodel, img, 384, device)
                gs = np.asarray([box_region_mean(prob384, b, scale, left2, top2)
                                 for b, _ in boxes], dtype=np.float32)

            if not boxes:
                continue
            cc_scores = []
            zero_scores = []
            for box, _ in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                v = crop_embed(dino, img, cx, cy, 192, device)
                xin = torch.cat([v, mu, v - mu]).unsqueeze(0).to(device)
                cc = float(torch.sigmoid(head(xin)).item())
                cc_scores.append(cc)
                zero_scores.append(float((v - mu).norm().item()))
            cc_scores = np.asarray(cc_scores)
            zero_scores = np.asarray(zero_scores)
            scores_map = {
                "cc_sel": cc_scores,
                "dino_zero": zero_scores,
                "cc_fuse": None,
            }
            if gs is not None:
                zcc = (cc_scores - cc_scores.mean()) / (cc_scores.std() + 1e-6)
                zgs = (gs - gs.mean()) / (gs.std() + 1e-6)
                scores_map["cc_fuse"] = zcc + zgs

            for n in names:
                s = scores_map[n]
                if s is None:
                    continue
                bi = int(np.argmax(s))
                bx = boxes[bi][0]
                cx = (bx[0] + bx[2]) // 2; cy = (bx[1] + bx[3]) // 2
                crop = crop_window(img, cx, cy, 384)
                arr = (crop - MEAN) / STD
                xin = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
                segp = torch.sigmoid(seg(xin))[0, 0].cpu().numpy()
                full = np.zeros((H, W), dtype=np.float32)
                place_max(full, segp, cx, cy, 384)
                if label == 1 and mask_path is not None:
                    mask = load_mask(mask_path)
                    update(stats[n], full > 0.5, mask)
                    gt = gtb.get(Path(img_path).stem)
                    if gt is not None:
                        sel_iou[n].append(iou(bx, gt))
                        gx = (gt[0] + gt[2]) // 2; gy = (gt[1] + gt[3]) // 2
                        if bx[0] <= gx <= bx[2] and bx[1] <= gy <= bx[3]:
                            sel_center[n] += 1
                        n_sel[n] += 1

    print(f"\n=== Context-contrast selector ({args.split}, {args.max_test} imgs) ===")
    for n in names:
        sm = summarize(stats[n])
        print(f"{n:10s}: IoU={sm['iou']:.4f} F1={sm['f1']:.4f} Dice={sm['dice']:.4f} "
              f"sel_mean_IoU={np.mean(sel_iou[n]) if sel_iou[n] else 0:.4f} "
              f"center_in_selected={100*sel_center[n]/max(n_sel[n],1):.1f}%")

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "metrics": {n: summarize(stats[n]) for n in names},
                   "sel_mean_iou": {n: float(np.mean(sel_iou[n])) if sel_iou[n] else 0.0 for n in names},
                   "sel_center_in": {n: float(sel_center[n] / max(n_sel[n], 1)) for n in names}},
                  f, indent=2)
    print("Saved:", out / "results.json")


if __name__ == "__main__":
    main()
