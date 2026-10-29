import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import V2, build_protocol_splits, load_meta  # noqa: E402


def bbox_map():
    out = {}
    for r in load_meta(V2 / "metadata.jsonl"):
        if r.get("split") == "testing" and "bbox_xyxy" in r:
            out[str(r["new_id"])] = r["bbox_xyxy"]
    return out


def overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    return ix * iy


def context_bbox(bbox, W, H, pad=0.6, min_pad=60):
    x1, y1, x2, y2 = bbox
    px = int(max(min_pad, (x2 - x1) * pad))
    py = int(max(min_pad, (y2 - y1) * pad))
    return max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)


@torch.no_grad()
def inpaint_error(pipe, image, bbox, size, steps, prompt, device, seed):
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = context_bbox(bbox, W, H)
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]

    mask = np.zeros((ch, cw), dtype=np.uint8)
    mask[max(0, y1 - cy1):max(0, y2 - cy1), max(0, x1 - cx1):max(0, x2 - cx1)] = 255

    crop_pil = Image.fromarray((crop * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize((size, size), Image.NEAREST)
    g = torch.Generator(device=device).manual_seed(seed)
    rec = pipe(prompt=prompt, image=crop_pil, mask_image=mask_pil,
               height=size, width=size, num_inference_steps=steps, generator=g).images[0]

    sx, sy = size / cw, size / ch
    bx1 = int((x1 - cx1) * sx); by1 = int((y1 - cy1) * sy)
    bx2 = max(bx1 + 1, int((x2 - cx1) * sx)); by2 = max(by1 + 1, int((y2 - cy1) * sy))

    orig = np.asarray(crop_pil, dtype=np.float32) / 255.0
    recn = np.asarray(rec, dtype=np.float32) / 255.0
    diff = np.abs(orig[by1:by2, bx1:bx2] - recn[by1:by2, bx1:bx2])
    return float(diff.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--prompt", default="a clear authentic financial receipt with crisp printed text")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test_gptimage")
    ap.add_argument("--out", default="results/diffusion_pilot")
    args = ap.parse_args()

    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading diffusion model:", args.model)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    raw = build_protocol_splits("p1")
    forged = [it for it in raw[args.split] if it[2] == 1]
    rng = random.Random(args.seed)
    bmap = bbox_map()

    forged_items = forged[:args.n]
    errors_t, errors_r = [], []
    used = []

    for idx, (img_path, mask_path, _) in enumerate(forged_items):
        stem = Path(img_path).stem
        bbox = bmap.get(stem)
        if bbox is None:
            continue
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = max(2, x2 - x1), max(2, y2 - y1)

        # random bbox on the same text line, non-overlapping
        cand = None
        for _ in range(60):
            xx = rng.randint(0, max(0, W - w))
            yy = max(0, min(H - h, y1))
            c = (xx, yy, xx + w, yy + h)
            if overlap(c, (x1, y1, x2, y2)) == 0:
                cand = c
                break
        if cand is None:
            continue

        e_t = inpaint_error(pipe, image, (x1, y1, x2, y2), args.size, args.steps,
                            args.prompt, device, args.seed + idx)
        e_r = inpaint_error(pipe, image, cand, args.size, args.steps,
                            args.prompt, device, args.seed + 10000 + idx)
        errors_t.append(e_t); errors_r.append(e_r)
        used.append({"stem": stem, "error_tampered": e_t, "error_random": e_r})
        print(f"[{len(errors_t)}] {stem}: tampered={e_t:.4f} random={e_r:.4f}")

    if not errors_t:
        print("No samples processed.")
        return

    y = np.array([1] * len(errors_t) + [0] * len(errors_r))
    s = np.array(errors_t + errors_r)
    auc = roc_auc_score(y, s)
    print(f"\nN={len(errors_t)} | mean tampered error={np.mean(errors_t):.4f} | "
          f"mean random error={np.mean(errors_r):.4f} | AUC={auc:.4f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pilot.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "auc": auc,
                   "mean_tampered": float(np.mean(errors_t)),
                   "mean_random": float(np.mean(errors_r)),
                   "samples": used}, f, indent=2)
    print("Saved:", out / "pilot.json")


if __name__ == "__main__":
    main()
