import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "figures" / "fig_tlhrs.png"


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def main():
    slide = [load(f"slide_s{i}/results.json") for i in range(3)]
    tlhrs = [s["methods"]["top1"]["iou_pooled"] for s in slide]
    oracle_gpt = float(np.mean([s["oracle_window_iou_mean"] for s in slide]))
    ideo = load("slide_ideo/results.json")
    ideo_tlhrs = ideo["methods"]["top1"]["iou_pooled"]
    ideo_oracle = ideo["oracle_window_iou_mean"]

    refine = load("refine/results.json")["iou_pooled"]
    loc = load("localize/results.json")["methods"]
    loc2 = load("localize2/results.json")["methods"]
    slide2 = load("slide2/results.json")

    ceilings = [load(f"patch_seg_gpt{s}/results.json")["tests"]["test_gptimage"]["full_iou"]
                for s in ["", "_s1", "_s2"]]
    ceiling = float(np.mean(ceilings))
    baseline = 0.195

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    names = ["full-image\nbaseline", "coarse\npeak", "+centroid", "+3x3\ngrid", "TL-HRS\n(dense)", "oracle\nwindow", "GT-box\nceiling"]
    vals = [baseline, refine["peak"], refine["centroid"], refine["grid3"],
            float(np.mean(tlhrs)), oracle_gpt, ceiling]
    colors = ["#8c8c8c", "#c9a227", "#c9a227", "#c9a227", "#2f6f9f", "#5b9bd5", "#2e7d32"]
    ax.bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("IoU (full image, GPT-Image-2)")
    ax.set_title("(a) Where the pipeline loses IoU", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    sub = [
        ("coarse peak", loc["peak"]["iou_pooled"]),
        ("centroid", loc2["gcent"]["iou_pooled"]),
        ("grid 3x3", loc["grid3"]["iou_pooled"]),
        ("grid 5x5", loc["grid5"]["iou_pooled"]),
        ("grid 7x7", loc2["grid7"]["iou_pooled"]),
        ("centroid+g5", loc2["gcent_g5"]["iou_pooled"]),
        ("zoom 768", loc["zoom2"]["iou_pooled"]),
        ("tile 2x2", loc["tile2"]["iou_pooled"]),
        ("pruned top-40", float(slide2["rows"]["40"]["iou_pooled"])),
        ("dense search", 0.5826),
    ]
    sub_sorted = sorted(sub, key=lambda z: z[1])
    ax.barh([z[0] for z in sub_sorted], [z[1] for z in sub_sorted],
            color=["#2f6f9f" if z[0] == "dense search" else "#9fb6c9" for z in sub_sorted])
    for i, z in enumerate(sub_sorted):
        ax.text(z[1] + 0.005, i, f"{z[1]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("IoU (150-image subset)")
    ax.set_title("(b) Localization variants", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    labels = ["GPT-Image-2\n(target)", "Ideogram\n(unseen)"]
    sel = [float(np.mean(tlhrs)), ideo_tlhrs]
    orc = [oracle_gpt, ideo_oracle]
    base = [baseline, 0.1144]
    x = np.arange(2)
    w = 0.26
    ax.bar(x - w, base, w, label="full-image", color="#8c8c8c")
    ax.bar(x, sel, w, label="TL-HRS", color="#2f6f9f")
    ax.bar(x + w, orc, w, label="oracle window", color="#5b9bd5")
    for xi, (b, s, o) in enumerate(zip(base, sel, orc)):
        ax.text(xi - w, b + 0.012, f"{b:.3f}", ha="center", fontsize=7.5)
        ax.text(xi, s + 0.012, f"{s:.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w, o + 0.012, f"{o:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("IoU")
    ax.set_title("(c) Same 100 labels, unseen generator", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print("Saved:", OUT)
    print("tlhrs", tlhrs, "ideo", ideo_tlhrs, "ceiling", ceiling)


if __name__ == "__main__":
    main()


