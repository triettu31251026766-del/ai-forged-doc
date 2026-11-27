import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG = Path(__file__).resolve().parents[0] / "figures" / "fig_pipeline.png"


def box(ax, x, y, w, h, text, fc="#e8f0f7", ec="#2f6f9f"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=1.0, color="#444"))


fig, ax = plt.subplots(figsize=(9, 3.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, 0.01, 0.38, 0.15, 0.24, "Document\n(native res)")
arrow(ax, 0.16, 0.50, 0.20, 0.50)
box(ax, 0.20, 0.38, 0.16, 0.24, "Coarse patch\nscan, stride 256")
arrow(ax, 0.36, 0.50, 0.40, 0.50)
box(ax, 0.40, 0.38, 0.15, 0.24, "Top-3\nwindows")
arrow(ax, 0.55, 0.50, 0.59, 0.50)
box(ax, 0.59, 0.38, 0.16, 0.24, "Local refine\n3x3, stride 96")
arrow(ax, 0.75, 0.50, 0.79, 0.50)
box(ax, 0.79, 0.38, 0.19, 0.24, "w* = argmax S(w)\n(adapted patch seg.)")

box(ax, 0.62, 0.04, 0.16, 0.22, "Image triage\n(CRC, FNR<=alpha)", fc="#f7efe0", ec="#c9a227")
arrow(ax, 0.86, 0.38, 0.72, 0.26)
box(ax, 0.82, 0.04, 0.17, 0.22, "Mask + pixel CRC\n(FNR certified)", fc="#eaf5ea", ec="#2e7d32")
arrow(ax, 0.92, 0.38, 0.90, 0.26)

ax.text(0.5, 0.95, "TL-HRS pipeline: coarse-to-fine search with risk-controlled outputs",
        ha="center", fontsize=9, weight="bold")
plt.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG, dpi=200, bbox_inches="tight", pad_inches=0.08)
print("saved", FIG)
