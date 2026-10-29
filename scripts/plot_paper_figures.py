import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "results" / "fewshot_multi"
MIT = ROOT / "results" / "fewshot_mit"
SHOT = ROOT / "results" / "fewshot"
OUT = ROOT / "paper" / "figures" / "fig_main_results.png"


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def iou(d, split):
    return d["tests"][split]["pixel"]["iou"]


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def main():
    # few-shot standard runs
    by_k = defaultdict(lambda: {"gpt": [], "gem": []})
    for folder in [FS]:
        for f in folder.glob("k*_s*.json"):
            if re.fullmatch(r"k\d+_s\d+", f.stem):
                m = re.match(r"k(\d+)_s(\d+)", f.stem)
                k = int(m.group(1))
                d = load_json(f)
                by_k[k]["gpt"].append(iou(d, "test_gptimage"))
                by_k[k]["gem"].append(iou(d, "test_in_domain"))

    ks = [0] + sorted(by_k.keys())
    gpt_m, gpt_s, gem_m, gem_s = [], [], [], []
    for k in ks:
        if k == 0:
            gpt_m.append(0.052); gpt_s.append(0.0)
            gem_m.append(0.669); gem_s.append(0.0)
        else:
            gm, gs = ms(by_k[k]["gpt"]); gpt_m.append(gm); gpt_s.append(gs)
            hm, hs = ms(by_k[k]["gem"]); gem_m.append(hm); gem_s.append(hs)

    # k=100 bars
    std_gpt, std_gem = [], []
    for f in FS.glob("k100_s*.json"):
        if re.fullmatch(r"k100_s\d+", f.stem):
            d = load_json(f)
            std_gpt.append(iou(d, "test_gptimage"))
            std_gem.append(iou(d, "test_in_domain"))

    rep_gpt, rep_gem = [], []
    for folder in [MIT, SHOT]:
        if not folder.exists():
            continue
        for f in folder.glob("*rp*.json"):
            if "_rp" in f.stem:
                d = load_json(f)
                rep_gpt.append(iou(d, "test_gptimage"))
                rep_gem.append(iou(d, "test_in_domain"))

    scr_gpt, scr_gem = [], []
    for folder in [SHOT, FS, MIT]:
        if not folder.exists():
            continue
        for f in folder.glob("*scratch*.json"):
            d = load_json(f)
            scr_gpt.append(iou(d, "test_gptimage"))
            scr_gem.append(iou(d, "test_in_domain"))

    labels, gpt_bars, gem_bars, gpt_err, gem_err = [], [], [], [], []
    for name, gp, ge in [("Standard", std_gpt, std_gem),
                         ("Replay", rep_gpt, rep_gem),
                         ("Scratch", scr_gpt, scr_gem)]:
        if not gp:
            continue
        gm, gs = ms(gp); hm, hs = ms(ge)
        labels.append(name); gpt_bars.append(gm); gem_bars.append(hm)
        gpt_err.append(gs); gem_err.append(hs)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.6))

    ax = axes[0]
    ax.errorbar(ks, gpt_m, yerr=gpt_s, marker="o", capsize=4, label="GPT-Image-2 (target)")
    ax.errorbar(ks, gem_m, yerr=gem_s, marker="s", capsize=4, label="Gemini (old)")
    ax.set_xlabel("Number of labelled target samples (k)")
    ax.set_ylabel("IoU")
    ax.set_xticks(ks)
    ax.set_title("(a) Few-shot adaptation")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, gpt_bars, w, yerr=gpt_err, capsize=4, label="GPT-Image-2 (target)")
    ax.bar(x + w / 2, gem_bars, w, yerr=gem_err, capsize=4, label="Gemini (old)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU")
    ax.set_title("(b) Forgetting at k=100")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()


