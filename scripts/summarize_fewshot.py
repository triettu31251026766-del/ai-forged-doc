import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [ROOT / "results" / "fewshot_multi", ROOT / "results" / "fewshot_mit", ROOT / "results" / "fewshot"]


def mean_std(vals):
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def main():
    groups = defaultdict(lambda: {"gpt_iou": [], "gpt_pixauc": [], "gem_iou": [], "ideo_iou": []})

    for folder in FOLDERS:
        if not folder.exists():
            continue
        for f in sorted(folder.glob("k*_s*.json")):
            name = f.stem
            m = re.match(r"k(\d+)_s(\d+)(?:_(scratch|fz|rp\d+))?", name)
            if not m:
                continue
            k = int(m.group(1))
            variant = m.group(3) or "standard"
            data = json.loads(f.read_text(encoding="utf-8"))
            tests = data.get("tests", {})
            gpt = tests.get("test_gptimage", {})
            gem = tests.get("test_in_domain", {})
            ideo = tests.get("test_ideogram", {})
            key = f"{variant}_k{k}"
            if gpt:
                groups[key]["gpt_iou"].append(gpt["pixel"]["iou"])
                groups[key]["gpt_pixauc"].append(gpt["pixel"].get("pixel_auc", 0))
            if gem:
                groups[key]["gem_iou"].append(gem["pixel"]["iou"])
            if ideo:
                groups[key]["ideo_iou"].append(ideo["pixel"]["iou"])

    print(f"{'group':>16} {'n':>2} {'GPT IoU':>18} {'GPT PixAUC':>18} {'Gemini IoU':>18} {'Ideogram IoU':>18}")
    for key in sorted(groups):
        g = groups[key]
        n = len(g["gpt_iou"])
        print(f"{key:>16} {n:>2} {mean_std(g['gpt_iou']):>18} {mean_std(g['gpt_pixauc']):>18} "
              f"{mean_std(g['gem_iou']):>18} {mean_std(g['ideo_iou']):>18}")


if __name__ == "__main__":
    main()
