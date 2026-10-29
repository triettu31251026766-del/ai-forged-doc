import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "calib"


def get(d, split, key="iou"):
    return d.get(split, {}).get(key, float("nan"))


def main():
    files = sorted(FOLDER.glob("*.json"))
    if not files:
        print("Chua co ket qua trong", FOLDER)
        return

    header = (f"{'tag':>30} {'T':>5} {'thr':>5} "
              f"{'GPTfx':>7} {'GPTcal':>7} {'GPT-AP':>7} {'GPT-R@1%':>8} "
              f"{'Gemfx':>7} {'Gemcal':>7} {'Ideofx':>7} {'Ideocal':>7}")
    print(header)
    print("-" * len(header))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        fx, cal = d.get("fixed", {}), d.get("calibrated", {})
        gpt_cal = cal.get("test_gptimage", {})
        tag = f.stem
        print(f"{tag:>30} {d.get('T', 1):>5.2f} {d.get('threshold', 0.5):>5.2f} "
              f"{get(fx, 'test_gptimage'):>7.4f} {get(cal, 'test_gptimage'):>7.4f} "
              f"{gpt_cal.get('image_ap', float('nan')):>7.4f} {gpt_cal.get('recall_at_1fpr', float('nan')):>8.4f} "
              f"{get(fx, 'test_in_domain'):>7.4f} {get(cal, 'test_in_domain'):>7.4f} "
              f"{get(fx, 'test_ideogram'):>7.4f} {get(cal, 'test_ideogram'):>7.4f}")


if __name__ == "__main__":
    main()
