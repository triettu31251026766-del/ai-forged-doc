import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "raw" / "AIForge-Doc-v1"
V2 = ROOT / "data" / "raw" / "AIForge-Doc-v2"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(name, base):
    rows = load(base / "metadata.jsonl")
    print("=" * 60)
    print(f"{name}: {len(rows)} records")

    by_split = Counter(r.get("split") for r in rows)
    print("split:", dict(by_split))

    tools = Counter(r.get("assigned_tool") for r in rows)
    print("tool:", dict(tools))

    combo = Counter((r.get("split"), r.get("assigned_tool")) for r in rows)
    print("split x tool:")
    for k, v in sorted(combo.items(), key=lambda x: str(x[0])):
        print(f"   {k}: {v}")

    src = Counter(r.get("source_dataset") for r in rows)
    print("source:", dict(src))

    # check new_id uniqueness + files exist
    ids = [r.get("new_id") for r in rows]
    print("unique new_id:", len(set(ids)), "/", len(ids))

    missing = 0
    for r in rows[:]:
        img = base / r["image"] if "image" in r else base / f"{r['split'].capitalize()}Set/images/{r['new_id']}.png"
        if not img.exists():
            missing += 1
    print("missing image files:", missing)


if __name__ == "__main__":
    summarize("v1", V1)
    summarize("v2", V2)
