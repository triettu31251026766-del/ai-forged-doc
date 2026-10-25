import json
import random
from pathlib import Path

from src.data.dataset import list_images

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data" / "raw" / "AIForge-Doc-v1"
V2 = ROOT / "data" / "raw" / "AIForge-Doc-v2"

GEMINI = "gemini-nano"
IDEOGRAM = "qwen-inpaint"
GPTIMAGE = "openai-gpt-image-2"


def load_meta(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def forged_items(meta, base, tool=None, split=None):
    base = Path(base)
    items = []
    for r in meta:
        if tool is not None and r.get("assigned_tool") != tool:
            continue
        if split is not None and r.get("split") != split:
            continue
        if r.get("image"):
            img = base / r["image"]
            msk = base / r["mask"]
        else:
            sd = "TrainingSet" if r.get("split") == "training" else "TestingSet"
            img = base / sd / "images" / f'{r["new_id"]}.png'
            msk = base / sd / "masks" / f'{r["new_id"]}.png'
        if img.exists() and msk.exists():
            items.append((img, msk, 1))
    return items


def authentic_items(base, split):
    base = Path(base)
    sd = "TrainingSet" if split == "training" else "TestingSet"
    d = base / sd / "authentic"
    if not d.exists():
        d = V1 / sd / "authentic"
    return [(p, None, 0) for p in list_images(d)]


def _balanced(forged, authentic, seed):
    rng = random.Random(seed)
    if len(authentic) >= len(forged):
        chosen = rng.sample(authentic, len(forged))
    else:
        chosen = authentic
    return forged + chosen


def build_generator_splits(seed=0):
    """Protocol generator-shift: train tren Gemini, test tren Gemini (in-domain),
    Ideogram (unseen) va GPT-Image-2 (unseen)."""
    v1_meta = load_meta(V1 / "metadata.jsonl")
    v2_meta = load_meta(V2 / "metadata.jsonl")

    gem_train = forged_items(v1_meta, V1, tool=GEMINI, split="training")
    gem_test = forged_items(v1_meta, V1, tool=GEMINI, split="testing")
    ideo_all = forged_items(v1_meta, V1, tool=IDEOGRAM)
    gpt_test = forged_items(v2_meta, V2, tool=GPTIMAGE, split="testing")

    auth_train = authentic_items(V1, "training")
    auth_test = authentic_items(V1, "testing")

    splits = {
        "train": _balanced(gem_train, auth_train, seed),
        "test_in_domain": _balanced(gem_test, auth_test, seed),
        "test_ideogram": _balanced(ideo_all, auth_test, seed),
        "test_gptimage": _balanced(gpt_test, auth_test, seed),
    }
    return splits


def build_v1_splits(seed=0):
    """Protocol don gian: train/test tren toan bo v1 (Gemini + Ideogram)."""
    v1_meta = load_meta(V1 / "metadata.jsonl")
    train_f = forged_items(v1_meta, V1, split="training")
    test_f = forged_items(v1_meta, V1, split="testing")
    auth_train = authentic_items(V1, "training")
    auth_test = authentic_items(V1, "testing")
    return {
        "train": _balanced(train_f, auth_train, seed),
        "test_in_domain": _balanced(test_f, auth_test, seed),
    }
