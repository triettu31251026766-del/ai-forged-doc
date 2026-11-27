# Bước 5 — Kaggle runbook (DocTamper DTD + SegFormer-B2)

Mục tiêu:
1. Chạy **DocTamper DTD** (official checkpoint) trên **613 ảnh GPT-Image-2 forged** → IoU/F1 để so với TruFor và TL-HRS.
2. Fine-tune **SegFormer-B2** với **đúng 100 nhãn** GPT (+100 authentic), protocol P1, so với U-Net ResNet-18 (IoU 0.195 ± 0.037).

Không chạy training trong phiên này. Tất cả lệnh dưới đây chạy trên Kaggle (Linux, GPU T4/P100).

---

## 0. Chuẩn bị dữ liệu trên Kaggle

Upload `AIForge-Doc-v1` và `AIForge-Doc-v2` dưới dạng Kaggle Dataset, ví dụ:

```
/kaggle/input/aiforgedoc/AIForge-Doc-v1
/kaggle/input/aiforgedoc/AIForge-Doc-v2
```

Clone repo và symlink:

```bash
git clone <repo-url> ai-forged-doc
cd ai-forged-doc
mkdir -p data/raw
ln -s /kaggle/input/aiforgedoc/AIForge-Doc-v1 data/raw/AIForge-Doc-v1
ln -s /kaggle/input/aiforgedoc/AIForge-Doc-v2 data/raw/AIForge-Doc-v2
```

Kiểm tra nhanh:

```bash
python - <<'PY'
from src.data.splits import build_protocol_splits
s = build_protocol_splits("p1")
print({k: len(v) for k, v in s.items()})
PY
```

Kỳ vọng: `train` ≈ 5824, `test_gptimage` = 1226, `test_in_domain` = 1454, `test_ideogram` = 170.

---

## 1. SegFormer-B2 với 100 nhãn

Cài thêm:

```bash
pip install -q transformers==4.44.2
```

### 1a. (Tuỳ chọn, đúng P1) train base trên Gemini

Nếu bỏ qua bước này, stage 2 là baseline “ADE → 100 nhãn”, không phải P1 adaptation.

```bash
python scripts/finetune_segformer_b2.py \
  --mode gemini --epochs 10 --batch-size 4 --size 384 \
  --out results/segformer_b2_gemini
```

Thời gian ước tính: 2–4 h trên T4 (5.8k ảnh, 384²). Có thể thêm `--max-train 1000` để smoke test.

### 1b. Fine-tune 100 nhãn

Có Gemini base:

```bash
python scripts/finetune_segformer_b2.py \
  --mode k100 --k 100 --seed 0 --epochs 10 --batch-size 4 --size 384 \
  --init results/segformer_b2_gemini/model.pth \
  --out results/segformer_b2_k100_s0
```

Không có Gemini base (baseline kiến trúc):

```bash
python scripts/finetune_segformer_b2.py \
  --mode k100 --k 100 --seed 0 --epochs 10 --batch-size 4 --size 384 \
  --out results/segformer_b2_k100_ade_s0
```

Lặp `--seed 1,2` nếu cần mean ± std (3 seed).

Kết quả: `results/segformer_*/results.json` gồm `iou_pooled`, `iou_mean`, `f1`, `image_auc`, `image_ap`, `recall_at_1fpr` cho GPT / Gemini / Ideogram.

So sánh với bài:
- U-Net full-image k=100: IoU **0.195 ± 0.037**, F1 **0.325**, Dice **0.228**.
- Nếu SegFormer-B2 vượt trần này, đó là kết quả dương; nếu không, báo cáo như negative result về kiến trúc.

---

## 2. DocTamper DTD trên 613 ảnh GPT

### 2a. Clone + weights

```bash
git clone https://github.com/qcf-568/DocTamper
```

Tải weights (Google Drive folder trong `DocTamper/models/Readme.md`) vào:

```
DocTamper/models/pths/dtd_doctamper.pth
DocTamper/models/pths/swin_imagenet.pt
DocTamper/models/pths/vph_imagenet.pt
```

Có thể dùng `gdown --folder <drive-folder-url>`.

### 2b. Dependencies (best-effort, không cài bản pin cũ)

Kaggle đã có torch 2.x / numpy / opencv / smp. Chỉ cài phần thiếu:

```bash
pip install -q jpegio lmdb timm==0.4.12 albumentations==1.3.0 \
  efficientnet_pytorch yacs six segmentation_models_pytorch
```

Nếu `jpegio` build fail: dừng, ghi log; giữ TruFor làm baseline công bố (bài đã có).

### 2c. Đóng gói 613 ảnh thành LMDB

Chạy từ repo `ai-forged-doc`:

```bash
python scripts/make_doctamper_lmdb.py \
  --out-dir DocTamper/models --name AIForgeGPT613 \
  --quality 75 --max-side 1024 --multiple 32
```

Ghi ra:
- `DocTamper/models/AIForgeGPT613/{data.mdb,lock.mdb}`
- `DocTamper/models/qt_table.pk`
- `DocTamper/models/pks/AIForgeGPT613_75.pk`
- `DocTamper/models/AIForgeGPT613_manifest.json`

Lưu ý: `--max-side 1024` và pad bội số 32 để tránh OOM; ghi rõ trong paper rằng DocTamper được cap 1024 px (giống TruFor).

### 2d. Chạy eval chính thức

```bash
cd DocTamper/models
CUDA_VISIBLE_DEVICES=0 python eval_dtd.py \
  --data_root ./ --lmdb_name AIForgeGPT613 \
  --pth pths/dtd_doctamper.pth --minq 75
```

Log cuối có dạng:

```
[val] iou:[... <bg> <tampered>] pre:... rec:... f1:...
```

Ghi lại:
- `iou[1]` = tampered IoU (pooled pixel)
- `pre`, `rec`, `f1`

Lưu vào `results/doctamper_dtd.json`:

```json
{
  "dataset": "AIForge-Doc v2 test, 613 forged GPT-Image-2",
  "max_side": 1024,
  "minq": 75,
  "iou_pooled": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "note": "official DTD checkpoint, zero-shot"
}
```

Sau đó tôi sẽ viết §6.9b/§6.15 so sánh DocTamper DTD vs TruFor vs TL-HRS.

---

## 3. Checklist mang kết quả về repo

- [ ] `results/segformer_b2_k100_s0/results.json`
- [ ] `results/segformer_b2_k100_s1/results.json`
- [ ] `results/segformer_b2_k100_s2/results.json`
- [ ] `results/doctamper_dtd.json`
- [ ] Log DocTamper (`eval_dtd` stdout)
- [ ] Ghi chú cấu hình: GPU, thời gian, `--max-side`, seed

Sau khi có 2 mục này: viết cover letter + Highlights cho ESWA.
