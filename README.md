# AI-Forged Financial Document Detection under Domain & Generator Shift

Phát hiện giấy tờ tài chính bị AI sinh/sửa (AI-forged), **dưới dịch chuyển miền và dịch chuyển bộ sinh**, bằng học tự giám sát và định vị vùng giả mạo có khả năng giải thích.

## Vấn đề

Kẻ gian ngày càng dùng **AI sinh ảnh (diffusion) và LLM inpainting** để sửa số tiền / trường thông tin trên hóa đơn, biên lai, hợp đồng, giấy tờ KYC. Các detector giả mạo truyền thống (DocTamper, TruFor) và cả VLM (GPT-4o) **gần như mù** với loại tấn công này (AUC ~0.5–0.56).

## Research gap

Các detector hiện tại:
1. Được huấn luyện trên thao tác Photoshop/splicing truyền thống;
2. Chưa phát hiện được giấy tờ bị chỉnh bằng AI sinh ảnh/LLM inpainting;
3. Và ngay cả khi phát hiện được, **chưa chứng minh chịu được khi gặp bộ sinh AI mới và loại giấy/điều kiện chụp mới**.

Bằng chứng:
- AIForge-Doc (arXiv 2602.20569)
- DOCFORGE-BENCH (arXiv 2603.01433)
- GPT-Image-2 (arXiv 2604.25213)

## Hai trục "shift" (trọng tâm đóng góp)

- **Domain shift (dịch chuyển miền):** train trên ảnh sạch (tiếng Anh, chụp phẳng) → test trên ảnh thực tế (nén JPEG, mờ, nghiêng, template/ngôn ngữ khác).
- **Generator shift (dịch chuyển bộ sinh):** train trên ảnh bị sửa bởi Gemini/Ideogram → test trên ảnh bị sửa bởi bộ sinh chưa từng thấy (GPT-Image-2, Stable Diffusion).

Mục tiêu: mô hình không học vẹt dấu vết của một bộ sinh/một miền, mà **tổng quát hoá** sang cái mới.

## Dữ liệu

- Ảnh thật: SROIE, CORD, WildReceipt, XFUND
- Ảnh bị AI sửa + mask: AIForge-Doc v1 + v2
- Tự sinh thêm bằng diffusion inpainting (Stable Diffusion) để làm tập "bộ sinh chưa thấy"

## Phương pháp

1. **Baseline**: U-Net ResNet-18 segmentation ở 384px (letterbox), train trên Gemini.
2. **Few-shot adaptation**: fine-tune với k = 5/25/100 nhãn từ bộ sinh mới → IoU 0.195 ± 0.037 (3 seed).
3. **Rehearsal**: 100 mẫu bộ sinh cũ → khôi phục in-domain 0.672 ± 0.006, không quên.
4. **TL-HRS** (đóng góp chính): adapt một **patch segmenter** native-resolution bằng đúng 100 nhãn (trần 0.459 → 0.677), rồi **quét dày cửa sổ 384px** (stride 96, chọn theo top-1% confidence) thay cho đỉnh coarse → **0.545 ± 0.012** trên bộ sinh chưa thấy (2.8× baseline, 82% trần).
5. **Đánh giá**: cross-generator (GPT-Image-2, Ideogram), label-efficiency, ablation định vị; giải thích bằng score maps.

## Kết quả chính (full forged test, metric full-image)

| | IoU |
|---|---|
| Full-image few-shot k=100 (3 seed) | 0.195 ± 0.037 |
| Patch adapted, GT box (trần) | 0.677 ± 0.009 |
| Đỉnh coarse + patch | 0.321 |
| **TL-HRS (quét dày, 3 seed)** | **0.545 ± 0.012** |
| Ideogram (bộ sinh chưa thấy) | 0.284 (vs 0.114 full-image; oracle 0.510) |
| Label efficiency: trần tại k=25 | 0.584 (86% của k=100) |

Chi tiết: `paper/main.md` (bản thảo EN), `paper/TOM_TAT_TIENG_VIET.md`, `docs/RESULTS.md` (nhật ký B1–B45).

## Thí nghiệm cốt lõi

| Thí nghiệm | Train | Test |
|---|---|---|
| Cross-generator | Gemini + Ideogram | GPT-Image-2 / Stable Diffusion (chưa thấy) |
| Cross-domain | Ảnh sạch, tiếng Anh | Ảnh thực tế: nén, mờ, tiếng Việt, template khác |

## Cấu trúc

```
ai-forged-doc/
├── data/        raw, processed
├── notebooks/   EDA, thử nghiệm
├── src/
│   ├── data/    xử lý dữ liệu, sinh ảnh AI
│   ├── models/  model, segmentation head
│   ├── train.py
│   └── evaluate.py
├── configs/     yaml thí nghiệm
├── scripts/     chạy hàng loạt
├── results/     kết quả, checkpoint
├── tests/       unit test
└── paper/       bản thảo
```

## Bắt đầu

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Tài liệu

- `paper/main.md` — bản thảo tiếng Anh (method TL-HRS)
- `paper/TOM_TAT_TIENG_VIET.md` — tóm tắt tiếng Việt
- `docs/RESULTS.md` — nhật ký toàn bộ thí nghiệm (B1–B45)
- `RECAP.md` — tổng hợp quyết định dự án
- `GLOSSARY.md` — thuật ngữ cần biết
