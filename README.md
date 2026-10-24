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

1. Pretrain tự giám sát (MAE/DINO) trên ảnh tài liệu chưa nhãn
2. Fine-tune định vị vùng giả mạo (segmentation)
3. Đánh giá **cross-generator** và **cross-domain**
4. Giải thích (Grad-CAM, bản đồ nhiệt)
5. So sánh baseline: TruFor, DocTamper, GPT-4o

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

- `RECAP.md` — tổng hợp toàn bộ quyết định
- `GLOSSARY.md` — thuật ngữ cần biết
