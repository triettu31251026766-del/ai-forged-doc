# RECAP — Tổng hợp quyết định dự án

## 1. Đề tài

**Phát hiện giấy tờ tài chính bị AI sinh/sửa (AI-forged) dưới dịch chuyển miền và dịch chuyển bộ sinh, bằng học tự giám sát và định vị vùng giả mạo có khả năng giải thích.**

- EN: *Detecting AI-Forged Financial Documents under Domain and Generator Shift via Self-Supervised, Explainable Localization*

## 2. Vì sao chọn (ngách, khác hướng cũ)

- Hướng cũ "phát hiện giấy tờ bị chỉnh bằng Photoshop/splicing" đã bão hòa (nhiều nhóm làm).
- Hướng này tập trung **loại tấn công mới**: giấy tờ bị **AI sinh/sửa bằng diffusion + LLM inpainting**.
- Bổ sung trục **domain & generator shift**: không chỉ phát hiện, mà còn chịu được bộ sinh/loại giấy mới.

## 3. Research gap

> Các detector giả mạo truyền thống huấn luyện trên Photoshop/splicing, gần như mù với giấy tờ do AI sinh/sửa; và ngay cả các phương pháp mới cũng chưa chứng minh được khả năng tổng quát hoá sang bộ sinh AI và miền dữ liệu chưa từng thấy.

Bằng chứng (arXiv):
1. **AIForge-Doc** (2602.20569): DocTamper AUC 0.563, GPT-4o 0.509 (đoán mò).
2. **DOCFORGE-BENCH** (2603.01433): "document forgery detection remains an unsolved problem".
3. **GPT-Image-2** (2604.25213): người thật phân biệt chỉ đạt 0.501.

## 4. Hai trục "shift"

- **Domain shift:** train ảnh sạch (tiếng Anh, chụp phẳng) → test ảnh thực tế (nén, mờ, nghiêng, template/ngôn ngữ khác).
- **Generator shift:** train ảnh bị sửa bởi Gemini/Ideogram → test ảnh bị sửa bởi bộ sinh chưa thấy (GPT-Image-2, Stable Diffusion).

## 5. Đầu vào / dữ liệu

- Ảnh thật: SROIE, CORD, WildReceipt, XFUND.
- Ảnh bị AI sửa + mask: AIForge-Doc (~4.061 ảnh) + AIForge-Doc v2 (~3.066 ảnh).
- Tự sinh thêm ~1.000–2.000 ảnh (diffusion inpainting) làm tập "bộ sinh chưa thấy".

## 6. Phương pháp / pipeline

```
Ảnh giấy tờ (thật + AI + mask)
 → pretrain tự giám sát (MAE/DINO)
 → fine-tune định vị vùng bị sửa (segmentation)
 → đánh giá cross-generator + cross-domain
 → giải thích (Grad-CAM)
 → so sánh TruFor / DocTamper / GPT-4o
 → đầu ra: thật/giả + bản đồ vùng nghi ngờ
```

## 7. Thí nghiệm cốt lõi

| Thí nghiệm | Train | Test |
|---|---|---|
| Cross-generator | Gemini + Ideogram | GPT-Image-2 / Stable Diffusion (chưa thấy) |
| Cross-domain | Ảnh sạch, tiếng Anh | Ảnh thực tế: nén, mờ, tiếng Việt, template khác |

## 8. Phạm vi & giai đoạn

| Giai đoạn | Nội dung |
|---|---|
| 1 (lõi) | Dữ liệu + baseline (ResNet/ViT) + so sánh detector cũ |
| 2 | Pretrain tự giám sát + định vị vùng sửa |
| 3 | Cross-generator + cross-domain + giải thích |

## 9. Phân công gợi ý

| Mảng | Phụ trách |
|---|---|
| Dữ liệu + sinh ảnh | [Điền] |
| Model + pretrain | [Điền] |
| Đánh giá + biểu đồ | [Điền] |
| Paper + repo | [Điền] |

## 10. Kế hoạch sinh dữ liệu (ước lượng)

- SD 1.5 inpainting (512px, ~20 steps): ~5–15s/ảnh trên RTX 5060.
- 1.000–2.000 ảnh: ~2–3 ngày chạy qua đêm.
- Setup script + lọc chất lượng: 2–4 ngày công.

## 11. Hạ tầng & phân công máy

| Máy | Vai trò | Ghi chú |
|---|---|---|
| **Laptop RTX 5060** (Windows, Python 3.13, torch CUDA 12.8) | Train chính + phát triển code | 8.55 GB VRAM |
| **MacBook Pro M4** | **Chỉ dùng để train** khi cần huy động thêm | Apple Silicon, dùng PyTorch MPS hoặc MLX |

- **Vai trò của nhóm trưởng (Từ Minh Triết):** **kiểm soát chất lượng** toàn bộ bài báo cáo/paper — không giao QC cho máy khác.
- **MacBook M4 chỉ là máy tính toán**, không tham gia quyết định nội dung/đánh giá chất lượng.
- **Tái lập giữa 2 máy:** luôn set `seed`, ghi rõ backend (`cuda` / `mps` / `cpu`), và dùng chung code + config trong repo. Kết quả train từ M4 đẩy về repo (không commit dữ liệu nặng).

## 12. Tài liệu liên quan

- `README.md` — tổng quan + quick start
- `GLOSSARY.md` — thuật ngữ cần biết
