# RESULTS — Nhật ký thí nghiệm

Ghi mọi kết quả để dùng cho paper. **Không xoá kết quả cũ**, chỉ thêm.

## Protocol

- **v1 / in-domain**: train + test trên AIForge-Doc v1 (Gemini + Ideogram).
- **generator-shift** (protocol chính):
  - Train: v1 **Gemini** (2.912 forged + 2.912 authentic).
  - Test in-domain: v1 Gemini test (727 forged + 727 authentic).
  - Test unseen-generator #1: v1 **Ideogram** (422 forged + 422 authentic).
  - Test unseen-generator #2: v2 **GPT-Image-2** (613 forged + 613 authentic).

## Kết quả

### B1 — Baselines (2026-10-25)

**B1.1 ResNet18, train v1 đầy đủ, 3 epochs, 384px**

| Epoch | Loss | v1 test AUC | v1 AP | v2 (GPT) AUC | v2 AP |
|---|---|---|---|---|---|
| 1 | 0.558 | 0.850 | 0.881 | 0.465 | 0.397 |
| 2 | 0.396 | 0.879 | 0.911 | 0.430 | 0.377 |
| 3 | 0.348 | **0.892** | **0.920** | **0.461** | **0.396** |

Nhận xét: model tốt trên v1 nhưng **sụp trên v2** (AUC < 0.5) → bằng chứng generator shift.

**B1.2 Smoke test — protocol generator (128 mẫu train, 1 epoch, 256px)**
- in-domain (Gemini): AUC 0.641 / AP 0.546
- Ideogram (unseen): AUC 0.450 / AP 0.410
- GPT-Image-2 (unseen): AUC 0.495 / AP 0.464

(Chỉ để kiểm pipeline, không phải kết quả thật.)

**B1.3 Baseline chính thức — protocol generator-shift (ResNet18, 3 epochs, 384px)**

Train: v1 Gemini (2.912 forged + 2.912 authentic).

| Epoch | Loss | in-domain (Gemini) AUC / AP | unseen Ideogram AUC / AP | unseen GPT-Image-2 AUC / AP |
|---|---|---|---|---|
| 1 | 0.574 | 0.8961 / 0.9210 | 0.6061 / 0.5843 | 0.4775 / 0.4878 |
| 2 | 0.342 | 0.9047 / 0.9326 | 0.5742 / 0.6103 | 0.4506 / 0.4870 |
| 3 | 0.273 | **0.9185 / 0.9419** | **0.5810 / 0.5885** | **0.4777 / 0.5065** |

Nhận xét: in-domain rất tốt (AUC 0.92) nhưng **sụp khi đổi generator** — Ideogram chỉ ~0.58, GPT-Image-2 ~0.48 (ngang/ngược đoán mò). Đây là **bằng chứng định lượng cho generator shift** — đóng góp chính của paper.

### B2 — Segmentation (U-Net ResNet18, 3 epochs, 384px) — 2026-10-25

Train: v1 Gemini (2.912 forged + 2.912 authentic, mask rỗng cho authentic).

| Epoch | Loss | in-domain IoU / F1 / Dice / ImgAUC | unseen Ideogram IoU / F1 / Dice / ImgAUC | unseen GPT-Image-2 IoU / F1 / Dice / ImgAUC |
|---|---|---|---|---|
| 1 | 1.275 | 0.454 / 0.625 / 0.500 / 0.885 | 0.076 / 0.141 / 0.060 / 0.489 | 0.026 / 0.050 / 0.053 / 0.548 |
| 2 | 0.951 | 0.424 / 0.596 / 0.537 / 0.665 | 0.047 / 0.090 / 0.080 / 0.685 | 0.039 / 0.076 / 0.057 / 0.438 |
| 3 | 0.775 | **0.669 / 0.802 / 0.719 / 0.946** | **0.098 / 0.179 / 0.109 / 0.546** | **0.052 / 0.099 / 0.104 / 0.520** |

Nhận xét:
- **In-domain định vị tốt** (IoU 0.67, F1 0.80, ImgAUC 0.95).
- **Cross-generator vẫn kém** (Ideogram IoU ~0.10, GPT ~0.05) → generator shift vẫn còn, kể cả với localization.
- Đây là động lực cho **self-supervised + domain generalization** (đóng góp chính).

### B3 — Segmentation + augmentation (2026-10-25)

Augmentation: HorizontalFlip, BrightnessContrast, GaussianBlur, GaussNoise, JPEG (quality 30–90).

| Epoch | in-domain IoU / F1 / Dice / ImgAUC | Ideogram IoU / F1 / Dice / ImgAUC | GPT-Image-2 IoU / F1 / Dice / ImgAUC |
|---|---|---|---|
| 1 | 0.404 / 0.576 / 0.416 / 0.836 | 0.052 / 0.100 / 0.037 / 0.502 | 0.012 / 0.024 / 0.012 / 0.509 |
| 2 | 0.541 / 0.702 / 0.572 / 0.892 | 0.105 / 0.189 / 0.087 / 0.487 | 0.038 / 0.072 / 0.054 / 0.567 |
| 3 | 0.589 / 0.741 / 0.637 / 0.894 | 0.104 / 0.188 / 0.084 / 0.527 | 0.054 / 0.103 / 0.084 / 0.525 |

So sánh epoch 3 (có vs không augmentation):

| Bộ test | Không aug (B2) | Có aug (B3) |
|---|---|---|
| in-domain IoU | 0.669 | 0.589 (giảm) |
| Ideogram IoU | 0.098 | 0.104 (tăng nhẹ) |
| GPT-Image-2 IoU | 0.052 | 0.054 (tăng nhẹ) |
| GPT ImgAUC | 0.520 | 0.525 |

Nhận xét: augmentation **giảm in-domain**, **chỉ nhích nhẹ cross-generator** → generator shift **không chỉ do nén/độ phân giải**; cần phương pháp mạnh hơn (multi-generator training / self-supervised / domain generalization).

## Cảnh báo cần kiểm tra

- Model có thể "ăn gian" nhờ khác biệt **độ phân giải / kiểu nén** giữa authentic và forged → cần augmentation (JPEG, resize, blur) để kiểm chứng.
- AUC < 0.5 nghĩa là dự đoán **ngược** (không phải chỉ kém).
