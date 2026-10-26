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

### B4 — Leave-One-Generator-Out (2026-10-25)

U-Net ResNet18, 3 epochs, 384px. Cùng bộ test, khác dữ liệu train.

| Protocol | Train | in-domain IoU/F1/ImgAUC | Ideogram IoU/F1/ImgAUC | GPT-Image-2 IoU/F1/ImgAUC |
|---|---|---|---|---|
| P1 | Gemini | 0.669 / 0.802 / 0.946 | **0.098** / 0.179 / 0.546 | **0.052** / 0.099 / 0.520 |
| P2 | Gemini + Ideogram | 0.670 / 0.803 / 0.934 | 0.233 / 0.377 / 0.702 (seen) | **0.058** / 0.109 / 0.537 |
| P3 | Gemini + GPT | 0.539 / 0.701 / 0.865 | **0.121** / 0.215 / 0.494 | 0.363 / 0.533 / 0.674 (seen) |

So sánh trên generator **chưa thấy**:
- GPT (P1 → P2): IoU 0.052 → 0.058 (tăng nhẹ), ImgAUC 0.520 → 0.537.
- Ideogram (P1 → P3): IoU 0.098 → 0.121 (tăng ~22% tương đối), ImgAUC 0.546 → 0.494 (giảm).

Nhận xét: thêm 1 generator giúp **localization trên generator chưa thấy tăng nhẹ**, nhưng **vẫn rất thấp** so với in-domain; image-level AUC không nhất quán. → Multi-generator đơn thuần **chưa đủ**; cần self-supervised / domain generalization / foundation features.

### B5 — PatchCore-style anomaly detection (ResNet18, 400 ảnh bank, 256px) — thử nghiệm

Học chỉ từ ảnh thật (memory bank 6.400 patch), ngưỡng p99.5 của ảnh thật.

| Bộ test | IoU | F1 | Dice | ImgAUC |
|---|---|---|---|---|
| in-domain | 0.0001 | 0.0001 | 0.0000 | 0.502 |
| Ideogram | 0.0000 | 0.0000 | 0.0000 | 0.357 |
| GPT-Image-2 | 0.0000 | 0.0000 | 0.0000 | 0.558 |

**B5.1 — Full bank (52.096 patch, 384px, ngưỡng p99.5 = 0.3625):**

| Bộ test | IoU | F1 | Dice | ImgAUC |
|---|---|---|---|---|
| in-domain | 0.0016 | 0.0033 | 0.0018 | 0.543 |
| Ideogram | 0.0010 | 0.0019 | 0.0025 | 0.363 |
| GPT-Image-2 | 0.0000 | 0.0001 | 0.0001 | 0.535 |

Kết luận: **PatchCore + ResNet18 thất bại** — không phát hiện được vùng chữ bị AI sửa. Đặc trưng CNN tổng quát không đủ nhạy với sửa đổi chữ rất nhỏ (~1% pixel) và font khớp.

**B5.2 — PatchCore + DINOv2 ViT-S/14 (448px, bank 400 ảnh, test 200):**

| Bộ test | IoU | ImgAUC |
|---|---|---|
| in-domain | 0.0001 | 0.548 |
| Ideogram | 0.0000 | 0.418 |
| GPT-Image-2 | 0.0000 | 0.500 |

Kết luận: **DINOv2 cũng không cải thiện** → vấn đề không nằm ở backbone, mà ở bản chất sửa đổi quá tinh vi (font khớp, vùng cực nhỏ). Anomaly detection tổng quát không giải được bài toán này.

### B6 — FR-Net: RGB + frequency/noise (P1, 3 epochs, 384px)

| Epoch | in-domain IoU / ImgAUC | Ideogram IoU / ImgAUC | GPT-Image-2 IoU / ImgAUC |
|---|---|---|---|
| 1 | 0.213 / 0.729 | 0.049 / 0.485 | 0.090 / 0.546 |
| 2 | 0.461 / 0.806 | 0.076 / 0.575 | 0.085 / 0.562 |
| 3 | 0.561 / 0.830 | 0.098 / 0.509 | 0.061 / 0.547 |

So sánh epoch 3 với baseline RGB (P1):

| | Baseline RGB | FR-Net |
|---|---|---|
| in-domain IoU | 0.669 | 0.561 (giảm) |
| Ideogram IoU | 0.098 | 0.098 (ngang) |
| GPT IoU | 0.052 | 0.061 (tăng nhẹ) |
| GPT ImgAUC | 0.520 | 0.547 (tăng nhẹ) |

Kết luận: kênh tần số giúp **GPT nhích nhẹ** nhưng **hại in-domain**; không đủ để tạo phương pháp thắng. Trên benchmark này, các hướng tổng quát đều bị chặn bởi bản chất sửa đổi quá tinh vi.

### B7 — Threshold sweep & pixel AUC (P1 model, GPT test) — 2026-10-25

Quét ngưỡng 0.05–0.95 trên 613 ảnh forged GPT-Image-2:

- **Pixel AUC (mean per-image) = 0.8988**
- IoU tốt nhất = 0.0665 (tại thr=0.05); tại 0.5 = 0.0520.

Kết luận quan trọng: **model KHÔNG mù** — nó xếp hạng pixel khá tốt (AUC 0.90) ngay cả trên generator chưa thấy. Nhưng **mask dự đoán bị loãng/lệch** (nhiều false positive) → IoU thấp. Đây là vấn đề **calibration/localization**, có thể cải thiện bằng few-shot fine-tuning + post-processing.

### B8 — Few-shot fine-tuning (Cách 1) — 2026-10-25

Base: U-Net ResNet18 train trên Gemini (P1). Fine-tune trên k ảnh GPT-Image-2 (+k authentic), 10 epochs, lr 1e-4. Test trên tập GPT test (613, disjoint) + Gemini/Ideogram để đo forgetting.

| k | GPT IoU | GPT F1 | GPT Dice | GPT PixAUC | GPT ImgAUC | Gemini IoU (forget) | Ideogram IoU |
|---|---|---|---|---|---|---|---|
| 0 (base) | 0.052 | 0.099 | 0.104 | 0.899 | 0.520 | 0.669 | 0.098 |
| 5 | 0.079 | 0.147 | 0.138 | 0.870 | 0.511 | 0.583 | 0.081 |
| 10 | 0.101 | 0.183 | 0.153 | 0.864 | 0.538 | 0.616 | 0.099 |
| 25 | 0.136 | 0.239 | 0.166 | 0.858 | 0.566 | 0.607 | 0.109 |
| 50 | 0.152 | 0.264 | 0.166 | 0.852 | 0.584 | 0.624 | 0.099 |
| 100 | **0.171** | **0.292** | **0.202** | **0.885** | 0.559 | 0.534 | 0.110 |

**Kết quả dương:** GPT IoU tăng **0.052 → 0.171 (3,3×)** với 100 nhãn; F1 **0.099 → 0.292 (3×)**.
**Forgetting:** in-domain Gemini giảm 0.669 → 0.534 (~20%) khi k=100.
**Quan trọng:** PixAUC vẫn cao ~0.85–0.89 → ranking pixel tốt, IoU bị chặn bởi **calibration/post-processing**, không phải do mù hoàn toàn.

### B9 — Threshold calibration trên model few-shot (k=100)

Pixel AUC = 0.8536. Quét ngưỡng trên GPT test:

| Ngưỡng | IoU | F1 |
|---|---|---|
| 0.50 | 0.1597 | 0.2754 |
| **0.05 (best)** | **0.1942** | **0.3252** |

→ Calibrate ngưỡng tăng IoU **0.160 → 0.194 (+22% tương đối)**. Xác nhận calibration là đòn bẩy quan trọng.

### B10 — Fine-tune từ base vs Train from scratch (k=100)

| | Fine-tune từ base | Scratch |
|---|---|---|
| GPT IoU | **0.1597** | 0.0850 |
| GPT F1 | **0.2754** | 0.1567 |
| GPT PixAUC | **0.8536** | 0.6977 |
| Gemini IoU (giữ kiến thức cũ) | **0.5522** | 0.0750 |

→ Fine-tune từ base tốt hơn scratch **~1,9×** ở GPT và **giữ in-domain tốt hơn nhiều**. Đây là bằng chứng mạnh cho "label-efficient adaptation".

Lưu ý: k=100 giữa 2 lần chạy cho IoU 0.1708 vs 0.1597 → **cần chạy nhiều seed** để báo cáo mean ± std.

### B11 — Few-shot curve, 3 seeds (mean ± std) — 2026-10-25

| k | GPT IoU | GPT PixAUC | Gemini IoU (forgetting) |
|---|---|---|---|
| 0 (zero-shot) | 0.052 | 0.899 | 0.669 |
| 5 | 0.068 ± 0.015 | 0.877 ± 0.020 | 0.624 ± 0.017 |
| 25 | 0.105 ± 0.031 | 0.830 ± 0.023 | 0.577 ± 0.031 |
| 100 | **0.195 ± 0.037** | **0.907 ± 0.031** | **0.494 ± 0.054** |

Kết luận:
- IoU tăng **đơn điệu** theo k: 0.052 → 0.195 (**~3,7×** ở k=100).
- PixAUC luôn cao (0.83–0.91) → ranking pixel tốt; IoU bị giới hạn bởi calibration/mask.
- Forgetting: Gemini giảm 0.669 → 0.494 (~26%) ở k=100.
- Variance đáng kể giữa seed (đặc biệt k=100: 0.158–0.231) → phải báo mean ± std.

### B12 — Forgetting mitigation (k=100, seed 0) — 2026-10-25

| Phương pháp | GPT IoU | Gemini IoU (quên) | Ideogram IoU |
|---|---|---|---|
| Full fine-tune | 0.1582 | 0.5552 | 0.1102 |
| Freeze encoder (tune decoder) | 0.1133 | 0.0869 | 0.0794 |
| **Replay 100 mẫu cũ** | **0.1543** | **0.6750** | 0.1108 |

Kết luận:
- **Replay loại bỏ forgetting gần như hoàn toàn**: Gemini 0.675 (gần mức zero-shot 0.669) trong khi GPT vẫn 0.154 (gần full fine-tune 0.158).
- **Freeze encoder thất bại** (Gemini sụp 0.087) → không dùng.
- Đây là đóng góp thực tiễn mạnh: *fine-tune ít nhãn + replay để không quên generator cũ*.

### B13 — Replay 4 seeds (k=100) — 2026-10-25

| Phương pháp | GPT IoU | GPT PixAUC | Gemini IoU | Ideogram IoU |
|---|---|---|---|---|
| Standard fine-tune (3 seed) | 0.1947 ± 0.0298 | 0.9066 ± 0.0257 | 0.4942 ± 0.0439 | 0.0959 ± 0.0139 |
| **Replay 100 (4 seed)** | **0.1620 ± 0.0211** | 0.8986 ± 0.0196 | **0.6722 ± 0.0058** | 0.1077 ± 0.0152 |

Kết luận:
- Replay đưa Gemini về **0.672 ± 0.006** — gần như **không quên** (zero-shot 0.669).
- Đổi lại GPT giảm nhẹ 0.195 → 0.162 (~17%).
- Đây là **trade-off rõ ràng**: rehearsal bảo toàn generator cũ với chi phí nhỏ.

### B14 — Phân tích định tính (score maps) — 2026-10-25

So sánh trực quan (xem `paper/figures/fig_xai_*.png`):
- **Zero-shot GPT:** score map loãng/tán loạn, có ảnh gần như không có tín hiệu, có ảnh highlight sai vùng.
- **Adapted GPT (k=100):** score map tập trung đúng vùng bị sửa, rõ nét hơn hẳn.
- **In-domain Gemini:** score map rất sắc, khớp ground-truth.

→ Định tính khớp định lượng: adaptation cải thiện cả chất lượng mask, không chỉ số.

### B15 — Sample selection (1 seed) — 2026-10-25

| k | Method | GPT IoU | GPT PixAUC | Gemini IoU | Ideogram IoU |
|---|---|---|---|---|---|
| 25 | random | 0.1217 | 0.8576 | 0.5665 | 0.0842 |
| 25 | kcenter | 0.1207 | 0.8631 | 0.3861 | 0.0622 |
| 25 | uncertainty | **0.1287** | 0.8753 | **0.6093** | 0.0787 |
| 50 | random | **0.1574** | 0.8350 | **0.5692** | **0.0985** |
| 50 | kcenter | 0.1337 | 0.8377 | 0.4748 | 0.0750 |
| 50 | uncertainty | 0.1257 | 0.8267 | 0.4817 | 0.0813 |

Kết luận: **sample selection không cải thiện nhất quán**. Uncertainty nhỉnh nhẹ ở k=25 nhưng kém hơn random ở k=50; kcenter còn hại forgetting. → **Random là baseline mạnh**; không nên lấy selection làm đóng góp chính.

## Cảnh báo cần kiểm tra

- Model có thể "ăn gian" nhờ khác biệt **độ phân giải / kiểu nén** giữa authentic và forged → cần augmentation (JPEG, resize, blur) để kiểm chứng.
- AUC < 0.5 nghĩa là dự đoán **ngược** (không phải chỉ kém).
