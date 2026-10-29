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

### B16 — Calibration-aware + Replay selection (k=100, seed 0) — 2026-10-26

| tag | T | thr | GPT fixed IoU | GPT cal IoU | GPT AP | GPT R@1%FPR | Gemini fixed | Gemini cal |
|---|---|---|---|---|---|---|---|---|
| both | 1.40 | 0.55 | 0.1308 | 0.1300 | 0.5402 | 0.0179 | 0.3915 | 0.3982 |
| both + replay hard | 1.40 | 0.05 | 0.1206 | 0.1267 | 0.5844 | 0.0163 | 0.4508 | 0.3181 |
| both + replay random | 1.50 | 0.05 | 0.0931 | 0.1169 | **0.5910** | **0.0245** | **0.6062** | 0.4590 |
| both + replay uncertainty | 1.50 | 0.25 | 0.0855 | 0.0909 | 0.5459 | 0.0098 | 0.5366 | 0.4861 |
| temp only | 1.40 | 0.50 | 0.1340 | 0.1340 | 0.5611 | 0.0163 | 0.4308 | 0.4308 |
| threshold only | 1.00 | 0.05 | 0.1230 | 0.1253 | 0.5538 | 0.0033 | 0.4312 | 0.3689 |

Nhận xét:
- **Temperature scaling không đổi IoU** (chỉ đổi ECE) — vì nó không đổi thứ hạng, chỉ đổi xác suất.
- **Threshold calibration giúp target generator rất ít**, và **làm hại Gemini** (cal < fixed ở replay) → **calibration mang tính generator-specific**.
- **Replay random**: giữ Gemini fixed 0.606 (gần zero-shot 0.669), GPT AP cao nhất 0.591; nhưng calibration trên GPT kéo Gemini cal xuống 0.459.
- Lưu ý: lần này train chỉ 160 mẫu (do tách 20% để cal) nên IoU thấp hơn standard k=100 (200 mẫu) — **không so trực tiếp**.

→ Kết luận: CAA dạng temperature+global threshold **không phải là đòn bẩy tốt**. Cần **threshold thích ứng theo từng ảnh** (per-image adaptive) để tránh lệch generator.

### B17 — Per-image adaptive threshold (top-p%) — 2026-10-26

| Model / split | fixed (0.5) IoU | p0.5% | p1% | p2% | p5% |
|---|---|---|---|---|---|
| Standard k=100, GPT | **0.1582** | 0.1210 | 0.1405 | 0.1298 | 0.0847 |
| Replay random, GPT (T=1.5) | 0.1079 | 0.1016 | **0.1185** | 0.1108 | 0.0744 |
| Standard k=100, Gemini | **0.5552** | 0.3109 | 0.3654 | 0.2898 | 0.1417 |

Kết luận:
- **Ngưỡng thích ứng theo ảnh KHÔNG thắng fixed 0.5** (chỉ nhỉnh nhẹ ở replay GPT: 0.108 → 0.119).
- Trên in-domain, adaptive **tệ hơn nhiều** (0.555 → 0.365).
- → **Threshold/calibration KHÔNG phải nút thắt.** Nút thắt là **độ chính xác định vị (localization precision)**, không phải ngưỡng.

### B18 — Error analysis: theo source & diện tích vùng sửa — 2026-10-26

**Theo source (mean IoU):**

| Source | Standard k=100 (GPT) | Replay random (GPT) | In-domain Gemini |
|---|---|---|---|
| CORD | **0.251** | 0.195 | 0.489 |
| WildReceipt | 0.151 | 0.092 | 0.568 |
| SROIE | 0.107 | 0.059 | 0.575 |
| XFUND | **0.013** | 0.010 | 0.372 |

**Theo diện tích vùng sửa (mean IoU):**

| Area | Standard (GPT) | In-domain Gemini |
|---|---|---|
| <0.5% | 0.170 | 0.413 |
| 0.5–1% | 0.131 | 0.572 |
| 1–2% | 0.160 | 0.619 |
| >2% | 0.189 | 0.577 |

Kết luận:
- **Cross-generator phụ thuộc mạnh vào loại giấy**: CORD (trường giá) tốt nhất, **XFUND (biểu mẫu đa ngôn ngữ) gần như thất bại**.
- In-domain: vùng sửa lớn hơn → IoU cao hơn (0.41 → 0.62).
- Cross-generator: ảnh hưởng diện tích **yếu/không đơn điệu** → lỗi không chỉ do vùng nhỏ, mà do **loại tài liệu/ngôn ngữ**.
- Replay giảm nhẹ hiệu năng target trên mọi source (đúng trade-off).

### B19 — Tiled native-resolution inference (negative) — 2026-10-26

| Setting | IoU | F1 | Dice | PixAUC | ImgAUC |
|---|---|---|---|---|---|
| Letterbox 384 (baseline) | 0.158 | 0.273 | 0.203 | 0.854 | 0.560 |
| Tiled 384/192, native (150 ảnh) | 0.0047 | 0.0094 | 0.0187 | 0.774 | 0.471 |

Kết luận: tiling ở độ phân giải gốc **thất bại** vì **lệch scale** — model được train trên ảnh thu nhỏ toàn trang, còn tile có chữ to hơn nhiều. Muốn tận dụng high-res phải **train lại ở patch scale**. Đây là hướng phương pháp (không phải chỉ inference).

### B21 — TL-HR Step 1: Patch-level upper bound (GT bbox) — 2026-10-26

Train patch segmenter (U-Net ResNet18) trên **context crop 384 native** quanh GT bbox của Gemini (jitter 0.25) + negatives (chữ thường cùng dòng + ảnh thật). Test dùng GT bbox (upper bound).

| Test | Full IoU | Full F1 | Patch IoU | PixAUC | PixAP | ImgAP | R@1%FPR |
|---|---|---|---|---|---|---|---|
| **GPT-Image-2 (unseen)** | **0.459** | 0.629 | 0.461 | 0.996 | 0.799 | 0.546 | 0.064 |
| Gemini (in-domain) | 0.557 | 0.715 | 0.568 | 0.996 | 0.889 | 0.608 | 0.053 |

So sánh: GPT full IoU **0.158 → 0.459 (2,9×)** so với U-Net letterbox.

Kết luận: **trần của phương pháp TL-HR rất cao**. Patch-level native-res training khắc phục đúng lỗi scale. Bước tiếp: thay GT bbox bằng **OCR proposals** (Bước 2) để thành phương pháp thật.

### B22 — TL-HR Step 2: OCR proposals (ghép tất cả box) — 2026-10-26

| Split | IoU | F1 | Dice | PixAUC | PixAP | ImgAP | R@1% | avg_boxes |
|---|---|---|---|---|---|---|---|---|
| GPT (60 ảnh) | 0.0514 | 0.0978 | 0.1366 | 0.9279 | 0.3331 | **0.6565** | **0.1852** | 23.3 |

Nhận xét:
- **Detection mạnh**: ImgAP 0.657 và R@1% 0.185 **cao hơn cả upper bound** (0.546 / 0.064).
- **Localization loãng**: ghép cả 23 box → nhiều false positive → IoU thấp.
- → Cần **chọn top-k box** theo phản hồi của patch model (không ghép tất cả).

### B23 — Proposal recall của OCR (GPT, 100 ảnh) — 2026-10-26

| Chỉ số | Giá trị |
|---|---|
| Center in OCR box | 82.0% |
| Center covered by 384 crop | **94.0%** |
| GT pixel coverage | **93.8%** |
| Best IoU (mean) | 0.544 |
| Recall@IoU≥0.1 / 0.3 / 0.5 | 78% / 75% / 66% |

By source: CORD 0.691, SROIE 0.701, WildReceipt 0.553, **XFUND 0.093 (coverage 62%)**.

Kết luận:
- **Proposal tốt**: 94% tâm vùng sửa được crop 384 phủ → patch model **có thể thấy** field.
- **Nút thắt là SELECTION**: chọn sai box trong 24 box.
- **XFUND là ngoại lệ**: OCR bỏ sót/không đọc được form đa ngôn ngữ.

### B24 — Chọn box theo score trong vùng (max/mean/top5) — 2026-10-26

| Score mode | top-1 IoU | top-1 F1 | top-1 Dice | PixAUC | PixAP |
|---|---|---|---|---|---|
| max | 0.1709 | 0.2918 | 0.2901 | 0.7782 | 0.3890 |
| mean | 0.1832 | 0.3096 | 0.2483 | 0.7113 | 0.3434 |
| **top5** | **0.2088** | **0.3455** | **0.3209** | 0.7970 | 0.4246 |

Kết luận:
- **top5 trong đúng vùng box** là mode tốt nhất → IoU 0.209 (cao hơn baseline 0.195 một chút).
- Nhưng vẫn xa trần 0.459 → **selection vẫn là nút thắt**.
- Cần **box classifier** chuyên dụng (thay vì dùng phản hồi segmenter) để chọn đúng field.

### B25 — Box classifier cho selection (GPT, 60 ảnh) — 2026-10-26

| Method | IoU | F1 | Dice | PixAUC | PixAP | ImgAP | R@1% | sel center |
|---|---|---|---|---|---|---|---|---|
| Box classifier top-1 | **0.2444** | 0.3928 | 0.3435 | 0.8435 | 0.4854 | **0.6755** | 0.1111 | **25.9%** |

So sánh: baseline 0.195; top5 selector 0.209; **box classifier 0.244**; trần GT bbox 0.459.

Nhận xét:
- Box classifier cải thiện **IoU** và **ImgAP** (tốt nhất tới giờ).
- Nhưng **selection vẫn yếu** (center_in_selected 25.9%) — crop classifier quá rộng, dễ bắt nhầm chữ thường.
- → Cần **digit-aware selection** (vùng sửa thường là số) + classifier crop chặt hơn.

### B26-B30 — TL-HR Step 3: selection nâng cao (GPT, 60 ảnh) — 2026-10-27

| Thí nghiệm | IoU | F1 | PixAUC | ImgAP | sel center | mean_dist | crop_hit |
|---|---|---|---|---|---|---|---|
| Oracle (GT center), 1 patch | 0.3141 | 0.4780 | 0.9680 | — | 100% | 0 | — |
| Oracle, grid3 best | **0.3300** | 0.5390 | 0.8397 | — | 100% | 0 | — |
| Oracle, grid3 union | 0.1393 | 0.2446 | 0.9686 | — | 100% | 0 | — |
| Box cls v1 (384), 1 patch | **0.2317** | 0.3762 | 0.8611 | 0.6700 | 29.6% | 225px | 81.5% |
| Box cls v2 (192, neg sửa), 1 patch | 0.1998 | 0.3333 | 0.8427 | 0.5243 | 59.3% | 188px | 78.1% |
| v2 + grid3 best | 0.1660 | 0.2847 | 0.8119 | 0.5243 | 59.3% | 188px | 74.1% |
| v1 + line-merge | 0.2229 | 0.3646 | 0.8420 | 0.6700 | 29.6% | 231px | 77.8% |
| v2 + line-merge | 0.2002 | 0.3336 | 0.8427 | 0.5243 | 59.3% | 197px | 74.1% |
| v2 + few-shot adapt k=100 | 0.1890 | 0.3179 | 0.7500 | 0.5675 | 44.4% | 263px | 59.3% |
| v1 + few-shot adapt k=100 | 0.2359 | 0.3818 | 0.8289 | 0.6672 | 29.6% | 205px | 74.1% |

Nhận xét (quan trọng):
- **Trần trên trên đúng 60 ảnh này là 0.314** (single) / **0.330** (grid best), không phải 0.459 — 0.459 là full test.
- GT bbox rộng (w p50 148, p90 381, p95 466; 10% > 384) → clipping là một phần giới hạn.
- **Grid union làm sụp IoU** (FP cộng dồn); grid **best** chỉ nhích oracle, hại pipeline.
- **Line-merge và few-shot adapt đều không cải thiện** localization.
- → **Tín hiệu "field bị sửa" mang tính tương phản với context**, không phải artifact cục bộ; classifier crop không bắt được.
- Pipeline thật tốt nhất hiện tại: **v1 single 0.232**; gap tới trần 0.314 = **localization**, không phải detection (ImgAP 0.67).

### B31 — Coarse-to-fine / context-consistency (không train, GPT 60 ảnh) — 2026-10-27

Dùng heatmap của mô hình few-shot toàn ảnh (k=100, letterbox 384) để chọn vùng, rồi patch_seg 384 native tinh chỉnh.

| Chiến lược | IoU | F1 | Dice | PixAUC | PixAP |
|---|---|---|---|---|---|
| **peak (đỉnh heatmap → patch native)** | **0.2388** | 0.3855 | 0.2327 | 0.6985 | 0.2807 |
| peak_refine (centroid 1 bước) | 0.2157 | 0.3548 | 0.2419 | 0.7232 | 0.3031 |
| peak_refine2 (centroid 2 bước) | 0.2025 | 0.3368 | 0.2544 | 0.7307 | 0.2994 |
| ocr_global (box có heatmap TB cao nhất) | 0.1245 | 0.2214 | 0.2090 | 0.7086 | 0.2575 |
| ocr_zscore (MAD z-score từng ảnh) | 0.1245 | 0.2214 | 0.2090 | 0.7086 | 0.2575 |
| fusion (global x patch) | 0.2094 | 0.3463 | 0.1624 | 0.6996 | 0.2502 |

So với v1 single **0.2317**. Kết luận:
- **Coarse-to-fine peak (0.2388) là pipeline tự động tốt nhất trên subset này** (nhỉnh hơn box classifier 0.232), và **không cần OCR** để chọn vùng — chỉ cần heatmap toàn ảnh → cắt 384 native → patch_seg.
- Pixel AUC thấp hơn (0.70) vì dự đoán chỉ phủ 1 patch (thưa), không phải vì xếp hạng pixel kém.
- `ocr_global` yếu (0.125): heatmap toàn ảnh 384 quá thô để **xếp hạng box**.
- `ocr_zscore` **trùng khít** `ocr_global` vì z-score là biến đổi **đơn điệu theo từng ảnh** → không đổi argmax. "Context-consistency" kiểu chuẩn hoá điểm là **vô nghĩa**; muốn có tác dụng phải so sánh **đặc trưng từng field** với các field khác, không phải chuẩn hoá scalar score.
- Vẫn xa trần 0.314 (subset) / 0.459 (full) → **selection/context-contrast vẫn là nút thắt**.
- **Self-refinement (centroid mask → cắt lại → patch_seg) làm IoU GIẢM** (0.239 → 0.216 → 0.203): centroid trôi sang chữ lân cận. → **lệch tâm không phải nút thắt chính**; chọn đúng field mới là.

### B32 — Context-contrast selector học được (DINOv2) — GPT 60 ảnh — 2026-10-27

Field embedding DINOv2 ViT-S; context = embedding **toàn ảnh**; head MLP([v, μ, v−μ]); train 1.200 ảnh Gemini (pos = field sửa, neg = crop cùng dòng không chồng).

| Selector | IoU | sel mean IoU(box,GT) | center_in_selected |
|---|---|---|---|
| cc_sel (học) | 0.1954 | 0.0969 | 11.1% |
| dino_zero (‖v−μ‖) | 0.0162 | 0.0000 | 0.0% |
| cc_fuse (cc + heatmap global) | **0.2392** | 0.2472 | 29.6% |

Kết luận:
- **Context mức doc-embedding quá thô**; head học "text vs background" chứ không học "field bị sửa vs field thường" → center 11%.
- `dino_zero` chọn sai hoàn toàn (0%).
- `cc_fuse` suy biến về global peak (0.239) → không thêm thông tin.
- → Cần context **cục bộ giữa các field lân cận ở native resolution**, không phải embedding toàn ảnh.

### B33 — Quick bundle + global-stage resolution — GPT 60 ảnh — 2026-10-27

| Chiến lược | IoU | Ghi chú |
|---|---|---|
| single_peak (384) | **0.2388–0.2422** | tốt nhất tự động |
| multipeak (top-3) | 0.2072–0.2237 | kém hơn |
| peak_ms (384/512/768) | 0.1804 | scale shift |
| multi_ms | 0.1588 | kém |
| handcraft field-consistency | 0.0772 | rất yếu |
| oracle_ms (GT center, best scale) | 0.3078 | **không vượt trần single 0.314** |
| peak @ gsize 768 | 0.1111 | global model train ở 384 → scale shift |
| ocr_global @ gsize 768 | 0.1574 | vẫn xa 0.239 |

Kết luận:
- **Đa thang không nâng trần** (oracle_ms 0.308 ≈ oracle single 0.314) → 384 native là đúng.
- Đa đỉnh/đa thang/handcraft đều < single peak.
- Global stage ở 768 **sụp do scale shift** → phải train lại nếu muốn high-res global.
- → Best automatic vẫn là **single_peak 0.239**; trần subset 0.314.

### B34 — Multi-Signal Fusion selector (rank fusion) — GPT 60 ảnh — 2026-10-27

Ba tín hiệu: `c` = field classifier (ResNet18, crop 384), `g` = heatmap global (few-shot U-Net 384), `p` = patch-segmenter response (native 384). Rank fusion đều nhau.

| Selector | IoU | F1 | Dice | PixAUC | PixAP | center |
|---|---|---|---|---|---|---|
| c | 0.2444 | 0.3928 | 0.3435 | 0.8435 | 0.4854 | 25.9% |
| g | 0.1245 | 0.2214 | 0.2090 | 0.7086 | 0.2575 | 22.2% |
| p | 0.1448 | 0.2529 | 0.2548 | 0.7407 | 0.3886 | 14.8% |
| c+g | 0.2360 | 0.3819 | 0.3019 | 0.7529 | 0.3606 | 25.9% |
| c+p | 0.2315 | 0.3759 | 0.3316 | 0.7967 | 0.4747 | 22.2% |
| g+p | 0.2200 | 0.3606 | 0.3143 | 0.7561 | 0.4061 | 25.9% |
| **c+g+p (MSF)** | **0.2561** | **0.4078** | **0.3395** | **0.8075** | 0.4403 | 25.9% |
| coarse-to-fine peak | 0.2388 | 0.3855 | 0.2327 | 0.6985 | 0.2807 | — |
| oracle (subset) | 0.3141 | 0.4780 | 0.3814 | 0.9680 | 0.5717 | 100% |

Kết luận:
- **MSF (c+g+p) = 0.2561** — tốt nhất tự động, vượt single peak (+0.017) và classifier (+0.012).
- Ba tín hiệu bổ trợ nhau: c mạnh nhất đơn lẻ, nhưng fusion cả ba mới tốt nhất (từng cặp còn kém c).
- Đạt **81.5% trần subset** (0.256/0.314).
- Không dùng digit-bonus (bản c có digit-bonus 0.15 chỉ 0.2317 — digit prior hại).

### B35 — MSF trên FULL GPT test (613 forged) — 2026-10-27

| Selector | IoU | F1 | Dice | PixAUC | PixAP | center |
|---|---|---|---|---|---|---|
| c | 0.1934 | 0.3241 | 0.3024 | 0.7901 | 0.4032 | 23.5% |
| g | 0.1572 | 0.2717 | 0.2496 | 0.7444 | 0.3245 | 26.8% |
| p | 0.1576 | 0.2723 | 0.2967 | 0.7364 | 0.3766 | 7.4% |
| c+g | 0.1754 | 0.2984 | 0.2763 | 0.7696 | 0.3642 | 28.4% |
| c+p | 0.1950 | 0.3263 | 0.3404 | 0.8034 | 0.4453 | 16.7% |
| g+p | 0.1842 | 0.3111 | 0.3132 | 0.7622 | 0.3970 | 18.3% |
| **c+g+p (MSF)** | **0.1988** | 0.3317 | 0.3313 | 0.8033 | 0.4305 | 23.5% |
| few-shot full-image (B12) | 0.195 ± 0.037 | — | — | — | — | — |
| TL-HR oracle (B21) | **0.459** | 0.629 | — | 0.996 | 0.799 | 100% |

**Kết luận (quan trọng, thay đổi hướng paper):**
- Trên full test, **mọi selector chỉ ~0.19–0.20**; MSF 0.1988 **không khác biệt có ý nghĩa** so với few-shot 0.195 ± 0.037.
- Số 0.239–0.256 trên **60 ảnh subset là nhiễu** (~30 ảnh forged) — **không dùng làm headline**.
- Trần oracle vẫn 0.459 → **automatic localization là bức tường thật**, không phải vấn đề chọn method.
- → Paper định vị là **empirical + ceiling/bottleneck analysis**; MSF là ablation/chẩn đoán, không phải "method thắng".

### B36 — Audit: checkpoint & subset (tại sao bỏ claim method) — 2026-10-27

| Checkpoint | Full GPT IoU |
|---|---|
| `results/fewshot/k100.pth` (TL-HR dùng) | 0.1597 |
| `fewshot_multi/k100_s0` | 0.1582 |
| `fewshot_multi/k100_s1` | **0.2313** |
| `fewshot_multi/k100_s2` | 0.1948 |
| mean ± std (3 seed) | 0.195 ± 0.037 |

Subset 60 ảnh: **27 forged**, thành phần lệch — CORD 12 (44% vs 32% full), WildReceipt 6 (22% vs 44% full), SROIE 5, XFUND 4.
Baseline trên subset: 0.1358 (k100.pth) / 0.1538 (s1) — **thấp hơn full**, tức subset khó hơn cho baseline nhưng selector lại cao hơn → **nhiễu n=27**.

Kết luận audit:
- **Không có bug code** trong các lần chạy method (2 lỗi import/unpack đã sửa trước các lần chạy báo cáo).
- TL-HR dùng **seed yếu nhất** (0.1597) trong khi baseline paper là mean 3 seed 0.195 → so sánh lệch chuẩn.
- Trên full test, MSF 0.199 nằm **trong khoảng seed** (0.158–0.231) → **không vượt baseline có ý nghĩa**.
- → Quyết định bỏ claim method là **đúng dữ liệu**, không phải sai sót. Muốn so công bằng phải chạy TL-HR **3 seed** (chưa làm).

### B37 — Kiểm tra "fair seed" cho MSF (full test 613) — 2026-10-27

| Global backbone (full IoU) | g | cgp (MSF) | Baseline cùng seed | Δ |
|---|---|---|---|---|
| `fewshot/k100.pth` (0.1597) | 0.1572 | **0.1988** | 0.1597 | **+0.039** |
| `fewshot_multi/k100_s1` (0.2313) | 0.1650 | **0.1948** | 0.2313 | **−0.036** |

Kết luận:
- "Cùng seed thắng +26%" ở seed yếu **không tái lập** ở seed mạnh (thực tế âm).
- **MSF ≈ 0.195 bất kể backbone** → không phải thiên lệch seed; không phải lỗi so sánh.
- Đòn còn lại: **fine-tune patch_seg trên k=100 GPT** + **soft-gate**.

### B38 — Few-shot patch segmenter trên k=100 GPT (trần mới) — 2026-10-27

Fine-tune `results/patch_seg/patch_seg.pth` trên 100 ảnh GPT train (crop GT) + 100 neg cùng dòng + 100 authentic + replay 100 Gemini forged + 100 authentic. 10 epoch, lr 1e-4, 500 patch.

| Test (GT bbox) | IoU | F1 | Dice | PixAUC | PixAP | n |
|---|---|---|---|---|---|---|
| **GPT-Image-2 (full)** | **0.6661** | 0.7996 | 0.7320 | 0.9842 | 0.8550 | 613 |
| Gemini (in-domain) | 0.5315 | 0.6941 | 0.4064 | 0.9441 | 0.8654 | 727 |

So với trần cũ (Gemini-only, B21): GPT **0.459 → 0.666 (+45%)**; Gemini 0.557 → 0.5315 (quên nhẹ, replay giữ).

→ **Trần TL-HR giờ là 0.666**. Pipeline tự động còn nhiều headroom; cần chạy soft-gate/selection với patch mới.

### B39 — Soft-gate + patch adapted (full GPT 613) — 2026-10-27

Global = `fewshot/k100.pth`; patch = `patch_seg_gpt` (fine-tune k=100 GPT, trần 0.666). Không OCR.

| Config | IoU | F1 | Dice |
|---|---|---|---|
| **K1, α=0.0 (peak + adapted patch, no gate)** | **0.5808** | 0.7348 | 0.4061 |
| K1, α=0.0, global seed 1 (0.2313) | 0.5680 | 0.7245 | 0.4254 |
| K2, α=0.0 | 0.5251 | 0.6886 | 0.4480 |
| K3, α=0.0 | 0.5091 | 0.6747 | 0.4572 |
| K1, α=0.3 | 0.3690 | 0.5391 | 0.2165 |
| K1, α=0.5 | 0.3456 | 0.5136 | 0.2033 |
| K1, α=0.7 | 0.3279 | 0.4939 | 0.1962 |
| K3, α=0.3, thr 0.4 (gate tốt nhất) | 0.3115 | 0.4751 | 0.2383 |
| global-only | 0.1357 | 0.2390 | 0.1776 |

Kết luận:
- **K1 + adapted patch = 0.5808** = **87% trần 0.666**, **2.98× baseline 0.195**, cùng 100 nhãn, không OCR.
- **Pixel gating bằng P_global làm GIẢM** (α↑ → IoU↓): global heatmap mờ/lệch phá mask sắc nét. Global chỉ dùng để **chọn đỉnh**.
- Đa đỉnh K2/K3 < K1 → top-1 peak là đủ.
- → Có **method thật**: TL-HR-PA (coarse peak + few-shot adapted patch).

### B41 — TL-HR-PA: 3 seed (hoàn tất) — 2026-10-27

| seed | trần patch (GT box) | TL-HR tự động (K1 α=0) |
|---|---|---|
| 0 | 0.6661 | 0.5808 |
| 1 | 0.6839 | 0.5653 |
| 2 | 0.6804 | 0.5391 |

Trần: **0.6768 ± 0.009**; auto: **0.5617 ± 0.021** (58% của trần, 2.9× baseline 0.195).
Patch s1 eval thêm Ideogram (n=85, P3 subset): 0.3687.

### B42 — Tổng quát hoá cross-generator: Ideogram (P1, 422 forged) — 2026-10-28

Coarse = `seg_p3` (Gemini+GPT, size 448), patch = `patch_seg_gpt` (s0), gsize 448.

| Config | IoU | F1 | Dice |
|---|---|---|---|
| global-only (cùng split) | 0.1144 | 0.2054 | 0.1325 |
| **K1, α=0 (TL-HR)** | **0.4490** | 0.6197 | 0.2131 |
| K1, α=0.3 / 0.5 / 0.7 | 0.3957 / 0.3768 / 0.3516 | | |
| K2 / K3, α=0 | 0.3295 / 0.3166 | | |

Kết luận: **3.9× baseline** trên generator chưa thấy; gating và đa đỉnh lại làm hại → kết luận về K1 α=0 tổng quát, không phải may mắn trên GPT.
Lưu ý: baseline 0.0916 trước đó là trên P3 subset 85 ảnh (khó hơn) — không trộn split khi báo cáo.

### B43 — Label efficiency của patch adaptation (trần GT box, full test GPT) — 2026-10-28

| k (nhãn GPT) | trần IoU | Gemini | so với full-image k tương ứng |
|---|---|---|---|
| 25 | 0.5839 | 0.5235 | 0.105 → **5.6×** |
| 50 | 0.5829 | 0.5270 | (~0.126) |
| 100 | 0.6768 ± 0.009 | 0.529 | 0.195 (mean 3 seed) |

Kết luận: **25 nhãn đã đạt 86% trần k=100**; TL-HR hưởng lợi từ nhãn mạnh hơn nhiều so với full-image few-shot. Đường cong phẳng 25→50 rồi nhảy ở 100 → 100 là điểm hợp lý cho headline.

### B44 — SỬA SAI METRIC + phương pháp mới "dense patch search" — 2026-10-28

**Lỗi metric**: `eval_softgate.py` tính IoU trong cửa sổ crop (`mask[y0:y1, x0:x1]`), không phải toàn ảnh
→ các số B40/B41/B42 (0.5808/0.5653/0.5391/0.4490) **bị thổi phồng, không dùng được**.

**Chẩn đoán (150 ảnh, metric full-image)**:
- peak coarse lệch tâm GT bbox: median **152px** (45% > 192px); mask được cửa sổ 384 phủ: 53% (tại peak) vs 97.8% (tại GT).
- Trần (GT box) 0.678; peak 0.355; zoom 768→384: 0.29 (**tệ hơn**, do scale shift); tile 2×2: 0.29 (tệ); grid3 0.429; grid5 0.465; grid7 0.487; centroid 0.368; TTA 0.350.
- Kết luận: nút thắt là **định vị coarse**, không phải phân đoạn.

**Sliding window (patch adapted, stride 96, chọn theo top-1% confidence)**:
| Bộ | windows/ảnh | top1 pooled | oracle window | trần |
|---|---|---|---|---|
| 150 ảnh | 218 | 0.5826 | 0.6546 | 0.678 |
| **613 ảnh (full, s0)** | 197 | **0.5546** | 0.6533 | 0.6661 |

- `meanpred` 0.5393, `max` 0.5316, `mass` 0.4720 → chọn top-1% confidence là tốt nhất.
- Pruning bằng coarse (top-K) **kém hơn** full search (K=40 → 0.544; K=5 → 0.431) → coarse không dùng làm prior.
- **Method**: few-shot patch adaptation + dense high-resolution search. 2.84× baseline, 83% trần, không OCR.

### B45 — Dense patch search: 3 seed + Ideogram (full test) — 2026-10-29

| seed | IoU pooled | IoU mean | oracle-window (mean) |
|---|---|---|---|
| 0 | 0.5546 | 0.5706 | 0.6533 |
| 1 | 0.5492 | 0.5477 | 0.6456 |
| 2 | 0.5320 | 0.5519 | 0.6653 |

**0.5452 ± 0.0115** (2.80× baseline 0.195; 82% trần 0.666). Score: top1% confidence.
Trên 150 ảnh: stride 96 = 0.5826, stride 128 = 0.5649; top1 > meanpred > max > mass.

**Ideogram (422 forged, patch s0, stride 96, 655 windows/ảnh)**:
| | pooled | mean |
|---|---|---|
| top1 | 0.2844 | 0.3808 |
| meanpred | 0.2761 | 0.3882 |
| max | 0.2821 | 0.3819 |
| oracle-window | — | 0.5096 |
| global-only (B42, gsize 448) | 0.1144 | — |

Kết luận: cross-generator **~2.5×** baseline nhưng gap lớn; detector chuyển được (oracle 0.51) nhưng **confidence score không hiệu chỉnh** sang generator mới → hướng cải thiện rõ ràng.

Lưu ý kỹ thuật: bản cũ cấp phát mảng full-size mỗi cửa sổ → ảnh 34MP làm chậm 7h. Đã sửa: IoU tính trên crop `inter/(pred_area+mask_area−inter)`.

### B46 — Image-level metrics (AP, Recall@1%FPR) + sửa trình bày — 2026-10-29

Score ảnh = trung bình top-1% probability (cùng hàm với TL-HRS), split P1 có 613 forged + 613 authentic.

| k | Image AUC | Image AP | Recall@1%FPR |
|---|---|---|---|
| 0 (zero-shot) | 0.526 | 0.547 | 0.015 |
| 5 | 0.529 ± 0.012 | 0.522 ± 0.010 | 0.014 ± 0.001 |
| 25 | 0.515 ± 0.024 | 0.518 ± 0.021 | 0.019 ± 0.006 |
| 100 | 0.565 ± 0.024 | 0.582 ± 0.016 | 0.042 ± 0.012 |

Zero-shot in-domain/Ideogram: Gemini AUC 0.913/AP 0.925/R@1% 0.553; Ideogram 0.559/0.559/0.038.
→ Image-level vẫn yếu dưới generator shift; TL-HRS là stage định vị, cần triage ảnh (ngưỡng 1% FPR) trước.
Script: `scripts/eval_image_metrics.py`; JSON: `results/image_metrics/`.

Sửa trình bày paper: hình chuyển dọc (Fig 1: 2 hàng, Fig 3: 3 hàng), công thức → HTML, mũi tên → `->`, điền ô trống Bảng 3.1/6.2, thêm mục **4.5 TL-HRS** (Method), đoạn xử lý ảnh authentic, xuất `paper/main_2.html/pdf` (A4, 18 trang).

## Cảnh báo cần kiểm tra

- Model có thể "ăn gian" nhờ khác biệt **độ phân giải / kiểu nén** giữa authentic và forged → cần augmentation (JPEG, resize, blur) để kiểm chứng.
- AUC < 0.5 nghĩa là dự đoán **ngược** (không phải chỉ kém).
