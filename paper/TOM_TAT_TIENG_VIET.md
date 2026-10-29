# TÓM TẮT TIẾNG VIỆT — BÁO CÁO NHÓM

## Tên đề tài

**Thích ứng tiết kiệm nhãn cho hệ thống phát hiện giấy tờ tài chính bị AI sửa, dưới dịch chuyển bộ sinh, bằng rehearsal và tìm kiếm dày độ phân giải cao.**
*(Label-Efficient Adaptation of AI-Forged Document Detectors under Generator Shift via Rehearsal and Dense High-Resolution Search)*

---

## 1. Vấn đề

- Kẻ gian dùng AI inpainting (Gemini, Ideogram, GPT-Image-2) sửa một con số trên hóa đơn/chứng từ trong vài giây.
- Detector huấn luyện trên forgery truyền thống (Photoshop) gần như mù với loại này.
- Detector huấn luyện trên **một** bộ sinh AI cũng **sụp** khi gặp bộ sinh mới (*generator shift*).
- Ngân hàng **không thể** thu thập hàng nghìn mẫu nhãn cho mỗi công cụ AI mới.

## 2. Câu hỏi nghiên cứu

> Cần **bao nhiêu mẫu nhãn** từ bộ sinh mới để khôi phục khả năng định vị, và làm sao **không quên** bộ sinh cũ?

## 3. Dữ liệu & giao thức

- Bộ **AIForge-Doc**: ảnh hóa đơn/biểu mẫu bị AI sửa, có mask pixel.
- 3 bộ sinh: Gemini, Ideogram, GPT-Image-2.
- Giao thức **leave-one-generator-out**:
  - Train Gemini → test Gemini (in-domain), Ideogram (unseen), GPT-Image-2 (unseen).

## 4. Phương pháp

- **Nền:** U-Net + ResNet18, train trên Gemini.
- **Few-shot adaptation:** fine-tune với \(k\) ảnh forged + \(k\) ảnh authentic từ bộ sinh mới.
- **Rehearsal:** thêm \(r=100\) mẫu từ bộ sinh cũ khi fine-tune để chống quên.
- **Calibration:** thử temperature scaling, threshold toàn cục, threshold thích ứng từng ảnh (kết quả: không hiệu quả).

## 5. Kết quả chính

### 5.1 Zero-shot sụp

| Test | IoU | Image AUC |
|---|---|---|
| Gemini (in-domain) | 0.669 | 0.946 |
| Ideogram (unseen) | 0.098 | 0.546 |
| GPT-Image-2 (unseen) | 0.052 | 0.520 |

### 5.2 Few-shot phục hồi (3 seed)

| k | GPT IoU | Gemini IoU (quên) |
|---|---|---|
| 0 | 0.052 | 0.669 |
| 5 | 0.068 ± 0.015 | 0.624 ± 0.017 |
| 25 | 0.105 ± 0.031 | 0.577 ± 0.031 |
| 100 | **0.195 ± 0.037** | **0.494 ± 0.054** |

→ IoU tăng **3,7 lần** với 100 nhãn.

### 5.3 Rehearsal chống quên (4 seed)

| Phương pháp | GPT IoU | Gemini IoU |
|---|---|---|
| Standard fine-tune | 0.195 ± 0.030 | 0.494 ± 0.044 |
| **Replay 100** | **0.162 ± 0.021** | **0.672 ± 0.006** |

→ Replay đưa Gemini về gần mức zero-shot (0.669), chỉ giảm GPT nhẹ (~17%).

### 5.4 Fine-tune vs train from scratch (k=100)

| | GPT IoU | Gemini IoU |
|---|---|---|
| Fine-tune từ base | **0.160** | **0.552** |
| Train from scratch | 0.085 | 0.075 |

→ Tiền huấn luyện trên bộ sinh cũ quan trọng: fine-tune tốt hơn ~1,9 lần.

### 5.5 Phương pháp TL-HRS (Text-Localized High-Resolution Search)

**Adapt patch** segmenter trên đúng k=100 nhãn GPT (+ replay 100 Gemini) → trần nâng 0.459 → 0.677. **Định vị**: thay vì lấy đỉnh coarse (lệch **152 px** median, crop chỉ phủ **53%** mask), quét dày cửa sổ **384 native, stride 96** (~197 cửa sổ/ảnh) và chọn cửa sổ có **mean top-1% probability** cao nhất. Không OCR, không gating, không selector học.

| Thiết lập (full GPT 613 ảnh) | IoU |
|---|---|
| Few-shot toàn ảnh (k=100) | 0.195 ± 0.037 |
| Patch zero-shot (GT box) | 0.459 |
| **Patch adapted (GT box) — trần** | **0.677 ± 0.009** |
| Đỉnh coarse + patch adapted | 0.321 |
| + centroid / grid 3×3 | 0.358 / 0.400 |
| **TL-HRS (quét dày, 3 seed)** | **0.545 ± 0.012** |
| — oracle window | 0.653 |
| Ideogram (bộ sinh chưa thấy) | 0.284 (oracle 0.510) |
| Ideogram full-image | 0.114 |

→ **0.545 = 82% trần, 2.8× baseline**, cùng 100 nhãn; quét dày hơn hẳn đỉnh coarse.
Ba phát hiện: **(1) adapt patch chuyển được, trần thật** (0.459 → 0.677); **(2) nút thắt là *nhìn vào đâu***, không phải phân đoạn (đỉnh coarse 0.321 → quét dày 0.545; oracle window 0.653); **(3) "thông minh" trong tìm kiếm không đáng** — zoom, tile, centroid, pruning bằng coarse đều kém hơn quét dày; coarse map không đủ tin cậy làm prior.

## 6. Phát hiện quan trọng

- **Pixel AUC vẫn cao (0.83–0.91)** dù IoU thấp → model vẫn **xếp hạng pixel tốt**, nhưng **mask định vị chưa chính xác**.
- **Calibration không phải nút thắt**: temperature không đổi IoU; threshold toàn cục/thích ứng không cải thiện ổn định; threshold thích ứng còn **hại in-domain**.
- → Nút thắt thật là **độ chính xác định vị (localization precision)**, không phải ngưỡng quyết định.
- **TL-HRS**: adapt patch với cùng 100 nhãn nâng trần **0.459 → 0.677**; quét dày native (stride 96) đạt **0.545 ± 0.012** (~2.8× baseline, 82% trần). Đỉnh coarse chỉ 0.321 và lệch 152 px; zoom/tile/centroid/pruning đều kém hơn quét dày; trên Ideogram đạt 0.284 (oracle 0.510) → confidence chưa hiệu chỉnh qua bộ sinh.

## 7. Đóng góp

1. Nghiên cứu có kiểm soát về **thích ứng tiết kiệm nhãn** dưới generator shift.
2. Đường cong few-shot (3 seed): **0.052 → 0.195**.
3. **Rehearsal** giữ bộ sinh cũ gần như không quên (0.672).
4. Bộ **kết quả âm trung thực** (calibration, adaptive threshold, sample selection, freeze encoder, PatchCore, FR-Net).
5. **Phương pháp TL-HRS**: adapt patch segmenter với cùng 100 nhãn (trần **0.459 → 0.677**) + quét dày cửa sổ native, đạt **0.545 ± 0.012** (~2.8× baseline, 82% trần).
6. **Ablation định vị**: đỉnh coarse lệch 152 px (phủ 53% mask); zoom, tile, centroid, grid, pruning bằng coarse, OCR/selector, MSF đều kém hơn quét dày; coarse map không đủ tin cậy để làm prior.
7. Bằng chứng định tính (score maps) + tổng quát hoá Ideogram (0.284 vs 0.114 full-image).

## 8. Hạn chế

- Dataset nhỏ, chủ yếu hóa đơn/biểu mẫu.
- Chỉ 3 bộ sinh.
- Chỉ sửa trường số (không gồm giấy tờ sinh từ đầu, chữ ký).
- Một kiến trúc (ResNet18 U-Net).
- IoU sau thích ứng vẫn thấp so với in-domain; TL-HRS đạt 0.545 ± 0.012 = 82% trần (0.677); chi phí suy luận ~197 cửa sổ/ảnh (~20 s/ảnh). Gap trên bộ sinh mới do confidence chưa hiệu chỉnh (Ideogram 0.284 vs oracle 0.510).

## 9. Hướng phát triển

- **Federated adaptation:** ngân hàng A gặp bộ sinh mới, thu 100 mẫu, chia sẻ **model update** (không chia sẻ dữ liệu thô) cho ngân hàng B, C.
- **Context-contrast localization:** chọn vùng bị sửa bằng cách so sánh mỗi trường với các trường lân cận trong cùng ảnh (thay vì phân loại cục bộ).
- Hiệu chỉnh confidence của patch qua các bộ sinh để đóng gap 0.284 → 0.510 trên Ideogram; scoring cửa sổ có ngữ cảnh thay vì độc lập.
- Thích ứng liên tục qua nhiều bộ sinh.
- Benchmark rộng hơn (loại giấy, ngôn ngữ).

## 10. Kết luận

> Một ít nhãn từ bộ sinh mới + rehearsal là đủ để **khôi phục định vị** và **không quên** bộ sinh cũ. Nút thắt còn lại là **độ chính xác định vị**, không phải calibration. **TL-HRS** — adapt patch segmenter với cùng 100 nhãn (trần **0.459 → 0.677**) rồi quét dày cửa sổ native — đạt **0.545 ± 0.012** (~2.8× baseline, 82% trần). Đỉnh coarse chỉ 0.321 vì lệch 152 px, và các biến thể "thông minh" (zoom, tile, grid, pruning) đều kém hơn. Trên Ideogram: 0.284 vs 0.114 baseline — detector tìm được vùng (oracle 0.510), nhưng confidence chưa hiệu chỉnh qua bộ sinh.

---

*Bản thảo tiếng Anh đầy đủ: `paper/main.md`. Bản thảo 1 (lưu cũ): `paper/ban_thao_1.md`.*
