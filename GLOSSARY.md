# GLOSSARY — Thuật ngữ cần biết

## 1. Nền tảng Machine Learning

| Thuật ngữ | Nghĩa |
|---|---|
| Machine Learning | Máy tự học quy luật từ dữ liệu |
| Supervised learning | Học có nhãn |
| Unsupervised learning | Học không nhãn |
| Self-supervised learning | Tự tạo "nhãn giả" từ dữ liệu |
| Classification | Phân loại (thật / giả) |
| Segmentation | Phân vùng từng pixel |
| Model / Training / Inference | Mô hình / huấn luyện / dự đoán |
| Overfitting / Generalization | Học vẹt / làm tốt trên dữ liệu mới |
| Baseline | Mô hình mốc so sánh |

## 2. Computer Vision cơ bản

| Thuật ngữ | Nghĩa |
|---|---|
| Image / RGB / Pixel / Resolution | Ảnh / kênh màu / điểm ảnh / độ phân giải |
| CNN | Mạng nơ-ron tích chập |
| Feature map | Bản đồ đặc trưng |
| Patch | Mảnh ảnh nhỏ |
| Backbone / Encoder | Phần trích xuất đặc trưng |
| Pretraining / Fine-tuning | Học trước / học lại với ít nhãn |
| Augmentation | Biến đổi ảnh để tăng dữ liệu |

## 3. Giả mạo & sinh ảnh

| Thuật ngữ | Nghĩa |
|---|---|
| Forgery / tampering | Giả mạo / chỉnh sửa ảnh |
| AI-generated / synthetic image | Ảnh do AI tạo / tổng hợp |
| Diffusion model | Mô hình sinh ảnh (Stable Diffusion...) |
| Inpainting | Vẽ lại/lấp một vùng ảnh |
| LLM | Mô hình ngôn ngữ lớn |
| Deepfake | Ảnh/video giả do AI |

## 4. Pháp y ảnh (Image Forensics)

| Thuật ngữ | Nghĩa |
|---|---|
| Image forgery detection | Phát hiện ảnh giả mạo |
| Localization | Định vị chỗ bị sửa |
| Mask | Nhãn pixel vùng bị sửa |
| Image-level / pixel-level | Phân loại cả ảnh / từng pixel |
| Copy-move / splicing | Sao chép-dán / ghép vùng |
| JPEG artifacts / noise residual | Dấu vết nén / nhiễu dư |
| ELA | Error Level Analysis |
| Frequency domain | Miền tần số |

## 5. Học tự giám sát

| Thuật ngữ | Nghĩa |
|---|---|
| Self-supervised learning | Học từ dữ liệu chưa nhãn |
| Pretext task | Nhiệm vụ tự tạo |
| Contrastive learning | Học bằng so sánh giống/khác |
| SimCLR / MoCo / BYOL / DINO | Các phương pháp phổ biến |
| MAE | Masked Autoencoder |
| Positive / negative pair | Cặp giống / khác |

## 6. Định vị & giải thích

| Thuật ngữ | Nghĩa |
|---|---|
| Localization / segmentation mask | Xác định vùng bị sửa |
| IoU / mIoU | Độ trùng khớp vùng |
| Explainable AI (XAI) | AI kèm lý do |
| Saliency map / heatmap | Bản đồ nhiệt |
| Grad-CAM | Kỹ thuật tạo bản đồ nhiệt |
| Attention map | Bản đồ chú ý |

## 7. Đánh giá

| Thuật ngữ | Nghĩa |
|---|---|
| TP / FP / TN / FN | Ma trận nhầm lẫn |
| Accuracy / Precision / Recall / F1 | Các chỉ số phân loại |
| AUC-ROC / AUC-PR | Chất lượng xếp hạng |
| Pixel-AUC / Pixel F1 | Đánh giá ở mức pixel |
| IoU / mIoU | Độ trùng khớp vùng |
| Calibration / ECE | Độ tin cậy của xác suất |
| Zero-shot | Dự đoán không cần huấn luyện |

## 8. Domain / tổng quát hoá

| Thuật ngữ | Nghĩa |
|---|---|
| Domain | Miền dữ liệu (giấy, bộ sinh AI, thiết bị) |
| Source / target domain | Miền nguồn / đích |
| Domain shift | Lệch phân phối |
| Generalization | Làm tốt trên miền mới |
| Cross-generator | Test bộ sinh AI khác bộ train |

## 9. Dữ liệu

| Dataset | Dùng cho |
|---|---|
| SROIE / CORD / WildReceipt | Ảnh hóa đơn, biên lai thật |
| XFUND | Biểu mẫu đa ngôn ngữ |
| AIForge-Doc (v1+v2) | Ảnh bị AI sửa + mask |
| DocTamper | Giả mạo truyền thống (so sánh) |

## 10. Công cụ & baseline

| Thuật ngữ | Vai trò |
|---|---|
| PyTorch / torchvision / timm | Framework + model ảnh |
| diffusers + Stable Diffusion | Sinh ảnh AI |
| OpenCV | Xử lý ảnh |
| segmentation-models-pytorch | Model segmentation |
| Grad-CAM / Captum | Giải thích |
| scikit-learn | Metrics, baseline |
| Git / GitHub | Quản lý code |
| TruFor / DocTamper | Baseline pháp y ảnh |
| GPT-4o / VLM | Baseline đa phương thức |
| Foundation model | Mô hình nền tảng |
