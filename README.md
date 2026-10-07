# Land Use and Land Cover Classification from Satellite Images using CNNs and Transfer Learning

Project môn Deep Learning – Nhóm 7, DSEB65A, Đại học Kinh tế Quốc dân.

Phân loại ảnh vệ tinh Sentinel-2 (EuroSAT RGB, 27.000 ảnh 64×64, 10 lớp) bằng 3 model xây bằng TensorFlow/Keras:
1. **Simple CNN** – Conv, Pooling, Fully Connected
2. **Complex CNN** – 4 CNN block tự thiết kế ([Conv–BatchNorm–ReLU] × 2 → MaxPool → Dropout), GlobalAveragePooling, Fully Connected
3. **ResNet50V2** – transfer learning từ ImageNet: đóng băng backbone, sau đó fine-tune stage cuối (conv5)

## Kết quả trên tập test
_(điền từ `results/test_metrics.csv` sau khi chạy notebook 05)_

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Số tham số |
|---|---|---|---|---|---|
| Simple CNN | | | | | |
| Complex CNN | | | | | |
| ResNet50V2 | | | | | |

## Cấu trúc thư mục
```
├── data/            splits.csv (ảnh gốc EuroSAT_RGB/ không đưa lên git – xem data/README.md)
├── notebooks/       01 → 05, chạy theo thứ tự
├── results/         lịch sử train, bảng kết quả, classification report
├── figures/         hình dùng trong báo cáo
├── models/          link Drive tới các model đã train
└── report/          báo cáo cuối cùng
```

| Notebook | Nội dung | Requirement |
|---|---|---|
| `01_data_preprocessing.ipynb` | Kiểm tra dữ liệu, EDA, chia train/val/test 70/15/15, chuẩn hóa, augmentation | Req 1 |
| `02_simple_cnn.ipynb` | Simple CNN | Req 2 |
| `03_complex_cnn.ipynb` | Complex CNN tự thiết kế + 3 biến thể | Req 3 |
| `04_transfer_learning.ipynb` | ResNet50V2: pha 1 (đóng băng) + pha 2 (fine-tune conv5) | Req 4 |
| `05_evaluation.ipynb` | Đánh giá 3 model trên tập test, so sánh, phân tích lỗi | Req 5 |

## Cách chạy lại
**Google Colab (khuyên dùng):**
1. Mở notebook trên Colab, chọn `Runtime → Change runtime type → T4 GPU`.
2. Chạy cell 0: tự kết nối Google Drive (thư mục `MyDrive/eurosat_project`) và tự tải dataset EuroSAT.
3. Chạy lần lượt `01` → `05`. Kết quả lưu vào `MyDrive/eurosat_project/` với cấu trúc giống repo.

**Máy cá nhân:** đặt ảnh vào `data/EuroSAT_RGB/` (xem `data/README.md`), cài thư viện `pip install -r requirements.txt`, rồi mở notebook từ thư mục `notebooks/`.

## Thành viên
| Họ tên | Phụ trách |
|---|---|
| | Dữ liệu & preprocessing, Simple CNN |
| | Complex CNN, GitHub repo |
| | Transfer learning, đánh giá & so sánh |

## Tài liệu tham khảo
- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE JSTARS*, 12(7), 2217–2226.
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). Identity Mappings in Deep Residual Networks (ResNetV2). *ECCV*.
- Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). *ICLR*.
