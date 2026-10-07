# Dữ liệu

**EuroSAT (RGB)** – 27.000 ảnh Sentinel-2, 64×64 pixel, 10 lớp land use / land cover.

- Nguồn: Helber et al. (2019), Zenodo record 7711810 – https://zenodo.org/records/7711810 (giấy phép MIT)
- File dùng: `EuroSAT_RGB.zip` (~95 MB)

## Đặt dữ liệu đúng chỗ
Sau khi giải nén, cấu trúc phải là:
```
data/
├── README.md
├── splits.csv            ← tạo bởi notebook 01 (được đưa lên git)
└── EuroSAT_RGB/          ← ảnh gốc (KHÔNG đưa lên git)
    ├── AnnualCrop/AnnualCrop_1.jpg ...
    ├── Forest/
    └── ... (10 thư mục lớp)
```

**Trên Colab:** cell **SETUP** ở đầu mỗi notebook tự tải và giải nén – không cần làm gì.

**Trên máy cá nhân:**
1. Tải https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1 (nên dùng Chrome; file đủ là ~95 MB).
2. Giải nén → được thư mục `EuroSAT_RGB`.
3. Kéo thư mục `EuroSAT_RGB` vào trong thư mục `data/` này.

Hoặc bằng Terminal (đứng ở thư mục gốc của repo):
```bash
curl -L -o data/EuroSAT_RGB.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
unzip -q data/EuroSAT_RGB.zip -d data/
```

## File trong thư mục này
- `splits.csv` – cách chia train/val/test (70/15/15, stratified, seed 42) **dùng chung cho cả 3 model**. Cột: `filepath`, `label`, `label_id`, `split`.
- Ảnh gốc và file zip đã bị chặn bởi `.gitignore` nên không bị đưa lên git.
