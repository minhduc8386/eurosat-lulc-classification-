# Data

**EuroSAT (RGB)** – 27,000 Sentinel-2 images, 64×64 pixels, 10 land use / land cover classes.

- Source: Helber et al. (2019) – https://github.com/phelber/EuroSAT
- Download: https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1 (~95 MB, MIT licence)

## Folder layout
```
data/
├── README.md
├── splits.csv            ← created by src/01_data_preprocessing.py (on GitHub)
└── EuroSAT_RGB/          ← raw images (NOT on GitHub)
    ├── AnnualCrop/AnnualCrop_1.jpg ...
    ├── Forest/
    └── ... (10 class folders)
```

## Getting the images
`python src/01_data_preprocessing.py` downloads and unzips the dataset automatically if `data/EuroSAT_RGB/` does not exist.

To do it by hand: download the zip above, unzip it, and move the `EuroSAT_RGB` folder into this `data/` folder.

## Files in this folder
- `splits.csv` – the train/val/test split (70/15/15, stratified, seed 42) **shared by all 3 models**. Columns: `filepath`, `label`, `label_id`, `split`.
- The raw images and the zip file are blocked by `.gitignore`, so they are never pushed to GitHub.
