import os
import hashlib
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import (PROJECT_DIR, DATA_DIR, SPLITS_CSV, RESULTS_DIR, FIGURES_DIR, DATASET_URL,
                    CLASS_NAMES, NUM_CLASSES, IMG_SIZE, RANDOM_STATE)
from data_utils import make_augmentation

tf.keras.utils.set_random_seed(RANDOM_STATE)

# ---------- download dataset if missing ----------
if not os.path.exists(DATA_DIR):
    zip_path = os.path.join(PROJECT_DIR, "data", "EuroSAT_RGB.zip")
    print("Downloading EuroSAT RGB...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(os.path.join(PROJECT_DIR, "data"))

# ---------- check images: corrupt, wrong size, not RGB, duplicate ----------
rows = []
for cls in CLASS_NAMES:
    for fname in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        path = os.path.join(DATA_DIR, cls, fname)
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
                mode = im.mode
            readable = True
        except Exception:
            width = height = mode = None
            readable = False
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        rows.append({"filepath": f"{cls}/{fname}", "label": cls, "readable": readable,
                     "width": width, "height": height, "mode": mode, "md5": md5})

df = pd.DataFrame(rows)
df["bad_size"] = df["readable"] & ((df["width"] != IMG_SIZE) | (df["height"] != IMG_SIZE))
df["bad_mode"] = df["readable"] & (df["mode"] != "RGB")
df["duplicate"] = df.duplicated("md5", keep="first")

check = pd.DataFrame({
    "Total images": [len(df)],
    "Corrupt": [int((~df["readable"]).sum())],
    "Wrong size": [int(df["bad_size"].sum())],
    "Not RGB": [int(df["bad_mode"].sum())],
    "Duplicates": [int(df["duplicate"].sum())],
})
check.to_csv(os.path.join(RESULTS_DIR, "data_check.csv"), index=False)
print(check.T.to_string(header=False))

# ---------- EDA ----------
counts = df[df["readable"]]["label"].value_counts().reindex(CLASS_NAMES)
print(counts.to_string())
print("Largest / smallest class ratio:", round(counts.max() / counts.min(), 2))

plt.figure(figsize=(10, 4))
counts.plot(kind="bar", color="steelblue")
plt.ylabel("Number of images")
plt.title("Images per class – EuroSAT RGB")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(10, 20))
for r, cls in enumerate(CLASS_NAMES):
    samples = df[(df["label"] == cls) & df["readable"]]["filepath"].sample(5, random_state=RANDOM_STATE)
    for c, fp in enumerate(samples):
        axes[r, c].imshow(Image.open(os.path.join(DATA_DIR, fp)))
        axes[r, c].axis("off")
    axes[r, 0].set_title(cls, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "sample_grid.png"), dpi=120)
plt.close()

# ---------- split 70 / 15 / 15 (stratified) ----------
clean = df[df["readable"] & ~df["bad_size"] & ~df["bad_mode"] & ~df["duplicate"]].copy()
clean["label_id"] = clean["label"].map({name: i for i, name in enumerate(CLASS_NAMES)})

train_df, temp_df = train_test_split(clean, test_size=0.30, stratify=clean["label"],
                                     random_state=RANDOM_STATE)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"],
                                   random_state=RANDOM_STATE)

train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"
splits = pd.concat([train_df, val_df, test_df])[["filepath", "label", "label_id", "split"]]
splits.to_csv(SPLITS_CSV, index=False)

summary = pd.crosstab(splits["label"], splits["split"])[["train", "val", "test"]]
summary.loc["TOTAL"] = summary.sum()
summary.to_csv(os.path.join(RESULTS_DIR, "split_summary.csv"))
print(summary.to_string())

# ---------- normalization ----------
sample_paths = train_df["filepath"].values[:500]
X_sample = np.stack([np.array(Image.open(os.path.join(DATA_DIR, fp))) for fp in sample_paths])
X_sample_norm = X_sample.astype("float32") / 255.0

print("Shape:", X_sample.shape)
print(f"Before: min = {X_sample.min()}, max = {X_sample.max()}")
print(f"After : min = {X_sample_norm.min():.2f}, max = {X_sample_norm.max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(X_sample.ravel(), bins=50, color="gray")
axes[0].set_title("Pixels before normalization [0, 255]")
axes[1].hist(X_sample_norm.ravel(), bins=50, color="steelblue")
axes[1].set_title("Pixels after normalization [0, 1]")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "normalization.png"), dpi=150)
plt.close()

# ---------- augmentation preview ----------
data_augmentation = make_augmentation()
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
for r in range(4):
    img = X_sample_norm[r * 50]
    axes[r, 0].imshow(img)
    axes[r, 0].set_title("Original", fontsize=9)
    for c in range(1, 6):
        aug = data_augmentation(img[None, ...], training=True)[0]
        axes[r, c].imshow(np.clip(np.asarray(aug), 0, 1))
        axes[r, c].set_title(f"Augmented {c}", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "augmented_samples.png"), dpi=120)
plt.close()
