import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential
from keras.layers import RandomFlip, RandomBrightness
from keras.utils import to_categorical

from config import DATA_DIR, SPLITS_CSV, NUM_CLASSES


def check_data_exists():
    assert os.path.exists(DATA_DIR), f"Images not found at {DATA_DIR}"
    assert os.path.exists(SPLITS_CSV), "splits.csv not found, run 01_data_preprocessing.py first"


def load_split(split_name):
    splits = pd.read_csv(SPLITS_CSV)
    df = splits[splits["split"] == split_name]
    X = np.stack([np.array(Image.open(os.path.join(DATA_DIR, fp))) for fp in df["filepath"]])
    X = X.astype("float32") / 255.0   # normalize to [0, 1]
    y = df["label_id"].values
    return X, y


def load_train_val():
    check_data_exists()
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    Y_train = to_categorical(y_train, NUM_CLASSES)
    Y_val = to_categorical(y_val, NUM_CLASSES)
    print("X_train:", X_train.shape, "| Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape, "| Y_val  :", Y_val.shape)
    return X_train, Y_train, X_val, Y_val


def make_augmentation():
    # only active during training
    return Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomBrightness(factor=0.1, value_range=(0, 1)),
    ], name="data_augmentation")
