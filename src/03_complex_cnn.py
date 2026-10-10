import os
import json
import shutil
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                          BatchNormalization, Activation, GlobalAveragePooling2D)

from config import IMG_SIZE, NUM_CLASSES, RANDOM_STATE, EPOCHS, MODELS_DIR, RESULTS_DIR
from data_utils import load_train_val, make_augmentation
from train_utils import train_and_save, plot_history

tf.keras.utils.set_random_seed(RANDOM_STATE)


def add_cnn_block(model, filters, use_batchnorm=True, dropout=0.25):
    # [Conv 3x3 -> BN -> ReLU] x 2 -> MaxPool -> Dropout
    for _ in range(2):
        model.add(Conv2D(filters, (3, 3), padding="same"))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))


def build_complex_cnn(name="complex_cnn", use_batchnorm=True, use_gap=True):
    model = Sequential(name=name)
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(make_augmentation())
    for filters in [32, 64, 128, 256]:      # 64 -> 32 -> 16 -> 8 -> 4
        add_cnn_block(model, filters, use_batchnorm)
    if use_gap:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    return model


build_complex_cnn().summary()

data = load_train_val()

variants = {
    "complex_cnn_v1": dict(use_batchnorm=True, use_gap=True),
    "complex_cnn_no_bn": dict(use_batchnorm=False, use_gap=True),
    "complex_cnn_flatten": dict(use_batchnorm=True, use_gap=False),
}
for name, settings in variants.items():
    print(f"\n===== {name} =====")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_complex_cnn(name=name, **settings)
    train_and_save(model, name, data, learning_rate=1e-3, epochs=EPOCHS)

# pick the variant with the lowest val_loss
rows = []
for name, settings in variants.items():
    with open(os.path.join(RESULTS_DIR, f"{name}_summary.json")) as f:
        rows.append({**json.load(f), **settings})
variants_df = pd.DataFrame(rows)
variants_df.to_csv(os.path.join(RESULTS_DIR, "complex_cnn_variants.csv"), index=False)

best_name = variants_df.loc[variants_df["best_val_loss"].idxmin(), "model"]
shutil.copy(os.path.join(MODELS_DIR, f"{best_name}.keras"),
            os.path.join(MODELS_DIR, "complex_cnn_best.keras"))
shutil.copy(os.path.join(RESULTS_DIR, f"{best_name}_history.csv"),
            os.path.join(RESULTS_DIR, "complex_cnn_best_history.csv"))

print(variants_df[["model", "params", "epochs_run", "best_epoch",
                   "best_val_loss", "best_val_accuracy"]].to_string(index=False))
print("Best variant:", best_name)

best_history = pd.read_csv(os.path.join(RESULTS_DIR, "complex_cnn_best_history.csv"))
plot_history(best_history, f"Complex CNN – {best_name}", "complex_cnn_curve.png")
