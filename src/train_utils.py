import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import MODELS_DIR, RESULTS_DIR, FIGURES_DIR, BATCH_SIZE, PATIENCE


def train_and_save(model, name, data, learning_rate, epochs, initial_epoch=0):
    X_train, Y_train, X_val, Y_val = data

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, f"{name}.keras"),
                                 monitor="val_loss", save_best_only=True)

    start = time.time()
    H = model.fit(X_train, Y_train,
                  validation_data=(X_val, Y_val),
                  batch_size=BATCH_SIZE, epochs=epochs, initial_epoch=initial_epoch,
                  callbacks=[early_stop, checkpoint], verbose=2)
    train_time = time.time() - start

    history = pd.DataFrame(H.history)
    history.insert(0, "epoch", [e + 1 for e in H.epoch])
    history.to_csv(os.path.join(RESULTS_DIR, f"{name}_history.csv"), index=False)

    best = history.loc[history["val_loss"].idxmin()]
    summary = {"model": name,
               "params": int(model.count_params()),
               "epochs_run": int(len(history)),
               "best_epoch": int(best["epoch"]),
               "best_val_loss": round(float(best["val_loss"]), 4),
               "best_val_accuracy": round(float(best["val_accuracy"]), 4),
               "train_time_sec": round(train_time, 1),
               "learning_rate": learning_rate}
    with open(os.path.join(RESULTS_DIR, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return history


def plot_history(history, title, save_name, mark_epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history["epoch"], history[metric], label=f"train {metric}")
        ax.plot(history["epoch"], history[f"val_{metric}"], label=f"val {metric}")
        if mark_epoch:
            ax.axvline(mark_epoch, color="gray", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150)
    plt.close(fig)
