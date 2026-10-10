import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

from config import CLASS_NAMES, BATCH_SIZE, MODELS_DIR, RESULTS_DIR, FIGURES_DIR
from data_utils import check_data_exists, load_split
from eval_utils import (compute_metrics, make_report, per_class_f1, plot_confusion_matrix,
                        top_confused_pairs)

check_data_exists()
X_test, y_test = load_split("test")
print("X_test:", X_test.shape)

# name: (model file, history file, short name for figures)
models = {
    "Simple CNN": ("simple_cnn", "simple_cnn_history", "simple"),
    "Complex CNN": ("complex_cnn_best", "complex_cnn_best_history", "complex"),
    "ResNet50V2": ("resnet50v2_finetuned", "resnet50v2_full_history", "resnet50v2"),
}


def train_time_minutes(name):
    if name == "ResNet50V2":
        files = ["resnet50v2_phase1", "resnet50v2_finetuned"]
    elif name == "Complex CNN":
        variants = pd.read_csv(os.path.join(RESULTS_DIR, "complex_cnn_variants.csv"))
        files = [variants.sort_values("best_val_loss").iloc[0]["model"]]
    else:
        files = ["simple_cnn"]
    seconds = 0
    for f in files:
        with open(os.path.join(RESULTS_DIR, f"{f}_summary.json")) as fh:
            seconds += json.load(fh)["train_time_sec"]
    return seconds / 60


# ---------- evaluate on the test set ----------
os.makedirs(os.path.join(RESULTS_DIR, "classification_reports"), exist_ok=True)
predictions = {}
confusion = {}
rows = []
for name, (model_file, _, short) in models.items():
    model = load_model(os.path.join(MODELS_DIR, f"{model_file}.keras"))
    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    predictions[name] = y_pred

    rows.append({"model": name, **compute_metrics(y_test, y_pred),
                 "params": model.count_params(), "train_time_min": train_time_minutes(name)})

    report = make_report(y_test, y_pred)
    with open(os.path.join(RESULTS_DIR, "classification_reports", f"{model_file}.txt"), "w") as f:
        f.write(report)
    print(f"\n===== {name} =====\n{report}")

    confusion[name] = plot_confusion_matrix(y_test, y_pred, f"Confusion matrix – {name}", f"cm_{short}.png")

test_metrics = pd.DataFrame(rows).round(4)
test_metrics.to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"), index=False)
print(test_metrics.to_string(index=False))

# ---------- per-class F1 ----------
f1_table = pd.DataFrame({name: per_class_f1(y_test, y_pred) for name, y_pred in predictions.items()},
                        index=CLASS_NAMES).round(4)
f1_table.to_csv(os.path.join(RESULTS_DIR, "per_class_f1.csv"))
print(f1_table.to_string())

f1_table.plot(kind="bar", figsize=(12, 4.5), width=0.8)
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.title("Per-class F1 – 3 models")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "per_class_f1.png"), dpi=150)
plt.close()

# ---------- validation curves of the 3 models ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, (_, history_file, _) in models.items():
    h = pd.read_csv(os.path.join(RESULTS_DIR, f"{history_file}.csv"))
    axes[0].plot(h["epoch"], h["val_loss"], label=name)
    axes[1].plot(h["epoch"], h["val_accuracy"], label=name)
axes[0].set_title("Validation loss")
axes[1].set_title("Validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "learning_curves_all.png"), dpi=150)
plt.close()

# ---------- error analysis of the best model ----------
best_name = test_metrics.sort_values("macro_f1", ascending=False).iloc[0]["model"]
top_pairs = top_confused_pairs(confusion[best_name])
top_pairs.to_csv(os.path.join(RESULTS_DIR, "top_confused_pairs.csv"), index=False)
print("Best model:", best_name)
print(top_pairs.to_string(index=False))

wrong = np.where(predictions[best_name] != y_test)[0][:16]
plt.figure(figsize=(12, 12))
for k, idx in enumerate(wrong):
    plt.subplot(4, 4, k + 1)
    plt.imshow(X_test[idx])
    plt.title(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[predictions[best_name][idx]]}", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "misclassified_examples.png"), dpi=150)
plt.close()
