import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

from config import CLASS_NAMES, NUM_CLASSES, FIGURES_DIR

LABELS = list(range(NUM_CLASSES))


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def make_report(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES,
                                 digits=4, zero_division=0)


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, title, save_name):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150)
    plt.close()
    return cm


def top_confused_pairs(cm, n=5):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], round(cm[i, j], 4))
             for i in LABELS for j in LABELS if cm[i, j] > 0]
    pairs = sorted(pairs, key=lambda p: -p[2])[:n]
    return pd.DataFrame(pairs, columns=["True class", "Predicted as", "Rate"])
