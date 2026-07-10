# models/evaluate.py
"""Compute and return classification metrics. Pure functions, no MLflow
imports -- called by train.py which handles the logging."""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RISK_LEVELS = ["safe", "warning", "critical"]


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    y_proba: np.ndarray,
    labels: list[str] = RISK_LEVELS,
) -> dict[str, float]:
    """Return a flat dict of classification metrics suitable for MLflow logging.

    Args:
        y_true: ground truth risk level strings.
        y_pred: predicted risk level strings.
        y_proba: predicted probabilities, shape (n_samples, n_classes).
        labels: ordered list of class labels matching y_proba columns.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # ROC-AUC: only meaningful when all classes are present in y_true.
    present = set(y_true)
    if len(present) > 1:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def confusion_matrix_png(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = RISK_LEVELS,
) -> str:
    """Render a confusion matrix and return it as a base64 PNG string.
    Saved as a file by train.py and logged to MLflow as an artifact."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
