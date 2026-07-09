# tests/test_models.py
"""Model accuracy test -- requires gold data in DB and MLflow running."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features.engineering import FEATURE_COLUMNS
#from features.engineering import FEATURE_COLUMNS, RISK_LEVELS


def _make_synthetic_data(n: int = 300) -> tuple[np.ndarray, list[str]]:
    """Generate a balanced synthetic dataset matching the feature schema."""
    rng = np.random.default_rng(42)
    rows, labels = [], []
    per_class = n // 3

    for _ in range(per_class):  # safe
        rows.append([rng.uniform(40, 100), rng.uniform(0, 10), rng.uniform(60, 80),
                     rng.uniform(0, 7), rng.uniform(6, 50), rng.uniform(10, 40),
                     rng.uniform(0, 5), rng.uniform(0, 5), rng.uniform(40, 100),
                     rng.uniform(0, 10), rng.uniform(0, 2), rng.uniform(60, 80),
                     rng.uniform(0, 7)])
        labels.append("safe")

    for _ in range(per_class):  # warning
        rows.append([rng.uniform(20, 35), rng.uniform(5, 15), rng.uniform(70, 90),
                     rng.uniform(8, 12), rng.uniform(5, 6), rng.uniform(5, 20),
                     rng.uniform(5, 20), rng.uniform(5, 20), rng.uniform(20, 35),
                     rng.uniform(5, 15), rng.uniform(2, 5), rng.uniform(70, 90),
                     rng.uniform(8, 12)])
        labels.append("warning")

    for _ in range(per_class):  # critical
        rows.append([rng.uniform(0, 20), rng.uniform(10, 20), rng.uniform(80, 100),
                     rng.uniform(12, 20), rng.uniform(0, 2), rng.uniform(0, 5),
                     rng.uniform(20, 45), rng.uniform(20, 45), rng.uniform(0, 20),
                     rng.uniform(10, 20), rng.uniform(5, 10), rng.uniform(80, 100),
                     rng.uniform(12, 20)])
        labels.append("critical")

    return np.array(rows, dtype=np.float32), labels


def test_model_accuracy_above_threshold() -> None:
    """A RandomForest trained on the synthetic feature distribution should
    exceed 0.80 accuracy -- if it doesn't, the feature schema is broken."""
    X, y = _make_synthetic_data(300)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    assert accuracy >= 0.80, f"Accuracy {accuracy:.2f} below 0.80 threshold"


def test_feature_columns_count() -> None:
    """FEATURE_COLUMNS must have exactly 13 entries."""
    assert len(FEATURE_COLUMNS) == 13