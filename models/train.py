# models/train.py
"""Training pipeline: loads gold features from Postgres, trains a
classifier, logs everything to MLflow, and registers the best model.
Run with: python -m models.train
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select
from xgboost import XGBClassifier

from config.settings import settings
from db.models import GoldTelemetryFeatures
from db.session import get_session
from features.engineering import TRAINING_COLUMNS as FEATURE_COLUMNS
from models.evaluate import RISK_LEVELS, compute_metrics, confusion_matrix_png

logger = structlog.get_logger(__name__)

LABEL_ORDER = RISK_LEVELS


def load_training_data() -> tuple[np.ndarray, list[str]]:
    """Load all gold rows from Postgres."""

    with get_session() as session:
        rows = session.execute(select(GoldTelemetryFeatures)).scalars().all()

    if not rows:
        raise RuntimeError("No gold rows found -- run the gold processor first.")

    X = np.array(
        [[getattr(r, col) for col in FEATURE_COLUMNS] for r in rows],
        dtype=np.float32,
    )

    y = [r.risk_level for r in rows]

    logger.info(
        "training_data_loaded",
        n_samples=len(y),
        class_counts={lvl: y.count(lvl) for lvl in LABEL_ORDER},
    )

    return X, y


def _train_random_forest(
    X_train: np.ndarray,
    y_train: list[str],
    class_weight: dict,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    return clf


def _train_xgboost(
    X_train: np.ndarray,
    y_train: list[str],
    le: LabelEncoder,
) -> XGBClassifier:
    """
    Train XGBoost using only the classes that actually exist in y_train.
    """

    y_enc = le.transform(y_train)

    # Compute balanced sample weights only for existing classes
    classes, counts = np.unique(y_enc, return_counts=True)

    total = counts.sum()

    class_weights = {
        cls: total / (len(classes) * count) for cls, count in zip(classes, counts, strict=False)
    }

    sample_weight = np.array([class_weights[c] for c in y_enc])

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_train,
        y_enc,
        sample_weight=sample_weight,
    )

    return clf


def run_training() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    X, y = load_training_data()

    if len(set(y)) < 2:
        raise RuntimeError(f"Only one class present in training data: {set(y)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ------------------------------------------------------------
    # CHANGED
    # Fit LabelEncoder ONLY on classes that actually exist.
    # ------------------------------------------------------------
    le = LabelEncoder()
    le.fit(y_train)

    labels_present = list(le.classes_)

    # ------------------------------------------------------------
    # RandomForest class weights
    # ------------------------------------------------------------
    counts = {lbl: y_train.count(lbl) for lbl in labels_present}

    total = len(y_train)

    class_weight = {lbl: total / (len(counts) * cnt) for lbl, cnt in counts.items()}

    candidates = []

    # ============================================================
    # Random Forest
    # ============================================================

    with mlflow.start_run(run_name="random_forest") as rf_run:
        mlflow.log_params(
            {
                "model_type": "RandomForest",
                "n_estimators": 200,
                "max_depth": 8,
                "feature_columns": FEATURE_COLUMNS,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        rf = _train_random_forest(
            X_train,
            y_train,
            class_weight,
        )

        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)

        metrics = compute_metrics(
            y_test,
            y_pred,
            y_proba,
            labels=labels_present,
        )

        mlflow.log_metrics(metrics)

        _log_artifacts(
            rf,
            y_test,
            y_pred,
            rf_run.info.run_id,
            "random_forest",
        )

        logger.info("rf_trained", **metrics)

        candidates.append(
            (
                "random_forest",
                rf,
                metrics,
            )
        )

    # ============================================================
    # XGBoost
    # ============================================================

    with mlflow.start_run(run_name="xgboost") as xgb_run:
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "feature_columns": FEATURE_COLUMNS,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        xgb = _train_xgboost(
            X_train,
            y_train,
            le,
        )

        y_pred_enc = xgb.predict(X_test)

        y_pred = le.inverse_transform(y_pred_enc)

        y_proba = xgb.predict_proba(X_test)

        metrics = compute_metrics(
            y_test,
            y_pred,
            y_proba,
            labels=labels_present,
        )

        mlflow.log_metrics(metrics)

        _log_artifacts(
            xgb,
            y_test,
            y_pred,
            xgb_run.info.run_id,
            "xgboost",
        )

        logger.info("xgb_trained", **metrics)

        candidates.append(
            (
                "xgboost",
                xgb,
                metrics,
            )
        )

    # ============================================================
    # Register best model
    # ============================================================

    best_name, best_model, best_metrics = max(
        candidates,
        key=lambda x: x[2]["f1_macro"],
    )

    logger.info(
        "best_model_selected",
        model=best_name,
        **best_metrics,
    )

    with mlflow.start_run(run_name=f"register_{best_name}"):
        mlflow.log_params(
            {
                "registered_model": best_name,
            }
        )

        mlflow.log_metrics(best_metrics)

        if isinstance(best_model, XGBClassifier):
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                name="model",
                registered_model_name="aerosense-risk",
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                registered_model_name="aerosense-risk",
            )

            logger.info(
                "model_registered",
                name="aerosense-risk",
            )


def _log_artifacts(
    model,
    y_true,
    y_pred,
    run_id,
    model_name,
):
    png_b64 = confusion_matrix_png(y_true, y_pred)

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "confusion_matrix.png"

        png_path.write_bytes(base64.b64decode(png_b64))

        mlflow.log_artifact(str(png_path))


if __name__ == "__main__":
    run_training()
