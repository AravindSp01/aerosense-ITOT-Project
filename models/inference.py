# models/inference.py
"""Model loading and inference. State is managed by the API layer --
never loaded at module import time."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import mlflow.sklearn
import numpy as np
import structlog
from mlflow.exceptions import MlflowException
from sklearn.base import ClassifierMixin

from config.settings import settings
from features.engineering import FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

_MODEL_NAME = "aerosense-risk"
_MODEL_ALIAS = "latest"

# Mutable state -- controlled by app.py
_model: ClassifierMixin | None = None
_model_version: str = "none"
MODEL_LOADED: bool = False


@dataclass
class PredictionResult:
    risk_level: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


def reload_model() -> bool:
    """Attempt to load/reload the latest registered model from MLflow.
    Returns True if a new version was loaded, False otherwise."""
    global _model, _model_version, MODEL_LOADED

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
        if not versions:
            return False

        latest_version = versions[0].version
        if latest_version == _model_version:
            return False  # already on this version, nothing to do

        model = mlflow.sklearn.load_model(f"models:/{_MODEL_NAME}/{_MODEL_ALIAS}")
        _model = model
        _model_version = latest_version
        MODEL_LOADED = True
        logger.info("model_loaded", name=_MODEL_NAME, version=_model_version)
        return True

    except (FileNotFoundError, OSError, pickle.UnpicklingError, MlflowException) as exc:
        logger.warning("model_load_failed", error=str(exc))
        return False


def predict(features: dict[str, float]) -> PredictionResult:
    if not MODEL_LOADED or _model is None:
        raise RuntimeError("Model not loaded -- run the training pipeline first.")

    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    predicted_label = str(_model.predict(x)[0])
    proba = _model.predict_proba(x)[0]
    classes = [str(c) for c in _model.classes_]
    prob_dict = dict(zip(classes, [float(p) for p in proba], strict=False))

    return PredictionResult(
        risk_level=predicted_label,
        confidence=float(max(proba)),
        probabilities=prob_dict,
        model_version=_model_version,
    )
