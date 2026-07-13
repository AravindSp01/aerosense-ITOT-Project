# models/inference.py
"""Load the registered aerosense-risk model from MLflow and expose a
predict() function. Model is loaded once at module import time -- never
per request."""

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


@dataclass
class PredictionResult:
    """Result of a single inference call."""

    risk_level: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


def _load_model() -> tuple[ClassifierMixin, str]:
    """Load the latest registered sklearn model from MLflow."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{_MODEL_NAME}/{_MODEL_ALIAS}")
        # Grab run_id from the MLflow client for versioning.
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
        version_str = versions[0].version if versions else "unknown"
        logger.info("inference_model_loaded", name=_MODEL_NAME, version=version_str)
        return model, version_str
    except Exception as exc:
        logger.error("inference_model_load_failed", error=str(exc))
        raise


try:
    _model, _model_version = _load_model()
    MODEL_LOADED = True
except (FileNotFoundError, OSError, pickle.UnpicklingError, MlflowException):
    logger.warning("No registered MLflow model found. API will start without a loaded model.")
    _model = None  # type: ignore[assignment]
    _model_version = "none"
    MODEL_LOADED = False


def predict(features: dict[str, float]) -> PredictionResult:
    """Run inference on a single feature dict.

    Args:
        features: dict with keys exactly matching FEATURE_COLUMNS.

    Returns:
        PredictionResult with risk_level, confidence, probabilities, model_version.

    Raises:
        RuntimeError: if the model failed to load at startup.
        KeyError: if a required feature column is missing from features.
    """
    if not MODEL_LOADED or _model is None:
        raise RuntimeError("Model not loaded -- run python -m models.train first.")

    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    predicted_label = str(_model.predict(x)[0])

    proba = _model.predict_proba(x)[0]
    classes = [str(c) for c in _model.classes_]
    prob_dict = dict(zip(classes, [float(p) for p in proba], strict=False))
    confidence = float(max(proba))

    return PredictionResult(
        risk_level=predicted_label,
        confidence=confidence,
        probabilities=prob_dict,
        model_version=_model_version,
    )
