# models/inference.py
"""Load the registered aerosense-risk model from MLflow and expose a
predict() function. Model is loaded once at import time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import mlflow
import numpy as np
import structlog
from mlflow.pyfunc import PyFuncModel, load_model

from config.settings import settings
from features.engineering import FEATURE_COLUMNS, RISK_LEVELS

logger = structlog.get_logger(__name__)

_MODEL_NAME = "aerosense-risk"
_MODEL_STAGE = "latest"  # use "Production" after manually promoting in MLflow UI


@dataclass
class PredictionResult:
    """Result of a single inference call."""

    risk_level: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


def _load_model() -> tuple[PyFuncModel, str]:
    """Load the registered model. Called once at module import."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    try:
        model = load_model(f"models:/{_MODEL_NAME}/{_MODEL_STAGE}")
        version = model.metadata.run_id[:8]
        logger.info("inference_model_loaded", name=_MODEL_NAME, version=version)
        return model, version
    except Exception as exc:
        logger.error("inference_model_load_failed", error=str(exc))
        raise


_model: PyFuncModel | None

try:
    _model, _model_version = _load_model()
    MODEL_LOADED = True
except Exception as exc:  # noqa: BLE001
    logger.error("model_load_failed", error=str(exc))
    _model = None
    _model_version = "none"
    MODEL_LOADED = False


def predict(features: dict[str, float]) -> PredictionResult:
    """Run inference on a single feature dict.

    Args:
        features: dict with keys matching FEATURE_COLUMNS.

    Returns:
        PredictionResult with risk_level, confidence, probabilities, model_version.

    Raises:
        RuntimeError: if the model failed to load at startup.
        KeyError: if a required feature column is missing.
    """
    if not MODEL_LOADED or _model is None:
        raise RuntimeError("Model is not loaded -- run models/train.py first.")

    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    raw = cast(Any, _model).predict(x)

    # MLflow pyfunc returns numpy array of predicted labels for sklearn models.
    predicted_label = str(raw[0])

    # For probabilities we need to call the underlying model directly.
    # MLflow pyfunc wraps sklearn, so the underlying estimator is accessible.
    try:
        underlying = _model._model_impl.python_model  # type: ignore[attr-defined]
        proba = underlying.predict_proba(x)[0]
        classes = underlying.classes_
    except AttributeError:
        # Fallback: uniform probabilities (e.g. if model type doesn't support proba).
        proba = np.array([1.0 / len(RISK_LEVELS)] * len(RISK_LEVELS))
        classes = RISK_LEVELS

    prob_dict = {str(cls): float(p) for cls, p in zip(classes, proba, strict=False)}
    confidence = float(max(proba))

    return PredictionResult(
        risk_level=predicted_label,
        confidence=confidence,
        probabilities=prob_dict,
        model_version=_model_version,
    )
