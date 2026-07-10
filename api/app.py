# api/app.py
"""FastAPI inference endpoint. Loads the risk model once at startup.
POST /predict  -- run inference on a gold feature row
GET  /health   -- liveness + model/db status
GET  /metrics  -- in-memory prediction counters"""

from __future__ import annotations

import time
from collections import defaultdict

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from db.session import get_session
from models.inference import MODEL_LOADED, PredictionResult, predict

logger = structlog.get_logger(__name__)

app = FastAPI(title="AeroSense Risk API", version="1.0.0")

_start_time = time.time()

# In-memory counters -- reset on restart.
_predictions_total: int = 0
_predictions_by_risk: dict[str, int] = defaultdict(int)
_confidence_sum: float = 0.0


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #


class FeatureInput(BaseModel):
    """13 features matching FEATURE_COLUMNS exactly."""

    battery: float = Field(..., ge=0.0, le=100.0)
    speed: float = Field(..., ge=0.0, le=100.0)
    motor_power: float = Field(..., ge=0.0)
    wind_force: float = Field(..., ge=0.0)
    lidar_distance: float = Field(..., ge=0.1, le=200.0)
    altitude: float = Field(..., ge=0.0, le=500.0)
    abs_roll: float = Field(..., ge=0.0)
    abs_pitch: float = Field(..., ge=0.0)
    battery_mean: float = Field(..., ge=0.0, le=100.0)
    speed_mean: float = Field(..., ge=0.0, le=100.0)
    altitude_std: float = Field(..., ge=0.0)
    motor_power_mean: float = Field(..., ge=0.0)
    wind_force_max: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    risk_level: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    predictions_total: int
    predictions_by_risk: dict[str, int]
    avg_confidence: float


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: FeatureInput) -> PredictResponse:
    """Run risk inference on a single feature row."""
    global _predictions_total, _confidence_sum

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python -m models.train first.",
        )

    try:
        result: PredictionResult = predict(body.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing feature: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Update counters.
    _predictions_total += 1
    _predictions_by_risk[result.risk_level] += 1
    _confidence_sum += result.confidence

    logger.info(
        "prediction_made", risk_level=result.risk_level, confidence=round(result.confidence, 3)
    )

    return PredictResponse(
        risk_level=result.risk_level,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_version=result.model_version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: model status + DB connectivity."""
    db_ok = False
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        db_ok = True
    except SQLAlchemyError as exc:
        logger.warning("health_db_check_failed", error=str(exc))

    return HealthResponse(
        status="ok",
        model_loaded=MODEL_LOADED,
        db_connected=db_ok,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """In-memory prediction counters since last restart."""
    avg_confidence = _confidence_sum / _predictions_total if _predictions_total > 0 else 0.0
    return MetricsResponse(
        predictions_total=_predictions_total,
        predictions_by_risk=dict(_predictions_by_risk),
        avg_confidence=round(avg_confidence, 3),
    )


@app.get("/")
def root() -> RedirectResponse:
    """Redirect root to the auto-generated API docs."""
    return RedirectResponse(url="/docs")
