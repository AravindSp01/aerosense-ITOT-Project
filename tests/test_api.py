# tests/test_api.py
"""API endpoint tests using FastAPI's TestClient -- no live model needed
for health/metrics; predict endpoint is tested with a mock."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from models.inference import PredictionResult

client = TestClient(app)

VALID_FEATURES = {
    "battery": 80.0,
    "speed": 5.0,
    "motor_power": 70.0,
    "wind_force": 2.0,
    "lidar_distance": 30.0,
    "altitude": 20.0,
    "abs_roll": 1.0,
    "abs_pitch": 0.5,
    "battery_mean": 80.0,
    "speed_mean": 5.0,
    "altitude_std": 0.3,
    "motor_power_mean": 70.0,
    "wind_force_max": 3.0,
}

MOCK_RESULT = PredictionResult(
    risk_level="safe",
    confidence=0.92,
    probabilities={"safe": 0.92, "warning": 0.06, "critical": 0.02},
    model_version="abc12345",
)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "db_connected" in body
    assert "uptime_seconds" in body


def test_metrics_returns_200() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "predictions_total" in body
    assert "predictions_by_risk" in body
    assert "avg_confidence" in body


def test_predict_returns_valid_risk_level() -> None:
    with patch("api.app.MODEL_LOADED", True), patch("api.app.predict", return_value=MOCK_RESULT):
        response = client.post("/predict", json=VALID_FEATURES)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in ("safe", "warning", "critical")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "probabilities" in body
    assert "model_version" in body


def test_predict_422_on_missing_feature() -> None:
    bad = dict(VALID_FEATURES)
    del bad["battery"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_on_out_of_range() -> None:
    bad = dict(VALID_FEATURES, battery=200.0)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_503_when_model_not_loaded() -> None:
    with patch("api.app.MODEL_LOADED", False):
        response = client.post("/predict", json=VALID_FEATURES)
    assert response.status_code == 503
