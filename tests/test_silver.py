# tests/test_silver.py
"""Unit tests for silver processor helpers -- no live DB needed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestion.silver_processor import _flatten
from db.models import BronzeTelemetry, SilverTelemetry
from validation.schemas import TelemetryMessage
from datetime import datetime, timezone

VALID_PAYLOAD = {
    "timestamp": "2026-07-03T06:32:40.251018+00:00",
    "mission_id": "m-001",
    "vehicle_id": "drone-alpha",
    "sim_time": 1.016,
    "position": {"x": -51.1, "y": -41.6, "z": 15.8},
    "gps": {"lat": -51.1, "lon": -41.6, "alt": 15.8},
    "imu": {"roll": -3.3, "pitch": 6.8, "yaw": 69.9},
    "gyro": {"x": 0.22, "y": -0.29, "z": -2.6},
    "speed": 0.0,
    "battery": 99.99,
    "motor_power": 71.5,
    "lidar_distance": 30.0,
    "camera": {"brightness": 0.77, "motion_blur": 0.0, "object_count": 2},
    "wind_force": 2.17,
}


def _make_bronze(payload: dict) -> BronzeTelemetry:
    """Build a minimal BronzeTelemetry instance for testing (no DB needed)."""
    b = BronzeTelemetry()
    b.id = 1
    b.raw_payload = payload
    b.timestamp = payload["timestamp"]
    return b


def test_flatten_produces_correct_fields() -> None:
    bronze = _make_bronze(VALID_PAYLOAD)
    validated = TelemetryMessage.model_validate(VALID_PAYLOAD)
    silver = _flatten(bronze, validated)
    assert silver.bronze_id == 1
    assert silver.mission_id == "m-001"
    assert silver.pos_z == pytest.approx(15.8)
    assert silver.camera_object_count == 2


def test_flatten_preserves_imu_degrees() -> None:
    """imu.roll in silver must be in degrees (already converted by sensor_mapper)."""
    bronze = _make_bronze(VALID_PAYLOAD)
    validated = TelemetryMessage.model_validate(VALID_PAYLOAD)
    silver = _flatten(bronze, validated)
    assert silver.roll == pytest.approx(-3.3)


def test_invalid_payload_raises_validation_error() -> None:
    bad_payload = dict(VALID_PAYLOAD, battery=200.0)
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate(bad_payload)


def test_processed_at_is_timezone_aware() -> None:
    bronze = _make_bronze(VALID_PAYLOAD)
    validated = TelemetryMessage.model_validate(VALID_PAYLOAD)
    silver = _flatten(bronze, validated)
    assert silver.processed_at.tzinfo is not None