# tests/test_validation.py
"""Unit tests for validation.schemas.TelemetryMessage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from validation.schemas import TelemetryMessage

VALID_PAYLOAD = {
    "timestamp": "2026-06-01T12:01:10Z",
    "mission_id": "m-001",
    "vehicle_id": "drone-alpha",
    "sim_time": 184.0,
    "position": {"x": 312.4, "y": 87.1, "z": 120.0},
    "gps": {"lat": 53.55, "lon": 10.00, "alt": 120.0},
    "imu": {"roll": 1.2, "pitch": -0.4, "yaw": 94.0},
    "gyro": {"x": 0.01, "y": -0.02, "z": 0.00},
    "speed": 8.3,
    "battery": 74.2,
    "motor_power": 62.5,
    "lidar_distance": 18.4,
    "camera": {"brightness": 0.72, "motion_blur": 0.15, "object_count": 3},
    "wind_force": 4.1,
}


def test_valid_message_passes() -> None:
    """A well-formed payload should validate without error."""
    message = TelemetryMessage.model_validate(VALID_PAYLOAD)
    assert message.vehicle_id == "drone-alpha"


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        ("battery", -1.0),
        ("battery", 101.0),
        ("speed", -5.0),
        ("speed", 150.0),
        ("lidar_distance", 0.0),
        ("lidar_distance", 250.0),
    ],
)
def test_out_of_range_top_level_fields_raise(field_path: str, bad_value: float) -> None:
    """Out-of-range scalar fields must raise ValidationError."""
    payload = dict(VALID_PAYLOAD, **{field_path: bad_value})
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate(payload)


def test_out_of_range_altitude_raises() -> None:
    """position.z outside [0, 500] must raise ValidationError."""
    payload = dict(VALID_PAYLOAD, position={"x": 0.0, "y": 0.0, "z": 999.0})
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate(payload)


def test_out_of_range_camera_brightness_raises() -> None:
    """camera.brightness outside [0, 1] must raise ValidationError."""
    payload = dict(VALID_PAYLOAD, camera={"brightness": 1.5, "motion_blur": 0.1, "object_count": 0})
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate(payload)


def test_naive_timestamp_raises() -> None:
    """A timestamp without timezone info must be rejected."""
    payload = dict(VALID_PAYLOAD, timestamp="2026-06-01T12:01:10")
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate(payload)
