# tests/test_bridge.py
"""Unit tests for bridge.sensor_mapper -- runs with no Webots/Kafka instance."""

from __future__ import annotations

import numpy as np
import pytest

from bridge.sensor_mapper import map_to_telemetry
from validation.schemas import TelemetryMessage

SAMPLE_KWARGS = {
    "gps": (53.55, 10.00, 120.0),
    "imu": (0.02, -0.01, 1.64),  # radians
    "gyro": (0.01, -0.02, 0.0),
    "compass": (0.99, 0.05, 0.0),
    "camera_image": None,
    "lidar_distance": 18.4,
    "battery": 74.2,
    "motor_power": 62.5,
    "sim_time": 184.0,
    "mission_id": "m-001",
    "vehicle_id": "drone-alpha",
    "position": (312.4, 87.1, 120.0),
    "speed": 8.3,
    "wind_force": 4.1,
}


def test_map_to_telemetry_produces_schema_valid_dict() -> None:
    """The mapper's output must validate cleanly against TelemetryMessage."""
    result = map_to_telemetry(**SAMPLE_KWARGS)
    message = TelemetryMessage.model_validate(result)
    assert message.vehicle_id == "drone-alpha"
    assert message.mission_id == "m-001"
    assert message.position.z == pytest.approx(120.0)


def test_map_to_telemetry_converts_radians_to_degrees() -> None:
    """imu roll/pitch/yaw in the output dict should be in degrees, not radians."""
    result = map_to_telemetry(**SAMPLE_KWARGS)
    assert result["imu"]["yaw"] == pytest.approx(np.degrees(1.64))


def test_map_to_telemetry_handles_missing_camera() -> None:
    """A None camera_image should yield zeroed-out camera metadata, not crash."""
    result = map_to_telemetry(**SAMPLE_KWARGS)
    assert result["camera"] == {"brightness": 0.0, "motion_blur": 0.0, "object_count": 0}


def test_map_to_telemetry_camera_brightness_from_image() -> None:
    """A bright uniform image should yield brightness close to 1.0."""
    bright_image = np.full((10, 10), 255, dtype=np.uint8)
    kwargs = dict(SAMPLE_KWARGS, camera_image=bright_image)
    result = map_to_telemetry(**kwargs)
    assert result["camera"]["brightness"] == pytest.approx(1.0, abs=0.01)


def test_map_to_telemetry_object_count_detects_blob() -> None:
    """A single bright square on a dark background should register as one object."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:12, 5:12] = 255  # 7x7 = 49 pixels, above _MIN_OBJECT_PIXELS
    kwargs = dict(SAMPLE_KWARGS, camera_image=image)
    result = map_to_telemetry(**kwargs)
    assert result["camera"]["object_count"] == 1
