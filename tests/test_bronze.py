# tests/test_bronze.py
"""Unit tests for the bronze consumer helpers -- no live Kafka or DB needed."""

from __future__ import annotations

import pytest

from ingestion.bronze_consumer import _make_record

SAMPLE_PAYLOAD = {
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


def test_make_record_scalar_fields() -> None:
    record = _make_record(SAMPLE_PAYLOAD, offset=42, partition=0)
    assert record.mission_id == "m-001"
    assert record.vehicle_id == "drone-alpha"
    assert record.sim_time == pytest.approx(1.016)
    assert record.kafka_offset == 42
    assert record.kafka_partition == 0


def test_make_record_raw_payload_preserved() -> None:
    record = _make_record(SAMPLE_PAYLOAD, offset=0, partition=0)
    assert record.raw_payload["camera"]["object_count"] == 2


def test_make_record_ingested_at_is_utc() -> None:
    record = _make_record(SAMPLE_PAYLOAD, offset=0, partition=0)
    assert record.ingested_at.tzinfo is not None
