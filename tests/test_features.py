# tests/test_features.py
"""Unit tests for features.engineering -- no DB or Kafka needed."""

from __future__ import annotations

import pytest
from features.engineering import assign_risk_level, engineer_features, FEATURE_COLUMNS


def _make_rows(n: int, battery: float = 80.0, wind: float = 2.0,
               lidar: float = 30.0) -> list[dict]:
    """Generate n synthetic silver rows for testing."""
    return [
        {
            "silver_id": i,
            "mission_id": "m-001",
            "vehicle_id": "drone-alpha",
            "sim_time": float(i),
            "battery": battery,
            "speed": 5.0,
            "motor_power": 70.0,
            "wind_force": wind,
            "lidar_distance": lidar,
            "pos_z": 20.0,
            "roll": 0.1,
            "pitch": -0.05,
        }
        for i in range(n)
    ]


def test_assign_risk_level_safe() -> None:
    assert assign_risk_level(battery=80.0, lidar_distance=30.0, wind_force=2.0) == "safe"


def test_assign_risk_level_warning_battery() -> None:
    assert assign_risk_level(battery=25.0, lidar_distance=30.0, wind_force=2.0) == "warning"


def test_assign_risk_level_warning_wind() -> None:
    assert assign_risk_level(battery=80.0, lidar_distance=30.0, wind_force=10.0) == "warning"


def test_assign_risk_level_critical_battery() -> None:
    assert assign_risk_level(battery=10.0, lidar_distance=30.0, wind_force=2.0) == "critical"


def test_assign_risk_level_critical_lidar() -> None:
    assert assign_risk_level(battery=80.0, lidar_distance=1.0, wind_force=2.0) == "critical"


def test_engineer_features_returns_expected_columns() -> None:
    rows = _make_rows(5)
    result = engineer_features(rows)
    assert len(result) == 5
    for col in FEATURE_COLUMNS:
        assert col in result[0], f"Missing feature column: {col}"


def test_engineer_features_empty_input() -> None:
    assert engineer_features([]) == []


def test_engineer_features_risk_level_matches_rules() -> None:
    rows = _make_rows(5, battery=10.0)
    result = engineer_features(rows)
    assert all(r["risk_level"] == "critical" for r in result)


def test_engineer_features_rolling_mean_converges() -> None:
    """battery_mean should equal battery when battery is constant."""
    rows = _make_rows(35, battery=75.0)
    result = engineer_features(rows)
    last = result[-1]
    assert last["battery_mean"] == pytest.approx(75.0, abs=0.01)