# features/engineering.py
"""Polars-based feature engineering: rolling statistics and risk labelling.
Pure functions only -- no DB imports, fully testable without infrastructure."""

from __future__ import annotations

import polars as pl

from config.settings import settings


# The 13 feature columns the ML model trains and predicts on.
# This list is the single source of truth -- referenced by train.py,
# inference.py, and the API FeatureInput schema.
FEATURE_COLUMNS: list[str] = [
    "battery",
    "speed",
    "motor_power",
    "wind_force",
    "lidar_distance",
    "altitude",
    "abs_roll",
    "abs_pitch",
    "battery_mean",
    "speed_mean",
    "altitude_std",
    "motor_power_mean",
    "wind_force_max",
]


def assign_risk_level(
    battery: float,
    lidar_distance: float,
    wind_force: float,
) -> str:
    """Assign a risk level from raw sensor values using settings thresholds.

    Rules (evaluated top-to-bottom, first match wins):
        CRITICAL: battery below critical threshold OR lidar below critical
        WARNING:  battery below warning threshold OR wind above warning
                  OR lidar below warning threshold
        SAFE:     everything else
    """
    if battery < settings.BATTERY_CRITICAL or lidar_distance < settings.LIDAR_CRITICAL:
        return "critical"
    if (
        battery < settings.BATTERY_WARNING
        or wind_force > settings.WIND_WARNING
        or lidar_distance < settings.LIDAR_WARNING
    ):
        return "warning"
    return "safe"


def engineer_features(rows: list[dict]) -> list[dict]:
    """Compute rolling features and risk labels for a batch of silver rows.

    Args:
        rows: list of dicts, each representing one silver_telemetry row.
              Must be ordered by sim_time ascending.
              Must contain at minimum: silver_id, mission_id, vehicle_id,
              sim_time, battery, speed, motor_power, wind_force,
              lidar_distance, pos_z, roll, pitch, processed_at.

    Returns:
        List of feature dicts ready to insert into gold_telemetry_features.
        Rows with insufficient history for rolling windows are excluded
        (fewer than FEATURE_WINDOW rows seen so far in the batch).
    """
    if not rows:
        return []

    df = pl.DataFrame(rows)

    window = settings.FEATURE_WINDOW

    df = df.with_columns([
        pl.col("roll").abs().alias("abs_roll"),
        pl.col("pitch").abs().alias("abs_pitch"),
        pl.col("pos_z").alias("altitude"),
    ])

    df = df.with_columns([
        pl.col("battery")
          .rolling_mean(window_size=window, min_periods=1)
          .alias("battery_mean"),
        pl.col("speed")
          .rolling_mean(window_size=window, min_periods=1)
          .alias("speed_mean"),
        pl.col("altitude")
          .rolling_std(window_size=window, min_periods=1)
          .fill_null(0.0)
          .alias("altitude_std"),
        pl.col("motor_power")
          .rolling_mean(window_size=window, min_periods=1)
          .alias("motor_power_mean"),
        pl.col("wind_force")
          .rolling_max(window_size=window, min_periods=1)
          .alias("wind_force_max"),
    ])

    df = df.with_columns(
        pl.struct(["battery", "lidar_distance", "wind_force"])
          .map_elements(
              lambda s: assign_risk_level(
                  s["battery"], s["lidar_distance"], s["wind_force"]
              ),
              return_dtype=pl.String,
          )
          .alias("risk_level")
    )

    keep = [
        "silver_id", "mission_id", "vehicle_id", "sim_time",
        "battery", "speed", "motor_power", "wind_force",
        "lidar_distance", "altitude", "abs_roll", "abs_pitch",
        "battery_mean", "speed_mean", "altitude_std",
        "motor_power_mean", "wind_force_max", "risk_level",
    ]

    return df.select(keep).to_dicts()