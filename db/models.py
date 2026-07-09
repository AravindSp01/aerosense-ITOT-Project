# db/models.py
"""SQLAlchemy ORM models. Bronze = raw, append-only telemetry as received."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BronzeTelemetry(Base):
    """Raw telemetry records written directly from Kafka. Never mutated after insert."""

    __tablename__ = "bronze_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kafka_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    kafka_partition: Mapped[int] = mapped_column(Integer, nullable=False)

    # Top-level scalar fields stored directly for query performance.
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    mission_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sim_time: Mapped[float] = mapped_column(Float, nullable=False)
    battery: Mapped[float] = mapped_column(Float, nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    motor_power: Mapped[float] = mapped_column(Float, nullable=False)
    wind_force: Mapped[float] = mapped_column(Float, nullable=False)
    lidar_distance: Mapped[float] = mapped_column(Float, nullable=False)

    # Full message stored as JSON for completeness (position, imu, gyro, camera, gps).
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False)

    processed_to_silver: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

class SilverTelemetry(Base):
    """Validated, flattened telemetry. One row per valid bronze record.
    Columns are typed and named for direct query use — no JSON blobs."""

    __tablename__ = "silver_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bronze_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    mission_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sim_time: Mapped[float] = mapped_column(Float, nullable=False)

    pos_x: Mapped[float] = mapped_column(Float, nullable=False)
    pos_y: Mapped[float] = mapped_column(Float, nullable=False)
    pos_z: Mapped[float] = mapped_column(Float, nullable=False)

    roll: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[float] = mapped_column(Float, nullable=False)
    yaw: Mapped[float] = mapped_column(Float, nullable=False)

    gyro_x: Mapped[float] = mapped_column(Float, nullable=False)
    gyro_y: Mapped[float] = mapped_column(Float, nullable=False)
    gyro_z: Mapped[float] = mapped_column(Float, nullable=False)

    speed: Mapped[float] = mapped_column(Float, nullable=False)
    battery: Mapped[float] = mapped_column(Float, nullable=False)
    motor_power: Mapped[float] = mapped_column(Float, nullable=False)
    lidar_distance: Mapped[float] = mapped_column(Float, nullable=False)
    wind_force: Mapped[float] = mapped_column(Float, nullable=False)

    camera_brightness: Mapped[float] = mapped_column(Float, nullable=False)
    camera_motion_blur: Mapped[float] = mapped_column(Float, nullable=False)
    camera_object_count: Mapped[int] = mapped_column(Integer, nullable=False)

    processed_to_gold: Mapped[bool] = mapped_column(
    Integer, nullable=False, default=0, server_default="0"
)


class GoldTelemetryFeatures(Base):
    """Feature-engineered rows ready for ML. One row per silver record
    that has enough history for rolling windows."""

    __tablename__ = "gold_telemetry_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    silver_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    mission_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sim_time: Mapped[float] = mapped_column(Float, nullable=False)

    # Raw features
    battery: Mapped[float] = mapped_column(Float, nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    motor_power: Mapped[float] = mapped_column(Float, nullable=False)
    wind_force: Mapped[float] = mapped_column(Float, nullable=False)
    lidar_distance: Mapped[float] = mapped_column(Float, nullable=False)
    altitude: Mapped[float] = mapped_column(Float, nullable=False)
    abs_roll: Mapped[float] = mapped_column(Float, nullable=False)
    abs_pitch: Mapped[float] = mapped_column(Float, nullable=False)

    # Rolling features (window = settings.FEATURE_WINDOW)
    battery_mean: Mapped[float] = mapped_column(Float, nullable=False)
    speed_mean: Mapped[float] = mapped_column(Float, nullable=False)
    altitude_std: Mapped[float] = mapped_column(Float, nullable=False)
    motor_power_mean: Mapped[float] = mapped_column(Float, nullable=False)
    wind_force_max: Mapped[float] = mapped_column(Float, nullable=False)

    # Label
    risk_level: Mapped[str] = mapped_column(String, nullable=False, index=True)