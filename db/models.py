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
