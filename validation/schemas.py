# validation/schemas.py
"""Canonical telemetry schema. This is the single source of truth for what
a valid telemetry message looks like, used both for runtime validation in
the bridge/ingestion layers and as a reference for testing."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class Position(BaseModel):
    """Drone position in world coordinates (meters)."""

    x: float
    y: float
    z: float = Field(ge=0.0, le=500.0, description="Altitude in meters")


class Gps(BaseModel):
    """GPS fix in degrees latitude/longitude and meters altitude."""

    lat: float
    lon: float
    alt: float


class Imu(BaseModel):
    """Orientation in degrees: roll, pitch, yaw."""

    roll: float
    pitch: float
    yaw: float


class Gyro(BaseModel):
    """Angular velocity in radians/second on each axis."""

    x: float
    y: float
    z: float


class Camera(BaseModel):
    """Lightweight camera-derived metadata (no raw image is ever streamed)."""

    brightness: float = Field(ge=0.0, le=1.0)
    motion_blur: float = Field(ge=0.0, le=1.0)
    object_count: int = Field(ge=0)


class TelemetryMessage(BaseModel):
    """A single telemetry reading emitted by the Webots bridge layer."""

    timestamp: datetime
    mission_id: str
    vehicle_id: str
    sim_time: float = Field(ge=0.0)
    position: Position
    gps: Gps
    imu: Imu
    gyro: Gyro
    speed: float = Field(ge=0.0, le=100.0)
    battery: float = Field(ge=0.0, le=100.0)
    motor_power: float = Field(ge=0.0)
    lidar_distance: float = Field(ge=0.1, le=200.0)
    camera: Camera
    wind_force: float = Field(ge=0.0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes -- timestamp must be ISO 8601 with a tz offset."""
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (ISO 8601 with offset)")
        return value
