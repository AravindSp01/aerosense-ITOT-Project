"""Silver processor: reads unprocessed bronze rows, validates and flattens
them into silver_telemetry. Runs as a polling loop on SILVER_BATCH_INTERVAL.
Run with: python -m ingestion.silver_processor"""

from __future__ import annotations

import signal
import time
from datetime import datetime, timezone

import structlog
from pydantic import ValidationError
from sqlalchemy import select

from config.settings import settings
from db.models import BronzeTelemetry, SilverTelemetry
from db.session import create_tables, get_session
from validation.schemas import TelemetryMessage

logger = structlog.get_logger(__name__)

_RUNNING = True


def _handle_signal(sig: int, _frame: object) -> None:
    global _RUNNING
    logger.info("shutdown_signal_received", signal=sig)
    _RUNNING = False


def _flatten(bronze: BronzeTelemetry, validated: TelemetryMessage) -> SilverTelemetry:
    """Build a SilverTelemetry row from a validated TelemetryMessage."""
    return SilverTelemetry(
        bronze_id=bronze.id,
        processed_at=datetime.now(timezone.utc),
        timestamp=bronze.timestamp,
        mission_id=validated.mission_id,
        vehicle_id=validated.vehicle_id,
        sim_time=validated.sim_time,
        pos_x=validated.position.x,
        pos_y=validated.position.y,
        pos_z=validated.position.z,
        roll=validated.imu.roll,
        pitch=validated.imu.pitch,
        yaw=validated.imu.yaw,
        gyro_x=validated.gyro.x,
        gyro_y=validated.gyro.y,
        gyro_z=validated.gyro.z,
        speed=validated.speed,
        battery=validated.battery,
        motor_power=validated.motor_power,
        lidar_distance=validated.lidar_distance,
        wind_force=validated.wind_force,
        camera_brightness=validated.camera.brightness,
        camera_motion_blur=validated.camera.motion_blur,
        camera_object_count=validated.camera.object_count,
    )


def process_batch() -> tuple[int, int]:
    """Process one batch of unprocessed bronze rows.

    Returns:
        (written, rejected) counts for this batch.
    """
    written = 0
    rejected = 0

    with get_session() as session:
        rows = (
            session.execute(
                select(BronzeTelemetry)
                .where(BronzeTelemetry.processed_to_silver == 0)
                .order_by(BronzeTelemetry.id)
                .limit(100)
            )
            .scalars()
            .all()
        )

        for bronze in rows:
            try:
                validated = TelemetryMessage.model_validate(bronze.raw_payload)
                silver = _flatten(bronze, validated)
                session.add(silver)
                bronze.processed_to_silver = True
                written += 1
            except ValidationError as exc:
                logger.warning(
                    "silver_validation_rejected",
                    bronze_id=bronze.id,
                    errors=exc.error_count(),
                )
                bronze.processed_to_silver = True  # mark done so we don't retry indefinitely
                rejected += 1
            except (ValueError, TypeError, KeyError) as exc:
                logger.error("silver_unexpected_error", bronze_id=bronze.id, error=str(exc))
                rejected += 1

    return written, rejected


def run() -> None:
    """Main processor loop. Polls every SILVER_BATCH_INTERVAL seconds."""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    create_tables()
    logger.info("silver_processor_started", interval=settings.SILVER_BATCH_INTERVAL)

    while _RUNNING:
        written, rejected = process_batch()

        if written or rejected:
            logger.info("silver_batch_done", written=written, rejected=rejected)
        else:
            pass

        time.sleep(settings.SILVER_BATCH_INTERVAL)

    logger.info("silver_processor_stopped")


if __name__ == "__main__":
    run()
