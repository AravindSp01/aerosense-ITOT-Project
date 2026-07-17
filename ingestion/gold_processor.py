"""Gold processor: reads unprocessed silver rows, engineers features via
Polars, and writes GoldTelemetryFeatures rows to Postgres.
Run with: python -m ingestion.gold_processor"""

from __future__ import annotations

import signal
import time
from datetime import datetime, timezone

import structlog
from sqlalchemy import select

from config.settings import settings
from db.models import GoldTelemetryFeatures, SilverTelemetry
from db.session import create_tables, get_session
from features.engineering import engineer_features

logger = structlog.get_logger(__name__)

_RUNNING = True


def _handle_signal(sig: int, _frame: object) -> None:
    global _RUNNING
    logger.info("shutdown_signal_received", signal=sig)
    _RUNNING = False


def _silver_row_to_dict(row: SilverTelemetry) -> dict:
    """Convert a SilverTelemetry ORM row to a plain dict for Polars."""
    return {
        "silver_id": row.id,
        "mission_id": row.mission_id,
        "vehicle_id": row.vehicle_id,
        "sim_time": row.sim_time,
        "battery": row.battery,
        "speed": row.speed,
        "motor_power": row.motor_power,
        "wind_force": row.wind_force,
        "lidar_distance": row.lidar_distance,
        "pos_z": row.pos_z,
        "roll": row.roll,
        "pitch": row.pitch,
    }


def process_batch() -> tuple[int, int]:
    """Process one batch of unprocessed silver rows.

    Returns:
        (written, skipped) counts for this batch.
    """
    written = 0
    skipped = 0

    with get_session() as session:
        rows = (
            session.execute(
                select(SilverTelemetry)
                .where(SilverTelemetry.processed_to_gold == 0)
                .order_by(SilverTelemetry.sim_time)
                .limit(200)
            )
            .scalars()
            .all()
        )

        if not rows:
            return 0, 0

        dicts = [_silver_row_to_dict(r) for r in rows]
        feature_rows = engineer_features(dicts)

        # silver_ids_done = {r.id for r in rows}
        silver_ids_featured = {f["silver_id"] for f in feature_rows}

        for feature in feature_rows:
            gold = GoldTelemetryFeatures(
                silver_id=feature["silver_id"],
                processed_at=datetime.now(timezone.utc),
                mission_id=feature["mission_id"],
                vehicle_id=feature["vehicle_id"],
                sim_time=feature["sim_time"],
                battery=feature["battery"],
                speed=feature["speed"],
                motor_power=feature["motor_power"],
                wind_force=feature["wind_force"],
                lidar_distance=feature["lidar_distance"],
                altitude=feature["altitude"],
                abs_roll=feature["abs_roll"],
                abs_pitch=feature["abs_pitch"],
                battery_mean=feature["battery_mean"],
                speed_mean=feature["speed_mean"],
                altitude_std=feature["altitude_std"],
                motor_power_mean=feature["motor_power_mean"],
                wind_force_max=feature["wind_force_max"],
                risk_level=feature["risk_level"],
            )
            session.add(gold)
            written += 1

        # Mark all fetched silver rows as processed regardless of whether
        # they produced a gold row (avoids reprocessing rows that were
        # excluded due to insufficient window history).
        for row in rows:
            row.processed_to_gold = True
            if row.id not in silver_ids_featured:
                skipped += 1

    return written, skipped


def run() -> None:
    """Main processor loop. Polls every GOLD_BATCH_INTERVAL seconds."""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    create_tables()
    logger.info("gold_processor_started", interval=settings.GOLD_BATCH_INTERVAL)

    while _RUNNING:
        written, skipped = process_batch()

        if written or skipped:
            logger.info("gold_batch_done", written=written, skipped=skipped)
        else:
            pass

        time.sleep(settings.GOLD_BATCH_INTERVAL)

    logger.info("gold_processor_stopped")


if __name__ == "__main__":
    run()
