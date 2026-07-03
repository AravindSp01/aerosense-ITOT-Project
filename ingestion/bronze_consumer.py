# ingestion/bronze_consumer.py
"""Kafka consumer: reads aerosense.telemetry, validates each message against
TelemetryMessage, and writes a BronzeTelemetry row to Postgres.
Run with: python -m ingestion.bronze_consumer"""

from __future__ import annotations

import json
import signal
from datetime import datetime, timezone

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from db.models import BronzeTelemetry
from db.session import create_tables, get_session
from validation.schemas import TelemetryMessage

logger = structlog.get_logger(__name__)

_RUNNING = True


def _handle_signal(sig: int, _frame: object) -> None:
    global _RUNNING
    logger.info("shutdown_signal_received", signal=sig)
    _RUNNING = False


def _make_record(
    msg_value: dict,
    offset: int,
    partition: int,
) -> BronzeTelemetry:
    """Build a BronzeTelemetry ORM row from a validated message dict."""
    return BronzeTelemetry(
        ingested_at=datetime.now(timezone.utc),
        kafka_offset=offset,
        kafka_partition=partition,
        timestamp=msg_value["timestamp"],
        mission_id=msg_value["mission_id"],
        vehicle_id=msg_value["vehicle_id"],
        sim_time=msg_value["sim_time"],
        battery=msg_value["battery"],
        speed=msg_value["speed"],
        motor_power=msg_value["motor_power"],
        wind_force=msg_value["wind_force"],
        lidar_distance=msg_value["lidar_distance"],
        raw_payload=msg_value,
    )


def run() -> None:
    """Main consumer loop. Blocks until SIGINT/SIGTERM."""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    create_tables()

    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP,
            "group.id": settings.KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # manual commit after successful DB write
        }
    )
    consumer.subscribe([settings.KAFKA_TOPIC])
    logger.info("bronze_consumer_started", topic=settings.KAFKA_TOPIC)

    try:
        while _RUNNING:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                err = msg.error()
                if err is not None:
                    code = err.code()
                    if code == KafkaError._PARTITION_EOF:
                        continue
                    if code == KafkaError.UNKNOWN_TOPIC_OR_PART:
                        logger.warning("topic_not_ready_retrying", topic=settings.KAFKA_TOPIC)
                        continue
                    raise KafkaException(err)

            raw = msg.value()
            if raw is None:
                continue

            offset = msg.offset()
            partition = msg.partition()

            try:
                payload = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
            except json.JSONDecodeError as exc:
                logger.error("invalid_json", offset=offset, error=str(exc))
                consumer.commit(message=msg)
                continue

            try:
                TelemetryMessage.model_validate(payload)
            except ValidationError as exc:
                logger.error("schema_validation_failed", offset=offset, errors=int(exc.error_count()))
                consumer.commit(message=msg)
                continue

            try:
                if offset is None or partition is None:
                    continue
                record = _make_record(payload, offset, partition)
                with get_session() as session:
                    session.add(record)
                consumer.commit(message=msg)
                logger.info(
                    "bronze_record_written",
                    mission_id=payload["mission_id"],
                    sim_time=payload["sim_time"],
                    offset=offset,
                )
            except SQLAlchemyError as exc:
                logger.error("db_write_failed", offset=offset, error=str(exc))
                # Do NOT commit -- message will be reprocessed on restart.

    finally:
        consumer.close()
        logger.info("bronze_consumer_stopped")


if __name__ == "__main__":
    run()
