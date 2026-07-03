"""Wrapper for Apache Kafka production to stream drone telemetry data safely."""

import json

import structlog
from confluent_kafka import KafkaException, KafkaError, Message, Producer
from builtins import BufferError

from config.settings import Settings

logger = structlog.get_logger()


class KafkaPublisher:
    """Manages connection and publishing lifecycle for streaming vehicle telemetry to Apache Kafka."""

    def __init__(self, settings: Settings) -> None:
        """Initializes the confluent-kafka Producer using global platform settings."""
        self.settings = settings
        conf = {
            "bootstrap.servers": self.settings.KAFKA_BOOTSTRAP,
            "client.id": "aerosense-bridge-publisher",
            "acks": "1",
            "message.timeout.ms": "5000",
        }
        self.producer = Producer(conf)

    def _delivery_report(self, err: KafkaError | None, msg: Message) -> None:
        """Callback triggered upon message acknowledgement or delivery failure."""
        if err is not None:
            logger.error(
                "Kafka message delivery failed",
                error=str(err),
                topic=msg.topic(),
            )
        else:
            logger.debug(
                "Kafka message delivered successfully",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    def publish(self, telemetry: dict[str, object]) -> None:
        """Publishes a validated telemetry dictionary to the Kafka cluster using vehicle_id as the key.

        This method is designed to be best-effort; it will swallow and log internal exceptions
        to prevent Kafka or network hiccups from crashing the core simulation loop.
        """
        try:
            vehicle_id: str = str(telemetry.get("vehicle_id") or "unknown-vehicle")
            payload: bytes = json.dumps(telemetry).encode("utf-8")
            key_bytes: bytes = vehicle_id.encode("utf-8")

            self.producer.produce(
                topic=self.settings.KAFKA_TOPIC,
                key=key_bytes,
                value=payload,
                callback=self._delivery_report,
            )
            # Serve delivery callback queue immediately (non-blocking)
            self.producer.poll(0)

        except (BufferError, KafkaException) as exc:
            logger.error(
                "Encountered Kafka error during telemetry publication",
                error=str(exc),
                vehicle_id=telemetry.get("vehicle_id"),
                sim_time=telemetry.get("sim_time"),
            )

    def flush(self, timeout: float = 1.0) -> int:
        """Synchronously flushes any buffered messages to guarantee delivery before shutdown."""
        return self.producer.flush(timeout)
