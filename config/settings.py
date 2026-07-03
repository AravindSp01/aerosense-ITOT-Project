# config/settings.py
"""Single source of truth for all configuration values, loaded from the
environment (and .env) via Pydantic BaseSettings. No module elsewhere
should hardcode a value that belongs here."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All AeroSense configuration, overridable via environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Kafka
    KAFKA_BOOTSTRAP: str = "localhost:9092"
    KAFKA_TOPIC: str = "aerosense.telemetry"
    KAFKA_GROUP_ID: str = "aerosense-consumer"

    # Postgres
    DATABASE_URL: str = (
        "postgresql+psycopg://postgres:postgres_secure_password@localhost:5432/aerosense"
    )

    # Workers
    SILVER_BATCH_INTERVAL: float = 5.0  # seconds between silver processing runs
    GOLD_BATCH_INTERVAL: float = 10.0  # seconds between gold processing runs
    FEATURE_WINDOW: int = 30  # rolling window size (readings)

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT: str = "aerosense-risk"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Risk thresholds (single source of truth, used in features/engineering.py)
    BATTERY_CRITICAL: float = 20.0
    BATTERY_WARNING: float = 35.0
    LIDAR_CRITICAL: float = 2.0
    LIDAR_WARNING: float = 5.0
    WIND_WARNING: float = 8.0


settings = Settings()
