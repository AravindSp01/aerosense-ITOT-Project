# AeroSense

> A autonomous sensor intelligence project: simulated drone telemetry streams through Kafka into a medallion data architecture, trains an ML risk model, and serves live predictions through a FastAPI endpoint and Streamlit dashboard.

---

## Architecture Overview

- **Webots** (local) simulates a Mavic 2 Pro drone flying autonomously through a physics environment
- **Bridge layer** reads onboard sensor data and publishes structured telemetry to **Kafka every second
- **Bronze consumer** lands raw Kafka messages into PostgreSQL as-is
- **Silver processor** validates and flattens bronze records into typed columns
- **Gold processor** applies feature engineering (Polars rolling windows) and attaches a risk label
- **MLflow** tracks training runs and stores the registered model
- **FastAPI** loads the best registered model and serves live risk predictions
- **Streamlit** pulls from PostgreSQL and the API to display the flight path, telemetry, and risk badge
- **Training pipeline** (`--profile pipeline`) runs data-processor → trainer → evaluator sequentially on demand

---

## Tech Stack

Simulation : Webots R2025a
Language : Python 3.10
Streaming : Apache Kafka 3.x (KRaft mode)
Validation : Pydantic v2
Feature Engineering : Polars
Database : PostgreSQL 16
DB Access : SQLAlchemy 2.0 + psycopg3
ML : scikit-learn (RandomForest), XGBoost
Experiment Tracking : MLflow 3.x
API : FastAPI + Uvicorn
Dashboard : Streamlit
Containerisation : Docker + Docker Compose
Testing : pytest + pytest-cov
Linting : ruff + mypy
CI : GitHub Actions

---

## Repository Structure

```
aerosense/
├── config/settings.py               # env vars via Pydantic BaseSettings
├── webots/
│   ├── worlds/aerosense.wbt         # webots world: terrain, obstacles, drone created usign VRML
│   └── controllers/drone_controller/
│       ├── drone_controller.py      # PID flight controller + bridge calls
│       └── runtime.ini              # tells Webots to use project venv
├── bridge/
│   ├── sensor_mapper.py             # raw Webots readings → TelemetryMessage dict
│   └── kafka_publisher.py           # wraps confluent_kafka.Producer
├── ingestion/
│   ├── bronze_consumer.py           # kafka → bronze_telemetry
│   ├── silver_processor.py          # bronze → silver_telemetry
│   └── gold_processor.py            # silver → gold_telemetry_features
├── validation/schemas.py            # pydantic TelemetryMessage + field validators
├── features/engineering.py          # polars rolling stats, risk labelling
├── db/
│   ├── models.py                    # ORM models: Bronze, Silver, Gold tables
│   └── session.py                   # engine, session factory, create_tables() using sqlalchemy
├── models/
│   ├── train.py                     # RandomForest + XGBoost, logged to MLflow
│   ├── evaluate.py                  # metrics: accuracy, F1, ROC-AUC, confusion matrix
│   └── inference.py                 # loads registered model, exposes predict()
├── api/app.py                       # FastAPI: /predict, /health, /metrics
├── dashboard/streamlit_app.py       # live telemetry + predictions + flight path
├── tests/
├── Dockerfile                       # multi-stage: base → data-pipeline / ml-api / dashboard
├── docker-compose.yml
├── requirements-*.txt
└── .env.example
```

---

## Prerequisites

- [Webots R2025a](https://cyberbotics.com/#download) — runs locally, outside Docker - better than static OT system telemetry data
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Setup

### 1. Clone and configure environment

```bash
git clone https://github.com/AravindSp01/aerosense-ITOT-Project.git
cd aerosense
cp .env.example .env
```

The defaults in `.env` work out of the box with `docker-compose.yml`. Adjust risk thresholds if needed:

```env
BATTERY_CRITICAL=20.0
BATTERY_WARNING=35.0
LIDAR_CRITICAL=2.0
LIDAR_WARNING=5.0
WIND_WARNING=8.0
```

### 2. Configure Webots to use your venv

Open `webots/controllers/drone_controller/runtime.ini` and set the paths for your machine:

```ini
# Windows
[environment variables with paths]
PYTHONPATH = C:\path\to\aerosense
[python]
COMMAND = C:\path\to\aerosense\venv\Scripts\python.exe

# macOS / Linux
[environment variables with paths]
PYTHONPATH = /path/to/aerosense
[python]
COMMAND = /path/to/aerosense/venv/bin/python
```

> **Note:** A local venv is only needed so Webots can run the drone controller. All other services run in Docker.

### 3. Start Docker services

```bash
docker compose up -d
```

First run takes 3–5 minutes to build the multi-stage image. Verify with `docker compose ps`.

### 4. Run the simulation

```bash
webots webots/worlds/aerosense.wbt
```

Press **Play (▶)**. Telemetry will begin streaming to Kafka immediately. Webots logs will help to verify this.

### 5. Train the model

Let Webots run for at least 5–10 minutes, then:

```bash
docker compose --profile pipeline run --rm trainer
docker compose restart api
```

### 6. Open the dashboards

Streamlit : http://localhost:8501
FastAPI docs : http://localhost:8000/docs
Kafka UI : http://localhost:8080
MLflow : http://localhost:5000

---

## Data Pipeline Detail

### Telemetry Schema

Every Kafka message conforms to `validation/schemas.py: TelemetryMessage`:


### Medallion Layers

**Bronze** — raw messages as received from Kafka, no changes. Kafka offset and partition stored for auditability.

**Silver** — validated against `TelemetryMessage`, flat typed columns, no JSON blobs. Invalid records are logged and skipped.

**Gold** — feature-engineered with Polars rolling windows over silver. Contains the 13 model input features plus a `risk_level` label.


## ML Model

Two classifiers are trained and compared: **Random Forest** (200 estimators, max depth 8) and **XGBoost** (200 estimators, max depth 6, lr 0.05). The higher macro F1 model is registered in MLflow as `aerosense-risk` and loaded by the API.

> Early missions produce mostly `safe` rows. Run longer simulations or lower `BATTERY_WARNING` in `.env` for a more balanced training set.

---

## Dockerfile

Single multi-stage build — all stages share a common base with system deps and `requirements-common.txt`.

`base` : common - all
`data-pipeline` : ingestion - `bronze-consumer`, `data-processor`
`ml-api` : ml + api (merged) - `api`, `trainer`, `evaluator`
`dashboard` : streamlit - `streamlit`

---

```bash
docker compose logs -f bronze-consumer
docker compose logs -f api

# Row counts per layer
docker exec -it aerosense-postgres psql -U postgres -d aerosense \
  -c "SELECT (SELECT COUNT(*) FROM bronze_telemetry) AS bronze,
             (SELECT COUNT(*) FROM silver_telemetry) AS silver,
             (SELECT COUNT(*) FROM gold_telemetry_features) AS gold;"

docker compose --profile pipeline up                # full pipeline
docker compose --profile pipeline run --rm trainer  # training only
docker compose restart api                          # reload model
docker compose down                                 # stop
docker compose down -v                              # stop + wipe volumes
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov --cov-report=term-missing
```

All tests are fully mocked — no Webots, Kafka, or database needed.

---

## Linting & CI

```bash
ruff check . --fix && ruff format .
mypy bridge/ ingestion/ validation/ config/ db/ models/ api/ features/
```

GitHub Actions runs on every push to `main`/`develop`: linting → type checking → pytest with coverage. Webots is excluded from CI (requires a display).

---

## Known Limitations

- **Navigation:** The drone follows a corrective arc rather than a straight line due to wind + proportional yaw control. Acceptable for data generation. 
- **LiDAR proxy:** The Mavic 2 Pro PROTO has no native LiDAR; `lidar_distance` is a fixed proxy value.
- **Class imbalance:** Default thresholds skew heavily toward `safe`. Longer runs or lower thresholds produce more `warning`/`critical` examples.

**Potential future additions:** sensor fusion, concept drift detection, automated retraining, fault injection, multi-drone support.

---
