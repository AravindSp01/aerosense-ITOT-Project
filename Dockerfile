# ==========================================
# STAGE 1: Global Base (System Dependencies)
# ==========================================
FROM python:3.10-slim AS base
WORKDIR /app
ENV PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=1000 \
    PYTHONUNBUFFERED=1

# Install shared system dependencies across all layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel

# Pre-bake common dependencies to build a highly efficient base cache layer
COPY requirements-common.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-common.txt

# ==========================================
# STAGE 2: Data Pipeline (Bronze & Processing)
# ==========================================
FROM base AS data-pipeline
COPY requirements-ingestion.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-ingestion.txt
COPY . .

# ==========================================
# STAGE 3: ML & Core API (Unified Heavy Layers)
# ==========================================
FROM base AS ml-api
ENV GIT_PYTHON_REFRESH=quiet
COPY requirements-ml.txt .
COPY requirements-api.txt .
# Merging these installs optimizes cache layers since they share numpy, scikit-learn, mlflow, and polars
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-ml.txt -r requirements-api.txt
COPY . .

# ==========================================
# STAGE 4: Streamlit UI Frontend
# ==========================================
FROM base AS dashboard
COPY requirements-streamlit.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-streamlit.txt
COPY . .
