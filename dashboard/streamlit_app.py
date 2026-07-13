# dashboard/streamlit_app.py
"""AeroSense live dashboard. Reads from Postgres (silver + gold) and calls
the FastAPI /predict endpoint. Never touches Kafka or Webots directly.
Run with: streamlit run dashboard/streamlit_app.py"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
import os
import time
from datetime import datetime
from json import JSONDecodeError

import requests
import streamlit as st
from requests.exceptions import RequestException
from sqlalchemy import desc, select, text

from db.models import GoldTelemetryFeatures, SilverTelemetry
from db.session import get_session
from features.engineering import FEATURE_COLUMNS

API_URL = os.environ.get("API_URL", "http://localhost:8000")

REFRESH_INTERVAL = 10  # seconds
TELEMETRY_HISTORY = 60  # number of readings to show in live charts
ALERTS_LIMIT = 10

RISK_BADGE = {
    "safe": "🟢 SAFE",
    "warning": "🟡 WARNING",
    "critical": "🔴 CRITICAL",
}
RISK_COLOUR = {
    "safe": "green",
    "warning": "orange",
    "critical": "red",
}

# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #


def load_flight_path(mission_id: str) -> list[dict]:
    """Load pos_x, pos_y, pos_z, sim_time from silver for the current mission."""
    with get_session() as session:
        rows = session.execute(
            select(
                SilverTelemetry.pos_x,
                SilverTelemetry.pos_y,
                SilverTelemetry.pos_z,
                SilverTelemetry.sim_time,
            )
            .where(SilverTelemetry.mission_id == mission_id)
            .order_by(SilverTelemetry.sim_time)
        ).all()
    return [
        {"pos_x": r.pos_x, "pos_y": r.pos_y, "pos_z": r.pos_z, "sim_time": r.sim_time} for r in rows
    ]


def load_live_telemetry(mission_id: str, limit: int = TELEMETRY_HISTORY) -> list[dict]:
    """Load last N silver rows for live charts."""
    with get_session() as session:
        rows = session.execute(
            select(
                SilverTelemetry.sim_time,
                SilverTelemetry.battery,
                SilverTelemetry.pos_z,
                SilverTelemetry.speed,
                SilverTelemetry.motor_power,
                SilverTelemetry.wind_force,
                SilverTelemetry.lidar_distance,
            )
            .where(SilverTelemetry.mission_id == mission_id)
            .order_by(desc(SilverTelemetry.sim_time))
            .limit(limit)
        ).all()
    return [
        {
            "sim_time": r.sim_time,
            "battery": r.battery,
            "altitude": r.pos_z,
            "speed": r.speed,
            "motor_power": r.motor_power,
            "wind_force": r.wind_force,
            "lidar_distance": r.lidar_distance,
        }
        for r in reversed(rows)
    ]


def load_latest_gold_row(mission_id: str) -> dict | None:
    """Load the most recent gold feature row for risk prediction."""
    with get_session() as session:
        row = session.execute(
            select(GoldTelemetryFeatures)
            .where(GoldTelemetryFeatures.mission_id == mission_id)
            .order_by(desc(GoldTelemetryFeatures.sim_time))
            .limit(1)
        ).scalar_one_or_none()
    if row is None:
        return None
    return {col: getattr(row, col) for col in FEATURE_COLUMNS}


def load_recent_alerts(mission_id: str, limit: int = ALERTS_LIMIT) -> list[dict]:
    """Load last N warning/critical gold rows."""
    with get_session() as session:
        rows = session.execute(
            select(
                GoldTelemetryFeatures.sim_time,
                GoldTelemetryFeatures.risk_level,
                GoldTelemetryFeatures.battery,
                GoldTelemetryFeatures.wind_force,
                GoldTelemetryFeatures.lidar_distance,
                GoldTelemetryFeatures.altitude,
            )
            .where(
                GoldTelemetryFeatures.mission_id == mission_id,
                GoldTelemetryFeatures.risk_level.in_(["warning", "critical"]),
            )
            .order_by(desc(GoldTelemetryFeatures.sim_time))
            .limit(limit)
        ).all()
    return [
        {
            "sim_time": r.sim_time,
            "risk_level": r.risk_level,
            "battery": r.battery,
            "wind_force": r.wind_force,
            "lidar_distance": r.lidar_distance,
            "altitude": r.altitude,
        }
        for r in rows
    ]


def get_current_mission_id() -> str | None:
    """Return the mission_id of the most recently active mission."""
    with get_session() as session:
        row = session.execute(
            text("SELECT mission_id FROM silver_telemetry ORDER BY id DESC LIMIT 1")
        ).fetchone()
    return row[0] if row else None


def call_predict(features: dict) -> dict | None:
    """Call FastAPI /predict. Returns None on any error."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=features, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except (RequestException, JSONDecodeError):
        pass
    return None


def check_api_health() -> bool:
    """Return True if the FastAPI server is reachable and model is loaded."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2.0)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except (RequestException, JSONDecodeError):
        return False


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AeroSense Dashboard",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 AeroSense — Live Telemetry Dashboard")

mission_id = get_current_mission_id()

if not mission_id:
    st.warning("No telemetry data found. Start the Webots simulation and consumers.")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

st.caption(
    f"Mission: `{mission_id}` · Refreshing every {REFRESH_INTERVAL}s · {datetime.now().strftime('%H:%M:%S')}"
)

# --------------------------------------------------------------------------- #
# Section 1: Flight Path
# --------------------------------------------------------------------------- #

st.subheader("✈️ Flight Path")

path_data = load_flight_path(mission_id)

if path_data:
    current_pos = path_data[-1]

    col_map, col_pos = st.columns([2, 1])

    with col_map:
        # Streamlit's native scatter_chart expects x/y column names.
        import pandas as pd

        path_df = pd.DataFrame(path_data)
        st.scatter_chart(path_df, x="pos_x", y="pos_y", size=20)

    with col_pos:
        st.metric("Current X", f"{current_pos['pos_x']:.1f} m")
        st.metric("Current Y", f"{current_pos['pos_y']:.1f} m")
        st.metric("Altitude", f"{current_pos['pos_z']:.1f} m")
        st.metric("Sim Time", f"{current_pos['sim_time']:.1f} s")
        st.metric("Data Points", len(path_data))
else:
    st.info("Waiting for flight path data...")

# --------------------------------------------------------------------------- #
# Section 2: Live Telemetry Charts
# --------------------------------------------------------------------------- #

st.subheader("📊 Live Telemetry")

telem = load_live_telemetry(mission_id)

if telem:
    import pandas as pd

    telem_df = pd.DataFrame(telem).set_index("sim_time")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Battery % & Altitude**")
        st.line_chart(telem_df[["battery", "altitude"]])
    with col_b:
        st.write("**Speed & Motor Power**")
        st.line_chart(telem_df[["speed", "motor_power"]])
else:
    st.info("Waiting for telemetry data...")

# --------------------------------------------------------------------------- #
# Section 3: Risk Prediction
# --------------------------------------------------------------------------- #

st.subheader("🎯 Risk Prediction")

api_ok = check_api_health()
gold_features = load_latest_gold_row(mission_id)

col_risk, col_conf = st.columns([1, 2])

if not api_ok:
    with col_risk:
        st.error("API offline — start uvicorn api.app:app")
elif gold_features is None:
    with col_risk:
        st.info("Waiting for gold feature data...")
else:
    prediction = call_predict(gold_features)
    if prediction:
        risk = prediction["risk_level"]
        confidence = prediction["confidence"]
        badge = RISK_BADGE.get(risk, risk.upper())
        colour = RISK_COLOUR.get(risk, "gray")

        with col_risk:
            st.markdown(
                f"<h1 style='color:{colour};'>{badge}</h1>",
                unsafe_allow_html=True,
            )
            st.caption(f"Model version: {prediction['model_version']}")

        with col_conf:
            st.write("**Confidence**")
            st.progress(confidence)
            st.caption(f"{confidence:.1%}")
            st.write("**Probabilities**")
            for level, prob in prediction["probabilities"].items():
                badge_small = RISK_BADGE.get(level, level)
                st.progress(prob, text=f"{badge_small}: {prob:.1%}")
    else:
        with col_risk:
            st.warning("Prediction call failed.")

# --------------------------------------------------------------------------- #
# Section 4: Sensor Quality
# --------------------------------------------------------------------------- #

st.subheader("🔬 Sensor Quality")

if telem:
    latest = telem[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        lidar = latest["lidar_distance"]
        st.metric("LiDAR Distance", f"{lidar:.1f} m")
        st.progress(min(lidar / 50.0, 1.0), text="clearance" if lidar > 5 else "⚠️ low clearance")

    with col2:
        wind = latest["wind_force"]
        st.metric("Wind Force", f"{wind:.2f} N")
        st.progress(min(wind / 15.0, 1.0), text="calm" if wind < 8 else "⚠️ high wind")

    with col3:
        mp = latest["motor_power"]
        st.metric("Motor Power", f"{mp:.1f}")
        st.progress(min(mp / 150.0, 1.0))
else:
    st.info("Waiting for sensor data...")

# --------------------------------------------------------------------------- #
# Section 5: Recent Alerts
# --------------------------------------------------------------------------- #

st.subheader("🚨 Recent Alerts")

alerts = load_recent_alerts(mission_id)

if alerts:
    import pandas as pd

    alerts_df = pd.DataFrame(alerts)
    alerts_df.columns = [
        "Sim Time",
        "Risk Level",
        "Battery %",
        "Wind (N)",
        "LiDAR (m)",
        "Altitude (m)",
    ]
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.success("No warnings or critical events recorded.")

# --------------------------------------------------------------------------- #
# Auto-refresh
# --------------------------------------------------------------------------- #

time.sleep(REFRESH_INTERVAL)
st.rerun()
