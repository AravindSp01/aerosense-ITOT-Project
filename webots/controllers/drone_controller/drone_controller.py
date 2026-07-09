# webots/controllers/drone_controller/drone_controller.py
"""Webots controller for the Mavic2Pro: PID-stabilized flight from start
to target, wind disturbance, battery model, and Kafka telemetry via the
bridge layer. Requires PYTHONPATH to include the aerosense project root
so that bridge/ and config/ are importable.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys

from controller import Supervisor  # type: ignore[import-not-found]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from bridge.kafka_publisher import KafkaPublisher
    from bridge.sensor_mapper import map_to_telemetry
    from config.settings import settings as _settings

    _BRIDGE_AVAILABLE = True
except ImportError as _err:
    _BRIDGE_AVAILABLE = False
    _BRIDGE_IMPORT_ERROR = str(_err)

MISSION_ID = os.environ.get("AEROSENSE_MISSION_ID", "m-001")
VEHICLE_ID = os.environ.get("AEROSENSE_VEHICLE_ID", "drone-alpha")

##--##
# Flight constants -- single source of truth for this file only.
##--##

K_VERTICAL_THRUST = 68.5  # from Cyberbotics official sample -- do not change # Power required to hover
K_VERTICAL_OFFSET = 0.6 # To compensate for altitude bias
K_VERTICAL_P = 3.0 # To compensate for altitude gain
K_ROLL_P = 50.0 # LR Stabilisation
K_PITCH_P = 30.0 # FB Stabilization
K_YAW_P = 1.0 # Heading correction
K_YAW_D = 1.2  # yaw-rate damping -- prevents circling

TARGET_X = 150.0
TARGET_Y = 150.0
CRUISE_ALTITUDE = 20.0
ARRIVAL_RADIUS = 5.0
TELEMETRY_PERIOD = 1.0  # seconds of sim time between telemetry ticks

MAX_PITCH_DISTURBANCE = -4.0 # Artificial forward tilt bias when moving

WIND_BASE_N = 0.6 # const wind
WIND_GUST_N = 1.8 # sudden gust sim
WIND_PERIOD_S = 6.0 #cycle repeat

BATTERY_CAPACITY = 100.0
BATTERY_DRAIN_RATE = 0.0006  # percent per motor-unit per second

LIDAR_DEFAULT = 30.0  # proxy value until a real distance sensor is added

# Lidar can be implemented in the VRML file and it provides a few options like "Range Image" and "Point Cloud (Adv)"
# Point could can help with 3D perception, obstacle mapping etc.
# Not integrated here since I'm running this on a laptop to keep cost $0


##--##
# Structured stdlib logger (structlog lives in the bridge layer, not here)
# Structuring the logs helps to easily ingest data to kafka, elasticsearch etc, and with the monitoring

# The drone controller uses a lightweight built-in JSON logger, while advanced structured logging (structlog) 
# is reserved for the external telemetry/bridge system to avoid coupling simulation logic with 
# infrastructure dependencies. 
##--##


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if hasattr(record, "extra_fields"): # If the log call included extra structured data, it gets merged in.
            payload.update(record.extra_fields)
        return json.dumps(payload)


def _get_logger() -> logging.Logger:
    log = logging.getLogger("aerosense.drone_controller")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_JsonFormatter())
        log.addHandler(h)
    return log


def _log(log: logging.Logger, msg: str, **fields) -> None:
    log.info(msg, extra={"extra_fields": fields})


##--##
# Pure flight helpers
##--##


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _vertical_input(target_alt: float, alt: float) -> float:
    clamped = _clamp(target_alt - alt + K_VERTICAL_OFFSET, -1.0, 1.0)
    return K_VERTICAL_P * (clamped**3.0)


def _battery_drain(mean_power: float, dt: float, drained: float) -> float:
    return min(BATTERY_CAPACITY, drained + mean_power * BATTERY_DRAIN_RATE * dt)


def _wind_force(sim_time: float, rng: random.Random) -> tuple[float, float, float]:
    phase = (2 * math.pi * sim_time) / WIND_PERIOD_S
    magnitude = WIND_BASE_N + WIND_GUST_N * max(0.0, math.sin(phase))
    angle = phase * 0.5 + rng.uniform(-0.2, 0.2)
    return (magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0)


##--##
# Main
##--##


def main() -> None:
    log = _get_logger()
    rng = random.Random(7)

    # Initialise bridge/Kafka -- failure is non-fatal, simulation continues.
    publisher = None
    if _BRIDGE_AVAILABLE:
        try:
            publisher = KafkaPublisher(settings=_settings)
        except Exception as exc:
            _log(log, "kafka_init_failed", error=str(exc))
    else:
        _log(
            log,
            "bridge_import_failed",
            error=_BRIDGE_IMPORT_ERROR,
            note="Set PYTHONPATH to the aerosense project root.",
        )

    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())

    # Devices
    gps = robot.getDevice("gps")
    gps.enable(ts)
    imu = robot.getDevice("inertial unit")
    imu.enable(ts)
    gyro = robot.getDevice("gyro")
    gyro.enable(ts)
    compass = robot.getDevice("compass")
    compass.enable(ts)

    camera = robot.getDevice("camera")
    if camera is not None:
        camera.enable(ts)
    else:
        _log(log, "camera_unavailable", note="Restart Webots to clear proto cache.")

    fl = robot.getDevice("front left propeller")
    fr = robot.getDevice("front right propeller")
    rl = robot.getDevice("rear left propeller")
    rr = robot.getDevice("rear right propeller")
    for m in [fl, fr, rl, rr]:
        m.setPosition(float("inf"))
        m.setVelocity(1.0)

    self_node = robot.getSelf()

    battery_drained = 0.0
    last_telem_time = 0.0
    last_position: tuple[float, float, float] | None = None
    prev_camera_image = None

    _log(
        log,
        "controller_started",
        target_x=TARGET_X,
        target_y=TARGET_Y,
        cruise_altitude=CRUISE_ALTITUDE,
        mission_id=MISSION_ID,
        vehicle_id=VEHICLE_ID,
        bridge=int(_BRIDGE_AVAILABLE),
    )

    while robot.step(ts) != -1:
        if robot.getTime() > 1.0:
            break

    while robot.step(ts) != -1:
        sim_time = robot.getTime()
        dt = ts / 1000.0

        # --- Sensors --------------------------------------------------- #
        pos_x, pos_y, pos_z = gps.getValues()  # Z is altitude in Webots
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()
        compass_vec = tuple(compass.getValues())

        # --- Wind ------------------------------------------------------ #
        fx, fy, fz = _wind_force(sim_time, rng)
        self_node.addForce([fx, fy, fz], False)
        wind_mag = math.hypot(fx, fy)

        # --- Navigation ------------------------------------------------ #
        dx = TARGET_X - pos_x
        dy = TARGET_Y - pos_y
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(desired_heading - yaw), math.cos(desired_heading - yaw))

        speed_factor = _clamp(distance / 5.0, 0.0, 1.0)
        roll_disturbance = _clamp(-heading_error * 0.5, -1.0, 1.0)
        pitch_disturbance = _clamp(
            MAX_PITCH_DISTURBANCE * speed_factor * math.cos(heading_error), -2.0, 0.0
        )
        yaw_disturbance = _clamp(heading_error * K_YAW_P - K_YAW_D * yaw_rate, -1.3, 1.3)

        # --- Stabilization (official Cyberbotics mixing equations) ------ #
        roll_input = K_ROLL_P * _clamp(roll, -1.0, 1.0) + roll_rate + roll_disturbance
        pitch_input = K_PITCH_P * _clamp(pitch, -1.0, 1.0) + pitch_rate + pitch_disturbance
        yaw_input = yaw_disturbance
        vert_input = _vertical_input(CRUISE_ALTITUDE, pos_z)

        fl_pwr = K_VERTICAL_THRUST + vert_input - roll_input + pitch_input - yaw_input
        fr_pwr = K_VERTICAL_THRUST + vert_input + roll_input + pitch_input + yaw_input
        rl_pwr = K_VERTICAL_THRUST + vert_input - roll_input - pitch_input + yaw_input
        rr_pwr = K_VERTICAL_THRUST + vert_input + roll_input - pitch_input - yaw_input

        fl.setVelocity(fl_pwr)
        fr.setVelocity(-fr_pwr)
        rl.setVelocity(-rl_pwr)
        rr.setVelocity(rr_pwr)

        mean_power = (abs(fl_pwr) + abs(fr_pwr) + abs(rl_pwr) + abs(rr_pwr)) / 4.0
        battery_drained = _battery_drain(mean_power, dt, battery_drained)
        battery_pct = BATTERY_CAPACITY - battery_drained

        # --- Telemetry tick -------------------------------------------- #
        if sim_time - last_telem_time >= TELEMETRY_PERIOD:
            last_telem_time = sim_time

            speed = 0.0
            if last_position is not None:
                speed = math.dist((pos_x, pos_y, pos_z), last_position) / TELEMETRY_PERIOD
            last_position = (pos_x, pos_y, pos_z)

            camera_image = None
            if camera is not None:
                try:
                    import numpy as np

                    raw = camera.getImage()
                    w, h = camera.getWidth(), camera.getHeight()
                    camera_image = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
                except Exception:
                    pass

            if publisher is not None:
                try:
                    telemetry = map_to_telemetry(
                        gps=(pos_x, pos_y, pos_z),
                        imu=(roll, pitch, yaw),
                        gyro=(roll_rate, pitch_rate, yaw_rate),
                        compass=compass_vec,
                        camera_image=camera_image,
                        lidar_distance=LIDAR_DEFAULT,
                        battery=battery_pct,
                        motor_power=mean_power,
                        sim_time=sim_time,
                        mission_id=MISSION_ID,
                        vehicle_id=VEHICLE_ID,
                        position=(pos_x, pos_y, pos_z),
                        speed=speed,
                        wind_force=wind_mag,
                        previous_camera_image=prev_camera_image,
                    )
                    publisher.publish(telemetry)
                    prev_camera_image = camera_image
                except Exception as exc:
                    _log(log, "telemetry_error", error=str(exc))

            _log(
                log,
                "telemetry_tick",
                sim_time=round(sim_time, 2),
                pos_x=round(pos_x, 2),
                pos_y=round(pos_y, 2),
                pos_z=round(pos_z, 2),
                speed=round(speed, 2),
                battery_pct=round(battery_pct, 2),
                motor_power=round(mean_power, 2),
                distance=round(distance, 2),
                wind=round(wind_mag, 3),
            )

        # --- Termination ----------------------------------------------- #
        if distance < ARRIVAL_RADIUS and battery_pct > 0:
            _log(
                log,
                "target_reached",
                sim_time=round(sim_time, 2),
                battery_pct=round(battery_pct, 2),
            )
            if publisher is not None:
                publisher.flush()
            break

        if battery_pct <= 0:
            _log(log, "battery_depleted", sim_time=round(sim_time, 2))
            if publisher is not None:
                publisher.flush()
            break


if __name__ == "__main__":
    main()
