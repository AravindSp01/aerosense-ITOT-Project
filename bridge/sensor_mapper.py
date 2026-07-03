# bridge/sensor_mapper.py
"""Translate raw Webots sensor readings into the canonical telemetry dict
(matching validation.schemas.TelemetryMessage). Pure functions only -- no
Kafka or Webots imports -- so this is fully testable without a running
Webots instance."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

# Threshold above the per-pixel mean used to count an "object" blob in the
# brightness-difference heuristic below. Not in config/settings.py because
# it is an internal implementation detail of the NumPy-only camera-metadata
# approximation, not an operationally tunable value.
_OBJECT_INTENSITY_DELTA = 40.0
_MIN_OBJECT_PIXELS = 25


def _camera_metadata(
    camera_image: np.ndarray | None,
    previous_image: np.ndarray | None,
) -> dict[str, float | int]:
    """Compute lightweight camera metadata without streaming raw pixels.

    NOTE: the spec calls for OpenCV-based contour counting for
    `object_count`. OpenCV is not in the approved tech stack, so this uses
    a NumPy-only approximation instead: pixels significantly darker/lighter
    than the image mean are treated as foreground, and contiguous
    foreground regions (4-connectivity, flood-fill) are counted as objects.
    This is coarser than real contour detection -- flag if this needs to
    be more accurate later.
    """
    if camera_image is None:
        return {"brightness": 0.0, "motion_blur": 0.0, "object_count": 0}

    grayscale = camera_image.astype(np.float64)
    if grayscale.ndim == 3:
        grayscale = grayscale.mean(axis=-1)

    brightness = float(np.clip(grayscale.mean() / 255.0, 0.0, 1.0))

    motion_blur = 0.0
    if previous_image is not None and previous_image.shape == camera_image.shape:
        prev_gray = previous_image.astype(np.float64)
        if prev_gray.ndim == 3:
            prev_gray = prev_gray.mean(axis=-1)
        diff = np.abs(grayscale - prev_gray)
        motion_blur = float(np.clip(diff.mean() / 255.0, 0.0, 1.0))

    object_count = _count_object_blobs(grayscale)

    return {
        "brightness": brightness,
        "motion_blur": motion_blur,
        "object_count": object_count,
    }


def _count_object_blobs(grayscale: np.ndarray) -> int:
    """Count contiguous foreground regions via a simple flood-fill.

    Foreground = pixels deviating from the image mean by more than
    `_OBJECT_INTENSITY_DELTA`. Regions smaller than `_MIN_OBJECT_PIXELS`
    are treated as noise and ignored.
    """
    mean_intensity = grayscale.mean()
    foreground = np.abs(grayscale - mean_intensity) > _OBJECT_INTENSITY_DELTA
    visited = np.zeros_like(foreground, dtype=bool)
    height, width = foreground.shape
    object_count = 0

    for row in range(height):
        for col in range(width):
            if not foreground[row, col] or visited[row, col]:
                continue
            # Flood-fill this region (iterative, 4-connected).
            stack = [(row, col)]
            visited[row, col] = True
            region_size = 0
            while stack:
                r, c = stack.pop()
                region_size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and foreground[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if region_size >= _MIN_OBJECT_PIXELS:
                object_count += 1

    return object_count


def map_to_telemetry(
    gps: tuple[float, float, float],
    imu: tuple[float, float, float],
    gyro: tuple[float, float, float],
    compass: tuple[float, float, float],
    camera_image: np.ndarray | None,
    lidar_distance: float,
    battery: float,
    motor_power: float,
    sim_time: float,
    mission_id: str,
    vehicle_id: str,
    position: tuple[float, float, float],
    speed: float,
    wind_force: float,
    previous_camera_image: np.ndarray | None = None,
) -> dict:
    """Map raw Webots device readings to a TelemetryMessage-shaped dict.

    Args:
        gps: (lat, lon, alt) -- Webots GPS device output reinterpreted as a
            geodetic fix (Webots GPS itself returns world x/y/z; this
            controller's caller is responsible for any coordinate
            conversion before calling this function).
        imu: (roll, pitch, yaw) in radians, as returned by InertialUnit.
        gyro: (x, y, z) angular velocity in radians/second.
        compass: (x, y, z) compass vector. Currently unused by the schema
            directly but accepted for forward compatibility (e.g. heading
            derivation in a future phase).
        camera_image: current camera frame as an (H, W) or (H, W, 3) array,
            or None if no camera is mounted/available.
        lidar_distance: distance-sensor reading in meters, used as a LiDAR
            proxy per the spec.
        battery: battery percentage, 0-100.
        motor_power: mean motor power draw.
        sim_time: simulated time in seconds since controller start.
        mission_id: identifier for the current mission/run.
        vehicle_id: identifier for this drone.
        position: (x, y, z) world position in meters.
        speed: current speed in meters/second.
        wind_force: magnitude of the currently applied wind disturbance.
        previous_camera_image: the prior tick's camera frame, used to
            compute motion_blur. Pass None on the first tick.

    Returns:
        A dict matching the shape of validation.schemas.TelemetryMessage,
        with `imu.roll/pitch/yaw` converted from radians to degrees to
        match the documented canonical schema.
    """
    roll_rad, pitch_rad, yaw_rad = imu
    pos_x, pos_y, pos_z = position
    gps_lat, gps_lon, gps_alt = gps
    gyro_x, gyro_y, gyro_z = gyro

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mission_id": mission_id,
        "vehicle_id": vehicle_id,
        "sim_time": sim_time,
        "position": {"x": pos_x, "y": pos_y, "z": pos_z},
        "gps": {"lat": gps_lat, "lon": gps_lon, "alt": gps_alt},
        "imu": {
            "roll": np.degrees(roll_rad),
            "pitch": np.degrees(pitch_rad),
            "yaw": np.degrees(yaw_rad),
        },
        "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
        "speed": speed,
        "battery": battery,
        "motor_power": motor_power,
        "lidar_distance": lidar_distance,
        "camera": _camera_metadata(camera_image, previous_camera_image),
        "wind_force": wind_force,
    }
