"""Per-mode motion behaviours: direct teleoperation and obstacle-assisted driving."""
import math
import numpy as np
from .avoidance import potential_field, scan_to_points


def lidar_angles(fov, resolution):
    """Robot-frame ray angles of a Webots Lidar range image.

    Webots orders rays from left to right: index 0 is at +fov/2 and the angle
    decreases with the index. A full-circle lidar spaces its n rays by fov/n
    (the first and last ray would otherwise coincide), others by fov/(n-1).
    """
    full_circle = abs(fov - 2 * math.pi) < 1e-3
    step = fov / resolution if full_circle or resolution < 2 else fov / (resolution - 1)
    return fov / 2 - step * np.arange(resolution)


def ray_spacing(angles):
    """Angular step between consecutive rays (rad)."""
    angles = np.asarray(angles, dtype=float)
    if angles.size < 2:
        return 2 * math.pi
    return float(np.median(np.abs((np.diff(angles) + np.pi) % (2 * np.pi) - np.pi)))


def sector_min(angles, ranges, center, half_width):
    """Closest finite return within +-half_width of center (robot frame); inf if none."""
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    offset = np.abs((angles - center + np.pi) % (2 * np.pi) - np.pi)
    selected = ranges[(offset <= half_width) & np.isfinite(ranges)]
    return float(selected.min()) if selected.size else math.inf


def clearance_scale(clearance, stop_distance, slow_distance):
    """1 when clearance >= slow_distance, falling linearly to 0 at stop_distance."""
    if slow_distance <= stop_distance:
        return 1.0 if clearance > stop_distance else 0.0
    return min(max((clearance - stop_distance) / (slow_distance - stop_distance), 0.0), 1.0)


def teleop_velocity(command, speed, steering, max_linear, max_angular, turn_fraction):
    """Gesture -> body velocity (v, omega) with no safety logic: the TELEOP baseline."""
    if command == "FORWARD":
        v = speed * max_linear
    elif command == "BACKWARD":
        v = -speed * max_linear
    elif command in ("LEFT", "RIGHT"):
        v = 0.0
    else:
        return 0.0, 0.0
    omega = steering * max_angular
    if command == "LEFT":
        omega = max(abs(omega), turn_fraction * max_angular)
    elif command == "RIGHT":
        omega = -max(abs(omega), turn_fraction * max_angular)
    return v, omega


def assisted_velocity(command, speed, steering, angles, ranges, max_linear, max_angular,
                      turn_fraction, influence_radius, repulsion_gain, stop_distance,
                      slow_distance, heading_gain=2.0):
    """The user's TELEOP intent, steered around nearby obstacles.

    Driving forward, the commanded direction is the attractive vector and every
    LiDAR return inside influence_radius repels (repulsion_gain is per radian of
    scan, so it does not depend on the LiDAR resolution). The robot follows the
    resulting heading, slows as the front clearance shrinks and stops at
    stop_distance. Reversing only checks the rear clearance; turning on the spot
    is always allowed.
    """
    v, omega = teleop_velocity(command, speed, steering, max_linear, max_angular, turn_fraction)
    angles = np.asarray(angles, dtype=float)
    if command == "FORWARD":
        ray_step = ray_spacing(angles)
        intent = (math.cos(steering * math.pi / 4), math.sin(steering * math.pi / 4))
        heading, _ = potential_field(intent, scan_to_points(angles, ranges, influence_radius),
                                     influence_radius, repulsion_gain * ray_step)
        front = sector_min(angles, ranges, 0.0, 0.5)
        v = (speed * max_linear * max(0.0, math.cos(heading))
             * clearance_scale(front, stop_distance, slow_distance))
        omega = min(max(heading_gain * heading, -max_angular), max_angular)
    elif command == "BACKWARD":
        v *= clearance_scale(sector_min(angles, ranges, math.pi, 0.5), stop_distance, slow_distance)
    return v, omega
