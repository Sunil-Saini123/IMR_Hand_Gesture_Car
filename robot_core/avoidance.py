"""Simple potential-field steering suggestion."""
import math
import numpy as np


def potential_field(goal_vector, obstacles_relative, influence_radius=1.0, repulsion_gain=0.8):
    """goal_vector is attractive vector in robot coordinates; obstacles are relative (x,y)."""
    fx, fy = goal_vector
    for ox, oy in obstacles_relative:
        distance = max(math.hypot(ox, oy), 1e-6)
        if distance < influence_radius:
            magnitude = repulsion_gain * (1/distance - 1/influence_radius) / (distance**2)
            fx -= magnitude * ox / distance
            fy -= magnitude * oy / distance
    return math.atan2(fy, fx), (fx, fy)


def scan_to_points(angles, ranges, max_distance):
    """Robot-frame (x, y) points of the scan returns closer than max_distance."""
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    near = np.isfinite(ranges) & (ranges < max_distance)
    return list(zip((ranges[near] * np.cos(angles[near])).tolist(),
                    (ranges[near] * np.sin(angles[near])).tolist()))
