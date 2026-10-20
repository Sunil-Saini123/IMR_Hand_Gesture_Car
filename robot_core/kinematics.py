"""Differential-drive forward and inverse kinematics."""
from dataclasses import dataclass
import math


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


def body_velocity(v_left: float, v_right: float, wheel_base: float) -> tuple[float, float]:
    if wheel_base <= 0:
        raise ValueError("wheel_base must be positive")
    return (v_right + v_left) / 2.0, (v_right - v_left) / wheel_base


def wheel_speeds(v: float, omega: float, wheel_radius: float, wheel_base: float,
                 max_wheel_speed: float) -> tuple[float, float]:
    """Body velocity (m/s, rad/s) -> wheel angular speeds (rad/s).

    When a wheel would exceed max_wheel_speed both wheels are scaled by the same
    factor, so the commanded turning radius is preserved.
    """
    if wheel_radius <= 0 or wheel_base <= 0:
        raise ValueError("wheel_radius and wheel_base must be positive")
    left = (v - omega * wheel_base / 2.0) / wheel_radius
    right = (v + omega * wheel_base / 2.0) / wheel_radius
    largest = max(abs(left), abs(right))
    if largest > max_wheel_speed > 0:
        scale = max_wheel_speed / largest
        left, right = left * scale, right * scale
    return left, right


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def integrate_pose(pose: Pose2D, v_left: float, v_right: float,
                   wheel_base: float, dt: float) -> Pose2D:
    if dt < 0:
        raise ValueError("dt must be non-negative")
    v, omega = body_velocity(v_left, v_right, wheel_base)
    mid = pose.theta + 0.5 * omega * dt
    return Pose2D(pose.x + v * math.cos(mid) * dt,
                  pose.y + v * math.sin(mid) * dt,
                  wrap_angle(pose.theta + omega * dt))
