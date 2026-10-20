"""Encoder-distance dead reckoning."""
from dataclasses import dataclass
import math
from .kinematics import Pose2D, wrap_angle


@dataclass
class Odometry:
    pose: Pose2D
    wheel_radius: float
    wheel_base: float
    previous_left_rad: float | None = None
    previous_right_rad: float | None = None

    def step(self, left_rad: float, right_rad: float) -> tuple[float, float]:
        """Integrate new encoder readings; return the (distance, dtheta) increment."""
        if self.previous_left_rad is None or self.previous_right_rad is None:
            self.previous_left_rad, self.previous_right_rad = left_rad, right_rad
            return 0.0, 0.0
        dl = self.wheel_radius * (left_rad - self.previous_left_rad)
        dr = self.wheel_radius * (right_rad - self.previous_right_rad)
        self.previous_left_rad, self.previous_right_rad = left_rad, right_rad
        distance = (dl + dr) / 2
        dtheta = (dr - dl) / self.wheel_base
        mid = self.pose.theta + dtheta / 2
        self.pose.x += distance * math.cos(mid)
        self.pose.y += distance * math.sin(mid)
        self.pose.theta = wrap_angle(self.pose.theta + dtheta)
        return distance, dtheta

    def update(self, left_rad: float, right_rad: float) -> Pose2D:
        self.step(left_rad, right_rad)
        return self.pose
