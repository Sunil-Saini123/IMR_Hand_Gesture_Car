"""Pose EKF: odometry-driven prediction and range-bearing landmark updates."""
import numpy as np


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PoseEKF:
    def __init__(self, x0=(0.0, 0.0, 0.0), sigma0=(0.02, 0.02, 0.03),
                 motion_noise=(0.04, 0.005, 0.05, 0.3)):
        """sigma0: initial standard deviations of (x, y, theta).

        motion_noise = (a1, a2, a3, a4): the variance of one odometry increment
        (d, dtheta) grows linearly with how far the wheels moved,
        var(d) = a1^2*|d| + a2^2*|dtheta| and var(dtheta) = a3^2*|d| + a4^2*|dtheta|
        (the random-walk form of Siegwart's wheel-odometry error model). So the
        uncertainty grows with motion, not with elapsed time, and a long run of
        small steps is not treated as many independent tiny errors. Wheel slip
        in turns is large and systematic, hence the large a4.
        """
        self.x = np.asarray(x0, dtype=float).reshape(3, 1)
        self.P = np.diag(np.square(np.asarray(sigma0, dtype=float)))
        self.alpha = tuple(float(a) for a in motion_noise)
        self.Q_floor = np.diag([1e-8, 1e-8, 1e-8])

    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

    @property
    def sigmas(self) -> tuple[float, float, float]:
        return tuple(float(s) for s in np.sqrt(np.diag(self.P)))

    def predict_increment(self, distance: float, dtheta: float):
        """Propagate the state by one odometry increment (midpoint motion model)."""
        x, y, theta = self.pose
        mid = theta + 0.5 * dtheta
        c, s = np.cos(mid), np.sin(mid)
        self.x[0, 0] = x + distance * c
        self.x[1, 0] = y + distance * s
        self.x[2, 0] = _wrap(theta + dtheta)
        F = np.array([[1.0, 0.0, -distance * s],
                      [0.0, 1.0, distance * c],
                      [0.0, 0.0, 1.0]])
        V = np.array([[c, -0.5 * distance * s],
                      [s, 0.5 * distance * c],
                      [0.0, 1.0]])
        a1, a2, a3, a4 = self.alpha
        var_d = a1 ** 2 * abs(distance) + a2 ** 2 * abs(dtheta)
        var_theta = a3 ** 2 * abs(distance) + a4 ** 2 * abs(dtheta)
        self.P = F @ self.P @ F.T + V @ np.diag([var_d, var_theta]) @ V.T + self.Q_floor
        return self.x[:, 0].copy()

    def predict(self, v: float, omega: float, dt: float):
        return self.predict_increment(v * dt, omega * dt)

    def inflate(self, sigmas):
        """Add independent uncertainty (x, y, theta standard deviations) to P."""
        self.P = self.P + np.diag(np.square(np.asarray(sigmas, dtype=float)))

    def _correct(self, innovation, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[2, 0] = _wrap(self.x[2, 0])
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return self.x[:, 0].copy()

    def update_linear(self, z, h, H, R, angle_rows=()):
        """EKF update: z=measurement, h=predicted measurement, H=Jacobian, R=noise covariance.

        Innovation rows listed in angle_rows are wrapped to [-pi, pi).
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        h = np.asarray(h, dtype=float).reshape(-1, 1)
        H, R = np.asarray(H, dtype=float), np.asarray(R, dtype=float)
        innovation = z - h
        for row in angle_rows:
            innovation[row, 0] = _wrap(innovation[row, 0])
        return self._correct(innovation, H, R)

    def update_range_bearing(self, z_range: float, z_bearing: float, landmark, R,
                             gate: float | None = None) -> bool:
        """Correct with one observation of a landmark at known map position.

        z_bearing is in the robot frame. If gate is given, the update is skipped
        when the innovation's Mahalanobis distance squared exceeds it
        (9.21 is the 99% chi-square bound for 2 degrees of freedom).
        Returns True if the update was applied.
        """
        x, y, theta = self.pose
        dx, dy = float(landmark[0]) - x, float(landmark[1]) - y
        q = dx * dx + dy * dy
        if q < 1e-9:
            return False
        r = np.sqrt(q)
        H = np.array([[-dx / r, -dy / r, 0.0],
                      [dy / q, -dx / q, -1.0]])
        R = np.asarray(R, dtype=float)
        innovation = np.array([[z_range - r],
                               [_wrap(z_bearing - _wrap(np.arctan2(dy, dx) - theta))]])
        if gate is not None:
            S = H @ self.P @ H.T + R
            if (innovation.T @ np.linalg.solve(S, innovation)).item() > gate:
                return False
        self._correct(innovation, H, R)
        return True
