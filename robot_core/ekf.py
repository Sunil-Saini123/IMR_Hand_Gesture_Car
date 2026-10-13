"""Pose EKF building block; integrate a real landmark/range measurement model."""
import numpy as np


class PoseEKF:
    def __init__(self, x0=(0.0, 0.0, 0.0)):
        self.x = np.asarray(x0, dtype=float).reshape(3, 1)
        self.P = np.diag([0.02, 0.02, 0.03])
        self.Q = np.diag([0.005, 0.005, 0.01])

    def predict(self, v: float, omega: float, dt: float):
        theta = float(self.x[2, 0])
        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] = (theta + omega * dt + np.pi) % (2*np.pi) - np.pi
        F = np.array([[1, 0, -v*np.sin(theta)*dt],
                      [0, 1, v*np.cos(theta)*dt],
                      [0, 0, 1]], dtype=float)
        self.P = F @ self.P @ F.T + self.Q * max(dt, 1e-3)
        return self.x[:, 0].copy()

    def update_linear(self, z, h, H, R):
        """EKF update: z=measurement, h=predicted measurement, H=Jacobian, R=noise covariance."""
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        h = np.asarray(h, dtype=float).reshape(-1, 1)
        H, R = np.asarray(H, dtype=float), np.asarray(R, dtype=float)
        innovation = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2*np.pi) - np.pi
        I = np.eye(3)
        self.P = (I-K@H) @ self.P @ (I-K@H).T + K@R@K.T
        return self.x[:, 0].copy()
