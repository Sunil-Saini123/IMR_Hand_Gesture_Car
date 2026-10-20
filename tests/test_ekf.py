import math
import numpy as np
from robot_core.ekf import PoseEKF

R = np.diag([0.03 ** 2, 0.03 ** 2])
LANDMARKS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)]


def measure(pose, landmark):
    dx, dy = landmark[0] - pose[0], landmark[1] - pose[1]
    bearing = math.atan2(dy, dx) - pose[2]
    return math.hypot(dx, dy), math.atan2(math.sin(bearing), math.cos(bearing))


def test_prediction_follows_odometry_and_grows_with_motion():
    ekf = PoseEKF((0.0, 0.0, 0.0), sigma0=(0.01, 0.01, 0.01))
    for _ in range(100):
        ekf.predict_increment(0.01, 0.0)
    x, y, theta = ekf.pose
    assert math.isclose(x, 1.0) and abs(y) < 1e-12 and theta == 0.0
    sx, sy, _ = ekf.sigmas
    assert sx > 0.01 and sy > 0.01


def test_stationary_robot_stays_certain():
    ekf = PoseEKF((0.0, 0.0, 0.0), sigma0=(0.01, 0.01, 0.01))
    before = ekf.P.copy()
    for _ in range(1000):
        ekf.predict(0.0, 0.0, 0.032)
    assert np.allclose(ekf.P, before, atol=2e-5)


def test_landmark_updates_correct_the_pose():
    truth = (0.0, 0.0, 0.0)
    ekf = PoseEKF((0.1, -0.06, 0.05), sigma0=(0.1, 0.1, 0.1))
    trace_before = np.trace(ekf.P)
    for _ in range(10):
        for landmark in LANDMARKS:
            assert ekf.update_range_bearing(*measure(truth, landmark), landmark, R, gate=9.21)
    x, y, theta = ekf.pose
    assert abs(x) < 0.01 and abs(y) < 0.01 and abs(theta) < 0.01
    assert np.trace(ekf.P) < trace_before / 10


def test_gate_rejects_outliers():
    ekf = PoseEKF((0.0, 0.0, 0.0), sigma0=(0.01, 0.01, 0.01))
    z_range, z_bearing = measure((0.0, 0.0, 0.0), LANDMARKS[0])
    assert not ekf.update_range_bearing(z_range, z_bearing + 1.0, LANDMARKS[0], R, gate=9.21)
    assert ekf.pose == (0.0, 0.0, 0.0)


def test_bearing_innovation_is_wrapped():
    ekf = PoseEKF((0.0, 0.0, math.pi - 0.01), sigma0=(0.01, 0.01, 0.05))
    ekf.update_linear([-math.pi + 0.01], [math.pi - 0.01], [[0.0, 0.0, 1.0]], [[0.05 ** 2]], angle_rows=(0,))
    # The 0.02 rad innovation must pull theta across +-pi, not swing it by almost 2*pi.
    assert abs(abs(ekf.pose[2]) - math.pi) < 0.02
