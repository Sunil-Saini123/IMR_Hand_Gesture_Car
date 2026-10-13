import math
from robot_core.kinematics import Pose2D, body_velocity, integrate_pose


def test_equal_wheels_move_straight():
    v, omega = body_velocity(1.0, 1.0, 0.5)
    assert v == 1.0 and omega == 0.0
    pose = integrate_pose(Pose2D(), 1.0, 1.0, 0.5, 1.0)
    assert math.isclose(pose.x, 1.0, abs_tol=1e-6)
    assert math.isclose(pose.y, 0.0, abs_tol=1e-6)


def test_opposite_wheels_rotate():
    _, omega = body_velocity(-1.0, 1.0, 0.5)
    assert math.isclose(omega, 4.0)
