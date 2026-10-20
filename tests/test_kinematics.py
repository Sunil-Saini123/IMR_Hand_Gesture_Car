import math
from robot_core.kinematics import Pose2D, body_velocity, integrate_pose, wheel_speeds


def test_equal_wheels_move_straight():
    v, omega = body_velocity(1.0, 1.0, 0.5)
    assert v == 1.0 and omega == 0.0
    pose = integrate_pose(Pose2D(), 1.0, 1.0, 0.5, 1.0)
    assert math.isclose(pose.x, 1.0, abs_tol=1e-6)
    assert math.isclose(pose.y, 0.0, abs_tol=1e-6)


def test_opposite_wheels_rotate():
    _, omega = body_velocity(-1.0, 1.0, 0.5)
    assert math.isclose(omega, 4.0)


def test_wheel_speeds_invert_body_velocity():
    left, right = wheel_speeds(0.05, 0.8, 0.02, 0.052, 100.0)
    v, omega = body_velocity(left * 0.02, right * 0.02, 0.052)
    assert math.isclose(v, 0.05) and math.isclose(omega, 0.8)


def test_wheel_speeds_saturate_without_changing_turn_radius():
    left, right = wheel_speeds(0.2, 1.0, 0.02, 0.052, 6.28)
    assert max(abs(left), abs(right)) <= 6.28 + 1e-9
    unsaturated = wheel_speeds(0.2, 1.0, 0.02, 0.052, 1e9)
    assert math.isclose(left / right, unsaturated[0] / unsaturated[1])
