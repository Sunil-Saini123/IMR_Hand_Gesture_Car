import math
from robot_core.kinematics import Pose2D
from robot_core.odometry import Odometry


def test_first_reading_only_initializes():
    odom = Odometry(Pose2D(1.0, 2.0, 0.5), wheel_radius=0.02, wheel_base=0.05)
    assert odom.step(10.0, 12.0) == (0.0, 0.0)
    assert (odom.pose.x, odom.pose.y, odom.pose.theta) == (1.0, 2.0, 0.5)


def test_straight_and_spin_increments():
    odom = Odometry(Pose2D(), wheel_radius=0.02, wheel_base=0.05)
    odom.step(0.0, 0.0)
    distance, dtheta = odom.step(10.0, 10.0)
    assert math.isclose(distance, 0.2) and dtheta == 0.0
    assert math.isclose(odom.pose.x, 0.2) and math.isclose(odom.pose.y, 0.0, abs_tol=1e-12)
    distance, dtheta = odom.step(10.0 - 1.0, 10.0 + 1.0)
    assert math.isclose(distance, 0.0, abs_tol=1e-12)
    assert math.isclose(dtheta, 2 * 0.02 / 0.05)


def test_update_returns_pose():
    odom = Odometry(Pose2D(), wheel_radius=0.02, wheel_base=0.05)
    odom.update(0.0, 0.0)
    pose = odom.update(5.0, 5.0)
    assert math.isclose(pose.x, 0.1)
