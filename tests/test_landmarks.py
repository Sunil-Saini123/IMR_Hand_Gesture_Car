import math
from robot_core.behaviors import lidar_angles
from robot_core.landmarks import detect_landmarks
from scan_sim import simulate_scan

ANGLES = lidar_angles(2 * math.pi, 360)
ARENA = [(-2.0, -2.0, -1.95, 2.0), (1.95, -2.0, 2.0, 2.0), (-2.0, -2.0, 2.0, -1.95), (-2.0, 1.95, 2.0, 2.0)]


def test_lidar_angles_follow_webots_order():
    assert math.isclose(ANGLES[0], math.pi)  # index 0 is the left-most ray, angles decrease
    assert math.isclose(ANGLES[180], 0.0, abs_tol=1e-12)
    assert ANGLES[90] > 0 > ANGLES[270]


def test_detects_post_with_range_and_bearing():
    post = (1.0, 0.5)
    ranges = simulate_scan((0.0, 0.0, 0.0), ANGLES, ARENA, [(*post, 0.05)])
    obs = detect_landmarks((0.0, 0.0, 0.0), ANGLES, ranges, [post], 0.05, 2.0)
    assert len(obs) == 1 and obs[0].index == 0
    assert abs(obs[0].range - math.hypot(*post)) < 0.02
    assert abs(obs[0].bearing - math.atan2(post[1], post[0])) < 0.02


def test_detects_from_slightly_wrong_pose_and_rotated_robot():
    post = (-0.8, 0.9)
    true_pose = (0.2, -0.1, 2.0)
    ranges = simulate_scan(true_pose, ANGLES, ARENA, [(*post, 0.05)])
    obs = detect_landmarks((0.25, -0.13, 2.05), ANGLES, ranges, [post], 0.05, 2.0)
    assert len(obs) == 1
    expected = math.atan2(post[1] - true_pose[1], post[0] - true_pose[0]) - true_pose[2]
    assert abs(obs[0].bearing - expected) < 0.02


def test_wall_where_a_post_should_be_is_rejected():
    wall = (0.95, -1.0, 1.05, 1.0)  # face at x = 0.95, exactly the predicted post surface range
    ranges = simulate_scan((0.0, 0.0, 0.0), ANGLES, [wall])
    assert detect_landmarks((0.0, 0.0, 0.0), ANGLES, ranges, [(1.0, 0.0)], 0.05, 2.0) == []


def test_out_of_range_or_hidden_posts_are_skipped():
    far_post = (1.9, 1.9)
    ranges = simulate_scan((-1.5, -1.5, 0.0), ANGLES, ARENA, [(*far_post, 0.05)])
    assert detect_landmarks((-1.5, -1.5, 0.0), ANGLES, ranges, [far_post], 0.05, 2.0) == []
    hidden = (1.0, 0.0)
    blocker = (0.4, -0.3, 0.5, 0.3)
    ranges = simulate_scan((0.0, 0.0, 0.0), ANGLES, [blocker], [(*hidden, 0.05)])
    assert detect_landmarks((0.0, 0.0, 0.0), ANGLES, ranges, [hidden], 0.05, 2.0) == []
