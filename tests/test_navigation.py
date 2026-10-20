"""Closed-loop navigation in a small simulated arena (perfect localization, analytic LiDAR)."""
import math
import numpy as np
from robot_core.behaviors import assisted_velocity, lidar_angles
from robot_core.kinematics import Pose2D, integrate_pose, wheel_speeds
from robot_core.navigation import FAILED, REACHED, GoalNavigator, NavigationParams, PathFollower
from robot_core.occupancy_grid import OccupancyGrid
from scan_sim import simulate_scan

ANGLES = lidar_angles(2 * math.pi, 120)
RADIUS, BASE, MAX_WHEEL, BODY = 0.02, 0.052, 6.28, 0.037
WALLS = [(-1.05, -1.05, -1.0, 1.05), (1.0, -1.05, 1.05, 1.05),
         (-1.05, -1.05, 1.05, -1.0), (-1.05, 1.0, 1.05, 1.05)]
CUP = [(0.2, -0.4, 0.25, 0.4), (-0.2, 0.35, 0.25, 0.4), (-0.2, -0.4, 0.25, -0.35)]  # opens towards -x


def clearance(pose, box):
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - pose.x, 0, pose.x - x1), max(y0 - pose.y, 0, pose.y - y1))


def navigate(goal, boxes, planner, start=(-0.7, 0.0, 0.0), sim_time=150.0, dt=0.032):
    """Returns (final state, reason, closest approach of the robot centre to any obstacle)."""
    obstacles = WALLS + boxes
    grid = OccupancyGrid(60, 60, 0.05)
    nav = GoalNavigator(NavigationParams())
    pose, t, step, closest = Pose2D(*start), 0.0, 0, math.inf
    nav.start(goal, "G", planner, 0.0)
    while t < sim_time and nav.active:
        p = (pose.x, pose.y, pose.theta)
        if step % 2 == 0:
            ranges = simulate_scan(p, ANGLES, obstacles)
            grid.update_scan(p, ANGLES, ranges, 2.0)
        v, omega = nav.update(p, grid, ANGLES, ranges, t)
        left, right = wheel_speeds(v, omega, RADIUS, BASE, MAX_WHEEL)
        pose = integrate_pose(pose, left * RADIUS, right * RADIUS, BASE, dt)
        closest = min([closest] + [clearance(pose, b) for b in obstacles])
        t, step = t + dt, step + 1
    return nav.state, nav.reason, closest


def test_astar_drives_around_a_box_without_touching_it():
    state, reason, closest = navigate((0.7, 0.0), [(-0.15, -0.5, 0.15, 0.5)], "astar")
    assert (state, reason) == (REACHED, "reached")
    assert closest > BODY


def test_astar_escapes_a_cup_that_traps_the_potential_field():
    assert navigate((0.6, 0.0), CUP, "astar")[0] == REACHED
    state, reason, _ = navigate((0.6, 0.0), CUP, "potential_field")
    assert state == FAILED and reason in ("stuck", "timeout")


def test_potential_field_reaches_an_open_goal():
    state, _, closest = navigate((0.7, 0.4), [], "potential_field")
    assert state == REACHED and closest > BODY


def test_follower_steers_back_onto_the_path():
    follower = PathFollower(0.12, 0.08, 2.0, 2.5, 0.8, 0.15)
    follower.set_path([(x, 0.0) for x in np.arange(0.0, 1.0, 0.05)])
    v, omega = follower.command((0.0, 0.02, 0.0))
    assert v > 0 and omega < 0
    assert follower.command((0.0, 0.0, math.pi))[0] == 0.0  # facing away: turn on the spot first


def test_assisted_mode_stops_in_front_of_a_wall():
    ranges = simulate_scan((0.0, 0.0, 0.0), ANGLES, [(0.06, -1.0, 0.1, 1.0)])
    v, _ = assisted_velocity("FORWARD", 1.0, 0.0, ANGLES, ranges, 0.12, 2.0, 0.45, 0.25, 0.01, 0.07, 0.2)
    assert v == 0.0
    clear = simulate_scan((0.0, 0.0, 0.0), ANGLES, WALLS)
    v, omega = assisted_velocity("FORWARD", 1.0, 0.0, ANGLES, clear, 0.12, 2.0, 0.45, 0.25, 0.01, 0.07, 0.2)
    assert math.isclose(v, 0.12) and abs(omega) < 1e-9
