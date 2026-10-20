"""Webots controller for the gesture-guided e-puck.

Each step: wheel odometry -> EKF prediction; LiDAR landmark posts -> EKF correction;
LiDAR scan -> occupancy grid; newest UDP command -> TELEOP / ASSISTED / GOAL behaviour;
IR last-resort stop -> wheel speeds. Supervisor ground truth is read only for logging
and trial scoring, never by the estimator.
"""
from __future__ import annotations
from datetime import datetime
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
# numpy's OpenBLAS reserves buffers for every CPU thread (~400 MB on a 12-thread laptop vs ~50 MB
# with one thread); the controller only multiplies 3x3 matrices, so one thread is plenty.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import numpy as np
    from robot_core.behaviors import assisted_velocity, lidar_angles, teleop_velocity
    from robot_core.commands import CommandLink
    from robot_core.config import load_config
    from robot_core.datalog import RunLogger
    from robot_core.ekf import PoseEKF
    from robot_core.kinematics import Pose2D, wheel_speeds
    from robot_core.landmarks import detect_landmarks
    from robot_core.navigation import FAILED, NAVIGATING, PAUSED, REACHED, GoalNavigator, NavigationParams
    from robot_core.occupancy_grid import OccupancyGrid
    from robot_core.odometry import Odometry
except ImportError as exc:
    sys.exit(f"robot_controller: {exc}. Webots runs controllers with the Python set in "
             f"Tools > Preferences > Python command; point it at a Python that has requirements.txt "
             f"installed, e.g. {REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'}")

from controller import Supervisor

FRONT_IR = (0, 1, 6, 7)  # ps0/ps7 face forward, ps1/ps6 point 45 degrees to each side


def ground_truth(node):
    """Supervisor pose (evaluation only)."""
    x, y, _ = node.getPosition()
    rotation = node.getOrientation()
    return x, y, math.atan2(rotation[3], rotation[0])


def main() -> None:
    cfg = load_config()
    net, body, loc = cfg["network"], cfg["robot"], cfg["localization"]
    mapping, nav_cfg, assist, log_cfg = cfg["mapping"], cfg["navigation"], cfg["assisted"], cfg["logging"]

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    left, right = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
    for motor in (left, right):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    left_encoder, right_encoder = robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor")
    left_encoder.enable(timestep)
    right_encoder.enable(timestep)
    proximity = [robot.getDevice(f"ps{i}") for i in range(8)]
    for sensor in proximity:
        sensor.enable(timestep)
    lidar = robot.getDevice(cfg["lidar"]["name"])
    if lidar is None:
        sys.exit("robot_controller: no LiDAR found. Open webots/worlds/gesture_arena.wbt from this "
                 "repository; its e-puck carries the LiDAR in turretSlot.")
    lidar.enable(timestep)
    angles = lidar_angles(lidar.getFov(), lidar.getHorizontalResolution())
    max_range = lidar.getMaxRange()
    self_node = robot.getSelf()

    radius, base, max_wheel = body["wheel_radius_m"], body["wheel_base_m"], body["max_wheel_speed_rad_s"]
    max_linear = radius * max_wheel
    max_angular = 2 * radius * max_wheel / base
    turn_fraction = cfg["teleop"]["turn_fraction"]
    start_pose = tuple(body["start_pose"])
    odometry = Odometry(Pose2D(*start_pose), radius, base)
    ekf = PoseEKF(start_pose, loc["initial_sigma"], loc["motion_noise"])
    landmark_noise = np.diag([loc["range_noise_m"] ** 2, loc["bearing_noise_rad"] ** 2])
    landmarks = [tuple(p) for p in loc["landmarks"]]
    grid = OccupancyGrid(mapping["width_cells"], mapping["height_cells"], mapping["resolution_m_per_cell"],
                         mapping["log_odds_hit"], mapping["log_odds_miss"])
    nav = GoalNavigator(NavigationParams(**{k: v for k, v in nav_cfg.items() if k != "planner"}))
    try:
        link = CommandLink(net["host"], net["port"], net["timeout_seconds"])
    except OSError as exc:
        sys.exit(f"robot_controller: cannot listen on {net['host']}:{net['port']} ({exc}). "
                 "Is another simulation already running?")
    logger = RunLogger(REPO_ROOT / log_cfg["directory"], log_cfg["flush_period_s"]) if log_cfg["enabled"] else None

    say = logger.event if logger else lambda message: print(message, flush=True)
    say(f"Robot controller listening at {net['host']}:{net['port']}; LiDAR {len(angles)} rays, "
        f"{max_range:.1f} m" + (f"; logging to {logger.run_dir}" if logger else ""))

    step = 0
    v = omega = 0.0
    update_period = loc["update_every_n_steps"] * timestep / 1000.0
    moved = math.inf  # forces a correction on the first update step
    landmarks_used, misses, rejected, lost_warned = 0, 0, 0, False
    finished_target = None  # (goal, planner) that just ended; FORWARD won't restart it until STOP re-arms
    trial = None            # ground-truth metrics of the goal in progress (evaluation only)
    trial_id, last_result = 0, None
    plans_logged = 0
    in_contact = False
    last_status = last_map_save = 0.0
    previous_gt = ground_truth(self_node)

    def start_goal(goal, goal_name, planner, log_trial, now, gt):
        nonlocal trial
        nav.start(goal, goal_name, planner, now)
        trial = {"method": planner, "goal_name": goal_name, "goal": goal, "start": gt[:2], "t0": now,
                 "length": 0.0, "collisions": 0, "error_sum": 0.0, "samples": 0, "log": log_trial}
        say(f"t={now:.1f} GOAL {goal_name or goal} with {planner}")

    def finish_goal(now, gt):
        nonlocal trial, trial_id, last_result, finished_target
        trial_id += 1
        finished_target = (nav.goal, nav.planner)
        last_result = {"goal_name": trial["goal_name"], "method": trial["method"],
                       "success": nav.state == REACHED, "outcome": nav.reason,
                       "time_s": round(now - trial["t0"], 2), "path_length_m": round(trial["length"], 3)}
        say(f"t={now:.1f} GOAL {trial['goal_name']}: {nav.reason} after {now - trial['t0']:.1f} s")
        if logger and trial["log"] and trial["length"] > 0.01:
            logger.log_trial({
                "timestamp": datetime.now().isoformat(timespec="seconds"), "run": logger.run_name,
                "method": trial["method"], "goal_name": trial["goal_name"],
                "goal_x": trial["goal"][0], "goal_y": trial["goal"][1],
                "start_x": trial["start"][0], "start_y": trial["start"][1],
                "success": nav.state == REACHED, "outcome": nav.reason,
                "time_s": now - trial["t0"], "path_length_m": trial["length"],
                "collisions": trial["collisions"],
                "final_error_m": math.hypot(gt[0] - trial["goal"][0], gt[1] - trial["goal"][1]),
                "mean_ekf_error_m": trial["error_sum"] / max(trial["samples"], 1)})
        trial = None

    try:
        while robot.step(timestep) != -1:
            step += 1
            t = robot.getTime()
            wall = time.monotonic()
            ranges = np.asarray(lidar.getRangeImage(), dtype=float)

            # Localization: odometry prediction, then landmark correction. Corrections are skipped
            # while the robot is parked: repeated scans from one spot are not independent
            # measurements and would make the filter overconfident.
            distance, dtheta = odometry.step(left_encoder.getValue(), right_encoder.getValue())
            ekf.predict_increment(distance, dtheta)
            moved += abs(distance) + abs(dtheta) * base / 2
            if step % loc["update_every_n_steps"] == 0 and moved >= loc["min_motion_m"]:
                moved = 0.0
                observations = detect_landmarks(ekf.pose, angles, ranges, landmarks,
                                                loc["landmark_radius_m"], max_range)
                landmarks_used = sum(ekf.update_range_bearing(o.range, o.bearing, landmarks[o.index],
                                                              landmark_noise, loc["gate_chi2"])
                                     for o in observations)
                if landmarks_used:
                    misses, rejected, lost_warned = 0, 0, False
                else:
                    misses += 1
                    rejected += bool(observations)
                    if rejected >= loc["relax_after_rejections"]:
                        ekf.inflate(loc["relax_sigma"])
                        rejected = 0
                        say(f"t={t:.1f} EKF: posts are visible but every match was gated out; widening the "
                            "covariance so the filter can re-converge.")
                    if misses * update_period > loc["lost_warning_seconds"] and not lost_warned:
                        say(f"t={t:.1f} WARNING: no landmark matched for {loc['lost_warning_seconds']:.0f} s of "
                            "driving; the pose estimate may be lost. Reset the simulation so the robot starts "
                            "at robot.start_pose.")
                        lost_warned = True
            pose = ekf.pose

            if step % mapping["update_every_n_steps"] == 0:
                grid.update_scan(pose, angles, ranges, max_range)

            # Command arbitration.
            cmd = link.poll(wall)
            stale = link.is_stale(wall)
            command = "STOP" if stale else cmd.command
            mode = cmd.mode
            gt = ground_truth(self_node)
            if (stale or mode != "GOAL") and (nav.active or finished_target):
                if nav.active:
                    nav.cancel()
                    trial = None
                finished_target = None
            if stale or mode == "TELEOP":
                v, omega = teleop_velocity(command, cmd.speed, cmd.steering, max_linear, max_angular,
                                           turn_fraction)
            elif mode == "ASSISTED":
                v, omega = assisted_velocity(command, cmd.speed, cmd.steering, angles, ranges, max_linear,
                                             assist["max_angular"], turn_fraction,
                                             assist["influence_radius_m"], assist["repulsion_gain"],
                                             assist["stop_distance_m"], assist["slow_distance_m"],
                                             assist["heading_gain"])
            else:
                planner = cmd.planner or nav_cfg["planner"]
                target = (cmd.goal, planner)
                if command == "STOP":
                    nav.pause(t)
                    finished_target = None
                elif cmd.goal is not None:
                    if nav.active and (nav.goal, nav.planner) != target:
                        restart = nav.state == NAVIGATING
                        nav.cancel()
                        trial = None
                        if restart:
                            start_goal(cmd.goal, cmd.goal_name, planner, cmd.log_trial, t, gt)
                    if command == "FORWARD":
                        if nav.state == PAUSED:
                            nav.resume(t)
                        elif not nav.active and target != finished_target:
                            start_goal(cmd.goal, cmd.goal_name, planner, cmd.log_trial, t, gt)
                v, omega = nav.update(pose, grid, angles, ranges, t)
                if logger and nav.plan_count != plans_logged:
                    plans_logged = nav.plan_count
                    logger.log_plan(t, nav.goal_name, nav.path)
                if trial and nav.state in (REACHED, FAILED):
                    finish_goal(t, gt)

            if mode != "TELEOP" and v > 0 and max(proximity[i].getValue() for i in FRONT_IR) \
                    > body["proximity_stop_threshold"]:
                v = 0.0
            wheel_left, wheel_right = wheel_speeds(v, omega, radius, base, max_wheel)
            left.setVelocity(wheel_left)
            right.setVelocity(wheel_right)

            # Evaluation bookkeeping from ground truth.
            finite = ranges[np.isfinite(ranges)]
            min_range = float(finite.min()) if finite.size else math.inf
            contact = min_range < body["body_radius_m"] + 0.005
            if trial:
                trial["length"] += math.hypot(gt[0] - previous_gt[0], gt[1] - previous_gt[1])
                trial["error_sum"] += math.hypot(gt[0] - pose[0], gt[1] - pose[1])
                trial["samples"] += 1
                if contact and not in_contact:
                    trial["collisions"] += 1
            in_contact, previous_gt = contact, gt

            if logger:
                sigma = ekf.sigmas
                logger.log_pose({"t": t, "mode": mode, "command": command, "nav_state": nav.state,
                                 "gt_x": gt[0], "gt_y": gt[1], "gt_theta": gt[2],
                                 "odom_x": odometry.pose.x, "odom_y": odometry.pose.y,
                                 "odom_theta": odometry.pose.theta,
                                 "ekf_x": pose[0], "ekf_y": pose[1], "ekf_theta": pose[2],
                                 "ekf_sx": sigma[0], "ekf_sy": sigma[1], "ekf_stheta": sigma[2],
                                 "landmarks": landmarks_used, "min_range": min(min_range, max_range),
                                 "v": v, "omega": omega}, wall)
                if t - last_map_save >= log_cfg["map_save_period_s"]:
                    logger.save_map(grid)
                    last_map_save = t

            if wall - last_status >= net["status_period_seconds"]:
                last_status = wall
                link.send_status({
                    "type": "status", "t": round(t, 2), "mode": mode, "command": command, "stale": stale,
                    "nav_state": nav.state, "nav_reason": nav.reason, "goal_name": nav.goal_name,
                    "planner": nav.planner, "pose": [round(p, 3) for p in pose],
                    "sigma": [round(s, 3) for s in ekf.sigmas], "landmarks": landmarks_used,
                    "localization_ok": misses * update_period <= loc["lost_warning_seconds"],
                    "min_range": round(min_range, 3) if math.isfinite(min_range) else None,
                    "trial_id": trial_id, "last_result": last_result})
    except Exception:
        if logger:
            logger.event(traceback.format_exc())
        raise
    finally:
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        if logger:
            logger.save_map(grid)
            logger.close()
        link.close()


if __name__ == "__main__":
    main()
