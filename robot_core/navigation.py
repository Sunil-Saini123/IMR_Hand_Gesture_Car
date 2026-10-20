"""Goal navigation: A* on the live occupancy map followed by a carrot controller,
or a purely reactive potential field (the comparison baseline)."""
from dataclasses import dataclass
import math
from .avoidance import potential_field, scan_to_points
from .behaviors import clearance_scale, ray_spacing, sector_min
from .kinematics import wrap_angle
from .planning import plan_path

IDLE, NAVIGATING, PAUSED, REACHED, FAILED = "IDLE", "NAVIGATING", "PAUSED", "REACHED", "FAILED"


def _clamp(value, limit):
    return min(max(value, -limit), limit)


@dataclass
class NavigationParams:
    occupied_threshold: float = 0.65
    inflation_m: float = 0.08
    replan_period_s: float = 1.0
    lookahead_m: float = 0.12
    goal_tolerance_m: float = 0.06
    max_linear: float = 0.08
    max_angular: float = 1.2
    heading_gain: float = 2.0
    rotate_in_place_rad: float = 0.8
    slow_radius_m: float = 0.15
    stop_distance_m: float = 0.07
    slow_distance_m: float = 0.2
    timeout_s: float = 150.0
    stuck_window_s: float = 15.0
    stuck_min_progress_m: float = 0.05
    no_path_timeout_s: float = 5.0
    pf_influence_m: float = 0.3
    pf_repulsion_gain: float = 0.02


class PathFollower:
    """Steer towards the first path point at least `lookahead` ahead (carrot following)."""

    def __init__(self, lookahead, max_linear, max_angular, heading_gain, rotate_in_place, slow_radius):
        self.lookahead, self.max_linear, self.max_angular = lookahead, max_linear, max_angular
        self.heading_gain, self.rotate_in_place, self.slow_radius = heading_gain, rotate_in_place, slow_radius
        self.path, self.index = [], 0

    def set_path(self, path):
        self.path, self.index = list(path), 0

    def command(self, pose):
        if not self.path:
            return 0.0, 0.0
        x, y, theta = pose
        window = range(self.index, min(len(self.path), self.index + 25))
        self.index = min(window, key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))
        tx, ty = self.path[-1]
        for px, py in self.path[self.index:]:
            if math.hypot(px - x, py - y) >= self.lookahead:
                tx, ty = px, py
                break
        alpha = wrap_angle(math.atan2(ty - y, tx - x) - theta)
        omega = _clamp(self.heading_gain * alpha, self.max_angular)
        if abs(alpha) > self.rotate_in_place:
            return 0.0, omega
        to_goal = math.hypot(self.path[-1][0] - x, self.path[-1][1] - y)
        v = self.max_linear * math.cos(alpha) * max(0.3, min(1.0, to_goal / self.slow_radius))
        return v, omega


class GoalNavigator:
    """IDLE -> NAVIGATING (<-> PAUSED) -> REACHED | FAILED, for one goal at a time."""

    def __init__(self, params: NavigationParams):
        self.p = params
        self.follower = PathFollower(params.lookahead_m, params.max_linear, params.max_angular,
                                     params.heading_gain, params.rotate_in_place_rad, params.slow_radius_m)
        self.state, self.reason = IDLE, ""
        self.goal, self.goal_name, self.planner = None, "", "astar"
        self.path, self.plan_count = [], 0

    @property
    def active(self):
        return self.state in (NAVIGATING, PAUSED)

    def start(self, goal, goal_name, planner, now):
        self.state, self.reason = NAVIGATING, ""
        self.goal, self.goal_name, self.planner = (float(goal[0]), float(goal[1])), goal_name, planner
        self.started = self.best_time = now
        self.best_distance = math.inf
        self.last_plan, self.replan_requested = -math.inf, True
        self.no_path_since, self.paused_at = None, None
        self.path = []
        self.follower.set_path([])

    def pause(self, now):
        if self.state == NAVIGATING:
            self.state, self.paused_at = PAUSED, now

    def resume(self, now):
        if self.state == PAUSED:
            shift = now - self.paused_at
            self.started += shift
            self.best_time += shift
            if self.no_path_since is not None:
                self.no_path_since += shift
            self.state, self.replan_requested = NAVIGATING, True

    def cancel(self):
        self.state, self.reason = IDLE, "cancelled"
        self.follower.set_path([])

    def _finish(self, state, reason):
        self.state, self.reason = state, reason
        self.follower.set_path([])
        return 0.0, 0.0

    def update(self, pose, grid, angles, ranges, now):
        """Body velocity (v, omega) for this control step."""
        if self.state != NAVIGATING:
            return 0.0, 0.0
        x, y, _ = pose
        distance = math.hypot(self.goal[0] - x, self.goal[1] - y)
        if distance <= self.p.goal_tolerance_m:
            return self._finish(REACHED, "reached")
        if now - self.started > self.p.timeout_s:
            return self._finish(FAILED, "timeout")
        if distance < self.best_distance - self.p.stuck_min_progress_m:
            self.best_distance, self.best_time = distance, now
        elif now - self.best_time > self.p.stuck_window_s:
            return self._finish(FAILED, "stuck")

        if self.planner == "potential_field":
            v, omega = self._potential_field(pose, angles, ranges, distance)
        else:
            v, omega = self._astar(pose, grid, now)
            if self.state != NAVIGATING:
                return 0.0, 0.0
        if v > 0:
            front = sector_min(angles, ranges, 0.0, 0.35)
            if front <= self.p.stop_distance_m:
                v, self.replan_requested = 0.0, True
            else:
                v *= max(clearance_scale(front, self.p.stop_distance_m, self.p.slow_distance_m), 0.25)
        return v, omega

    def _astar(self, pose, grid, now):
        since_plan = now - self.last_plan
        if since_plan >= self.p.replan_period_s or (self.replan_requested and since_plan >= 0.25):
            self.path = plan_path(grid, pose[:2], self.goal, self.p.occupied_threshold, self.p.inflation_m)
            self.last_plan, self.replan_requested = now, False
            self.plan_count += 1
            self.follower.set_path(self.path)
            if self.path:
                self.no_path_since = None
            elif self.no_path_since is None:
                self.no_path_since = now
        if not self.path:
            if now - self.no_path_since > self.p.no_path_timeout_s:
                return self._finish(FAILED, "no path")
            return 0.0, 0.4 * self.p.max_angular  # turn on the spot to observe more of the map
        return self.follower.command(pose)

    def _potential_field(self, pose, angles, ranges, distance):
        x, y, theta = pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        c, s = math.cos(theta), math.sin(theta)
        attraction = ((c * dx + s * dy) / distance, (-s * dx + c * dy) / distance)
        heading, _ = potential_field(attraction, scan_to_points(angles, ranges, self.p.pf_influence_m),
                                     self.p.pf_influence_m, self.p.pf_repulsion_gain * ray_spacing(angles))
        omega = _clamp(self.p.heading_gain * heading, self.p.max_angular)
        if abs(heading) > self.p.rotate_in_place_rad:
            return 0.0, omega
        v = self.p.max_linear * math.cos(heading) * max(0.3, min(1.0, distance / self.p.slow_radius_m))
        return v, omega
