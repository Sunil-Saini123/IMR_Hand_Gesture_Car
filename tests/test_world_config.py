"""The Webots world and config.yaml describe the same arena."""
import math
import pytest
from robot_core.config import load_config
from robot_core.occupancy_grid import OccupancyGrid
from robot_core.planning import plan_path
from scan_sim import epuck_pose, world_obstacles, world_solids

CFG = load_config()


def test_solids_rest_on_the_floor():
    # Catches Y-up coordinates in this Z-up (ENU) world: everything must sit on z = 0.
    for solid in world_solids():
        z, geom = solid["translation"][2], solid["geometry"]
        height = geom[3] if geom[0] == "box" else geom[1]
        assert math.isclose(z - height / 2, 0.0, abs_tol=1e-6), solid["name"]


def test_landmarks_and_start_pose_match_config():
    posts = sorted((s["translation"][:2], s["geometry"][2]) for s in world_solids()
                   if s["name"].startswith("landmark"))
    assert [p for p, _ in posts] == sorted(tuple(p) for p in CFG["localization"]["landmarks"])
    assert all(math.isclose(r, CFG["localization"]["landmark_radius_m"]) for _, r in posts)
    x, y, yaw = epuck_pose()
    sx, sy, syaw = CFG["robot"]["start_pose"]
    assert math.isclose(x, sx) and math.isclose(y, sy) and math.isclose(yaw, syaw, abs_tol=1e-6)


def rasterized_world():
    grid = OccupancyGrid(CFG["mapping"]["width_cells"], CFG["mapping"]["height_cells"],
                         CFG["mapping"]["resolution_m_per_cell"])
    boxes, circles = world_obstacles()
    for r in range(grid.height):
        for c in range(grid.width):
            x, y = grid.grid_to_world(r, c)
            inside = any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes) or \
                any(math.hypot(x - cx, y - cy) <= cr for cx, cy, cr in circles)
            grid.log_odds[r, c] = 4.0 if inside else -4.0
    return grid


@pytest.fixture(scope="module")
def world_grid():
    return rasterized_world()


def test_goals_are_clear_of_obstacles():
    boxes, circles = world_obstacles()
    margin = CFG["robot"]["body_radius_m"] + CFG["navigation"]["inflation_m"]
    for name, (gx, gy) in CFG["goals"].items():
        for x0, y0, x1, y1 in boxes:
            gap = math.hypot(max(x0 - gx, 0, gx - x1), max(y0 - gy, 0, gy - y1))
            assert gap > margin, (name, (x0, y0, x1, y1))
        for cx, cy, r in circles:
            assert math.hypot(gx - cx, gy - cy) - r > margin, (name, (cx, cy))


def test_every_trial_leg_has_a_path_in_the_true_map(world_grid):
    names = ["HOME"] + CFG["trials"]["sequence"]
    for start, goal in zip(names, names[1:]):
        path = plan_path(world_grid, CFG["goals"][start], CFG["goals"][goal],
                         CFG["navigation"]["occupied_threshold"], CFG["navigation"]["inflation_m"])
        assert path, f"{start} -> {goal}"
