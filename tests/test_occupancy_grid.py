import math
import numpy as np
from robot_core.behaviors import lidar_angles
from robot_core.occupancy_grid import OccupancyGrid
from scan_sim import simulate_scan


def test_world_grid_round_trip():
    grid = OccupancyGrid(width=40, height=40, resolution=0.05)
    for x, y in [(0.0, 0.0), (0.5, -0.25), (-0.95, 0.9)]:
        assert np.allclose(grid.grid_to_world(*grid.world_to_grid(x, y)), (x, y))


def test_scan_marks_wall_occupied_and_space_free():
    grid = OccupancyGrid(width=60, height=60, resolution=0.05)
    angles = lidar_angles(2 * math.pi, 360)
    wall = (1.0, -2.0, 1.2, 2.0)
    ranges = simulate_scan((0.0, 0.0, 0.0), angles, [wall], max_range=1.5)
    grid.update_scan((0.0, 0.0, 0.0), angles, ranges, max_range=1.5)
    p = grid.probabilities()
    assert p[grid.world_to_grid(1.0, 0.0)] > 0.65
    assert p[grid.world_to_grid(0.5, 0.0)] < 0.5
    assert p[grid.world_to_grid(1.3, 0.0)] == 0.5  # behind the wall: never observed


def test_rays_without_returns_only_clear_space():
    grid = OccupancyGrid(width=60, height=60, resolution=0.05)
    angles = lidar_angles(2 * math.pi, 90)
    ranges = np.full(90, np.inf)
    ranges[0] = np.nan
    grid.update_scan((0.0, 0.0, 0.0), angles, ranges, max_range=1.0)
    p = grid.probabilities()
    assert (p <= 0.5).all()
    assert p[grid.world_to_grid(0.9, 0.0)] < 0.5
    assert p[grid.world_to_grid(1.2, 0.0)] == 0.5


def test_save_and_load(tmp_path):
    grid = OccupancyGrid(width=20, height=10, resolution=0.1)
    grid.log_odds[3, 4] = 2.0
    grid.save(tmp_path / "map.npz")
    loaded = OccupancyGrid.load(tmp_path / "map.npz")
    assert (loaded.width, loaded.height, loaded.resolution) == (20, 10, 0.1)
    assert np.array_equal(loaded.log_odds, grid.log_odds)
