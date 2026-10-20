import numpy as np
from robot_core.occupancy_grid import OccupancyGrid
from robot_core.planning import astar, inflate, path_length, plan_path


def test_astar_finds_path():
    grid = np.zeros((10,10), dtype=float)
    grid[4,1:8] = 1.0
    grid[4,5] = 0.0
    path = astar(grid, (1,1), (8,8))
    assert path and path[0] == (1,1) and path[-1] == (8,8)
    assert path_length(path) > 0


def test_astar_rejects_blocked_goal():
    grid = np.zeros((5,5), dtype=float)
    grid[4,4] = 1.0
    assert astar(grid, (0,0), (4,4)) == []


def test_inflate_grows_a_disk():
    occupied = np.zeros((9, 9), dtype=bool)
    occupied[4, 4] = True
    grown = inflate(occupied, 2)
    assert grown[4, 2] and grown[2, 4] and grown[5, 5]
    assert not grown[2, 2] and not grown[4, 1]


def wall_grid():
    grid = OccupancyGrid(width=60, height=60, resolution=0.05)
    for y in np.arange(-0.6, 0.61, 0.05):  # wall along x = 0 from y = -0.6 to 0.6
        grid.log_odds[grid.world_to_grid(0.0, y)] = 4.0
    return grid


def test_plan_path_goes_around_inflated_wall():
    grid = wall_grid()
    path = plan_path(grid, (-0.8, 0.0), (0.8, 0.0), inflation_m=0.1)
    assert path and np.allclose(path[0], (-0.8, 0.0)) and np.allclose(path[-1], (0.8, 0.0))
    crossing = [p for p in path if abs(p[0]) < 0.05]
    assert crossing and all(abs(y) > 0.6 + 0.1 - 1e-9 for _, y in crossing)
    assert path_length(path) > 1.6


def test_plan_path_from_inside_inflation_and_to_blocked_goal():
    grid = wall_grid()
    assert plan_path(grid, (-0.05, 0.0), (-0.8, 0.0), inflation_m=0.1)
    assert plan_path(grid, (-0.8, 0.0), (0.05, 0.0), inflation_m=0.1) == []
    assert len(plan_path(grid, (-0.8, 0.0), (-0.8, 0.0))) == 1
