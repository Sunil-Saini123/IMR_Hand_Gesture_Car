import numpy as np
from robot_core.planning import astar, path_length


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
