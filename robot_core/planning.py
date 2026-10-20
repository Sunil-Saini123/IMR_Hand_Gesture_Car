"""A* on a 2D occupancy-probability grid."""
from collections import deque
import heapq
import math
import numpy as np

MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
         (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(grid, start, goal, occupied_threshold=0.65):
    """Shortest 8-connected path of (row, col) cells; [] if there is none.

    Cells with value >= occupied_threshold are obstacles, and diagonal moves may
    not cut the corner of an obstacle.
    """
    grid = np.asarray(grid)
    rows, cols = grid.shape
    start, goal = tuple(start), tuple(goal)
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols):
            return []
    free = (grid < occupied_threshold).tolist()
    if not free[start[0]][start[1]] or not free[goal[0]][goal[1]]:
        return []
    gr, gc = goal
    frontier = [(math.hypot(gr - start[0], gc - start[1]), 0.0, start)]
    parent = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, g, cur = heapq.heappop(frontier)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        if g > cost[cur]:
            continue
        r, c = cur
        for dr, dc, step in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr][nc]:
                continue
            if dr and dc and not (free[r + dr][c] and free[r][c + dc]):
                continue
            new_cost = g + step
            if new_cost < cost.get((nr, nc), math.inf):
                cost[(nr, nc)] = new_cost
                parent[(nr, nc)] = cur
                heapq.heappush(frontier, (new_cost + math.hypot(gr - nr, gc - nc), new_cost, (nr, nc)))
    return []


def path_length(path):
    return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:]))


def inflate(occupied, radius_cells):
    """Grow a boolean obstacle mask by a disk of radius_cells (configuration space)."""
    occupied = np.asarray(occupied, dtype=bool)
    out = occupied.copy()
    rows, cols = occupied.shape
    r = int(radius_cells)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if (dr == 0 and dc == 0) or dr * dr + dc * dc > r * r:
                continue
            out[max(0, dr):rows + min(0, dr), max(0, dc):cols + min(0, dc)] |= \
                occupied[max(0, -dr):rows + min(0, -dr), max(0, -dc):cols + min(0, -dc)]
    return out


def escape_route(blocked, occupied, start, max_cells):
    """Fewest-step route from start, over cells that are not occupied, to the nearest
    cell outside the inflated zone (searching at most max_cells away); [] if none."""
    rows, cols = blocked.shape
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if not blocked[cur]:
            route = []
            while cur is not None:
                route.append(cur)
                cur = parent[cur]
            return route[::-1]
        for dr, dc, _ in MOVES:
            nxt = (cur[0] + dr, cur[1] + dc)
            if (0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in parent and not occupied[nxt]
                    and max(abs(nxt[0] - start[0]), abs(nxt[1] - start[1])) <= max_cells):
                parent[nxt] = cur
                queue.append(nxt)
    return []


def plan_path(grid, start_xy, goal_xy, occupied_threshold=0.65, inflation_m=0.08):
    """A* on an OccupancyGrid between world points; returns world waypoints or [].

    Obstacles are inflated by inflation_m and unknown cells (p = 0.5) count as
    free. A robot that has ended up inside the inflated zone (e.g. after stopping
    close to a wall) first takes the shortest way out of it; an inflated goal
    cell makes planning fail.
    """
    occupied = grid.probabilities() >= occupied_threshold
    radius = int(math.ceil(inflation_m / grid.resolution))
    blocked = inflate(occupied, radius)
    start = grid.world_to_grid(*start_xy)
    goal = grid.world_to_grid(*goal_xy)
    if not (grid.in_bounds(*start) and grid.in_bounds(*goal)):
        return []
    lead_in = []
    if blocked[start]:
        route = escape_route(blocked, occupied, start, radius + 2)
        if not route:
            return []
        lead_in, start = route[:-1], route[-1]
    cells = astar(blocked.astype(np.float32), start, goal, occupied_threshold=0.5)
    if not cells:
        return []
    return [grid.grid_to_world(r, c) for r, c in lead_in + cells]
