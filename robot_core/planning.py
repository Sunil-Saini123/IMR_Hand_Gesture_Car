"""A* on a 2D occupancy-probability grid."""
import heapq
import math


def astar(grid, start, goal, occupied_threshold=0.65):
    rows, cols = grid.shape
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols):
            return []
    if grid[start] >= occupied_threshold or grid[goal] >= occupied_threshold:
        return []
    frontier = [(0.0, start)]
    parent = {start: None}
    cost = {start: 0.0}
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
             (1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dr, dc, step in moves:
            nxt = (cur[0]+dr, cur[1]+dc)
            r, c = nxt
            if not (0 <= r < rows and 0 <= c < cols) or grid[nxt] >= occupied_threshold:
                continue
            if dr and dc and (grid[cur[0]+dr, cur[1]] >= occupied_threshold or
                              grid[cur[0], cur[1]+dc] >= occupied_threshold):
                continue
            new_cost = cost[cur] + step
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                parent[nxt] = cur
                heapq.heappush(frontier, (new_cost + math.hypot(goal[0]-r, goal[1]-c), nxt))
    return []


def path_length(path):
    return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:]))
