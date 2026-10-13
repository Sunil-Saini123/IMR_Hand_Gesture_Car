"""Log-odds occupancy grid updated from 2D range rays."""
import math
import numpy as np


class OccupancyGrid:
    def __init__(self, width=120, height=120, resolution=0.05,
                 hit=0.85, miss=-0.4, clamp_min=-5.0, clamp_max=5.0):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.width, self.height, self.resolution = width, height, resolution
        self.hit, self.miss = hit, miss
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
        self.log_odds = np.zeros((height, width), dtype=np.float32)

    def world_to_grid(self, x, y):
        col = int(round(x / self.resolution + self.width / 2))
        row = int(round(self.height / 2 - y / self.resolution))
        return row, col

    @staticmethod
    def line(r0, c0, r1, c1):
        dx, dy = abs(c1-c0), abs(r1-r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err, r, c = dx-dy, r0, c0
        while True:
            yield r, c
            if r == r1 and c == c1:
                break
            e2 = 2*err
            if e2 > -dy:
                err -= dy
                c += sx
            if e2 < dx:
                err += dx
                r += sy

    def update_ray(self, pose, angle, distance, max_range=3.0):
        x, y, theta = pose
        usable = min(max(distance, 0.0), max_range)
        ex = x + usable * math.cos(theta + angle)
        ey = y + usable * math.sin(theta + angle)
        r0, c0 = self.world_to_grid(x, y)
        r1, c1 = self.world_to_grid(ex, ey)
        cells = list(self.line(r0, c0, r1, c1))
        for r, c in cells[:-1]:
            if 0 <= r < self.height and 0 <= c < self.width:
                self.log_odds[r, c] = np.clip(self.log_odds[r, c] + self.miss,
                                              self.clamp_min, self.clamp_max)
        if distance < max_range and cells:
            r, c = cells[-1]
            if 0 <= r < self.height and 0 <= c < self.width:
                self.log_odds[r, c] = np.clip(self.log_odds[r, c] + self.hit,
                                              self.clamp_min, self.clamp_max)

    def probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))
