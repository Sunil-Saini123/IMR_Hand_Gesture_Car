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

    def grid_to_world(self, row, col):
        """Centre of cell (row, col) in world coordinates."""
        return (col - self.width / 2) * self.resolution, (self.height / 2 - row) * self.resolution

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

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

    def _flat_indices(self, xs, ys):
        cols = np.rint(xs / self.resolution + self.width / 2).astype(int)
        rows = np.rint(self.height / 2 - ys / self.resolution).astype(int)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        return np.unique(rows[inside] * self.width + cols[inside])

    def update_scan(self, pose, angles, ranges, max_range):
        """Vectorized update from a whole scan (angles in the robot frame).

        Each cell receives at most one hit or one miss per scan. Rays with no
        return (inf, or >= max_range) only clear free space; NaN rays are skipped.
        """
        x, y, theta = pose
        angles = np.asarray(angles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
        valid = ~np.isnan(ranges)
        angles, ranges = angles[valid], ranges[valid]
        hit = np.isfinite(ranges) & (ranges < max_range)
        usable = np.clip(np.where(np.isfinite(ranges), ranges, max_range), 0.0, max_range)
        world = theta + angles
        cos, sin = np.cos(world), np.sin(world)

        step = self.resolution / 2
        samples = np.arange(int(math.ceil(max_range / step)) + 1) * step
        free = samples[None, :] < (usable[:, None] - step)
        px = x + samples[None, :] * cos[:, None]
        py = y + samples[None, :] * sin[:, None]
        free_cells = self._flat_indices(px[free], py[free])
        hit_cells = self._flat_indices(x + usable[hit] * cos[hit], y + usable[hit] * sin[hit])
        free_cells = np.setdiff1d(free_cells, hit_cells, assume_unique=True)

        flat = self.log_odds.reshape(-1)
        flat[free_cells] += self.miss
        flat[hit_cells] += self.hit
        np.clip(self.log_odds, self.clamp_min, self.clamp_max, out=self.log_odds)

    def probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def save(self, path):
        np.savez_compressed(path, log_odds=self.log_odds, resolution=self.resolution,
                            width=self.width, height=self.height)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        grid = cls(width=int(data["width"]), height=int(data["height"]),
                   resolution=float(data["resolution"]))
        grid.log_odds = data["log_odds"].astype(np.float32)
        return grid
