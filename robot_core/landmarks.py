"""Detect known cylindrical landmark posts in a planar LiDAR scan."""
from dataclasses import dataclass
import math
import numpy as np
from .behaviors import ray_spacing


@dataclass(frozen=True)
class LandmarkObservation:
    index: int      # position in the landmark list
    range: float    # robot centre -> post centre (m)
    bearing: float  # robot frame (rad), counter-clockwise positive


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def detect_landmarks(pose, angles, ranges, landmarks, radius, max_range,
                     range_gate=0.3, bearing_gate=0.25, min_contrast=0.1):
    """Look for each mapped post near where the current pose estimate predicts it.

    Inside the predicted bearing window, the closest return within range_gate of
    the predicted range seeds a cluster of neighbouring rays that hit the same
    post (all closer than r_min + radius). The cluster counts as a post only if
    it is no wider than a post of that radius would be and the rays on both
    sides are at least min_contrast farther away, which rejects walls and box
    faces. Data association is by prediction, so the pose estimate must already
    be roughly right; the EKF's Mahalanobis gate removes the remaining outliers.
    """
    x, y, theta = pose
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    n = len(ranges)
    if n < 3:
        return []
    step = ray_spacing(angles)
    full_circle = abs(n * step - 2 * math.pi) < step / 2
    finite = np.isfinite(ranges)
    observations = []
    for index, (lx, ly) in enumerate(landmarks):
        dx, dy = lx - x, ly - y
        distance = math.hypot(dx, dy)
        predicted_range = distance - radius
        if predicted_range <= 0.02 or predicted_range > max_range - 0.05:
            continue
        predicted_bearing = _wrap(math.atan2(dy, dx) - theta)
        half_width = math.asin(min(1.0, radius / distance))
        offset = np.abs(_wrap(angles - predicted_bearing))
        candidates = np.flatnonzero((offset <= half_width + bearing_gate) & finite
                                    & (np.abs(ranges - predicted_range) <= range_gate))
        if candidates.size == 0:
            continue
        seed = int(candidates[np.argmin(ranges[candidates])])
        r_min = ranges[seed]
        limit = r_min + radius + 0.01

        def neighbour(i, direction):
            j = i + direction
            if full_circle:
                return j % n
            return j if 0 <= j < n else None

        cluster = [seed]
        edges = []
        for direction in (-1, 1):
            i = seed
            while True:
                j = neighbour(i, direction)
                if j is None or j == seed:
                    edges.append(None)
                    break
                if finite[j] and ranges[j] < limit:
                    cluster.append(j)
                    i = j
                    if len(cluster) > n // 4:
                        break
                else:
                    edges.append(j)
                    break
        expected_width = 2 * math.asin(min(1.0, radius / (r_min + radius)))
        if len(cluster) * step > expected_width + 2 * step:
            continue
        if len(edges) != 2 or any(e is not None and finite[e] and ranges[e] < limit + min_contrast
                                  for e in edges):
            continue
        bearing = math.atan2(np.sin(angles[cluster]).mean(), np.cos(angles[cluster]).mean())
        observations.append(LandmarkObservation(index, float(r_min + radius), float(bearing)))
    return observations
