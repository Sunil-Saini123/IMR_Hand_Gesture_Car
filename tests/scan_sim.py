"""Test helpers: an analytic 2D LiDAR and a reader for the Webots world's solids."""
from pathlib import Path
import re
import numpy as np

WORLD = Path(__file__).resolve().parents[1] / "webots" / "worlds" / "gesture_arena.wbt"
NUMBER = r"(-?[\d.]+(?:e-?\d+)?)"


def simulate_scan(pose, angles, boxes=(), circles=(), max_range=2.0):
    """Ranges from pose to axis-aligned boxes (xmin, ymin, xmax, ymax) and circles (cx, cy, r).

    Rays that hit nothing within max_range return inf, like a Webots Lidar.
    """
    x, y, theta = pose
    world = theta + np.asarray(angles, dtype=float)
    dx, dy = np.cos(world), np.sin(world)
    best = np.full(world.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for xmin, ymin, xmax, ymax in boxes:
            tx1, tx2 = (xmin - x) / dx, (xmax - x) / dx
            ty1, ty2 = (ymin - y) / dy, (ymax - y) / dy
            near = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            far = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            hit = (near <= far) & (near > 0)
            best = np.where(hit & (near < best), near, best)
        for cx, cy, r in circles:
            fx, fy = x - cx, y - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - r * r)
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            hit = (disc >= 0) & (t > 0)
            best = np.where(hit & (t < best), t, best)
    best[best > max_range] = np.inf
    return best


def world_solids(path=WORLD):
    """Top-level Solids of the world: dicts with name, translation and box size or cylinder."""
    text = Path(path).read_text(encoding="utf-8")
    geometry = {}
    for name, sx, sy, sz in re.findall(rf"DEF (\w+) Box {{\s*size {NUMBER} {NUMBER} {NUMBER}", text):
        geometry[name] = ("box", float(sx), float(sy), float(sz))
    for name, h, r in re.findall(rf"DEF (\w+) Cylinder {{\s*height {NUMBER}\s*radius {NUMBER}", text):
        geometry[name] = ("cylinder", float(h), float(r))
    solids = []
    for match in re.finditer(r"^(?:DEF \w+ )?Solid {", text, re.MULTILINE):
        depth, end = 0, match.end() - 1
        for end in range(match.end() - 1, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                break
        block = text[match.start():end]
        tx, ty, tz = map(float, re.search(rf"translation {NUMBER} {NUMBER} {NUMBER}", block).groups())
        solids.append({"name": re.search(r'^  name "([^"]+)"', block, re.MULTILINE).group(1),
                       "translation": (tx, ty, tz),
                       "geometry": geometry[re.search(r"boundingObject USE (\w+)", block).group(1)]})
    return solids


def world_obstacles(path=WORLD):
    """2D footprints of the world: (boxes, circles) in simulate_scan's format."""
    boxes, circles = [], []
    for solid in world_solids(path):
        (x, y, _), geom = solid["translation"], solid["geometry"]
        if geom[0] == "box":
            boxes.append((x - geom[1] / 2, y - geom[2] / 2, x + geom[1] / 2, y + geom[2] / 2))
        else:
            circles.append((x, y, geom[2]))
    return boxes, circles


def epuck_pose(path=WORLD):
    text = Path(path).read_text(encoding="utf-8")
    block = text[text.index("E-puck {"):]
    x, y, _ = map(float, re.search(rf"translation {NUMBER} {NUMBER} {NUMBER}", block).groups())
    rotation = re.search(rf"\n  rotation {NUMBER} {NUMBER} {NUMBER} {NUMBER}", block)
    yaw = 0.0
    if rotation:
        ax, ay, az, angle = map(float, rotation.groups())
        yaw = angle * np.sign(az) if abs(az) > 0.999 else float("nan")
    return x, y, yaw
