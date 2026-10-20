"""Render a run's occupancy map with landmarks, goals, driven paths and the A* plans.

Usage: python evaluation/plot_map.py [data/runs/<stamp>]   (default: newest run with a map)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from robot_core.config import REPO_ROOT, load_config
from robot_core.occupancy_grid import OccupancyGrid


def first_plan_per_goal(plans_path):
    """The initial plan of every goal segment (consecutive plans with the same goal)."""
    if not plans_path.exists():
        return []
    segments, current = [], None
    for line in plans_path.read_text(encoding="utf-8").splitlines():
        plan = json.loads(line)
        if plan["goal_name"] != current and plan["path"]:
            segments.append(plan)
            current = plan["goal_name"]
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation" / "plots" / "map.png")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        maps = sorted((REPO_ROOT / "data" / "runs").glob("*/occupancy.npz"))
        if not maps:
            raise SystemExit("No saved map found; run the simulation first (logs go to data/runs/).")
        run_dir = maps[-1].parent
    cfg = load_config()
    grid = OccupancyGrid.load(run_dir / "occupancy.npz")
    res, w, h = grid.resolution, grid.width, grid.height
    extent = [(-w / 2 - 0.5) * res, (w / 2 - 0.5) * res, (-h / 2 + 0.5) * res, (h / 2 + 0.5) * res]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(grid.probabilities(), cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent)
    log = run_dir / "pose_log.csv"
    if log.exists():
        df = pd.read_csv(log)
        ax.plot(df.gt_x, df.gt_y, color="tab:green", lw=1.2, label="Driven path (ground truth)")
        ax.plot(df.ekf_x, df.ekf_y, color="tab:orange", lw=0.8, ls="--", label="EKF estimate")
    for i, plan in enumerate(first_plan_per_goal(run_dir / "plans.jsonl")):
        xs, ys = zip(*plan["path"])
        ax.plot(xs, ys, color="tab:blue", lw=0.9, alpha=0.8, label="Initial A* plan" if i == 0 else None)
    lx, ly = zip(*cfg["localization"]["landmarks"])
    ax.scatter(lx, ly, s=60, marker="o", facecolors="none", edgecolors="gold", linewidths=2, label="Landmarks")
    for name, (gx, gy) in cfg["goals"].items():
        ax.scatter([gx], [gy], marker="*", s=120, color="tab:red")
        ax.annotate(name, (gx, gy), textcoords="offset points", xytext=(6, 6), color="tab:red")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Occupancy map ({run_dir.name})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
