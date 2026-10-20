"""Localization error of odometry and the EKF against Supervisor ground truth.

Usage: python evaluation/analyze_pose_log.py [path/to/pose_log.csv]
Defaults to the newest data/runs/*/pose_log.csv. Needs columns gt_x, gt_y, odom_x, odom_y;
ekf_x, ekf_y (and ekf_sx, ekf_sy for the 2-sigma bound) are optional.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS = REPO_ROOT / "evaluation" / "plots"


def latest_run_file(name):
    files = sorted((REPO_ROOT / "data" / "runs").glob(f"*/{name}"))
    return files[-1] if files else None


def summarize(label, error):
    print(f"{label:9s} mean {error.mean():.3f} m | RMSE {np.sqrt((error ** 2).mean()):.3f} m | "
          f"max {error.max():.3f} m | final {error.iloc[-1]:.3f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, help="pose log (default: newest run)")
    parser.add_argument("--out-dir", type=Path, default=PLOTS)
    args = parser.parse_args()
    path = args.csv or latest_run_file("pose_log.csv")
    if path is None:
        raise SystemExit("No pose log found; run the simulation first (logs go to data/runs/).")
    df = pd.read_csv(path)
    required = {"gt_x", "gt_y", "odom_x", "odom_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    has_ekf = {"ekf_x", "ekf_y"}.issubset(df.columns)
    time = df["t"] if "t" in df.columns else pd.Series(np.arange(len(df)), name="sample")
    df["odom_error"] = np.hypot(df.gt_x - df.odom_x, df.gt_y - df.odom_y)
    print(f"{path} ({len(df)} samples, {time.iloc[-1] - time.iloc[0]:.1f} {'s' if 't' in df else 'samples'})")
    summarize("Odometry", df.odom_error)
    if has_ekf:
        df["ekf_error"] = np.hypot(df.gt_x - df.ekf_x, df.gt_y - df.ekf_y)
        summarize("EKF", df.ekf_error)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(time, df.odom_error, label="Odometry")
    if has_ekf:
        ax.plot(time, df.ekf_error, label="EKF")
        if {"ekf_sx", "ekf_sy"}.issubset(df.columns):
            ax.plot(time, 2 * np.hypot(df.ekf_sx, df.ekf_sy), "k--", lw=0.8, label="EKF 2σ bound")
    ax.set_xlabel("Simulation time (s)" if "t" in df else "Sample")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Localization error vs. ground truth")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "pose_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(df.gt_x, df.gt_y, "k", lw=1.6, label="Ground truth")
    ax.plot(df.odom_x, df.odom_y, lw=1, label="Odometry")
    if has_ekf:
        ax.plot(df.ekf_x, df.ekf_y, lw=1, label="EKF")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectories")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "trajectory.png", dpi=160)
    plt.close(fig)
    print("Saved", args.out_dir / "pose_error.png", "and", args.out_dir / "trajectory.png")


if __name__ == "__main__":
    main()
