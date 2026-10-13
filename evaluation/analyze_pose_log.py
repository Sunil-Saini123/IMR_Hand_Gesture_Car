"""Plot localization error from CSV columns gt_x,gt_y,odom_x,odom_y and optional ekf_x,ekf_y."""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="evaluation/plots/pose_error.png")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = {"gt_x","gt_y","odom_x","odom_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    df["odom_error"] = ((df.gt_x-df.odom_x)**2 + (df.gt_y-df.odom_y)**2)**0.5
    plt.plot(df.index, df.odom_error, label="Odometry")
    if {"ekf_x","ekf_y"}.issubset(df.columns):
        df["ekf_error"] = ((df.gt_x-df.ekf_x)**2 + (df.gt_y-df.ekf_y)**2)**0.5
        plt.plot(df.index, df.ekf_error, label="EKF")
    plt.xlabel("Sample"); plt.ylabel("Position error (m)")
    plt.title("Localization error"); plt.grid(True); plt.legend()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=160)
    print("Saved", args.out)


if __name__ == "__main__": main()
