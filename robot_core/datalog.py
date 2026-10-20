"""CSV/JSON logging of controller runs and navigation trials."""
import csv
from datetime import datetime
import json
from pathlib import Path

POSE_FIELDS = ["t", "mode", "command", "nav_state", "gt_x", "gt_y", "gt_theta",
               "odom_x", "odom_y", "odom_theta", "ekf_x", "ekf_y", "ekf_theta",
               "ekf_sx", "ekf_sy", "ekf_stheta", "landmarks", "min_range", "v", "omega"]
TRIAL_FIELDS = ["timestamp", "run", "method", "goal_name", "goal_x", "goal_y", "start_x", "start_y",
                "success", "outcome", "time_s", "path_length_m", "collisions", "final_error_m",
                "mean_ekf_error_m"]


def _cell(value):
    return f"{value:.4f}" if isinstance(value, float) else value


class RunLogger:
    """Writes data/runs/<stamp>/{pose_log.csv, plans.jsonl, occupancy.npz, events.log} and
    appends one row per finished navigation trial to data/path_trials.csv."""

    def __init__(self, data_dir, flush_period_s=1.0):
        data_dir = Path(data_dir)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = data_dir / "runs" / stamp
        suffix = 1
        while self.run_dir.exists():
            suffix += 1
            self.run_dir = data_dir / "runs" / f"{stamp}-{suffix}"
        self.run_dir.mkdir(parents=True)
        self.run_name = self.run_dir.name
        self.trials_path = data_dir / "path_trials.csv"
        self.flush_period_s = flush_period_s
        self._last_flush = 0.0
        self._pose_file = open(self.run_dir / "pose_log.csv", "w", newline="", encoding="utf-8")
        self._pose_writer = csv.DictWriter(self._pose_file, POSE_FIELDS)
        self._pose_writer.writeheader()
        self._plans_file = open(self.run_dir / "plans.jsonl", "w", encoding="utf-8")

    def log_pose(self, row, now):
        self._pose_writer.writerow({k: _cell(row.get(k, "")) for k in POSE_FIELDS})
        if now - self._last_flush >= self.flush_period_s:
            self.flush()
            self._last_flush = now

    def event(self, message):
        """Print a controller message and keep it in events.log (the Webots console may be hidden)."""
        print(message, flush=True)
        with open(self.run_dir / "events.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")

    def log_plan(self, t, goal_name, path):
        self._plans_file.write(json.dumps({"t": round(t, 3), "goal_name": goal_name,
                                           "path": [[round(x, 3), round(y, 3)] for x, y in path]}) + "\n")

    def log_trial(self, row):
        new_file = not self.trials_path.exists()
        with open(self.trials_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, TRIAL_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: _cell(row.get(k, "")) for k in TRIAL_FIELDS})

    def save_map(self, grid):
        grid.save(self.run_dir / "occupancy.npz")

    def flush(self):
        self._pose_file.flush()
        self._plans_file.flush()

    def close(self):
        self._pose_file.close()
        self._plans_file.close()
