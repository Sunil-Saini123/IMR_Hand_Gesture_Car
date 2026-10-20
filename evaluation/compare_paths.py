"""Summarize navigation trials from data/path_trials.csv (written by the controller).

Required columns: method, path_length_m, time_s, success. goal_name, collisions and
final_error_m are reported when present. --plot saves evaluation/plots/path_comparison.png.
"""
import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS = REPO_ROOT / "data" / "path_trials.csv"


def is_success(row):
    return row.get("success", "").strip().lower() in ("true", "1", "yes")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize(rows):
    """Per-method statistics; time and path length average over successful trials only."""
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    summary = {}
    for method, trials in groups.items():
        successes = [r for r in trials if is_success(r)]
        summary[method] = {
            "trials": len(trials),
            "success_rate": len(successes) / len(trials),
            "mean_path_length_m": mean([float(r["path_length_m"]) for r in successes if r.get("path_length_m")]),
            "mean_time_s": mean([float(r["time_s"]) for r in successes if r.get("time_s")]),
            "collisions": sum(int(float(r["collisions"])) for r in trials if r.get("collisions")),
            "mean_final_error_m": mean([float(r["final_error_m"]) for r in successes if r.get("final_error_m")]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, default=TRIALS)
    parser.add_argument("--plot", action="store_true", help="save a bar chart to evaluation/plots/")
    args = parser.parse_args()
    if not args.csv.exists():
        print(f"{args.csv} not found. Run GOAL-mode trials first (evaluation/run_trials.py), or create it "
              "with method,path_length_m,time_s,success columns.")
        return
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No trials recorded yet.")
        return
    summary = summarize(rows)
    for method, s in summary.items():
        print(f"{method}: trials={s['trials']}, success_rate={s['success_rate']:.1%}, "
              f"collisions={s['collisions']}")
        print(f"  successful trials: mean path_length_m {s['mean_path_length_m']:.3f}, "
              f"mean time_s {s['mean_time_s']:.1f}, mean final_error_m {s['mean_final_error_m']:.3f}")
    if "goal_name" in rows[0]:
        print("Per goal (successes/trials):")
        for goal in sorted({r["goal_name"] for r in rows}):
            cells = []
            for method in summary:
                trials = [r for r in rows if r["method"] == method and r["goal_name"] == goal]
                if trials:
                    cells.append(f"{method} {sum(map(is_success, trials))}/{len(trials)}")
            print(f"  {goal}: " + ", ".join(cells))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods = list(summary)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for ax, key, label in zip(axes, ("success_rate", "mean_path_length_m", "mean_time_s"),
                                  ("Success rate", "Mean path length (m)", "Mean time (s)")):
            ax.bar(methods, [summary[m][key] for m in methods], color=["#4C72B0", "#DD8452"][:len(methods)])
            ax.set_title(label)
            ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()
        out = REPO_ROOT / "evaluation" / "plots" / "path_comparison.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        print("Saved", out)


if __name__ == "__main__":
    main()
