"""Summarize path trials from data/path_trials.csv."""
import csv, os


def main():
    path = "data/path_trials.csv"
    if not os.path.exists(path):
        print("Create data/path_trials.csv with method,path_length_m,time_s,success columns.")
        return
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): groups.setdefault(row["method"], []).append(row)
    for method, rows in groups.items():
        success = sum(r.get("success","").lower() in ("true","1","yes") for r in rows) / len(rows)
        print(f"{method}: trials={len(rows)}, success_rate={success:.1%}")
        for col in ("path_length_m", "time_s"):
            vals = [float(r[col]) for r in rows if r.get(col)]
            if vals: print(f"  mean {col}: {sum(vals)/len(vals):.3f}")


if __name__ == "__main__": main()
