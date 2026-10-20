"""Drive scripted GOAL-mode trials over UDP, with no webcam needed.

Start the simulation first (robot at HOME), then run:
    python evaluation/run_trials.py [--planners astar potential_field] [--repeats 1]
Each planner drives the legs in config.yaml trials.sequence. The controller logs every
trial to data/path_trials.csv. If a leg fails, the robot is first brought to that
leg's goal with A* (not logged), so every planner drives the same legs.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_core.commands import PLANNERS
from robot_core.config import load_config

SEND_PERIOD_S = 0.05


class Link:
    def __init__(self, host, port):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.status = None

    def exchange(self, packet):
        """Send one packet and return the newest status received so far."""
        self.sock.sendto(json.dumps(packet).encode("utf-8"), self.address)
        try:
            while True:
                self.status = json.loads(self.sock.recvfrom(8192)[0].decode("utf-8"))
        except (OSError, ValueError):  # nothing queued, or Windows reset after the controller closed
            pass
        return self.status

    def hold(self, packet, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.exchange(packet)
            time.sleep(SEND_PERIOD_S)
        return self.status


def drive(link, name, goal, planner, log_trial, timeout_s):
    """Navigate to one goal; returns the controller's result dict, or None on timeout."""
    base = {"mode": "GOAL", "goal": list(goal), "goal_name": name, "planner": planner,
            "log_trial": log_trial, "speed": 0.4, "steering": 0.0}
    status = link.hold({**base, "command": "STOP"}, 0.6)  # re-arm and learn the current trial id
    if status is None:
        raise SystemExit("No status from the controller. Is the simulation running and not paused?")
    start_id = status["trial_id"]
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        status = link.exchange({**base, "command": "FORWARD"})
        result = status.get("last_result") if status else None
        if status and status["trial_id"] > start_id and result and result["goal_name"] == name:
            return result
        time.sleep(SEND_PERIOD_S)
    link.hold({**base, "command": "STOP"}, 0.3)
    return None


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=list(PLANNERS))
    parser.add_argument("--sequence", nargs="+", default=cfg["trials"]["sequence"],
                        help="goal names to visit in order (default: trials.sequence)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--leg-timeout", type=float, default=400.0, help="wall-clock seconds per leg")
    args = parser.parse_args()
    goals = cfg["goals"]
    unknown = [g for g in args.sequence if g not in goals]
    if unknown:
        raise SystemExit(f"Unknown goals {unknown}; known: {list(goals)}")

    link = Link(cfg["network"]["host"], cfg["network"]["port"])
    results = []
    try:
        for repeat in range(args.repeats):
            for planner in args.planners:
                for name in args.sequence:
                    result = drive(link, name, goals[name], planner, True, args.leg_timeout)
                    outcome = result["outcome"] if result else "runner timeout"
                    print(f"[{repeat + 1}/{args.repeats}] {planner:15s} -> {name}: {outcome}"
                          + (f" in {result['time_s']:.1f} s, {result['path_length_m']:.2f} m" if result else ""),
                          flush=True)
                    results.append((planner, name, bool(result and result["success"])))
                    if not (result and result["success"]):
                        recovery = drive(link, name, goals[name], "astar", False, args.leg_timeout)
                        if not (recovery and recovery["success"]):
                            raise SystemExit(f"Could not bring the robot to {name}; stopping the run.")
    finally:
        link.hold({"command": "STOP", "mode": "TELEOP", "speed": 0.0}, 0.5)
    for planner in args.planners:
        legs = [ok for p, _, ok in results if p == planner]
        if legs:
            print(f"{planner}: {sum(legs)}/{len(legs)} legs reached")
    print("Trials were appended to data/path_trials.csv; summarize them with evaluation/compare_paths.py")


if __name__ == "__main__":
    main()
