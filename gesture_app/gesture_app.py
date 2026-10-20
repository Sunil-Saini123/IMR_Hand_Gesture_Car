"""Standalone OpenCV + MediaPipe gesture app; communicates with Webots over UDP.

Run from the repository root:  python gesture_app/gesture_app.py [--classifier rules|rf|auto]
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Must be set before numpy loads: OpenBLAS otherwise reserves buffers for every CPU thread
# (~400 MB on a 12-thread laptop), which is enough to exhaust memory next to Webots.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import mediapipe as mp
from gesture_app.features import GestureFilter, GestureRecognizer, landmarks_to_array
from robot_core.config import REPO_ROOT, load_config

MODES = ["TELEOP", "ASSISTED", "GOAL"]
PLANNERS = ["astar", "potential_field"]
DRIVE_KEYS = {"w": "FORWARD", "s": "BACKWARD", "a": "LEFT", "d": "RIGHT", " ": "STOP"}
GREEN, CYAN, WHITE, ORANGE = (0, 255, 0), (255, 255, 0), (255, 255, 255), (0, 165, 255)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", choices=["auto", "rules", "rf"],
                        help="gesture classifier (default: gesture.classifier in config.yaml)")
    parser.add_argument("--model", help="Random Forest model path (default: gesture.model_path)")
    parser.add_argument("--camera", type=int, help="webcam index (default: gesture.camera_index)")
    return parser.parse_args()


def put(frame, text, row, color=WHITE, scale=0.5):
    # Outline from 1-px offset copies: OpenCV 5 widens glyphs with thickness, so a thicker black
    # copy would no longer line up with the coloured text.
    x, y = 10, 24 + 22 * row
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    args = parse_args()
    cfg = load_config()
    gcfg, net = cfg["gesture"], cfg["network"]
    host, port = net["host"], net["port"]
    recognizer = GestureRecognizer(args.classifier or gcfg["classifier"],
                                   REPO_ROOT / (args.model or gcfg["model_path"]), gcfg["min_confidence"])
    goals = [(name, (float(x), float(y))) for name, (x, y) in cfg["goals"].items()]
    goal_index, planner, mode = 0, cfg["navigation"]["planner"], "TELEOP"

    cap = cv2.VideoCapture(gcfg["camera_index"] if args.camera is None else args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gcfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gcfg["height"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam; check permissions or camera index.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    mp_hands = mp.solutions.hands
    draw = mp.solutions.drawing_utils
    gestures = GestureFilter(gcfg["vote_window_frames"], gcfg["stable_frames"])
    key_command, key_until = None, 0.0
    status, status_time = None, 0.0
    print(f"Classifier: {recognizer.name}. Point index finger: up=FORWARD, down=BACKWARD, left=LEFT, right=RIGHT.")
    print("Open palm/fist=STOP; W/A/S/D keyboard fallback; SPACE stops; M mode; 1-5 goal; P planner; Q quits.")
    try:
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.55, min_tracking_confidence=0.55) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                observed, confidence = "NONE", 0.0
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    observed, confidence = recognizer.predict(landmarks_to_array(hand))
                    draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                previous = gestures.active
                active = gestures.update(observed)
                if mode == "GOAL" and active != previous and active in ("LEFT", "RIGHT"):
                    goal_index = (goal_index + (1 if active == "RIGHT" else -1)) % len(goals)

                now = time.monotonic()
                key = cv2.waitKey(1) & 0xFF
                ch = chr(key).lower() if key < 128 else ""
                if ch == "q":
                    break
                if ch == "m":
                    mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                    key_command = None
                elif ch == "p":
                    planner = PLANNERS[(PLANNERS.index(planner) + 1) % len(PLANNERS)]
                elif ch.isdigit() and 1 <= int(ch) <= len(goals):
                    goal_index = int(ch) - 1
                elif mode == "GOAL" and ch in ("a", "d"):
                    goal_index = (goal_index + (1 if ch == "d" else -1)) % len(goals)
                elif ch in DRIVE_KEYS:
                    key_command, key_until = DRIVE_KEYS[ch], now + gcfg["key_hold_seconds"]

                command = key_command if key_command and now < key_until else active
                goal_name, goal = goals[goal_index]
                packet = {"command": command,
                          "speed": 0.0 if command in ("STOP", "NONE") else gcfg["default_speed"],
                          "steering": 0.0, "mode": mode, "goal": list(goal), "goal_name": goal_name,
                          "planner": planner, "timestamp": now}
                sock.sendto(json.dumps(packet).encode("utf-8"), (host, port))
                try:
                    while True:
                        status, status_time = json.loads(sock.recvfrom(8192)[0].decode("utf-8")), now
                except (OSError, ValueError):  # nothing queued, or Windows reset after the controller closed
                    pass

                put(frame, f"Gesture: {observed} ({confidence:.2f}) | Command: {command} | Mode: {mode}", 0, GREEN)
                if mode == "GOAL":
                    put(frame, f"Goal {goal_index + 1}/{len(goals)}: {goal_name} ({goal[0]:.2f}, {goal[1]:.2f})"
                               f" | Planner: {planner}", 1, CYAN)
                    put(frame, "Point up: go | palm/fist: pause | point left/right: change goal", 2, CYAN, 0.42)
                else:
                    put(frame, "Point up/down/left/right to drive | open palm or fist: stop", 1, CYAN, 0.42)
                if status and now - status_time < 1.0:
                    x, y, _ = status["pose"]
                    nav = status["nav_state"] + (f" ({status['nav_reason']})" if status["nav_reason"] else "")
                    put(frame, f"Robot: {nav} | EKF ({x:.2f}, {y:.2f}) | landmarks {status['landmarks']}",
                        3, WHITE if status["localization_ok"] else ORANGE, 0.45)
                else:
                    put(frame, "Robot: no status (is the simulation running?)", 3, ORANGE, 0.45)
                put(frame, "W/A/S/D drive | SPACE stop | M mode | 1-5 goal | P planner | Q quit",
                    4, WHITE, 0.4)
                cv2.imshow("Gesture-Guided Robot", frame)
    finally:
        try:
            sock.sendto(json.dumps({"command": "STOP", "speed": 0, "mode": mode}).encode(), (host, port))
        except OSError:
            pass
        cap.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
