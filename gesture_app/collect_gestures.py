"""Record labelled MediaPipe hand landmarks to data/gesture_samples.csv.

Keys: 1-5 choose the label, R starts/pauses recording, Q quits. Only frames seen
while recording is on are saved, so changing gestures between labels does not
produce mislabelled rows.
"""
import argparse
from collections import Counter
import csv
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # see gesture_app.py: saves ~350 MB of memory

import cv2
import mediapipe as mp
from gesture_app.features import LABELS
from robot_core.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "data" / "gesture_samples.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, help="webcam index (default: gesture.camera_index)")
    args = parser.parse_args()
    camera = load_config()["gesture"]["camera_index"] if args.camera is None else args.camera

    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            counts.update(row["label"] for row in csv.DictReader(f))
    new_file = not OUT.exists()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam; check permissions or camera index.")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")])
        with mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0) as hands:
            label, recording = LABELS[0], False
            print("1=STOP, 2=FORWARD, 3=LEFT, 4=RIGHT, 5=BACKWARD, R=record on/off, Q=quit")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(1) & 0xFF
                if ord("1") <= key < ord("1") + len(LABELS):
                    label = LABELS[key - ord("1")]
                elif key in (ord("r"), ord("R")):
                    recording = not recording
                elif key in (ord("q"), ord("Q")):
                    break
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    if recording:
                        row = [label]
                        for point in hand.landmark:
                            row.extend([point.x, point.y, point.z])
                        writer.writerow(row)
                        counts[label] += 1
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                state = "REC" if recording else "paused"
                cv2.putText(frame, f"Label: {label} [{state}] | samples: {counts[label]}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (0, 255, 0), 2)
                cv2.putText(frame, " ".join(f"{name}:{counts[name]}" for name in LABELS), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(frame, "1-5 label | R record | Q quit", (10, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow("Collect Gestures", frame)
    cap.release()
    cv2.destroyAllWindows()
    print("Saved samples to", OUT, dict(counts))


if __name__ == "__main__":
    main()
