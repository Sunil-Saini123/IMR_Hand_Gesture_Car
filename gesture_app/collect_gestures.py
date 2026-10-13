"""Record MediaPipe landmarks to CSV. Press 1..5 to label visible hand samples."""
import csv, os
import cv2
import mediapipe as mp

OUT = os.path.join("data", "gesture_samples.csv")
LABELS = ["STOP", "FORWARD", "LEFT", "RIGHT", "BACKWARD"]


def main():
    os.makedirs("data", exist_ok=True)
    new_file = not os.path.exists(OUT)
    cap = cv2.VideoCapture(0)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x","y","z")])
        with mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0) as hands:
            label = "STOP"
            print("1=STOP, 2=FORWARD, 3=LEFT, 4=RIGHT, 5=BACKWARD, Q=quit")
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(1) & 0xff
                if ord("1") <= key <= ord("5"): label = LABELS[key-ord("1")]
                if key in (ord("q"), ord("Q")): break
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    row = [label]
                    for point in hand.landmark: row.extend([point.x, point.y, point.z])
                    writer.writerow(row)
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Label: {label} | 1-5 change label | Q quit", (10,25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.imshow("Collect Gestures", frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
