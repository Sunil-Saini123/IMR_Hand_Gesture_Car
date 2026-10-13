"""Standalone OpenCV + MediaPipe gesture app; communicates with Webots over UDP."""
import json
import socket
import time
import cv2
import mediapipe as mp

HOST, PORT = "127.0.0.1", 5005
STABLE_FRAMES = 4
DEFAULT_SPEED = 0.4


def classify_hand(hand_landmarks):
    """Classify a single index-finger pointing direction.

    The camera image is mirrored before MediaPipe sees it, so horizontal
    directions match the user's view: point left for LEFT and right for RIGHT.
    """
    p = hand_landmarks.landmark

    def distance(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    index_extended = distance(p[8], p[5]) > distance(p[6], p[5]) * 1.15
    other_fingers_curled = all(
        distance(tip, p[5]) < distance(mcp, p[5]) * 1.35
        for tip, mcp in ((p[12], p[9]), (p[16], p[13]), (p[20], p[17]))
    )
    if not index_extended or not other_fingers_curled:
        return "STOP"

    dx = p[8].x - p[5].x
    dy = p[8].y - p[5].y
    if abs(dx) > abs(dy) * 1.2:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy < 0:
        return "FORWARD"
    return "BACKWARD"


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam; check permissions or camera index.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mp_hands = mp.solutions.hands
    draw = mp.solutions.drawing_utils
    candidate, count, active = "STOP", 0, "STOP"
    mode = "TELEOP"
    print("Point index finger: up=FORWARD, down=BACKWARD, left=LEFT, right=RIGHT.")
    print("Open palm/fist=STOP; W/A/S/D keyboard fallback; SPACE stops; M mode; Q quits.")
    try:
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.55, min_tracking_confidence=0.55) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                observed = "STOP"
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    observed = classify_hand(hand)
                    draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if observed == candidate:
                    count += 1
                else:
                    candidate, count = observed, 1
                if count >= STABLE_FRAMES:
                    active = candidate

                key = cv2.waitKey(1) & 0xFF
                override = None
                if key in (ord("w"), ord("W")): override = "FORWARD"
                elif key in (ord("s"), ord("S")): override = "BACKWARD"
                elif key in (ord("a"), ord("A")): override = "LEFT"
                elif key in (ord("d"), ord("D")): override = "RIGHT"
                elif key == 32: override = "STOP"
                elif key in (ord("m"), ord("M")):
                    mode = {"TELEOP":"ASSISTED", "ASSISTED":"GOAL", "GOAL":"TELEOP"}[mode]
                elif key in (ord("q"), ord("Q")): break

                command = override or active
                packet = {"command": command, "speed": 0.0 if command == "STOP" else DEFAULT_SPEED,
                          "steering": 0.0, "mode": mode, "timestamp": time.monotonic()}
                sock.sendto(json.dumps(packet).encode("utf-8"), (HOST, PORT))
                cv2.putText(frame, f"Gesture: {observed} | Command: {command}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.putText(frame, f"Point: up/down/left/right | Open palm/fist: stop | Mode: {mode}",
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
                cv2.imshow("Gesture-Guided Robot", frame)
    finally:
        try:
            sock.sendto(json.dumps({"command":"STOP","speed":0,"mode":mode}).encode(), (HOST, PORT))
        except OSError:
            pass
        cap.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
