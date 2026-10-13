"""Webots controller: UDP command receiver, e-puck motor control and simple safety."""
from __future__ import annotations
import json
import socket
import time
from controller import Robot

HOST, PORT = "127.0.0.1", 5005
COMMAND_TIMEOUT_S = 0.5
MAX_WHEEL_RAD_S = 6.28
PROXIMITY_STOP_THRESHOLD = 85.0  # Must be tuned for your installed e-puck model.


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)

    sensors = []
    for i in range(8):
        try:
            sensor = robot.getDevice(f"ps{i}")
            sensor.enable(timestep)
            sensors.append(sensor)
        except Exception:
            pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(False)
    last_packet = 0.0
    command = {"command": "STOP", "speed": 0.0, "steering": 0.0, "mode": "TELEOP"}
    print(f"Robot controller listening at {HOST}:{PORT}")

    try:
        while robot.step(timestep) != -1:
            now = time.monotonic()
            try:
                payload, _ = sock.recvfrom(2048)
                obj = json.loads(payload.decode("utf-8"))
                if isinstance(obj, dict):
                    command = obj
                    last_packet = now
            except BlockingIOError:
                pass
            except (ValueError, UnicodeDecodeError):
                pass

            stale = now - last_packet > COMMAND_TIMEOUT_S
            gesture = str(command.get("command", "STOP")).upper()
            speed = clamp(float(command.get("speed", 0.0)), 0.0, 1.0)
            steering = clamp(float(command.get("steering", 0.0)), -1.0, 1.0)
            mode = str(command.get("mode", "TELEOP")).upper()
            front = [sensors[i].getValue() for i in (0, 1, 6, 7)] if len(sensors) >= 8 else []
            obstacle_close = bool(front) and max(front) > PROXIMITY_STOP_THRESHOLD

            if stale or gesture == "STOP":
                vl = vr = 0.0
            elif mode in ("ASSISTED", "GOAL") and obstacle_close and gesture in ("FORWARD", "GOAL"):
                vl = vr = 0.0
            else:
                base = speed * MAX_WHEEL_RAD_S
                if gesture == "BACKWARD":
                    base = -base
                elif gesture in ("LEFT", "RIGHT"):
                    base = 0.0
                elif gesture not in ("FORWARD", "BACKWARD"):
                    base = 0.0
                turn = steering * MAX_WHEEL_RAD_S
                if gesture == "LEFT":
                    turn = max(abs(turn), 0.45 * MAX_WHEEL_RAD_S)
                elif gesture == "RIGHT":
                    turn = -max(abs(turn), 0.45 * MAX_WHEEL_RAD_S)
                vl = clamp(base - turn, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)
                vr = clamp(base + turn, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)

            left.setVelocity(vl)
            right.setVelocity(vr)
    finally:
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        sock.close()


if __name__ == "__main__":
    main()
