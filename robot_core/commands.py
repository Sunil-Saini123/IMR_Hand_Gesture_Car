"""UDP command protocol shared by the gesture app, trial runner and controller."""
from dataclasses import dataclass
import json
import math
import socket

COMMANDS = ("STOP", "NONE", "FORWARD", "BACKWARD", "LEFT", "RIGHT")  # NONE = no hand visible
MODES = ("TELEOP", "ASSISTED", "GOAL")
PLANNERS = ("astar", "potential_field")


@dataclass(frozen=True)
class Command:
    command: str = "STOP"
    speed: float = 0.0
    steering: float = 0.0
    mode: str = "TELEOP"
    goal: tuple[float, float] | None = None
    goal_name: str = ""
    planner: str | None = None
    log_trial: bool = True


def _number(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def parse_command(obj) -> Command | None:
    """Validate a decoded JSON packet; returns None if it is not a JSON object.

    Unknown commands become STOP and unknown modes TELEOP, so a malformed
    packet can only ever make the robot stop.
    """
    if not isinstance(obj, dict):
        return None
    command = str(obj.get("command", "STOP")).upper()
    mode = str(obj.get("mode", "TELEOP")).upper()
    goal = obj.get("goal")
    if isinstance(goal, (list, tuple)) and len(goal) == 2:
        gx, gy = _number(goal[0], None), _number(goal[1], None)
        goal = (gx, gy) if gx is not None and gy is not None else None
    else:
        goal = None
    planner = obj.get("planner")
    return Command(
        command=command if command in COMMANDS else "STOP",
        speed=min(max(_number(obj.get("speed"), 0.0), 0.0), 1.0),
        steering=min(max(_number(obj.get("steering"), 0.0), -1.0), 1.0),
        mode=mode if mode in MODES else "TELEOP",
        goal=goal,
        goal_name=str(obj.get("goal_name", ""))[:32],
        planner=planner if planner in PLANNERS else None,
        log_trial=obj.get("log_trial", True) is not False,
    )


class CommandLink:
    """Non-blocking UDP endpoint with a stale-packet watchdog and status replies."""

    def __init__(self, host, port, timeout_s=0.5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.timeout_s = timeout_s
        self.command = Command()
        self.peer = None
        self.last_packet_time = -math.inf

    def poll(self, now: float) -> Command:
        """Drain every queued datagram and keep the newest valid command.

        Reading only one packet per step would build up lag whenever the
        simulation runs slower than the sender.
        """
        while True:
            try:
                payload, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except ConnectionResetError:
                # Windows reports an earlier status reply that reached a closed port here.
                continue
            except OSError:
                break
            try:
                cmd = parse_command(json.loads(payload.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                continue
            if cmd is not None:
                self.command, self.peer, self.last_packet_time = cmd, addr, now
        return self.command

    def is_stale(self, now: float) -> bool:
        return now - self.last_packet_time > self.timeout_s

    def send_status(self, status: dict):
        if self.peer is None:
            return
        try:
            self.sock.sendto(json.dumps(status).encode("utf-8"), self.peer)
        except OSError:
            pass

    def close(self):
        self.sock.close()
