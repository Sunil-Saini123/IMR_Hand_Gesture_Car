import json
import socket
import time
import pytest
from robot_core.commands import Command, CommandLink, parse_command


def test_parse_validates_and_clamps():
    cmd = parse_command({"command": "forward", "speed": 3, "steering": -9, "mode": "goal",
                         "goal": [1, "2.5"], "goal_name": "A", "planner": "potential_field",
                         "log_trial": False})
    assert cmd == Command("FORWARD", 1.0, -1.0, "GOAL", (1.0, 2.5), "A", "potential_field", False)


def test_parse_falls_back_to_safe_defaults():
    cmd = parse_command({"command": "JUMP", "speed": "fast", "mode": "TURBO", "goal": [1],
                         "planner": "rrt"})
    assert (cmd.command, cmd.speed, cmd.mode, cmd.goal, cmd.planner) == ("STOP", 0.0, "TELEOP", None, None)
    assert parse_command(["FORWARD"]) is None
    assert parse_command({"speed": float("nan")}).speed == 0.0


@pytest.fixture
def link():
    endpoint = CommandLink("127.0.0.1", 0, timeout_s=0.5)
    yield endpoint
    endpoint.close()


def send(sock, port, packet):
    sock.sendto(json.dumps(packet).encode("utf-8"), ("127.0.0.1", port))


def test_poll_drains_queue_and_keeps_newest(link):
    port = link.sock.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        for command in ("LEFT", "RIGHT", "FORWARD"):
            send(client, port, {"command": command, "speed": 0.4})
        client.sendto(b"not json", ("127.0.0.1", port))
        time.sleep(0.05)
        assert link.poll(10.0).command == "FORWARD"
        assert not link.is_stale(10.4)
        assert link.is_stale(10.6)


def test_status_goes_back_to_sender(link):
    port = link.sock.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(1.0)
        send(client, port, {"command": "STOP"})
        time.sleep(0.05)
        link.poll(0.0)
        link.send_status({"nav_state": "IDLE"})
        assert json.loads(client.recvfrom(1024)[0]) == {"nav_state": "IDLE"}


def test_no_packets_means_stale(link):
    assert link.poll(0.0) == Command()
    assert link.is_stale(0.0)
