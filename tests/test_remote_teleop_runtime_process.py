"""Two isolated processes using the production robot/operator runtime services."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

PEER = Path(__file__).parent / "helpers" / "remote_teleop_runtime_service_peer.py"


def _make_certificate(tmp_path: Path) -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the runtime service TLS test")
    certificate = tmp_path / "certificate.pem"
    private_key = tmp_path / "private-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    if os.name != "nt":
        private_key.chmod(0o600)
    return certificate, private_key


class RuntimeProcess:
    def __init__(self, arguments: list[str]) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(PEER), *arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=PEER.parents[2],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._messages: queue.Queue[dict[str, object]] = queue.Queue()
        assert self.process.stdout is not None
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            value = json.loads(line)
            assert isinstance(value, dict)
            self._messages.put(value)

    def receive(self, event: str, timeout: float = 8.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self._messages.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if message.get("event") == event:
                return message
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise AssertionError(
            f"runtime peer did not emit {event!r}; exit={self.process.poll()}, stderr={stderr[-1500:]}"
        )

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command": command}) + "\n")
        self.process.stdin.flush()

    def status(self) -> dict[str, object]:
        self.send("status")
        return self.receive("status")["status"]

    def close(self, *, abrupt: bool = False) -> None:
        if self.process.poll() is not None:
            return
        if abrupt:
            self.process.terminate()
        else:
            try:
                self.send("shutdown")
                self.receive("shutdown", timeout=2.0)
            except (AssertionError, BrokenPipeError):
                self.process.terminate()
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3.0)


def test_two_runtime_services_stream_then_robot_stops_locally_on_operator_loss(
    tmp_path: Path,
) -> None:
    certificate, private_key = _make_certificate(tmp_path)
    robot = RuntimeProcess(
        [
            "robot",
            "--state-root",
            str(tmp_path / "robot-instance"),
            "--certificate",
            str(certificate),
            "--private-key",
            str(private_key),
        ]
    )
    operator: RuntimeProcess | None = None
    try:
        ready = robot.receive("ready")
        operator = RuntimeProcess(
            [
                "operator",
                "--state-root",
                str(tmp_path / "operator-instance"),
                "--control-uri",
                f"wss://127.0.0.1:{ready['control_port']}",
                "--fingerprint",
                str(ready["certificate_fingerprint"]),
                "--pairing-token",
                str(ready["pairing_token"]),
            ]
        )
        started = operator.receive("ready")
        assert started["status"]["state"] == "streaming"
        assert started["paired"]["credential_id"].startswith("operator-")

        deadline = time.monotonic() + 4.0
        robot_status: dict[str, object] = {}
        operator_status: dict[str, object] = {}
        while time.monotonic() < deadline:
            operator.send("browser_heartbeat")
            operator.receive("browser_heartbeat")
            robot_status = robot.status()
            operator_status = operator.status()
            active = robot_status["service"]["active"]
            executed = (
                active["executor"]["counters"].get("action_executed", 0) if isinstance(active, dict) else 0
            )
            if executed >= 1:
                break
            time.sleep(0.02)

        robot_service = robot_status["service"]
        operator_service = operator_status["service"]
        active = robot_service["active"]
        assert active["executor"]["counters"]["action.admitted"] >= 1
        assert active["executor"]["counters"]["action_executed"] >= 1
        assert active["executor"]["observation"] == robot_status["follower"]["positions"]
        assert robot_status["follower"]["positions"] != {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
        }
        assert operator_service["last_sequence"] >= 1
        assert robot_status["registry"] == {
            **robot_status["registry"],
            "held": True,
            "state": "active",
            "kind": "remote_teleoperation",
        }
        assert operator_status["registry"] == {
            **operator_status["registry"],
            "held": True,
            "state": "active",
            "kind": "remote_operator",
        }
        assert robot_service["active"]["owner_credential_id"] == operator_service["credential_id"]

        # Abrupt process termination simultaneously removes the operator,
        # authenticated control channel, and action network stream. The robot
        # runtime must complete its local STOP without help from that process.
        operator.close(abrupt=True)
        operator = None
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            robot_status = robot.status()
            if robot_status["follower"]["stop_reasons"]:
                break
            time.sleep(0.02)

        last_stop = robot_status["service"]["last_stop"]
        assert last_stop["reason"] == "control_channel_lost"
        assert last_stop["state"] == "idle"
        assert last_stop["lease_released"] is True
        assert last_stop["safety"]["software_dispatch_halted"] is True
        assert last_stop["safety"]["hardware_stop_completed"] is True
        assert last_stop["safety"]["hardware_close_completed"] is True
        assert last_stop["safety"]["torque_off_confirmed"] is True
        assert robot_status["follower"]["connected"] is False
        assert robot_status["follower"]["stop_reasons"][-1] == "control_channel_lost"
        assert robot_status["registry"] == {
            **robot_status["registry"],
            "held": False,
            "state": "idle",
            "kind": None,
            "owner": None,
        }
    finally:
        if operator is not None:
            operator.close()
        robot.close()
