"""Authenticated two-process TLS/UDP remote-teleoperation fault matrix."""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from makermodslab.remote_teleop.pairing import certificate_sha256_fingerprint
from makermodslab.remote_teleop.transport import UdpActionReceiver, UdpSessionSender

PEER = Path(__file__).parent / "helpers" / "remote_teleop_process_peer.py"


class _ObservedSocket:
    def __init__(self, inner: socket.socket, receive_waiting: threading.Event) -> None:
        self.inner = inner
        self.receive_waiting = receive_waiting

    def recvfrom(self, size: int):
        self.receive_waiting.set()
        return self.inner.recvfrom(size)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def test_udp_receiver_refreshes_session_after_blocking_receive_begins() -> None:
    receive_waiting = threading.Event()

    def socket_factory(*args, **kwargs):
        return _ObservedSocket(socket.socket(*args, **kwargs), receive_waiting)

    receiver = UdpActionReceiver(
        "127.0.0.1",
        0,
        lambda _raw: None,
        allow_loopback=True,
        socket_factory=socket_factory,
    )
    key = b"k" * 32
    port = receiver.start()
    sender = UdpSessionSender("127.0.0.1", port)
    try:
        # Proves the receiver has already snapshotted its pre-session state and
        # entered recvfrom(). begin_session must still authorize this first probe.
        assert receive_waiting.wait(1.0)
        receiver.begin_session(
            session_id="race-session",
            executor_generation=1,
            key_id="race-key",
            key=key,
        )
        sender.prove_endpoint(
            session_id="race-session",
            executor_generation=1,
            key_id="race-key",
            key=key,
        )
        assert receiver.status()["endpoint_bound"] is True
        assert receiver.status()["counters"] == {"endpoint_bound": 1}
    finally:
        sender.close()
        receiver.close()


def _make_certificate(tmp_path: Path) -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the two-process TLS trial")
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


class PeerProcess:
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

    def receive(self, event: str, timeout: float = 4.0) -> dict[str, object]:
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
            f"peer did not emit {event!r}; exit={self.process.poll()}, stderr={stderr[-1000:]}"
        )

    def send(self, command: str, **values: object) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command": command, **values}) + "\n")
        self.process.stdin.flush()

    def request(self, command: str, event: str, **values: object) -> dict[str, object]:
        self.send(command, **values)
        return self.receive(event)

    def close(self, *, abrupt: bool = False) -> None:
        if self.process.poll() is None:
            if abrupt:
                self.process.terminate()
            else:
                try:
                    self.send("shutdown")
                    self.receive("shutdown", timeout=1.0)
                except (AssertionError, BrokenPipeError):
                    self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


class ProcessTrial:
    def __init__(self, root: Path, certificate: Path, private_key: Path) -> None:
        self.root = root
        self.certificate = certificate
        self.private_key = private_key
        self.fingerprint = certificate_sha256_fingerprint(certificate)
        self.robot: PeerProcess | None = None
        self.operator: PeerProcess | None = None

    def start(
        self,
        *,
        pair: bool = True,
        drop: str = "",
        delay: str = "",
        duplicate: str = "",
    ) -> tuple[PeerProcess, PeerProcess]:
        self.robot = PeerProcess(
            [
                "robot",
                "--state-root",
                str(self.root),
                "--certificate",
                str(self.certificate),
                "--private-key",
                str(self.private_key),
            ]
        )
        ready = self.robot.receive("ready")
        self.operator = PeerProcess(
            [
                "operator",
                "--state-root",
                str(self.root),
                "--control-uri",
                f"wss://127.0.0.1:{ready['control_port']}",
                "--fingerprint",
                self.fingerprint,
                "--drop",
                drop,
                "--delay",
                delay,
                "--duplicate",
                duplicate,
            ]
        )
        self.operator.receive("ready")
        try:
            self.operator.request(
                "connect",
                "connected",
                pairing_token=ready["pairing_token"] if pair else None,
            )
        except AssertionError as exc:
            status = self.robot.request("status", "status")["status"]
            self.close()
            raise AssertionError(f"operator connection failed; robot_status={status}") from exc
        return self.robot, self.operator

    def close(self, *, abrupt_operator: bool = False) -> None:
        if self.operator is not None:
            self.operator.close(abrupt=abrupt_operator)
        if self.robot is not None:
            self.robot.close()


def _wait_for_stop(robot: PeerProcess, expected_reason: str, timeout: float = 2.5) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = robot.request("status", "status")
        last = response["status"]
        follower = last["follower"]
        if follower["stop_reasons"]:
            break
        time.sleep(0.02)
    assert last["follower"]["stop_reasons"][-1] == expected_reason
    safety = last["executor"]["safety"]
    assert safety == {
        **safety,
        "stop_accepted": True,
        "software_dispatch_halted": True,
        "disable_requested": True,
        "hardware_stop_completed": True,
        "hardware_close_completed": True,
        "torque_off_confirmed": True,
        "fault_lockout": False,
        "faults": [],
    }
    assert last["executor"]["dispatch_enabled"] is False
    assert last["follower"]["connected"] is False
    return last


def test_two_clean_processes_pass_authenticated_tls_udp_fault_matrix(tmp_path: Path) -> None:
    certificate, private_key = _make_certificate(tmp_path)

    # Normal, packet loss/reorder/duplicate, concurrent session, and acknowledged STOP.
    normal = ProcessTrial(tmp_path / "normal", certificate, private_key)
    robot, operator = normal.start(drop="3", delay="4", duplicate="5")
    try:
        for _ in range(4):
            operator.request("send_action", "action_sent")
        operator.request("flush_reordered", "reordered_flushed")
        duplicate = operator.request("duplicate_session", "duplicate_refused")
        assert duplicate["error"] != ""
        time.sleep(0.08)
        robot_status = robot.request("status", "status")["status"]
        operator_status = operator.request("status", "status")["status"]
        assert robot_status["executor"]["counters"]["action.admitted"] == 2
        assert robot_status["udp"]["counters"]["datagram_rejected"] == 2
        assert operator_status["proxy"]["counters"] == {
            "delayed": 1,
            "delayed_forwarded": 1,
            "dropped": 1,
            "duplicated": 1,
            "forwarded": 3,
            "reverse_forwarded": 1,
        }
        operator.request("save_old_action", "old_action_saved")
        stop = operator.request("stop", "stop_ack")
        assert stop["receipt"]["reason"] == "process_trial_operator_stop"
        _wait_for_stop(robot, "process_trial_operator_stop")
    finally:
        normal.close()

    # Restart with persisted pairing only. The old session datagram is rejected.
    restarted = ProcessTrial(tmp_path / "normal", certificate, private_key)
    robot, operator = restarted.start(pair=False)
    try:
        operator.request("replay_old_action", "old_action_replayed")
        operator.request("send_action", "action_sent")
        time.sleep(0.08)
        status = robot.request("status", "status")["status"]
        assert status["udp"]["counters"]["datagram_rejected"] == 1
        assert status["executor"]["counters"]["action.admitted"] == 1
        operator.request("stop", "stop_ack")
    finally:
        restarted.close()

    scenarios = (
        ("action", "pause_actions", "action_watchdog_timeout"),
        ("browser", "browser_loss", "operator_browser_lost"),
        ("process-signal", "process_loss", "operator_process_lost"),
        ("control", "pause_control", "control_heartbeat_timeout"),
        ("network", "network_down", "action_watchdog_timeout"),
    )
    for name, fault, expected in scenarios:
        trial = ProcessTrial(tmp_path / name, certificate, private_key)
        robot, operator = trial.start()
        try:
            operator.request("start_actions", "actions_started")
            time.sleep(0.08)
            if fault == "pause_actions":
                operator.request("pause_actions", "actions_paused")
            elif fault == "browser_loss":
                operator.request("browser_loss", "browser_loss_sent")
            elif fault == "process_loss":
                operator.request("process_loss", "process_loss_sent")
            elif fault == "pause_control":
                operator.request("pause_control", "control_paused")
            else:
                operator.request("network_down", "network_down")
            _wait_for_stop(robot, expected)
        finally:
            trial.close()

    # An unannounced operator crash still causes a robot-local control stop.
    crashed = ProcessTrial(tmp_path / "crash", certificate, private_key)
    robot, operator = crashed.start()
    try:
        operator.request("start_actions", "actions_started")
        operator.close(abrupt=True)
        crashed.operator = None
        _wait_for_stop(robot, "control_channel_lost")
    finally:
        crashed.close()

    # A fresh in-session clock mapping that drifts is stopped, never remapped.
    drifted = ProcessTrial(tmp_path / "clock-drift", certificate, private_key)
    robot, operator = drifted.start()
    try:
        operator.request("start_actions", "actions_started")
        checked = operator.request("clock_drift", "clock_checked")
        assert checked["valid"] is False
        _wait_for_stop(robot, "clock_sync_violation")
    finally:
        drifted.close()
