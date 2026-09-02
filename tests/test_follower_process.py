"""Bounded process-supervisor proofs for the live follower boundary."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from makermodslab.remote_teleop.adapters.follower_process import (
    SO101_ACTION_KEYS,
    FollowerWorkerError,
    FollowerWorkerTimeoutError,
    SO101FollowerProcessDriver,
    WorkerTimeouts,
    _request,
    _worker_main,
)
from makermodslab.remote_teleop.adapters.lerobot_follower import SO101FollowerDriver
from makermodslab.remote_teleop.contracts import SessionSpec
from makermodslab.remote_teleop.executor import JointLimit, RemoteExecutor
from makermodslab.servo_health.sampler import PROTOCOL_VERSION, SOURCE_REVISION
from makermodslab.servo_health.service import ServoHealthService
from tests.test_remote_teleop_adapters import _calibration_data, _FakeFollower


class _ProcessFakeAdapter:
    """Constructed in the child; behavior is selected by a plain dict."""

    joint_names = SO101_ACTION_KEYS

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        marker = Path(str(config["marker_path"]))
        marker.write_text(str(os.getpid()), encoding="utf-8")
        self.positions = OrderedDict((key, 50.0 if key == "gripper.pos" else 0.0) for key in self.joint_names)
        self.status = {
            "connected": False,
            "connect_transient_torque_risk": True,
        }

    def connect(self) -> None:
        if self.config.get("hang_connect"):
            while True:
                time.sleep(1)
        self.status["connected"] = True

    def observe(self):
        if self.config.get("hang_observe"):
            while True:
                time.sleep(1)
        return dict(self.positions)

    def execute(self, positions):
        if self.config.get("hang_execute"):
            while True:
                time.sleep(1)
        self.positions = OrderedDict(positions)
        return dict(self.positions)

    def stop(self, reason):
        if self.config.get("hang_stop"):
            while True:
                time.sleep(1)
        confirmed = self.config.get("torque_off_confirmed", True)
        return {
            "reason": reason,
            "disable_requested": True,
            "hardware_stop_completed": True,
            "torque_off_confirmed": confirmed,
            "verification": "fake_register_readback" if confirmed is not None else "fake_unknown",
            "fault": None if confirmed is True else "fake_torque_fault",
        }

    def close(self):
        if self.config.get("hang_close"):
            while True:
                time.sleep(1)
        self.status["connected"] = False
        return {"close_requested": True, "close_completed": True, "fault": None}

    def sample_health(self):
        if not self.config.get("health"):
            return None
        return {
            "protocol_version": PROTOCOL_VERSION,
            "source_revision": SOURCE_REVISION,
            "owner": "remote_teleoperation",
            "arm": "follower",
            "read_only": True,
            "available": True,
            "complete": False,
            "last_error": None,
            "motors": [],
        }


def _process_fake_factory(config, expected_id, expected_digest, expected_serial_binding):
    assert expected_id is None or isinstance(expected_id, str)
    assert expected_digest is None or isinstance(expected_digest, str)
    assert expected_serial_binding is None or isinstance(expected_serial_binding, str)
    return _ProcessFakeAdapter(config)


def _unproved_open_handle_factory(config, expected_id, expected_digest, expected_serial_binding):
    fake = _FakeFollower(Path(str(config["calibration_path"])))
    return SO101FollowerDriver(
        config,
        expected_calibration_id=expected_id,
        expected_calibration_digest=expected_digest,
        expected_serial_binding=expected_serial_binding,
        device_factory=lambda _config: fake,
    )


def _timeouts(**changes: float) -> WorkerTimeouts:
    values = {
        "connect_s": 3.0,
        "observe_s": 0.08,
        "execute_s": 0.08,
        "stop_s": 0.08,
        "close_s": 0.08,
        "terminate_s": 0.05,
    }
    values.update(changes)
    return WorkerTimeouts(**values)


def _driver(tmp_path: Path, **config: object) -> SO101FollowerProcessDriver:
    return SO101FollowerProcessDriver(
        {"marker_path": str(tmp_path / "constructed.pid"), **config},
        timeouts=_timeouts(),
        process_context=multiprocessing.get_context("spawn"),
        adapter_factory=_process_fake_factory,
    )


def _executor(driver: SO101FollowerProcessDriver) -> RemoteExecutor:
    limits = OrderedDict(
        (key, JointLimit(-200, 200, 100, 500))
        if key != "gripper.pos"
        else (key, JointLimit(0, 100, 100, 500))
        for key in driver.joint_names
    )
    return RemoteExecutor(driver, limits)


class _ScriptedWorkerConnection:
    def __init__(self) -> None:
        self.request = _request(1, "connect", {})
        self.responses: list[object] = []
        self.closed = False

    def recv(self):
        if self.request is None:
            raise EOFError
        request = self.request
        self.request = None
        return request

    def send(self, value) -> None:
        self.responses.append(value)

    def close(self) -> None:
        self.closed = True


def test_worker_finally_after_unproved_binding_and_disconnect_failure_writes_no_motor(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "follower.json"
    calibration_path.write_text(json.dumps(_calibration_data()), encoding="utf-8")
    fake = _FakeFollower(calibration_path)
    fake.bus.disconnect_error = True
    constructed: list[SO101FollowerDriver] = []

    def factory(config, expected_id, expected_digest, expected_serial_binding):
        driver = SO101FollowerDriver(
            config,
            expected_calibration_id=expected_id,
            expected_calibration_digest=expected_digest,
            expected_serial_binding=expected_serial_binding,
            device_factory=lambda _config: fake,
        )
        constructed.append(driver)
        return driver

    connection = _ScriptedWorkerConnection()
    _worker_main(
        connection,
        {},
        None,
        None,
        "claimed-adapter",
        factory,
    )

    assert len(constructed) == 1
    assert connection.responses[0]["ok"] is False
    assert connection.responses[0]["error_code"] == "connect_RuntimeError"
    assert connection.closed is True
    assert fake.bus.disable_calls == []
    assert fake.bus.write_calls == []
    assert fake.bus.enable_calls == 0
    # connect() cleanup and worker-finally close both attempted handle-only
    # disconnects; neither path inferred or requested a torque write.
    assert fake.bus.disconnect_calls == [False, False]


def test_process_driver_reports_fault_lockout_when_default_handle_proof_is_unavailable(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "follower.json"
    calibration_path.write_text(json.dumps(_calibration_data()), encoding="utf-8")
    driver = SO101FollowerProcessDriver(
        {"calibration_path": str(calibration_path)},
        expected_serial_binding="claimed-adapter",
        timeouts=_timeouts(),
        process_context=multiprocessing.get_context("spawn"),
        adapter_factory=_unproved_open_handle_factory,
    )

    with pytest.raises(FollowerWorkerError, match="connect failed"):
        driver.connect()

    assert driver.fault_lockout is True
    assert driver.status["connected"] is False
    assert driver.status["stop_receipt"] is None
    assert driver.status["close_receipt"] is None


def _spec() -> SessionSpec:
    return SessionSpec(
        source_id="operator",
        rig_id="rig",
        rig_digest="a" * 64,
        leader_calibration_id="leader",
        leader_calibration_digest="b" * 64,
        follower_calibration_id="follower",
        follower_calibration_digest="c" * 64,
        joint_names=SO101_ACTION_KEYS,
        units=("degree", "degree", "degree", "degree", "degree", "percent"),
    )


def test_process_starts_only_on_connect_and_child_owns_adapter(tmp_path: Path) -> None:
    marker = tmp_path / "constructed.pid"
    driver = _driver(tmp_path)
    assert driver.worker_pid is None
    assert not marker.exists()

    driver.connect()
    assert driver.worker_pid is not None
    assert int(marker.read_text(encoding="utf-8")) == driver.worker_pid
    assert driver.worker_pid != os.getpid()
    assert tuple(driver.observe()) == SO101_ACTION_KEYS

    targets = OrderedDict((key, 60.0 if key == "gripper.pos" else 10.0) for key in SO101_ACTION_KEYS)
    assert driver.execute(targets) == targets
    stop = driver.stop("test")
    assert stop["torque_off_confirmed"] is True
    assert stop["worker_terminated"] is False
    close = driver.close()
    assert close["close_completed"] is True
    assert driver.worker_pid is None


def test_hanging_stop_is_terminated_with_unknown_torque_and_bounded_caller(tmp_path: Path) -> None:
    driver = _driver(tmp_path, hang_stop=True)
    executor = _executor(driver)
    executor.open_session(_spec())

    started = time.monotonic()
    receipt = executor.stop("network_loss")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    safety = receipt["safety"]
    assert safety["software_dispatch_halted"] is True
    assert safety["torque_off_confirmed"] is None
    assert safety["hardware_close_completed"] is False
    assert safety["fault_lockout"] is True
    assert safety["stop_receipt"]["worker_terminated"] is True
    assert safety["stop_receipt"]["fault"] == "follower_worker_unresponsive"
    assert driver.worker_pid is None


def test_hanging_execute_is_bounded_and_subsequent_stop_never_claims_torque_off(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path, hang_execute=True)
    driver.connect()
    targets = OrderedDict((key, 60.0 if key == "gripper.pos" else 10.0) for key in SO101_ACTION_KEYS)

    started = time.monotonic()
    with pytest.raises(FollowerWorkerTimeoutError):
        driver.execute(targets)
    assert time.monotonic() - started < 0.5
    assert driver.worker_pid is None

    receipt = driver.stop("execute_timeout")
    assert receipt["torque_off_confirmed"] is None
    assert receipt["worker_terminated"] is True
    assert receipt["fault_lockout"] is True
    close = driver.close()
    assert close["close_completed"] is False


def test_hanging_observe_is_bounded_and_latches_unknown_torque(tmp_path: Path) -> None:
    driver = _driver(tmp_path, hang_observe=True)
    driver.connect()
    started = time.monotonic()
    with pytest.raises(FollowerWorkerTimeoutError):
        driver.observe()
    assert time.monotonic() - started < 0.5
    receipt = driver.stop("observation_timeout")
    assert receipt["torque_off_confirmed"] is None
    assert receipt["fault_lockout"] is True


def test_hanging_close_is_terminated_without_becoming_close_evidence(tmp_path: Path) -> None:
    driver = _driver(tmp_path, hang_close=True)
    driver.connect()
    assert driver.stop("test")["torque_off_confirmed"] is True
    started = time.monotonic()
    receipt = driver.close()
    assert time.monotonic() - started < 0.5
    assert receipt["close_completed"] is False
    assert receipt["fault_lockout"] is True
    assert driver.worker_pid is None


def test_responsive_unknown_torque_receipt_latches_fault_without_false_confirmation(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path, torque_off_confirmed=None)
    driver.connect()
    receipt = driver.stop("watchdog")
    assert receipt["torque_off_confirmed"] is None
    assert receipt["worker_terminated"] is False
    assert receipt["fault_lockout"] is True
    assert driver.fault_lockout is True
    assert driver.close()["close_completed"] is True


def test_hanging_connect_is_bounded_and_never_restarts_implicitly(tmp_path: Path) -> None:
    driver = SO101FollowerProcessDriver(
        {"marker_path": str(tmp_path / "constructed.pid"), "hang_connect": True},
        timeouts=_timeouts(connect_s=0.08),
        process_context=multiprocessing.get_context("spawn"),
        adapter_factory=_process_fake_factory,
    )
    started = time.monotonic()
    with pytest.raises(FollowerWorkerTimeoutError):
        driver.connect()
    assert time.monotonic() - started < 0.8
    assert driver.worker_pid is None
    assert driver.fault_lockout is True
    with pytest.raises(RuntimeError, match="fault lockout"):
        driver.connect()


def test_child_bus_owner_republishes_health_and_detaches_on_close(tmp_path: Path) -> None:
    health = ServoHealthService()
    driver = SO101FollowerProcessDriver(
        {"marker_path": str(tmp_path / "constructed.pid"), "health": True},
        timeouts=_timeouts(),
        process_context=multiprocessing.get_context("spawn"),
        adapter_factory=_process_fake_factory,
        health_service=health,
    )
    driver.connect()
    driver.observe()
    snapshot = health.snapshot()
    assert snapshot["owner"] == "remote_teleoperation"
    assert snapshot["arms"][0]["read_only"] is True

    assert driver.stop("test")["torque_off_confirmed"] is True
    assert driver.close()["close_completed"] is True
    assert health.snapshot()["available"] is False
