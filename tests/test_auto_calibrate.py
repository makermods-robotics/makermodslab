# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for makermodslab.auto_calibrate — subprocess manager (process mocked)."""

from __future__ import annotations

import os
import subprocess
import threading

import pytest

import makermodslab.auto_calibrate as auto_calibrate
from makermodslab.vendor.feetech_autocal import auto_calibrate_script as acs


class StoppableFakeProc:
    """Process double for the stop path.

    stdout yields one line then blocks until the process "dies"; wait()
    honors timeouts like subprocess.Popen.wait. With ignore_sigterm=True the
    process only dies on kill() — the hardware failure mode (main thread
    wedged in a C-level serial call, SIGTERM's KeyboardInterrupt never
    materializes).
    """

    def __init__(self, ignore_sigterm: bool = False) -> None:
        self._dead = threading.Event()
        self.ignore_sigterm = ignore_sigterm
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

        def _gen():
            yield "Stage 0: init\n"
            self._dead.wait()

        self.stdout = _gen()

    def wait(self, timeout: float | None = None) -> int:
        if not self._dead.wait(timeout):
            raise subprocess.TimeoutExpired(cmd="auto-calibrate", timeout=timeout or 0)
        self.returncode = -9 if self.killed else 130
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignore_sigterm:
            self._dead.set()

    def kill(self) -> None:
        self.killed = True
        self._dead.set()


def _start_with_fake_proc(
    monkeypatch: pytest.MonkeyPatch, proc: StoppableFakeProc, released_ports: list[str]
):
    """Start a manager on a fake process, with the fallback release recorded."""
    import makermodslab.auto_calibrate as ac

    popen_kwargs: dict = {}

    def _popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return proc

    monkeypatch.setattr(ac.subprocess, "Popen", _popen)
    monkeypatch.setattr(ac, "_release_arm_torque", lambda port: (released_ports.append(port), [])[1])
    # Keep the escalation fast in tests.
    monkeypatch.setattr(ac, "_STOP_GRACE_S", 0.2)
    monkeypatch.setattr(ac, "_STOP_KILL_WAIT_S", 0.2)

    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="my_arm"))
    assert result["success"] is True
    return mgr, popen_kwargs


def _join_stop(mgr) -> None:
    if mgr._stop_thread is not None:
        mgr._stop_thread.join(timeout=5)
    if mgr._thread is not None:
        mgr._thread.join(timeout=5)


def test_auto_calibration_blocked_when_teleoperation_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-calibration must refuse to start while teleop owns the same
    serial bus, rather than opening a second connection on a live port."""
    import makermodslab.auto_calibrate as ac

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="c"))
    assert result["success"] is False
    assert "Teleoperation" in result["message"]


def test_auto_calibration_blocked_when_recording_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="c"))
    assert result["success"] is False
    assert "Recording" in result["message"]


def test_auto_calibration_blocked_when_inference_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="c"))
    assert result["success"] is False
    assert "Inference" in result["message"]


def test_auto_calibration_blocked_when_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="c"))
    assert result["success"] is False
    assert "Calibration" in result["message"]


def test_auto_calibration_blocked_when_wiggle_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="c"))
    assert result["success"] is False
    assert "wiggle" in result["message"].lower()


def test_auto_calibration_rejects_bad_device() -> None:
    import makermodslab.auto_calibrate as ac

    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="bogus", port="/dev/x", config_file="c"))
    assert result["success"] is False


def test_auto_calibration_rejects_empty_port() -> None:
    import makermodslab.auto_calibrate as ac

    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="", config_file="c"))
    assert result["success"] is False


def test_auto_calibration_status_idle() -> None:
    import makermodslab.auto_calibrate as ac

    status = ac.AutoCalibrationManager().get_status()
    assert status["status"] == "idle"
    assert status["active"] is False
    assert status["logs"] == []


def test_auto_calibration_launches_captures_logs_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful subprocess run captures its stdout and ends 'completed'."""
    import makermodslab.auto_calibrate as ac

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = iter(["Stage 0: init\n", "calibration done\n"])

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(ac.subprocess, "Popen", lambda *a, **k: FakeProc())

    mgr = ac.AutoCalibrationManager()
    # No robot_name -> no record write-back, so no filesystem needed.
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/x", config_file="my_arm"))
    assert result["success"] is True

    if mgr._thread is not None:
        mgr._thread.join(timeout=2)

    status = mgr.get_status()
    assert status["status"] == "completed"
    assert status["active"] is False
    assert any("Stage 0" in line for line in status["logs"])


def test_stop_graceful_reaches_stopped_and_releases_torque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal stop: SIGTERM, the script exits on its own within the grace
    period, no kill — and the fallback release still runs (belt and braces)."""
    proc = StoppableFakeProc()
    released: list[str] = []
    mgr, popen_kwargs = _start_with_fake_proc(monkeypatch, proc, released)

    result = mgr.stop()
    assert result["success"] is True
    _join_stop(mgr)

    status = mgr.get_status()
    assert status["status"] == "stopped"
    assert status["active"] is False
    assert status["error"] is None
    assert proc.terminated is True
    assert proc.killed is False
    assert released == ["/dev/arm"]
    # The child must not inherit the server's stdin: a TTY there makes the
    # script interactive and its "press Enter" prompts block forever.
    assert popen_kwargs.get("stdin") == subprocess.DEVNULL


def test_stop_kills_process_that_ignores_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hardware failure mode: the script never exits on SIGTERM (wedged in
    a C-level serial call). The manager must escalate to SIGKILL, release
    torque directly over the port, and still end on a terminal status —
    previously it stayed 'active' forever with the arm energized."""
    proc = StoppableFakeProc(ignore_sigterm=True)
    released: list[str] = []
    mgr, _ = _start_with_fake_proc(monkeypatch, proc, released)

    result = mgr.stop()
    assert result["success"] is True
    # A second stop while the escalation runs must not double-spawn workers.
    assert mgr.stop()["success"] is False
    _join_stop(mgr)

    status = mgr.get_status()
    assert status["status"] == "stopped"
    assert status["active"] is False
    assert proc.terminated is True
    assert proc.killed is True
    assert released == ["/dev/arm"]


def test_stop_and_wait_blocks_until_torque_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_and_wait() must not return until the escalation has actually
    finished releasing the arm — a caller like server shutdown that's about
    to exit the process can't rely on the background stop thread to keep
    running after it's gone, unlike a Stop-button caller."""
    proc = StoppableFakeProc()
    released: list[str] = []
    mgr, _ = _start_with_fake_proc(monkeypatch, proc, released)

    mgr.stop_and_wait(timeout=5)

    status = mgr.get_status()
    assert status["status"] == "stopped"
    assert status["active"] is False
    assert proc.terminated is True
    assert released == ["/dev/arm"]


def test_stop_and_wait_when_idle_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing running, nothing to wait for — must return immediately rather
    than hang, since server shutdown calls this unconditionally."""
    mgr = auto_calibrate.AutoCalibrationManager()
    mgr.stop_and_wait(timeout=5)
    assert mgr.get_status()["active"] is False


def test_stop_surfaces_failed_torque_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the fallback release fails, the status must end terminal (not
    frozen) with an unmistakable torque warning for the UI."""
    import makermodslab.auto_calibrate as ac

    proc = StoppableFakeProc(ignore_sigterm=True)
    released: list[str] = []
    mgr, _ = _start_with_fake_proc(monkeypatch, proc, released)
    monkeypatch.setattr(
        ac,
        "_release_arm_torque",
        lambda port: [f"TORQUE MAY STILL BE ENABLED on {port} — unplug power to release."],
    )

    assert mgr.stop()["success"] is True
    _join_stop(mgr)

    status = mgr.get_status()
    assert status["status"] == "failed"
    assert status["active"] is False
    assert "TORQUE MAY STILL BE ENABLED" in status["error"]
    assert "/dev/arm" in status["error"]


# ---------------------------------------------------------------------------
# Success-only persistence: a non-successful run must leave no phantom file
# ---------------------------------------------------------------------------


def _point_robots_base_at(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    """Redirect the module-level robots calibration base the manager reads.

    auto_calibrate binds CALIBRATION_BASE_PATH_ROBOTS at import, so patching
    cfg alone wouldn't reach it — patch the name on the module directly."""
    monkeypatch.setattr(auto_calibrate, "CALIBRATION_BASE_PATH_ROBOTS", base)


def _plant_follower_file(base: str, stem: str) -> str:
    """Create the file the subprocess would have written for a follower run."""
    path = os.path.join(base, "so_follower", f"{stem}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{}")
    return path


def _follower_file_path(base: str, stem: str) -> str:
    """Where the subprocess WOULD write, without creating it.

    Stray-file tests must let the run itself create the file (see
    `_write_during_run`): a file planted before `start()` is, by construction,
    indistinguishable from the user's previous calibration under the same name,
    and the cleanup deliberately keeps those (P0-1 / C1)."""
    path = os.path.join(base, "so_follower", f"{stem}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def test_stray_file_removed_when_run_fails_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A run that exits non-zero after the subprocess already wrote its file
    must not leave a phantom library entry behind.

    The file is written by the fake subprocess DURING the run (not planted
    before `start`), so it is genuinely this run's output — which is what makes
    it a stray rather than the user's previous calibration (P0-1 / C1).
    """
    _point_robots_base_at(monkeypatch, str(tmp_path))
    stray = _follower_file_path(str(tmp_path), "my_arm")

    class FailProc:
        def __init__(self) -> None:
            self.stdout = iter(["Stage 0: init\n"])

        def wait(self) -> int:
            with open(stray, "w") as f:  # the subprocess wrote it, then failed
                f.write("{}")
            return 1

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: FailProc())

    mgr = auto_calibrate.AutoCalibrationManager()
    mgr.start(auto_calibrate.AutoCalibrationRequest(device_type="robot", port="/dev/x", config_file="my_arm"))
    if mgr._thread is not None:
        mgr._thread.join(timeout=2)

    assert mgr.get_status()["status"] == "failed"
    assert not os.path.exists(stray)


def test_stray_file_removed_when_post_processing_fails(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A successful subprocess whose post-processing (_finalize_success) raises
    must clean up the file the subprocess wrote — the run is 'failed', so its
    name must not persist as a phantom entry.

    As above, the fake subprocess writes the file during the run so it is this
    run's output rather than a pre-existing calibration.
    """
    _point_robots_base_at(monkeypatch, str(tmp_path))
    stray = _follower_file_path(str(tmp_path), "my_arm")

    class OkProc:
        def __init__(self) -> None:
            self.stdout = iter(["calibration done\n"])

        def wait(self) -> int:
            with open(stray, "w") as f:  # the subprocess succeeded and wrote it
                f.write("{}")
            return 0

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: OkProc())

    mgr = auto_calibrate.AutoCalibrationManager()

    def _boom() -> None:
        raise RuntimeError("record write-back exploded")

    monkeypatch.setattr(mgr, "_finalize_success", _boom)
    mgr.start(auto_calibrate.AutoCalibrationRequest(device_type="robot", port="/dev/x", config_file="my_arm"))
    if mgr._thread is not None:
        mgr._thread.join(timeout=2)

    assert mgr.get_status()["status"] == "failed"
    assert not os.path.exists(stray)


def test_stray_file_removed_on_stop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A stopped run must leave no phantom file even if a stop landed just after
    the subprocess wrote it.

    The file appears AFTER `start()` — i.e. the running subprocess wrote it —
    which is what distinguishes it from the user's previous calibration under the
    same name (P0-1 / C1).
    """
    _point_robots_base_at(monkeypatch, str(tmp_path))
    stray = _follower_file_path(str(tmp_path), "my_arm")

    proc = StoppableFakeProc()
    released: list[str] = []
    mgr, _ = _start_with_fake_proc(monkeypatch, proc, released)

    with open(stray, "w") as f:  # written mid-run, just before the stop lands
        f.write("{}")

    assert mgr.stop()["success"] is True
    _join_stop(mgr)

    assert mgr.get_status()["status"] == "stopped"
    assert not os.path.exists(stray)


def test_completed_during_grace_keeps_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """If the run finishes successfully during the stop grace period, its saved
    file must be KEPT — the terminal status is 'completed', not a stop."""
    _point_robots_base_at(monkeypatch, str(tmp_path))
    kept = _plant_follower_file(str(tmp_path), "my_arm")

    mgr = auto_calibrate.AutoCalibrationManager()
    mgr._request = auto_calibrate.AutoCalibrationRequest(
        device_type="robot", port="/dev/arm", config_file="my_arm"
    )
    mgr.status = auto_calibrate.AutoCalibrationStatus(active=False, status="completed", message="done")
    monkeypatch.setattr(auto_calibrate, "_release_arm_torque", lambda port: [])

    class _DoneProc:
        def wait(self, timeout=None) -> int:
            return 0

        def terminate(self) -> None:
            pass

    mgr._thread = None
    mgr._stop_worker(_DoneProc(), "/dev/arm")

    assert mgr.get_status()["status"] == "completed"
    assert os.path.exists(kept)


def test_release_arm_torque_disables_all_motors(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    class _FakeReleaseBus:
        instances: list[_FakeReleaseBus] = []

        def __init__(self, port: str, motors: dict) -> None:
            self.port = port
            self.motors = motors
            self.disabled: list[str] = []
            self.disconnected = False
            _FakeReleaseBus.instances.append(self)

        def connect(self, handshake: bool = True) -> None:
            assert handshake is False

        def disable_torque(self, motor: str, num_retry: int = 0) -> None:
            self.disabled.append(motor)

        def disconnect(self, disable_torque: bool = True) -> None:
            self.disconnected = True

    monkeypatch.setattr(ac, "FeetechMotorsBus", _FakeReleaseBus)
    problems = ac._release_arm_torque("/dev/arm")

    assert problems == []
    bus = _FakeReleaseBus.instances[-1]
    # All six SO-101 motors released, then the port freed again.
    assert len(bus.disabled) == 6
    assert bus.disconnected is True


# ---------------------------------------------------------------------------
# Vendored script: graceful stop (freeze -> return to start -> release)
# ---------------------------------------------------------------------------


class _FakeScriptBus:
    """Bus double for the vendored script's interrupt/graceful-stop path.

    Reads come from `registers` ({reg: {motor: value}}, default 0); every
    mutation is appended to `calls` so tests can assert ordering.
    """

    def __init__(self, registers: dict[str, dict[str, int]] | None = None) -> None:
        self.motors = dict.fromkeys(acs.MOTOR_NAMES)
        self.registers = registers or {}
        self.calls: list[tuple] = []
        self.sync_write_error: BaseException | None = None
        self.limits = (0, acs.FULL_TURN - 1)
        self.torque_recovery_fail: set[str] = set()
        self.read_fail: set[str] = set()

    def read(self, reg: str, motor: str, normalize: bool = True) -> int:
        if motor in self.read_fail:
            raise ConnectionError("no response")
        return self.registers.get(reg, {}).get(motor, 0)

    def write(self, reg: str, motor: str, value: int, normalize: bool = True) -> None:
        self.calls.append(("write", reg, motor, value))

    def sync_write(self, reg: str, values: dict, normalize: bool = True) -> None:
        if self.sync_write_error is not None:
            raise self.sync_write_error
        self.calls.append(("sync_write", reg, dict(values)))

    def sync_write_pos_ex(self, values: dict) -> None:
        self.calls.append(("pos_ex", dict(values)))

    def read_position_limits(self, motor: str) -> tuple[int, int]:
        return self.limits

    def _write_torque_with_recovery(
        self, motor: str, value: int, retries: int = 3, interval_s: float = 0.5
    ) -> None:
        if motor in self.torque_recovery_fail:
            raise RuntimeError("overload persists")
        self.calls.append(("torque_recovery", motor, value))

    def safe_disable_all(self) -> None:
        self.calls.append(("safe_disable_all",))

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect",))


class _FakeClock:
    """Simulated time for the vendored script: sleep() advances monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Drive the script's time.sleep/time.monotonic off a simulated clock."""
    clock = _FakeClock()
    monkeypatch.setattr(acs.time, "sleep", clock.sleep)
    monkeypatch.setattr(acs.time, "monotonic", clock.monotonic)
    return clock


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record time.sleep durations (as seen by the vendored script) without sleeping."""
    sleeps: list[float] = []
    monkeypatch.setattr(acs.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_interrupt_freezes_and_returns_before_releasing(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """On KeyboardInterrupt the script must freeze all motion FIRST (halt
    velocity mode, hold current pose at working torque), then drive back to the
    startup pose, and only then disable torque — no more instant free-fall."""
    bus = _FakeScriptBus(
        registers={
            "Present_Position": dict.fromkeys(acs.MOTOR_NAMES, 1000),
            "Homing_Offset": dict.fromkeys(acs.MOTOR_NAMES, 0),
            "Moving": dict.fromkeys(acs.MOTOR_NAMES, 0),  # return arrives instantly
        }
    )
    monkeypatch.setattr(acs, "_connect_and_clear", lambda port: bus)

    def body(b) -> None:
        raise KeyboardInterrupt

    code = acs._run_with_bus("/dev/arm", False, body)

    assert code == 130
    names = [c[0] for c in bus.calls]
    # Freeze: Goal_Velocity=0 broadcast, then per-motor holds, then the two
    # pos_ex writes (freeze goals + return goals), then the release.
    assert names[0] == "sync_write"
    assert bus.calls[0][1] == "Goal_Velocity"
    assert names.count("pos_ex") == 2
    last_pos_ex = len(names) - 1 - names[::-1].index("pos_ex")
    assert names.index("safe_disable_all") > last_pos_ex
    assert names[-1] == "disconnect"
    # Freeze holds at NORMAL working torque, not a sag value.
    torque_writes = [c for c in bus.calls if c[0] == "write" and c[1] == "Torque_Limit"]
    assert {c[3] for c in torque_writes} == {acs.DEFAULT_TORQUE_LIMIT}
    # Every motor is re-energized through the overload-clearing recovery path.
    recoveries = [c for c in bus.calls if c[0] == "torque_recovery"]
    assert len(recoveries) == 6
    assert {c[2] for c in recoveries} == {1}
    # Return arrived -> no fallback hold.
    assert acs.STOP_HOLD_S not in no_sleep


def test_return_to_rest_pose_reexpresses_targets_against_current_offset(no_sleep: list[float]) -> None:
    """The startup pose is captured offset-independently; the return must
    re-express it against the CURRENT Homing_Offset (calibration rewrites it)."""
    bus = _FakeScriptBus(
        registers={
            "Present_Position": {"shoulder_lift": 500},
            "Homing_Offset": {"shoulder_lift": 100},
        }
    )
    pose = acs._capture_rest_pose(bus)
    assert pose["shoulder_lift"] == 600  # (500 + 100) % 4096

    # Calibration later rewrote the offset; same physical pose, new raw target.
    # Success is position-based (within POSITION_TOLERANCE), so simulate the
    # motor sitting at the re-expressed target when the return polls it.
    bus.registers["Homing_Offset"]["shoulder_lift"] = -50
    bus.registers["Present_Position"]["shoulder_lift"] = 650
    arrived = acs._return_to_rest_pose(bus, pose, acs.RETURN_TO_REST_BUDGET_S)

    assert arrived is True
    pos_ex = [c for c in bus.calls if c[0] == "pos_ex"]
    assert len(pos_ex) == 1
    target, speed, _acc = pos_ex[0][1]["shoulder_lift"]
    assert target == 650  # (600 - (-50)) % 4096
    assert speed == acs.RETURN_POS_SPEED


def test_graceful_stop_stalled_return_falls_back_to_hold(fake_clock: _FakeClock) -> None:
    """A return making no progress (Moving stays 1, position never changes)
    must be declared a stall after RETURN_STALL_WINDOW_S — well before the
    absolute ceiling — and fall back to the freeze-hold before the release."""
    bus = _FakeScriptBus(registers={"Moving": dict.fromkeys(acs.MOTOR_NAMES, 1)})

    acs._graceful_stop(bus, {"shoulder_lift": 600})

    assert acs.STOP_HOLD_S in fake_clock.sleeps
    # The stall detector fired; the 10s ceiling never had to matter.
    assert fake_clock.now < acs.RETURN_TO_REST_BUDGET_S


def test_return_slower_than_old_flat_budget_still_completes(fake_clock: _FakeClock) -> None:
    """A healthy-but-slow return must run to completion — termination is
    progress-based, so steady motion is never cut short by a timer (the old
    flat 5s cutoff would have aborted this one mid-motion)."""
    bus = _FakeScriptBus(registers={"Moving": {"shoulder_lift": 1}})
    start, target, rate = 500, 2000, 200.0  # ticks, ticks, ticks/s -> ~7.5s of travel
    base_read = _FakeScriptBus.read

    def read(reg: str, motor: str, normalize: bool = True) -> int:
        if reg == "Present_Position" and motor == "shoulder_lift":
            return min(target, int(start + rate * fake_clock.now))
        return base_read(bus, reg, motor, normalize)

    bus.read = read  # type: ignore[method-assign]

    arrived = acs._return_to_rest_pose(bus, {"shoulder_lift": target}, acs.RETURN_TO_REST_BUDGET_S)

    assert arrived is True
    assert fake_clock.now > 5.0  # kept going past the old flat budget
    assert fake_clock.now < acs.RETURN_TO_REST_BUDGET_S  # ceiling still untouched


def test_freeze_clears_overload_and_skips_dead_motor(
    no_sleep: list[float], capsys: pytest.CaptureFixture
) -> None:
    """The freeze must disable torque first (a Stop usually lands with one
    motor stalled mid-probe with its overload latch set — a bare
    Torque_Enable=1 NAKs on it), then re-enable via the recovery path. A motor
    that stays dead is skipped LOUDLY rather than aborting the freeze —
    hardware showed a silently-limp motor stalls the whole return."""
    bus = _FakeScriptBus()
    bus.torque_recovery_fail = {"elbow_flex"}

    acs._freeze_arm(bus)

    # Overload latch cleared (Torque_Enable=0) on every motor before enabling.
    latch_clears = [c for c in bus.calls if c[0] == "write" and c[1] == "Torque_Enable" and c[3] == 0]
    assert len(latch_clears) == 6
    # The dead motor is excluded from the hold goals; the other five freeze.
    pos_ex = [c for c in bus.calls if c[0] == "pos_ex"]
    assert len(pos_ex) == 1
    assert "elbow_flex" not in pos_ex[0][1]
    assert len(pos_ex[0][1]) == 5
    assert "could not be re-energized" in capsys.readouterr().out


def test_graceful_stop_diagnoses_a_stalled_return(
    fake_clock: _FakeClock, capsys: pytest.CaptureFixture
) -> None:
    """The fallback must say WHY it was taken — hardware logs previously could
    not distinguish 'no pose' from 'stall' from 'ceiling'."""
    bus = _FakeScriptBus(registers={"Moving": dict.fromkeys(acs.MOTOR_NAMES, 1)})

    acs._graceful_stop(bus, {"shoulder_lift": 600})

    out = capsys.readouterr().out
    assert "Fallback: return stalled after" in out
    assert "ticks" in out


def test_graceful_stop_diagnoses_a_missing_pose(no_sleep: list[float], capsys: pytest.CaptureFixture) -> None:
    bus = _FakeScriptBus()

    acs._graceful_stop(bus, {})

    assert "no start pose was captured" in capsys.readouterr().out


def test_return_reports_settled_short_instead_of_success(
    fake_clock: _FakeClock, capsys: pytest.CaptureFixture
) -> None:
    """All-Moving==0 with a motor far from target must NOT count as returned
    (the bench symptom: 'not sure the starting position was right') — it is
    its own 'settled short' outcome, with the per-motor deltas."""
    bus = _FakeScriptBus(
        registers={
            "Present_Position": {"shoulder_lift": 100},  # sits 500 ticks short
            "Moving": dict.fromkeys(acs.MOTOR_NAMES, 0),
        }
    )

    arrived = acs._return_to_rest_pose(bus, {"shoulder_lift": 600}, acs.RETURN_TO_REST_BUDGET_S)

    assert arrived is False
    out = capsys.readouterr().out
    assert "settled short of target" in out
    assert "shoulder_lift=500" in out


def test_return_success_reports_per_motor_deltas(
    no_sleep: list[float], capsys: pytest.CaptureFixture
) -> None:
    """A successful return logs |final - target| per motor so a subtly-off
    landing is diagnosable from the log panel."""
    bus = _FakeScriptBus(registers={"Present_Position": {"shoulder_lift": 595}})

    arrived = acs._return_to_rest_pose(bus, {"shoulder_lift": 595}, acs.RETURN_TO_REST_BUDGET_S)

    assert arrived is True
    out = capsys.readouterr().out
    assert "Return complete: max delta" in out
    assert "shoulder_lift=" in out


def test_capture_rest_pose_reports_missing_motors(capsys: pytest.CaptureFixture) -> None:
    """A motor absent from the capture (comm error) is never returned on a
    stop — which looks exactly like 'the starting position wasn't right' — so
    the capture must say what it got and what it is missing."""
    bus = _FakeScriptBus()
    bus.read_fail = {"wrist_roll"}

    pose = acs._capture_rest_pose(bus)

    assert "wrist_roll" not in pose
    assert len(pose) == 5
    out = capsys.readouterr().out
    assert "Start pose captured for:" in out
    assert "MISSING: wrist_roll" in out


def test_nearest_wrap_target_takes_the_short_arc() -> None:
    """The rest target is only defined mod FULL_TURN; the return must pick the
    wrap representative nearest the current position (clamped into the
    firmware limits) so a seam-adjacent pose is a short arc, not a long trek
    through the hard stops."""
    # Target near the top of the range, joint sitting just past the 0 seam:
    # the -96 representative (== 4000 physically) is nearest; clamped to 0.
    assert acs._nearest_wrap_target(4000, 100, 0, 4095) == 0
    # Mirror image: target near 0, joint near 4095 -> clamp at the top.
    assert acs._nearest_wrap_target(96, 4050, 0, 4095) == 4095
    # No seam involved: the plain representative wins.
    assert acs._nearest_wrap_target(650, 500, 0, 4095) == 650
    # Calibrated limit window is respected.
    assert acs._nearest_wrap_target(650, 500, 700, 3000) == 700


def test_graceful_stop_without_captured_pose_holds_then_falls_through(no_sleep: list[float]) -> None:
    """No startup pose (capture failed) -> freeze, hold, no return attempt."""
    bus = _FakeScriptBus()

    acs._graceful_stop(bus, {})

    # Only the freeze pos_ex — no return goals were written.
    assert [c[0] for c in bus.calls if c[0] == "pos_ex"] == ["pos_ex"]
    assert acs.STOP_HOLD_S in no_sleep


def test_graceful_stop_second_interrupt_skips_everything(no_sleep: list[float]) -> None:
    """A second Stop (KeyboardInterrupt mid-sequence) means NOW: _graceful_stop
    must swallow it and return immediately so the release runs instantly."""
    bus = _FakeScriptBus()
    bus.sync_write_error = KeyboardInterrupt()

    acs._graceful_stop(bus, {"shoulder_lift": 600})  # must not raise

    assert no_sleep == []  # no return settle, no fallback hold


def test_graceful_stop_bus_error_falls_through_immediately(no_sleep: list[float]) -> None:
    """A dead bus mid-freeze must not raise and must not delay the release."""
    bus = _FakeScriptBus()

    # _freeze_arm tolerates a COMM_ERR broadcast failure; a harder failure in
    # pos_ex is caught by _graceful_stop's blanket guard.
    def _boom(values: dict) -> None:
        raise OSError("port vanished")

    bus.sync_write_pos_ex = _boom  # type: ignore[method-assign]
    acs._graceful_stop(bus, {})  # must not raise

    assert acs.STOP_HOLD_S not in no_sleep


def test_stop_sequence_budget_stays_inside_sigterm_grace() -> None:
    """freeze (<0.5s) + return ceiling (10s backstop, never expected to matter
    — termination is progress-based) + fallback hold (3s) + 6-motor release
    and disconnect (~2s) must fit the manager's SIGTERM grace, or a healthy
    graceful stop gets SIGKILLed mid-landing."""
    freeze_and_release_margin_s = 3.0
    assert (
        acs.RETURN_TO_REST_BUDGET_S + acs.STOP_HOLD_S + freeze_and_release_margin_s
        <= auto_calibrate._STOP_GRACE_S
    )


def test_release_arm_torque_reports_connect_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.auto_calibrate as ac

    class _DeadBus:
        def __init__(self, port: str, motors: dict) -> None:
            self.port = port

        def connect(self, handshake: bool = True) -> None:
            raise ConnectionError("port busy")

    monkeypatch.setattr(ac, "FeetechMotorsBus", _DeadBus)
    problems = ac._release_arm_torque("/dev/arm")

    assert len(problems) == 1
    assert "TORQUE MAY STILL BE ENABLED" in problems[0]
    assert "/dev/arm" in problems[0]


# ---------------------------------------------------------------------------
# Concurrent batch auto-calibration (subset of arms, partial success)
# ---------------------------------------------------------------------------


def _arm(device_type: str = "robot", port: str = "/dev/a", name: str = "arm_a", arm: str = "left"):
    """Build one AutoCalibrationBatchArm."""
    return auto_calibrate.AutoCalibrationBatchArm(
        device_type=device_type, port=port, config_file=name, arm=arm
    )


def _port_of(popen_args) -> str:
    """Extract the --port value from a Popen command list."""
    argv = popen_args[0]
    return argv[argv.index("--port") + 1]


class _ExitProc:
    """Fake process that emits one line then exits with a fixed code."""

    def __init__(self, code: int, port: str) -> None:
        self.stdout = iter([f"Stage 0: init on {port}\n"])
        self._code = code

    def wait(self, timeout=None) -> int:
        return self._code

    def terminate(self) -> None:
        pass


def _no_name_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing on disk is 'taken' — the pre-check passes for every arm."""
    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: False)


def _join_batch(mgr) -> None:
    for runner in mgr._runners:
        if runner._thread is not None:
            runner._thread.join(timeout=2)
        if runner._stop_thread is not None:
            runner._stop_thread.join(timeout=5)


def test_batch_blocked_when_teleoperation_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch launch must refuse up front (before starting any arm) while
    teleop owns a serial bus, rather than partially launching some arms."""
    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="n1")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "Teleoperation" in result["message"]
    assert mgr._runners == []  # nothing launched


def test_batch_blocked_when_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="n1")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "Calibration" in result["message"]


def test_batch_blocked_when_wiggle_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="n1")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "wiggle" in result["message"].lower()


def test_batch_rejects_empty() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=[]))
    assert result["success"] is False


def test_batch_rejects_more_than_four() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port=f"/dev/a{i}", name=f"n{i}") for i in range(5)]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "4" in result["message"]


def test_batch_rejects_duplicate_ports() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/shared", name="n1"), _arm(port="/dev/shared", name="n2")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "port" in result["message"].lower()


def test_batch_rejects_duplicate_same_side_names() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    # Same device_type + same name (different ports) collide on one file path.
    arms = [
        _arm(device_type="robot", port="/dev/a", name="dup"),
        _arm(device_type="robot", port="/dev/b", name="dup"),
    ]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert "dup" in result["message"]


def test_batch_same_name_different_side_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leader and a follower may share a name — they save to different dirs."""
    _no_name_taken(monkeypatch)
    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: _ExitProc(0, _port_of(a)))
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [
        _arm(device_type="teleop", port="/dev/lead", name="shared"),
        _arm(device_type="robot", port="/dev/follow", name="shared"),
    ]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is True
    _join_batch(mgr)


def test_batch_name_taken_precheck_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing name is rejected up front (before any subprocess launches)."""
    launched: list = []
    monkeypatch.setattr(
        auto_calibrate.subprocess, "Popen", lambda *a, **k: launched.append(a) or _ExitProc(0, "x")
    )
    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: stem == "taken_arm")
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="fresh"), _arm(port="/dev/b", name="taken_arm")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False
    assert result["code"] == "name_taken"
    assert "taken_arm" in result["names"]
    assert launched == []  # fail-fast: no hardware touched


def test_batch_name_taken_bypassed_by_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: _ExitProc(0, _port_of(a)))
    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: True)
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="n1"), _arm(port="/dev/b", name="n2")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms, overwrite=True))
    assert result["success"] is True
    _join_batch(mgr)


def test_batch_threads_motor_power_into_torque_limit_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The robot's torque slider (percent) reaches the vendored script as
    --torque-limit percent×10, on every arm; the flag is absent when the
    request carries no motor_power (the script's default applies)."""
    _no_name_taken(monkeypatch)
    argvs: list[list[str]] = []

    def _popen(*a, **k):
        argvs.append(a[0])
        return _ExitProc(0, _port_of(a))

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", _popen)

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="arm_a"), _arm(device_type="teleop", port="/dev/b", name="arm_b")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms, motor_power=25))
    assert result["success"] is True
    _join_batch(mgr)
    for argv in argvs:
        assert argv[argv.index("--torque-limit") + 1] == "250"

    argvs.clear()
    mgr2 = auto_calibrate.AutoCalibrationBatchManager()
    result2 = mgr2.start(auto_calibrate.AutoCalibrationBatchRequest(arms=[_arm(port="/dev/c", name="arm_c")]))
    assert result2["success"] is True
    _join_batch(mgr2)
    assert "--torque-limit" not in argvs[0]


def test_batch_launches_concurrently_and_all_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three arms launch simultaneously, each on its own port, and all complete
    — no robot_name so no filesystem write-back is needed."""
    _no_name_taken(monkeypatch)
    ports_seen: list[str] = []

    def _popen(*a, **k):
        port = _port_of(a)
        ports_seen.append(port)
        return _ExitProc(0, port)

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", _popen)

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [
        _arm(port="/dev/a", name="arm_a"),
        _arm(port="/dev/b", name="arm_b"),
        _arm(device_type="teleop", port="/dev/c", name="arm_c"),
    ]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is True
    assert result["total"] == 3 and result["launched"] == 3
    _join_batch(mgr)

    status = mgr.get_status()
    assert status["total"] == 3
    assert status["completed"] == 3
    assert status["failed"] == 0
    assert status["active"] is False
    # One subprocess per distinct port.
    assert sorted(ports_seen) == ["/dev/a", "/dev/b", "/dev/c"]
    # Per-arm status carries identity; combined logs are per-arm prefixed.
    names = {a["name"] for a in status["arms"]}
    assert names == {"arm_a", "arm_b", "arm_c"}
    assert any("[arm_a]" in line for line in status["logs"])


def test_batch_partial_success_one_arm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """One arm's subprocess exits non-zero; the OTHERS still complete. The
    failed arm is reported failed, its stray file removed, its name not saved."""
    _no_name_taken(monkeypatch)
    removed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auto_calibrate,
        "_remove_stray_calibration_file",
        lambda dt, stem: removed.append((dt, stem)),
    )

    def _popen(*a, **k):
        port = _port_of(a)
        # The arm on /dev/bad fails; the rest succeed.
        return _ExitProc(1 if port == "/dev/bad" else 0, port)

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", _popen)

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [
        _arm(port="/dev/ok1", name="ok1"),
        _arm(port="/dev/bad", name="bad"),
        _arm(port="/dev/ok2", name="ok2"),
    ]
    assert mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))["success"] is True
    _join_batch(mgr)

    status = mgr.get_status()
    assert status["completed"] == 2
    assert status["failed"] == 1
    by_name = {a["name"]: a for a in status["arms"]}
    assert by_name["ok1"]["status"] == "completed"
    assert by_name["ok2"]["status"] == "completed"
    assert by_name["bad"]["status"] == "failed"
    # Only the failed arm's stray file is cleaned up.
    assert removed == [("robot", "bad")]


def test_batch_launch_failure_does_not_block_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """If one arm's Popen raises, that arm is 'failed' but the others launch."""
    _no_name_taken(monkeypatch)

    def _popen(*a, **k):
        port = _port_of(a)
        if port == "/dev/boom":
            raise OSError("could not open serial port")
        return _ExitProc(0, port)

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", _popen)

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/good", name="good"), _arm(port="/dev/boom", name="boom")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is True
    assert result["launched"] == 1
    _join_batch(mgr)

    status = mgr.get_status()
    by_name = {a["name"]: a for a in status["arms"]}
    assert by_name["good"]["status"] == "completed"
    assert by_name["boom"]["status"] == "failed"


def test_batch_all_launches_fail_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_name_taken(monkeypatch)

    def _popen(*a, **k):
        raise OSError("no serial port")

    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", _popen)
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="a"), _arm(port="/dev/b", name="b")]
    result = mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))
    assert result["success"] is False


def test_batch_stop_stops_all_and_releases_each_torque(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop terminates every running arm and releases EACH arm's torque over its
    own port; a per-arm terminal status ('stopped') results."""
    released: list[str] = []
    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: False)
    monkeypatch.setattr(
        auto_calibrate,
        "_release_arm_torque",
        lambda port: (released.append(port), [])[1],
    )
    monkeypatch.setattr(auto_calibrate, "_STOP_GRACE_S", 0.2)
    monkeypatch.setattr(auto_calibrate, "_STOP_KILL_WAIT_S", 0.2)

    procs = {"/dev/a": StoppableFakeProc(), "/dev/b": StoppableFakeProc()}
    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: procs[_port_of(a)])

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="a"), _arm(port="/dev/b", name="b")]
    assert mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))["success"] is True

    assert mgr.stop()["success"] is True
    _join_batch(mgr)

    status = mgr.get_status()
    assert status["active"] is False
    assert {a["status"] for a in status["arms"]} == {"stopped"}
    assert sorted(released) == ["/dev/a", "/dev/b"]
    assert procs["/dev/a"].terminated and procs["/dev/b"].terminated


def test_batch_stop_and_wait_blocks_until_every_arm_releases_torque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_and_wait() must not return until every arm's escalation has
    actually finished, and must signal every arm up front rather than one at
    a time — otherwise a later arm would keep driving its motors for the
    length of every earlier arm's whole stop sequence."""
    released: list[str] = []
    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: False)
    monkeypatch.setattr(
        auto_calibrate,
        "_release_arm_torque",
        lambda port: (released.append(port), [])[1],
    )
    monkeypatch.setattr(auto_calibrate, "_STOP_GRACE_S", 0.2)
    monkeypatch.setattr(auto_calibrate, "_STOP_KILL_WAIT_S", 0.2)

    procs = {"/dev/a": StoppableFakeProc(), "/dev/b": StoppableFakeProc()}
    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: procs[_port_of(a)])

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="a"), _arm(port="/dev/b", name="b")]
    assert mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))["success"] is True

    mgr.stop_and_wait(timeout=5)

    status = mgr.get_status()
    assert status["active"] is False
    assert {a["status"] for a in status["arms"]} == {"stopped"}
    assert sorted(released) == ["/dev/a", "/dev/b"]
    assert procs["/dev/a"].terminated and procs["/dev/b"].terminated


def test_batch_stop_and_wait_when_idle_is_a_no_op() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    mgr.stop_and_wait(timeout=5)
    assert mgr.get_status()["active"] is False


def test_batch_stop_and_wait_races_manual_batch_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A manual 'Stop batch' button click and server shutdown's stop_and_wait()
    can hit the same batch manager at nearly the same instant — the frontend
    poll loop isn't gated on shutdown. Each arm's own lock must still let only
    one of the two callers actually start that arm's stop escalation; the
    loser must not terminate the process a second time or release its torque
    a second time, and stop_and_wait() must still block until every arm's
    escalation has genuinely finished, whichever caller started it."""
    terminate_calls = {"/dev/a": 0, "/dev/b": 0}
    released: list[str] = []

    class CountingProc(StoppableFakeProc):
        def __init__(self, port: str) -> None:
            super().__init__()
            self._port = port

        def terminate(self) -> None:
            terminate_calls[self._port] += 1
            super().terminate()

    monkeypatch.setattr(auto_calibrate, "_calibration_name_taken", lambda dt, stem: False)
    monkeypatch.setattr(
        auto_calibrate,
        "_release_arm_torque",
        lambda port: (released.append(port), [])[1],
    )
    monkeypatch.setattr(auto_calibrate, "_STOP_GRACE_S", 0.2)
    monkeypatch.setattr(auto_calibrate, "_STOP_KILL_WAIT_S", 0.2)

    procs = {"/dev/a": CountingProc("/dev/a"), "/dev/b": CountingProc("/dev/b")}
    monkeypatch.setattr(auto_calibrate.subprocess, "Popen", lambda *a, **k: procs[_port_of(a)])

    mgr = auto_calibrate.AutoCalibrationBatchManager()
    arms = [_arm(port="/dev/a", name="a"), _arm(port="/dev/b", name="b")]
    assert mgr.start(auto_calibrate.AutoCalibrationBatchRequest(arms=arms))["success"] is True

    start_barrier = threading.Barrier(2)

    def _manual_stop() -> None:
        start_barrier.wait()
        mgr.stop()  # the frontend "Stop" button

    def _shutdown_stop() -> None:
        start_barrier.wait()
        mgr.stop_and_wait(timeout=5)  # server shutdown_event()

    manual_thread = threading.Thread(target=_manual_stop)
    shutdown_thread = threading.Thread(target=_shutdown_stop)
    manual_thread.start()
    shutdown_thread.start()
    manual_thread.join(timeout=5)
    shutdown_thread.join(timeout=5)
    assert not manual_thread.is_alive() and not shutdown_thread.is_alive()

    status = mgr.get_status()
    assert status["active"] is False
    assert {a["status"] for a in status["arms"]} == {"stopped"}
    assert terminate_calls == {"/dev/a": 1, "/dev/b": 1}
    assert sorted(released) == ["/dev/a", "/dev/b"]


def test_batch_stop_when_idle_is_rejected() -> None:
    mgr = auto_calibrate.AutoCalibrationBatchManager()
    assert mgr.stop()["success"] is False


def test_batch_status_idle() -> None:
    status = auto_calibrate.AutoCalibrationBatchManager().get_status()
    assert status["active"] is False
    assert status["arms"] == []
    assert status["total"] == 0
    assert status["completed"] == 0
    assert status["failed"] == 0


# ---------------------------------------------------------------------------
# Calibration-file cleanup: only remove what THIS run created (P0-1 / C1).
#
# The subprocess writes its --save output only at the natural end of a fully
# successful run, so on any stopped/failed run the file at that path is the
# user's previous, still-valid calibration. Deleting it used to leave the arm
# with neither a calibration file nor its servo EEPROM.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_created_calibration_paths():
    """The created-paths registry is module-level (one autocal run per process in
    production); keep tests independent of each other."""
    auto_calibrate._RUN_CREATED_CALIBRATION_PATHS.clear()
    yield
    auto_calibrate._RUN_CREATED_CALIBRATION_PATHS.clear()


def test_stray_cleanup_keeps_a_calibration_file_that_predates_the_run(tmp_lerobot_home) -> None:
    """The re-calibrate flow: the name is already in use, the run fails/stops."""
    stem = "home"
    path = auto_calibrate._subprocess_output_path("robot", stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write('{"shoulder_pan": {"homing_offset": -142}}')

    # Baseline recorded at run start (the file already existed) …
    auto_calibrate._note_calibration_file_baseline("robot", stem)
    # … then the run does not succeed.
    auto_calibrate._remove_stray_calibration_file("robot", stem)

    assert os.path.exists(path), "the previous valid calibration must survive a failed re-run"


def test_stray_cleanup_removes_a_file_this_run_created(tmp_lerobot_home) -> None:
    """Control: "never delete the user's file" must not become "never delete"."""
    stem = "fresh"
    path = auto_calibrate._subprocess_output_path("robot", stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    auto_calibrate._note_calibration_file_baseline("robot", stem)  # nothing there yet
    with open(path, "w") as fh:  # the run wrote it, then died
        fh.write("{}")
    auto_calibrate._remove_stray_calibration_file("robot", stem)

    assert not os.path.exists(path), "a phantom file this run created should still be cleaned up"


# ---------------------------------------------------------------------------
# Servo-EEPROM snapshot/restore in the vendored script (P0-1).
#
# Stage 0 zeroes Homing_Offset and opens the limits to (0, 4095) behind Lock=1
# (EEPROM) before any measurement. Upstream restored nothing on a Stop, so one
# Stop left the arm uncalibrated AND without its firmware end-stop clamp.
# ---------------------------------------------------------------------------

_PRE_HOMING = -142
_PRE_LIMITS = (512, 3600)


class _CalibrationBus:
    """Minimal bus double recording register access in order."""

    def __init__(self) -> None:
        self.motors = dict.fromkeys(acs.MOTOR_NAMES)
        self.homing = dict.fromkeys(acs.MOTOR_NAMES, _PRE_HOMING)
        self.limits = dict.fromkeys(acs.MOTOR_NAMES, _PRE_LIMITS)
        self.log: list[tuple] = []

    def read(self, reg: str, motor: str, normalize: bool = True):
        value = self.homing[motor] if reg == "Homing_Offset" else 0
        self.log.append(("read", reg, motor, value))
        return value

    def write(self, reg: str, motor: str, value, normalize: bool = True) -> None:
        self.log.append(("write", reg, motor, value))
        if reg == "Homing_Offset":
            self.homing[motor] = value

    def read_position_limits(self, motor: str):
        self.log.append(("read", "Position_Limits", motor, self.limits[motor]))
        return self.limits[motor]

    def write_position_limits(self, motor: str, minimum: int, maximum: int) -> None:
        self.log.append(("write", "Position_Limits", motor, (minimum, maximum)))
        self.limits[motor] = (minimum, maximum)


@pytest.fixture
def _no_script_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acs.time, "sleep", lambda *_a, **_k: None)


def test_stage0_reads_calibration_before_it_overwrites_it(_no_script_sleep) -> None:
    """The snapshot must be taken BEFORE the wipe, per motor.

    Ordering is the whole point: a snapshot taken after Stage 0 records the
    zeroed values and would "restore" garbage, which no restore test would catch.
    """
    bus = _CalibrationBus()

    acs._run_init(bus, interactive=False)

    for motor in acs.MOTOR_NAMES:
        entries = [e for e in bus.log if e[2] == motor]
        first_write = next(
            i
            for i, e in enumerate(entries)
            if e[0] == "write" and e[1] in ("Homing_Offset", "Position_Limits")
        )
        reads_before = [
            e
            for e in entries[:first_write]
            if e[0] == "read" and e[1] in ("Homing_Offset", "Position_Limits")
        ]
        assert reads_before, f"{motor}: Stage 0 destroyed the calibration without reading it first"

    # And the wipe itself still happens — Stage 0's job is unchanged.
    assert all(v == 0 for v in bus.homing.values())
    assert all(v == (0, 4095) for v in bus.limits.values())


def test_restore_calibration_puts_back_the_pre_run_values(_no_script_sleep) -> None:
    """After a stopped/failed run the servos are left as they were found."""
    bus = _CalibrationBus()
    acs._run_init(bus, interactive=False)
    assert all(v == 0 for v in bus.homing.values()), "precondition: Stage 0 wiped the offsets"

    acs._restore_calibration(bus)

    assert bus.homing == dict.fromkeys(acs.MOTOR_NAMES, _PRE_HOMING)
    assert bus.limits == dict.fromkeys(acs.MOTOR_NAMES, _PRE_LIMITS), (
        "the firmware travel clamp must come back too — (0, 4095) leaves the joints unprotected"
    )


def test_restore_calibration_is_a_noop_without_a_snapshot() -> None:
    """Never invent values: with nothing snapshotted, write nothing.

    Guards the failure mode where a restore on a path that never ran Stage 0
    would push zeroes into EEPROM itself.
    """
    acs._PRE_RUN_CALIBRATION.clear()
    bus = _CalibrationBus()

    acs._restore_calibration(bus)

    assert not [e for e in bus.log if e[0] == "write"]


def test_restore_calibration_survives_an_unresponsive_motor(_no_script_sleep) -> None:
    """Best-effort: this runs on an already-failing path and must never raise.

    One motor that NAKs must not stop the other five from being restored.
    """
    bus = _CalibrationBus()
    acs._run_init(bus, interactive=False)

    dead = acs.MOTOR_NAMES[0]
    original_write = bus.write

    def _flaky(reg, motor, value, normalize=True):
        if motor == dead:
            raise ConnectionError("no response")
        original_write(reg, motor, value, normalize)

    bus.write = _flaky

    acs._restore_calibration(bus)  # must not raise

    for motor in acs.MOTOR_NAMES[1:]:
        assert bus.homing[motor] == _PRE_HOMING, f"{motor} should still have been restored"


def test_stray_cleanup_without_a_baseline_keeps_the_file(tmp_lerobot_home) -> None:
    """No baseline recorded → assume the file predates the run and keep it.

    The safe answer is the default: a phantom library entry is cosmetic, a
    deleted calibration is not.
    """
    stem = "unknown"
    path = auto_calibrate._subprocess_output_path("robot", stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{}")

    auto_calibrate._remove_stray_calibration_file("robot", stem)

    assert os.path.exists(path)
