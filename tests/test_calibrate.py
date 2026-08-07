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
"""Tests for makermodslab.calibrate — manager initial state, request schema, and the
post-recording centering guard."""

from __future__ import annotations

import time

import pytest

from makermodslab.calibrate import final_motor_ranges, find_off_center_joints


def test_final_motor_ranges_forces_wrist_roll_full_turn() -> None:
    # Wrist_roll's swept sliver is discarded for the full turn (matching
    # upstream lerobot); other joints keep their recorded ranges.
    mins = {"shoulder_pan": 900, "wrist_roll": 2000}
    maxes = {"shoulder_pan": 3200, "wrist_roll": 2010}
    assert final_motor_ranges(mins, maxes) == {
        "shoulder_pan": (900, 3200),
        "wrist_roll": (0, 4095),
    }


def test_calibration_request_dataclass_round_trip() -> None:
    from makermodslab.calibrate import CalibrationRequest

    req = CalibrationRequest(
        device_type="teleop",
        port="/dev/ttyUSB0",
        config_file="my_calib",
    )
    assert req.device_type == "teleop"
    assert req.port == "/dev/ttyUSB0"
    assert req.config_file == "my_calib"
    assert req.robot_name is None


def test_calibration_manager_starts_idle() -> None:
    from makermodslab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    status = mgr.get_status()

    assert isinstance(status, CalibrationStatus)
    assert status is mgr.status
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate already running

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_start_calibration_blocked_when_teleoperation_active(monkeypatch) -> None:
    """Calibration must refuse to start while teleop owns the same serial
    bus, rather than opening a second connection on a live port."""
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result["success"] is False
    assert "Teleoperation" in result["message"]


def test_start_calibration_blocked_when_recording_active(monkeypatch) -> None:
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result["success"] is False
    assert "Recording" in result["message"]


def test_start_calibration_blocked_when_inference_active(monkeypatch) -> None:
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result["success"] is False
    assert "Inference" in result["message"]


def test_start_calibration_blocked_when_auto_calibration_active(monkeypatch) -> None:
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result["success"] is False
    assert "Auto-calibration" in result["message"]


def test_start_calibration_blocked_when_wiggle_active(monkeypatch) -> None:
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result["success"] is False
    assert "wiggle" in result["message"].lower()


def test_start_calibration_refuses_existing_config_without_overwrite(tmp_lerobot_home) -> None:
    """Completing calibration saves <config_file>.json; if that name already
    exists, start must refuse (code=name_taken) unless overwrite=True — so no
    file is silently clobbered, and no hardware is touched."""
    from pathlib import Path

    from makermodslab.calibrate import CalibrationManager, CalibrationRequest
    from makermodslab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "taken.json").write_text("{}")

    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="taken")
    )
    assert result.get("success") is False
    assert result.get("code") == "name_taken"
    # The guard returns before activating or spawning the worker thread.
    assert mgr.status.calibration_active is False
    assert mgr.calibration_thread is None


def test_start_calibration_check_and_claim_is_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """Two callers racing start_calibration must not both win.

    `calibration_active` is read (the "already active?" guard) and later
    written (claiming the slot) as two separate, unguarded steps — a second
    caller landing in between sees the same False the first one saw, and both
    proceed to spawn a worker thread against the same arm. This stalls a real
    thread between the read and the write (at `calibration_dir_for_device`,
    already on that path) so the test controls the exact interleaving a lock
    around the whole check-and-claim must prevent."""
    import threading

    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    # Avoid spawning the real hardware-touching worker; only start_calibration
    # (the check-and-claim path) is under test here.
    monkeypatch.setattr(mgr, "_calibration_worker", lambda request: None)

    reached_stall = threading.Event()
    release_stall = threading.Event()

    def _stalling_calibration_dir_for_device(device_type: str) -> str | None:
        reached_stall.set()
        assert release_stall.wait(timeout=5), "test setup: release never signalled"
        return None

    monkeypatch.setattr(
        "makermodslab.calibrate.calibration_dir_for_device", _stalling_calibration_dir_for_device
    )

    request = CalibrationRequest(device_type="robot", port="/dev/null", config_file="race", overwrite=True)
    results: dict[str, dict] = {}

    def _call(key: str) -> None:
        results[key] = mgr.start_calibration(request)

    t1 = threading.Thread(target=_call, args=("t1",))
    t1.start()
    assert reached_stall.wait(timeout=5), "t1 never reached the stall point"

    t2 = threading.Thread(target=_call, args=("t2",))
    t2.start()
    # Give t2 a chance to reach (and, if the bug is present, race past) the
    # same unguarded check t1 already passed.
    time.sleep(0.05)

    release_stall.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()

    outcomes = sorted(r["success"] for r in results.values())
    assert outcomes == [False, True], (
        f"expected exactly one of the two concurrent starts to win, got {results}"
    )


def test_start_calibration_resets_calibration_committed(monkeypatch, tmp_lerobot_home) -> None:
    """The manager is a long-lived singleton reused across sessions, so a
    stale True left over from a prior successful run must not survive into a
    new one — otherwise _cleanup_device would treat this run's own cancelled
    attempt as already committed and skip the homing-offset/position-limit
    rollback entirely."""
    from makermodslab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    # Avoid spawning the real hardware-touching worker; only the
    # check-and-claim / reset path is under test here.
    monkeypatch.setattr(mgr, "_calibration_worker", lambda request: None)
    mgr._calibration_committed = True  # leftover from a prior successful run

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )

    assert result["success"] is True
    assert mgr._calibration_committed is False


def test_find_off_center_joints_passes_centered_ranges() -> None:
    """Ranges whose midpoints sit on the raw-tick center (2047) all pass."""
    ranges = {
        "shoulder_pan": (1047, 3047),  # midpoint exactly 2047
        "shoulder_lift": (1500, 2600),  # midpoint 2050, well within tolerance
        "elbow_flex": (1000, 3000),
        "wrist_flex": (1200, 2900),
    }
    assert find_off_center_joints(ranges) == []


def test_find_off_center_joints_names_the_skewed_joint() -> None:
    """A range lying almost entirely to one side of 2047 is flagged by name."""
    ranges = {
        "shoulder_pan": (1047, 3047),  # centered, passes
        "shoulder_lift": (2000, 3600),  # midpoint 2800, 753 off vs 320 allowed
    }
    assert find_off_center_joints(ranges) == ["shoulder_lift"]


def test_find_off_center_joints_exempts_gripper_and_wrist_roll() -> None:
    """Gripper is legitimately homed closed, and wrist_roll is a full-turn
    motor upstream — both skip the check no matter how skewed their range is."""
    ranges = {
        "gripper": (2000, 3500),  # midpoint 2750, would fail if checked
        "wrist_roll": (2500, 4000),  # midpoint 3250, would fail if checked
    }
    assert find_off_center_joints(ranges) == []


def test_find_off_center_joints_tolerance_boundary() -> None:
    """Deviation equal to 20% of the range width passes; one tick more fails."""
    # Width 2000 -> 400 ticks allowed. Midpoint 2447 deviates by exactly 400.
    assert find_off_center_joints({"elbow_flex": (1447, 3447)}) == []
    # Midpoint 2448 deviates by 401 — just over the line.
    assert find_off_center_joints({"elbow_flex": (1448, 3448)}) == ["elbow_flex"]


class _FakeBus:
    """Records register writes; enough surface for _cleanup_device."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str, int]] = []

    def write(self, register: str, motor: str, value: int) -> None:
        self.writes.append((register, motor, value))


class _BrokenBus(_FakeBus):
    """Every write raises, simulating a bus that's already gone unreachable."""

    def write(self, register: str, motor: str, value: int) -> None:
        raise RuntimeError("bus unreachable")


class _PartiallyBrokenBus(_FakeBus):
    """Only the named motor's writes fail; every other motor succeeds."""

    def __init__(self, failing_motor: str) -> None:
        super().__init__()
        self._failing_motor = failing_motor

    def write(self, register: str, motor: str, value: int) -> None:
        if motor == self._failing_motor:
            raise RuntimeError(f"bus rejected write for {motor}")
        super().write(register, motor, value)


class _FakeDevice:
    def __init__(self, bus) -> None:
        self.bus = bus
        self.disconnected = False

    def disconnect(self) -> None:
        self.disconnected = True


def test_cleanup_device_restores_homing_offsets_when_not_committed() -> None:
    """Regression for C2: _step_homing writes new Homing_Offset values to the
    servo's EEPROM immediately (step 1), well before _complete_calibration
    ever saves the on-disk calibration file (step 3). A calibration cancelled
    or errored in between used to leave the servo permanently diverged from
    the last-saved file — the next session's arm-identity fingerprint check
    (see makermodslab/arm_identity.py) would then either hard-refuse to start or
    silently decode positions against a stale offset. _cleanup_device must
    restore the pre-calibration baseline whenever the run never committed."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _FakeBus()
    device = _FakeDevice(bus)
    mgr.device = device
    mgr._original_homing_offsets = {"shoulder_pan": 15, "wrist_roll": -42}
    mgr._calibration_committed = False  # cancelled/errored before saving

    mgr._cleanup_device()

    assert ("Homing_Offset", "shoulder_pan", 15) in bus.writes
    assert ("Homing_Offset", "wrist_roll", -42) in bus.writes
    assert device.disconnected is True
    assert mgr.device is None


def test_cleanup_device_restores_position_limits_when_not_committed() -> None:
    """Regression: reset_calibration() (called by _step_homing) doesn't just
    zero Homing_Offset — it also zeroes Min_Position_Limit and maxes out
    Max_Position_Limit for every motor on the bus. A cancelled run must roll
    those back too, or the next session's arm-identity check still sees a
    servo diverged from the saved calibration file, just via a different
    pair of registers."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _FakeBus()
    device = _FakeDevice(bus)
    mgr.device = device
    mgr._original_homing_offsets = {"shoulder_pan": 15}
    mgr._original_min_position_limits = {"shoulder_pan": 120}
    mgr._original_max_position_limits = {"shoulder_pan": 3980}
    mgr._calibration_committed = False

    mgr._cleanup_device()

    assert ("Min_Position_Limit", "shoulder_pan", 120) in bus.writes
    assert ("Max_Position_Limit", "shoulder_pan", 3980) in bus.writes
    assert device.disconnected is True


def test_cleanup_device_restores_remaining_motors_after_one_write_fails() -> None:
    """One motor's restore write failing must not abort the writes still
    queued for the remaining motors — the old single try/except around the
    whole loop meant one bad write left every motor after it in iteration
    order still diverged: a partially-rolled-back arm, which is worse than a
    fully-diverged one because it's much harder to notice and diagnose."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _PartiallyBrokenBus(failing_motor="elbow_flex")
    device = _FakeDevice(bus)
    mgr.device = device
    mgr._original_homing_offsets = {
        "shoulder_pan": 15,
        "elbow_flex": 7,
        "wrist_roll": -42,
    }
    mgr._calibration_committed = False

    mgr._cleanup_device()  # must not raise despite elbow_flex's write failing

    assert ("Homing_Offset", "shoulder_pan", 15) in bus.writes
    assert ("Homing_Offset", "wrist_roll", -42) in bus.writes
    assert not any(write[1] == "elbow_flex" for write in bus.writes)
    assert device.disconnected is True
    assert mgr.device is None


def test_cleanup_device_does_not_restore_once_calibration_committed() -> None:
    """A calibration that reached _complete_calibration (file saved to disk)
    must not have its final homing offsets touched during cleanup — EEPROM
    and the file already agree at that point."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _FakeBus()
    mgr.device = _FakeDevice(bus)
    mgr._original_homing_offsets = {"shoulder_pan": 15}
    mgr._calibration_committed = True

    mgr._cleanup_device()

    assert bus.writes == []


def test_cleanup_device_skips_restore_when_no_baseline_was_captured() -> None:
    """If the baseline read itself failed (device unreachable at connect
    time), there's nothing to roll back to — cleanup must not crash or invent
    a value."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _FakeBus()
    mgr.device = _FakeDevice(bus)
    mgr._original_homing_offsets = {}
    mgr._calibration_committed = False

    mgr._cleanup_device()  # must not raise

    assert bus.writes == []


def test_cleanup_device_still_disconnects_if_restore_write_fails() -> None:
    """A failed rollback write (e.g. the bus just dropped) must be logged, not
    swallowed silently and not allowed to block the device disconnect."""
    from makermodslab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    bus = _BrokenBus()
    device = _FakeDevice(bus)
    mgr.device = device
    mgr._original_homing_offsets = {"shoulder_pan": 15}
    mgr._calibration_committed = False

    mgr._cleanup_device()  # must not raise despite the write failing

    assert device.disconnected is True
    assert mgr.device is None
