# Copyright 2026 MakerMods. All rights reserved.
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
"""Tests for makermodslab.wiggle — gripper wiggle port finder (hardware mocked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from makermodslab.wiggle import plan_wiggle


async def test_wiggle_gripper_rejects_empty_port() -> None:
    from makermodslab.wiggle import wiggle_gripper

    result = await wiggle_gripper("   ")
    assert result == {"success": False, "message": "No port provided."}


async def test_wiggle_gripper_blocked_when_already_wiggling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second concurrent wiggle must refuse rather than opening a second
    connection on the same port a live wiggle is already driving."""
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "already in progress" in result["message"]


async def test_wiggle_gripper_blocked_when_teleoperation_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "Teleoperation" in result["message"]


async def test_wiggle_gripper_blocked_when_recording_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "Recording" in result["message"]


async def test_wiggle_gripper_blocked_when_inference_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "Inference" in result["message"]


async def test_wiggle_gripper_blocked_when_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "Calibration" in result["message"]


async def test_wiggle_gripper_blocked_when_auto_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "Auto-calibration" in result["message"]


async def test_wiggle_gripper_blocked_when_replay_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.wiggle import wiggle_gripper

    monkeypatch.setattr("makermodslab.replay.replay_active", True)
    result = await wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert "replay" in result["message"].lower()


async def test_wiggle_gripper_clears_wiggle_active_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wiggle_active must be reset once the drive finishes, so a later wiggle
    isn't wrongly refused forever."""
    import makermodslab.wiggle as wiggle

    monkeypatch.setattr(wiggle, "_wiggle_gripper_sync", lambda port: None)

    result = await wiggle.wiggle_gripper("/dev/fake")
    assert result["success"] is True
    assert wiggle.wiggle_active is False


async def test_wiggle_gripper_clears_wiggle_active_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wiggle_active must be reset even when the drive raises, so one failed
    wiggle can't wedge every later wiggle attempt shut."""
    import makermodslab.wiggle as wiggle

    def boom(port: str) -> None:
        raise RuntimeError("no device on port")

    monkeypatch.setattr(wiggle, "_wiggle_gripper_sync", boom)

    result = await wiggle.wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert wiggle.wiggle_active is False


async def test_wiggle_gripper_timeout_keeps_flag_set_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `asyncio.wait_for` timing out does NOT stop the underlying
    thread (asyncio.to_thread work keeps running after the wrapper gives up on
    it) — the same orphaned-worker shape as the teleoperation bug in
    tests/test_teleoperate.py::test_second_stop_timeout_keeps_thread_reference.
    If wiggle_active clears the instant the 15s wait_for times out, a later
    start (teleop/calibration/etc.) can open a second connection on the same
    port a still-running wiggle thread is still driving.
    """
    import threading
    import time

    import makermodslab.wiggle as wiggle

    thread_still_running = threading.Event()
    thread_finished = threading.Event()

    def slow_wiggle(port: str) -> None:
        thread_still_running.set()
        time.sleep(0.2)  # outlives the shortened timeout below
        thread_finished.set()

    monkeypatch.setattr(wiggle, "_wiggle_gripper_sync", slow_wiggle)
    monkeypatch.setattr(wiggle, "_WIGGLE_TIMEOUT_S", 0.05)

    result = await wiggle.wiggle_gripper("/dev/fake")

    assert result["success"] is False
    assert "timed out" in result["message"]
    # The key regression assertion: the background thread is still actually
    # running (it just started sleeping), so the flag must still be set.
    assert thread_still_running.is_set()
    assert not thread_finished.is_set()
    assert wiggle.wiggle_active is True

    # Once the real thread actually finishes, the flag must clear.
    assert thread_finished.wait(timeout=2.0)
    for _ in range(100):
        if not wiggle.wiggle_active:
            break
        time.sleep(0.01)
    assert wiggle.wiggle_active is False


def test_wiggle_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the blocking hardware call so the happy path runs without a device.
    monkeypatch.setattr("makermodslab.wiggle._wiggle_gripper_sync", lambda port: None)

    response = client.post("/wiggle", json={"port": "/dev/fake"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "/dev/fake" in body["message"]


def test_wiggle_endpoint_reports_hardware_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(port: str) -> None:
        raise RuntimeError("no device on port")

    monkeypatch.setattr("makermodslab.wiggle._wiggle_gripper_sync", boom)

    response = client.post("/wiggle", json={"port": "/dev/fake"})
    # Logical failures stay HTTP 200 with success=False (like other handlers).
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "no device on port" in body["message"]


# plan_wiggle cases mirror the 2026-07-02 hardware incident: EEPROM position
# limits written by a prior calibration no longer bracketed the gripper's
# resting position, so the naive 0-4095 jog clamped and moved the wrong way.


def test_plan_wiggle_centered_position_wiggles_in_place() -> None:
    assert plan_wiggle(2000, 0, 4095) == (2200, 1800, 2000)


def test_plan_wiggle_position_beyond_max_limit_refuses_motion() -> None:
    # Real case: gripper parked at 3676 with programmed limits 1387-2707;
    # "+200" used to clamp down to 2707 and jog the wrong way first.
    with pytest.raises(ValueError, match="cannot return"):
        plan_wiggle(3676, 1387, 2707)


def test_plan_wiggle_position_near_min_limit_preserves_start() -> None:
    # Real case: gripper at 1461 with min limit 1387 — the -200 jog clamped.
    high, low, rest = plan_wiggle(1461, 1387, 2707)
    assert (high, low, rest) == (1661, 1387, 1461)


def test_plan_wiggle_limits_wider_than_factory_range_are_clamped() -> None:
    high, low, rest = plan_wiggle(50, -500, 9000)
    assert low >= 0
    assert high <= 4095
    assert rest == 50


def test_plan_wiggle_too_narrow_window_raises_legible_error() -> None:
    with pytest.raises(ValueError, match="too narrow"):
        plan_wiggle(2000, 1900, 2100)


@pytest.mark.parametrize("current", [1387, 1461, 2000, 2690, 2707])
def test_plan_wiggle_preserves_closed_open_and_partial_positions(current: int) -> None:
    high, low, rest = plan_wiggle(current, 1387, 2707)
    assert 1387 <= low < high <= 2707
    assert rest == current


class GripperBus:
    """A servo that follows goals, with an optional one-shot transport failure."""

    def __init__(self, current, fail_jog=False):
        self.current = current
        self.initial = current
        self.fail_jog = fail_jog
        self.goals = []
        self.is_connected = False
        self.disconnected = False

    def connect(self, **kwargs):
        self.is_connected = True

    def disconnect(self, **kwargs):
        self.disconnected = True
        self.is_connected = False

    def read(self, name, motor, **kwargs):
        assert motor == "gripper"
        return {"Min_Position_Limit": 1387, "Max_Position_Limit": 2707, "Present_Position": self.current}[
            name
        ]

    def sync_read(self, name, motor, **kwargs):
        return {motor: self.read(name, motor)}

    def write(self, name, motor, value, **kwargs):
        assert motor == "gripper"
        if name != "Goal_Position":
            return
        self.goals.append(value)
        if self.fail_jog and value != self.initial:
            self.fail_jog = False
            raise OSError("jog failed")
        self.current = value

    def enable_torque(self, motor):
        assert motor == "gripper"

    def disable_torque(self):
        pass


@pytest.mark.parametrize("current", [1387, 1461, 2000, 2707])
@pytest.mark.parametrize("fail_jog", [False, True])
def test_feetech_wiggle_returns_before_disconnect(monkeypatch, current, fail_jog):
    import makermodslab.wiggle as wiggle

    bus = GripperBus(current, fail_jog)
    monkeypatch.setattr(wiggle, "FeetechMotorsBus", lambda **kwargs: bus)
    monkeypatch.setattr(wiggle.time, "sleep", lambda _: None)
    if fail_jog:
        with pytest.raises(OSError, match="jog failed"):
            wiggle._wiggle_gripper_sync("fake")
    else:
        wiggle._wiggle_gripper_sync("fake")
    assert bus.goals[-1] == current
    assert bus.current == current
    assert bus.disconnected


@pytest.mark.parametrize(
    "family,current", [("maker", -120.1), ("maker", -2.5), ("metal", 0.0), ("metal", 100.0)]
)
@pytest.mark.parametrize("fail_jog", [False, True])
def test_can_wiggle_only_drives_gripper_and_returns(monkeypatch, family, current, fail_jog):
    import importlib

    import makermodslab.wiggle as wiggle
    from makermodslab import can_wiggle

    bus = GripperBus(current, fail_jog)
    protocol, class_name = (
        ("robstride", "RobstrideMotorsBus") if family == "maker" else ("damiao", "DamiaoMotorsBus")
    )

    def make_bus(**kwargs):
        assert list(kwargs["motors"]) == ["gripper"]
        assert kwargs["motors"]["gripper"].id == 7
        return bus

    monkeypatch.setattr(importlib.import_module(f"lerobot.motors.{protocol}"), class_name, make_bus)
    monkeypatch.setattr(wiggle.time, "sleep", lambda _: None)
    if fail_jog:
        with pytest.raises(OSError, match="jog failed"):
            can_wiggle.drive_gripper_wiggle(
                can_wiggle._open_gripper_bus(family, "fake"),
                can_wiggle.gripper_limits(family),
                sleep=lambda _: None,
            )
    else:
        can_wiggle.drive_gripper_wiggle(
            can_wiggle._open_gripper_bus(family, "fake"),
            can_wiggle.gripper_limits(family),
            sleep=lambda _: None,
        )
    assert bus.goals[-1] == current
    assert bus.current == current
    assert bus.disconnected


def test_wiggle_endpoint_routes_can_families(client, monkeypatch):
    import makermodslab.wiggle as wiggle

    calls = []
    monkeypatch.setattr(wiggle, "wiggle_active", False)
    monkeypatch.setattr(
        "makermodslab.can_wiggle._run_and_clear_flag", lambda family, port: calls.append((port, family))
    )
    for family in ("maker", "metal"):
        result = client.post(
            "/api/v1/maker/wiggle-gripper", json={"port": "fake", "arm_type": family, "device_type": "robot"}
        )
        assert result.json()["success"] is True
        wiggle.wiggle_active = False
    assert calls == [("fake", "maker"), ("fake", "metal")]


def test_wiggle_outside_limits_does_not_send_any_goal(monkeypatch):
    import makermodslab.wiggle as wiggle

    bus = GripperBus(3676)
    monkeypatch.setattr(wiggle, "FeetechMotorsBus", lambda **kwargs: bus)
    with pytest.raises(ValueError, match="cannot return"):
        wiggle._wiggle_gripper_sync("fake")
    assert bus.goals == []
    assert bus.disconnected


def test_return_timeout_is_reported(monkeypatch):
    import makermodslab.wiggle as wiggle

    ticks = iter([0.0, 0.5, 1.0, 2.0])
    monkeypatch.setattr(wiggle.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(wiggle.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="did not return"):
        wiggle._wait_for_rest(lambda: 1500, rest=2000, tolerance=10)
