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


def test_plan_wiggle_position_beyond_max_limit_plans_inside_window() -> None:
    # Real case: gripper parked at 3676 with programmed limits 1387-2707;
    # "+200" used to clamp down to 2707 and jog the wrong way first.
    high, low, rest = plan_wiggle(3676, 1387, 2707)
    assert (high, low, rest) == (2707, 2307, 2507)


def test_plan_wiggle_position_near_min_limit_shifts_up() -> None:
    # Real case: gripper at 1461 with min limit 1387 — the -200 jog clamped.
    high, low, rest = plan_wiggle(1461, 1387, 2707)
    assert (high, low, rest) == (1787, 1387, 1587)


def test_plan_wiggle_limits_wider_than_factory_range_are_clamped() -> None:
    high, low, rest = plan_wiggle(50, -500, 9000)
    assert low >= 0
    assert high <= 4095
    assert rest == 200


def test_plan_wiggle_too_narrow_window_raises_legible_error() -> None:
    with pytest.raises(ValueError, match="too narrow"):
        plan_wiggle(2000, 1900, 2100)
