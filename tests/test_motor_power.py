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
"""Tests for makermodslab.motor_power — percent→register mapping, the mocked-bus
apply path (mirrors the FakeRobot/FakeBus patterns in tests/mocks.py), and
the one-shot supply-voltage read (hardware mocked, like tests/test_wiggle.py)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


class _FakeBus:
    """Minimal Feetech bus stand-in: records writes, can fail per motor.

    ``max_torque`` maps motor → the EEPROM Max_Torque_Limit value ``read``
    returns (default 1000; lerobot's configure() sets the gripper to 500)."""

    def __init__(
        self,
        motors: list[str],
        fail: tuple[str, ...] = (),
        port: str = "/dev/fake",
        max_torque: dict[str, int] | None = None,
    ) -> None:
        self.motors = dict.fromkeys(motors)
        self.port = port
        self.writes: list[tuple[str, str, Any, bool, int]] = []
        self._fail = set(fail)
        self._max_torque = max_torque or {}

    def write(
        self, data_name: str, motor: str, value: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        if motor in self._fail:
            raise RuntimeError("write failed")
        self.writes.append((data_name, motor, value, normalize, num_retry))

    def read(self, data_name: str, motor: str, *, normalize: bool = True) -> int:
        if motor in self._fail:
            raise RuntimeError("read failed")
        assert data_name == "Max_Torque_Limit"
        return self._max_torque.get(motor, 1000)


class _FakeArm:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus


class _FakeBimanual:
    def __init__(self, left: _FakeBus, right: _FakeBus) -> None:
        self.left_arm = _FakeArm(left)
        self.right_arm = _FakeArm(right)


def test_torque_limit_from_percent_scales_and_clamps() -> None:
    from makermodslab.motor_power import torque_limit_from_percent

    assert torque_limit_from_percent(100) == 1000
    assert torque_limit_from_percent(55) == 550
    assert torque_limit_from_percent(10) == 100
    # Clamped inputs: below range, above range, junk.
    assert torque_limit_from_percent(5) == 100
    assert torque_limit_from_percent(1000) == 1000
    # Junk/missing (None) falls back to DEFAULT_MOTOR_POWER (38 = the auto-cal
    # torque level, not full power) → 380. This is a fixed constant in
    # utils.config, not a persisted/per-machine read.
    assert torque_limit_from_percent(None) == 380


def test_reset_torque_limit_reseeds_ram_from_eeprom_per_motor() -> None:
    """Every motor's RAM Torque_Limit is restored to its own EEPROM
    Max_Torque_Limit — the exact power-on state (stock lerobot torque)."""
    from makermodslab.motor_power import FOLLOWER, reset_torque_limit

    bus = _FakeBus(
        ["shoulder_pan", "elbow_flex", "gripper"],
        max_torque={"gripper": 500},  # lerobot's configure() stamps 500 there
    )
    warnings = reset_torque_limit(_FakeArm(bus), FOLLOWER)

    assert warnings == []
    # RAM register only — never a write to the EEPROM "Max_Torque_Limit".
    assert bus.writes == [
        ("Torque_Limit", "shoulder_pan", 1000, False, 2),
        ("Torque_Limit", "elbow_flex", 1000, False, 2),
        ("Torque_Limit", "gripper", 500, False, 2),
    ]


def test_reset_torque_limit_failure_warns_but_does_not_abort() -> None:
    """One bad motor must not stop the others, and the failure surfaces as a
    warning message (that motor keeps its previous limit) — never an exception."""
    from makermodslab.motor_power import FOLLOWER, reset_torque_limit

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"], fail=("elbow_flex",))
    warnings = reset_torque_limit(_FakeArm(bus), FOLLOWER)

    written = [w[1] for w in bus.writes]
    assert written == ["shoulder_pan", "gripper"]
    assert len(warnings) == 1
    assert "elbow_flex" in warnings[0]
    assert "Torque_Limit" in warnings[0]
    assert "/dev/fake" in warnings[0]


def test_reset_torque_limit_covers_both_bimanual_arms() -> None:
    from makermodslab.motor_power import FOLLOWER, reset_torque_limit

    left = _FakeBus(["gripper"], port="/dev/left", max_torque={"gripper": 500})
    right = _FakeBus(["gripper"], port="/dev/right", max_torque={"gripper": 500})
    warnings = reset_torque_limit(_FakeBimanual(left, right), FOLLOWER)

    assert warnings == []
    assert left.writes == [("Torque_Limit", "gripper", 500, False, 2)]
    assert right.writes == [("Torque_Limit", "gripper", 500, False, 2)]


def test_reset_torque_limit_handles_none_device() -> None:
    from makermodslab.motor_power import FOLLOWER, reset_torque_limit

    assert reset_torque_limit(None, FOLLOWER) == []


# ---------------------------------------------------------------------------
# clear_goal_velocity: reset the RAM speed cap (Goal_Velocity=0) at session
# start so a leftover cap (auto-cal fold/unfold=1000, rest-pose return=400)
# can't throttle the next session. Mirrors apply_motor_power's shape.
# ---------------------------------------------------------------------------


def test_clear_goal_velocity_zeroes_every_follower_motor() -> None:
    from makermodslab.motor_power import FOLLOWER, clear_goal_velocity

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"])
    warnings = clear_goal_velocity(_FakeArm(bus), FOLLOWER)

    assert warnings == []
    assert len(bus.writes) == 3
    for data_name, _motor, value, normalize, num_retry in bus.writes:
        assert data_name == "Goal_Velocity"
        assert value == 0  # 0 = uncapped servo default
        assert normalize is False
        assert num_retry > 0
    assert {w[1] for w in bus.writes} == {"shoulder_pan", "elbow_flex", "gripper"}


def test_clear_goal_velocity_covers_both_bimanual_arms() -> None:
    from makermodslab.motor_power import FOLLOWER, clear_goal_velocity

    left = _FakeBus(["gripper"], port="/dev/left")
    right = _FakeBus(["gripper"], port="/dev/right")
    warnings = clear_goal_velocity(_FakeBimanual(left, right), FOLLOWER)

    assert warnings == []
    assert left.writes == [("Goal_Velocity", "gripper", 0, False, 2)]
    assert right.writes == [("Goal_Velocity", "gripper", 0, False, 2)]


def test_clear_goal_velocity_failure_warns_but_does_not_abort() -> None:
    """One bad motor must not stop the others, and the failure surfaces as a
    warning message (the motor keeps its leftover cap) — never an exception, so
    the session start still proceeds."""
    from makermodslab.motor_power import FOLLOWER, clear_goal_velocity

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"], fail=("elbow_flex",))
    warnings = clear_goal_velocity(_FakeArm(bus), FOLLOWER)

    written = [w[1] for w in bus.writes]
    assert written == ["shoulder_pan", "gripper"]  # the good motors were still cleared
    assert len(warnings) == 1
    assert "elbow_flex" in warnings[0]
    assert "Goal_Velocity" in warnings[0]
    assert "/dev/fake" in warnings[0]


def test_clear_goal_velocity_handles_none_device() -> None:
    from makermodslab.motor_power import FOLLOWER, clear_goal_velocity

    assert clear_goal_velocity(None, FOLLOWER) == []


def test_session_request_models_have_no_motor_power_field() -> None:
    """The torque slider drives AUTO-CALIBRATION only; session requests
    (teleop/record/inference) no longer carry a motor_power knob. A stale
    client still sending the field is ignored, not an error."""
    from makermodslab.auto_calibrate import AutoCalibrationBatchRequest, AutoCalibrationRequest
    from makermodslab.record import RecordingRequest
    from makermodslab.rollout import InferenceRequest
    from makermodslab.teleoperate import TeleoperateRequest

    for model in (TeleoperateRequest, RecordingRequest, InferenceRequest):
        assert "motor_power" not in model.model_fields

    # Auto-calibration requests DO carry it (percent, None = script default).
    assert AutoCalibrationRequest.model_fields["motor_power"].default is None
    assert AutoCalibrationBatchRequest.model_fields["motor_power"].default is None

    teleop = TeleoperateRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="L",
        follower_config="F",
        motor_power=40,  # stale-client field: ignored
    )
    assert not hasattr(teleop, "motor_power")


def test_voltage_from_raw_scales_tenths_of_a_volt() -> None:
    """Present_Voltage is in 0.1 V units (121 raw = 12.1 V)."""
    from makermodslab.motor_power import voltage_from_raw

    assert voltage_from_raw(121) == 12.1
    assert voltage_from_raw(0) == 0.0


async def test_read_supply_voltage_rejects_empty_port() -> None:
    from makermodslab.motor_power import read_supply_voltage

    result = await read_supply_voltage("   ")
    assert result == {"success": False, "message": "No port provided."}


def test_supply_voltage_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the blocking hardware call so the happy path runs without a device.
    monkeypatch.setattr("makermodslab.motor_power._read_voltage_sync", lambda port: 12.1)

    response = client.get("/supply-voltage", params={"port": "/dev/fake"})
    assert response.status_code == 200
    assert response.json() == {"success": True, "voltage": 12.1}


def test_supply_voltage_endpoint_reports_hardware_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(port: str) -> float:
        raise RuntimeError("no device on port")

    monkeypatch.setattr("makermodslab.motor_power._read_voltage_sync", boom)

    response = client.get("/supply-voltage", params={"port": "/dev/fake"})
    # Logical failures stay HTTP 200 with success=False (like other handlers).
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "no device on port" in body["message"]


# -- the leader gate -------------------------------------------------------
#
# The runtime half of the fix. The static half is
# tests/test_motor_power_call_sites.py, which checks that call sites STATE a
# side; this checks what happens when one states "leader".


def test_clear_goal_velocity_refuses_a_leader_without_writing() -> None:
    """The defect this gate exists for: a leader reached a follower-only register.

    The refusal must be total — not a best-effort write that mostly fails — so
    the bus is asserted untouched, not merely "no exception raised"."""
    from makermodslab.motor_power import LEADER, clear_goal_velocity

    bus = _FakeBus(["shoulder_pan", "gripper"])
    warnings = clear_goal_velocity(_FakeArm(bus), LEADER)

    assert len(warnings) == 1
    assert "follower-only" in warnings[0]
    assert bus.writes == [], "a refused call must not touch the bus at all"


def test_clear_goal_velocity_still_writes_for_a_stated_follower() -> None:
    """The gate must not be so eager that it breaks the legitimate caller —
    every session start depends on this write landing."""
    from makermodslab.motor_power import FOLLOWER, clear_goal_velocity

    bus = _FakeBus(["shoulder_pan"])
    assert clear_goal_velocity(_FakeArm(bus), FOLLOWER) == []
    assert [(w[0], w[1], w[2]) for w in bus.writes] == [("Goal_Velocity", "shoulder_pan", 0)]


def test_reset_torque_limit_serves_the_leader_because_coaching_drives_it() -> None:
    """`reset_torque_limit` must NOT inherit the leader refusal.

    Coaching glides the leader to the follower's pose under its own torque, so a
    leader still capped by an earlier auto-calibration cannot carry its own arm.
    The two registers are asymmetric; that asymmetry is the whole point of
    passing `side` rather than banning leaders wholesale."""
    from makermodslab.motor_power import LEADER, reset_torque_limit

    bus = _FakeBus(["shoulder_pan"], max_torque={"shoulder_pan": 900})
    assert reset_torque_limit(_FakeArm(bus), LEADER) == []
    assert [(w[0], w[1], w[2]) for w in bus.writes] == [("Torque_Limit", "shoulder_pan", 900)]


def test_an_unknown_side_is_rejected_rather_than_guessed() -> None:
    """Fail closed. A typo must not fall through to the permissive branch —
    that is how an optional, prose-defaulted parameter failed the first time."""
    import pytest

    from makermodslab.motor_power import clear_goal_velocity, reset_torque_limit

    bus = _FakeBus(["shoulder_pan"])
    for helper in (clear_goal_velocity, reset_torque_limit):
        with pytest.raises(ValueError, match="side must be one of"):
            helper(_FakeArm(bus), "follower arm")  # the old prose label
    assert bus.writes == []
