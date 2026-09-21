"""Start-pose capture and the ease-in target mapping in `makermodslab.drtc._pose`.

Pure-helper test: no LiveKit, no serial port, no sleeps. `_pose` is the half of
the `robot_sync` teardown/ease-in that does NOT import `livekit.portal`, which
is precisely so these can run in ordinary CI.

Deliberately NOT covered here, per the repo's tests policy: the poll-until-
arrived loop inside `rest_pose.return_to_rest_pose` (it settles for
RETURN_SETTLE_S and is already covered by the replay tests, and driving it here
would mean sleeping), and anything that needs a real bus.
"""

from __future__ import annotations

import pytest

from makermodslab import replay
from makermodslab.drtc import _pose


class FakeBus:
    """The slice of a motors bus these helpers touch."""

    def __init__(self, motors, positions=None, fail=False):
        self.motors = dict.fromkeys(motors)
        self._positions = positions or dict.fromkeys(motors, 100)
        self._fail = fail
        self.writes: list[tuple] = []

    def sync_read(self, register, normalize=False):
        if self._fail:
            raise RuntimeError("bus is dark")
        if register == "Goal_Velocity":
            return dict.fromkeys(self.motors, 0)
        return dict(self._positions)

    def sync_write(self, register, values, normalize=False):
        self.writes.append((register, dict(values)))


class FakeArm:
    def __init__(self, bus):
        self.bus = bus


class FakeBimanual:
    def __init__(self, left, right):
        self.left_arm = FakeArm(left)
        self.right_arm = FakeArm(right)


SO101_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture
def feetech(monkeypatch):
    """Treat FakeBus as the Feetech bus type.

    `feetech_buses` gates on the real class so a Dynamixel (Koch/OMX) arm never
    gets `rest_pose`'s Feetech-unit constants applied to it — see _pose.py. The
    gate is a module-level tuple exactly so a test can substitute a fake."""
    monkeypatch.setattr(_pose, "FEETECH_BUS_TYPES", (FakeBus,))


# --- bus discovery ----------------------------------------------------------


def test_a_single_arm_exposes_one_bus(feetech):
    bus = FakeBus(SO101_MOTORS)

    assert _pose.feetech_buses(FakeArm(bus)) == [bus]


def test_a_bimanual_robot_exposes_both_buses(feetech):
    left, right = FakeBus(SO101_MOTORS), FakeBus(SO101_MOTORS)

    assert _pose.feetech_buses(FakeBimanual(left, right)) == [left, right]


def test_a_non_feetech_arm_exposes_no_bus():
    """Koch/OMX are Dynamixel: RETURN_POS_SPEED and the tick tolerances mean
    something else entirely there, so every caller must see an empty list and
    skip rather than drive the arm in the wrong units."""
    assert _pose.feetech_buses(FakeArm(FakeBus(SO101_MOTORS))) == []


def test_a_missing_device_exposes_no_bus(feetech):
    assert _pose.feetech_buses(None) == []


# --- start-pose capture -----------------------------------------------------


def test_the_captured_start_pose_excludes_the_gripper(feetech):
    bus = FakeBus(SO101_MOTORS, positions=dict.fromkeys(SO101_MOTORS, 2048))

    [(captured_bus, pose)] = _pose.capture_start_poses(FakeArm(bus))

    assert captured_bus is bus
    assert "gripper" not in pose
    assert set(pose) == set(SO101_MOTORS) - {"gripper"}


def test_a_dark_bus_captures_an_empty_pose_rather_than_raising(feetech):
    """A session must not fail to start over an optional nicety; an empty pose
    simply makes the return a no-op that reports `no-pose`."""
    [(_bus, pose)] = _pose.capture_start_poses(FakeArm(FakeBus(SO101_MOTORS, fail=True)))

    assert pose == {}


def test_both_arms_are_captured_for_a_bimanual_robot(feetech):
    poses = _pose.capture_start_poses(FakeBimanual(FakeBus(SO101_MOTORS), FakeBus(SO101_MOTORS)))

    assert len(poses) == 2
    assert all("gripper" not in pose for _bus, pose in poses)


# --- action -> bus target mapping -------------------------------------------


def test_suffixed_action_keys_are_mapped_onto_bare_motor_names():
    """lerobot's action features carry `<motor>.pos`; `bus.motors` does not.
    Passing the unconverted dict through matches zero motors and yields
    "no-pose" without ever writing a Goal_Position."""
    bus = FakeBus(SO101_MOTORS)
    action = {f"{m}.pos": float(i) for i, m in enumerate(SO101_MOTORS)}

    assert _pose.bus_keyed(action, bus) == {m: float(i) for i, m in enumerate(SO101_MOTORS)}


def test_bare_action_keys_still_map():
    bus = FakeBus(["shoulder_pan"])

    assert _pose.bus_keyed({"shoulder_pan": 1.0}, bus) == {"shoulder_pan": 1.0}


def test_the_suffixed_key_wins_when_both_forms_are_present():
    bus = FakeBus(["shoulder_pan"])

    keyed = _pose.bus_keyed({"shoulder_pan.pos": 1.0, "shoulder_pan": 9.0}, bus)

    assert keyed == {"shoulder_pan": 1.0}


def test_motors_absent_from_the_action_are_left_out():
    bus = FakeBus(SO101_MOTORS)

    assert _pose.bus_keyed({"elbow_flex.pos": 3.0}, bus) == {"elbow_flex": 3.0}


def test_the_ease_in_target_includes_the_gripper():
    """The exclusion is about the CAPTURED pose, not the target: the policy's
    own first action drives the gripper like every other joint."""
    bus = FakeBus(SO101_MOTORS)
    action = {f"{m}.pos": 1.0 for m in SO101_MOTORS}

    assert "gripper" in _pose.bus_keyed(action, bus)


# --- ease-in refusals (the branches that never touch the bus) ---------------


def test_the_ease_in_is_skipped_for_a_non_feetech_arm():
    arrived, reason = _pose.ease_to_action(FakeArm(FakeBus(SO101_MOTORS)), {"shoulder_pan.pos": 1.0})

    assert not arrived
    assert reason == "unsupported (no Feetech bus)"


def test_the_ease_in_is_skipped_for_a_bimanual_robot(feetech):
    """A BiSO robot's action keys are `left_`/`right_` prefixed while each
    sub-arm's motors are bare, so `bus_keyed` would silently match nothing per
    bus. Refuse rather than guess a prefix convention — return-to-rest has no
    such problem, it works per bus in raw ticks."""
    device = FakeBimanual(FakeBus(SO101_MOTORS), FakeBus(SO101_MOTORS))

    arrived, reason = _pose.ease_to_action(device, {"left_shoulder_pan.pos": 1.0})

    assert not arrived
    assert reason == "unsupported (2 buses)"


def test_the_ease_in_reports_no_pose_when_nothing_matches(feetech):
    arrived, reason = _pose.ease_to_action(FakeArm(FakeBus(SO101_MOTORS)), {"unrelated.pos": 1.0})

    assert not arrived
    assert reason == "no-pose"


def test_no_bus_is_written_to_on_any_refusal_path(feetech):
    bus = FakeBus(SO101_MOTORS)

    _pose.ease_to_action(FakeArm(bus), {"unrelated.pos": 1.0})

    assert bus.writes == []


# --- return-to-rest fan-out -------------------------------------------------


def test_returning_with_no_captured_buses_is_a_no_op():
    assert _pose.return_to_start_poses([]) == []


# --- the normalized-unit constants ------------------------------------------


def test_the_ease_constants_match_their_replay_twins():
    """Hand-mirrored, so pin them: `replay` derives both values and this module
    restates them to avoid importing the FastAPI-side session machinery into a
    subprocess. Change one, change the other."""
    assert _pose.EASE_ARRIVE_TOLERANCE == replay.EASE_ARRIVE_TOLERANCE
    assert _pose.EASE_STALL_MIN_PROGRESS == replay.EASE_STALL_MIN_PROGRESS
