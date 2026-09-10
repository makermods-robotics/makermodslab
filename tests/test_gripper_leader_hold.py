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
"""Fake-servo tests for the leader gripper hold state machine: no UART, no hardware.

Follower gripper convention (Metal): ``gripper.pos`` degrees, 0 = closed,
increasing = open. "Over-squeezing" is the operator commanding a smaller value
than the follower can physically reach, so ``present - commanded`` grows.
"""

from types import SimpleNamespace

import pytest

from makermodslab.gripper_leader_hold import GripperLeaderHold

GID = 6


class FakeServoBus:
    """Records set_angle / read_angle / unlock, scriptable current raw angle."""

    def __init__(self, raw_deg=12.0):
        self.raw_deg = raw_deg
        self.set_angle_calls = []
        self.unlock_calls = []
        self.read_calls = []
        self.fail_unlock = False

    def read_angle(self, servo_id, multi_turn=True):
        self.read_calls.append((servo_id, multi_turn))
        return SimpleNamespace(raw_deg=self.raw_deg, filtered_deg=self.raw_deg, reliable=True)

    def set_angle(self, servo_id, angle_deg, multi_turn=False, interval_ms=0):
        self.set_angle_calls.append((servo_id, angle_deg, multi_turn, interval_ms))

    def unlock(self, servo_id):
        self.unlock_calls.append(servo_id)
        if self.fail_unlock:
            raise RuntimeError("bus gone")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_hold(bus=None, clock=None, **kw):
    bus = bus or FakeServoBus()
    clock = clock or Clock()
    hold = GripperLeaderHold(bus, GID, gap_deg=6.0, clock=clock, **kw)
    return hold, bus, clock


def drive_to_hold(hold, clock, commanded=4.0, present=12.0):
    """Run the two ticks that take a fresh machine into a live hold."""
    hold.step(commanded=commanded, present=present)  # arms the debounce
    clock.advance(0.2)
    hold.step(commanded=commanded, present=present)  # debounce elapsed -> engage
    assert hold.phase == "holding"


def test_sustained_gap_engages_a_rigid_multi_turn_hold():
    hold, bus, clock = make_hold()
    bus.raw_deg = 11.5

    # Operator commanding 4 deg, follower stuck at 12 -> gap 8 deg, past the 6 deg threshold.
    hold.step(commanded=4.0, present=12.0)
    assert hold.phase == "monitoring"  # debounce not elapsed yet
    assert bus.set_angle_calls == []

    clock.advance(0.2)  # past engage_debounce_s
    hold.step(commanded=4.0, present=12.0)

    assert hold.phase == "holding"
    # Held at the leader gripper's measured raw angle, in the multi_turn frame,
    # with a nonzero interval (the frame trap in the servo-hold reference).
    assert bus.read_calls[-1] == (GID, True)
    servo_id, angle, multi_turn, interval_ms = bus.set_angle_calls[-1]
    assert servo_id == GID
    assert angle == 11.5
    assert multi_turn is True
    assert interval_ms > 0


def test_brief_gap_spike_does_not_engage():
    hold, bus, clock = make_hold()
    hold.step(commanded=4.0, present=12.0)
    clock.advance(0.05)
    hold.step(commanded=10.0, present=12.0)  # gap gone before debounce elapsed
    clock.advance(0.2)
    hold.step(commanded=10.0, present=12.0)
    assert hold.phase == "monitoring"
    assert bus.set_angle_calls == []


def test_hold_reasserts_at_a_bounded_rate_not_every_step():
    hold, bus, clock = make_hold(assert_hz=7.0)
    drive_to_hold(hold, clock)
    n_after_engage = len(bus.set_angle_calls)
    assert n_after_engage == 1

    # Many rapid steps within one re-assert period add no writes.
    for _ in range(20):
        clock.advance(0.001)
        hold.step(commanded=4.0, present=12.0)
    assert len(bus.set_angle_calls) == n_after_engage

    clock.advance(0.2)  # past 1/assert_hz
    hold.step(commanded=4.0, present=12.0)
    assert len(bus.set_angle_calls) == n_after_engage + 1
    assert bus.set_angle_calls[-1][2] is True  # still the multi_turn frame


def test_hold_pulses_open_to_probe_after_the_pulse_interval():
    hold, bus, clock = make_hold(hold_pulse_s=0.4)
    drive_to_hold(hold, clock)
    assert bus.unlock_calls == []

    clock.advance(0.41)
    hold.step(commanded=4.0, present=12.0)
    assert hold.phase == "probing"
    assert bus.unlock_calls == [GID]


def test_probe_detecting_operator_opening_releases_without_re_engaging():
    hold, bus, clock = make_hold(hold_pulse_s=0.4, probe_release_deg=3.0)
    drive_to_hold(hold, clock)  # engage at commanded=4
    clock.advance(0.41)
    hold.step(commanded=4.0, present=12.0)  # -> probing, leader now limp
    reads_before = len(bus.read_calls)

    clock.advance(0.05)
    hold.step(commanded=9.0, present=12.0)  # operator pulled the handle open 5 deg

    assert hold.phase == "monitoring"
    assert bus.unlock_calls[-1] == GID
    assert len(bus.read_calls) == reads_before  # no fresh hold capture


def test_probe_with_operator_still_squeezing_re_engages():
    hold, bus, clock = make_hold(hold_pulse_s=0.4, probe_window_s=0.1)
    drive_to_hold(hold, clock)
    clock.advance(0.41)
    hold.step(commanded=4.0, present=12.0)  # -> probing
    bus.raw_deg = 11.9  # servo drifted a touch while limp

    clock.advance(0.11)
    hold.step(commanded=4.2, present=12.0)  # still hard against the wall

    assert hold.phase == "holding"
    assert bus.set_angle_calls[-1][1] == 11.9  # re-captured the fresh angle


def test_follower_catching_up_releases_the_hold():
    hold, bus, clock = make_hold(release_gap_deg=2.0)
    drive_to_hold(hold, clock)

    # Object removed: the follower closes to the commanded position.
    hold.step(commanded=4.0, present=5.0)
    assert hold.phase == "monitoring"
    assert bus.unlock_calls[-1] == GID


def test_release_unlocks_and_swallows_bus_errors():
    hold, bus, clock = make_hold()
    drive_to_hold(hold, clock)
    bus.fail_unlock = True

    hold.release()  # must not raise
    hold.release()  # idempotent

    assert bus.unlock_calls[-1] == GID
    assert hold.phase == "monitoring"


def test_reset_restarts_the_engage_debounce():
    hold, bus, clock = make_hold()
    hold.step(commanded=4.0, present=12.0)
    hold.reset()  # e.g. follower still doing its startup sync
    clock.advance(0.2)
    hold.step(commanded=4.0, present=12.0)
    assert hold.phase == "monitoring"  # debounce clock restarted by reset

    clock.advance(0.2)
    hold.step(commanded=4.0, present=12.0)
    assert hold.phase == "holding"


def test_missing_present_reading_never_engages():
    hold, bus, clock = make_hold()
    clock.advance(1.0)
    hold.step(commanded=4.0, present=None)
    assert hold.phase == "monitoring"
    assert bus.set_angle_calls == []


# --- install_gripper_leader_hold: the send_action wrapper ---------------------

from makermodslab.gripper_leader_hold import (  # noqa: E402
    install_gripper_leader_hold,
    resolve_request_gripper_leader_hold,
)


class FakeFollowerBus:
    def __init__(self, gripper_present=12.0):
        self.gripper_present = gripper_present
        self.raise_on_read = False

    def sync_read(self, data_name):
        assert data_name == "Present_Position"
        if self.raise_on_read:
            raise RuntimeError("bus hiccup")
        return {"shoulder_pan": 0.0, "gripper": self.gripper_present}


class FakeFollower:
    def __init__(self, gripper_present=12.0):
        self.bus = FakeFollowerBus(gripper_present)
        self._synced = True
        self.sent = []

    def send_action(self, action):
        self.sent.append(dict(action))
        return dict(action)


class FakeLeader:
    def __init__(self):
        self.bus = FakeServoBus(raw_deg=11.5)
        self.config = SimpleNamespace(joint_ids={"shoulder_pan": 1, "gripper": GID})


METAL = SimpleNamespace(supports_gripper_leader_hold=True)


def install(robot, leader, gap_deg=6.0, clock=None, **kw):
    clock = clock or Clock()
    controller = install_gripper_leader_hold(
        robot, leader, METAL, gap_deg, clock=clock, present_poll_s=0.0, **kw
    )
    return controller, clock


def test_disabled_leaves_send_action_untouched():
    robot, leader = FakeFollower(), FakeLeader()
    assert install_gripper_leader_hold(robot, leader, METAL, None) is None
    robot.send_action({"gripper.pos": 4.0})
    assert not hasattr(robot, "_gripper_leader_hold")
    assert leader.bus.set_angle_calls == []


def test_unsupported_family_is_refused():
    robot, leader = FakeFollower(), FakeLeader()
    with pytest.raises(ValueError, match="Metal"):
        install_gripper_leader_hold(robot, leader, SimpleNamespace(supports_gripper_leader_hold=False), 6.0)


def test_bimanual_is_refused_for_now():
    robot = SimpleNamespace(left_arm=FakeFollower(), right_arm=FakeFollower(), send_action=lambda a: a)
    with pytest.raises(ValueError, match="bimanual"):
        install_gripper_leader_hold(robot, FakeLeader(), METAL, 6.0)


def test_wrapper_forwards_the_action_and_engages_on_sustained_over_squeeze():
    robot, leader = FakeFollower(gripper_present=12.0), FakeLeader()
    controller, clock = install(robot, leader)

    out = robot.send_action({"gripper.pos": 4.0, "shoulder_pan.pos": 10.0})
    assert out == {"gripper.pos": 4.0, "shoulder_pan.pos": 10.0}  # untouched passthrough
    assert robot.sent[-1]["gripper.pos"] == 4.0

    clock.advance(0.2)
    robot.send_action({"gripper.pos": 4.0, "shoulder_pan.pos": 10.0})

    assert controller.phase == "holding"
    assert leader.bus.set_angle_calls[-1][0] == GID
    assert leader.bus.set_angle_calls[-1][2] is True  # multi_turn frame
    assert robot._gripper_leader_hold is controller


def test_wrapper_holds_off_while_the_follower_is_still_syncing():
    robot, leader = FakeFollower(gripper_present=12.0), FakeLeader()
    robot._synced = False
    controller, clock = install(robot, leader)

    for _ in range(5):
        clock.advance(0.2)
        robot.send_action({"gripper.pos": 4.0})
    assert controller.phase == "monitoring"
    assert leader.bus.set_angle_calls == []


def test_wrapper_survives_a_follower_read_failure():
    robot, leader = FakeFollower(), FakeLeader()
    robot.bus.raise_on_read = True
    controller, clock = install(robot, leader)
    clock.advance(0.2)
    robot.send_action({"gripper.pos": 4.0})  # must not raise
    assert controller.phase == "monitoring"


def test_resolve_request_inherits_saved_gap_unless_explicit(monkeypatch):
    from makermodslab.teleoperate import TeleoperateRequest

    monkeypatch.setattr(
        "makermodslab.gripper_leader_hold.get_robot_record",
        lambda _: {"gripper_leader_hold_gap_deg": 7.0},
    )
    inherited = TeleoperateRequest(
        leader_port="l",
        follower_port="f",
        leader_config="l",
        follower_config="f",
        arm_type="metal",
        robot_name="saved",
    )
    resolve_request_gripper_leader_hold(inherited)
    assert inherited.gripper_leader_hold_gap_deg == 7.0

    explicit = TeleoperateRequest(
        leader_port="l",
        follower_port="f",
        leader_config="l",
        follower_config="f",
        arm_type="metal",
        robot_name="saved",
        gripper_leader_hold_gap_deg=None,
    )
    resolve_request_gripper_leader_hold(explicit)
    assert explicit.gripper_leader_hold_gap_deg is None


def test_resolve_request_rejects_an_invalid_persisted_gap(monkeypatch):
    from makermodslab.api_errors import ApiError
    from makermodslab.teleoperate import TeleoperateRequest

    monkeypatch.setattr(
        "makermodslab.gripper_leader_hold.get_robot_record",
        lambda _: {"gripper_leader_hold_gap_deg": float("nan")},
    )
    request = TeleoperateRequest(
        leader_port="l",
        follower_port="f",
        leader_config="l",
        follower_config="f",
        arm_type="metal",
        robot_name="saved",
    )
    with pytest.raises(ApiError):
        resolve_request_gripper_leader_hold(request)
