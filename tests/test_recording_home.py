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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from makermodslab import maker_rest_pose, record, recording_home
from makermodslab.arms import registry


class Arm:
    def __init__(self):
        self.positions = {"shoulder_pan": 12.0, "shoulder_lift": -12.0, "gripper": -50.0}
        self.bus = SimpleNamespace(motors=dict.fromkeys(self.positions))
        self.config = SimpleNamespace(joint_limits={"shoulder_pan": (-90, 90), "shoulder_lift": (-90, -3.2)})
        self.sent = []

    def get_observation(self):
        return {f"{motor}.pos": value for motor, value in self.positions.items()}

    def send_action(self, action):
        self.sent.append(action)
        self.positions.update({key.removesuffix(".pos"): value for key, value in action.items()})


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    fake = SimpleNamespace(
        monotonic=lambda: now[0], perf_counter=lambda: now[0], time=lambda: now[0], sleep=sleep
    )
    monkeypatch.setattr(maker_rest_pose, "time", fake)
    monkeypatch.setattr(record, "time", fake)
    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", sleep)
    return now


@pytest.mark.parametrize("bimanual", [False, True])
def test_zero_targets_obey_limits_preserve_gripper_and_use_home_rate(clock, bimanual):
    arms = [Arm(), Arm()] if bimanual else [Arm()]
    robot = SimpleNamespace(left_arm=arms[0], right_arm=arms[1]) if bimanual else arms[0]
    speed = registry.get("maker").recording_home_speed_deg_s
    assert speed == pytest.approx(35.15625)
    arrived, targets = recording_home.return_recording_home(robot, speed, lambda: False)
    assert arrived
    assert len(targets) == len(arms)
    for arm, pose in targets:
        assert pose == {"shoulder_pan": 0, "shoulder_lift": -3.2}
        assert arm.positions["gripper"] == -50.0
        assert all("gripper.pos" not in command for command in arm.sent)
        assert arm.positions["shoulder_pan"] == pytest.approx(0)
        assert 12.0 - arm.sent[0]["shoulder_pan.pos"] <= speed / 30


def test_invalid_second_arm_prevents_any_home_commands(clock):
    left, right = Arm(), Arm()
    right.config.joint_limits["shoulder_lift"] = (10, -10)
    assert not recording_home.return_recording_home(
        SimpleNamespace(left_arm=left, right_arm=right), 30, lambda: False
    )[0]
    assert not left.sent and not right.sent


def test_stop_during_home_prevents_success(clock):
    arm = Arm()
    arrived, _ = recording_home.return_recording_home(arm, 30, lambda: bool(arm.sent))
    assert not arrived
    assert len(arm.sent) == 1


def test_home_reset_ignores_leader_but_honors_stop(clock):
    arm, leader, realign = Arm(), Mock(), Mock()
    events = {"paused": False, "stop_recording": False}
    observations = []

    def observe(obs):
        observations.append(obs)
        if len(observations) == 3:
            events["stop_recording"] = True

    record._reset_loop_with_pause(
        arm,
        leader,
        events,
        30,
        None,
        None,
        10,
        realign=realign,
        observation_callback=observe,
        hold_position=True,
    )
    assert len(observations) == 3
    assert not arm.sent
    leader.get_action.assert_not_called()
    realign.assert_not_called()


@pytest.mark.parametrize("home_ok", [True, False])
def test_per_task_returns_home_before_naming_or_saves_on_failure(monkeypatch, clock, home_ok):
    events = {"stop_recording": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "saved_episodes", 0)
    monkeypatch.setattr(record, "discard_requested", False)
    calls = []
    dataset = SimpleNamespace(
        save_episode=lambda: calls.append("save"), clear_episode_buffer=lambda: calls.append("clear")
    )

    def preview():
        calls.append("naming")
        events["episode_task_submitted"] = "pick"

    def prepare():
        calls.append("prepare")
        return True

    def finish():
        calls.append("home")
        return home_ok

    record._record_task_episodes(
        SimpleNamespace(dataset=SimpleNamespace(num_episodes=2)),
        dataset,
        events,
        lambda task: calls.append("capture"),
        prepare,
        preview,
        finish_episode=finish,
    )
    if home_ok:
        assert calls == [
            "naming",
            "prepare",
            "capture",
            "home",
            "naming",
            "save",
            "prepare",
            "capture",
            "home",
            "save",
        ]
    else:
        assert calls == ["naming", "prepare", "capture", "home", "save"]


def test_maker_home_overwrites_old_goal_even_when_observation_is_already_home(clock):
    arm = Arm()
    arm.positions.update(shoulder_pan=0.0, shoulder_lift=-3.2)
    # The previous recording target can still be far away while feedback lags.
    arm.sent.append({"shoulder_pan.pos": 60.0, "shoulder_lift.pos": -40.0})
    before = len(arm.sent)
    assert recording_home.return_recording_home(arm, 35.15625, lambda: False)[0]
    assert len(arm.sent) > before
    assert arm.sent[-1] == {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -3.2}
    assert all("gripper.pos" not in command for command in arm.sent[before:])


@pytest.mark.parametrize("bimanual", [False, True])
def test_metal_home_uses_pinned_limits_and_only_follower_commands(clock, bimanual):
    from lerobot.robots.metal_follower import MetalFollowerConfig

    arms = [Arm() for _ in range(2 if bimanual else 1)]
    for arm in arms:
        arm.config = MetalFollowerConfig(port="unused")
        arm.positions = {
            motor: max(low, min(high, 12.0)) for motor, (low, high) in arm.config.joint_limits.items()
        }
        arm.positions["gripper"] = 50.0
        arm.bus = SimpleNamespace(motors=dict.fromkeys(arm.positions))
    robot = SimpleNamespace(left_arm=arms[0], right_arm=arms[1]) if bimanual else arms[0]
    family = registry.get("metal")
    assert family.recording_realign_speed_deg_s == 60.0
    assert family.recording_home_speed_deg_s == pytest.approx(35.15625)
    assert family.recording_home_is_rest_pose is False
    ok, targets = recording_home.return_recording_home(
        robot, family.recording_home_speed_deg_s, lambda: False
    )
    assert ok
    for arm, target in targets:
        assert target == dict.fromkeys(set(arm.positions) - {"gripper"}, 0.0)
        assert arm.positions["gripper"] == 50.0
        assert all("gripper.pos" not in action for action in arm.sent)
        assert all(value == 0.0 for motor, value in arm.positions.items() if motor != "gripper")


def test_metal_home_finishes_with_zero_velocity_using_pinned_follower(monkeypatch, clock):
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig, metal_follower as driver

    monkeypatch.setattr(driver, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    arm = MetalFollower.__new__(MetalFollower)
    arm.id = "mock-metal"
    arm.config = MetalFollowerConfig(port="unused")
    arm._synced = True
    arm._joint_motor_names = list(arm.config.motor_can_ids)
    arm._resolved_gains = dict(arm.config.gains)
    arm._reset_velocity_feedforward()
    arm.cameras = {}
    positions = {motor: max(low, min(high, 12.0)) for motor, (low, high) in arm.config.joint_limits.items()}
    positions["gripper"] = 50.0
    commands = []

    def write(frame):
        commands.append(frame)
        positions.update({motor: command[2] for motor, command in frame.items()})

    arm.bus = SimpleNamespace(
        is_connected=True,
        motors=dict.fromkeys(positions),
        sync_read=lambda *a, **k: dict(positions),
        sync_write_metal=write,
    )
    family = registry.get("metal")
    ok, _ = recording_home.return_recording_home(
        arm, family.recording_home_speed_deg_s, lambda: False, hold=family.hold_recording_home
    )
    assert ok
    assert any(command[3] != 0 for frame in commands[:-1] for command in frame.values())
    assert all(command[3] == 0 for command in commands[-1].values())
    assert all(command[:2] == arm.config.gains[motor] for motor, command in commands[-1].items())
    assert all("gripper" not in frame for frame in commands)
    assert positions["gripper"] == 50.0


def test_home_hold_failure_returns_false_for_graceful_cleanup(clock):
    def broken_hold(targets):
        raise RuntimeError("hold command failed")

    ok, _ = recording_home.return_recording_home(Arm(), 35.15625, lambda: False, hold=broken_hold)
    assert ok is False
