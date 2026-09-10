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
"""Fake motor tests: no CAN connection or hardware operations."""

from types import SimpleNamespace

import pytest

from makermodslab.gripper_soft_limit import install_gripper_soft_limit, validate_gripper_closing_error


class Bus:
    def __init__(self, position=40):
        self.position = position
        self.timestamp = 1000.0
        self.missing = False
        self.malformed = False
        self.writes = []
        self._motor_types = {"gripper": "fake"}

    def _refresh_motor(self, motor):
        assert motor == "gripper"
        return None if self.missing else SimpleNamespace(timestamp=self.timestamp, data=b"fake")

    def _decode_motor_state(self, data, motor_type):
        if self.malformed:
            raise ValueError("bad packet")
        return self.position, 0, 0, 0, 0

    def sync_write(self, register, values):
        self.writes.append(dict(values))

    def sync_write_metal(self, commands):
        self.writes.append(dict(commands))


class Follower:
    def __init__(self, position=40, ff=False):
        self.bus = Bus(position)
        self.config = SimpleNamespace(joint_limits={"gripper": (0.0, 137.5)})
        self.ff = ff

    def send_action(self, action):
        # Model an upstream startup limiter which tries to close farther than
        # this feature allows. The adapter must run AFTER this processing.
        values = {k.removesuffix(".pos"): v for k, v in action.items()}
        if self.ff:
            self.bus.sync_write_metal({k: (20, 0.6, v, -35, 0) for k, v in values.items()})
        else:
            self.bus.sync_write("Goal_Position", values)
        return action


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr("makermodslab.gripper_soft_limit.time.time", lambda: 1000.0)


def enabled(position=40, ff=False):
    robot = Follower(position, ff)
    bus = robot.bus
    install_gripper_soft_limit(robot, SimpleNamespace(supports_gripper_soft_limit=True), 3.0)
    return robot, bus


@pytest.mark.parametrize("ff", [False, True])
def test_stalled_gripper_never_accumulates_and_arms_unchanged(ff):
    robot, bus = enabled(ff=ff)
    for _ in range(10):
        sent = robot.send_action({"gripper.pos": 0, "joint_1.pos": 90})
        assert sent == {"gripper.pos": 37, "joint_1.pos": 90}
        if ff:
            assert bus.writes[-1] == {"gripper": (20, 0.6, 37, 0, 0), "joint_1": (20, 0.6, 90, -35, 0)}
        else:
            assert bus.writes[-1] == {"gripper": 37, "joint_1": 90}
    bus.position = 30
    assert robot.send_action({"gripper.pos": 0})["gripper.pos"] == 27
    assert robot.send_action({"gripper.pos": 80})["gripper.pos"] == 80
    assert robot.send_action({"gripper.pos": 29})["gripper.pos"] == 29


@pytest.mark.parametrize("failure", ["missing", "malformed", "stale", "future", "nan", "range"])
def test_feedback_failure_sends_no_batch_and_restores_wrapper(failure):
    robot, bus = enabled()
    if failure in ("missing", "malformed"):
        setattr(bus, failure, True)
    elif failure == "stale":
        bus.timestamp = 999.8
    elif failure == "future":
        bus.timestamp = 1001
    elif failure == "nan":
        bus.position = float("nan")
    else:
        bus.position = 200
    with pytest.raises((ValueError, ConnectionError)):
        robot.send_action({"gripper.pos": 0, "joint_1.pos": 20})
    assert bus.writes == []
    assert robot.bus.active is False


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_setting(value):
    with pytest.raises(ValueError):
        validate_gripper_closing_error(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_target_rejected_before_upstream(value):
    robot, bus = enabled()
    with pytest.raises(ValueError):
        robot.send_action({"gripper.pos": value})
    assert not bus.writes


def test_disabled_is_identical_and_unsupported_refused():
    robot = Follower()
    original = robot.bus
    install_gripper_soft_limit(robot, SimpleNamespace(supports_gripper_soft_limit=False), None)
    assert robot.bus is original
    assert robot.send_action({"gripper.pos": 0}) == {"gripper.pos": 0}
    with pytest.raises(ValueError):
        install_gripper_soft_limit(robot, SimpleNamespace(supports_gripper_soft_limit=False), 3)


def test_bimanual_installs_each_follower_and_direct_cleanup_passthrough():
    left, right = Follower(20), Follower(50, ff=True)
    robot = SimpleNamespace(left_arm=left, right_arm=right)
    install_gripper_soft_limit(robot, SimpleNamespace(supports_gripper_soft_limit=True), 3)
    assert left.send_action({"gripper.pos": 0})["gripper.pos"] == 17
    assert right.send_action({"gripper.pos": 0})["gripper.pos"] == 47
    # Direct stop/rest bus commands retain the pre-existing cleanup behavior.
    left.bus.sync_write("Goal_Position", {"gripper": 0})
    assert left.bus.writes[-1] == {"gripper": 0}


@pytest.mark.parametrize("bimanual", [False, True])
def test_real_record_loop_saves_sent_gripper_targets(monkeypatch, bimanual):
    from unittest.mock import MagicMock

    import numpy as np

    from lerobot.scripts import lerobot_record
    from lerobot.teleoperators import Teleoperator
    from makermodslab.gripper_soft_limit import record_limited_gripper_actions

    left, _ = enabled(position=40)
    if bimanual:
        right, _ = enabled(position=60, ff=True)
        robot = SimpleNamespace(left_arm=left, right_arm=right, name="bi_metal_follower")

        def send(action):
            return {
                **{
                    "left_" + k: v
                    for k, v in left.send_action(
                        {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
                    ).items()
                },
                **{
                    "right_" + k: v
                    for k, v in right.send_action(
                        {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
                    ).items()
                },
            }

        robot.send_action = send
        targets = {"right_gripper.pos": 0, "left_joint_1.pos": 90, "left_gripper.pos": 0}
        expected = [57, 90, 37]
    else:
        robot = left
        robot.name = "metal_follower"
        targets = {"joint_1.pos": 90, "gripper.pos": 0}
        expected = [90, 37]
    robot.get_observation = lambda: targets.copy()
    events = {"exit_early": False}
    frames = []

    def add(frame):
        frames.append(frame)
        events["exit_early"] = True

    features = {"action": {"dtype": "float32", "shape": (len(targets),), "names": list(targets)}}
    dataset = SimpleNamespace(features=features, fps=30, add_frame=add)
    record_limited_gripper_actions(dataset, robot)
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = targets
    monkeypatch.setattr(lerobot_record, "precise_sleep", lambda _: None)
    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda x: x[0].copy(),
        robot_action_processor=lambda x: x[0].copy(),
        robot_observation_processor=lambda x: x,
        dataset=dataset,
        teleop=teleop,
        control_time_s=1,
        single_task="grasp",
    )
    np.testing.assert_array_equal(frames[0]["action"], expected)
    assert targets[list(targets)[-1]] == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_named_request_inherits_saved_limit_unless_explicit(monkeypatch, explicit):
    from makermodslab.gripper_soft_limit import resolve_request_gripper_soft_limit
    from makermodslab.teleoperate import TeleoperateRequest

    monkeypatch.setattr(
        "makermodslab.gripper_soft_limit.get_robot_record", lambda _: {"gripper_closing_error_deg": 4}
    )
    fields = {"gripper_closing_error_deg": None} if explicit else {}
    request = TeleoperateRequest(
        leader_port="l",
        follower_port="f",
        leader_config="l",
        follower_config="f",
        arm_type="metal",
        robot_name="saved",
        **fields,
    )
    resolve_request_gripper_soft_limit(request)
    assert request.gripper_closing_error_deg == (None if explicit else 4)


def test_invalid_persisted_limit_refused(monkeypatch):
    from makermodslab.api_errors import ApiError
    from makermodslab.gripper_soft_limit import resolve_request_gripper_soft_limit
    from makermodslab.teleoperate import TeleoperateRequest

    monkeypatch.setattr(
        "makermodslab.gripper_soft_limit.get_robot_record",
        lambda _: {"gripper_closing_error_deg": float("nan")},
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
        resolve_request_gripper_soft_limit(request)
