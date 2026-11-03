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
"""Maker gravity feed-forward: the model, its safeguards and the MIT frames, on a fake CAN bus."""

import logging
import math
import time
import tomllib
import types
from pathlib import Path

import can
import numpy as np
import pytest

from lerobot.robots.bi_maker_follower import BiMakerFollower, BiMakerFollowerConfig
from lerobot.robots.maker_follower import MakerFollower, MakerFollowerConfig, MakerFollowerConfigBase
from makermodslab import maker_can, maker_gravity, remote_can, rollout
from makermodslab.arms.maker import MAKER
from makermodslab.arms.metal import METAL
from makermodslab.arms.so101 import SO101
from makermodslab.maker_gravity import ARM_JOINTS, GravityFeedforward, GravityModel, shipped_model
from makermodslab.maker_gripper_hold import MakerHoldingBus
from makermodslab.maker_rest_pose import return_maker_to_pose
from makermodslab.maker_teleop_safety import MakerTeleopSafety
from tests.test_maker_gripper_hold import Device

REPO = Path(__file__).resolve().parent.parent
RAISED = [5.88, -52.67, 52.31, 3.78, 8.85, 4.41]
REFERENCE = [
    (RAISED, [4.3e-08, 1.587526618, 4.681853633, 1.356082357, 0.007371863, -0.000594413]),
    (
        [3.0, -0.3, 10.7, -2.6, 2.9, 2.7],
        [-2e-08, -0.742514025, 4.357497869, 1.348319781, 0.000822936, -0.000443027],
    ),
    ([0, 0, 0, 0, 0, 0], [-1.1e-08, -0.417932457, 4.682138685, 1.365119916, 9.777e-06, 5.4583e-05]),
    ([90, -90, 90, 0, 0, 0], [0.0, 4.680102117, 4.682138647, 1.365119915, 9.788e-06, 5.4583e-05]),
    (
        [-120, -60, 120, 45, -60, 90],
        [4.1e-08, -2.459659448, 0.092139589, -0.350959264, 0.35018096, 0.00101021],
    ),
    ([30, -150, 200, -40, 70, -130], [1.57e-07, 6.54907588, 2.133302941, 1.076258227, 1e-09, 0.00842321]),
    (
        [150, -10, 5, 100, -90, 150],
        [3.5e-08, -1.756668756, 3.266274473, -0.160439469, 0.410769331, -0.009984233],
    ),
    (
        [-45, -120, 30, -65, 78, -153],
        [5e-08, 3.031222621, 0.482950535, -0.920484693, 0.104127816, -0.007291715],
    ),
    (
        [10, -80, 236, 104, -97, 20],
        [-1.22e-07, -4.543230506, -3.655606967, -0.054530978, -0.385022526, -0.008421651],
    ),
    ([0, -175, 3, 0, 0, 0], [1.7e-08, 0.642939037, -4.437547311, -1.348129982, -1.974e-06, -5.5391e-05]),
]


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeBus:
    def __init__(self):
        self.motors = dict.fromkeys((*ARM_JOINTS, "gripper"))
        self.last_feedback_time = dict.fromkeys(self.motors)
        self._last_known_states = {motor: {"position": 0.0} for motor in self.motors}
        self._gains = {motor: {"kp": 30.0, "kd": 2.0} for motor in self.motors}
        self.calls = []

    def _get_motors_list(self, motors):
        if motors is None:
            return list(self.motors)
        return [motors] if isinstance(motors, str) else motors

    def write(self, data_name, motor, value):
        self.calls.append(("write", data_name, motor, value))

    def sync_write(self, data_name, values):
        self.calls.append(("sync_write", data_name, dict(values)))

    def enable_torque(self, motors=None, num_retry=0):
        self.calls.append(("enable", motors))

    def disable_torque(self, motors=None, num_retry=0):
        self.calls.append(("disable", motors))

    def _mit_control(self, motor, kp, kd, position, velocity, torque):
        self.calls.append(("mit", motor, kp, kd, position, velocity, torque))

    def measure(self, pose, at):
        for motor, position in zip(ARM_JOINTS, pose, strict=True):
            self._last_known_states[motor]["position"] = position
            self.last_feedback_time[motor] = at


def full_nm(pose, scale=1.0):
    return shipped_model().holding_nm(pose) * scale


def rig(scale=1.0, enable=True):
    bus, clock = FakeBus(), Clock()
    feedforward = GravityFeedforward(bus, shipped_model(), scale, clock=clock)
    if enable:
        bus.enable_torque()
    bus.calls.clear()
    return bus, clock, feedforward


def send_frame(bus, pose):
    start = len(bus.calls)
    for motor, value in zip(ARM_JOINTS, pose, strict=True):
        bus.write("Goal_Position", motor, value)
    return torques(bus.calls[start:])


def torques(calls):
    return [call[6] if call[0] == "mit" else 0.0 for call in calls]


def warm_up(bus, clock, pose):
    for _ in range(11):
        send_frame(bus, pose)
        clock.now += 0.05
        if bus.last_feedback_time["shoulder_pan"] is not None:
            bus.measure(pose, clock.now)


@pytest.mark.parametrize("pose,expected", REFERENCE)
def test_model_matches_reference(pose, expected):
    assert shipped_model().holding_nm(pose).tolist() == pytest.approx(expected, abs=2e-9)


def test_a_whole_turn_is_the_same_pose():
    turned = [q + 360 * (-1) ** i for i, q in enumerate(RAISED)]
    assert shipped_model().holding_nm(turned).tolist() == pytest.approx(
        shipped_model().holding_nm(RAISED).tolist(), abs=1e-9
    )


def test_wheel_layout_reads_the_urdf_from_dist(monkeypatch, tmp_path):
    shipped_model.cache_clear()
    monkeypatch.setattr(maker_gravity, "URDF_PATHS", (tmp_path / "public.urdf", maker_gravity.URDF_PATHS[1]))
    try:
        assert shipped_model().holding_nm(RAISED).tolist() == pytest.approx(REFERENCE[0][1], abs=2e-9)
    finally:
        shipped_model.cache_clear()
    package_data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    assert package_data["frontend"] == ["dist/**/*"]
    assert maker_gravity.URDF_PATHS[1] == REPO / "frontend" / "dist" / "maker-urdf" / "robot.urdf"


def test_missing_urdf_leaves_the_driver_as_built(monkeypatch, tmp_path, caplog):
    shipped_model.cache_clear()
    monkeypatch.setattr(maker_gravity, "URDF_PATHS", (tmp_path / "none.urdf",))
    try:
        with caplog.at_level(logging.ERROR):
            robot = MakerFollower(MakerFollowerConfig(port="/dev/fake"))
    finally:
        shipped_model.cache_clear()
    assert not hasattr(robot, "gravity_feedforward")
    assert "write" not in vars(robot.bus)
    assert "feed-forward is OFF" in caplog.text


@pytest.mark.parametrize(
    "old,new,match",
    [
        ('<origin xyz="0.00013418072 0.00023828401 0.046440916" rpy="0 0 0"', None, "inertial rpy"),
        ('<axis xyz="0 0 1"', '<axis xyz="0 1 0"', "joint axis"),
    ],
)
def test_a_urdf_the_model_cannot_represent_is_refused(tmp_path, old, new, match):
    source = maker_gravity.URDF_PATHS[0].read_text()
    assert old in source
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(source.replace(old, new or old.replace('rpy="0 0 0"', 'rpy="0 0.1 0"')))
    with pytest.raises(ValueError, match=match):
        GravityModel(urdf)


def test_ease_in_starts_at_zero_and_reaches_full_after_half_a_second():
    bus, clock, _ = rig()
    assert send_frame(bus, RAISED) == [0.0] * 6
    for tick in range(1, 51):
        clock.now += 0.01
        sent = send_frame(bus, RAISED)
        assert sent[2] == pytest.approx(full_nm(RAISED)[2] * tick * 0.02)
    clock.now += 0.01
    assert send_frame(bus, RAISED)[2] == pytest.approx(full_nm(RAISED)[2])


def test_a_stall_advances_the_ease_by_at_most_fifty_ms():
    bus, clock, _ = rig()
    send_frame(bus, RAISED)
    clock.now += 30.0
    assert send_frame(bus, RAISED)[2] == pytest.approx(full_nm(RAISED)[2] * 0.1)
    clock.now -= 5.0
    assert send_frame(bus, RAISED)[2] == pytest.approx(full_nm(RAISED)[2] * 0.1)


def test_torque_on_restarts_the_ease():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    assert send_frame(bus, RAISED)[2] == pytest.approx(full_nm(RAISED)[2])
    bus.enable_torque()
    assert send_frame(bus, RAISED) == [0.0] * 6


def test_frames_sent_before_torque_on_or_after_torque_off_carry_nothing():
    bus, clock, _ = rig(enable=False)
    assert send_frame(bus, RAISED) == [0.0] * 6
    bus.enable_torque()
    warm_up(bus, clock, RAISED)
    bus.disable_torque()
    clock.now += 0.05
    assert send_frame(bus, RAISED) == [0.0] * 6


def test_measured_pose_when_every_joint_replied_within_150_ms():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    bus.measure(REFERENCE[4][0], clock.now - 0.15)
    assert send_frame(bus, RAISED) == pytest.approx(
        np.clip(full_nm(REFERENCE[4][0]), -maker_gravity.LIMIT_NM, maker_gravity.LIMIT_NM).tolist()
    )


@pytest.mark.parametrize("stale", ARM_JOINTS)
def test_one_stale_joint_falls_back_to_the_goals(stale):
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    bus.measure(REFERENCE[4][0], clock.now)
    bus.last_feedback_time[stale] = clock.now - 0.151
    assert send_frame(bus, RAISED) == pytest.approx(full_nm(RAISED).tolist())


def test_non_finite_measurement_or_future_stamp_falls_back_to_the_goals():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    bus.measure(REFERENCE[4][0], clock.now)
    bus._last_known_states["elbow_flex"]["position"] = math.nan
    assert send_frame(bus, RAISED) == pytest.approx(full_nm(RAISED).tolist())
    bus.measure(REFERENCE[4][0], clock.now + 1.0)
    assert send_frame(bus, RAISED) == pytest.approx(full_nm(RAISED).tolist())


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_each_joint_is_capped(sign):
    bus, clock, feedforward = rig()
    bus.measure(RAISED, clock.now)
    warm_up(bus, clock, RAISED)
    feedforward.model = types.SimpleNamespace(holding_nm=lambda pose: np.full(6, sign * 100.0))
    assert send_frame(bus, RAISED) == [sign * limit for limit in [0.5, 15.0, 7.5, 2.0, 0.5, 0.5]]


def test_the_lift_cap_clears_the_model_at_any_scale_inside_the_soft_limits():
    bus, clock, _ = rig(scale=maker_gravity.MAX_SCALE)
    pose = [0, -5, 168, 0, 0, 0]
    warm_up(bus, clock, pose)
    assert full_nm(pose)[1] == pytest.approx(-9.9765, abs=0.001)
    assert send_frame(bus, pose)[1] == pytest.approx(full_nm(pose, maker_gravity.MAX_SCALE)[1])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_goal_sends_zero_torque(bad):
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    clock.now += 1.0
    pose = list(RAISED)
    pose[0] = bad
    assert send_frame(bus, pose)[1:] == [0.0] * 5


def test_scale_multiplies_the_torque():
    bus, clock, _ = rig(scale=0.5)
    warm_up(bus, clock, RAISED)
    assert send_frame(bus, RAISED) == pytest.approx(full_nm(RAISED, 0.5).tolist())


@pytest.mark.parametrize("scale", [-0.1, 1.21, math.nan, math.inf])
def test_scale_out_of_range_is_refused(scale):
    with pytest.raises(ValueError, match="scale"):
        GravityFeedforward(FakeBus(), shipped_model(), scale)


def test_gripper_frames_carry_nothing_and_do_not_start_a_frame():
    bus, clock, feedforward = rig()
    warm_up(bus, clock, RAISED)
    level = feedforward._level
    bus.calls.clear()
    for _ in range(3):
        bus.write("Goal_Position", "gripper", -10.0)
    assert [c[0] for c in bus.calls] == ["write"] * 3
    assert feedforward._level == level
    assert "gripper" not in feedforward._retained


def test_other_writes_pass_through():
    bus, _, _ = rig()
    bus.write("Kp", "elbow_flex", 90.0)
    assert bus.calls == [("write", "Kp", "elbow_flex", 90.0)]


def test_the_model_runs_once_per_frame_on_fresh_feedback():
    bus, clock, feedforward = rig()
    warm_up(bus, clock, RAISED)
    runs = []
    holding_nm = feedforward.model.holding_nm
    feedforward.model = types.SimpleNamespace(holding_nm=lambda pose: runs.append(pose) or holding_nm(pose))
    bus.measure(RAISED, clock.now)
    send_frame(bus, RAISED)
    assert len(runs) == 1


def test_single_joint_frames_use_the_other_joints_measured_position():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    bus.measure(RAISED, clock.now - 1.0)
    bus.calls.clear()
    bus.write("Goal_Position", "elbow_flex", RAISED[2])
    assert torques(bus.calls)[0] == pytest.approx(full_nm(RAISED)[2])


def test_no_goal_and_no_feedback_means_no_torque():
    bus, clock, feedforward = rig()
    bus.write("Goal_Position", "elbow_flex", 50.0)
    clock.now += 0.05
    bus.calls.clear()
    bus.write("Goal_Position", "elbow_flex", 50.0)
    assert feedforward._level > 0
    assert torques(bus.calls) == [0.0]


def test_disable_clears_the_retained_torque_first():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    send_frame(bus, RAISED)
    bus.calls.clear()
    bus.disable_torque()
    assert bus.calls[:6] == [
        ("mit", motor, 30.0, 2.0, goal, 0, 0) for motor, goal in zip(ARM_JOINTS, RAISED, strict=True)
    ]
    assert bus.calls[6:] == [("disable", None)]
    bus.calls.clear()
    bus.disable_torque()
    assert bus.calls == [("disable", None)]


def test_a_failed_clear_never_blocks_the_disable():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    send_frame(bus, RAISED)
    bus.calls.clear()

    def boom(*args):
        raise OSError("bus gone")

    bus._mit_control = boom
    bus.disable_torque()
    assert bus.calls == [("disable", None)]


def test_a_zero_effort_sync_write_leaves_nothing_to_clear():
    bus, clock, _ = rig()
    warm_up(bus, clock, RAISED)
    send_frame(bus, RAISED)
    bus.sync_write("Goal_Position", dict.fromkeys(bus.motors, 0.0))
    bus.calls.clear()
    bus.disable_torque()
    assert bus.calls == [("disable", None)]


def test_gripper_torque_toggles_leave_the_arm_alone():
    bus, clock, feedforward = rig()
    warm_up(bus, clock, RAISED)
    level = feedforward._level
    bus.enable_torque("gripper")
    bus.disable_torque("gripper")
    assert feedforward._level == level
    assert send_frame(bus, RAISED)[2] != 0.0


def _mit_frames(messages):
    return {
        msg.arbitration_id: msg
        for msg in messages
        if msg.arbitration_id in range(1, 8) and bytes(msg.data[:6]) != bytes([255] * 6)
    }


def _decode(msg):
    tmax = 17 if msg.arbitration_id in (2, 3) else 14
    d = msg.data
    return {
        "deg": math.degrees(((d[0] << 8 | d[1]) / 65535 * 2 - 1) * 12.57),
        "kp": ((d[3] & 0xF) << 8 | d[4]) / 4095 * 500,
        "tau": ((d[6] & 0xF) << 8 | d[7]) / 4095 * 2 * tmax - tmax,
    }


def _action(device):
    names = (*ARM_JOINTS, "gripper")
    return {f"{name}.pos": device.positions[i] for i, name in enumerate(names, 1)}


def _arm_pose(device):
    return [device.positions[i] for i in range(1, 7)]


@pytest.fixture
def wired(monkeypatch):
    device = Device()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr("makermodslab.maker_gripper_hold.QUERY_TIMEOUT_S", 0.005)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    robots = []

    def build(holding=False):
        robot = MakerFollower(MakerFollowerConfig(port="/dev/fake"))
        if holding:
            robot.bus = MakerHoldingBus(robot, "maker-gravity", 0.5)
        robots.append(robot)
        return robot

    yield build, device
    for robot in robots:
        if robot.bus.is_connected:
            robot.disconnect()


def _warm(robot, device):
    clock = Clock(now=time.time())
    feedforward = robot.gravity_feedforward
    base = feedforward.bus
    feedforward._clock = clock
    for _ in range(11):
        robot.send_action(_action(device))
        clock.now += 0.05
        for motor in ARM_JOINTS:
            base.last_feedback_time[motor] = clock.now
    feedforward._clock = time.time
    for motor in ARM_JOINTS:
        base.last_feedback_time[motor] = time.time()
    device.messages.clear()


@pytest.mark.parametrize("holding", [False, True])
def test_send_action_puts_the_torque_on_the_wire(wired, holding):
    build, device = wired
    robot = build(holding=holding)
    robot.connect(calibrate=False)
    _warm(robot, device)
    robot.send_action(_action(device))
    frames = _mit_frames(device.messages)
    expected = shipped_model().holding_nm(_arm_pose(device))
    for index, (motor, nm) in enumerate(zip(ARM_JOINTS, expected, strict=True), 1):
        frame = _decode(frames[index])
        assert frame["tau"] == pytest.approx(
            min(maker_gravity.LIMIT_NM[index - 1], max(-maker_gravity.LIMIT_NM[index - 1], nm)), abs=0.03
        )
        assert frame["deg"] == pytest.approx(device.positions[index], abs=0.05)
        assert frame["kp"] == pytest.approx(robot.config.gains[motor][0], abs=0.2)
    assert any(abs(_decode(frames[i])["tau"]) > 0.5 for i in (2, 3, 4))
    if not holding:
        assert _decode(frames[7])["tau"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("via", ["teleop_guard", "recovery_drive", "remote_can"])
def test_each_session_wrapper_reaches_the_wire(wired, via):
    build, device = wired
    robot = build()
    robot.connect(calibrate=False)
    _warm(robot, device)
    guard = MakerTeleopSafety(robot)
    action = _action(device)
    if via == "teleop_guard":
        guard.send_action(action)
    elif via == "recovery_drive":
        action.pop("gripper.pos")
        guard.recovery_drive().send_action(action)
    else:
        remote_can.send_action(robot, action, action)
    assert abs(_decode(_mit_frames(device.messages)[3])["tau"]) > 0.5


def test_return_to_rest_carries_the_torque(wired):
    build, device = wired
    robot = build()
    robot.connect(calibrate=False)
    _warm(robot, device)
    pose = {name: device.positions[i] for i, name in enumerate(ARM_JOINTS, 1)}
    return_maker_to_pose(robot, pose, ensure_target=True, ceiling_s=0.2)
    frames = _mit_frames(device.messages)
    assert abs(_decode(frames[3])["tau"]) > 0.5


def test_bimanual_arms_each_carry_their_own(monkeypatch, tmp_path):
    devices = {"/dev/left": Device(), "/dev/right": Device()}
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: devices[kwargs["channel"]])
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    robot = BiMakerFollower(
        BiMakerFollowerConfig(
            id="bi",
            calibration_dir=tmp_path,
            left_arm_config=MakerFollowerConfigBase(port="/dev/left"),
            right_arm_config=MakerFollowerConfigBase(port="/dev/right"),
        )
    )
    assert robot.left_arm.gravity_feedforward is not robot.right_arm.gravity_feedforward
    robot.connect(calibrate=False)
    try:
        for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
            _warm(arm, devices[f"/dev/{side}"])
        robot.send_action(
            {
                f"{side}_{key}": value
                for side in ("left", "right")
                for key, value in _action(devices["/dev/left"]).items()
            }
        )
        for device in devices.values():
            assert abs(_decode(_mit_frames(device.messages)[3])["tau"]) > 0.5
    finally:
        robot.disconnect()


def test_neutralize_frames_stay_zero_effort_and_are_not_re_armed(wired):
    build, device = wired
    robot = build()
    robot.connect(calibrate=False)
    _warm(robot, device)
    robot.send_action(_action(device))
    device.messages.clear()
    remote_can.neutralize(robot)
    for index in range(1, 8):
        frame = _decode(_mit_frames(device.messages)[index])
        assert frame["tau"] == pytest.approx(0.0, abs=0.01)
        assert frame["kp"] == 0.0
    device.messages.clear()
    robot.bus.disable_torque()
    assert not _mit_frames(device.messages)


def test_engaging_a_hosted_arm_starts_from_zero_torque(wired):
    build, device = wired
    robot = build()
    robot.connect(calibrate=False)
    _warm(robot, device)
    robot.send_action(_action(device))
    device.messages.clear()
    remote_can.engage(robot)
    frames = [m for m in device.messages if m.arbitration_id == 3 and bytes(m.data[:6]) != bytes([255] * 6)]
    assert frames
    assert all(_decode(frame)["tau"] == pytest.approx(0.0, abs=0.01) for frame in frames)


def test_torque_off_sends_a_zero_torque_frame_first(wired):
    build, device = wired
    robot = build()
    robot.connect(calibrate=False)
    _warm(robot, device)
    robot.send_action(_action(device))
    device.messages.clear()
    robot.bus.disable_torque()
    elbow = [m for m in device.messages if m.arbitration_id == 3]
    assert _decode(elbow[0])["tau"] == pytest.approx(0.0, abs=0.01)
    assert bytes(elbow[-1].data) == bytes([255] * 7 + [0xFD])


def test_scale_zero_leaves_the_driver_as_built(wired, monkeypatch):
    monkeypatch.setattr(MAKER, "gravity_feedforward_scale", 0.0)
    build, device = wired
    robot = build()
    assert not hasattr(robot, "gravity_feedforward")
    assert "write" not in vars(robot.bus)
    robot.connect(calibrate=False)
    robot.send_action(_action(device))
    assert all(
        _decode(frame)["tau"] == pytest.approx(0.0, abs=0.01)
        for frame in _mit_frames(device.messages).values()
    )


def test_only_the_maker_family_asks_for_gravity():
    assert MAKER.gravity_feedforward_scale == 1.0
    assert 0.0 <= MAKER.gravity_feedforward_scale <= maker_gravity.MAX_SCALE
    assert METAL.gravity_feedforward_scale == 0.0
    assert SO101.gravity_feedforward_scale == 0.0


def test_other_robstride_buses_and_the_metal_follower_are_untouched():
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig

    assert "write" not in vars(MAKER.gripper_bus("/dev/fake"))
    metal = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    assert not hasattr(metal, "gravity_feedforward")
    assert "write" not in vars(metal.bus)


def test_install_is_idempotent():
    maker_can.install()
    installed = MakerFollower.__init__
    maker_can.install()
    assert MakerFollower.__init__ is installed
    assert installed._makermodslab_gravity is True


@pytest.mark.parametrize(
    "module",
    [
        "maker_rollout.py",
        "eval_runner.py",
        "dagger_runner.py",
        "drtc/robot_rtc.py",
        "drtc/robot_sync.py",
        "utils/robot_factory.py",
    ],
)
def test_every_process_entry_point_installs_the_hook(module):
    assert "_install_maker_can()" in (REPO / "makermodslab" / module).read_text()


@pytest.mark.parametrize("module", ["teleoperate.py", "record.py", "replay.py", "remote_host.py"])
def test_server_flows_build_through_the_factory(module):
    assert ".utils.robot_factory import" in (REPO / "makermodslab" / module).read_text()


def test_rollout_subprocess_runs_the_entry_point_with_the_hook(monkeypatch):
    monkeypatch.setattr(rollout, "_rollout_cli_args", lambda *args: [])
    assert rollout._build_rollout_cmd(None, "policy", [])[1:3] == ["-m", "makermodslab.maker_rollout"]
