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
"""Protocol-level fake CAN; no adapter, motor, thread or service is opened."""

import math
import struct
import time
from types import SimpleNamespace

import pytest

from makermodslab.gripper_current_limit import MetalGripperBus


class Wire:
    def __init__(self):
        self.messages = []
        self.queue = []
        self.mode = 1
        self.enabled = False
        self.drop_force = False
        self.disable_param_only = False
        self.mode_write_ignored = False
        self.missing_joint = False
        self.params = {
            14: b"7018",
            21: struct.pack("<f", 6.28),
            22: struct.pack("<f", 30),
            23: struct.pack("<f", 20),
        }

    def reply(self, data):
        self.queue.append(SimpleNamespace(arbitration_id=0x17, data=data, timestamp=time.time()))

    def state(self):
        self.reply(bytes([0x17 if self.enabled else 7, 0x80, 0, 0, 0, 0, 20, 20]))

    def send(self, msg):
        self.messages.append((msg.arbitration_id, bytes(msg.data)))
        data = bytes(msg.data)
        if msg.arbitration_id not in (7, 0x307, 0x7FF):
            if not self.missing_joint:
                self.queue.append(
                    SimpleNamespace(
                        arbitration_id=msg.arbitration_id + 0x10,
                        data=bytes([0x10 | msg.arbitration_id, 0x80, 0, 0, 0, 0, 20, 20]),
                        timestamp=time.time(),
                    )
                )
            return
        if msg.arbitration_id == 0x7FF:
            if data[2] == 0xCC:
                self.state()
                return
            if data[2] == 0x55:
                assert data[3] == 10  # No firmware/current/scale/flash writes.
                if not self.mode_write_ignored:
                    self.mode = struct.unpack("<I", data[4:])[0]
                self.reply(data)
            else:
                assert data[2] == 0x33
                value = struct.pack("<I", self.mode) if data[3] == 10 else self.params[data[3]]
                self.reply(data[:4] + value)
        elif data == bytes([0xFF] * 7 + [0xFD]):
            self.enabled = False
            # A delayed mode query may be waiting on the same feedback ID.
            self.reply(struct.pack("<HBBI", 7, 0x33, 10, self.mode))
            if not self.disable_param_only:
                self.state()
        elif data == bytes([0xFF] * 7 + [0xFC]):
            self.enabled = True
            self.state()
        elif msg.arbitration_id == 0x307:
            assert self.enabled and self.mode == 4
            if not self.drop_force:
                self.state()
        else:
            raise AssertionError((msg.arbitration_id, data))

    def shutdown(self):
        pass

    def recv(self, timeout):
        return self.queue.pop(0) if self.queue else None


class BaseBus:
    def __init__(self):
        self.canbus = Wire()
        self._motor_types = {"gripper": SimpleNamespace(name="METAL_JHI")}
        self.motors = {"joint_1": object(), "gripper": object()}
        self.calls = []
        self.is_connected = False

    def _recv_motor_response(self, expected_recv_id, timeout):
        msg = self.canbus.recv(timeout)
        return msg if msg and msg.arbitration_id == expected_recv_id else None

    def _get_motor_id(self, name):
        return 7 if name == "gripper" else 1

    def _get_motor_recv_id(self, name):
        return 0x17 if name == "gripper" else 0x11

    def _get_motor_name(self, name):
        return {7: "gripper", 1: "joint_1"}.get(name, name)

    def _get_motors_list(self, motors):
        return list(self.motors) if motors is None else [motors] if isinstance(motors, str) else motors

    def connect(self, handshake):
        self.calls.append(("connect", handshake))
        self.is_connected = True

    def _handshake(self):
        self.calls.append(("handshake",))

    def enable_torque(self, motors, num_retry=0):
        self.calls.append(("enable", motors))

    def disable_torque(self, motors, num_retry=0):
        self.calls.append(("disable", motors))

    def disconnect(self, disable_torque):
        self.calls.append(("disconnect", disable_torque))
        self.is_connected = False

    def _decode_motor_state(self, data, motor_type):
        return 30.0, 0, 0, 20, 20

    def _process_response(self, name, msg):
        pass

    def sync_write(self, register, values):
        self.calls.append((register, values))

    def sync_write_metal(self, commands):
        self.calls.append(("mit", commands))

    def write(self, register, motor, value):
        self.calls.append((register, motor, value))


def make():
    base = BaseBus()
    return MetalGripperBus(base, 0.12349, 50), base


def test_configures_before_enable_then_sends_current_limited_initial_hold():
    bus, base = make()
    bus.connect()
    frames = base.canbus.messages
    mode_write = next(i for i, (_, data) in enumerate(frames) if data[:4] == bytes([7, 0, 0x55, 10]))
    enable = next(i for i, (_, data) in enumerate(frames) if data[-1] == 0xFC)
    assert mode_write < enable
    assert frames[enable + 1][0] == 0x307
    position, velocity, current = struct.unpack("<fHH", frames[enable + 1][1])
    assert position == pytest.approx(math.radians(30))
    assert velocity == int(math.radians(50) * 100)
    assert current == 1234  # Never round a requested ceiling UP.
    assert base.calls == [("connect", False)]
    assert bus.sent_gripper == 30
    count = len(frames)
    bus.enable_torque()  # Stock follower's second enable must not reseed/re-enable.
    assert all(data[-1] != 0xFC for can_id, data in frames[count:] if can_id == 7)


@pytest.mark.parametrize("bad", ["firmware", "scales", "mode", "mode_readback"])
def test_preflight_failure_never_enables_and_cleanup_restores_only_disabled(bad):
    bus, base = make()
    if bad == "firmware":
        base.canbus.params[14] = b"7012"
    elif bad == "scales":
        base.canbus.params[21] = struct.pack("<f", 12.5)
    elif bad == "mode":
        base.canbus.mode = 2
    else:
        base.canbus.mode_write_ignored = True
    with pytest.raises((ValueError, ConnectionError)):
        bus.connect()
    assert not any(data[-1] == 0xFC for _, data in base.canbus.messages)
    assert not any(call[0] == "enable" for call in base.calls)
    bus.disconnect()
    assert base.calls[-1] == ("disconnect", False)


@pytest.mark.parametrize("method", ["sync", "ff", "write", "private", "batch"])
def test_all_gripper_paths_use_force_position_arms_unchanged(method):
    bus, base = make()
    bus.connect()
    if method == "sync":
        bus.sync_write("Goal_Position", {"gripper": 10, "joint_1": 45})
        assert base.calls[-1] == ("Goal_Position", {"joint_1": 45})
    elif method == "ff":
        bus.sync_write_metal({"gripper": (20, 0.6, 10, -30, 9), "joint_1": (20, 0.6, 45, -30, 9)})
        assert base.calls[-1] == ("mit", {"joint_1": (20, 0.6, 45, -30, 9)})
    elif method == "write":
        bus.write("Goal_Position", 7, 10)
    elif method == "private":
        bus._mit_control(7, 20, 0.6, 10, -30, 9)
    else:
        bus._mit_control_batch({7: (20, 0.6, 10, -30, 9)})
    assert base.canbus.messages[-1][0] == 0x307
    assert struct.unpack("<fHH", base.canbus.messages[-1][1])[0] == pytest.approx(math.radians(10))
    assert bus.sent_gripper == 10


def test_disable_confirmation_precedes_mode_restore_even_disconnect_false():
    bus, base = make()
    bus.connect()
    start = len(base.canbus.messages)
    bus.disconnect(disable_torque=False)
    frames = base.canbus.messages[start:]
    assert frames[0][1] == bytes([0xFF] * 7 + [0xFD])
    assert frames[1][1] == struct.pack("<HBBI", 7, 0x55, 10, 1)
    assert base.canbus.mode == 1
    assert not base.canbus.enabled


def test_parameter_echo_cannot_falsely_confirm_disable_or_restore_mode():
    bus, base = make()
    bus.connect()
    base.canbus.disable_param_only = True
    start = len(base.canbus.messages)
    with pytest.raises(ConnectionError):
        bus.disconnect()
    assert not any(data[2] == 0x55 for can_id, data in base.canbus.messages[start:] if can_id == 0x7FF)
    assert base.canbus.mode == 4
    assert base.calls[-1] == ("disconnect", False)


def test_force_feedback_failure_inhibits_later_commands_and_no_arm_batch():
    bus, base = make()
    bus.connect()
    base.canbus.drop_force = True
    start = len(base.calls)
    with pytest.raises(ConnectionError):
        bus.sync_write("Goal_Position", {"gripper": 10, "joint_1": 50})
    assert len(base.calls) == start
    start = len(base.canbus.messages)
    with pytest.raises(ConnectionError):
        bus.write("Goal_Position", "gripper", 20)
    assert len(base.canbus.messages) == start
    bus.disconnect()
    assert base.canbus.mode == 1


def test_ordinary_metal_refuses_leftover_mode4_before_handshake():
    base = BaseBus()
    base.canbus.mode = 4
    bus = MetalGripperBus(base)
    with pytest.raises(ValueError):
        bus.connect()
    assert not base.canbus.enabled
    assert not any(call[0] == "handshake" for call in base.calls)


def test_recovery_open_never_enables_or_configures_and_reconnect_rechecks():
    bus, base = make()
    bus.connect(handshake=False)
    assert base.canbus.messages == []
    bus.disconnect()
    bus.connect()
    bus.disconnect()
    base.canbus.params[14] = b"9999"
    start = len(base.canbus.messages)
    with pytest.raises(ValueError):
        bus.connect()
    assert not any(data[-1] == 0xFC for _, data in base.canbus.messages[start:])


def test_missing_arm_joint_still_fails_handshake():
    bus, base = make()
    base.canbus.missing_joint = True
    with pytest.raises(ConnectionError, match="did not confirm enable"):
        bus.connect()
    bus.disconnect()
    assert not base.canbus.enabled
    assert base.canbus.mode == 1


@pytest.mark.parametrize("ff", [False, True])
@pytest.mark.parametrize("soft", [None, 3.0])
def test_actual_pinned_metal_follower_and_bus_with_fake_can(monkeypatch, tmp_path, ff, soft):
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
    from makermodslab.arms import METAL
    from makermodslab.gripper_soft_limit import install_gripper_soft_limit

    wire = Wire()
    monkeypatch.setattr("can.interface.Bus", lambda **kwargs: wire)
    config = MetalFollowerConfig(
        port="fake",
        calibration_dir=tmp_path,
        id="current-limit-test",
        velocity_feedforward=ff,
        startup_sync_speed_deg=None,
    )
    robot = MetalFollower(config)
    assert robot.bus._motor_types["gripper"].name == "METAL_JHI"
    install_gripper_soft_limit(robot, METAL, soft, 0.2, 50)
    robot.connect(calibrate=False)
    start = len(wire.messages)
    sent = robot.send_action({"gripper.pos": 45.0, "shoulder_pan.pos": 20.0})
    control = [(can_id, data) for can_id, data in wire.messages[start:] if can_id != 0x7FF]
    assert [can_id for can_id, _ in control] == [0x307, 1]
    assert struct.unpack("<fHH", control[0][1]) == pytest.approx(
        (math.radians(45), int(math.radians(50) * 100), 2000)
    )
    assert sent == {"gripper.pos": 45.0, "shoulder_pan.pos": 20.0}
    # Direct stop/rest control cannot accidentally restore MIT on the gripper.
    robot.bus.write("Goal_Position", "gripper", 40)
    assert wire.messages[-1][0] == 0x307
    robot.disconnect()
    assert wire.mode == 1
    assert not wire.enabled
