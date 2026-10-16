"""The Metal gripper motor answers late, twice, or not at all; no hardware.

Measured on two arms and two adapters: motor 7 often emits its reply to
request N only when the NEXT frame appears on the bus, sometimes delivers that
held reply twice, and occasionally never answers. A stale reply can also
survive into the next process and pop out on its first transmit.
"""

import can
import pytest

from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
from makermodslab import metal_gripper as grip
from makermodslab.metal_gripper_hold import MetalHoldingBus
from tests.test_metal_gripper import Device
from tests.test_metal_gripper_hold import HoldingDevice

ENABLE = bytes([255] * 7 + [0xFC])
DISABLE = bytes([255] * 7 + [0xFD])


class LateReplies:
    """Hold each gripper (0x17) reply until the next frame reaches the bus."""

    def __init__(self):
        super().__init__()
        self.held = []
        self.late = True
        self.duplicate = False
        self.lose = set()  # indexes of gripper replies that are never emitted
        self.mute = False  # the gripper stops answering entirely
        self.mute_params = False
        self.frozen = set()  # registers that ack a write but keep their value
        self.refuse_enable = False
        self.count = 0

    def send(self, msg):
        # The new frame on the bus releases whatever the gripper was holding.
        self.replies.extend(self.held)
        self.held = []
        d = bytes(msg.data)
        write = msg.arbitration_id == 0x7FF and d[2] == 0x55 and d[3] in self.frozen
        old = self.registers.get(d[3]) if write else None
        if self.refuse_enable and msg.arbitration_id == 7 and d == ENABLE:
            # The motor ignores the enable: it answers as a disabled motor.
            msg = can.Message(arbitration_id=7, is_extended_id=False, data=DISABLE)
        queued = len(self.replies)
        super().send(msg)
        if write:
            self.registers[d[3]] = old
        fresh = [self.replies.pop() for _ in range(len(self.replies) - queued)][::-1]
        for reply in fresh:
            if reply.arbitration_id != 0x17:
                self.replies.append(reply)
                continue
            index, self.count = self.count, self.count + 1
            param = msg.arbitration_id == 0x7FF and d[2] in (0x33, 0x55)
            if self.mute or index in self.lose or (param and self.mute_params):
                continue
            if not self.late:
                self.replies.append(reply)
            else:
                self.held += [reply, reply] if self.duplicate else [reply]


class LateDevice(LateReplies, Device):
    pass


class LateHoldingDevice(LateReplies, HoldingDevice):
    pass


def stale_state(status, temperature):
    msg = Device().state()
    msg.data[0] = status << 4 | 7
    msg.data[6] = msg.data[7] = temperature
    return msg


@pytest.fixture
def late_rig(monkeypatch):
    device = LateHoldingDevice()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr(grip, "QUERY_TIMEOUT_S", 0.05)
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    bus = MetalHoldingBus(robot.bus, "late", 0.5, (0.0, 137.5), gains=(20.0, 0.6))
    robot.bus = bus
    yield robot, bus, device
    if bus._base.canbus is not None:
        device.mute = device.mute_params = False
        device.frozen.clear()
        bus.disconnect()


@pytest.mark.parametrize("duplicate", [False, True])
def test_holding_starts_and_stops_with_every_gripper_reply_one_frame_late(late_rig, duplicate):
    _, bus, device = late_rig
    device.duplicate = duplicate
    bus.connect()
    assert device.registers[9] == 10000 and device.registers[10] == 1
    assert bus._original_timeout == 0
    # The held reply to the zero-gain preload (disabled) pops out when the
    # enable frame hits the bus; it must not be read as the enable's answer.
    bus.enable_torque("gripper")
    assert device.status == 1 and bus._enabled and not bus._error
    bus.disconnect()
    assert device.status == 0
    assert device.registers[9] == 0
    assert device.closed


def test_lost_reply_is_provoked_by_a_flush_not_waited_out(late_rig):
    _, bus, device = late_rig
    # The first answer is lost and the re-issued request's answer is held:
    # only a flush frame gets it onto the bus inside the timeout.
    device.lose = {0}
    bus.connect()
    bus.enable_torque("gripper")
    assert device.status == 1 and not bus._error
    flushes = [m for m in device.messages if m.arbitration_id == 0x7FF and m.data[:3] == bytes([1, 0, 0xCC])]
    assert flushes, "a held reply must be provoked with a read-only refresh of another joint"


def test_stale_replies_from_a_previous_process_are_not_this_sessions_answers(late_rig):
    _, bus, device = late_rig
    stale_ack = can.Message(
        arbitration_id=0x17, is_extended_id=False, data=bytes([7, 0, 0x33, 9]) + (10000).to_bytes(4, "little")
    )
    device.held = [stale_state(1, 60), stale_ack]
    device.replies.extend([stale_state(1, 60)])
    bus.connect()
    assert bus._original_timeout == 0
    assert bus.temperature_c == 30 and bus._status == 0
    bus.disconnect()
    assert device.registers[9] == 0


def test_register_that_does_not_retain_a_write_still_fails_closed(late_rig):
    _, bus, device = late_rig
    device.duplicate = True
    device.frozen = {9}
    with pytest.raises(grip.GripperSafetyError, match="did not retain"):
        bus.connect()
    assert all(bytes(m.data) != ENABLE for m in device.messages)
    assert device.status == 0 and device.closed


def test_silent_gripper_fails_closed_and_is_still_sent_the_release(late_rig):
    _, bus, device = late_rig
    device.mute = True
    with pytest.raises((ConnectionError, grip.GripperSafetyError), match="No fresh gripper response"):
        bus.connect()
    assert all(bytes(m.data) != ENABLE for m in device.messages)
    assert any(m.arbitration_id == 7 and bytes(m.data) == DISABLE for m in device.messages)
    assert device.closed


def test_motor_that_refuses_enable_is_not_confirmed_by_a_late_frame(late_rig):
    _, bus, device = late_rig
    device.duplicate = True
    device.refuse_enable = True
    bus.connect()
    with pytest.raises(grip.GripperSafetyError, match="did not enable"):
        bus.enable_torque("gripper")
    assert device.status == 0 and bus._error and not bus._enabled


def test_enabled_gripper_is_released_even_when_its_mode_is_unreadable(late_rig):
    _, bus, device = late_rig
    bus.connect()
    bus.enable_torque("gripper")
    device.mute_params = True
    with pytest.raises(grip.GripperSafetyError, match="cleanup"):
        bus.disconnect()
    assert device.status == 0
    assert device.closed


def test_stale_feedback_stops_the_hold_even_with_flushes(late_rig):
    _, bus, device = late_rig
    bus.connect()
    bus.enable_torque("gripper")
    with bus._lock:
        device.mute = True
        with pytest.raises(ConnectionError, match="No fresh gripper response"):
            bus.sync_write("Goal_Position", {"gripper": 0})
        assert bus._error and not bus._enabled


def test_punctual_replies_add_no_flush_frames_and_queued_frames_are_dropped(monkeypatch):
    device = Device()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr(grip, "HOLD_INTERVAL_S", 60)
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    bus = grip.MetalGripperBus(robot.bus, "punctual", 0.5, (0.0, 137.5), speed_limit_deg_s=120)
    bus.connect()
    try:
        device.messages.clear()
        # An answer to some earlier request, still queued: not this refresh's.
        device.replies.append(stale_state(0, 50))
        device.temperature = 35
        bus._refresh()
        bus._param(9)
        assert bus.temperature_c == 35
        assert [m.data[0] for m in device.messages] == [7, 7]
    finally:
        bus.disconnect()
