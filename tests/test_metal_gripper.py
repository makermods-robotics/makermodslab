"""Exercise real Damiao packet routing with a simulated CAN device; no hardware."""

import math
import struct
import time
from collections import deque

import can
import pytest

from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
from makermodslab import metal_gripper as grip
from makermodslab.gripper_settings import validate_gripper_current


class Device:
    def __init__(self):
        self.registers = {9: 0, 10: 1, 59: 20.0}
        self.status = 0
        self.temperature = 30
        self.messages = []
        self.replies = deque()
        self.drop = False
        self.refuse_mode4 = False
        self.closed = False

    def state(self, motor=7):
        # Known Metal wire ranges; gripper at 20 degrees, other joints at 0.
        q = int(((math.radians(20) if motor == 7 else 0) + 6.28) / 12.56 * 65535)
        status = self.status if motor == 7 else 1
        data = bytes(
            [status << 4 | motor, q >> 8, q & 255, 0x7F, 0xF7, 0xFF, self.temperature, self.temperature]
        )
        return can.Message(arbitration_id=motor + 0x10, is_extended_id=False, data=data)

    def send(self, msg):
        self.messages.append(msg)
        d = bytes(msg.data)
        if msg.arbitration_id == 0x7FF:
            motor, op, rid = d[0], d[2], d[3]
            if op == 0xCC:
                reply = self.state(motor)
            else:
                assert op in (0x33, 0x55), "No save/zero/unknown writes allowed"
                if op == 0x55:
                    assert rid in (9, 10), "Only mode/watchdog writes allowed"
                    value = struct.unpack("<I", d[4:])[0]
                    if not (rid == 10 and value == 4 and self.refuse_mode4):
                        self.registers[rid] = value
                fmt = "f" if rid in (21, 22, 23, 59) else "I"
                reply = can.Message(
                    arbitration_id=0x17,
                    is_extended_id=False,
                    data=d[:4] + struct.pack("<" + fmt, self.registers[rid]),
                )
        else:
            motor = msg.arbitration_id & 0xFF
            if motor == 7:
                expected = 0x307 if self.registers[10] == 4 else 7
                assert msg.arbitration_id == expected, "Gripper routed into the wrong mode"
            if d[:7] == bytes([255] * 7):
                assert d[7] in (0xFC, 0xFD)
                if motor == 7:
                    self.status = 1 if d[7] == 0xFC else 0
            reply = self.state(motor)
        if not self.drop:
            self.replies.append(reply)

    def recv(self, timeout=None):
        if self.replies:
            return self.replies.popleft()
        time.sleep(min(timeout or 0, 0.001))
        return None

    def shutdown(self):
        self.closed = True


@pytest.fixture
def rig(monkeypatch):
    device = Device()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr(grip, "QUERY_TIMEOUT_S", 0.01)
    # Leave worker running but waiting unless a test explicitly exercises it.
    monkeypatch.setattr(grip, "HOLD_INTERVAL_S", 60)
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    bus = grip.MetalGripperBus(
        robot.bus,
        "test",
        0.5,
        (0.0, 137.5),
        speed_limit_deg_s=robot.config.velocity_ff_max_deg_s,
    )
    robot.bus = bus
    yield robot, bus, device
    if bus._base.canbus is not None:
        device.drop = False
        bus.disconnect()


@pytest.mark.parametrize("value", [True, False, "1", float("nan"), float("inf"), -1, 0, 0.09, 2.01])
def test_invalid_cap_never_becomes_unlimited(value):
    with pytest.raises(ValueError):
        validate_gripper_current(value)


def test_encoding_is_physical_units_and_never_rounds_above_cap():
    data = grip.encode_force_position(90, 0.35, 0.5, 20.522388458)
    pos, speed, current = struct.unpack("<fHH", data)
    assert pos == pytest.approx(math.pi / 2)
    assert speed == 35
    assert current == 243
    assert current / 10000 * 20.522388458 <= 0.5


def test_prepare_does_not_enable_and_restores_mode_and_watchdog(rig):
    _, bus, device = rig
    bus.connect()
    assert device.status == 0
    assert device.registers[10] == 4
    assert device.registers[9] == 10000
    assert all(bytes(m.data) != bytes([255] * 7 + [0xFC]) for m in device.messages)
    bus.disconnect()
    assert device.registers[10] == 1
    assert device.registers[9] == 0
    assert device.status == 0
    assert device.closed


@pytest.mark.parametrize("velocity_ff", [True, False])
def test_real_follower_send_action_routes_only_gripper_to_force_position(rig, velocity_ff):
    robot, bus, device = rig
    robot.config.velocity_feedforward = velocity_ff
    robot.config.startup_sync_speed_deg = None
    robot.connect(calibrate=False)
    device.messages.clear()
    robot.send_action({f"{name}.pos": 0.0 for name in bus.motors})
    control = [m for m in device.messages if m.arbitration_id != 0x7FF]
    assert {m.arbitration_id for m in control} == {1, 2, 3, 4, 5, 6, 0x307}
    gripper = next(m for m in control if m.arbitration_id == 0x307)
    _, speed, current = struct.unpack("<fHH", gripper.data)
    assert speed == 209  # 120 deg/s, floored to 0.01 rad/s wire units
    assert current == 250  # still 0.5 A; speed correction must not raise current


def test_live_current_tuning_does_not_change_tracking_speed(rig):
    _, bus, device = rig
    bus.connect()
    bus.enable_torque("gripper")
    for limit in (0.1, 0.5, 1.0):
        grip.apply_live_current("test", limit)
        _, speed, current = struct.unpack("<fHH", device.messages[-1].data)
        assert speed == 209
        assert current == int(limit / 20.0 * 10000)


def test_fixed_ceiling_does_not_derate_when_motor_warms(rig):
    _, bus, device = rig
    bus.connect()
    bus.enable_torque("gripper")
    for temperature in (30, 45, 50, 54):
        device.temperature = temperature
        bus._refresh()
        bus.sync_write("Goal_Position", {"gripper": 0.0})
        assert bus.effective_current_a == 0.5
        assert struct.unpack("<fHH", device.messages[-1].data)[2] == 250


def test_temperature_trips_and_cannot_be_cleared_by_another_action(rig):
    _, bus, device = rig
    bus.connect()
    bus.enable_torque("gripper")
    device.temperature = 55
    with pytest.raises(grip.GripperSafetyError, match="55 C"):
        bus.sync_write("Goal_Position", {"gripper": 0.0})
    assert device.status == 0
    device.temperature = 30
    with pytest.raises(grip.GripperSafetyError):
        bus.enable_torque("gripper")
    assert device.status == 0


def test_lost_feedback_trips_and_does_not_send_other_joint_commands(rig):
    _, bus, device = rig
    bus.connect()
    bus.enable_torque("gripper")
    device.drop = True
    device.messages.clear()
    with pytest.raises(ConnectionError):
        bus.sync_write_metal({"gripper": (20, 0.6, 0, 0, 0), "shoulder_pan": (20, 0.6, 0, 0, 0)})
    assert all(m.arbitration_id in (0x307, 0x7FF) for m in device.messages)
    assert bus._error


def test_unsupported_mode_aborts_before_enable(rig):
    _, bus, device = rig
    device.refuse_mode4 = True
    with pytest.raises(grip.GripperSafetyError):
        bus.connect()
    assert device.status == 0
    assert device.registers[10] == 1
    assert device.registers[9] == 0
    assert device.closed


def test_apply_changes_active_command_and_preserves_other_motors(rig):
    _, bus, device = rig
    bus.connect()
    bus.enable_torque("gripper")
    device.messages.clear()
    assert grip.apply_live_current("test", 0.8) == "live"
    assert bus.limit_a == 0.8
    assert all(m.arbitration_id in (0x307, 0x7FF) for m in device.messages)
    assert struct.unpack("<fHH", device.messages[-1].data)[2] == 400
    with pytest.raises(grip.GripperSafetyError, match="Stop"):
        grip.apply_live_current("test", None)
    assert bus.limit_a == 0.8


def test_worker_protects_hold_during_recording_pause(rig, monkeypatch):
    _, bus, device = rig
    monkeypatch.setattr(grip, "HOLD_INTERVAL_S", 0.005)
    bus.connect()
    bus.enable_torque("gripper")
    device.temperature = 56
    bus._thread.join(timeout=1)
    assert not bus._thread.is_alive()
    assert device.status == 0
    assert bus._error


def test_bimanual_installation_loads_saved_limit(tmp_lerobot_home):
    from lerobot.robots.bi_metal_follower import BiMetalFollower, BiMetalFollowerConfig
    from lerobot.robots.metal_follower import MetalFollowerConfigBase
    from makermodslab.utils.config import save_robot_record

    save_robot_record("two", {"arm_type": "metal", "gripper_current_limit_a": 0.7})
    robot = BiMetalFollower(
        BiMetalFollowerConfig(
            left_arm_config=MetalFollowerConfigBase(port="/dev/left", velocity_ff_max_deg_s=90),
            right_arm_config=MetalFollowerConfigBase(port="/dev/right", velocity_ff_max_deg_s=120),
        )
    )
    grip.install_metal_gripper(robot, "two")
    assert robot.left_arm.bus.limit_a == 0.7
    assert robot.right_arm.bus.limit_a == 0.7
    assert robot.left_arm.bus.speed_limit_rad_s == pytest.approx(math.radians(90))
    assert robot.right_arm.bus.speed_limit_rad_s == pytest.approx(math.radians(120))


def test_api_saves_valid_limit_and_refuses_invalid_and_other_families(client, tmp_lerobot_home):
    assert (
        client.post(
            "/robots/grip?create=true", json={"arm_type": "metal", "gripper_hold_torque_nm": None}
        ).status_code
        == 200
    )
    response = client.post("/robots/grip", json={"gripper_current_limit_a": 0.6})
    assert response.status_code == 200
    assert response.json()["gripper_application"] == "next_session"
    assert client.get("/robots/grip").json()["robot"]["gripper_current_limit_a"] == 0.6
    assert client.post("/robots/grip", json={"gripper_current_limit_a": 5}).status_code == 400
    assert client.post("/robots/grip", json={"gripper_current_limit_a": True}).status_code == 400
    assert client.get("/robots/grip").json()["robot"]["gripper_current_limit_a"] == 0.6
    client.post("/robots/so?create=true", json={"arm_type": "so101"})
    assert client.post("/robots/so", json={"gripper_current_limit_a": 0.5}).status_code == 400


def test_recording_installs_saved_limit_before_connect(monkeypatch, tmp_lerobot_home):
    from types import SimpleNamespace

    from makermodslab.record import record_with_web_events
    from makermodslab.utils.config import save_robot_record

    save_robot_record("record-grip", {"arm_type": "metal", "gripper_current_limit_a": 0.7})
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    monkeypatch.setattr("lerobot.robots.make_robot_from_config", lambda cfg: robot)

    class ReachedProcessorsError(Exception):
        pass

    def processors():
        assert isinstance(robot.bus, grip.MetalGripperBus)
        assert robot.bus.limit_a == 0.7
        assert robot.bus._base.canbus is None
        raise ReachedProcessorsError

    monkeypatch.setattr("lerobot.processor.make_default_processors", processors)
    cfg = SimpleNamespace(robot=robot.config, teleop=None, _makermodslab_robot_name="record-grip")
    with pytest.raises(ReachedProcessorsError):
        record_with_web_events(cfg, {})


def test_api_live_apply_and_save_failure_release(rig, client, tmp_lerobot_home, monkeypatch):
    from makermodslab.utils.config import save_robot_record

    _, bus, device = rig
    save_robot_record("test", {"arm_type": "metal", "gripper_current_limit_a": 0.5})
    bus.connect()
    bus.enable_torque("gripper")
    response = client.post("/robots/test", json={"gripper_current_limit_a": 0.7})
    assert response.status_code == 200
    assert response.json()["gripper_application"] == "live"
    assert bus.limit_a == 0.7
    assert client.get("/robots/test").json()["robot"]["gripper_current_limit_a"] == 0.7
    state = client.get("/api/v1/robots/test/gripper-status").json()["grippers"][0]
    assert state["enabled"] and state["effective_current_a"] == 0.7

    def fail_save(*args, **kwargs):
        raise OSError("simulated disk error")

    monkeypatch.setattr("makermodslab.server.save_robot_record", fail_save)
    response = client.post("/robots/test", json={"gripper_current_limit_a": 0.6})
    assert response.status_code == 500
    assert device.status == 0
    assert bus._error
    assert client.get("/robots/test").json()["robot"]["gripper_current_limit_a"] == 0.7


def test_delayed_parameter_ack_is_not_state_feedback(rig):
    _, bus, device = rig
    bus.connect()
    device.temperature = 40
    device.replies.append(
        can.Message(
            arbitration_id=0x17,
            is_extended_id=False,
            data=struct.pack("<HBBI", 7, 0x55, 9, 10000),
        )
    )
    bus._refresh()
    assert bus.temperature_c == 40


def test_failure_cleanup_reopen_never_configures_or_enables(rig):
    _, bus, device = rig
    bus.connect(handshake=False)
    assert device.registers == {9: 0, 10: 1, 59: 20.0}
    assert all(m.arbitration_id == 0x7FF and m.data[2] == 0x33 for m in device.messages)
    bus.disconnect()
    assert device.status == 0
    assert device.closed


def test_disconnect_attempts_other_joint_stops_if_gripper_unreachable(rig):
    _, bus, device = rig
    bus.connect()
    device.drop = True
    device.messages.clear()
    with pytest.raises(grip.GripperSafetyError, match="cleanup"):
        bus.disconnect()
    stops = {m.arbitration_id for m in device.messages if bytes(m.data) == bytes([255] * 7 + [0xFD])}
    assert {1, 2, 3, 4, 5, 6} <= stops
    assert device.closed
