"""RS00 wire/lifecycle tests use the pinned real Maker follower and a fake CAN bus."""

import math
import struct
import time
from collections import deque
from types import SimpleNamespace

import can
import pytest

from lerobot.robots.maker_follower import MakerFollower, MakerFollowerConfig
from makermodslab import metal_gripper as grip
from makermodslab.maker_gripper_hold import CAN_TIMEOUT_INDEX, WATCHDOG_TICKS, MakerHoldingBus
from makermodslab.metal_gripper_hold import HoldingController


class Device:
    def __init__(self):
        self.messages = []
        self.queue = deque()
        self.enabled = set()
        self.registers = {0x7005: 0, CAN_TIMEOUT_INDEX: 0}
        self.positions = {
            i: sum(bounds) / 2 for i, bounds in enumerate(MakerFollowerConfig().joint_limits.values(), 1)
        }
        self.effort = 0.0
        self.temperature = 25.0
        self.flags = 0
        self.drop = False
        self.ignore_stop = False
        self.closed = False

    def state(self, motor):
        def encode(value, maximum, bits):
            return int((value + maximum) / (2 * maximum) * ((1 << bits) - 1))

        # The tests assert gripper units independently of the driver's encoder.
        q = encode(math.radians(self.positions[motor]), 12.57, 16)
        v = encode(0, 33, 12)
        torque = encode(self.effort if motor == 7 else 0, 14, 12)
        temp = int(self.temperature * 10)
        status = (0x80 if motor in self.enabled else 0) | (self.flags if motor == 7 else 0)
        return can.Message(
            arbitration_id=0xFD,
            is_extended_id=False,
            data=[
                motor,
                q >> 8,
                q & 255,
                v >> 4,
                (v & 15) << 4 | torque >> 8,
                torque & 255,
                status | temp >> 8,
                temp & 255,
            ],
        )

    def send(self, msg):
        self.messages.append(msg)
        mid, payload = msg.arbitration_id, bytes(msg.data)
        if mid & 0x700 in (0x300, 0x400):
            index = struct.unpack("<H", payload[:2])[0]
            if mid & 0x700 == 0x400:
                self.registers[index] = struct.unpack("<I", payload[4:])[0]
            response = can.Message(
                arbitration_id=mid,
                is_extended_id=False,
                data=payload[:4] + struct.pack("<I", self.registers[index]),
            )
        else:
            if payload == bytes([255] * 7 + [0xFC]):
                self.enabled.add(mid)
            elif payload == bytes([255] * 7 + [0xFD]) and not self.ignore_stop:
                self.enabled.discard(mid)
            response = self.state(mid)
        if not self.drop:
            self.queue.append(response)

    def recv(self, timeout=0):
        return self.queue.popleft() if self.queue else None

    def shutdown(self):
        self.closed = True


@pytest.fixture
def rig(monkeypatch):
    device = Device()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr("makermodslab.maker_gripper_hold.QUERY_TIMEOUT_S", 0.005)
    robot = MakerFollower(MakerFollowerConfig(port="/dev/fake"))
    bus = MakerHoldingBus(robot, "maker-hold", 0.5)
    robot.bus = bus
    yield robot, bus, device
    if bus._base.canbus is not None:
        device.drop = False
        device.flags = 0
        device.ignore_stop = False
        bus.disconnect()


def mit_commands(device):
    return [m for m in device.messages if m.arbitration_id == 7 and bytes(m.data[:6]) != bytes([255] * 6)]


def target(msg):
    return math.degrees(((msg.data[0] << 8 | msg.data[1]) / 65535 * 2 - 1) * 12.57)


@pytest.mark.parametrize("rate", [20, 50, 100])
@pytest.mark.parametrize("torque", [0.1, 0.5, 2.0])
def test_maker_closing_effort_converges_and_opening_releases(rate, torque):
    controller = HoldingController(torque, (-2.1, -0.04), closing_direction=1)
    position = command = -0.7
    for n in range(rate * 8):
        effort = 20 * (command - position)
        command = controller.update(-0.04, position, 0, effort, 20, n / rate)
        assert -2.1 <= command <= -0.04
        if n > rate * 2:
            assert controller.holding
            assert effort == pytest.approx(torque, abs=0.025)
    assert controller.update(-1.8, position, 0, torque, 20, 8) == -1.8
    assert not controller.holding


def test_opening_effort_and_moving_closure_do_not_engage():
    c = HoldingController(0.5, (-2.1, -0.04), closing_direction=1)
    for n in range(20):
        assert c.update(-0.04, -0.7, 0, -2, 20, n / 50) == -0.04
        assert not c.holding
    for n in range(20, 40):
        c.update(-0.04, -0.7, 1, 2, 20, n / 50)
        assert not c.holding


def test_connect_does_not_enable_and_checks_robstride_parameters(rig):
    _, bus, device = rig
    bus.connect()
    assert not device.enabled
    assert bus.temperature_c == pytest.approx(25)
    assert device.registers[CAN_TIMEOUT_INDEX] == 0
    assert {m.arbitration_id for m in device.messages} == {7, 0x307}
    bus.disconnect()
    assert device.closed


def test_real_follower_routes_goals_and_arms_restores_watchdog(rig):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    assert device.enabled == set(range(1, 8))
    assert bus._enabled and bus.temperature_c == 25
    assert device.registers[CAN_TIMEOUT_INDEX] == WATCHDOG_TICKS
    robot.config.startup_sync_speed_deg = None
    with bus._lock:
        device.messages.clear()
        robot.send_action({"gripper.pos": -35.0, "wrist_roll.pos": 0.0})
        command = mit_commands(device)[-1]
        assert target(command) == pytest.approx(-35, abs=0.03)
        assert ((command.data[3] & 15) << 8 | command.data[4]) / 4095 * 500 == pytest.approx(20, abs=0.13)
        assert any(m.arbitration_id == 6 for m in device.messages)
        assert all(
            bytes(m.data) != bytes([255] * 7 + [0xFB]) for m in device.messages if m.arbitration_id == 7
        )
    robot.disconnect()
    assert not device.enabled
    assert device.registers[CAN_TIMEOUT_INDEX] == 0
    assert not bus._thread.is_alive()
    assert grip.gripper_status("maker-hold") == []


@pytest.mark.parametrize("offset", [-360, 360])
def test_full_turn_offset_keeps_limits_and_commands_in_raw_coordinates(rig, offset):
    robot, bus, device = rig
    device.positions[7] = -60 - offset
    robot.connect(calibrate=False)
    assert robot._turn_offset["gripper"] == offset
    assert bus.joint_limits == pytest.approx(tuple(x - offset for x in robot.config.joint_limits["gripper"]))
    robot.config.startup_sync_speed_deg = None
    with bus._lock:
        robot.send_action({"gripper.pos": -30})
        assert target(mit_commands(device)[-1]) == pytest.approx(-30 - offset, abs=0.03)


def test_worker_regulates_during_pause_and_live_setting_applies(rig):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    before = len(mit_commands(device))
    deadline = time.monotonic() + 1
    while len(mit_commands(device)) <= before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(mit_commands(device)) > before
    assert grip.apply_live_hold_torque("maker-hold", 0.7) == "live"
    assert grip.gripper_status("maker-hold")[0]["hold_torque_nm"] == 0.7
    with pytest.raises(grip.GripperSafetyError):
        grip.apply_live_hold_torque("maker-hold", None)


@pytest.mark.parametrize("fault", ["heat", "warning", "fault", "disabled", "timeout"])
def test_runtime_fault_latches_stop(rig, fault):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    with bus._lock:
        if fault == "heat":
            device.temperature = 55
        elif fault in ("warning", "fault"):
            device.flags = 0x10 if fault == "warning" else 0x20
        elif fault == "disabled":
            device.enabled.discard(7)
        else:
            device.drop = True
        with pytest.raises((grip.GripperSafetyError, ConnectionError)):
            bus.write("Goal_Position", "gripper", -20)
        assert bus._error and not bus._enabled and bus._stop.is_set()
        device.drop = False
        device.flags = 0
        device.temperature = 25
        with pytest.raises(grip.GripperSafetyError):
            bus.enable_torque("gripper")


@pytest.mark.parametrize("gain,value", [("Kp", 25), ("Kd", 1), ("Goal_Velocity", 3)])
def test_gain_or_mode_changes_cannot_bypass_controller(rig, gain, value):
    robot, bus, _ = rig
    robot.connect(calibrate=False)
    with pytest.raises(grip.GripperSafetyError):
        bus.write(gain, "gripper", value)
    assert not bus._enabled


def test_unconfirmed_stop_leaves_watchdog_armed(rig):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    device.ignore_stop = True
    with pytest.raises(grip.GripperSafetyError, match="cleanup"):
        bus.disconnect()
    assert device.registers[CAN_TIMEOUT_INDEX] == WATCHDOG_TICKS
    assert device.closed


@pytest.mark.parametrize("temperature", [45, 55])
def test_warm_start_refused_without_enable(rig, temperature):
    _, bus, device = rig
    device.temperature = temperature
    with pytest.raises(grip.GripperSafetyError):
        bus.connect()
    assert not device.enabled and device.closed


def test_missing_parameter_support_fails_closed(rig):
    _, bus, device = rig
    original = device.send

    def send(msg):
        original(msg)
        if msg.arbitration_id == 0x307:
            device.queue.clear()

    device.send = send
    with pytest.raises(ConnectionError):
        bus.connect()
    assert not device.enabled and device.closed


def test_defaults_bimanual_and_explicit_opt_out(tmp_lerobot_home):
    from makermodslab.utils.config import save_robot_record

    save_robot_record("maker-two", {"arm_type": "maker"})
    arms = [MakerFollower(MakerFollowerConfig(port=f"/dev/fake{i}")) for i in range(2)]
    robot = SimpleNamespace(left_arm=arms[0], right_arm=arms[1])
    grip.install_metal_gripper(robot, "maker-two")
    assert all(isinstance(arm.bus, MakerHoldingBus) for arm in arms)
    save_robot_record("maker-off", {"arm_type": "maker", "gripper_hold_torque_nm": None})
    other = MakerFollower(MakerFollowerConfig(port="/dev/off"))
    original = other.bus
    grip.install_metal_gripper(other, "maker-off")
    assert other.bus is original
    unnamed = MakerFollower(MakerFollowerConfig(port="/dev/unnamed"))
    grip.install_metal_gripper(unnamed, "")
    assert isinstance(unnamed.bus, MakerHoldingBus)


def test_maker_api_allows_holding_but_refuses_metal_current_mode(client, tmp_lerobot_home):
    assert client.post("/robots/maker?create=true", json={"arm_type": "maker"}).status_code == 200
    response = client.post("/robots/maker", json={"gripper_hold_torque_nm": 0.7})
    assert response.status_code == 200
    assert response.json()["robot"]["gripper_hold_torque_nm"] == 0.7
    response = client.post(
        "/robots/maker", json={"gripper_hold_torque_nm": None, "gripper_current_limit_a": 0.5}
    )
    assert response.status_code == 400
    assert client.get("/robots/maker").json()["robot"]["gripper_hold_torque_nm"] == 0.7


def test_recording_installs_maker_adapter_before_connect(monkeypatch):
    from makermodslab.record import record_with_web_events

    robot = MakerFollower(MakerFollowerConfig(port="/dev/fake"))
    monkeypatch.setattr("lerobot.robots.make_robot_from_config", lambda cfg: robot)

    class PreparedError(Exception):
        pass

    def processors():
        assert isinstance(robot.bus, MakerHoldingBus)
        assert robot.bus.canbus is None
        raise PreparedError

    monkeypatch.setattr("lerobot.processor.make_default_processors", processors)
    with pytest.raises(PreparedError):
        record_with_web_events(SimpleNamespace(robot=robot.config, teleop=None), {})


def test_teleoperation_installs_maker_adapter_before_connect(monkeypatch):
    from makermodslab import teleoperate

    robot = MakerFollower(MakerFollowerConfig(port="/dev/fake"))
    monkeypatch.setattr(teleoperate, "build_single_configs", lambda req: (robot.config, object()))
    monkeypatch.setattr(teleoperate, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(teleoperate, "make_teleoperator_from_config", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(teleoperate, "de_energize_can_device", lambda *a: None)

    def connect(**kwargs):
        assert isinstance(robot.bus, MakerHoldingBus)
        assert robot.bus.canbus is None
        raise ConnectionError("prepared maker holding")

    monkeypatch.setattr(robot, "connect", connect)
    request = teleoperate.TeleoperateRequest(
        leader_port="/dev/star",
        follower_port="/dev/fake",
        leader_config="L",
        follower_config="F",
        arm_type="maker",
    )
    monkeypatch.setattr(teleoperate, "_cleanup_after_setup_failure", lambda *a: None)
    with pytest.raises(RuntimeError, match="Maker follower") as error:
        teleoperate._connect_can(request)
    assert str(error.value.__cause__) == "prepared maker holding"


def test_stale_queued_state_cannot_acknowledge_a_new_command(rig):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    with bus._lock:
        device.queue.append(device.state(7))
        device.drop = True
        with pytest.raises(ConnectionError):
            bus.write("Goal_Position", "gripper", -30)
        assert bus._error


def test_watchdog_readback_mismatch_prevents_enable(rig):
    robot, bus, device = rig
    original = device.send

    def send(msg):
        original(msg)
        if msg.arbitration_id == 0x407:
            device.registers[CAN_TIMEOUT_INDEX] = 0

    device.send = send
    with pytest.raises(grip.GripperSafetyError, match="write did not verify"):
        robot.connect(calibrate=False)
    assert not device.enabled and device.closed


def test_reenable_starts_at_current_position_without_old_hold_target(rig):
    robot, bus, device = rig
    robot.connect(calibrate=False)
    with bus._lock:
        bus.write("Goal_Position", "gripper", -20)
        bus.disable_torque("gripper")
    # Let the previous worker exit before starting another one.
    bus._thread.join(timeout=1)
    device.positions[7] = -90
    with bus._lock:
        bus.enable_torque("gripper")
        assert bus._goal == pytest.approx(-90, abs=0.03)
        assert bus.controller.last_command is None
        assert not bus.controller.holding
        bus._force(bus._goal)
        assert target(mit_commands(device)[-1]) == pytest.approx(-90, abs=0.05)
