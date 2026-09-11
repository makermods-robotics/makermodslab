"""No hardware: MIT wire tests and a stationary rigid-contact simulation."""

import math
import time

import can
import pytest

from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
from makermodslab import metal_gripper as grip
from makermodslab.gripper_settings import validate_gripper_hold_torque
from makermodslab.metal_gripper_hold import HoldingController, MetalHoldingBus
from tests.test_metal_gripper import Device


class HoldingDevice(Device):
    def __init__(self):
        super().__init__()
        self.registers.update({21: 6.28, 22: 30.0, 23: 20.0})
        self.effort = 0.0

    def state(self, motor=7):
        msg = super().state(motor)
        tau = int((self.effort + 20) / 40 * 4095) if motor == 7 else 2047
        msg.data[4] = (msg.data[4] & 0xF0) | (tau >> 8)
        msg.data[5] = tau & 255
        return msg


@pytest.fixture
def hold_rig(monkeypatch):
    device = HoldingDevice()
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: device)
    monkeypatch.setattr(grip, "QUERY_TIMEOUT_S", 0.01)
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    bus = MetalHoldingBus(robot.bus, "hold", 0.5, (0.0, 137.5), gains=(20.0, 0.6))
    robot.bus = bus
    yield robot, bus, device
    if bus._base.canbus is not None:
        device.drop = False
        bus.disconnect()


def decode(msg):
    d = msg.data
    return (
        ((d[3] & 15) << 8 | d[4]) / 4095 * 500,
        (d[5] << 4 | d[6] >> 4) / 4095 * 5,
        (d[0] << 8 | d[1]) / 65535 * 12.56 - 6.28,
        (d[2] << 4 | d[3] >> 4) / 4095 * 60 - 30,
        ((d[6] & 15) << 8 | d[7]) / 4095 * 40 - 20,
    )


def test_free_motion_is_unfiltered_and_opening_releases():
    c = HoldingController(0.5, (0, 2.4))
    for n in range(10):
        assert c.update(0.1, 0.7, -1, -2, 20, n * 0.02) == 0.1
        assert not c.holding
    for n in range(10, 20):
        c.update(0.1, 0.7, 0, -2, 20, n * 0.02)
    assert c.holding
    assert c.update(1.0, 0.7, 0, -2, 20, 0.4) == 1.0
    assert not c.holding


@pytest.mark.parametrize("hold_nm", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("rate_hz", [20, 50, 100])
def test_stationary_object_effort_converges_without_chasing_closed_leader(hold_nm, rate_hz):
    c = HoldingController(hold_nm, (0, 2.4))
    position, command, kp = 0.7, 0.7, 20
    for n in range(rate_hz * 8):
        effort = kp * (command - position)
        command = c.update(0.0, position, 0.0, effort, kp, n / rate_hz)
        assert 0 <= command <= 2.4
        if n > rate_hz * 2:
            assert c.holding
            assert abs(effort) <= hold_nm + 0.02
    assert c.holding
    assert kp * (command - position) == pytest.approx(-hold_nm, abs=0.02)
    assert command > 0.5  # Leader is fully closed; target stays near contact.


def test_single_effort_spike_and_opening_load_do_not_trigger_holding():
    c = HoldingController(0.5, (0, 2.4))
    c.update(0.0, 0.7, 0, -3, 20, 0)
    for n in range(1, 20):
        c.update(0.0, 0.7, 0, 0, 20, n * 0.02)
    assert not c.holding
    for n in range(20, 40):
        assert c.update(1.5, 0.7, 0, 3, 20, n * 0.02) == 1.5
    assert not c.holding


@pytest.mark.parametrize("value", [True, "0.5", float("nan"), float("inf"), 0, 2.01])
def test_invalid_holding_torque(value):
    with pytest.raises(ValueError):
        validate_gripper_hold_torque(value)


def test_prepare_stays_mit_and_does_not_enable(hold_rig):
    _, bus, device = hold_rig
    bus.connect()
    assert device.status == 0
    assert device.registers[10] == 1
    assert device.registers[9] == 10000
    assert all(m.arbitration_id != 0x307 for m in device.messages)
    assert all(bytes(m.data) != bytes([255] * 7 + [0xFC]) for m in device.messages)
    bus.disconnect()
    assert device.registers[9] == 0 and device.status == 0


@pytest.mark.parametrize("velocity_ff", [True, False])
def test_real_send_action_keeps_normal_gains_and_mit_targets(hold_rig, velocity_ff):
    robot, bus, device = hold_rig
    robot.config.velocity_feedforward = velocity_ff
    robot.config.startup_sync_speed_deg = None
    robot.connect(calibrate=False)
    with bus._lock:
        device.messages.clear()
        robot.send_action({f"{m}.pos": 40.0 if m == "gripper" else 0 for m in bus.motors})
        packets = [m for m in device.messages if m.arbitration_id == 7 and m.data[:7] != bytes([255] * 7)]
        kp, kd, pos, _, ff = decode(packets[-1])
        assert kp == pytest.approx(20, abs=0.13)
        assert kd == pytest.approx(0.6, abs=0.002)
        assert math.degrees(pos) == pytest.approx(40, abs=0.02)
        assert ff == pytest.approx(0, abs=0.01)
        assert {1, 2, 3, 4, 5, 6, 7} <= {m.arbitration_id for m in device.messages}
        assert not bus.controller.holding


def test_worker_holds_and_zeroes_velocity_feedforward_during_pause(hold_rig):
    _, bus, device = hold_rig
    bus.connect()
    bus.enable_torque("gripper")
    device.effort = -2
    bus.sync_write_metal({"gripper": (20, 0.6, 0, -100, -0.2)})
    deadline = time.monotonic() + 1
    while not bus.controller.holding and time.monotonic() < deadline:
        time.sleep(0.01)
    with bus._lock:
        assert bus.controller.holding
        bus._force(0)
        kp, kd, pos, vel, ff = decode(device.messages[-1])
        assert kp == pytest.approx(20, abs=0.13)
        assert kd == pytest.approx(0.6, abs=0.002)
        assert abs(vel) < 0.01 and abs(ff) < 0.01
        assert pos > 0


def test_faults_and_stale_feedback_disable_and_latch(hold_rig):
    _, bus, device = hold_rig
    bus.connect()
    bus.enable_torque("gripper")
    with bus._lock:
        device.drop = True
        with pytest.raises(ConnectionError):
            bus.sync_write("Goal_Position", {"gripper": 0})
        assert bus._error and not bus._enabled
        device.drop = False
        with pytest.raises(grip.GripperSafetyError):
            bus.enable_torque("gripper")


def test_mismatched_torque_scale_fails_before_enable(hold_rig):
    _, bus, device = hold_rig
    device.registers[23] = 10.0
    with pytest.raises(grip.GripperSafetyError, match="scaling"):
        bus.connect()
    assert device.status == 0 and device.closed


def test_api_switches_only_idle_and_tunes_holding_live(hold_rig, client, tmp_lerobot_home):
    from makermodslab.utils.config import save_robot_record

    _, bus, device = hold_rig
    save_robot_record("hold", {"arm_type": "metal", "gripper_current_limit_a": 0.5})
    assert client.post("/robots/hold", json={"gripper_hold_torque_nm": 0.5}).status_code == 400
    r = client.post("/robots/hold", json={"gripper_current_limit_a": None, "gripper_hold_torque_nm": 0.5})
    assert r.status_code == 200
    assert r.json()["robot"]["gripper_hold_torque_nm"] == 0.5
    bus.connect()
    bus.enable_torque("gripper")
    r = client.post("/robots/hold", json={"gripper_hold_torque_nm": 0.7})
    assert r.status_code == 200 and r.json()["gripper_application"] == "live"
    assert bus.controller.torque_nm == 0.7
    assert client.post("/robots/hold", json={"gripper_hold_torque_nm": None}).status_code == 409
    assert client.post("/robots/hold", json={"gripper_hold_torque_nm": True}).status_code == 400
    assert client.post("/robots/hold", json={"gripper_current_limit_a": 0.5}).status_code == 400
    assert device.status == 1
    # Clearing an already-null legacy current field must not partially apply
    # torque then report a failed current-mode switch.
    r = client.post("/robots/hold", json={"gripper_hold_torque_nm": 0.6, "gripper_current_limit_a": None})
    assert r.status_code == 200
    assert r.json()["robot"]["gripper_hold_torque_nm"] == bus.controller.torque_nm == 0.6
    state = client.get("/api/v1/robots/hold/gripper-status").json()["grippers"][0]
    assert state["hold_torque_nm"] == 0.6


def test_save_failure_after_live_hold_change_releases(hold_rig, client, tmp_lerobot_home, monkeypatch):
    from makermodslab.utils.config import save_robot_record

    _, bus, device = hold_rig
    save_robot_record("hold", {"arm_type": "metal", "gripper_hold_torque_nm": 0.5})
    bus.connect()
    bus.enable_torque("gripper")

    def fail_save(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("makermodslab.server.save_robot_record", fail_save)
    assert client.post("/robots/hold", json={"gripper_hold_torque_nm": 0.6}).status_code == 500
    assert device.status == 0 and bus._error


def test_temperature_stops_without_changing_torque_setting(hold_rig):
    _, bus, device = hold_rig
    bus.connect()
    bus.enable_torque("gripper")
    with bus._lock:
        device.temperature = 54
        bus.sync_write("Goal_Position", {"gripper": 0})
        assert bus.controller.torque_nm == 0.5
        device.temperature = 55
        with pytest.raises(grip.GripperSafetyError, match="55 C"):
            bus.sync_write("Goal_Position", {"gripper": 0})
        assert device.status == 0 and bus._error
        assert bus.controller.torque_nm == 0.5


def test_bimanual_holding_installs_on_both_followers(tmp_lerobot_home):
    from lerobot.robots.bi_metal_follower import BiMetalFollower, BiMetalFollowerConfig
    from lerobot.robots.metal_follower import MetalFollowerConfigBase
    from makermodslab.utils.config import save_robot_record

    save_robot_record("two", {"arm_type": "metal", "gripper_hold_torque_nm": 0.5})
    robot = BiMetalFollower(
        BiMetalFollowerConfig(
            left_arm_config=MetalFollowerConfigBase(port="/dev/left"),
            right_arm_config=MetalFollowerConfigBase(port="/dev/right"),
        )
    )
    grip.install_metal_gripper(robot, "two")
    for arm in (robot.left_arm, robot.right_arm):
        assert isinstance(arm.bus, MetalHoldingBus)
        assert arm.bus.controller.torque_nm == 0.5


def test_installed_in_recording_before_connect(monkeypatch, tmp_lerobot_home):
    from types import SimpleNamespace

    from makermodslab.record import record_with_web_events
    from makermodslab.utils.config import save_robot_record

    save_robot_record("hold", {"arm_type": "metal", "gripper_hold_torque_nm": 0.5})
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    monkeypatch.setattr("lerobot.robots.make_robot_from_config", lambda cfg: robot)

    class PreparedError(Exception):
        pass

    def processors():
        assert isinstance(robot.bus, MetalHoldingBus)
        assert robot.bus.controller.torque_nm == 0.5
        assert robot.bus.canbus is None
        raise PreparedError

    monkeypatch.setattr("lerobot.processor.make_default_processors", processors)
    with pytest.raises(PreparedError):
        record_with_web_events(
            SimpleNamespace(robot=robot.config, teleop=None, _makermodslab_robot_name="hold"), {}
        )


@pytest.mark.parametrize("family,expected", [("metal", 0.5), ("maker", None), ("so101", None)])
def test_new_robot_default_is_metal_only(client, tmp_lerobot_home, family, expected):
    response = client.post("/robots/default?create=true", json={"arm_type": family})
    assert response.status_code == 200
    assert response.json()["robot"]["gripper_hold_torque_nm"] == expected
    assert client.get("/robots/default").json()["robot"]["gripper_hold_torque_nm"] == expected


def test_missing_legacy_field_defaults_but_null_and_current_selection_are_preserved(tmp_lerobot_home):
    import json
    from pathlib import Path

    from makermodslab.utils import config

    config.save_robot_record("old", {"arm_type": "metal"})
    path = Path(config._robot_record_path("old"))
    record = json.loads(path.read_text())
    record.pop("gripper_hold_torque_nm")
    path.write_text(json.dumps(record))
    assert config.get_robot_record("old")["gripper_hold_torque_nm"] == 0.5
    record["gripper_current_limit_a"] = 0.7
    path.write_text(json.dumps(record))
    assert config.get_robot_record("old")["gripper_hold_torque_nm"] is None
    record["gripper_current_limit_a"] = None
    record["gripper_hold_torque_nm"] = None
    path.write_text(json.dumps(record))
    config.save_robot_record("old", {"motor_power": 50})
    assert config.get_robot_record("old")["gripper_hold_torque_nm"] is None


def test_unnamed_metal_session_uses_holding_default():
    robot = MetalFollower(MetalFollowerConfig(port="/dev/fake"))
    grip.install_metal_gripper(robot, "")
    assert isinstance(robot.bus, MetalHoldingBus)
    assert robot.bus.controller.torque_nm == 0.5
