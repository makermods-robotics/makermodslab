"""Hardware-free proofs for the live SO-101 adapter boundary."""

from __future__ import annotations

import json
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.robots.so_follower import SO101Follower
from makermodslab.remote_teleop.adapters.common import derive_so101_joint_schema
from makermodslab.remote_teleop.adapters.lerobot_follower import SO101FollowerDriver
from makermodslab.remote_teleop.adapters.lerobot_leader import SO101LeaderAdapter
from makermodslab.remote_teleop.calibration_identity import (
    IdentityError,
    calibration_identity,
    derive_rig_digest,
    verify_leader_allowlist,
)
from makermodslab.remote_teleop.contracts import SessionSpec
from makermodslab.remote_teleop.executor import JointLimit, RemoteExecutor

MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _calibration_data() -> dict[str, dict[str, int]]:
    return {
        name: {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for index, name in enumerate(MOTORS, 1)
    }


class _FakeBus:
    def __init__(self) -> None:
        self.motors = OrderedDict(
            (
                name,
                SimpleNamespace(
                    id=index,
                    model="sts3215",
                    norm_mode=SimpleNamespace(value="range_0_100" if name == "gripper" else "degrees"),
                ),
            )
            for index, name in enumerate(MOTORS, 1)
        )
        self.model_resolution_table = {"sts3215": 4096}
        self.is_connected = False
        self.torque = dict.fromkeys(MOTORS, 0)
        self.disable_errors: set[str] = set()
        self.read_errors: set[str] = set()
        self.disconnect_calls: list[bool] = []
        self.disable_calls: list[tuple[str | None, int]] = []
        self.disconnect_error = False
        self.connect_calls = 0
        self.enable_calls = 0
        self.configure_calls = 0
        self.write_calls: list[tuple[str, str, int]] = []
        self.write_error_register: str | None = None
        self.block_disable_entered: threading.Event | None = None
        self.release_disable: threading.Event | None = None

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disable_torque(self, motor=None, num_retry=0) -> None:
        self.disable_calls.append((motor, num_retry))
        if self.block_disable_entered is not None and self.release_disable is not None:
            self.block_disable_entered.set()
            self.release_disable.wait(timeout=2)
        targets = MOTORS if motor is None else (motor,)
        for name in targets:
            if name in self.disable_errors:
                raise OSError("injected disable failure")
            self.torque[name] = 0

    def enable_torque(self, motors=None, num_retry=0) -> None:
        self.enable_calls += 1
        targets = MOTORS if motors is None else tuple(motors)
        for name in targets:
            self.torque[name] = 1

    def configure_motors(self) -> None:
        self.configure_calls += 1

    def write(self, register, motor, value) -> None:
        if register == self.write_error_register:
            raise OSError("injected register write failure")
        self.write_calls.append((register, motor, value))

    def read(self, register, motor, *, normalize, num_retry):
        assert register == "Torque_Enable"
        assert normalize is False
        if motor in self.read_errors:
            raise OSError("injected read failure")
        return self.torque[motor]

    def disconnect(self, disable_torque=True) -> None:
        self.disconnect_calls.append(disable_torque)
        if self.disconnect_error:
            raise OSError("injected disconnect failure")
        self.is_connected = False


class _FakeFollower:
    def __init__(self, calibration_path: Path) -> None:
        self.id = calibration_path.stem
        self.calibration_fpath = calibration_path
        self.calibration = {name: SimpleNamespace(**fields) for name, fields in _calibration_data().items()}
        self.bus = _FakeBus()
        self.cameras = {}
        self.config = SimpleNamespace(
            position_p_coefficient=16,
            position_i_coefficient=0,
            position_d_coefficient=32,
        )
        self.is_calibrated = True
        self.connect_calls = 0
        self.sent: list[dict[str, float]] = []
        self.positions = OrderedDict((f"{name}.pos", 50.0 if name == "gripper" else 0.0) for name in MOTORS)

    @property
    def action_features(self):
        return OrderedDict((f"{name}.pos", float) for name in MOTORS)

    def connect(self, *, calibrate):
        self.connect_calls += 1
        raise AssertionError("remote follower must not call upstream connect()")

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        clean = dict(action)
        self.sent.append(clean)
        self.positions = OrderedDict(clean)
        return clean


class _FakeLeader(_FakeFollower):
    def connect(self, *, calibrate):
        assert calibrate is False
        self.connect_calls += 1
        self.bus.is_connected = True

    def get_action(self):
        return dict(self.positions)

    def disconnect(self):
        self.bus.disconnect()


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    path = tmp_path / "arm-a.json"
    path.write_text(json.dumps(_calibration_data()), encoding="utf-8")
    return path


def test_calibration_digest_is_canonical_and_duplicate_keys_fail(tmp_path: Path) -> None:
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    one.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    two.write_text('{\n  "a":1,"b":2\n}', encoding="utf-8")
    assert calibration_identity(one).digest == calibration_identity(two).digest

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"motor": 1, "motor": 2}', encoding="utf-8")
    with pytest.raises(IdentityError, match="duplicate JSON key") as exc:
        calibration_identity(duplicate)
    assert exc.value.code == "calibration_duplicate_key"


def test_unreadable_calibration_traceback_does_not_disclose_absolute_path(tmp_path: Path) -> None:
    private_path = tmp_path / "private-owner" / "missing-calibration.json"

    try:
        calibration_identity(private_path)
    except IdentityError as exc:
        rendered = "".join(traceback.format_exception(exc))
        assert exc.code == "calibration_unreadable"
        assert str(private_path) not in rendered
        assert exc.__cause__ is None
    else:  # pragma: no cover - the path intentionally does not exist
        raise AssertionError("missing calibration unexpectedly loaded")


def test_identity_allowlist_and_rig_digest_are_robot_authoritative(tmp_path: Path) -> None:
    leader_path = tmp_path / "leader.json"
    follower_path = tmp_path / "follower.json"
    leader_path.write_text('{"joint":{"id":1}}', encoding="utf-8")
    follower_path.write_text('{"joint":{"id":2}}', encoding="utf-8")
    leader = calibration_identity(leader_path)
    follower = calibration_identity(follower_path)
    verify_leader_allowlist(leader, {leader.calibration_id: leader.digest})
    with pytest.raises(IdentityError) as exc:
        verify_leader_allowlist(leader, {leader.calibration_id: "0" * 64})
    assert exc.value.code == "leader_calibration_digest_mismatch"

    limits = {"joint.pos": JointLimit(-1, 1, 2, 3)}
    digest = derive_rig_digest(
        arm_family="so101",
        topology="single",
        joint_schema=[{"action_key": "joint.pos", "unit": "degree"}],
        leader=leader,
        follower=follower,
        limits=limits,
    )
    assert len(digest) == 64
    changed = derive_rig_digest(
        arm_family="so101",
        topology="single",
        joint_schema=[{"action_key": "joint.pos", "unit": "degree"}],
        leader=leader,
        follower=follower,
        limits={"joint.pos": JointLimit(-2, 1, 2, 3)},
    )
    assert changed != digest


def test_follower_connect_is_disarmed_arm_preloads_observation_and_stop_is_verified(
    calibration_path: Path,
) -> None:
    fake = _FakeFollower(calibration_path)
    identity = calibration_identity(calibration_path)
    driver = SO101FollowerDriver(
        SimpleNamespace(),
        expected_calibration_id=identity.calibration_id,
        expected_calibration_digest=identity.digest,
        device_factory=lambda config: fake,
    )

    assert fake.connect_calls == 0
    driver.connect()
    assert driver.status["connect_transient_torque_risk"] is False
    assert fake.connect_calls == 0
    assert fake.bus.connect_calls == 1
    assert fake.bus.enable_calls == 0
    assert fake.bus.configure_calls == 1
    assert set(fake.bus.torque.values()) == {0}
    assert driver.last_stop_receipt is not None
    assert driver.last_stop_receipt.torque_off_confirmed is True
    assert driver.last_stop_receipt.hardware_stop_completed is True
    assert driver.last_stop_receipt.connect_transient_torque_risk is False

    # The free arm can move after negotiation; ARM must preload a fresh read,
    # not the connect-time sample.
    fake.positions["shoulder_pan.pos"] = 5.0
    target = OrderedDict((key, 10.0 if key != "gripper.pos" else 60.0) for key in driver.joint_names)
    current = driver.execute(target)
    assert current != target
    assert current["shoulder_pan.pos"] == 5.0
    assert fake.sent == [current]
    assert set(fake.bus.torque.values()) == {1}

    assert driver.execute(target) == target
    assert fake.sent[-1] == target
    receipt = driver.stop("operator_stop")
    assert receipt["torque_off_confirmed"] is True
    assert receipt["hardware_stop_completed"] is True
    assert receipt["connect_transient_torque_risk"] is False
    assert receipt["verification"] == "feetech_torque_enable_readback"
    close = driver.close()
    assert close["close_completed"] is True
    assert fake.bus.disconnect_calls == [False]


def test_follower_connect_applies_exact_pinned_register_contract(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    driver.connect()

    common = [("Operating_Mode", motor, 0) for motor in MOTORS]
    expected: list[tuple[str, str, int]] = []
    for item in common:
        expected.extend(
            (
                item,
                ("P_Coefficient", item[1], 16),
                ("I_Coefficient", item[1], 0),
                ("D_Coefficient", item[1], 32),
            )
        )
        if item[1] == "gripper":
            expected.extend(
                (
                    ("Max_Torque_Limit", "gripper", 500),
                    ("Protection_Current", "gripper", 250),
                    ("Overload_Torque", "gripper", 25),
                )
            )
    assert fake.bus.write_calls == expected
    assert fake.bus.enable_calls == 0
    driver.close()


def test_follower_contract_drift_fails_before_opening_bus(
    calibration_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeFollower(calibration_path)

    def changed_configure(self) -> None:
        self.bus.write("P_Coefficient", "gripper", 99)

    monkeypatch.setattr(SO101Follower, "configure", changed_configure)
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    with pytest.raises(RuntimeError, match="configuration contract changed"):
        driver.connect()
    assert fake.bus.connect_calls == 0
    assert fake.bus.enable_calls == 0


def test_follower_adapter_swap_after_claim_closes_without_torque_writes(
    calibration_path: Path,
) -> None:
    fake = _FakeFollower(calibration_path)
    driver = SO101FollowerDriver(
        SimpleNamespace(),
        expected_serial_binding="claimed-adapter",
        opened_binding_verifier=lambda _bus, _binding: False,
        device_factory=lambda config: fake,
    )

    with pytest.raises(RuntimeError, match="identity could not be confirmed"):
        driver.connect()

    assert fake.bus.connect_calls == 1
    assert fake.bus.write_calls == []
    assert fake.bus.enable_calls == 0
    assert fake.bus.torque == dict.fromkeys(MOTORS, 0)
    assert fake.bus.disconnect_calls == [False]


def test_binding_mismatch_worker_cleanup_and_disconnect_failure_never_write_motors(
    calibration_path: Path,
) -> None:
    fake = _FakeFollower(calibration_path)
    fake.bus.disconnect_error = True
    driver = SO101FollowerDriver(
        SimpleNamespace(),
        expected_serial_binding="claimed-adapter",
        opened_binding_verifier=lambda _bus, _binding: False,
        device_factory=lambda config: fake,
    )

    with pytest.raises(RuntimeError, match="identity could not be confirmed"):
        driver.connect()
    # This is the same cleanup pair invoked by the process worker's finally
    # block after a failed connect or lost channel.
    stop = driver.stop("worker_channel_closed")
    close = driver.close()

    assert stop["disable_requested"] is False
    assert stop["torque_off_confirmed"] is None
    assert stop["hardware_stop_completed"] is False
    assert stop["fault"] == "opened_adapter_identity_unconfirmed"
    assert close["close_completed"] is False
    assert fake.bus.disable_calls == []
    assert fake.bus.write_calls == []
    assert fake.bus.enable_calls == 0


def test_follower_stop_never_claims_completion_without_a_connected_bus(
    calibration_path: Path,
) -> None:
    disconnected = _FakeFollower(calibration_path)
    disconnected_driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: disconnected)
    disconnected_receipt = disconnected_driver.stop("not_connected")
    assert disconnected_receipt["verification"] == "bus_disconnected_no_readback"
    assert disconnected_receipt["hardware_stop_completed"] is False
    assert disconnected_receipt["torque_off_confirmed"] is None

    unavailable = _FakeFollower(calibration_path)
    unavailable_driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: unavailable)
    unavailable.bus = None
    unavailable_receipt = unavailable_driver.stop("bus_unavailable")
    assert unavailable_receipt["fault"] == "feetech_bus_unavailable"
    assert unavailable_receipt["hardware_stop_completed"] is False
    assert unavailable_receipt["torque_off_confirmed"] is None


def test_follower_configuration_failure_stops_and_closes_without_enabling_torque(
    calibration_path: Path,
) -> None:
    fake = _FakeFollower(calibration_path)
    fake.bus.write_error_register = "I_Coefficient"
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    with pytest.raises(OSError, match="injected register write failure"):
        driver.connect()
    assert fake.bus.enable_calls == 0
    assert set(fake.bus.torque.values()) == {0}
    assert fake.bus.disconnect_calls == [False]
    assert driver.last_stop_receipt is not None
    assert driver.last_stop_receipt.torque_off_confirmed is True
    assert driver.last_close_receipt is not None
    assert driver.last_close_receipt.close_completed is True


def test_follower_rejects_cameras_before_opening_bus(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    fake.cameras = {"wrist": SimpleNamespace(is_connected=False)}
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    with pytest.raises(RuntimeError, match="without cameras"):
        driver.connect()
    assert fake.bus.connect_calls == 0
    assert fake.bus.enable_calls == 0


def test_follower_servo_calibration_mismatch_stops_and_closes(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    fake.is_calibrated = False
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    with pytest.raises(RuntimeError, match="calibration does not match"):
        driver.connect()
    assert fake.bus.enable_calls == 0
    assert set(fake.bus.torque.values()) == {0}
    assert fake.bus.disconnect_calls == [False]


def test_follower_stop_reports_false_and_unknown_without_inference(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    driver.connect()

    fake.bus.disable_errors.add("wrist_roll")
    fake.bus.torque["wrist_roll"] = 1
    receipt = driver.stop("watchdog")
    assert receipt["torque_off_confirmed"] is False
    assert receipt["fault"] == "torque_still_enabled"

    fake.bus.disable_errors.clear()
    fake.bus.read_errors.add("gripper")
    receipt = driver.stop("retry")
    assert receipt["torque_off_confirmed"] is None
    assert str(receipt["fault"]).startswith("torque_state_unknown")
    driver.close()


def test_follower_rejects_wrong_order_boolean_and_out_of_range(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    schema = derive_so101_joint_schema(fake)
    values = OrderedDict(fake.positions)
    reversed_values = OrderedDict(reversed(tuple(values.items())))
    with pytest.raises(ValueError, match="exact ordered"):
        schema.validate_positions(reversed_values)
    values["shoulder_pan.pos"] = True
    with pytest.raises(ValueError, match="must be numeric"):
        schema.validate_positions(values)
    values["shoulder_pan.pos"] = 181.0
    with pytest.raises(ValueError, match="commissioned range"):
        schema.validate_positions(values)


def test_leader_returns_raw_timestamped_exact_schema(calibration_path: Path) -> None:
    fake = _FakeLeader(calibration_path)
    leader = SO101LeaderAdapter(
        SimpleNamespace(),
        clock_ns=lambda: 123456,
        device_factory=lambda config: fake,
    )
    leader.connect()
    sample = leader.read()
    assert tuple(sample.positions) == leader.joint_schema.action_keys
    assert sample.sampled_monotonic_ns == 123456
    leader.close()
    assert fake.bus.is_connected is False


def test_identity_mismatch_opens_no_device(calibration_path: Path) -> None:
    fake = _FakeFollower(calibration_path)
    with pytest.raises(IdentityError) as exc:
        SO101FollowerDriver(
            SimpleNamespace(),
            expected_calibration_id=calibration_path.stem,
            expected_calibration_digest="0" * 64,
            device_factory=lambda config: fake,
        )
    assert exc.value.code == "follower_calibration_digest_mismatch"
    assert fake.connect_calls == 0


def test_blocking_feetech_disable_has_no_in_process_stop_deadline(calibration_path: Path) -> None:
    """Evidence gate: a blocking SDK call requires a killable worker process."""
    fake = _FakeFollower(calibration_path)
    driver = SO101FollowerDriver(SimpleNamespace(), device_factory=lambda config: fake)
    limits = OrderedDict(
        (key, JointLimit(-200, 200, 100, 500))
        if key != "gripper.pos"
        else (key, JointLimit(0, 100, 100, 500))
        for key in driver.joint_names
    )
    executor = RemoteExecutor(driver, limits)
    digest = calibration_identity(calibration_path).digest
    executor.open_session(
        SessionSpec(
            source_id="operator",
            rig_id="rig",
            rig_digest="a" * 64,
            leader_calibration_id="leader",
            leader_calibration_digest="b" * 64,
            follower_calibration_id=calibration_path.stem,
            follower_calibration_digest=digest,
            joint_names=driver.joint_names,
            units=driver.joint_schema.units,
        )
    )
    fake.bus.block_disable_entered = threading.Event()
    fake.bus.release_disable = threading.Event()
    thread = threading.Thread(target=executor.stop, args=("network_loss",), daemon=True)
    thread.start()
    assert fake.bus.block_disable_entered.wait(timeout=0.2)
    # Software authority is revoked synchronously before vendor I/O, but the
    # caller and physical STOP receipt are still blocked in-process.
    status = executor.status()
    assert status["dispatch_enabled"] is False
    assert status["authority"]["state"] == "idle"
    thread.join(timeout=0.05)
    assert thread.is_alive(), "an in-process Feetech call unexpectedly became killable"
    assert executor.wait_until_halted(timeout=0.01) is False
    fake.bus.release_disable.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert executor.wait_until_halted(timeout=0.01) is True


def test_remote_hardware_imports_are_confined_to_adapter_modules() -> None:
    package = Path(__file__).parents[1] / "makermodslab" / "remote_teleop"
    offenders: list[str] = []
    for path in package.rglob("*.py"):
        if path.name in {"lerobot_follower.py", "lerobot_leader.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "SO101Follower(" in text or "SO101Leader(" in text:
            offenders.append(str(path.relative_to(package)))
    assert offenders == []


def test_single_side_config_builders_never_touch_the_opposite_side(monkeypatch) -> None:
    from makermodslab.utils import robot_factory

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        robot_factory,
        "setup_follower_calibration_file",
        lambda name, arm_type: calls.append(("follower", name)) or "follower-id",
    )
    monkeypatch.setattr(
        robot_factory,
        "setup_leader_calibration_file",
        lambda name, arm_type: calls.append(("leader", name)) or "leader-id",
    )
    monkeypatch.setattr(
        robot_factory,
        "SO101FollowerConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        robot_factory,
        "SO101LeaderConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    follower = robot_factory.build_follower_config(
        SimpleNamespace(follower_config="follower.json", follower_port="/dev/follower")
    )
    assert follower.id == "follower-id"
    assert calls == [("follower", "follower.json")]

    calls.clear()
    leader = robot_factory.build_leader_config(
        SimpleNamespace(leader_config="leader.json", leader_port="/dev/leader")
    )
    assert leader.id == "leader-id"
    assert calls == [("leader", "leader.json")]
