from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from makermodslab.hardware_lease import HardwareLeaseRegistry
from makermodslab.remote_teleop.calibration_identity import CalibrationIdentity
from makermodslab.remote_teleop.clock_sync import FrozenClockMapping
from makermodslab.remote_teleop.commissioning import CommissioningRecord, CommissioningStore
from makermodslab.remote_teleop.config import RemoteRoleConfigStore
from makermodslab.remote_teleop.executor import JointLimit
from makermodslab.remote_teleop.fault_journal import HardwareFaultRecord, RemoteFaultJournal
from makermodslab.remote_teleop.robot_service import (
    PreparedRobotProfile,
    RemoteRobotService,
    _profile_recovery_identity,
)
from makermodslab.remote_teleop.simulation import SimulatedFollower

JOINTS = ("joint_a", "joint_b")
SAFEGUARDS = {
    "arm_secured": True,
    "workspace_clear": True,
    "physical_power_cutoff_reachable": True,
    "acknowledge_live_torque_enable_risk": True,
}


def serial_bindings(ports):
    return dict.fromkeys(ports, "test-physical-adapter")


class FakeUdp:
    bound_port = 7444

    def __init__(self) -> None:
        self.session = None
        self.dispatch_stopped = False

    def begin_session(self, **kwargs) -> None:
        self.session = kwargs

    def stop_dispatch(self) -> None:
        self.dispatch_stopped = True

    def status(self):
        return {"endpoint_bound": True, "counters": {}, "listening": True}

    def close(self) -> None:
        self.dispatch_stopped = True


class UnknownStopFollower(SimulatedFollower):
    def stop(self, reason: str):
        self.stop_reasons.append(reason)
        return {
            "disable_requested": True,
            "torque_off_confirmed": None,
            "verification": "unavailable",
            "fault_lockout": True,
        }

    def close(self):
        self.connected = False
        return {"close_completed": False, "fault": "unknown_close", "fault_lockout": True}


class CommissioningFollower(SimulatedFollower):
    @property
    def child_status(self):
        return {
            "device": {"digest": "e" * 64},
            "stop_receipt": {
                "hardware_stop_completed": True,
                "torque_off_confirmed": True,
                "fault": None,
            },
        }


class BlockingStopFollower(SimulatedFollower):
    def __init__(self, joint_names):
        super().__init__(joint_names)
        self.stop_started = threading.Event()
        self.release_stop = threading.Event()

    def stop(self, reason: str):
        self.stop_started.set()
        assert self.release_stop.wait(2.0)
        return super().stop(reason)


def profile() -> PreparedRobotProfile:
    limits = {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }
    return PreparedRobotProfile(
        follower_config=SimpleNamespace(port="/dev/test-follower"),
        follower_calibration=CalibrationIdentity("follower-cal", "c" * 64),
        leader_calibration=CalibrationIdentity("leader-cal", "b" * 64),
        rig_id="rig-1",
        rig_digest="a" * 64,
        joint_names=JOINTS,
        units=("rad", "rad"),
        limits=limits,
        limits_digest="d" * 64,
        device_identity_digest="e" * 64,
        follower_port="/dev/test-follower",
    )


def configured_service(follower_factory, tmp_path) -> tuple[RemoteRobotService, HardwareLeaseRegistry]:
    registry = HardwareLeaseRegistry()
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        follower_factory=follower_factory,
        serial_binding_resolver=serial_bindings,
    )
    service._config = SimpleNamespace(
        recording_enabled=False,
        action_rate_hz=50,
        action_watchdog_ms=200,
        first_action_deadline_ms=1000,
        control_deadline_ms=1000,
        browser_deadline_ms=2000,
        bind_address="127.0.0.1",
        udp_port=7444,
    )
    service._profile = replace(profile(), follower_serial_binding="test-physical-adapter")
    service._udp = FakeUdp()
    return service, registry


def test_remote_profile_recovery_is_bound_to_the_physical_follower() -> None:
    old = profile()
    moved = replace(
        old,
        follower_config=SimpleNamespace(port="/dev/moved-follower"),
        follower_port="/dev/moved-follower",
    )

    def same_adapter(ports):
        return dict.fromkeys(ports, "same-adapter")

    def replacement(ports):
        return dict.fromkeys(ports, "replacement-adapter")

    original = _profile_recovery_identity(old, binding_resolver=same_adapter)
    assert original == _profile_recovery_identity(moved, binding_resolver=same_adapter)
    assert original != _profile_recovery_identity(moved, binding_resolver=replacement)
    unbound = _profile_recovery_identity(old, binding_resolver=lambda _ports: {})
    assert unbound.recovery_kind == "so101_physical_recovery"


def open_session(service: RemoteRobotService):
    credential = service.credentials.issue("test operator", now_ns=1)
    requested = profile().for_credential(credential.credential_id).expected_spec
    return service._open_session(
        requested,
        FrozenClockMapping(0, 0, 0, 16),
        credential.credential_id,
    )


def test_browser_loss_stops_robot_and_releases_only_after_safe_close(tmp_path) -> None:
    follower = SimulatedFollower(JOINTS)
    service, registry = configured_service(lambda *_args: follower, tmp_path)
    result = open_session(service)
    assert registry.snapshot().kind == "remote_teleoperation"

    heartbeat = service._heartbeat(
        result.grant.session_id,
        result.grant.executor_generation,
        True,
        False,
    )
    assert heartbeat["watchdog_remaining_ms"] == 0
    assert follower.stop_reasons == ["operator_browser_lost"]
    assert registry.snapshot().state == "idle"
    assert service.status()["last_stop"]["safety"]["torque_off_confirmed"] is True


def test_unknown_torque_and_incomplete_close_retain_fault_lease(tmp_path) -> None:
    follower = UnknownStopFollower(JOINTS)
    service, registry = configured_service(lambda *_args: follower, tmp_path)
    result = open_session(service)

    receipt = service._stop_session(
        result.grant.session_id,
        result.grant.executor_generation,
        "network_loss",
    )
    assert receipt["lease_released"] is False
    assert receipt["state"] == "fault_lockout"
    assert receipt["safety"]["torque_off_confirmed"] is None
    assert receipt["safety"]["hardware_close_completed"] is False
    assert registry.snapshot().state == "unresolved"
    assert RemoteFaultJournal(tmp_path).load() is not None


def test_concurrent_acknowledged_stop_waits_for_final_hardware_receipt(tmp_path) -> None:
    follower = BlockingStopFollower(JOINTS)
    service, registry = configured_service(lambda *_args: follower, tmp_path)
    open_session(service)
    receipts: list[dict[str, object]] = []

    first = threading.Thread(target=lambda: receipts.append(service.local_stop("first_stop")))
    first.start()
    assert follower.stop_started.wait(1.0)
    second = threading.Thread(target=lambda: receipts.append(service.local_stop("second_stop")))
    second.start()
    time.sleep(0.05)
    assert second.is_alive()

    follower.release_stop.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert len(receipts) == 2
    assert all(receipt["state"] == "idle" for receipt in receipts)
    assert all(receipt["safety"]["torque_off_confirmed"] is True for receipt in receipts)
    assert registry.snapshot().state == "idle"


def test_disable_cancels_in_progress_follower_start_before_it_can_publish(tmp_path) -> None:
    follower = SimulatedFollower(JOINTS)
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(*_args):
        factory_started.set()
        assert release_factory.wait(2.0)
        return follower

    service, registry = configured_service(blocking_factory, tmp_path)
    open_errors: list[Exception] = []

    def run_open() -> None:
        try:
            open_session(service)
        except Exception as exc:  # cancellation is the expected result
            open_errors.append(exc)

    opener = threading.Thread(target=run_open)
    opener.start()
    assert factory_started.wait(1.0)

    disabled: list[dict[str, object]] = []
    disabler = threading.Thread(target=lambda: disabled.append(service.disable()))
    disabler.start()
    deadline = time.monotonic() + 1.0
    while service._opening is not None and not service._opening.cancel.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.005)
    release_factory.set()
    opener.join(timeout=2.0)
    disabler.join(timeout=2.0)

    assert not opener.is_alive() and not disabler.is_alive()
    assert open_errors
    assert disabled[0]["state"] == "disabled"
    assert service.status()["live_hardware_enabled"] is False
    assert service._executor is None
    assert registry.snapshot().state == "idle"
    assert follower.stop_reasons == ["session_open_failed"]


def test_live_enable_refuses_profile_without_commissioning(tmp_path) -> None:
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=HardwareLeaseRegistry(),
        profile_builder=lambda _config: profile(),
        serial_binding_resolver=serial_bindings,
    )
    service._load_robot_config = lambda: SimpleNamespace()

    try:
        service.enable()
    except RuntimeError as exc:
        assert "has not passed" in str(exc)
    else:  # pragma: no cover - proves no listener can be reached first
        raise AssertionError("uncommissioned live enable unexpectedly succeeded")
    assert service.status()["listener"] is None


def test_commissioning_is_profile_bound_and_safe_close_releases_registry(tmp_path) -> None:
    registry = HardwareLeaseRegistry()
    follower = CommissioningFollower(JOINTS)
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        profile_builder=lambda _config: profile(),
        follower_factory=lambda *_args: follower,
        serial_binding_resolver=serial_bindings,
    )
    service._load_robot_config = lambda: SimpleNamespace()

    result = service.commission(SAFEGUARDS)

    assert result["commissioning"]["profile_digest"]
    assert CommissioningStore(tmp_path).require(profile()).device_identity_digest == "e" * 64
    assert registry.snapshot().state == "idle"
    assert follower.stop_reasons == ["secured_arm_commissioning"]


def test_missing_stable_serial_binding_blocks_commissioning_before_device_open(tmp_path) -> None:
    registry = HardwareLeaseRegistry()
    opened: list[object] = []
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        profile_builder=lambda _config: profile(),
        follower_factory=lambda *_args: opened.append(object()),
        serial_binding_resolver=lambda _ports: {},
    )
    service._load_robot_config = lambda: SimpleNamespace()

    with pytest.raises(RuntimeError, match="stable, unique USB serial binding"):
        service.commission(SAFEGUARDS)

    assert opened == []
    assert registry.snapshot().state == "idle"
    assert CommissioningStore(tmp_path).load() is None


def test_missing_stable_serial_binding_blocks_listener_before_start(tmp_path) -> None:
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=HardwareLeaseRegistry(),
        profile_builder=lambda _config: profile(),
        serial_binding_resolver=lambda _ports: {},
    )
    service._load_robot_config = lambda: SimpleNamespace()
    CommissioningStore(tmp_path).save(CommissioningRecord.from_profile(profile()))

    with pytest.raises(RuntimeError, match="stable, unique USB serial binding"):
        service.enable()

    assert service.status()["listener"] is None


def test_unsafe_commissioning_persists_and_restart_restores_lockout(tmp_path) -> None:
    registry = HardwareLeaseRegistry()
    follower = UnknownStopFollower(JOINTS)
    follower.child_status = CommissioningFollower(JOINTS).child_status
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        profile_builder=lambda _config: profile(),
        follower_factory=lambda *_args: follower,
        serial_binding_resolver=serial_bindings,
    )
    service._load_robot_config = lambda: SimpleNamespace()

    try:
        service.commission(SAFEGUARDS)
    except RuntimeError as exc:
        assert "fault lockout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsafe commissioning unexpectedly succeeded")
    assert registry.snapshot().state == "unresolved"
    assert RemoteFaultJournal(tmp_path).load() is not None

    restarted_registry = HardwareLeaseRegistry()
    restarted = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=restarted_registry,
    )
    assert restarted_registry.snapshot().state == "unresolved"
    assert restarted.status()["durable_fault"]["fault_lockout"] is True


def test_matching_secured_recovery_clears_durable_fault(tmp_path) -> None:
    selected = profile()
    CommissioningStore(tmp_path).save(CommissioningRecord.from_profile(selected))
    RemoteFaultJournal(tmp_path).save(
        HardwareFaultRecord.from_profile(
            selected,
            reason_code="shutdown_unconfirmed",
            fault_codes=("torque_state_unknown",),
            hardware_stop_completed=True,
            device_closed=True,
            torque_off_confirmed=False,
        )
    )
    registry = HardwareLeaseRegistry()
    follower = CommissioningFollower(JOINTS)
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        profile_builder=lambda _config: selected,
        follower_factory=lambda *_args: follower,
        serial_binding_resolver=serial_bindings,
    )
    service._load_robot_config = lambda: SimpleNamespace()

    result = service.recover_fault(SAFEGUARDS)

    assert result["recovered"] is True
    assert registry.snapshot().state == "idle"
    assert RemoteFaultJournal(tmp_path).load() is None


def test_same_process_remote_fault_recovery_adopts_original_unresolved_lease(tmp_path) -> None:
    selected = profile()
    CommissioningStore(tmp_path).save(CommissioningRecord.from_profile(selected))
    unsafe = UnknownStopFollower(JOINTS)
    recovered = CommissioningFollower(JOINTS)
    followers = iter((unsafe, recovered))
    service, registry = configured_service(lambda *_args: next(followers), tmp_path)
    service.profile_builder = lambda _config: selected
    service._load_robot_config = lambda: SimpleNamespace()
    result = open_session(service)
    original_lease_id = registry.snapshot().lease_id

    service._stop_session(
        result.grant.session_id,
        result.grant.executor_generation,
        "network_loss",
    )
    assert registry.snapshot().state == "unresolved"
    assert registry.snapshot().pending_unresolved is False
    service._udp = None

    outcome = service.recover_fault(SAFEGUARDS)

    assert outcome["recovered"] is True
    assert original_lease_id is not None
    assert registry.snapshot().state == "idle"
    assert registry.snapshot().pending_unresolved is False


def test_crash_only_central_intent_recovers_without_secondary_fault_record(tmp_path) -> None:
    selected = profile()
    CommissioningStore(tmp_path).save(CommissioningRecord.from_profile(selected))
    journal = tmp_path / "central" / "hardware-lease.json"
    first_process = HardwareLeaseRegistry(journal_path=journal)
    first_process.claim(
        "remote_teleoperation",
        "credential:private",
        recovery=_profile_recovery_identity(selected, binding_resolver=serial_bindings),
    )

    restarted_registry = HardwareLeaseRegistry(journal_path=journal)
    follower = CommissioningFollower(JOINTS)
    restarted = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=restarted_registry,
        profile_builder=lambda _config: selected,
        follower_factory=lambda *_args: follower,
        serial_binding_resolver=serial_bindings,
    )
    restarted._load_robot_config = lambda: SimpleNamespace()

    assert RemoteFaultJournal(tmp_path).load() is None
    assert restarted.status()["durable_fault"]["fault_lockout"] is True
    assert restarted.status()["durable_fault"]["central_intent_unresolved"] is True

    outcome = restarted.recover_fault(SAFEGUARDS)

    assert outcome["recovered"] is True
    assert restarted_registry.snapshot().state == "idle"
    assert journal.exists() is False


def test_restart_queues_durable_fault_behind_current_owner_without_idle_gap(tmp_path) -> None:
    selected = profile()
    RemoteFaultJournal(tmp_path).save(
        HardwareFaultRecord.from_profile(
            selected,
            reason_code="shutdown_unconfirmed",
            fault_codes=("torque_state_unknown",),
            hardware_stop_completed=True,
            device_closed=True,
            torque_off_confirmed=False,
        )
    )
    registry = HardwareLeaseRegistry()
    current = registry.claim("recording", "existing-worker")

    restarted = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
    )

    assert registry.snapshot().pending_unresolved is True
    try:
        restarted.assert_no_durable_fault()
    except RuntimeError as exc:
        assert "fault recovery" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("durable fault was bypassed while another feature held hardware")

    registry.release(
        current,
        {
            "safe": True,
            "device_closed": True,
            "torque_off": True,
            "evidence": "existing worker stopped safely",
        },
    )
    snapshot = registry.snapshot()
    assert snapshot.state == "unresolved"
    assert snapshot.kind == "so101_physical_recovery"
    assert snapshot.owner == "durable:robot-fault"
    assert restarted.status()["hardware_registry"]["pending_unresolved"] is False
