"""Explicitly enabled robot-host runtime for single-arm SO-101 teleoperation."""

from __future__ import annotations

import hashlib
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from types import SimpleNamespace

from ..hardware_lease import (
    HardwareLeaseHeld,
    HardwareLeaseRegistry,
    HardwareLeaseToken,
    HardwareRecoveryIdentity,
    HardwareReleaseReceipt,
    hardware_lease_registry,
    safe_hardware_receipt,
)
from ..hardware_recovery_identity import so101_hardware_recovery_identity
from ..serial_port_identity import SerialBindingResolver, resolve_serial_port_bindings
from ..utils.config import get_robot_record, is_robot_record_clean
from ..utils.robot_factory import build_follower_config
from .adapters import SO101FollowerDriver, SO101FollowerProcessDriver
from .calibration_identity import CalibrationIdentity, canonical_json, derive_rig_digest
from .clock_sync import FrozenClockMapping
from .commissioning import (
    CommissioningRecord,
    CommissioningStore,
    profile_commissioning_digest,
)
from .config import RemoteRoleConfigStore, RobotRoleConfig
from .contracts import SessionSpec
from .control_server import (
    RobotControlCallbacks,
    RobotControlProtocol,
    RobotSessionProfile,
    SessionOpenResult,
    TlsControlServer,
)
from .executor import FollowerDriver, JointLimit, RemoteExecutor
from .fault_journal import (
    FaultJournalError,
    HardwareFaultRecord,
    RemoteFaultJournal,
)
from .pairing import PairingAuthority, RobotCredentialStore, certificate_sha256_fingerprint
from .recording import BoundedSessionRecorder
from .runtime import AsyncLoopThread
from .transport import UdpActionReceiver
from .watchdog import RobotLivenessWatchdog, WatchdogDeadlines, WatchdogRunner

STARTUP_STOP_TIMEOUT_S = 20.0


class _RobotStartupCancelledError(RuntimeError):
    """An accepted local STOP invalidated an in-flight session open."""


@dataclass
class _OpeningRobotSession:
    generation: int
    credential_id: str
    cancel: threading.Event
    done: threading.Event
    stop_reason: str | None = None
    lease: HardwareLeaseToken | None = None
    driver: FollowerDriver | None = None
    executor: RemoteExecutor | None = None
    recorder: BoundedSessionRecorder | None = None


@dataclass(frozen=True)
class PreparedRobotProfile:
    follower_config: object
    follower_calibration: CalibrationIdentity
    leader_calibration: CalibrationIdentity
    rig_id: str
    rig_digest: str
    joint_names: tuple[str, ...]
    units: tuple[str, ...]
    limits: Mapping[str, JointLimit]
    limits_digest: str
    device_identity_digest: str
    follower_port: str | None = None
    follower_serial_binding: str | None = None

    def for_credential(self, credential_id: str) -> RobotSessionProfile:
        return RobotSessionProfile(
            SessionSpec(
                source_id=credential_id,
                rig_id=self.rig_id,
                rig_digest=self.rig_digest,
                leader_calibration_id=self.leader_calibration.calibration_id,
                leader_calibration_digest=self.leader_calibration.digest,
                follower_calibration_id=self.follower_calibration.calibration_id,
                follower_calibration_digest=self.follower_calibration.digest,
                joint_names=self.joint_names,
                units=self.units,
            ),
            self.limits_digest,
        )


def _robot_request(config: RobotRoleConfig) -> SimpleNamespace:
    record = get_robot_record(config.robot_name)
    if record is None:
        raise ValueError(f"no saved robot named {config.robot_name!r}")
    if record.get("mode") != "single" or record.get("arm_type") != "so101":
        raise ValueError("remote teleoperation v1 requires a single-arm SO-101 robot")
    if not is_robot_record_clean(record, arms="follower"):
        raise ValueError("the selected follower needs a port and an existing calibration")
    return SimpleNamespace(
        arm_type="so101",
        follower_port=record["follower_port"],
        follower_config=f"{record['follower_config']}.json",
    )


def prepare_robot_profile(config: RobotRoleConfig) -> PreparedRobotProfile:
    """Inspect configured artifacts and the unconnected SO-101 contract."""
    follower_config = build_follower_config(_robot_request(config))
    adapter = SO101FollowerDriver(follower_config)
    follower = adapter.calibration_identity
    leader = CalibrationIdentity(config.leader_calibration_id, config.leader_calibration_digest)
    limits = {
        joint.action_key: JointLimit(
            joint.commissioned_minimum,
            joint.commissioned_maximum,
            config.max_velocity_per_s,
            config.max_acceleration_per_s2,
        )
        for joint in adapter.joint_schema.joints
    }
    rig_digest = derive_rig_digest(
        arm_family="so101",
        topology="single",
        joint_schema=adapter.joint_schema.public(),
        leader=leader,
        follower=follower,
        limits=limits,
    )
    limits_body = [
        {
            "joint": joint,
            "minimum": limit.minimum,
            "maximum": limit.maximum,
            "max_velocity_per_s": limit.max_velocity_per_s,
            "max_acceleration_per_s2": limit.max_acceleration_per_s2,
        }
        for joint, limit in limits.items()
    ]
    limits_digest = hashlib.sha256(canonical_json(limits_body)).hexdigest()
    rig_id = "so101-" + hashlib.sha256(config.robot_name.encode("utf-8")).hexdigest()[:16]
    return PreparedRobotProfile(
        follower_config=follower_config,
        follower_calibration=follower,
        leader_calibration=leader,
        rig_id=rig_id,
        rig_digest=rig_digest,
        joint_names=adapter.joint_schema.action_keys,
        units=adapter.joint_schema.units,
        limits=limits,
        limits_digest=limits_digest,
        device_identity_digest=adapter.device_identity.digest,
        follower_port=getattr(follower_config, "port", None),
    )


def _process_follower(
    follower_config: object,
    calibration_id: str,
    calibration_digest: str,
    serial_binding: str | None,
) -> FollowerDriver:
    return SO101FollowerProcessDriver(
        follower_config,
        expected_calibration_id=calibration_id,
        expected_calibration_digest=calibration_digest,
        expected_serial_binding=serial_binding,
    )


def _remote_recovery_identity(
    *,
    rig_digest: str,
    profile_digest: str,
    follower_port: str | None = None,
    binding_resolver: SerialBindingResolver = resolve_serial_port_bindings,
) -> HardwareRecoveryIdentity:
    """Bind remote recovery to its profile and physical follower adapter."""
    if isinstance(follower_port, str) and follower_port.strip():
        return so101_hardware_recovery_identity(
            (follower_port,),
            recovery_kind="remote_recovery",
            unbound_recovery_kind="so101_physical_recovery",
            profile_digest=profile_digest,
            binding_resolver=binding_resolver,
        )
    return HardwareRecoveryIdentity.from_targets(
        "so101_physical_recovery",
        "so101",
        rig_digest,
        profile_digest=profile_digest,
    )


def _profile_recovery_identity(
    profile: PreparedRobotProfile,
    *,
    binding_resolver: SerialBindingResolver = resolve_serial_port_bindings,
) -> HardwareRecoveryIdentity:
    if profile.follower_serial_binding is not None:
        bound_port = profile.follower_port or getattr(profile.follower_config, "port", None)

        def frozen_binding_resolver(ports):
            return {target: profile.follower_serial_binding for target in ports if target == bound_port}

        binding_resolver = frozen_binding_resolver
    return _remote_recovery_identity(
        rig_digest=profile.rig_digest,
        profile_digest=profile_commissioning_digest(profile),
        follower_port=profile.follower_port or getattr(profile.follower_config, "port", None),
        binding_resolver=binding_resolver,
    )


def _record_recovery_identity(record: HardwareFaultRecord) -> HardwareRecoveryIdentity:
    return _remote_recovery_identity(
        rig_digest=record.rig_digest,
        profile_digest=record.profile_digest,
    )


class RemoteRobotService:
    """Robot-local authority; construction performs no I/O and starts no threads."""

    def __init__(
        self,
        *,
        config_store: RemoteRoleConfigStore | None = None,
        registry: HardwareLeaseRegistry = hardware_lease_registry,
        profile_builder: Callable[[RobotRoleConfig], PreparedRobotProfile] = prepare_robot_profile,
        follower_factory: Callable[[object, str, str, str | None], FollowerDriver] = _process_follower,
        commissioning_store: CommissioningStore | None = None,
        fault_journal: RemoteFaultJournal | None = None,
        serial_binding_resolver: SerialBindingResolver = resolve_serial_port_bindings,
        allow_loopback: bool = False,
    ) -> None:
        self.config_store = config_store or RemoteRoleConfigStore()
        self.registry = registry
        self.profile_builder = profile_builder
        self.follower_factory = follower_factory
        self.serial_binding_resolver = serial_binding_resolver
        self.allow_loopback = allow_loopback
        self.commissioning = commissioning_store or CommissioningStore(self.config_store.root)
        self.fault_journal = fault_journal or RemoteFaultJournal(self.config_store.root)
        self.credentials = RobotCredentialStore(self.config_store.root)
        self.pairing = PairingAuthority(self.credentials)
        self._lock = threading.RLock()
        self._loop = AsyncLoopThread("remote-teleop-robot-control")
        self._config: RobotRoleConfig | None = None
        self._profile: PreparedRobotProfile | None = None
        self._control: TlsControlServer | None = None
        self._udp: UdpActionReceiver | None = None
        self._executor: RemoteExecutor | None = None
        self._lease: HardwareLeaseToken | None = None
        self._watchdog: RobotLivenessWatchdog | None = None
        self._watchdog_runner: WatchdogRunner | None = None
        self._executor_thread: threading.Thread | None = None
        self._recorder: BoundedSessionRecorder | None = None
        self._owner_credential_id: str | None = None
        self._faulted_executor: RemoteExecutor | None = None
        self._faulted_driver: FollowerDriver | None = None
        self._faulted_lease: HardwareLeaseToken | None = None
        self._lifecycle_generation = 0
        self._opening: _OpeningRobotSession | None = None
        self._stopping = False
        self._stop_done = threading.Event()
        self._stop_done.set()
        self._last_receipt: dict[str, object] | None = None
        self._fault: str | None = None
        self._durable_fault: HardwareFaultRecord | None = None
        self._fault_journal_error: str | None = None
        self._reconcile_durable_fault()

    def _recovery_identity(self, profile: PreparedRobotProfile) -> HardwareRecoveryIdentity:
        return _profile_recovery_identity(
            profile,
            binding_resolver=self.serial_binding_resolver,
        )

    def _bind_profile(self, profile: PreparedRobotProfile) -> PreparedRobotProfile:
        """Freeze one pre-open binding for both the lease and child worker."""
        port = profile.follower_port or getattr(profile.follower_config, "port", None)
        if not isinstance(port, str) or not port.strip():
            return replace(profile, follower_serial_binding=None)
        bindings = self.serial_binding_resolver((port,))
        return replace(profile, follower_serial_binding=bindings.get(port))

    @staticmethod
    def _require_stable_serial_binding(profile: PreparedRobotProfile) -> None:
        if not isinstance(profile.follower_serial_binding, str) or not profile.follower_serial_binding:
            raise RuntimeError("the remote follower requires one stable, unique USB serial binding")

    def _reconcile_durable_fault(self) -> HardwareLeaseToken | None:
        """Install the durable latch now or atomically behind an existing owner.

        A service can be constructed while another feature still holds the
        process-wide registry.  Queueing the latch inside that registry makes
        the current owner's safe release promote directly to ``unresolved``;
        there is no idle release-to-claim window for a third feature to win.
        """
        self._fault_journal_error = None
        try:
            record = self.fault_journal.load()
        except FaultJournalError:
            record = None
            self._fault_journal_error = "hardware fault journal is invalid"
        if self._fault_journal_error is not None:
            self._durable_fault = None
            self._fault = self._fault_journal_error
            return None
        if record is None:
            snapshot = self.registry.snapshot()
            recovery = snapshot.recovery or {}
            if snapshot.state == "unresolved" and recovery.get("recovery_kind") in {
                "remote_recovery",
                "so101_physical_recovery",
            }:
                self._fault = "prior remote process exited without terminal safe-close evidence"
            return None
        snapshot = self.registry.snapshot()
        snapshot_recovery = snapshot.recovery or {}
        if snapshot.state == "unresolved" and snapshot_recovery.get("recovery_kind") in {
            "remote_recovery",
            "so101_physical_recovery",
        }:
            # The central journal already carries the physical binding. The
            # secondary fault record adds diagnostics but must not replace it
            # with a profile-only identity after restart.
            self._durable_fault = record
            self._fault = record.reason_code
            return None
        recovery = _record_recovery_identity(record)
        try:
            try:
                configured = self._load_robot_config()
            except Exception:
                # Test/injected builders may be fully self-contained. The
                # production builder still rejects this placeholder, leaving
                # the record in nonautomatic physical recovery lockout.
                configured = SimpleNamespace()
            profile = self._bind_profile(self.profile_builder(configured))
        except Exception:
            pass
        else:
            if profile_commissioning_digest(profile) == record.profile_digest:
                recovery = self._recovery_identity(profile)
        lease = self.registry.install_unresolved_latch(
            kind=recovery.recovery_kind,
            owner="durable:robot-fault",
            reason="durable robot hardware fault requires local evidence-backed recovery",
            receipt=HardwareReleaseReceipt(
                safe=False,
                device_closed=record.device_closed if record is not None else False,
                torque_off=(record.torque_off_confirmed if record is not None else None),
                evidence="durable hardware fault journal",
            ),
            recovery=recovery,
        )
        self._faulted_lease = lease
        self._durable_fault = record
        self._fault = record.reason_code if record is not None else "hardware fault journal is invalid"
        return lease

    def assert_no_durable_fault(self) -> None:
        """Fail closed whenever a durable or invalid hardware journal exists."""
        self._reconcile_durable_fault()
        if self._fault_journal_error is not None:
            raise RuntimeError("repair the invalid hardware fault journal before continuing")
        snapshot = self.registry.snapshot()
        recovery = snapshot.recovery or {}
        if self._durable_fault is not None or (
            snapshot.state == "unresolved"
            and recovery.get("recovery_kind") in {"remote_recovery", "so101_physical_recovery"}
        ):
            raise RuntimeError("complete local secured-arm fault recovery before continuing")

    def _load_robot_config(self) -> RobotRoleConfig:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "robot" or not isinstance(loaded[1], RobotRoleConfig):
            raise ValueError("configure this host as Remote robot first")
        return loaded[1]

    @staticmethod
    def _driver_status(driver: FollowerDriver) -> Mapping[str, object]:
        status = getattr(driver, "child_status", None)
        if status is None:
            status = getattr(driver, "status", None)
        if callable(status):
            status = status()
        if not isinstance(status, Mapping):
            raise RuntimeError("follower supplied no commissioning status")
        return status

    def _verify_secured_probe(
        self,
        driver: FollowerDriver,
        profile: PreparedRobotProfile,
        observation: Mapping[str, float],
    ) -> None:
        if tuple(driver.joint_names) != profile.joint_names or tuple(observation) != profile.joint_names:
            raise RuntimeError("follower schema changed during secured-arm probe")
        status = self._driver_status(driver)
        device = status.get("device")
        initial_stop = status.get("stop_receipt")
        if not isinstance(device, Mapping) or device.get("digest") != profile.device_identity_digest:
            raise RuntimeError("connected follower device identity does not match the selected profile")
        if not isinstance(initial_stop, Mapping):
            raise RuntimeError("follower supplied no initial torque-off receipt")
        if (
            initial_stop.get("hardware_stop_completed") is not True
            or initial_stop.get("torque_off_confirmed") is not True
            or initial_stop.get("fault") not in {None, ""}
        ):
            raise RuntimeError("follower did not connect in a confirmed torque-off state")

    @staticmethod
    def _teardown_probe(driver: FollowerDriver, reason: str) -> dict[str, object]:
        faults: list[str] = []
        try:
            stop = driver.stop(reason)
        except Exception as exc:
            stop = None
            faults.append(f"stop:{type(exc).__name__}")
        try:
            close = driver.close()
        except Exception as exc:
            close = None
            faults.append(f"close:{type(exc).__name__}")
        stop_completed = isinstance(stop, Mapping) and stop.get("hardware_stop_completed") is True
        close_completed = isinstance(close, Mapping) and close.get("close_completed") is True
        torque = stop.get("torque_off_confirmed") if isinstance(stop, Mapping) else None
        if isinstance(stop, Mapping) and stop.get("fault") not in {None, ""}:
            faults.append("stop:reported_fault")
        if isinstance(close, Mapping) and close.get("fault") not in {None, ""}:
            faults.append("close:reported_fault")
        safe = stop_completed and close_completed and torque is True and not faults
        return {
            "stop_accepted": True,
            "software_dispatch_halted": True,
            "disable_requested": isinstance(stop, Mapping) and stop.get("disable_requested") is True,
            "hardware_stop_completed": stop_completed,
            "hardware_close_completed": close_completed,
            "torque_off_confirmed": torque if isinstance(torque, bool) else None,
            "fault_lockout": not safe,
            "faults": faults,
            "stop_receipt": dict(stop) if isinstance(stop, Mapping) else None,
            "close_receipt": dict(close) if isinstance(close, Mapping) else None,
        }

    @staticmethod
    def _fault_codes(safety: Mapping[str, object]) -> tuple[str, ...]:
        codes: list[str] = []
        if safety.get("hardware_stop_completed") is not True:
            codes.append("hardware_stop_unconfirmed")
        if safety.get("hardware_close_completed") is not True:
            codes.append("device_close_unconfirmed")
        torque = safety.get("torque_off_confirmed")
        if torque is False:
            codes.append("torque_still_enabled")
        elif torque is not True:
            codes.append("torque_state_unknown")
        if not codes:
            codes.append("adapter_fault_lockout")
        return tuple(codes)

    def _persist_fault(
        self,
        profile: PreparedRobotProfile,
        *,
        reason_code: str,
        safety: Mapping[str, object],
    ) -> HardwareFaultRecord | None:
        try:
            record = HardwareFaultRecord.from_profile(
                profile,
                reason_code=reason_code,
                fault_codes=self._fault_codes(safety),
                hardware_stop_completed=safety.get("hardware_stop_completed") is True,
                device_closed=safety.get("hardware_close_completed") is True,
                torque_off_confirmed=safety.get("torque_off_confirmed") is True,
            )
            self.fault_journal.save(record)
        except Exception:
            self._fault_journal_error = "hardware fault could not be written durably"
            return None
        self._durable_fault = record
        self._fault_journal_error = None
        return record

    @staticmethod
    def _require_physical_safeguards(attestations: Mapping[str, object]) -> None:
        required = {
            "arm_secured",
            "workspace_clear",
            "physical_power_cutoff_reachable",
            "acknowledge_live_torque_enable_risk",
        }
        if set(attestations) != required or any(attestations[name] is not True for name in required):
            raise ValueError("all secured-arm physical safeguards must be explicitly acknowledged")

    def commission(self, attestations: Mapping[str, object]) -> dict[str, object]:
        """Run the local, secured-arm, no-motion proof for the exact profile."""
        self.assert_no_durable_fault()
        self._require_physical_safeguards(attestations)
        with self._lock:
            if (
                self._control is not None
                or self._udp is not None
                or self._opening is not None
                or self._executor is not None
                or self._faulted_lease is not None
            ):
                raise RuntimeError("disable remote teleoperation and resolve hardware faults first")
        config = self._load_robot_config()
        profile = self._bind_profile(self.profile_builder(config))
        self._require_stable_serial_binding(profile)
        lease = self.registry.claim(
            "remote_commissioning",
            "local:secured-arm",
            recovery=self._recovery_identity(profile),
        )
        driver: FollowerDriver | None = None
        probe_error: str | None = None
        safety: dict[str, object] | None = None
        try:
            driver = self.follower_factory(
                profile.follower_config,
                profile.follower_calibration.calibration_id,
                profile.follower_calibration.digest,
                profile.follower_serial_binding,
            )
            driver.connect()
            observation = driver.observe()
            self._verify_secured_probe(driver, profile, observation)
        except Exception as exc:
            probe_error = type(exc).__name__
        finally:
            if driver is not None:
                safety = self._teardown_probe(driver, "secured_arm_commissioning")

        if driver is None:
            self.registry.release(
                lease,
                safe_hardware_receipt(
                    "commissioning ended before a follower driver was constructed",
                    torque_off=None,
                    torque_not_applicable=True,
                ),
            )
        else:
            assert safety is not None
            safe = (
                safety["hardware_stop_completed"] is True
                and safety["hardware_close_completed"] is True
                and safety["torque_off_confirmed"] is True
                and safety["fault_lockout"] is False
            )
            if not safe:
                self._persist_fault(
                    profile,
                    reason_code="commissioning_teardown_unconfirmed",
                    safety=safety,
                )
            self.registry.release(
                lease,
                HardwareReleaseReceipt(
                    safe=safe,
                    device_closed=safety["hardware_close_completed"] is True,
                    torque_off=(
                        safety["torque_off_confirmed"]
                        if isinstance(safety["torque_off_confirmed"], bool)
                        else None
                    ),
                    evidence=(
                        "secured-arm commissioning follower closed safely"
                        if safe
                        else "secured-arm commissioning follower safety was not confirmed"
                    ),
                ),
            )
            if not safe:
                with self._lock:
                    self._fault = "secured-arm commissioning ended in hardware fault lockout"
                    self._faulted_driver = driver
                    self._faulted_lease = lease
                    self._last_receipt = {
                        "reason": "secured_arm_commissioning",
                        "safety": safety,
                        "lease_released": False,
                        "state": "fault_lockout",
                    }
                raise RuntimeError("commissioning safety could not be confirmed; fault lockout retained")

        if probe_error is not None:
            raise RuntimeError(f"secured-arm commissioning probe failed ({probe_error or 'unknown'})")
        record = CommissioningRecord.from_profile(profile)
        self.commissioning.save(record)
        with self._lock:
            self._last_receipt = {
                "reason": "secured_arm_commissioning",
                "safety": safety,
                "lease_released": True,
                "state": "idle",
            }
        return {"commissioning": record.public(), "runtime": self.status()}

    def recover_fault(self, attestations: Mapping[str, object]) -> dict[str, object]:
        """Clear a durable latch only after repeating the full secured-arm proof."""
        self._require_physical_safeguards(attestations)
        with self._lock:
            if self._control is not None or self._udp is not None or self._opening is not None:
                raise RuntimeError("disable the remote robot listener before recovery")
        self._reconcile_durable_fault()
        if self._fault_journal_error is not None:
            raise RuntimeError("repair the invalid hardware fault journal before recovery")
        durable = self._durable_fault
        config = self._load_robot_config()
        profile = self._bind_profile(self.profile_builder(config))
        recovery = self._recovery_identity(profile)
        if recovery.recovery_kind != "remote_recovery":
            raise RuntimeError(
                "the follower lacks a stable USB serial binding; automatic recovery is refused"
            )
        if durable is not None and profile_commissioning_digest(profile) != durable.profile_digest:
            raise RuntimeError("configured profile does not match the durable hardware fault")
        snapshot = self.registry.snapshot()
        if snapshot.state != "unresolved" or snapshot.recovery != recovery.public():
            if durable is None and snapshot.state != "unresolved":
                raise RuntimeError("no durable robot hardware fault requires recovery")
            raise RuntimeError("durable fault recovery is waiting for the current hardware owner")
        lease = self.registry.begin_recovery(
            "local:secured-arm-recovery",
            expected=recovery,
        )
        driver: FollowerDriver | None = None
        probe_error: str | None = None
        safety: dict[str, object] | None = None
        try:
            self.commissioning.require(profile)
            driver = self.follower_factory(
                profile.follower_config,
                profile.follower_calibration.calibration_id,
                profile.follower_calibration.digest,
                profile.follower_serial_binding,
            )
            driver.connect()
            observation = driver.observe()
            self._verify_secured_probe(driver, profile, observation)
        except Exception as exc:
            probe_error = type(exc).__name__
        finally:
            if driver is not None:
                safety = self._teardown_probe(driver, "secured_arm_recovery")

        recovered = (
            probe_error is None
            and safety is not None
            and safety["hardware_stop_completed"] is True
            and safety["hardware_close_completed"] is True
            and safety["torque_off_confirmed"] is True
            and safety["fault_lockout"] is False
        )
        if recovered:
            assert safety is not None
            self.registry.release(
                lease,
                HardwareReleaseReceipt.safe_close(
                    torque_off=True,
                    evidence="secured-arm recovery confirmed follower torque off and device closed",
                ),
            )
            self.fault_journal.clear_after_recovery()
            with self._lock:
                self._fault = None
                self._faulted_driver = None
                self._faulted_lease = None
                self._durable_fault = None
                self._fault_journal_error = None
                self._last_receipt = {
                    "reason": "secured_arm_recovery",
                    "safety": safety,
                    "lease_released": True,
                    "state": "idle",
                }
            return {"recovered": True, "runtime": self.status()}

        if safety is not None and safety.get("fault_lockout") is True:
            self._persist_fault(
                profile,
                reason_code="recovery_teardown_unconfirmed",
                safety=safety,
            )
        self.registry.mark_unresolved(
            lease,
            "secured-arm recovery did not produce complete matching safe-close evidence",
        )
        with self._lock:
            self._fault = "secured-arm recovery did not complete"
            self._faulted_driver = driver
            self._faulted_lease = lease
        raise RuntimeError(f"secured-arm recovery failed ({probe_error or 'unsafe_teardown'})")

    def enable(self) -> dict[str, object]:
        self.assert_no_durable_fault()
        with self._lock:
            if self._control is not None or self._udp is not None:
                raise RuntimeError("remote robot listener is already enabled")
        config = self._load_robot_config()
        profile = self._bind_profile(self.profile_builder(config))
        self._require_stable_serial_binding(profile)

        # Preparing a profile constructs no serial connection, but it still
        # participates in the same registry so it cannot inspect a changing
        # calibration/device contract while another arm feature owns hardware.
        token = self.registry.claim(
            "remote_configuration",
            "local:robot-enable",
            recovery=self._recovery_identity(profile),
        )
        try:
            self.commissioning.require(profile)
        finally:
            self.registry.release(
                token,
                safe_hardware_receipt(
                    "profile built without opening a device",
                    torque_off=None,
                    torque_not_applicable=True,
                ),
            )

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(config.tls_certificate_path, config.tls_private_key_path)
        udp = UdpActionReceiver(
            config.bind_address,
            config.udp_port,
            self._on_datagram,
            allow_loopback=self.allow_loopback,
        )
        udp_port = udp.start()
        callbacks = RobotControlCallbacks(
            session_profile=self._session_profile,
            open_session=self._open_session,
            stop_session=self._stop_session,
            session_status=self._session_status,
            heartbeat=self._heartbeat,
            udp_probe_status=self._udp_probe_status,
            control_lost=self._control_lost,
        )
        control = TlsControlServer(
            config.bind_address,
            config.control_port,
            context,
            lambda: RobotControlProtocol(
                robot_id=config.node_id,
                credentials=self.credentials,
                pairing=self.pairing,
                callbacks=callbacks,
            ),
            allow_loopback=self.allow_loopback,
            heartbeat_deadline_s=config.control_deadline_ms / 1000,
        )
        try:
            control_port = self._loop.submit(control.start(), timeout=5.0)
        except Exception:
            udp.close()
            self._loop.close()
            raise
        with self._lock:
            self._config = replace(config, control_port=control_port, udp_port=udp_port)
            self._profile = profile
            self._udp = udp
            self._control = control
            self._fault = None
        return self.status()

    def _session_profile(self, credential_id: str) -> RobotSessionProfile:
        with self._lock:
            profile = self._profile
        if profile is None:
            raise RuntimeError("remote robot listener is disabled")
        return profile.for_credential(credential_id)

    def _open_session(
        self,
        spec: SessionSpec,
        clock: FrozenClockMapping,
        credential_id: str,
    ) -> SessionOpenResult:
        self.assert_no_durable_fault()
        with self._lock:
            config = self._config
            profile = self._profile
            udp = self._udp
            if config is None or profile is None or udp is None:
                raise RuntimeError("remote robot listener is disabled")
            self._require_stable_serial_binding(profile)
            if self._executor is not None or self._lease is not None or self._opening is not None:
                raise RuntimeError("a remote robot session is already active")
            self._lifecycle_generation += 1
            opening = _OpeningRobotSession(
                generation=self._lifecycle_generation,
                credential_id=credential_id,
                cancel=threading.Event(),
                done=threading.Event(),
            )
            self._opening = opening
        lease: HardwareLeaseToken | None = None
        recorder: BoundedSessionRecorder | None = None
        driver: FollowerDriver | None = None
        executor: RemoteExecutor | None = None
        try:
            self._require_opening(opening)
            lease = self.registry.claim(
                "remote_teleoperation",
                f"credential:{credential_id}",
                recovery=self._recovery_identity(profile),
            )
            with self._lock:
                opening.lease = lease
            self._require_opening(opening)
            driver = self.follower_factory(
                profile.follower_config,
                profile.follower_calibration.calibration_id,
                profile.follower_calibration.digest,
                profile.follower_serial_binding,
            )
            with self._lock:
                opening.driver = driver
            self._require_opening(opening)
            if tuple(driver.joint_names) != profile.joint_names:
                raise RuntimeError("follower driver schema changed after profile negotiation")
            if config.recording_enabled:
                recorder = BoundedSessionRecorder(
                    self.config_store.root / "recordings",
                    "pending-" + uuid.uuid4().hex,
                )
                recorder.start(
                    {
                        "rig_id": profile.rig_id,
                        "rig_digest": profile.rig_digest,
                        "clock": clock.public(),
                        "joint_names": list(profile.joint_names),
                        "units": list(profile.units),
                    }
                )
                with self._lock:
                    opening.recorder = recorder
                self._require_opening(opening)
            executor = RemoteExecutor(
                driver,
                profile.limits,
                tick_hz=config.action_rate_hz,
                watchdog_ns=config.action_watchdog_ms * 1_000_000,
                first_action_deadline_ns=config.first_action_deadline_ms * 1_000_000,
                mode="live",
                recorder=recorder,
            )
            with self._lock:
                opening.executor = executor
            self._require_opening(opening)
            grant = executor.open_session(
                spec,
                clock_offset_ns=clock.robot_minus_operator_ns,
                clock_uncertainty_ns=clock.uncertainty_ns,
            )
            self._require_opening(opening)
            udp.begin_session(
                session_id=grant.session_id,
                executor_generation=grant.executor_generation,
                key_id=grant.key_id,
                key=grant.action_key,
            )
            self._require_opening(opening)
            watchdog = RobotLivenessWatchdog(
                self._watchdog_stop,
                deadlines=WatchdogDeadlines(
                    action_ns=config.action_watchdog_ms * 1_000_000,
                    first_action_ns=config.first_action_deadline_ms * 1_000_000,
                    control_ns=config.control_deadline_ms * 1_000_000,
                    browser_ns=config.browser_deadline_ms * 1_000_000,
                ),
            )
            watchdog.arm()
            runner = WatchdogRunner(watchdog)
            runner.start()
            executor_thread = threading.Thread(
                target=self._run_executor,
                args=(executor,),
                name="remote-teleop-executor",
                daemon=True,
            )
            with self._lock:
                if self._opening is not opening or opening.cancel.is_set():
                    runner.close()
                    raise _RobotStartupCancelledError(opening.stop_reason or "robot session start cancelled")
                # The connection authenticated before entering this blocking
                # method, but revocation can race follower construction.  The
                # credential lock stays held through live-authority
                # publication, giving revoke and publish one strict order.
                with self.credentials.active_guard(credential_id) as credential_active:
                    if not credential_active:
                        opening.stop_reason = "operator_credential_revoked"
                        opening.cancel.set()
                        runner.close()
                        raise _RobotStartupCancelledError("operator credential revoked during startup")
                    self._lease = lease
                    self._executor = executor
                    self._owner_credential_id = credential_id
                    self._watchdog = watchdog
                    self._watchdog_runner = runner
                    self._executor_thread = executor_thread
                    self._recorder = recorder
                    self._stopping = False
                    self._opening = None
            executor_thread.start()
            opening.done.set()
            return SessionOpenResult(grant, config.bind_address, udp.bound_port or config.udp_port)
        except Exception as exc:
            safety: Mapping[str, object] | None = None
            if executor is not None:
                safety = executor.stop(f"session_open_failed:{type(exc).__name__}").get("safety")
            elif driver is not None:
                try:
                    stop_receipt = driver.stop("session_open_failed")
                except Exception:
                    stop_receipt = None
                try:
                    close_receipt = driver.close()
                except Exception:
                    close_receipt = None
                safety = {
                    "torque_off_confirmed": (
                        stop_receipt.get("torque_off_confirmed")
                        if isinstance(stop_receipt, Mapping)
                        else None
                    ),
                    "hardware_close_completed": (
                        close_receipt.get("close_completed") is True
                        if isinstance(close_receipt, Mapping)
                        else False
                    ),
                }
            if lease is None:
                lease_released = True
            elif driver is None:
                self.registry.release(
                    lease,
                    safe_hardware_receipt(
                        "session open ended before a follower driver was constructed",
                        torque_off=None,
                        torque_not_applicable=True,
                    ),
                )
                lease_released = not self.registry.is_token_current(lease)
            else:
                lease_released = self._release_lease(
                    lease,
                    safety,
                    f"session open failed: {type(exc).__name__}",
                )
            if recorder is not None:
                recorder.close({"event": "session.open_failed", "fault": type(exc).__name__})
            if opening.cancel.is_set() or not lease_released:
                stop_reason = (
                    opening.stop_reason
                    if opening.cancel.is_set()
                    else f"session_open_failed:{type(exc).__name__}"
                )
                receipt = {
                    "stop_accepted": True,
                    "reason": stop_reason or "robot_session_start_cancelled",
                    "safety": dict(safety or {}),
                    "lease_released": lease_released,
                    "state": "idle" if lease_released else "fault_lockout",
                }
                with self._lock:
                    self._last_receipt = receipt
                    if not lease_released:
                        self._fault = "safe follower startup teardown was not confirmed"
                        self._faulted_executor = executor
                        self._faulted_driver = driver
                        self._faulted_lease = lease
            raise
        finally:
            with self._lock:
                if self._opening is opening:
                    self._opening = None
            opening.done.set()

    def _require_opening(self, opening: _OpeningRobotSession) -> None:
        with self._lock:
            current = self._opening is opening
            cancelled = opening.cancel.is_set()
        if not current or cancelled:
            raise _RobotStartupCancelledError(opening.stop_reason or "robot session start cancelled")

    def _run_executor(self, executor: RemoteExecutor) -> None:
        period = 1.0 / executor.tick_hz
        try:
            while executor.authority.grant is not None:
                started = time.monotonic()
                executor.tick()
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        except Exception as exc:
            # An unexpected runner failure is itself loss of the action
            # process. Revoke and finalize through the same physical STOP path
            # instead of leaving authority live until another watchdog notices.
            with suppress(Exception):
                self._stop_current(f"executor_thread:{type(exc).__name__}")
        finally:
            if executor.authority.grant is None:
                executor.wait_until_halted(timeout=2.0)
                self._finalize_stopped_executor(executor, "executor_stopped")

    def _on_datagram(self, raw: bytes) -> None:
        with self._lock:
            executor = self._executor
            watchdog = self._watchdog
        if executor is None or watchdog is None:
            raise RuntimeError("no remote robot session is active")
        executor.submit_datagram(raw)
        watchdog.mark_action()

    def _validate_session(self, session_id: str, generation: int) -> RemoteExecutor:
        with self._lock:
            executor = self._executor
        grant = executor.authority.grant if executor is not None else None
        if grant is None or grant.session_id != session_id or grant.executor_generation != generation:
            raise RuntimeError("stale or foreign remote robot session")
        return executor

    def _heartbeat(
        self,
        session_id: str,
        generation: int,
        operator_process_live: bool,
        browser_live: bool,
    ) -> Mapping[str, object]:
        self._validate_session(session_id, generation)
        with self._lock:
            watchdog = self._watchdog
        if watchdog is None:
            raise RuntimeError("robot watchdog is unavailable")
        watchdog.mark_control(
            operator_process_live=operator_process_live,
            browser_live=browser_live,
        )
        status = watchdog.status()
        remaining_values = [
            float(value)
            for value in (
                status.get("action_remaining_ms"),
                status.get("control_remaining_ms"),
                status.get("browser_remaining_ms"),
            )
            if isinstance(value, (int, float))
        ]
        remaining = min(remaining_values, default=0.0)
        return {"watchdog_remaining_ms": remaining}

    def _udp_probe_status(self, session_id: str, generation: int) -> bool:
        self._validate_session(session_id, generation)
        with self._lock:
            udp = self._udp
        return udp is not None and udp.status()["endpoint_bound"] is True

    def _session_status(self, session_id: str, generation: int) -> Mapping[str, object]:
        executor = self._validate_session(session_id, generation)
        return self._active_status(executor)

    def _control_lost(self, session_id: str, generation: int, reason: str) -> None:
        try:
            self._validate_session(session_id, generation)
        except RuntimeError:
            return
        self._stop_current(reason)

    def _watchdog_stop(self, reason: str) -> None:
        self._stop_current(reason)

    def _stop_session(self, session_id: str, generation: int, reason: str) -> Mapping[str, object]:
        self._validate_session(session_id, generation)
        return self._stop_current(reason)

    def local_stop(self, reason: str = "robot_local_stop") -> dict[str, object]:
        return self._stop_current(reason)

    def _stop_current(self, reason: str) -> dict[str, object]:
        concurrent_stop: threading.Event | None = None
        with self._lock:
            opening = self._opening
            if opening is not None:
                opening.stop_reason = reason
                opening.cancel.set()
                if opening.lease is not None and self.registry.is_token_current(opening.lease):
                    self.registry.request_stop(opening.lease, reason)
            executor = self._executor
            lease = self._lease
            udp = self._udp
            runner = self._watchdog_runner
            watchdog = self._watchdog
            if executor is None:
                if opening is not None:
                    # The opener owns teardown. Waiting outside the lock lets it
                    # finish a bounded connect/STOP/close sequence exactly once.
                    pass
                else:
                    return dict(
                        self._last_receipt
                        or {
                            "stop_accepted": False,
                            "duplicate": True,
                            "state": "idle" if self._fault is None else "fault_lockout",
                        }
                    )
            if executor is not None and self._stopping:
                concurrent_stop = self._stop_done
            if executor is None:
                runner = None
            elif concurrent_stop is None:
                self._stopping = True
                self._stop_done.clear()
                if udp is not None:
                    udp.stop_dispatch()
                if lease is not None and self.registry.is_token_current(lease):
                    self.registry.request_stop(lease, reason)
                if watchdog is not None:
                    watchdog.disarm()
        if concurrent_stop is not None:
            if concurrent_stop.wait(STARTUP_STOP_TIMEOUT_S):
                with self._lock:
                    return dict(
                        self._last_receipt
                        or {
                            "stop_accepted": True,
                            "duplicate": True,
                            "state": "idle" if self._fault is None else "fault_lockout",
                        }
                    )
            return {
                "stop_accepted": True,
                "duplicate": True,
                "state": "stopping",
                "safety": executor.status()["safety"],
            }
        if opening is not None and executor is None:
            if not opening.done.wait(STARTUP_STOP_TIMEOUT_S):
                return {
                    "stop_accepted": True,
                    "state": "stopping",
                    "reason": reason,
                    "safety": {
                        "software_dispatch_halted": True,
                        "torque_off_confirmed": None,
                        "fault_lockout": True,
                    },
                }
            with self._lock:
                return dict(
                    self._last_receipt
                    or {
                        "stop_accepted": True,
                        "reason": reason,
                        "state": "idle" if self._fault is None else "fault_lockout",
                    }
                )
        assert executor is not None
        if runner is not None:
            runner.close()
        transition = executor.stop(reason)
        executor.wait_until_halted(timeout=2.0)
        return self._finalize_stopped_executor(executor, reason, transition)

    def _finalize_stopped_executor(
        self,
        executor: RemoteExecutor,
        reason: str,
        transition: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if executor is not self._executor:
                return dict(self._last_receipt or {"duplicate": True, "state": "idle"})
            lease = self._lease
            recorder = self._recorder
            safety = executor.status()["safety"]
            self._watchdog = None
            self._watchdog_runner = None
            self._executor_thread = None
            self._recorder = None
            self._owner_credential_id = None
        lease_released = True
        if lease is not None and self.registry.is_token_current(lease):
            lease_released = self._release_lease(lease, safety, reason)
        if recorder is not None:
            recorder.close({"event": "session.stopped", "reason": reason, "safety": safety})
        receipt = {
            **dict(transition or {}),
            "stop_accepted": True,
            "reason": reason,
            "safety": safety,
            "lease_released": lease_released,
            "state": "idle" if lease_released else "fault_lockout",
        }
        with self._lock:
            self._last_receipt = receipt
            self._stopping = False
            if lease_released:
                self._executor = None
                self._lease = None
                self._faulted_executor = None
                self._faulted_driver = None
                self._faulted_lease = None
            else:
                self._fault = "safe follower close was not confirmed"
                self._executor = None
                self._lease = None
                self._faulted_executor = executor
                self._faulted_driver = executor.follower
                self._faulted_lease = lease
            self._stop_done.set()
        return dict(receipt)

    def _release_lease(
        self,
        lease: HardwareLeaseToken,
        safety: Mapping[str, object] | None,
        evidence: str,
    ) -> bool:
        torque = safety.get("torque_off_confirmed") if safety is not None else None
        closed = safety.get("hardware_close_completed") is True if safety is not None else False
        safe = torque is True and closed and safety.get("fault_lockout") is not True
        if not safe:
            with self._lock:
                profile = self._profile
            if profile is not None:
                self._persist_fault(
                    profile,
                    reason_code="shutdown_unconfirmed",
                    safety=safety or {},
                )
        self.registry.release(
            lease,
            HardwareReleaseReceipt(
                safe=safe,
                device_closed=closed,
                torque_off=torque if isinstance(torque, bool) else None,
                evidence=evidence,
            ),
        )
        return not self.registry.is_token_current(lease)

    def commissioning_status(self) -> dict[str, object]:
        try:
            return self.commissioning.public()
        except Exception:
            return {
                "commissioned": False,
                "record": None,
                "error": "commissioning record is invalid",
            }

    def fault_status(self) -> dict[str, object]:
        try:
            public = self.fault_journal.public()
        except Exception:
            public = {"fault_lockout": True, "record": None}
        registry = self.registry.snapshot()
        recovery = registry.recovery or {}
        central_remote_fault = registry.state == "unresolved" and recovery.get("recovery_kind") in {
            "remote_recovery",
            "so101_physical_recovery",
        }
        public["fault_lockout"] = public.get("fault_lockout") is True or central_remote_fault
        public["central_intent_unresolved"] = central_remote_fault
        if self._fault_journal_error is not None:
            public["error"] = self._fault_journal_error
        return public

    def open_pairing_window(self) -> dict[str, object]:
        with self._lock:
            config = self._config
            control = self._control
        if config is None or control is None or control.bound_port is None:
            raise RuntimeError("enable the remote robot listener before pairing")
        fingerprint = certificate_sha256_fingerprint(config.tls_certificate_path)
        return self.pairing.open_local_window(
            local_request=True,
            robot_address=config.bind_address,
            control_port=control.bound_port,
            certificate_fingerprint=fingerprint,
        ).manual()

    def revoke_credential(self, credential_id: str) -> bool:
        revoked = self.credentials.revoke(credential_id)
        if not revoked:
            return False
        with self._lock:
            control = self._control
            opening = self._opening
            opening_owned = opening is not None and opening.credential_id == credential_id
            if opening_owned:
                opening.stop_reason = "operator_credential_revoked"
                opening.cancel.set()
                if opening.lease is not None and self.registry.is_token_current(opening.lease):
                    self.registry.request_stop(opening.lease, "operator_credential_revoked")
            active = self._owner_credential_id == credential_id
        if control is not None:
            with suppress(Exception):
                self._loop.submit(control.revoke_credential(credential_id), timeout=5.0)
        if active or opening_owned:
            self._stop_current("operator_credential_revoked")
        return True

    def _active_status(self, executor: RemoteExecutor) -> dict[str, object]:
        with self._lock:
            udp = self._udp
            watchdog = self._watchdog
            owner = self._owner_credential_id
        status = executor.status()
        return {
            "state": status["authority"]["state"],
            "owner_credential_id": owner,
            "executor": status,
            "udp": udp.status() if udp is not None else None,
            "watchdog": watchdog.status() if watchdog is not None else None,
        }

    def status(self) -> dict[str, object]:
        with self._lock:
            config = self._config
            control = self._control
            udp = self._udp
            executor = self._executor
            opening = self._opening
            faulted_executor = self._faulted_executor
            faulted_driver = self._faulted_driver
            faulted_lease = self._faulted_lease
            fault = self._fault
            last = dict(self._last_receipt) if self._last_receipt else None
        registry = self.registry.snapshot()
        if executor is not None:
            active = self._active_status(executor)
            state = active["state"]
        else:
            active = None
            state = (
                "starting"
                if opening is not None
                else "fault_lockout"
                if fault
                else ("idle" if control is not None else "disabled")
            )
        return {
            "protocol_version": "makermodslab.remote-robot-service.v1",
            "role": "robot",
            "runtime_enabled": control is not None and udp is not None,
            "live_hardware_enabled": (
                executor is not None
                or bool(opening and opening.driver is not None)
                or faulted_driver is not None
            ),
            "state": state,
            "listener": (
                {
                    "control_port": control.bound_port,
                    "udp_port": udp.bound_port if udp is not None else None,
                    "exact_bind_configured": config is not None,
                }
                if control is not None
                else None
            ),
            "active": active,
            "fault": fault,
            "last_stop": last,
            "commissioning": self.commissioning_status(),
            "durable_fault": self.fault_status(),
            "hardware_registry": {
                "held": registry.held,
                "state": registry.state,
                "kind": registry.kind,
                "owner": registry.owner,
                "generation": registry.generation,
                "pending_unresolved": registry.pending_unresolved,
                "pending_kind": registry.pending_kind,
                "pending_owner": registry.pending_owner,
            },
            "fault_resource": (
                {
                    "lease_retained": faulted_lease is not None,
                    "executor": faulted_executor.status() if faulted_executor is not None else None,
                    "driver_retained": faulted_driver is not None,
                }
                if faulted_lease is not None or faulted_driver is not None
                else None
            ),
            "credentials": self.credentials.public_entries(),
        }

    def disable(self) -> dict[str, object]:
        receipt = self._stop_current("robot_role_disabled")
        if receipt.get("state") == "stopping":
            raise RuntimeError("robot session startup teardown did not finish; listener remains enabled")
        with self._lock:
            control = self._control
            udp = self._udp
            self._control = None
            self._udp = None
            self._config = None
            self._profile = None
        self.pairing.close()
        if control is not None:
            self._loop.submit(control.close(), timeout=3.0)
        if udp is not None:
            udp.close()
        self._loop.close()
        return self.status()

    def shutdown(self) -> None:
        with suppress(RuntimeError, HardwareLeaseHeld):
            self.disable()
