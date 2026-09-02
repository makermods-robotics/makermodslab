"""Operator-host runtime: one SO-101 leader and one authenticated action stream."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace

from ..hardware_lease import (
    HardwareLeaseRegistry,
    HardwareLeaseToken,
    HardwareReleaseReceipt,
    hardware_lease_registry,
)
from ..hardware_recovery_identity import hardware_recovery_identity
from ..utils.config import get_robot_record
from ..utils.robot_factory import build_leader_config
from .adapters import SO101LeaderAdapter
from .config import OperatorRoleConfig, RemoteRoleConfigStore
from .contracts import ActionSample, SessionSpec, encode_action
from .control_client import OperatorSessionGrant, PinnedControlClient
from .pairing import IssuedCredential, OperatorCredentialVault
from .runtime import AsyncLoopThread
from .transport import UdpSessionSender

BROWSER_LIVENESS_S = 1.25
ACTION_LIFETIME_NS = 250_000_000
STARTUP_STOP_TIMEOUT_S = 20.0
LOCAL_IO_CLOSE_TIMEOUT_S = 2.0


class _OperatorStartupCancelledError(RuntimeError):
    """An accepted local STOP invalidated an in-flight operator start."""


@dataclass
class _StartingOperator:
    generation: int
    cancel: threading.Event
    done: threading.Event
    stop_reason: str | None = None


def _leader_request(config: OperatorRoleConfig) -> SimpleNamespace:
    record = get_robot_record(config.leader_robot_name)
    if record is None:
        raise ValueError(f"no saved robot named {config.leader_robot_name!r}")
    if record.get("mode") != "single" or record.get("arm_type") != "so101":
        raise ValueError("remote teleoperation v1 requires a single-arm SO-101 leader")
    if not record.get("leader_port") or not record.get("leader_config"):
        raise ValueError("the selected leader needs a port and calibration")
    return SimpleNamespace(
        arm_type="so101",
        leader_port=record["leader_port"],
        leader_config=f"{record['leader_config']}.json",
    )


def _leader_factory(config: OperatorRoleConfig, expected_id: str, expected_digest: str):
    return SO101LeaderAdapter(
        build_leader_config(_leader_request(config)),
        expected_calibration_id=expected_id,
        expected_calibration_digest=expected_digest,
    )


def _profile_spec(profile: Mapping[str, object]) -> SessionSpec:
    expected = {
        "source_id",
        "rig_id",
        "rig_digest",
        "follower_calibration_id",
        "follower_calibration_digest",
        "allowed_leader_calibration_id",
        "allowed_leader_calibration_digest",
        "joint_names",
        "units",
        "limits_digest",
    }
    if set(profile) != expected:
        raise ValueError("robot session profile has missing or extra fields")
    return SessionSpec(
        source_id=profile["source_id"],
        rig_id=profile["rig_id"],
        rig_digest=profile["rig_digest"],
        leader_calibration_id=profile["allowed_leader_calibration_id"],
        leader_calibration_digest=profile["allowed_leader_calibration_digest"],
        follower_calibration_id=profile["follower_calibration_id"],
        follower_calibration_digest=profile["follower_calibration_digest"],
        joint_names=tuple(profile["joint_names"]),
        units=tuple(profile["units"]),
    )


class RemoteOperatorService:
    """Explicit local lifecycle. Construction opens neither network nor hardware."""

    def __init__(
        self,
        *,
        config_store: RemoteRoleConfigStore | None = None,
        registry: HardwareLeaseRegistry = hardware_lease_registry,
        client_factory: Callable[[str, str], PinnedControlClient] = PinnedControlClient,
        leader_factory: Callable[[OperatorRoleConfig, str, str], object] = _leader_factory,
        sender_factory: Callable[[str, int], UdpSessionSender] = UdpSessionSender,
        leader_port_resolver: Callable[[OperatorRoleConfig], str] = lambda config: (
            _leader_request(config).leader_port
        ),
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config_store = config_store or RemoteRoleConfigStore()
        self.registry = registry
        self.client_factory = client_factory
        self.leader_factory = leader_factory
        self.sender_factory = sender_factory
        self.leader_port_resolver = leader_port_resolver
        self.clock_ns = clock_ns
        self.vault = OperatorCredentialVault(self.config_store.root)
        self._lock = threading.RLock()
        self._loop = AsyncLoopThread("remote-teleop-operator")
        self._config: OperatorRoleConfig | None = None
        self._client: PinnedControlClient | None = None
        self._leader: object | None = None
        self._sender: UdpSessionSender | None = None
        self._lease: HardwareLeaseToken | None = None
        self._grant: OperatorSessionGrant | None = None
        self._spec: SessionSpec | None = None
        self._leader_io_idle = threading.Event()
        self._leader_io_idle.set()
        self._faulted_leader: object | None = None
        self._faulted_lease: HardwareLeaseToken | None = None
        self._action_task: asyncio.Task[None] | None = None
        self._clock_task: asyncio.Task[None] | None = None
        self._browser_last_ns: int | None = None
        self._sequence = 0
        self._last_action_ns: int | None = None
        self._clock_uncertainty_ns: int | None = None
        self._credential_id: str | None = None
        self._lifecycle_generation = 0
        self._starting: _StartingOperator | None = None
        self._state = "idle"
        self._fault: str | None = None
        self._last_stop: dict[str, object] | None = None

    def _load_config(self) -> OperatorRoleConfig:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "operator" or not isinstance(loaded[1], OperatorRoleConfig):
            raise ValueError("configure this host as Remote operator first")
        return loaded[1]

    def pair(self, pairing_token: str, operator_label: str) -> dict[str, object]:
        config = self._load_config()

        async def run() -> dict[str, object]:
            client = self.client_factory(config.control_uri, config.certificate_fingerprint)
            try:
                await client.connect()
                hello = await client.hello(config.node_id)
                if hello.get("robot_id") != config.robot_id:
                    raise RuntimeError("paired robot identity does not match local configuration")
                if hello.get("pairing_allowed") is not True:
                    raise RuntimeError("robot pairing window is closed")
                issued = await client.pair(pairing_token, operator_label)
                self.vault.put(config.robot_id, issued)
                with self._lock:
                    self._credential_id = issued.credential_id
                return issued.public()
            finally:
                await client.close()

        return self._loop.submit(run(), timeout=10.0)

    def start(self) -> dict[str, object]:
        config = self._load_config()
        credential = self.vault.get(config.robot_id)
        if credential is None:
            with self._lock:
                self._state = "fault"
                self._fault = "paired operator credential is unavailable"
            raise RuntimeError("pair this operator with the robot before connecting")
        with self._lock:
            if self._state not in {"idle", "fault"} or self._lease is not None or self._starting is not None:
                raise RuntimeError("remote operator is already active")
            self._lifecycle_generation += 1
            starting = _StartingOperator(
                generation=self._lifecycle_generation,
                cancel=threading.Event(),
                done=threading.Event(),
            )
            self._starting = starting
            self._state = "connecting"
            self._fault = None
            self._browser_last_ns = self.clock_ns()
        try:
            return self._loop.submit(self._start_async(config, credential, starting), timeout=30.0)
        except Exception as exc:
            with self._lock:
                if self._starting is starting:
                    self._state = "fault"
                    # Adapter/network exception text may contain a device path,
                    # address, or credential-bearing URL. Status exposes only
                    # the stable failure class.
                    self._fault = f"operator_start:{type(exc).__name__}"
            raise

    async def _start_async(
        self,
        config: OperatorRoleConfig,
        credential: IssuedCredential,
        starting: _StartingOperator | None = None,
    ) -> dict[str, object]:
        if starting is None:
            # Focused integration tests and embedders may invoke the coroutine
            # directly. They still receive the same cancellable lifecycle.
            with self._lock:
                if self._starting is not None:
                    raise RuntimeError("remote operator is already starting")
                self._lifecycle_generation += 1
                starting = _StartingOperator(
                    generation=self._lifecycle_generation,
                    cancel=threading.Event(),
                    done=threading.Event(),
                )
                self._starting = starting
                self._state = "connecting"
        client = self.client_factory(config.control_uri, config.certificate_fingerprint)
        with self._lock:
            self._require_starting_locked(starting)
            self._config = config
            self._client = client
        lease: HardwareLeaseToken | None = None
        leader: object | None = None
        sender: UdpSessionSender | None = None
        grant: OperatorSessionGrant | None = None
        spec: SessionSpec | None = None
        try:
            await client.connect()
            self._require_starting(starting)
            hello = await client.hello(config.node_id, credential)
            self._require_starting(starting)
            if hello.get("robot_id") != config.robot_id or hello.get("authenticated") is not True:
                raise RuntimeError("robot rejected the paired operator credential")
            mapping, samples = await client.synchronize_clocks()
            self._require_starting(starting)
            profile = await client.profile()
            spec = _profile_spec(profile)
            self._require_starting(starting)

            # Claim before constructing or opening the leader. This host's local
            # recording/calibration/inference flows use the same registry.
            lease = await asyncio.to_thread(
                self.registry.claim,
                "remote_operator",
                f"robot:{config.robot_id}",
                recovery=hardware_recovery_identity(
                    "so101",
                    target_ports=(),
                    feetech_ports=(self.leader_port_resolver(config),),
                ),
            )
            with self._lock:
                self._lease = lease
            self._require_starting(starting)
            leader = self.leader_factory(
                config,
                spec.leader_calibration_id,
                spec.leader_calibration_digest,
            )
            with self._lock:
                self._leader = leader
            self._require_starting(starting)
            self._leader_io_idle.clear()
            await asyncio.to_thread(self._run_leader_io, leader.connect)
            self._require_starting(starting)
            schema = leader.joint_schema
            if tuple(schema.action_keys) != spec.joint_names or tuple(schema.units) != spec.units:
                raise RuntimeError("local leader joint schema does not match the robot profile")
            grant = await client.open_session(spec, samples)
            with self._lock:
                self._grant = grant
                self._spec = spec
            self._require_starting(starting)
            sender = self.sender_factory(grant.udp_host, grant.udp_port)
            with self._lock:
                self._sender = sender
            self._require_starting(starting)
            await asyncio.to_thread(
                sender.prove_endpoint,
                session_id=grant.session_id,
                executor_generation=grant.executor_generation,
                key_id=grant.key_id,
                key=grant.action_key(),
            )
            self._require_starting(starting)
            if await client.udp_probe_status(grant.key_id) is not True:
                raise RuntimeError("robot did not bind the authenticated UDP endpoint")
            self._require_starting(starting)

            with self._lock:
                self._require_starting_locked(starting)
                self._sequence = 0
                self._last_action_ns = None
                self._clock_uncertainty_ns = mapping.uncertainty_ns
                # Negotiation may legitimately take longer than the browser
                # liveness budget. Start it only once streaming is ready.
                self._browser_last_ns = self.clock_ns()
                self._credential_id = credential.credential_id
                self._state = "streaming"
                self._fault = None

            client.start_heartbeats(
                browser_live=self._browser_live,
                on_failure=self._heartbeat_failed,
            )
            self._action_task = asyncio.create_task(
                self._action_loop(), name="remote-teleop-operator-actions"
            )
            self._clock_task = asyncio.create_task(self._clock_check_loop(), name="remote-teleop-clock-check")
            with self._lock:
                self._require_starting_locked(starting)
                self._starting = None
            return self.status()
        except _OperatorStartupCancelledError:
            await self._abort_start(
                starting, starting.stop_reason or "operator_start_cancelled", failed=False
            )
            return self.status()
        except asyncio.CancelledError:
            # AsyncLoopThread timeouts/shutdowns revoke startup authority too.
            # Consume this cancellation long enough to publish the local
            # close/lease receipt, then propagate it to the caller.
            starting.stop_reason = starting.stop_reason or "operator_start_cancelled"
            starting.cancel.set()
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            await self._abort_start(starting, starting.stop_reason, failed=True)
            raise
        except Exception as exc:
            await self._abort_start(
                starting,
                f"operator_start_failed:{type(exc).__name__}",
                failed=True,
            )
            raise
        finally:
            starting.done.set()

    def _require_starting_locked(self, starting: _StartingOperator) -> None:
        if self._starting is not starting or starting.cancel.is_set():
            raise _OperatorStartupCancelledError(starting.stop_reason or "operator start cancelled")

    def _require_starting(self, starting: _StartingOperator) -> None:
        with self._lock:
            self._require_starting_locked(starting)

    async def _abort_start(
        self,
        starting: _StartingOperator,
        reason: str,
        *,
        failed: bool,
    ) -> None:
        with self._lock:
            client = self._client
            sender = self._sender
            leader = self._leader
            lease = self._lease
            grant = self._grant

        remote_receipt: dict[str, object] | None = None
        confirmation_available = False
        cleanup_faults: list[str] = []
        if lease is not None and self.registry.is_token_current(lease):
            self.registry.request_stop(lease, reason)
        if grant is not None and client is not None:
            try:
                remote_receipt = await client.stop(reason)
                confirmation_available = True
            except Exception as exc:
                cleanup_faults.append(f"robot_confirmation:{type(exc).__name__}")
        if sender is not None:
            try:
                sender.close()
            except Exception as exc:
                cleanup_faults.append(f"sender_close:{type(exc).__name__}")
        if client is not None:
            try:
                await asyncio.wait_for(client.close(), timeout=LOCAL_IO_CLOSE_TIMEOUT_S)
            except Exception as exc:
                cleanup_faults.append(f"control_close:{type(exc).__name__}")
        leader_closed = await asyncio.to_thread(
            self._leader_io_idle.wait,
            LOCAL_IO_CLOSE_TIMEOUT_S,
        )
        if not leader_closed:
            cleanup_faults.append("leader_io:TimeoutError")
        if leader is not None and leader_closed:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(leader.close),
                    timeout=LOCAL_IO_CLOSE_TIMEOUT_S,
                )
            except Exception as exc:
                leader_closed = False
                cleanup_faults.append(f"leader_close:{type(exc).__name__}")

        lease_released = True
        if lease is not None and self.registry.is_token_current(lease):
            self.registry.release(
                lease,
                HardwareReleaseReceipt(
                    safe=leader_closed,
                    device_closed=leader_closed,
                    torque_off=None,
                    torque_not_applicable=True,
                    evidence=(
                        "operator leader closed during cancelled start"
                        if leader_closed
                        else "operator leader close was not confirmed during cancelled start"
                    ),
                ),
            )
            lease_released = not self.registry.is_token_current(lease)

        state = "fault" if failed or not leader_closed or not lease_released else "idle"
        receipt = {
            "reason": reason,
            "robot_confirmation_available": confirmation_available,
            "robot_receipt": remote_receipt,
            "leader_closed": leader_closed,
            "lease_released": lease_released,
            "cleanup_faults": cleanup_faults,
            "state": state,
        }
        with self._lock:
            if self._starting is starting:
                self._config = None
                self._client = None
                self._leader = None
                self._sender = None
                self._lease = None
                self._grant = None
                self._spec = None
                self._browser_last_ns = None
                self._starting = None
                self._state = state
                self._fault = reason if failed else (cleanup_faults[0] if cleanup_faults else None)
                if state == "fault":
                    self._faulted_leader = leader
                    self._faulted_lease = lease
                else:
                    self._faulted_leader = None
                    self._faulted_lease = None
                self._last_stop = receipt

    def _run_leader_io(self, operation: Callable[[], object]) -> object:
        try:
            return operation()
        finally:
            self._leader_io_idle.set()

    def _browser_live(self) -> bool:
        with self._lock:
            last = self._browser_last_ns
            state = self._state
        return (
            state == "streaming"
            and last is not None
            and self.clock_ns() - last < int(BROWSER_LIVENESS_S * 1_000_000_000)
        )

    def browser_heartbeat(self) -> dict[str, object]:
        with self._lock:
            if self._state != "streaming":
                raise RuntimeError("remote operator is not streaming")
            self._browser_last_ns = self.clock_ns()
        return {"accepted": True, "browser_live": True}

    async def _action_loop(self) -> None:
        while True:
            with self._lock:
                config = self._config
                leader = self._leader
                sender = self._sender
                grant = self._grant
                spec = self._spec
                state = self._state
            if (
                state != "streaming"
                or config is None
                or leader is None
                or sender is None
                or grant is None
                or spec is None
            ):
                return
            if not self._browser_live():
                # Heartbeat sends browser_live=false; robot performs the
                # authoritative local stop. No more actions leave this host.
                return
            started = time.monotonic()
            try:
                self._leader_io_idle.clear()
                raw = await asyncio.to_thread(self._run_leader_io, leader.read)
                self._sequence += 1
                source_ns = raw.sampled_monotonic_ns
                sample = ActionSample(
                    session_id=grant.session_id,
                    source_id=spec.source_id,
                    executor_generation=grant.executor_generation,
                    rig_id=spec.rig_id,
                    rig_digest=spec.rig_digest,
                    leader_calibration_id=spec.leader_calibration_id,
                    leader_calibration_digest=spec.leader_calibration_digest,
                    follower_calibration_id=spec.follower_calibration_id,
                    follower_calibration_digest=spec.follower_calibration_digest,
                    sequence=self._sequence,
                    source_monotonic_ns=source_ns,
                    expires_at_source_monotonic_ns=source_ns + ACTION_LIFETIME_NS,
                    joint_names=spec.joint_names,
                    units=spec.units,
                    positions=tuple(raw.positions[joint] for joint in spec.joint_names),
                )
                await asyncio.to_thread(
                    sender.send,
                    encode_action(sample, key_id=grant.key_id, key=grant.action_key()),
                )
                with self._lock:
                    self._last_action_ns = self.clock_ns()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                asyncio.create_task(self._fail_async(f"leader_or_udp:{type(exc).__name__}"))
                return
            await asyncio.sleep(max(0.0, 1.0 / config.action_rate_hz - (time.monotonic() - started)))

    async def _clock_check_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            with self._lock:
                client = self._client
                if self._state != "streaming" or client is None:
                    return
            try:
                mapping, samples = await client.synchronize_clocks()
                if not await client.check_clock(samples):
                    raise RuntimeError("robot rejected clock drift")
                with self._lock:
                    self._clock_uncertainty_ns = mapping.uncertainty_ns
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                asyncio.create_task(self._fail_async(f"clock:{type(exc).__name__}"))
                return

    def _heartbeat_failed(self, exc: Exception) -> None:
        asyncio.create_task(self._fail_async(f"control:{type(exc).__name__}"))

    async def _fail_async(self, reason: str) -> None:
        with self._lock:
            if self._state in {"stopping", "idle", "fault"}:
                return
            self._fault = reason
        await self._cleanup_async(reason, contact_robot=False)

    def stop(self, reason: str = "operator_stop") -> dict[str, object]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason is required")
        if not self._loop.active:
            return dict(self._last_stop or {"duplicate": True, "state": self._state})
        return self._loop.submit(
            self._cleanup_async(reason.strip(), contact_robot=True),
            timeout=STARTUP_STOP_TIMEOUT_S + 10.0,
        )

    async def _cleanup_async(self, reason: str, *, contact_robot: bool) -> dict[str, object]:
        with self._lock:
            starting = self._starting
            if starting is not None:
                starting.stop_reason = reason
                starting.cancel.set()
                self._state = "stopping"
                if self._lease is not None and self.registry.is_token_current(self._lease):
                    self.registry.request_stop(self._lease, reason)
                starting_done = starting.done
            else:
                starting_done = None
            if self._state == "stopping":
                if starting_done is None:
                    return {"duplicate": True, "state": "stopping"}
            else:
                self._state = "stopping"
        if starting_done is not None:
            completed = await asyncio.to_thread(starting_done.wait, STARTUP_STOP_TIMEOUT_S)
            if not completed:
                return {
                    "reason": reason,
                    "robot_confirmation_available": False,
                    "leader_closed": False,
                    "lease_released": False,
                    "state": "stopping",
                }
            with self._lock:
                return dict(
                    self._last_stop
                    or {
                        "reason": reason,
                        "robot_confirmation_available": False,
                        "leader_closed": self._leader is None,
                        "lease_released": self._lease is None,
                        "state": self._state,
                    }
                )

        with self._lock:
            self._state = "stopping"
            action_task = self._action_task
            clock_task = self._clock_task
            client = self._client
            sender = self._sender
            leader = self._leader
            lease = self._lease
            if lease is not None and self.registry.is_token_current(lease):
                self.registry.request_stop(lease, reason)
            self._action_task = None
            self._clock_task = None
        current = asyncio.current_task()
        for task in (action_task, clock_task):
            if task is not None and task is not current:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        remote_receipt: dict[str, object] | None = None
        confirmation_available = False
        cleanup_faults: list[str] = []
        if contact_robot and client is not None:
            try:
                remote_receipt = await client.stop(reason)
                confirmation_available = True
            except Exception as exc:
                cleanup_faults.append(f"robot_confirmation:{type(exc).__name__}")
        if sender is not None:
            try:
                sender.close()
            except Exception as exc:
                cleanup_faults.append(f"sender_close:{type(exc).__name__}")
        if client is not None:
            try:
                await asyncio.wait_for(client.close(), timeout=LOCAL_IO_CLOSE_TIMEOUT_S)
            except Exception as exc:
                cleanup_faults.append(f"control_close:{type(exc).__name__}")
        leader_closed = await asyncio.to_thread(
            self._leader_io_idle.wait,
            LOCAL_IO_CLOSE_TIMEOUT_S,
        )
        if not leader_closed:
            cleanup_faults.append("leader_io:TimeoutError")
        if leader is not None and leader_closed:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(leader.close),
                    timeout=LOCAL_IO_CLOSE_TIMEOUT_S,
                )
            except Exception as exc:
                leader_closed = False
                cleanup_faults.append(f"leader_close:{type(exc).__name__}")

        lease_released = True
        if lease is not None and self.registry.is_token_current(lease):
            self.registry.release(
                lease,
                HardwareReleaseReceipt(
                    safe=leader_closed,
                    device_closed=leader_closed,
                    torque_off=None,
                    torque_not_applicable=True,
                    evidence=(
                        "operator leader closed"
                        if leader_closed
                        else "operator leader close was not confirmed"
                    ),
                ),
            )
            lease_released = not self.registry.is_token_current(lease)

        final_state = "idle" if leader_closed and lease_released else "fault"
        receipt = {
            "reason": reason,
            "robot_confirmation_available": confirmation_available,
            "robot_receipt": remote_receipt,
            "leader_closed": leader_closed,
            "lease_released": lease_released,
            "cleanup_faults": cleanup_faults,
            "state": final_state,
        }
        with self._lock:
            self._config = None
            self._client = None
            self._leader = None
            self._sender = None
            self._lease = None
            self._grant = None
            self._spec = None
            self._browser_last_ns = None
            self._state = final_state
            self._fault = cleanup_faults[0] if cleanup_faults else None
            if final_state == "fault":
                self._faulted_leader = leader
                self._faulted_lease = lease
            else:
                self._faulted_leader = None
                self._faulted_lease = None
            self._last_stop = receipt
        return dict(receipt)

    def status(self) -> dict[str, object]:
        with self._lock:
            grant = self._grant
            faulted_leader = self._faulted_leader
            faulted_lease = self._faulted_lease
            return {
                "protocol_version": "makermodslab.remote-operator-service.v1",
                "role": "operator",
                "state": self._state,
                "runtime_enabled": self._state not in {"idle", "fault"},
                "live_hardware_enabled": self._leader is not None,
                "credential_id": self._credential_id,
                "session": (
                    {
                        "session_id": grant.session_id,
                        "executor_generation": grant.executor_generation,
                        "key_id": grant.key_id,
                    }
                    if grant is not None
                    else None
                ),
                "browser_live": self._browser_live(),
                "last_sequence": self._sequence if grant is not None else None,
                "last_action_monotonic_ns": self._last_action_ns,
                "clock_uncertainty_ns": self._clock_uncertainty_ns,
                "fault": self._fault,
                "last_stop": dict(self._last_stop) if self._last_stop else None,
                "fault_resource": (
                    {
                        "lease_retained": faulted_lease is not None,
                        "leader_retained": faulted_leader is not None,
                    }
                    if faulted_lease is not None or faulted_leader is not None
                    else None
                ),
            }

    def shutdown(self) -> None:
        with suppress(Exception):
            self.stop("server_shutdown")
        self._loop.close()
