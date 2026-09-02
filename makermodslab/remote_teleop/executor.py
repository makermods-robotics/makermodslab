"""Fixed-rate, latest-value follower execution with robot-local fail-safe stop."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from .authority import RemoteSessionAuthority, SessionGrant
from .contracts import ActionSample, SessionSpec, decode_action


class FollowerDriver(Protocol):
    """The only boundary allowed to own and command the follower device."""

    joint_names: tuple[str, ...]

    def connect(self) -> None: ...

    def observe(self) -> Mapping[str, float]: ...

    def execute(self, positions: Mapping[str, float]) -> Mapping[str, float]: ...

    def stop(self, reason: str) -> Mapping[str, object] | None: ...

    def close(self) -> Mapping[str, object] | None: ...


class EventRecorder(Protocol):
    def __call__(self, event: Mapping[str, object]) -> bool: ...


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float
    max_velocity_per_s: float
    max_acceleration_per_s2: float

    def __post_init__(self) -> None:
        values = (
            self.minimum,
            self.maximum,
            self.max_velocity_per_s,
            self.max_acceleration_per_s2,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("joint limits must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("joint minimum must be below maximum")
        if self.max_velocity_per_s <= 0 or self.max_acceleration_per_s2 <= 0:
            raise ValueError("joint velocity and acceleration limits must be positive")


class RemoteExecutor:
    """One action owner, one latest slot, and an immediately revocable writer.

    Device I/O never runs while ``_lock`` is held. STOP first revokes the
    robot-minted generation and disables dispatch under that lock, then asks
    the adapter to disable and close the hardware. A blocking vendor SDK can
    therefore never prevent software revocation, although a live adapter must
    still provide a bounded or killable hardware boundary before it can claim
    a physical stop deadline.
    """

    def __init__(
        self,
        follower: FollowerDriver,
        limits: Mapping[str, JointLimit],
        *,
        tick_hz: int = 50,
        watchdog_ns: int = 200_000_000,
        first_action_deadline_ns: int = 1_000_000_000,
        mode: str = "live",
        clock_ns: Callable[[], int] = time.monotonic_ns,
        recorder: EventRecorder | None = None,
    ) -> None:
        if not 10 <= tick_hz <= 200:
            raise ValueError("tick_hz must be in [10,200]")
        if not 20_000_000 <= watchdog_ns <= 2_000_000_000:
            raise ValueError("watchdog_ns must be in [20ms,2s]")
        if not watchdog_ns <= first_action_deadline_ns <= 5_000_000_000:
            raise ValueError("first_action_deadline_ns must be between watchdog and 5s")
        if mode not in {"live", "simulation"}:
            raise ValueError("mode must be live or simulation")
        if tuple(limits) != tuple(follower.joint_names):
            raise ValueError("limit ordering must match follower joint ordering")
        self.follower = follower
        self.limits = dict(limits)
        self.tick_hz = tick_hz
        self.watchdog_ns = watchdog_ns
        self.first_action_deadline_ns = first_action_deadline_ns
        self.mode = mode
        self.clock_ns = clock_ns
        self.recorder = recorder
        self.authority = RemoteSessionAuthority(max_action_age_ns=min(watchdog_ns, 150_000_000))
        self._lock = threading.RLock()
        self._latest: ActionSample | None = None
        self._latest_received_ns: int | None = None
        self._latest_latency_ns: int | None = None
        self._session_opened_ns: int | None = None
        self._last_tick_ns: int | None = None
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        self._observation: dict[str, float] = {}
        self._events: deque[dict[str, object]] = deque(maxlen=512)
        self._counters: Counter[str] = Counter()
        self._recorder_failures = 0
        self._connected = False
        self._dispatch_enabled = False
        self._teardown_in_progress = False
        self._opening = False
        self._lifecycle_token = 0
        self._halted = threading.Event()
        self._halted.set()
        self._safety: dict[str, object] = self._idle_safety()

    @staticmethod
    def _idle_safety() -> dict[str, object]:
        return {
            "stop_accepted": False,
            "software_dispatch_halted": True,
            "disable_requested": False,
            "hardware_stop_completed": True,
            "hardware_close_completed": True,
            "torque_off_confirmed": None,
            "fault_lockout": False,
            "faults": [],
        }

    def open_session(
        self,
        spec: SessionSpec,
        *,
        clock_offset_ns: int = 0,
        clock_uncertainty_ns: int = 0,
    ) -> SessionGrant:
        with self._lock:
            if self.authority.grant is not None or self._opening:
                raise RuntimeError("remote executor already has an active session")
            if self._teardown_in_progress or self._safety.get("fault_lockout") is True:
                raise RuntimeError("remote executor is stopped in fault lockout")
            self._opening = True
            self._lifecycle_token += 1
            token = self._lifecycle_token
            self._halted.clear()

        connected = False
        try:
            self.follower.connect()
            connected = True
            observed = self._clean_positions(self.follower.observe())
            initial_torque_off = self._connected_torque_off_status()
        except Exception:
            if connected:
                with suppress(Exception):
                    self.follower.close()
            with self._lock:
                if self._lifecycle_token == token:
                    self._opening = False
                    self._halted.set()
            raise

        with self._lock:
            cancelled = not self._opening or self._lifecycle_token != token
            if not cancelled:
                now = self.clock_ns()
                grant = self.authority.open_session(
                    spec,
                    now_ns=now,
                    clock_offset_ns=clock_offset_ns,
                    clock_uncertainty_ns=clock_uncertainty_ns,
                )
                self._opening = False
                self._connected = True
                self._dispatch_enabled = True
                self._teardown_in_progress = False
                self._safety = {
                    "stop_accepted": False,
                    "software_dispatch_halted": False,
                    "disable_requested": False,
                    "hardware_stop_completed": False,
                    "hardware_close_completed": False,
                    "torque_off_confirmed": initial_torque_off,
                    "fault_lockout": False,
                    "faults": [],
                }
                self._positions = observed
                self._observation = dict(observed)
                self._velocities = dict.fromkeys(self.follower.joint_names, 0.0)
                self._latest = None
                self._latest_received_ns = None
                self._latest_latency_ns = None
                self._session_opened_ns = now
                self._last_tick_ns = now
                self._emit(
                    {
                        "event": "session.active",
                        "session_id": grant.session_id,
                        "executor_generation": grant.executor_generation,
                        "monotonic_ns": now,
                    }
                )
                return grant

        try:
            self.follower.close()
        finally:
            self._halted.set()
        raise RuntimeError("remote executor opening was cancelled")

    def _connected_torque_off_status(self) -> bool | None:
        """Project adapter evidence without inventing an initial torque state."""
        if self.mode == "simulation":
            return None
        status = getattr(self.follower, "child_status", None)
        if status is None:
            status = getattr(self.follower, "status", None)
        if callable(status):
            status = status()
        if not isinstance(status, Mapping):
            return None
        receipt = status.get("stop_receipt")
        if not isinstance(receipt, Mapping):
            return None
        torque_off = receipt.get("torque_off_confirmed")
        return torque_off if isinstance(torque_off, bool) else None

    def submit_datagram(self, raw: bytes) -> None:
        sample = decode_action(raw, key_lookup=self.authority.key_for)
        now = self.clock_ns()
        with self._lock:
            # Local physical bounds are part of admission. An invalid target
            # must not advance the sequence high-water mark.
            self._validate_targets(sample)
            latency_ns = self.authority.accept(sample, received_monotonic_ns=now)
            if not self._dispatch_enabled:
                raise RuntimeError("remote executor dispatch is halted")
            self._latest = sample
            self._latest_received_ns = now
            self._latest_latency_ns = latency_ns
            self._emit(
                {
                    "event": "action.admitted",
                    "session_id": sample.session_id,
                    "sequence": sample.sequence,
                    "source_positions": list(sample.positions),
                    "received_monotonic_ns": now,
                    "estimated_one_way_ns": latency_ns,
                }
            )

    def _validate_targets(self, sample: ActionSample) -> None:
        if sample.joint_names != tuple(self.limits):
            self._counters["action_rejected_joint_schema"] += 1
            raise ValueError("action joint ordering does not match configured limits")
        for joint, position in zip(sample.joint_names, sample.positions, strict=True):
            limit = self.limits[joint]
            if not limit.minimum <= position <= limit.maximum:
                self._counters["action_rejected_position"] += 1
                raise ValueError(f"target for {joint} is outside its configured limits")

    def _clean_positions(self, values: Mapping[str, float]) -> dict[str, float]:
        if set(values) != set(self.follower.joint_names):
            raise ValueError("follower observation changed joint membership")
        clean = {}
        for joint in self.follower.joint_names:
            value = values[joint]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"follower {joint} is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"follower {joint} is not finite")
            clean[joint] = number
        return clean

    def tick(self) -> Mapping[str, float] | None:
        now = self.clock_ns()
        stop_reason: str | None = None
        with self._lock:
            grant = self.authority.grant
            if grant is None or not self._dispatch_enabled:
                return None
            if self._latest_received_ns is None:
                opened = self._session_opened_ns if self._session_opened_ns is not None else now
                if now - opened >= self.first_action_deadline_ns:
                    stop_reason = "first_action_watchdog"
            elif now - self._latest_received_ns >= self.watchdog_ns:
                stop_reason = "stale_command_watchdog"
            if stop_reason is None and self._latest is None:
                return None
            latest = self._latest
            previous_tick = self._last_tick_ns if self._last_tick_ns is not None else now
            positions = dict(self._positions)
            velocities = dict(self._velocities)

        if stop_reason is not None:
            self.stop(stop_reason)
            return None
        assert latest is not None
        elapsed_s = (now - previous_tick) / 1_000_000_000
        if elapsed_s <= 0:
            raise RuntimeError("executor clock did not advance")
        # Watchdog and action freshness use the uncapped wall clock above.
        # Motion shaping does not: a delayed scheduler tick must never earn a
        # larger position or acceleration step than one configured control
        # period. The remaining target is approached on later ticks instead.
        dt_s = min(elapsed_s, 1.0 / self.tick_hz)
        target = dict(zip(latest.joint_names, latest.positions, strict=True))
        admitted: dict[str, float] = {}
        next_velocities: dict[str, float] = {}
        for joint in self.follower.joint_names:
            current = positions[joint]
            velocity = velocities[joint]
            limit = self.limits[joint]
            desired_velocity = max(
                -limit.max_velocity_per_s,
                min(limit.max_velocity_per_s, (target[joint] - current) / dt_s),
            )
            max_delta_velocity = limit.max_acceleration_per_s2 * dt_s
            next_velocity = max(
                velocity - max_delta_velocity,
                min(velocity + max_delta_velocity, desired_velocity),
            )
            next_position = current + next_velocity * dt_s
            if target[joint] >= current:
                next_position = min(next_position, target[joint])
            else:
                next_position = max(next_position, target[joint])
            admitted[joint] = max(limit.minimum, min(limit.maximum, next_position))
            next_velocities[joint] = next_velocity

        try:
            executed = self._clean_positions(self.follower.execute(admitted))
            observation = self._clean_positions(self.follower.observe())
        except Exception as exc:
            self.stop(f"follower_io:{type(exc).__name__}")
            raise

        with self._lock:
            current_grant = self.authority.grant
            if (
                current_grant is None
                or current_grant.session_id != grant.session_id
                or not self._dispatch_enabled
            ):
                self._counters["action_completed_after_stop"] += 1
                self._emit(
                    {
                        "event": "action.completed_after_stop",
                        "session_id": latest.session_id,
                        "sequence": latest.sequence,
                        "monotonic_ns": self.clock_ns(),
                    }
                )
                return None
            self._last_tick_ns = now
            self._velocities = next_velocities
            self._positions = dict(executed)
            self._observation = dict(observation)
            # The live SO-101 adapter returns from execute only after its
            # torque-enable readback succeeds.  Once a command completed,
            # reporting torque off would therefore be false. Simulation has
            # no physical torque state and remains explicitly unknown.
            if self.mode == "live":
                self._safety["torque_off_confirmed"] = False
            self._counters["action_executed"] += 1
            self._emit(
                {
                    "event": "action.executed",
                    "session_id": latest.session_id,
                    "sequence": latest.sequence,
                    "source_positions": list(latest.positions),
                    "admitted_positions": [admitted[joint] for joint in self.follower.joint_names],
                    "executed_positions": [executed[joint] for joint in self.follower.joint_names],
                    "observation_positions": [observation[joint] for joint in self.follower.joint_names],
                    "executed_monotonic_ns": now,
                    "network_latency_ns": self._latest_latency_ns,
                    "command_age_at_execution_ns": (
                        None
                        if self._latest_latency_ns is None or self._latest_received_ns is None
                        else self._latest_latency_ns + now - self._latest_received_ns
                    ),
                }
            )
            return dict(executed)

    def stop(self, reason: str = "explicit_stop") -> dict[str, object]:
        now = self.clock_ns()
        with self._lock:
            opening = self._opening
            self._opening = False
            self._lifecycle_token += 1
            transition = self.authority.stop(reason=reason, now_ns=now)
            duplicate = bool(transition.get("duplicate"))
            self._dispatch_enabled = False
            self._latest = None
            self._latest_received_ns = None
            self._latest_latency_ns = None
            self._session_opened_ns = None
            if duplicate and not opening and not self._connected and not self._teardown_in_progress:
                return {**transition, "safety": self._safety_snapshot()}
            if self._teardown_in_progress:
                return {**transition, "safety": self._safety_snapshot()}
            self._teardown_in_progress = True
            self._safety = {
                "stop_accepted": True,
                "software_dispatch_halted": True,
                "disable_requested": False,
                "hardware_stop_completed": False,
                "hardware_close_completed": False,
                "torque_off_confirmed": None,
                "fault_lockout": True,
                "faults": [],
            }
            self._emit(
                {
                    "event": "session.stop_accepted",
                    "reason": reason,
                    "monotonic_ns": now,
                    "next_generation": transition.get("next_generation"),
                }
            )

        safety = self._teardown_hardware(reason)
        with self._lock:
            self._connected = False
            self._teardown_in_progress = False
            self._safety = safety
            self._halted.set()
            self._emit(
                {
                    "event": "session.stopped",
                    "reason": reason,
                    "monotonic_ns": self.clock_ns(),
                    "next_generation": transition.get("next_generation"),
                    "safety": dict(safety),
                }
            )
            return {**transition, "safety": self._safety_snapshot()}

    @staticmethod
    def _receipt_value(receipt: Mapping[str, object] | None, key: str) -> object | None:
        return receipt.get(key) if isinstance(receipt, Mapping) else None

    def _teardown_hardware(self, reason: str) -> dict[str, object]:
        faults: list[str] = []
        stop_receipt: Mapping[str, object] | None = None
        close_receipt: Mapping[str, object] | None = None
        try:
            stop_receipt = self.follower.stop(reason)
            reported = self._receipt_value(stop_receipt, "hardware_stop_completed")
            if not isinstance(reported, bool):
                reported = self._receipt_value(stop_receipt, "stop_completed")
            stop_completed = reported if isinstance(reported, bool) else False
        except Exception as exc:
            stop_completed = False
            faults.append(f"stop:{type(exc).__name__}")
        try:
            close_receipt = self.follower.close()
            reported = self._receipt_value(close_receipt, "close_completed")
            close_completed = reported if isinstance(reported, bool) else False
        except Exception as exc:
            close_completed = False
            faults.append(f"close:{type(exc).__name__}")

        for stage, receipt in (("stop", stop_receipt), ("close", close_receipt)):
            fault = self._receipt_value(receipt, "fault")
            if isinstance(fault, str) and fault:
                faults.append(f"{stage}:{fault}")

        confirmations = [
            value
            for value in (
                self._receipt_value(stop_receipt, "torque_off_confirmed"),
                self._receipt_value(close_receipt, "torque_off_confirmed"),
            )
            if isinstance(value, bool)
        ]
        # Any explicit nonzero readback defeats an affirmative result from a
        # different stage. Close normally supplies no torque evidence at all.
        torque_off_confirmed: bool | None
        if False in confirmations:
            torque_off_confirmed = False
        elif True in confirmations:
            torque_off_confirmed = True
        else:
            torque_off_confirmed = None
        disable_requested = any(
            self._receipt_value(receipt, "disable_requested") is True
            for receipt in (stop_receipt, close_receipt)
        )
        receipt_fault_lockout = any(
            self._receipt_value(receipt, "fault_lockout") is True for receipt in (stop_receipt, close_receipt)
        )
        return {
            "stop_accepted": True,
            "software_dispatch_halted": True,
            "disable_requested": disable_requested,
            "hardware_stop_completed": stop_completed,
            "hardware_close_completed": close_completed,
            "torque_off_confirmed": torque_off_confirmed,
            "fault_lockout": (
                bool(faults)
                or receipt_fault_lockout
                or not stop_completed
                or not close_completed
                or torque_off_confirmed is not True
            ),
            "faults": faults,
            "stop_receipt": dict(stop_receipt) if isinstance(stop_receipt, Mapping) else None,
            "close_receipt": dict(close_receipt) if isinstance(close_receipt, Mapping) else None,
        }

    def wait_until_halted(self, timeout: float | None = None) -> bool:
        """Wait for hardware teardown, without weakening stop acceptance."""
        return self._halted.wait(timeout)

    def _safety_snapshot(self) -> dict[str, object]:
        copy = dict(self._safety)
        copy["faults"] = list(self._safety.get("faults", []))
        return copy

    def _emit(self, event: Mapping[str, object]) -> None:
        copy = dict(event)
        self._events.append(copy)
        name = copy.get("event")
        if isinstance(name, str):
            self._counters[name] += 1
        if self.recorder is not None:
            try:
                if self.recorder(copy) is not True:
                    self._recorder_failures += 1
            except Exception:
                self._recorder_failures += 1

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "protocol_version": "makermodslab.remote-executor.v1",
                "mode": self.mode,
                "authority": self.authority.snapshot(),
                "tick_hz": self.tick_hz,
                "watchdog_ms": self.watchdog_ns / 1_000_000,
                "first_action_deadline_ms": self.first_action_deadline_ns / 1_000_000,
                "connected": self._connected,
                "dispatch_enabled": self._dispatch_enabled,
                "teardown_in_progress": self._teardown_in_progress,
                "latest_received_monotonic_ns": self._latest_received_ns,
                "observation": dict(self._observation) if self._observation else None,
                "safety": self._safety_snapshot(),
                "counters": dict(self._counters),
                "recorder_failures": self._recorder_failures,
                "recent_events": [dict(event) for event in self._events],
            }
