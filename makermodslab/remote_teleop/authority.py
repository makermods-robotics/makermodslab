"""Robot-minted, single-source remote-session authority."""

from __future__ import annotations

import secrets
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .contracts import ActionSample, SessionSpec


class SessionAuthorityError(RuntimeError):
    """The robot host refused a session transition or action."""


@dataclass(frozen=True)
class SessionGrant:
    session_id: str
    executor_generation: int
    key_id: str
    action_key: bytes = field(repr=False)
    opened_monotonic_ns: int
    clock_offset_ns: int
    clock_uncertainty_ns: int
    spec: SessionSpec

    def public(self) -> dict[str, object]:
        """Status-safe fields only. The action key never appears in status."""
        return {
            "session_id": self.session_id,
            "executor_generation": self.executor_generation,
            "key_id": self.key_id,
            "opened_monotonic_ns": self.opened_monotonic_ns,
            "source_id": self.spec.source_id,
            "rig_id": self.spec.rig_id,
            "joint_names": list(self.spec.joint_names),
            "units": list(self.spec.units),
        }


class RemoteSessionAuthority:
    """One robot-side action owner, one generation, one sequence window."""

    def __init__(self, *, max_action_age_ns: int = 150_000_000) -> None:
        if not 1_000_000 <= max_action_age_ns <= 1_000_000_000:
            raise ValueError("max_action_age_ns must be in [1ms,1s]")
        self.max_action_age_ns = max_action_age_ns
        self._lock = threading.RLock()
        self._generation = 0
        self._grant: SessionGrant | None = None
        self._highest_sequence: int | None = None
        self._last_stop: dict[str, object] | None = None
        self._counters: Counter[str] = Counter()

    @property
    def grant(self) -> SessionGrant | None:
        with self._lock:
            return self._grant

    def open_session(
        self,
        spec: SessionSpec,
        *,
        now_ns: int,
        clock_offset_ns: int = 0,
        clock_uncertainty_ns: int = 0,
    ) -> SessionGrant:
        if now_ns < 0 or clock_uncertainty_ns < 0:
            raise ValueError("session clocks must be non-negative")
        with self._lock:
            if self._grant is not None:
                self._counters["session_rejected_duplicate"] += 1
                raise SessionAuthorityError("a remote action session is already active")
            self._generation += 1
            session_id = uuid.uuid4().hex
            grant = SessionGrant(
                session_id=session_id,
                executor_generation=self._generation,
                key_id=f"action-{uuid.uuid4().hex}",
                action_key=secrets.token_bytes(32),
                opened_monotonic_ns=now_ns,
                clock_offset_ns=clock_offset_ns,
                clock_uncertainty_ns=clock_uncertainty_ns,
                spec=spec,
            )
            self._grant = grant
            self._highest_sequence = None
            self._last_stop = None
            self._counters["session_opened"] += 1
            return grant

    def key_for(self, key_id: str) -> bytes | None:
        with self._lock:
            grant = self._grant
            return grant.action_key if grant is not None and grant.key_id == key_id else None

    def accept(self, sample: ActionSample, *, received_monotonic_ns: int) -> int:
        """Validate identity/time/sequence, returning estimated one-way latency."""
        with self._lock:
            grant = self._grant
            if grant is None:
                self._reject("no_session")
            assert grant is not None
            spec = grant.spec
            checks = {
                "session": sample.session_id == grant.session_id,
                "source": sample.source_id == spec.source_id,
                "generation": sample.executor_generation == grant.executor_generation,
                "rig": sample.rig_id == spec.rig_id and sample.rig_digest == spec.rig_digest,
                "leader_calibration": sample.leader_calibration_id == spec.leader_calibration_id
                and sample.leader_calibration_digest == spec.leader_calibration_digest,
                "follower_calibration": sample.follower_calibration_id == spec.follower_calibration_id
                and sample.follower_calibration_digest == spec.follower_calibration_digest,
                "joint_schema": sample.joint_names == spec.joint_names and sample.units == spec.units,
            }
            failed = next((name for name, valid in checks.items() if not valid), None)
            if failed is not None:
                self._reject(failed)
            source_at_robot = sample.source_monotonic_ns + grant.clock_offset_ns
            expiry_at_robot = sample.expires_at_source_monotonic_ns + grant.clock_offset_ns
            uncertainty = grant.clock_uncertainty_ns
            if source_at_robot - uncertainty > received_monotonic_ns:
                self._reject("future_timestamp")
            age = max(0, received_monotonic_ns - source_at_robot)
            if age + uncertainty > self.max_action_age_ns:
                self._reject("stale")
            if received_monotonic_ns + uncertainty >= expiry_at_robot:
                self._reject("expired")
            if self._highest_sequence is not None and sample.sequence <= self._highest_sequence:
                self._reject("sequence")
            self._highest_sequence = sample.sequence
            self._counters["action_admitted"] += 1
            return age

    def _reject(self, reason: str) -> None:
        self._counters[f"action_rejected_{reason}"] += 1
        raise SessionAuthorityError(f"remote action rejected: {reason}")

    def stop(self, *, reason: str, now_ns: int) -> dict[str, object]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason is required")
        with self._lock:
            grant = self._grant
            if grant is None:
                return {
                    **(self._last_stop or {"reason": reason, "stopped_monotonic_ns": now_ns}),
                    "duplicate": True,
                }
            prior_generation = grant.executor_generation
            self._generation += 1
            self._grant = None
            self._highest_sequence = None
            self._last_stop = {
                "session_id": grant.session_id,
                "reason": reason.strip(),
                "revoked_generation": prior_generation,
                "next_generation": self._generation,
                "stopped_monotonic_ns": now_ns,
                "duplicate": False,
            }
            self._counters["session_stopped"] += 1
            return dict(self._last_stop)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": "active" if self._grant is not None else "idle",
                "executor_generation": self._generation,
                "session": self._grant.public() if self._grant else None,
                "highest_sequence": self._highest_sequence,
                "last_stop": dict(self._last_stop) if self._last_stop else None,
                "counters": dict(self._counters),
            }
