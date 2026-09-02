"""Versioned, authenticated action values for split-host teleoperation."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

PROTOCOL_VERSION = "makermodslab.remote-teleop.v1"
MESSAGE_TYPE = "action.sample"
MAX_DATAGRAM_BYTES = 4096
MAX_ACTION_LIFETIME_NS = 250_000_000
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_UINT64_MAX = 2**64 - 1


class ActionContractError(ValueError):
    """A remote action is malformed, unauthenticated, or incompatible."""


class ActionSource(Protocol):
    """Provider-neutral source: leader, policy, or test double."""

    source_id: str

    def next_action(self, deadline_monotonic_ns: int) -> ActionSample | None: ...

    def close(self, reason: str) -> None: ...


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ActionContractError(f"{name} must be a path-free identifier")
    return value


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ActionContractError(f"{name} must be a SHA-256 digest")
    return value


def _uint64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX:
        raise ActionContractError(f"{name} must be an unsigned 64-bit integer")
    return value


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ActionContractError(f"{name} must be a non-empty list")
    result = tuple(_identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ActionContractError(f"{name} must not contain duplicates")
    return result


def _units(values: Sequence[str], count: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ActionContractError("units must match joint_names")
    result = []
    for value in values:
        if not isinstance(value, str) or not value or len(value.encode()) > 32:
            raise ActionContractError("units must be short non-empty strings")
        result.append(value)
    return tuple(result)


def _positions(values: Sequence[float], count: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ActionContractError("positions must match joint_names")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ActionContractError("positions must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ActionContractError("positions must be finite")
        result.append(number)
    return tuple(result)


def _canonical(value: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    except (TypeError, ValueError) as exc:
        raise ActionContractError("action is not canonical JSON") from exc
    if len(encoded) > MAX_DATAGRAM_BYTES:
        raise ActionContractError("action datagram exceeds the size bound")
    return encoded


@dataclass(frozen=True)
class SessionSpec:
    """Robot-verified identities and exact action vocabulary for one session."""

    source_id: str
    rig_id: str
    rig_digest: str
    leader_calibration_id: str
    leader_calibration_digest: str
    follower_calibration_id: str
    follower_calibration_digest: str
    joint_names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "source_id",
            "rig_id",
            "leader_calibration_id",
            "follower_calibration_id",
        ):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))
        for field in ("rig_digest", "leader_calibration_digest", "follower_calibration_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        joints = _names(self.joint_names, "joint_names")
        object.__setattr__(self, "joint_names", joints)
        object.__setattr__(self, "units", _units(self.units, len(joints)))


@dataclass(frozen=True)
class ActionSample:
    """One latest-value action proposal; the robot still decides whether to execute."""

    session_id: str
    source_id: str
    executor_generation: int
    rig_id: str
    rig_digest: str
    leader_calibration_id: str
    leader_calibration_digest: str
    follower_calibration_id: str
    follower_calibration_digest: str
    sequence: int
    source_monotonic_ns: int
    expires_at_source_monotonic_ns: int
    joint_names: tuple[str, ...]
    units: tuple[str, ...]
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in (
            "session_id",
            "source_id",
            "rig_id",
            "leader_calibration_id",
            "follower_calibration_id",
        ):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))
        for field in ("rig_digest", "leader_calibration_digest", "follower_calibration_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self, "executor_generation", _uint64(self.executor_generation, "executor_generation")
        )
        object.__setattr__(self, "sequence", _uint64(self.sequence, "sequence"))
        sent = _uint64(self.source_monotonic_ns, "source_monotonic_ns")
        expires = _uint64(self.expires_at_source_monotonic_ns, "expires_at_source_monotonic_ns")
        if expires <= sent or expires - sent > MAX_ACTION_LIFETIME_NS:
            raise ActionContractError("action lifetime must be in (0, 250ms]")
        joints = _names(self.joint_names, "joint_names")
        object.__setattr__(self, "joint_names", joints)
        object.__setattr__(self, "units", _units(self.units, len(joints)))
        object.__setattr__(self, "positions", _positions(self.positions, len(joints)))

    def body(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": MESSAGE_TYPE,
            "session_id": self.session_id,
            "source_id": self.source_id,
            "executor_generation": self.executor_generation,
            "rig_id": self.rig_id,
            "rig_digest": self.rig_digest,
            "leader_calibration_id": self.leader_calibration_id,
            "leader_calibration_digest": self.leader_calibration_digest,
            "follower_calibration_id": self.follower_calibration_id,
            "follower_calibration_digest": self.follower_calibration_digest,
            "sequence": self.sequence,
            "source_monotonic_ns": self.source_monotonic_ns,
            "expires_at_source_monotonic_ns": self.expires_at_source_monotonic_ns,
            "joint_names": list(self.joint_names),
            "units": list(self.units),
            "positions": list(self.positions),
        }

    @classmethod
    def from_body(cls, body: Mapping[str, object]) -> ActionSample:
        expected = {
            "protocol_version",
            "message_type",
            "session_id",
            "source_id",
            "executor_generation",
            "rig_id",
            "rig_digest",
            "leader_calibration_id",
            "leader_calibration_digest",
            "follower_calibration_id",
            "follower_calibration_digest",
            "sequence",
            "source_monotonic_ns",
            "expires_at_source_monotonic_ns",
            "joint_names",
            "units",
            "positions",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise ActionContractError("action has missing or extra fields")
        if body["protocol_version"] != PROTOCOL_VERSION or body["message_type"] != MESSAGE_TYPE:
            raise ActionContractError("unsupported action protocol")
        values = dict(body)
        values.pop("protocol_version")
        values.pop("message_type")
        return cls(**values)


def encode_action(sample: ActionSample, *, key_id: str, key: bytes) -> bytes:
    """Canonical JSON plus HMAC-SHA256; safe to place in one UDP datagram."""
    _identifier(key_id, "key_id")
    if not isinstance(key, bytes) or len(key) < 32:
        raise ActionContractError("action key must contain at least 256 bits")
    unsigned = {**sample.body(), "key_id": key_id}
    tag = hmac.new(key, _canonical(unsigned), hashlib.sha256).hexdigest()
    return _canonical({**unsigned, "auth_tag": tag})


def decode_action(raw: bytes, *, key_lookup: Callable[[str], bytes | None]) -> ActionSample:
    """Authenticate before constructing the typed sample; reject non-canonical input."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_DATAGRAM_BYTES:
        raise ActionContractError("action datagram is empty or oversized")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ActionContractError("action datagram is invalid JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("key_id"), str):
        raise ActionContractError("action authentication fields are missing")
    tag = body.get("auth_tag")
    if not isinstance(tag, str):
        raise ActionContractError("action authentication fields are missing")
    key = key_lookup(body["key_id"])
    if key is None:
        raise ActionContractError("unknown action key")
    unsigned = dict(body)
    unsigned.pop("auth_tag")
    expected = hmac.new(key, _canonical(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag, expected):
        raise ActionContractError("action authentication failed")
    unsigned.pop("key_id")
    sample = ActionSample.from_body(unsigned)
    if encode_action(sample, key_id=body["key_id"], key=key) != raw:
        raise ActionContractError("action datagram is not canonical")
    return sample
