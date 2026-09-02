"""Canonical, path-free calibration and rig identities.

The network contract carries digests, never local calibration paths.  This
module is pure: it reads explicitly selected artifacts and contains no device
or socket operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_DIGEST_VERSION = "calibration-digest-v1"
RIG_DIGEST_VERSION = "rig-digest-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class IdentityError(ValueError):
    """A stable identity-class failure safe to return over the control API."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityError("calibration_duplicate_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise IdentityError("calibration_non_finite", f"non-finite JSON number: {value}")


def _assert_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IdentityError("identity_non_finite", "identity contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise IdentityError("identity_invalid_key", "identity object keys must be strings")
            _assert_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite(child)


def canonical_json(value: object) -> bytes:
    """Strict UTF-8 canonical JSON used by every identity digest."""
    _assert_finite(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IdentityError("identity_not_json", "identity value is not strict JSON") from exc


def load_strict_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        # Local absolute paths are not part of the public/network error
        # contract and must not survive as a chained traceback cause.
        raise IdentityError("calibration_unreadable", "selected calibration cannot be read") from None
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except IdentityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise IdentityError("calibration_invalid_json", "selected calibration is invalid JSON") from None


@dataclass(frozen=True)
class CalibrationIdentity:
    calibration_id: str
    digest: str
    algorithm: str = CALIBRATION_DIGEST_VERSION

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.calibration_id):
            raise IdentityError("calibration_id_invalid", "calibration id must be path-free")
        if not _DIGEST.fullmatch(self.digest):
            raise IdentityError("calibration_digest_invalid", "calibration digest must be SHA-256")
        if self.algorithm != CALIBRATION_DIGEST_VERSION:
            raise IdentityError("calibration_algorithm_invalid", "unsupported calibration digest version")

    def public(self) -> dict[str, str]:
        return {
            "calibration_id": self.calibration_id,
            "digest": self.digest,
            "algorithm": self.algorithm,
        }


def calibration_identity(path: Path, calibration_id: str | None = None) -> CalibrationIdentity:
    """Hash one selected artifact after strict parse and canonical re-encoding."""
    value = load_strict_json(path)
    if not isinstance(value, Mapping) or not value:
        raise IdentityError("calibration_invalid_shape", "calibration must be a non-empty object")
    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    return CalibrationIdentity(calibration_id or path.stem, digest)


def verify_calibration_identity(
    actual: CalibrationIdentity,
    *,
    expected_id: str,
    expected_digest: str,
    side: str,
) -> None:
    """Refuse an unexpected local artifact with a stable, non-secret code."""
    if actual.calibration_id != expected_id:
        raise IdentityError(f"{side}_calibration_id_mismatch", f"{side} calibration id does not match")
    if actual.digest != expected_digest:
        raise IdentityError(
            f"{side}_calibration_digest_mismatch", f"{side} calibration digest does not match"
        )


def verify_leader_allowlist(
    actual: CalibrationIdentity,
    allowed: Mapping[str, str],
) -> None:
    """Robot-side verification of the paired operator calibration."""
    digest = allowed.get(actual.calibration_id)
    if digest is None:
        raise IdentityError("leader_calibration_not_allowed", "leader calibration is not paired")
    verify_calibration_identity(
        actual,
        expected_id=actual.calibration_id,
        expected_digest=digest,
        side="leader",
    )


def derive_rig_digest(
    *,
    arm_family: str,
    topology: str,
    joint_schema: Sequence[Mapping[str, object]],
    leader: CalibrationIdentity,
    follower: CalibrationIdentity,
    limits: Mapping[str, object],
) -> str:
    """Robot-authoritative digest over identities, schema, and enforced limits."""
    if arm_family != "so101" or topology != "single":
        raise IdentityError("rig_scope_unsupported", "the live v1 rig must be single-arm SO-101")

    normalized_limits: list[dict[str, object]] = []
    for action_key, limit in limits.items():
        try:
            normalized_limits.append(
                {
                    "action_key": action_key,
                    "minimum": float(limit.minimum),
                    "maximum": float(limit.maximum),
                    "max_velocity_per_s": float(limit.max_velocity_per_s),
                    "max_acceleration_per_s2": float(limit.max_acceleration_per_s2),
                }
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise IdentityError("rig_limits_invalid", "rig limit object is incomplete") from exc

    body = {
        "digest_version": RIG_DIGEST_VERSION,
        "calibration_digest_version": CALIBRATION_DIGEST_VERSION,
        "arm_family": arm_family,
        "topology": topology,
        "joint_schema": list(joint_schema),
        "leader_calibration": leader.public(),
        "follower_calibration": follower.public(),
        "limits": normalized_limits,
    }
    return hashlib.sha256(canonical_json(body)).hexdigest()
