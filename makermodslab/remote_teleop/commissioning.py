"""Owner-private proof that one robot profile passed secured-arm checks."""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .calibration_identity import canonical_json

COMMISSIONING_VERSION = "makermodslab.remote-so101-commissioning.v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class CommissioningError(RuntimeError):
    """The selected live profile has no matching secured-arm proof."""


class CommissionableProfile(Protocol):
    rig_id: str
    rig_digest: str
    limits_digest: str
    device_identity_digest: str
    joint_names: tuple[str, ...]
    units: tuple[str, ...]
    follower_calibration: object
    leader_calibration: object


def profile_commissioning_digest(profile: CommissionableProfile) -> str:
    """Bind approval to every identity, joint, unit, and enforced limit."""
    import hashlib

    body = {
        "version": COMMISSIONING_VERSION,
        "rig_id": profile.rig_id,
        "rig_digest": profile.rig_digest,
        "limits_digest": profile.limits_digest,
        "device_identity_digest": profile.device_identity_digest,
        "joint_names": list(profile.joint_names),
        "units": list(profile.units),
        "follower_calibration_id": profile.follower_calibration.calibration_id,
        "follower_calibration_digest": profile.follower_calibration.digest,
        "leader_calibration_id": profile.leader_calibration.calibration_id,
        "leader_calibration_digest": profile.leader_calibration.digest,
    }
    return hashlib.sha256(canonical_json(body)).hexdigest()


@dataclass(frozen=True)
class CommissioningRecord:
    version: str
    profile_digest: str
    rig_id: str
    rig_digest: str
    follower_calibration_id: str
    follower_calibration_digest: str
    leader_calibration_id: str
    leader_calibration_digest: str
    limits_digest: str
    device_identity_digest: str
    commissioned_at_utc: str
    checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != COMMISSIONING_VERSION:
            raise CommissioningError("commissioning record version is unsupported")
        for name in (
            "profile_digest",
            "rig_digest",
            "follower_calibration_digest",
            "leader_calibration_digest",
            "limits_digest",
            "device_identity_digest",
        ):
            if not _DIGEST.fullmatch(getattr(self, name)):
                raise CommissioningError(f"commissioning {name} is invalid")
        if not self.rig_id or not self.follower_calibration_id or not self.leader_calibration_id:
            raise CommissioningError("commissioning identity is incomplete")
        if self.checks != (
            "arm_secured",
            "power_removal_reachable",
            "workspace_clear",
            "profile_verified",
            "follower_connected_disarmed",
            "initial_observation_read",
            "torque_off_readback",
            "device_closed",
        ):
            raise CommissioningError("commissioning checks are incomplete")
        try:
            parsed = datetime.fromisoformat(self.commissioned_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CommissioningError("commissioning timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise CommissioningError("commissioning timestamp must include UTC")

    @classmethod
    def from_profile(
        cls,
        profile: CommissionableProfile,
    ) -> CommissioningRecord:
        return cls(
            version=COMMISSIONING_VERSION,
            profile_digest=profile_commissioning_digest(profile),
            rig_id=profile.rig_id,
            rig_digest=profile.rig_digest,
            follower_calibration_id=profile.follower_calibration.calibration_id,
            follower_calibration_digest=profile.follower_calibration.digest,
            leader_calibration_id=profile.leader_calibration.calibration_id,
            leader_calibration_digest=profile.leader_calibration.digest,
            limits_digest=profile.limits_digest,
            device_identity_digest=profile.device_identity_digest,
            commissioned_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            checks=(
                "arm_secured",
                "power_removal_reachable",
                "workspace_clear",
                "profile_verified",
                "follower_connected_disarmed",
                "initial_observation_read",
                "torque_off_readback",
                "device_closed",
            ),
        )

    def public(self) -> dict[str, object]:
        return asdict(self)


class CommissioningStore:
    """Atomic record storage. There is deliberately no unchecked import API."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "so101-commissioning.json"

    def load(self) -> CommissioningRecord | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise CommissioningError("commissioning record must not be a symlink")
        if os.name != "nt" and stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise CommissioningError("commissioning record must be owner-only")
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CommissioningError("commissioning record is unreadable") from exc
        if not isinstance(body, dict) or set(body) != set(CommissioningRecord.__dataclass_fields__):
            raise CommissioningError("commissioning record has an unsupported shape")
        if not isinstance(body.get("checks"), list):
            raise CommissioningError("commissioning checks are invalid")
        body["checks"] = tuple(body["checks"])
        try:
            return CommissioningRecord(**body)
        except TypeError as exc:
            raise CommissioningError("commissioning record is invalid") from exc

    def require(self, profile: CommissionableProfile) -> CommissioningRecord:
        record = self.load()
        if record is None:
            raise CommissioningError("this SO-101 profile has not passed the secured-arm commissioning check")
        if record.profile_digest != profile_commissioning_digest(profile):
            raise CommissioningError(
                "commissioning does not match the current calibrations, schema, or limits"
            )
        return record

    def save(self, record: CommissioningRecord) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.root.chmod(0o700)
        descriptor, name = tempfile.mkstemp(prefix=".commissioning-", dir=self.root)
        temporary = Path(name)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            payload = json.dumps(
                record.public(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                self.path.chmod(0o600)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def invalidate(self) -> None:
        """Conservatively require a fresh proof after robot config is saved."""
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def public(self) -> dict[str, object]:
        record = self.load()
        return {
            "commissioned": record is not None,
            "record": None if record is None else record.public(),
        }
