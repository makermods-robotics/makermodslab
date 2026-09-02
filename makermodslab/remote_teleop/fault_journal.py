"""Durable, path-free lockout evidence for unresolved arm shutdowns."""

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

from .commissioning import profile_commissioning_digest

FAULT_JOURNAL_VERSION = "makermodslab.remote-so101-fault.v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class FaultJournalError(RuntimeError):
    """The durable hardware fault latch is invalid or permission-unsafe."""


class FaultableProfile(Protocol):
    rig_id: str
    rig_digest: str
    limits_digest: str
    device_identity_digest: str
    joint_names: tuple[str, ...]
    units: tuple[str, ...]
    follower_calibration: object
    leader_calibration: object


@dataclass(frozen=True)
class HardwareFaultRecord:
    """Sanitized safety facts only; never exception text, paths, or secrets."""

    version: str
    profile_digest: str
    rig_id: str
    rig_digest: str
    follower_calibration_id: str
    follower_calibration_digest: str
    reason_code: str
    fault_codes: tuple[str, ...]
    hardware_stop_completed: bool
    device_closed: bool
    torque_off_confirmed: bool
    occurred_at_utc: str

    def __post_init__(self) -> None:
        if self.version != FAULT_JOURNAL_VERSION:
            raise FaultJournalError("hardware fault record version is unsupported")
        for name in (
            "profile_digest",
            "rig_digest",
            "follower_calibration_digest",
        ):
            if not _DIGEST.fullmatch(getattr(self, name)):
                raise FaultJournalError(f"hardware fault {name} is invalid")
        if not self.rig_id or not self.follower_calibration_id:
            raise FaultJournalError("hardware fault identity is incomplete")
        if not _CODE.fullmatch(self.reason_code):
            raise FaultJournalError("hardware fault reason code is invalid")
        if not self.fault_codes or any(not _CODE.fullmatch(code) for code in self.fault_codes):
            raise FaultJournalError("hardware fault codes are invalid")
        if len(set(self.fault_codes)) != len(self.fault_codes):
            raise FaultJournalError("hardware fault codes must be unique")
        if self.hardware_stop_completed and self.device_closed and self.torque_off_confirmed:
            raise FaultJournalError("a fully safe close must not create a fault latch")
        try:
            parsed = datetime.fromisoformat(self.occurred_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FaultJournalError("hardware fault timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise FaultJournalError("hardware fault timestamp must include UTC")

    @classmethod
    def from_profile(
        cls,
        profile: FaultableProfile,
        *,
        reason_code: str,
        fault_codes: tuple[str, ...],
        hardware_stop_completed: bool,
        device_closed: bool,
        torque_off_confirmed: bool,
    ) -> HardwareFaultRecord:
        return cls(
            version=FAULT_JOURNAL_VERSION,
            profile_digest=profile_commissioning_digest(profile),
            rig_id=profile.rig_id,
            rig_digest=profile.rig_digest,
            follower_calibration_id=profile.follower_calibration.calibration_id,
            follower_calibration_digest=profile.follower_calibration.digest,
            reason_code=reason_code,
            fault_codes=fault_codes,
            hardware_stop_completed=hardware_stop_completed,
            device_closed=device_closed,
            torque_off_confirmed=torque_off_confirmed,
            occurred_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    def public(self) -> dict[str, object]:
        return asdict(self)


class RemoteFaultJournal:
    """Atomic owner-private persistence for the fail-closed hardware latch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "unresolved-hardware.json"

    def load(self) -> HardwareFaultRecord | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise FaultJournalError("hardware fault record must not be a symlink")
        if os.name != "nt" and stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise FaultJournalError("hardware fault record must be owner-only")
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FaultJournalError("hardware fault record is unreadable") from exc
        if not isinstance(body, dict) or set(body) != set(HardwareFaultRecord.__dataclass_fields__):
            raise FaultJournalError("hardware fault record has an unsupported shape")
        if not isinstance(body.get("fault_codes"), list):
            raise FaultJournalError("hardware fault codes are invalid")
        body["fault_codes"] = tuple(body["fault_codes"])
        try:
            return HardwareFaultRecord(**body)
        except TypeError as exc:
            raise FaultJournalError("hardware fault record is invalid") from exc

    def save(self, record: HardwareFaultRecord) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.root.chmod(0o700)
        descriptor, name = tempfile.mkstemp(prefix=".hardware-fault-", dir=self.root)
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

    def clear_after_recovery(self) -> None:
        """Remove the latch only after the caller has safe-close evidence."""
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def public(self) -> dict[str, object]:
        record = self.load()
        return {"fault_lockout": record is not None, "record": None if record is None else record.public()}
