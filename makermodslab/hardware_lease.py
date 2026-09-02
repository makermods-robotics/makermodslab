# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Process-wide authority for opening or commanding arm hardware.

Feature-local ``*_active`` flags are useful status projections, but cannot
make a check-then-open sequence atomic across modules.  This registry is the
single authority boundary.  A start path claims before opening a device; a
stop path only records teardown intent; and the hardware-owning finalizer is
the only path allowed to release the claim.

An unsafe or unconfirmed finalizer retains the lease in ``unresolved`` state.
That deliberately blocks every new claimant until an evidence-backed recovery
marks the hardware safe.  Losing a worker must never manufacture an idle arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .api_errors import ErrorCode


@dataclass(frozen=True, slots=True)
class HardwareLeaseToken:
    lease_id: str
    generation: int
    kind: str
    owner: str
    resource: str


@dataclass(frozen=True, slots=True)
class HardwareRecoveryIdentity:
    """Sanitized identity and authority required to recover one hardware claim.

    Device paths never enter the durable journal. Callers prove they are
    recovering the same hardware by presenting an identity derived from the
    recovery mechanism, arm family, and canonical target strings.
    """

    recovery_kind: str
    arm_family: str
    target_digests: tuple[str, ...]
    profile_digest: str | None = None
    _targets: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.recovery_kind.strip() or not self.arm_family.strip():
            raise ValueError("recovery kind and arm family must be non-empty")
        if not self.target_digests:
            raise ValueError("hardware recovery requires at least one target")
        for digest in self.target_digests:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("hardware recovery target must be a SHA-256 digest")
        if self.profile_digest is not None:
            normalized = self.profile_digest.removeprefix("sha256:")
            if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
                raise ValueError("hardware recovery profile must be a SHA-256 digest")
            object.__setattr__(self, "profile_digest", normalized)

    @classmethod
    def from_targets(
        cls,
        recovery_kind: str,
        arm_family: str,
        *targets: str,
        profile_digest: str | None = None,
    ) -> HardwareRecoveryIdentity:
        clean = tuple(
            sorted({target.strip() for target in targets if isinstance(target, str) and target.strip()})
        )
        if not clean:
            raise ValueError("hardware recovery targets must be non-empty")
        digests = tuple(
            sorted(
                hashlib.sha256(
                    f"makermodslab-hardware-target-v1\0{arm_family}\0{target}".encode()
                ).hexdigest()
                for target in clean
            )
        )
        return cls(recovery_kind, arm_family, digests, profile_digest, clean)

    @classmethod
    def from_bound_targets(
        cls,
        recovery_kind: str,
        arm_family: str,
        targets: Mapping[str, str],
        *,
        profile_digest: str | None = None,
    ) -> HardwareRecoveryIdentity:
        """Bind private paths to stable, non-public hardware identifiers.

        ``targets`` maps each raw local device path to a stable identifier
        resolved without opening the arm (for example a USB VID/PID/serial
        tuple). Only domain-separated hashes of those identifiers enter the
        public journal; the paths remain in the separate owner-private target
        map. Duplicate stable identifiers are rejected because they cannot
        distinguish two simultaneously claimed devices.
        """
        clean: dict[str, str] = {}
        for raw_target, raw_binding in targets.items():
            if not isinstance(raw_target, str) or not isinstance(raw_binding, str):
                raise ValueError("hardware recovery targets and bindings must be strings")
            target = raw_target.strip()
            binding = raw_binding.strip()
            if not target or not binding:
                raise ValueError("hardware recovery targets and bindings must be non-empty")
            if target in clean:
                raise ValueError("hardware recovery targets must be unique")
            clean[target] = binding
        if not clean:
            raise ValueError("hardware recovery requires at least one bound target")
        if len(set(clean.values())) != len(clean):
            raise ValueError("hardware recovery bindings must be unique")
        digests = tuple(
            sorted(
                hashlib.sha256(
                    f"makermodslab-hardware-binding-v1\0{arm_family}\0{binding}".encode()
                ).hexdigest()
                for binding in clean.values()
            )
        )
        return cls(
            recovery_kind,
            arm_family,
            digests,
            profile_digest,
            tuple(sorted(clean)),
        )

    @classmethod
    def from_public(cls, value: Mapping[str, Any]) -> HardwareRecoveryIdentity:
        expected = {"recovery_kind", "arm_family", "target_digests", "profile_digest"}
        if set(value) != expected or not isinstance(value.get("target_digests"), list):
            raise ValueError("hardware recovery identity has an unsupported shape")
        recovery_kind = value.get("recovery_kind")
        arm_family = value.get("arm_family")
        target_digests = value.get("target_digests")
        profile_digest = value.get("profile_digest")
        if (
            not isinstance(recovery_kind, str)
            or not isinstance(arm_family, str)
            or not isinstance(target_digests, list)
            or not all(isinstance(item, str) for item in target_digests)
            or (profile_digest is not None and not isinstance(profile_digest, str))
        ):
            raise ValueError("hardware recovery identity has invalid field types")
        return cls(
            recovery_kind=recovery_kind,
            arm_family=arm_family,
            target_digests=tuple(target_digests),
            profile_digest=profile_digest,
        )

    def public(self) -> dict[str, object]:
        return {
            "recovery_kind": self.recovery_kind,
            "arm_family": self.arm_family,
            "target_digests": list(self.target_digests),
            "profile_digest": self.profile_digest,
        }

    def private_targets(self) -> tuple[str, ...]:
        """Raw local targets for the owner-private recovery map only."""
        return self._targets

    def attach_private_targets(self, targets: tuple[str, ...]) -> HardwareRecoveryIdentity:
        """Attach owner-private paths already bound by the target-map record.

        The caller must first verify that the target map carries this exact
        public recovery identity. Bound identities intentionally cannot be
        recomputed from a path alone; the recovery handler resolves the live
        hardware binding again before it may adopt the lease or open a bus.
        """
        clean = tuple(
            sorted({target.strip() for target in targets if isinstance(target, str) and target.strip()})
        )
        if not clean or len(clean) != len(targets):
            raise ValueError("hardware recovery target map contains invalid targets")
        return HardwareRecoveryIdentity(
            self.recovery_kind,
            self.arm_family,
            self.target_digests,
            self.profile_digest,
            clean,
        )


@dataclass(frozen=True, slots=True)
class StopClaim:
    accepted: bool
    first_request: bool
    reason: str
    requested_at_monotonic: float


@dataclass(frozen=True, slots=True)
class HardwareReleaseReceipt:
    """Evidence supplied by the worker that actually closed the hardware.

    ``safe`` is the finalizer's explicit conclusion.  ``device_closed`` must
    be true because releasing while a worker still owns an open bus recreates
    the race this registry exists to prevent.  ``torque_off`` may be ``None``
    only for a read-only/encoder-only operation that declares
    ``torque_not_applicable`` in its evidence.
    """

    safe: bool
    device_closed: bool
    torque_off: bool | None = None
    evidence: str = ""
    torque_not_applicable: bool = False

    @classmethod
    def safe_close(
        cls,
        *,
        torque_off: bool | None,
        evidence: str,
        torque_not_applicable: bool = False,
    ) -> HardwareReleaseReceipt:
        return cls(
            safe=True,
            device_closed=True,
            torque_off=torque_off,
            evidence=evidence,
            torque_not_applicable=torque_not_applicable,
        )


@dataclass(frozen=True, slots=True)
class HardwareLeaseSnapshot:
    resource: str
    held: bool
    lease_id: str | None = None
    generation: int = 0
    kind: str | None = None
    owner: str | None = None
    state: str = "idle"
    acquired_at_monotonic: float | None = None
    stop_requested_at_monotonic: float | None = None
    stop_reason: str | None = None
    unresolved_reason: str | None = None
    receipt: dict[str, Any] | None = None
    pending_unresolved: bool = False
    pending_kind: str | None = None
    pending_owner: str | None = None
    recovery: dict[str, object] | None = None
    journal_error: str | None = None


class HardwareLeaseHeld(RuntimeError):  # noqa: N818 - public domain exception name
    def __init__(self, snapshot: HardwareLeaseSnapshot) -> None:
        self.snapshot = snapshot
        owner = snapshot.kind or "unknown"
        state = snapshot.state
        super().__init__(f"arm hardware is held by {owner} ({state})")


class HardwareLeaseTokenError(RuntimeError):
    pass


class HardwareLeaseJournalError(RuntimeError):
    pass


_HARDWARE_JOURNAL_VERSION = 1
_RECOVERY_TARGETS_VERSION = 1


def _default_hardware_journal_path() -> Path:
    override = os.environ.get("MAKERMODSLAB_HARDWARE_LEASE_ROOT")
    if override:
        root = Path(override)
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "MakerModsLab"
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise HardwareLeaseJournalError("LOCALAPPDATA is required for hardware safety state")
        root = Path(local) / "MakerModsLab"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "MakerModsLab"
    return root / "hardware-lease.json"


class _HardwareLeaseJournal:
    """Owner-private, atomic record of a claim that lacks safe-close evidence."""

    def __init__(self, path: Path) -> None:
        # Resolve existing ancestor aliases once, before any state transition.
        # All later operations use this canonical parent plus a validated
        # directory descriptor, so retargeting an override symlink cannot move
        # the journal. ``strict=False`` preserves a not-yet-created leaf.
        canonical_parent = path.parent.resolve(strict=False)
        self.path = canonical_parent / path.name
        self.targets_path = self.path.with_name(f"{self.path.stem}-targets.json")
        self.target_error: str | None = None

    @staticmethod
    def _validate_parent_ancestry(parent: Path) -> None:
        """Reject symlinked or replaceable POSIX ancestors."""
        if os.name == "nt":
            return
        absolute = parent.absolute()
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            try:
                info = current.lstat()
            except OSError as exc:
                raise HardwareLeaseJournalError("hardware safety directory ancestry is unavailable") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise HardwareLeaseJournalError(
                    "hardware safety directory ancestry must not contain symbolic links"
                )
            # A sticky temporary directory permits the owner to protect its
            # own private child. Any other group/world-writable ancestor can
            # redirect the path before the final O_NOFOLLOW open.
            if stat.S_IMODE(info.st_mode) & 0o022 and not (info.st_mode & stat.S_ISVTX):
                raise HardwareLeaseJournalError("hardware safety directory ancestry is not owner-controlled")

    def _open_private_parent(self, *, create: bool) -> int | None:
        parent = self.path.parent
        try:
            if create:
                parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if os.name != "nt":
                    current = parent.lstat()
                    if (
                        stat.S_ISLNK(current.st_mode)
                        or not stat.S_ISDIR(current.st_mode)
                        or (hasattr(os, "getuid") and current.st_uid != os.getuid())
                    ):
                        raise HardwareLeaseJournalError("hardware safety directory must be owner-controlled")
                    parent.chmod(0o700)
            elif not parent.exists():
                return None
            self._validate_parent_ancestry(parent)
            flags = os.O_RDONLY
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(parent, flags)
        except OSError as exc:
            raise HardwareLeaseJournalError("hardware safety directory could not be opened securely") from exc
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            os.close(descriptor)
            raise HardwareLeaseJournalError("hardware safety directory is invalid")
        if os.name != "nt" and (
            stat.S_IMODE(info.st_mode) & 0o077 or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            os.close(descriptor)
            raise HardwareLeaseJournalError("hardware safety directory must be owner-only")
        return descriptor

    def _read_private_json(self, path: Path, label: str) -> dict[str, Any] | None:
        parent_descriptor = self._open_private_parent(create=False)
        if parent_descriptor is None:
            return None
        descriptor = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                if os.name == "nt":
                    descriptor = os.open(path, flags)
                else:
                    descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
            except FileNotFoundError:
                return None
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise HardwareLeaseJournalError(f"{label} must be a regular file")
            if os.name != "nt" and (
                stat.S_IMODE(info.st_mode) & 0o077 or (hasattr(os, "getuid") and info.st_uid != os.getuid())
            ):
                raise HardwareLeaseJournalError(f"{label} must be owner-only")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
                descriptor = -1
                body = json.load(handle)
        except HardwareLeaseJournalError:
            raise
        except (OSError, ValueError) as exc:
            raise HardwareLeaseJournalError(f"{label} is unreadable") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_descriptor)
        if not isinstance(body, dict):
            raise HardwareLeaseJournalError(f"{label} has an unsupported shape")
        return body

    def load(self) -> tuple[dict[str, Any], HardwareRecoveryIdentity] | None:
        body = self._read_private_json(self.path, "hardware safety journal")
        if body is None:
            return None
        expected = {"version", "state", "source_kind", "recovery", "receipt"}
        if set(body) != expected:
            raise HardwareLeaseJournalError("hardware safety journal has an unsupported shape")
        if body.get("version") != _HARDWARE_JOURNAL_VERSION:
            raise HardwareLeaseJournalError("hardware safety journal version is unsupported")
        if body.get("state") not in {"active", "unresolved"}:
            raise HardwareLeaseJournalError("hardware safety journal state is invalid")
        if not isinstance(body.get("source_kind"), str) or not body["source_kind"].strip():
            raise HardwareLeaseJournalError("hardware safety journal source is invalid")
        if not isinstance(body.get("recovery"), dict):
            raise HardwareLeaseJournalError("hardware safety journal recovery identity is invalid")
        try:
            recovery = HardwareRecoveryIdentity.from_public(body["recovery"])
        except (TypeError, ValueError) as exc:
            raise HardwareLeaseJournalError("hardware safety journal recovery identity is invalid") from exc
        receipt = body.get("receipt")
        if receipt is not None:
            receipt_fields = {"safe", "device_closed", "torque_off", "torque_not_applicable"}
            if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
                raise HardwareLeaseJournalError("hardware safety journal receipt is invalid")
            if (
                not isinstance(receipt["safe"], bool)
                or not isinstance(receipt["device_closed"], bool)
                or not isinstance(receipt["torque_not_applicable"], bool)
                or (receipt["torque_off"] is not None and not isinstance(receipt["torque_off"], bool))
            ):
                raise HardwareLeaseJournalError("hardware safety journal receipt is invalid")
        self.target_error = None
        try:
            target_body = self._read_private_json(
                self.targets_path,
                "hardware recovery target map",
            )
            if target_body is not None:
                if set(target_body) != {"version", "recovery", "targets"}:
                    raise HardwareLeaseJournalError("hardware recovery target map has an unsupported shape")
                targets = target_body.get("targets")
                if (
                    target_body.get("version") != _RECOVERY_TARGETS_VERSION
                    or target_body.get("recovery") != recovery.public()
                    or not isinstance(targets, list)
                    or not targets
                    or not all(isinstance(target, str) and target.strip() for target in targets)
                ):
                    raise HardwareLeaseJournalError("hardware recovery target map is invalid")
                recovery = recovery.attach_private_targets(tuple(targets))
        except (HardwareLeaseJournalError, TypeError, ValueError) as exc:
            # The path-free journal remains authoritative. A missing or
            # damaged convenience map never makes the arm idle; the owner can
            # still supply exact targets to a local recovery endpoint.
            self.target_error = str(exc)
        return body, recovery

    def _atomic_write(self, path: Path, payload: Mapping[str, Any], prefix: str) -> None:
        parent_descriptor = self._open_private_parent(create=True)
        assert parent_descriptor is not None
        descriptor = -1
        temporary: Path | None = None
        temporary_name: str | None = None
        try:
            if os.name == "nt":
                descriptor, name = tempfile.mkstemp(prefix=prefix, dir=self.path.parent)
                temporary = Path(name)
            else:
                temporary_name = f"{prefix}{secrets.token_hex(16)}"
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
                descriptor = -1
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "nt":
                assert temporary is not None
                os.replace(temporary, path)
            else:
                assert temporary_name is not None
                os.replace(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.fsync(parent_descriptor)
        except OSError as exc:
            raise HardwareLeaseJournalError("hardware safety state could not be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()
            if temporary_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
            os.close(parent_descriptor)

    def save(
        self,
        *,
        state: str,
        source_kind: str,
        recovery: HardwareRecoveryIdentity,
        receipt: HardwareReleaseReceipt | None = None,
    ) -> None:
        if state not in {"active", "unresolved"}:
            raise ValueError("only active or unresolved hardware state is durable")
        durable_receipt = None
        if receipt is not None:
            durable_receipt = {
                "safe": receipt.safe,
                "device_closed": receipt.device_closed,
                "torque_off": receipt.torque_off,
                "torque_not_applicable": receipt.torque_not_applicable,
            }
        payload = {
            "version": _HARDWARE_JOURNAL_VERSION,
            "state": state,
            "source_kind": source_kind,
            "recovery": recovery.public(),
            "receipt": durable_receipt,
        }
        targets = recovery.private_targets()
        if targets:
            self._atomic_write(
                self.targets_path,
                {
                    "version": _RECOVERY_TARGETS_VERSION,
                    "recovery": recovery.public(),
                    "targets": list(targets),
                },
                ".hardware-targets-",
            )
        self._atomic_write(self.path, payload, ".hardware-lease-")
        self.target_error = None

    def _unlink_and_sync(self, path: Path, label: str) -> None:
        parent_descriptor = self._open_private_parent(create=False)
        if parent_descriptor is None:
            return
        try:
            try:
                if os.name == "nt":
                    info = path.lstat()
                else:
                    info = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISLNK(info.st_mode):
                raise HardwareLeaseJournalError(f"{label} must not be a symbolic link")
            if os.name == "nt":
                path.unlink()
            else:
                os.unlink(path.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
        except HardwareLeaseJournalError:
            raise
        except OSError as exc:
            raise HardwareLeaseJournalError(f"{label} could not be cleared") from exc
        finally:
            os.close(parent_descriptor)

    def clear(self) -> None:
        # Remove the convenience map first. If the process crashes between
        # unlinks, the authoritative journal remains and restart stays locked.
        self._unlink_and_sync(self.targets_path, "hardware recovery target map")
        self._unlink_and_sync(self.path, "hardware safety journal")
        self.target_error = None


class HardwareLeaseRegistry:
    """One-resource lease registry with optional crash-durable safety intent."""

    def __init__(
        self,
        resource: str = "arm_hardware",
        *,
        journal_path: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._resource = resource
        self._journal = _HardwareLeaseJournal(journal_path) if journal_path is not None else None
        self._journal_error: str | None = None
        self._generation = 0
        self._token: HardwareLeaseToken | None = None
        self._source_kind: str | None = None
        self._recovery: HardwareRecoveryIdentity | None = None
        self._state = "idle"
        self._acquired_at: float | None = None
        self._stop_requested_at: float | None = None
        self._stop_reason: str | None = None
        self._unresolved_reason: str | None = None
        self._receipt: dict[str, Any] | None = None
        self._pending_latch: tuple[str, str, str, HardwareReleaseReceipt, HardwareRecoveryIdentity] | None = (
            None
        )
        self._restore_journal()

    def claim(
        self,
        kind: str,
        owner: str,
        resource: str = "arm_hardware",
        *,
        recovery: HardwareRecoveryIdentity | None = None,
    ) -> HardwareLeaseToken:
        if not kind.strip() or not owner.strip():
            raise ValueError("kind and owner must be non-empty")
        if resource != self._resource:
            raise ValueError(f"unsupported hardware resource: {resource}")
        with self._lock:
            if self._token is not None:
                raise HardwareLeaseHeld(self._snapshot_locked())
            if self._journal_error is not None:
                raise HardwareLeaseJournalError(self._journal_error)
            selected_recovery = recovery or HardwareRecoveryIdentity.from_targets(
                "manual_recovery",
                "unknown",
                f"{kind}:{owner}",
            )
            self._persist_locked("active", kind, selected_recovery)
            self._generation += 1
            token = HardwareLeaseToken(
                lease_id=secrets.token_hex(16),
                generation=self._generation,
                kind=kind,
                owner=owner,
                resource=resource,
            )
            self._token = token
            self._source_kind = kind
            self._recovery = selected_recovery
            self._state = "active"
            self._acquired_at = time.monotonic()
            self._stop_requested_at = None
            self._stop_reason = None
            self._unresolved_reason = None
            self._receipt = None
            return token

    def request_stop(self, token: HardwareLeaseToken, reason: str) -> StopClaim:
        with self._lock:
            self._require_current_locked(token)
            now = time.monotonic()
            first = self._stop_requested_at is None
            if first:
                self._stop_requested_at = now
                self._stop_reason = reason
                if self._state != "unresolved":
                    self._state = "stopping"
            return StopClaim(
                accepted=True,
                first_request=first,
                reason=self._stop_reason or reason,
                requested_at_monotonic=self._stop_requested_at or now,
            )

    def begin_recovery(
        self,
        owner: str,
        *,
        expected: HardwareRecoveryIdentity,
    ) -> HardwareLeaseToken:
        """Authorize the explicit recovery path without silently clearing a latch.

        Recovery may adopt an unresolved token because its whole purpose is to
        gather the safe-close evidence that token lacks.  An active/stopping
        worker is never adoptable.  When idle, recovery takes an ordinary new
        lease before it opens the bus.
        """
        with self._lock:
            if self._token is None:
                return self.claim(
                    expected.recovery_kind,
                    owner,
                    self._resource,
                    recovery=expected,
                )
            if self._state != "unresolved":
                raise HardwareLeaseHeld(self._snapshot_locked())
            if self._recovery != expected:
                raise HardwareLeaseHeld(self._snapshot_locked())
            if expected.private_targets() and self._recovery.private_targets() != expected.private_targets():
                durable_receipt = (
                    self._normalize_receipt(self._receipt) if self._receipt is not None else None
                )
                # Repair a missing/stale private target map before hardware is
                # opened. The public identity already matched exactly; for a
                # bound identity the recovery handler has also re-resolved the
                # live stable device identifier. A failed recovery will now
                # retain these newly proved paths across another restart.
                self._persist_locked(
                    "unresolved",
                    self._source_kind or expected.recovery_kind,
                    expected,
                    durable_receipt,
                )
                self._recovery = expected
            self._state = "recovering"
            return self._token

    def install_unresolved_latch(
        self,
        *,
        kind: str,
        owner: str,
        reason: str,
        receipt: HardwareReleaseReceipt | Mapping[str, Any],
        recovery: HardwareRecoveryIdentity,
    ) -> HardwareLeaseToken | None:
        """Install a durable latch now or atomically behind the current owner.

        A journal may be discovered while an earlier feature still owns the
        registry. Merely retrying later creates a release-to-claim gap. This
        method queues the latch under the same lock, so a safe release promotes
        it directly to ``unresolved`` before any claimant can observe idle.
        """
        if not kind.strip() or not owner.strip() or not reason.strip():
            raise ValueError("durable latch identity and reason must be non-empty")
        normalized = self._normalize_receipt(receipt)
        with self._lock:
            if self._token is not None and self._state in {"unresolved", "recovering"}:
                if self._recovery == recovery:
                    return self._token
                raise HardwareLeaseHeld(self._snapshot_locked())
            if self._token is not None and (
                self._token.kind == kind and self._token.owner == owner and self._recovery == recovery
            ):
                return self._token
            pending = (kind, owner, reason, normalized, recovery)
            if self._pending_latch is not None and self._pending_latch != pending:
                raise HardwareLeaseHeld(self._snapshot_locked())
            if self._token is not None:
                self._pending_latch = pending
                return None
            return self._activate_pending_locked(pending)

    def release(
        self,
        token: HardwareLeaseToken,
        receipt: HardwareReleaseReceipt | Mapping[str, Any],
    ) -> None:
        normalized = self._normalize_receipt(receipt)
        with self._lock:
            self._require_current_locked(token)
            self._receipt = asdict(normalized)
            confirmed_torque = normalized.torque_off is True or (
                normalized.torque_off is None and normalized.torque_not_applicable
            )
            if not normalized.safe or not normalized.device_closed or not confirmed_torque:
                self._state = "unresolved"
                self._unresolved_reason = normalized.evidence or "safe hardware close was not confirmed"
                self._persist_locked(
                    "unresolved",
                    self._source_kind or token.kind,
                    self._require_recovery_locked(),
                    normalized,
                )
                return
            if self._pending_latch is not None:
                pending = self._pending_latch
                # ``_activate_pending_locked`` atomically replaces the current
                # intent on disk. Clearing first would create a crash window in
                # which neither claim survived restart.
                self._activate_pending_locked(pending)
                self._pending_latch = None
            else:
                try:
                    self._clear_journal_locked()
                except HardwareLeaseJournalError:
                    self._state = "unresolved"
                    self._unresolved_reason = (
                        "safe-close evidence exists but its durable intent could not clear"
                    )
                    raise
                self._clear_current_locked()

    def mark_unresolved(
        self,
        token: HardwareLeaseToken,
        reason: str,
        receipt: HardwareReleaseReceipt | Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._require_current_locked(token)
            self._state = "unresolved"
            self._unresolved_reason = reason
            normalized = None
            if receipt is not None:
                normalized = self._normalize_receipt(receipt)
                self._receipt = asdict(normalized)
            self._persist_locked(
                "unresolved",
                self._source_kind or token.kind,
                self._require_recovery_locked(),
                normalized,
            )

    def resolve_unresolved(
        self,
        token: HardwareLeaseToken,
        receipt: HardwareReleaseReceipt | Mapping[str, Any],
    ) -> None:
        """Clear a fault latch only with the same full safe-close evidence."""
        with self._lock:
            self._require_current_locked(token)
            if self._state != "unresolved":
                raise HardwareLeaseTokenError("lease is not unresolved")
        self.release(token, receipt)

    def snapshot(self) -> HardwareLeaseSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def is_token_current(self, token: HardwareLeaseToken | None) -> bool:
        with self._lock:
            return token is not None and self._token == token

    def recovery_targets(self, recovery_kind: str) -> tuple[str, ...]:
        """Return raw targets only to an explicit local recovery handler.

        They never enter snapshots, API status, logs, or the shareable fault
        journal. An empty tuple means the owner must re-enter exact targets.
        """
        with self._lock:
            if (
                self._token is None
                or self._state not in {"unresolved", "recovering"}
                or self._recovery is None
                or self._recovery.recovery_kind != recovery_kind
            ):
                return ()
            return self._recovery.private_targets()

    def _reset_for_tests(self) -> None:
        """Test isolation only; production code must resolve retained faults."""
        with self._lock:
            self._clear_locked()
            self._journal_error = None
            if self._journal is not None:
                self._journal.clear()

    def _require_current_locked(self, token: HardwareLeaseToken) -> None:
        if self._token != token:
            raise HardwareLeaseTokenError("stale or foreign hardware lease token")

    def _snapshot_locked(self) -> HardwareLeaseSnapshot:
        token = self._token
        return HardwareLeaseSnapshot(
            resource=self._resource,
            held=token is not None,
            lease_id=token.lease_id if token else None,
            generation=token.generation if token else self._generation,
            kind=token.kind if token else None,
            owner=token.owner if token else None,
            state=self._state,
            acquired_at_monotonic=self._acquired_at,
            stop_requested_at_monotonic=self._stop_requested_at,
            stop_reason=self._stop_reason,
            unresolved_reason=self._unresolved_reason,
            receipt=dict(self._receipt) if self._receipt is not None else None,
            pending_unresolved=self._pending_latch is not None,
            pending_kind=(self._pending_latch[0] if self._pending_latch is not None else None),
            pending_owner=(self._pending_latch[1] if self._pending_latch is not None else None),
            recovery=(self._recovery.public() if self._recovery is not None else None),
            journal_error=self._journal_error,
        )

    def _clear_current_locked(self) -> None:
        self._token = None
        self._source_kind = None
        self._recovery = None
        self._state = "idle"
        self._acquired_at = None
        self._stop_requested_at = None
        self._stop_reason = None
        self._unresolved_reason = None
        self._receipt = None

    def _clear_locked(self) -> None:
        self._clear_current_locked()
        self._pending_latch = None

    def _activate_pending_locked(
        self,
        pending: tuple[str, str, str, HardwareReleaseReceipt, HardwareRecoveryIdentity],
    ) -> HardwareLeaseToken:
        kind, owner, reason, receipt, recovery = pending
        self._persist_locked("unresolved", kind, recovery, receipt)
        self._generation += 1
        token = HardwareLeaseToken(
            lease_id=secrets.token_hex(16),
            generation=self._generation,
            kind=kind,
            owner=owner,
            resource=self._resource,
        )
        self._token = token
        self._source_kind = kind
        self._recovery = recovery
        self._state = "unresolved"
        self._acquired_at = time.monotonic()
        self._stop_requested_at = None
        self._stop_reason = None
        self._unresolved_reason = reason
        self._receipt = asdict(receipt)
        return token

    def _require_recovery_locked(self) -> HardwareRecoveryIdentity:
        recovery = self._recovery
        if recovery is None:
            raise HardwareLeaseJournalError("hardware recovery identity is unavailable")
        return recovery

    def _persist_locked(
        self,
        state: str,
        source_kind: str,
        recovery: HardwareRecoveryIdentity,
        receipt: HardwareReleaseReceipt | None = None,
    ) -> None:
        journal = self._journal
        if journal is None:
            return
        try:
            journal.save(
                state=state,
                source_kind=source_kind,
                recovery=recovery,
                receipt=receipt,
            )
        except HardwareLeaseJournalError as exc:
            self._journal_error = str(exc)
            raise
        self._journal_error = None

    def _clear_journal_locked(self) -> None:
        journal = self._journal
        if journal is None:
            return
        try:
            journal.clear()
        except HardwareLeaseJournalError as exc:
            self._journal_error = str(exc)
            raise
        self._journal_error = None

    def _restore_journal(self) -> None:
        journal = self._journal
        if journal is None:
            return
        try:
            record = journal.load()
        except HardwareLeaseJournalError as exc:
            self._journal_error = str(exc)
            self._generation += 1
            self._token = HardwareLeaseToken(
                lease_id=secrets.token_hex(16),
                generation=self._generation,
                kind="hardware_recovery",
                owner="durable:invalid-hardware-journal",
                resource=self._resource,
            )
            self._source_kind = "unknown"
            self._state = "unresolved"
            self._acquired_at = time.monotonic()
            self._unresolved_reason = "invalid durable hardware safety journal requires repair"
            return
        if record is None:
            return
        record, recovery = record
        if journal.target_error is not None:
            self._journal_error = journal.target_error
        durable_receipt = record.get("receipt")
        receipt = None
        if isinstance(durable_receipt, dict):
            receipt = HardwareReleaseReceipt(evidence="durable hardware safety journal", **durable_receipt)
        self._generation += 1
        self._token = HardwareLeaseToken(
            lease_id=secrets.token_hex(16),
            generation=self._generation,
            kind=recovery.recovery_kind,
            owner="durable:hardware-fault",
            resource=self._resource,
        )
        self._source_kind = record["source_kind"]
        self._recovery = recovery
        self._state = "unresolved"
        self._acquired_at = time.monotonic()
        self._unresolved_reason = (
            "prior process exited without terminal safe-close evidence"
            if record["state"] == "active"
            else "durable hardware fault requires compatible local recovery"
        )
        self._receipt = asdict(receipt) if receipt is not None else None

    @staticmethod
    def _normalize_receipt(
        receipt: HardwareReleaseReceipt | Mapping[str, Any],
    ) -> HardwareReleaseReceipt:
        if isinstance(receipt, HardwareReleaseReceipt):
            return receipt
        return HardwareReleaseReceipt(**dict(receipt))


hardware_lease_registry = HardwareLeaseRegistry(journal_path=_default_hardware_journal_path())


_BUSY_CODES: dict[str, ErrorCode] = {
    "teleoperation": ErrorCode.ROBOT_BUSY_TELEOPERATION,
    "remote_teleoperation": ErrorCode.ROBOT_BUSY_TELEOPERATION,
    "recording": ErrorCode.ROBOT_BUSY_RECORDING,
    "inference": ErrorCode.ROBOT_BUSY_INFERENCE,
    "replay": ErrorCode.ROBOT_BUSY_REPLAY,
    "calibration": ErrorCode.ROBOT_BUSY_CALIBRATION,
    "auto_calibration": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
    "wiggle": ErrorCode.ROBOT_BUSY_WIGGLE,
}


def held_response(exc: HardwareLeaseHeld, *, include_status: bool = False) -> dict[str, Any]:
    """Compatibility response for feature handlers that historically return dicts."""
    snapshot = exc.snapshot
    holder = snapshot.kind or "another operation"
    response: dict[str, Any] = {
        "success": False,
        "message": f"The arm hardware is held by {holder}. Stop it and wait for safe release first.",
        "code": _BUSY_CODES.get(holder, ErrorCode.SESSION_HELD),
        "details": {
            "holder": {
                "kind": holder,
                "owner": snapshot.owner,
                "state": snapshot.state,
                "generation": snapshot.generation,
            }
        },
    }
    if include_status:
        response["status_code"] = 409
    return response


def safe_hardware_receipt(
    evidence: str,
    *,
    torque_off: bool | None = True,
    torque_not_applicable: bool = False,
) -> HardwareReleaseReceipt:
    """Concise finalizer helper used by the migrated feature modules."""
    return HardwareReleaseReceipt.safe_close(
        torque_off=torque_off,
        evidence=evidence,
        torque_not_applicable=torque_not_applicable,
    )
