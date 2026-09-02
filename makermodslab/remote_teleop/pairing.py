"""One-time enrollment and owner-private credential persistence."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import ssl
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


class PairingError(RuntimeError):
    """Pairing or credential verification failed without disclosing a secret."""


def private_application_data_dir() -> Path:
    """Return the user-private application data root without creating it."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise PairingError("LOCALAPPDATA is required for credential storage")
        base = Path(local)
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "MakerModsLab" / "remote_teleop"


def _token_urlsafe(size: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(size)).rstrip(b"=").decode("ascii")


def _secret_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not 40 <= len(value) <= 128:
        raise PairingError("credential material is malformed")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise PairingError("credential material is malformed") from exc
    if len(decoded) != 32:
        raise PairingError("credential material is malformed")
    return decoded


def _fingerprint_bytes(certificate_der: bytes) -> str:
    if not isinstance(certificate_der, bytes) or not certificate_der:
        raise PairingError("TLS certificate is empty")
    return "sha256:" + hashlib.sha256(certificate_der).hexdigest()


def certificate_sha256_fingerprint(certificate: bytes | str | Path) -> str:
    """Hash a DER certificate or a PEM certificate/path in display-safe form."""
    if isinstance(certificate, Path):
        raw = certificate.read_bytes()
    elif isinstance(certificate, str):
        if "-----BEGIN CERTIFICATE-----" in certificate:
            raw = certificate.encode("ascii")
        else:
            path = Path(certificate)
            raw = path.read_bytes()
    elif isinstance(certificate, bytes):
        raw = certificate
    else:
        raise PairingError("TLS certificate input is invalid")
    if b"-----BEGIN CERTIFICATE-----" in raw:
        try:
            der = ssl.PEM_cert_to_DER_cert(raw.decode("ascii"))
            raw = der if isinstance(der, bytes) else der.encode("latin1")
        except (UnicodeDecodeError, ValueError) as exc:
            raise PairingError("TLS certificate PEM is invalid") from exc
    return _fingerprint_bytes(raw)


def normalize_fingerprint(value: str) -> str:
    if not isinstance(value, str):
        raise PairingError("TLS certificate fingerprint is invalid")
    compact = value.lower().replace(":", "")
    if compact.startswith("sha256"):
        compact = compact[6:]
    if len(compact) != 64 or any(char not in "0123456789abcdef" for char in compact):
        raise PairingError("TLS certificate fingerprint is invalid")
    return "sha256:" + compact


def verify_certificate_fingerprint(certificate_der: bytes, expected: str) -> None:
    actual = certificate_sha256_fingerprint(certificate_der)
    if not hmac.compare_digest(actual, normalize_fingerprint(expected)):
        raise PairingError("TLS certificate fingerprint mismatch")


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise PairingError("credential directory must not be a symbolic link")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        path.chmod(0o700)
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PairingError("credential directory is not owner-only")


def _read_private_json(path: Path) -> dict[str, object]:
    parent = path.parent
    if parent.exists():
        if parent.is_symlink():
            raise PairingError("credential directory must not be a symbolic link")
        if os.name != "nt" and stat.S_IMODE(parent.stat().st_mode) & 0o077:
            raise PairingError("credential directory is not owner-only")
    if not path.exists():
        return {"version": 1, "entries": {}}
    if path.is_symlink():
        raise PairingError("credential file must not be a symbolic link")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PairingError("credential file is not owner-only")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PairingError("credential file is unreadable") from exc
    if not isinstance(body, dict) or set(body) != {"version", "entries"}:
        raise PairingError("credential file has an unsupported shape")
    if body["version"] != 1 or not isinstance(body["entries"], dict):
        raise PairingError("credential file has an unsupported version")
    return body


def _write_private_json(path: Path, body: dict[str, object]) -> None:
    _ensure_private_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _credential_digest(secret: bytes, salt: bytes) -> str:
    return hashlib.sha256(b"makermodslab.remote-teleop.credential.v1\x00" + salt + secret).hexdigest()


@dataclass(frozen=True)
class IssuedCredential:
    credential_id: str
    secret: str = field(repr=False)
    operator_label: str

    def public(self) -> dict[str, str]:
        return {"credential_id": self.credential_id, "operator_label": self.operator_label}


class RobotCredentialStore:
    """Robot-side hash-only credential registry with explicit revocation."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or private_application_data_dir()
        self.path = self.root / "robot-credentials.json"
        self._lock = threading.RLock()

    def issue(self, operator_label: str, *, now_ns: int) -> IssuedCredential:
        label = operator_label.strip() if isinstance(operator_label, str) else ""
        if not label or len(label.encode()) > 128:
            raise PairingError("operator label must be a short non-empty string")
        credential_id = "operator-" + secrets.token_hex(16)
        secret = _token_urlsafe()
        salt = secrets.token_bytes(16)
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entries[credential_id] = {
                "operator_label": label,
                "salt": salt.hex(),
                "digest": _credential_digest(_secret_bytes(secret), salt),
                "created_monotonic_ns": now_ns,
                "revoked": False,
            }
            _write_private_json(self.path, body)
        return IssuedCredential(credential_id, secret, label)

    def authenticate(self, credential_id: str, secret: str) -> bool:
        try:
            decoded = _secret_bytes(secret)
        except PairingError:
            return False
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entry = entries.get(credential_id)
            if not isinstance(entry, dict) or entry.get("revoked") is not False:
                return False
            try:
                salt = bytes.fromhex(entry["salt"])
                expected = entry["digest"]
            except (KeyError, TypeError, ValueError):
                raise PairingError("credential file contains malformed material") from None
            return isinstance(expected, str) and hmac.compare_digest(
                expected, _credential_digest(decoded, salt)
            )

    def is_active(self, credential_id: str) -> bool:
        """Return whether an issued credential still has action authority.

        Established TLS connections use this hash-independent check before
        every authenticated request.  The same check is also safe to call at
        the robot service's final session-publication boundary.
        """
        if not isinstance(credential_id, str) or not credential_id:
            return False
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entry = entries.get(credential_id)
            return isinstance(entry, dict) and entry.get("revoked") is False

    @contextlib.contextmanager
    def active_guard(self, credential_id: str) -> Iterator[bool]:
        """Hold revocation ordering stable across one authority publication."""
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entry = entries.get(credential_id)
            yield isinstance(entry, dict) and entry.get("revoked") is False

    def revoke(self, credential_id: str) -> bool:
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entry = entries.get(credential_id)
            if not isinstance(entry, dict):
                return False
            entry["revoked"] = True
            _write_private_json(self.path, body)
            return True

    def public_entries(self) -> list[dict[str, object]]:
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            return [
                {
                    "credential_id": credential_id,
                    "operator_label": entry.get("operator_label"),
                    "revoked": entry.get("revoked"),
                }
                for credential_id, entry in sorted(entries.items())
                if isinstance(entry, dict)
            ]


class OperatorCredentialVault:
    """Operator-side raw credentials; never use this store on the robot host."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or private_application_data_dir()
        self.path = self.root / "operator-credentials.json"
        self._lock = threading.RLock()

    def put(self, robot_id: str, credential: IssuedCredential) -> None:
        if not robot_id or len(robot_id.encode()) > 128:
            raise PairingError("robot id is invalid")
        _secret_bytes(credential.secret)
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entries[robot_id] = {
                "credential_id": credential.credential_id,
                "secret": credential.secret,
                "operator_label": credential.operator_label,
            }
            _write_private_json(self.path, body)

    def get(self, robot_id: str) -> IssuedCredential | None:
        with self._lock:
            body = _read_private_json(self.path)
            entries = body["entries"]
            assert isinstance(entries, dict)
            entry = entries.get(robot_id)
            if not isinstance(entry, dict):
                return None
            try:
                return IssuedCredential(
                    credential_id=entry["credential_id"],
                    secret=entry["secret"],
                    operator_label=entry["operator_label"],
                )
            except (KeyError, TypeError):
                raise PairingError("operator credential vault contains malformed material") from None


@dataclass(frozen=True)
class PairingPayload:
    robot_address: str
    control_port: int
    certificate_fingerprint: str
    pairing_token: str = field(repr=False)
    expires_monotonic_ns: int

    def manual(self) -> dict[str, object]:
        return {
            "protocol": "makermodslab.remote-pairing.v1",
            "robot_address": self.robot_address,
            "control_port": self.control_port,
            "certificate_fingerprint": self.certificate_fingerprint,
            "pairing_token": self.pairing_token,
            "expires_monotonic_ns": self.expires_monotonic_ns,
        }


class PairingAuthority:
    """One short-lived, single-use window opened only by a robot-local action."""

    def __init__(
        self,
        credentials: RobotCredentialStore,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        window_ns: int = 120_000_000_000,
        max_attempts: int = 5,
    ) -> None:
        if not 10_000_000_000 <= window_ns <= 600_000_000_000 or not 1 <= max_attempts <= 20:
            raise ValueError("pairing window bounds are invalid")
        self.credentials = credentials
        self.clock_ns = clock_ns
        self.window_ns = window_ns
        self.max_attempts = max_attempts
        self._lock = threading.RLock()
        self._token_digest: bytes | None = None
        self._expires_ns = 0
        self._attempts = 0

    def open_local_window(
        self,
        *,
        local_request: bool,
        robot_address: str,
        control_port: int,
        certificate_fingerprint: str,
    ) -> PairingPayload:
        if local_request is not True:
            raise PairingError("pairing can only be opened from the robot host")
        if not robot_address or not 1 <= control_port <= 65535:
            raise PairingError("pairing endpoint is invalid")
        fingerprint = normalize_fingerprint(certificate_fingerprint)
        now = self.clock_ns()
        with self._lock:
            if self._token_digest is not None and now < self._expires_ns:
                raise PairingError("pairing window is already open")
            token = _token_urlsafe()
            self._token_digest = hashlib.sha256(token.encode("ascii")).digest()
            self._expires_ns = now + self.window_ns
            self._attempts = 0
        return PairingPayload(robot_address, control_port, fingerprint, token, self._expires_ns)

    def is_open(self) -> bool:
        with self._lock:
            return self._token_digest is not None and self.clock_ns() < self._expires_ns

    def exchange(self, pairing_token: str, operator_label: str) -> IssuedCredential:
        now = self.clock_ns()
        presented = hashlib.sha256(
            pairing_token.encode("ascii", errors="ignore") if isinstance(pairing_token, str) else b""
        ).digest()
        with self._lock:
            expected = self._token_digest
            if expected is None or now >= self._expires_ns:
                self._token_digest = None
                raise PairingError("pairing window is closed")
            self._attempts += 1
            if self._attempts > self.max_attempts:
                self._token_digest = None
                raise PairingError("pairing attempt limit exceeded")
            if not hmac.compare_digest(presented, expected):
                raise PairingError("pairing token is invalid")
            # Consume before touching disk so an I/O failure cannot make the token reusable.
            self._token_digest = None
        return self.credentials.issue(operator_label, now_ns=now)

    def close(self) -> None:
        with self._lock:
            self._token_digest = None
            self._expires_ns = 0
            self._attempts = 0
