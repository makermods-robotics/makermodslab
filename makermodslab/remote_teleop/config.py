"""Private local configuration for the two remote-teleoperation roles."""

from __future__ import annotations

import contextlib
import json
import os
import re
import ssl
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from .pairing import normalize_fingerprint, private_application_data_dir
from .transport import validate_private_bind_address

CONFIG_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RemoteConfigError(ValueError):
    """Local role configuration is incomplete or permission-unsafe."""


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RemoteConfigError(f"{name} must be a path-free identifier")
    return value


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise RemoteConfigError(f"{name} must be a SHA-256 digest")
    return value


def _port(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise RemoteConfigError(f"{name} must be in [1,65535]")
    return value


def _private_file(path_text: str, name: str, *, private: bool) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RemoteConfigError(f"{name} must be an existing absolute regular file")
    if private and os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RemoteConfigError(f"{name} must be owner-only (chmod 600)")
    return str(path)


@dataclass(frozen=True)
class RobotRoleConfig:
    node_id: str
    robot_name: str
    bind_address: str
    control_port: int
    udp_port: int
    tls_certificate_path: str
    tls_private_key_path: str
    leader_calibration_id: str
    leader_calibration_digest: str
    action_rate_hz: int = 50
    action_watchdog_ms: int = 200
    first_action_deadline_ms: int = 1000
    control_deadline_ms: int = 1000
    browser_deadline_ms: int = 2000
    max_velocity_per_s: float = 60.0
    max_acceleration_per_s2: float = 300.0
    recording_enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, "node_id"))
        if not isinstance(self.robot_name, str) or not self.robot_name.strip():
            raise RemoteConfigError("robot_name is required")
        object.__setattr__(self, "bind_address", str(validate_private_bind_address(self.bind_address)))
        object.__setattr__(self, "control_port", _port(self.control_port, "control_port"))
        object.__setattr__(self, "udp_port", _port(self.udp_port, "udp_port"))
        object.__setattr__(
            self,
            "tls_certificate_path",
            _private_file(self.tls_certificate_path, "TLS certificate", private=False),
        )
        object.__setattr__(
            self,
            "tls_private_key_path",
            _private_file(self.tls_private_key_path, "TLS private key", private=True),
        )
        # Loading the pair proves both files are readable and match without
        # starting a listener.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(self.tls_certificate_path, self.tls_private_key_path)
        except (OSError, ssl.SSLError) as exc:
            raise RemoteConfigError("TLS certificate and private key cannot be loaded") from exc
        object.__setattr__(
            self, "leader_calibration_id", _identifier(self.leader_calibration_id, "leader calibration id")
        )
        object.__setattr__(
            self,
            "leader_calibration_digest",
            _digest(self.leader_calibration_digest, "leader calibration digest"),
        )
        if not 10 <= self.action_rate_hz <= 100:
            raise RemoteConfigError("action_rate_hz must be in [10,100]")
        deadlines = {
            "action_watchdog_ms": (self.action_watchdog_ms, 20, 2000),
            "first_action_deadline_ms": (self.first_action_deadline_ms, 20, 5000),
            "control_deadline_ms": (self.control_deadline_ms, 100, 5000),
            "browser_deadline_ms": (self.browser_deadline_ms, 100, 10000),
        }
        for name, (value, minimum, maximum) in deadlines.items():
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise RemoteConfigError(f"{name} must be in [{minimum},{maximum}]")
        if self.first_action_deadline_ms < self.action_watchdog_ms:
            raise RemoteConfigError("first action deadline cannot be shorter than action watchdog")
        if self.browser_deadline_ms < self.control_deadline_ms:
            raise RemoteConfigError("browser deadline cannot be shorter than control deadline")
        if self.max_velocity_per_s <= 0 or self.max_acceleration_per_s2 <= 0:
            raise RemoteConfigError("velocity and acceleration limits must be positive")

    def public(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("tls_certificate_path")
        body.pop("tls_private_key_path")
        body["tls_certificate_configured"] = True
        body["tls_private_key_configured"] = True
        return body


@dataclass(frozen=True)
class OperatorRoleConfig:
    node_id: str
    robot_id: str
    leader_robot_name: str
    control_uri: str
    certificate_fingerprint: str
    action_rate_hz: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, "node_id"))
        object.__setattr__(self, "robot_id", _identifier(self.robot_id, "robot_id"))
        if not isinstance(self.leader_robot_name, str) or not self.leader_robot_name.strip():
            raise RemoteConfigError("leader_robot_name is required")
        parsed = urlparse(self.control_uri)
        if parsed.scheme != "wss" or not parsed.hostname or parsed.path not in ("", "/"):
            raise RemoteConfigError("control_uri must be wss://host:port without a path")
        try:
            address = validate_private_bind_address(parsed.hostname)
        except Exception as exc:
            raise RemoteConfigError("control_uri host must be an exact private IP address") from exc
        port = parsed.port
        if port is None:
            raise RemoteConfigError("control_uri must include the robot control port")
        host = f"[{address}]" if address.version == 6 else str(address)
        object.__setattr__(self, "control_uri", f"wss://{host}:{_port(port, 'control port')}")
        object.__setattr__(
            self,
            "certificate_fingerprint",
            normalize_fingerprint(self.certificate_fingerprint),
        )
        if not 10 <= self.action_rate_hz <= 100:
            raise RemoteConfigError("action_rate_hz must be in [10,100]")

    def public(self) -> dict[str, object]:
        return asdict(self)


class RemoteRoleConfigStore:
    """Atomic owner-private persistence; runtime enablement is never stored."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or private_application_data_dir()
        self.path = self.root / "role-config.json"

    def load(self) -> tuple[str, RobotRoleConfig | OperatorRoleConfig] | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise RemoteConfigError("remote role configuration must not be a symlink")
        if os.name != "nt" and stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise RemoteConfigError("remote role configuration must be owner-only")
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RemoteConfigError("remote role configuration is unreadable") from exc
        if not isinstance(body, dict) or set(body) != {"version", "role", "config"}:
            raise RemoteConfigError("remote role configuration has an unsupported shape")
        if body["version"] != CONFIG_VERSION or not isinstance(body["config"], dict):
            raise RemoteConfigError("remote role configuration has an unsupported version")
        if body["role"] == "robot":
            return "robot", RobotRoleConfig(**body["config"])
        if body["role"] == "operator":
            return "operator", OperatorRoleConfig(**body["config"])
        raise RemoteConfigError("remote role configuration has an unsupported role")

    def save_robot(self, config: RobotRoleConfig) -> None:
        self._save("robot", asdict(config))

    def save_operator(self, config: OperatorRoleConfig) -> None:
        self._save("operator", asdict(config))

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def public(self) -> dict[str, object]:
        loaded = self.load()
        if loaded is None:
            return {"configured": False, "role": None, "config": None, "runtime_enabled": False}
        role, config = loaded
        return {
            "configured": True,
            "role": role,
            "config": config.public(),
            # Deliberately hard-coded: persisted config can never arm or listen.
            "runtime_enabled": False,
        }

    def _save(self, role: str, config: dict[str, object]) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.root.chmod(0o700)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".role-config-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            payload = json.dumps(
                {"version": CONFIG_VERSION, "role": role, "config": config},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
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
