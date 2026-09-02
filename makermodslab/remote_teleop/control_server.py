"""Explicit-start robot TLS WebSocket control service and state machine."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .authority import SessionGrant
from .clock_sync import (
    ClockDriftMonitor,
    ClockSample,
    ClockSyncError,
    FrozenClockMapping,
    select_clock_mapping,
)
from .contracts import SessionSpec
from .control_protocol import (
    MAX_CONTROL_FRAME_BYTES,
    ControlMessage,
    ControlProtocolError,
    decode_control_message,
    encode_control_message,
    make_error,
)
from .pairing import PairingAuthority, RobotCredentialStore
from .transport import validate_private_bind_address


class ControlStateError(RuntimeError):
    """A valid frame arrived in a state where it cannot have authority."""


@dataclass(frozen=True)
class SessionOpenResult:
    grant: SessionGrant
    udp_host: str
    udp_port: int


@dataclass(frozen=True)
class RobotSessionProfile:
    """Robot-authoritative public identities needed to construct a session request."""

    expected_spec: SessionSpec
    limits_digest: str

    def __post_init__(self) -> None:
        digest = self.limits_digest.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("limits_digest must be a SHA-256 digest")

    def public(self) -> dict[str, object]:
        spec = self.expected_spec
        return {
            "source_id": spec.source_id,
            "rig_id": spec.rig_id,
            "rig_digest": spec.rig_digest,
            "follower_calibration_id": spec.follower_calibration_id,
            "follower_calibration_digest": spec.follower_calibration_digest,
            "allowed_leader_calibration_id": spec.leader_calibration_id,
            "allowed_leader_calibration_digest": spec.leader_calibration_digest,
            "joint_names": list(spec.joint_names),
            "units": list(spec.units),
            "limits_digest": self.limits_digest,
        }

    def verify_requested(self, requested: SessionSpec) -> None:
        expected = self.expected_spec
        checks = (
            ("source", requested.source_id == expected.source_id),
            (
                "rig_identity",
                requested.rig_id == expected.rig_id and requested.rig_digest == expected.rig_digest,
            ),
            (
                "leader_calibration",
                requested.leader_calibration_id == expected.leader_calibration_id
                and requested.leader_calibration_digest == expected.leader_calibration_digest,
            ),
            (
                "follower_calibration",
                requested.follower_calibration_id == expected.follower_calibration_id
                and requested.follower_calibration_digest == expected.follower_calibration_digest,
            ),
            (
                "joint_schema",
                requested.joint_names == expected.joint_names and requested.units == expected.units,
            ),
        )
        failed = next((name for name, valid in checks if not valid), None)
        if failed is not None:
            raise ControlStateError(f"session profile mismatch: {failed}")


@dataclass(frozen=True)
class RobotControlCallbacks:
    """Integration boundary; only these callbacks may reach the robot executor."""

    session_profile: Callable[[str], RobotSessionProfile]
    open_session: Callable[[SessionSpec, FrozenClockMapping, str], SessionOpenResult]
    stop_session: Callable[[str, int, str], Mapping[str, object]]
    session_status: Callable[[str, int], Mapping[str, object]]
    heartbeat: Callable[[str, int, bool, bool], Mapping[str, object]]
    udp_probe_status: Callable[[str, int], bool]
    control_lost: Callable[[str, int, str], object]


def _spec_from_payload(value: object) -> SessionSpec:
    expected = {
        "source_id",
        "rig_id",
        "rig_digest",
        "leader_calibration_id",
        "leader_calibration_digest",
        "follower_calibration_id",
        "follower_calibration_digest",
        "joint_names",
        "units",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ControlStateError("session spec has missing or extra fields")
    return SessionSpec(**value)


class RobotControlProtocol:
    """One authenticated connection; robot callbacks mint all session authority."""

    def __init__(
        self,
        *,
        robot_id: str,
        credentials: RobotCredentialStore,
        pairing: PairingAuthority,
        callbacks: RobotControlCallbacks,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.robot_id = robot_id
        self.credentials = credentials
        self.pairing = pairing
        self.callbacks = callbacks
        self.clock_ns = clock_ns
        self._hello_seen = False
        self._credential_id: str | None = None
        self._session: tuple[str, int] | None = None
        self._clock_monitor: ClockDriftMonitor | None = None
        self._loss_reported = False
        self._authorization_invalidated = False

    @property
    def session_active(self) -> bool:
        return self._session is not None

    @property
    def authenticated(self) -> bool:
        return self._credential_id is not None and not self._authorization_invalidated

    @property
    def credential_id(self) -> str | None:
        return self._credential_id

    @property
    def authorization_invalidated(self) -> bool:
        return self._authorization_invalidated

    def _require_authenticated(self) -> str:
        if self._credential_id is None:
            raise ControlStateError("operator credential is not authenticated")
        if self._authorization_invalidated or not self.credentials.is_active(self._credential_id):
            self._authorization_invalidated = True
            raise ControlStateError("operator credential is no longer active")
        return self._credential_id

    def _require_session(self, message: ControlMessage) -> tuple[str, int]:
        current = self._session
        presented = (message.session_id, message.executor_generation)
        if current is None:
            raise ControlStateError("no robot session is active")
        if presented != current:
            raise ControlStateError("stale or foreign robot session authority")
        return current

    async def handle(self, message: ControlMessage) -> ControlMessage:
        """Handle one strict request. Exceptions are converted to redacted errors."""
        was_authenticated = self.authenticated
        try:
            return await self._handle(message)
        except Exception as exc:
            # Precise state errors are useful after authentication. Before it,
            # keep the response uniform so the control socket cannot be used
            # to enumerate robot state. Adapter/storage exceptions can also
            # contain device paths or secret material and are always redacted.
            detail = (
                str(exc)
                if isinstance(exc, ControlStateError) and was_authenticated
                else "request could not be completed"
            )
            return make_error(
                message.request_id,
                code="request_refused",
                detail=detail,
                failed_message_type=message.message_type,
            )

    async def _handle(self, message: ControlMessage) -> ControlMessage:
        kind = message.message_type
        if kind == "hello":
            if self._hello_seen:
                raise ControlStateError("hello was already received")
            self._hello_seen = True
            credential_id = message.payload["credential_id"]
            credential_secret = message.payload["credential_secret"]
            if credential_id is None and credential_secret is None:
                authenticated = False
            elif isinstance(credential_id, str) and isinstance(credential_secret, str):
                authenticated = self.credentials.authenticate(credential_id, credential_secret)
                if authenticated:
                    self._credential_id = credential_id
                    self._authorization_invalidated = False
            else:
                raise ControlStateError("credential id and secret must be supplied together")
            return ControlMessage(
                "hello_ack",
                message.request_id,
                {
                    "robot_id": self.robot_id,
                    "authenticated": authenticated,
                    "pairing_allowed": self.pairing.is_open(),
                },
            )

        if not self._hello_seen:
            raise ControlStateError("hello is required before other control messages")

        if kind == "pair":
            if self._credential_id is not None:
                raise ControlStateError("connection is already authenticated")
            issued = await asyncio.to_thread(
                self.pairing.exchange,
                message.payload["pairing_token"],
                message.payload["operator_label"],
            )
            self._credential_id = issued.credential_id
            self._authorization_invalidated = False
            return ControlMessage(
                "pair_ack",
                message.request_id,
                {
                    "credential_id": issued.credential_id,
                    "credential_secret": issued.secret,
                    "operator_label": issued.operator_label,
                },
            )

        self._require_authenticated()

        if kind == "clock_probe":
            t0 = message.payload["operator_send_ns"]
            if isinstance(t0, bool) or not isinstance(t0, int) or t0 < 0:
                raise ControlStateError("operator clock probe is invalid")
            t1 = self.clock_ns()
            t2 = self.clock_ns()
            return ControlMessage(
                "clock_reply",
                message.request_id,
                {"operator_send_ns": t0, "robot_receive_ns": t1, "robot_send_ns": t2},
            )

        if kind == "profile":
            credential_id = self._require_authenticated()
            profile = await asyncio.to_thread(self.callbacks.session_profile, credential_id)
            return ControlMessage(
                "profile_ack",
                message.request_id,
                {"profile": profile.public()},
            )

        if kind == "session_open":
            if self._session is not None:
                raise ControlStateError("this connection already has an active session")
            credential_id = self._require_authenticated()
            spec = _spec_from_payload(message.payload["spec"])
            profile = await asyncio.to_thread(self.callbacks.session_profile, credential_id)
            profile.verify_requested(spec)
            raw_samples = message.payload["clock_samples"]
            if not isinstance(raw_samples, list) or len(raw_samples) != 16:
                raise ControlStateError("session open requires exactly 16 clock samples")
            samples = [ClockSample.from_dict(value) for value in raw_samples]
            mapping = select_clock_mapping(samples)
            result = await asyncio.to_thread(self.callbacks.open_session, spec, mapping, credential_id)
            # Revocation can occur while the blocking robot adapter starts in
            # the worker thread.  Never publish authority to this connection
            # after that credential has lost authorization.
            try:
                self._require_authenticated()
            except ControlStateError:
                await asyncio.to_thread(
                    self.callbacks.stop_session,
                    result.grant.session_id,
                    result.grant.executor_generation,
                    "operator_credential_revoked",
                )
                raise
            grant = result.grant
            self._session = (grant.session_id, grant.executor_generation)
            self._clock_monitor = ClockDriftMonitor(mapping)
            return ControlMessage(
                "session_grant",
                message.request_id,
                {
                    "session": grant.public(),
                    "action_key_base64": base64.b64encode(grant.action_key).decode("ascii"),
                    "udp_host": result.udp_host,
                    "udp_port": result.udp_port,
                    "clock": mapping.public(),
                },
            )

        if kind == "clock_check":
            session_id, generation = self._require_session(message)
            monitor = self._clock_monitor
            raw_samples = message.payload["clock_samples"]
            if monitor is None or not isinstance(raw_samples, list) or len(raw_samples) != 16:
                raise ControlStateError("clock check requires exactly 16 samples")
            samples = [ClockSample.from_dict(value) for value in raw_samples]
            try:
                frozen = monitor.validate(samples)
                valid = True
            except ClockSyncError:
                await asyncio.to_thread(
                    self.callbacks.stop_session,
                    session_id,
                    generation,
                    "clock_sync_violation",
                )
                frozen = monitor.mapping
                valid = False
                self._session = None
                self._clock_monitor = None
                self._loss_reported = True
            return ControlMessage(
                "clock_check_ack",
                message.request_id,
                {"valid": valid, "clock": frozen.public()},
                session_id,
                generation,
            )

        if kind == "udp_probe_ready":
            session_id, generation = self._require_session(message)
            grant_key_id = message.payload["key_id"]
            if not isinstance(grant_key_id, str):
                raise ControlStateError("UDP action key id is invalid")
            bound = await asyncio.to_thread(self.callbacks.udp_probe_status, session_id, generation)
            return ControlMessage(
                "udp_endpoint_bound",
                message.request_id,
                {"bound": bound},
                session_id,
                generation,
            )

        if kind == "heartbeat":
            session_id, generation = self._require_session(message)
            process_live = message.payload["operator_process_live"]
            browser_live = message.payload["browser_live"]
            if not isinstance(process_live, bool) or not isinstance(browser_live, bool):
                raise ControlStateError("heartbeat liveness fields must be booleans")
            if not process_live or not browser_live:
                reason = "operator_process_lost" if not process_live else "operator_browser_lost"
                await asyncio.to_thread(
                    self.callbacks.stop_session,
                    session_id,
                    generation,
                    reason,
                )
                self._session = None
                self._clock_monitor = None
                self._loss_reported = True
                return ControlMessage(
                    "heartbeat_ack",
                    message.request_id,
                    {"robot_monotonic_ns": self.clock_ns(), "watchdog_remaining_ms": 0},
                    session_id,
                    generation,
                )
            heartbeat = await asyncio.to_thread(
                self.callbacks.heartbeat,
                session_id,
                generation,
                process_live,
                browser_live,
            )
            remaining = heartbeat.get("watchdog_remaining_ms")
            return ControlMessage(
                "heartbeat_ack",
                message.request_id,
                {"robot_monotonic_ns": self.clock_ns(), "watchdog_remaining_ms": remaining},
                session_id,
                generation,
            )

        if kind == "status":
            session_id, generation = self._require_session(message)
            status = await asyncio.to_thread(self.callbacks.session_status, session_id, generation)
            return ControlMessage(
                "status",
                message.request_id,
                {"status": dict(status)},
                session_id,
                generation,
            )

        if kind == "stop":
            session_id, generation = self._require_session(message)
            reason = message.payload["reason"]
            if not isinstance(reason, str) or not reason.strip() or len(reason.encode()) > 256:
                raise ControlStateError("STOP reason must be a short non-empty string")
            receipt = await asyncio.to_thread(
                self.callbacks.stop_session, session_id, generation, reason.strip()
            )
            self._session = None
            self._clock_monitor = None
            self._loss_reported = True
            return ControlMessage(
                "stop_ack",
                message.request_id,
                {"receipt": dict(receipt)},
                session_id,
                generation,
            )

        raise ControlStateError(f"{kind} is not a client request")

    async def invalidate_credential(self, credential_id: str) -> bool:
        """Invalidate this protocol before its transport is closed."""
        if self._credential_id != credential_id:
            return False
        self._authorization_invalidated = True
        return True

    async def connection_lost(self, reason: str) -> None:
        session = self._session
        if session is None or self._loss_reported:
            return
        self._loss_reported = True
        self._session = None
        self._clock_monitor = None
        await asyncio.to_thread(self.callbacks.control_lost, *session, reason)


class TlsControlServer:
    """Dedicated TLS WebSocket listener; nothing starts until ``start`` is awaited."""

    def __init__(
        self,
        bind_address: str,
        port: int,
        tls_context: ssl.SSLContext,
        protocol_factory: Callable[[], RobotControlProtocol],
        *,
        allow_loopback: bool = False,
        heartbeat_deadline_s: float = 1.0,
        negotiation_idle_deadline_s: float = 10.0,
    ) -> None:
        self.bind_ip = validate_private_bind_address(bind_address, allow_loopback=allow_loopback)
        if (
            not 0 <= port <= 65535
            or not 0.1 <= heartbeat_deadline_s <= 10
            or not 1 <= negotiation_idle_deadline_s <= 60
        ):
            raise ValueError("control port or connection deadlines are out of bounds")
        if not isinstance(tls_context, ssl.SSLContext):
            raise TypeError("an explicit TLS server context is required")
        self.port = port
        self.tls_context = tls_context
        self.protocol_factory = protocol_factory
        self.heartbeat_deadline_s = heartbeat_deadline_s
        self.negotiation_idle_deadline_s = negotiation_idle_deadline_s
        self._server: Any = None
        self._connections: dict[RobotControlProtocol, Any] = {}

    @property
    def bound_port(self) -> int | None:
        server = self._server
        if server is None or not server.sockets:
            return None
        return int(server.sockets[0].getsockname()[1])

    async def start(self) -> int:
        if self._server is not None:
            raise RuntimeError("TLS control server is already started")
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._handle_connection,
            str(self.bind_ip),
            self.port,
            ssl=self.tls_context,
            compression=None,
            ping_interval=None,
            max_size=MAX_CONTROL_FRAME_BYTES,
            max_queue=4,
        )
        assert self.bound_port is not None
        return self.bound_port

    async def _handle_connection(self, websocket: Any) -> None:
        protocol = self.protocol_factory()
        self._connections[protocol] = websocket
        close_reason = "control_channel_lost"
        try:
            while True:
                timeout = (
                    self.heartbeat_deadline_s if protocol.session_active else self.negotiation_idle_deadline_s
                )
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                except TimeoutError:
                    close_reason = "control_heartbeat_timeout"
                    break
                try:
                    request = decode_control_message(raw)
                except ControlProtocolError as exc:
                    response = make_error(
                        "invalid-frame",
                        code="invalid_frame",
                        detail=str(exc),
                    )
                    await websocket.send(encode_control_message(response))
                    close_reason = "invalid_control_frame"
                    break
                response = await protocol.handle(request)
                await websocket.send(encode_control_message(response))
                if protocol.authorization_invalidated:
                    close_reason = "operator_credential_revoked"
                    break
        except asyncio.CancelledError:
            close_reason = "control_server_shutdown"
            raise
        except Exception:
            # Connection exceptions never become status claims; the robot callback owns STOP.
            close_reason = "control_channel_lost"
        finally:
            self._connections.pop(protocol, None)
            await protocol.connection_lost(close_reason)
            with contextlib.suppress(Exception):
                await websocket.close()

    async def revoke_credential(self, credential_id: str) -> int:
        """Invalidate and close every live protocol for one credential."""
        matches = [
            (protocol, websocket)
            for protocol, websocket in tuple(self._connections.items())
            if protocol.credential_id == credential_id
        ]
        for protocol, websocket in matches:
            with contextlib.suppress(Exception):
                await protocol.invalidate_credential(credential_id)
            with contextlib.suppress(Exception):
                await websocket.close(code=1008, reason="operator credential revoked")
        return len(matches)

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
