"""Pinned-certificate operator client for the dedicated robot control socket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ssl
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .clock_sync import ClockSample, FrozenClockMapping, select_clock_mapping
from .contracts import SessionSpec
from .control_protocol import (
    MAX_CONTROL_FRAME_BYTES,
    ControlMessage,
    decode_control_message,
    encode_control_message,
)
from .pairing import IssuedCredential, verify_certificate_fingerprint


class ControlClientError(RuntimeError):
    """The pinned robot control connection or a request failed."""


@dataclass(frozen=True)
class OperatorSessionGrant:
    session_id: str
    executor_generation: int
    key_id: str
    action_key_base64: str = field(repr=False)
    udp_host: str
    udp_port: int
    clock: FrozenClockMapping

    def action_key(self) -> bytes:
        try:
            decoded = base64.b64decode(self.action_key_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ControlClientError("robot action key is malformed") from exc
        if len(decoded) != 32:
            raise ControlClientError("robot action key is malformed")
        return decoded


def _spec_body(spec: SessionSpec) -> dict[str, object]:
    return {
        "source_id": spec.source_id,
        "rig_id": spec.rig_id,
        "rig_digest": spec.rig_digest,
        "leader_calibration_id": spec.leader_calibration_id,
        "leader_calibration_digest": spec.leader_calibration_digest,
        "follower_calibration_id": spec.follower_calibration_id,
        "follower_calibration_digest": spec.follower_calibration_digest,
        "joint_names": list(spec.joint_names),
        "units": list(spec.units),
    }


class PinnedControlClient:
    """Sequential request client; fingerprint verification precedes all credentials."""

    def __init__(
        self,
        uri: str,
        certificate_fingerprint: str,
        *,
        ssl_context: ssl.SSLContext | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        request_timeout_s: float = 3.0,
    ) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "wss" or not parsed.hostname or parsed.path not in ("", "/"):
            raise ValueError("control URI must be a wss:// host and port without a path")
        self.uri = uri
        self.certificate_fingerprint = certificate_fingerprint
        if ssl_context is None:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        self.ssl_context = ssl_context
        self.clock_ns = clock_ns
        if not 0.1 <= request_timeout_s <= 10:
            raise ValueError("control request timeout must be in [0.1s,10s]")
        self.request_timeout_s = request_timeout_s
        self._connection: Any = None
        self._request_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._session: tuple[str, int] | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def connect(self) -> None:
        if self._connection is not None:
            raise ControlClientError("control client is already connected")
        from websockets.asyncio.client import connect

        connection = await connect(
            self.uri,
            ssl=self.ssl_context,
            compression=None,
            ping_interval=None,
            max_size=MAX_CONTROL_FRAME_BYTES,
            max_queue=4,
            proxy=None,
        )
        try:
            ssl_object = connection.transport.get_extra_info("ssl_object")
            certificate = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
            if not isinstance(certificate, bytes):
                raise ControlClientError("robot connection did not present a TLS certificate")
            verify_certificate_fingerprint(certificate, self.certificate_fingerprint)
        except Exception:
            await connection.close()
            raise
        self._connection = connection

    async def request(self, message: ControlMessage) -> ControlMessage:
        connection = self._connection
        if connection is None:
            raise ControlClientError("control client is not connected")
        async with self._request_lock:
            try:
                await asyncio.wait_for(
                    connection.send(encode_control_message(message)),
                    timeout=self.request_timeout_s,
                )
                raw = await asyncio.wait_for(
                    connection.recv(),
                    timeout=self.request_timeout_s,
                )
            except TimeoutError as exc:
                raise ControlClientError("robot control request timed out") from exc
            response = decode_control_message(raw)
        if response.request_id != message.request_id:
            raise ControlClientError("robot control response request id does not match")
        if response.message_type == "error":
            raise ControlClientError(f"robot refused {message.message_type}: {response.payload['code']}")
        return response

    def _request_id(self) -> str:
        return "request-" + uuid.uuid4().hex

    async def hello(self, node_id: str, credential: IssuedCredential | None = None) -> dict[str, object]:
        response = await self.request(
            ControlMessage(
                "hello",
                self._request_id(),
                {
                    "node_id": node_id,
                    "credential_id": credential.credential_id if credential else None,
                    "credential_secret": credential.secret if credential else None,
                },
            )
        )
        return dict(response.payload)

    async def pair(self, pairing_token: str, operator_label: str) -> IssuedCredential:
        response = await self.request(
            ControlMessage(
                "pair",
                self._request_id(),
                {"pairing_token": pairing_token, "operator_label": operator_label},
            )
        )
        return IssuedCredential(
            response.payload["credential_id"],
            response.payload["credential_secret"],
            response.payload["operator_label"],
        )

    async def synchronize_clocks(
        self, *, sample_count: int = 16
    ) -> tuple[FrozenClockMapping, list[ClockSample]]:
        if sample_count != 16:
            raise ValueError("control protocol v1 requires exactly 16 operator clock probes")
        samples: list[ClockSample] = []
        for _ in range(sample_count):
            t0 = self.clock_ns()
            response = await self.request(
                ControlMessage(
                    "clock_probe",
                    self._request_id(),
                    {"operator_send_ns": t0},
                )
            )
            t3 = self.clock_ns()
            samples.append(
                ClockSample(
                    operator_send_ns=t0,
                    robot_receive_ns=response.payload["robot_receive_ns"],
                    robot_send_ns=response.payload["robot_send_ns"],
                    operator_receive_ns=t3,
                )
            )
        return select_clock_mapping(samples), samples

    async def profile(self) -> dict[str, object]:
        response = await self.request(ControlMessage("profile", self._request_id(), {}))
        profile = response.payload["profile"]
        if not isinstance(profile, dict):
            raise ControlClientError("robot session profile is malformed")
        return dict(profile)

    async def check_clock(self, samples: Iterable[ClockSample]) -> bool:
        session_id, generation = self._require_session()
        collected = list(samples)
        if len(collected) != 16:
            raise ValueError("control protocol v1 requires exactly 16 clock-check samples")
        response = await self.request(
            ControlMessage(
                "clock_check",
                self._request_id(),
                {"clock_samples": [sample.to_dict() for sample in collected]},
                session_id,
                generation,
            )
        )
        valid = response.payload["valid"] is True
        if not valid:
            self._session = None
        return valid

    async def open_session(self, spec: SessionSpec, samples: Iterable[ClockSample]) -> OperatorSessionGrant:
        response = await self.request(
            ControlMessage(
                "session_open",
                self._request_id(),
                {"spec": _spec_body(spec), "clock_samples": [sample.to_dict() for sample in samples]},
            )
        )
        session = response.payload["session"]
        clock = response.payload["clock"]
        if not isinstance(session, dict) or not isinstance(clock, dict):
            raise ControlClientError("robot session grant is malformed")
        mapping = FrozenClockMapping(
            robot_minus_operator_ns=clock["robot_minus_operator_ns"],
            uncertainty_ns=clock["uncertainty_ns"],
            selected_round_trip_ns=clock["selected_round_trip_ns"],
            sample_count=clock["sample_count"],
        )
        grant = OperatorSessionGrant(
            session_id=session["session_id"],
            executor_generation=session["executor_generation"],
            key_id=session["key_id"],
            action_key_base64=response.payload["action_key_base64"],
            udp_host=response.payload["udp_host"],
            udp_port=response.payload["udp_port"],
            clock=mapping,
        )
        grant.action_key()
        self._session = (grant.session_id, grant.executor_generation)
        return grant

    async def udp_probe_status(self, key_id: str) -> bool:
        session_id, generation = self._require_session()
        response = await self.request(
            ControlMessage(
                "udp_probe_ready",
                self._request_id(),
                {"key_id": key_id},
                session_id,
                generation,
            )
        )
        return response.payload["bound"] is True

    def _require_session(self) -> tuple[str, int]:
        if self._session is None:
            raise ControlClientError("no robot session is active")
        return self._session

    async def heartbeat(self, *, browser_live: bool = True) -> dict[str, object]:
        session_id, generation = self._require_session()
        response = await self.request(
            ControlMessage(
                "heartbeat",
                self._request_id(),
                {
                    "operator_monotonic_ns": self.clock_ns(),
                    "operator_process_live": True,
                    "browser_live": browser_live,
                },
                session_id,
                generation,
            )
        )
        return dict(response.payload)

    def start_heartbeats(
        self,
        *,
        interval_s: float = 0.25,
        browser_live: Callable[[], bool] = lambda: True,
        on_failure: Callable[[Exception], object] | None = None,
    ) -> None:
        if not 0.05 <= interval_s <= 0.5:
            raise ValueError("heartbeat interval must be in [50ms,500ms]")
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            raise ControlClientError("heartbeats are already active")

        async def run() -> None:
            while True:
                try:
                    await self.heartbeat(browser_live=browser_live())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if on_failure is not None:
                        on_failure(exc)
                    return
                await asyncio.sleep(interval_s)

        self._heartbeat_task = asyncio.create_task(run(), name="remote-teleop-control-heartbeat")

    async def stop(self, reason: str) -> dict[str, object]:
        session_id, generation = self._require_session()
        response = await self.request(
            ControlMessage(
                "stop",
                self._request_id(),
                {"reason": reason},
                session_id,
                generation,
            )
        )
        self._session = None
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
        return dict(response.payload["receipt"])

    async def close(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        connection = self._connection
        self._connection = None
        self._session = None
        if connection is not None:
            await connection.close()
