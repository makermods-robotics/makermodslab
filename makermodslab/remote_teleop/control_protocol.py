"""Strict, bounded control-channel messages for split-host teleoperation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

CONTROL_PROTOCOL_VERSION = "makermodslab.remote-control.v1"
MAX_CONTROL_FRAME_BYTES = 65_536
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ControlProtocolError(ValueError):
    """A control frame is malformed, oversized, or invalid in this protocol."""


_SESSION_BOUND = {
    "udp_probe_ready",
    "udp_endpoint_bound",
    "clock_check",
    "clock_check_ack",
    "heartbeat",
    "heartbeat_ack",
    "status",
    "observation",
    "stop",
    "stop_ack",
}

# Payloads are deliberately exact. Adding a field requires a protocol version decision.
_PAYLOAD_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "hello": (frozenset({"node_id", "credential_id", "credential_secret"}), frozenset()),
    "hello_ack": (
        frozenset({"robot_id", "authenticated", "pairing_allowed"}),
        frozenset(),
    ),
    "pair": (frozenset({"pairing_token", "operator_label"}), frozenset()),
    "pair_ack": (
        frozenset({"credential_id", "credential_secret", "operator_label"}),
        frozenset(),
    ),
    "clock_probe": (frozenset({"operator_send_ns"}), frozenset()),
    "clock_reply": (
        frozenset({"operator_send_ns", "robot_receive_ns", "robot_send_ns"}),
        frozenset(),
    ),
    "profile": (frozenset(), frozenset()),
    "profile_ack": (frozenset({"profile"}), frozenset()),
    "session_open": (frozenset({"spec", "clock_samples"}), frozenset()),
    "session_grant": (
        frozenset({"session", "action_key_base64", "udp_host", "udp_port", "clock"}),
        frozenset(),
    ),
    "udp_probe_ready": (frozenset({"key_id"}), frozenset()),
    "udp_endpoint_bound": (frozenset({"bound"}), frozenset()),
    "clock_check": (frozenset({"clock_samples"}), frozenset()),
    "clock_check_ack": (frozenset({"valid", "clock"}), frozenset()),
    "heartbeat": (
        frozenset({"operator_monotonic_ns", "operator_process_live", "browser_live"}),
        frozenset(),
    ),
    "heartbeat_ack": (
        frozenset({"robot_monotonic_ns", "watchdog_remaining_ms"}),
        frozenset(),
    ),
    "status": (frozenset({"status"}), frozenset()),
    "observation": (frozenset({"observation", "robot_monotonic_ns"}), frozenset()),
    "stop": (frozenset({"reason"}), frozenset()),
    "stop_ack": (frozenset({"receipt"}), frozenset()),
    "error": (frozenset({"code", "detail"}), frozenset({"failed_message_type"})),
}

_SECRET_KEYS = {
    "credential_secret",
    "pairing_token",
    "action_key_base64",
    "action_key",
    "private_key",
    "tls_private_key",
}


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {key: "[redacted]" if key in _SECRET_KEYS else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _check_json_value(value: object, *, depth: int = 0) -> None:
    if depth > 12:
        raise ControlProtocolError("control payload nesting exceeds the bound")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControlProtocolError("control payload contains a non-finite number")
        return
    if isinstance(value, list):
        if len(value) > 1024:
            raise ControlProtocolError("control payload list exceeds the bound")
        for item in value:
            _check_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256 or any(not isinstance(key, str) for key in value):
            raise ControlProtocolError("control payload object exceeds the bound")
        for item in value.values():
            _check_json_value(item, depth=depth + 1)
        return
    raise ControlProtocolError("control payload contains a non-JSON value")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ControlProtocolError(f"{name} must be a path-free identifier")
    return value


@dataclass(frozen=True, repr=False)
class ControlMessage:
    message_type: str
    request_id: str
    payload: Mapping[str, object]
    session_id: str | None = None
    executor_generation: int | None = None

    def __post_init__(self) -> None:
        if self.message_type not in _PAYLOAD_FIELDS:
            raise ControlProtocolError("unsupported control message type")
        _identifier(self.request_id, "request_id")
        if not isinstance(self.payload, Mapping):
            raise ControlProtocolError("control payload must be an object")
        required, optional = _PAYLOAD_FIELDS[self.message_type]
        keys = frozenset(self.payload)
        if not required <= keys or not keys <= required | optional:
            raise ControlProtocolError(f"{self.message_type} payload has missing or extra fields")
        payload = dict(self.payload)
        _check_json_value(payload)
        object.__setattr__(self, "payload", MappingProxyType(payload))
        if self.message_type in _SESSION_BOUND:
            _identifier(self.session_id, "session_id")
            generation = self.executor_generation
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise ControlProtocolError("executor_generation must be a positive integer")
        elif self.session_id is not None or self.executor_generation is not None:
            raise ControlProtocolError("non-session control message contains session authority")

    def body(self, *, redact_secrets: bool = False) -> dict[str, object]:
        payload = _redact(dict(self.payload)) if redact_secrets else dict(self.payload)
        assert isinstance(payload, dict)
        body: dict[str, object] = {
            "protocol_version": CONTROL_PROTOCOL_VERSION,
            "message_type": self.message_type,
            "request_id": self.request_id,
            "payload": payload,
        }
        if self.session_id is not None:
            body["session_id"] = self.session_id
            body["executor_generation"] = self.executor_generation
        return body

    def redacted(self) -> dict[str, object]:
        return self.body(redact_secrets=True)

    def __repr__(self) -> str:
        return f"ControlMessage({self.redacted()!r})"


def encode_control_message(message: ControlMessage) -> str:
    try:
        encoded = json.dumps(
            message.body(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ControlProtocolError("control message is not canonical JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError("control frame exceeds 64 KiB")
    return encoded


def decode_control_message(raw: str | bytes) -> ControlMessage:
    if isinstance(raw, bytes):
        if len(raw) > MAX_CONTROL_FRAME_BYTES:
            raise ControlProtocolError("control frame exceeds 64 KiB")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ControlProtocolError("control frame is not UTF-8") from exc
    if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError("control frame is empty or oversized")
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ControlProtocolError("control frame is invalid JSON") from exc
    if not isinstance(body, dict):
        raise ControlProtocolError("control frame must be an object")
    common = {"protocol_version", "message_type", "request_id", "payload"}
    session = {"session_id", "executor_generation"}
    message_type = body.get("message_type")
    expected = common | session if message_type in _SESSION_BOUND else common
    if set(body) != expected:
        raise ControlProtocolError("control frame has missing or extra fields")
    if body.get("protocol_version") != CONTROL_PROTOCOL_VERSION:
        raise ControlProtocolError("unsupported control protocol version")
    message = ControlMessage(
        message_type=message_type,
        request_id=body.get("request_id"),
        payload=body.get("payload"),
        session_id=body.get("session_id"),
        executor_generation=body.get("executor_generation"),
    )
    if encode_control_message(message) != raw:
        raise ControlProtocolError("control frame is not canonical JSON")
    return message


def make_error(
    request_id: str,
    *,
    code: str,
    detail: str,
    failed_message_type: str | None = None,
) -> ControlMessage:
    payload: dict[str, Any] = {"code": _identifier(code, "code"), "detail": detail[:512]}
    if failed_message_type is not None:
        payload["failed_message_type"] = failed_message_type
    return ControlMessage("error", request_id, payload)
