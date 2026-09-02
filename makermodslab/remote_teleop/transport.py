"""Endpoint-bound UDP action transport and deterministic test adapters."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import socket
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from typing import Protocol

from .contracts import MAX_DATAGRAM_BYTES

UDP_PROBE_VERSION = "makermodslab.remote-udp-probe.v1"
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class UdpTransportError(RuntimeError):
    """A UDP endpoint, session probe, or datagram violated the action lane."""


class DatagramTransport(Protocol):
    def send(self, payload: bytes) -> None: ...

    def close(self) -> None: ...


class UdpActionSender:
    """Small provider-neutral sender; it never knows about serial hardware."""

    def __init__(self, host: str, port: int, *, sock: socket.socket | None = None) -> None:
        if not host or not 1 <= port <= 65535:
            raise ValueError("a host and valid UDP port are required")
        self._target = (host, port)
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload or len(payload) > MAX_DATAGRAM_BYTES:
            raise ValueError("datagram payload must be non-empty bytes within the size bound")
        self._socket.sendto(payload, self._target)

    def close(self) -> None:
        self._socket.close()


def validate_private_bind_address(
    address: str, *, allow_loopback: bool = False
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Require an exact private/tailnet IP literal; binding later proves assignment."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as exc:
        raise UdpTransportError("bind address must be an exact IP literal") from exc
    if parsed.is_unspecified or parsed.is_multicast:
        raise UdpTransportError("wildcard and multicast bind addresses are forbidden")
    if parsed.is_loopback:
        if not allow_loopback:
            raise UdpTransportError("loopback is allowed only for local simulation")
        return parsed
    if parsed.is_link_local:
        raise UdpTransportError("link-local addresses are not valid remote-control interfaces")
    if not (parsed.is_private or (isinstance(parsed, ipaddress.IPv4Address) and parsed in _CGNAT)):
        raise UdpTransportError("bind address must be private or in the Tailscale CGNAT range")
    return parsed


def _probe_unsigned(
    message_type: str,
    *,
    session_id: str,
    executor_generation: int,
    key_id: str,
    nonce: str,
) -> dict[str, object]:
    return {
        "protocol_version": UDP_PROBE_VERSION,
        "message_type": message_type,
        "session_id": session_id,
        "executor_generation": executor_generation,
        "key_id": key_id,
        "nonce": nonce,
    }


def _canonical_probe(body: dict[str, object]) -> bytes:
    try:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    except (TypeError, ValueError) as exc:
        raise UdpTransportError("UDP probe is not canonical JSON") from exc
    if len(raw) > MAX_DATAGRAM_BYTES:
        raise UdpTransportError("UDP probe exceeds the datagram size bound")
    return raw


def encode_udp_probe(
    *,
    session_id: str,
    executor_generation: int,
    key_id: str,
    nonce: str,
    key: bytes,
    acknowledgement: bool = False,
) -> bytes:
    if not isinstance(key, bytes) or len(key) < 32:
        raise UdpTransportError("UDP probe key must contain at least 256 bits")
    if not all(isinstance(value, str) and value for value in (session_id, key_id, nonce)):
        raise UdpTransportError("UDP probe identity is invalid")
    if (
        isinstance(executor_generation, bool)
        or not isinstance(executor_generation, int)
        or executor_generation < 1
    ):
        raise UdpTransportError("UDP probe generation is invalid")
    message_type = "action.udp_probe_ack" if acknowledgement else "action.udp_probe"
    unsigned = _probe_unsigned(
        message_type,
        session_id=session_id,
        executor_generation=executor_generation,
        key_id=key_id,
        nonce=nonce,
    )
    tag = hmac.new(key, _canonical_probe(unsigned), hashlib.sha256).hexdigest()
    return _canonical_probe({**unsigned, "auth_tag": tag})


def decode_udp_probe(
    raw: bytes,
    *,
    session_id: str,
    executor_generation: int,
    key_id: str,
    key: bytes,
    acknowledgement: bool = False,
) -> str:
    if not raw or len(raw) > MAX_DATAGRAM_BYTES:
        raise UdpTransportError("UDP probe is empty or oversized")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise UdpTransportError("UDP probe is invalid JSON") from exc
    expected_fields = {
        "protocol_version",
        "message_type",
        "session_id",
        "executor_generation",
        "key_id",
        "nonce",
        "auth_tag",
    }
    if not isinstance(body, dict) or set(body) != expected_fields:
        raise UdpTransportError("UDP probe has missing or extra fields")
    message_type = "action.udp_probe_ack" if acknowledgement else "action.udp_probe"
    if (
        body["protocol_version"] != UDP_PROBE_VERSION
        or body["message_type"] != message_type
        or body["session_id"] != session_id
        or body["executor_generation"] != executor_generation
        or body["key_id"] != key_id
        or not isinstance(body["nonce"], str)
        or not isinstance(body["auth_tag"], str)
    ):
        raise UdpTransportError("UDP probe session identity is invalid")
    unsigned = dict(body)
    tag = unsigned.pop("auth_tag")
    expected_tag = hmac.new(key, _canonical_probe(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag, expected_tag):
        raise UdpTransportError("UDP probe authentication failed")
    canonical = encode_udp_probe(
        session_id=session_id,
        executor_generation=executor_generation,
        key_id=key_id,
        nonce=body["nonce"],
        key=key,
        acknowledgement=acknowledgement,
    )
    if canonical != raw:
        raise UdpTransportError("UDP probe is not canonical JSON")
    return body["nonce"]


class UdpActionReceiver:
    """Explicit-start receiver pinned to one authenticated session endpoint."""

    def __init__(
        self,
        bind_address: str,
        port: int,
        on_datagram: Callable[[bytes], None],
        *,
        allow_loopback: bool = False,
        socket_factory: Callable[..., socket.socket] = socket.socket,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        invalid_per_second: int = 20,
    ) -> None:
        self.bind_ip = validate_private_bind_address(bind_address, allow_loopback=allow_loopback)
        if not 0 <= port <= 65535 or not 1 <= invalid_per_second <= 1000:
            raise ValueError("UDP port or invalid-packet rate limit is out of bounds")
        self.port = port
        self.on_datagram = on_datagram
        self.socket_factory = socket_factory
        self.clock_ns = clock_ns
        self.invalid_per_second = invalid_per_second
        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._session: tuple[str, int, str, bytes] | None = None
        self._endpoint: tuple[str, int] | None = None
        self._invalid_windows: dict[str, deque[int]] = {}
        self._counters: Counter[str] = Counter()

    @property
    def bound_port(self) -> int | None:
        with self._lock:
            if self._socket is None:
                return None
            return int(self._socket.getsockname()[1])

    def begin_session(
        self,
        *,
        session_id: str,
        executor_generation: int,
        key_id: str,
        key: bytes,
    ) -> None:
        # Encoding once gives all identity/key validation in one place.
        encode_udp_probe(
            session_id=session_id,
            executor_generation=executor_generation,
            key_id=key_id,
            nonce="validation",
            key=key,
        )
        with self._lock:
            self._session = (session_id, executor_generation, key_id, bytes(key))
            self._endpoint = None
            self._invalid_windows.clear()

    def start(self) -> int:
        with self._lock:
            if self._socket is not None:
                raise RuntimeError("UDP action receiver is already started")
            family = socket.AF_INET6 if self.bind_ip.version == 6 else socket.AF_INET
            sock = self.socket_factory(family, socket.SOCK_DGRAM)
            try:
                sock.bind((str(self.bind_ip), self.port))
                sock.settimeout(0.05)
            except Exception:
                sock.close()
                raise
            self._socket = sock
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="remote-teleop-udp", daemon=True)
            self._thread.start()
            return int(sock.getsockname()[1])

    def _invalid_blocked(self, source_ip: str, now_ns: int) -> bool:
        window = self._invalid_windows.setdefault(source_ip, deque())
        cutoff = now_ns - 1_000_000_000
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.invalid_per_second:
            self._counters["invalid_rate_limited"] += 1
            return True

        return False

    def _record_invalid(self, source_ip: str, now_ns: int) -> None:
        window = self._invalid_windows.setdefault(source_ip, deque())
        window.append(now_ns)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                sock = self._socket
            if sock is None:
                return
            try:
                raw, source = sock.recvfrom(MAX_DATAGRAM_BYTES + 1)
            except TimeoutError:
                continue
            except OSError:
                return
            source_endpoint = (source[0], source[1])
            if len(raw) > MAX_DATAGRAM_BYTES:
                self._counters["oversized"] += 1
                continue
            # A receiver normally starts before a session is minted. Refresh
            # these values after the blocking read so a begin_session() that
            # races with recvfrom() cannot make the first valid probe look as
            # though no session exists. Conversely, a concurrent STOP remains
            # fail-closed because the post-read snapshot observes the cleared
            # session and will not dispatch the datagram.
            with self._lock:
                session = self._session
                endpoint = self._endpoint
            if session is None:
                self._counters["no_session"] += 1
                continue
            session_id, generation, key_id, key = session
            if endpoint is None:
                source_key = f"{source[0]}:{source[1]}"
                now_ns = self.clock_ns()
                if self._invalid_blocked(source_key, now_ns):
                    continue
                try:
                    nonce = decode_udp_probe(
                        raw,
                        session_id=session_id,
                        executor_generation=generation,
                        key_id=key_id,
                        key=key,
                    )
                except UdpTransportError:
                    self._record_invalid(source_key, now_ns)
                    self._counters["probe_rejected"] += 1
                    continue
                with self._lock:
                    if self._session == session and self._endpoint is None:
                        self._endpoint = source_endpoint
                        endpoint = source_endpoint
                        self._counters["endpoint_bound"] += 1
                acknowledgement = encode_udp_probe(
                    session_id=session_id,
                    executor_generation=generation,
                    key_id=key_id,
                    nonce=nonce,
                    key=key,
                    acknowledgement=True,
                )
                sock.sendto(acknowledgement, source)
                continue
            if source_endpoint != endpoint:
                source_key = f"{source[0]}:{source[1]}"
                now_ns = self.clock_ns()
                if not self._invalid_blocked(source_key, now_ns):
                    self._record_invalid(source_key, now_ns)
                    self._counters["endpoint_rejected"] += 1
                continue
            source_key = f"{source[0]}:{source[1]}"
            now_ns = self.clock_ns()
            if self._invalid_blocked(source_key, now_ns):
                continue
            try:
                self.on_datagram(raw)
                self._counters["datagram_dispatched"] += 1
            except Exception:
                # The executor records the detailed rejection; the UDP loop stays alive.
                self._record_invalid(source_key, now_ns)
                self._counters["datagram_rejected"] += 1

    def stop_dispatch(self) -> None:
        """Atomically prevent future action dispatch before hardware teardown."""
        with self._lock:
            self._session = None
            self._endpoint = None

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "listening": self._socket is not None,
                "bound_port": self.bound_port,
                "session_configured": self._session is not None,
                "endpoint_bound": self._endpoint is not None,
                "counters": dict(self._counters),
            }

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            sock = self._socket
            self._socket = None
            self._session = None
            self._endpoint = None
        if sock is not None:
            sock.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


class UdpSessionSender:
    """Operator sender whose source endpoint is proven before actions flow."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        sock: socket.socket | None = None,
    ) -> None:
        if not host or not 1 <= port <= 65535:
            raise ValueError("a UDP host and valid port are required")
        self._target = (host, port)
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._proven = False

    def prove_endpoint(
        self,
        *,
        session_id: str,
        executor_generation: int,
        key_id: str,
        key: bytes,
        timeout_s: float = 1.0,
    ) -> None:
        nonce = hashlib.sha256(str(time.monotonic_ns()).encode()).hexdigest()[:32]
        probe = encode_udp_probe(
            session_id=session_id,
            executor_generation=executor_generation,
            key_id=key_id,
            nonce=nonce,
            key=key,
        )
        prior_timeout = self._socket.gettimeout()
        try:
            self._socket.settimeout(timeout_s)
            self._socket.sendto(probe, self._target)
            reply, source = self._socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
            if source[0] != self._target[0] or source[1] != self._target[1]:
                raise UdpTransportError("UDP probe acknowledgement came from another endpoint")
            acknowledged_nonce = decode_udp_probe(
                reply,
                session_id=session_id,
                executor_generation=executor_generation,
                key_id=key_id,
                key=key,
                acknowledgement=True,
            )
            if not hmac.compare_digest(acknowledged_nonce, nonce):
                raise UdpTransportError("UDP probe acknowledgement nonce does not match")
            self._proven = True
        finally:
            self._socket.settimeout(prior_timeout)

    def send(self, payload: bytes) -> None:
        if not self._proven:
            raise UdpTransportError("UDP endpoint has not been proven for this session")
        if not isinstance(payload, bytes) or not payload or len(payload) > MAX_DATAGRAM_BYTES:
            raise UdpTransportError("action datagram is empty or oversized")
        self._socket.sendto(payload, self._target)

    def close(self) -> None:
        self._proven = False
        self._socket.close()


class UdpFaultProxy:
    """Explicit-start bidirectional UDP proxy with deterministic action faults."""

    def __init__(
        self,
        listen_address: str,
        listen_port: int,
        target_address: str,
        target_port: int,
        *,
        allow_loopback: bool = False,
        drop_numbers: set[int] | None = None,
        duplicate_numbers: set[int] | None = None,
        delay_numbers: set[int] | None = None,
        corrupt_numbers: set[int] | None = None,
    ) -> None:
        self.listen_ip = validate_private_bind_address(listen_address, allow_loopback=allow_loopback)
        self.target_ip = validate_private_bind_address(target_address, allow_loopback=allow_loopback)
        if self.listen_ip.version != self.target_ip.version:
            raise ValueError("UDP fault proxy endpoints must use the same IP family")
        if not 0 <= listen_port <= 65535 or not 1 <= target_port <= 65535:
            raise ValueError("UDP fault proxy port is invalid")
        self.listen_port = listen_port
        self.target = (str(self.target_ip), target_port)
        self.drop_numbers = drop_numbers or set()
        self.duplicate_numbers = duplicate_numbers or set()
        self.delay_numbers = delay_numbers or set()
        self.corrupt_numbers = corrupt_numbers or set()
        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client: tuple[str, int] | None = None
        self._forwarding = True
        self._number = 0
        self._delayed: deque[bytes] = deque()
        self._counters: Counter[str] = Counter()

    @property
    def bound_port(self) -> int | None:
        with self._lock:
            return int(self._socket.getsockname()[1]) if self._socket is not None else None

    def start(self) -> int:
        with self._lock:
            if self._socket is not None:
                raise RuntimeError("UDP fault proxy is already started")
            family = socket.AF_INET6 if self.listen_ip.version == 6 else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_DGRAM)
            try:
                sock.bind((str(self.listen_ip), self.listen_port))
                sock.settimeout(0.05)
            except Exception:
                sock.close()
                raise
            self._socket = sock
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="remote-teleop-udp-fault-proxy", daemon=True
            )
            self._thread.start()
            return int(sock.getsockname()[1])

    def set_forwarding(self, enabled: bool) -> None:
        with self._lock:
            self._forwarding = bool(enabled)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                sock = self._socket
            if sock is None:
                return
            try:
                raw, source = sock.recvfrom(MAX_DATAGRAM_BYTES + 1)
            except TimeoutError:
                continue
            except OSError:
                return
            endpoint = (source[0], source[1])
            with self._lock:
                if endpoint == self.target:
                    client = self._client
                    if self._forwarding and client is not None:
                        sock.sendto(raw, client)
                        self._counters["reverse_forwarded"] += 1
                    continue
                self._client = endpoint
                self._number += 1
                number = self._number
                if not self._forwarding or number in self.drop_numbers:
                    self._counters["dropped"] += 1
                    continue
                if number in self.delay_numbers:
                    self._delayed.append(raw)
                    self._counters["delayed"] += 1
                    continue
                if number in self.corrupt_numbers and raw:
                    raw = bytes([raw[0] ^ 1]) + raw[1:]
                    self._counters["corrupted"] += 1
                sock.sendto(raw, self.target)
                self._counters["forwarded"] += 1
                if number in self.duplicate_numbers:
                    sock.sendto(raw, self.target)
                    self._counters["duplicated"] += 1

    def flush_delayed(self, *, newest_first: bool = True) -> None:
        with self._lock:
            sock = self._socket
            if sock is None:
                raise RuntimeError("UDP fault proxy is not started")
            while self._delayed:
                raw = self._delayed.pop() if newest_first else self._delayed.popleft()
                sock.sendto(raw, self.target)
                self._counters["delayed_forwarded"] += 1

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "listening": self._socket is not None,
                "bound_port": self.bound_port,
                "client_observed": self._client is not None,
                "forwarding": self._forwarding,
                "pending_delayed": len(self._delayed),
                "counters": dict(self._counters),
            }

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            sock = self._socket
            self._socket = None
            self._client = None
            self._delayed.clear()
        if sock is not None:
            sock.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


class FaultInjectingDatagramTransport:
    """Deterministic packet loss/reorder/duplication for hardware-free tests."""

    def __init__(
        self,
        receiver,
        *,
        drop_numbers: set[int] | None = None,
        duplicate_numbers: set[int] | None = None,
        delay_numbers: set[int] | None = None,
    ) -> None:
        self.receiver = receiver
        self.drop_numbers = drop_numbers or set()
        self.duplicate_numbers = duplicate_numbers or set()
        self.delay_numbers = delay_numbers or set()
        self._number = 0
        self._delayed: deque[bytes] = deque()

    def send(self, payload: bytes) -> None:
        self._number += 1
        number = self._number
        if number in self.drop_numbers:
            return
        if number in self.delay_numbers:
            self._delayed.append(payload)
            return
        self.receiver(payload)
        if number in self.duplicate_numbers:
            self.receiver(payload)

    def flush_reordered(self) -> None:
        while self._delayed:
            self.receiver(self._delayed.pop())

    def close(self) -> None:
        self._delayed.clear()
