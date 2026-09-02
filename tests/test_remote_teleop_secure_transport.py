from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from makermodslab.remote_teleop.authority import RemoteSessionAuthority
from makermodslab.remote_teleop.clock_sync import (
    ClockDriftMonitor,
    ClockSample,
    ClockSyncError,
    select_clock_mapping,
)
from makermodslab.remote_teleop.contracts import SessionSpec
from makermodslab.remote_teleop.control_client import PinnedControlClient
from makermodslab.remote_teleop.control_protocol import (
    ControlMessage,
    ControlProtocolError,
    decode_control_message,
    encode_control_message,
)
from makermodslab.remote_teleop.control_server import (
    RobotControlCallbacks,
    RobotControlProtocol,
    RobotSessionProfile,
    SessionOpenResult,
    TlsControlServer,
)
from makermodslab.remote_teleop.pairing import (
    OperatorCredentialVault,
    PairingAuthority,
    PairingError,
    RobotCredentialStore,
    certificate_sha256_fingerprint,
    verify_certificate_fingerprint,
)
from makermodslab.remote_teleop.transport import (
    UdpActionReceiver,
    UdpFaultProxy,
    UdpSessionSender,
    UdpTransportError,
    validate_private_bind_address,
)
from makermodslab.remote_teleop.watchdog import RobotLivenessWatchdog, WatchdogDeadlines

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


class FakeClock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def session_spec() -> SessionSpec:
    return SessionSpec(
        source_id="operator-1",
        rig_id="rig-1",
        rig_digest=DIGEST_A,
        leader_calibration_id="leader-cal-1",
        leader_calibration_digest=DIGEST_B,
        follower_calibration_id="follower-cal-1",
        follower_calibration_digest=DIGEST_C,
        joint_names=("joint_a", "joint_b"),
        units=("rad", "rad"),
    )


def clock_samples(*, offset_ns: int = 9_000_000_000, delay_ns: int = 2_000_000) -> list[ClockSample]:
    return [
        ClockSample(
            operator_send_ns=1_000_000_000 + index * 20_000_000,
            robot_receive_ns=1_000_000_000 + index * 20_000_000 + offset_ns + delay_ns,
            robot_send_ns=1_000_000_000 + index * 20_000_000 + offset_ns + delay_ns + 10_000,
            operator_receive_ns=1_000_000_000 + index * 20_000_000 + 2 * delay_ns + 10_000,
        )
        for index in range(16)
    ]


def test_clock_sync_uses_lowest_rtt_freezes_mapping_and_rejects_drift() -> None:
    samples = clock_samples()
    samples[0] = ClockSample(1_000_000_000, 10_001_000_000, 10_001_010_000, 1_002_010_000)
    mapping = select_clock_mapping(samples)
    assert mapping.robot_minus_operator_ns == 9_000_000_000
    assert mapping.sample_count == 16
    assert mapping.uncertainty_ns == 2_000_000
    assert mapping.robot_interval_for_operator_ns(5_000_000_000) == (
        13_998_000_000,
        14_002_000_000,
    )

    monitor = ClockDriftMonitor(mapping, drift_tolerance_ns=100_000)
    assert monitor.validate(clock_samples(offset_ns=9_000_050_000)) is mapping
    with pytest.raises(ClockSyncError, match="drift"):
        monitor.validate(clock_samples(offset_ns=9_100_000_000))
    with pytest.raises(ClockSyncError, match="requires 16"):
        select_clock_mapping(samples[:15])


def test_pairing_is_single_use_hash_only_private_and_revocable(tmp_path: Path) -> None:
    clock = FakeClock()
    robot_store = RobotCredentialStore(tmp_path / "robot")
    pairing = PairingAuthority(robot_store, clock_ns=clock)
    payload = pairing.open_local_window(
        local_request=True,
        robot_address="100.64.0.10",
        control_port=7443,
        certificate_fingerprint="sha256:" + "a" * 64,
    )
    issued = pairing.exchange(payload.pairing_token, "operator laptop")
    assert robot_store.authenticate(issued.credential_id, issued.secret)
    with pytest.raises(PairingError, match="closed"):
        pairing.exchange(payload.pairing_token, "replay")

    stored = robot_store.path.read_text()
    assert issued.secret not in stored
    assert stat.S_IMODE(robot_store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(robot_store.root.stat().st_mode) == 0o700

    vault = OperatorCredentialVault(tmp_path / "operator")
    vault.put("robot-1", issued)
    assert vault.get("robot-1") == issued
    assert issued.secret in vault.path.read_text()
    assert robot_store.revoke(issued.credential_id)
    assert not robot_store.authenticate(issued.credential_id, issued.secret)


def test_pairing_window_requires_local_action_expires_and_rate_limits(tmp_path: Path) -> None:
    clock = FakeClock()
    pairing = PairingAuthority(RobotCredentialStore(tmp_path), clock_ns=clock, max_attempts=2)
    with pytest.raises(PairingError, match="robot host"):
        pairing.open_local_window(
            local_request=False,
            robot_address="100.64.0.10",
            control_port=7443,
            certificate_fingerprint="b" * 64,
        )
    payload = pairing.open_local_window(
        local_request=True,
        robot_address="100.64.0.10",
        control_port=7443,
        certificate_fingerprint="b" * 64,
    )
    with pytest.raises(PairingError, match="already open"):
        pairing.open_local_window(
            local_request=True,
            robot_address="100.64.0.10",
            control_port=7443,
            certificate_fingerprint="b" * 64,
        )
    with pytest.raises(PairingError, match="invalid"):
        pairing.exchange("wrong", "operator")
    with pytest.raises(PairingError, match="invalid"):
        pairing.exchange("still-wrong", "operator")
    with pytest.raises(PairingError, match="limit"):
        pairing.exchange(payload.pairing_token, "operator")


def test_expired_pairing_window_can_be_replaced_locally(tmp_path: Path) -> None:
    clock = FakeClock()
    pairing = PairingAuthority(RobotCredentialStore(tmp_path), clock_ns=clock)
    first = pairing.open_local_window(
        local_request=True,
        robot_address="100.64.0.10",
        control_port=7443,
        certificate_fingerprint="b" * 64,
    )
    clock.advance_ms(120_001)
    replacement = pairing.open_local_window(
        local_request=True,
        robot_address="100.64.0.10",
        control_port=7443,
        certificate_fingerprint="b" * 64,
    )
    assert replacement.pairing_token != first.pairing_token
    with pytest.raises(PairingError, match="invalid"):
        pairing.exchange(first.pairing_token, "operator")


def test_credential_store_fails_closed_on_unsafe_permissions(tmp_path: Path) -> None:
    root = tmp_path / "unsafe"
    root.mkdir(mode=0o755)
    store = RobotCredentialStore(root)
    with pytest.raises(PairingError, match="directory is not owner-only"):
        store.issue("operator", now_ns=1)


def test_control_frames_are_strict_canonical_bounded_and_redacted() -> None:
    message = ControlMessage(
        "hello",
        "request-1",
        {
            "node_id": "operator-1",
            "credential_id": "cred-1",
            "credential_secret": "do-not-log-material",
        },
    )
    raw = encode_control_message(message)
    assert decode_control_message(raw) == message
    assert message.redacted()["payload"]["credential_secret"] == "[redacted]"
    assert "do-not-log-material" not in repr(message)
    with pytest.raises(ControlProtocolError, match="extra"):
        ControlMessage(
            "hello",
            "request-1",
            {
                "node_id": "operator-1",
                "credential_id": None,
                "credential_secret": None,
                "surprise": True,
            },
        )
    noncanonical = json.dumps(json.loads(raw), indent=2)
    with pytest.raises(ControlProtocolError, match="canonical"):
        decode_control_message(noncanonical)


@pytest.mark.asyncio
async def test_control_state_details_are_redacted_before_authentication(tmp_path: Path) -> None:
    callbacks = RobotControlCallbacks(
        session_profile=lambda _credential_id: RobotSessionProfile(session_spec(), "d" * 64),
        open_session=lambda *_args: pytest.fail("session must not open"),
        stop_session=lambda *_args: pytest.fail("session must not stop"),
        session_status=lambda *_args: {},
        heartbeat=lambda *_args: {},
        udp_probe_status=lambda *_args: False,
        control_lost=lambda *_args: None,
    )
    protocol = RobotControlProtocol(
        robot_id="robot-1",
        credentials=RobotCredentialStore(tmp_path),
        pairing=PairingAuthority(RobotCredentialStore(tmp_path)),
        callbacks=callbacks,
    )
    refused = await protocol.handle(ControlMessage("profile", "request-profile", {}))
    assert refused.message_type == "error"
    assert refused.payload["detail"] == "request could not be completed"
    assert "hello" not in refused.payload["detail"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda watchdog, clock: clock.advance_ms(1001), "control_heartbeat_timeout"),
        (
            lambda watchdog, clock: (
                clock.advance_ms(900),
                watchdog.mark_control(now_ns=clock()),
                clock.advance_ms(101),
            ),
            "first_action_timeout",
        ),
        (
            lambda watchdog, clock: (
                watchdog.mark_action(now_ns=clock()),
                watchdog.mark_control(now_ns=clock()),
                clock.advance_ms(201),
            ),
            "action_watchdog_timeout",
        ),
    ],
)
def test_robot_watchdog_stops_locally_once(mutate, reason: str) -> None:
    clock = FakeClock()
    stopped: list[str] = []
    watchdog = RobotLivenessWatchdog(
        stopped.append,
        clock_ns=clock,
        deadlines=WatchdogDeadlines(
            action_ns=200_000_000,
            first_action_ns=1_000_000_000,
            control_ns=1_000_000_000,
            browser_ns=2_000_000_000,
        ),
    )
    watchdog.arm()
    mutate(watchdog, clock)
    assert watchdog.poll() == reason
    assert watchdog.poll() == reason
    assert stopped == [reason]


def test_robot_watchdog_accepts_explicit_operator_and_browser_loss() -> None:
    clock = FakeClock()
    stopped: list[str] = []
    watchdog = RobotLivenessWatchdog(stopped.append, clock_ns=clock)
    watchdog.arm()
    assert watchdog.mark_control(operator_process_live=False) == "operator_process_lost"
    assert stopped == ["operator_process_lost"]
    watchdog.arm()
    assert watchdog.mark_control(browser_live=False) == "operator_browser_lost"
    assert stopped[-1] == "operator_browser_lost"


def test_udp_receiver_requires_private_exact_bind_and_authenticated_endpoint() -> None:
    with pytest.raises(UdpTransportError, match="wildcard"):
        validate_private_bind_address("0.0.0.0")
    with pytest.raises(UdpTransportError, match="loopback"):
        validate_private_bind_address("127.0.0.1")
    assert str(validate_private_bind_address("127.0.0.1", allow_loopback=True)) == "127.0.0.1"
    assert str(validate_private_bind_address("100.64.0.20")) == "100.64.0.20"

    received: list[bytes] = []
    event = threading.Event()

    def receive(payload: bytes) -> None:
        received.append(payload)
        event.set()

    receiver = UdpActionReceiver("127.0.0.1", 0, receive, allow_loopback=True)
    assert receiver.status()["listening"] is False
    key = b"k" * 32
    receiver.begin_session(session_id="session-1", executor_generation=1, key_id="key-1", key=key)
    port = receiver.start()
    proxy = UdpFaultProxy(
        "127.0.0.1",
        0,
        "127.0.0.1",
        port,
        allow_loopback=True,
        drop_numbers={2},
        delay_numbers={3},
        duplicate_numbers={4},
        corrupt_numbers={5},
    )
    proxy_port = proxy.start()
    sender = UdpSessionSender("127.0.0.1", proxy_port)
    spoof = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.prove_endpoint(
            session_id="session-1",
            executor_generation=1,
            key_id="key-1",
            key=key,
        )
        spoof.sendto(b"spoof", ("127.0.0.1", port))
        sender.send(b"dropped")
        sender.send(b"delayed")
        sender.send(b"duplicated")
        assert event.wait(1.0)
        deadline = time.monotonic() + 1
        while len(received) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [b"duplicated", b"duplicated"]
        proxy.flush_delayed()
        deadline = time.monotonic() + 1
        while len(received) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received[-1] == b"delayed"
        sender.send(b"corrupted")
        deadline = time.monotonic() + 1
        while len(received) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received[-1] == b"borrupted"
        deadline = time.monotonic() + 1
        while receiver.status()["counters"].get("endpoint_rejected") != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.status()["counters"]["endpoint_rejected"] == 1
    finally:
        spoof.close()
        sender.close()
        proxy.close()
        receiver.close()


@pytest.mark.asyncio
async def test_robot_profile_rejects_identity_mismatch_before_opening_session(tmp_path: Path) -> None:
    credentials = RobotCredentialStore(tmp_path)
    credential = credentials.issue("operator", now_ns=time.monotonic_ns())
    pairing = PairingAuthority(credentials)
    opened: list[SessionSpec] = []
    callbacks = RobotControlCallbacks(
        session_profile=lambda _credential_id: RobotSessionProfile(session_spec(), "d" * 64),
        open_session=lambda requested, *_args: opened.append(requested),
        stop_session=lambda *_args: {},
        session_status=lambda *_args: {},
        heartbeat=lambda *_args: {},
        udp_probe_status=lambda *_args: False,
        control_lost=lambda *_args: None,
    )
    protocol = RobotControlProtocol(
        robot_id="robot-1",
        credentials=credentials,
        pairing=pairing,
        callbacks=callbacks,
    )
    hello = await protocol.handle(
        ControlMessage(
            "hello",
            "request-hello",
            {
                "node_id": "operator-1",
                "credential_id": credential.credential_id,
                "credential_secret": credential.secret,
            },
        )
    )
    assert hello.payload["authenticated"] is True
    requested = session_spec()
    request_body = {
        "source_id": requested.source_id,
        "rig_id": requested.rig_id,
        "rig_digest": requested.rig_digest,
        "leader_calibration_id": requested.leader_calibration_id,
        "leader_calibration_digest": "f" * 64,
        "follower_calibration_id": requested.follower_calibration_id,
        "follower_calibration_digest": requested.follower_calibration_digest,
        "joint_names": list(requested.joint_names),
        "units": list(requested.units),
    }
    refused = await protocol.handle(
        ControlMessage(
            "session_open",
            "request-open",
            {"spec": request_body, "clock_samples": [item.to_dict() for item in clock_samples()]},
        )
    )
    assert refused.message_type == "error"
    assert "leader_calibration" in refused.payload["detail"]
    assert opened == []


def _make_test_certificate(tmp_path: Path) -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the real TLS integration test")
    certificate = tmp_path / "certificate.pem"
    private_key = tmp_path / "private-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return certificate, private_key


@pytest.mark.asyncio
async def test_two_instances_complete_pinned_tls_pair_session_heartbeat_and_stop(tmp_path: Path) -> None:
    certificate, private_key = _make_test_certificate(tmp_path)
    fingerprint = certificate_sha256_fingerprint(certificate)
    verify_certificate_fingerprint(ssl.PEM_cert_to_DER_cert(certificate.read_text()), fingerprint)

    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, private_key)
    credentials = RobotCredentialStore(tmp_path / "credentials")
    pairing = PairingAuthority(credentials)
    authority = RemoteSessionAuthority()
    stopped: list[str] = []

    def open_session(spec, mapping, _credential_id):
        grant = authority.open_session(
            spec,
            now_ns=time.monotonic_ns(),
            clock_offset_ns=mapping.robot_minus_operator_ns,
            clock_uncertainty_ns=mapping.uncertainty_ns,
        )
        return SessionOpenResult(grant, "127.0.0.1", 9000)

    def stop_session(session_id, generation, reason):
        grant = authority.grant
        assert grant is not None
        assert (session_id, generation) == (grant.session_id, grant.executor_generation)
        stopped.append(reason)
        return authority.stop(reason=reason, now_ns=time.monotonic_ns())

    callbacks = RobotControlCallbacks(
        session_profile=lambda _credential_id: RobotSessionProfile(session_spec(), "d" * 64),
        open_session=open_session,
        stop_session=stop_session,
        session_status=lambda _session, _generation: authority.snapshot(),
        heartbeat=lambda _session, _generation, _process, _browser: {"watchdog_remaining_ms": 200},
        udp_probe_status=lambda _session, _generation: True,
        control_lost=lambda _session, _generation, reason: (
            stopped.append(reason),
            authority.stop(reason=reason, now_ns=time.monotonic_ns()),
        ),
    )
    server = TlsControlServer(
        "127.0.0.1",
        0,
        tls,
        lambda: RobotControlProtocol(
            robot_id="robot-1",
            credentials=credentials,
            pairing=pairing,
            callbacks=callbacks,
        ),
        allow_loopback=True,
    )
    assert server.bound_port is None
    port = await server.start()
    payload = pairing.open_local_window(
        local_request=True,
        robot_address="127.0.0.1",
        control_port=port,
        certificate_fingerprint=fingerprint,
    )
    client = PinnedControlClient(f"wss://127.0.0.1:{port}", fingerprint)
    timeout_client = PinnedControlClient(f"wss://127.0.0.1:{port}", fingerprint)
    try:
        await client.connect()
        hello = await client.hello("operator-1")
        assert hello == {"robot_id": "robot-1", "authenticated": False, "pairing_allowed": True}
        credential = await client.pair(payload.pairing_token, "operator laptop")
        assert credentials.authenticate(credential.credential_id, credential.secret)
        profile = await client.profile()
        assert profile["follower_calibration_digest"] == DIGEST_C
        assert profile["allowed_leader_calibration_digest"] == DIGEST_B
        mapping, samples = await client.synchronize_clocks()
        assert mapping.sample_count == 16
        grant = await client.open_session(session_spec(), samples)
        assert grant.key_id.startswith("action-")
        assert await client.udp_probe_status(grant.key_id)
        _fresh_mapping, fresh_samples = await client.synchronize_clocks()
        assert await client.check_clock(fresh_samples)
        heartbeat = await client.heartbeat()
        assert heartbeat["watchdog_remaining_ms"] == 200
        receipt = await client.stop("operator_stop")
        assert receipt["reason"] == "operator_stop"
        assert stopped == ["operator_stop"]
        await client.close()

        await timeout_client.connect()
        authenticated = await timeout_client.hello("operator-2", credential)
        assert authenticated["authenticated"] is True
        _mapping, timeout_samples = await timeout_client.synchronize_clocks()
        await timeout_client.open_session(session_spec(), timeout_samples)
        deadline = time.monotonic() + 2
        while "control_heartbeat_timeout" not in stopped and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert stopped[-1] == "control_heartbeat_timeout"
        assert authority.grant is None
    finally:
        await client.close()
        await timeout_client.close()
        await server.close()


@pytest.mark.asyncio
async def test_tls_fingerprint_mismatch_sends_no_credentials(tmp_path: Path) -> None:
    certificate, private_key = _make_test_certificate(tmp_path)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, private_key)
    credentials = RobotCredentialStore(tmp_path / "credentials")
    pairing = PairingAuthority(credentials)
    callbacks = RobotControlCallbacks(
        session_profile=lambda _credential_id: RobotSessionProfile(session_spec(), "d" * 64),
        open_session=lambda *_args: pytest.fail("session must not open"),
        stop_session=lambda *_args: pytest.fail("session must not stop"),
        session_status=lambda *_args: {},
        heartbeat=lambda *_args: {},
        udp_probe_status=lambda *_args: False,
        control_lost=lambda *_args: None,
    )
    server = TlsControlServer(
        "127.0.0.1",
        0,
        tls,
        lambda: RobotControlProtocol(
            robot_id="robot-1",
            credentials=credentials,
            pairing=pairing,
            callbacks=callbacks,
        ),
        allow_loopback=True,
    )
    port = await server.start()
    client = PinnedControlClient(f"wss://127.0.0.1:{port}", "f" * 64)
    try:
        with pytest.raises(PairingError, match="fingerprint mismatch"):
            await client.connect()
        assert not client.connected
        assert not credentials.path.exists()
    finally:
        await client.close()
        await server.close()
