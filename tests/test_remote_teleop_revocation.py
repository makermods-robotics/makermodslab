from __future__ import annotations

import ssl
import threading
import time
from types import SimpleNamespace

import pytest

from makermodslab.hardware_lease import HardwareLeaseRegistry
from makermodslab.remote_teleop.calibration_identity import CalibrationIdentity
from makermodslab.remote_teleop.clock_sync import FrozenClockMapping
from makermodslab.remote_teleop.config import RemoteRoleConfigStore
from makermodslab.remote_teleop.control_protocol import ControlMessage
from makermodslab.remote_teleop.control_server import (
    RobotControlCallbacks,
    RobotControlProtocol,
    TlsControlServer,
)
from makermodslab.remote_teleop.executor import JointLimit
from makermodslab.remote_teleop.pairing import PairingAuthority, RobotCredentialStore
from makermodslab.remote_teleop.robot_service import PreparedRobotProfile, RemoteRobotService
from makermodslab.remote_teleop.simulation import SimulatedFollower

JOINTS = ("joint_a", "joint_b")


class FakeUdp:
    bound_port = 7444

    def begin_session(self, **kwargs) -> None:
        self.session = kwargs

    def stop_dispatch(self) -> None:
        pass

    def status(self):
        return {"endpoint_bound": True, "counters": {}, "listening": True}


class FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def close(self, *, code=1000, reason="") -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason


def prepared_profile() -> PreparedRobotProfile:
    limits = {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }
    return PreparedRobotProfile(
        follower_config=object(),
        follower_calibration=CalibrationIdentity("follower-cal", "c" * 64),
        leader_calibration=CalibrationIdentity("leader-cal", "b" * 64),
        rig_id="rig-1",
        rig_digest="a" * 64,
        joint_names=JOINTS,
        units=("rad", "rad"),
        limits=limits,
        limits_digest="d" * 64,
        device_identity_digest="e" * 64,
        follower_serial_binding="test-physical-adapter",
    )


def configured_service(follower_factory, tmp_path):
    registry = HardwareLeaseRegistry()
    service = RemoteRobotService(
        config_store=RemoteRoleConfigStore(tmp_path),
        registry=registry,
        follower_factory=follower_factory,
    )
    service._config = SimpleNamespace(
        recording_enabled=False,
        action_rate_hz=50,
        action_watchdog_ms=200,
        first_action_deadline_ms=1000,
        control_deadline_ms=1000,
        browser_deadline_ms=2000,
        bind_address="127.0.0.1",
        udp_port=7444,
    )
    service._profile = prepared_profile()
    service._udp = FakeUdp()
    return service, registry


def open_as(service: RemoteRobotService, credential_id: str):
    spec = prepared_profile().for_credential(credential_id).expected_spec
    return service._open_session(spec, FrozenClockMapping(0, 0, 0, 16), credential_id)


def callbacks(profile: PreparedRobotProfile) -> RobotControlCallbacks:
    return RobotControlCallbacks(
        session_profile=lambda credential_id: profile.for_credential(credential_id),
        open_session=lambda *_args: pytest.fail("revoked credential reached session open"),
        stop_session=lambda *_args: {},
        session_status=lambda *_args: {},
        heartbeat=lambda *_args: {},
        udp_probe_status=lambda *_args: False,
        control_lost=lambda *_args: None,
    )


@pytest.mark.asyncio
async def test_idle_authenticated_protocol_rechecks_revocation_and_refuses_request(tmp_path) -> None:
    credentials = RobotCredentialStore(tmp_path)
    credential = credentials.issue("operator", now_ns=time.monotonic_ns())
    protocol = RobotControlProtocol(
        robot_id="robot-1",
        credentials=credentials,
        pairing=PairingAuthority(credentials),
        callbacks=callbacks(prepared_profile()),
    )
    hello = await protocol.handle(
        ControlMessage(
            "hello",
            "hello-1",
            {
                "node_id": "operator-1",
                "credential_id": credential.credential_id,
                "credential_secret": credential.secret,
            },
        )
    )
    assert hello.payload["authenticated"] is True

    assert credentials.revoke(credential.credential_id)
    refused = await protocol.handle(ControlMessage("profile", "profile-1", {}))

    assert refused.message_type == "error"
    assert refused.payload["detail"] == "operator credential is no longer active"
    assert protocol.authorization_invalidated is True
    assert protocol.authenticated is False


@pytest.mark.asyncio
async def test_server_revocation_invalidates_and_closes_every_matching_protocol(tmp_path) -> None:
    credentials = RobotCredentialStore(tmp_path)
    credential = credentials.issue("operator", now_ns=time.monotonic_ns())
    pairing = PairingAuthority(credentials)
    protocol = RobotControlProtocol(
        robot_id="robot-1",
        credentials=credentials,
        pairing=pairing,
        callbacks=callbacks(prepared_profile()),
    )
    await protocol.handle(
        ControlMessage(
            "hello",
            "hello-1",
            {
                "node_id": "operator-1",
                "credential_id": credential.credential_id,
                "credential_secret": credential.secret,
            },
        )
    )
    websocket = FakeWebSocket()
    server = TlsControlServer(
        "127.0.0.1",
        0,
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
        lambda: protocol,
        allow_loopback=True,
    )
    server._connections[protocol] = websocket

    assert credentials.revoke(credential.credential_id)
    assert await server.revoke_credential(credential.credential_id) == 1

    assert protocol.authorization_invalidated is True
    assert websocket.closed is True
    assert websocket.close_code == 1008
    assert websocket.close_reason == "operator credential revoked"


def test_revoking_active_credential_stops_robot_locally_and_releases_lease(tmp_path) -> None:
    follower = SimulatedFollower(JOINTS)
    service, registry = configured_service(lambda *_args: follower, tmp_path)
    credential = service.credentials.issue("operator", now_ns=time.monotonic_ns())
    open_as(service, credential.credential_id)

    assert service.revoke_credential(credential.credential_id) is True

    assert service.credentials.is_active(credential.credential_id) is False
    assert service._executor is None
    assert service.status()["last_stop"]["reason"] == "operator_credential_revoked"
    assert follower.stop_reasons == ["operator_credential_revoked"]
    assert registry.snapshot().state == "idle"


def test_revocation_racing_blocked_follower_start_cannot_publish_authority(tmp_path) -> None:
    follower = SimulatedFollower(JOINTS)
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(*_args):
        factory_started.set()
        assert release_factory.wait(2.0)
        return follower

    service, registry = configured_service(blocking_factory, tmp_path)
    credential = service.credentials.issue("operator", now_ns=time.monotonic_ns())
    open_errors: list[Exception] = []
    revoke_results: list[bool] = []

    def run_open() -> None:
        try:
            open_as(service, credential.credential_id)
        except Exception as exc:
            open_errors.append(exc)

    opener = threading.Thread(target=run_open)
    opener.start()
    assert factory_started.wait(1.0)
    revoker = threading.Thread(
        target=lambda: revoke_results.append(service.revoke_credential(credential.credential_id))
    )
    revoker.start()

    deadline = time.monotonic() + 1.0
    while service._opening is not None and not service._opening.cancel.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.005)
    release_factory.set()
    opener.join(timeout=2.0)
    revoker.join(timeout=2.0)

    assert not opener.is_alive() and not revoker.is_alive()
    assert revoke_results == [True]
    assert open_errors
    assert service.credentials.is_active(credential.credential_id) is False
    assert service._executor is None
    assert service._owner_credential_id is None
    assert service.status()["last_stop"]["reason"] == "operator_credential_revoked"
    assert follower.stop_reasons == ["session_open_failed"]
    assert registry.snapshot().state == "idle"
