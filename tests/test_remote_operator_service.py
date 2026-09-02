from __future__ import annotations

import base64
import threading
import time
from types import SimpleNamespace

from makermodslab.hardware_lease import HardwareLeaseRegistry
from makermodslab.remote_teleop.adapters.common import RawLeaderSample
from makermodslab.remote_teleop.clock_sync import FrozenClockMapping
from makermodslab.remote_teleop.contracts import decode_action
from makermodslab.remote_teleop.control_client import OperatorSessionGrant
from makermodslab.remote_teleop.operator_service import RemoteOperatorService
from makermodslab.remote_teleop.pairing import IssuedCredential

JOINTS = ("joint_a", "joint_b")
KEY = b"k" * 32


def public_profile() -> dict[str, object]:
    return {
        "source_id": "credential-1",
        "rig_id": "rig-1",
        "rig_digest": "a" * 64,
        "follower_calibration_id": "follower-cal",
        "follower_calibration_digest": "c" * 64,
        "allowed_leader_calibration_id": "leader-cal",
        "allowed_leader_calibration_digest": "b" * 64,
        "joint_names": list(JOINTS),
        "units": ["rad", "rad"],
        "limits_digest": "d" * 64,
    }


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.stop_reasons: list[str] = []
        self.heartbeat_callback = None

    async def connect(self) -> None:
        pass

    async def hello(self, node_id, credential=None):
        return {
            "robot_id": "robot-1",
            "authenticated": credential is not None,
            "pairing_allowed": True,
        }

    async def pair(self, token, label):
        return IssuedCredential("credential-1", base64.urlsafe_b64encode(KEY).decode(), label)

    async def synchronize_clocks(self):
        return FrozenClockMapping(0, 0, 0, 16), [object()] * 16

    async def profile(self):
        return public_profile()

    async def open_session(self, spec, samples):
        return OperatorSessionGrant(
            "session-1",
            1,
            "key-1",
            base64.b64encode(KEY).decode(),
            "127.0.0.1",
            7444,
            FrozenClockMapping(0, 0, 0, 16),
        )

    async def udp_probe_status(self, key_id):
        return True

    def start_heartbeats(self, *, browser_live, on_failure):
        self.heartbeat_callback = browser_live

    async def stop(self, reason):
        self.stop_reasons.append(reason)
        return {"state": "idle", "safety": {"torque_off_confirmed": True}}

    async def close(self):
        self.closed = True


class FakeLeader:
    joint_schema = SimpleNamespace(action_keys=JOINTS, units=("rad", "rad"))

    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def read(self) -> RawLeaderSample:
        return RawLeaderSample({"joint_a": 0.25, "joint_b": -0.25}, time.monotonic_ns())

    def close(self) -> None:
        self.closed = True


class FakeSender:
    def __init__(self) -> None:
        self.proven = False
        self.sent: list[bytes] = []
        self.closed = False

    def prove_endpoint(self, **kwargs) -> None:
        self.proven = True

    def send(self, payload: bytes) -> None:
        assert self.proven
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


def test_operator_owns_only_leader_encodes_actions_and_reports_stop(tmp_path) -> None:
    client = FakeClient()
    leader = FakeLeader()
    sender = FakeSender()
    registry = HardwareLeaseRegistry()
    service = RemoteOperatorService(
        registry=registry,
        config_store=SimpleNamespace(root=tmp_path),
        client_factory=lambda *_args: client,
        leader_factory=lambda *_args: leader,
        sender_factory=lambda *_args: sender,
        leader_port_resolver=lambda _config: "/dev/test-leader",
    )
    config = SimpleNamespace(
        control_uri="wss://127.0.0.1:7443",
        certificate_fingerprint="a" * 64,
        node_id="operator-1",
        robot_id="robot-1",
        action_rate_hz=50,
    )
    credential = IssuedCredential("credential-1", base64.urlsafe_b64encode(KEY).decode(), "operator")
    service._browser_last_ns = time.monotonic_ns()
    status = service._loop.submit(service._start_async(config, credential))
    assert status["state"] == "streaming"
    deadline = time.monotonic() + 1
    while not sender.sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sender.sent
    sample = decode_action(sender.sent[-1], key_lookup=lambda _key_id: KEY)
    assert sample.positions == (0.25, -0.25)
    assert registry.snapshot().kind == "remote_operator"

    receipt = service.stop("operator_stop")
    assert receipt["robot_confirmation_available"] is True
    assert receipt["leader_closed"] is True
    assert receipt["lease_released"] is True
    assert client.stop_reasons == ["operator_stop"]
    assert registry.snapshot().state == "idle"
    service.shutdown()


def test_browser_budget_starts_after_slow_negotiation(tmp_path) -> None:
    now = [0]

    class SlowClient(FakeClient):
        async def synchronize_clocks(self):
            now[0] += 5_000_000_000
            return await super().synchronize_clocks()

    client = SlowClient()
    service = RemoteOperatorService(
        registry=HardwareLeaseRegistry(),
        config_store=SimpleNamespace(root=tmp_path),
        client_factory=lambda *_args: client,
        leader_factory=lambda *_args: FakeLeader(),
        sender_factory=lambda *_args: FakeSender(),
        leader_port_resolver=lambda _config: "/dev/test-leader",
        clock_ns=lambda: now[0],
    )
    config = SimpleNamespace(
        control_uri="wss://127.0.0.1:7443",
        certificate_fingerprint="a" * 64,
        node_id="operator-1",
        robot_id="robot-1",
        action_rate_hz=50,
    )
    credential = IssuedCredential("credential-1", base64.urlsafe_b64encode(KEY).decode(), "operator")
    service._browser_last_ns = now[0]
    service._loop.submit(service._start_async(config, credential))

    assert client.heartbeat_callback is not None
    assert client.heartbeat_callback() is True
    service.stop("test_complete")
    service.shutdown()


def test_stop_cancels_in_progress_leader_start_before_streaming(tmp_path) -> None:
    connect_started = threading.Event()
    release_connect = threading.Event()

    class BlockingLeader(FakeLeader):
        def connect(self) -> None:
            connect_started.set()
            assert release_connect.wait(2.0)
            super().connect()

    client = FakeClient()
    leader = BlockingLeader()
    registry = HardwareLeaseRegistry()
    config = SimpleNamespace(
        control_uri="wss://127.0.0.1:7443",
        certificate_fingerprint="a" * 64,
        node_id="operator-1",
        robot_id="robot-1",
        action_rate_hz=50,
    )
    service = RemoteOperatorService(
        registry=registry,
        config_store=SimpleNamespace(root=tmp_path),
        client_factory=lambda *_args: client,
        leader_factory=lambda *_args: leader,
        sender_factory=lambda *_args: FakeSender(),
        leader_port_resolver=lambda _config: "/dev/test-leader",
    )
    service._load_config = lambda: config
    service.vault.put(
        config.robot_id,
        IssuedCredential("credential-1", base64.urlsafe_b64encode(KEY).decode(), "operator"),
    )

    started: list[dict[str, object]] = []
    stopped: list[dict[str, object]] = []
    starter = threading.Thread(target=lambda: started.append(service.start()))
    starter.start()
    assert connect_started.wait(1.0)
    stopper = threading.Thread(target=lambda: stopped.append(service.stop("operator_local_stop")))
    stopper.start()
    deadline = time.monotonic() + 1.0
    while service._starting is not None and not service._starting.cancel.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.005)
    release_connect.set()
    starter.join(timeout=2.0)
    stopper.join(timeout=2.0)

    assert not starter.is_alive() and not stopper.is_alive()
    assert started[0]["state"] == "idle"
    assert stopped[0]["state"] == "idle"
    assert service.status()["live_hardware_enabled"] is False
    assert leader.closed is True
    assert client.closed is True
    assert registry.snapshot().state == "idle"
    service.shutdown()
