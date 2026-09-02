"""Exact, multi-target CAN crash-recovery proofs."""

from __future__ import annotations

import hashlib
import logging

import pytest
from pydantic import ValidationError

from makermodslab.api_errors import ApiError
from makermodslab.can_recovery import (
    ReleaseCanTorqueRequest,
    handle_release_can_torque,
)
from makermodslab.hardware_lease import (
    HardwareLeaseRegistry,
    HardwareRecoveryIdentity,
    hardware_lease_registry,
)


class FakeCanBus:
    def __init__(
        self,
        *,
        acknowledge: set[str] | None = None,
        close_fails: bool = False,
    ) -> None:
        self.motors = {"joint_1": object(), "joint_2": object()}
        self.is_connected = False
        self.acknowledge = set(self.motors) if acknowledge is None else acknowledge
        self.close_fails = close_fails
        self.calls: list[tuple[object, ...]] = []

    def _get_motor_recv_id(self, motor: str) -> int:
        return {"joint_1": 0x11, "joint_2": 0x12}[motor]

    def _recv_motor_response(self, expected_recv_id=None, timeout=0.001):
        motor = {0x11: "joint_1", 0x12: "joint_2"}.get(expected_recv_id)
        return object() if motor in self.acknowledge else None

    def connect(self, *, handshake: bool = True) -> None:
        self.calls.append(("connect", handshake))
        self.is_connected = True

    def disable_torque(self) -> None:
        self.calls.append(("disable_torque",))
        for motor in self.motors:
            self._recv_motor_response(expected_recv_id=self._get_motor_recv_id(motor))

    def disconnect(self, *, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))
        self.is_connected = False
        if self.close_fails:
            raise OSError("injected close failure on /dev/private")


class FakeFollower:
    def __init__(self, bus: FakeCanBus) -> None:
        self.bus = bus


def _test_binding(port: str) -> str:
    return hashlib.sha256(f"test-can-binding-v1\0{port}".encode()).hexdigest()


def identity(family: str, *ports: str) -> HardwareRecoveryIdentity:
    return HardwareRecoveryIdentity.from_bound_targets(
        "can_recovery",
        family,
        {port: _test_binding(port) for port in ports},
    )


def install_unresolved(
    family: str,
    *ports: str,
    kind: str = "teleoperation",
    owner: str = "lost-owner",
) -> None:
    token = hardware_lease_registry.claim(
        kind,
        owner,
        recovery=identity(family, *ports),
    )
    hardware_lease_registry.mark_unresolved(token, "injected crash")


@pytest.fixture(autouse=True)
def stable_can_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import can_recovery

    monkeypatch.setattr(
        can_recovery,
        "resolve_can_port_bindings",
        lambda ports: {port: _test_binding(port) for port in ports},
    )


@pytest.mark.parametrize(
    "body",
    [
        {"arm_type": "metal", "port": ""},
        {"arm_type": "metal", "ports": []},
        {"arm_type": "metal", "ports": ["/dev/a", " /dev/a "]},
        {"arm_type": "metal", "port": "/dev/a", "ports": ["/dev/a"]},
    ],
)
def test_request_rejects_ambiguous_or_invalid_explicit_targets(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ReleaseCanTorqueRequest.model_validate(body)


def test_matching_multi_target_recovery_requires_ack_and_close_on_every_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    ports = ("/dev/private-left", "/dev/private-right")
    install_unresolved("metal", *ports)
    buses = {port: FakeCanBus() for port in ports}
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: FakeFollower(buses[port]),
    )

    result = handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="metal", ports=list(ports)))

    assert result["success"] is True
    assert result["problems"] == []
    assert all(port not in str(result) for port in ports)
    assert hardware_lease_registry.snapshot().held is False
    for bus in buses.values():
        assert bus.calls == [("connect", False), ("disable_torque",), ("disconnect", False)]


def test_omitted_targets_use_owner_private_retained_map_after_restart(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    ports = ("/dev/can-left", "/dev/can-right")
    journal = tmp_path / "hardware-lease.json"
    first = HardwareLeaseRegistry(journal_path=journal)
    first.claim("inference", "lost", recovery=identity("maker", *ports))
    restarted = HardwareLeaseRegistry(journal_path=journal)
    buses = {port: FakeCanBus() for port in ports}
    monkeypatch.setattr(can_recovery, "hardware_lease_registry", restarted)
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: FakeFollower(buses[port]),
    )

    result = handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="maker"))

    assert result["success"] is True
    assert restarted.snapshot().state == "idle"
    assert journal.exists() is False


def test_missing_retained_targets_requires_exact_reentry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    journal = tmp_path / "hardware-lease.json"
    first = HardwareLeaseRegistry(journal_path=journal)
    first.claim("inference", "lost", recovery=identity("metal", "/dev/private"))
    journal.with_name("hardware-lease-targets.json").unlink()
    restarted = HardwareLeaseRegistry(journal_path=journal)
    opened: list[str] = []
    monkeypatch.setattr(can_recovery, "hardware_lease_registry", restarted)
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: opened.append(port),
    )

    with pytest.raises(ApiError, match="exact ports"):
        handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="metal"))

    assert opened == []
    assert restarted.snapshot().state == "unresolved"


def test_identity_mismatch_opens_nothing_and_retains_lockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    install_unresolved("metal", "/dev/left", "/dev/right")
    opened: list[str] = []
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: opened.append(port),
    )

    with pytest.raises(ApiError):
        handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="metal", ports=["/dev/left"]))

    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_rebound_path_to_replacement_same_family_bus_never_clears_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    port = "/dev/rebound-can"
    token = hardware_lease_registry.claim(
        "teleoperation",
        "lost-owner",
        recovery=HardwareRecoveryIdentity.from_targets(
            "can_physical_recovery",
            "metal",
            port,
        ),
    )
    hardware_lease_registry.mark_unresolved(token, "injected crash")
    opened: list[str] = []
    monkeypatch.setattr(
        can_recovery,
        "resolve_can_port_bindings",
        lambda ports: dict.fromkeys(ports, "replacement-adapter-binding"),
    )
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, target: opened.append(target),
    )

    with pytest.raises(ApiError, match="physical/manual recovery"):
        handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="metal", port=port))

    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_adapter_swap_after_open_skips_disable_and_retains_lockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    port = "/dev/can-swap"
    install_unresolved("maker", port)
    bus = FakeCanBus()
    calls = 0

    def changing_binding(ports):
        nonlocal calls
        calls += 1
        binding = _test_binding(port) if calls == 1 else "replacement-after-open"
        return dict.fromkeys(ports, binding)

    monkeypatch.setattr(can_recovery, "resolve_can_port_bindings", changing_binding)
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, _target: FakeFollower(bus),
    )

    result = handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="maker", port=port))

    assert result["success"] is False
    assert any("identity changed" in problem for problem in result["problems"])
    assert ("disable_torque",) not in bus.calls
    assert bus.calls == [("connect", False), ("disconnect", False)]
    assert hardware_lease_registry.snapshot().state == "unresolved"


@pytest.mark.parametrize("kind", ["calibration", "diagnostic"])
def test_ambiguous_star_uart_or_mixed_claim_never_opens_a_can_follower(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import can_recovery

    private_port = "/dev/ambiguous-target"
    install_unresolved("maker", private_port, kind=kind)
    opened: list[str] = []
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: opened.append(port),
    )

    with pytest.raises(ApiError, match="does not prove"):
        handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="maker", port=private_port))

    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_ambiguous_claim_409_never_exposes_path_like_owner(client) -> None:
    private_owner = "calibration:/Users/private/devices/tty-secret"
    private_port = "/dev/ambiguous-target"
    install_unresolved("maker", private_port, kind="calibration", owner=private_owner)

    response = client.post(
        "/api/v1/arms/release-torque",
        json={"arm_type": "maker", "port": private_port},
    )

    assert response.status_code == 409
    assert private_owner not in response.text
    assert "/Users/private" not in response.text
    assert "tty-secret" not in response.text


def test_one_failed_target_does_not_skip_others_or_expose_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from makermodslab import can_recovery

    ports = ("/dev/secret-left", "/dev/secret-right")
    install_unresolved("metal", *ports)
    buses = {
        ports[0]: FakeCanBus(acknowledge={"joint_1"}),
        ports[1]: FakeCanBus(close_fails=True),
    }
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, port: FakeFollower(buses[port]),
    )

    with caplog.at_level(logging.INFO, logger="makermodslab.can_recovery"):
        result = handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="metal", ports=list(ports)))

    assert result["success"] is False
    assert hardware_lease_registry.snapshot().state == "unresolved"
    assert all(("disable_torque",) in bus.calls for bus in buses.values())
    assert all(port not in str(result) for port in ports)
    assert all(port not in caplog.text for port in ports)
    assert any("acknowledged" in problem for problem in result["problems"])
    assert any("close" in problem for problem in result["problems"])


def test_legacy_single_port_request_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import can_recovery

    bus = FakeCanBus()
    monkeypatch.setattr(
        can_recovery,
        "_build_follower_device",
        lambda _family, _port: FakeFollower(bus),
    )

    result = handle_release_can_torque(ReleaseCanTorqueRequest(arm_type="maker", port=" /dev/legacy "))

    assert result["success"] is True
    assert "/dev/legacy" not in str(result)


def test_versioned_route_accepts_multi_target_request_without_echoing_paths(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import server

    captured: list[ReleaseCanTorqueRequest] = []

    def fake_handler(request: ReleaseCanTorqueRequest):
        captured.append(request)
        return {"success": True, "message": "safe", "problems": []}

    monkeypatch.setattr(server, "handle_release_can_torque", fake_handler)
    response = client.post(
        "/api/v1/arms/release-torque",
        json={"arm_type": "metal", "ports": ["/dev/private-left", "/dev/private-right"]},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "safe", "problems": []}
    assert captured[0].port is None
    assert captured[0].ports == ["/dev/private-left", "/dev/private-right"]
    assert "/dev/private" not in response.text
