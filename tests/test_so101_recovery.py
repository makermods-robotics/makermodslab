from __future__ import annotations

import pytest
from pydantic import ValidationError

from makermodslab.api_errors import ApiError
from makermodslab.hardware_lease import (
    HardwareRecoveryIdentity,
    hardware_lease_registry,
)
from makermodslab.so101_recovery import (
    ReleaseSo101TorqueRequest,
    handle_release_so101_torque,
)


class FakeFeetechBus:
    motors = {"shoulder_pan": object(), "gripper": object()}

    def __init__(
        self,
        *,
        readbacks: dict[str, int] | None = None,
        close_fails: bool = False,
    ) -> None:
        self.readbacks = readbacks or dict.fromkeys(self.motors, 0)
        self.close_fails = close_fails
        self.calls: list[tuple[object, ...]] = []

    def connect(self, *, handshake: bool = True) -> None:
        self.calls.append(("connect", handshake))

    def write(
        self,
        register: str,
        motor: str,
        value: int,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        self.calls.append(("write", register, motor, value, normalize, num_retry))

    def read(
        self,
        register: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> int:
        self.calls.append(("read", register, motor, normalize, num_retry))
        return self.readbacks[motor]

    def disconnect(self, *, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))
        if self.close_fails:
            raise OSError("injected close failure")


def binding(port: str) -> str:
    return "binding:" + port


def identity(*ports: str, bindings: dict[str, str] | None = None) -> HardwareRecoveryIdentity:
    return HardwareRecoveryIdentity.from_bound_targets(
        "so101_recovery",
        "so101",
        bindings or {port: binding(port) for port in ports},
    )


def install_unresolved(*ports: str) -> None:
    token = hardware_lease_registry.claim(
        "teleoperation",
        "test-owner",
        recovery=identity(*ports),
    )
    hardware_lease_registry.mark_unresolved(token, "injected crash")


@pytest.fixture(autouse=True)
def fake_serial_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import so101_recovery

    monkeypatch.setattr(
        so101_recovery,
        "resolve_serial_port_bindings",
        lambda ports: {port: binding(port) for port in ports},
    )
    monkeypatch.setattr(so101_recovery, "opened_serial_binding_matches", lambda _bus, _binding: True)


@pytest.mark.parametrize("ports", [[], [""], [" /dev/a ", "/dev/a"]])
def test_request_requires_an_exact_nonempty_unique_port_list(ports: list[str]) -> None:
    with pytest.raises(ValidationError):
        ReleaseSo101TorqueRequest(ports=ports)


def test_matching_multiport_recovery_disables_verifies_and_closes_every_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    ports = ("/dev/follower", "/dev/leader")
    install_unresolved(*ports)
    buses = {port: FakeFeetechBus() for port in ports}
    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", buses.__getitem__)

    result = handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=list(ports)))

    assert result == {
        "success": True,
        "message": "Every requested SO-101 bus confirmed torque disabled and closed.",
        "confirmed_ports": 2,
        "problems": [],
    }
    assert hardware_lease_registry.snapshot().held is False
    for bus in buses.values():
        assert bus.calls[0] == ("connect", False)
        for motor in bus.motors:
            assert ("write", "Torque_Enable", motor, 0, False, 5) in bus.calls
            assert ("read", "Torque_Enable", motor, False, 5) in bus.calls
        assert bus.calls[-1] == ("disconnect", False)


def test_restart_can_use_owner_private_retained_targets_without_reentry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery
    from makermodslab.hardware_lease import HardwareLeaseRegistry

    ports = ("/dev/follower-by-id", "/dev/leader-by-id")
    journal = tmp_path / "hardware-lease.json"
    first_process = HardwareLeaseRegistry(journal_path=journal)
    first_process.claim("teleoperation", "lost", recovery=identity(*ports))
    restarted = HardwareLeaseRegistry(journal_path=journal)
    buses = {port: FakeFeetechBus() for port in ports}
    monkeypatch.setattr(so101_recovery, "hardware_lease_registry", restarted)
    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", buses.__getitem__)

    result = handle_release_so101_torque(ReleaseSo101TorqueRequest())

    assert result["success"] is True
    assert restarted.snapshot().state == "idle"
    assert journal.exists() is False


def test_wrong_adapter_at_retained_path_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    port = "/dev/follower"
    install_unresolved(port)
    opened: list[str] = []
    monkeypatch.setattr(
        so101_recovery,
        "resolve_serial_port_bindings",
        lambda ports: dict.fromkeys(ports, "replacement-adapter"),
    )
    monkeypatch.setattr(
        so101_recovery,
        "_build_feetech_bus",
        lambda path: opened.append(path),
    )

    with pytest.raises(ApiError):
        handle_release_so101_torque(ReleaseSo101TorqueRequest())

    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_same_adapter_at_new_explicit_path_can_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    old_path = "/dev/old-follower"
    new_path = "/dev/new-follower"
    stable_binding = "same-physical-adapter"
    token = hardware_lease_registry.claim(
        "teleoperation",
        "test-owner",
        recovery=identity(old_path, bindings={old_path: stable_binding}),
    )
    hardware_lease_registry.mark_unresolved(token, "injected crash")
    bus = FakeFeetechBus()
    monkeypatch.setattr(
        so101_recovery,
        "resolve_serial_port_bindings",
        lambda ports: dict.fromkeys(ports, stable_binding),
    )
    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", lambda path: bus)

    result = handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=[new_path]))

    assert result["success"] is True
    assert hardware_lease_registry.snapshot().held is False


def test_adapter_swap_after_resolution_writes_nothing_and_retains_lockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    port = "/dev/follower"
    install_unresolved(port)
    bus = FakeFeetechBus()
    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", lambda _path: bus)
    monkeypatch.setattr(so101_recovery, "opened_serial_binding_matches", lambda _bus, _binding: False)

    result = handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=[port]))

    assert result["success"] is False
    assert hardware_lease_registry.snapshot().state == "unresolved"
    assert bus.calls == [("connect", False), ("disconnect", False)]


def test_missing_usb_serial_never_opens_or_auto_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    install_unresolved("/dev/follower")
    opened: list[str] = []
    monkeypatch.setattr(so101_recovery, "resolve_serial_port_bindings", lambda _ports: {})
    monkeypatch.setattr(
        so101_recovery,
        "_build_feetech_bus",
        lambda path: opened.append(path),
    )

    with pytest.raises(ApiError, match="stable USB serial"):
        handle_release_so101_torque(ReleaseSo101TorqueRequest())

    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_port_identity_mismatch_opens_nothing_and_retains_lockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    install_unresolved("/dev/follower", "/dev/leader")
    opened: list[str] = []
    monkeypatch.setattr(
        so101_recovery,
        "_build_feetech_bus",
        lambda port: opened.append(port),
    )

    with pytest.raises(ApiError) as exc_info:
        handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=["/dev/follower"]))

    assert exc_info.value.status_code == 409
    assert opened == []
    assert hardware_lease_registry.snapshot().state == "unresolved"


def test_bus_open_error_retains_lockout_without_exposing_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    private_port = "/dev/private-follower"
    install_unresolved(private_port)

    def fail_open(_port: str):
        raise ConnectionError(f"failed to open {private_port}")

    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", fail_open)
    result = handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=[private_port]))

    assert result["success"] is False
    assert result["problems"] == ["port[0] open: ConnectionError"]
    assert private_port not in str(result)
    assert private_port not in str(hardware_lease_registry.snapshot())
    assert hardware_lease_registry.snapshot().state == "unresolved"


@pytest.mark.parametrize(
    ("readbacks", "close_fails"),
    [({"shoulder_pan": 0, "gripper": 1}, False), ({"shoulder_pan": 0, "gripper": 0}, True)],
)
def test_unverified_torque_or_close_failure_retains_durable_lockout(
    monkeypatch: pytest.MonkeyPatch,
    readbacks: dict[str, int],
    close_fails: bool,
) -> None:
    from makermodslab import so101_recovery

    install_unresolved("/dev/follower")
    bus = FakeFeetechBus(readbacks=readbacks, close_fails=close_fails)
    monkeypatch.setattr(so101_recovery, "_build_feetech_bus", lambda _port: bus)

    result = handle_release_so101_torque(ReleaseSo101TorqueRequest(ports=["/dev/follower"]))

    assert result["success"] is False
    assert result["confirmed_ports"] == 0
    assert hardware_lease_registry.snapshot().state == "unresolved"
    assert bus.calls[-1] == ("disconnect", False)


def test_versioned_route_is_typed_and_does_not_echo_raw_ports(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import server

    captured: list[ReleaseSo101TorqueRequest] = []

    def fake_handler(request: ReleaseSo101TorqueRequest):
        captured.append(request)
        return {
            "success": True,
            "message": "safe",
            "confirmed_ports": 1,
            "problems": [],
        }

    monkeypatch.setattr(server, "handle_release_so101_torque", fake_handler)
    response = client.post(
        "/api/v1/arms/so101/recover-torque",
        json={"ports": ["/dev/private-follower"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "safe",
        "confirmed_ports": 1,
        "problems": [],
    }
    assert captured[0].ports == ["/dev/private-follower"]
    assert "/dev/private-follower" not in response.text


def test_identity_mismatch_error_never_echoes_owner_path_or_usb_serial(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import so101_recovery

    private_path = "/private/owner/follower"
    private_serial = "SECRET-USB-SERIAL"
    token = hardware_lease_registry.claim(
        "teleoperation",
        f"local:{private_path}",
        recovery=identity(private_path, bindings={private_path: private_serial}),
    )
    hardware_lease_registry.mark_unresolved(token, "injected crash")
    monkeypatch.setattr(
        so101_recovery,
        "resolve_serial_port_bindings",
        lambda ports: dict.fromkeys(ports, "different-adapter"),
    )

    response = client.post(
        "/api/v1/arms/so101/recover-torque",
        json={"ports": [private_path]},
    )

    assert response.status_code == 409
    assert private_path not in response.text
    assert private_serial not in response.text
