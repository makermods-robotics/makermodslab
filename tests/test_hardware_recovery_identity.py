"""Recovery identities attached to every ordinary hardware claim."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from makermodslab.hardware_lease import HardwareRecoveryIdentity
from makermodslab.hardware_recovery_identity import (
    _socketcan_binding,
    hardware_recovery_identity,
    resolve_can_port_bindings,
)

_CLAIM_MODULES = (
    "teleoperate.py",
    "record.py",
    "replay.py",
    "rollout.py",
    "calibrate.py",
    "auto_calibrate.py",
    "zero_calibrate.py",
    "wiggle.py",
    "identify.py",
    "maker_ports.py",
    "motor_power.py",
)


def test_ordinary_so101_identity_is_physical_only_without_universal_post_open_proof() -> None:
    ports = ("/dev/follower", "/dev/leader")
    bindings = {port: f"stable-binding-{index}" for index, port in enumerate(ports)}
    identity = hardware_recovery_identity(
        "so101",
        target_ports=(ports[0],),
        feetech_ports=(ports[1],),
        binding_resolver=lambda _ports: bindings,
    )
    assert identity == HardwareRecoveryIdentity.from_targets(
        "so101_physical_recovery",
        "so101",
        *ports,
    )
    public = identity.public()
    assert public["recovery_kind"] == "so101_physical_recovery"
    assert public["arm_family"] == "so101"
    assert all(port not in str(public) for port in ports)


def test_so101_without_complete_usb_identity_uses_nonautomatic_recovery_kind() -> None:
    identity = hardware_recovery_identity(
        "so101",
        target_ports=("/dev/follower",),
        binding_resolver=lambda _ports: {},
    )

    assert identity.recovery_kind == "so101_physical_recovery"


@pytest.mark.parametrize("family", ["maker", "metal"])
def test_can_identity_is_physical_only_even_with_preopen_adapter_identity(family: str) -> None:
    ports = ("/dev/can-left", "/dev/can-right")
    bindings = {port: f"stable-can-binding-{index}" for index, port in enumerate(ports)}
    identity = hardware_recovery_identity(
        family,
        target_ports=ports,
        feetech_ports=("/dev/star-leader",),
        binding_resolver=lambda _ports: bindings,
    )
    assert identity == HardwareRecoveryIdentity.from_targets(
        "can_physical_recovery",
        family,
        *ports,
    )
    assert "/dev" not in str(identity.public())


@pytest.mark.parametrize("family", ["maker", "metal"])
def test_can_without_complete_adapter_identity_requires_physical_recovery(family: str) -> None:
    identity = hardware_recovery_identity(
        family,
        target_ports=("/dev/can-left", "/dev/can-right"),
        binding_resolver=lambda _ports: {"/dev/can-left": "only-one-binding"},
    )

    assert identity.recovery_kind == "can_physical_recovery"
    assert "/dev" not in str(identity.public())


def test_can_binding_resolution_covers_serial_and_usb_backed_socketcan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab import hardware_recovery_identity as recovery_identity

    monkeypatch.setattr(
        recovery_identity,
        "_socketcan_binding",
        lambda interface: "socketcan-usb-binding" if interface == "can0" else None,
    )

    bindings = resolve_can_port_bindings(
        ("/dev/ttyUSB0", "can0", "vcan0"),
        serial_resolver=lambda _ports: {"/dev/ttyUSB0": "serial-can-usb-binding"},
    )

    assert bindings == {
        "/dev/ttyUSB0": "serial-can-usb-binding",
        "can0": "socketcan-usb-binding",
    }


def test_socketcan_binding_rejects_path_traversal_without_sysfs_lookup() -> None:
    assert _socketcan_binding("../can0") is None


def test_every_non_remote_production_claim_supplies_typed_recovery_identity() -> None:
    package = Path(__file__).parents[1] / "makermodslab"
    claims = 0
    for relative in _CLAIM_MODULES:
        tree = ast.parse((package / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "claim" or not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "hardware_lease_registry":
                continue
            claims += 1
            recovery = next((kw.value for kw in node.keywords if kw.arg == "recovery"), None)
            assert isinstance(recovery, ast.Call), f"{relative} claim omits recovery identity"
            assert isinstance(recovery.func, ast.Name)
            assert recovery.func.id == "hardware_recovery_identity"
    assert claims == 13
