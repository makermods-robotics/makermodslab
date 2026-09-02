"""Typed, path-sanitized recovery identities for ordinary hardware claims."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from .hardware_lease import HardwareRecoveryIdentity
from .serial_port_identity import SerialBindingResolver, resolve_serial_port_bindings
from .utils.config import normalize_arm_type


def _read_sysfs_identity(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _socketcan_binding(interface: str) -> str | None:
    """Return an opaque USB-backed socketCAN identity or refuse to guess."""
    if not sys.platform.startswith("linux") or not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
        return None
    try:
        current = (Path("/sys/class/net") / interface / "device").resolve(strict=True)
    except OSError:
        return None
    for parent in (current, *current.parents):
        serial = _read_sysfs_identity(parent / "serial")
        vendor = _read_sysfs_identity(parent / "idVendor")
        product = _read_sysfs_identity(parent / "idProduct")
        if serial and vendor and product:
            material = f"makermodslab-usb-socketcan-binding-v1\0{vendor.lower()}\0{product.lower()}\0{serial}"
            return hashlib.sha256(material.encode("utf-8")).hexdigest()
        if parent == Path("/sys"):
            break
    return None


def resolve_can_port_bindings(
    ports: Iterable[str],
    *,
    serial_resolver: SerialBindingResolver = resolve_serial_port_bindings,
) -> dict[str, str]:
    """Resolve stable serial-CAN/socketCAN identities without opening a bus.

    Serial adapters use USB VID/PID/serial enumeration. Linux socketCAN is
    accepted only when its backing sysfs device exposes the same three stable
    fields. Virtual/native/reusable interface names are deliberately omitted.
    """
    targets = tuple(sorted({port.strip() for port in ports if isinstance(port, str) and port.strip()}))
    bindings = serial_resolver(targets)
    for target in targets:
        if target in bindings or target.startswith("/dev/"):
            continue
        binding = _socketcan_binding(target)
        if binding is not None:
            bindings[target] = binding
    return bindings


def can_hardware_recovery_identity(
    ports: Iterable[str],
    arm_family: str,
    *,
    binding_resolver: SerialBindingResolver = resolve_can_port_bindings,
) -> HardwareRecoveryIdentity:
    """Require physical recovery until the opened CAN handle can attest identity.

    USB enumeration can identify the adapter currently assigned to a path or
    socketCAN interface, but it cannot prove that the already-open driver
    handle belongs to that adapter across an adversarial hot-swap race. Keep
    the pre-open resolver available for explicit checks without minting crash
    recovery authority from it.
    """
    targets = tuple(sorted({port.strip() for port in ports if isinstance(port, str) and port.strip()}))
    if not targets:
        raise ValueError("CAN recovery requires at least one target")
    # Deliberately do not use ``binding_resolver`` to authorize recovery. A
    # trustworthy implementation must attest through the opened handle, not
    # merely re-enumerate the name before or after open.
    return HardwareRecoveryIdentity.from_targets(
        "can_physical_recovery",
        arm_family,
        *targets,
    )


def so101_hardware_recovery_identity(
    ports: Iterable[str],
    *,
    recovery_kind: str = "so101_recovery",
    unbound_recovery_kind: str = "so101_physical_recovery",
    profile_digest: str | None = None,
    binding_resolver: SerialBindingResolver = resolve_serial_port_bindings,
) -> HardwareRecoveryIdentity:
    """Use physical adapter bindings when every supplied SO-101 port has one.

    A missing USB serial or VID/PID does not prevent clean operation. It does
    produce a distinct fail-closed recovery kind that automatic recovery paths
    refuse after an unclean process exit.
    """
    targets = tuple(sorted({port.strip() for port in ports if isinstance(port, str) and port.strip()}))
    if not targets:
        raise ValueError("SO-101 recovery requires at least one serial port")
    bindings = binding_resolver(targets)
    if set(bindings) == set(targets):
        return HardwareRecoveryIdentity.from_bound_targets(
            recovery_kind,
            "so101",
            bindings,
            profile_digest=profile_digest,
        )
    return HardwareRecoveryIdentity.from_targets(
        unbound_recovery_kind,
        "so101",
        *targets,
        profile_digest=profile_digest,
    )


def hardware_recovery_identity(
    arm_type: object,
    *,
    target_ports: Iterable[str],
    feetech_ports: Iterable[str] = (),
    binding_resolver: SerialBindingResolver = resolve_can_port_bindings,
) -> HardwareRecoveryIdentity:
    """Describe the recovery boundary for an ordinary hardware claim.

    CAN recovery de-energizes follower buses only. Ordinary SO-101 owners do
    not yet share a universal post-open descriptor proof, so their durable
    identities remain physical-only and cover every Feetech target, including
    leader arms. ``HardwareRecoveryIdentity`` hashes paths before persistence.
    """
    family = normalize_arm_type(arm_type)
    primary_targets = tuple(target_ports)
    targets = primary_targets + tuple(feetech_ports) if family == "so101" else primary_targets
    if family == "so101":
        # Ordinary owners do not yet share a universal post-open descriptor
        # attestation hook. Keep their durable crash state physical-only rather
        # than minting automatic recovery authority from a pre-open pathname.
        return HardwareRecoveryIdentity.from_targets(
            "so101_physical_recovery",
            "so101",
            *targets,
        )
    return can_hardware_recovery_identity(
        targets,
        family,
        binding_resolver=binding_resolver,
    )
