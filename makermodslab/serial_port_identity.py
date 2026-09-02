"""Stable, path-sanitized identities for USB serial arm adapters."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SerialPortInfo(Protocol):
    device: str
    serial_number: str | None
    vid: int | None
    pid: int | None


SerialPortEnumerator = Callable[[], Iterable[SerialPortInfo]]
SerialBindingResolver = Callable[[Iterable[str]], dict[str, str]]
OpenedSerialBindingProvider = Callable[[object], str | None]
OpenedSerialBindingVerifier = Callable[[object, str], bool]


@dataclass(frozen=True)
class MacOSSerialRegistryCandidate:
    """One IOKit serial client tied to a concrete character device."""

    device_number: int
    serial_registry_id: int
    usb_registry_id: int
    binding: str


MacOSSerialRegistryEnumerator = Callable[[], Iterable[MacOSSerialRegistryCandidate]]


def _enumerate_serial_ports() -> Iterable[SerialPortInfo]:
    from serial.tools import list_ports

    return list_ports.comports()


def _binding_digest(info: SerialPortInfo) -> str | None:
    return _binding_digest_components(info.serial_number, info.vid, info.pid)


def _binding_digest_components(
    serial_number: object,
    vid: object,
    pid: object,
) -> str | None:
    if not isinstance(serial_number, str) or not serial_number.strip():
        return None
    if (
        isinstance(vid, bool)
        or not isinstance(vid, int)
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or not 0 <= vid <= 0xFFFF
        or not 0 <= pid <= 0xFFFF
    ):
        return None
    # Return only an opaque digest. Neither the raw USB serial nor its device
    # path can escape through identity snapshots, errors, logs, or callers.
    material = f"makermodslab-usb-serial-binding-v1\0{vid:04x}\0{pid:04x}\0{serial_number.strip()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _serial_handle(bus: object) -> object | None:
    return getattr(getattr(bus, "port_handler", None), "ser", None)


def linux_opened_serial_binding(
    bus: object,
    *,
    sysfs_root: Path = Path("/sys"),
    platform_name: str = sys.platform,
    fstat_descriptor: Callable[[int], object] = os.fstat,
) -> str | None:
    """Derive USB identity from an already-open Linux character descriptor.

    No pathname supplied by the caller participates. The descriptor's device
    number selects ``/sys/dev/char/<major>:<minor>/device`` and USB attributes
    are read from that device's ancestry. Any unsupported platform, race,
    incomplete identity, or non-character descriptor fails closed.
    """
    if platform_name != "linux":
        return None
    serial_handle = _serial_handle(bus)
    fileno = getattr(serial_handle, "fileno", None)
    if not callable(fileno) or getattr(serial_handle, "is_open", True) is not True:
        return None
    try:
        descriptor_fd = fileno()
        before = fstat_descriptor(descriptor_fd)
    except (OSError, TypeError, ValueError):
        return None
    if not stat.S_ISCHR(before.st_mode):
        return None
    device_link = (
        sysfs_root / "dev" / "char" / f"{os.major(before.st_rdev)}:{os.minor(before.st_rdev)}" / "device"
    )
    try:
        device = device_link.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    for ancestor in (device, *device.parents):
        try:
            serial_number = (ancestor / "serial").read_text(encoding="utf-8").strip()
            vendor_text = (ancestor / "idVendor").read_text(encoding="ascii").strip()
            product_text = (ancestor / "idProduct").read_text(encoding="ascii").strip()
            binding = _binding_digest_components(
                serial_number,
                int(vendor_text, 16),
                int(product_text, 16),
            )
        except (OSError, UnicodeError, ValueError):
            binding = None
        if binding is not None:
            try:
                after = fstat_descriptor(descriptor_fd)
                stable_device = device_link.resolve(strict=True)
            except (OSError, RuntimeError):
                return None
            if stat.S_ISCHR(after.st_mode) and after.st_rdev == before.st_rdev and stable_device == device:
                return binding
            return None
        if ancestor == sysfs_root:
            break
    return None


def _enumerate_macos_serial_registry() -> Iterable[MacOSSerialRegistryCandidate]:
    # Importing this module loads macOS frameworks, so it must remain behind
    # the platform dispatch boundary.
    from .macos_serial_registry import enumerate_serial_registry

    return enumerate_serial_registry()


def macos_opened_serial_binding(
    bus: object,
    *,
    platform_name: str = sys.platform,
    fstat_descriptor: Callable[[int], object] = os.fstat,
    enumerate_registry: MacOSSerialRegistryEnumerator = _enumerate_macos_serial_registry,
) -> str | None:
    """Derive USB identity from an already-open macOS character descriptor.

    IOKit's callout node is used only to tie a live registry entry to the open
    descriptor's device number. The configured pathname never participates.
    Registry entry IDs and the complete unique USB identity must survive a
    second scan; unplug/replug, ambiguity, and duplicate serials fail closed.
    """
    if platform_name != "darwin":
        return None
    serial_handle = _serial_handle(bus)
    fileno = getattr(serial_handle, "fileno", None)
    if not callable(fileno) or getattr(serial_handle, "is_open", True) is not True:
        return None
    try:
        descriptor_fd = fileno()
        before = fstat_descriptor(descriptor_fd)
    except (OSError, TypeError, ValueError):
        return None
    if not stat.S_ISCHR(before.st_mode):
        return None

    try:
        first_scan = tuple(enumerate_registry())
    except Exception:
        return None
    first_matches = tuple(candidate for candidate in first_scan if candidate.device_number == before.st_rdev)
    if len(first_matches) != 1:
        return None
    selected = first_matches[0]
    if sum(candidate.binding == selected.binding for candidate in first_scan) != 1:
        return None

    try:
        after = fstat_descriptor(descriptor_fd)
        second_scan = tuple(enumerate_registry())
    except Exception:
        return None
    if not stat.S_ISCHR(after.st_mode) or after.st_rdev != before.st_rdev:
        return None
    second_matches = tuple(candidate for candidate in second_scan if candidate.device_number == after.st_rdev)
    if len(second_matches) != 1 or second_matches[0] != selected:
        return None
    if sum(candidate.binding == selected.binding for candidate in second_scan) != 1:
        return None
    return selected.binding


def opened_serial_binding(
    bus: object,
    *,
    platform_name: str = sys.platform,
) -> str | None:
    """Dispatch opened-handle verification to the current host platform."""
    if platform_name == "linux":
        return linux_opened_serial_binding(bus, platform_name=platform_name)
    if platform_name == "darwin":
        return macos_opened_serial_binding(bus, platform_name=platform_name)
    return None


def resolve_serial_port_bindings(
    ports: Iterable[str],
    *,
    enumerate_ports: SerialPortEnumerator = _enumerate_serial_ports,
) -> dict[str, str]:
    """Resolve supplied paths to stable USB VID/PID/serial digests.

    Missing, ambiguous, or incompletely identified paths are omitted. Callers
    requiring recovery authority must demand an entry for every target before
    opening hardware. Enumeration performs no serial-port open or device I/O.
    """
    requested = tuple(sorted({port.strip() for port in ports if isinstance(port, str) and port.strip()}))
    if not requested:
        return {}
    requested_set = set(requested)
    candidates: dict[str, list[str]] = {port: [] for port in requested}
    binding_paths: dict[str, set[str]] = {}
    for info in enumerate_ports():
        device = getattr(info, "device", None)
        if not isinstance(device, str) or not device:
            continue
        binding = _binding_digest(info)
        if binding is None:
            continue
        binding_paths.setdefault(binding, set()).add(device)
        if device in requested_set:
            candidates[device].append(binding)
    return {
        port: bindings[0]
        for port, bindings in candidates.items()
        if len(bindings) == 1 and len(binding_paths[bindings[0]]) == 1
    }


def opened_serial_binding_matches(
    bus: object,
    expected_binding: str,
    *,
    binding_provider: OpenedSerialBindingProvider = opened_serial_binding,
) -> bool:
    """Compare the expected binding only with identity from the open handle."""
    if not isinstance(expected_binding, str) or not expected_binding:
        return False
    try:
        opened_binding = binding_provider(bus)
    except Exception:
        return False
    return opened_binding == expected_binding
