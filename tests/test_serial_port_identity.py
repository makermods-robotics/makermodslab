from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import makermodslab.serial_port_identity as serial_port_identity
from makermodslab.serial_port_identity import (
    MacOSSerialRegistryCandidate,
    linux_opened_serial_binding,
    macos_opened_serial_binding,
    opened_serial_binding,
    opened_serial_binding_matches,
    resolve_serial_port_bindings,
)


def port(device: str, *, serial: str | None, vid: int | None, pid: int | None):
    return SimpleNamespace(device=device, serial_number=serial, vid=vid, pid=pid)


def test_resolver_requires_vid_pid_and_nonempty_usb_serial() -> None:
    inventory = [
        port("/dev/complete", serial="ABC-123", vid=0x1A86, pid=0x55D4),
        port("/dev/no-serial", serial=None, vid=0x1A86, pid=0x55D4),
        port("/dev/no-vid", serial="ABC-124", vid=None, pid=0x55D4),
        port("/dev/no-pid", serial="ABC-125", vid=0x1A86, pid=None),
    ]

    result = resolve_serial_port_bindings(
        [item.device for item in inventory],
        enumerate_ports=lambda: inventory,
    )

    assert set(result) == {"/dev/complete"}


def test_same_adapter_binding_is_stable_across_path_change_and_opaque() -> None:
    serial = "PRIVATE-SERIAL-987"
    old = port("/dev/cu.old", serial=serial, vid=0x1A86, pid=0x55D4)
    new = port("/dev/cu.new", serial=serial, vid=0x1A86, pid=0x55D4)

    old_binding = resolve_serial_port_bindings([old.device], enumerate_ports=lambda: [old])[old.device]
    new_binding = resolve_serial_port_bindings([new.device], enumerate_ports=lambda: [new])[new.device]

    assert old_binding == new_binding
    assert len(old_binding) == 64
    assert serial not in old_binding
    assert old.device not in old_binding


def test_duplicate_inventory_entry_is_ambiguous_and_omitted() -> None:
    entries = [
        port("/dev/cu.arm", serial="A", vid=1, pid=2),
        port("/dev/cu.arm", serial="B", vid=1, pid=2),
    ]

    assert resolve_serial_port_bindings(["/dev/cu.arm"], enumerate_ports=lambda: entries) == {}


def test_duplicate_physical_identity_on_an_unrequested_path_denies_authority() -> None:
    entries = [
        port("/dev/cu.requested", serial="DUPLICATE", vid=1, pid=2),
        port("/dev/cu.other", serial="DUPLICATE", vid=1, pid=2),
    ]

    assert resolve_serial_port_bindings(["/dev/cu.requested"], enumerate_ports=lambda: entries) == {}


def test_restored_path_and_reused_device_number_cannot_override_opened_binding() -> None:
    claimed = port("/dev/cu.arm", serial="ADAPTER-A", vid=1, pid=2)
    opened = port("/dev/cu.arm", serial="ADAPTER-B", vid=1, pid=2)
    expected_a = resolve_serial_port_bindings(
        [claimed.device],
        enumerate_ports=lambda: [claimed],
    )[claimed.device]
    opened_b = resolve_serial_port_bindings(
        [opened.device],
        enumerate_ports=lambda: [opened],
    )[opened.device]
    bus = SimpleNamespace()

    assert not opened_serial_binding_matches(
        bus,
        expected_a,
        # Even if pathname enumeration has returned to A and the kernel has
        # reused the same device number, authority comes only from the injected
        # identity of the descriptor that actually opened B.
        binding_provider=lambda _bus: opened_b,
    )


def test_unavailable_open_handle_provider_fails_closed() -> None:
    assert not opened_serial_binding_matches(
        SimpleNamespace(),
        "a" * 64,
        binding_provider=lambda _bus: None,
    )


def test_linux_provider_derives_binding_from_open_descriptor_sysfs_ancestry(tmp_path: Path) -> None:
    usb_device = tmp_path / "sys" / "devices" / "usb1" / "1-1"
    tty_device = usb_device / "1-1:1.0" / "ttyUSB0"
    tty_device.mkdir(parents=True)
    (usb_device / "serial").write_text("OPENED-ADAPTER\n", encoding="utf-8")
    (usb_device / "idVendor").write_text("1a86\n", encoding="ascii")
    (usb_device / "idProduct").write_text("55d4\n", encoding="ascii")
    device_number = os.makedev(188, 7)
    device_link = tmp_path / "sys" / "dev" / "char" / "188:7"
    device_link.mkdir(parents=True)
    (device_link / "device").symlink_to(tty_device, target_is_directory=True)
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=device_number)
    bus = SimpleNamespace(
        port_handler=SimpleNamespace(
            ser=SimpleNamespace(is_open=True, fileno=lambda: 42),
        )
    )
    expected_info = port("ignored", serial="OPENED-ADAPTER", vid=0x1A86, pid=0x55D4)
    expected = resolve_serial_port_bindings(
        [expected_info.device],
        enumerate_ports=lambda: [expected_info],
    )[expected_info.device]

    assert (
        linux_opened_serial_binding(
            bus,
            sysfs_root=tmp_path / "sys",
            platform_name="linux",
            fstat_descriptor=lambda _fd: descriptor,
        )
        == expected
    )
    assert linux_opened_serial_binding(bus, platform_name="darwin") is None


def _open_bus() -> SimpleNamespace:
    return SimpleNamespace(
        port_handler=SimpleNamespace(
            ser=SimpleNamespace(is_open=True, fileno=lambda: 42),
        )
    )


def _mac_candidate(
    *,
    device_number: int,
    serial_registry_id: int = 100,
    usb_registry_id: int = 200,
    binding: str = "a" * 64,
) -> MacOSSerialRegistryCandidate:
    return MacOSSerialRegistryCandidate(
        device_number=device_number,
        serial_registry_id=serial_registry_id,
        usb_registry_id=usb_registry_id,
        binding=binding,
    )


def test_macos_provider_binds_open_descriptor_to_stable_unique_iokit_chain() -> None:
    device_number = os.makedev(9, 41)
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=device_number)
    selected = _mac_candidate(device_number=device_number)

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=lambda: (selected,),
        )
        == selected.binding
    )


@pytest.mark.parametrize(
    "first_scan,second_scan",
    [
        # A device swap changes both registry entries and USB identity.
        (
            (_mac_candidate(device_number=os.makedev(9, 42)),),
            (
                _mac_candidate(
                    device_number=os.makedev(9, 42),
                    serial_registry_id=101,
                    usb_registry_id=201,
                    binding="b" * 64,
                ),
            ),
        ),
        # A disconnected adapter disappears from IOKit before verification.
        (
            (_mac_candidate(device_number=os.makedev(9, 42)),),
            (),
        ),
        # Even an adapter with the same reported identity is a new IOKit chain.
        (
            (_mac_candidate(device_number=os.makedev(9, 42)),),
            (
                _mac_candidate(
                    device_number=os.makedev(9, 42),
                    serial_registry_id=101,
                    usb_registry_id=201,
                ),
            ),
        ),
    ],
)
def test_macos_provider_rejects_iokit_change_during_verification(
    first_scan: tuple[MacOSSerialRegistryCandidate, ...],
    second_scan: tuple[MacOSSerialRegistryCandidate, ...],
) -> None:
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(9, 42))
    scans = iter((first_scan, second_scan))

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=lambda: next(scans),
        )
        is None
    )


def test_macos_provider_rejects_duplicate_usb_identity_on_other_device() -> None:
    opened_device = os.makedev(9, 43)
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=opened_device)
    inventory = (
        _mac_candidate(device_number=opened_device),
        _mac_candidate(
            device_number=os.makedev(9, 44),
            serial_registry_id=102,
            usb_registry_id=202,
        ),
    )

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=lambda: inventory,
        )
        is None
    )


def test_macos_provider_rejects_missing_or_failed_iokit_inventory() -> None:
    opened_device = os.makedev(9, 48)
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=opened_device)

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=lambda: (),
        )
        is None
    )

    def failed_inventory() -> tuple[MacOSSerialRegistryCandidate, ...]:
        raise OSError("IOKit unavailable")

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=failed_inventory,
        )
        is None
    )


def test_macos_provider_rejects_ambiguous_device_number() -> None:
    opened_device = os.makedev(9, 45)
    descriptor = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=opened_device)
    inventory = (
        _mac_candidate(device_number=opened_device),
        _mac_candidate(
            device_number=opened_device,
            serial_registry_id=103,
            usb_registry_id=203,
            binding="c" * 64,
        ),
    )

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: descriptor,
            enumerate_registry=lambda: inventory,
        )
        is None
    )


def test_macos_provider_rejects_descriptor_change_between_scans() -> None:
    before = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(9, 46))
    after = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(9, 47))
    descriptor_stats = iter((before, after))
    selected = _mac_candidate(device_number=before.st_rdev)

    assert (
        macos_opened_serial_binding(
            _open_bus(),
            platform_name="darwin",
            fstat_descriptor=lambda _fd: next(descriptor_stats),
            enumerate_registry=lambda: (selected,),
        )
        is None
    )


def test_opened_binding_dispatches_macos_without_loading_linux_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        serial_port_identity,
        "macos_opened_serial_binding",
        lambda _bus, *, platform_name: "d" * 64,
    )
    monkeypatch.setattr(
        serial_port_identity,
        "linux_opened_serial_binding",
        lambda _bus, *, platform_name: pytest.fail("Linux provider should not run"),
    )

    assert opened_serial_binding(SimpleNamespace(), platform_name="darwin") == "d" * 64
    assert opened_serial_binding(SimpleNamespace(), platform_name="win32") is None


def test_opened_binding_dispatches_linux_without_loading_macos_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        serial_port_identity,
        "linux_opened_serial_binding",
        lambda _bus, *, platform_name: "e" * 64,
    )
    monkeypatch.setattr(
        serial_port_identity,
        "macos_opened_serial_binding",
        lambda _bus, *, platform_name: pytest.fail("macOS provider should not run"),
    )

    assert opened_serial_binding(SimpleNamespace(), platform_name="linux") == "e" * 64


@pytest.mark.skipif(sys.platform != "darwin", reason="requires the local IOKit frameworks")
def test_macos_iokit_inventory_loads_without_opening_a_device() -> None:
    from makermodslab.macos_serial_registry import enumerate_serial_registry

    assert isinstance(enumerate_serial_registry(), tuple)
