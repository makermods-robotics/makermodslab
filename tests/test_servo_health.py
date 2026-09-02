from __future__ import annotations

from types import SimpleNamespace

import pytest

from makermodslab.servo_health.maintenance import (
    MaintenanceLeaseManager,
    MaintenanceUnavailableError,
)
from makermodslab.servo_health.sampler import (
    DIAGNOSTIC_REGISTERS,
    FeetechHealthSampler,
)
from makermodslab.servo_health.service import ServoHealthService


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    def advance(self) -> None:
        self.now += 100_000_000


class FakeBus:
    def __init__(self) -> None:
        self.motors = {
            "shoulder": SimpleNamespace(id=1, model="sts3215"),
            "elbow": SimpleNamespace(id=2, model="sts3250"),
        }
        self.fail_register: str | None = None
        self.reads: list[str] = []

    def sync_read(self, register, **_kwargs):
        self.reads.append(register)
        if register == self.fail_register:
            raise OSError("injected diagnostic failure")
        values = {
            "Present_Position": (2048, 1024),
            "Present_Velocity": (10, 0x8005),
            "Present_Load": (100, 0x0400 | 250),
            "Present_Current": (20, 40),
            "Present_Voltage": (120, 118),
            "Present_Temperature": (30, 31),
            "Moving": (1, 0),
            "Torque_Enable": (1, 0),
            "Status": (0, 0x0C),
        }[register]
        return dict(zip(self.motors, values, strict=True))


def complete_sampler(bus: FakeBus, clock: FakeClock) -> FeetechHealthSampler:
    sampler = FeetechHealthSampler(bus, owner="teleoperation", clock_ns=clock)
    for _ in DIAGNOSTIC_REGISTERS:
        assert sampler.sample_one()
        clock.advance()
    return sampler


def test_motorlab_units_models_faults_and_torque_state() -> None:
    bus = FakeBus()
    clock = FakeClock()
    snapshot = complete_sampler(bus, clock).snapshot()
    assert snapshot["complete"] is True
    shoulder, elbow = snapshot["motors"]
    assert shoulder["id"] == 1
    assert shoulder["model"] == "STS3215"
    assert shoulder["position_degrees"] == pytest.approx(180.044, abs=0.001)
    assert shoulder["velocity_rpm_estimate"] == 7.32
    assert shoulder["current_a"] == 0.13
    assert shoulder["voltage_v"] == 12.0
    assert shoulder["temperature_c"] == 30
    assert shoulder["moving"] is True
    assert shoulder["torque_enabled"] is True
    assert elbow["velocity_raw"] == -5
    assert elbow["load_raw"] == 250
    assert elbow["faults"] == ["over_temperature", "over_current"]


def test_diagnostic_failure_keeps_value_null_and_surfaces_error() -> None:
    bus = FakeBus()
    bus.fail_register = "Present_Current"
    clock = FakeClock()
    snapshot = complete_sampler(bus, clock).snapshot()
    assert snapshot["complete"] is False
    assert snapshot["communication_errors"] == 1
    assert snapshot["last_error"] == "OSError"
    assert all(motor["current_a"] is None for motor in snapshot["motors"])
    assert all(motor["estimated_torque_nm"] is None for motor in snapshot["motors"])


def test_http_service_reads_cache_without_bus_transaction() -> None:
    bus = FakeBus()
    clock = FakeClock()
    sampler = complete_sampler(bus, clock)
    service = ServoHealthService()
    service.attach("teleoperation:left", sampler)
    before = list(bus.reads)
    snapshot = service.snapshot()
    assert snapshot["available"] is True
    assert bus.reads == before
    service.detach("teleoperation:left")
    assert service.snapshot()["available"] is False


def test_maintenance_is_exclusive_and_requires_exact_identity_readback() -> None:
    current_holder = ["teleoperation"]
    manager = MaintenanceLeaseManager(lambda: current_holder[0])
    with pytest.raises(MaintenanceUnavailableError, match="teleoperation"):
        manager.acquire(owner="operator", device_identity="usb:exact", operation="assign_id")

    current_holder[0] = None
    lease = manager.acquire(owner="operator", device_identity="usb:exact", operation="assign_id")
    with pytest.raises(MaintenanceUnavailableError, match="another"):
        manager.acquire(owner="other", device_identity="usb:exact", operation="write_eeprom")
    with pytest.raises(MaintenanceUnavailableError, match="identity changed"):
        manager.validate_receipt(
            lease.lease_id,
            device_identity="usb:different",
            before={"ID": 1},
            after={"ID": 2},
            readback={"ID": 2},
        )
    with pytest.raises(MaintenanceUnavailableError, match="readback"):
        manager.validate_receipt(
            lease.lease_id,
            device_identity="usb:exact",
            before={"ID": 1},
            after={"ID": 2},
            readback={"ID": 3},
        )
    receipt = manager.validate_receipt(
        lease.lease_id,
        device_identity="usb:exact",
        before={"ID": 1},
        after={"ID": 2},
        readback={"ID": 2},
    )
    assert receipt["before"] == {"ID": 1}
    assert receipt["readback"] == {"ID": 2}
    manager.release(lease.lease_id)
