"""Exclusive maintenance lease contract. No hardware operation is implemented."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

WRITE_OPERATIONS = frozenset({"assign_id", "center_encoder", "write_eeprom", "change_mode", "bench_motion"})


class MaintenanceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaintenanceLease:
    lease_id: str
    owner: str
    device_identity: str
    operation: str
    opened_monotonic_ns: int


class MaintenanceLeaseManager:
    """Mutual exclusion and receipt validation for future maintenance writes."""

    def __init__(self, hardware_owner: Callable[[], str | None]) -> None:
        self._hardware_owner = hardware_owner
        self._lock = threading.RLock()
        self._lease: MaintenanceLease | None = None

    @property
    def lease(self) -> MaintenanceLease | None:
        with self._lock:
            return self._lease

    def acquire(self, *, owner: str, device_identity: str, operation: str) -> MaintenanceLease:
        if not owner or not device_identity:
            raise ValueError("owner and exact device identity are required")
        if operation not in WRITE_OPERATIONS:
            raise ValueError("unsupported maintenance operation")
        with self._lock:
            holder = self._hardware_owner()
            if holder is not None:
                raise MaintenanceUnavailableError(f"hardware is owned by {holder}")
            if self._lease is not None:
                raise MaintenanceUnavailableError("another maintenance lease is active")
            self._lease = MaintenanceLease(
                lease_id=uuid.uuid4().hex,
                owner=owner,
                device_identity=device_identity,
                operation=operation,
                opened_monotonic_ns=time.monotonic_ns(),
            )
            return self._lease

    def validate_receipt(
        self,
        lease_id: str,
        *,
        device_identity: str,
        before: Mapping[str, int],
        after: Mapping[str, int],
        readback: Mapping[str, int],
    ) -> dict[str, object]:
        """Validate the shape future write adapters must receipt; performs no write."""
        with self._lock:
            lease = self._lease
            if lease is None or lease.lease_id != lease_id:
                raise MaintenanceUnavailableError("maintenance lease is not active")
            if lease.device_identity != device_identity:
                raise MaintenanceUnavailableError("maintenance device identity changed")
            if not before or set(after) != set(before) or dict(readback) != dict(after):
                raise MaintenanceUnavailableError("maintenance readback does not match the requested change")
            return {
                "lease_id": lease.lease_id,
                "owner": lease.owner,
                "operation": lease.operation,
                "device_identity": lease.device_identity,
                "before": dict(before),
                "after": dict(after),
                "readback": dict(readback),
            }

    def release(self, lease_id: str) -> None:
        with self._lock:
            if self._lease is None or self._lease.lease_id != lease_id:
                raise MaintenanceUnavailableError("maintenance lease is not active")
            self._lease = None
