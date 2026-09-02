"""Thread-safe cache read by HTTP; bus owners publish, handlers never sample."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .sampler import PROTOCOL_VERSION, SOURCE_REVISION, FeetechHealthSampler, unavailable_snapshot


class ServoHealthService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._samplers: dict[str, FeetechHealthSampler] = {}
        self._published: dict[str, dict[str, Any]] = {}

    def attach(self, key: str, sampler: FeetechHealthSampler) -> None:
        with self._lock:
            if key in self._samplers or key in self._published:
                raise RuntimeError(f"servo-health publisher {key!r} already exists")
            self._samplers[key] = sampler

    def publish(self, key: str, snapshot: Mapping[str, Any]) -> None:
        """Accept a snapshot relayed by the child process that owns the bus."""
        if (
            not key
            or snapshot.get("protocol_version") != PROTOCOL_VERSION
            or snapshot.get("source_revision") != SOURCE_REVISION
            or snapshot.get("read_only") is not True
            or not isinstance(snapshot.get("owner"), str)
            or not isinstance(snapshot.get("motors"), list)
        ):
            raise ValueError("servo-health publisher snapshot is invalid")
        with self._lock:
            if key in self._samplers:
                raise RuntimeError(f"servo-health publisher {key!r} already exists")
            self._published[key] = deepcopy(dict(snapshot))

    def detach(self, key: str) -> None:
        with self._lock:
            self._samplers.pop(key, None)
            self._published.pop(key, None)

    def sample_owned_buses(self) -> None:
        """Called from the hardware owner's loop, never from an HTTP thread."""
        with self._lock:
            samplers = tuple(self._samplers.values())
        for sampler in samplers:
            sampler.sample_one()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samplers = tuple(self._samplers.values())
            published = tuple(deepcopy(value) for value in self._published.values())
        if not samplers and not published:
            return unavailable_snapshot()
        arms = [sampler.snapshot() for sampler in samplers]
        arms.extend(published)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "source_revision": SOURCE_REVISION,
            "read_only": True,
            "available": any(arm["available"] for arm in arms),
            "complete": all(arm["complete"] for arm in arms),
            "owner": arms[0]["owner"],
            "arms": arms,
            "last_error": next((arm["last_error"] for arm in arms if arm["last_error"]), None),
            "maintenance": {
                "state": "disabled",
                "write_operations": [
                    "assign_id",
                    "center_encoder",
                    "write_eeprom",
                    "change_mode",
                    "bench_motion",
                ],
            },
        }


servo_health_service = ServoHealthService()
