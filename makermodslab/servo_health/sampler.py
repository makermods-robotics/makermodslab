"""Read Feetech health through the bus owner; never open a port or write.

Conversions, status labels, and model metadata are adapted from NORI MotorLab
(MIT) revision 2d8fe277ee9fae8603b2e2f5ee6564b5969ff8b5. See
THIRD_PARTY_NOTICES.md and licenses/NORI-MotorLab-MIT.txt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

SOURCE_REVISION = "Nori-Robotics/feetech-motorlab@2d8fe277ee9fae8603b2e2f5ee6564b5969ff8b5"
PROTOCOL_VERSION = "makermodslab.servo-health.v1"
CURRENT_SCALE_A_PER_LSB = 0.0065
VELOCITY_RPM_PER_RAW = 0.732
KGFCM_TO_NM = 0.0980665
STATUS_BITS = {
    0x01: "voltage",
    0x02: "angle_sensor",
    0x04: "over_temperature",
    0x08: "over_current",
    0x20: "overload",
}
MODEL_PROFILES = {
    "sts3215": {"label": "STS3215", "model_number": 777, "torque_constant_kgcm_per_a": 11.0},
    "sts3250": {"label": "STS3250", "model_number": 2825, "torque_constant_kgcm_per_a": 11.0},
    "sts3095": {"label": "STS3095", "model_number": 2569, "torque_constant_kgcm_per_a": None},
    "sts_series": {"label": "STS series", "model_number": None, "torque_constant_kgcm_per_a": None},
}
DIAGNOSTIC_REGISTERS = (
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Current",
    "Present_Voltage",
    "Present_Temperature",
    "Moving",
    "Torque_Enable",
    "Status",
)


class ExistingFeetechBus(Protocol):
    motors: Mapping[str, Any]

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Any]: ...


def _integral(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = int(value)
    if float(value) != result:
        raise ValueError(f"{name} must be integral")
    return result


def _sign_magnitude(value: int, sign_bit: int) -> int:
    sign = 1 << sign_bit
    magnitude = value & (sign - 1)
    return -magnitude if value & sign else magnitude


def _load(value: int) -> tuple[int, float]:
    magnitude = abs(_sign_magnitude(value, 10))
    return magnitude, min(100.0, magnitude * 100.0 / 1000.0)


class FeetechHealthSampler:
    """Incremental telemetry reader called only from the bus owner's loop."""

    def __init__(
        self,
        bus: ExistingFeetechBus,
        *,
        owner: str,
        arm: str = "left",
        clock_ns=time.monotonic_ns,
        register_interval_ns: int = 100_000_000,
    ) -> None:
        if not owner or not arm:
            raise ValueError("owner and arm labels are required")
        if not 50_000_000 <= register_interval_ns <= 5_000_000_000:
            raise ValueError("register interval must be in [50ms,5s]")
        if not bus.motors:
            raise ValueError("a sampler requires the owner's populated bus")
        self.bus = bus
        self.owner = owner
        self.arm = arm
        self.clock_ns = clock_ns
        self.register_interval_ns = register_interval_ns
        self._lock = threading.RLock()
        self._joint_names = tuple(bus.motors)
        self._identities = {
            name: {
                "id": int(bus.motors[name].id),
                "model_key": str(bus.motors[name].model).lower(),
            }
            for name in self._joint_names
        }
        ids = [identity["id"] for identity in self._identities.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("servo IDs must be unique on one bus")
        self._values: dict[str, dict[str, int]] = {}
        self._cursor = 0
        self._next_sample_ns = 0
        self._last_success_ns: int | None = None
        self._last_error: str | None = None
        self._register_errors: dict[str, str] = {}
        self._communication_errors = 0

    def sample_one(self) -> bool:
        """Read at most one register group. Returns whether a read was attempted."""
        now = self.clock_ns()
        with self._lock:
            if now < self._next_sample_ns:
                return False
            register = DIAGNOSTIC_REGISTERS[self._cursor]
            self._cursor = (self._cursor + 1) % len(DIAGNOSTIC_REGISTERS)
            self._next_sample_ns = now + self.register_interval_ns
            try:
                values = self.bus.sync_read(register, normalize=False)
                if set(values) != set(self._joint_names):
                    raise ValueError(f"{register} response changed joint membership")
                clean = {
                    joint: _integral(values[joint], f"{register}.{joint}") for joint in self._joint_names
                }
            except Exception as exc:
                self._communication_errors += 1
                # Bus exceptions may contain a contributor's serial path.
                # Status exposes the failure class, never vendor exception text.
                self._last_error = type(exc).__name__
                self._register_errors[register] = self._last_error
                return True
            self._values[register] = clean
            self._last_success_ns = now
            self._register_errors.pop(register, None)
            self._last_error = list(self._register_errors.values())[-1] if self._register_errors else None
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock_ns()
            motors = []
            for joint in self._joint_names:
                identity = self._identities[joint]
                model = MODEL_PROFILES.get(
                    identity["model_key"],
                    {
                        "label": identity["model_key"].upper(),
                        "model_number": None,
                        "torque_constant_kgcm_per_a": None,
                    },
                )
                raw = {register: values.get(joint) for register, values in self._values.items()}
                position_raw = raw.get("Present_Position")
                velocity_encoded = raw.get("Present_Velocity")
                load_encoded = raw.get("Present_Load")
                current_raw = raw.get("Present_Current")
                status = raw.get("Status")
                velocity = None if velocity_encoded is None else _sign_magnitude(velocity_encoded, 15)
                load_raw, load_percent = (None, None) if load_encoded is None else _load(load_encoded)
                current_a = None if current_raw is None else abs(current_raw) * CURRENT_SCALE_A_PER_LSB
                torque_constant = model["torque_constant_kgcm_per_a"]
                estimated_torque_nm = (
                    None
                    if current_a is None or torque_constant is None
                    else current_a * float(torque_constant) * KGFCM_TO_NM
                )
                motors.append(
                    {
                        "joint": joint,
                        "id": identity["id"],
                        "model": model["label"],
                        "model_number": model["model_number"],
                        "position_raw": position_raw,
                        "position_degrees": (
                            None if position_raw is None else round(position_raw * 360.0 / 4095.0, 3)
                        ),
                        "velocity_raw": velocity,
                        "velocity_rpm_estimate": (
                            None if velocity is None else round(velocity * VELOCITY_RPM_PER_RAW, 3)
                        ),
                        "load_raw": load_raw,
                        "load_percent": None if load_percent is None else round(load_percent, 3),
                        "current_a": None if current_a is None else round(current_a, 4),
                        "estimated_torque_nm": (
                            None if estimated_torque_nm is None else round(estimated_torque_nm, 4)
                        ),
                        "voltage_v": (
                            None
                            if raw.get("Present_Voltage") is None
                            else round(raw["Present_Voltage"] * 0.1, 2)
                        ),
                        "temperature_c": raw.get("Present_Temperature"),
                        "moving": None if raw.get("Moving") is None else bool(raw["Moving"]),
                        "torque_enabled": (
                            None if raw.get("Torque_Enable") is None else bool(raw["Torque_Enable"])
                        ),
                        "status_code": status,
                        "faults": None
                        if status is None
                        else [label for bit, label in STATUS_BITS.items() if status & bit],
                        "complete": all(register in raw for register in DIAGNOSTIC_REGISTERS),
                    }
                )
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "source_revision": SOURCE_REVISION,
                "owner": self.owner,
                "arm": self.arm,
                "read_only": True,
                "available": self._last_success_ns is not None,
                "complete": len(self._values) == len(DIAGNOSTIC_REGISTERS),
                "sampled_monotonic_ns": self._last_success_ns,
                "age_ms": None
                if self._last_success_ns is None
                else max(0, now - self._last_success_ns) / 1_000_000,
                "communication_errors": self._communication_errors,
                "last_error": self._last_error,
                "registers_complete": [
                    register for register in DIAGNOSTIC_REGISTERS if register in self._values
                ],
                "registers_total": len(DIAGNOSTIC_REGISTERS),
                "motors": motors,
            }
            return deepcopy(result)


def unavailable_snapshot(message: str = "No Feetech bus owner is publishing health") -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_revision": SOURCE_REVISION,
        "read_only": True,
        "available": False,
        "complete": False,
        "owner": None,
        "arms": [],
        "last_error": message,
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
