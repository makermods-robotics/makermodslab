"""Typed values shared by the SO-101 leader and follower adapters."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from ..calibration_identity import canonical_json


@dataclass(frozen=True)
class JointDefinition:
    name: str
    action_key: str
    unit: str
    commissioned_minimum: float
    commissioned_maximum: float

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "action_key": self.action_key,
            "unit": self.unit,
            "commissioned_minimum": self.commissioned_minimum,
            "commissioned_maximum": self.commissioned_maximum,
        }


@dataclass(frozen=True)
class JointSchema:
    joints: tuple[JointDefinition, ...]

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError("joint schema cannot be empty")
        keys = self.action_keys
        names = tuple(joint.name for joint in self.joints)
        if len(set(keys)) != len(keys) or len(set(names)) != len(names):
            raise ValueError("joint schema cannot contain duplicates")

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(joint.action_key for joint in self.joints)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(joint.unit for joint in self.joints)

    def public(self) -> list[dict[str, object]]:
        return [joint.public() for joint in self.joints]

    def validate_positions(self, values: Mapping[str, object]) -> dict[str, float]:
        if not isinstance(values, Mapping) or tuple(values) != self.action_keys:
            raise ValueError("positions must have the exact ordered SO-101 action schema")
        clean: dict[str, float] = {}
        for joint in self.joints:
            value = values[joint.action_key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{joint.action_key} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{joint.action_key} must be finite")
            if not joint.commissioned_minimum <= number <= joint.commissioned_maximum:
                raise ValueError(f"{joint.action_key} is outside its commissioned range")
            clean[joint.action_key] = number
        return clean


@dataclass(frozen=True)
class DeviceIdentity:
    device_type: str
    digest: str
    motor_ids: tuple[tuple[str, int, str], ...]

    @classmethod
    def from_device(cls, device_type: str, device: object) -> DeviceIdentity:
        bus = getattr(device, "bus", None)
        motors = getattr(bus, "motors", None)
        if not isinstance(motors, Mapping) or not motors:
            raise ValueError("SO-101 device exposes no motor contract")
        rows = tuple((name, int(motor.id), str(motor.model)) for name, motor in motors.items())
        body = {"device_type": device_type, "motors": [list(row) for row in rows]}
        return cls(device_type, hashlib.sha256(canonical_json(body)).hexdigest(), rows)

    def public(self) -> dict[str, object]:
        return {
            "device_type": self.device_type,
            "digest": self.digest,
            "motors": [
                {"name": name, "id": motor_id, "model": model} for name, motor_id, model in self.motor_ids
            ],
        }


@dataclass(frozen=True)
class RawLeaderSample:
    positions: Mapping[str, float]
    sampled_monotonic_ns: int


@dataclass(frozen=True)
class StopHardwareReceipt:
    reason: str
    disable_requested: bool
    torque_off_confirmed: bool | None
    verification: str
    fault: str | None = None
    connect_transient_torque_risk: bool = True
    hardware_stop_completed: bool = False

    def public(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "disable_requested": self.disable_requested,
            "torque_off_confirmed": self.torque_off_confirmed,
            "verification": self.verification,
            "hardware_stop_completed": self.hardware_stop_completed,
            "fault": self.fault,
            "connect_transient_torque_risk": self.connect_transient_torque_risk,
        }


@dataclass(frozen=True)
class CloseReceipt:
    close_requested: bool
    close_completed: bool
    fault: str | None = None

    def public(self) -> dict[str, object]:
        return {
            "close_requested": self.close_requested,
            "close_completed": self.close_completed,
            "fault": self.fault,
        }


def derive_so101_joint_schema(device: object) -> JointSchema:
    """Derive order, units, and ranges from the concrete LeRobot action bus."""
    features = getattr(device, "action_features", None)
    bus = getattr(device, "bus", None)
    motors = getattr(bus, "motors", None)
    calibration = getattr(device, "calibration", None)
    if not isinstance(features, Mapping) or not isinstance(motors, Mapping):
        raise ValueError("SO-101 device action contract is unavailable")
    expected_keys = tuple(f"{name}.pos" for name in motors)
    if tuple(features) != expected_keys or any(features[key] is not float for key in expected_keys):
        raise ValueError("SO-101 action_features do not match the motor contract")
    if not isinstance(calibration, Mapping) or set(calibration) != set(motors):
        raise ValueError("SO-101 calibration does not match the motor contract")

    joints: list[JointDefinition] = []
    for name, motor in motors.items():
        cal = calibration[name]
        norm_mode = getattr(getattr(motor, "norm_mode", None), "value", "")
        if norm_mode == "degrees":
            resolution = getattr(bus, "model_resolution_table", {}).get(motor.model)
            if not isinstance(resolution, int) or resolution <= 1:
                raise ValueError(f"SO-101 model resolution is unavailable for {name}")
            midpoint = (float(cal.range_min) + float(cal.range_max)) / 2.0
            minimum = (float(cal.range_min) - midpoint) * 360.0 / (resolution - 1)
            maximum = (float(cal.range_max) - midpoint) * 360.0 / (resolution - 1)
            unit = "degree"
        elif norm_mode == "range_m100_100":
            minimum, maximum, unit = -100.0, 100.0, "normalized_-100_100"
        elif norm_mode == "range_0_100":
            minimum, maximum, unit = 0.0, 100.0, "percent"
        else:
            raise ValueError(f"unsupported SO-101 normalization mode for {name}")
        if not minimum < maximum:
            raise ValueError(f"invalid commissioned range for {name}")
        joints.append(JointDefinition(name, f"{name}.pos", unit, minimum, maximum))
    return JointSchema(tuple(joints))
