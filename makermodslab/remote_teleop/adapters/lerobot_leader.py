"""Raw SO-101 leader boundary for the operator host."""

from __future__ import annotations

import time
from collections.abc import Callable

from lerobot.teleoperators.so_leader import SO101Leader

from ..calibration_identity import (
    CalibrationIdentity,
    calibration_identity,
    verify_calibration_identity,
)
from .common import DeviceIdentity, JointSchema, RawLeaderSample, derive_so101_joint_schema


class SO101LeaderAdapter:
    """Own only one leader and publish timestamped provider-neutral values."""

    def __init__(
        self,
        config: object,
        *,
        expected_calibration_id: str | None = None,
        expected_calibration_digest: str | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        device_factory: Callable[[object], object] = SO101Leader,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._device = device_factory(config)
        self.calibration_identity: CalibrationIdentity = calibration_identity(
            self._device.calibration_fpath,
            str(self._device.id),
        )
        if (expected_calibration_id is None) != (expected_calibration_digest is None):
            raise ValueError("expected leader calibration id and digest must be provided together")
        if expected_calibration_id is not None and expected_calibration_digest is not None:
            verify_calibration_identity(
                self.calibration_identity,
                expected_id=expected_calibration_id,
                expected_digest=expected_calibration_digest,
                side="leader",
            )
        self.joint_schema: JointSchema = derive_so101_joint_schema(self._device)
        self.device_identity = DeviceIdentity.from_device("so101_leader", self._device)
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("SO-101 leader adapter is already connected")
        try:
            self._device.connect(calibrate=False)
            if not self._device.is_calibrated:
                raise RuntimeError("SO-101 leader calibration does not match the connected servos")
            # Force one exact-schema sample before reporting readiness.
            self._read_positions()
            self._connected = True
        except Exception:
            self._close_device()
            raise

    def read(self) -> RawLeaderSample:
        if not self._connected:
            raise RuntimeError("SO-101 leader adapter is disconnected")
        positions = self._read_positions()
        return RawLeaderSample(positions, self._clock_ns())

    def close(self) -> None:
        self._close_device()

    def _read_positions(self) -> dict[str, float]:
        action = self._device.get_action()
        return self.joint_schema.validate_positions(action)

    def _close_device(self) -> None:
        bus = getattr(self._device, "bus", None)
        if bus is not None and getattr(bus, "is_connected", False):
            self._device.disconnect()
        self._connected = False
