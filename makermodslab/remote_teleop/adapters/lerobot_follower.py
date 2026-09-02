"""Commissioned SO-101 follower boundary for the robot host.

No network frame reaches this object directly.  ``RemoteExecutor`` is its only
command caller, and this adapter is the only remote module that imports the
concrete LeRobot follower.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import logging
import textwrap
import tokenize
from collections.abc import Callable, Mapping

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower

from ...serial_port_identity import OpenedSerialBindingVerifier, opened_serial_binding_matches
from ...servo_health.sampler import FeetechHealthSampler
from ..calibration_identity import (
    CalibrationIdentity,
    calibration_identity,
    verify_calibration_identity,
)
from .common import (
    CloseReceipt,
    DeviceIdentity,
    JointSchema,
    StopHardwareReceipt,
    derive_so101_joint_schema,
)

logger = logging.getLogger(__name__)

# Semantic token stream of ``SO101Follower.configure`` in the pinned LeRobot revision
# b968c0c015e16699a225070f677dc644543707f0.  Remote teleoperation reproduces
# this register contract without using LeRobot's torque-restoring context
# manager.  Fail closed if the pin changes so new safety-relevant register
# writes cannot silently bypass this adapter.
_PINNED_CONFIGURE_SHA256 = (  # gitleaks:allow -- semantic source digest, not a credential
    "075539f8270fa8b19e6dae656a38aaef89aae146a4c0039d72762b4fb9d70615"
)


def _assert_pinned_configuration_contract() -> None:
    try:
        source = textwrap.dedent(inspect.getsource(SO101Follower.configure))
        insignificant = {
            tokenize.ENCODING,
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }
        semantic_tokens = tuple(
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in insignificant
        )
    except (OSError, TypeError, tokenize.TokenError) as exc:
        raise RuntimeError("cannot verify the pinned LeRobot follower configuration contract") from exc
    digest = hashlib.sha256("\0".join(semantic_tokens).encode("utf-8")).hexdigest()
    if digest != _PINNED_CONFIGURE_SHA256:
        raise RuntimeError(
            "pinned LeRobot follower configuration contract changed; "
            "review the disarmed remote configuration sequence"
        )


class SO101FollowerDriver:
    """A single SO-101 follower with readback-backed safe-stop receipts."""

    connect_transient_torque_risk = False

    def __init__(
        self,
        config: object,
        *,
        expected_calibration_id: str | None = None,
        expected_calibration_digest: str | None = None,
        expected_serial_binding: str | None = None,
        opened_binding_verifier: OpenedSerialBindingVerifier = opened_serial_binding_matches,
        device_factory: Callable[[object], object] = SO101Follower,
    ) -> None:
        self.config = config
        self._device = device_factory(config)
        self._expected_serial_binding = expected_serial_binding
        self._opened_binding_verifier = opened_binding_verifier
        self._binding_confirmed = expected_serial_binding is None
        self.calibration_identity: CalibrationIdentity = calibration_identity(
            self._device.calibration_fpath,
            str(self._device.id),
        )
        if (expected_calibration_id is None) != (expected_calibration_digest is None):
            raise ValueError("expected follower calibration id and digest must be provided together")
        if expected_calibration_id is not None and expected_calibration_digest is not None:
            verify_calibration_identity(
                self.calibration_identity,
                expected_id=expected_calibration_id,
                expected_digest=expected_calibration_digest,
                side="follower",
            )
        self.joint_schema: JointSchema = derive_so101_joint_schema(self._device)
        self.joint_names = self.joint_schema.action_keys
        self.device_identity = DeviceIdentity.from_device("so101_follower", self._device)
        self._connected = False
        self._execution_enabled = False
        self._current_positions: dict[str, float] = {}
        self.last_stop_receipt: StopHardwareReceipt | None = None
        self.last_close_receipt: CloseReceipt | None = None
        self._health_sampler: FeetechHealthSampler | None = None

    @property
    def status(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "execution_enabled": self._execution_enabled,
            "connect_transient_torque_risk": self.connect_transient_torque_risk,
            "calibration": self.calibration_identity.public(),
            "device": self.device_identity.public(),
            "joint_schema": self.joint_schema.public(),
            "servo_health_available": self._health_sampler is not None,
            "stop_receipt": None if self.last_stop_receipt is None else self.last_stop_receipt.public(),
            "close_receipt": None if self.last_close_receipt is None else self.last_close_receipt.public(),
        }

    def connect(self) -> None:
        """Connect and configure DISARMED without calling LeRobot ``connect``.

        The pinned upstream configuration uses a context manager that restores
        torque on exit.  This boundary instead performs the same register
        writes directly while torque remains readback-confirmed off.  Only the
        first ``execute`` call made by ``RemoteExecutor`` may enable torque.
        """
        if self._connected:
            raise RuntimeError("SO-101 follower adapter is already connected")
        try:
            _assert_pinned_configuration_contract()
            cameras = getattr(self._device, "cameras", None) or {}
            if cameras:
                raise RuntimeError("remote SO-101 follower must be configured without cameras")
            self._device.bus.connect()
            if self._expected_serial_binding is not None:
                self._binding_confirmed = self._opened_binding_verifier(
                    self._device.bus,
                    self._expected_serial_binding,
                )
                if not self._binding_confirmed:
                    raise RuntimeError("opened follower adapter identity could not be confirmed")
            receipt = self._disable_torque("connect_safe")
            self.last_stop_receipt = receipt
            if receipt.torque_off_confirmed is not True:
                raise RuntimeError("SO-101 follower torque-off could not be confirmed after bus connect")
            if not self._device.is_calibrated:
                raise RuntimeError("SO-101 follower calibration does not match the connected servos")
            self._configure_disarmed()
            receipt = self._verify_torque_off("connect_configured")
            self.last_stop_receipt = receipt
            if receipt.torque_off_confirmed is not True:
                raise RuntimeError("SO-101 follower torque-off could not be confirmed after configuration")
            self._connected = True
            self._execution_enabled = False
            self._current_positions = self._read_positions()
            try:
                self._health_sampler = FeetechHealthSampler(
                    self._device.bus,
                    owner="remote_teleoperation",
                    arm="follower",
                )
            except Exception:
                # Diagnostics are supplemental; they never make a safe
                # follower connection fail or create another bus owner.
                self._health_sampler = None
        except Exception:
            try:
                if self._binding_confirmed:
                    self.last_stop_receipt = self._disable_torque("connect_failed")
            finally:
                self.last_close_receipt = self._close_device()
            raise

    def _configure_disarmed(self) -> None:
        """Apply the pinned LeRobot register contract without restoring torque."""
        bus = self._device.bus
        config = self._device.config
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, config.position_p_coefficient)
            bus.write("I_Coefficient", motor, config.position_i_coefficient)
            bus.write("D_Coefficient", motor, config.position_d_coefficient)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)

    def _verify_torque_off(self, reason: str) -> StopHardwareReceipt:
        confirmed = self._read_torque_state(expected=0)
        return StopHardwareReceipt(
            reason,
            disable_requested=True,
            torque_off_confirmed=confirmed,
            verification=(
                "feetech_torque_enable_readback"
                if confirmed is True
                else "feetech_torque_enable_readback_nonzero"
                if confirmed is False
                else "feetech_torque_enable_readback_unavailable"
            ),
            fault=(
                None
                if confirmed is True
                else "torque_still_enabled"
                if confirmed is False
                else "torque_state_unknown"
            ),
            connect_transient_torque_risk=False,
            hardware_stop_completed=True,
        )

    def observe(self) -> Mapping[str, float]:
        if not self._connected:
            raise RuntimeError("SO-101 follower adapter is disconnected")
        self._current_positions = self._read_positions()
        return dict(self._current_positions)

    def execute(self, positions: Mapping[str, float]) -> Mapping[str, float]:
        if not self._connected:
            raise RuntimeError("SO-101 follower adapter is disconnected")
        targets = self.joint_schema.validate_positions(positions)
        if not self._execution_enabled:
            # ARM transition: program the observed pose before torque is
            # enabled.  The remote target is intentionally deferred until the
            # next executor tick, preventing a jump to a stale EEPROM goal.
            # Re-read here because the torque-free arm may have moved since
            # session negotiation observed it.
            self._current_positions = self._read_positions()
            self._device.send_action(dict(self._current_positions))
            self._device.bus.enable_torque(num_retry=5)
            enabled = self._read_torque_state(expected=1)
            if enabled is not True:
                raise RuntimeError("SO-101 follower torque enable could not be confirmed")
            self._execution_enabled = True
            return dict(self._current_positions)

        executed = self._device.send_action(targets)
        clean = self.joint_schema.validate_positions(executed)
        self._current_positions = clean
        return dict(clean)

    def sample_health(self) -> Mapping[str, object] | None:
        sampler = self._health_sampler
        if sampler is None or not self._connected:
            return None
        sampler.sample_one()
        return sampler.snapshot()

    def stop(self, reason: str) -> Mapping[str, object]:
        receipt = self._disable_torque(reason)
        self._execution_enabled = False
        self.last_stop_receipt = receipt
        return receipt.public()

    def close(self) -> Mapping[str, object]:
        receipt = self._close_device()
        self.last_close_receipt = receipt
        return receipt.public()

    def _read_positions(self) -> dict[str, float]:
        observation = self._device.get_observation()
        if not isinstance(observation, Mapping):
            raise ValueError("SO-101 follower returned a non-mapping observation")
        try:
            positions = {key: observation[key] for key in self.joint_names}
        except KeyError as exc:
            raise ValueError("SO-101 follower observation is missing an action joint") from exc
        return self.joint_schema.validate_positions(positions)

    def _disable_torque(self, reason: str) -> StopHardwareReceipt:
        if self._expected_serial_binding is not None and self._binding_confirmed is not True:
            return StopHardwareReceipt(
                reason,
                disable_requested=False,
                torque_off_confirmed=None,
                verification="opened_adapter_identity_unconfirmed",
                fault="opened_adapter_identity_unconfirmed",
                connect_transient_torque_risk=False,
                hardware_stop_completed=False,
            )
        bus = getattr(self._device, "bus", None)
        motors = getattr(bus, "motors", None)
        if bus is None or not isinstance(motors, Mapping):
            return StopHardwareReceipt(
                reason,
                disable_requested=False,
                torque_off_confirmed=None,
                verification="unavailable",
                fault="feetech_bus_unavailable",
                connect_transient_torque_risk=False,
                hardware_stop_completed=False,
            )
        if not getattr(bus, "is_connected", False):
            return StopHardwareReceipt(
                reason,
                disable_requested=True,
                torque_off_confirmed=None,
                verification="bus_disconnected_no_readback",
                fault=None,
                connect_transient_torque_risk=False,
                hardware_stop_completed=False,
            )

        write_errors: list[str] = []
        for motor in motors:
            try:
                bus.disable_torque(motor, num_retry=5)
            except Exception as exc:  # continue so one bad servo cannot strand the others
                write_errors.append(f"{motor}:{type(exc).__name__}")

        confirmed = self._read_torque_state(expected=0)
        if confirmed is True:
            verification = (
                "feetech_torque_enable_readback_after_write_error"
                if write_errors
                else "feetech_torque_enable_readback"
            )
            fault = None
        elif confirmed is False:
            verification = "feetech_torque_enable_readback_nonzero"
            fault = "torque_still_enabled"
        else:
            verification = "feetech_torque_enable_readback_unavailable"
            suffix = ",".join(write_errors)
            fault = "torque_state_unknown" if not suffix else f"torque_state_unknown:{suffix}"
        return StopHardwareReceipt(
            reason,
            True,
            confirmed,
            verification,
            fault,
            connect_transient_torque_risk=False,
            hardware_stop_completed=True,
        )

    def _read_torque_state(self, *, expected: int) -> bool | None:
        bus = self._device.bus
        saw_mismatch = False
        saw_unknown = False
        for motor in bus.motors:
            try:
                value = bus.read("Torque_Enable", motor, normalize=False, num_retry=5)
            except Exception:
                saw_unknown = True
                continue
            if value != expected:
                saw_mismatch = True
        if saw_mismatch:
            return False
        if saw_unknown:
            return None
        return True

    def _close_device(self) -> CloseReceipt:
        bus = getattr(self._device, "bus", None)
        faults: list[str] = []
        # Do not ask LeRobot to infer safety during close.  stop() already made
        # the explicit per-servo request and read it back; close only releases
        # handles, even when the receipt is unresolved.
        if bus is not None and getattr(bus, "is_connected", False):
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                faults.append(f"bus:{type(exc).__name__}")
        for name, camera in (getattr(self._device, "cameras", None) or {}).items():
            if getattr(camera, "is_connected", False):
                try:
                    camera.disconnect()
                except Exception as exc:
                    faults.append(f"camera_{name}:{type(exc).__name__}")
        self._connected = False
        self._execution_enabled = False
        self._health_sampler = None
        if faults:
            logger.error("SO-101 follower close incomplete: %s", ",".join(faults))
        return CloseReceipt(True, not faults, None if not faults else ",".join(faults))
