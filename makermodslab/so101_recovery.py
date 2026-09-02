"""Explicit local torque-off recovery for crashed SO-101/Feetech owners.

This is not a normal robot connection. It opens only the exact serial buses
bound into a durable ``so101_recovery`` lease, with the Feetech handshake
disabled, writes ``Torque_Enable=0`` to every expected servo, verifies every
readback, and closes every bus. No upstream SO101 robot is constructed and no
configuration or torque-enable path is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .api_errors import ApiError, ErrorCode
from .hardware_lease import (
    HardwareLeaseHeld,
    HardwareRecoveryIdentity,
    hardware_lease_registry,
    safe_hardware_receipt,
)
from .serial_port_identity import opened_serial_binding_matches, resolve_serial_port_bindings


class ReleaseSo101TorqueRequest(BaseModel):
    """Exact serial targets from the durable SO-101 recovery identity."""

    ports: list[str] | None = Field(default=None, min_length=1)

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, ports: list[str] | None) -> list[str] | None:
        if ports is None:
            return None
        clean = [port.strip() for port in ports]
        if any(not port for port in clean):
            raise ValueError("every SO-101 recovery port must be non-empty")
        if len(set(clean)) != len(clean):
            raise ValueError("SO-101 recovery ports must be unique")
        return clean


@dataclass(frozen=True, slots=True)
class _PortRecoveryResult:
    binding_confirmed: bool
    torque_off: bool
    closed: bool
    problems: tuple[str, ...]


def _build_feetech_bus(port: str):
    """Build the raw bus without constructing an upstream SO101 robot."""
    from lerobot.motors.feetech import FeetechMotorsBus

    from .vendor.feetech_autocal.calibration_defaults import SO_FOLLOWER_MOTORS

    return FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())


def _recover_port(port: str, index: int, expected_binding: str) -> _PortRecoveryResult:
    problems: list[str] = []
    connected = False
    binding_confirmed = False
    closed = False
    readbacks: list[bool] = []
    label = f"port[{index}]"
    try:
        bus = _build_feetech_bus(port)
    except Exception as exc:
        return _PortRecoveryResult(
            False,
            False,
            False,
            (f"{label} open: {type(exc).__name__}",),
        )
    try:
        # handshake=False opens transport only. In particular, it does not
        # ping/configure the arm or invoke the SO101 torque-enable sequence.
        bus.connect(handshake=False)
        connected = True
        binding_confirmed = opened_serial_binding_matches(bus, expected_binding)
        if not binding_confirmed:
            problems.append(f"{label} identity: opened adapter could not be confirmed")
        else:
            for motor in bus.motors:
                try:
                    bus.write("Torque_Enable", motor, 0, normalize=False, num_retry=5)
                except Exception as exc:
                    problems.append(f"{label} disable {motor}: {type(exc).__name__}")
            for motor in bus.motors:
                try:
                    value = bus.read("Torque_Enable", motor, normalize=False, num_retry=5)
                except Exception as exc:
                    problems.append(f"{label} verify {motor}: {type(exc).__name__}")
                    readbacks.append(False)
                else:
                    confirmed = not isinstance(value, bool) and value == 0
                    # Some test/SDK shims return bool for one-byte registers. False
                    # is still an exact zero readback; True is energized.
                    if isinstance(value, bool):
                        confirmed = value is False
                    readbacks.append(confirmed)
                    if not confirmed:
                        problems.append(f"{label} verify {motor}: torque still enabled")
    except Exception as exc:
        problems.append(f"{label} open: {type(exc).__name__}")
    finally:
        try:
            # Attempt close even when connect raised: a serial open can fail
            # after acquiring the file descriptor but before returning. Torque
            # was already handled above, so never invoke an implicit write pass.
            bus.disconnect(disable_torque=False)
            closed = True
        except Exception as exc:
            problems.append(f"{label} close: {type(exc).__name__}")

    torque_off = binding_confirmed and connected and len(readbacks) == len(bus.motors) and all(readbacks)
    return _PortRecoveryResult(binding_confirmed, torque_off, closed, tuple(problems))


def _held_error(exc: HardwareLeaseHeld) -> ApiError:
    snapshot = exc.snapshot
    return ApiError(
        status_code=409,
        detail="The requested SO-101 ports do not match the recoverable hardware claim.",
        code=ErrorCode.SESSION_HELD,
        details={
            "holder": {
                "kind": snapshot.kind,
                "session_id": None,
                "state": snapshot.state,
            }
        },
    )


def handle_release_so101_torque(request: ReleaseSo101TorqueRequest) -> dict[str, Any]:
    """Verify torque-off and safe close on every exact Feetech target."""
    if not hardware_lease_registry.snapshot().held:
        # Compatibility protection for a process upgraded while a legacy
        # feature-local flag was already active.
        from .sessions import _held_by

        legacy_holder = _held_by()
        if legacy_holder is not None:
            raise ApiError(
                status_code=409,
                detail="The robot hardware is held by an active local session.",
                code=ErrorCode.SESSION_HELD,
                details={"holder": {"kind": legacy_holder, "session_id": None}},
            )

    ports = tuple(request.ports or hardware_lease_registry.recovery_targets("so101_recovery"))
    if not ports:
        raise ApiError(
            status_code=409,
            detail="No retained SO-101 recovery targets are available; enter the exact ports.",
            code=ErrorCode.SESSION_HELD,
        )
    bindings = resolve_serial_port_bindings(ports)
    if set(bindings) != set(ports):
        raise ApiError(
            status_code=409,
            detail=(
                "Every requested SO-101 port must expose a stable USB serial and VID/PID "
                "before automatic recovery."
            ),
            code=ErrorCode.SESSION_HELD,
        )
    expected = HardwareRecoveryIdentity.from_bound_targets(
        "so101_recovery",
        "so101",
        bindings,
    )
    try:
        lease_token = hardware_lease_registry.begin_recovery(
            "local:so101-recovery",
            expected=expected,
        )
    except HardwareLeaseHeld as exc:
        raise _held_error(exc) from exc

    results: list[_PortRecoveryResult] = []
    try:
        for index, port in enumerate(ports):
            results.append(_recover_port(port, index, bindings[port]))
    except Exception as exc:  # defensive: never strand a recovering lease
        evidence = f"SO-101 recovery encountered an internal {type(exc).__name__}"
        hardware_lease_registry.mark_unresolved(
            lease_token,
            evidence,
            {
                "safe": False,
                "device_closed": False,
                "torque_off": False,
                "evidence": evidence,
            },
        )
        return {
            "success": False,
            "message": "SO-101 recovery remains locked out after an internal error.",
            "confirmed_ports": sum(result.torque_off and result.closed for result in results),
            "problems": [f"recovery: {type(exc).__name__}"],
        }
    problems = [problem for result in results for problem in result.problems]
    all_bindings_confirmed = all(result.binding_confirmed for result in results)
    all_torque_off = all(result.torque_off for result in results)
    all_closed = all(result.closed for result in results)
    if problems or not all_bindings_confirmed or not all_torque_off or not all_closed:
        evidence = "SO-101 recovery did not confirm torque-off and close on every requested bus"
        hardware_lease_registry.mark_unresolved(
            lease_token,
            evidence,
            {
                "safe": False,
                "device_closed": all_closed,
                "torque_off": all_torque_off,
                "evidence": evidence,
            },
        )
        return {
            "success": False,
            "message": "SO-101 recovery remains locked out; not every bus was confirmed safe.",
            "confirmed_ports": sum(result.torque_off and result.closed for result in results),
            "problems": problems,
        }

    hardware_lease_registry.release(
        lease_token,
        safe_hardware_receipt("every SO-101 bus confirmed torque disabled and closed"),
    )
    return {
        "success": True,
        "message": "Every requested SO-101 bus confirmed torque disabled and closed.",
        "confirmed_ports": len(results),
        "problems": [],
    }
