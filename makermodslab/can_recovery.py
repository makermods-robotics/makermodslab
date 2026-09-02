# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""De-energize exact CAN follower targets after an unclean owner exit.

Every normal stop path releases torque on its way out. What none of them can
reach is an arm whose PROCESS died: a SIGKILL or power loss leaves Damiao
motors holding their last MIT command indefinitely (RobStride at least times
out), and the next server process starts with no device object, no session,
and no cleanup owing — just a rigid arm on a port. This module is the
deliberate, user-invoked answer: open every bound follower bus WITHOUT the
energizing handshake, require every motor to acknowledge disable, then close.

Deliberately NOT a session. It must be usable exactly when session state is
wrecked, so it adopts only a durable recovery capability for the exact CAN
family and complete target set. It cannot clear an SO-101, Star-UART, or
different-target latch. SO-101 has its own readback-backed recovery route.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .api_errors import ApiError, ErrorCode
from .hardware_lease import (
    HardwareLeaseHeld,
    HardwareRecoveryIdentity,
    hardware_lease_registry,
    safe_hardware_receipt,
)
from .hardware_recovery_identity import resolve_can_port_bindings

logger = logging.getLogger(__name__)


class ReleaseCanTorqueRequest(BaseModel):
    # This route constructs CAN followers only. SO-101 uses the separate
    # Feetech recovery route and Star-UART leaders remain outside this route.
    arm_type: Literal["maker", "metal"]
    # Legacy single-target spelling remains accepted.
    port: str | None = None
    # Exact multi-target spelling for bimanual and other grouped CAN claims.
    ports: list[str] | None = Field(default=None, min_length=1)

    @field_validator("port")
    @classmethod
    def validate_port(cls, port: str | None) -> str | None:
        if port is None:
            return None
        clean = port.strip()
        if not clean:
            raise ValueError("CAN recovery port must be non-empty")
        return clean

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, ports: list[str] | None) -> list[str] | None:
        if ports is None:
            return None
        clean = [port.strip() for port in ports]
        if any(not port for port in clean):
            raise ValueError("every CAN recovery port must be non-empty")
        if len(set(clean)) != len(clean):
            raise ValueError("CAN recovery ports must be unique")
        return clean

    @model_validator(mode="after")
    def select_one_explicit_shape(self) -> ReleaseCanTorqueRequest:
        if self.port is not None and self.ports is not None:
            raise ValueError("use either port or ports for CAN recovery, not both")
        return self


@dataclass(frozen=True, slots=True)
class _TargetRecoveryResult:
    torque_off: bool
    closed: bool
    problems: tuple[str, ...]


def _build_follower_device(arm_type: str, port: str):
    """The follower device whose bus gets de-energized.

    Built through the same factory helpers the calibration flows use, with a
    throwaway id — construction does no device I/O, and the calibration file
    (if any) is irrelevant to a torque release.
    """
    from lerobot.robots import make_robot_from_config

    from .utils.robot_factory import maker_follower_config, metal_follower_config

    builder = metal_follower_config if arm_type == "metal" else maker_follower_config
    return make_robot_from_config(builder(port, "recovery"))


def _recover_target(
    arm_type: str,
    port: str,
    index: int,
    expected_binding: str,
) -> _TargetRecoveryResult:
    """Disable one CAN follower without letting vendor logs expose its path."""
    # Both pinned CAN drivers log their raw channel during connect. The path is
    # owner-private recovery material, so suppress that exact driver logger
    # across direct I/O and publish a sanitized result after it exits.
    driver_logger_name = (
        "lerobot.motors.damiao.damiao" if arm_type == "metal" else "lerobot.motors.robstride.robstride"
    )
    driver_logger = logging.getLogger(driver_logger_name)
    was_disabled = driver_logger.disabled
    driver_logger.disabled = True
    try:
        return _recover_target_silenced(arm_type, port, index, expected_binding)
    finally:
        driver_logger.disabled = was_disabled


def _recover_target_silenced(
    arm_type: str,
    port: str,
    index: int,
    expected_binding: str,
) -> _TargetRecoveryResult:
    """Perform one acknowledgement-backed recovery while logging is suppressed."""
    label = f"target[{index}]"
    problems: list[str] = []
    try:
        device = _build_follower_device(arm_type, port)
        bus = device.bus
    except Exception as exc:
        return _TargetRecoveryResult(False, False, (f"{label} construct: {type(exc).__name__}",))

    connected = bool(getattr(bus, "is_connected", False))
    closed = False
    disable_completed = False
    identity_proven = False
    expected_acks: set[object] = set()
    received_acks: set[object] = set()
    original_receiver = None
    receiver_was_instance_attribute = False
    try:
        if not connected:
            bus.connect(handshake=False)
            connected = True

        # Re-resolve after open so a pathname/interface rebound between lease
        # adoption and bus construction cannot authorize a replacement adapter.
        observed_binding = resolve_can_port_bindings((port,)).get(port)
        identity_proven = observed_binding == expected_binding
        if not identity_proven:
            problems.append(f"{label} adapter identity changed after open")
        else:
            motors = tuple((getattr(bus, "motors", None) or {}).keys())
            receiver = getattr(bus, "_recv_motor_response", None)
            receiver_id = getattr(bus, "_get_motor_recv_id", None)
            if not motors or not callable(receiver) or not callable(receiver_id):
                problems.append(f"{label} exposes no per-motor disable acknowledgement")
            else:
                try:
                    expected_acks = {receiver_id(motor) for motor in motors}
                except Exception as exc:
                    problems.append(f"{label} acknowledgement identity: {type(exc).__name__}")
                if None in expected_acks or len(expected_acks) != len(motors):
                    problems.append(f"{label} acknowledgement identities are incomplete")
                    expected_acks = set()
                elif expected_acks:
                    original_receiver = receiver
                    receiver_was_instance_attribute = hasattr(
                        bus, "__dict__"
                    ) and "_recv_motor_response" in vars(bus)

                    def tracked_receiver(*args, **kwargs):
                        response = original_receiver(*args, **kwargs)
                        expected = kwargs.get("expected_recv_id")
                        if expected is None and args:
                            expected = args[0]
                        if response is not None and expected in expected_acks:
                            received_acks.add(expected)
                        return response

                    bus._recv_motor_response = tracked_receiver

            try:
                bus.disable_torque()
                disable_completed = True
            except Exception as exc:
                problems.append(f"{label} disable: {type(exc).__name__}; TORQUE MAY STILL BE ENABLED")
            finally:
                if original_receiver is not None:
                    if receiver_was_instance_attribute:
                        bus._recv_motor_response = original_receiver
                    else:
                        del bus._recv_motor_response

            if expected_acks and received_acks != expected_acks:
                problems.append(f"not every motor on {label} acknowledged the disable command")
    except Exception as exc:
        problems.append(f"{label} open: {type(exc).__name__}")
    finally:
        try:
            # Torque was handled explicitly; do not trigger another implicit
            # disable while the transport is closing.
            bus.disconnect(disable_torque=False)
            closed = True
        except Exception as exc:
            problems.append(f"{label} close: {type(exc).__name__}")

    torque_off = (
        connected
        and identity_proven
        and disable_completed
        and bool(expected_acks)
        and received_acks == expected_acks
    )
    return _TargetRecoveryResult(torque_off, closed, tuple(problems))


def _held_error(exc: HardwareLeaseHeld) -> ApiError:
    snapshot = exc.snapshot
    return ApiError(
        status_code=409,
        detail="The requested CAN targets do not match the recoverable hardware claim.",
        code=ErrorCode.SESSION_HELD,
        details={
            "holder": {
                "kind": snapshot.kind,
                # Owners of ordinary local claims may contain raw device
                # paths. Recovery errors are public API payloads, so expose
                # only the sanitized claim kind/state.
                "session_id": None,
                "state": snapshot.state,
            }
        },
    )


def handle_release_can_torque(request: ReleaseCanTorqueRequest) -> dict[str, Any]:
    """Recover every exact CAN target without exposing private device paths.

    Returns ``{"success", "message", "problems"}``; success means every
    disable landed. Refuses with 409 session.held while any feature's active
    flag holds the hardware.
    """
    # Compatibility flags are status projections, not authority. They still
    # guard older/injected callers that predate registry claims (including a
    # worker already live during a rolling server upgrade). Only consult them
    # while the authoritative registry is idle; an unresolved registry lease
    # is exactly what this recovery operation is allowed to adopt.
    if not hardware_lease_registry.snapshot().held:
        from .sessions import _held_by

        legacy_holder = _held_by()
        if legacy_holder is not None:
            raise ApiError(
                status_code=409,
                detail=(f"The robot hardware is held by an active {legacy_holder} session. Stop it first."),
                code=ErrorCode.SESSION_HELD,
                details={"holder": {"kind": legacy_holder, "session_id": None}},
            )

    ports = tuple(
        request.ports
        or ([request.port] if request.port is not None else [])
        or hardware_lease_registry.recovery_targets("can_recovery")
    )
    if hardware_lease_registry.recovery_targets("can_physical_recovery"):
        raise ApiError(
            status_code=409,
            detail=(
                "The retained CAN claim requires physical/manual recovery; "
                "software recovery cannot clear its fault lockout."
            ),
            code=ErrorCode.SESSION_HELD,
        )
    if not ports:
        raise ApiError(
            status_code=409,
            detail="No retained CAN recovery targets are available; enter the exact ports.",
            code=ErrorCode.SESSION_HELD,
        )
    snapshot = hardware_lease_registry.snapshot()
    if snapshot.held and snapshot.kind in {"calibration", "diagnostic"}:
        # These older claim shapes may identify a Star-UART leader or a mixed
        # discovery candidate set. A target digest proves which path, not which
        # bus protocol lives there, so this follower-only route must not guess.
        raise ApiError(
            status_code=409,
            detail=(
                "The retained claim does not prove that every target is a CAN follower; "
                "use a side-aware recovery route."
            ),
            code=ErrorCode.SESSION_HELD,
            details={
                "holder": {
                    "kind": snapshot.kind,
                    "session_id": None,
                    "state": snapshot.state,
                }
            },
        )
    bindings = resolve_can_port_bindings(ports)
    if set(bindings) != set(ports):
        raise ApiError(
            status_code=409,
            detail=(
                "Stable CAN adapter identity is unavailable for every target; "
                "retain lockout and use physical/manual recovery."
            ),
            code=ErrorCode.SESSION_HELD,
        )
    recovery = HardwareRecoveryIdentity.from_bound_targets(
        "can_recovery",
        request.arm_type,
        bindings,
    )
    try:
        lease_token = hardware_lease_registry.begin_recovery(
            "local:can-recovery",
            expected=recovery,
        )
    except HardwareLeaseHeld as exc:
        raise _held_error(exc) from exc

    family = "Metal" if request.arm_type == "metal" else "Maker"
    logger.info("Releasing torque on %d %s CAN recovery target(s)", len(ports), family)
    results: list[_TargetRecoveryResult] = []
    try:
        for index, port in enumerate(ports):
            results.append(_recover_target(request.arm_type, port, index, bindings[port]))
    except Exception as exc:  # defensive: never strand a recovering lease
        evidence = f"CAN recovery encountered an internal {type(exc).__name__}"
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
            "message": "CAN recovery remains locked out after an internal error.",
            "problems": [f"recovery: {type(exc).__name__}"],
        }
    problems = [problem for result in results for problem in result.problems]
    all_torque_off = all(result.torque_off for result in results)
    all_closed = all(result.closed for result in results)
    if problems or not all_torque_off or not all_closed:
        evidence = "CAN recovery did not confirm torque-off and close on every requested bus"
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
        logger.warning(
            "CAN recovery retained fault lockout; %d of %d target(s) were fully confirmed",
            sum(result.torque_off and result.closed for result in results),
            len(results),
        )
        return {
            "success": False,
            "message": "CAN recovery remains locked out; not every bus was confirmed safe.",
            "problems": problems,
        }
    hardware_lease_registry.release(
        lease_token,
        safe_hardware_receipt("CAN recovery confirmed torque disabled and bus closed"),
    )
    logger.info("CAN recovery confirmed all %d target(s) safe", len(results))
    return {
        "success": True,
        "message": f"Every requested {family} CAN follower confirmed torque disabled and closed.",
        "problems": [],
    }
