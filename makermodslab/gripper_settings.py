"""Settings for the local Metal gripper current-limit experiment."""

import math

DEFAULT_GRIPPER_CURRENT_A = 0.5
MIN_GRIPPER_CURRENT_A = 0.1
MAX_GRIPPER_CURRENT_A = 2.0
GRIPPER_STOP_C = 55.0
GRIPPER_RESTART_C = 45.0
DEFAULT_GRIPPER_HOLD_TORQUE_NM = 0.5


def supports_gripper_effort_control(arm_type: str) -> bool:
    from .arms import registry

    try:
        return registry.get(arm_type).supports_gripper_effort_control
    except registry.UnknownArmType:
        return False


def default_gripper_hold_torque(arm_type: str, current_limit: object = None) -> float | None:
    """Default only Metal followers without an explicitly selected current mode."""
    if supports_gripper_effort_control(arm_type) and current_limit is None:
        return DEFAULT_GRIPPER_HOLD_TORQUE_NM
    return None


def validate_gripper_hold_torque(value: object) -> float | None:
    """Experimental motor-output holding effort, not an instantaneous limit."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.1 <= value <= 2.0
    ):
        raise ValueError("Gripper holding torque must be between 0.1 and 2.0 N·m, or null to disable.")
    return float(value)


def validate_gripper_current(value: object) -> float | None:
    """None disables the experiment; invalid enabled settings never fail open."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not MIN_GRIPPER_CURRENT_A <= value <= MAX_GRIPPER_CURRENT_A
    ):
        raise ValueError("Gripper current limit must be between 0.1 and 2.0 A, or null to disable.")
    return float(value)
