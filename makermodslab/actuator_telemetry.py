"""Read Maker feedback already cached by the control loop. No bus I/O or motor commands."""

from __future__ import annotations

import math
import time

from .thermal_limits import CRITICAL_AT_C, STOP_AT_C, thermal_policy

STALE_AFTER_S = 1.0


def finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def temperature_status(temperature: float | None) -> str:
    """Classify at the software cutoff and the manufacturer-reported reference."""
    if temperature is None:
        return "NO DATA"
    if temperature >= CRITICAL_AT_C:
        return "CRITICAL"
    if temperature >= STOP_AT_C:
        return "OVERHEATING"
    return "OK"


def cached_maker_telemetry(robot, now: float | None = None) -> dict:
    """Copy the current motor cache; never connect, refresh, read or write a bus.

    Call from the same thread that owns the robot, after an existing observation.
    An uninitialized cache is NOT a zero-temperature sample. The driver reports
    feedback temperature as temp_mos; the RS02 July 2026 MIT manual labels this
    field winding temperature. No separate board temperature is available here.
    """
    now = time.time() if now is None else now
    bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
    arms = (("left", robot.left_arm), ("right", robot.right_arm)) if bimanual else (("follower", robot),)
    samples = []
    for side, arm in arms:
        bus = getattr(arm, "bus", None)
        # Holding wrappers delegate to the actual RobStride cache.
        bus = getattr(bus, "_base", bus)
        states = getattr(bus, "_last_known_states", {})
        times = getattr(bus, "last_feedback_time", {})
        for name in getattr(bus, "motors", {}):
            state = dict(states.get(name, {}))
            stamp = finite_number(times.get(name))
            age = now - stamp if stamp is not None else None
            temperature = finite_number(state.get("temp_mos"))
            torque = finite_number(state.get("torque"))
            reason = ""
            if stamp is None or stamp <= 0:
                status, temperature, torque, age = "NO DATA", None, None, None
            elif age < 0:
                status, reason = "INVALID", "Feedback timestamp is in the future"
            elif temperature is None or not 0 <= temperature <= 409.5 or torque is None:
                # Do not guess firmware format by masking a large reading: newer
                # MIT status bits need an explicitly compatible driver decoder.
                status, reason = "INVALID", "Invalid feedback; check driver temperature decoding"
                temperature = None
            elif age > STALE_AFTER_S:
                status = "STALE"
            else:
                status = temperature_status(temperature)
            samples.append(
                {
                    "actuator": f"{side}.{name}",
                    "temperature_c": temperature,
                    "torque_nm": torque,
                    "feedback_ts": stamp,
                    "feedback_age_s": age,
                    "status": status,
                    "reason": reason,
                }
            )
    return {"timestamp": now, "stale_after_s": STALE_AFTER_S, "actuators": samples, **thermal_policy()}
