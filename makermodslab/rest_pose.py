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
"""Return an arm to a captured rest pose before releasing torque.

Stop paths that cut torque instantly drop the arm under gravity. The STS3215
has no torque-control mode for true gravity compensation, so the graceful
alternative is: capture the pose the arm started the session in, and on a
normal stop drive it back there — progress-based (a slow return is never cut
short, only a stuck one gives up), gently, and abortable — then release.

MIRROR NOTE: the vendored auto-calibration script carries its own twin of this
logic (makermodslab/vendor/feetech_autocal/auto_calibrate_script.py, "Graceful stop"
section). It runs as a standalone subprocess and cannot import makermodslab cleanly,
so the two are kept mirrored by hand — change one, check the other. The
vendored twin additionally re-expresses targets against Homing_Offset (the
calibration rewrites offsets mid-run); THIS module stores raw Present_Position
ticks, which is valid precisely because teleoperation never touches
Homing_Offset mid-session.

Register facts (pinned lerobot, lerobot/motors/feetech/tables.py):
``Goal_Position`` (42), ``Goal_Velocity`` (46, the profile speed in position
mode), ``Present_Position`` (56, read-only), ``Moving`` (66, read-only).
"""

import contextlib
import logging
import threading
import time

from lerobot.motors import MotorNormMode

logger = logging.getLogger(__name__)

# Bounds lerobot's own bus.sync_write/sync_read clamp normalized values into
# (motors_bus.py's _unnormalize/_normalize) — DEGREES mode is left out on
# purpose, since lerobot doesn't clamp that one either.
_NORMALIZED_BOUNDS: dict[MotorNormMode, tuple[float, float]] = {
    MotorNormMode.RANGE_M100_100: (-100.0, 100.0),
    MotorNormMode.RANGE_0_100: (0.0, 100.0),
}


def _clamp_to_representable_range(motors: dict, targets: dict[str, int | float]) -> dict[str, int | float]:
    """Clamp each normalized target into the range the bus can actually
    write/read back (see _NORMALIZED_BOUNDS). A dataset-recorded action can
    fall outside a motor's calibrated range; lerobot clamps it on both the
    write and read path, so comparing against the unclamped value would
    compare against a position the bus can never report — the arrival check
    would never pass no matter how close the arm gets."""
    clamped = dict(targets)
    for motor, value in targets.items():
        bounds = _NORMALIZED_BOUNDS.get(getattr(motors.get(motor), "norm_mode", None))
        if bounds is not None:
            lo, hi = bounds
            clamped[motor] = min(hi, max(lo, float(value)))
    return clamped


# Gentle return speed (ticks/s-ish register units; teleop snaps run at full).
RETURN_POS_SPEED = 400
RETURN_SETTLE_S = 0.3  # let motion start before polling
RETURN_POLL_S = 0.05
# Stall detection: the return may be slow, never stuck. Give up only when the
# aggregate distance-to-target fails to shrink by more than
# RETURN_STALL_MIN_PROGRESS ticks for a full RETURN_STALL_WINDOW_S window.
RETURN_STALL_WINDOW_S = 1.5
RETURN_STALL_MIN_PROGRESS = 10  # ticks, summed over all motors
# Absolute ceiling, not a working limit: termination is progress-based, so
# this must never matter in practice — it only bounds a pathological loop.
RETURN_CEILING_S = 10.0
# Arrived when every motor is within this many ticks of its target (matches
# the auto-calibration mixin's POSITION_TOLERANCE).
RETURN_ARRIVE_TOLERANCE = 20


def capture_rest_pose(bus, normalize: bool = False) -> dict[str, int | float]:
    """Present_Position of every motor on one bus, or {} on failure.

    Call at session start, before any motion. Raw ticks (normalize=False,
    the default) are directly replayable as Goal_Position later because
    nothing in a teleop/record session rewrites Homing_Offset (see module
    docstring). normalize=True instead captures the same normalized units
    robot.send_action() uses — for a caller (replay's ease-in) whose target
    is itself an action dict, not a raw-ticks pose. Never raises — a session
    must not fail to start over an optional nicety.
    """
    try:
        return dict(bus.sync_read("Present_Position", normalize=normalize))
    except Exception as e:
        port = getattr(bus, "port", None) or "unknown port"
        logger.warning(f"Could not capture the rest pose on {port}: {e}")
        return {}


def return_to_rest_pose(
    bus,
    rest_pose: dict[str, int | float],
    abort_event: threading.Event | None = None,
    label: str = "arm",
    normalize: bool = False,
    tolerance: float | None = None,
    stall_min_progress: float | None = None,
) -> tuple[bool, str]:
    """Drive one bus's motors back to ``rest_pose``, then report how it went.

    Returns ``(arrived, reason)`` with reason starting with one of
    ``returned`` (with per-motor |final - target| deltas), ``settled`` (every
    motor stopped — Moving == 0 — but short of target: a latched or weak
    motor, reported with its deltas), ``stalled``, ``ceiling``, ``cut-short``
    (abort_event set — a second stop or a new session start), ``no-pose``, or
    ``comm-error: ...``. Success requires every motor within ``tolerance``
    (default RETURN_ARRIVE_TOLERANCE) of its target — Moving == 0 alone is
    NOT success, precisely because a weak motor can sit motionless far from
    its target and that must not be reported as "returned".

    Torque must still be enabled (call BEFORE force_disable_torque).

    ``normalize`` selects the unit space for both the Goal_Position write and
    the Present_Position polling comparison — False (default, raw ticks) for
    a pose captured by capture_rest_pose(bus) (unchanged behavior for every
    existing caller); True for a target already expressed in
    robot.send_action()'s normalized units (e.g. a dataset's recorded
    `action` frame), where the bus's own calibration-aware
    normalize/unnormalize handles any needed conversion and clamps
    out-of-range values rather than raising — no manual tick conversion
    needed. A target outside the bus's representable range (e.g. a recorded
    action beyond a motor's calibrated span) is itself clamped before both
    the write and the arrival comparison, so a saturated joint is compared
    against the value the bus will actually report back, not the unreachable
    original. Never raises.

    ``stall_min_progress`` (default RETURN_STALL_MIN_PROGRESS, raw ticks) is
    the aggregate distance-to-target the return must close within
    RETURN_STALL_WINDOW_S to avoid being reported as stalled — like
    ``tolerance``, it does NOT auto-scale with ``normalize``: the raw-ticks
    constant is ~15x too strict a bar in normalized-unit space (10 ticks of
    required progress vs. 10 normalized units, a much larger physical
    distance), so a normalized caller should pass its own value sized for its
    unit space rather than inherit the raw-ticks default.

    On EVERY exit path the gentle RETURN_POS_SPEED profile cap written into the
    motors' RAM Goal_Velocity is reset to 0 (uncapped) before returning, so it
    can't linger and throttle the next session — the RAM-persistent speed-cap
    hazard that makermodslab/motor_power.clear_goal_velocity also guards at session
    start (see _restore_goal_velocity).
    """
    motors = getattr(bus, "motors", None) or {}
    targets = {m: v for m, v in rest_pose.items() if m in motors}
    if not targets:
        logger.info(f"Rest-pose return skipped for the {label}: no captured pose")
        return False, "no-pose"
    if normalize:
        # Clamp before both the write and the arrival comparison, so the
        # comparison is against what the bus will actually report back for a
        # saturated joint (see _clamp_to_representable_range).
        targets = _clamp_to_representable_range(motors, targets)
    try:
        for motor in targets:
            bus.write("Goal_Velocity", motor, RETURN_POS_SPEED, normalize=False)
        bus.sync_write("Goal_Position", targets, normalize=normalize)
    except Exception as e:
        logger.warning(f"Rest-pose return failed to start for the {label}: {e}")
        # The gentle RETURN_POS_SPEED cap may already be stamped on some motors;
        # clear it so it can't throttle the next session (see _restore_goal_velocity).
        _restore_goal_velocity(bus, targets, label)
        return False, f"comm-error: {e}"

    try:
        return _run_return_loop(
            bus,
            targets,
            abort_event,
            normalize=normalize,
            tolerance=RETURN_ARRIVE_TOLERANCE if tolerance is None else tolerance,
            stall_min_progress=RETURN_STALL_MIN_PROGRESS
            if stall_min_progress is None
            else stall_min_progress,
        )
    finally:
        # Belt and braces: whatever the outcome (returned / settled / stalled /
        # ceiling / cut-short), we wrote the gentle RETURN_POS_SPEED cap into the
        # motors' RAM Goal_Velocity above. Reset it to 0 before the caller
        # releases torque, so a slow-return speed cap can't linger and throttle
        # the next teleop/record/inference session on this power-up. Best-effort.
        _restore_goal_velocity(bus, targets, label)


def _restore_goal_velocity(bus, targets: dict[str, int | float], label: str = "arm") -> None:
    """Reset Goal_Velocity to 0 (uncapped) on the motors the return just drove.

    The return writes a gentle RETURN_POS_SPEED profile cap into each motor's
    RAM Goal_Velocity; that value is RAM-persistent across sessions (only a
    power cycle resets it), so leaving it stamped would silently throttle the
    next session's follower moves (the same leftover-cap mechanism the vendored
    auto-cal fold/unfold=1000 and this module's return=400 both create — see
    makermodslab/motor_power.clear_goal_velocity, the primary session-start guard).
    Best-effort: a failed reset is logged and ignored — the release must run.
    """
    try:
        bus.sync_write("Goal_Velocity", dict.fromkeys(targets, 0), normalize=False)
    except Exception as e:
        logger.warning(f"Could not reset the return speed cap (Goal_Velocity) for the {label}: {e}")


def _run_return_loop(
    bus,
    targets: dict[str, int | float],
    abort_event: threading.Event | None,
    normalize: bool = False,
    tolerance: float = RETURN_ARRIVE_TOLERANCE,
    stall_min_progress: float = RETURN_STALL_MIN_PROGRESS,
) -> tuple[bool, str]:
    """The poll-until-arrived loop of return_to_rest_pose (see its docstring).

    Split out so return_to_rest_pose can wrap every exit path in a finally that
    resets the Goal_Velocity cap without duplicating the reset at each return.
    """
    time.sleep(RETURN_SETTLE_S)
    t0 = time.monotonic()
    best_dist: float | None = None
    last_progress_t = t0
    remaining = -1.0
    distances: dict[str, float] = {}
    unit = "units" if normalize else "ticks"
    while time.monotonic() - t0 < RETURN_CEILING_S:
        if abort_event is not None and abort_event.is_set():
            return False, "cut-short"
        try:
            positions = bus.sync_read("Present_Position", normalize=normalize)
        except Exception:
            positions = {}
        now = time.monotonic()
        if positions:
            if normalize:
                distances = {
                    m: abs(float(positions[m]) - float(t)) for m, t in targets.items() if m in positions
                }
            else:
                distances = {m: abs(int(positions[m]) - int(t)) for m, t in targets.items() if m in positions}
            remaining = sum(distances.values())
            if distances and all(d <= tolerance for d in distances.values()):
                return True, f"returned: max delta {max(distances.values())} {unit} ({_deltas(distances)})"
            if best_dist is None or remaining < best_dist - stall_min_progress:
                best_dist = remaining
                last_progress_t = now
        if now - last_progress_t > RETURN_STALL_WINDOW_S:
            detail = f"remaining {remaining} {unit} ({_deltas(distances)})"
            # Distinguish a motor still fighting (stalled) from one that gave
            # up and sits motionless short of target (settled: a latched or
            # weak motor) — a one-shot Moving read, only on this exit path.
            with contextlib.suppress(Exception):
                moving = bus.sync_read("Moving", normalize=False)
                if all(moving.get(m, 1) == 0 for m in targets):
                    return False, f"settled short of target after {now - t0:.1f}s, {detail}"
            return False, f"stalled after {now - t0:.1f}s, {detail}"
        time.sleep(RETURN_POLL_S)
    return False, f"ceiling ({RETURN_CEILING_S:.0f}s), remaining {remaining} {unit}"


def _deltas(distances: dict[str, int]) -> str:
    """Compact per-motor |final - target| report, e.g. 'shoulder_pan=4, ...'."""
    return ", ".join(f"{m}={d}" for m, d in distances.items()) or "no readable motors"
