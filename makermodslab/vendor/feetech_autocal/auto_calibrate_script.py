#!/usr/bin/env python

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

"""
Feetech STS servo auto-calibration (with unfold).

Full flow (single command):
  Stage 0  Initialization: stop all servos, Lock=1, configure PID/acceleration, enable torque
  Stage 2  Unfold joints 2-4 (skippable via --unfold-angle 0)
  Stage 3  Calibrate servos 2-6 (5 -> 6 -> 4 -> 3 -> 2)
  Stage 4  Calibrate servo 1 shoulder_pan last and return to mid
  Stage 5  Wait for user confirmation, then release torque

Usage examples:

  lerobot-auto-calibrate-feetech --port COM3
  lerobot-auto-calibrate-feetech --port COM3 --save
  lerobot-auto-calibrate-feetech --port COM3 --unfold-angle 0
  lerobot-auto-calibrate-feetech --port COM3 --save --robot-id default
  lerobot-auto-calibrate-feetech --port COM3 --unfold-only   # debug arm-unfold only (Stage 0 + Stage 2)
"""

import argparse
import contextlib
import signal
import sys
import time
from collections.abc import Callable

import draccus
from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# Vendored auto-calibration: upstream FeetechMotorsBus + the calibration mixin.
from .bus import COMM_ERR, AutoCalBus as FeetechMotorsBus
from .calibration_defaults import (
    CALIBRATE_FIRST,
    CALIBRATE_REST,
    DEFAULT_ACCELERATION,
    FULL_TURN,
    DEFAULT_D_COEFFICIENT,
    DEFAULT_I_COEFFICIENT,
    DEFAULT_MAX_TORQUE,
    DEFAULT_P_COEFFICIENT,
    DEFAULT_POS_SPEED,
    DEFAULT_TIMEOUT,
    DEFAULT_TORQUE_LIMIT,
    DEFAULT_UNFOLD_ANGLE,
    DEFAULT_UNFOLD_TIMEOUT,
    DEFAULT_VELOCITY_LIMIT,
    HOMING_OFFSET_MAX_MAG,
    MOTOR_NAMES,
    POSITION_TOLERANCE,
    SO_FOLLOWER_MOTORS,
    STS_HALF_TURN_RAW,
    UNFOLD_ORDER,
    motor_label,
)

# ====================== Graceful stop (freeze -> return to start -> release) ======================

# NOTE: makermodslab/rest_pose.py carries the MakerMods Lab-side twin of this return logic
# (teleoperation stop). This script runs as a standalone subprocess and cannot
# import makermodslab cleanly, so the two are kept mirrored by hand — change one,
# check the other.
#
# On a mid-calibration stop the arm is usually extended in mid-air with torque on
# (possibly still moving in velocity mode); cutting Torque_Enable instantly makes
# it free-fall. The STS3215 has no torque-control mode for true gravity
# compensation, so the stop sequence instead: (1) FREEZES all motion immediately
# — the user sees the Stop register right away — (2) drives the arm back to the
# pose it was in when the script started, at a gentle speed, for as long as it
# keeps making progress, and (3) only then releases torque. If the return can't
# run (pose not captured, bus error) or stalls, the arm holds its frozen pose
# briefly and is then released wherever it is.
#
# Budget: freeze (~1s) + return (<= the RETURN_TO_REST_BUDGET_S ceiling) +
# fallback hold (STOP_HOLD_S) + 6-motor release/disconnect (~1-2s) must stay
# inside the manager's SIGTERM grace, makermodslab.auto_calibrate._STOP_GRACE_S —
# sized to 18s for exactly this worst case.
#
# ABSOLUTE ceiling on the return, not a working limit: termination is
# progress-based (see the stall constants below), so a healthy-but-slow return
# is never cut short and this ceiling must never matter in practice — it
# exists so the manager's SIGKILL can't land mid-motion.
RETURN_TO_REST_BUDGET_S = 10.0
RETURN_POS_SPEED = 400  # gentle return speed (fold/unfold move at DEFAULT_POS_SPEED=1000)
RETURN_SETTLE_S = 0.3  # let motion start before polling (mirrors _fold_arm)
RETURN_POLL_S = 0.05
# Stall detection: the return is allowed to be slow, never to be stuck. Give up
# only when the aggregate distance-to-target fails to shrink by more than
# RETURN_STALL_MIN_PROGRESS ticks for a full RETURN_STALL_WINDOW_S window.
RETURN_STALL_WINDOW_S = 1.5
RETURN_STALL_MIN_PROGRESS = 10  # ticks, summed over all motors
STOP_HOLD_S = 3.0  # fallback: stay frozen this long before releasing

# Working torque (RAM Torque_Limit) used while the script drives the arm —
# init, unfold, probing, and the freeze-on-stop all write this value. Runtime-
# selectable via --torque-limit (MakerMods Lab threads the robot's torque slider
# through it); main() overwrites this global before any stage runs.
TORQUE_LIMIT = DEFAULT_TORQUE_LIMIT


def _capture_rest_pose(bus: FeetechMotorsBus) -> dict[str, int]:
    """Offset-independent pose of every motor, to return to on an interrupt.

    Captured as (Present_Position + Homing_Offset) % FULL_TURN — the same
    invariant as _record_reference_position — because calibration rewrites
    Homing_Offset, so a raw Present_Position captured now would point at a
    different physical pose later. Best-effort: unreadable motors are simply
    absent from the result.
    """
    pose: dict[str, int] = {}
    for motor in bus.motors:
        try:
            pr = bus.read("Present_Position", motor, normalize=False)
            ho = bus.read("Homing_Offset", motor, normalize=False)
            pose[motor] = (pr + ho) % FULL_TURN
        except COMM_ERR:
            continue
    # Say what was captured: a motor silently absent here (comm error) is
    # never returned on a stop, which on the bench looks exactly like "the
    # starting position wasn't right".
    missing = [m for m in bus.motors if m not in pose]
    line = f"Start pose captured for: {', '.join(pose) if pose else 'NO MOTORS'}"
    if missing:
        line += f" (MISSING: {', '.join(missing)} — these will NOT be returned on a stop)"
    print(line)
    return pose


def _freeze_arm(bus: FeetechMotorsBus) -> None:
    """Halt all motion NOW: hold the current pose at normal working torque.

    Zero velocity-mode motion first (a motor interrupted mid limit-probe keeps
    pushing otherwise), then per motor: servo mode (the same mode-switch
    pattern _fold_arm and write_pos_ex_and_wait use mid-run), working
    Torque_Limit, torque cleared-and-re-enabled, goal = current position.
    ~1s for 6 motors; a per-motor comm error skips that motor rather than
    aborting the freeze.

    The disable-settle-enable torque cycle matters: the Stop usually lands
    while one motor sits stalled against a hard stop with its OVERLOAD latch
    set, and a bare Torque_Enable=1 NAKs on such a servo — leaving it limp,
    which then stalls the whole return-to-start (the arm cannot move a dead
    joint back; observed on hardware as "only ever the fallback"). Disabling
    first clears the latch, exactly like the mixin's _clear_and_enable_torque;
    _write_torque_with_recovery retries the enable with a longer settle if it
    still NAKs. The ~50ms per-motor limp window is imperceptible (the other
    five motors keep holding).
    """
    with contextlib.suppress(COMM_ERR):
        bus.sync_write("Goal_Velocity", dict.fromkeys(bus.motors, 0))
    values: dict[str, tuple[int, int, int]] = {}
    for motor in bus.motors:
        try:
            bus.write("Operating_Mode", motor, 0)
            bus.write("Torque_Limit", motor, TORQUE_LIMIT, normalize=False)
            with contextlib.suppress(COMM_ERR):
                bus.write("Torque_Enable", motor, 0)  # clear a mid-probe overload latch
            time.sleep(0.05)
            pos = bus.read("Present_Position", motor, normalize=False)
            bus._write_torque_with_recovery(motor, 1, retries=2, interval_s=0.2)
            values[motor] = (pos, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION)
        except COMM_ERR:
            print(f"  Freeze: {motor_label(motor)} could not be re-energized; it will not hold or return.")
            continue
    if values:
        bus.sync_write_pos_ex(values)


def _nearest_wrap_target(base: int, present: int, min_limit: int, max_limit: int) -> int:
    """Pick the wrap-equivalent of `base` nearest `present`, inside the limits.

    Geometry: the encoder is a FULL_TURN-tick circle, so the re-expressed rest
    target `(encoder - ho_now) % FULL_TURN` is only defined modulo FULL_TURN —
    base, base - FULL_TURN and base + FULL_TURN all name the same physical
    pose. But Goal_Position drives a straight line through register space, so
    the wrong representative can send the joint the long way around, through
    the calibration hard stops. Choose the representative nearest the joint's
    current position, then clamp into the firmware's current
    Min/Max_Position_Limit window: a clamped seam-adjacent target still lands
    a short arc from the pose instead of trekking toward the far one.
    """
    candidates = (base - FULL_TURN, base, base + FULL_TURN)
    target = min(candidates, key=lambda candidate: abs(candidate - present))
    return max(min_limit, min(max_limit, target))


def _return_to_rest_pose(bus: FeetechMotorsBus, rest_pose: dict[str, int], ceiling_s: float) -> bool:
    """Drive the arm back to the captured start pose; True once it arrives.

    Targets are re-expressed against each motor's CURRENT Homing_Offset (see
    _capture_rest_pose) using the near, in-limits wrap representative (see
    _nearest_wrap_target). Motion is simultaneous and gentle.

    Termination is progress-based, not timed: the return runs for as long as
    the aggregate distance-to-target keeps shrinking (by more than
    RETURN_STALL_MIN_PROGRESS ticks per RETURN_STALL_WINDOW_S window), and
    succeeds ONLY when every motor sits within POSITION_TOLERANCE of its
    target. All-Moving==0 is deliberately NOT success: a latched or weak motor
    can sit motionless far from target, and reporting that as "returned" is
    exactly the bench symptom of "the starting position wasn't right" — it is
    reported as a distinct "settled short" outcome instead. A healthy-but-slow
    return is never cut short; `ceiling_s` is only the absolute backstop
    (RETURN_TO_REST_BUDGET_S) that keeps the manager's SIGKILL from landing
    mid-motion.
    """
    values: dict[str, tuple[int, int, int]] = {}
    for motor, encoder in rest_pose.items():
        try:
            ho = bus.read("Homing_Offset", motor, normalize=False)
            present = bus.read("Present_Position", motor, normalize=False)
            try:
                # Mid-calibration these are (0, 4095) from stage 0 until a
                # motor's calibrated limits land — which are written right
                # after its new Homing_Offset, so they are in the same
                # offset-applied space as our re-expressed target. Clamping is
                # therefore consistent at every interrupt point.
                min_limit, max_limit = bus.read_position_limits(motor)
            except COMM_ERR:
                min_limit, max_limit = 0, FULL_TURN - 1
            base = (encoder - ho) % FULL_TURN
            target = _nearest_wrap_target(base, present, min_limit, max_limit)
            values[motor] = (target, RETURN_POS_SPEED, DEFAULT_ACCELERATION)
        except COMM_ERR:
            continue
    if not values:
        print("Fallback: could not compute any return target (comm errors on every motor).")
        return False
    targets = {motor: v[0] for motor, v in values.items()}
    bus.sync_write_pos_ex(values)
    time.sleep(RETURN_SETTLE_S)
    t0 = time.monotonic()
    best_dist: int | None = None
    last_progress_t = t0
    remaining = -1
    distances: dict[str, int] = {}
    while time.monotonic() - t0 < ceiling_s:
        remaining = 0
        distances = {}
        readable = False
        arrived = True
        for motor, target in targets.items():
            try:
                pos = bus.read("Present_Position", motor, normalize=False)
            except COMM_ERR:
                arrived = False
                continue
            readable = True
            distance = abs(pos - target)
            distances[motor] = distance
            remaining += distance
            if distance > POSITION_TOLERANCE:
                arrived = False
        now = time.monotonic()
        if readable and arrived:
            detail = ", ".join(f"{m}={d}" for m, d in distances.items())
            print(f"Return complete: max delta {max(distances.values())} ticks ({detail}).")
            return True
        if readable and (best_dist is None or remaining < best_dist - RETURN_STALL_MIN_PROGRESS):
            best_dist = remaining
            last_progress_t = now
        if now - last_progress_t > RETURN_STALL_WINDOW_S:
            # No meaningful progress for a full window. Say which motors are
            # still out — and whether they are still fighting (stalled) or
            # sitting motionless short of target (settled: latched/weak) — so
            # the next hardware log is conclusive.
            detail = ", ".join(f"{m}={d}" for m, d in distances.items() if d > POSITION_TOLERANCE)
            outcome = "stalled"
            with contextlib.suppress(COMM_ERR):
                if all(bus.read("Moving", m, normalize=False) == 0 for m in targets):
                    outcome = "settled short of target"
            print(
                f"Fallback: return {outcome} after {now - t0:.1f}s — "
                f"remaining distance {remaining} ticks ({detail or 'unreadable motors'})."
            )
            return False
        time.sleep(RETURN_POLL_S)
    # Absolute ceiling — never expected in practice (see docstring).
    print(f"Fallback: return hit the {ceiling_s:.0f}s ceiling with {remaining} ticks remaining.")
    return False


def _graceful_stop(bus: FeetechMotorsBus, rest_pose: dict[str, int]) -> None:
    """Interrupt sequence: freeze in place, then return to the start pose (or hold).

    The caller runs safe_disable_all afterwards on EVERY path — this function
    only decides where the arm is when torque drops. All six motors take part,
    gripper included: during calibration nothing is ever gripped, so there is
    nothing a moving gripper could drop.

    Strictly best-effort: any failure falls through immediately so this can
    never prevent — or delay beyond the manager's SIGTERM grace — the real
    release. A SECOND KeyboardInterrupt (another Stop / SIGTERM) anywhere in
    here means "stop NOW": skip the rest and release instantly.

    SPEED-CAP NOTE: the freeze/return here (and the fold/unfold on a NORMAL
    completion) leave a nonzero Goal_Velocity stamped in the servos' RAM. That
    register is NOT reset here on purpose: the fold path leaves speeds stamped
    by design, safe_disable_all follows on every path anyway (it only cuts
    torque, it does not touch Goal_Velocity), and — crucially — this subprocess
    can't reach into the next feature's session. The leftover cap is cleared
    where it matters, at the NEXT session's start, by
    makermodslab/motor_power.clear_goal_velocity (the primary, durable fix). Adding a
    reset here would be redundant on the stop path and would fight the fold
    path, so the MakerMods Lab side owns it.
    """
    try:
        print("\nStop requested: freezing the arm in place...")
        _freeze_arm(bus)
        returned = False
        if rest_pose:
            print("Returning the arm to its starting position...")
            returned = _return_to_rest_pose(bus, rest_pose, RETURN_TO_REST_BUDGET_S)
        else:
            print("Fallback: no start pose was captured at startup.")
        if returned:
            print("Arm returned to its starting position.")
        else:
            print(f"Holding position for {STOP_HOLD_S:.0f}s before releasing...")
            time.sleep(STOP_HOLD_S)
    except (Exception, KeyboardInterrupt):
        # Includes a broken stdout pipe from the prints: skip straight to the
        # caller's safe_disable_all rather than dying with torque on.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feetech servo auto-calibration (with unfold): full flow in one command."
    )
    parser.add_argument(
        "--port", type=str, required=True,
        help="Serial port path, e.g. COM3 or /dev/ttyUSB0",
    )
    parser.add_argument(
        "--motor", type=str, choices=MOTOR_NAMES, default=None,
        help="Test only this servo (skip unfold); if not specified, test all 6 servos in order",
    )

    cal = parser.add_argument_group("Calibration parameters")
    cal.add_argument(
        "--velocity-limit", type=int, default=DEFAULT_VELOCITY_LIMIT,
        help=f"Calibration limit-probing speed (constant-speed mode Goal_Velocity), default {DEFAULT_VELOCITY_LIMIT}",
    )
    cal.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"Calibration single-direction limit wait timeout (seconds), default {DEFAULT_TIMEOUT}",
    )
    cal.add_argument(
        "--torque-limit", type=int, default=DEFAULT_TORQUE_LIMIT,
        help=(
            "Working torque (RAM Torque_Limit, 0-1000 scale) used while driving the arm "
            f"during calibration (init, unfold, freeze-on-stop). Default {DEFAULT_TORQUE_LIMIT}"
        ),
    )
    unfold = parser.add_argument_group("Unfold parameters")
    unfold.add_argument(
        "--unfold-only", action="store_true",
        help="Run only the arm unfold (Stage 0 init + Stage 2 unfold), no calibration; for debugging unfold logic",
    )
    unfold.add_argument(
        "--unfold-angle", type=float, default=DEFAULT_UNFOLD_ANGLE,
        help=f"Unfold angle (degrees); set to 0 to skip unfold. Default {DEFAULT_UNFOLD_ANGLE}",
    )
    unfold.add_argument(
        "--unfold-timeout", type=float, default=DEFAULT_UNFOLD_TIMEOUT,
        help=f"Per-motion timeout for unfold (seconds), default {DEFAULT_UNFOLD_TIMEOUT}",
    )
    out = parser.add_argument_group("Output (same path and format as manual calibration)")
    out.add_argument(
        "--save", action="store_true",
        help="Write calibration data to servo EEPROM and save to the same local path as manual calibration (draccus format)",
    )
    out.add_argument(
        "--robot-id", type=str, default="default",
        help="Robot id used in the saved filename; path is .../calibration/robots/<robot_type>/<robot_id>.json. Must match config.id used when starting the arm.",
    )
    out.add_argument(
        "--robot-type", type=str, default="so_follower",
        choices=["so_follower", "so_leader"],
        help="Robot type for calibration file path: 'so_follower' (default) or 'so_leader'",
    )

    return parser.parse_args()


# ====================== Unfold-related ======================

def _unfold_joints(
    bus: FeetechMotorsBus,
    unfold_angle: float,
    unfold_timeout: float,
    unfold_directions: dict[str, str | None] | None = None,
) -> None:
    """Unfold joints 2-4 to avoid mechanical interference during calibration.
    If unfold_directions is provided, record each joint's unfold direction."""
    print(f"\n{'='*20} Stage 2: Unfold joints 2-4 ({unfold_angle}°) {'='*20}")
    for motor in UNFOLD_ORDER:
        direction, _ = bus.unfold_single_joint(motor, unfold_angle, unfold_timeout)
        if unfold_directions is not None and direction is not None:
            unfold_directions[motor] = direction
    print("\n  Unfold complete; joints 2-4 are lifted. Per-joint unfold directions:")
    if unfold_directions is not None:
        for motor in UNFOLD_ORDER:
            direction = unfold_directions.get(motor, "unknown")
            print(f"    {motor_label(motor)}: unfold direction = {direction}")


def _fold_arm(
    bus: FeetechMotorsBus,
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_unfold_directions: dict[str, str | None],
    *,
    motors: list[str] | None = None,
    unfold: bool = False,
    unfold_per_motor: dict[str, bool] | None = None,
) -> None:
    """Fold the specified joints, or fully unfold them. Multiple servos move simultaneously.

    Fold (unfold=False): forward-unfold -> fold target = range_max; reverse -> range_min;
    gripper fixed at range_min.
    Unfold (unfold=True): targets are reversed; forward -> range_min, reverse -> range_max;
    gripper fixed at range_max.
    motors: list of servos to move; if None or empty, use the default order
    (shoulder_lift -> elbow_flex -> wrist_flex -> gripper).
    unfold_per_motor: optional, per-joint fold(False)/unfold(True); joints not listed use 'unfold'.
    If None, all use 'unfold'.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
    fold_order = default_order if not motors else motors
    title = "Fold/Unfold arm" if unfold_per_motor else (("Unfold" if unfold else "Fold") + " arm")
    print(f"\n{'='*20} {title} (simultaneous) {'='*20}")
    values: dict[str, tuple[int, int, int]] = {}

    for motor in fold_order:
        if motor not in all_mins or motor not in all_maxes:
            continue
        per_unfold = unfold_per_motor.get(motor, unfold) if unfold_per_motor is not None else unfold
        direction = all_unfold_directions.get(motor)
        # First compute fold-end and unfold-end, then pick one based on per_unfold
        if motor == "gripper":
            fold_end = all_mins[motor]
            unfold_end = all_maxes[motor]
        else:
            fold_end = all_maxes[motor] if direction == "reverse" else all_mins[motor]
            unfold_end = all_mins[motor] if direction == "reverse" else all_maxes[motor]
        target = unfold_end if per_unfold else fold_end
        label = "range_max" if target == all_maxes[motor] else "range_min"
        if motor == "gripper":
            label += "(gripper)" if per_unfold else "(gripper forward)"
        action_m = "Unfold" if per_unfold else "Fold"
        values[motor] = (target, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION)
        bus.write("Operating_Mode", motor, 0)  # servo mode
        try:
            pos = bus.read("Present_Position", motor, normalize=False)
            print(f"  {motor_label(motor)} current pos={pos}, {action_m} to {label}={target}.")
        except COMM_ERR:
            print(f"  {motor_label(motor)} failed to read current pos, {action_m} to {label}={target}.")
    if not values:
        action = "Unfold" if unfold else "Fold"
        print(f"  No valid servos, skipping {action}.\n")
        return
    bus.sync_write_pos_ex(values)
    time.sleep(0.3)
    # Poll until all stopped
    timeout_s = 10.0
    poll_s = 0.05
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            if all(bus.read("Moving", m, normalize=False) == 0 for m in values):
                break
        except COMM_ERR:
            pass
        time.sleep(poll_s)
    done_label = "Fold/Unfold" if unfold_per_motor else ("Unfold" if unfold else "Fold")
    for m in values:
        try:
            pos = bus.read("Present_Position", m, normalize=False)
            print(f"  {motor_label(m)} after {done_label}: end pos={pos}, arrived")
        except COMM_ERR:
            print(f"  {motor_label(m)} after {done_label}: failed to read pos, arrived")
    print(f"  {done_label} complete.\n")


def _move_arm_by_angle(
    bus: FeetechMotorsBus,
    all_unfold_directions: dict[str, str | None],
    angle_deg: float,
    *,
    fold: bool = False,
    motors: list[str] | None = None,
    all_mins: dict[str, int] | None = None,
    all_maxes: dict[str, int] | None = None,
) -> None:
    """Move the given joints by the specified degrees relative to current position, in either the
    unfold or the fold direction. Does not probe direction; relies on all_unfold_directions.

    Direction matches _fold_arm: forward-unfold -> position increases for unfold, decreases for fold;
    reverse-unfold -> the opposite.
    fold: False = unfold direction, True = fold direction.
    motors: list of servos to move; if None or empty, uses the default order
    (shoulder_lift -> elbow_flex -> wrist_flex).
    all_mins / all_maxes: optional; if provided, the target position is clamped to them.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    move_order = default_order if not motors else motors
    angle_steps = int(angle_deg / 360.0 * FULL_TURN)
    direction_label = "fold" if fold else "unfold"
    print(f"\n{'='*20} Relative {direction_label} {angle_deg:.1f}° from current pos {'='*20}")
    for motor in move_order:
        if all_mins is not None and all_maxes is not None and (motor not in all_mins or motor not in all_maxes):
            continue
        try:
            present = bus.read("Present_Position", motor, normalize=False)
        except COMM_ERR:
            print(f"  Warning: {motor_label(motor)} failed to read current pos, skipping")
            continue
        direction = all_unfold_directions.get(motor)
        # Matches _fold_arm: forward -> unfold increases pos, fold decreases; reverse -> opposite
        if fold:
            target = present - angle_steps if direction == "forward" else present + angle_steps
        else:
            target = present + angle_steps if direction == "forward" else present - angle_steps
        if all_mins is not None and all_maxes is not None and motor in all_mins and motor in all_maxes:
            target = max(all_mins[motor], min(all_maxes[motor], target))
        print(f"  {motor_label(motor)} {direction_label} {angle_deg:.1f}°: pos {present} -> {target}")
        ok = bus.write_pos_ex_and_wait(
            motor, target, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION,
            timeout_s=DEFAULT_UNFOLD_TIMEOUT, poll_interval_s=0.05,
        )
        if not ok:
            print(f"  Warning: {motor_label(motor)} motion timeout, holding current position")
        else:
            print(f"  {motor_label(motor)} arrived")
    print(f"  {direction_label} complete.\n")


# ====================== Calibration-related ======================


def _record_reference_position(
    bus: FeetechMotorsBus,
    motor_name: str,
    out: dict[str, int],
) -> None:
    """Read the servo's current reference position (Present_Position + Homing_Offset) % FULL_TURN
    and write it to out[motor_name]; on read failure, leave out unchanged."""
    try:
        pr = bus.read("Present_Position", motor_name, normalize=False)
        ho = bus.read("Homing_Offset", motor_name, normalize=False)
        out[motor_name] = (pr + ho) % FULL_TURN
    except COMM_ERR:
        pass


def _calibrate_motors(
    bus: FeetechMotorsBus,
    motor_names: list[str],
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    ccw_first: bool = False,
    unfold_directions: dict[str, str | None] | None = None,
    reference_positions: dict[str, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Calibrate a group of servos (uniformly via measure_ranges_of_motion_multi, then write back
    and return to mid). Returns {motor_name: (range_min, range_max, mid_raw)}.
    If unfold_directions is provided and both servos 2 and 3 are calibrated together: both move
    together; servo 2 fully folds, servo 3 fully unfolds along its unfold direction. Otherwise
    follow the original return-to-mid logic.
    If reference_positions is provided: for those servos, use the reference position to pick the
    arc during calibration, skipping limit back-off and re-read."""
    if not motor_names:
        return {}
    raw_results = bus.measure_ranges_of_motion_multi(
        motor_names,
        velocity_limit=velocity_limit,
        timeout_s=timeout_s,
        ccw_first=ccw_first,
        reference_positions=reference_positions,
    )
    print("Preparing to write registers")
    result: dict[str, tuple[int, int, int]] = {}
    for m in motor_names:
        rmin, rmax, mid_raw, _raw_min_meas, _raw_max_meas, homing_offset = raw_results[m]
        print(f"  {motor_label(m)}: post-offset range_min={rmin}, range_max={rmax}, mid={mid_raw}, Homing_Offset register={homing_offset}")
        time.sleep(0.05)
        try:
            ho_before = bus.read("Homing_Offset", m, normalize=False)
            min_before = bus.read("Min_Position_Limit", m, normalize=False)
            max_before = bus.read("Max_Position_Limit", m, normalize=False)
            print(f"  {motor_label(m)} pre-write: Min_Position_Limit={min_before}, Max_Position_Limit={max_before}, Homing_Offset={ho_before}")
        except COMM_ERR:
            print(f"  {motor_label(m)} pre-write: register read failed")
        bus.safe_write("Homing_Offset", m, homing_offset, normalize=False)
        bus.safe_write_position_limits(m, rmin, rmax)
        time.sleep(0.1)
        try:
            ho_after = bus.read("Homing_Offset", m, normalize=False)
            min_after = bus.read("Min_Position_Limit", m, normalize=False)
            max_after = bus.read("Max_Position_Limit", m, normalize=False)
            print(f"  {motor_label(m)} post-write: Min_Position_Limit={min_after}, Max_Position_Limit={max_after}, Homing_Offset={ho_after}")
        except COMM_ERR:
            print(f"  {motor_label(m)} post-write: register read failed")
        time.sleep(0.1)
        do_2_3_together = (
            unfold_directions is not None
            and "shoulder_lift" in motor_names
            and "elbow_flex" in motor_names
        )
        if m == "wrist_roll":
            pass
        elif do_2_3_together and m in ("shoulder_lift", "elbow_flex"):
            # Servos 2 and 3 lock torque
            pass
        else:
            bus.go_to_mid(m)
        result[m] = (rmin, rmax, mid_raw)

    return result


# ====================== Connect and init (shared) ======================

def _connect_and_clear(port: str) -> FeetechMotorsBus:
    """Create the bus, clear residual Overload, then connect for real. Raises on failure."""
    bus = FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())
    bus.connect(handshake=False)
    print("Clearing residual servo state...")
    all_zero = {m: 0 for m in MOTOR_NAMES}
    for _ in range(3):
        try:
            bus.sync_write("Goal_Velocity", all_zero)
        except COMM_ERR:
            pass
        try:
            bus.sync_write("Torque_Enable", all_zero)
        except COMM_ERR:
            pass
        time.sleep(0.2)
    bus.disconnect(disable_torque=False)
    time.sleep(0.2)
    bus.connect()
    print("All servos ready.")
    return bus


def _run_with_bus(
    port: str,
    interactive: bool,
    body: Callable[[FeetechMotorsBus], None],
) -> int:
    """After connecting the bus, run body(bus); uniformly handle connect failures, KeyboardInterrupt,
    Exception, and disconnect. Returns 0 on success, 1 on error, 130 on user interrupt."""
    try:
        bus = _connect_and_clear(port)
    except Exception as e:
        print(f"Connect failed: {e}", file=sys.stderr)
        return 1
    rest_pose: dict[str, int] = {}
    try:
        # Captured before any motion, so a Stop mid-run can drive the arm back
        # to exactly where the user left it (inside the try: a SIGTERM during
        # the capture itself must still release torque below).
        rest_pose = _capture_rest_pose(bus)
        body(bus)
    except KeyboardInterrupt:
        # Freeze/return first (best-effort, see _graceful_stop), then release
        # torque BEFORE printing: if stdout's pipe is gone the print raises
        # and would otherwise skip the release.
        _graceful_stop(bus, rest_pose)
        bus.safe_disable_all()
        # MakerMods Lab patch (P0-1): this is what a Stop takes. Put the servos'
        # own calibration back now that they're limp and the arm has finished
        # returning to rest — the return runs in the Stage-0 frame that
        # _graceful_stop already accounts for, so restoring after it (rather
        # than before) keeps that logic untouched.
        _restore_calibration(bus)
        print("\nUser interrupt; releasing all servos...")
        return 130
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        bus.safe_disable_all()
        # MakerMods Lab patch (P0-1): a failed run destroyed the calibration just
        # as thoroughly as a stopped one.
        _restore_calibration(bus)
        if interactive:
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
        return 1
    finally:
        # Don't let a failed disconnect (e.g. a motor still in overload NAKs
        # the disable-torque write it performs) replace the return code with
        # a traceback — torque was already released on every path above.
        try:
            bus.disconnect()
        except Exception as e:
            print(f"Disconnect failed: {e}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# MakerMods Lab patch (design-debt P0-1): snapshot/restore of the servos' own
# calibration registers.
#
# Stage 0 zeroes Homing_Offset and opens the travel limits to (0, 4095) on every
# motor BEFORE a single measurement is taken, and does it behind Lock=1 — i.e.
# into EEPROM, so it survives a power cycle. Upstream writes real calibration
# back only at the natural end of a fully successful run; both terminal failure
# paths (KeyboardInterrupt, which is what a Stop sends, and a generic exception)
# released torque and returned without restoring anything. A single Stop
# therefore left the arm with no calibration and no firmware end-stop clamp, and
# nothing in MakerMods Lab rewrites calibration on a later session (it connects
# with calibrate=False), so every reading stayed silently shifted.
#
# Kept deliberately small: one snapshot taken where the destruction happens, one
# restore on the paths that aren't a successful run, and no change to the success
# path (which legitimately writes fresh calibration).
_PRE_RUN_CALIBRATION: dict[str, dict[str, object]] = {}


def _snapshot_calibration(bus: FeetechMotorsBus) -> None:
    """Record each motor's pre-run Homing_Offset and travel limits.

    Best-effort and never raises: a motor that won't answer is simply absent from
    the snapshot and is left alone on restore. Reading is free of side effects,
    so doing this even on runs that go on to succeed costs only ~12 register
    reads."""
    _PRE_RUN_CALIBRATION.clear()
    for m in MOTOR_NAMES:
        entry: dict[str, object] = {}
        try:
            entry["Homing_Offset"] = bus.read("Homing_Offset", m, normalize=False)
        except Exception as e:  # noqa: BLE001 - a motor that won't answer just isn't restorable
            print(f"  [Warning] Could not snapshot Homing_Offset on {m}: {e}")
        try:
            entry["limits"] = bus.read_position_limits(m)
        except Exception as e:  # noqa: BLE001
            print(f"  [Warning] Could not snapshot position limits on {m}: {e}")
        if entry:
            _PRE_RUN_CALIBRATION[m] = entry


def _restore_calibration(bus: FeetechMotorsBus) -> None:
    """Put the pre-run calibration back after a stopped or failed run.

    Call AFTER torque is released. Homing_Offset re-frames what the servo
    considers its current position, so writing it while torque is still enabled
    invites a jump as the motor chases its old goal in the new frame — the
    restore is not urgent, and doing it limp is the safe order.

    Best-effort throughout: this runs on a path that is already failing, and it
    must never replace the original error (or block the disconnect) with one of
    its own."""
    if not _PRE_RUN_CALIBRATION:
        return
    print("Restoring the pre-run servo calibration...")
    for m, entry in _PRE_RUN_CALIBRATION.items():
        homing = entry.get("Homing_Offset")
        if homing is not None:
            try:
                bus.write("Homing_Offset", m, homing, normalize=False)
            except Exception as e:  # noqa: BLE001
                print(f"  [Warning] Could not restore Homing_Offset on {m}: {e}", file=sys.stderr)
        limits = entry.get("limits")
        if isinstance(limits, tuple) and len(limits) == 2:
            try:
                bus.write_position_limits(m, limits[0], limits[1])
            except Exception as e:  # noqa: BLE001
                print(f"  [Warning] Could not restore position limits on {m}: {e}", file=sys.stderr)
    _PRE_RUN_CALIBRATION.clear()


# Stage 0 init: per-register write -> read -> compare, table-driven; special items
# (limits, Torque_Enable) handled separately. A function, not a constant, so the
# Torque_Limit entry picks up the runtime-selected TORQUE_LIMIT.
def _init_checks() -> list[tuple[str, int]]:
    return [
        ("Lock", 1),
        ("Return_Delay_Time", 0),
        ("Operating_Mode", 0),
        ("Max_Torque_Limit", DEFAULT_MAX_TORQUE),
        ("Torque_Limit", TORQUE_LIMIT),
        ("Acceleration", DEFAULT_ACCELERATION),
        ("P_Coefficient", DEFAULT_P_COEFFICIENT),
        ("I_Coefficient", DEFAULT_I_COEFFICIENT),
        ("D_Coefficient", DEFAULT_D_COEFFICIENT),
        ("Homing_Offset", 0),
    ]


def _run_init(bus: FeetechMotorsBus, *, interactive: bool = True) -> None:
    """Stage 0: Lock=1, PID, limits, Homing_Offset, enable torque. On parameter anomaly,
    if interactive, wait for Enter."""
    print(f"\n{'='*20} Stage 0: Initialization {'='*20}")
    # MakerMods Lab patch (P0-1): capture the calibration this stage is about to
    # destroy, BEFORE the first motor is touched. Without this there is nothing
    # for the failure paths in _run_with_bus to restore from.
    _snapshot_calibration(bus)
    for m in MOTOR_NAMES:
        print(f"Configuring servo: {motor_label(m)}")
        try:
            bus.write("Torque_Enable", m, 0)
            time.sleep(0.05)
        except COMM_ERR:
            pass
        param_set_ok = True
        try:
            for reg, expected in _init_checks():
                bus.write(reg, m, expected, normalize=(reg != "Homing_Offset"))
                time.sleep(0.01)
                got = bus.read(reg, m, normalize=False)
                if got != expected:
                    print(f"  [Warning] {reg} setting failed on {m}: set value={expected}, read value={got}")
                    param_set_ok = False
            # Limits: separate write/read/compare
            bus.write_position_limits(m, 0, 4095)
            time.sleep(0.05)
            limits = bus.read_position_limits(m)
            if limits != (0, 4095):
                print(f"  [Warning] Position_Limits setting failed on {m}: set value=(0, 4095), read value={limits}")
                param_set_ok = False
            time.sleep(0.2)
            # Finally enable torque
            bus.write("Torque_Enable", m, 1)
            time.sleep(0.05)
            te_read = bus.read("Torque_Enable", m, normalize=False)
            if te_read != 1:
                print(f"  [Warning] Torque_Enable enable failed on {m}: set value=1, read value={te_read}")
                param_set_ok = False
            time.sleep(0.1)
        except Exception as e:
            print(f"  [Exception] Error setting parameters on {m}: {e}")
            param_set_ok = False
        if not param_set_ok and interactive:
            try:
                input("  [Warning] Parameter setting/verification has anomalies; check wiring and power, press Enter to force-continue...")
            except Exception:
                pass
    print(
        f"Initialized and torque enabled (P={DEFAULT_P_COEFFICIENT}, "
        f"Acc={DEFAULT_ACCELERATION}, Torque={TORQUE_LIMIT})."
    )


# ====================== Public entry points (full calibration / unfold-only / single servo) ======================


def _apply_calibration_results(
    results: dict[str, tuple[int, int, int]],
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_mids: dict[str, int],
    motor_list: list[str],
) -> None:
    """Apply _calibrate_motors return values to all_mins / all_maxes / all_mids."""
    for m in motor_list:
        all_mins[m], all_maxes[m], all_mids[m] = results[m]


def run_full_calibration(
    port: str,
    *,
    save: bool = False,
    robot_id: str = "default",
    robot_type: str = "so_follower",
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    unfold_timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Full calibration flow: init -> servos 2-6 (with arm-lift to avoid obstruction) ->
    servo 1 shoulder_pan calibrated last -> fold.
    If save is True: write to servo EEPROM and save to the same path/format as manual calibration
    (draccus, loaded by the arm at startup).

    For CLI or teleop programs to call. Returns 0 on success, 1 on error, 130 on user interrupt.
    """

    def body(bus: FeetechMotorsBus) -> None:
        all_mins: dict[str, int] = {}
        all_maxes: dict[str, int] = {}
        all_mids: dict[str, int] = {}
        all_unfold_directions: dict[str, str | None] = {}
        all_reference_positions: dict[str, int] = {}
        _run_init(bus, interactive=interactive)
        # Lift servo 4 by 80 degrees
        direction, _ = bus.unfold_single_joint("wrist_flex", 80, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["wrist_flex"] = direction
        time.sleep(0.1)
        # Lift servos 2 and 3 and record reference positions
        # (Present_Position + Homing_Offset; used to pick the arc during calibration)
        direction, _ = bus.unfold_single_joint("shoulder_lift", 15, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["shoulder_lift"] = direction
        _record_reference_position(bus, "shoulder_lift", all_reference_positions)
        direction, _ = bus.unfold_single_joint("elbow_flex", 30, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["elbow_flex"] = direction
        _record_reference_position(bus, "elbow_flex", all_reference_positions)
        time.sleep(0.1)
        # Fold: retract shoulder_lift and elbow_flex
        for m in ["shoulder_lift", "elbow_flex"]:
            bus.go_to_mid(m)
            time.sleep(0.1)
        # Use multi-servo calibration for servos 2 and 3; the first rotation direction is the
        # opposite of each servo's lift direction.
        # forward-lifted -> CCW first; reverse-lifted -> CW first; default is CCW first when not recorded.
        ccw_first_2_3 = {
            "shoulder_lift": all_unfold_directions.get("shoulder_lift") != "reverse",
            "elbow_flex": all_unfold_directions.get("elbow_flex") != "reverse",
        }
        print(f"\n{'='*20} Calibrating servos 2 and 3 (multi-servo, opposite of lift direction) {'='*20}")
        results_2_3 = _calibrate_motors(
            bus, ["shoulder_lift", "elbow_flex"],
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            ccw_first=ccw_first_2_3,
            unfold_directions=all_unfold_directions,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(results_2_3, all_mins, all_maxes, all_mids, ["shoulder_lift", "elbow_flex"])
        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions, motors=["shoulder_lift", "elbow_flex"])

        time.sleep(0.1)
        # Stage 3: Calibrate the remaining servos 4, 5, 6 (multi-servo simultaneous, with arm lift to avoid obstruction)
        print(f"\n{'='*20} Stage 3: Calibrate servos 4-6 (multi-servo simultaneous) {'='*20}")
        _move_arm_by_angle(bus, all_unfold_directions, 80, fold=False, motors=["elbow_flex"], all_mins=all_mins, all_maxes=all_maxes)
        CALIBRATE_REST_REMAINING = ["wrist_roll", "gripper", "wrist_flex"]
        results_rest = _calibrate_motors(
            bus, CALIBRATE_REST_REMAINING,
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(results_rest, all_mins, all_maxes, all_mids, CALIBRATE_REST_REMAINING)
        time.sleep(0.1)
        # Fold servo 3, fully unfold servo 4 (executed together in one call)
        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions,
            motors=["elbow_flex", "wrist_flex","gripper"],
            unfold_per_motor={"elbow_flex": False, "wrist_flex": True, "gripper": False})
        # Stage 4: Calibrate servo 1 shoulder_pan last
        print(f"\n{'='*20} Stage 4: Calibrate {motor_label('shoulder_pan')} (servo 1) and return to mid {'='*20}")
        results_pan = _calibrate_motors(
            bus, ["shoulder_pan"], velocity_limit=velocity_limit, timeout_s=timeout_s
        )
        _apply_calibration_results(results_pan, all_mins, all_maxes, all_mids, ["shoulder_pan"])
        time.sleep(0.1)
        motors_calibrated = CALIBRATE_REST + CALIBRATE_FIRST
        print(f"\n{'='*20} Calibration results {'='*20}")
        for name in motors_calibrated:
            offset = all_mids[name] - STS_HALF_TURN_RAW
            print(
                f"  {motor_label(name)}: min={all_mins[name]}, max={all_maxes[name]}, "
                f"mid={all_mids[name]}, offset={offset}"
            )


        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions)
           # Before persistence: unlock EEPROM (Lock=0) and restore all servos to servo mode (Operating_Mode=0)
        for name in bus.motors:
            bus.write("Lock", name, 0)
            time.sleep(0.01)
            bus.write("Operating_Mode", name, 0)
            time.sleep(0.01)
        time.sleep(1)
        if interactive:
            bus.safe_disable_all()
            print("\nCalibration complete.")

        if save:
            print(f"\n{'='*20} Persistence (same scheme as manual calibration) {'='*20}")
            bus.safe_disable_all()
            cal = {}
            for name in motors_calibrated:
                m = SO_FOLLOWER_MOTORS[name]
                offset = all_mids[name] - STS_HALF_TURN_RAW
                offset = max(-HOMING_OFFSET_MAX_MAG, min(HOMING_OFFSET_MAX_MAG, offset))
                cal[name] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=offset,
                    range_min=all_mins[name],
                    range_max=all_maxes[name],
                )
            bus.write_calibration(cal, cache=True)
            print("Wrote calibration to servo EEPROM.")
            # Same path and format as manual calibration; loaded by the arm at startup.
            calibration_fpath = HF_LEROBOT_CALIBRATION / "robots" / robot_type / f"{robot_id}.json"
            calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(calibration_fpath, "w") as f, draccus.config_type("json"):
                draccus.dump(cal, f, indent=4)
            print(f"Wrote calibration to: {calibration_fpath}")
        # No _soft_release needed here (unlike the interrupt path): _fold_arm
        # above already parked the arm in its folded rest pose, so cutting
        # torque from here is a no-drop release.
        print("Releasing all servos...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def unfold_joints(
    port: str,
    angle_deg: float,
    *,
    timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Run only Stage 0 init + unfold joints 2-4 to the given angle. For debugging the unfold.
    Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        all_unfold_directions: dict[str, str | None] = {}
        if angle_deg > 0:
            _unfold_joints(bus, angle_deg, timeout_s, all_unfold_directions)
            print("  Unfold complete; unfold directions:")
            for motor, direction in all_unfold_directions.items():
                print(f"    {motor_label(motor)}: {direction}")
            if interactive:
                input("  Press Enter to release torque and exit...")
        else:
            print("  Unfold angle is 0, skipping unfold.")
            if interactive:
                input("  Press Enter to release torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def calibrate_single_motor(
    port: str,
    motor_name: str,
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Run only Stage 0 + calibrate the given servo, no fold, no save. For testing.
    Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        print(f"\n{'='*20} Calibrating {motor_label(motor_name)} {'='*20}")
        _calibrate_motors(bus, [motor_name], velocity_limit=velocity_limit, timeout_s=timeout_s)
        time.sleep(0.1)
        if interactive:
            input("  Calibration complete. Press Enter to release torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


# ====================== CLI entry ======================

def _handle_sigterm(signum, frame) -> None:
    """MakerMods Lab's Stop button terminates this subprocess with SIGTERM; raise
    KeyboardInterrupt so _run_with_bus releases torque (safe_disable_all)
    instead of dying with the arm still energized."""
    raise KeyboardInterrupt


def main() -> int:
    """CLI: based on arguments, invoke full calibration, unfold-only, or single-servo calibration."""
    global TORQUE_LIMIT
    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = parse_args()
    # Clamp to a drivable window: below ~100 the arm can't move its own weight,
    # above DEFAULT_MAX_TORQUE the register is out of range.
    TORQUE_LIMIT = max(100, min(DEFAULT_MAX_TORQUE, args.torque_limit))
    if TORQUE_LIMIT != DEFAULT_TORQUE_LIMIT:
        print(f"Working torque: Torque_Limit={TORQUE_LIMIT} (default {DEFAULT_TORQUE_LIMIT})")
    # MakerMods Lab runs this as a subprocess (no TTY). Without a TTY the "press Enter"
    # prompts have no one to answer them, so run non-interactively — every
    # input() is guarded by `interactive` and torque still releases afterwards.
    interactive = sys.stdin.isatty()
    print(f"Serial port: {args.port}")
    if getattr(args, "unfold_only", False):
        print("Arm unfold only (--unfold-only): Stage 0 init + Stage 2 unfold, no calibration")
        print(f"Unfold angle: {args.unfold_angle}°")
        return unfold_joints(
            args.port,
            args.unfold_angle,
            timeout_s=args.unfold_timeout,
            interactive=interactive,
        )
    if args.motor is not None:
        print(f"Single-servo mode: {args.motor}")
        return calibrate_single_motor(
            args.port,
            args.motor,
            velocity_limit=args.velocity_limit,
            timeout_s=args.timeout,
            interactive=interactive,
        )
    print(f"Full calibration: {CALIBRATE_FIRST + CALIBRATE_REST}")
    return run_full_calibration(
        args.port,
        save=args.save,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        velocity_limit=args.velocity_limit,
        timeout_s=args.timeout,
        unfold_timeout_s=args.unfold_timeout,
        interactive=interactive,
    )


if __name__ == "__main__":
    sys.exit(main())
