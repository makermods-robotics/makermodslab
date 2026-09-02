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
"""Gentle return-to-pose for a Maker arm — the CAN counterpart to rest_pose.py.

``rest_pose.py`` eases an SO-101 home by writing a profile velocity
(``Goal_Velocity``) and a target (``Goal_Position``) into Feetech RAM and then
polling until the servos arrive. A RobStride joint has neither register: it is
commanded in MIT position mode, one setpoint per control step, and it goes
wherever you point it as fast as its gains allow. Pointing a synced arm straight
at a far-away pose makes it SNAP there.

So the motion has to be shaped here, in software, by interpolating the setpoint:
walk the target from where the arm is to where it should be at a bounded angular
rate, one step per tick. This is the same shape lerobot's own
``RolloutStrategy._return_to_initial_position`` uses for the inference teardown
(linear interpolation over ``send_action``), which is why inference already
lands a Maker arm correctly and teleop/record/replay needed this module.

Why an arm must be returned at all, rather than just dropped:

A Maker follower has no brakes. Cutting torque anywhere except near its resting
pose lets the whole arm fall under gravity — onto the bench, onto whatever it
was holding, or through its own cable loom. Torque-off is the vendor's *safe
state* only once the arm is somewhere it can safely rest. Getting it there is
this module's job; ``torque.release_maker_torque`` then does the release.

Arrival is judged by CONVERGENCE, not by tolerance alone. A joint held in MIT
position control settles with a standing error proportional to its load —
measured 3-5 deg on ``wrist_flex`` across runs on a real arm — so waiting for
the error to reach zero waits forever and would burn the whole ceiling on every
healthy stop. Once the worst joint stops improving, being within
``MAKER_RETURN_SETTLE_DEG`` counts as arrived.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Angular rate cap for the return, in degrees per second per joint. 30 deg/s is
# the vendor's own gentle rate: MakerFollowerConfig.startup_sync_speed_deg caps
# teleop's initial alignment at 1 deg per control step, and the control loop
# runs at 30 Hz. Reusing it means a stop moves the arm no faster than a teleop
# session's first seconds already do.
MAKER_RETURN_SPEED_DEG_S = 30.0

# Setpoint rate for the interpolation. Matches the recording/teleop loop rate,
# so each step is one ordinary control tick.
MAKER_RETURN_FPS = 30.0

# Absolute ceiling on the whole return, mirroring rest_pose.RETURN_CEILING_S.
# A stop must never hang: past this the caller releases torque regardless.
MAKER_RETURN_CEILING_S = 10.0

# A joint this close to target has arrived outright.
MAKER_RETURN_TOLERANCE_DEG = 2.0

# ...and this close counts as arrived once the arm has stopped improving. Wider
# than the tolerance by about the standing error observed on the most heavily
# loaded joint, with margin (measured 3-5 deg on wrist_flex).
MAKER_RETURN_SETTLE_DEG = 6.0

# Convergence detection, applied after the interpolation ramp has finished:
# less than this much improvement in the worst joint, for this many consecutive
# polls, means the arm has gone as far as its gains will take it.
MAKER_RETURN_STALL_PROGRESS_DEG = 0.25
MAKER_RETURN_STALL_POLLS = 15

# Share of the ceiling the interpolation ramp may use, leaving the rest for the
# settle check. See return_maker_to_pose.
_RAMP_CEILING_FRACTION = 0.6


def maker_follower_arms(robot) -> list[tuple[object, str]]:
    """The individually drivable follower arm(s) of a Maker robot, with labels.

    A bimanual Maker follower wraps two ``MakerFollower`` sub-arms, each with
    its own CAN bus and its own UNPREFIXED action keys. Returning them through
    the bimanual wrapper would need the ``left_``/``right_`` prefixes; driving
    the sub-arms directly avoids that entirely and lets the two return
    concurrently on their separate buses.
    """
    left = getattr(robot, "left_arm", None)
    right = getattr(robot, "right_arm", None)
    if left is not None and right is not None:
        return [(left, "left follower arm"), (right, "right follower arm")]
    return [(robot, "follower arm")]


def capture_maker_pose(device, include_gripper: bool = False) -> dict[str, float]:
    """Current joint angles in degrees, keyed by BARE motor name.

    ``get_observation()`` keys them "<motor>.pos"; the bare name is what
    ``return_maker_to_pose`` compares and re-suffixes, matching how
    ``send_action`` strips the suffix at the robot->bus boundary.

    The gripper is excluded by default, for the same reason the SO-101 rest
    pose excludes it: at stop time it may be holding something, and driving it
    back to its (likely open) starting width would drop that object mid-return.
    """
    try:
        observation = device.get_observation()
    except Exception as e:
        logger.warning(f"Could not capture the Maker arm's pose: {e}")
        return {}
    pose: dict[str, float] = {}
    for key, value in observation.items():
        if not key.endswith(".pos") or not isinstance(value, (int, float)):
            continue
        motor = key[: -len(".pos")]
        if motor == "gripper" and not include_gripper:
            continue
        pose[motor] = float(value)
    return pose


def maker_targets_from_action(
    robot, action: dict, include_gripper: bool = False
) -> list[tuple[object, dict[str, float]]]:
    """Split a ROBOT-level action dict into per-sub-arm target poses.

    ``return_maker_to_pose`` drives the individually drivable arms of
    ``maker_follower_arms`` and speaks their BARE motor names; a bimanual
    robot's action dict is keyed ``left_<motor>.pos`` / ``right_<motor>.pos``
    (see ``BiMakerFollower.send_action``, which splits on exactly those
    prefixes). This converts one into the other, so a target computed from the
    leader through the ordinary processors can be handed to the rate-bounded
    return unchanged.

    The result pairs with ``maker_follower_arms`` device for device. Poses may
    be empty (an action carrying no ``.pos`` key for that side); callers drop
    those rather than asking for a return with nothing to return to.

    The gripper is excluded by default, matching ``capture_maker_pose``: a
    rate-bounded walk of the jaws would squeeze or drop whatever they hold,
    and the gripper is the one joint a snap does not endanger.
    """
    left = getattr(robot, "left_arm", None)
    right = getattr(robot, "right_arm", None)
    bimanual = left is not None and right is not None
    sides = [(left, "left_"), (right, "right_")] if bimanual else [(robot, "")]

    targets: list[tuple[object, dict[str, float]]] = []
    for device, prefix in sides:
        pose: dict[str, float] = {}
        for key, value in action.items():
            if prefix and not key.startswith(prefix):
                continue
            if not key.endswith(".pos") or not isinstance(value, (int, float)):
                continue
            motor = key[len(prefix) : -len(".pos")]
            if motor == "gripper" and not include_gripper:
                continue
            pose[motor] = float(value)
        targets.append((device, pose))
    return targets


def _read_pose(device) -> dict[str, float]:
    """Bare-name degrees, including the gripper — used to measure progress."""
    return capture_maker_pose(device, include_gripper=True)


def return_maker_to_pose(
    device,
    pose: dict[str, float],
    abort_event: threading.Event | None = None,
    label: str = "follower arm",
    speed_deg_s: float = MAKER_RETURN_SPEED_DEG_S,
    ceiling_s: float = MAKER_RETURN_CEILING_S,
    target_label: str = "its start pose",
) -> tuple[bool, str]:
    """Walk a Maker arm back to ``pose`` at a bounded rate, then confirm it landed.

    ``pose`` is bare-name degrees, as ``capture_maker_pose`` returns. Joints
    absent from it are left alone — that is how the gripper keeps its grip
    through a stop.

    ``target_label`` only names the destination in the log lines. It defaults to
    the teardown wording because that is this function's original caller; the
    mid-session re-alignment (``record._realign_follower_to_leader``) walks the
    arm to the LEADER's current pose, which is not a start pose at all.

    Never raises: this runs on teardown paths where the caller's next move is
    to release torque regardless, and an exception here would skip that. A
    failure is reported as ``(False, reason)`` and logged.

    ``abort_event`` cuts the return short (a second stop press). The arm is left
    where it got to, which is still nearer the rest pose than where it started.
    """
    if not pose:
        return False, "no-pose"

    try:
        start = _read_pose(device)
    except Exception as e:
        logger.error(f"Could not read the {label} to plan its return: {e}")
        return False, "unreadable"

    targets = {m: v for m, v in pose.items() if m in start}
    if not targets:
        logger.warning(f"None of the {label}'s recorded joints are readable; skipping the return")
        return False, "no-pose"

    max_delta = max(abs(start[m] - v) for m, v in targets.items())
    if max_delta <= MAKER_RETURN_TOLERANCE_DEG:
        return True, ""

    # Distance sets duration, so the RATE is what stays bounded. A fixed
    # duration (lerobot's 3s) would make a long return fast and a short one
    # slow; capping the rate instead means every return feels the same.
    # Capped at a FRACTION of the ceiling, not the whole of it: the ramp only
    # commands the setpoints, and the settle check afterwards is what decides
    # whether the arm actually landed. A ramp allowed to consume the entire
    # budget would leave nothing for that check, so a blocked joint would
    # report a bare "timed out" instead of naming itself.
    duration_s = min(max_delta / max(speed_deg_s, 1e-6), ceiling_s * _RAMP_CEILING_FRACTION)
    steps = max(int(duration_s * MAKER_RETURN_FPS), 1)
    period = 1.0 / MAKER_RETURN_FPS
    deadline = time.monotonic() + ceiling_s

    logger.info(
        "Returning the %s to %s: %.1f deg worst-case over %.1fs",
        label,
        target_label,
        max_delta,
        duration_s,
    )

    try:
        for step in range(1, steps + 1):
            if abort_event is not None and abort_event.is_set():
                return False, "cut-short"
            if time.monotonic() > deadline:
                break
            t = step / steps
            device.send_action({f"{m}.pos": start[m] * (1 - t) + v * t for m, v in targets.items()})
            time.sleep(period)

        # The ramp finished commanding the target; now let the joints actually
        # settle onto it and decide whether they got close enough.
        best = float("inf")
        stalled = 0
        described = ""
        while time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                return False, "cut-short"
            device.send_action({f"{m}.pos": v for m, v in targets.items()})
            time.sleep(period)
            try:
                current = _read_pose(device)
            except Exception:
                continue  # transient CAN read miss; keep holding and re-read
            deltas = {m: abs(current[m] - v) for m, v in targets.items() if m in current}
            if not deltas:
                return False, "no-pose"
            motor, delta = max(deltas.items(), key=lambda kv: kv[1])
            described = f"{motor} still {delta:.1f} deg away"
            if delta <= MAKER_RETURN_TOLERANCE_DEG:
                return True, ""
            if best - delta >= MAKER_RETURN_STALL_PROGRESS_DEG:
                best = delta
                stalled = 0
                continue
            best = min(best, delta)
            stalled += 1
            if stalled < MAKER_RETURN_STALL_POLLS:
                continue
            # Converged as far as its gains allow.
            if delta <= MAKER_RETURN_SETTLE_DEG:
                logger.info("The %s settled at %s (%s)", label, target_label, described)
                return True, "settled"
            logger.warning("The %s stopped short of %s: %s", label, target_label, described)
            return False, described

        return False, described or "timed out"
    except Exception as e:
        # Documented never-raises: the caller is about to cut torque and must
        # not be skipped by an exception from the courtesy return.
        logger.error(f"Error returning the {label} to {target_label}: {e}")
        return False, str(e)


def return_maker_arms_to_rest(
    rest_poses: list[tuple[object, dict[str, float]]],
    abort_event: threading.Event | None = None,
    target_label: str = "its start pose",
) -> list[tuple[bool, str]]:
    """Return every captured Maker arm concurrently, then wait for all of them.

    Mirrors ``teleoperate._return_followers_to_rest``: the two arms of a
    bimanual rig sit on separate CAN buses, so returning them in series would
    take twice as long for no reason and leave the second arm hanging under
    gravity while the first moves.

    Returns each arm's ``(ok, reason)`` verdict, in the order given. The stop
    paths ignore it — they release torque either way, and a failure has already
    been logged by then — but the mid-session re-alignment reports its own
    outcome, and a thread whose join timed out must not be mistaken for a
    success, so the verdicts are collected rather than discarded.
    """
    if not rest_poses:
        return []
    if len(rest_poses) == 1:
        device, pose = rest_poses[0]
        return [return_maker_to_pose(device, pose, abort_event=abort_event, target_label=target_label)]

    results: list[tuple[bool, str]] = [(False, "did not finish")] * len(rest_poses)

    def _run(index: int, device: object, pose: dict[str, float]) -> None:
        results[index] = return_maker_to_pose(
            device,
            pose,
            abort_event=abort_event,
            label=f"Maker follower arm {index + 1}",
            target_label=target_label,
        )

    threads = [
        threading.Thread(
            target=_run,
            args=(i, device, pose),
            name=f"maker-return-{i}",
            daemon=True,
        )
        for i, (device, pose) in enumerate(rest_poses)
    ]
    for t in threads:
        t.start()
    # Bounded by each return's own ceiling; the join guard is belt-and-braces
    # so a wedged CAN read can't hold the stop path open indefinitely.
    for t in threads:
        t.join(timeout=MAKER_RETURN_CEILING_S + 2.0)
    return results
