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

"""Start-pose capture, first-action ease-in and return-to-rest for `robot_sync`.

Split out of `robot_sync` so it stays importable WITHOUT the `drtc` extra: the
entrypoint imports `livekit.portal` (an FFI dylib) at module top, this does not,
so the helpers below are unit-testable in ordinary CI. It does import
`makermodslab.rest_pose`, which needs only `lerobot.motors` — a hard dependency
of the Lab, not part of the optional extra.

**Feetech only, deliberately.** `robot_sync` registers `so_follower`,
`bi_so_follower`, `koch_follower` and `omx_follower` with draccus; the CAN arms
(`maker_follower` / `metal_follower`) are not registered there at all, so a CAN
arm is simply unreachable through this entrypoint and `maker_rest_pose.py` has
no call site here. Koch and OMX ARE reachable and are Dynamixel, not Feetech:
their control table names overlap enough that `rest_pose` would appear to work,
but `RETURN_POS_SPEED` and `RETURN_ARRIVE_TOLERANCE` are Feetech-unit constants
that mean something else entirely there. So :func:`feetech_buses` gates on the
bus type and everything else no-ops loudly for a non-Feetech arm rather than
guessing.

MIRROR NOTE: `bus_keyed` and `feetech_buses` are hand-mirrored twins of
`replay._bus_keyed` and `teleoperate._device_buses` / `motor_power._device_buses`
(the latter pair already mirror each other by hand, for a documented cycle
reason). Importing `replay` or `teleoperate` here would drag the FastAPI-side
session machinery into a subprocess that must stay a thin robot client, so the
duplication is deliberate — but it is duplication, and consolidating the three
`_device_buses` copies plus the two ease constants into one shared helper module
is a worthwhile follow-up. Change one, check the others.
"""

from __future__ import annotations

import logging
import threading

from lerobot.motors.feetech import FeetechMotorsBus

from ..rest_pose import capture_rest_pose, return_to_rest_pose

logger = logging.getLogger(__name__)

# Bus types `rest_pose`'s register names and unit constants are valid for. A
# module-level tuple rather than a bare isinstance so tests can substitute a
# fake bus class without a real serial port.
FEETECH_BUS_TYPES: tuple[type, ...] = (FeetechMotorsBus,)

# Ease-in tolerances, in the SAME normalized units as robot.send_action() — NOT
# raw ticks. Hand-mirrored from `replay.EASE_ARRIVE_TOLERANCE` /
# `EASE_STALL_MIN_PROGRESS`, which carry the full derivation: normalized units
# are coarser than ticks (~11-19 ticks per unit on this arm), so 2.0 units is
# ~23-39 ticks — looser than rest_pose's raw-ticks RETURN_ARRIVE_TOLERANCE of
# 20, not tighter; and rest_pose's raw-ticks RETURN_STALL_MIN_PROGRESS would be
# a ~15x too strict progress bar in normalized space. Kept local rather than
# imported from `replay` (see the module docstring's MIRROR NOTE).
EASE_ARRIVE_TOLERANCE = 2.0
EASE_STALL_MIN_PROGRESS = 1.0

# The gripper is excluded from the captured start pose, matching teleoperation
# and recording rather than replay: at stop time the policy may have left it
# holding something, and driving it back to its (likely open) starting width
# would drop the object mid-return. Replay includes it because the dataset
# drives the gripper and its start width is part of the pose being restored —
# not the case here, where the pose is "wherever the operator left the arm".
GRIPPER = "gripper"


def _device_buses(device) -> list:
    """The motor bus(es) of a robot device.

    A single-arm device exposes ``.bus``; a bimanual BiSO device exposes
    ``left_arm``/``right_arm`` sub-arms which each carry their own bus.
    (Hand-mirrored from `teleoperate._device_buses` — see the module docstring.)
    """
    if device is None:
        return []
    arms = [
        arm
        for arm in (getattr(device, "left_arm", None), getattr(device, "right_arm", None))
        if arm is not None
    ]
    targets = arms if arms else [device]
    return [target.bus for target in targets if getattr(target, "bus", None) is not None]


def feetech_buses(device) -> list:
    """The device's buses that `rest_pose`'s constants are actually valid for.

    Empty for a Dynamixel arm (Koch/OMX), which is the signal every caller here
    uses to skip rather than to guess — see the module docstring."""
    return [bus for bus in _device_buses(device) if isinstance(bus, FEETECH_BUS_TYPES)]


def capture_start_poses(device) -> list[tuple[object, dict[str, int | float]]]:
    """`(bus, pose)` per Feetech bus, raw ticks, gripper excluded.

    Call right after `robot.connect()`, before anything moves. Never raises —
    `capture_rest_pose` already swallows a comm failure and returns `{}`, and a
    session must not fail to start over an optional nicety; a `{}` pose simply
    makes the return a no-op that reports `no-pose`."""
    return [
        (bus, {m: v for m, v in capture_rest_pose(bus, normalize=False).items() if m != GRIPPER})
        for bus in feetech_buses(device)
    ]


def bus_keyed(action: dict[str, float], bus) -> dict[str, float]:
    """Re-key an action dict onto the plain motor names the bus uses.

    lerobot's robot-level action feature names carry a `<motor>.pos` suffix, but
    `bus.motors` — and therefore `rest_pose`'s target filtering — is keyed by
    bare motor name. Passing an unconverted action dict straight to
    `return_to_rest_pose` matches zero motors and yields "no-pose" without ever
    writing a Goal_Position. This mirrors what `robot.send_action()` already
    does internally at the same robot->bus boundary (`key.removesuffix(".pos")`);
    the bare-name fallback keeps an unsuffixed action dict working too."""
    keyed: dict[str, float] = {}
    for motor in bus.motors:
        for key in (f"{motor}.pos", motor):
            if key in action:
                keyed[motor] = action[key]
                break
    return keyed


def ease_to_action(
    device,
    action: dict[str, float],
    abort_event: threading.Event | None = None,
    label: str = "follower arm",
) -> tuple[bool, str]:
    """Drive the arm to `action` at the gentle profile speed, then report.

    This is the FIRST-ACTION EASE-IN: without it the first `send_action` after
    connect steps the arm straight to the policy's first commanded pose at full
    speed, which is the snap-to-pose family of issue analysed for teleop and
    record on 2026-09-01. `lerobot-rollout` does not ramp on entry either (its
    `ActionInterpolator` explicitly runs the first step raw, "no previous action
    yet"), so this is an improvement on the local sibling rather than parity
    with it — but lerobot DOES ramp on EXIT, over a 3 s interpolation in
    `RolloutStrategy._return_to_initial_position`, which is the same shape in
    the other direction.

    Reuses `rest_pose.return_to_rest_pose` with `normalize=True` — the same
    arbitrary-target primitive `replay` uses for its frame-0 approach, with the
    same normalized-unit tolerances. Returns `(arrived, reason)`; `cut-short`
    means the abort event fired (a stop landed during the ease).

    Single-bus only. A bimanual BiSO robot's action keys are `left_`/`right_`
    prefixed while each sub-arm's `bus.motors` are bare, so `bus_keyed` would
    silently match nothing per bus; rather than guess a prefix convention here,
    the ease-in reports `unsupported` and the caller skips it. Return-to-rest
    below has no such problem — it works per bus in raw ticks."""
    buses = feetech_buses(device)
    if len(buses) != 1:
        detail = "no Feetech bus" if not buses else f"{len(buses)} buses"
        logger.warning(f"First-action ease-in skipped for the {label}: {detail}")
        return False, f"unsupported ({detail})"
    bus = buses[0]
    targets = bus_keyed(action, bus)
    if not targets:
        logger.warning(f"First-action ease-in skipped for the {label}: no motor matched the action")
        return False, "no-pose"
    return return_to_rest_pose(
        bus,
        targets,
        abort_event=abort_event,
        label=label,
        normalize=True,
        tolerance=EASE_ARRIVE_TOLERANCE,
        stall_min_progress=EASE_STALL_MIN_PROGRESS,
    )


def ensure_uncapped(device, label: str = "follower arm") -> None:
    """Clear the RAM Goal_Velocity profile cap the ease-in wrote, and verify it took.

    `return_to_rest_pose` stamps the gentle `RETURN_POS_SPEED` into every
    driven motor's RAM `Goal_Velocity` and clears it again in its own `finally`
    — but that restore is deliberately best-effort, so a single dropped serial
    write would leave a 400 profile cap throttling the ENTIRE run: slow but
    accurate-looking, with nothing anywhere to say why. The register is
    RAM-persistent across sessions (only a power cycle resets it), so it would
    also outlive this process. Read back and retry once, then say so loudly.

    Hand-mirrored from `replay._ensure_uncapped`, which guards the identical
    hazard between its frame-0 ease-in and playback (see the module docstring's
    MIRROR NOTE)."""
    for bus in feetech_buses(device):
        _clear_goal_velocity(bus, label)
        try:
            stuck = {m: v for m, v in bus.sync_read("Goal_Velocity", normalize=False).items() if v}
        except Exception as e:
            logger.warning(f"Could not verify the speed cap was cleared for the {label}: {e}")
            continue
        if not stuck:
            continue
        logger.warning(f"Speed cap survived on the {label} ({stuck}) — retrying before the run")
        _clear_goal_velocity(bus, label)
        try:
            still = {m: v for m, v in bus.sync_read("Goal_Velocity", normalize=False).items() if v}
            if still:
                logger.error(
                    f"The {label} is still speed-capped ({still}); this run will move slower than "
                    "the policy commands. A power cycle clears the register if the write keeps failing."
                )
        except Exception:
            pass


def _clear_goal_velocity(bus, label: str) -> None:
    """Best-effort `Goal_Velocity = 0` (uncapped) on every motor of one bus."""
    try:
        bus.sync_write("Goal_Velocity", dict.fromkeys(bus.motors, 0), normalize=False)
    except Exception as e:
        logger.warning(f"Could not clear the speed cap (Goal_Velocity) for the {label}: {e}")


def return_to_start_poses(
    poses: list[tuple[object, dict[str, int | float]]],
    abort_event: threading.Event | None = None,
    label: str = "follower arm",
) -> list[tuple[bool, str]]:
    """Drive every captured bus back to its start pose, in raw ticks.

    Torque must still be enabled — call BEFORE `robot.disconnect()`, which is
    what releases it. A second STOP sets `abort_event` and each return comes
    back `cut-short`, leaving the arm nearer rest than it started."""
    results = []
    for index, (bus, pose) in enumerate(poses):
        bus_label = label if len(poses) == 1 else f"{label} (bus {index + 1}/{len(poses)})"
        arrived, reason = return_to_rest_pose(bus, pose, abort_event=abort_event, label=bus_label)
        level = logger.info if arrived else logger.warning
        level(f"Rest-pose return for the {bus_label}: {reason}")
        results.append((arrived, reason))
    return results
