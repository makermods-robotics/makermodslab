# Copyright 2026 MakerMods. All rights reserved.
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
"""Identify a CAN arm by moving only its gripper, then returning to its start.

The family builds a bus with just the gripper motor. Capture its position
without the energizing handshake, validate it against the soft limits, then
jog using bounded MIT setpoints. Every exit attempts to return before torque
release and bus cleanup. The shared wiggle mutex lasts until the worker exits.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable

from .api_errors import ErrorCode
from .arms import registry as arm_registry
from .session_events import notify_session_changed
from .utils.config import normalize_arm_type

logger = logging.getLogger(__name__)

# Maximum excursion from the captured position. Near a limit the stroke is
# one-sided, so a closed/open gripper returns to exactly where it started.
WIGGLE_STROKE_DEG = 10.0

WIGGLE_REPEATS = 3
# Whole-run budget, mirroring wiggle._WIGGLE_TIMEOUT_S.
WIGGLE_TIMEOUT_S = 15.0

# MIT gains for the stroke: the follower's own gripper gains
# (MetalFollowerConfigBase.gains["gripper"], deliberately low so grip force
# stays compliant) — a wiggle must not squeeze harder than teleop does.
WIGGLE_KP = 20.0
WIGGLE_KD = 0.6

_GRIPPER = "gripper"


def plan_can_wiggle(
    current_deg: float,
    limits: tuple[float, float],
    stroke_deg: float = WIGGLE_STROKE_DEG,
) -> tuple[float, float, float]:
    """Keep the captured rest position and clip each excursion to the limits.

    Refuse an out-of-range start because the limits would prevent returning.
    """
    lo, hi = sorted(limits)
    if hi - lo < 2 * stroke_deg:
        raise ValueError(f"Gripper limits {limits} leave no room for a {stroke_deg:g} deg stroke.")
    if not math.isfinite(current_deg) or not lo <= current_deg <= hi:
        raise ValueError("Gripper is outside its position limits. Recalibrate before wiggling.")
    return min(current_deg + stroke_deg, hi), max(current_deg - stroke_deg, lo), current_deg


def choose_identification(
    family,
    device_type: str,
    motion_result: dict | None,
    leader_kind: str | None = None,
) -> str | None:
    """Which identification to run next, as a pure decision.

    ``motion_result`` is the outcome of the motion gesture already attempted
    (None when it was not attempted). The gesture stays the first attempt
    wherever the family allows it; when it was refused (a Damiao follower, or
    an energized leader) or did not find a port, a family with a gripper
    wiggle falls back to it. Returns ``"wiggle"`` for that fallback, ``None``
    when nothing else can be tried (the motion result stands as the answer,
    success or not).
    """
    if motion_result is not None and motion_result.get("success") and motion_result.get("port"):
        return None
    if not getattr(family, "supports_gripper_wiggle", False):
        return None
    if device_type == "teleop" and not family.leader_holds_torque(leader_kind):
        # A Star leader is an encoder-only arm on its own protocol: the probe
        # tells it apart and the gesture works on it — and it has no motor to
        # wiggle. Nothing to fall back to.
        return None
    return "wiggle"


def gripper_limits(arm_type: str) -> tuple[float, float]:
    """The follower's soft limits for the gripper, in degrees (the leader is
    the same arm, so they are its limits too)."""
    family = arm_registry.get(normalize_arm_type(arm_type))
    limits = family._device_classes().follower_base(port="unused").joint_limits[_GRIPPER]
    return float(limits[0]), float(limits[1])


def _open_gripper_bus(arm_type: str, port: str):
    """Ask the family for a bus carrying ONLY its gripper motor."""
    family = arm_registry.get(normalize_arm_type(arm_type))
    return family.gripper_bus(port)


def drive_gripper_wiggle(
    bus,
    limits: tuple[float, float],
    sleep: Callable[[float], None] = time.sleep,
    gains: tuple[float, float] = (WIGGLE_KP, WIGGLE_KD),
) -> None:
    """Capture, jog, return, and disable the gripper even when a write fails."""
    from .torque import de_energize_can_bus
    from .wiggle import _wait_for_rest

    try:
        bus.connect(handshake=False)
        current = float(bus.read("Present_Position", _GRIPPER))
        high, low, rest = plan_can_wiggle(current, limits)
        bus.write("Kp", _GRIPPER, gains[0])
        bus.write("Kd", _GRIPPER, gains[1])

        def move(target: float) -> None:
            start = float(bus.read("Present_Position", _GRIPPER))
            steps = max(1, math.ceil(abs(target - start) / 1.5))
            for step in range(1, steps + 1):
                bus.write("Goal_Position", _GRIPPER, start + (target - start) * step / steps)
                sleep(0.05)

        try:
            bus.enable_torque(_GRIPPER)
            bus.write("Goal_Position", _GRIPPER, current)
            for _ in range(WIGGLE_REPEATS):
                move(high)
                move(low)
        finally:
            move(rest)
            _wait_for_rest(lambda: bus.read("Present_Position", _GRIPPER), rest, tolerance=1.0)
    finally:
        problems = de_energize_can_bus(bus, "gripper wiggle")
        if problems:
            raise RuntimeError(" ".join(problems))


def _busy_refusal() -> dict | None:
    """The reciprocal mutex, phrased for a wiggle (nothing to stop — wait)."""
    from . import wiggle as _wiggle
    from .sessions import held_by

    if _wiggle.wiggle_active:
        return {
            "success": False,
            "message": "A gripper wiggle is already in progress.",
            "code": ErrorCode.ROBOT_BUSY_WIGGLE,
        }
    holder = held_by()
    if holder is not None:
        return {
            "success": False,
            "message": f"{holder.replace('_', ' ').capitalize()} is currently active — wait for it to stop before wiggling.",
            "code": _BUSY_CODES.get(holder, ErrorCode.ROBOT_NOT_READY),
        }
    from . import jobs as _jobs

    if (training := _jobs.training_is_active()) is not None:
        return {
            "success": False,
            "message": f"Training run '{training}' is using this machine — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_TRAINING,
        }
    return None


_BUSY_CODES = {
    "teleoperation": ErrorCode.ROBOT_BUSY_TELEOPERATION,
    "recording": ErrorCode.ROBOT_BUSY_RECORDING,
    "inference": ErrorCode.ROBOT_BUSY_INFERENCE,
    "replay": ErrorCode.ROBOT_BUSY_REPLAY,
    "calibration": ErrorCode.ROBOT_BUSY_CALIBRATION,
    "auto_calibration": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
    "wiggle": ErrorCode.ROBOT_BUSY_WIGGLE,
}


def _run_and_clear_flag(arm_type: str, port: str) -> None:
    """The blocking wiggle, clearing ``wiggle_active`` on the thread's REAL
    exit (not the async wrapper's timeout) — see wiggle._run_wiggle_and_clear_flag."""
    from . import wiggle as _wiggle

    try:
        family = arm_registry.get(normalize_arm_type(arm_type))
        config = family._device_classes().follower_base(port=port)
        drive_gripper_wiggle(
            _open_gripper_bus(arm_type, port), gripper_limits(arm_type), gains=config.gains[_GRIPPER]
        )
    finally:
        _wiggle.wiggle_active = False
        notify_session_changed("wiggle", False)


async def wiggle_can_gripper(
    arm_type: str, device_type: str, port: str, leader_kind: str | None = None
) -> dict:
    """Wiggle the gripper of the Damiao arm on ``port`` so the user can see which it is.

    ``device_type`` names which side the caller is trying to identify; a
    Star leader (encoder-only, no motor to move) is refused outright. Returns
    ``{"success", "message"}`` (+ ``"code"`` on a busy refusal); logical
    failures are reported, not raised, so the route stays HTTP 200 like the
    other hardware handlers.
    """
    from . import wiggle as _wiggle

    if not port or not port.strip():
        return {"success": False, "message": "No port provided."}
    family = arm_registry.get(normalize_arm_type(arm_type))
    if device_type == "teleop" and not family.leader_holds_torque(leader_kind):
        return {
            "success": False,
            "message": (
                "The Star Arm 102 leader has no motors to wiggle. Its port answers the leader "
                "protocol probe on its own."
            ),
        }
    refusal = _busy_refusal()
    if refusal is not None:
        return refusal

    _wiggle.wiggle_active = True
    notify_session_changed("wiggle", True)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_run_and_clear_flag, arm_type, port.strip()),
            timeout=WIGGLE_TIMEOUT_S,
        )
        return {
            "success": True,
            "message": f"Wiggled the gripper on {port}. The arm whose jaws moved is on this port.",
        }
    except TimeoutError:
        # The worker keeps unwinding (and clears the flag itself); only our
        # wait gave up. Same reasoning as wiggle.wiggle_gripper.
        return {
            "success": False,
            "message": (
                f"Wiggle timed out after {WIGGLE_TIMEOUT_S:.0f}s — is the arm powered on and the port correct?"
            ),
        }
    except Exception as e:
        logger.exception("CAN gripper wiggle failed")
        return {"success": False, "message": f"Failed to wiggle the gripper: {e}"}
