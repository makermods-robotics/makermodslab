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
"""Gripper wiggle for a Damiao CAN arm — the identification of last resort.

``wiggle.py`` finds an SO-101 by jogging the gripper on a port so the user can
see which physical arm is on it. A Metal rig needs the same thing in exactly
one situation the probe and the gesture cannot cover: when its leader is a
second Metal arm (leader kind "metal"), every port answers the Damiao
protocol — ``maker_ports.probe_maker_ports`` cannot say which is the leader
and which the follower, and ``identify_maker_arm_by_motion`` is refused for
every Damiao device because opening its bus to watch a joint energizes the
motors mid-gesture. Driving ONE port's gripper a small stroke and asking the
user which arm moved is what is left, and it is safe where the gesture is
not: the bus is opened with ONLY the gripper motor on it, so the handshake
(the Damiao enable command, sent per motor) energizes nothing else; the
stroke stays inside the gripper's soft limits with a wide margin; and the
gripper is explicitly disabled again before the bus closes, with every
failure routed through ``torque.de_energize_can_bus`` so a handshake that
raised partway never leaves a motor held.

The mutex is ``wiggle.wiggle_active`` — this IS a wiggle, so it shares the
legacy module's flag and its ``robot.busy.wiggle`` discriminant rather than
adding a discriminant of its own — and it is refused while any feature holds
the bus (``sessions.held_by``), exactly like the Feetech wiggle.

``choose_identification`` is the pure decision the identify flow makes:
motion first wherever it is allowed, the wiggle when motion is refused or
found nothing on a family that supports it. Kept free of I/O so the fallback
is unit-testable without threads or hardware.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from .api_errors import ErrorCode
from .arms import registry as arm_registry
from .session_events import notify_session_changed
from .utils.config import normalize_arm_type

logger = logging.getLogger(__name__)

# The stroke, in degrees of gripper travel, each side of where the jaws are
# now: 8 deg is clearly visible on the Metal gripper (its full travel is 137.5
# deg of jaw opening) and small enough that a gripper holding an object at
# rest neither drops it nor crushes it. A gripper parked at or past a limit
# is first pulled the margin's width inside it (plan_can_wiggle).
WIGGLE_STROKE_DEG = 8.0

# Distance kept from EITHER soft limit. The vendor's gripper table documents
# jaw opening only up to 116.4 deg although the limit is 137.5, so the
# stroke never goes anywhere near the top of the range.
WIGGLE_LIMIT_MARGIN_DEG = 15.0

WIGGLE_REPEATS = 3
# Time the jaws get to move each way. MIT position control at the gains
# below reaches an 8 deg step comfortably within this.
WIGGLE_DWELL_S = 0.35
# Whole-run budget, mirroring wiggle._WIGGLE_TIMEOUT_S.
WIGGLE_TIMEOUT_S = 15.0

# MIT gains for the stroke: the follower's own gripper gains
# (MetalFollowerConfigBase.gains["gripper"], deliberately low so grip force
# stays compliant) — a wiggle must not squeeze harder than teleop does.
WIGGLE_KP = 20.0
WIGGLE_KD = 0.6

# Classic CAN at 1 Mbps over slcan, the wiring every Metal config defaults to.
_CAN_BITRATE = 1_000_000

_GRIPPER = "gripper"


def plan_can_wiggle(
    current_deg: float,
    limits: tuple[float, float],
    stroke_deg: float = WIGGLE_STROKE_DEG,
    margin_deg: float = WIGGLE_LIMIT_MARGIN_DEG,
) -> tuple[float, float, float]:
    """Plan a (high, low, rest) jog inside the gripper's soft limits.

    The window is the limits shrunk by ``margin_deg`` on each side; ``rest``
    is the current angle clamped so the whole stroke fits inside that window
    (a gripper parked at a limit is pulled in first, never pushed further
    out). Raises when the window is too narrow for the stroke.
    """
    lo = min(limits) + margin_deg
    hi = max(limits) - margin_deg
    if hi - lo < 2 * stroke_deg:
        raise ValueError(
            f"Gripper limits {limits} leave no room for a {stroke_deg:g} deg stroke inside a "
            f"{margin_deg:g} deg margin."
        )
    rest = min(max(float(current_deg), lo + stroke_deg), hi - stroke_deg)
    return rest + stroke_deg, rest - stroke_deg, rest


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
    """A Damiao bus carrying ONLY the gripper motor, built the way the
    follower builds its own (ids, model, wiring) and connected — which, on
    Damiao, enables that one motor and nothing else."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.damiao import DamiaoMotorsBus
    from lerobot.robots.metal_follower.metal_follower import MOTOR_MODELS

    family = arm_registry.get(normalize_arm_type(arm_type))
    send_id, recv_id = family._device_classes().follower_base(port=port).motor_can_ids[_GRIPPER]
    model = MOTOR_MODELS[_GRIPPER]
    motor = Motor(send_id, model, MotorNormMode.DEGREES)
    motor.recv_id = recv_id
    motor.motor_type_str = model
    return DamiaoMotorsBus(
        port=port,
        motors={_GRIPPER: motor},
        can_interface="slcan",
        use_can_fd=False,
        bitrate=_CAN_BITRATE,
        data_bitrate=None,
    )


def drive_gripper_wiggle(
    bus,
    limits: tuple[float, float],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Connect ``bus`` (gripper only), jog the jaws, and leave the motor DISABLED.

    Every exit — the normal one and any exception between the handshake and
    the last write — ends in ``de_energize_can_bus``: reopen without the
    handshake if the bus looks dead, broadcast the disable, close. That is
    what turns "the handshake energized the gripper" into "the gripper is
    limp again", whatever happened in between.
    """
    from .torque import de_energize_can_bus

    try:
        bus.connect()  # handshake=True: enables the ONE motor on this bus
        current = float(bus.read("Present_Position", _GRIPPER))
        high, low, rest = plan_can_wiggle(current, limits)
        for _ in range(WIGGLE_REPEATS):
            for target in (high, low):
                bus.sync_write_metal({_GRIPPER: (WIGGLE_KP, WIGGLE_KD, target, 0.0, 0.0)})
                sleep(WIGGLE_DWELL_S)
        bus.sync_write_metal({_GRIPPER: (WIGGLE_KP, WIGGLE_KD, rest, 0.0, 0.0)})
        sleep(WIGGLE_DWELL_S)
    finally:
        problems = de_energize_can_bus(bus, "gripper wiggle")
        if problems:
            # de_energize_can_bus already logged loudly; make the wiggle's
            # result say it too.
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
        drive_gripper_wiggle(_open_gripper_bus(arm_type, port), gripper_limits(arm_type))
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
