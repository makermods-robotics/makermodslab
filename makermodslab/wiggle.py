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
"""
Wiggle-to-find-port: drive the gripper on a given serial port a few times so the
user can see which physical arm is on that port. SO-101 uses raw servo positions;
Maker and Metal use a gripper-only CAN bus and their configured position limits.
Every jog returns to its captured starting position before disconnecting.
"""

import asyncio
import logging
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .api_errors import ErrorCode
from .session_events import notify_session_changed

logger = logging.getLogger(__name__)

# ~200 encoder steps (of 4096) is a small but clearly visible movement.
_WIGGLE_OFFSET = 200
_WIGGLE_REPEATS = 3
_WIGGLE_TIMEOUT_S = 15.0

# True while a wiggle is actually driving the gripper (set just before the
# blocking drive, cleared in a finally so it can never stick set on a
# timeout/exception). Wiggle has no "stop" button — it's a brief, bounded
# one-shot action — so this is the self-check half of the reciprocal mutex
# with teleop/record/inference/calibration/auto-calibration (see CLAUDE.md).
wiggle_active = False


def plan_wiggle(
    current: int, min_limit: int, max_limit: int, offset: int = _WIGGLE_OFFSET
) -> tuple[int, int, int]:
    """Plan a (high, low, rest) jog that stays inside the servo's programmed limits.

    Any prior calibration writes Min/Max_Position_Limit into the servo EEPROM and the
    firmware silently clamps Goal_Position to them — a jog planned against the factory
    0-4095 range can then move the wrong way. Keep the original rest position,
    even at a limit. Refuse to move if firmware limits would prevent returning.
    """
    lo = max(min_limit, 0)
    hi = min(max_limit, 4095)
    if hi - lo < 2 * offset:
        raise ValueError(
            f"Gripper's programmed position limits ({min_limit}-{max_limit}) are too narrow "
            "to wiggle in. Recalibrate this arm and try again."
        )
    if not lo <= current <= hi:
        raise ValueError(
            "Gripper is outside its programmed position limits; cannot return to its "
            "original position. Recalibrate this arm before wiggling."
        )
    return min(current + offset, hi), max(current - offset, lo), current


def _wiggle_gripper_sync(port: str) -> None:
    """Connect to the gripper (motor id 6) on `port` and wiggle it in place.

    Reads the current raw position and the servo's programmed position limits, then
    jogs +/- _WIGGLE_OFFSET inside those limits a few times. Blocking; run in a
    worker thread.
    """
    bus = FeetechMotorsBus(
        port=port,
        motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    try:
        bus.connect()
        current = bus.sync_read("Present_Position", "gripper", normalize=False)["gripper"]
        min_limit = bus.read("Min_Position_Limit", "gripper", normalize=False)
        max_limit = bus.read("Max_Position_Limit", "gripper", normalize=False)

        high, low, rest = plan_wiggle(current, min_limit, max_limit)

        try:
            # Load the captured position before torque so an old goal cannot
            # snap the gripper open when it was initially unpowered.
            bus.write("Goal_Position", "gripper", rest, normalize=False)
            bus.enable_torque("gripper")
            for _ in range(_WIGGLE_REPEATS):
                bus.write("Goal_Position", "gripper", high, normalize=False)
                time.sleep(0.3)
                bus.write("Goal_Position", "gripper", low, normalize=False)
                time.sleep(0.3)
        finally:
            # Also attempt the return when a jog fails partway through.
            bus.write("Goal_Position", "gripper", rest, normalize=False)
            _wait_for_rest(
                lambda: bus.sync_read("Present_Position", "gripper", normalize=False)["gripper"],
                rest,
                tolerance=10,
            )
    finally:
        bus.disconnect()


def _wait_for_rest(read_position, rest: float, tolerance: float) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
        if abs(read_position() - rest) <= tolerance:
            return
    raise RuntimeError("Gripper did not return to its original position.")


async def wiggle_gripper(port: str) -> dict:
    """
    Run the wiggle in a worker thread with a timeout. Returns a result dict
    ({"success": bool, "message": str}) — logical failures (port busy, arm off)
    are reported, not raised, so the endpoint stays HTTP 200 like the rest of the
    feature handlers.

    Guarded by the same reciprocal mutex as every other feature that drives the
    servos (see CLAUDE.md): refuses while teleop/record/inference/calibration/
    auto-calibration is active, and refuses a second concurrent wiggle via
    ``wiggle_active`` (there is no shared lock, just a self-check like the
    other five). Wiggle has no "stop" button — it's a brief, bounded one-shot
    action — so the rejection messages tell the caller to wait rather than to
    stop something.
    """
    global wiggle_active

    if not port or not port.strip():
        return {"success": False, "message": "No port provided."}

    # Lazy imports to dodge circular imports at module load time (matches the
    # existing pattern in teleoperate.py/record.py/rollout.py).
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        remote_host as _remote_host,
        remote_inference as _remote_inference,
        remote_teleoperate as _remote_teleoperate,
        replay as _replay,
        rollout as _rollout,
        teleoperate as _teleoperate,
    )

    if wiggle_active:
        return {
            "success": False,
            "message": "A gripper wiggle is already in progress.",
            "code": ErrorCode.ROBOT_BUSY_WIGGLE,
        }
    if _remote_host.hosting_active:
        return {
            "success": False,
            "message": "This robot is hosted for remote teleoperation. Stop hosting first.",
            "code": ErrorCode.ROBOT_BUSY_HOSTING,
        }
    if _remote_teleoperate.remote_teleoperation_active:
        return {
            "success": False,
            "message": "Remote teleoperation is currently active. Stop it first.",
            "code": ErrorCode.ROBOT_BUSY_REMOTE_TELEOPERATION,
        }
    if _teleoperate.teleoperation_active:
        return {
            "success": False,
            "message": "Teleoperation is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_TELEOPERATION,
        }
    if _record.recording_active:
        return {
            "success": False,
            "message": "Recording is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_RECORDING,
        }
    if _rollout.inference_active:
        return {
            "success": False,
            "message": "Inference is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_INFERENCE,
        }
    if _remote_inference.remote_inference_is_active():
        return {
            "success": False,
            "message": "Remote inference is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_REMOTE_INFERENCE,
        }
    if _calibrate.calibration_is_active():
        return {
            "success": False,
            "message": "Calibration is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
        }
    if _auto_calibrate.auto_calibration_is_active():
        return {
            "success": False,
            "message": "Auto-calibration is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
        }
    if _replay.replay_active:
        return {
            "success": False,
            "message": "Replay is currently active — wait for it to stop before wiggling.",
            "code": ErrorCode.ROBOT_BUSY_REPLAY,
        }
    # Lazy, because jobs imports this module back the same way.
    from . import jobs as _jobs

    if (training := _jobs.training_is_active()) is not None:
        return {
            "success": False,
            "message": (
                f"Training run '{training}' is using this machine — wait for it to stop before wiggling."
            ),
            "code": ErrorCode.ROBOT_BUSY_TRAINING,
        }

    wiggle_active = True
    # The claim above is the real state transition — broadcast the hint so
    # every WS client learns the robot is busy (wiggle has no status
    # endpoint of its own; consumers see the busy state via start refusals).
    notify_session_changed("wiggle", True)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_run_wiggle_and_clear_flag, port.strip()),
            timeout=_WIGGLE_TIMEOUT_S,
        )
        return {"success": True, "message": f"Wiggled the gripper on {port}."}
    except TimeoutError:
        # `wait_for` timing out only stops US from waiting — it does NOT stop
        # the thread `to_thread` submitted, which keeps driving the gripper
        # (and holding the port) in the background for however long it
        # actually takes to unwind. Do NOT clear wiggle_active here: that's
        # `_run_wiggle_and_clear_flag`'s job, tied to the thread's real exit,
        # so the mutex stays honest for other features' start checks.
        return {
            "success": False,
            "message": "Wiggle timed out after 15s — is the arm powered on and the port correct?",
        }
    except Exception as e:
        logger.exception("Wiggle failed")
        return {"success": False, "message": f"Failed to wiggle the gripper: {e}"}


def _run_wiggle_and_clear_flag(port: str) -> None:
    """Run the blocking wiggle, then clear ``wiggle_active`` — from inside the
    worker thread, so the flag reflects the thread's REAL exit rather than
    `wiggle_gripper`'s async wrapper giving up on a `wait_for` timeout (see the
    comment at that timeout's except clause).
    """
    global wiggle_active
    try:
        _wiggle_gripper_sync(port)
    finally:
        wiggle_active = False
        # Final release, tied to the thread's REAL exit (like the flag) so
        # the hint can never claim idle while the port is still held.
        notify_session_changed("wiggle", False)
