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
Calibration module for the web interface.

This module provides calibration functionality similar to the CLI calibrate.py,
but adapted for the web interface with step-by-step guidance.
"""

import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import (
    Robot,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    make_teleoperator_from_config,
)
from lerobot.utils.utils import init_logging

from .api_errors import ErrorCode
from .hardware_lease import (
    HardwareLeaseHeld,
    HardwareLeaseToken,
    hardware_lease_registry,
    held_response,
    safe_hardware_receipt,
)
from .hardware_recovery_identity import hardware_recovery_identity
from .session_events import notify_session_changed
from .utils.config import calibration_dir_for_device, save_robot_record

logger = logging.getLogger(__name__)

# Raw-tick center of the 12-bit (0-4095) Feetech encoder. Step 1 homing offsets
# the start pose to read ~2047, so a calibration that really started from the
# documented middle position records a range whose midpoint sits near this value.
ENCODER_MID_TICK = 2047

# Allowed deviation of a recorded range's midpoint from ENCODER_MID_TICK, as a
# fraction of the recorded range width.
CENTERING_TOLERANCE = 0.2

# Joints exempt from the centering check: users legitimately home the gripper
# closed (~580 ticks of midpoint deviation on real data), and wrist_roll is a
# full-turn motor upstream (lerobot's CLI calibration forces its range to
# 0-4095 instead of recording it), so its recorded midpoint carries no meaning.
CENTERING_EXEMPT_MOTORS = frozenset({"gripper", "wrist_roll"})

# wrist_roll is a continuous full-turn joint. Official lerobot-calibrate
# behavior (lerobot/robots/so_follower/so_follower.py::calibrate, same in
# so_leader.py): the user is told to move all joints EXCEPT wrist_roll, and its
# range is unconditionally hardcoded to the full turn — a continuous joint has
# no min/max, only the homing offset from the middle pose. Sweeping it crosses
# the encoder wrap (a ~4096 single-frame jump), so it is exempt from the
# discontinuity check and any recorded range is discarded.
FULL_TURN_MOTORS = frozenset({"wrist_roll"})
FULL_TURN_RANGE = (0, 4095)


def final_motor_ranges(mins: dict[str, int], maxes: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Recorded (min, max) per motor, with full-turn joints forced to 0-4095."""
    return {
        motor: (FULL_TURN_RANGE if motor in FULL_TURN_MOTORS else (mins[motor], maxes[motor]))
        for motor in mins
    }


def find_off_center_joints(ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Return the joints whose recorded range is not centered on the start pose.

    `ranges` maps motor name to (range_min, range_max) in raw ticks. A joint
    passes when |ENCODER_MID_TICK - (range_min + range_max) / 2| is at most
    CENTERING_TOLERANCE of the range width; a larger deviation means the joint
    started near one of its limits rather than mid-range, so the homing offsets
    captured in step 1 would skew the saved calibration.
    """
    offending = []
    for motor, (range_min, range_max) in ranges.items():
        if motor in CENTERING_EXEMPT_MOTORS:
            continue
        midpoint = (range_min + range_max) / 2
        if abs(ENCODER_MID_TICK - midpoint) > CENTERING_TOLERANCE * (range_max - range_min):
            offending.append(motor)
    return offending


class CalibrationCenteringError(Exception):
    """Raised when the recorded ranges show calibration didn't start mid-pose.

    Detected after range recording, before anything is saved: if a joint's
    recorded range is heavily skewed to one side of the raw-tick center, the
    arm was not in the documented middle position when calibration began.
    """


class CalibrationDiscontinuityError(Exception):
    """Raised when a motor position reading jumps across the encoder wrap-around.

    The Feetech encoder is 12-bit (0-4095); if calibration starts with a joint
    near a boundary, moving it past 0 or 4095 produces a single-frame delta of
    ~4096. The user-side fix is to start with all joints in the middle of their
    range, as documented in the SO-101 docs.
    """


@dataclass
class CalibrationStatus:
    """Status information for calibration process"""

    calibration_active: bool = False
    status: str = "idle"  # "idle", "connecting", "recording", "completed", "error", "stopping"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0  # Current calibration step
    total_steps: int = 1  # Total number of calibration steps
    current_positions: dict[str, float] = None
    recorded_ranges: dict[str, dict[str, float]] = None  # {motor: {min: val, max: val, current: val}}


@dataclass
class CalibrationRequest:
    """Request parameters for starting calibration"""

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None  # When set, write port + config back into the robot record on success
    overwrite: bool = False  # Must be explicitly true to replace an existing config file of the same name
    arm: Literal["left", "right"] = (
        "left"  # Which arm of a bimanual robot; "left" is also the single-arm pair
    )


class CalibrationManager:
    """Manages calibration process for the web interface"""

    def __init__(self):
        self.status = CalibrationStatus()
        self.device: Robot | Teleoperator | None = None
        self.calibration_thread: threading.Thread | None = None
        self.stop_calibration = False
        # RLock (not Lock): start_calibration holds this across its whole
        # check-and-claim critical section, which itself calls
        # _update_status — a plain Lock would deadlock on that reentrant
        # acquisition from the same thread.
        self._status_lock = threading.RLock()
        self._step_complete = threading.Event()
        self._recording_active = False
        self._start_positions = {}
        self._mins = {}
        self._maxes = {}
        self._homing_offsets = {}
        self._current_request: CalibrationRequest | None = None
        # Rollback baseline for a cancelled/failed run (see _calibration_worker
        # and _cleanup_device): the servo's Homing_Offset and Min/Max_Position_Limit
        # values as they stood before this session's _step_homing touched them
        # (reset_calibration() zeroes/maxes out all three, not just the homing
        # offset), and whether the run reached a real commit (on-disk file
        # written) that supersedes them.
        self._original_homing_offsets: dict[str, int] = {}
        self._original_min_position_limits: dict[str, int] = {}
        self._original_max_position_limits: dict[str, int] = {}
        self._calibration_committed = False
        # Guards _cleanup_device's body against two *callers of
        # _cleanup_device* overlapping — whichever thread arrives second finds
        # self.device already None and no-ops. That is a narrower guarantee
        # than "no concurrent bus access": the actual hazard it exists for is
        # in stop_calibration_process, where join(timeout=5.0) can expire
        # while the worker thread is still alive and mid read/write on the
        # bus (e.g. blocked in a sync_read retry inside
        # _step_range_recording). The request-handling thread then forces a
        # cleanup call — restoring registers and disconnecting — while the
        # worker thread may still be actively using that same device. This
        # lock only prevents that forced cleanup from double-running against
        # the worker's own eventual _cleanup_and_finish; it does not, and
        # cannot, stop the worker's un-locked bus I/O from overlapping the
        # forced restore+disconnect.
        self._cleanup_lock = threading.Lock()
        self._hardware_lease_token: HardwareLeaseToken | None = None

        # Initialize logging
        init_logging()

    def get_status(self) -> CalibrationStatus:
        """Get current calibration status"""
        with self._status_lock:
            # Update current positions if we're recording and device is connected
            if self.status.status == "recording" and self.device and self.device.is_connected:
                try:
                    # Try reading positions with quick retry on port contention
                    positions = None
                    for attempt in range(2):  # Quick retry for status updates
                        try:
                            positions = self.device.bus.sync_read("Present_Position", normalize=False)
                            break
                        except Exception as read_error:
                            if "Port is in use" in str(read_error) and attempt < 1:
                                time.sleep(0.005)  # Very short delay
                                continue
                            else:
                                raise read_error

                    if positions:
                        # Update recorded ranges
                        if not self.status.recorded_ranges:
                            self.status.recorded_ranges = {}

                        for motor, pos in positions.items():
                            # Filter out invalid readings (0, negative, or extreme values)
                            if pos <= 0 or pos >= 5000:
                                continue  # Skip invalid readings

                            if motor in FULL_TURN_MOTORS:
                                # Report the range that will actually be saved
                                # (forced full turn), not the swept sliver —
                                # the live marker still tracks `current`.
                                full_min, full_max = FULL_TURN_RANGE
                                self.status.recorded_ranges[motor] = {
                                    "min": full_min,
                                    "max": full_max,
                                    "current": pos,
                                }
                            elif motor not in self.status.recorded_ranges:
                                self.status.recorded_ranges[motor] = {"min": pos, "max": pos, "current": pos}
                            else:
                                self.status.recorded_ranges[motor]["current"] = pos
                                self.status.recorded_ranges[motor]["min"] = min(
                                    self.status.recorded_ranges[motor]["min"], pos
                                )
                                self.status.recorded_ranges[motor]["max"] = max(
                                    self.status.recorded_ranges[motor]["max"], pos
                                )
                except Exception as e:
                    # Reduce log spam by using debug level for expected port contention
                    if "Port is in use" in str(e):
                        logger.debug(f"Port busy during position read: {e}")
                    else:
                        logger.warning(f"Failed to read positions: {e}")

            return self.status

    def _update_status(self, **kwargs):
        """Update calibration status thread-safely"""
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    def start_calibration(self, request: CalibrationRequest) -> dict[str, Any]:
        """Start calibration process"""
        try:
            # The "already active?" check and the claim (calibration_active =
            # True below) must be atomic: held under one lock acquisition, not
            # two separate ones. Without this, two concurrent callers can both
            # read calibration_active as False before either sets it True, and
            # both spawn a worker thread against the same arm.
            with self._status_lock:
                if self.status.calibration_active:
                    return {
                        "success": False,
                        "message": "Calibration already active",
                        "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
                    }

                # Mutex with every other feature that drives the same serial bus
                # (see CLAUDE.md's "State model & mutual exclusion"). Lazy
                # imports to dodge circular imports at module load time (matches
                # the existing pattern in teleoperate.py/record.py/rollout.py).
                from . import (
                    auto_calibrate as _auto_calibrate,
                    record as _record,
                    replay as _replay,
                    rollout as _rollout,
                    teleoperate as _teleoperate,
                    wiggle as _wiggle,
                )

                if _teleoperate.teleoperation_active:
                    return {
                        "success": False,
                        "message": "Teleoperation is currently active. Stop it first.",
                        "code": ErrorCode.ROBOT_BUSY_TELEOPERATION,
                    }
                if _record.recording_active:
                    return {
                        "success": False,
                        "message": "Recording is currently active. Stop it first.",
                        "code": ErrorCode.ROBOT_BUSY_RECORDING,
                    }
                if _rollout.inference_active:
                    return {
                        "success": False,
                        "message": "Inference is currently active. Stop it first.",
                        "code": ErrorCode.ROBOT_BUSY_INFERENCE,
                    }
                if _auto_calibrate.auto_calibration_is_active():
                    return {
                        "success": False,
                        "message": "Auto-calibration is currently active. Stop it first.",
                        "code": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
                    }
                if _wiggle.wiggle_active:
                    return {
                        "success": False,
                        "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
                        "code": ErrorCode.ROBOT_BUSY_WIGGLE,
                    }
                if _replay.replay_active:
                    return {
                        "success": False,
                        "message": "Replay is currently active. Stop it first.",
                        "code": ErrorCode.ROBOT_BUSY_REPLAY,
                    }

                # Lazy, because jobs imports this module back the same way.
                from . import jobs as _jobs

                if (training := _jobs.training_is_active()) is not None:
                    return {
                        "success": False,
                        "message": (f"Training run '{training}' is using this machine. Stop it first."),
                        "code": ErrorCode.ROBOT_BUSY_TRAINING,
                    }

                # Refuse to silently overwrite an existing config file. Completing a
                # calibration saves "<config_file>.json"; if that name is taken, the
                # caller must pass overwrite=True (after confirming) or pick another
                # name. Lets the frontend warn before any data is clobbered.
                config_dir = calibration_dir_for_device(request.device_type)
                if config_dir is not None and not request.overwrite:
                    stem = (
                        request.config_file[:-5]
                        if request.config_file.endswith(".json")
                        else request.config_file
                    )
                    if os.path.exists(os.path.join(config_dir, f"{stem}.json")):
                        return {
                            "success": False,
                            "code": "name_taken",
                            "message": f"A calibration named '{stem}' already exists. Overwrite it or choose a different name.",
                        }

                try:
                    lease_token = hardware_lease_registry.claim(
                        "calibration",
                        f"local:{request.device_type}:{request.port}",
                        recovery=hardware_recovery_identity(
                            "so101",
                            target_ports=(request.port,),
                        ),
                    )
                except HardwareLeaseHeld as exc:
                    return held_response(exc)
                self._hardware_lease_token = lease_token

                # Reset status and clear any previous calibration data
                self._start_positions = {}
                self._mins = {}
                self._maxes = {}
                self._homing_offsets = {}
                self._original_homing_offsets = {}
                self._original_min_position_limits = {}
                self._original_max_position_limits = {}
                self._calibration_committed = False

                self._update_status(
                    calibration_active=True,
                    status="connecting",
                    device_type=request.device_type,
                    error=None,
                    message=f"Starting calibration for {request.device_type}",
                    step=0,
                    current_positions=None,
                    recorded_ranges=None,
                )
                self._current_request = request

                # Start calibration in a separate thread
                self.calibration_thread = threading.Thread(
                    target=self._calibration_worker, args=(request,), daemon=True
                )
                self.stop_calibration = False
                self._step_complete.clear()
                self.calibration_thread.start()

            # The claim above (calibration_active=True under the lock) is the
            # real state transition — broadcast the hint so every WS client
            # refetches /calibration-status.
            notify_session_changed("calibration", True, phase="connecting")
            return {"success": True, "message": "Calibration started"}

        except Exception as e:
            logger.error(f"Error starting calibration: {e}")
            self._update_status(
                calibration_active=False, status="error", error=str(e), message="Failed to start calibration"
            )
            # Undo any claim hint: a start that failed after claiming must not
            # leave clients believing a session is live. (Harmless when the
            # failure predates the claim — clients refetch and see idle.)
            notify_session_changed("calibration", False, phase="error")
            token = self._hardware_lease_token
            if token is not None and hardware_lease_registry.is_token_current(token):
                worker_alive = self.calibration_thread is not None and self.calibration_thread.is_alive()
                snapshot = hardware_lease_registry.snapshot()
                if not worker_alive and self.device is None and snapshot.state == "active":
                    hardware_lease_registry.release(
                        token,
                        safe_hardware_receipt(
                            "calibration request thread failed before opening hardware",
                            torque_off=None,
                            torque_not_applicable=True,
                        ),
                    )
                    self._hardware_lease_token = None
                else:
                    hardware_lease_registry.mark_unresolved(
                        token,
                        f"calibration startup failed before its worker finalized: {e}",
                    )
            return {"success": False, "message": str(e)}

    def complete_step(self) -> dict[str, Any]:
        """Complete the current calibration step"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            if self.status.status == "recording":
                # Complete recording step
                self._recording_active = False
                self._step_complete.set()
                return {"success": True, "message": "Range recording completed"}

            else:
                return {"success": False, "message": f"Cannot complete step in status: {self.status.status}"}

        except Exception as e:
            logger.error(f"Error completing step: {e}")
            return {"success": False, "message": str(e)}

    def stop_calibration_process(self) -> dict[str, Any]:
        """Stop calibration process"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping calibration process...")
            token = self._hardware_lease_token
            if token is not None and hardware_lease_registry.is_token_current(token):
                hardware_lease_registry.request_stop(token, "operator_stop")
            self.stop_calibration = True
            self._recording_active = False
            self._step_complete.set()  # Unblock any waiting step

            self._update_status(status="stopping", message="Stopping calibration...")

            # Wait for thread to finish
            if self.calibration_thread and self.calibration_thread.is_alive():
                self.calibration_thread.join(timeout=5.0)

            # Ensure cleanup is called if thread didn't finish properly
            if self.calibration_thread and self.calibration_thread.is_alive():
                reason = "Calibration worker did not finish within 5 seconds; hardware state is unknown"
                logger.warning(reason)
                if token is not None and hardware_lease_registry.is_token_current(token):
                    hardware_lease_registry.mark_unresolved(token, reason)
                return {
                    "success": True,
                    "shutting_down": True,
                    "warning": reason,
                    "message": "Calibration stop requested; safe release is not yet confirmed",
                }

            logger.info("Calibration stop completed")
            return {"success": True, "message": "Calibration stopped"}

        except Exception as e:
            logger.error(f"Error stopping calibration: {e}")
            token = self._hardware_lease_token
            if token is not None and hardware_lease_registry.is_token_current(token):
                hardware_lease_registry.mark_unresolved(token, f"calibration stop failed: {e}")
            return {"success": False, "message": str(e)}

    def _calibration_worker(self, request: CalibrationRequest):
        """Worker thread for calibration process"""
        try:
            logger.info(f"Starting calibration worker for {request.device_type}")

            # Create device configuration
            if request.device_type == "robot":
                from lerobot.robots.so_follower import SO101FollowerConfig

                config = SO101FollowerConfig(port=request.port, id=request.config_file)
            elif request.device_type == "teleop":
                from lerobot.teleoperators.so_leader import SO101LeaderConfig

                config = SO101LeaderConfig(port=request.port, id=request.config_file)
            else:
                raise ValueError(f"Unknown device type: {request.device_type}")

            self._update_status(status="connecting", message="Connecting to device...")

            # Create and connect device
            if request.device_type == "robot":
                self.device = make_robot_from_config(config)
            else:
                self.device = make_teleoperator_from_config(config)

            logger.info("Connecting to device...")
            self.device.connect(calibrate=False)

            if self.stop_calibration:
                logger.info("Calibration stopped after device connection")
                self._cleanup_and_finish("Calibration cancelled")
                return

            # Capture the servo's current Homing_Offset and Min/Max_Position_Limit
            # as the rollback baseline BEFORE _step_homing runs — it calls
            # reset_calibration(), which immediately zeroes Homing_Offset and
            # Min_Position_Limit and maxes out Max_Position_Limit in EEPROM for
            # every motor on the bus, so this is the last point the prior
            # (possibly still-in-use) values are readable. If the run is
            # cancelled or errors before _complete_calibration commits new
            # values to disk, _cleanup_device restores this baseline so the
            # servo never diverges from the last-saved calibration file.
            #
            # A failed read here isn't caught locally: the device is freshly
            # connected and nothing has been written yet, so there's nothing
            # cheaper to retry into. Letting it propagate to this method's own
            # except Exception below aborts the run with a clear error instead
            # of silently proceeding with no rollback safety net for the one
            # run that most needs it.
            self._original_homing_offsets = {
                motor: self.device.bus.read("Homing_Offset", motor, normalize=False)
                for motor in self.device.bus.motors
            }
            self._original_min_position_limits = {
                motor: self.device.bus.read("Min_Position_Limit", motor, normalize=False)
                for motor in self.device.bus.motors
            }
            self._original_max_position_limits = {
                motor: self.device.bus.read("Max_Position_Limit", motor, normalize=False)
                for motor in self.device.bus.motors
            }

            # Start Step 1: Homing
            self._step_homing()

            if self.stop_calibration:
                logger.info("Calibration stopped after homing step")
                self._cleanup_and_finish("Calibration cancelled")
                return

            # Start Step 2: Range Recording
            self._step_range_recording()

            if self.stop_calibration:
                logger.info("Calibration stopped after recording step")
                self._cleanup_and_finish("Calibration cancelled")
                return

            # Complete calibration
            self._complete_calibration()

            logger.info("Calibration completed successfully")
            self._cleanup_and_finish("Calibration completed successfully", status="completed")

        except (CalibrationCenteringError, CalibrationDiscontinuityError) as e:
            logger.error(f"Calibration aborted: {e}")
            self._update_status(error=str(e))
            self._cleanup_and_finish(str(e), status="error")
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            logger.error(traceback.format_exc())
            # Ensure cleanup happens even on error
            self._cleanup_and_finish(f"Calibration failed: {e}", status="error")
        finally:
            # Ensure we always clean up and reset the active flag
            logger.info("Calibration worker thread finishing")
            if self.status.calibration_active:
                logger.warning(
                    "Worker thread ending but calibration still marked as active - forcing cleanup"
                )
                self._cleanup_and_finish("Calibration stopped", status="idle")

    def _step_homing(self):
        """Auto-capture homing offsets from the device's current position."""
        logger.info("Setting homing offsets from current position")

        # Disable torque to allow manual movement during recording
        self.device.bus.disable_torque()
        for motor in self.device.bus.motors:
            self.device.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self.device.bus.reset_calibration()
        actual_positions = self.device.bus.sync_read("Present_Position", normalize=False)
        logger.info(f"Current positions for homing: {actual_positions}")

        self._homing_offsets = self.device.bus._get_half_turn_homings(actual_positions)
        logger.info(f"Calculated homing offsets: {self._homing_offsets}")

        for motor, offset in self._homing_offsets.items():
            self.device.bus.write("Homing_Offset", motor, offset)

    def _step_range_recording(self):
        """Record range of motion as the user moves all joints."""
        logger.info("Starting range recording step")

        # Initialize range tracking with retry and validation
        self._start_positions = {}
        for attempt in range(5):  # Try multiple times to get valid initial positions
            try:
                positions = self.device.bus.sync_read("Present_Position", normalize=False)
                # Validate initial positions
                valid_positions = {}
                for motor, pos in positions.items():
                    if pos > 0 and pos < 5000:  # Valid range
                        valid_positions[motor] = pos

                if len(valid_positions) == len(positions):  # All positions are valid
                    self._start_positions = valid_positions
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: Got invalid initial positions, retrying...")
                    time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to read initial positions: {e}")
                time.sleep(0.1)

        if not self._start_positions:
            raise RuntimeError("Could not get valid initial positions after multiple attempts")

        logger.info(f"Starting positions for range recording: {self._start_positions}")

        self._mins = self._start_positions.copy()
        self._maxes = self._start_positions.copy()
        logger.info(f"Initialized mins: {self._mins}")
        logger.info(f"Initialized maxes: {self._maxes}")

        self._update_status(
            status="recording",
            step=1,
            message="Move every joint EXCEPT the wrist roll through its FULL range of motion - from minimum to maximum. Leave the wrist roll near the middle: it rotates continuously and its range is set automatically.",
            recorded_ranges={
                motor: (
                    {"min": FULL_TURN_RANGE[0], "max": FULL_TURN_RANGE[1], "current": pos}
                    if motor in FULL_TURN_MOTORS
                    else {"min": pos, "max": pos, "current": pos}
                )
                for motor, pos in self._start_positions.items()
            },
        )

        self._recording_active = True
        prev_positions: dict[str, int] = dict(self._start_positions)

        # Record positions until user completes step
        while not self._step_complete.is_set() and not self.stop_calibration:
            try:
                # Try reading positions with retry on port contention
                positions = None
                for attempt in range(3):  # Try up to 3 times
                    try:
                        positions = self.device.bus.sync_read("Present_Position", normalize=False)
                        break  # Success, exit retry loop
                    except Exception as read_error:
                        if "Port is in use" in str(read_error) and attempt < 2:
                            time.sleep(0.01)  # Short delay before retry
                            continue
                        else:
                            raise read_error  # Re-raise if not port contention or final attempt

                if positions:
                    # Validate the readings - filter out invalid/zero values
                    valid_positions = {}
                    for motor, pos in positions.items():
                        # Filter out clearly invalid readings (0, negative, or extreme values)
                        if pos > 0 and pos < 5000:  # Reasonable range for motor positions
                            valid_positions[motor] = pos
                        else:
                            logger.debug(f"Filtered invalid position for {motor}: {pos}")

                    # Only update if we have valid readings
                    if valid_positions:
                        for motor, pos in valid_positions.items():
                            # Full-turn joints legitimately cross the encoder
                            # wrap when rolled — no discontinuity to detect.
                            if (
                                motor not in FULL_TURN_MOTORS
                                and motor in prev_positions
                                and abs(pos - prev_positions[motor]) > 2000
                            ):
                                raise CalibrationDiscontinuityError(
                                    "Motor discontinuity detected. Make sure to start "
                                    "the calibration with the robot in a middle position "
                                    "- all joints in the middle of their ranges."
                                )
                            prev_positions[motor] = pos
                            if motor in self._mins:
                                self._mins[motor] = min(self._mins[motor], pos)
                                self._maxes[motor] = max(self._maxes[motor], pos)

                time.sleep(0.05)  # 20Hz update rate
            except CalibrationDiscontinuityError:
                raise
            except Exception as e:
                if "Port is in use" in str(e):
                    logger.debug(f"Port busy during position read: {e}")
                else:
                    logger.warning(f"Error reading positions during recording: {e}")
                # Increase sleep time on error to reduce port contention
                time.sleep(0.2)

        if self.stop_calibration:
            logger.info("Range recording step cancelled due to stop request")
            return

        # Log the final recorded ranges for debugging
        logger.info("Final recorded ranges:")
        for motor in self._mins:
            logger.info(
                f"  {motor}: min={self._mins[motor]}, max={self._maxes[motor]}, range={self._maxes[motor] - self._mins[motor]}"
            )

        # Validate ranges. Full-turn joints are exempt: their recorded sweep is
        # discarded for the forced 0-4095 range, and NOT moving them is the
        # documented procedure.
        same_min_max = [
            motor
            for motor in self._mins
            if motor not in FULL_TURN_MOTORS and self._mins[motor] == self._maxes[motor]
        ]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values: {same_min_max}")

        # Check for insufficient range movement (less than 100 motor steps)
        insufficient_range = []
        for motor in self._mins:
            if motor in FULL_TURN_MOTORS:
                continue
            range_diff = self._maxes[motor] - self._mins[motor]
            if range_diff < 100:  # Less than 100 motor steps seems insufficient
                insufficient_range.append(f"{motor}: {range_diff}")

        if insufficient_range:
            logger.warning(
                f"Some motors may not have been moved through sufficient range: {insufficient_range}"
            )
            logger.warning("Consider moving all joints through their full range of motion during calibration")

        self._step_complete.clear()
        logger.info("Range recording step completed")

    def _complete_calibration(self):
        """Complete the calibration and save results"""
        logger.info("Completing calibration...")

        # Centering guard: fail before anything is written if the recorded
        # ranges show the arm didn't start from the middle pose (see
        # find_off_center_joints). The worker's error path cleans up, so no
        # half-written calibration file is left behind.
        off_center = find_off_center_joints(
            {motor: (self._mins[motor], self._maxes[motor]) for motor in self._mins}
        )
        if off_center:
            raise CalibrationCenteringError(
                f"Start pose wasn't the middle position: {', '.join(off_center)}. "
                "Re-run calibration starting from the middle pose."
            )

        # Log motor information for debugging
        logger.info("Motor configuration:")
        for motor, m in self.device.bus.motors.items():
            logger.info(f"  {motor}: ID={m.id}, Model={m.model}")

        # Create calibration dict. Full-turn joints get the forced 0-4095 range
        # (matching upstream lerobot), not whatever sliver was swept.
        ranges = final_motor_ranges(self._mins, self._maxes)
        calibration = {}
        for motor, m in self.device.bus.motors.items():
            range_min, range_max = ranges[motor]
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=self._homing_offsets[motor],
                range_min=range_min,
                range_max=range_max,
            )
            logger.info(
                f"Calibration for {motor}: "
                f"ID={m.id}, "
                f"homing_offset={self._homing_offsets[motor]}, "
                f"range_min={range_min}, "
                f"range_max={range_max}"
            )

        # Write and save calibration
        self.device.calibration = calibration
        self.device.bus.write_calibration(calibration)
        self.device._save_calibration()

        logger.info(f"Calibration saved to {self.device.calibration_fpath}")

        # EEPROM and the on-disk file now agree — nothing left for
        # _cleanup_device to roll back, regardless of what happens below.
        self._calibration_committed = True

        # Robot-record write-back: if this calibration was launched from a tile,
        # update the robot's port + config field for the side that was just calibrated.
        request = self._current_request
        if request is not None and request.robot_name:
            # Store the config as a STEM (no .json) — that's the canonical
            # user-facing name and the id lerobot uses; the extension is only the
            # on-disk filename. (Records used to store "<name>.json"; reads now
            # normalize old ones, so this stays consistent.)
            config_stem = (
                request.config_file[:-5] if request.config_file.endswith(".json") else request.config_file
            )
            # Pick the record fields for this side AND arm. For a bimanual robot
            # the right arm writes the right_* fields; "left" is also the single
            # robot's only pair.
            is_right = request.arm == "right"
            if request.device_type == "teleop":
                port_field = "right_leader_port" if is_right else "leader_port"
                config_field = "right_leader_config" if is_right else "leader_config"
            else:
                port_field = "right_follower_port" if is_right else "follower_port"
                config_field = "right_follower_config" if is_right else "follower_config"
            patch = {port_field: request.port, config_field: config_stem}
            try:
                save_robot_record(request.robot_name, patch, allow_create=False)
            except Exception as e:
                logger.warning(f"Robot-record write-back failed for {request.robot_name}: {e}")

    def _cleanup_and_finish(self, message: str, status: str = "completed"):
        """Clean up and finish calibration"""
        problems = self._cleanup_device()
        self._recording_active = False
        self._update_status(calibration_active=False, status=status, message=message)
        token = self._hardware_lease_token
        if token is not None and hardware_lease_registry.is_token_current(token):
            if problems:
                hardware_lease_registry.mark_unresolved(
                    token,
                    " ".join(problems),
                    {
                        "safe": False,
                        "device_closed": not any("disconnect" in problem.lower() for problem in problems),
                        "torque_off": None,
                        "evidence": " ".join(problems),
                    },
                )
            else:
                hardware_lease_registry.release(
                    token,
                    safe_hardware_receipt(
                        "calibration device closed; torque was disabled for the manual procedure"
                    ),
                )
                self._hardware_lease_token = None
        # Final release: every terminal path (completed / cancelled / error /
        # forced stop) funnels through here, after the device cleanup ran.
        notify_session_changed("calibration", False, phase=status)

    def _cleanup_device(self) -> list[str]:
        """Clean up device connection"""
        problems: list[str] = []
        # Serializes against a concurrent forced cleanup from
        # stop_calibration_process (see its comment, and _cleanup_lock's
        # docstring in __init__): only the first caller does real work, a
        # second concurrent call finds self.device already None below.
        with self._cleanup_lock:
            try:
                if self.device:
                    # A run that never committed (cancelled, or errored before
                    # _complete_calibration saved) may have already pushed new
                    # Homing_Offset, Min_Position_Limit, and Max_Position_Limit
                    # values to the servo in _step_homing (reset_calibration()
                    # touches all three). Left as-is, the servo would permanently
                    # diverge from the last-saved calibration file — the next
                    # session's arm-identity check would then either hard-refuse
                    # to start or, worse, silently decode positions against a
                    # stale offset. Restore the pre-calibration baseline so a
                    # cancelled run leaves the physical arm exactly as it found it.
                    #
                    # Each motor's write is its own try/except: one motor's write
                    # failing must not abort the writes still queued for the
                    # remaining motors, which would otherwise leave the arm
                    # partially rolled back — some motors restored, others still
                    # diverged — which is worse than leaving all of them diverged,
                    # since it is much harder to notice and diagnose.
                    if not self._calibration_committed and self._original_homing_offsets:
                        logger.info(
                            "Calibration did not complete — restoring pre-calibration "
                            f"homing offsets: {self._original_homing_offsets}, "
                            f"min position limits: {self._original_min_position_limits}, "
                            f"max position limits: {self._original_max_position_limits}"
                        )
                        for register, baseline in (
                            ("Homing_Offset", self._original_homing_offsets),
                            ("Min_Position_Limit", self._original_min_position_limits),
                            ("Max_Position_Limit", self._original_max_position_limits),
                        ):
                            for motor, value in baseline.items():
                                try:
                                    self.device.bus.write(register, motor, value)
                                except Exception as e:
                                    problems.append(f"Failed to restore {register} for {motor}: {e}")
                                    logger.error(
                                        f"Failed to restore {register}={value} for {motor} "
                                        f"after cancelled calibration: {e}"
                                    )

                    logger.info("Disconnecting device...")
                    self.device.disconnect()
                    self.device = None
            except Exception as e:
                problems.append(f"Failed to disconnect calibration device: {e}")
                logger.error(f"Error disconnecting device: {e}")
            return problems


# Global calibration manager instance
calibration_manager = CalibrationManager()


def calibration_is_active() -> bool:
    """True while ANY calibration session owns a bus.

    Reads the singletons' real state (no separately-tracked boolean) so other
    feature modules' reciprocal mutex checks (see CLAUDE.md) can't drift from
    the managers' own status.

    Covers both calibration flows: the SO-101's step-by-step range sweep in
    this module, and the Maker arm's zero-pose flow in ``zero_calibrate``.
    They are separate managers because the procedures share nothing, but from
    the mutual-exclusion standpoint they are one fact — "a calibration owns
    this bus" — so every existing reciprocal check gets the Maker flow for
    free, with no new ``robot.busy.*`` discriminant to register.
    """
    from .zero_calibrate import zero_calibration_is_active

    return calibration_manager.status.calibration_active or zero_calibration_is_active()
