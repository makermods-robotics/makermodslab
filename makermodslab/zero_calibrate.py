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
"""Zero-pose calibration: the Maker arm's only calibration flow.

The SO-101 needs a range sweep — its Feetech servos know nothing about where
the joint stops are, so ``calibrate.py`` walks the user through moving every
joint end to end and records what it sees. The Maker arm needs nothing of the
kind: its joint limits are fixed constants measured once against the arm's
mechanical stops (``MakerFollowerConfig.joint_limits``, and the Star 102
leader's ``joint_ranges`` copied from them). The only thing calibration has to
establish is **where zero is**.

So the whole procedure is: torque off, the user poses the arm by hand, we tell
the motors "this is zero", and we write out a calibration file whose ranges
come from the config. One step, no sweep, no driving.

This is a web reimplementation of lerobot's ``MakerFollower.calibrate()`` and
``RebotArm102Leader.calibrate()``, which are the same procedure but block on
``input()`` — unusable from a request thread. The waiting is a
``threading.Event`` the ``/complete-calibration-step`` route sets instead.

Deliberately NOT a new session kind. From the mutual-exclusion standpoint the
fact that matters is "a calibration owns this bus", which is what
``calibrate.calibration_is_active()`` already answers — it ORs this manager's
state in, so every existing reciprocal check keeps working without a new
``robot.busy.*`` discriminant, a new ``STARTABLE_KINDS`` entry, or a new
options schema. The sessions surface routes kind ``calibration`` here or to
``calibrate.py`` based on the robot record's ``arm_type``.

The two device families differ in exactly one place, ``_set_zero``:

* the Maker **follower** speaks CAN, and one whole-bus ``set_zero_position()``
  zeroes every RobStride motor at once;
* the Star 102 **leader** speaks FashionStar UART, and each servo has to be
  unlocked and given ``set_origin_point`` individually.

Everything either side of that is shared.
"""

import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

from lerobot.motors import MotorCalibration
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.utils import init_logging

from .api_errors import ErrorCode
from .session_events import notify_session_changed
from .utils.config import calibration_dir_for_device, save_robot_record

logger = logging.getLogger(__name__)

# Settle time after unlocking a FashionStar servo before writing its origin
# point. Mirrors lerobot's own `_SETTLE_SEC` in rebot_102_leader.py — the servo
# needs a moment between the unlock and the write or the origin lands on a
# stale reading.
_SETTLE_SEC = 0.01

# How long the user gets to pose the arm before the run gives up and releases
# the bus. Generous, because posing a 7-DOF arm by hand is not quick and an
# abandoned run costs nothing but an open port — but bounded, so a forgotten
# browser tab cannot hold the bus forever.
_POSE_TIMEOUT_S = 15 * 60.0


def zero_pose_instructions(arm_type: object) -> str:
    """The physical zero pose to ask the user for, per CAN family.

    The two poses are OPPOSITES on the gripper (Maker: fully open; Metal:
    closed), so showing one family's text to the other zeroes the gripper at
    the wrong end of its travel. Wording mirrors each family's own
    ``calibrate()`` prompt in lerobot.
    """
    if arm_type == "metal":
        return (
            "Move the arm by hand to its ZERO POSE — standing upright, all "
            "joints at 0 degrees, gripper closed — then confirm."
        )
    return (
        "Move the arm by hand to its ZERO POSE — folded against the base, "
        "gripper fully open — then confirm."
    )


@dataclass
class ZeroCalibrationStatus:
    """Status of a zero-pose calibration run.

    Field-for-field compatible with ``calibrate.CalibrationStatus`` where the
    two overlap, so ``/calibration-status`` can serve either flow and the
    frontend can read one shape. ``recorded_ranges`` is always None here (there
    is no sweep to report) and ``total_steps`` is always 1.
    """

    calibration_active: bool = False
    # idle | connecting | awaiting_zero | saving | completed | error | stopping
    status: str = "idle"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0
    total_steps: int = 1
    current_positions: dict[str, float] | None = None
    recorded_ranges: dict[str, dict[str, float]] | None = None
    # The zero flow's own field: True once the arm is connected, torque is off
    # and we are waiting for the user to confirm the pose. The frontend shows
    # its "move the arm to its zero pose" panel on this.
    awaiting_pose: bool = False


@dataclass
class ZeroCalibrationRequest:
    """Request parameters for starting a zero-pose calibration.

    Same shape as ``calibrate.CalibrationRequest`` so ``sessions.py`` can build
    one request model and dispatch on arm type alone.
    """

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None
    overwrite: bool = False
    arm: Literal["left", "right"] = "left"
    # Which CAN family this run calibrates. Decides the device configs built
    # in _connect, the zero-pose text shown to the user, and the library the
    # name-collision check reads. Defaults to maker so a request built before
    # the Metal arm existed is unchanged.
    arm_type: Literal["maker", "metal"] = "maker"


class ZeroCalibrationManager:
    """Owns the one live zero-pose calibration run."""

    def __init__(self):
        self.status = ZeroCalibrationStatus()
        self.device = None
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        # RLock for the same reason calibrate.CalibrationManager uses one: the
        # start path holds it across a critical section that itself calls
        # _update_status.
        self._status_lock = threading.RLock()
        self._pose_confirmed = threading.Event()
        self._current_request: ZeroCalibrationRequest | None = None
        self._cleanup_lock = threading.Lock()
        init_logging()

    # -- status ------------------------------------------------------------

    def get_status(self) -> ZeroCalibrationStatus:
        """Current status, with live joint positions while awaiting the pose.

        Reading positions is what makes the frontend's live readout work while
        the user moves the arm. Torque is off throughout, so this is a pure
        read — it cannot move the arm, and a failed read is logged and skipped
        rather than aborting the run the user is in the middle of.
        """
        with self._status_lock:
            if self.status.status == "awaiting_zero" and self.device is not None:
                try:
                    self.status.current_positions = self._read_positions()
                except Exception as e:
                    logger.debug(f"Position read during zero calibration failed: {e}")
            return self.status

    def _update_status(self, **kwargs):
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    # -- device helpers ----------------------------------------------------

    def _is_follower(self) -> bool:
        return self._current_request is not None and self._current_request.device_type == "robot"

    def _read_positions(self) -> dict[str, float]:
        """Current joint angles in degrees, keyed by motor name.

        The Maker follower and the Star leader expose a private raw-position
        reader with the same name and the same contract (``{motor: degrees}``),
        which is what lerobot's own ``calibrate()`` logs before zeroing. The
        Metal follower has no such reader, so fall back to the bus's plain
        ``sync_read("Present_Position")`` — same units, same keys. Either way
        this is a read on a torque-off bus; a failure is the caller's to skip.
        """
        reader = getattr(self.device, "_read_raw_positions", None)
        if reader is not None:
            return {m: float(v) for m, v in reader().items()}
        bus = getattr(self.device, "bus", None)
        sync_read = getattr(bus, "sync_read", None)
        if sync_read is None:
            return {}
        return {m: float(v) for m, v in sync_read("Present_Position").items()}

    def _set_zero(self) -> None:
        """Tell the motors that where they are now is zero."""
        bus = self.device.bus
        if self._is_follower():
            # CAN (RobStride and Damiao alike): one broadcast zeroes every
            # motor on the bus.
            bus.set_zero_position()
            # Mirror what MakerFollower.calibrate() resets alongside the zero,
            # so the freshly zeroed arm is not still carrying the previous
            # zero's multi-turn bookkeeping (which would make send_action
            # refuse with a stale-zero error).
            for attr, value in (
                ("_turn_offset", dict.fromkeys(getattr(self.device, "_joint_motor_names", []), 0.0)),
                ("_stale_zero", {}),
                ("_last_positions", {}),
            ):
                if hasattr(self.device, attr):
                    setattr(self.device, attr, value)
            return

        # FashionStar UART: no broadcast — unlock and origin each servo in turn.
        for motor_id in self._current_request_joint_ids().values():
            bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
            bus.set_origin_point(motor_id)

    def _current_request_joint_ids(self) -> dict[str, int]:
        return dict(self.device.config.joint_ids)

    def _build_calibration(self) -> dict[str, MotorCalibration]:
        """The calibration file's contents: fixed ranges from the config.

        ``homing_offset`` is 0 for every joint and that is correct, not a
        placeholder — the zero now lives INSIDE the motor (RobStride's zero
        position / FashionStar's origin point), so there is no software offset
        left to apply on top. It is also why the Feetech EEPROM fingerprint in
        ``arm_identity.py`` can say nothing about a Maker arm: every Maker
        calibration file has the same all-zero offsets.
        """
        config = self.device.config
        if self._is_follower():
            ids = config.motor_can_ids
            ranges = config.joint_limits
            default = (-360.0, 360.0)
        else:
            ids = config.joint_ids
            ranges = {m: tuple(v) for m, v in config.joint_ranges.items()}
            default = (-360.0, 360.0)

        calibration: dict[str, MotorCalibration] = {}
        for motor_name, motor_id in ids.items():
            # The two CAN followers disagree about the id field's shape:
            # Maker motor_can_ids are plain ints, Metal's are (send, recv)
            # tuples. MotorCalibration.id is an int, and lerobot's own
            # MetalFollower.calibrate() stores the SEND id.
            if isinstance(motor_id, tuple):
                motor_id = motor_id[0]
            range_min, range_max = ranges.get(motor_name, default)
            calibration[motor_name] = MotorCalibration(
                id=motor_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )
        return calibration

    # -- lifecycle ---------------------------------------------------------

    def start(self, request: ZeroCalibrationRequest) -> dict[str, Any]:
        """Claim the bus and spawn the worker. Same response shape as
        ``CalibrationManager.start_calibration``."""
        try:
            # Check-and-claim atomically, for the same reason the SO-101
            # manager does: two concurrent callers must not both spawn a
            # worker against the same arm.
            with self._status_lock:
                if self.status.calibration_active:
                    return {
                        "success": False,
                        "message": "Calibration already active",
                        "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
                    }

                busy = _other_feature_busy()
                if busy is not None:
                    return busy

                # Refuse to silently clobber an existing calibration of the
                # same name — same contract as the SO-101 flow.
                config_dir = calibration_dir_for_device(request.device_type, request.arm_type)
                if config_dir is not None and not request.overwrite:
                    stem = request.config_file.removesuffix(".json")
                    if os.path.exists(os.path.join(config_dir, f"{stem}.json")):
                        return {
                            "success": False,
                            "code": "name_taken",
                            "message": (
                                f"A calibration named '{stem}' already exists. "
                                "Overwrite it or choose a different name."
                            ),
                        }

                self._update_status(
                    calibration_active=True,
                    status="connecting",
                    device_type=request.device_type,
                    error=None,
                    message="Connecting to the arm...",
                    step=0,
                    awaiting_pose=False,
                    current_positions=None,
                    recorded_ranges=None,
                )
                self._current_request = request
                self.stop_requested = False
                self._pose_confirmed.clear()
                self.thread = threading.Thread(target=self._worker, args=(request,), daemon=True)
                self.thread.start()

            notify_session_changed("calibration", True, phase="connecting")
            return {"success": True, "message": "Zero calibration started"}

        except Exception as e:
            logger.error(f"Error starting zero calibration: {e}")
            self._update_status(
                calibration_active=False,
                status="error",
                error=str(e),
                message="Failed to start calibration",
            )
            notify_session_changed("calibration", False, phase="error")
            return {"success": False, "message": str(e)}

    def complete_step(self) -> dict[str, Any]:
        """The user confirmed the arm is in its zero pose."""
        if not self.status.calibration_active:
            return {"success": False, "message": "No calibration active"}
        if self.status.status != "awaiting_zero":
            return {
                "success": False,
                "message": f"Cannot complete step in status: {self.status.status}",
            }
        self._pose_confirmed.set()
        return {"success": True, "message": "Zero pose confirmed"}

    def stop(self) -> dict[str, Any]:
        """Cancel the run and release the bus."""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping zero calibration...")
            self.stop_requested = True
            # Unblock the worker's wait so it can notice stop_requested and
            # exit through its own cleanup rather than sitting out the timeout.
            self._pose_confirmed.set()
            self._update_status(status="stopping", message="Stopping calibration...")

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            if self.thread and self.thread.is_alive():
                logger.warning("Zero calibration thread did not finish within timeout, forcing cleanup")

            self._finish("Calibration stopped", status="idle")
            return {"success": True, "message": "Calibration stopped"}
        except Exception as e:
            logger.error(f"Error stopping zero calibration: {e}")
            self._finish("Calibration stopped with error", status="error")
            return {"success": False, "message": str(e)}

    # -- worker ------------------------------------------------------------

    def _worker(self, request: ZeroCalibrationRequest):
        try:
            logger.info(f"Zero calibration worker starting for {request.device_type} on {request.port}")
            self._connect(request)

            if self.stop_requested:
                self._finish("Calibration cancelled", status="idle")
                return

            self._update_status(
                status="awaiting_zero",
                awaiting_pose=True,
                step=1,
                message=zero_pose_instructions(request.arm_type),
            )
            notify_session_changed("calibration", True, phase="awaiting_zero")

            # Torque is off and stays off for this entire wait: the user is
            # physically moving the arm.
            if not self._pose_confirmed.wait(timeout=_POSE_TIMEOUT_S):
                raise TimeoutError(
                    "Timed out waiting for the zero pose to be confirmed. "
                    "Start the calibration again when you're ready to pose the arm."
                )
            if self.stop_requested:
                self._finish("Calibration cancelled", status="idle")
                return

            self._update_status(
                status="saving", awaiting_pose=False, message="Setting zero and saving calibration..."
            )
            self._commit()

            logger.info("Zero calibration completed successfully")
            self._finish("Calibration completed successfully", status="completed")

        except Exception as e:
            logger.error(f"Zero calibration error: {e}")
            logger.error(traceback.format_exc())
            self._update_status(error=str(e))
            self._finish(f"Calibration failed: {e}", status="error")
        finally:
            if self.status.calibration_active:
                logger.warning("Zero calibration worker ending while still active - forcing cleanup")
                self._finish("Calibration stopped", status="idle")

    def _connect(self, request: ZeroCalibrationRequest):
        """Open the bus with torque OFF, ready for the user to pose the arm."""
        from .utils.robot_factory import (  # local import: avoids a cycle at module load
            maker_follower_config,
            maker_leader_config,
            metal_follower_config,
            metal_leader_config,
        )

        is_metal = request.arm_type == "metal"
        if request.device_type == "robot":
            builder = metal_follower_config if is_metal else maker_follower_config
            config = builder(request.port, request.config_file)
            self.device = make_robot_from_config(config)
            label = "Metal" if is_metal else "Maker"
            self._update_status(message=f"Connecting to the {label} follower arm...")
            # NOT device.connect(): both followers' connect() finishes by
            # calling enable_torque(), which would lock the arm rigid exactly
            # when the user needs to move it by hand. Open the bus directly
            # and disable torque, which is what lerobot's own calibrate() does
            # internally. The disable is not optional for Metal even here:
            # the Damiao HANDSHAKE inside bus.connect() is itself the enable
            # command, so the arm comes up energized and this write is what
            # frees it for the user's hands.
            self.device.bus.connect()
            self.device.bus.disable_torque()
        else:
            builder = metal_leader_config if is_metal else maker_leader_config
            config = builder(request.port, request.config_file)
            self.device = make_teleoperator_from_config(config)
            self._update_status(message="Connecting to the Star Arm 102 leader arm...")
            # The leader's bus is constructed inside connect(), so there is no
            # bus-only path — but there is nothing to disable either: its
            # joints hold encoders and no motors. connect(calibrate=False)
            # leaves it unlocked and back-drivable, which is the state we want.
            self.device.connect(calibrate=False)

    def _commit(self):
        """Set zero on the motors, then write the calibration file."""
        pre_zero = {}
        try:
            pre_zero = self._read_positions()
        except Exception as e:
            logger.warning(f"Could not read pre-zero positions: {e}")
        for motor, value in pre_zero.items():
            # Logged so an offset against the PREVIOUS zero stays recoverable
            # from the logs if this one turns out to have been taken in the
            # wrong pose — same reason lerobot's own calibrate() logs them.
            logger.info(f"Pre-zero position of {motor}: {value:.2f} deg")

        self._set_zero()
        logger.info("Arm zero position set.")

        calibration = self._build_calibration()
        self.device.calibration = calibration
        # The follower's CAN bus can hold the calibration; the leader's
        # FashionStar handle has no write_calibration at all (its zero lives in
        # the servos via set_origin_point above), so the file is the whole
        # record on that side.
        writer = getattr(getattr(self.device, "bus", None), "write_calibration", None)
        if writer is not None:
            writer(calibration)
        self.device._save_calibration()
        logger.info(f"Calibration saved to {self.device.calibration_fpath}")

        self._write_back_record()

    def _write_back_record(self):
        """Point the robot record's slot at this calibration, as calibrate.py does."""
        request = self._current_request
        if request is None or not request.robot_name:
            return
        config_stem = request.config_file.removesuffix(".json")
        is_right = request.arm == "right"
        if request.device_type == "teleop":
            port_field = "right_leader_port" if is_right else "leader_port"
            config_field = "right_leader_config" if is_right else "leader_config"
        else:
            port_field = "right_follower_port" if is_right else "follower_port"
            config_field = "right_follower_config" if is_right else "follower_config"
        try:
            save_robot_record(
                request.robot_name,
                {port_field: request.port, config_field: config_stem},
                allow_create=False,
            )
        except Exception as e:
            logger.warning(f"Robot-record write-back failed for {request.robot_name}: {e}")

    # -- teardown ----------------------------------------------------------

    def _finish(self, message: str, status: str = "completed"):
        self._release_device()
        with self._status_lock:
            was_active = self.status.calibration_active
            self._update_status(
                calibration_active=False,
                status=status,
                awaiting_pose=False,
                message=message,
                current_positions=None,
            )
            self._current_request = None
        if was_active:
            notify_session_changed("calibration", False, phase=status)

    def _release_device(self):
        """Disconnect the arm, never raising.

        Guarded by ``_cleanup_lock`` for the same reason the SO-101 manager
        guards its own: ``stop()``'s join can time out while the worker is
        still mid-bus, and the request thread then forces a release that must
        not double-run against the worker's own eventual one.
        """
        with self._cleanup_lock:
            device, self.device = self.device, None
            if device is None:
                return
            try:
                device.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting after zero calibration: {e}")


def _other_feature_busy() -> dict[str, Any] | None:
    """The mutual-exclusion gate, mirroring calibrate.py's list exactly.

    Lazy imports for the same reason every other feature module uses them:
    these modules import each other, so a top-level import would close a cycle
    at load time.
    """
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
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
    if _calibrate.calibration_manager.status.calibration_active:
        return {
            "success": False,
            "message": "Calibration is currently active. Stop it first.",
            "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
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
    return None


# Global manager instance, mirroring calibrate.calibration_manager.
zero_calibration_manager = ZeroCalibrationManager()


def zero_calibration_is_active() -> bool:
    """True while a zero-pose calibration owns a bus.

    Read by ``calibrate.calibration_is_active()``, which is what every other
    feature's reciprocal mutex check already calls — so this flow participates
    in mutual exclusion without any of them needing to know it exists.
    """
    return zero_calibration_manager.status.calibration_active
