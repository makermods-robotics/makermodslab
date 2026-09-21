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
"""The generic step-wizard calibration manager, for every ``steps`` family.

A family whose ``calibration_kind`` is ``"steps"`` (the CAN arms — Maker and
Metal — whose only calibration is a zero pose set by hand with torque off;
and any extension family that declares the kind) runs its OWN procedure,
``ArmFamily.calibrate``, on this manager's worker thread. The family drives
the wizard through a ``CalibrationUI``: ``step(text, ...)`` publishes a step
and BLOCKS until the user confirms it from the browser
(``/complete-calibration-step``), ``message(text)`` shows a transient status
while the family works. The procedure's text, images and hardware writes all
live in the family (``arms/can_common.py`` for the zero pose); nothing here
knows what a step means.

The manager owns everything a family must not: the mutual-exclusion gate,
the calibration-name collision check, connecting through
``family.open_for_calibration`` (torque OFF), writing the returned
calibration exactly as the zero flow always wrote it (``device.calibration``,
the bus's ``write_calibration`` when it has one, ``_save_calibration``), the
robot-record write-back, the session-events phases (``connecting``,
``awaiting_step`` on EVERY step, and the finish), the per-step timeout, and
— on EVERY exit, including a stop and an error — releasing the device: for a
follower, ``family.release_torque`` first (belt and braces; the bus was
opened torque-off but a Metal handshake energizes), then disconnect. The
leader is human-held and never energized here, so it is only disconnected.

Deliberately NOT a new session kind. From the mutual-exclusion standpoint the
fact that matters is "a calibration owns this bus", which is what
``calibrate.calibration_is_active()`` already answers — it ORs this manager's
state in, so every existing reciprocal check keeps working without a new
``robot.busy.*`` discriminant, a new ``STARTABLE_KINDS`` entry, or a new
options schema. The sessions surface routes kind ``calibration`` here or to
``calibrate.py`` by the family's ``calibration_kind``.
"""

import logging
import os
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Literal

from lerobot.utils.utils import init_logging

from .api_errors import ApiError, ErrorCode
from .arm_capabilities import require_known_arm_type
from .arms.base import ArmFamily, CalibrationAborted, leader_kwargs
from .session_events import notify_session_changed
from .utils.config import calibration_dir_for_device, save_robot_record

logger = logging.getLogger(__name__)

# How long the user gets to confirm ONE step before the run gives up and
# releases the bus. Generous, because posing a 7-DOF arm by hand is not quick
# and an abandoned run costs nothing but an open port — but bounded, so a
# forgotten browser tab cannot hold the bus forever.
_POSE_TIMEOUT_S = 15 * 60.0


@dataclass
class StepCalibrationStatus:
    """Status of a step-wizard calibration run.

    Field-for-field compatible with ``calibrate.CalibrationStatus`` where the
    two overlap, so ``/calibration-status`` can serve either flow and the
    frontend can read one shape. ``recorded_ranges`` is always None here
    (there is no sweep to report). ``total_steps`` is unknown up front —
    the family decides how many steps it publishes — so while running it
    equals ``step`` and the wizard shows "Step N", never "N of M".
    """

    calibration_active: bool = False
    # idle | connecting | awaiting_step | saving | completed | error | stopping
    status: str = "idle"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0
    total_steps: int = 1
    current_positions: dict[str, float] | None = None
    recorded_ranges: dict[str, dict[str, float]] | None = None
    # The current step's image (a served URL the family named), if any.
    image_url: str | None = None
    # True while the current step asked for the live joint readout; positions
    # are read (torque off) only then.
    live_positions: bool = False


@dataclass
class StepCalibrationRequest:
    """Request parameters for starting a step-wizard calibration.

    Same shape as ``calibrate.CalibrationRequest`` so ``sessions.py`` can build
    one request model and dispatch on the family's calibration kind alone.
    """

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None
    overwrite: bool = False
    arm: Literal["left", "right"] = "left"
    # Which family this run calibrates: it decides the device configs, the
    # procedure, and the library the name-collision check reads. Any
    # registered id (an extension's family rides this same request);
    # start() refuses an unknown one and a family whose kind is not
    # "steps". Defaults to maker so a request built before the Metal arm
    # existed is unchanged.
    arm_type: str = "maker"
    # Which of the family's leaders a "teleop" run calibrates (the record's
    # leader_kind; None = the family's default). Passed to the family only
    # when it offers a choice (arms.base.leader_kwargs).
    leader_kind: str | None = None


class _ManagerUI:
    """The ``CalibrationUI`` handed to the family: every call lands on the
    manager's status and blocks (``step``) on the confirmation event."""

    def __init__(self, manager: "StepCalibrationManager") -> None:
        self._manager = manager

    def step(self, text: str, *, image_url: str | None = None, live_positions: bool = False) -> None:
        manager = self._manager
        if manager.stop_requested:
            raise CalibrationAborted("Calibration stopped")
        # clear() and the publish are one critical section, and so are
        # complete_step's check and set(): a confirm that lands between them
        # would otherwise carry over to the step published next.
        with manager._status_lock:
            manager._step_confirmed.clear()
            step = manager.status.step + 1
            manager._update_status(
                status="awaiting_step",
                step=step,
                total_steps=step,
                message=text,
                image_url=image_url,
                live_positions=live_positions,
                current_positions=None,
            )
        # Every step is a phase the UI refetches on, not only the first.
        notify_session_changed("calibration", True, phase="awaiting_step")

        # Torque is off for this entire wait: the user is at the arm.
        if not manager._step_confirmed.wait(timeout=_POSE_TIMEOUT_S):
            raise CalibrationAborted(
                f"Timed out waiting for step {step} to be confirmed. "
                "Start the calibration again when you're ready."
            )
        if manager.stop_requested:
            raise CalibrationAborted("Calibration stopped")
        # Confirmed: the family is working until its next step (or its
        # return), so no input is expected and the Next button is not live.
        manager._update_status(status="saving", live_positions=False, current_positions=None)

    def message(self, text: str) -> None:
        self._manager._update_status(
            status="saving", message=text, live_positions=False, current_positions=None
        )


class StepCalibrationManager:
    """Owns the one live step-wizard calibration run."""

    def __init__(self):
        self.status = StepCalibrationStatus()
        self.device = None
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        # RLock for the same reason calibrate.CalibrationManager uses one: the
        # start path holds it across a critical section that itself calls
        # _update_status.
        self._status_lock = threading.RLock()
        self._step_confirmed = threading.Event()
        self._current_request: StepCalibrationRequest | None = None
        self._family: ArmFamily | None = None
        self._cleanup_lock = threading.Lock()
        init_logging()

    # -- status ------------------------------------------------------------

    def get_status(self) -> StepCalibrationStatus:
        """Current status, with live joint positions while a live-positions step is up.

        Reading positions is what makes the frontend's live readout work while
        the user moves the arm. Torque is off throughout, so this is a pure
        read — it cannot move the arm, and a failed read is logged and skipped
        rather than aborting the run the user is in the middle of.
        """
        with self._status_lock:
            if (
                self.status.status == "awaiting_step"
                and self.status.live_positions
                and self.device is not None
                and self._family is not None
            ):
                try:
                    self.status.current_positions = self._family.read_positions(self.device)
                except Exception as e:
                    logger.debug(f"Position read during step calibration failed: {e}")
            return self.status

    def _update_status(self, **kwargs):
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    # -- lifecycle ---------------------------------------------------------

    def start(self, request: StepCalibrationRequest) -> dict[str, Any]:
        """Claim the bus and spawn the worker. Same response shape as
        ``CalibrationManager.start_calibration`` — except the two argument
        refusals below, which RAISE (400) the way sessions'
        _build_auto_calibration_request does for the mirror case, so the
        app-wide handler renders the coded body."""
        from .arms import registry as arm_registry
        from .utils.config import normalize_arm_type

        # Outside the try: an ApiError must reach the caller, not be folded
        # into the generic "Failed to start" dict below.
        require_known_arm_type(request.arm_type)
        family = arm_registry.get(normalize_arm_type(request.arm_type))
        if family.calibration_kind != "steps":
            raise ApiError(
                status_code=400,
                detail=(
                    f"The {family.short_label} is calibrated by its family's '{family.calibration_kind}' "
                    "procedure, not a step wizard. Run that calibration instead."
                ),
                code=ErrorCode.ROBOT_NOT_READY,
            )
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
                config_dir = calibration_dir_for_device(
                    request.device_type, request.arm_type, request.leader_kind
                )
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
                    total_steps=1,
                    current_positions=None,
                    recorded_ranges=None,
                    image_url=None,
                    live_positions=False,
                )
                self._current_request = request
                self._family = family
                self.stop_requested = False
                self._step_confirmed.clear()
                self.thread = threading.Thread(target=self._worker, args=(request,), daemon=True)
                self.thread.start()

            notify_session_changed("calibration", True, phase="connecting")
            return {"success": True, "message": "Calibration started"}

        except Exception as e:
            logger.error(f"Error starting step calibration: {e}")
            self._update_status(
                calibration_active=False,
                status="error",
                error=str(e),
                message="Failed to start calibration",
            )
            notify_session_changed("calibration", False, phase="error")
            return {"success": False, "message": str(e)}

    def complete_step(self, step: int | None = None) -> dict[str, Any]:
        """The user confirmed the current step.

        ``step`` is the step number the client is confirming; when given it
        must be the one on screen. A duplicate click that reaches the server
        after the family has already moved on to the next step must not
        confirm THAT step — with a multi-step family that is a hardware write
        in a pose the user never confirmed. The check and the set() are one
        critical section with the worker's clear-and-publish for the same
        reason.
        """
        with self._status_lock:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}
            if self.status.status != "awaiting_step":
                return {
                    "success": False,
                    "message": f"Cannot complete step in status: {self.status.status}",
                }
            if step is not None and step != self.status.step:
                return {
                    "success": False,
                    "message": f"Step {step} is not the current step ({self.status.step}); nothing confirmed.",
                }
            if step is None and self.status.step > 1:
                # A bodiless confirm is the pre-wizard contract (one-step
                # families, older clients). Past step 1 it cannot be told from
                # a stale duplicate, so a multi-step run requires the step.
                return {
                    "success": False,
                    "message": (
                        f"This calibration is on step {self.status.step}; the confirm must name "
                        "the step it is for."
                    ),
                }
            self._step_confirmed.set()
        return {"success": True, "message": "Step confirmed"}

    def stop(self) -> dict[str, Any]:
        """Cancel the run and release the bus."""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping step calibration...")
            self.stop_requested = True
            # Unblock the worker's wait so it can notice stop_requested and
            # exit through its own cleanup rather than sitting out the timeout.
            self._step_confirmed.set()
            self._update_status(status="stopping", message="Stopping calibration...")

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            if self.thread and self.thread.is_alive():
                logger.warning("Step calibration thread did not finish within timeout, forcing cleanup")

            with self._status_lock:
                if not self.status.calibration_active:
                    # The worker finished on its own while we joined — either
                    # it honoured the stop (idle) or it was already past its
                    # last confirm and wrote the file (completed). Report
                    # THAT rather than overwriting it with "stopped": the
                    # record slot may just have been repointed.
                    return {"success": True, "message": self.status.message}

            self._finish("Calibration stopped", status="idle")
            return {"success": True, "message": "Calibration stopped"}
        except Exception as e:
            logger.error(f"Error stopping step calibration: {e}")
            self._finish("Calibration stopped with error", status="error")
            return {"success": False, "message": str(e)}

    # -- worker ------------------------------------------------------------

    def _worker(self, request: StepCalibrationRequest):
        family = self._family
        try:
            logger.info(f"Step calibration worker starting for {request.device_type} on {request.port}")
            side = "follower" if request.device_type == "robot" else "leader"
            self._update_status(message=f"Connecting to the {family.short_label} {side} arm...")
            self.device = family.open_for_calibration(
                request.device_type,
                request.port,
                request.config_file,
                **leader_kwargs(family, request.leader_kind),
            )

            if self.stop_requested:
                self._finish("Calibration cancelled", status="idle")
                return

            try:
                calibration = family.calibrate(self.device, request.device_type, _ManagerUI(self))
            except CalibrationAborted as abort:
                if self.stop_requested:
                    self._finish("Calibration cancelled", status="idle")
                    return
                # Not a user stop: the per-step timeout.
                raise TimeoutError(str(abort)) from abort
            # No stop check here, deliberately: the family has returned, so
            # its procedure ran to the end — a CAN motor already holds the
            # new zero (RobStride's set-zero and FashionStar's origin point
            # persist in the motor). Dropping the file now would leave the
            # hardware and the library disagreeing, which is worse than
            # honouring a stop a moment late. A stop that arrived during the
            # last step's wait was honoured inside ui.step.
            self._update_status(
                status="saving", message="Saving calibration...", live_positions=False, current_positions=None
            )
            self._commit(calibration)

            logger.info("Step calibration completed successfully")
            self._finish("Calibration completed successfully", status="completed")

        except Exception as e:
            logger.error(f"Step calibration error: {e}")
            logger.error(traceback.format_exc())
            self._update_status(error=str(e))
            self._finish(f"Calibration failed: {e}", status="error")
        finally:
            if self.status.calibration_active:
                logger.warning("Step calibration worker ending while still active - forcing cleanup")
                self._finish("Calibration stopped", status="idle")

    def _commit(self, calibration: dict[str, Any]):
        """Write the calibration the family returned, then the record slot."""
        self.device.calibration = calibration
        # The follower's CAN bus can hold the calibration; the leader's
        # FashionStar handle has no write_calibration at all (its zero lives in
        # the servos), so the file is the whole record on that side.
        writer = getattr(getattr(self.device, "bus", None), "write_calibration", None)
        if writer is not None:
            writer(calibration)
        self.device._save_calibration()
        logger.info(f"Calibration saved to {getattr(self.device, 'calibration_fpath', '?')}")

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
                message=message,
                current_positions=None,
                live_positions=False,
            )
            self._current_request = None
        if was_active:
            notify_session_changed("calibration", False, phase=status)

    def _release_device(self):
        """Release torque (follower only), then disconnect — never raising.

        The follower's bus was opened torque-off, but "stopped means
        de-energized" is checked, not assumed: ``family.release_torque`` runs
        before the disconnect on every exit path (a Metal handshake energizes,
        and a family's procedure may have enabled a motor). A Star leader is
        human-held and never energized here — a release would be a write to
        servos the family never enabled — so it is only disconnected; an
        ENERGIZED leader (the Metal leader, a Damiao arm) gets the follower's
        release. Guarded by ``_cleanup_lock`` for the same reason the
        SO-101 manager guards its own: ``stop()``'s join can time out while
        the worker is still mid-bus, and the request thread then forces a
        release that must not double-run against the worker's own eventual one.
        """
        with self._cleanup_lock:
            device, self.device = self.device, None
            if device is None:
                return
            family = self._family
            request = self._current_request
            # A follower always; a leader only when the family says the
            # selected one holds torque (the Metal leader: a Damiao arm whose
            # handshake energized it exactly as the follower's did).
            energized_leader = (
                self.status.device_type == "teleop"
                and family is not None
                and request is not None
                and family.leader_holds_torque(request.leader_kind)
            )
            if family is not None and (self.status.device_type == "robot" or energized_leader):
                side = "follower" if self.status.device_type == "robot" else "leader"
                try:
                    family.release_torque(device, f"{family.short_label} {side}")
                except Exception as e:
                    logger.warning(f"Error releasing torque after step calibration: {e}")
            try:
                device.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting after step calibration: {e}")


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
step_calibration_manager = StepCalibrationManager()


def step_calibration_is_active() -> bool:
    """True while a step-wizard calibration owns a bus.

    Read by ``calibrate.calibration_is_active()``, which is what every other
    feature's reciprocal mutex check already calls — so this flow participates
    in mutual exclusion without any of them needing to know it exists.
    """
    return step_calibration_manager.status.calibration_active
