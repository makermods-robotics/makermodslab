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
Auto-calibration: run the vendored Feetech auto-calibration script as a
subprocess (it drives the arm under torque to find each joint's range), stream
its logs, and on success copy the file to MakerMods Lab's expected path + write the
config back into the robot record.

The vendored script (makermodslab/vendor/feetech_autocal) always saves to
``.../calibration/robots/<robot_type>/<id>.json``. For a follower that IS MakerMods Lab's
follower dir; for a leader we copy it to ``.../teleoperators/so_leader``.
"""

import logging
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass

from pydantic import BaseModel

from lerobot.motors.feetech import FeetechMotorsBus

from .api_errors import ErrorCode
from .motor_power import torque_limit_from_percent
from .session_events import notify_session_changed
from .torque import force_disable_bus_torque
from .utils.config import (
    CALIBRATION_BASE_PATH_ROBOTS,
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    save_robot_record,
)
from .vendor.feetech_autocal.calibration_defaults import SO_FOLLOWER_MOTORS

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 1000
_SCRIPT_MODULE = "makermodslab.vendor.feetech_autocal.auto_calibrate_script"

# Stop escalation timings. The script's own SIGTERM cleanup (KeyboardInterrupt
# -> _graceful_stop -> safe_disable_all) first freezes the arm (~1s), then
# drives it back to its starting pose — progress-based, giving up on stall,
# with RETURN_TO_REST_BUDGET_S = 10s as an absolute ceiling — or holds the
# freeze (STOP_HOLD_S = 3s) before the 6-motor release + disconnect (~1-2s,
# more on a bus that's mid overload). Worst case ~15-16s, so 18s of grace
# before concluding the script is wedged (e.g. blocked in a C-level serial
# call, where the raised KeyboardInterrupt never materializes) and killing
# it. Keep this above the vendored script's summed stop budget
# (makermodslab/vendor/feetech_autocal/auto_calibrate_script.py) so a SIGKILL can
# never land on a healthy graceful stop mid-motion.
_STOP_GRACE_S = 18.0
_STOP_KILL_WAIT_S = 5.0
_READER_JOIN_S = 5.0
# Absolute ceiling for stop_and_wait(): the sum of every step in the escalation
# above, plus margin. Used by callers (server shutdown) that need the arm
# actually de-energized before they let the process exit, instead of just
# handing the sequence to its background thread and moving on.
_STOP_AND_WAIT_CEILING_S = _STOP_GRACE_S + _STOP_KILL_WAIT_S + _READER_JOIN_S + 2.0


class AutoCalibrationRequest(BaseModel):
    device_type: str  # "teleop" (leader) or "robot" (follower)
    port: str
    config_file: str
    robot_name: str | None = None
    arm: str = "left"  # "left" (also single) or "right"
    # Calibration drive torque, as a percentage of full power (the robot's
    # torque slider; clamped server-side to 10-100). Passed to the vendored
    # script as --torque-limit (percent × 10). None = the script's default.
    motor_power: int | None = None


class AutoCalibrationBatchArm(BaseModel):
    """One arm in a concurrent auto-calibration batch. Same fields as the
    single-arm request minus robot_name (batch-level, shared by every arm)."""

    device_type: str  # "teleop" (leader) or "robot" (follower)
    port: str
    config_file: str
    arm: str = "left"  # "left" (also single) or "right"


class AutoCalibrationBatchRequest(BaseModel):
    arms: list[AutoCalibrationBatchArm]
    robot_name: str | None = None
    overwrite: bool = False
    # Batch-level calibration drive torque (percent, 10-100), applied to every
    # arm in the batch. None = the vendored script's default.
    motor_power: int | None = None


@dataclass
class AutoCalibrationStatus:
    active: bool = False
    status: str = "idle"  # idle | running | stopping | completed | failed | stopped
    message: str = ""
    error: str | None = None


def _stem(name: str) -> str:
    return name[: -len(".json")] if name.endswith(".json") else name


def _subprocess_output_path(device_type: str, config_stem: str) -> str:
    """The file the vendored subprocess writes with --save.

    It always saves under ``.../calibration/robots/<robot_type>/<id>.json``:
    for a follower that IS MakerMods Lab's real library dir; for a leader it's a scratch
    location (robots/so_leader) that _finalize_success later copies into the real
    leader library (teleoperators/so_leader). Used to remove a stray file left by
    a run that didn't finish cleanly, so a failed/aborted/dropped auto-calibration
    never leaves a phantom library entry (follower) or scratch file (leader)."""
    robot_type = "so_follower" if device_type == "robot" else "so_leader"
    return os.path.join(CALIBRATION_BASE_PATH_ROBOTS, robot_type, f"{config_stem}.json")


def _remove_stray_calibration_file(device_type: str, config_stem: str) -> None:
    """Best-effort removal of the subprocess's --save output on a non-success
    run. The subprocess only writes it at the natural end of a fully-successful
    calibration (an interrupt raises before its save block), so on the normal
    failed/stopped path there is nothing to remove; this is the belt-and-braces
    guard for the case where the process wrote the file and then exited non-zero
    (or post-processing failed), which would otherwise leave a phantom entry."""
    path = _subprocess_output_path(device_type, config_stem)
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed stray auto-calibration file from a non-successful run: {path}")
    except OSError as e:
        logger.warning(f"Could not remove stray auto-calibration file {path}: {e}")


def _calibration_name_taken(device_type: str, config_stem: str) -> bool:
    """True if a calibration file already exists under the name the run would
    --save. Mirrors the single-arm library layout: leaders live under
    teleoperators/so_leader, followers under robots/so_follower."""
    if device_type == "teleop":
        path = os.path.join(LEADER_CONFIG_PATH, f"{config_stem}.json")
    else:
        path = os.path.join(FOLLOWER_CONFIG_PATH, f"{config_stem}.json")
    return os.path.exists(path)


def _release_arm_torque(port: str) -> list[str]:
    """Fallback torque release, run from THIS process after the calibration
    subprocess is dead (so the serial port is free again).

    The script's own SIGTERM handler releases torque on a clean stop, but a
    killed or wedged script leaves the arm energized (rigid) — so reconnect a
    fresh bus to the port and disable torque motor by motor, with retries.
    Returns problem descriptions; empty means every motor was released.

    Deliberately INSTANT — no freeze/return-to-start like the script's own
    graceful stop: this is the emergency path after the child died or was
    killed, the bus state is unknown, and the priority is de-energizing the
    arm, not landing it nicely.
    """
    try:
        bus = FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())
        # No handshake: a mid-calibration motor can be in overload and slow to
        # answer pings, but still accept the Torque_Enable=0 writes below.
        bus.connect(handshake=False)
    except Exception as e:
        message = (
            f"TORQUE MAY STILL BE ENABLED on {port} — could not reconnect to release the arm ({e}). "
            "The arm can stay rigid; unplug its power to release it."
        )
        logger.error(message)
        return [message]
    try:
        return force_disable_bus_torque(bus, "auto-calibration arm")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception as e:
            logger.warning(f"Auto-calibration fallback release: disconnect failed: {e}")


class _AutoCalArmRunner:
    """Runs the auto-calibration subprocess for ONE arm and tracks its state +
    logs. Owns all the per-arm machinery (subprocess, log reader, success
    finalization, escalating stop, fallback torque release) that was previously
    inlined in AutoCalibrationManager. Both the single-arm manager (one runner)
    and the concurrent batch manager (a runner per arm) drive it.

    Each runner has its own lock and threads, so several run fully independently
    and in parallel: one arm's failure, timeout, or wedged teardown never blocks
    another's."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._request: AutoCalibrationRequest | None = None
        self.status = AutoCalibrationStatus()

    def start(self, request: AutoCalibrationRequest) -> dict:
        with self._lock:
            if self.status.active:
                return {
                    "success": False,
                    "message": "Auto-calibration is already running",
                    "code": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
                }

            # Mutex with every other feature that drives the same serial bus
            # (see CLAUDE.md's "State model & mutual exclusion"). Lazy
            # imports to dodge circular imports at module load time (matches
            # the existing pattern in teleoperate.py/record.py/rollout.py).
            from . import (
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
            if _calibrate.calibration_is_active():
                return {
                    "success": False,
                    "message": "Calibration is currently active. Stop it first.",
                    "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
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

            if request.device_type not in ("teleop", "robot"):
                return {"success": False, "message": "Invalid device type"}
            if not request.port:
                return {"success": False, "message": "No port provided"}

            config_stem = _stem(request.config_file)
            robot_type = "so_follower" if request.device_type == "robot" else "so_leader"
            command = [
                sys.executable,
                "-m",
                _SCRIPT_MODULE,
                "--port",
                request.port,
                "--save",
                "--robot-id",
                config_stem,
                "--robot-type",
                robot_type,
            ]
            if request.motor_power is not None:
                # The robot's torque slider drives the calibration torque:
                # percent (clamped 10-100) → raw Torque_Limit register units.
                command += ["--torque-limit", str(torque_limit_from_percent(request.motor_power))]

            self._logs.clear()
            self._request = request
            self._stop_thread = None
            try:
                self._proc = subprocess.Popen(
                    command,
                    # DEVNULL, not inherited: with the server launched from a
                    # terminal the child would see a TTY on stdin and run
                    # interactively — its "press Enter" prompts would then
                    # block forever on a terminal nobody is answering.
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as e:
                logger.error(f"Failed to launch auto-calibration: {e}")
                self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
                return {"success": False, "message": str(e)}

            self.status = AutoCalibrationStatus(
                active=True, status="running", message="Auto-calibration running…"
            )
            self._thread = threading.Thread(
                target=self._run, name=f"auto-calibration:{request.port}", daemon=True
            )
            self._thread.start()
            # The claim above is the real state transition — broadcast the
            # hint so every WS client refetches the auto-calibration status.
            # (_notify_autocal_transition takes no locks, so calling it while
            # holding self._lock is safe.)
            _notify_autocal_transition(phase="running")
            return {"success": True, "message": "Auto-calibration started"}

    def _run(self) -> None:
        proc = self._proc
        assert proc is not None
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._logs.append(line.rstrip("\n"))
        except Exception as e:
            logger.warning(f"Auto-calibration log reader error: {e}")
        code = proc.wait()

        request = self._request
        with self._lock:
            if code == 0:
                try:
                    self._finalize_success()
                    self.status = AutoCalibrationStatus(
                        active=False, status="completed", message="Auto-calibration complete"
                    )
                except Exception as e:
                    logger.error(f"Auto-calibration post-processing failed: {e}")
                    # Post-processing failed after the subprocess wrote its file
                    # — don't leave that file behind as a phantom under a failed
                    # status. (The record write-back never assigns the name on
                    # this path either: it runs inside _finalize_success, which
                    # aborted.)
                    if request is not None:
                        _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                    self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
            elif self.status.status in ("stopping", "stopped"):
                # Stop path: _stop_worker owns the terminal status — it still
                # has to run the fallback torque release after the process is
                # gone, and only then flips the status to stopped/failed. It
                # also removes any stray output file. Leave "stopping"
                # (active=True) so the UI keeps polling.
                pass
            else:
                # Plain failure (nonzero exit, connection dropped): the
                # subprocess writes its file only on a fully-successful run, so
                # normally there's nothing here — but remove it defensively in
                # case the process wrote then died, so no phantom entry remains.
                if request is not None:
                    _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                self.status = AutoCalibrationStatus(
                    active=False, status="failed", error=f"Auto-calibration exited with code {code}"
                )
            self._proc = None
        # This arm's subprocess is gone: completed/failed here (a release),
        # or still "stopping" while _stop_worker owns the terminal status (in
        # which case the aggregate keeps active=True and _stop_worker emits
        # the final release itself).
        _notify_autocal_transition(phase=self.status.status)

    def _finalize_success(self) -> None:
        """Copy the leader file to MakerMods Lab's path (if needed) and write the config
        back into the robot record for the calibrated side + arm."""
        request = self._request
        if request is None:
            return
        config_stem = _stem(request.config_file)

        # The script saves leaders under robots/so_leader; MakerMods Lab reads leaders
        # from teleoperators/so_leader. Copy it over.
        if request.device_type == "teleop":
            src = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "so_leader", f"{config_stem}.json")
            dst = os.path.join(LEADER_CONFIG_PATH, f"{config_stem}.json")
            if os.path.exists(src):
                os.makedirs(LEADER_CONFIG_PATH, exist_ok=True)
                shutil.copy2(src, dst)
                logger.info(f"Copied auto-calibrated leader config to {dst}")

        # Write port + config back into the robot record (arm-aware), mirroring
        # the manual calibration write-back.
        if request.robot_name:
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
                logger.warning(f"Auto-calibration write-back failed for {request.robot_name}: {e}")

    def stop(self) -> dict:
        """Request a stop. Escalation runs in a worker thread so this returns
        immediately; the status is guaranteed to reach a terminal state
        (stopped/failed) even if the subprocess has to be killed."""
        with self._lock:
            if not self.status.active or self._proc is None:
                return {"success": False, "message": "No auto-calibration is running"}
            if self.status.status == "stopping":
                return {"success": False, "message": "Auto-calibration is already stopping"}
            self.status.status = "stopping"
            # The script's graceful stop freezes the arm, drives it back to its
            # starting pose, and only then releases torque — up to ~10s. Say so,
            # or the wait reads as an unresponsive Stop button.
            self.status.message = "Stopping — returning the arm to its starting position…"
            proc = self._proc
            port = self._request.port if self._request is not None else ""
            self._stop_thread = threading.Thread(
                target=self._stop_worker, args=(proc, port), name="auto-calibration-stop", daemon=True
            )
            self._stop_thread.start()
        # Still driving/holding the arm until the escalation finishes — a
        # phase of this session, not idle yet.
        _notify_autocal_transition(phase="stopping")
        return {"success": True, "message": "Stopping auto-calibration"}

    def stop_and_wait(self, timeout: float = _STOP_AND_WAIT_CEILING_S) -> None:
        """Like stop(), but blocks until the escalation has actually finished
        de-energizing the arm, instead of handing it to the background thread
        and returning right away.

        stop() returns immediately on purpose — it's built for the HTTP
        endpoint, where the frontend polls status while the SIGTERM -> grace
        -> SIGKILL -> torque-release sequence runs in the background. That's
        wrong for a caller that's about to let the process exit (server
        shutdown): the background thread doesn't get to keep running after
        the process is gone, so the arm could be left energized. This joins
        that thread first so the caller only returns once it's done, or the
        ceiling has passed.

        A no-op when nothing is running, or when a stop is already in
        flight — either way, if a stop thread exists, this still waits for
        it, so shutdown never walks away mid-stop."""
        self.stop()
        thread = self._stop_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _stop_worker(self, proc: subprocess.Popen, port: str) -> None:
        """Escalating stop: SIGTERM → grace period for the script's own torque
        release → SIGKILL → direct fallback torque release from this process.

        SIGTERM makes the script raise KeyboardInterrupt and release torque
        itself, but if its main thread is wedged in a C-level serial call the
        exception never materializes and the process won't die — observed on
        hardware. SIGKILL can't be caught, so after it the arm is assumed
        energized and we release torque directly over the (now free) port.
        Always ends on a terminal status so the UI can never freeze mid-stop.
        """
        killed = False
        problems: list[str] = []
        try:
            try:
                proc.terminate()
            except Exception as e:
                logger.warning(f"Error terminating auto-calibration: {e}")
            try:
                proc.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                killed = True
                logger.warning(
                    f"Auto-calibration did not exit within {_STOP_GRACE_S}s of SIGTERM; killing it"
                )
                self._logs.append(f"Stop: process did not exit within {_STOP_GRACE_S:.0f}s; killing it.")
                try:
                    proc.kill()
                except Exception as e:
                    logger.warning(f"Error killing auto-calibration: {e}")
                try:
                    proc.wait(timeout=_STOP_KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    # Unkillable = stuck in an uninterruptible kernel call; the
                    # port is still held, so the release below will fail loudly.
                    logger.error("Auto-calibration process survived SIGKILL")

            # Let the reader thread drain the last logs and observe the exit.
            reader = self._thread
            if reader is not None:
                reader.join(timeout=_READER_JOIN_S)

            # Belt and braces: release torque from here even when the script
            # exited cleanly — a killed or wedged script leaves torque enabled,
            # and re-disabling already-released motors is harmless.
            problems = _release_arm_torque(port)
            if killed and not problems:
                self._logs.append("Process killed; torque released directly over the port.")
        except Exception as e:
            logger.error(f"Auto-calibration stop worker failed: {e}")
            problems.append(
                f"TORQUE MAY STILL BE ENABLED on {port} — the stop sequence failed ({e}). "
                "The arm can stay rigid; unplug its power to release it."
            )
        finally:
            with self._lock:
                if self.status.status == "completed":
                    # The run finished during the grace period; keep the result
                    # (and its saved file — this was a success, not a stop).
                    pass
                else:
                    # A stopped run must leave no phantom library entry. The
                    # subprocess only writes its file at the natural end of a
                    # fully-successful run, so a stop mid-calibration normally
                    # wrote nothing — remove it defensively in case a stop
                    # landed just after the save block.
                    request = self._request
                    if request is not None:
                        _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                    if problems:
                        self.status = AutoCalibrationStatus(
                            active=False,
                            status="failed",
                            message="Auto-calibration stopped",
                            error=" ".join(problems),
                        )
                    else:
                        self.status = AutoCalibrationStatus(
                            active=False, status="stopped", message="Auto-calibration stopped"
                        )
                self._proc = None
            # Final release for the stop path: the escalation (and the
            # fallback torque release) has finished, whatever the outcome.
            _notify_autocal_transition(phase=self.status.status)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active": self.status.active,
                "status": self.status.status,
                "message": self.status.message,
                "error": self.status.error,
                "logs": list(self._logs),
            }

    def arm_status(self) -> dict:
        """Per-arm status enriched with this arm's identity — for the batch view.
        A superset of get_status(): includes name/port/device_type/arm."""
        req = self._request
        status = self.get_status()
        status.update(
            {
                "name": _stem(req.config_file) if req is not None else "",
                "port": req.port if req is not None else "",
                "device_type": req.device_type if req is not None else "",
                "arm": req.arm if req is not None else "left",
            }
        )
        return status


class AutoCalibrationManager(_AutoCalArmRunner):
    """Single-arm auto-calibration.

    A thin subclass of _AutoCalArmRunner so the existing single-arm endpoints
    (/start-auto-calibration, /stop-auto-calibration, /auto-calibration-status)
    keep their exact behavior and every previously-inlined attribute/method
    (_proc, _thread, _request, status, _finalize_success, _stop_worker, …) stays
    directly reachable — the machinery just became shareable with the batch
    path. No overrides: it exists to name the single-arm role."""


class AutoCalibrationBatchManager:
    """Runs auto-calibration on a USER-SELECTED SUBSET of arms CONCURRENTLY.

    Each arm gets its own _AutoCalArmRunner (own subprocess, own serial port,
    own lock and threads), so the arms run fully in parallel with independent
    outcomes: one arm erroring, timing out, or wedging its teardown never blocks
    another. Auto-cal is parallelizable precisely because each arm's subprocess
    drives its own arm on its own bus with no human motion required — unlike the
    human-in-the-loop manual flow, which is sequential.

    Partial success is the norm: report each arm's terminal status plus overall
    counts. A name persists only on that arm's success (the runner's
    _finalize_success write-back), and a failed arm's stray --save file is
    cleaned up by the runner, exactly as in the single-arm path."""

    _MAX_ARMS = 4

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runners: list[_AutoCalArmRunner] = []
        self._robot_name: str | None = None

    def _active(self) -> bool:
        return any(r.get_status()["active"] for r in self._runners)

    def start(self, request: AutoCalibrationBatchRequest) -> dict:
        with self._lock:
            if self._active():
                return {
                    "success": False,
                    "message": "A batch auto-calibration is already running",
                    "code": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
                }

            # Mutex with every other feature that drives the same serial bus
            # (see CLAUDE.md's "State model & mutual exclusion"). Checked
            # up front, before launching any arm, so a doomed batch refuses
            # cleanly instead of partially launching some arms before a
            # later arm's own runner.start() hits the same check mid-loop
            # (belt-and-braces with that per-runner check). Lazy imports to
            # dodge circular imports at module load time.
            from . import (
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
            if _calibrate.calibration_is_active():
                return {
                    "success": False,
                    "message": "Calibration is currently active. Stop it first.",
                    "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
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

            arms = request.arms
            # --- Fail-fast validation, before touching any hardware. ---
            if not arms:
                return {"success": False, "message": "No arms selected"}
            if len(arms) > self._MAX_ARMS:
                return {
                    "success": False,
                    "message": f"At most {self._MAX_ARMS} arms can be calibrated at once",
                }
            for arm in arms:
                if arm.device_type not in ("teleop", "robot"):
                    return {"success": False, "message": "Invalid device type"}
                if not arm.port:
                    return {"success": False, "message": "Every arm needs a port"}

            # Distinct ports: two subprocesses cannot share one serial bus.
            ports = [a.port for a in arms]
            if len(set(ports)) != len(ports):
                return {
                    "success": False,
                    "message": "Each arm needs its own port — two arms share a port",
                }

            # Distinct names within the same device_type/side (same file path).
            side_names: set[tuple[str, str]] = set()
            for a in arms:
                key = (a.device_type, _stem(a.config_file))
                if key in side_names:
                    return {
                        "success": False,
                        "message": f"Two arms would save to the same name '{_stem(a.config_file)}'",
                    }
                side_names.add(key)

            # Name-taken pre-check across all selected arms (unless overwrite).
            if not request.overwrite:
                taken = [
                    _stem(a.config_file)
                    for a in arms
                    if _calibration_name_taken(a.device_type, _stem(a.config_file))
                ]
                if taken:
                    return {
                        "success": False,
                        "code": "name_taken",
                        "message": "Calibration name already exists: " + ", ".join(sorted(set(taken))),
                        "names": sorted(set(taken)),
                    }

            # --- Launch every arm concurrently. Each runner spawns its own
            # subprocess + reader thread, so this loop returns quickly and the
            # arms run in parallel. If a launch fails, that arm is 'failed' and
            # the rest still run. ---
            self._robot_name = request.robot_name
            self._runners = [_AutoCalArmRunner() for _ in arms]
            launched = 0
            for runner, arm in zip(self._runners, arms, strict=True):
                req = AutoCalibrationRequest(
                    device_type=arm.device_type,
                    port=arm.port,
                    config_file=arm.config_file,
                    robot_name=request.robot_name,
                    arm=arm.arm,
                    motor_power=request.motor_power,
                )
                result = runner.start(req)
                # Give the runner its identity even if the launch failed, so the
                # status view can still name the failed arm.
                runner._request = req
                if result.get("success"):
                    launched += 1
                else:
                    runner.status = AutoCalibrationStatus(
                        active=False, status="failed", error=result.get("message", "launch failed")
                    )

            if launched == 0:
                return {"success": False, "message": "No arm could be launched"}
            return {
                "success": True,
                "message": f"Auto-calibration started on {launched} arm(s)",
                "total": len(arms),
                "launched": launched,
            }

    def stop(self) -> dict:
        """Stop ALL arms: request a stop on each running runner. Each runner's
        stop escalates (SIGTERM → SIGKILL) and releases that arm's torque in its
        own worker thread, so a stall in one arm's teardown never blocks the
        others."""
        with self._lock:
            if not self._active():
                return {"success": False, "message": "No batch auto-calibration is running"}
            stopped = 0
            for runner in self._runners:
                if runner.get_status()["active"] and runner.stop().get("success"):
                    stopped += 1
            return {"success": True, "message": f"Stopping {stopped} arm(s)"}

    def stop_and_wait(self, timeout: float = _STOP_AND_WAIT_CEILING_S) -> None:
        """Like stop(), but blocks until every arm's escalation has actually
        finished, instead of returning as soon as they're kicked off. See
        _AutoCalArmRunner.stop_and_wait for why a caller like server shutdown
        needs this instead of stop().

        Every arm is signalled to stop first, before waiting on any of
        them — arms run their escalation concurrently, so stopping one at a
        time here would leave the others still driving their motors for the
        length of every earlier arm's stop sequence."""
        with self._lock:
            runners = list(self._runners)
        active_runners = [r for r in runners if r.get_status()["active"]]
        for runner in active_runners:
            runner.stop()
        for runner in active_runners:
            thread = runner._stop_thread
            if thread is not None:
                thread.join(timeout=timeout)

    def get_status(self) -> dict:
        with self._lock:
            arms = [r.arm_status() for r in self._runners]
            total = len(arms)
            completed = sum(1 for a in arms if a["status"] == "completed")
            failed = sum(1 for a in arms if a["status"] in ("failed", "stopped"))
            active = any(a["active"] for a in arms)
            return {
                "active": active,
                "arms": arms,
                "total": total,
                "completed": completed,
                "failed": failed,
                # Combined log stream, per-arm prefixed so a single panel can
                # show everything at once; each arm also carries its own logs.
                "logs": [f"[{a['name'] or a['port']}] {line}" for a in arms for line in a["logs"]],
            }


auto_calibration_manager = AutoCalibrationManager()
auto_calibration_batch_manager = AutoCalibrationBatchManager()


def _notify_autocal_transition(phase: str | None = None) -> None:
    """Broadcast an auto-calibration `session_changed` hint.

    `active` is the AGGREGATE across the single-arm manager and every batch
    runner — one batch arm finishing while another still runs must not read as
    "auto-calibration over" — mirroring auto_calibration_is_active(), which is
    what the other features' mutex checks consult. Reads the status dataclass
    attributes directly (no runner locks, unlike get_status()) so it is safe
    to call from inside a runner's own locked section without self-deadlock;
    a hint can tolerate that raciness because consumers refetch the status
    endpoints rather than trusting the payload."""
    active = auto_calibration_manager.status.active or any(
        runner.status.active for runner in auto_calibration_batch_manager._runners
    )
    notify_session_changed("auto_calibration", active, phase=phase)


def auto_calibration_is_active() -> bool:
    """True while EITHER the single-arm manager or a batch run owns a serial
    bus. Reads the singletons' real state so other feature modules'
    reciprocal mutex checks (see CLAUDE.md) can't drift out of sync."""
    return auto_calibration_manager.status.active or auto_calibration_batch_manager._active()
