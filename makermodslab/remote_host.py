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

"""Hosting — the STATION side of remote teleoperation (session kind
``hosting``).

"Available for remote teleop": the station opens the record's follower(s)
WITH the record's cameras, joins its own LiveKit room (sfu.py) as Portal's
``Robot`` through the ``lerobot-teleoperator-livekit`` plugin, publishes
state + frames at ``fps``, and executes whatever the room's ACTIVE operator
sends. It never opens a leader. The session is deliberately operator-
agnostic: a laptop's leader today (remote_teleoperate.py), a policy worker
in the remote-inference phase, both against this unchanged session — Portal
arbitrates who is active.

The worker loop is the local teleop loop with the devices swapped:
``teleop_device.get_action()`` → ``robot.send_action()`` where the "teleop
device" is the network. Portal's plugin preserves lerobot's Teleoperator
contract, so ``send_feedback(robot.get_observation())`` is the only added
line. One knob, ``fps``, paces frames, state and action application alike.

Safety is teleoperate.py's, unchanged: the follower's start pose is captured
after connect and driven back on a normal stop (SO-101 via rest_pose,
CAN arms via maker_rest_pose), torque is released explicitly, a second stop
aborts the return. Operator loss without lease loss (a WireGuard blip)
freezes the follower at its last goal — exactly what a still leader does in
local teleop. Abandonment is the session lease's job: the hosting session is
owned by the station's own browser tab, and the expiry watchdog stops it.

The plugin import is lazy: ``remote`` is an optional extra (its Portal wheel
is Python-3.12/Linux/Apple-Silicon only until our fork retags it), and a
station without it refuses with system.extra_missing rather than failing at
import time and taking every other flow with it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from . import sfu
from .api_errors import ErrorCode
from .arm_capabilities import uses_feetech_bus
from .arm_identity import verify_devices
from .maker_rest_pose import capture_maker_pose, maker_follower_arms, return_maker_arms_to_rest
from .motor_power import FOLLOWER, clear_goal_velocity, reset_torque_limit
from .rest_pose import capture_rest_pose
from .session_events import notify_session_changed
from .teleoperate import (
    _SO101_URDF_JOINTS,
    _STS3215_MAX_RES,
    _device_buses,
    _return_followers_to_rest,
    _safe_disconnect,
    force_disable_torque,
    force_disconnect_partial,
)
from .torque import de_energize_can_device, release_maker_torque
from .utils.config import get_instance_id
from .utils.errors import classify_outcome, format_exception, friendly_hint
from .utils.robot_factory import build_follower_config

logger = logging.getLogger(__name__)

# Portal's lerobot plugin is the `remote` extra; this is the module we probe
# and the install target the UI offers.
REMOTE_EXTRA_PROBE_MODULE = "lerobot_teleoperator_livekit"
REMOTE_EXTRA_INSTALL_TARGET = "makermodslab[remote]"
REMOTE_EXTRA_HINT = (
    "The `remote` extra (LiveKit Portal's lerobot plugins) is not installed. "
    "Install it from Settings → Optional extras, or `uv pip install 'makermodslab[remote]'`, then restart."
)

# Identity the station joins its own room with. One Robot per room (Portal's
# model), so a fixed identity is right; operators get unique ones.
ROBOT_IDENTITY = "robot"

_BROADCAST_INTERVAL_S = 0.05  # 20 Hz joint broadcast, same as local teleop

hosting_active = False
hosting_thread: threading.Thread | None = None
releasing = False
current_descriptor: dict[str, Any] | None = None
# The live plugin instance (LiveKitTeleoperator) — read by the status handler
# for the room's active operator; None outside a session.
current_teleop: Any = None
last_cleanup_error: str | None = None
last_session_outcome: str | None = None
last_session_error: str | None = None
_state_lock = threading.Lock()
_release_now = threading.Event()


class HostingRequest(BaseModel):
    """Everything the station needs, resolved server-side from the robot
    record by sessions._build_hosting_request — never from a client."""

    follower_port: str
    follower_config: str
    mode: str = "single"
    right_follower_port: str = ""
    right_follower_config: str = ""
    robot_name: str = ""
    arm_type: str = "so101"
    # {camera name: record settings} (utils/config.record_cameras_by_name).
    cameras: dict[str, dict] = {}
    fps: int = 30
    video_codec: str = "H264"
    skip_identity_check: bool = False


# --- pure helpers (unit-tested) ---------------------------------------------


def remote_extra_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(REMOTE_EXTRA_PROBE_MODULE) is not None
    except (ImportError, ValueError):
        return False


def split_features(observation_features: dict) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """lerobot ``observation_features`` → (motor names, {camera: (h, w)}).

    Motors are the scalar-typed ``<motor>.pos`` keys with the suffix
    stripped (a bimanual robot's ``left_``/``right_`` prefixes stay — they
    ARE the wire names); cameras are the tuple-shaped ``(h, w, c)`` entries.
    Order is preserved: Portal fingerprints the schema by field order.
    """
    motors: list[str] = []
    cameras: dict[str, tuple[int, int]] = {}
    for key, value in observation_features.items():
        if isinstance(value, tuple) and len(value) >= 2:
            cameras[key] = (int(value[0]), int(value[1]))
        elif isinstance(key, str) and key.endswith(".pos"):
            motors.append(key[: -len(".pos")])
    return motors, cameras


def joint_ranges_deg(calibrations: dict[str, dict]) -> dict[str, float]:
    """Per-motor calibrated full travel in degrees, for the SO-101 URDF
    mapping on the operator side (mirrors teleoperate._motor_fraction).

    ``calibrations`` is {prefix: lerobot calibration dict} — ``{"": cal}``
    single, ``{"left_": cal_l, "right_": cal_r}`` bimanual. Only body
    (DEGREES-mode) joints appear; the gripper is a 0–100 percentage.
    """
    out: dict[str, float] = {}
    for prefix, cal in calibrations.items():
        for motor in _SO101_URDF_JOINTS:
            if motor == "gripper":
                continue
            entry = (cal or {}).get(motor)
            if entry is None:
                continue
            full = (entry.range_max - entry.range_min) * 360.0 / _STS3215_MAX_RES
            if full > 0:
                out[f"{prefix}{motor}"] = full
    return out


def observation_to_urdf_joints(
    observation: dict[str, Any], ranges_deg: dict[str, float], prefix: str = ""
) -> dict[str, float]:
    """A lerobot observation's ``.pos`` values → URDF joint radians, using
    the published ranges instead of a calibration object. Same affine map
    as teleoperate.get_joint_positions_from_robot; joints without a range
    render their raw degrees (uncalibrated fallback)."""
    joints: dict[str, float] = {}
    for motor, (urdf_joint, lower, upper, sign) in _SO101_URDF_JOINTS.items():
        key = f"{prefix}{motor}.pos"
        if key not in observation:
            joints[urdf_joint] = 0.0
            continue
        value = float(observation[key])
        if motor == "gripper":
            frac: float | None = value / 100.0
        else:
            full = ranges_deg.get(f"{prefix}{motor}")
            frac = 0.5 + value / full if full else None
        if frac is None:
            joints[urdf_joint] = value * math.pi / 180.0
            continue
        frac = min(1.0, max(0.0, frac))
        if sign < 0:
            frac = 1.0 - frac
        joints[urdf_joint] = lower + frac * (upper - lower)
    return joints


def observation_degrees(observation: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """CAN-arm readout: ``.pos`` values (already degrees) keyed by bare motor
    name, one arm's prefix stripped — the shape of teleoperate.get_maker_joint_degrees."""
    out: dict[str, float] = {}
    for key, value in observation.items():
        if not key.endswith(".pos"):
            continue
        motor = key[: -len(".pos")]
        if prefix:
            if not motor.startswith(prefix):
                continue
            motor = motor[len(prefix) :]
        elif motor.startswith(("left_", "right_")):
            continue
        if isinstance(value, (int, float)):
            out[motor] = float(value)
    return out


def build_descriptor(
    request: HostingRequest,
    *,
    room: str,
    motors: list[str],
    cameras: dict[str, tuple[int, int]],
    ranges_deg: dict[str, float],
) -> dict[str, Any]:
    """The hosting descriptor (schemas/remote.HostingDescriptor) minus the
    request-derived ``url`` and the live ``active_operator``, which the
    status handler fills per read."""
    return {
        "robot": request.robot_name,
        "arm_type": request.arm_type,
        "mode": request.mode,
        "room": room,
        "fps": request.fps,
        "video_codec": request.video_codec,
        "motors": list(motors),
        "cameras": [{"name": name, "width": w, "height": h} for name, (h, w) in cameras.items()],
        "joint_ranges_deg": dict(ranges_deg),
    }


# --- hardware ----------------------------------------------------------------


def _connect_follower(request: HostingRequest):
    """Open the follower(s) with cameras. Returns (robot, warnings).

    Mirrors record.py's follower path: ``robot.connect(calibrate=False)``
    opens the bus AND the cameras in one call (never interactive
    recalibration — a headless worker would hang on input()). The Feetech
    preflights (identity guard, calibration write, torque-limit/velocity
    resets) apply to the SO-101 only; a CAN follower's own connect() does
    its equivalent, and a failed one is de-energized before raising because
    on Metal the handshake IS the enable command.
    """
    from lerobot.robots import make_robot_from_config

    from .record import _build_camera_configs, _platform_backend

    camera_configs = _build_camera_configs(request.cameras, _platform_backend())
    robot_config = build_follower_config(request, cameras=camera_configs)
    robot = make_robot_from_config(robot_config)
    feetech = uses_feetech_bus(request.arm_type)
    label = "follower arm"
    try:
        logger.info(
            f"Connecting to the follower arm(s) on {request.follower_port} with {len(camera_configs)} camera(s)..."
        )
        try:
            robot.connect(calibrate=False)
        except Exception as e:
            if not feetech:
                de_energize_can_device(robot, label)
            raise RuntimeError(
                f"Could not connect to the follower arm on {request.follower_port} (or one of its cameras). "
                "Make sure it's plugged in and powered on, and that no browser preview holds the camera."
            ) from e
        warnings = verify_devices(((robot, "follower"),), skip=request.skip_identity_check or not feetech)
        if feetech:
            warnings += reset_torque_limit(robot, FOLLOWER)
            warnings += clear_goal_velocity(robot, FOLLOWER)
        return robot, warnings
    except Exception:
        force_disconnect_partial(robot, label)
        raise


def _release_follower(robot, request: HostingRequest) -> str | None:
    if uses_feetech_bus(request.arm_type):
        problems = force_disable_torque(robot, "follower arm")
    else:
        problems = release_maker_torque(robot, "follower arm")
    error = _safe_disconnect(robot, "follower arm")
    if error:
        problems.append(error)
    return " ".join(problems) if problems else None


def _calibrations_by_prefix(robot, request: HostingRequest) -> dict[str, dict]:
    if not uses_feetech_bus(request.arm_type):
        return {}
    if request.mode == "bimanual":
        return {"left_": robot.left_arm.calibration, "right_": robot.right_arm.calibration}
    return {"": getattr(robot, "calibration", None) or {}}


# --- session ----------------------------------------------------------------


def handle_start_hosting(request: HostingRequest, websocket_manager=None) -> dict[str, Any]:
    """Claim the hardware, connect the follower + cameras, join the room, and
    stream from a worker thread. Connect failures are reported to the caller
    synchronously (nothing is claimed on failure), as every feature does."""
    global hosting_active, hosting_thread, releasing, current_descriptor, current_teleop
    global last_cleanup_error, last_session_outcome, last_session_error
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        jobs as _jobs,
        record as _record,
        remote_teleoperate as _remote_teleoperate,
        replay as _replay,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    _teleoperate.finish_pending_release()
    _record.finish_pending_release()
    with _state_lock:
        if hosting_active:
            return {
                "success": False,
                "message": "Hosting is already active",
                "code": ErrorCode.ROBOT_BUSY_HOSTING,
            }
        if hosting_thread is not None and hosting_thread.is_alive():
            return {
                "success": False,
                "message": "The arm from the previous hosting session is still being released. Try again in a few seconds.",
                "code": ErrorCode.ROBOT_BUSY_RELEASING,
            }
        for active, message, code in (
            (
                _teleoperate.teleoperation_active,
                "Teleoperation is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_TELEOPERATION,
            ),
            (
                _remote_teleoperate.remote_teleoperation_active,
                "Remote teleoperation is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_REMOTE_TELEOPERATION,
            ),
            (
                _record.recording_active,
                "Recording is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_RECORDING,
            ),
            (
                _rollout.inference_active,
                "Inference is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_INFERENCE,
            ),
            (
                _calibrate.calibration_is_active(),
                "Calibration is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_CALIBRATION,
            ),
            (
                _auto_calibrate.auto_calibration_is_active(),
                "Auto-calibration is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
            ),
            (
                _wiggle.wiggle_active,
                "A gripper wiggle is currently in progress. Wait for it to finish.",
                ErrorCode.ROBOT_BUSY_WIGGLE,
            ),
            (
                _replay.replay_active,
                "Replay is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_REPLAY,
            ),
        ):
            if active:
                return {"success": False, "message": message, "code": code}
        if (training := _jobs.training_is_active()) is not None:
            return {
                "success": False,
                "message": f"Training run '{training}' is using this machine. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_TRAINING,
            }
        # Preconditions that are NOT hardware: no room to join without the
        # SFU, no plugin without the extra. Checked before the claim so a
        # refusal never emits a session event.
        if not sfu.sfu_enabled():
            return {
                "success": False,
                "status_code": 409,
                "message": "No LiveKit SFU is running on this node. Restart it with `makermodslab --sfu` to host.",
                "code": ErrorCode.SFU_DISABLED,
            }
        if not remote_extra_available():
            return {
                "success": False,
                "status_code": 409,
                "message": REMOTE_EXTRA_HINT,
                "code": ErrorCode.SYSTEM_EXTRA_MISSING,
            }

        hosting_active = True
        last_cleanup_error = None
        last_session_outcome = None
        last_session_error = None
        releasing = False
        current_descriptor = None
        _release_now.clear()

    notify_session_changed("hosting", True)

    robot = None
    teleop = None
    try:
        from lerobot_teleoperator_livekit import LiveKitTeleoperator, LiveKitTeleoperatorConfig

        robot, warnings = _connect_follower(request)

        # Wire contract from the connected robot: what lerobot says it has.
        motors, cameras = split_features(dict(robot.observation_features))
        ranges = joint_ranges_deg(_calibrations_by_prefix(robot, request))
        room = sfu.default_room(get_instance_id())
        api_key, api_secret = sfu.api_keys()
        token, _expires = sfu.mint_token(
            api_key=api_key, api_secret=api_secret, identity=ROBOT_IDENTITY, room=room, role="robot"
        )
        from livekit.portal import VideoCodec

        teleop = LiveKitTeleoperator(
            LiveKitTeleoperatorConfig(
                url=sfu.local_url(),
                token=token,
                session=room,
                fps=request.fps,
                video_codec=VideoCodec[request.video_codec],
            ),
            robot=robot,
        )
        teleop.connect()
        descriptor = build_descriptor(request, room=room, motors=motors, cameras=cameras, ranges_deg=ranges)

        # Start pose for the return on stop — followers only, gripper
        # excluded (it may be holding something at stop time).
        if uses_feetech_bus(request.arm_type):
            follower_rest_poses = [
                (bus, {m: v for m, v in capture_rest_pose(bus).items() if m != "gripper"})
                for bus in _device_buses(robot)
            ]
            maker_rest_poses = []
        else:
            follower_rest_poses = []
            maker_rest_poses = [(arm, capture_maker_pose(arm)) for arm, _label in maker_follower_arms(robot)]

        with _state_lock:
            current_descriptor = descriptor
            current_teleop = teleop
    except Exception as e:
        logger.error(f"Hosting setup failed: {e}")
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as disconnect_error:
                logger.warning(f"Portal disconnect after a failed start: {disconnect_error}")
        cleanup_error = _release_follower(robot, request) if robot is not None else None
        with _state_lock:
            hosting_active = False
            last_cleanup_error = cleanup_error
            last_session_error = format_exception(e)
            last_session_outcome = classify_outcome(False, last_session_error)
        notify_session_changed("hosting", False)
        message = str(e)
        if cleanup_error:
            message += f" (cleanup: {cleanup_error})"
        return {"success": False, "message": message, "code": ErrorCode.HARDWARE_CONNECT_FAILED}

    def worker() -> None:
        global hosting_active, releasing, current_descriptor, current_teleop, last_cleanup_error
        global last_session_outcome, last_session_error
        period = 1.0 / max(1, request.fps)
        is_bimanual = request.mode == "bimanual"
        feetech = uses_feetech_bus(request.arm_type)
        stopped_normally = False
        loop_error: str | None = None
        last_broadcast = 0.0
        try:
            while hosting_active:
                tick = time.monotonic()
                observation = robot.get_observation()
                teleop.send_feedback(observation)
                action = teleop.get_action()
                if action:
                    robot.send_action(action)
                now = time.time()
                if now - last_broadcast >= _BROADCAST_INTERVAL_S:
                    try:
                        joint_data: dict[str, Any] = {"type": "joint_update", "timestamp": now}
                        if feetech:
                            joint_data["joints"] = observation_to_urdf_joints(
                                observation, ranges, prefix="left_" if is_bimanual else ""
                            )
                            if is_bimanual:
                                joint_data["joints_right"] = observation_to_urdf_joints(
                                    observation, ranges, prefix="right_"
                                )
                        else:
                            joint_data["joints"] = {}
                            joint_data["joints_deg"] = observation_degrees(
                                observation, prefix="left_" if is_bimanual else ""
                            )
                            if is_bimanual:
                                joint_data["joints_deg_right"] = observation_degrees(
                                    observation, prefix="right_"
                                )
                        if websocket_manager and websocket_manager.active_connections:
                            websocket_manager.broadcast_joint_data_sync(joint_data)
                        last_broadcast = now
                    except Exception as e:
                        logger.error(f"Error broadcasting joint data: {e}")
                sleep_for = period - (time.monotonic() - tick)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            stopped_normally = True
        except Exception as e:
            logger.error(f"Error during hosting loop: {e}")
            loop_error = format_exception(e)
        finally:
            # Leave the room FIRST so no late action lands mid-return.
            try:
                teleop.disconnect()
            except Exception as e:
                logger.warning(f"Portal disconnect on stop: {e}")
            if stopped_normally and not _release_now.is_set():
                releasing = True
                notify_session_changed("hosting", True, phase="releasing")
                _return_followers_to_rest(follower_rest_poses, _release_now)
                return_maker_arms_to_rest(maker_rest_poses, _release_now)
            cleanup_error = _release_follower(robot, request)
            with _state_lock:
                last_cleanup_error = cleanup_error
                last_session_error = loop_error or cleanup_error
                last_session_outcome = classify_outcome(stopped_normally, last_session_error)
                hosting_active = False
                releasing = False
                current_descriptor = None
                current_teleop = None
            logger.info("Hosting stopped")
            notify_session_changed("hosting", False)

    hosting_thread = threading.Thread(target=worker, name="hosting", daemon=True)
    hosting_thread.start()
    result: dict[str, Any] = {"success": True, "message": "Hosting started"}
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def handle_stop_hosting() -> dict[str, Any]:
    """Same two-press contract as teleoperation: first stop → the follower
    returns to its start pose then goes limp (reported as releasing);
    second stop during the return → release now and wait."""
    global hosting_active, hosting_thread

    worker = hosting_thread
    if hosting_active:
        logger.info("Stop hosting triggered")
        hosting_active = False
        if worker is None or not worker.is_alive():
            hosting_thread = None
            if last_cleanup_error:
                return {
                    "success": True,
                    "message": "Hosting stopped, but releasing the arm reported a problem",
                    "warning": last_cleanup_error,
                }
            return {"success": True, "message": "Hosting stopped"}
        return {
            "success": True,
            "releasing": True,
            "message": "Hosting stopped — the arm returns to its starting position, then goes limp. Press Stop again to release it now.",
        }
    if worker is not None and worker.is_alive():
        logger.info("Second stop during the rest-pose return — releasing the arm now")
        _release_now.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            return {
                "success": True,
                "message": "Release requested, but the worker has not shut down yet",
                "warning": "The hosting worker did not shut down within 5s, so the arm may not have been released.",
            }
        hosting_thread = None
        if last_cleanup_error:
            return {
                "success": True,
                "message": "Arm released, but the release reported a problem",
                "warning": last_cleanup_error,
            }
        return {"success": True, "message": "Arm released"}
    return {"success": False, "message": "No hosting session is active"}


def _active_operator() -> str | None:
    """The room's active operator, via the plugin's Portal (a private handle
    until the plugin exposes it) — None when unknown."""
    try:
        portal = getattr(current_teleop, "_portal", None)
        return portal.active_operator() if portal is not None else None
    except Exception:
        return None


def handle_hosting_status(request_host: str) -> dict[str, Any]:
    """GET /api/v1/hosting. The descriptor's URL is derived from the host the
    CALLER reached this API on (sfu.sfu_url) — the address known to be
    routable from where they sit."""
    with _state_lock:
        descriptor = dict(current_descriptor) if current_descriptor else None
        active = hosting_active
    if descriptor is not None:
        descriptor["url"] = sfu.sfu_url(request_host)
        descriptor["active_operator"] = _active_operator()
    return {
        "hosting_active": active,
        "hosting": descriptor,
        "releasing": releasing,
        "last_cleanup_error": last_cleanup_error,
        "outcome": last_session_outcome,
        "error": last_session_error,
        "hint": friendly_hint(last_session_error),
        "message": "Returning the arm to its rest position…"
        if releasing
        else "Hosting status retrieved successfully",
    }
