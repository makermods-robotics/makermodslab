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

"""Remote teleoperation — the OPERATOR side (session kind
``remote_teleoperation``).

This node's leader drives a STATION's follower. The station must already be
hosting (remote_host.py — "Available for remote teleop" pressed there); this
side never starts anything on the station. Start sequence: resolve the
station in the node registry → read its hosting descriptor (room, codec,
fps, motors, cameras — the wire contract) → check the arm family and motor
set against our leader (Portal silently drops packets whose schema
fingerprint differs, so a mismatch is refused here with a name, not
discovered as a dead arm) → fetch an ``operator`` token from the station's
token route → open the leader → join the room through the
``lerobot-robot-livekit`` plugin, which presents the remote follower as a
local lerobot ``Robot``. The loop is then local teleop's loop unchanged:
``leader.get_action()`` → ``robot.send_action()``, paced by the station's
fps.

What the browser sees: the WebSocket joint broadcast carries the REMOTE
follower's joints (from the synced observation, using the ranges the station
published) so the 3D viewer renders the real arm; the remote cameras are
re-streamed as MJPEG on /api/v1/remote-teleoperation/camera/{name} from the
frames the plugin already delivers — one extra hop, and the existing camera
tiles work unchanged. A later phase can have the browser join the room
itself with a viewer token.

Stopping is local: leave the room, release the (human-held, torque-off)
leader. The station's follower simply holds its last goal when its operator
leaves — the same thing a still leader does — and returns to rest when the
STATION's hosting session stops.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from .api_errors import ErrorCode
from .arm_capabilities import uses_feetech_bus
from .arm_identity import verify_devices
from .nodes import NodeNotFoundError, NodeUnreachableError, PeerJobRefusalError, node_registry
from .remote_host import (
    REMOTE_EXTRA_HINT,
    observation_degrees,
    observation_to_urdf_joints,
    remote_extra_available,
)
from .session_events import notify_session_changed
from .teleoperate import _device_buses, _safe_disconnect, force_disable_torque, force_disconnect_partial
from .utils.config import get_instance_id
from .utils.errors import classify_outcome, format_exception, friendly_hint
from .utils.robot_factory import build_leader_config

logger = logging.getLogger(__name__)

_BROADCAST_INTERVAL_S = 0.05  # 20 Hz, like local teleop
_STOP_JOIN_TIMEOUT_S = 5.0

remote_teleoperation_active = False
remote_teleoperation_thread: threading.Thread | None = None
current_station: dict[str, Any] | None = None
current_room: str | None = None
current_cameras: list[str] = []
last_cleanup_error: str | None = None
last_session_outcome: str | None = None
last_session_error: str | None = None
_state_lock = threading.Lock()
# Latest RGB frame per remote camera, for the MJPEG re-stream.
_frames: dict[str, Any] = {}
_frames_lock = threading.Lock()
_metrics_source: Any = None


class RemoteTeleoperateRequest(BaseModel):
    """Leader fields resolved from the local robot record + the station's
    instance id (sessions._build_remote_teleoperation_request)."""

    leader_port: str
    leader_config: str
    mode: str = "single"
    right_leader_port: str = ""
    right_leader_config: str = ""
    robot_name: str = ""
    arm_type: str = "so101"
    station: str
    skip_identity_check: bool = False


# --- pure helpers (unit-tested) ---------------------------------------------


def leader_motors(action_features: dict) -> list[str]:
    """The leader's ``<motor>.pos`` action keys, suffix stripped, in order —
    what Portal will fingerprint on our side."""
    return [k[: -len(".pos")] for k in action_features if isinstance(k, str) and k.endswith(".pos")]


def schema_mismatch(descriptor: dict[str, Any], arm_type: str, motors: list[str] | None) -> str | None:
    """Why this leader cannot drive the described follower, or None.

    The Star leader preset carries ITS follower family's joint mapping, so
    a family mismatch is refused before the leader is even opened; the
    motor set is checked once the leader is connected."""
    if descriptor.get("arm_type") != arm_type:
        return (
            f"The station's robot is a {descriptor.get('arm_type')!r} arm but this leader is "
            f"set up for {arm_type!r} — the families must match."
        )
    if motors is not None and list(descriptor.get("motors") or []) != list(motors):
        return (
            f"Motor sets differ: the station publishes {list(descriptor.get('motors') or [])}, "
            f"this leader produces {list(motors)}."
        )
    return None


def metrics_summary(portal_metrics: Any) -> dict[str, Any] | None:
    """Portal's metrics record → the status payload's ms numbers."""
    if portal_metrics is None:
        return None
    try:
        rtt = portal_metrics.rtt
        sync = portal_metrics.sync

        def ms(value: Any) -> float | None:
            return None if not value else round(float(value) / 1000.0, 2)

        return {
            "rtt_ms_last": ms(getattr(rtt, "rtt_us_last", None)),
            "rtt_ms_mean": ms(getattr(rtt, "rtt_us_mean", None)),
            "rtt_ms_p95": ms(getattr(rtt, "rtt_us_p95", None)),
            "observations": int(getattr(sync, "observations_emitted", 0) or 0),
            "states_dropped": int(getattr(sync, "states_dropped", 0) or 0),
        }
    except Exception:
        return None


def _refusal(status_code: int, message: str, code: ErrorCode | str) -> dict[str, Any]:
    return {"success": False, "status_code": status_code, "message": message, "code": code}


# --- station handshake -------------------------------------------------------


def _fetch_station(instance_id: str) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
    """(station, descriptor) from the registry, or a refusal dict."""
    try:
        peer = node_registry.resolve(instance_id)
        status = node_registry.fetch_peer_hosting(instance_id)
    except NodeNotFoundError:
        return _refusal(
            404, f"No registered node with instance id {instance_id!r}.", ErrorCode.NODE_NOT_FOUND
        )
    except NodeUnreachableError as exc:
        return _refusal(502, f"The station could not be reached: {exc}", ErrorCode.NODE_UNREACHABLE)
    descriptor = status.get("hosting") if isinstance(status, dict) else None
    if not status.get("hosting_active") or not isinstance(descriptor, dict):
        return _refusal(
            409,
            "The station is not hosting a robot. Press “Available for remote teleop” there first.",
            ErrorCode.NODE_NOT_HOSTING,
        )
    station = {"instance_id": instance_id, "name": peer.name, "url": peer.url}
    return station, descriptor


def _fetch_token(instance_id: str, room: str) -> str | dict[str, Any]:
    identity = f"operator-{get_instance_id()[:12]}"
    try:
        body = node_registry.request_peer_sfu_token(
            instance_id, {"identity": identity, "room": room, "role": "operator"}
        )
    except NodeUnreachableError as exc:
        return _refusal(
            502, f"The station could not be reached for a room token: {exc}", ErrorCode.NODE_UNREACHABLE
        )
    except PeerJobRefusalError as exc:
        detail = exc.body.get("detail") if isinstance(exc.body, dict) else None
        code = exc.body.get("code") if isinstance(exc.body, dict) else None
        return _refusal(
            exc.status_code,
            f"The station refused a room token: {detail or exc.body}",
            code or ErrorCode.NODE_UNREACHABLE,
        )
    return body["token"]


# --- hardware ----------------------------------------------------------------


def _connect_leader(request: RemoteTeleoperateRequest):
    """Open the leader(s). Returns (leader, warnings). ``connect(calibrate=
    False)`` never drops into interactive recalibration; the Feetech identity
    guard and calibration write apply to the SO-101 only."""
    from lerobot.teleoperators import make_teleoperator_from_config

    leader = make_teleoperator_from_config(build_leader_config(request))
    feetech = uses_feetech_bus(request.arm_type)
    try:
        logger.info(f"Connecting to the leader arm(s) on {request.leader_port}...")
        try:
            leader.connect(calibrate=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the leader arm on {request.leader_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e
        warnings = verify_devices(((leader, "leader"),), skip=request.skip_identity_check or not feetech)
        if feetech:
            for bus in _device_buses(leader):
                bus.write_calibration(bus.calibration)
        return leader, warnings
    except Exception:
        force_disconnect_partial(leader, "leader arm")
        raise


def _release_leader(leader, request: RemoteTeleoperateRequest) -> str | None:
    problems = force_disable_torque(leader, "leader arm") if uses_feetech_bus(request.arm_type) else []
    error = _safe_disconnect(leader, "leader arm")
    if error:
        problems.append(error)
    return " ".join(problems) if problems else None


# --- session ----------------------------------------------------------------


def handle_start_remote_teleoperation(
    request: RemoteTeleoperateRequest, websocket_manager=None
) -> dict[str, Any]:
    global remote_teleoperation_active, remote_teleoperation_thread, current_station, current_room
    global current_cameras, last_cleanup_error, last_session_outcome, last_session_error, _metrics_source
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        jobs as _jobs,
        record as _record,
        remote_host as _remote_host,
        replay as _replay,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    _teleoperate.finish_pending_release()
    _record.finish_pending_release()
    with _state_lock:
        if remote_teleoperation_active:
            return {
                "success": False,
                "message": "Remote teleoperation is already active",
                "code": ErrorCode.ROBOT_BUSY_REMOTE_TELEOPERATION,
            }
        for active, message, code in (
            (
                _teleoperate.teleoperation_active,
                "Teleoperation is currently active. Stop it first.",
                ErrorCode.ROBOT_BUSY_TELEOPERATION,
            ),
            (
                _remote_host.hosting_active,
                "This robot is hosted for remote teleoperation. Stop hosting first.",
                ErrorCode.ROBOT_BUSY_HOSTING,
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
        if not remote_extra_available():
            return _refusal(409, REMOTE_EXTRA_HINT, ErrorCode.SYSTEM_EXTRA_MISSING)

        # Station handshake BEFORE the claim: a station that is down or not
        # hosting is a refusal, not a session that died.
        fetched = _fetch_station(request.station)
        if isinstance(fetched, dict):
            return fetched
        station, descriptor = fetched
        reason = schema_mismatch(descriptor, request.arm_type, None)
        if reason:
            return _refusal(409, reason, ErrorCode.ROBOT_SCHEMA_MISMATCH)
        token = _fetch_token(request.station, descriptor["room"])
        if isinstance(token, dict):
            return token

        remote_teleoperation_active = True
        last_cleanup_error = None
        last_session_outcome = None
        last_session_error = None
        current_station = station
        current_room = descriptor["room"]
        current_cameras = [c["name"] for c in descriptor.get("cameras") or []]
        _metrics_source = None
        with _frames_lock:
            _frames.clear()

    notify_session_changed("remote_teleoperation", True)

    leader = None
    robot = None
    try:
        from lerobot_robot_livekit import LiveKitRobot, LiveKitRobotConfig
        from livekit.portal import VideoCodec

        leader, warnings = _connect_leader(request)
        motors = leader_motors(dict(leader.action_features))
        reason = schema_mismatch(descriptor, request.arm_type, motors)
        if reason:
            raise _SchemaMismatchError(reason)

        cameras = descriptor.get("cameras") or []
        observation_features: dict[str, Any] = {f"{m}.pos": float for m in descriptor["motors"]}
        for cam in cameras:
            observation_features[cam["name"]] = (cam["height"], cam["width"], 3)
        robot = LiveKitRobot(
            LiveKitRobotConfig(
                url=descriptor["url"],
                token=token,
                session=descriptor["room"],
                fps=int(descriptor["fps"]),
                camera_names=tuple(c["name"] for c in cameras),
                camera_width=int(cameras[0]["width"]) if cameras else 640,
                camera_height=int(cameras[0]["height"]) if cameras else 480,
                video_codec=VideoCodec[descriptor["video_codec"]],
                observation_features=observation_features,
                auto_claim_control=True,
            ),
            teleop=leader,
        )
        robot.connect()
        _metrics_source = robot
    except _SchemaMismatchError as e:
        cleanup_error = _release_leader(leader, request) if leader is not None else None
        _end_session(False, None, cleanup_error)
        return _refusal(409, str(e), ErrorCode.ROBOT_SCHEMA_MISMATCH)
    except Exception as e:
        logger.error(f"Remote teleoperation setup failed: {e}")
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as disconnect_error:
                logger.warning(f"Portal disconnect after a failed start: {disconnect_error}")
        cleanup_error = _release_leader(leader, request) if leader is not None else None
        _end_session(False, format_exception(e), cleanup_error)
        message = str(e)
        if cleanup_error:
            message += f" (cleanup: {cleanup_error})"
        return {"success": False, "message": message, "code": ErrorCode.HARDWARE_CONNECT_FAILED}

    ranges = dict(descriptor.get("joint_ranges_deg") or {})
    fps = int(descriptor["fps"])
    feetech = uses_feetech_bus(request.arm_type)
    is_bimanual = request.mode == "bimanual"
    camera_names = [c["name"] for c in cameras]

    def worker() -> None:
        period = 1.0 / max(1, fps)
        stopped_normally = False
        loop_error: str | None = None
        last_broadcast = 0.0
        try:
            while remote_teleoperation_active:
                tick = time.monotonic()
                observation = robot.get_observation()
                action = leader.get_action()
                robot.send_action(action)
                if observation:
                    with _frames_lock:
                        for name in camera_names:
                            frame = observation.get(name)
                            if frame is not None:
                                _frames[name] = frame
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
            logger.error(f"Error during remote teleoperation loop: {e}")
            loop_error = format_exception(e)
        finally:
            try:
                robot.disconnect()
            except Exception as e:
                logger.warning(f"Portal disconnect on stop: {e}")
            cleanup_error = _release_leader(leader, request)
            _end_session(stopped_normally, loop_error, cleanup_error)
            logger.info("Remote teleoperation stopped")

    remote_teleoperation_thread = threading.Thread(target=worker, name="remote-teleoperation", daemon=True)
    remote_teleoperation_thread.start()
    result: dict[str, Any] = {
        "success": True,
        "message": f"Remote teleoperation started against {station['name'] or station['url']}",
    }
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


class _SchemaMismatchError(RuntimeError):
    pass


def _end_session(stopped_normally: bool, loop_error: str | None, cleanup_error: str | None) -> None:
    global remote_teleoperation_active, current_station, current_room, current_cameras
    global last_cleanup_error, last_session_outcome, last_session_error, _metrics_source
    with _state_lock:
        last_cleanup_error = cleanup_error
        last_session_error = loop_error or cleanup_error
        last_session_outcome = classify_outcome(stopped_normally, last_session_error)
        remote_teleoperation_active = False
        current_station = None
        current_room = None
        current_cameras = []
        _metrics_source = None
    with _frames_lock:
        _frames.clear()
    notify_session_changed("remote_teleoperation", False)


def handle_stop_remote_teleoperation() -> dict[str, Any]:
    """Leave the room and release the leader. Nothing to return to rest on
    this side (the leader is human-held), so the stop waits — bounded — for
    the worker's cleanup and reports it."""
    global remote_teleoperation_active, remote_teleoperation_thread

    worker = remote_teleoperation_thread
    if not remote_teleoperation_active:
        return {"success": False, "message": "No remote teleoperation session is active"}
    logger.info("Stop remote teleoperation triggered")
    remote_teleoperation_active = False
    if worker is not None and worker.is_alive():
        worker.join(timeout=_STOP_JOIN_TIMEOUT_S)
        if worker.is_alive():
            return {
                "success": True,
                "message": "Stop requested, but the worker has not shut down yet",
                "warning": "The remote teleoperation worker did not shut down within 5s.",
            }
    remote_teleoperation_thread = None
    if last_cleanup_error:
        return {
            "success": True,
            "message": "Remote teleoperation stopped, but releasing the leader reported a problem",
            "warning": last_cleanup_error,
        }
    return {"success": True, "message": "Remote teleoperation stopped"}


def handle_remote_teleoperation_status() -> dict[str, Any]:
    with _state_lock:
        active = remote_teleoperation_active
        station = dict(current_station) if current_station else None
        room = current_room
        cameras = list(current_cameras)
        source = _metrics_source
    metrics = None
    if source is not None:
        try:
            metrics = metrics_summary(source.metrics())
        except Exception:
            metrics = None
    return {
        "remote_teleoperation_active": active,
        "station": station,
        "room": room,
        "cameras": cameras,
        "metrics": metrics,
        "last_cleanup_error": last_cleanup_error,
        "outcome": last_session_outcome,
        "error": last_session_error,
        "hint": friendly_hint(last_session_error),
        "message": "Remote teleoperation status retrieved successfully",
    }


# --- camera re-stream --------------------------------------------------------


def latest_frame(name: str):
    with _frames_lock:
        return _frames.get(name)


def camera_stream(name: str, fps: int = 15, jpeg_quality: int = 70) -> Iterator[bytes]:
    """multipart/x-mixed-replace MJPEG of one remote camera, from the frames
    the Portal plugin delivers (RGB → BGR for cv2). Ends when the session
    does. Paced at `fps`; a frame is re-sent only when a newer one arrived."""
    import cv2

    period = 1.0 / max(1, fps)
    last_sent = None
    while remote_teleoperation_active:
        frame = latest_frame(name)
        if frame is not None and frame is not last_sent:
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                last_sent = frame
        time.sleep(period)
