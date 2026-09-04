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
``hosting``), and station mode (``makermodslab --host <robot>``).

The station opens the record's follower(s) WITH the record's cameras, joins
its own LiveKit room (sfu.py) as Portal's ``Robot`` through the
``lerobot-teleoperator-livekit`` plugin, publishes state + frames at ``fps``,
and executes what the seated operator sends. It never opens a leader, and
it is deliberately operator-agnostic: a laptop's leader today
(remote_teleoperate.py), a policy worker in the remote-inference phase.

**Parked and engaged.** A hosting session has a resting state it falls back
to on its own. *Parked*: the follower sits at its rest pose with torque OFF;
cameras and state still stream, the room is open, the station listens.
*Engaged*: torque on, following the seated operator. Transitions carry the
safety — parked → engaged when an operator takes the seat (automatically on
join, or on an explicit ``engage`` RPC after a Home), always through a soft
start that blends from the follower's present pose to the leader's over
``SOFT_START_S``; engaged → parked on the operator's Home, on a ``release``
(the operator ending its session — immediate), on a silent operator loss
after ``GRACE_S`` (a Tailscale path change or a laptop sleep reconnects well
inside it; a reconnect with the same identity resumes the seat), on an
action-stream stall of ``GRACE_S`` with the operator still present, and on
the station's own stop. Every park goes through the same return-to-rest
path local teleop uses, then releases torque.

**Single seat.** The station's own room token caps the room at robot + one
operator (the SFU enforces it), the token route refuses a second operator
token while the seat is held (server.py), and operators never self-claim —
the station assigns Portal's active operator when the seat is free and
clears it when the seat empties. SeatMonitor is that policy, pure and
clock-injected; the worker thread applies its decisions to the hardware.

**Station mode** (``--host``): the process hosts the named robot parked from
startup with no browser and no lease (the engaged → parked timeouts do the
lease's job for an unattended arm), and re-arms hosting after any local
session ends. A local flow started at the station preempts a parked,
unseated hosting session automatically (sessions.handle_start_session) —
"local wins when idle"; a seated operator is a held session like any other.

SO-101 only in this release: parking toggles torque mid-session, and the
CAN arms' torque semantics (the Metal handshake IS the enable) need their
own treatment. The plugin import is lazy: ``remote`` is an optional extra.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from . import sfu
from .api_errors import ApiError, ErrorCode
from .arm_capabilities import uses_feetech_bus
from .arm_identity import verify_devices
from .motor_power import FOLLOWER, clear_goal_velocity, reset_torque_limit
from .rest_pose import RETURN_CEILING_S, capture_rest_pose
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
from .utils.config import get_instance_id
from .utils.errors import classify_outcome, format_exception, friendly_hint
from .utils.robot_factory import build_follower_config
from .utils.system import REMOTE_INSTALL_HINT

logger = logging.getLogger(__name__)

# Portal's lerobot plugin is the `remote` extra; the probe module and the
# install target are utils/system.py's (one place for the pins).
REMOTE_EXTRA_PROBE_MODULE = "lerobot_teleoperator_livekit"
REMOTE_EXTRA_HINT = (
    "The `remote` extra (LiveKit Portal's lerobot plugins) is not installed. " + REMOTE_INSTALL_HINT
)

# Identity the station joins its own room with. One Robot per room (Portal's
# model), so a fixed identity is right; operators get unique ones.
ROBOT_IDENTITY = "robot"
# Robot + one operator: the SFU-enforced half of the single seat.
ROOM_MAX_PARTICIPANTS = 2

GRACE_S = 15.0
SOFT_START_S = 1.0
# An action counts as "live" for auto re-engage if it arrived within this.
_LIVE_ACTION_S = 1.0
_BROADCAST_INTERVAL_S = 0.05  # 20 Hz joint broadcast, same as local teleop
_STATION_RETRY_S = 3.0
_STATION_BACKOFF_S = 15.0
STATION_ROBOT_ENV = "MAKERMODSLAB_HOST_ROBOT"
STATION_ENV = "MAKERMODSLAB_STATION"

PHASES = ("parked", "engaging", "engaged", "parking")

hosting_active = False
hosting_thread: threading.Thread | None = None
releasing = False
phase: str = "parked"
current_descriptor: dict[str, Any] | None = None
# The live plugin instance (LiveKitTeleoperator) — None outside a session.
current_teleop: Any = None
seat: SeatMonitor | None = None
station_mode = False
station_robot: str | None = None
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


class SeatMonitor:
    """The single operator seat and the engage/park policy — pure state,
    injected clock, no hardware. The worker feeds it room events, RPCs and
    action arrivals and applies the decision it returns: ``"engage"`` or
    ``"park"`` (or None).

    Rules: the first operator to join takes the seat and is engaged; the
    seat holder rejoining (a reconnect inside the grace window) is
    re-engaged; anyone else is ignored (the token route and the room cap
    keep them out anyway). ``home`` parks and HOLDS — the leader's stream
    keeps arriving, so only an explicit ``engage`` (or a rejoin) resumes.
    ``release`` parks at once and frees the seat. A silent loss frees the
    seat after ``grace_s``; an action-stream stall of ``grace_s`` with the
    operator still present parks but keeps the seat, and the stream coming
    back re-engages by itself (no hold). A stall that goes on for a second
    ``grace_s`` frees the seat too: the SFU reports a hard-crashed operator
    late (its peer-connection timeout is longer than ours), and a seat held
    for a laptop that is gone would keep the next operator out.
    """

    def __init__(self, grace_s: float = GRACE_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.grace_s = grace_s
        self.clock = clock
        self.seat: str | None = None
        self.lost_at: float | None = None
        self.last_action_at: float | None = None
        self.hold = False
        self.stalled_at: float | None = None

    def operator_joined(self, identity: str) -> str | None:
        if self.seat is not None and identity != self.seat:
            return None
        self.seat = identity
        self.lost_at = None
        self.hold = False
        self.last_action_at = self.clock()
        return "engage"

    def operator_left(self, identity: str) -> None:
        if identity == self.seat:
            self.lost_at = self.clock()

    def action_received(self) -> None:
        self.last_action_at = self.clock()
        self.stalled_at = None

    def command(self, name: str, caller: str) -> str | None:
        """An RPC from `caller`; only the seat holder is heard."""
        if self.seat is None or caller != self.seat:
            return None
        if name == "home":
            self.hold = True
            return "park"
        if name == "engage":
            self.hold = False
            self.last_action_at = self.clock()
            return "engage"
        if name == "release":
            self.seat = None
            self.lost_at = None
            self.hold = False
            return "park"
        return None

    def tick(self, engaged: bool) -> str | None:
        now = self.clock()
        if self.seat is not None and self.lost_at is not None and now - self.lost_at >= self.grace_s:
            self.seat = None
            self.lost_at = None
            self.hold = False
            return "park" if engaged else None
        if engaged:
            if (
                self.lost_at is None
                and self.last_action_at is not None
                and now - self.last_action_at >= self.grace_s
            ):
                self.stalled_at = now
                return "park"
            return None
        if self.seat is not None and self.stalled_at is not None and now - self.stalled_at >= self.grace_s:
            self.seat = None
            self.lost_at = None
            self.hold = False
            self.stalled_at = None
            return None
        # Parked with a present, un-held seat holder whose stream is live:
        # re-engage (the stall ended).
        if (
            self.seat is not None
            and self.lost_at is None
            and not self.hold
            and self.last_action_at is not None
            and now - self.last_action_at <= _LIVE_ACTION_S
        ):
            return "engage"
        return None


def soft_start_blend(elapsed_s: float, duration_s: float = SOFT_START_S) -> float:
    """0 → 1 over `duration_s` with a smoothstep, so the follower eases out
    of its present pose instead of stepping toward the leader's."""
    if duration_s <= 0 or elapsed_s >= duration_s:
        return 1.0
    x = max(0.0, elapsed_s / duration_s)
    return x * x * (3.0 - 2.0 * x)


def blend_action(start: dict[str, float], target: dict[str, float], blend: float) -> dict[str, float]:
    """Per-key interpolation from the follower's present pose to the leader's
    action; keys the start pose lacks pass through untouched."""
    out: dict[str, float] = {}
    for key, value in target.items():
        base = start.get(key)
        out[key] = value if base is None else base + (value - base) * blend
    return out


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
    per-read fields (``url``, ``active_operator``, ``phase``,
    ``station_mode``) the status handler fills."""
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
    recalibration — a headless worker would hang on input()). The identity
    guard, calibration write and the torque-limit/velocity resets are the
    SO-101 preflights.
    """
    from lerobot.robots import make_robot_from_config

    from .record import _build_camera_configs, _platform_backend

    camera_configs = _build_camera_configs(request.cameras, _platform_backend())
    robot = make_robot_from_config(build_follower_config(request, cameras=camera_configs))
    try:
        logger.info(
            f"Connecting to the follower arm(s) on {request.follower_port} with {len(camera_configs)} camera(s)..."
        )
        try:
            robot.connect(calibrate=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the follower arm on {request.follower_port} (or one of its cameras). "
                "Make sure it's plugged in and powered on, and that no browser preview holds the camera."
            ) from e
        warnings = verify_devices(((robot, "follower"),), skip=request.skip_identity_check)
        warnings += reset_torque_limit(robot, FOLLOWER)
        warnings += clear_goal_velocity(robot, FOLLOWER)
        return robot, warnings
    except Exception:
        force_disconnect_partial(robot, "follower arm")
        raise


def _release_follower(robot) -> str | None:
    problems = force_disable_torque(robot, "follower arm")
    error = _safe_disconnect(robot, "follower arm")
    if error:
        problems.append(error)
    return " ".join(problems) if problems else None


def _set_torque(robot, on: bool) -> None:
    for bus in _device_buses(robot):
        if on:
            bus.enable_torque()
        else:
            bus.disable_torque()


def _energize_at_present(robot) -> dict[str, float]:
    """Torque on WITHOUT a snap: write the present pose as the goal first,
    while the motors are still limp, then enable torque (a Feetech servo
    holds its stale Goal_Position the instant torque comes back — see
    dagger_runner._restore_torque). Refuses to energize if the pose can't be
    read. Returns the present pose, the soft start's origin."""
    observation = robot.get_observation()
    present = {k: float(v) for k, v in observation.items() if k.endswith(".pos")}
    if not present:
        raise RuntimeError("could not read the follower's pose before energizing")
    robot.send_action(present)
    _set_torque(robot, True)
    return present


def _calibrations_by_prefix(robot, request: HostingRequest) -> dict[str, dict]:
    if request.mode == "bimanual":
        return {"left_": robot.left_arm.calibration, "right_": robot.right_arm.calibration}
    return {"": getattr(robot, "calibration", None) or {}}


def _set_phase(new_phase: str) -> None:
    global phase
    phase = new_phase
    # The phase is a session phase: every WS client refetches the status.
    notify_session_changed("hosting", True, phase=new_phase)


# --- session ----------------------------------------------------------------


def handle_start_hosting(request: HostingRequest, websocket_manager=None) -> dict[str, Any]:
    """Claim the hardware, connect the follower + cameras, join the room
    parked, and run the seat/engage state machine from a worker thread.
    Connect failures are reported synchronously (nothing is claimed)."""
    global hosting_active, hosting_thread, releasing, current_descriptor, current_teleop, seat, phase
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
        # Preconditions that are NOT hardware, checked before the claim so a
        # refusal never emits a session event.
        if not uses_feetech_bus(request.arm_type):
            return {
                "success": False,
                "status_code": 400,
                "message": "Hosting supports the SO-101 in this release: parking toggles torque mid-session, "
                "and the CAN arms need their own treatment.",
                "code": ErrorCode.ROBOT_NOT_READY,
            }
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
        phase = "parked"
        current_descriptor = None
        _release_now.clear()

    notify_session_changed("hosting", True, phase="parked")

    robot = None
    teleop = None
    events: queue.Queue = queue.Queue()
    try:
        from lerobot_teleoperator_livekit import LiveKitTeleoperator, LiveKitTeleoperatorConfig
        from livekit.portal import VideoCodec

        robot, warnings = _connect_follower(request)

        # Wire contract from the connected robot: what lerobot says it has.
        motors, cameras = split_features(dict(robot.observation_features))
        ranges = joint_ranges_deg(_calibrations_by_prefix(robot, request))
        room = sfu.default_room(get_instance_id())
        api_key, api_secret = sfu.api_keys()
        token, _expires = sfu.mint_token(
            api_key=api_key,
            api_secret=api_secret,
            identity=ROBOT_IDENTITY,
            room=room,
            role="robot",
            max_participants=ROOM_MAX_PARTICIPANTS,
        )
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
        # Room events and RPCs arrive on Portal's loop thread; they only
        # enqueue — the worker thread is the one that touches the bus.
        portal = teleop._portal  # the plugin keeps it private; see module docstring
        portal.on_operator_joined(lambda identity: events.put(("joined", identity)))
        portal.on_operator_left(lambda identity: events.put(("left", identity)))
        for rpc in ("home", "engage", "release"):

            def _handler(data, _name=rpc) -> str:
                events.put((_name, data.caller_identity))
                return "ok"

            portal.register_rpc_method(rpc, _handler)
        descriptor = build_descriptor(request, room=room, motors=motors, cameras=cameras, ranges_deg=ranges)

        # Rest pose = where the arm is now (the station places it resting
        # before hosting starts); parked means torque off right there. The
        # gripper is excluded from the return (it may hold something).
        follower_rest_poses = [
            (bus, {m: v for m, v in capture_rest_pose(bus).items() if m != "gripper"})
            for bus in _device_buses(robot)
        ]
        _set_torque(robot, False)

        with _state_lock:
            current_descriptor = descriptor
            current_teleop = teleop
            seat = SeatMonitor()
    except Exception as e:
        logger.error(f"Hosting setup failed: {e}")
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as disconnect_error:
                logger.warning(f"Portal disconnect after a failed start: {disconnect_error}")
        cleanup_error = _release_follower(robot) if robot is not None else None
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
        global hosting_active, releasing, current_descriptor, current_teleop, seat, phase
        global last_cleanup_error, last_session_outcome, last_session_error
        period = 1.0 / max(1, request.fps)
        is_bimanual = request.mode == "bimanual"
        monitor = seat
        stopped_normally = False
        loop_error: str | None = None
        last_broadcast = 0.0
        last_action_ts: int | None = None
        soft_start: tuple[dict[str, float], float] | None = None  # (origin pose, t0)

        def engage() -> None:
            nonlocal soft_start
            if phase in ("engaged", "engaging"):
                return
            _set_phase("engaging")
            origin = _energize_at_present(robot)
            soft_start = (origin, time.monotonic())
            logger.info(f"Engaged for operator {monitor.seat!r} (soft start {SOFT_START_S:.1f}s)")

        def park(reason: str) -> None:
            nonlocal soft_start
            soft_start = None
            if phase == "parked":
                return
            _set_phase("parking")
            logger.info(f"Parking: {reason}")
            _return_followers_to_rest(follower_rest_poses, _release_now)
            _set_torque(robot, False)
            _set_phase("parked")

        def set_active(identity: str | None) -> None:
            try:
                teleop._run(portal.set_active_operator(identity))
            except Exception as e:
                logger.warning(f"set_active_operator({identity!r}) failed: {e}")

        try:
            while hosting_active:
                tick = time.monotonic()
                observation = robot.get_observation()
                teleop.send_feedback(observation)

                # Latest action, with the sender's clock so a still leader
                # (identical values every frame) never reads as a stall.
                raw = portal.get_action()
                action: dict[str, float] | None = None
                if raw is not None and raw.timestamp_us != last_action_ts:
                    last_action_ts = raw.timestamp_us
                    monitor.action_received()
                    action = teleop.get_action() or None

                # Room events + RPCs → seat policy → decisions.
                decisions: list[str] = []
                while True:
                    try:
                        name, identity = events.get_nowait()
                    except queue.Empty:
                        break
                    if name == "joined":
                        had_seat = monitor.seat
                        decision = monitor.operator_joined(identity)
                        if decision and had_seat != identity:
                            set_active(identity)
                            logger.info(f"Operator {identity!r} took the seat")
                    elif name == "left":
                        monitor.operator_left(identity)
                        decision = None
                        logger.info(f"Operator {identity!r} left (grace {GRACE_S:.0f}s)")
                    else:
                        decision = monitor.command(name, identity)
                        if name == "release" and decision:
                            set_active(None)
                            logger.info(f"Operator {identity!r} released the seat")
                    if decision:
                        decisions.append(decision)
                seat_before = monitor.seat
                timed = monitor.tick(engaged=phase in ("engaged", "engaging"))
                if timed:
                    decisions.append(timed)
                if seat_before is not None and monitor.seat is None and timed is not None:
                    set_active(None)
                    logger.info(f"Operator {seat_before!r} did not return within {GRACE_S:.0f}s — seat freed")
                elif seat_before is not None and monitor.seat is None:
                    set_active(None)

                if "park" in decisions:
                    park("operator command or loss")
                elif "engage" in decisions:
                    engage()

                if action and phase in ("engaging", "engaged"):
                    if soft_start is not None:
                        origin, t0 = soft_start
                        blend = soft_start_blend(time.monotonic() - t0)
                        robot.send_action(blend_action(origin, action, blend))
                        if blend >= 1.0:
                            soft_start = None
                            _set_phase("engaged")
                    else:
                        robot.send_action(action)

                now = time.time()
                if now - last_broadcast >= _BROADCAST_INTERVAL_S:
                    try:
                        joint_data: dict[str, Any] = {"type": "joint_update", "timestamp": now}
                        joint_data["joints"] = observation_to_urdf_joints(
                            observation, ranges, prefix="left_" if is_bimanual else ""
                        )
                        if is_bimanual:
                            joint_data["joints_right"] = observation_to_urdf_joints(
                                observation, ranges, prefix="right_"
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
            if phase != "parked" and not _release_now.is_set():
                releasing = True
                notify_session_changed("hosting", True, phase="releasing")
                _return_followers_to_rest(follower_rest_poses, _release_now)
            cleanup_error = _release_follower(robot)
            with _state_lock:
                last_cleanup_error = cleanup_error
                last_session_error = loop_error or cleanup_error
                last_session_outcome = classify_outcome(stopped_normally, last_session_error)
                hosting_active = False
                releasing = False
                phase = "parked"
                current_descriptor = None
                current_teleop = None
                seat = None
            logger.info("Hosting stopped")
            notify_session_changed("hosting", False)

    hosting_thread = threading.Thread(target=worker, name="hosting", daemon=True)
    hosting_thread.start()
    result: dict[str, Any] = {"success": True, "message": "Hosting started (parked)"}
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def handle_stop_hosting() -> dict[str, Any]:
    """Same two-press contract as teleoperation when the arm is engaged:
    first stop → return to rest then release (reported as releasing); second
    stop during the return → release now. A parked arm is already at rest
    with torque off, so the stop is immediate."""
    global hosting_active, hosting_thread

    worker = hosting_thread
    if hosting_active:
        logger.info("Stop hosting triggered")
        was_parked = phase == "parked"
        hosting_active = False
        if worker is None or not worker.is_alive():
            hosting_thread = None
            return {"success": True, "message": "Hosting stopped"}
        if was_parked:
            worker.join(timeout=5.0)
            hosting_thread = None if not worker.is_alive() else hosting_thread
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


def seat_holder() -> str | None:
    """Identity holding the operator seat, or None (idle, or not hosting)."""
    monitor = seat
    return monitor.seat if (hosting_active and monitor is not None) else None


def yield_for_local(timeout_s: float = 10.0) -> bool:
    """Station mode's "local wins when idle": stop a PARKED, UNSEATED hosting
    session so a flow started at the station can take the hardware, and wait
    for the release. False when hosting is engaged/seated (a held session —
    the caller refuses like any other) or the release did not land in time.
    """
    if not hosting_active:
        return True
    if phase != "parked" or seat_holder() is not None:
        return False
    logger.info("A local session preempts the parked hosting session")
    handle_stop_hosting()
    worker = hosting_thread
    if worker is not None:
        worker.join(timeout=timeout_s)
        return not worker.is_alive()
    return True


def handle_hosting_status(request_host: str) -> dict[str, Any]:
    """GET /api/v1/hosting. The descriptor's URL is derived from the host the
    CALLER reached this API on (sfu.sfu_url) — the address known to be
    routable from where they sit."""
    with _state_lock:
        descriptor = dict(current_descriptor) if current_descriptor else None
        active = hosting_active
        current_phase = phase
    if descriptor is not None:
        descriptor["url"] = sfu.sfu_url(request_host)
        descriptor["active_operator"] = seat_holder()
        descriptor["phase"] = current_phase
        descriptor["station_mode"] = station_mode
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


# --- station mode ------------------------------------------------------------


def hostable_robots() -> list[str]:
    """Saved robots this station could host: follower side set up (the arm
    scope hosting drives) and an SO-101 (this release's family)."""
    from .utils.config import is_robot_record_clean, list_robot_records

    return [
        r["name"]
        for r in list_robot_records()
        if uses_feetech_bus(r.get("arm_type")) and is_robot_record_clean(r, arms="follower")
    ]


def pick_station_robot(remembered: str | None, hostable: list[str]) -> str | None:
    """Which robot a station should host: the remembered choice if it is
    still hostable, else the ONLY hostable robot (a machine with one robot
    needs no picker), else None — the UI chooses."""
    if remembered and remembered in hostable:
        return remembered
    if len(hostable) == 1:
        return hostable[0]
    return None


def set_station_robot(robot: str | None) -> dict[str, Any]:
    """The station UI's choice: remember `robot` (None clears it) and re-arm
    hosting on it. A parked, unseated hosting session of another robot yields
    (the supervisor re-hosts within seconds); an engaged/seated one is a held
    session — refused with session.held so an operator is never dropped by a
    click at the station."""
    global station_robot
    from .utils.config import get_robot_record, is_robot_record_clean, is_valid_robot_name, save_station_robot

    if robot is not None:
        record = get_robot_record(robot) if is_valid_robot_name(robot) else None
        if record is None:
            raise ApiError(404, f"No robot named {robot!r}.", code=ErrorCode.ROBOT_NOT_FOUND)
        if not uses_feetech_bus(record.get("arm_type")):
            raise ApiError(
                400,
                "Hosting supports the SO-101 in this release.",
                code=ErrorCode.ROBOT_NOT_READY,
            )
        if not is_robot_record_clean(record, arms="follower"):
            raise ApiError(
                400,
                f"Robot {robot!r} is not set up for hosting: its follower arm needs a port and a calibration.",
                code=ErrorCode.ROBOT_NOT_READY,
            )
    current = current_descriptor["robot"] if (hosting_active and current_descriptor) else None
    if hosting_active and current != robot and not yield_for_local():
        holder = seat_holder()
        raise ApiError(
            409,
            f"An operator ({holder!r}) is driving {current!r} right now. Change the hosted robot once they leave.",
            code=ErrorCode.SESSION_HELD,
            details={"holder": {"kind": "hosting", "session_id": None}},
        )
    save_station_robot(robot)
    with _state_lock:
        station_robot = robot
    return handle_station_status()


def handle_station_status() -> dict[str, Any]:
    """GET /api/v1/station: the posture, the chosen robot, what could be
    hosted, and whether hosting is up right now."""
    return {
        "station_mode": station_mode,
        "robot": station_robot,
        "hostable": hostable_robots(),
        "hosting_active": hosting_active,
        "phase": phase if hosting_active else None,
    }


def start_station_mode(robot_name: str | None, websocket_manager=None) -> threading.Thread:
    """`--host [robot]`: this machine is a station. A daemon thread re-arms
    hosting of the chosen robot whenever nothing holds the hardware, through
    the normal session front door (owner-less, so no lease — the
    parked/engaged timeouts do the lease's job for an unattended arm).

    The choice: an explicit `--host <robot>` is remembered (station.json); a
    bare `--host` uses the remembered one, else the only hostable robot,
    else nothing until the station's UI picks (set_station_robot). Refusals
    are logged and retried with a backoff, so a camera that is unplugged at
    boot comes back later."""
    global station_mode, station_robot
    from .utils.config import load_station_robot, save_station_robot

    station_mode = True
    if robot_name:
        save_station_robot(robot_name)
    station_robot = robot_name or load_station_robot()

    def supervisor() -> None:
        global station_robot
        from .schemas.sessions import SessionStartBody
        from .sessions import handle_start_session, held_by

        delay = _STATION_RETRY_S
        announced_idle = False
        while station_mode:
            time.sleep(delay)
            delay = _STATION_RETRY_S
            if hosting_active or held_by() is not None:
                continue
            if hosting_thread is not None and hosting_thread.is_alive():
                continue  # previous session still releasing
            try:
                hostable = hostable_robots()
            except Exception as exc:
                logger.warning(f"Station mode: could not list robots: {exc}")
                delay = _STATION_BACKOFF_S
                continue
            chosen = pick_station_robot(station_robot, hostable)
            if chosen is None:
                if not announced_idle:
                    logger.info(
                        "Station mode: no robot to host yet — pick one in the station's UI"
                        + (f" (hostable: {hostable})" if hostable else " (none is set up for hosting)")
                    )
                    announced_idle = True
                delay = _STATION_BACKOFF_S
                continue
            announced_idle = False
            if chosen != station_robot:
                station_robot = chosen
                save_station_robot(chosen)
            try:
                handle_start_session(
                    SessionStartBody(kind="hosting", robot=chosen, options={}), websocket_manager
                )
                logger.info(f"Station mode: hosting {chosen!r}")
            except ApiError as exc:
                logger.warning(
                    f"Station mode: could not host {chosen!r} ({exc.detail}); retrying in {_STATION_BACKOFF_S:.0f}s"
                )
                delay = _STATION_BACKOFF_S
            except Exception as exc:
                logger.exception(f"Station mode: unexpected failure hosting {chosen!r}: {exc}")
                delay = _STATION_BACKOFF_S

    thread = threading.Thread(target=supervisor, name="station-mode", daemon=True)
    thread.start()
    return thread


def stop_station_mode() -> None:
    """Switch the station supervisor off (server shutdown): its next tick
    exits instead of re-arming hosting while the process winds down."""
    global station_mode
    station_mode = False


def stop_hosting_for_shutdown() -> bool:
    """Shutdown's stop for a live hosting session — the same stop its own Stop
    control runs, but WAITED FOR (bounded), because a stop that outlives the
    process it runs in is not a stop. An engaged arm returns to rest first
    (the worker's own park path, ceiling `RETURN_CEILING_S`), then torque is
    released; past the ceiling the release is forced, the way a second Stop
    press does it. Returns True when there was a session to stop."""
    worker = hosting_thread
    if not hosting_active and (worker is None or not worker.is_alive()):
        return False
    handle_stop_hosting()
    worker = hosting_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=RETURN_CEILING_S + 5.0)
        if worker.is_alive():
            logger.warning("Hosting worker did not finish its return within the ceiling — releasing now")
            _release_now.set()
            worker.join(timeout=5.0)
    return True
