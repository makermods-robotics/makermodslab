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

"""Physical episode replay: drives the connected follower through a recorded
dataset episode's `action` column, open-loop, real-time paced, no policy and
no cameras involved.

Mirrors `teleoperate.py` in shape — in-process worker thread (there's no
model/vision to isolate, unlike rollout.py's subprocess), single global
session, mutex with every other feature that owns the same serial bus (see
CLAUDE.md's "State model & mutual exclusion"). Connects the follower
*synchronously* (mirrors handle_start_teleoperation, not
handle_start_inference's background-connect — replay's connect is fast, no
multi-minute Hub model download to hide behind a background worker), so a
connection failure is reported straight back to the caller; only the
ease-in + playback loop runs in the background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .api_errors import ErrorCode
from .arm_capabilities import ARM_TYPE_LABEL, arm_type_from_robot_type, uses_feetech_bus
from .arm_identity import verify_devices
from .datasets import get_episode_action_series, read_dataset_robot_type
from .maker_rest_pose import capture_maker_pose, return_maker_to_pose
from .motor_power import FOLLOWER, clear_goal_velocity, reset_torque_limit
from .rest_pose import (
    RETURN_CEILING_S,
    _clamp_to_representable_range,
    capture_rest_pose,
    return_to_rest_pose,
)
from .session_events import notify_session_changed
from .teleoperate import _cleanup_after_setup_failure, force_disable_torque
from .torque import release_maker_torque
from .utils.config import get_robot_record, setup_follower_calibration_file

logger = logging.getLogger(__name__)

# v1 is single-arm only — a bimanual dataset/robot mismatch is a confusing
# error otherwise (left_/right_-prefixed action keys read as "every joint is
# missing"), and bimanual replay needs its own two-bus ease-in/playback shape
# this module doesn't implement yet.
_SINGLE_ARM_MODE = "single"

# Ease-in tolerance, in the SAME normalized units as robot.send_action() (not
# raw ticks — see rest_pose.py's normalize=True path). Numerically smaller
# than RETURN_ARRIVE_TOLERANCE's raw-ticks value (2.0 vs 20), but NOT looser
# in real terms: normalized units are coarser than ticks (roughly 11-19
# ticks per unit on this arm), so 2.0 units is actually ~23-39 ticks —
# looser than RETURN_ARRIVE_TOLERANCE, not tighter. Validate on real
# hardware before trusting this default (see the plan's manual hardware
# verification step) — arm joints and the gripper share this one tolerance
# despite having different normalized ranges (~-100..100 vs 0..100 by
# default), same simplification RETURN_ARRIVE_TOLERANCE already makes across
# differently-geared joints.
EASE_ARRIVE_TOLERANCE = 2.0

# Stall-progress threshold for the ease-in, in the same normalized units as
# EASE_ARRIVE_TOLERANCE — rest_pose.RETURN_STALL_MIN_PROGRESS is in raw
# ticks and would be ~15x too strict a bar here (see return_to_rest_pose's
# docstring). Keeps the same 0.5x-of-arrival-tolerance ratio as the raw-ticks
# defaults (RETURN_STALL_MIN_PROGRESS=10 is half of RETURN_ARRIVE_TOLERANCE=20).
EASE_STALL_MIN_PROGRESS = 1.0

# How far a *settled* ease-in may still be from frame 0 and still be good
# enough to start playback, in the same normalized units as
# EASE_ARRIVE_TOLERANCE (~9-10 deg on the arm joints).
#
# EASE_ARRIVE_TOLERANCE's 2.0 units is ~22-39 ticks (~2-3 deg), which sits
# INSIDE the standing error an STS3215 holds in position mode at lerobot's
# P=16 — and frame 0 of a recorded episode routinely parks joints against
# their calibrated endpoints (shoulder_lift recorded at -103 and clamped to
# the -100 hard stop, elbow near +96, a gripper under a torque cap), which is
# exactly where that error is largest. Teleop and record run the same return
# loop and merely LOG a "settled" verdict; replay's ease-in was the one place
# it was fatal, and exactness doesn't matter here: the ease-in exists only so
# frame 0 doesn't snap the arm from wherever it happened to be, so a joint a
# few units short just means the first send_action moves it a few more
# degrees. A `settled` whose largest per-joint |present - target| is at or
# under this bound is therefore accepted (loudly) and playback starts; a
# `settled` with a LARGER residual — and every `stalled` / `ceiling` /
# `comm-error` / `no-pose` verdict — stays fatal, because those mean a stuck
# or latched joint, or an arm posed genuinely far from frame 0.
EASE_SETTLED_MAX_RESIDUAL = 10.0

# How far behind real time a frame may be before it is dropped rather than
# fired late. Below this, ordinary jitter is absorbed by the pacing wait; above
# it the frame is stale and sending it would only stream an out-of-date setpoint
# at uncapped speed. ~1.5 frames at the 30fps these datasets record at.
_MAX_FRAME_LAG_S = 0.05

# Retries for the idempotent write_calibration + configure pair at connect, to
# absorb a dropped serial packet (see _connect_follower). Deliberately small: a
# genuinely dead bus should still fail fast rather than stall the caller.
_CONNECT_ATTEMPTS = 3
_CONNECT_RETRY_DELAY_S = 0.25

# Bound on how long stop_and_wait() waits for the worker to actually finish
# releasing the arm before forcing an immediate release — mirrors
# teleoperate.py's _STOP_AND_WAIT_TIMEOUT_S shape (RETURN_CEILING_S is
# rest_pose.py's own poll-loop ceiling), but budgets TWO returns rather than
# one: unlike teleoperation, a SIGTERM landing during replay's ease-in can
# owe the ease-in's own RETURN_CEILING_S-bounded return-to-episode-start AND
# THEN the stopping-phase's RETURN_CEILING_S-bounded return-to-session-start
# in series (see _replay_worker's ease-in fall-through). The abort_event fix
# makes the ease-in leg near-instant in the common case, but this is the
# last-resort guard for precisely the case where something didn't abort as
# designed — size it for the worst case, not the common one.
_STOP_AND_WAIT_TIMEOUT_S = 2 * RETURN_CEILING_S + 5.0


class ReplayRequest(BaseModel):
    repo_id: str
    episode_index: int
    follower_port: str
    follower_config: str
    # Robot record name, used only to look up the record's `mode` for the
    # bimanual-rejection guard — the port/config above are what actually
    # drive the connection.
    robot_name: str = ""
    # Hardware family: "so101" or "maker" — selects the follower config class
    # and the calibration library the follower_config name resolves in.
    arm_type: str = "so101"
    skip_identity_check: bool = False


replay_active: bool = False
replay_thread: threading.Thread | None = None
_state_lock = threading.Lock()
_replay_started_at: float | None = None
# Module-scoped (not local to _replay_worker) so handle_stop_replay can
# actually signal it — a stop_event that only the worker itself could see or
# set left a stop pressed during the ease-in with nothing to notice it. Reset
# at the start of every new session (see handle_start_replay).
_stop_event = threading.Event()
# {phase, episode_index, elapsed_s, duration_s, error, hint} — see
# handle_replay_status. phase is one of: idle | easing_in | playing |
# stopping | done | error.
_replay_meta: dict[str, Any] = {}


def _load_robot_record(robot_name: str) -> dict[str, Any] | None:
    """Thin wrapper so tests can monkeypatch the record lookup without
    touching disk — get_robot_record already does the real lookup."""
    return get_robot_record(robot_name)


def handle_start_replay(request: ReplayRequest, websocket_manager=None) -> dict[str, Any]:
    """Validate, connect the follower synchronously, and spawn the
    ease-in + playback worker. Returns a dict the route layer turns into a
    JSON response or HTTPException (status_code key present on every
    failure), mirroring rollout.py's InferenceRequest response shape."""
    global replay_active, replay_thread, _replay_started_at, _replay_meta

    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        remote_host as _remote_host,
        remote_teleoperate as _remote_teleoperate,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    with _state_lock:
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_TELEOPERATION,
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_RECORDING,
            }
        if _rollout.inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_INFERENCE,
            }
        if _calibrate.calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Calibration is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_CALIBRATION,
            }
        if _auto_calibrate.auto_calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Auto-calibration is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
            }
        if _wiggle.wiggle_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
                "code": ErrorCode.ROBOT_BUSY_WIGGLE,
            }
        if _remote_host.hosting_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "This robot is hosted for remote teleoperation. Stop hosting first.",
                "code": ErrorCode.ROBOT_BUSY_HOSTING,
            }
        if _remote_teleoperate.remote_teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Remote teleoperation is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_REMOTE_TELEOPERATION,
            }
        if replay_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Replay is already active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_REPLAY,
            }
        if replay_thread is not None and replay_thread.is_alive():
            return {
                "success": False,
                "status_code": 409,
                "message": "The previous replay session is still shutting down. Try again in a few seconds.",
                "code": ErrorCode.ROBOT_BUSY_RELEASING,
            }

        # Lazy, because jobs imports this module back the same way. Replay was
        # the one robot feature PR #83 left out of the training mutex; it
        # drives the follower over the same USB bus every other feature does.
        from . import jobs as _jobs

        if (training := _jobs.training_is_active()) is not None:
            return {
                "success": False,
                "status_code": 409,
                "message": f"Training run '{training}' is using this machine. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_TRAINING,
            }

        record = _load_robot_record(request.robot_name)
        if record is not None and record.get("mode") == "bimanual":
            return {
                "success": False,
                "status_code": 400,
                "message": "Bimanual replay isn't supported yet — select a single-arm robot.",
            }
        action_series = get_episode_action_series(request.repo_id, request.episode_index)
        if action_series is None:
            return {
                "success": False,
                "status_code": 400,
                "message": "Could not read this episode's recorded actions — it may not be downloaded locally yet.",
            }

        # Joint NAMES can match while the arm family doesn't: Maker and Metal
        # share all seven, and only their units differ, so neither the
        # action_features comparison below nor the frame-0 bus keying can tell
        # them apart. lerobot tags the recording robot in the dataset's
        # meta/info.json, so check that FIRST — before _connect_follower, so a
        # refused replay never energizes the arm.
        #
        # An untagged, imported or community dataset resolves to None, and that
        # stays SILENT (same rule the merge / cross-arm fine-tune warnings
        # follow): "not established" is not "mismatched", and the joint-name
        # checks below remain the guard for those, exactly as before.
        dataset_arm = arm_type_from_robot_type(read_dataset_robot_type(request.repo_id))
        if dataset_arm is not None and dataset_arm != request.arm_type:
            robot_label = ARM_TYPE_LABEL.get(request.arm_type, f"a {request.arm_type} arm")
            return {
                "success": False,
                "status_code": 400,
                "message": (
                    f"This dataset was recorded on {ARM_TYPE_LABEL[dataset_arm]}, but the "
                    f"connected robot is {robot_label} — replay it on a matching robot."
                ),
            }

        try:
            robot, identity_warnings = _connect_follower(request)
        except Exception as e:
            return {"success": False, "status_code": 500, "message": str(e)}

        if set(action_series["action_names"]) != set(robot.action_features.keys()):
            _cleanup_after_setup_failure(robot, None, "follower arm", "leader arm")
            return {
                "success": False,
                "status_code": 400,
                "message": "This dataset's joints don't match the connected robot.",
            }

        # The check above compares against robot.action_features, which is
        # `<motor>.pos`-suffixed — it therefore cannot catch a target that
        # re-keys onto zero bus motors. Verify the ease-in target the worker
        # will actually build covers every motor, so a naming mismatch fails
        # here with a clear message instead of surfacing as a bare "no-pose"
        # from the worker after the arm has already been energized.
        if action_series["values"]:
            frame0 = dict(zip(action_series["action_names"], action_series["values"][0], strict=True))
            missing = set(robot.bus.motors) - set(_bus_keyed(frame0, robot.bus))
            if missing:
                _cleanup_after_setup_failure(robot, None, "follower arm", "leader arm")
                return {
                    "success": False,
                    "status_code": 400,
                    "message": (
                        "This dataset's joint names don't map onto the connected robot's motors "
                        f"({', '.join(sorted(missing))})."
                    ),
                }

        replay_active = True
        _stop_event.clear()
        _replay_started_at = time.time()
        _replay_meta = {
            "phase": "easing_in",
            "episode_index": request.episode_index,
            "duration_s": action_series["timestamps"][-1] if action_series["timestamps"] else 0.0,
            "error": None,
            "hint": None,
        }

    # The claim above is the real state transition — broadcast the hint so
    # every WS client (any page, any remote UI) refetches /replay-status.
    notify_session_changed("replay", True, phase="easing_in")

    worker = threading.Thread(
        target=_replay_worker,
        args=(robot, action_series, websocket_manager, request.arm_type),
        name="replay-worker",
        daemon=True,
    )
    replay_thread = worker
    worker.start()

    response: dict[str, Any] = {"success": True, "message": "Replay starting"}
    if identity_warnings:
        response["warning"] = " ".join(identity_warnings)
    return response


def _connect_can_follower(request: ReplayRequest):
    """Connect a CAN follower (Maker or Metal) for replay.

    Much shorter than the SO-101 path below because the CAN followers'
    connect() already does everything that path hand-rolls — bus open,
    calibration load and registration, MIT gain write, torque enable.
    `calibrate=False` for the same reason teleoperate._connect_can passes it:
    the default would drop an uncalibrated device into an `input()`-blocked
    calibrate() and hang this thread forever. A failed connect is
    de-energized explicitly — load-bearing on Metal, whose handshake IS the
    motor enable command and whose connect() has no cleanup of its own.

    Connecting also RE-ARMS the arm's slow initial sync (`_synced = False`),
    which keeps the arm's own slow initial sync as a second safety net under
    the rate-bounded approach maker_rest_pose already applies.
    """
    from lerobot.robots import make_robot_from_config

    from .torque import de_energize_can_device
    from .utils.robot_factory import maker_follower_config, metal_follower_config

    is_metal = request.arm_type == "metal"
    family = "Metal" if is_metal else "Maker"
    builder = metal_follower_config if is_metal else maker_follower_config
    follower_id = setup_follower_calibration_file(request.follower_config, request.arm_type)
    robot = make_robot_from_config(builder(request.follower_port, follower_id))
    try:
        robot.connect(calibrate=False)
    except Exception as e:
        de_energize_can_device(robot, f"{family} follower arm")
        raise RuntimeError(
            f"Could not connect to the {family} follower arm on {request.follower_port}. "
            "Check that the CAN adapter is plugged in, the arm is powered, and the "
            "motors are in MIT mode, then try again."
        ) from e
    # No identity fingerprint, torque-limit reset or speed-cap clear: all three
    # are Feetech register operations (see arm_capabilities.uses_feetech_bus),
    # and a RobStride arm's drive effort comes from the MIT gains connect()
    # just wrote instead.
    return robot, []


def _connect_follower(request: ReplayRequest):
    """Connect and configure the follower for replay — mirrors
    teleoperate.py's single-arm connect sequence (write_calibration →
    configure → reset_torque_limit → clear_goal_velocity), follower-only.
    Raises on a connection or hard identity-mismatch failure; the caller
    (handle_start_replay) is responsible for cleanup on that path."""
    if not uses_feetech_bus(request.arm_type):
        return _connect_can_follower(request)

    follower_id = setup_follower_calibration_file(request.follower_config, request.arm_type)
    robot = SO101Follower(SO101FollowerConfig(port=request.follower_port, id=follower_id))
    try:
        robot.bus.connect()
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to the follower arm on {request.follower_port}. "
            "Make sure it's plugged in and powered on, then try again."
        ) from e

    identity_warnings = verify_devices(((robot, "follower"),), skip=request.skip_identity_check)

    # A dropped serial packet during configure() ("Failed to write 'Lock' ...
    # no status packet") turned roughly one start in twenty into a hard 500,
    # on a bus that was otherwise healthy — lerobot's configure_motors() retries
    # only once internally. Both steps are idempotent register writes, so retry
    # the pair rather than surfacing a transient glitch as a failed start.
    for attempt in range(_CONNECT_ATTEMPTS):
        try:
            robot.bus.write_calibration(robot.calibration)
            robot.configure()
            break
        except Exception as e:
            if attempt == _CONNECT_ATTEMPTS - 1:
                raise RuntimeError(
                    f"The follower arm on {request.follower_port} stopped responding while being "
                    f"configured ({e}). Check the power and the daisy-chain cabling, then try again."
                ) from e
            logger.warning(
                "Follower configure failed (%s); retrying (%d/%d)", e, attempt + 2, _CONNECT_ATTEMPTS
            )
            time.sleep(_CONNECT_RETRY_DELAY_S)

    identity_warnings += reset_torque_limit(robot, FOLLOWER)
    identity_warnings += clear_goal_velocity(robot, FOLLOWER)
    return robot, identity_warnings


def _bus_keyed(action: dict[str, float], bus) -> dict[str, float]:
    """Re-key an action dict onto the plain motor names the bus uses.

    A dataset's `action` feature names carry lerobot's robot-level `<motor>.pos`
    suffix (SO101Follower._motors_ft), but `bus.motors` — and therefore
    rest_pose's target filtering — is keyed by bare motor name. Passing an
    unconverted action dict straight to return_to_rest_pose matches zero motors
    and yields "no-pose" without ever writing a Goal_Position. This mirrors what
    robot.send_action() already does internally at the same robot→bus boundary
    (`key.removesuffix(".pos")`); the bare-name fallback keeps a dataset that
    recorded unsuffixed names working too.
    """
    keyed: dict[str, float] = {}
    for motor in bus.motors:
        for key in (f"{motor}.pos", motor):
            if key in action:
                keyed[motor] = action[key]
                break
    return keyed


def _ease_in_residual(bus, targets: dict[str, float]) -> dict[str, float] | None:
    """Per-joint |present - target| after a `settled` ease-in, normalized units.

    Read fresh rather than parsed out of return_to_rest_pose's reason string:
    the reason is prose meant for humans, and (bool, str) is the contract every
    teleop/record caller of return_to_rest_pose depends on, so the verdict is
    not the place to smuggle numbers through. Compares against the SAME clamped
    targets the return itself drove to (_clamp_to_representable_range) — a
    recorded action beyond a motor's calibrated span is clamped on both the
    write and the read path, so an unclamped comparison would measure a joint
    resting perfectly on its hard stop as several units off.

    Returns None when the read fails or reports no motor we have a target for;
    the caller must treat that as "unknown", i.e. keep the fatal path.
    """
    clamped = _clamp_to_representable_range(getattr(bus, "motors", None) or {}, targets)
    try:
        positions = bus.sync_read("Present_Position", normalize=True)
    except Exception as e:
        logger.warning(f"Could not measure how far the ease-in settled from the first frame: {e}")
        return None
    residual = {
        motor: abs(float(positions[motor]) - float(target))
        for motor, target in clamped.items()
        if motor in positions
    }
    return residual or None


def _ensure_uncapped(robot: SO101Follower, label: str) -> None:
    """Clear the RAM Goal_Velocity profile cap and verify it actually took.

    clear_goal_velocity (like rest_pose's own restore) is deliberately
    best-effort so a failed write can't abort a session — but for playback a
    silently-surviving cap is not a degraded nicety, it throttles every move of
    the whole episode. Read back and retry once, and say so loudly if a cap
    survives, rather than replaying at a fraction of the recorded speed with no
    indication why.
    """
    clear_goal_velocity(robot, FOLLOWER, label)
    try:
        caps = robot.bus.sync_read("Goal_Velocity", normalize=False)
    except Exception as e:
        logger.warning(f"Could not verify the speed cap was cleared for the {label}: {e}")
        return
    stuck = {m: v for m, v in caps.items() if v}
    if not stuck:
        return
    logger.warning(f"Speed cap survived on the {label} ({stuck}) — retrying before playback")
    clear_goal_velocity(robot, FOLLOWER, label)
    try:
        still = {m: v for m, v in robot.bus.sync_read("Goal_Velocity", normalize=False).items() if v}
        if still:
            logger.error(
                f"The {label} is still speed-capped ({still}); this replay will run slower than "
                "recorded. A power cycle clears the register if the write keeps failing."
            )
    except Exception:
        pass


def _replay_worker(
    robot,
    action_series: dict[str, Any],
    websocket_manager,
    arm_type: str = "so101",
) -> None:
    """Ease the arm to the episode's first frame, then stream send_action()
    calls at the episode's recorded pace, broadcasting live joint feedback
    over the existing joint-data websocket every frame. On stop or
    completion, gently return to the pose captured at the start and release
    torque — mirrors teleoperation_worker's graceful-stop shape exactly.

    `arm_type` selects the ease-in and teardown machinery: an SO-101 gets the
    Feetech profile-velocity return and an explicit torque release, a Maker arm
    gets the interpolated MIT setpoint in maker_rest_pose.return_maker_to_pose.
    The PLAYBACK loop between them is identical for both — it is plain
    `send_action` on the dataset's action column, exactly as lerobot's own
    arm-agnostic `lerobot-replay` does it."""
    global replay_active

    action_names = action_series["action_names"]
    timestamps = action_series["timestamps"]
    frames = action_series["values"]

    def _stop_check() -> bool:
        return not replay_active or _stop_event.is_set()

    # Both arm types capture where the session started and are driven back to
    # it before torque is released — a Maker arm has no brakes, so releasing it
    # mid-episode drops it. Only the mechanism differs: a Feetech profile
    # velocity for the SO-101 (rest_pose.py), an interpolated MIT setpoint for
    # the Maker arm (maker_rest_pose.py). The Maker capture INCLUDES the
    # gripper, matching what capture_rest_pose does here for the SO-101 —
    # replay drives the gripper from the dataset, so its start width is part of
    # the pose being restored.
    feetech = uses_feetech_bus(arm_type)
    if feetech:
        start_pose = capture_rest_pose(robot.bus, normalize=False)
    else:
        start_pose = capture_maker_pose(robot, include_gripper=True)

    try:
        if frames:
            frame0 = dict(zip(action_names, frames[0], strict=True))
            if feetech:
                arrived, reason = return_to_rest_pose(
                    robot.bus,
                    _bus_keyed(frame0, robot.bus),
                    abort_event=_stop_event,
                    label="follower arm",
                    normalize=True,
                    tolerance=EASE_ARRIVE_TOLERANCE,
                    stall_min_progress=EASE_STALL_MIN_PROGRESS,
                )
            else:
                # Same primitive as the stop return below, in the other
                # direction. It bounds the RATE rather than trusting the arm's
                # startup sync, so it behaves identically whether or not that
                # sync is still armed — and the convergence check is what makes
                # it tolerate the standing error a healthy joint holds.
                arrived, reason = return_maker_to_pose(
                    robot,
                    {k.removesuffix(".pos"): v for k, v in frame0.items()},
                    abort_event=_stop_event,
                    label="follower arm",
                )
            # A `settled` verdict says every motor STOPPED short of target, which
            # on this arm is the ordinary outcome of easing into a frame 0 that
            # parks joints on their endpoints — not a fault. Measure how short it
            # actually stopped and start playback anyway when that is within
            # EASE_SETTLED_MAX_RESIDUAL (Feetech only: the CAN return already
            # judges by convergence and has no equivalent verdict).
            settled_too_far: tuple[str, float] | None = None
            if not arrived and feetech and reason.startswith("settled"):
                residual = _ease_in_residual(robot.bus, _bus_keyed(frame0, robot.bus))
                if residual is not None:
                    worst_joint, worst = max(residual.items(), key=lambda item: item[1])
                    if worst <= EASE_SETTLED_MAX_RESIDUAL:
                        logger.warning(
                            "Ease-in settled %.1f normalized units short of the episode's first "
                            "frame on %s (%s), within the %.1f-unit bound — starting playback; the "
                            "first frames will carry those joints the rest of the way",
                            worst,
                            worst_joint,
                            ", ".join(f"{m}={d:.1f}" for m, d in sorted(residual.items())),
                            EASE_SETTLED_MAX_RESIDUAL,
                        )
                        arrived = True
                    else:
                        settled_too_far = (worst_joint, worst)

            if not arrived and reason != "cut-short":
                with _state_lock:
                    _replay_meta["phase"] = "error"
                    _replay_meta["error"] = f"Could not reach the episode's starting position ({reason})"
                    # Surface the actual per-motor reason (return_to_rest_pose
                    # already names the joint and its delta) instead of a
                    # generic guess — the frontend shows hint in preference to
                    # error, so a static "may be posed too far" message was
                    # hiding the real diagnosis from the user.
                    if settled_too_far is not None:
                        worst_joint, worst = settled_too_far
                        _replay_meta["hint"] = (
                            "The arm didn't settle at this episode's starting position: "
                            f"{worst_joint} stopped {worst:.1f} normalized units away, past the "
                            f"{EASE_SETTLED_MAX_RESIDUAL:.0f}-unit bound for starting playback "
                            f"({reason}). Try again, or reposition it closer first."
                        )
                    else:
                        _replay_meta["hint"] = (
                            f"The arm didn't settle at this episode's starting position ({reason}). "
                            "Try again, or reposition it closer first."
                        )
                # Fall through to the graceful return below instead of
                # returning here — an ease-in that didn't arrive still left
                # the arm mid-air under torque, and the drop straight to
                # force_disable_torque in `finally` is exactly what the return
                # below exists to avoid.
            elif reason == "cut-short" or _stop_check():
                # A stop landed during (or right as) the ease-in — skip
                # playback and fall through to the same graceful return every
                # other stop path uses, instead of dropping the arm wherever
                # the ease-in left it.
                pass
            else:
                # The ease-in stamps RETURN_POS_SPEED into every motor's RAM
                # Goal_Velocity and clears it again in its own finally — but that
                # restore is best-effort, so a single dropped serial write leaves a
                # 400 profile cap throttling the ENTIRE playback (slow but
                # accurate-looking, with nothing in the UI to say why). Re-clear and
                # verify here so playback can never inherit the ease-in's cap.
                # Feetech only: the ease-in above stamps a RAM Goal_Velocity
                # profile cap that must not survive into playback. The Maker
                # ease-in writes no such register — it borrows the arm's own
                # startup sync — so there is nothing to clear.
                if feetech:
                    _ensure_uncapped(robot, "follower arm")

                with _state_lock:
                    _replay_meta["phase"] = "playing"
                notify_session_changed("replay", True, phase="playing")

                t0 = time.monotonic()
                skipped = 0
                last_i = len(timestamps) - 1
                for i, (ts, values) in enumerate(zip(timestamps, frames, strict=True)):
                    if _stop_check():
                        break
                    target_t = t0 + ts
                    now = time.monotonic()
                    # Falling behind (a slow bus read, a GC pause, a serial retry)
                    # used to make every overdue frame fire back-to-back with no
                    # wait, streaming far-apart setpoints at uncapped speed — the
                    # arm lurches through them instead of following the recorded
                    # path. Drop stale frames instead: the arm is then always
                    # driven to the setpoint that matches wall-clock time. The
                    # final frame is never dropped, so the episode still ends on
                    # its recorded pose.
                    if now > target_t + _MAX_FRAME_LAG_S and i != last_i:
                        skipped += 1
                        continue
                    while True:
                        now = time.monotonic()
                        if now >= target_t or _stop_check():
                            break
                        time.sleep(min(0.01, target_t - now))
                    if _stop_check():
                        break

                    action = dict(zip(action_names, values, strict=True))
                    robot.send_action(action)

                    with _state_lock:
                        _replay_meta["elapsed_s"] = time.monotonic() - t0

                    if websocket_manager is not None and getattr(
                        websocket_manager, "active_connections", None
                    ):
                        try:
                            observation = robot.get_observation()
                            websocket_manager.broadcast_joint_data_sync(
                                {
                                    "type": "replay_joint_update",
                                    "joints": observation,
                                    "timestamp": time.time(),
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Could not broadcast replay joint feedback: {e}")

                played = time.monotonic() - t0
                if skipped:
                    logger.warning(
                        "Replay dropped %d/%d frames to stay in real time (played %.2fs vs %.2fs recorded) "
                        "— the control loop could not keep up",
                        skipped,
                        len(timestamps),
                        played,
                        timestamps[-1],
                    )
                else:
                    logger.info(
                        "Replay played %d frames in %.2fs (recorded %.2fs)",
                        len(timestamps),
                        played,
                        timestamps[-1],
                    )
                with _state_lock:
                    _replay_meta["frames_dropped"] = skipped
                    _replay_meta["played_s"] = played

        with _state_lock:
            if _replay_meta.get("phase") != "error":
                _replay_meta["phase"] = "stopping"
        # Playback is over but the arm is still energized for the return —
        # a phase of this session, not idle yet.
        notify_session_changed("replay", True, phase=_replay_meta.get("phase"))
        if feetech:
            return_to_rest_pose(robot.bus, start_pose, label="follower arm")
        else:
            return_maker_to_pose(robot, start_pose, abort_event=_stop_event, label="follower arm")
    except Exception as e:
        logger.error(f"Replay worker error: {e}")
        with _state_lock:
            _replay_meta["phase"] = "error"
            _replay_meta["error"] = str(e)
    finally:
        if feetech:
            force_disable_torque(robot, "follower arm")
            try:
                robot.bus.disconnect(disable_torque=False)
            except Exception as e:
                logger.warning(f"Could not disconnect the follower after replay: {e}")
        else:
            release_maker_torque(robot, "CAN follower arm")
            try:
                # disconnect() (not bus.disconnect(disable_torque=False)):
                # MakerFollower.disconnect honours disable_torque_on_disconnect,
                # and a torque-off settle is this arm's documented safe state.
                robot.disconnect()
            except Exception as e:
                logger.warning(f"Could not disconnect the CAN follower after replay: {e}")
        with _state_lock:
            replay_active = False
            if _replay_meta.get("phase") not in ("error",):
                _replay_meta["phase"] = "done"
            final_phase = _replay_meta.get("phase")
        # Final release: torque released and port freed, including error paths.
        notify_session_changed("replay", False, phase=final_phase)


def handle_replay_status() -> dict[str, Any]:
    with _state_lock:
        # Freeze once the session ends: computing this unconditionally made a
        # finished run keep ticking upward, so a dead session read as live.
        if not replay_active:
            elapsed_s = _replay_meta.get("played_s") or 0.0
        else:
            elapsed_s = time.time() - _replay_started_at if _replay_started_at else 0.0
        return {
            "replay_active": replay_active,
            "phase": _replay_meta.get("phase", "idle"),
            "episode_index": _replay_meta.get("episode_index"),
            "elapsed_s": elapsed_s,
            "duration_s": _replay_meta.get("duration_s"),
            "error": _replay_meta.get("error"),
            "hint": _replay_meta.get("hint"),
            # Fidelity of the run just played: a nonzero drop count or a
            # played_s well past duration_s means the arm did NOT follow the
            # recorded motion, which otherwise looks identical to success.
            "frames_dropped": _replay_meta.get("frames_dropped"),
            "played_s": _replay_meta.get("played_s"),
        }


def handle_stop_replay() -> dict[str, Any]:
    global replay_active
    with _state_lock:
        if not replay_active:
            return {"success": False, "status_code": 409, "message": "No replay is active"}
        replay_active = False
        _stop_event.set()
        _replay_meta["phase"] = "stopping"
    return {"success": True, "message": "Replay stopping"}


def stop_and_wait(timeout: float = _STOP_AND_WAIT_TIMEOUT_S) -> None:
    """Stop replay and block until the worker has actually released the arm
    — for the FastAPI shutdown hook, which has no UI to poll and no "press
    Stop again" gesture available. Mirrors teleoperate.py's stop_and_wait.
    A no-op when idle."""
    if replay_active:
        handle_stop_replay()
    worker = replay_thread
    if worker is None or not worker.is_alive():
        return
    worker.join(timeout=timeout)
    if worker.is_alive():
        logger.warning("Replay worker did not finish releasing within %.0fs", timeout)
