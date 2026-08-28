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

import collections
import json
import logging
import shutil
import threading
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.motors.motors_bus import MotorsBus

# Import the main record functionality to reuse it
from lerobot.scripts.lerobot_record import RecordConfig

from .api_errors import ErrorCode
from .arm_capabilities import arm_type_of_robot_config, uses_feetech_bus
from .arm_identity import ArmIdentityError, verify_devices
from .camera_preview import camera_preview_manager
from .datasets import (
    _lerobot_cache_root,
    invalidate_dataset_listing_cache,
    invalidate_hub_status,
    push_dataset_to_hub,
)
from .motor_power import clear_goal_velocity, reset_torque_limit
from .rest_pose import RETURN_CEILING_S, capture_rest_pose
from .session_events import notify_session_changed
from .teleoperate import (
    _device_buses,
    _return_followers_to_rest,
    force_disable_torque,
    force_disconnect_partial,
)
from .torque import release_maker_torque
from .utils.config import (
    CameraResolutionError,
    load_robot_cameras,
    validate_dataset_repo_id,
    with_makermodslab_tag,
)
from .utils.errors import classify_outcome, format_exception, friendly_hint
from .utils.robot_factory import build_bimanual_configs, build_single_configs

logger = logging.getLogger(__name__)

# Default pixel format for USB cameras when the request doesn't pin one.
# OpenCV/V4L2 otherwise negotiates uncompressed YUYV, whose isochronous USB
# bandwidth exhausts the controller at ~2-3 concurrent cameras — the UVC driver
# then fails STREAMON with the misleading "VIDIOC_STREAMON: No space left on
# device" (ENOSPC = USB bandwidth, not disk), silently dropping a 3rd/4th camera
# from the UI. MJPG is ~10x smaller and lets the full rig stream. macOS already
# negotiates MJPEG, so this only changes Linux behavior. An explicit per-camera
# fourcc (e.g. a deliberate YUYV choice from the UI) still wins.
_DEFAULT_FOURCC = "MJPG"

# --- Motor bus read retries ----------------------------------------------------
# lerobot's SO-10x follower/leader call bus.sync_read("Present_Position") with
# the default num_retry=0, so a single missed reply kills the whole recording
# session ("[TxRxResult] There is no status packet!"). Replies do get missed:
# on hosts where the arm serial adapters share a USB bus with streaming
# cameras (e.g. the Jetson rig, cameras and CH34x adapters on the same hubs),
# isochronous camera traffic occasionally delays a bulk serial reply past the
# packet timeout. Position reads are idempotent, and a retry costs only the
# packet timeout (single-digit ms at 1 Mbps) *when a read actually glitched* —
# invisible at a 30 Hz loop. Patch the class-level default (process-wide);
# an explicit num_retry from any caller still wins.
_BUS_SYNC_READ_RETRIES = 2
_original_sync_read = MotorsBus.sync_read

# Robot-connect attempts before giving up on transient camera turbulence (see
# the retry loop below and connect_retry_attempt above).
_CONNECT_ATTEMPTS = 3


def _sync_read_with_default_retries(
    self, data_name, motors=None, *, normalize=True, num_retry=_BUS_SYNC_READ_RETRIES
):
    return _original_sync_read(self, data_name, motors, normalize=normalize, num_retry=num_retry)


if MotorsBus.sync_read.__name__ != "_sync_read_with_default_retries":  # idempotent under --reload
    MotorsBus.sync_read = _sync_read_with_default_retries

# --- Recording log capture (bounded ring buffer) ------------------------------
# The record flow logs progress through the Python `logger` rather than a
# subprocess, so its output only ever reached the uvicorn console. To surface it
# on the Record page we attach a bounded logging.Handler to the loggers that
# carry recording output while a session is active, and expose the buffer via a
# polled GET endpoint. The deque is capacity-capped (maxlen) so memory can never
# grow unbounded no matter how chatty or long a session is.
_RECORD_LOG_MAX_LINES = 500
# Loggers whose records describe a recording session: this module's logger,
# lerobot's own record/control loggers (the "Recording episode N", save/reset
# lines come from there; attaching to the shared "lerobot" ancestor captures
# any lerobot child logger via propagation), and teleoperate's connect/
# teardown logger — force_disconnect_partial's warnings and torque-disable
# errors otherwise only ever reach the server console, invisible to an
# operator watching the Record page during a failed/retried connect.
_RECORD_LOG_LOGGER_NAMES = (__name__, "lerobot", "makermodslab.teleoperate")


class _RingBufferLogHandler(logging.Handler):
    """A logging.Handler that keeps only the last N formatted records.

    Backed by a `collections.deque(maxlen=...)`, so it is O(1) per record and
    strictly bounded in memory. `snapshot()` returns the current lines; the
    handler's own lock guards concurrent emit/read."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self._buffer: collections.deque[str] = collections.deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        # deque.append is atomic under the GIL; acquire the handler lock anyway
        # to stay consistent with snapshot() and satisfy Handler conventions.
        with self.lock:
            self._buffer.append(line)

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self._buffer)


# The single active handler (None when no session has attached one). Guarded by
# _record_log_lock for attach/detach; the handler's own lock guards its buffer.
_record_log_handler: _RingBufferLogHandler | None = None
_record_log_lock = threading.Lock()


def _attach_record_log_handler() -> None:
    """Start capturing recording logs into a fresh bounded ring buffer.

    Detaches any previous handler first (so a new session starts clean) and
    attaches a new one to the recording-related loggers. Called at the start of
    each session."""
    global _record_log_handler
    with _record_log_lock:
        _detach_record_log_handler_locked()
        handler = _RingBufferLogHandler(_RECORD_LOG_MAX_LINES)
        for name in _RECORD_LOG_LOGGER_NAMES:
            lg = logging.getLogger(name)
            lg.addHandler(handler)
            # Ensure INFO records reach the handler even if the logger's own
            # level would otherwise filter them; never lower it below INFO.
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
        _record_log_handler = handler


def _detach_record_log_handler_locked() -> None:
    """Remove the active handler from all loggers. Caller holds _record_log_lock."""
    global _record_log_handler
    if _record_log_handler is None:
        return
    for name in _RECORD_LOG_LOGGER_NAMES:
        logging.getLogger(name).removeHandler(_record_log_handler)
    # Keep the buffer around (don't null the handler) so the log stays readable
    # after the session ends until the next session attaches a fresh one.


def handle_recording_log(max_lines: int = _RECORD_LOG_MAX_LINES) -> dict[str, Any]:
    """Return the tail of the current/most-recent recording session's log.

    Read-only and bounded: at most `max_lines` trailing lines from the ring
    buffer (itself capped at _RECORD_LOG_MAX_LINES). Never raises — before any
    session has captured logs the buffer is empty, so the route stays 200."""
    with _record_log_lock:
        handler = _record_log_handler
    if handler is None:
        return {"logs": ""}
    lines = handler.snapshot()
    tail = lines[-max_lines:] if max_lines > 0 else lines
    return {"logs": "\n".join(tail)}


# Global variables for recording state
recording_active = False
recording_thread: threading.Thread | None = None
recording_events = None  # Events dict for controlling recording session
recording_config = None  # Store recording configuration
recording_start_time = None  # Track when recording started
session_end_elapsed_seconds = None  # Final session duration after the run ends
current_episode = 1  # Track current episode number
saved_episodes = 0  # Track how many episodes have been saved
current_phase = "preparing"  # Track current phase: "preparing", "recording", "resetting", "completed"
phase_start_time = None  # Track when current phase started


def _set_phase(phase: str) -> None:
    """Record the session phase and broadcast the transition as a
    `session_changed` hint (see makermodslab/session_events.py).

    Pure assignment plus a droppable notification — every consumer refetches
    /recording-status rather than trusting the hint, so this changes no
    behavior. Phase transitions go through here; the flag claim/release sites
    emit their own hints where the active flag actually flips."""
    global current_phase
    current_phase = phase
    notify_session_changed("recording", recording_active, phase=phase)


# Set only while current_phase == "reconnecting_robot" (the teardown and
# backoff sleep between connect retries below); 0 otherwise — including on a
# successful connect and on the terminal failure, so a reader can treat
# "connect_retry_attempt > 0" as "a retry is in flight right now". Lets the
# frontend show which retry attempt that is ("Camera hiccup, retrying (2/3)…")
# instead of one static "Connecting arm & cameras…" label for a window that
# can run past ten seconds across multiple component-wise teardowns.
connect_retry_attempt = 0
# Total time spent paused during the CURRENT reset phase, credited on resume
# (see handle_resume_recording). Reset to 0.0 at the start of every new reset
# phase (both reset-phase call sites in record_with_web_events) AND at the
# following recording-phase entry (top of the while loop in
# record_with_web_events), so a leftover value from a finished reset phase
# never leaks into the next episode's phase_elapsed_seconds. Consumers
# (handle_recording_status) must ALSO gate their use of this value on
# current_phase == "resetting" — the two resets alone aren't sufficient,
# since this is a stateless global any thread's status read can observe
# between the reset-phase end and the next reset-phase start.
paused_accum_seconds: float = 0.0
# Wall-clock time.time() the current pause began; None while not paused. Kept
# separate from the reset loop's own internal perf_counter()-based pacing —
# this pair exists purely so handle_recording_status can freeze the
# frontend-facing countdown, independent of how the tick loop paces itself.
pause_started_at: float | None = None
# True when the most recent session saved zero episodes and its (freshly
# created) dataset directory was discarded. Surfaced in the session-end status
# so the frontend can tell the user nothing was kept (see Upload.tsx).
last_session_discarded_empty = False
# Terminal error taxonomy of the most recent session, surfaced with the
# session-end status (the in-process twin of rollout's exited payload).
# `last_session_outcome` is "ok" | "ran_with_warning" | "failed" (None before
# any session ends); `last_session_error` is the caught exception formatted as
# "Type: message" — recording runs in-process, so the worker's catch site holds
# the actual exception object and no log forensics are needed. Classified by
# catch site: an exception AFTER the recording loop finished its real work
# (current_phase already "completed" — episodes saved / stop honored) means
# only teardown tripped, which must NOT be reported as a failed session.
last_session_outcome: str | None = None
last_session_error: str | None = None
# Warn-but-allow arm-identity findings from the current session's guard (see
# makermodslab/arm_identity.py). The guard runs inside the recording worker (after the
# start response has already been sent), so the messages are surfaced through
# the /recording-status payload instead of the start response.
identity_warnings: list[str] = []
# Guards the start path so two concurrent POST /start-recording calls cannot
# both pass the active-flag check.
_state_lock = threading.Lock()

# True while the session's cleanup is driving the follower(s) back to their
# session-start pose (and on through the release): the recording loop is over
# but the arms are still energized (holding position, then moving home) and the
# serial ports are still held. Surfaced in the status payload so the UI isn't
# lying about the arm's state. No timed hold anymore — a normal stop returns the
# follower to where the session started, then releases (same as teleop; see
# makermodslab/rest_pose.py).
releasing = False
# Cuts the post-stop return short: set by a second stop request ("release now")
# or by a new start request that needs the serial ports. Cleared on start.
_release_now = threading.Event()

# Set by a QUIT (stop-without-saving): the user ended the session and asked to
# throw the recording away. For a FRESH session this deletes the whole stamped
# dataset directory in the worker's finally (even episodes already saved this
# session — quit discards everything this session created); for a RESUME session
# it is a no-op on disk (the pre-existing dataset is never touched — lerobot
# committed its earlier episodes as they saved, and the in-flight episode is
# already dropped by the mid-episode stop path). Reset to False under the start
# lock so a quit from one session can never leak into the next. Read by the
# worker thread, set by the request thread — a plain bool is safe under the GIL,
# same as the other recording_events flags.
discard_requested = False


def finish_pending_release(timeout: float = 10.0) -> bool:
    """Cut a pending torque-release grace short and wait for its cleanup.

    Called by start paths (recording and teleoperation) so a start arriving
    during the grace window releases the arms and frees the serial ports
    immediately instead of failing port-busy for the rest of the grace.
    Returns True when no recording worker is running afterwards; False when a
    live session is still recording or the worker did not exit in time.
    """
    worker = recording_thread
    if worker is None or not worker.is_alive():
        return True
    if not releasing:
        # A live recording session, not a pending release — the caller's
        # mutex check will report "already active".
        return False
    _release_now.set()
    worker.join(timeout=timeout)
    return not worker.is_alive()


# Bound on how long stop_and_wait() waits for the graceful end-of-session
# (return-to-rest + torque release) to finish on its own before forcing an
# immediate release. Mirrors teleoperate.py's own derivation: rest_pose's
# RETURN_CEILING_S already bounds the return motion itself; the margin covers
# episode finalization, the torque-release writes, disconnect, and
# thread-join overhead that follow it.
_STOP_AND_WAIT_TIMEOUT_S = RETURN_CEILING_S + 5.0


def stop_and_wait(timeout: float = _STOP_AND_WAIT_TIMEOUT_S) -> None:
    """Stop recording and block until the worker has actually released the
    arm(s), for a caller with no UI to poll and no "press Stop again"
    gesture available (shutdown).

    handle_stop_recording()'s first call is deliberately fire-and-forget: it
    only sets the stop_recording/exit_early events and returns immediately,
    relying on the recording worker to notice them, finish the session
    (saving whatever episodes already completed), then return to rest and
    release. Neither a status poll nor a second Stop press exists at
    shutdown. This composes both: trigger the stop, then join the worker —
    giving it its normal grace to finish gracefully, and only forcing an
    immediate release (the same "second stop" mechanism a human pressing
    Stop twice would trigger) if it hasn't finished within the bound.
    Discards nothing: a shutdown mid-session should keep whatever episodes
    were already recorded, not throw them away. A no-op when idle.
    """
    if recording_active and not releasing:
        # recording_active stays True for the worker's entire lifetime --
        # unlike teleoperation_active, which flips False the instant a stop
        # is requested -- so a session already mid return-to-rest (releasing)
        # still has recording_active=True. Calling handle_stop_recording()
        # again in that state would hit its "releasing" branch and
        # force-release immediately, aborting a return that may finish
        # gracefully on its own well within the timeout below.
        handle_stop_recording(discard=False)
    worker = recording_thread
    if worker is None or not worker.is_alive():
        return
    worker.join(timeout=timeout)
    if not worker.is_alive():
        return
    logger.warning(
        "Recording worker did not finish its graceful release within %.0fs; forcing release now",
        timeout,
    )
    _release_now.set()
    worker.join(timeout=5.0)
    if worker.is_alive():
        logger.warning("Recording worker still alive after forcing release; giving up the wait")


class RecordingRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
    # Bimanual: the primary pair above is the LEFT arm; these add the right arm.
    mode: str = "single"
    right_leader_port: str = ""
    right_follower_port: str = ""
    right_leader_config: str = ""
    right_follower_config: str = ""
    # Robot record name. Two jobs:
    #   1. The session's CAMERAS are resolved from this record, server-side
    #      (utils/config.load_robot_cameras) — the request carries no camera
    #      payload at all. Blank ⇒ a camera-less recording; a non-blank name
    #      with no record on disk is a 400, never a silent camera-less run.
    #   2. Bimanual only: the BiSO staging base id — it decides the on-disk
    #      staging dir, not which calibration drives which arm. Blank/invalid
    #      falls back to DEFAULT_BIMANUAL_BASE.
    # A `cameras` dict used to live on this model. It was removed (the panel is
    # read-only now); pydantic ignores unknown fields, so an older frontend's
    # payload still parses and its stale camera set is correctly ignored.
    robot_name: str = ""
    # Hardware family: "so101" (Feetech serial) or "maker" (RobStride CAN
    # follower + Star Arm 102 leader). Decides which lerobot config classes the
    # robot factory builds, which calibration library the names resolve in, and
    # which of the Feetech-only safety helpers apply. Defaults to so101 so a
    # request from a client that predates the Maker arm is unchanged.
    arm_type: str = "so101"
    dataset_repo_id: str
    single_task: str
    num_episodes: int = 5
    episode_time_s: int = 30
    reset_time_s: int = 10
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    tags: list[str] = []
    private: bool = False
    resume: bool = False
    streaming_encoding: bool = True
    test_mode: bool = False  # Skip robot connection for testing
    # Escape hatch for the arm-identity guard (see makermodslab/arm_identity.py):
    # when true, record even if the connected arms don't match their calibrations.
    skip_identity_check: bool = False


class UploadRequest(BaseModel):
    dataset_repo_id: str
    tags: list[str] = []
    private: bool = False


class DatasetInfoRequest(BaseModel):
    dataset_repo_id: str


def _platform_backend():
    """Pin the OpenCV backend per-platform so the index→camera mapping matches
    what the /available-cameras thumbnails were captured with. cv2.CAP_ANY can
    pick different backends across calls on macOS, silently reordering cameras
    between the modal preview and the recording."""
    import platform

    from lerobot.cameras.configs import Cv2Backends

    system = platform.system()
    if system == "Darwin":
        return Cv2Backends.AVFOUNDATION
    if system == "Linux":
        return Cv2Backends.V4L2
    if system == "Windows":
        # DirectShow, matching the order /available-cameras enumerates (via
        # pygrabber) so a camera_index always opens the previewed device.
        return Cv2Backends.DSHOW
    return Cv2Backends.ANY


def _build_camera_configs(cameras: dict, default_backend) -> dict:
    """Convert the session camera dict into OpenCVCameraConfig objects.

    The dict is `{camera name: settings}` as resolved from the robot record by
    utils/config.load_robot_cameras — it never comes from the request.

    `backend` (a Cv2Backends name) and `fourcc` (a 4-char code) are optional per
    camera; when omitted `backend` falls back to `default_backend` and `fourcc`
    to MJPG (`_DEFAULT_FOURCC`) so multi-camera USB rigs don't exhaust isochronous
    bandwidth on Linux (see `_DEFAULT_FOURCC`). An explicit per-camera fourcc wins.
    Cameras are addressed by their cv2 integer `camera_index`.
    """
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig

    camera_configs: dict = {}
    for camera_name, camera_data in cameras.items():
        if camera_data.get("type") != "opencv":
            logger.warning(
                f"⚠️ CAMERA CONFIG: Unsupported camera type '{camera_data.get('type')}' for {camera_name}"
            )
            continue

        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or _DEFAULT_FOURCC

        camera_configs[camera_name] = OpenCVCameraConfig(
            index_or_path=camera_data.get("camera_index", 0),
            backend=backend,
            fps=camera_data.get("fps"),
            width=camera_data.get("width"),
            height=camera_data.get("height"),
            fourcc=fourcc,
        )
        logger.info(
            f"✅ CAMERA CONFIG: {camera_name} -> OpenCVCameraConfig("
            f"index={camera_data.get('camera_index')}, backend={backend.name}, "
            f"{camera_data.get('width')}x{camera_data.get('height')}@{camera_data.get('fps')}fps, "
            f"fourcc={fourcc})"
        )
    return camera_configs


def _is_transient_camera_error(msg: str) -> bool:
    """True when a robot-connect failure is transient camera-session
    turbulence, curable by a clean re-connect, rather than a real failure.

    An AVCaptureSession opened into another session's asynchronous teardown
    (e.g. right after a hot-unplug) intermittently comes up wrong —
    forensically established 2026-07-09:
      * "failed to set fps=30 (actual_fps=5.0)" — cold-open fps read-back
      * "timed out waiting for frame" — session came up frame-dead (opens
        fine, background reader never receives a frame)
      * "read thread is not running" — the same frame-dead session after its
        background reader has already died (see below)

    What reaches us, and what doesn't. Only `connect()`'s own exceptions land
    here; anything raised inside a camera's background reader thread does not.
    That matters for the wrong-native-format case (session lands 640x360
    instead of 640x480): lerobot detects it in `_postprocess_image`, which
    runs *only* in `_read_loop`, so its "do not match configured" text never
    propagates — after 12 consecutive mismatches the loop dies and the caller
    sees `async_read`'s "read thread is not running" instead (or, if the
    reader is merely stalled, the timeout above). "do not match configured" is
    kept in the tuple as a cheap guard in case upstream ever surfaces it
    synchronously, but do not expect it to fire; "read thread is not running"
    is the marker that actually catches that case today.

    NOT included: "failed to set capture_" (width/height mismatch). That
    string is also produced when a camera simply doesn't support the
    configured resolution — a permanent misconfiguration, not turbulence —
    and `utils/errors.py::friendly_hint` classifies it that way for the
    operator. Retrying it would burn ~4s of backoff plus three connects on a
    failure that cannot succeed before showing the correct "click Auto" hint.

    Note "failed to set fps" has the *same* permanent-misconfiguration mode
    (an operator can type an fps the device can't do, in the same settings
    panel), so it is retried on the optimistic reading and then, if it never
    recovers, handed to `friendly_hint`'s matching fps branch for the same
    "click Auto" guidance. Keep those two in sync: any marker added here that
    can also be a permanent misconfiguration needs a `friendly_hint` branch,
    or the operator waits out the retries and gets no advice at the end.
    """
    low = msg.lower()
    return any(
        marker in low
        for marker in (
            "failed to set fps",
            "do not match configured",
            "read thread is not running",
            "timed out waiting for frame",
        )
    )


def create_record_config(request: RecordingRequest, cameras: dict | None = None) -> RecordConfig:
    """Create a RecordConfig from the recording request.

    `cameras` is the session camera dict already resolved from the robot record
    (handle_start_recording resolves it inside its lock so a bad robot name is a
    400 before the session claims anything). Left None it is resolved here from
    `request.robot_name`, so a direct caller gets the same record-driven set.
    """
    # Convert the session camera dict into OpenCVCameraConfig objects. Backend
    # defaults to the platform pin unless the record pins one per camera.
    if cameras is None:
        cameras = load_robot_cameras(request.robot_name)
    camera_configs = _build_camera_configs(cameras, _platform_backend())

    if request.mode == "bimanual":
        # Build a lerobot BiSO leader+follower pair (config assembly + calibration
        # staging in build_bimanual_configs). Cameras go on the left follower arm
        # (exposed prefixed "left_*").
        robot_config, teleop_config = build_bimanual_configs(request, cameras=camera_configs)
    else:
        robot_config, teleop_config = build_single_configs(request, cameras=camera_configs)

    # Create dataset config
    dataset_config = DatasetRecordConfig(
        repo_id=request.dataset_repo_id,
        # Explicit local root, ALWAYS. For fresh sessions this is identical to
        # the root=None default (HF_LEROBOT_HOME/<repo_id>), but lerobot's
        # resume path REFUSES root=None (it would write into the revision-safe
        # Hub snapshot cache) — and every MakerMods Lab helper that discards/deletes
        # sessions already resolves datasets against this exact location, so
        # pinning it here keeps create, resume, and cleanup on one path. The
        # repo_id is final by now (the fresh-session timestamp is stamped
        # before this runs).
        root=str(_lerobot_cache_root() / request.dataset_repo_id),
        single_task=request.single_task,
        num_episodes=request.num_episodes,
        episode_time_s=request.episode_time_s,
        reset_time_s=request.reset_time_s,
        fps=request.fps,
        video=request.video,
        push_to_hub=request.push_to_hub,
        # Upstream typing: tags is `list[str] | None`. None when push is off
        # keeps the lerobot default.
        tags=with_makermodslab_tag(request.tags) if request.push_to_hub else None,
        private=request.private,
        streaming_encoding=request.streaming_encoding,
        # lerobot 0.6.0 shim: vcodec moved off DatasetRecordConfig into the
        # nested RGBEncoderConfig. "auto" resolves a hardware encoder on this
        # machine (RGBEncoderConfig.__post_init__ probes the local FFmpeg) —
        # h264_videotoolbox on macOS, h264_nvenc/hevc_nvenc on Jetson/Linux-
        # NVIDIA — falling back to software libsvtav1 (a logged warning, no
        # error) when none exists. This matters because streaming_encoding is on
        # by default: on Jetson-class CPUs, software SVT-AV1 can run below
        # realtime, fill the encoder queue, and block the capture loop — the
        # suspected cause of record-lag. Hardware encoding keeps encode off the
        # capture thread's critical path.
        rgb_encoder=RGBEncoderConfig(vcodec="auto"),
    )

    # Create the main record config
    record_config = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        resume=request.resume,
        display_data=False,  # Don't display data in API mode
        play_sounds=False,  # Don't play sounds in API mode
    )

    return record_config


def handle_start_recording(request: RecordingRequest) -> dict[str, Any]:
    """Handle start recording request by using the existing record() function.

    Returns a dict — the route layer turns a rejection into an HTTPException
    using the "status_code" key when present (409 for an active-session
    conflict, 400 for a malformed dataset name), defaulting to 500 otherwise."""
    global \
        recording_active, \
        releasing, \
        recording_thread, \
        recording_events, \
        recording_config, \
        recording_start_time, \
        session_end_elapsed_seconds, \
        current_episode, \
        saved_episodes, \
        current_phase, \
        phase_start_time, \
        last_session_discarded_empty, \
        last_session_outcome, \
        last_session_error, \
        identity_warnings, \
        discard_requested, \
        connect_retry_attempt

    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        replay as _replay,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    # Claim the active flag under the lock so two concurrent starts can't both
    # pass the precondition check.
    logger.info(
        "Recording start requested: dataset=%r, task=%r, resume=%s, mode=%s",
        request.dataset_repo_id,
        request.single_task,
        request.resume,
        getattr(request, "mode", "single"),
    )

    # A previous session (recording or teleop) may still be holding torque for
    # its release grace — cut it short so this start doesn't fail on a busy
    # serial port. Best effort: failures are surfaced by the checks below.
    finish_pending_release()
    _teleoperate.finish_pending_release()

    with _state_lock:
        if recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": (
                    "The previous session is still releasing the arms. Try again in a few seconds."
                    if releasing
                    else "Recording is already active"
                ),
                "code": ErrorCode.ROBOT_BUSY_RELEASING if releasing else ErrorCode.ROBOT_BUSY_RECORDING,
            }
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_TELEOPERATION,
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
        if _replay.replay_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Replay is currently active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_REPLAY,
            }
        # Refuse a malformed dataset name up front (before claiming the flag or
        # touching hardware). Rejecting beats silent sanitization: "whoo/" used to
        # smuggle in a namespace and land the dataset at "user/whoo/".
        name_ok, name_reason = validate_dataset_repo_id(request.dataset_repo_id)
        if not name_ok:
            logger.warning(
                "Rejected recording start: invalid dataset name %r (%s)",
                request.dataset_repo_id,
                name_reason,
            )
            return {
                "success": False,
                "status_code": 400,
                "message": name_reason,
                "code": ErrorCode.REQUEST_INVALID_NAME,
            }
        # Resolve the session's cameras from the robot record NAMED BY THE
        # REQUEST — the request itself carries no camera payload. Done here,
        # before the flag is claimed, so a wrong robot name or an ambiguous
        # camera set is a plain 400 instead of a session that starts and
        # silently records no video.
        try:
            session_cameras = load_robot_cameras(request.robot_name)
        except CameraResolutionError as exc:
            logger.warning("Rejected recording start: %s", exc)
            return {"success": False, "status_code": 400, "message": str(exc)}
        # Per-session state reset, under the same lock that claims the active
        # flag: a stale _release_now from a previous session's double-stop
        # would otherwise cut EVERY later release grace short until the server
        # restarts (regression-tested in tests/test_record.py).
        recording_active = True
        releasing = False
        _release_now.clear()
        recording_thread = None
        recording_events = None
        recording_config = None
        recording_start_time = None
        session_end_elapsed_seconds = None
        current_episode = 1
        saved_episodes = 0
        current_phase = "preparing"
        phase_start_time = None
        connect_retry_attempt = 0
        last_session_discarded_empty = False
        last_session_outcome = None
        last_session_error = None
        identity_warnings = []
        # Clear any quit-discard request from a previous session under the same
        # lock that claims the active flag (mirrors the _release_now reset), so a
        # stale flag can never make this fresh session delete its own dataset.
        discard_requested = False

    # The claim above is the real state transition — broadcast the hint so
    # every WS client (any page, any remote UI) refetches /recording-status.
    notify_session_changed("recording", True, phase="preparing")

    # Backend camera previews hold the cv2 devices; hand them over before we open
    # the same indices. Ordered deliberately: `recording_active` is already True
    # (set under the lock above), so /camera-preview now 409s and no client can
    # re-acquire a device in the gap between this release and the recorder's
    # open. Doing it the other way round leaves that race open, and a preview
    # still holding index 0 starves the recorder (OpenCVCamera(0) actual_fps=5.0).
    camera_preview_manager.stop_all()

    # Start capturing this session's logs into a fresh bounded ring buffer so the
    # Record page can display them (detaches any previous session's handler).
    _attach_record_log_handler()

    try:
        # The name is already validated (validate_dataset_repo_id in the lock), so
        # no sanitization is needed here. Stamp the repo_id with a timestamp
        # (matches lerobot-record CLI behavior) so each session lands in a unique
        # directory and the frontend gets the final id back in the response.
        if not request.resume and request.dataset_repo_id:
            request.dataset_repo_id = f"{request.dataset_repo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting recording for dataset: {request.dataset_repo_id}")
        logger.info(f"Task: {request.single_task}")

        recording_config = request
        recording_events = {
            "exit_early": False,  # Right arrow key -> "Skip to next episode" button
            "stop_recording": False,  # ESC key -> "Stop recording" button
            "rerecord_episode": False,  # Left arrow key -> "Re-record episode" button
            "paused": False,  # Reset-phase pause/resume (mouse-only, no keyboard shortcut)
        }

        record_config = create_record_config(request, cameras=session_cameras)

        def recording_worker():
            global \
                recording_active, \
                recording_start_time, \
                session_end_elapsed_seconds, \
                current_phase, \
                phase_start_time, \
                current_episode, \
                saved_episodes, \
                last_session_discarded_empty, \
                last_session_outcome, \
                last_session_error, \
                paused_accum_seconds, \
                pause_started_at
            recording_start_time = time.time()
            current_episode = 1
            saved_episodes = 0
            paused_accum_seconds = 0.0
            pause_started_at = None

            try:
                logger.info(
                    "Recording session started: dataset=%s task=%r episodes=%d",
                    request.dataset_repo_id,
                    request.single_task,
                    request.num_episodes,
                )

                # Give the frontend's camera streams time to release the
                # underlying devices before lerobot tries to open them.
                if session_cameras:
                    logger.info(
                        "Waiting for camera resources to be released (cameras: %s)",
                        list(session_cameras.keys()),
                    )
                    time.sleep(2.0)

                # Bimanual: the sub-arm ids are BiSO staging aliases, so give the
                # identity guard the real library stems (in arm-iteration order:
                # left follower, right follower, left leader, right leader).
                identity_config_names = (
                    [
                        request.follower_config,
                        request.right_follower_config,
                        request.leader_config,
                        request.right_leader_config,
                    ]
                    if request.mode == "bimanual"
                    else None
                )
                dataset = record_with_web_events(
                    record_config,
                    recording_events,
                    skip_identity_check=request.skip_identity_check,
                    identity_config_names=identity_config_names,
                )
                logger.info(f"Recording completed successfully. Dataset has {dataset.num_episodes} episodes")
                last_session_outcome = "ok"
            except Exception as exc:
                logger.exception("Recording session failed")
                # In-process error taxonomy: this catch site holds the actual
                # exception object, so format it straight into the session-end
                # status (no log forensics — that's rollout's subprocess-only
                # step). Classified by catch-site phase: current_phase is
                # "completed" only when the recording loop already finished its
                # real work (all episodes saved / user stop honored) before the
                # raise — i.e. only teardown/cleanup tripped (e.g. a gripper
                # overload on torque disable). That session ran fine, so it's a
                # warning, NOT a failure — the episodes are on disk. Any other
                # phase (connecting_*, recording, resetting, stopping) means
                # the session's work was cut short: a real failure.
                last_session_error = format_exception(exc)
                work_completed = current_phase == "completed"
                last_session_outcome = classify_outcome(work_completed, last_session_error)
                if not work_completed:
                    _set_phase("error")
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)
            finally:
                if current_phase != "error":
                    _set_phase("completed")
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)

                # Dataset cleanup, ordered here — after record_with_web_events
                # has returned/raised, which released torque and disconnected in
                # its own finally — so cleanup never blocks the hardware release.
                # Best-effort: both helpers swallow their own errors and never
                # re-raise, so the original error path is preserved. Resume
                # sessions are never deleted (guarded inside both helpers).
                if discard_requested:
                    # QUIT: the user threw the recording away. For a fresh
                    # session this removes the whole stamped directory — even
                    # episodes saved earlier this session — because quit means
                    # discard everything this session created. A resume session
                    # keeps its pre-existing dataset (helper no-ops on resume).
                    last_session_discarded_empty = _discard_session_dataset(
                        request.dataset_repo_id, request.resume
                    )
                elif saved_episodes == 0:
                    # Not a quit, but the session created a fresh dataset it
                    # never wrote an episode into (interrupted/failed, or every
                    # take re-recorded away). Confirmed against the on-disk
                    # episode count inside the helper; resume sessions untouched.
                    last_session_discarded_empty = _discard_empty_dataset(
                        request.dataset_repo_id, request.resume
                    )

                recording_active = False
                # Final release: record_with_web_events' own finally already
                # released torque and disconnected before this ran, so the
                # ports are free now — including on error paths.
                notify_session_changed("recording", False, phase=current_phase)
                recording_start_time = None
                phase_start_time = None
                current_episode = 1
                # saved_episodes is deliberately left as-is (not reset here):
                # handle_recording_status reports it in the terminal
                # (session_ended) status, and the next session already resets it
                # under the lock in handle_start_recording. Zeroing it here would
                # erase the count before any poll can read it.
                logger.info("Recording session ended")

        recording_thread = threading.Thread(target=recording_worker, name="recording-worker", daemon=True)
        recording_thread.start()

        return {
            "success": True,
            "message": "Recording started successfully",
            "dataset_id": request.dataset_repo_id,
            "num_episodes": request.num_episodes,
        }

    except Exception as e:
        recording_active = False
        # The claim above already broadcast active=True; undo the hint now
        # that the failed start released the flag.
        notify_session_changed("recording", False)
        logger.error(f"Failed to start recording: {e}")
        return {"success": False, "message": f"Failed to start recording: {str(e)}"}


def handle_stop_recording(discard: bool = False) -> dict[str, Any]:
    """End the recording session. Two flavours, selected by ``discard``:

    * ``discard=False`` (DONE / keep): finalize now, keeping every episode saved
      so far. The in-progress (incomplete) episode is dropped by the mid-episode
      stop path, but all completed episodes stay on disk and the frontend
      navigates on to the upload page.
    * ``discard=True`` (QUIT / don't save): end without keeping the recording.
      A FRESH session's whole stamped dataset directory is removed in the
      worker's finally; a RESUME session keeps its pre-existing dataset (only the
      in-flight episode is dropped). This is also the path an unintentional page
      exit (back button / tab close) takes.

    Either way the arm returns to its session-start pose, then releases torque.
    A second stop while the session-end cleanup is holding torque for the release
    grace cuts the hold short ("release now").
    """
    global current_phase, phase_start_time, discard_requested

    if releasing:
        _release_now.set()
        logger.info("Second stop during the release grace — releasing the arms now")
        return {
            "success": True,
            "message": "Releasing the arms now",
            "session_ending": True,
        }

    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}

    if discard:
        discard_requested = True
    recording_events["stop_recording"] = True
    recording_events["exit_early"] = True
    _set_phase("stopping")
    phase_start_time = None
    logger.info("Stop recording triggered from web interface (discard=%s)", discard)
    if discard:
        message = (
            "Quitting without saving. The recording is being discarded; the arm returns to its "
            "starting position, then goes limp."
        )
    else:
        message = (
            "Recording stop requested. Saved episodes are kept. When the session ends, the arm "
            "returns to its starting position, then goes limp — press Stop again to release it "
            "immediately."
        )
    return {
        "success": True,
        "message": message,
        "session_ending": True,
        "discard": discard,
    }


def handle_exit_early() -> dict[str, Any]:
    """Handle exit early request - replaces right arrow key"""
    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    if _reset_paused():
        # Design decision: skip-to-next-episode is disabled while paused (the
        # operator must explicitly resume first). Enforced here, not just in
        # the advertised available_controls.exit_early, since the frontend's
        # gate is 1-second-stale via polling and this closes that race at
        # its root.
        return {
            "success": False,
            "message": "Cannot skip to next episode while the reset phase is paused",
            "current_phase": current_phase,
        }
    recording_events["exit_early"] = True
    # Tracking flag that record_loop won't reset, so the worker can tell
    # "user pressed skip" from "control_time_s elapsed naturally".
    recording_events["_exit_early_triggered"] = True
    logger.info("Exit early triggered (current phase: %s)", current_phase)
    phase_name = "recording phase" if current_phase == "recording" else "reset phase"
    return {
        "success": True,
        "message": f"Exit early triggered successfully for {phase_name}",
        "current_phase": current_phase,
        "events_state": dict(recording_events),
    }


def handle_rerecord_episode() -> dict[str, Any]:
    """Handle rerecord episode request - replaces left arrow key"""
    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    recording_events["rerecord_episode"] = True
    recording_events["exit_early"] = True
    logger.info("Re-record episode triggered")
    return {
        "success": True,
        "message": "Re-record episode requested successfully",
        "events_state": dict(recording_events),
    }


def handle_pause_recording() -> dict[str, Any]:
    """Pause the reset gap. Can be requested during the recording phase too —
    that ARMS a pending pause: the current episode keeps recording
    uninterrupted, and the reset phase that follows starts already paused
    (teleop passthrough frozen, follower holds position, ignores the leader)
    instead of needing a fresh click once resetting begins. Only takes
    effect on the reset countdown once resetting actually starts (see the
    reset-phase preamble in record_with_web_events, which seeds
    pause_started_at at that point if arming already happened). No-ops
    outside recording/resetting and when already paused/armed, so a stray or
    duplicate request is always safe."""
    global pause_started_at

    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    if current_phase not in ("recording", "resetting"):
        return {
            "success": False,
            "message": "Pause is only available while recording or during the reset gap",
        }
    if recording_events.get("paused", False):
        return {"success": True, "message": "Already paused", "events_state": dict(recording_events)}

    recording_events["paused"] = True
    if current_phase == "resetting":
        pause_started_at = time.time()
    # else: armed during the recording phase — pause_started_at stays None
    # until the reset phase actually begins, since the countdown that
    # matters hasn't started yet.
    logger.info("Pause requested (phase=%s)", current_phase)
    return {"success": True, "message": "Pause set", "events_state": dict(recording_events)}


def handle_resume_recording() -> dict[str, Any]:
    """Resume a paused reset gap, or cancel a pause armed during the
    recording phase before the reset gap it targets even begins. No-ops
    outside recording/resetting and when not currently paused/armed."""
    global paused_accum_seconds, pause_started_at

    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    if current_phase not in ("recording", "resetting"):
        return {
            "success": False,
            "message": "Resume is only available while recording or during the reset gap",
        }
    if not recording_events.get("paused", False):
        return {"success": True, "message": "Not paused", "events_state": dict(recording_events)}

    recording_events["paused"] = False
    if pause_started_at is not None:
        paused_accum_seconds += time.time() - pause_started_at
        pause_started_at = None
    logger.info("Pause cleared (phase=%s)", current_phase)
    return {"success": True, "message": "Pause cleared", "events_state": dict(recording_events)}


def _reset_paused() -> bool:
    """True only while a live session is genuinely paused DURING THE RESET
    PHASE — i.e. the pause is actually freezing the countdown/passthrough
    right now, as opposed to merely being armed for an upcoming reset gap
    while still in the recording phase. Phase-gated (not just an
    events-dict flag check) so a stale paused=True left over from a reset
    loop that exited without clearing it (e.g. via exit_early) can never
    leak into a later phase's available_controls or status — see the
    loop-exit cleanup in record_with_web_events, which is defense in depth
    on top of this gate, not a substitute for it."""
    return bool(recording_events and current_phase == "resetting" and recording_events.get("paused", False))


def _pause_armed() -> bool:
    """True whenever paused is set at all, regardless of phase — covers both
    a live pause during the reset phase and a pause armed during the
    recording phase for the upcoming reset gap. Used to gate
    pause_recording/resume_recording's availability across both phases;
    _reset_paused() remains the narrower, phase-gated predicate for anything
    that must only react to an ACTIVE freeze (the exit_early guard, the
    status `paused` field). Gated on recording_active, not just the
    events-dict flag: a pause armed on an episode's recording phase can be
    left stranded True if that episode turns out to be the session's last
    (no reset phase follows to run the unconditional `paused = False`
    cleanup in record_with_web_events) — without this gate, the stale flag
    would keep reporting pause_armed=True in the completed/stopped status
    until the next session's handle_start_recording rebuilds the dict."""
    return bool(recording_active and recording_events and recording_events.get("paused", False))


def handle_recording_status() -> dict[str, Any]:
    """Handle recording status request"""
    # If recording is not active and phase is completed or error, indicate session has ended
    session_ended = not recording_active and current_phase in ["completed", "error"]

    # Log when session has ended to help debug frontend polling
    if session_ended:
        if current_phase == "error":
            logger.info(
                "📡 RECORDING STATUS REQUEST: Session failed with error - frontend should stop polling"
            )
            print("📡 STATUS CHANGE: Frontend is still polling after session error - should stop now")
        else:
            logger.info("📡 RECORDING STATUS REQUEST: Session has ended - frontend should stop polling")
            print("📡 STATUS CHANGE: Frontend is still polling after session end - should stop now")

    status = {
        "recording_active": recording_active,
        "current_phase": current_phase,  # "preparing", "recording", "resetting", "completed"
        # Only meaningful while current_phase == "reconnecting_robot": which
        # connect attempt is about to be retried, out of _CONNECT_ATTEMPTS.
        "connect_retry_attempt": connect_retry_attempt,
        "connect_retry_max": _CONNECT_ATTEMPTS,
        "session_ended": session_ended,  # New field to indicate session completion
        # True during the post-session rest-pose return: the recording loop is
        # over but the arm is still energized and driving back to its
        # session-start pose before torque is released.
        "releasing": releasing,
        # True only during the reset phase, and only while a pause is
        # actively freezing it — see handle_pause_recording/handle_resume_recording.
        "paused": _reset_paused(),
        # True whenever paused is set at all, including a pause armed during
        # the recording phase for the upcoming reset gap (not yet actually
        # freezing anything). Drives the frontend's Pause/Resume toggle,
        # which is available in both phases.
        "pause_armed": _pause_armed(),
        "available_controls": {
            "stop_recording": recording_active,  # ESC key replacement
            "exit_early": recording_active
            and not _reset_paused(),  # Right arrow key replacement; disabled while paused
            "rerecord_episode": recording_active
            and current_phase == "recording",  # Only during recording phase
            "pause_recording": recording_active
            and current_phase in ("recording", "resetting")
            and not _pause_armed(),
            "resume_recording": recording_active
            and current_phase in ("recording", "resetting")
            and _pause_armed(),
        },
        "message": "Recording session failed with error - check logs"
        if current_phase == "error"
        else (
            "Returning the arm to its rest position…"
            if releasing
            else (
                "Recording session has ended - stop polling"
                if session_ended
                else "Recording status retrieved successfully"
            )
        ),
    }

    # Always echo the stamped dataset id whenever a config exists, so the frontend
    # can read the actual on-disk repo_id (post stamp) for upload navigation.
    if recording_config:
        status["dataset_repo_id"] = recording_config.dataset_repo_id

    # When the session has ended, tell the frontend honestly whether anything
    # was kept. A session that saved zero episodes had its (freshly created)
    # dataset directory discarded — the post-recording page shows a "nothing was
    # saved" variant and does NOT link the (now-gone) repo id.
    if session_ended:
        status["discarded_empty"] = last_session_discarded_empty
        # Terminal error taxonomy (the in-process twin of rollout's exited
        # payload): `outcome` classifies the ended session (ok |
        # ran_with_warning | failed), `error` is the caught exception's
        # "Type: message" text, and `hint` a plain-language headline for the
        # common SO-101 failures. ran_with_warning = the episodes are safe on
        # disk and only teardown tripped — the frontend styles it amber, not
        # as a failed session.
        status["outcome"] = last_session_outcome
        status["error"] = last_session_error
        status["hint"] = friendly_hint(last_session_error)
        # The frontend's exit handoff reads this to show/pass along how many
        # episodes actually got saved — it must survive past recording_active
        # flipping False, not just live-poll while recording is in progress.
        status["saved_episodes"] = saved_episodes

    # Warn-but-allow arm-identity findings (the guard runs in the worker, after
    # the start response) — the frontend shows these as a non-blocking toast.
    if identity_warnings:
        status["warning"] = " ".join(identity_warnings)

    # Add episode information if recording is active
    if recording_active and recording_config:
        status["current_episode"] = current_episode
        status["total_episodes"] = recording_config.num_episodes
        status["saved_episodes"] = saved_episodes  # Track completed episodes

        # Add session start time if available
        if recording_start_time:
            status["session_start_time"] = recording_start_time
            status["session_elapsed_seconds"] = int(time.time() - recording_start_time)

        # Add phase timing information
        if phase_start_time:
            status["phase_start_time"] = phase_start_time
            elapsed = time.time() - phase_start_time
            # Only the reset phase ever accumulates paused time; gate the
            # adjustment on phase so a leftover paused_accum_seconds/
            # pause_started_at from a finished reset phase (this is a
            # stateless read of module globals, so it can race a phase
            # transition) never corrupts the recording phase's countdown.
            if current_phase == "resetting":
                elapsed -= paused_accum_seconds
                if pause_started_at is not None:
                    elapsed -= time.time() - pause_started_at
            status["phase_elapsed_seconds"] = int(elapsed)

            # Add phase time limits
            if current_phase == "recording":
                status["phase_time_limit_s"] = recording_config.episode_time_s
            elif current_phase == "resetting":
                status["phase_time_limit_s"] = recording_config.reset_time_s
    elif session_end_elapsed_seconds is not None:
        status["session_elapsed_seconds"] = session_end_elapsed_seconds

    return status


def handle_delete_dataset(request: DatasetInfoRequest) -> dict[str, Any]:
    """Remove a recorded dataset's directory from local disk."""
    from pathlib import Path

    from lerobot.utils.constants import HF_LEROBOT_HOME

    repo_id = request.dataset_repo_id
    root = Path(HF_LEROBOT_HOME).resolve()
    target = (root / repo_id).resolve()

    # Reject path traversal: target must stay strictly inside HF_LEROBOT_HOME.
    if target == root or root not in target.parents:
        return {"success": False, "message": "Invalid dataset path"}

    # Don't yank the directory out from under an active writer/reader. Reuses
    # the full rename busy-guard (recording / merge / upload / local training)
    # instead of the old upload-only check; lazy import to avoid the
    # datasets<->record cycle documented in _dataset_in_use.
    from .datasets import _dataset_in_use

    in_use = _dataset_in_use(repo_id)
    if in_use is not None:
        return {"success": False, "message": in_use}

    if not target.exists():
        return {"success": False, "message": f"Dataset not found on disk: {repo_id}"}

    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.error(f"Failed to delete dataset {repo_id}: {e}")
        return {"success": False, "message": f"Failed to delete dataset: {e}"}

    # The listing just changed — drop the cached /datasets listing so the delete
    # reflects immediately instead of after the TTL. Also drop the cached
    # Hub-existence answer: get_hub_status memoizes "local_only" (and "on_hub")
    # for the process lifetime, so without this a dataset checked before the
    # delete would keep reporting "local_only" forever instead of "absent".
    invalidate_dataset_listing_cache()
    invalidate_hub_status(repo_id)

    logger.info(f"Deleted dataset directory {target}")
    return {"success": True, "message": f"Deleted {repo_id}"}


def _discard_empty_dataset(repo_id: str, resume: bool) -> bool:
    """Remove the directory of a session that saved zero episodes.

    Interrupted/failed recording sessions used to leave 0-episode datasets on
    disk — hundreds of MB of video, invisible in the picker (empties are
    hidden) yet still consuming space. When a session ends having saved no
    episodes, delete the directory THIS session created.

    Guards (all must hold before anything is removed):
      * ``resume`` is False — a resume/append session writes into a
        pre-existing dataset, so we must NEVER delete it even at zero *new*
        episodes. Only non-resume sessions stamp a fresh timestamped directory
        (see handle_start_recording), so only those are ours to discard.
      * The directory reports zero episodes — confirmed against ``meta/info.json``'s
        ``total_episodes`` (the same signal the picker uses), not just the
        in-memory counter.
      * The path stays strictly inside the LeRobot cache root (traversal guard,
        mirroring handle_delete_dataset).

    Best-effort: any failure is logged as a warning and swallowed — this runs
    during session-end cleanup and must never mask an original error or block
    the hardware release. Returns True iff the directory was removed.
    """
    if resume:
        # Append-into-existing: the dataset predates this session. Never delete.
        return False
    if not repo_id:
        return False

    root = _lerobot_cache_root().resolve()
    try:
        target = (root / repo_id).resolve()
    except OSError:
        return False

    # Reject path traversal: target must stay strictly inside the cache root.
    if target == root or root not in target.parents:
        return False
    if not target.is_dir():
        return False

    # Confirm zero episodes against the on-disk metadata, not just the counter.
    info_path = target / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, ValueError):
        # No readable info.json — the dataset was never created far enough to
        # hold episodes. Treat as empty and clean it up.
        info = {}
    if info.get("total_episodes"):
        return False

    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.warning(f"Failed to remove empty dataset {repo_id}: {e}")
        return False

    # Invalidate the cached Hub-existence probe (cheap correctness — the
    # repo no longer exists here) and the cached /datasets listing.
    invalidate_hub_status(repo_id)
    invalidate_dataset_listing_cache()

    logger.info(f"Removed empty dataset {repo_id} — no episodes were saved.")
    return True


def _discard_session_dataset(repo_id: str, resume: bool) -> bool:
    """Remove the whole directory of a FRESH session the user QUIT without saving.

    Unlike :func:`_discard_empty_dataset`, this deletes even when episodes were
    already saved earlier in THIS session — a quit means discard everything this
    session created, not just the empty case.

    Guards (all must hold before anything is removed), the resume guard FIRST and
    load-bearing:
      * ``resume`` is False — a resume/append session writes into a PRE-EXISTING
        dataset, so quitting must NEVER delete it. lerobot commits episodes as
        they save, so a resume quit keeps every already-saved episode and only
        drops the in-flight one (handled by the mid-episode stop path). Only
        non-resume sessions stamp a fresh timestamped directory
        (handle_start_recording), so only those are ours to delete.
      * The path stays strictly inside the LeRobot cache root (traversal guard,
        mirroring handle_delete_dataset / _discard_empty_dataset).

    Best-effort: any failure is logged as a warning and swallowed — this runs
    during session-end cleanup and must never mask an original error or block the
    hardware release. Returns True iff the directory was removed.
    """
    if resume:
        # Append-into-existing: the dataset predates this session. Never delete.
        return False
    if not repo_id:
        return False

    root = _lerobot_cache_root().resolve()
    try:
        target = (root / repo_id).resolve()
    except OSError:
        return False

    # Reject path traversal: target must stay strictly inside the cache root.
    if target == root or root not in target.parents:
        return False
    if not target.is_dir():
        return False

    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.warning(f"Failed to discard quit session dataset {repo_id}: {e}")
        return False

    # The repo no longer exists locally — drop its cached Hub-existence probe and
    # the cached /datasets listing so the discard reflects immediately.
    invalidate_hub_status(repo_id)
    invalidate_dataset_listing_cache()

    logger.info(f"Discarded dataset {repo_id} — session was quit without saving.")
    return True


def _upload_auth_error(exc: Exception) -> dict[str, str] | None:
    """If ``exc`` is a Hub auth failure, return the friendly {message, docs_url}
    the frontend shows for it; else None. Kept separate so the sync worker and
    any future caller map the 401 identically."""
    err_text = str(exc).lower()
    looks_like_auth = any(
        m in err_text
        for m in ("401", "you must be authenticated", "authentication required", "huggingfacehub_token")
    )
    if looks_like_auth:
        return {
            "message": (
                "You're not logged into the Hugging Face Hub. Run `hf auth login` in your "
                "terminal, then retry."
            ),
            "docs_url": "https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication",
        }
    return None


class UploadManager:
    """Runs one dataset upload at a time in a background thread.

    ``push_to_hub`` copies 100+ MB of video/parquet over the network and takes
    minutes, so we run it off the request thread (same start/poll shape as
    MergeManager) rather than block the browser on a multi-minute HTTP request
    that a navigation-away would abort mid-push. One upload at a time: a second
    concurrent start for any repo is refused (409-mapped by the route). The
    per-repo status lets the info card / picker row poll "is *my* dataset
    uploading?" and survive navigation.
    """

    def __init__(self) -> None:
        self.state: str = "idle"  # "idle" | "running" | "done" | "error"
        self.repo_id: str | None = None
        self.message: str | None = None
        self.dataset_url: str | None = None
        self.docs_url: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, request: UploadRequest) -> dict[str, Any]:
        repo_id = request.dataset_repo_id
        with self._lock:
            if self.state == "running":
                return {
                    "started": False,
                    "repo_id": self.repo_id,
                    "message": f"An upload is already running for {self.repo_id}",
                }
            # Refuse a dataset another operation is actively writing — pushing a
            # half-written directory would ship a corrupt dataset. Reuses the
            # rename busy-guard (recording / merge / local training); lazy import
            # to avoid the datasets<->record cycle documented in _dataset_in_use.
            from .datasets import _dataset_in_use

            in_use = _dataset_in_use(repo_id)
            if in_use is not None:
                return {"started": False, "repo_id": repo_id, "message": in_use}
            self.state = "running"
            self.repo_id = repo_id
            self.message = f"Uploading {repo_id} to the Hub…"
            self.dataset_url = None
            self.docs_url = None

        self._thread = threading.Thread(
            target=self._worker, args=(request,), name="upload-worker", daemon=True
        )
        self._thread.start()
        return {"started": True, "repo_id": repo_id, "message": "Upload started"}

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = {
                "state": self.state,
                "repo_id": self.repo_id,
                "message": self.message,
                "dataset_url": self.dataset_url,
            }
            if self.docs_url is not None:
                status["docs_url"] = self.docs_url
            return status

    def _worker(self, request: UploadRequest) -> None:
        repo_id = request.dataset_repo_id
        try:
            tags = with_makermodslab_tag(request.tags)
            logger.info(f"Uploading to HuggingFace Hub with tags: {tags}, private: {request.private}")
            # A locally-recorded dataset's id is bare; push_dataset_to_hub
            # resolves it and returns the id it actually landed under, which is
            # what the success message and url below must name — the bare one
            # would send the user to a 404.
            hub_repo_id = push_dataset_to_hub(repo_id, tags=tags, private=request.private)
            logger.info(f"Dataset {hub_repo_id} uploaded successfully to HuggingFace Hub")

            # push_dataset_to_hub already dropped the cached Hub facts for this
            # dataset (existence + summary), so the card's next hub-status check
            # flips to "On Hub". The merged /datasets listing is this site's to
            # drop: it isn't a Hub fact, but the newly-pushed repo should appear
            # in it immediately rather than after the TTL.
            invalidate_dataset_listing_cache()

            with self._lock:
                self.state = "done"
                self.message = f"Dataset {hub_repo_id} uploaded successfully to the Hugging Face Hub"
                self.dataset_url = f"https://huggingface.co/datasets/{hub_repo_id}"
                self.docs_url = None
        except Exception as e:
            logger.error(f"Error uploading dataset {repo_id}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            auth = _upload_auth_error(e)
            with self._lock:
                self.state = "error"
                self.dataset_url = None
                if auth is not None:
                    self.message = auth["message"]
                    self.docs_url = auth["docs_url"]
                else:
                    self.message = f"Failed to upload dataset: {e}"
                    self.docs_url = None


upload_manager = UploadManager()


def handle_upload_dataset(request: UploadRequest) -> dict[str, Any]:
    """Start a background upload of a local dataset to the Hub.

    Returns immediately with ``{started, repo_id, message}`` — the actual push
    runs in a worker thread; poll /upload-status for progress. ``started`` is
    False when an upload is already running or the dataset is busy being
    written (recording / merge / training)."""
    return upload_manager.start(request)


def handle_upload_status() -> dict[str, Any]:
    """Current upload state (idle | running | done | error) + repo_id, message,
    and dataset_url once done."""
    return upload_manager.get_status()


def _reset_loop_with_pause(
    robot,
    teleop,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    control_time_s: float,
) -> None:
    """Reset-phase tick loop: same per-tick shape as lerobot's record_loop
    (lerobot.scripts.lerobot_record.record_loop) called with dataset=None —
    read the leader, send it to the follower, no dataset write, no display —
    but additionally checks events["paused"] every tick, which record_loop has
    no primitive for and which can't be added without patching that vendored
    dependency.

    While paused: skips the leader-to-follower step entirely (the follower
    holds its last commanded position and ignores the leader — "freeze
    passthrough"), and holds the loop's own control_time_s budget rather than
    spending it, so the operator gets the full reset_time_s of active gap
    time regardless of how long they stayed paused. Pausing is indefinite —
    this loop enforces no timeout of its own.

    Never blocks on an unbounded wait while paused: sleeps one control tick
    at a time and rechecks events["exit_early"]/events["stop_recording"] every
    iteration BEFORE the pause check, so Stop remains responsive within about
    one tick even while paused. handle_stop_recording always sets both flags
    together, so the stop_recording check is redundant in practice today —
    it's here so this loop's own contract (matching the design spec: check
    both every tick) doesn't depend on caller discipline.
    """
    from lerobot.utils.robot_utils import precise_sleep

    control_interval = 1 / fps
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events.get("exit_early", False):
            events["exit_early"] = False
            break

        if events.get("stop_recording", False):
            break

        if events.get("paused", False):
            precise_sleep(control_interval)
            # Paused time isn't spent from control_time_s: push the reference
            # point forward by the tick we just spent paused.
            start_episode_t += time.perf_counter() - start_loop_t
            timestamp = time.perf_counter() - start_episode_t
            continue

        if teleop is not None:
            obs = robot.get_observation()
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


def record_with_web_events(
    cfg: RecordConfig,
    web_events: dict,
    skip_identity_check: bool = False,
    identity_config_names: list[str] | None = None,
) -> LeRobotDataset:
    """
    Implement recording with phase tracking - exactly mirrors original record() function behavior

    `identity_config_names` (bimanual only) are the real library calibration stems
    — [left_follower, right_follower, left_leader, right_leader] — so the arm
    identity guard compares against the library instead of the BiSO staging alias
    ids ("<base>_left"/"<base>_right"). None for single-arm (id is the real stem).
    """
    import time

    from lerobot.common.control_utils import (
        sanity_check_dataset_robot_compatibility,
    )
    from lerobot.datasets import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.utils import log_say

    global current_phase, phase_start_time, current_episode, saved_episodes, releasing
    global identity_warnings, connect_retry_attempt
    global paused_accum_seconds, pause_started_at

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Read the arm type back off the assembled config rather than taking it as
    # a parameter, so it can never disagree with the devices actually built.
    # Everything below that touches a Feetech register by name is gated on it.
    feetech = uses_feetech_bus(arm_type_of_robot_config(cfg.robot))

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            # lerobot 0.6.0: vcodec moved off DatasetRecordConfig into the nested
            # rgb_encoder/depth_encoder configs; resume/create now take those.
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
            if num_cameras > 0
            else 0,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # lerobot's sanity_check_dataset_name requires a namespaced "user/name"
        # id and crashes on the bare names we allow for local recording without
        # an HF login. Inline its only rule that applies here (policy is None):
        # an eval_ prefix is reserved for policy-evaluation recordings.
        if cfg.dataset.repo_id.rsplit("/", 1)[-1].startswith("eval_"):
            raise ValueError(
                f"Dataset name '{cfg.dataset.repo_id}' begins with 'eval_', which is reserved "
                "for policy-evaluation datasets recorded through the rollout flow."
            )
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            # lerobot 0.6.0: vcodec moved off DatasetRecordConfig into the nested
            # rgb_encoder/depth_encoder configs; resume/create now take those.
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    # 🔧 ROBOT CONNECTION: Connect with enhanced error handling for camera conflicts.
    #
    # A camera can read back a degraded fps on the FIRST open — macOS AVFoundation
    # reports e.g. actual_fps=5.0 when the device was just released by a browser
    # preview / enumeration probe and the OS-level release (which is asynchronous)
    # hasn't settled. lerobot's _validate_fps then raises before any warmup frame.
    # The device isn't broken: a cold re-open a moment later negotiates 30fps
    # cleanly, so retry the transient fps failure a couple of times.
    # Refine the coarse "preparing" phase into a named substep so the UI can say
    # "Connecting arm & cameras…" instead of one opaque "preparing session".
    # robot.connect() opens the follower bus AND its cameras in one call, so we
    # can't cleanly split arm-connect from camera-warmup — they share this
    # substep. (Both surface through the same current_phase global the status
    # handler already returns; no new plumbing.)
    _set_phase("connecting_robot")
    phase_start_time = time.time()
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            logger.info(
                "🔧 ROBOT CONNECTION: Attempting to connect robot (attempt %d/%d)...",
                attempt,
                _CONNECT_ATTEMPTS,
            )
            # Calibration is already on disk (loaded via the configs above), so never
            # let connect() drop into interactive recalibration — that would hang the
            # headless record thread (the "stuck on preparing session" symptom).
            robot.connect(calibrate=False)
            logger.info("✅ ROBOT CONNECTION: Robot connected successfully")
            break
        except Exception as e:
            msg = str(e)
            transient_camera = _is_transient_camera_error(msg)
            logger.error(f"❌ ROBOT CONNECTION: Failed to connect robot: {e}")
            # If robot connection fails due to camera conflict, provide clear error
            if (
                "camera" in msg.lower()
                or "device" in msg.lower()
                or "busy" in msg.lower()
                or transient_camera
            ):
                logger.error(
                    "💡 ROBOT CONNECTION: Camera connection failure - resource conflict or cold-open session turbulence"
                )
                logger.error(
                    "💡 ROBOT CONNECTION: Make sure frontend camera streams are released before recording"
                )
            will_retry = attempt < _CONNECT_ATTEMPTS and transient_camera
            if will_retry:
                # Enter the retry substep BEFORE the teardown, not after: the
                # component-wise release below is the slow part (each wedged
                # camera's disconnect joins a read thread that is waiting out
                # a frame timeout), so bracketing only the 2s sleep would
                # leave the operator staring at a static "Connecting arm &
                # cameras…" for the window this substep exists to explain.
                # Logged at INFO through this module's own logger (already in
                # _RECORD_LOG_LOGGER_NAMES) so it reaches the visible
                # Record-page log panel, not just the server console.
                _set_phase("reconnecting_robot")
                # The attempt about to run, not the one that just failed —
                # same convention as the log line below, so the Record page's
                # dialog and its log panel can't disagree about the number
                # (they sit one above the other).
                connect_retry_attempt = attempt + 1
                logger.info(
                    "🔁 ROBOT CONNECTION: Camera hiccup, retrying (%d/%d) after OS release settles...",
                    attempt + 1,
                    _CONNECT_ATTEMPTS,
                )
            # Drop any half-open handles from this failed attempt so the retry
            # starts from a clean device — and so a *terminal* failure doesn't
            # leak the opened bus and camera read threads into the rest of the
            # process. Must be the component-wise teardown, not
            # robot.disconnect(): connect() dies with the later cameras still
            # unopened, and lerobot's all-or-nothing is_connected guard makes
            # disconnect() a no-op raise in exactly that state (see
            # force_disconnect_partial).
            force_disconnect_partial(robot, "robot")
            if will_retry:
                # Let the OS release settle past the turbulence window before
                # re-rolling the connect.
                time.sleep(2.0)
                _set_phase("connecting_robot")
                # Back to a plain connect attempt: keep the "0 unless we are
                # in reconnecting_robot" contract this field documents, so a
                # later reader can't mistake a healthy session for a retrying
                # one.
                connect_retry_attempt = 0
                continue
            raise

    if teleop is not None:
        # Second detectable substep of the preparing window: the leader bus.
        _set_phase("connecting_teleop")
        phase_start_time = time.time()
        try:
            logger.info("🔧 TELEOP CONNECTION: Attempting to connect teleoperator...")
            # calibrate=False for the same reason as the robot connect above —
            # and critically for the identity guard below: with the default
            # calibrate=True, an EEPROM/file mismatch (exactly the swapped-port
            # case) drops into SOLeader.calibrate(), which writes the wrong
            # JSON into the servos' EEPROM (or hangs on input() in this
            # headless thread). The calibration write happens explicitly in
            # _write_calibration below, after the guard has read the EEPROM.
            teleop.connect(calibrate=False)
            logger.info("✅ TELEOP CONNECTION: Teleoperator connected successfully")
        except Exception as e:
            logger.error(f"❌ TELEOP CONNECTION: Failed to connect teleoperator: {e}")
            # The robot connected fine a moment ago; release both so a leader
            # failure can't strand the follower's bus and camera threads for
            # the rest of the process (partial teardown for the same reason as
            # the robot-connect path above).
            force_disconnect_partial(robot, "robot")
            force_disconnect_partial(teleop, "teleop")
            raise

    # Arm-identity guard: read-only check that each connected arm matches its
    # assigned calibration, BEFORE _write_calibration below can stamp a wrong
    # file into a swapped arm's EEPROM and before any action is sent. On a hard
    # mismatch, release the arms (torque was never enabled) and let the worker's
    # error path surface the message via the recording status.
    try:
        identity_warnings = verify_devices(
            ((robot, "follower"), (teleop, "leader")),
            # A Maker arm has no EEPROM fingerprint to compare — its zero lives
            # inside the RobStride motors and its calibration writes
            # homing_offset=0 for every joint — so the guard is skipped whole
            # rather than left to fail open per arm.
            skip=skip_identity_check or not feetech,
            config_names=identity_config_names,
        )
    except ArmIdentityError:
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        raise

    # Ensure calibration is properly loaded and applied to the devices
    logger.info("Applying calibration to devices")

    # Write calibration to motors' memory (similar to teleoperation code). A
    # single-arm device has its own .bus/.calibration; a bimanual BiSO device
    # exposes left_arm/right_arm sub-arms instead, so write each of those.
    def _write_calibration(device, label: str) -> None:
        if device is None:
            return
        sub_arms = [
            a
            for a in (getattr(device, "left_arm", None), getattr(device, "right_arm", None))
            if a is not None
        ]
        targets = sub_arms if sub_arms else [device]
        wrote = False
        for target in targets:
            if hasattr(target, "bus") and getattr(target, "calibration", None) is not None:
                try:
                    target.bus.write_calibration(target.calibration)
                    wrote = True
                except Exception as e:
                    logger.error(f"Error writing {label} calibration: {e}")
        if wrote:
            logger.info(f"{label.capitalize()} calibration applied successfully")
        else:
            logger.warning(
                f"{label.capitalize()} bus or calibration not available - calibration may not be applied"
            )

    if feetech:
        _write_calibration(robot, "robot")
        _write_calibration(teleop, "teleop")
    else:
        # The Maker devices' own connect() already registered their calibration
        # on the bus. Re-writing it here would also hit the Star 102 leader,
        # whose bus is a FashionStar handle with no write_calibration at all —
        # its zero is set by set_origin_point during calibration and lives in
        # the servos, not in a file the bus reloads.
        logger.info("Maker arm: calibration registered by connect(); skipping the explicit write")

    # Stock session torque (RAM Torque_Limit re-seeded from EEPROM) — the
    # follower only, never the human-held leader. Clears any torque cap a
    # previous auto-calibration left in RAM; a failed write degrades to the
    # previous limit (logged inside) and must not abort the session.
    if feetech:
        reset_torque_limit(robot, "follower arm")
        # Clear any leftover Goal_Velocity speed cap a previous arm-driving feature
        # stamped in RAM (auto-cal fold/unfold=1000, rest-pose return=400); the
        # follower only, never the human-held leader. See makermodslab/motor_power.py.
        clear_goal_velocity(robot, "follower arm")

    # Capture the follower's rest pose now — after connect/configure/identity
    # guard, before the recording loop moves anything — so a normal stop can
    # drive it back to where the user left it (same as teleop; see
    # makermodslab/rest_pose.py). Followers only (a bimanual BiSO robot exposes two
    # follower buses), NEVER the human-held leader. The gripper is excluded: at
    # stop time it may be holding an object, and returning it to its (likely
    # open) starting width would drop the object mid-return.
    follower_rest_poses = (
        [
            (bus, {m: v for m, v in capture_rest_pose(bus).items() if m != "gripper"})
            for bus in _device_buses(robot)
        ]
        if feetech
        else []
    )

    # Start with episode 1 - but track it properly
    current_episode = 1
    saved_episodes = 0  # Track how many episodes we've actually saved

    # Set once the session ends on a user/planned path (stop button, all
    # episodes recorded) — those get the torque-release grace below. An
    # exception (camera death, unplugged bus, ...) leaves it False so the
    # release is attempted immediately: the bus may already be unreachable.
    ended_normally = False

    try:
        while saved_episodes < cfg.dataset.num_episodes:
            # RECORDING PHASE - with dataset (matches original record.py exactly)
            _set_phase("recording")
            phase_start_time = time.time()
            # Defense in depth alongside the phase gate in handle_recording_status:
            # a reset phase's pause bookkeeping must never survive into the
            # recording phase that follows it, whether this is the very first
            # recording phase of the session or the one after a reset gap.
            paused_accum_seconds = 0.0
            pause_started_at = None
            logger.info(f"Starting recording phase for episode {current_episode}")
            logger.info(f"Events state at start of recording phase: {web_events}")
            print(
                f"🎬 STATUS CHANGE: Starting recording phase for episode {current_episode}/{cfg.dataset.num_episodes}"
            )

            log_say(f"Recording episode {current_episode}", cfg.play_sounds)

            # Add a tracking flag that won't be reset by record_loop
            web_events["_exit_early_triggered"] = False
            logger.info(f"Recording phase - calling record_loop with events: {web_events}")

            record_loop(
                robot=robot,
                events=web_events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            logger.info(f"Recording phase completed - events state: {web_events}")

            # Stop pressed mid-episode: discard the in-progress (incomplete)
            # episode and end the session immediately — no save, no reset phase,
            # no re-record. Checked BEFORE the exit-early/timeout classification
            # below, because handle_stop_recording sets exit_early (so record_loop
            # returns early) but deliberately does NOT set _exit_early_triggered.
            # Without this short-circuit the early return would be misclassified
            # as a timeout, flip rerecord_episode on, and drag the user through a
            # full reset phase before honoring the stop. Previously saved episodes
            # stay saved; the empty-dataset discard path handles the case where
            # this was episode 1 and nothing was ever saved.
            if web_events["stop_recording"]:
                logger.info(
                    "🛑 STOP RECORDING requested during recording phase - "
                    "discarding incomplete episode and ending session"
                )
                print(
                    "🛑 STATUS CHANGE: Stopped by user during recording - "
                    "incomplete episode discarded, ending session"
                )
                dataset.clear_episode_buffer()
                break

            # Check if exit_early was triggered (use our tracking flag)
            recording_interrupted_by_exit_early = web_events.get("_exit_early_triggered", False)
            if recording_interrupted_by_exit_early:
                logger.info("🟡 RECORDING PHASE INTERRUPTED BY EXIT_EARLY - proceeding to save episode")
                print(
                    f"🟡 STATUS CHANGE: Recording phase interrupted by user - episode {current_episode} data collected"
                )
                # Reset our tracking flag
                web_events["_exit_early_triggered"] = False
            else:
                # Recording completed due to timeout - trigger re-record behavior
                logger.info("⏰ RECORDING PHASE COMPLETED DUE TO TIMEOUT - triggering re-record")
                print(
                    f"⏰ STATUS CHANGE: Recording timeout reached for episode {current_episode} - re-recording"
                )
                web_events["rerecord_episode"] = True

            # Handle rerecord logic first (before saving)
            if web_events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                print(
                    f"🔄 STATUS CHANGE: Re-recording episode {current_episode} (episode number stays the same)"
                )
                web_events["rerecord_episode"] = False
                web_events["exit_early"] = False
                dataset.clear_episode_buffer()

                # Go through reset phase before re-recording (don't increment episode counters)
                # RESET PHASE - without dataset (matches original record.py exactly)
                _set_phase("resetting")
                phase_start_time = time.time()
                paused_accum_seconds = 0.0
                pause_started_at = None
                logger.info(f"Starting reset phase for re-record of episode {current_episode}")
                logger.info(f"Events state at start of reset phase: {web_events}")
                print(f"🔄 STATUS CHANGE: Starting reset phase for episode {current_episode}")

                log_say("Reset the environment", cfg.play_sounds)

                # Reset exit_early at the start of each phase. Don't clear
                # `paused` unconditionally: an operator may have armed a
                # pause during the just-finished recording phase for this
                # exact reset gap (see handle_pause_recording) — preserve it
                # and start its wall-clock accounting now that the countdown
                # it affects has actually begun.
                web_events["exit_early"] = False
                pause_started_at = time.time() if web_events.get("paused", False) else None
                logger.info(f"Reset phase - calling reset loop with events: {web_events}")

                _reset_loop_with_pause(
                    robot=robot,
                    teleop=teleop,
                    events=web_events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    control_time_s=cfg.dataset.reset_time_s,
                )

                # The loop may have exited (e.g. via exit_early/stop) while
                # still paused. Clear the actual pause state here rather than
                # relying solely on the phase-aware _reset_paused() gate, so a
                # paused flag can never survive past this reset phase's end.
                web_events["paused"] = False
                pause_started_at = None

                logger.info(f"Reset phase completed - events state: {web_events}")

                # Check if reset was interrupted by exit_early
                if web_events["exit_early"]:
                    logger.info("🟡 RESET PHASE INTERRUPTED BY EXIT_EARLY during re-record")
                    print("🟡 STATUS CHANGE: Reset phase interrupted by user during re-record")
                    web_events["exit_early"] = False

                # Check if stop recording was requested during re-record reset phase
                if web_events["stop_recording"]:
                    logger.info("🛑 STOP RECORDING requested during re-record reset phase - ending session")
                    print(
                        "🛑 STATUS CHANGE: Stop recording requested during re-record reset - ending session"
                    )
                    break

                # Don't increment current_episode or saved_episodes - we're re-recording the same episode
                continue

            # Save episode immediately after recording phase (matches expected flow)
            logger.info(f"💾 Saving episode {current_episode}...")
            print(f"💾 STATUS CHANGE: Saving episode {current_episode}")
            dataset.save_episode()
            logger.info(f"✅ Episode {current_episode} saved successfully")
            print(f"✅ STATUS CHANGE: Episode {current_episode} saved successfully")

            # Increment episode counters after successful save
            saved_episodes += 1
            current_episode += 1

            # Check if we should stop recording
            if web_events["stop_recording"]:
                print("🛑 STATUS CHANGE: Recording manually stopped by user")
                break

            # Check if we've completed all episodes
            if saved_episodes >= cfg.dataset.num_episodes:
                break

            # Execute reset phase to prepare for next episode
            # Skip reset for the last episode that was just saved
            if saved_episodes < cfg.dataset.num_episodes:
                # RESET PHASE - without dataset (matches original record.py exactly)
                _set_phase("resetting")
                phase_start_time = time.time()
                paused_accum_seconds = 0.0
                pause_started_at = None
                logger.info(f"Starting reset phase for next episode {current_episode}")
                logger.info(f"Events state at start of reset phase: {web_events}")
                print(f"🔄 STATUS CHANGE: Starting reset phase for episode {current_episode}")

                log_say("Reset the environment", cfg.play_sounds)

                # Reset exit_early at the start of each phase. Don't clear
                # `paused` unconditionally: an operator may have armed a
                # pause during the just-finished recording phase for this
                # exact reset gap (see handle_pause_recording) — preserve it
                # and start its wall-clock accounting now that the countdown
                # it affects has actually begun.
                web_events["exit_early"] = False
                pause_started_at = time.time() if web_events.get("paused", False) else None
                logger.info(f"Reset phase - calling reset loop with events: {web_events}")

                _reset_loop_with_pause(
                    robot=robot,
                    teleop=teleop,
                    events=web_events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    control_time_s=cfg.dataset.reset_time_s,
                )

                # The loop may have exited (e.g. via exit_early/stop) while
                # still paused. Clear the actual pause state here rather than
                # relying solely on the phase-aware _reset_paused() gate, so a
                # paused flag can never survive past this reset phase's end.
                web_events["paused"] = False
                pause_started_at = None

                logger.info(f"Reset phase completed - events state: {web_events}")

                # Check if reset was interrupted by exit_early
                if web_events["exit_early"]:
                    logger.info("🟡 RESET PHASE INTERRUPTED BY EXIT_EARLY - proceeding to next episode")
                    print("🟡 STATUS CHANGE: Reset phase interrupted by user - proceeding to next episode")
                    web_events["exit_early"] = False

                # Check if stop recording was requested during reset phase
                if web_events["stop_recording"]:
                    logger.info("🛑 STOP RECORDING requested during reset phase - ending session")
                    print("🛑 STATUS CHANGE: Stop recording requested during reset - ending session")
                    break

        # Recording completed
        _set_phase("completed")
        phase_start_time = None
        print("🏁 STATUS CHANGE: Recording session completed - all episodes finished")
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        ended_normally = True

    finally:
        try:
            if ended_normally and not _release_now.is_set():
                # User-initiated stop / planned session end: no timed hold — the
                # servos hold their last goal on their own in position mode — so
                # drive the follower(s) straight back to their session-start
                # pose, then release (same behavior as the teleop / auto-cal
                # stop). A second stop (release-now) skips/aborts the return;
                # error exits skip this — the bus may be gone, release ASAP.
                releasing = True
                # Still energized and holding the ports — a phase of this
                # session, not idle yet (the worker's finally emits the final
                # release hint once cleanup is done).
                notify_session_changed("recording", True, phase="releasing")
                _return_followers_to_rest(follower_rest_poses, _release_now)
            # Belt and braces: disable torque explicitly before disconnect, so a
            # failure inside disconnect() can't leave an arm energized (rigid).
            # force_disable_torque logs any failure at ERROR level with the port.
            if feetech:
                force_disable_torque(robot, "robot")
                force_disable_torque(teleop, "teleop")
            else:
                release_maker_torque(robot, "Maker follower arm")
            robot.disconnect()
            if teleop:
                teleop.disconnect()
        finally:
            releasing = False

    if cfg.dataset.push_to_hub:
        # Same bare-id hazard as the UploadManager path, and the same single
        # workaround: cfg.dataset.repo_id is the local (bare) id, and calling
        # dataset.push_to_hub directly would let create_repo resolve it while
        # the upload_folder right after it doesn't — stranding an empty repo.
        # Unreachable from the browser today (the recording dialog sends
        # push_to_hub: false and lets the user upload afterwards), but this is
        # an HTTP API a non-UI caller can set.
        push_dataset_to_hub(cfg.dataset.repo_id, tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset
