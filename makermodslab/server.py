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

import asyncio
import concurrent.futures
import contextlib
import ctypes
import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from lerobot.policies.factory import make_policy_config

# Module objects (not from-imports) so live attribute access always sees the
# current value (e.g. globals a feature module rebinds at runtime need attribute
# lookup at call time, not a bound name frozen at import).
from . import (
    datasets as dataset_browser,
    models as model_browser,
    record as record_state,
    rollout as rollout_state,
)

# Import our custom calibration functionality
from .auto_calibrate import (
    AutoCalibrationBatchRequest,
    AutoCalibrationRequest,
    auto_calibration_batch_manager,
    auto_calibration_manager,
)
from .calibrate import CalibrationRequest, calibration_manager
from .camera_identity import resolve_cv2_index
from .camera_preview import CameraOpenError, camera_preview_manager
from .identify import identify_arm_by_motion
from .jobs import (
    DatasetNotOnHubError,
    JobAlreadyContinuedError,
    JobAlreadyRunningError,
    JobHasChildrenError,
    JobNotFoundError,
    JobNotRunningError,
    JobTarget,
    _list_local_checkpoints,
    job_registry,
)
from .merge import MergeRequest, handle_merge_status, handle_start_merge
from .motor_power import read_supply_voltage

# Import our custom recording functionality
from .record import (
    DatasetInfoRequest,
    RecordingRequest,
    UploadRequest,
    handle_delete_dataset,
    handle_exit_early,
    handle_pause_recording,
    handle_recording_log,
    handle_recording_status,
    handle_rerecord_episode,
    handle_resume_recording,
    handle_start_recording,
    handle_stop_recording,
    handle_upload_dataset,
    handle_upload_status,
    stop_and_wait as stop_recording_and_wait,
)
from .rollout import (
    InferenceRequest,
    handle_inference_log,
    handle_inference_status,
    handle_next_episode,
    handle_start_inference,
    handle_stop_episode,
    handle_stop_inference,
)

# Import our custom teleoperation functionality
from .teleoperate import (
    TeleoperateRequest,
    handle_start_teleoperation,
    handle_stop_teleoperation,
    handle_teleoperation_status,
    stop_and_wait as stop_teleoperation_and_wait,
)

# Training is now job-based; see app/jobs.py.
from .train import TrainingRequest
from .update import handle_run_update, handle_update_check
from .utils.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    add_dismissed_hub_job,
    add_hidden_dataset,
    add_hidden_model,
    add_saved_custom_dataset,
    add_saved_custom_model,
    calibration_dir_for_device,
    clear_config_references,
    config_slot_conflict,
    delete_robot_record,
    find_available_ports,
    get_default_robot_port,
    get_dismissed_hub_jobs,
    get_robot_record,
    get_saved_robot_port,
    is_robot_record_clean,
    is_valid_robot_name,
    list_robot_records,
    port_slot_conflict,
    prune_dismissed_hub_jobs,
    remove_hidden_dataset,
    remove_hidden_model,
    remove_saved_custom_dataset,
    remove_saved_custom_model,
    rename_calibration_config,
    rename_robot_record,
    save_imported_calibration,
    save_robot_record,
)
from .utils.hf_auth import (
    cached_whoami,
    handle_hf_auth_status,
    handle_hf_login,
    hf_hub_offline,
    shared_hf_api,
)
from .utils.system import (
    handle_get_policy_extra,
    handle_get_training_extra,
    handle_get_wandb_extra,
    handle_install_policy_extra,
    handle_install_policy_extra_status,
    handle_install_training_extra,
    handle_install_training_extra_status,
    handle_install_wandb_extra,
    handle_install_wandb_extra_status,
    open_folder_in_file_browser,
    warn_if_cuda_mismatch,
)
from .wiggle import wiggle_gripper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# High-frequency read-only status polls (~2 Hz each from the frontend) that
# drown the uvicorn access log and bury real warnings (a torque warning was
# once lost in this noise). Successful GETs to these EXACT paths (query string
# ignored) are dropped from the access log; non-GETs, other paths (including
# subpaths like /jobs/{id}/logs), and error responses still log.
_QUIET_STATUS_POLL_PATHS = {
    "/auto-calibration-status",
    "/auto-calibration-batch-status",
    "/calibration-status",
    "/teleoperation-status",
    "/recording-status",
    "/jobs",
}


class _StatusPollAccessFilter(logging.Filter):
    """Drop uvicorn.access records for successful high-frequency status polls.

    uvicorn.access records carry args = (client_addr, method, full_path,
    http_version, status_code); anything else passes through untouched.
    Only affects the access log — app-level loggers are not filtered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _, method, full_path, _, status_code = args
        if method != "GET" or not isinstance(status_code, int):
            return True
        # Errors (and redirects-gone-wrong) must still log.
        if status_code >= 400:
            return True
        path = str(full_path).split("?", 1)[0]
        return path not in _QUIET_STATUS_POLL_PATHS


logging.getLogger("uvicorn.access").addFilter(_StatusPollAccessFilter())


class StartTrainingBody(BaseModel):
    """Wrapping body for POST /jobs/training. Adds optional target spec."""

    config: TrainingRequest
    target: JobTarget | None = None

    @classmethod
    def from_legacy(cls, raw: dict) -> "StartTrainingBody":
        """Accept the old request shape (TrainingRequest fields at top level)
        as well as the new shape ({config: ..., target: ...}).
        """
        if "config" in raw and isinstance(raw["config"], dict):
            return cls.model_validate(raw)
        # Legacy: top-level training fields, no target.
        return cls(config=TrainingRequest.model_validate(raw))


# Cache for HF Jobs hardware flavors (5-minute TTL)
_flavors_cache: dict = {"data": None, "fetched_at": 0.0}
_FLAVOR_CACHE_TTL_SECONDS = 300.0


app = FastAPI()

# In dev mode the React app runs on :8080 while the API runs on :8000; in
# prod they share an origin and CORS is unnecessary. allow_credentials with
# a wildcard origin is rejected by browsers, so we drop it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Get the path to the lerobot root directory (3 levels up from this script)
LEROBOT_PATH = str(Path(__file__).parent.parent.parent.parent)
logger.info(f"LeRobot path: {LEROBOT_PATH}")


class ConnectionManager:
    def __init__(self):
        # Each websocket is bound to the asyncio loop that accepted it; sends
        # from the broadcast worker thread must be marshaled onto that loop.
        self.active_connections: dict[WebSocket, asyncio.AbstractEventLoop] = {}
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False
        # Guards `active_connections` since the broadcast worker thread also
        # mutates it on send failure.
        self._connections_lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._connections_lock:
            self.active_connections[websocket] = asyncio.get_running_loop()
            count = len(self.active_connections)
        logger.info(f"WebSocket connected. Total connections: {count}")

        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        """Remove a connection and stop the worker if none remain.

        Only called from request-handler context (the endpoint's cleanup and
        server shutdown), never from the broadcast worker — the worker uses
        _drop_connection so it can't end up joining its own thread.
        """
        with self._connections_lock:
            if self.active_connections.pop(websocket, None) is not None:
                count = len(self.active_connections)
                logger.info(f"WebSocket disconnected. Total connections: {count}")
            else:
                count = len(self.active_connections)

        if count == 0 and self.is_running:
            self.stop_broadcast_thread()

    def _drop_connection(self, websocket: WebSocket):
        """Forget a connection whose send failed, without stopping the worker.

        The endpoint's receive loop notices the disconnect independently and
        its cleanup calls disconnect(), which is where thread stop happens.
        """
        with self._connections_lock:
            if self.active_connections.pop(websocket, None) is not None:
                count = len(self.active_connections)
                logger.info(f"Dropped unreachable WebSocket. Total connections: {count}")

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data"""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
        self.broadcast_thread.start()
        logger.info("📡 Broadcast thread started")

    def stop_broadcast_thread(self):
        """Signal the worker thread to stop. Never joins.

        Joining here is unsafe in both directions: from the uvicorn event
        loop it can stall the loop while the worker waits on a send it
        scheduled onto that same loop, and from the worker itself it would
        be a self-join. The daemon worker notices the cleared flag (or a
        newer thread replacing it) within its 0.1 s queue timeout and exits.
        """
        self.is_running = False
        self.broadcast_thread = None
        logger.info("📡 Broadcast thread stop requested")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data"""
        me = threading.current_thread()
        # The identity check makes a rapid stop→start cycle safe: if a new
        # worker has been started, this one exits even though is_running is
        # True again.
        while self.is_running and self.broadcast_thread is me:
            try:
                # Get data from queue with timeout
                data = self.broadcast_queue.get(timeout=0.1)
                if data is None:  # Poison pill to stop
                    break

                self._send_to_all_connections(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")

        logger.info("📡 Broadcast thread stopped")

    def _send_to_all_connections(self, data: dict[str, Any]):
        """Send data to all active connections, each on its own event loop.

        Runs on the broadcast worker thread: every send is submitted to the
        loop that accepted the websocket via run_coroutine_threadsafe (a
        websocket's ASGI channel is not usable from any other loop). All
        sends are submitted before any is waited on so one slow client
        doesn't delay the others.
        """
        with self._connections_lock:
            connections = list(self.active_connections.items())
        if not connections:
            return

        pending = []
        for connection, loop in connections:
            try:
                future = asyncio.run_coroutine_threadsafe(connection.send_json(data), loop)
            except Exception as e:  # loop closed or shutting down
                logger.error(f"Error scheduling send to WebSocket: {e}")
                self._drop_connection(connection)
            else:
                pending.append((connection, future))

        for connection, future in pending:
            try:
                future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Error sending data to WebSocket: {e}")
                future.cancel()
                self._drop_connection(connection)

    def broadcast_joint_data_sync(self, data: dict[str, Any]):
        """Thread-safe method to queue data for broadcasting"""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")

    def notify_jobs_changed(self) -> None:
        """Push a 'jobs_changed' event to all WS clients so they refetch.

        Called from JobRegistry on submit / watchdog finalisation / delete.
        Skipped silently if no clients are connected — the frontend does an
        initial fetch on mount, so a missed broadcast is self-healing.
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait({"type": "jobs_changed", "timestamp": time.time()})

    def notify_job_progress(self, snapshots: list[dict]) -> None:
        """Push a 'job_progress' event with per-running-job snapshots.

        Fired from the JobRegistry watchdog (~1Hz) while jobs are running so
        the dashboard's progress bar updates live without refetching /jobs
        (let alone /jobs/hub, which hits the HF API on every call).
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(
                    {"type": "job_progress", "jobs": snapshots, "timestamp": time.time()}
                )


manager = ConnectionManager()
job_registry.set_on_change(manager.notify_jobs_changed)
job_registry.set_on_progress(manager.notify_job_progress)


# Frontend policy_type -> lerobot registry name. In this lerobot pin the names
# match 1:1 (pi0_fast registers as "pi0_fast", not the older "pi0fast").
# reward_classifier is NOT a policy in this pin: it registers under the
# separate RewardModelConfig registry (lerobot/rewards/), so make_policy_config
# raises for it and it reports available=False below. Keep in sync with
# POLICY_TYPE_OPTIONS in frontend/src/components/training/types.ts.
_POLICY_TYPE_TO_LEROBOT = {
    "act": "act",
    "diffusion": "diffusion",
    "pi0": "pi0",
    "pi05": "pi05",
    "smolvla": "smolvla",
    "tdmpc": "tdmpc",
    "vqbet": "vqbet",
    "pi0_fast": "pi0_fast",
    "gaussian_actor": "gaussian_actor",
    "reward_classifier": "reward_classifier",
}

# Optimizer preset class name -> frontend optimizer_type value.
_OPTIMIZER_CLASS_TO_NAME = {
    "adamw": "adamw",
    "adam": "adam",
    "multiadam": "multi_adam",
    "sgd": "sgd",
}


def _optimizer_name_from_preset(preset) -> str:
    """Derive the optimizer_type value from the preset config class name.

    e.g. AdamWConfig -> "adamw", MultiAdamConfig -> "multi_adam". Falls back to
    the lowercased class name (with a trailing "config" stripped) for unknown
    types so we never crash on an optimizer we haven't mapped.
    """
    name = type(preset).__name__.lower()
    if name.endswith("config"):
        name = name[: -len("config")]
    return _OPTIMIZER_CLASS_TO_NAME.get(name, name)


@app.get("/policy-optimizer-defaults")
def get_policy_optimizer_defaults():
    """Return each policy's optimizer preset (lr / weight_decay / grad_clip_norm
    + optimizer type) so the training UI can show the real "policy default"
    instead of a generic placeholder.

    Every frontend policy_type is included. `available` says whether this
    lerobot pin can construct the policy config at all — false means a training
    run with that type is doomed at policy construction, so the UI disables the
    button (e.g. reward_classifier, which isn't a policy in this pin). Policies
    whose config exists but whose optimizer preset can't be read stay available
    with a null entry in `defaults`.
    """
    defaults: dict[str, Any] = {}
    available: dict[str, bool] = {}
    for frontend_name, lerobot_name in _POLICY_TYPE_TO_LEROBOT.items():
        try:
            config = make_policy_config(lerobot_name)
        except Exception as e:
            logger.warning(
                "Policy %r (lerobot %r) is unavailable in this lerobot install: %s",
                frontend_name,
                lerobot_name,
                e,
            )
            available[frontend_name] = False
            defaults[frontend_name] = None
            continue
        available[frontend_name] = True
        try:
            preset = config.get_optimizer_preset()
            defaults[frontend_name] = {
                "optimizer": _optimizer_name_from_preset(preset),
                "lr": preset.lr,
                "weight_decay": preset.weight_decay,
                "grad_clip_norm": preset.grad_clip_norm,
            }
        except Exception as e:
            logger.warning(
                "No optimizer preset for policy %r (lerobot %r): %s",
                frontend_name,
                lerobot_name,
                e,
            )
            defaults[frontend_name] = None

    return {"defaults": defaults, "available": available}


@app.post("/move-arm")
def teleoperate_arm(request: TeleoperateRequest):
    """Start teleoperation of the robot arm"""
    return handle_start_teleoperation(request, manager)


@app.post("/stop-teleoperation")
def stop_teleoperation():
    """Stop the current teleoperation session"""
    return handle_stop_teleoperation()


@app.get("/teleoperation-status")
def teleoperation_status():
    """Get the current teleoperation status"""
    return handle_teleoperation_status()


@app.post("/start-inference")
def start_inference(request: InferenceRequest):
    result = handle_start_inference(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start inference"),
        )
    return result


@app.post("/stop-inference")
def stop_inference():
    """Abort the whole session. In evaluation mode (eval_episodes > 1) this ends
    the run wherever it is and reports the partial tally with NO accuracy — the
    per-episode control is /inference-episode-stop."""
    result = handle_stop_inference()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop inference"),
        )
    return result


@app.post("/inference-episode-stop")
def inference_episode_stop():
    """Evaluation mode only: end the CURRENT episode early and score it a
    SUCCESS ("the robot did the task"). The session stays up and moves into its
    reset phase. 409 when no evaluation episode is running."""
    result = handle_stop_episode()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop the episode"),
        )
    return result


@app.post("/inference-next-episode")
def inference_next_episode():
    """Evaluation mode only: leave the reset phase and start the next episode.
    The reset is user-ended (no auto-timer). 409 unless an evaluation is parked
    waiting for a reset."""
    result = handle_next_episode()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start the next episode"),
        )
    return result


@app.get("/inference-status")
def inference_status():
    return handle_inference_status()


@app.get("/inference-log")
def inference_log():
    """Tail of the active/most-recent rollout's log file (read-only, bounded).

    Returns {logs, log_path, belongs_to}; empty logs (not an error) when no run
    has produced output yet, so the frontend can poll unconditionally.

    `belongs_to` is "active" (the running session's own log), "last_run" (the
    most recent finished run of this server process) or null (nothing to show) —
    the caller must not present a "last_run" log as the live session's output."""
    return handle_inference_log()


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running"""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.get("/hf-auth-status")
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


class HfLoginBody(BaseModel):
    token: str


@app.post("/hf-auth/login")
def hf_auth_login(body: HfLoginBody):
    """Persist a pasted HF token (validated against whoami) for this user."""
    try:
        return handle_hf_login(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/datasets")
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()


@app.get("/datasets/info")
def datasets_info(repo_id: str):
    """Detail card for one dataset. Local cache first (full detail: episodes,
    cameras, tasks, size on disk — ``source: "local"``); a dataset with no
    local copy falls back to a Hub summary read from its meta/info.json
    (episodes/frames/fps/robot/cameras, no tasks/size — ``source: "hub"``).
    404 only when NEITHER resolves (offline / unknown repo). repo_id is a query
    param because repo ids contain '/'."""
    info = dataset_browser.get_local_dataset_info(repo_id)
    if info is None:
        info = dataset_browser.get_hub_dataset_info(repo_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{repo_id}' not found in the local cache")
    return info


@app.get("/datasets/episodes")
def datasets_episodes(repo_id: str):
    """Per-episode index/length/duration/tasks for the dataset viewer window.
    404 when the dataset isn't local or predates the v3.0 parquet episode
    layout (meta/episodes/chunk-*/file-*.parquet) the viewer reads."""
    episodes = dataset_browser.list_episode_summaries(repo_id)
    if episodes is None:
        raise HTTPException(status_code=404, detail=f"No viewable episode list for '{repo_id}'")
    return episodes


@app.get("/datasets/episode-joints")
def datasets_episode_joints(repo_id: str, episode_index: int):
    """Per-frame timestamp + joint (observation.state) values for one episode,
    for the dataset viewer's joint-position chart."""
    series = dataset_browser.get_episode_joint_series(repo_id, episode_index)
    if series is None:
        raise HTTPException(
            status_code=404, detail=f"No joint data for episode {episode_index} of '{repo_id}'"
        )
    return series


@app.get("/datasets/episode-video")
def datasets_episode_video(repo_id: str, episode_index: int, camera: str):
    """The mp4 backing one camera's footage for one episode, served straight
    off disk. FileResponse handles Range requests, so the <video> element can
    seek without downloading the whole file first."""
    video_path = dataset_browser.get_episode_video_path(repo_id, episode_index, camera)
    if video_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No video for camera '{camera}', episode {episode_index} of '{repo_id}'",
        )
    return FileResponse(video_path, media_type="video/mp4")


class DatasetEpisodeDeleteBody(BaseModel):
    repo_id: str
    episode_index: int


@app.post("/datasets/episode-delete")
def datasets_episode_delete(body: DatasetEpisodeDeleteBody):
    """Delete one episode from a locally-cached dataset, rewriting it via
    lerobot's delete_episodes. Never touches the Hub — if this dataset is
    also on the Hub, its published copy is left exactly as it was; use
    "Upload to Hub" to push the edited version manually. Refuses (409) if
    the dataset is being recorded, merged, uploaded, or trained on locally;
    400 if the index is invalid or it's the dataset's only episode."""
    try:
        return dataset_browser.delete_local_episode(body.repo_id, body.episode_index)
    except dataset_browser.DatasetEpisodeDeleteError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


@app.get("/datasets/hub-status")
def datasets_hub_status(repo_id: str):
    """Whether a dataset repo with this id exists on the Hub.

    Fetched lazily by the info card (separate from /datasets/info) so it never
    blocks the card render. Degrades to status "unknown" offline/unauthenticated
    — see get_hub_status. repo_id is a query param because repo ids contain '/'.
    """
    return dataset_browser.get_hub_status(repo_id)


@app.get("/datasets/hub-settings")
def datasets_hub_settings(repo_id: str):
    """Current Hub-side visibility + tags for a dataset, for pre-filling the
    post-upload editor. Returns ``{repo_id, private, tags}``. 400 offline;
    403/502 on a Hub failure. repo_id is a query param (repo ids contain '/')."""
    try:
        return dataset_browser.get_hub_settings(repo_id)
    except dataset_browser.DatasetHubEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class DatasetVisibilityBody(BaseModel):
    repo_id: str
    private: bool


@app.post("/datasets/visibility")
def datasets_visibility(body: DatasetVisibilityBody):
    """Flip a Hub dataset's visibility (public <-> private). MUTATES the live
    repo. 400 offline; 403 when the token can't write the namespace; 502 on any
    other Hub failure. Invalidates the cached hub-status so the card re-reads."""
    try:
        return dataset_browser.set_dataset_visibility(body.repo_id, body.private)
    except dataset_browser.DatasetHubEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class DatasetTagsBody(BaseModel):
    repo_id: str
    tags: list[str]


@app.post("/datasets/tags")
def datasets_tags(body: DatasetTagsBody):
    """Replace a Hub dataset card's ``tags:`` metadata. User tags run through
    with_makermodslab_tag first, so the required org tags are never dropped. MUTATES
    the live card. 400 offline; 403 no write permission; 502 other Hub failure.
    Returns the final tag list actually written."""
    try:
        return dataset_browser.set_dataset_tags(body.repo_id, body.tags)
    except dataset_browser.DatasetHubEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class DatasetRenameBody(BaseModel):
    repo_id: str
    new_name: str


@app.post("/datasets/rename")
def datasets_rename(body: DatasetRenameBody):
    """Rename a locally-cached dataset by moving its directory.

    `new_name` is the NAME PART ONLY — the namespace prefix stays fixed, so
    `ns/old` renamed to `new` becomes `ns/new`. Refuses (409) if the dataset is
    being recorded, merged, or trained on locally. Returns the new repo_id.
    """
    try:
        new_repo_id = dataset_browser.rename_local_dataset(body.repo_id, body.new_name)
    except dataset_browser.DatasetRenameError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    return {"success": True, "repo_id": new_repo_id}


class CustomDatasetRequest(BaseModel):
    repo_id: str


# A Hub dataset id is namespace/name; allow word chars, dot, and dash in each.
_CUSTOM_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


@app.post("/datasets/custom")
def datasets_save_custom(request: CustomDatasetRequest):
    """Pin a typed Hub dataset repo id so it persists in the picker listing.

    Called when the user selects "Use org/name" for a dataset that isn't in
    their own namespace and has no local copy. Idempotent. Invalidates the
    listing cache so the pinned dataset appears immediately.
    """
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub dataset id as namespace/name.")
    add_saved_custom_dataset(repo_id)
    # AUTO-UNHIDE: re-adding a repo the user previously removed from the list
    # must make it visible again — otherwise the pin lands behind the hidden
    # filter and the "added" dataset never appears.
    remove_hidden_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    return {"success": True, "repo_id": repo_id}


@app.delete("/datasets/custom")
def datasets_remove_custom(request: CustomDatasetRequest):
    """Unpin a saved custom dataset (does NOT touch the Hub or any local copy)."""
    repo_id = request.repo_id.strip()
    removed = remove_saved_custom_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    return {"success": removed, "repo_id": repo_id}


@app.post("/datasets/hide")
def datasets_hide(request: CustomDatasetRequest):
    """Hide a Hub dataset from the picker listing ("remove from list").

    NEVER deletes or mutates the Hub repo — it's a persistent local filter for
    hub rows the user's own namespace listing keeps returning (a pinned row is
    unpinned instead; a local copy is deleted instead). Re-pinning via
    POST /datasets/custom auto-unhides. Invalidates the listing cache only (the
    hub-status cache is untouched — the repo's Hub state didn't change)."""
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub dataset id as namespace/name.")
    add_hidden_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    dataset_browser.invalidate_hub_dataset_info(repo_id)
    return {"success": True, "repo_id": repo_id}


@app.delete("/datasets/hide")
def datasets_unhide(request: CustomDatasetRequest):
    """Unhide a dataset so it reappears in the listing (does NOT touch the Hub)."""
    repo_id = request.repo_id.strip()
    removed = remove_hidden_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    return {"success": removed, "repo_id": repo_id}


class DatasetDownloadRequest(BaseModel):
    repo_id: str


@app.post("/datasets/download")
def datasets_download(request: DatasetDownloadRequest):
    """Download a Hub dataset into the local cache in the background.

    Returns immediately with {started, repo_id, message}; poll
    /datasets/download-status for progress. The dataset lands in the flat cache
    layout so the listing source flips to "both" on completion. 400 for a
    malformed repo id; 409 when a download is already running."""
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub dataset id as namespace/name.")
    result = dataset_browser.download_manager.start(repo_id)
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Download could not be started"))
    return result


@app.get("/datasets/download-status")
def datasets_download_status():
    """Current download state (idle | running | done | error) + repo_id, message,
    and error once failed. Polled by the info card so a download survives
    navigation."""
    return dataset_browser.download_manager.get_status()


class DatasetImportRequest(BaseModel):
    path: str
    name: str | None = None


@app.post("/datasets/import")
def datasets_import(request: DatasetImportRequest):
    """Import a LeRobot dataset folder already on the server machine by COPYING
    it into the local cache (the user's source folder is left intact).

    Validates the folder is a LeRobot dataset with episodes and the target name.
    400 invalid source/name; 404 no such folder; 409 target already exists.
    Copy is synchronous — the request blocks until it completes."""
    try:
        return dataset_browser.import_local_dataset(request.path, request.name)
    except dataset_browser.DatasetImportError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


@app.post("/datasets/merge")
def datasets_merge(request: MergeRequest):
    """Aggregate 2+ datasets into a new local dataset in the background."""
    return handle_start_merge(request)


@app.get("/datasets/merge/status")
def datasets_merge_status():
    """Current merge state + drained log lines (idle | running | done | error)."""
    return handle_merge_status()


@app.websocket("/ws/joint-data")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔗 New WebSocket connection attempt")
    try:
        await manager.connect(websocket)
        logger.info("✅ WebSocket connection established")

        while True:
            # Keep the connection alive and wait for messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Handle any incoming messages if needed
                logger.debug(f"Received WebSocket message: {data}")
            except TimeoutError:
                # No message received, continue
                pass
            except WebSocketDisconnect:
                logger.info("🔌 WebSocket client disconnected")
                break

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("🧹 WebSocket connection cleaned up")


@app.post("/start-recording")
def start_recording(request: RecordingRequest):
    """Start a dataset recording session.

    Refuses (409) if recording, teleoperation, inference, calibration,
    auto-calibration, or a gripper wiggle is already active on this robot;
    400 for a malformed dataset name."""
    result = handle_start_recording(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start recording"),
        )
    return result


@app.post("/stop-recording")
def stop_recording(discard: bool = False):
    """End the current recording session.

    ``discard`` is a query parameter (not a body) so the browser's page-leave
    safety net can POST it via a keepalive/beacon "simple request" during unload
    without tripping a CORS preflight. ``discard=false`` keeps every saved
    episode (Done); ``discard=true`` throws the recording away (Quit / an
    unintentional page exit) — see handle_stop_recording."""
    return handle_stop_recording(discard=discard)


@app.get("/recording-status")
def recording_status():
    """Get the current recording status"""
    return handle_recording_status()


@app.get("/recording-log")
def recording_log():
    """Tail of the current/most-recent recording session's log (read-only,
    bounded ring buffer). Returns {logs}; empty (not an error) before a session
    has captured anything, so the frontend can poll unconditionally."""
    return handle_recording_log()


@app.post("/recording-exit-early")
def recording_exit_early():
    """Skip to next episode (replaces right arrow key)"""
    return handle_exit_early()


@app.post("/recording-rerecord-episode")
def recording_rerecord_episode():
    """Re-record current episode (replaces left arrow key)"""
    return handle_rerecord_episode()


@app.post("/recording-pause")
def recording_pause():
    """Pause the reset-phase gap between episodes (mouse-only, no keyboard
    shortcut). No-ops outside the reset phase — see handle_pause_recording."""
    return handle_pause_recording()


@app.post("/recording-resume")
def recording_resume():
    """Resume a paused reset-phase gap. No-ops if not currently paused —
    see handle_resume_recording."""
    return handle_resume_recording()


@app.post("/upload-dataset")
def upload_dataset(request: UploadRequest):
    """Start a background upload of a local dataset to the Hub.

    Returns immediately with {started, repo_id, message}; poll /upload-status
    for progress. 409 when an upload is already running (frontend maps it to a
    "an upload is already running" toast)."""
    result = handle_upload_dataset(request)
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Upload could not be started"))
    return result


@app.get("/upload-status")
def upload_status():
    """Current upload state + repo_id, message, and dataset_url once done."""
    return handle_upload_status()


@app.post("/delete-dataset")
def delete_dataset(request: DatasetInfoRequest):
    """Remove a recorded dataset directory from local disk."""
    return handle_delete_dataset(request)


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================
# A datasets-style browser for trained policies. Local models are the final
# checkpoint of each completed local training run (read from the job registry);
# Hub models are the user's LeRobot policy repos. See makermodslab/models.py.


@app.get("/models")
def models_list():
    """List trained models available to the user — local runs + Hub repos.

    Each entry carries a `source` field: "local", "hub", or "both" (a local run
    that was also pushed to the Hub). Mirrors GET /datasets."""
    return model_browser.list_all_models()


@app.get("/models/info")
def models_info(id: str):
    """Detail card for one model: policy type, base dataset, steps, size, and the
    local path (local) or Hub repo (hub). `id` is a local run id or a Hub repo id
    (a query param because repo ids contain '/'). 404 when neither resolves."""
    info = model_browser.get_model_info(id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{id}' not found")
    return info


class ModelUploadBody(BaseModel):
    id: str
    repo_id: str | None = None


@app.post("/models/upload")
def models_upload(body: ModelUploadBody):
    """Push a local run's final checkpoint to the Hub as a PUBLIC, MakerModsLab-tagged
    model repo. MUTATES the Hub (creates/updates the repo). 400 offline; 403 when
    the token can't write the namespace; 404 when the local model has no saved
    checkpoint; 502 on any other Hub failure. Returns {repo_id, url, tags}."""
    try:
        return model_browser.upload_local_model(body.id, body.repo_id)
    except model_browser.ModelError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class ModelDeleteBody(BaseModel):
    id: str


@app.post("/models/delete")
def models_delete(body: ModelDeleteBody):
    """Delete a local model — its training run's output dir (strictly sandboxed
    under outputs/train/). Never touches the Hub. 400 unsafe/non-local; 404
    unknown; 409 when the run is still training; 502 on a delete failure."""
    try:
        return model_browser.delete_local_model(body.id)
    except model_browser.ModelError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class CustomModelRequest(BaseModel):
    repo_id: str


@app.post("/models/custom")
def models_save_custom(request: CustomModelRequest):
    """Pin a Hub model repo id so it persists in the /models listing.

    The models mirror of POST /datasets/custom (same repo-id shape, same
    idempotence). Invalidates the model listing cache so the pin appears
    immediately."""
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub model id as namespace/name.")
    add_saved_custom_model(repo_id)
    # AUTO-UNHIDE: mirrors POST /datasets/custom — re-adding a hidden repo must
    # make it visible again.
    remove_hidden_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    return {"success": True, "repo_id": repo_id}


@app.delete("/models/custom")
def models_remove_custom(request: CustomModelRequest):
    """Unpin a saved custom model (does NOT touch the Hub or any local copy)."""
    repo_id = request.repo_id.strip()
    removed = remove_saved_custom_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    return {"success": removed, "repo_id": repo_id}


@app.post("/models/hide")
def models_hide(request: CustomModelRequest):
    """Hide a Hub model from the picker listing ("remove from list").

    NEVER deletes or mutates the Hub repo — a persistent local filter, the
    models mirror of POST /datasets/hide. Re-pinning via POST /models/custom
    auto-unhides. Invalidates the listing cache only."""
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub model id as namespace/name.")
    add_hidden_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    model_browser.invalidate_model_hub_info(repo_id)
    return {"success": True, "repo_id": repo_id}


@app.delete("/models/hide")
def models_unhide(request: CustomModelRequest):
    """Unhide a model so it reappears in the listing (does NOT touch the Hub)."""
    repo_id = request.repo_id.strip()
    removed = remove_hidden_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    return {"success": removed, "repo_id": repo_id}


class ModelDownloadRequest(BaseModel):
    repo_id: str


@app.post("/models/download")
def models_download(request: ModelDownloadRequest):
    """Download a Hub model checkpoint into the local models dir in the
    background. Returns immediately with {started, repo_id, message}; poll
    /models/download-status for progress. On completion the listing source flips
    to "both" and inference can run on it offline. 400 for a malformed repo id;
    409 when a download is already running (shared one-at-a-time budget with the
    dataset downloader — each manager runs its own single download)."""
    repo_id = request.repo_id.strip()
    if not _CUSTOM_REPO_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Enter a Hub model id as namespace/name.")
    result = model_browser.model_download_manager.start(repo_id)
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Download could not be started"))
    return result


@app.get("/models/download-status")
def models_download_status():
    """Current model-download state (idle | running | done | error) + repo_id,
    message, and error once failed. Polled by the model info card so a download
    survives navigation. Mirrors /datasets/download-status."""
    return model_browser.model_download_manager.get_status()


class ModelImportRequest(BaseModel):
    path: str
    name: str | None = None


@app.post("/models/import")
def models_import(request: ModelImportRequest):
    """Import a policy checkpoint folder already on the server machine by
    COPYING it into the local models dir (the source folder is left intact).

    Validates the folder is a checkpoint (config.json or checkpoints tree) and
    the target name. 400 invalid source/name; 404 no such folder; 409 target
    already exists. Copy is synchronous — the request blocks until it completes.
    Mirrors POST /datasets/import."""
    try:
        return model_browser.import_local_model(request.path, request.name)
    except model_browser.ModelError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


# ============================================================================
# JOB ENDPOINTS
# ============================================================================


def _job_label(job_id: str) -> str:
    """A run named the way its row names it: `#46 'alias' (id)`.

    For refusal messages that have to point at a *different* run than the one
    the user acted on — "delete X first" is only actionable if X is findable in
    the list. All three parts earn their place: the NUMBER is what the UI shows
    and what a person can repeat back; the NAME is shared by every run on a
    resume chain, so it locates the lineage but not the run; the ID is the
    unambiguous one and the only one that survives a rename.

    Degrades a part at a time — an unnumbered record (pre-backfill) or an
    unnamed one just drops that piece, and an id the registry no longer holds
    falls back to the bare id.
    """
    try:
        record = job_registry.get(job_id)
    except JobNotFoundError:
        return repr(job_id)
    name = (record.display_name or record.name or "").strip()
    label = f"{name!r} ({job_id})" if name else repr(job_id)
    return f"#{record.job_number} {label}" if record.job_number > 0 else label


def _is_finished_run(job_id: str) -> bool:
    """True when the run reached its target — i.e. holds finished training.

    Used to keep refusal messages from casually recommending its deletion. An
    unresolvable id reads as False: the message degrades to the plainer advice
    rather than inventing a reason to keep something that may not exist.
    """
    try:
        return job_registry.get(job_id).state == "done"
    except JobNotFoundError:
        return False


@app.post("/jobs/training", status_code=201)
async def create_training_job(req: Request):
    raw = await req.json()
    body = StartTrainingBody.from_legacy(raw)
    cfg = body.config
    # Soft warning (not a block): lerobot saves/logs on `step % freq == 0`, so a
    # frequency larger than the total step count means the action never fires —
    # no checkpoint gets saved / no metrics logged. Almost always a config
    # mistake, but we still let the run proceed.
    if cfg.steps:
        if cfg.save_freq > cfg.steps:
            logger.warning(
                "save_freq (%d) exceeds steps (%d) — no checkpoint will be saved.",
                cfg.save_freq,
                cfg.steps,
            )
        if cfg.log_freq > cfg.steps:
            logger.warning(
                "log_freq (%d) exceeds steps (%d) — no metrics will be logged.",
                cfg.log_freq,
                cfg.steps,
            )
    # Hard block (not a warning): when resuming, the total step count must be
    # strictly above the checkpoint's step — lerobot requires --steps be raised
    # above the resumed checkpoint, and steps == checkpoint would train nothing.
    #
    # This is the FAST half only, and cannot be the whole guard: it reads the
    # request's `resume_from_step`, which is None whenever the caller picked
    # "latest checkpoint" and left the step for the registry to resolve. Those
    # requests walk straight past this. JobRegistry.start re-asks the same
    # question at the bottom of its resume block, once the step is a number;
    # that one is the authority, and it raises ValueError -> the 400 below.
    if cfg.resume_from_step is not None and cfg.steps <= cfg.resume_from_step:
        logger.warning(
            "Rejecting resume: steps (%d) <= checkpoint step (%d).",
            cfg.steps,
            cfg.resume_from_step,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Total steps ({cfg.steps}) must be greater than the checkpoint's "
                f"step ({cfg.resume_from_step}) to continue training."
            ),
        )
    # Local preflight (belt-and-braces), the mirror of the cloud
    # DatasetNotOnHubError guard: a LOCAL run with no --dataset.root makes
    # lerobot auto-download the dataset from the Hub at start. When the Hub is
    # offline (HF_HUB_OFFLINE) that download can't happen — it hangs or dies
    # with a raw traceback — so reject up front with an actionable message
    # instead of starting a doomed job. Purely offline flag + local filesystem
    # check; no network call (no repo_exists/whoami). A RESUME run inherits its
    # dataset via config_path and doesn't re-download, but dataset_repo_id is a
    # required field so we can't distinguish resume by its absence; the guard is
    # a no-op on resume anyway because the resumed dataset is by definition
    # already local (it was trained on locally before), so is_dataset_available_
    # locally returns True and nothing is blocked.
    runner = body.target.runner if body.target is not None else "local"
    if runner == "local" and cfg.dataset_repo_id and hf_hub_offline():
        from .datasets import is_dataset_available_locally

        if not is_dataset_available_locally(cfg.dataset_repo_id):
            # 400 (matching this endpoint's other preflight rejections — the
            # resume-steps guard above and the ValueError->400 below), NOT 409:
            # startTrainingJob (jobsApi.ts) rewrites EVERY 409 into "Another
            # training is already running", which would mask this message. 400
            # lets FastAPI's `detail` reach the toast verbatim.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset '{cfg.dataset_repo_id}' isn't available locally and the "
                    "Hub is offline (HF_HUB_OFFLINE) — it can't be downloaded for a "
                    "local run. Start MakerMods Lab without --offline (or with Hub access) to "
                    "fetch it, or record/obtain the dataset locally first."
                ),
            )
    try:
        record = job_registry.start(body.config, body.target)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job already running: {exc}") from exc
    except DatasetNotOnHubError as exc:
        # Cloud run on a local-only dataset. 409: the caller must upload the
        # dataset first (the browser flow does this automatically before
        # submitting, so this fires for non-UI callers).
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobAlreadyContinuedError as exc:
        # Sticks only: the source already has a continuation, so a second one
        # would fork it. 409 for the same reason the mid-chain delete refusal
        # is a 409 — a conflict with existing state, not a malformed request —
        # and routed here the same way, with the ids turned into a message at
        # this layer rather than baked into the exception.
        #
        # The message has to TEACH the way out, because the way out is not
        # obvious from the refusal — but which way out is honest depends on what
        # is standing in the way, and getting that wrong is worse than saying
        # nothing. Two shapes:
        #
        #  - ONE unfinished continuation (every lineage the sticks rule can
        #    create): deleting it is cheap and correct, so say so.
        #  - a LEGACY FORK, or a continuation that ran to completion: "delete
        #    the continuation(s) first" is advice to throw away work. On the
        #    user's own registry this fired for a parent whose two children
        #    included a finished 30k run — the single run in that lineage
        #    nobody should delete. Recommend fine-tune instead: it starts a
        #    fresh schedule from the same weights, is not restricted to one per
        #    run, and needs nothing deleted.
        continued_by = ", ".join(_job_label(cid) for cid in exc.child_ids)
        source = _job_label(exc.job_id)
        finished = [_job_label(cid) for cid in exc.child_ids if _is_finished_run(cid)]
        if len(exc.child_ids) == 1 and not finished:
            remedy = f"A run can be continued once, so delete {continued_by} first, then resume {source}."
        else:
            cost = (
                f", including the finished {'runs' if len(finished) > 1 else 'run'} {', '.join(finished)}"
                if finished
                else ""
            )
            remedy = (
                f"A run can be continued once, so resuming {source} would mean first "
                f"deleting {continued_by}{cost}. Fine-tune from {source}'s checkpoint "
                "instead — that starts a fresh schedule from the same weights, needs "
                "nothing deleted, and is not limited to one per run."
            )
        raise HTTPException(
            status_code=409,
            detail=f"{source} was already continued by {continued_by}. {remedy}",
        ) from exc
    except ValueError as exc:
        # e.g. "flavor is required when runner is hf_cloud"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


class ImportModelRequest(BaseModel):
    source: str
    name: str | None = None


@app.post("/jobs/import", status_code=201)
def import_model(body: ImportModelRequest):
    """Register an external model (local dir or HF repo) as a pseudo-job.

    Importing an already-registered source is idempotent: the registry
    returns the EXISTING record (id and display alias preserved), and the
    response carries `already_imported: true` with a 200 (not 201) so the
    frontend can say "already imported" instead of pretending a new entry
    was created."""
    try:
        existing = job_registry.find_imported(body.source)
        record = job_registry.register_imported(body.source, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if existing is not None and existing.id == record.id:
        payload = record.model_dump(mode="json")
        payload["already_imported"] = True
        return JSONResponse(status_code=200, content=payload)
    return record


@app.get("/jobs")
def list_jobs(limit: int = 10):
    return {"jobs": job_registry.list(limit=limit)}


# A MakerMods Lab cloud-training run repo is named "<policy>_<namespace>_<dataset>_<ts>"
# where the trailing "_YYYY-MM-DD_HH-MM-SS" is stamped by _generate_job_id()
# (jobs.py). We match on that timestamp suffix rather than the policy prefix so
# the pattern stays policy-agnostic as new policy types are added. Used to pull
# MakerMods Lab's OWN empty/untagged run repos into the /jobs/hub listing without also
# surfacing a user's unrelated personal models.
_RUN_REPO_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

# Hub job stages still doing work. Mirrors HUB_ACTIVE_STAGES in the frontend
# (jobsApi.ts); a dismissed id in one of these stages is NOT hidden from the
# listing, so a live run can never be dismissed out of sight.
_HUB_ACTIVE_STAGES = {"RUNNING", "QUEUED", "SCHEDULING"}


def _hub_job_stage(ji) -> str:
    """Uppercased status stage of a huggingface_hub JobInfo ('' when absent)."""
    return (ji.status.stage or "").upper() if ji.status else ""


# Errors a per-author Hub model listing may raise that must degrade to "empty for
# this author" instead of 500ing /jobs/hub. httpx.HTTPError is the base of
# ConnectError / TimeoutException / TransportError — what a GFW-killed TLS
# connection raises — plus HfHubHTTPError for HTTP-status failures and OSError for
# lower-level socket failures.
_HUB_MODEL_LISTING_ERRORS = (HfHubHTTPError, httpx.HTTPError, OSError)

# Bounded per-author fan-out for the model listing. Small cap (a handful of
# authors). The timeout is the OVERALL fan-out budget — the single deadline the
# whole batch must finish within (authors run concurrently, so overall ≈
# per-author). It is the ONLY timeout in the stack: the shared HfApi httpx
# client is built with timeout=None, so a blackholed connection would otherwise
# stall /jobs/hub until the OS TCP layer gives up. 5s lets a merely-slow Hub
# succeed while a hung author is abandoned fast.
_HUB_MODEL_FANOUT_MAX_WORKERS = 8
_HUB_MODEL_FANOUT_TIMEOUT_S = 5.0

# Short-TTL cache for the /jobs/hub response. Startup + navigation re-hit this in
# quick succession; caching avoids re-fanning-out to the (slow/flaky) Hub each
# time. TTL uses time.monotonic() (app runtime, immune to wall-clock jumps).
_HUB_JOBS_CACHE_TTL_S = 45.0
_hub_jobs_cache_lock = threading.Lock()
_hub_jobs_cache: dict[str, Any] | None = None  # {"at": monotonic, "value": {...}}


def invalidate_hub_jobs_cache() -> None:
    """Drop the cached /jobs/hub response so the next call re-fetches. Called
    after a Hub model delete so the removal reflects immediately."""
    global _hub_jobs_cache
    with _hub_jobs_cache_lock:
        _hub_jobs_cache = None


def _list_author_models(api, author: str) -> list:
    """All MakerMods Lab-relevant model repos for one author, as a materialized list.

    Collapses what used to be TWO calls per author (a `filter="lerobot"` call plus
    an unfiltered fallback) into ONE unfiltered `list_models(author=...)` call,
    filtering client-side. A repo qualifies if EITHER:

      * it carries the `lerobot` library tag (what push_to_hub stamps), OR
      * its name matches the MakerMods Lab run-repo pattern (the "_<timestamp>" suffix) —
        this pulls in the empty repos a crashed cloud run pre-creates but never
        tags, which the untracked-cleanup path exists to delete.

    This is the same union the old two-pass code produced, at half the calls.
    The generator is materialized here (inside the fan-out worker) so the network
    I/O happens under the per-author timeout budget.
    """
    out = []
    for m in api.list_models(author=author, limit=200, expand=["lastModified", "private", "tags"]):
        tags = getattr(m, "tags", None) or []
        name = m.id.split("/", 1)[-1]
        if "lerobot" in tags or _RUN_REPO_RE.search(name):
            out.append(m)
    return out


def _fan_out_model_authors(authors: list[str], call) -> list:
    """Run `call(author)` for each author concurrently, gathering the results —
    the /jobs/hub twin of datasets._fan_out_hub_authors (kept separate for its
    own error tuple + log wording). Per-author failures are logged and
    swallowed; the whole batch runs under ONE overall deadline
    (_HUB_MODEL_FANOUT_TIMEOUT_S) so a hung connection (the underlying httpx
    client has timeout=None) can't stall the endpoint. Returns the successful
    authors' results in author order."""
    if not authors:
        return []

    results: list = [None] * len(authors)
    max_workers = min(_HUB_MODEL_FANOUT_MAX_WORKERS, len(authors))
    # Deliberately NOT `with ThreadPoolExecutor(...)`: the context-manager exit
    # JOINS the workers, so a hung author would stall us at the `with` exit even
    # after the as_completed deadline fired. shutdown(wait=False,
    # cancel_futures=True) cancels queued-not-started work; an already-running
    # hung thread is left to die with its socket — a bounded leak (the OS TCP
    # timeout eventually reaps it), which beats blocking the endpoint on it.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_to_idx = {pool.submit(call, author): i for i, author in enumerate(authors)}
    try:
        for future in concurrent.futures.as_completed(future_to_idx, timeout=_HUB_MODEL_FANOUT_TIMEOUT_S):
            idx = future_to_idx[future]
            author = authors[idx]
            try:
                results[idx] = future.result()
            except _HUB_MODEL_LISTING_ERRORS as exc:
                logger.warning("list_models(%s) failed: %s", author, exc)
            except Exception as exc:  # noqa: BLE001 - listings are best-effort; never 500
                logger.warning("list_models(%s) failed unexpectedly: %s", author, exc)
    except concurrent.futures.TimeoutError:
        unfinished = [authors[i] for f, i in future_to_idx.items() if not f.done()]
        logger.warning(
            "Hub model fan-out exceeded %ss; giving up on authors: %s",
            _HUB_MODEL_FANOUT_TIMEOUT_S,
            ", ".join(unfinished) or "(none)",
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return [r for r in results if r is not None]


@app.get("/jobs/hub")
def list_hub_jobs():
    """List the user's HF Cloud compute Jobs and their uploaded LeRobot model
    repos on huggingface.co.

    Returns 200 with empty lists when no token is configured so the frontend
    can render an unauthenticated empty state without surfacing an error.

    Declared before `/jobs/{job_id}` so FastAPI's first-match routing doesn't
    treat "hub" as a job id.
    """
    global _hub_jobs_cache

    now = time.monotonic()
    with _hub_jobs_cache_lock:
        if _hub_jobs_cache is not None and (now - _hub_jobs_cache["at"]) < _HUB_JOBS_CACHE_TTL_S:
            return _hub_jobs_cache["value"]

    info = cached_whoami()
    if info is None:
        # Not cached: unauthenticated is cheap to recompute and self-heals the
        # moment a token appears.
        return {"authenticated": False, "jobs": [], "models": []}
    api = shared_hf_api()

    authors: list[str] = []
    if info.get("name"):
        authors.append(info["name"])
    for o in info.get("orgs", []) or []:
        if isinstance(o, dict) and o.get("name"):
            authors.append(o["name"])

    jobs_permission = True
    jobs_listed = True
    try:
        # list_jobs() returns a lazy pagination generator — materialize it here
        # so any HTTP error (e.g. 403 when the token lacks the job.read scope)
        # is raised and caught inside this try, not later while building the
        # response, which would escape as an unhandled 500.
        jobs = list(api.list_jobs())
    except Exception as exc:
        logger.warning("list_jobs failed: %s", exc)
        jobs = []
        jobs_listed = False
        # A 401/403 means the token is valid but lacks the job.read scope —
        # surface that to the frontend so it can show a hint instead of a
        # silently-empty list. Other failures are treated as transient.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            jobs_permission = False

    # Drop jobs the user dismissed from the UI — but only in a terminal stage:
    # an id whose job is still active stays visible, so a live run can't be
    # dismissed out of sight. Ids that have fallen out of the Hub listing are
    # pruned so the file doesn't grow forever; skipped when list_jobs() failed,
    # otherwise a transient outage would forget every dismissal.
    dismissed = get_dismissed_hub_jobs()
    if jobs_listed:
        prune_dismissed_hub_jobs({ji.id for ji in jobs})
    if dismissed:
        jobs = [ji for ji in jobs if ji.id not in dismissed or _hub_job_stage(ji) in _HUB_ACTIVE_STAGES]

    seen_models: set[str] = set()
    models: list[dict] = []

    def _add(m) -> None:
        if m.id in seen_models:
            return
        seen_models.add(m.id)
        models.append(
            {
                "repo_id": m.id,
                "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                "private": bool(getattr(m, "private", False)),
            }
        )

    # Fan out the per-author model listing concurrently (bounded pool, one
    # OVERALL deadline — see _fan_out_model_authors). Each author is ONE
    # unfiltered list_models call, filtered client-side by _list_author_models
    # (union of the `lerobot` tag and the MakerMods Lab run-repo naming). Each author's
    # call is guarded so a GFW-killed connection / hung socket / slow author
    # degrades to "no models from that author" instead of sinking the batch or
    # stalling the endpoint. Results are deduped by _add() in author order,
    # preserving the original merge semantics.
    for author_models in _fan_out_model_authors(authors, lambda author: _list_author_models(api, author)):
        for m in author_models or []:
            _add(m)
    models.sort(key=lambda m: m["last_modified"] or "", reverse=True)

    response = {
        "authenticated": True,
        "jobs_permission": jobs_permission,
        "jobs": [
            {
                "id": ji.id,
                "created_at": ji.created_at.isoformat() if ji.created_at else None,
                "docker_image": ji.docker_image,
                "space_id": ji.space_id,
                "flavor": ji.flavor,
                "status": ({"stage": ji.status.stage, "message": ji.status.message} if ji.status else None),
                "owner": ji.owner.name if ji.owner else None,
                "url": ji.url,
            }
            for ji in jobs
        ],
        "models": models,
    }

    with _hub_jobs_cache_lock:
        _hub_jobs_cache = {"at": time.monotonic(), "value": response}
    return response


@app.delete("/jobs/hub/models/{repo_id:path}")
def delete_hub_model(repo_id: str):
    """Permanently delete a model repo from the Hugging Face Hub.

    Scoped to model repos under the authenticated user's own namespace — used
    to clean up orphaned repos (e.g. an empty repo left behind by a crashed
    cloud run). This destroys weights on the Hub; it is not a local-record
    deletion.

    Semantics:
    - A missing repo (404 from the Hub) is treated as already-gone success,
      mirroring the idempotent robot-delete convention.
    - Repos NOT under the caller's own username are refused up front with a
      clear message (the Hub would 403 anyway; fail fast).
    - Auth/permission failures (401/403) surface the friendly "token needs
      write access" message.

    The `/jobs/hub` listing is cached backend-side for a short TTL; this delete
    invalidates that cache (see invalidate_hub_jobs_cache) so the removed repo
    disappears immediately when the frontend re-fetches.
    """
    info = cached_whoami()
    username = info.get("name") if info else None
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Add a Hugging Face token with write access first.",
        )

    # Only allow deleting repos the caller owns (namespace == their username).
    # An org-owned repo (username/... mismatch) is refused rather than 403ing.
    namespace = repo_id.split("/", 1)[0] if "/" in repo_id else ""
    if namespace != username:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Refusing to delete {repo_id!r}: it is not under your namespace "
                f"({username!r}). You can only delete your own model repos."
            ),
        )

    api = shared_hf_api()
    try:
        # missing_ok=True: a repo that's already gone (404) is a no-op success,
        # so re-issuing the delete is idempotent.
        api.delete_repo(repo_id, repo_type="model", missing_ok=True)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your Hugging Face token can't delete this repo. It needs "
                    "write access to your namespace — re-log in with a write token."
                ),
            ) from exc
        logger.warning("delete_repo(%s) failed: %s", repo_id, exc)
        raise HTTPException(status_code=502, detail=f"Hub delete failed: {exc}") from exc

    # The listing changed — drop the cached /jobs/hub response so the removed
    # repo doesn't linger until the TTL expires.
    invalidate_hub_jobs_cache()
    return {"status": "success", "repo_id": repo_id}


@app.post("/jobs/hub/jobs/{job_id}/dismiss")
def dismiss_hub_job(job_id: str):
    """Hide a Hub job from the /jobs/hub listing.

    The HF Jobs API has no delete — a finished job stays in list_jobs()
    indefinitely — so "removing" a dead untracked job from the UI is a local,
    persisted hide (utils/config.DISMISSED_HUB_JOBS_FILE), not a Hub mutation.
    The listing keeps showing a dismissed id while its stage is still active
    (RUNNING/QUEUED/SCHEDULING); it disappears once the job reaches a terminal
    stage. Ids that later drop out of the Hub listing are pruned automatically.
    """
    if not add_dismissed_hub_job(job_id):
        raise HTTPException(status_code=400, detail="Job id can't be empty.")
    return {"status": "success", "job_id": job_id.strip()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"logs": logs}


@app.get("/jobs/{job_id}/log-file")
def get_job_log_file(job_id: str):
    """Return the entire on-disk log file for a job. Drains the live queue too
    so the next /logs poll returns only lines that arrived after this call."""
    try:
        logs = job_registry.read_persisted_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    # Best-effort drain so the frontend doesn't double-display.
    with contextlib.suppress(JobNotFoundError):
        job_registry.drain_logs(job_id)
    return {"logs": logs}


@app.get("/jobs/{job_id}/metrics-history")
def get_job_metrics_history(job_id: str):
    """Return the per-step loss/lr/grad-norm series reconstructed from the
    job's log.jsonl. Used to seed the monitoring charts so curves persist
    across page reloads, navigation, and MakerMods Lab restarts."""
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"points": points}


@app.get("/jobs/{job_id}/checkpoints")
def get_job_checkpoints(job_id: str):
    """List the checkpoints saved for this job, ascending by step."""
    try:
        return {"checkpoints": job_registry.list_checkpoints(job_id)}
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/policy-config")
def get_checkpoint_policy_config(job_id: str, step: int):
    """Return the UX-relevant slice of a checkpoint's pretrained_model config:
    policy_type, image_features (per-camera height/width), requires_task, and
    the flat state_dim/action_dim (6 = single arm, 12 = bimanual) the inference
    modal uses to flag a single-arm/bimanual mismatch."""
    try:
        return job_registry.get_policy_config_summary(job_id, step)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/download")
def download_checkpoint(job_id: str, step: int):
    """Stream a zip of a local checkpoint's `pretrained_model/` directory.

    This bundles the portable, importable model (config.json + weights +
    pre/post-processors) — NOT the large `training_state/` optimizer dir.
    Hub-hosted models are downloadable from their HF page, so only local runs
    are supported here.
    """
    try:
        record = job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc

    if record.runner != "local":
        raise HTTPException(
            status_code=400,
            detail="Only local checkpoints can be downloaded; Hub models are available on their HF page.",
        )

    # The pretrained_model dir comes from _list_local_checkpoints (which resolves
    # it under record.output_dir/checkpoints/<step>), not from user input, so
    # path traversal isn't a concern. Match on the int step, never a raw path.
    checkpoint = next((c for c in _list_local_checkpoints(record.output_dir) if c.step == step), None)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} has no checkpoint at step {step}")

    pretrained_dir = Path(checkpoint.ref)

    buffer = io.BytesIO()
    # safetensors weights are already incompressible, so DEFLATE would burn CPU
    # for ~no gain; store uncompressed.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(pretrained_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(pretrained_dir).as_posix())
    buffer.seek(0)

    # Build a filesystem-safe filename from the job's display alias (falling
    # back to its name) + step, then to the job id if sanitising leaves
    # nothing usable.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", record.display_name or record.name).strip("_")
    if not safe_name:
        safe_name = job_id
    filename = f"{safe_name}_step_{step}.zip"

    logger.info("Downloading checkpoint for job %s at step %d", job_id, step)

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class RenameJobBody(BaseModel):
    new_name: str


@app.post("/jobs/{job_id}/rename")
def rename_job(job_id: str, body: RenameJobBody):
    """Set a job's display alias (shown in place of the auto-generated name).

    Metadata-only: never moves the output directory or rewrites the run id /
    hub repo id — those are the job's immutable identity (resume lineage,
    imported-model dedup, and remote HF/W&B names key off them). Validation
    (trim, reject empty, is_valid-style character guard) lives in
    JobRegistry.rename; unlike calibration/robot renames, aliases are
    display-only and need not be unique.
    """
    try:
        return job_registry.rename(job_id, body.new_name)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_registry.stop(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is not running") from exc


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        record = job_registry.get(job_id)
        job_registry.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first") from exc
    except JobHasChildrenError as exc:
        # Mid-chain delete: name the runs that continue from this one so the
        # user can work inwards from the tip instead of guessing.
        continued_by = ", ".join(repr(cid) for cid in exc.child_ids)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job {job_id!r} was continued by {continued_by}, which would be left "
                "pointing at a deleted run. Delete the continuation(s) first."
            ),
        ) from exc
    # Deleting a tracked cloud run removes the local record, but its Hub job
    # would resurface in /jobs/hub as an untracked card on the next poll (the
    # HF Jobs API has no delete). Mark it dismissed so the removal sticks.
    if record.hf_job_id:
        add_dismissed_hub_job(record.hf_job_id)


def _format_accelerator(accelerator) -> str | None:
    """Render a JobHardwareInfo's accelerator as a label — 2× Nvidia A100 —
    or None on a CPU flavor.

    huggingface_hub returns this field as a JobAccelerator OBJECT (type, model,
    quantity, vram, manufacturer), not a string. Forwarding it raw put a nested
    dict on the wire under a field the frontend types as `string`, which the
    hardware dropdown then interpolated into its label as "[object Object]".
    Flattened here rather than in the frontend so `vram` — the one number that
    says whether a policy will fit — survives as its own field instead of being
    buried in a shape nothing declared.
    """
    if accelerator is None:
        return None
    quantity = str(getattr(accelerator, "quantity", "") or "").strip()
    model = str(getattr(accelerator, "model", "") or "").strip()
    manufacturer = str(getattr(accelerator, "manufacturer", "") or "").strip()
    name = " ".join(part for part in (manufacturer, model) if part)
    if not name:
        # A future hub version could rename the fields out from under us; a
        # plain str() still beats a dict the UI would render as [object Object].
        return str(accelerator)
    return f"{quantity}× {name}" if quantity and quantity != "1" else name


@app.get("/jobs/runners/hardware")
def get_runners_hardware():
    """Return HF Jobs flavor catalog + auth state for the TargetCard.

    Both the flavors list and the whoami result are cached in-process to
    keep this endpoint cheap (it can be re-fetched whenever auth state
    changes). The whoami cache is invalidated on login.
    """
    # Offline mode disables every Hub write, so the cloud-training flow can't
    # upload a local-only dataset. Surface it here (same fetch TargetCard uses)
    # so the UI can keep Start disabled and explain why for those datasets.
    offline = hf_hub_offline()
    info = cached_whoami()
    if info is None or not info.get("name"):
        return {"authenticated": False, "username": None, "flavors": [], "offline": offline}
    username: str = info["name"]
    api = shared_hf_api()

    now = time.time()
    if _flavors_cache["data"] is None or now - _flavors_cache["fetched_at"] > _FLAVOR_CACHE_TTL_SECONDS:
        try:
            hw_list = api.list_jobs_hardware()
        except Exception as exc:
            logger.warning("list_jobs_hardware failed: %s", exc)
            return {"authenticated": True, "username": username, "flavors": [], "offline": offline}
        _flavors_cache["data"] = [
            {
                "name": h.name,
                "pretty_name": h.pretty_name,
                "cpu": h.cpu,
                "ram": h.ram,
                "accelerator": _format_accelerator(h.accelerator),
                "vram": getattr(h.accelerator, "vram", None),
                "unit_cost_usd": h.unit_cost_usd,
                "unit_label": h.unit_label,
            }
            for h in hw_list
        ]
        _flavors_cache["fetched_at"] = now

    return {
        "authenticated": True,
        "username": username,
        "flavors": _flavors_cache["data"],
        "offline": offline,
    }


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================


@app.get("/system/training-extra")
def get_training_extra():
    """Return whether the LeRobot training extra (accelerate) is importable."""
    return handle_get_training_extra()


@app.post("/system/training-extra/install")
def install_training_extra():
    """Spawn `pip install accelerate` as a background subprocess. No-op if already running."""
    return handle_install_training_extra()


@app.get("/system/training-extra/install-status")
def install_training_extra_status():
    """Return current install state plus any pending log lines (drained on read)."""
    return handle_install_training_extra_status()


@app.get("/system/wandb-extra")
def get_wandb_extra():
    """Return whether the `wandb` package is importable in this MakerMods Lab process."""
    return handle_get_wandb_extra()


@app.post("/system/wandb-extra/install")
def install_wandb_extra():
    """Spawn `pip install wandb` as a background subprocess. No-op if already running."""
    return handle_install_wandb_extra()


@app.get("/system/wandb-extra/install-status")
def install_wandb_extra_status():
    """Return current wandb install state plus any pending log lines (drained on read)."""
    return handle_install_wandb_extra_status()


@app.get("/system/policy-extra/{policy_type}")
def get_policy_extra(policy_type: str):
    """Whether the optional LeRobot extra a policy needs (e.g. transformers for
    smolvla/pi0, diffusers for diffusion) is importable. Core policies report available."""
    return handle_get_policy_extra(policy_type)


@app.post("/system/policy-extra/{policy_type}/install")
def install_policy_extra(policy_type: str):
    """Spawn `pip install lerobot[<extra>]` for the policy's extra in the background."""
    return handle_install_policy_extra(policy_type)


@app.get("/system/policy-extra/{policy_type}/install-status")
def install_policy_extra_status(policy_type: str):
    """Return the policy extra's install state plus any pending log lines (drained on read)."""
    return handle_install_policy_extra_status(policy_type)


@app.get("/system/update-check")
def update_check():
    """Report whether a newer MakerMods Lab commit exists on GitHub (cached, silent on failure)."""
    return handle_update_check()


@app.post("/system/update")
def run_update():
    """Run the pip upgrade in-process; the user must restart MakerMods Lab afterwards."""
    return handle_run_update()


# Replay is rendered by the embedded lerobot/visualize_dataset Space; no backend routes needed.


# ============================================================================
# Calibration endpoints
@app.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start calibration process"""
    return calibration_manager.start_calibration(request)


@app.post("/stop-calibration")
def stop_calibration():
    """Stop calibration process"""
    return calibration_manager.stop_calibration_process()


@app.get("/calibration-status")
def calibration_status():
    """Get current calibration status"""
    from dataclasses import asdict

    status = calibration_manager.get_status()
    return asdict(status)


@app.post("/complete-calibration-step")
def complete_calibration_step():
    """Complete the current calibration step"""
    return calibration_manager.complete_step()


# --- Auto-calibration (drives the arm under torque; runs the vendored script) ---


@app.post("/start-auto-calibration")
def start_auto_calibration(request: AutoCalibrationRequest):
    """Start auto-calibration as a subprocess. The arm moves on its own."""
    return auto_calibration_manager.start(request)


@app.post("/stop-auto-calibration")
def stop_auto_calibration():
    """Stop a running auto-calibration."""
    return auto_calibration_manager.stop()


@app.get("/auto-calibration-status")
def auto_calibration_status():
    """Current auto-calibration state + streamed log lines."""
    return auto_calibration_manager.get_status()


@app.post("/start-auto-calibration-batch")
def start_auto_calibration_batch(request: AutoCalibrationBatchRequest):
    """Auto-calibrate a user-selected subset of arms CONCURRENTLY. Each arm runs
    its own subprocess on its own serial port with an independent outcome
    (partial success). Validated up front (1-4 arms, distinct ports, distinct
    same-side names, name-taken pre-check) before any hardware is touched."""
    return auto_calibration_batch_manager.start(request)


@app.post("/stop-auto-calibration-batch")
def stop_auto_calibration_batch():
    """Stop ALL running arms of a batch auto-calibration, releasing each arm's
    torque independently."""
    return auto_calibration_batch_manager.stop()


@app.get("/auto-calibration-batch-status")
def auto_calibration_batch_status():
    """Per-arm status + logs and overall counts for a batch auto-calibration."""
    return auto_calibration_batch_manager.get_status()


@app.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str):
    """Get all calibration config files for a specific device type"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # Get all JSON files in the config directory
        configs = []
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    file_path = os.path.join(config_path, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)

                    configs.append(
                        {
                            "name": config_name,
                            "filename": file,
                            "size": file_size,
                            "modified": modified_time,
                        }
                    )

        return {"success": True, "configs": configs, "device_type": device_type}

    except Exception as e:
        logger.error(f"Error getting calibration configs: {e}")
        return {"success": False, "message": str(e)}


@app.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str):
    """Delete a calibration config file"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # config_name is interpolated into a filename, so reject path-traversal
        # characters (/, \, ..) before touching the filesystem. Defense-in-depth:
        # FastAPI path params already block a literal "/", but not "\" or "..".
        # Reuses the same guard already applied to robot-record deletes.
        if not is_valid_robot_name(config_name):
            return {"success": False, "message": "Invalid configuration name"}

        # Construct the file path
        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        # Delete the file. This dir IS the location lerobot reads calibrations
        # from (setup_calibration_files' source == target), so removing the file
        # removes the only copy — nothing stale can silently keep working.
        os.remove(file_path)
        logger.info(f"Deleted calibration config: {file_path}")

        # Unassign every robot record that still pointed at this config, so
        # those arms return to the "needs calibration" state instead of
        # dangling on a missing file. The response lists them so the UI can
        # refresh the affected robots.
        unassigned = clear_config_references(device_type, config_name)
        if unassigned:
            robots = ", ".join(u["robot"] for u in unassigned)
            message = (
                f"Configuration '{config_name}' deleted. Robot(s) {robots} now need calibration before use."
            )
        else:
            message = f"Configuration '{config_name}' deleted successfully"

        return {
            "success": True,
            "message": message,
            "unassigned": unassigned,
        }

    except Exception as e:
        logger.error(f"Error deleting calibration config: {e}")
        return {"success": False, "message": str(e)}


@app.get("/calibration-configs/{device_type}/{config_name}/download")
def download_calibration_config(device_type: str, config_name: str):
    """
    Download one arm's calibration as a raw lerobot calibration JSON file.

    The file IS lerobot's own calibration file (no MakerMods Lab wrapper), so it's
    drop-in: shareable, hand-copyable, and re-importable anywhere. The arm's
    side/name are supplied by the caller on re-import, not stored in the file.
    """
    if device_type == "robot":
        config_path = FOLLOWER_CONFIG_PATH
    elif device_type == "teleop":
        config_path = LEADER_CONFIG_PATH
    else:
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid device type"})

    # config_name is interpolated into a filename, so reject path-traversal
    # characters before touching the filesystem (same guard as delete).
    if not is_valid_robot_name(config_name):
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Invalid configuration name"}
        )

    # Robot records store config names WITH the .json extension while this
    # resource is otherwise stem-based; accept either form so callers that pass
    # `robot.leader_config` ("so101.json") don't resolve to "so101.json.json".
    if config_name.endswith(".json"):
        config_name = config_name[: -len(".json")]

    file_path = os.path.join(config_path, f"{config_name}.json")
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404, content={"success": False, "message": "Configuration file not found"}
        )

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except OSError as e:
        logger.error(f"Error reading calibration config {file_path}: {e}")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{config_name}.json"'},
    )


@app.post("/calibration-configs/{device_type}/upload")
def upload_calibration_config(device_type: str, body: dict):
    """
    Import a calibration into a side's config dir. Body: {"name": "...",
    "data": {<raw lerobot calibration>}}. The data is shape-validated; an
    existing name is never overwritten (409 → caller renames).
    """
    name = (body or {}).get("name", "")
    data = (body or {}).get("data")
    if not isinstance(name, str):
        return JSONResponse(status_code=400, content={"success": False, "message": "name must be a string"})

    ok, reason, saved = save_imported_calibration(device_type, name, data)
    if ok:
        return {"success": True, "name": saved}

    if reason == "invalid_device":
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid device type"})
    if reason == "invalid_name":
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Invalid configuration name"}
        )
    if reason == "name_taken":
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": f"A config named '{saved}' already exists. Choose a different name.",
            },
        )
    if reason.startswith("invalid_data:"):
        return JSONResponse(status_code=400, content={"success": False, "message": reason.split(":", 1)[1]})
    return JSONResponse(status_code=500, content={"success": False, "message": "Import failed"})


@app.post("/calibration-configs/{device_type}/{config_name}/rename")
def rename_calibration_config_endpoint(device_type: str, config_name: str, body: dict):
    """
    Rename a calibration config file. Body: {"new_name": "..."}. Never
    overwrites; robot records referencing the old name are repointed.
    """
    new_name = (body or {}).get("new_name", "")
    if not isinstance(new_name, str):
        return JSONResponse(
            status_code=400, content={"success": False, "message": "new_name must be a string"}
        )

    ok, reason = rename_calibration_config(device_type, config_name, new_name)
    if ok:
        return {"success": True, "name": new_name.strip().removesuffix(".json")}

    status_code, message = {
        "invalid_device": (400, "Invalid device type"),
        "invalid_name": (400, "Invalid configuration name"),
        "not_found": (404, "Configuration file not found"),
        "name_taken": (409, "A config with that name already exists. Choose a different name."),
    }.get(reason, (500, "Rename failed"))
    return JSONResponse(status_code=status_code, content={"success": False, "message": message})


class OpenCalibrationFolderRequest(BaseModel):
    device_type: str  # "teleop" (leader) or "robot" (follower)


@app.post("/open-calibration-folder")
def open_calibration_folder(request: OpenCalibrationFolderRequest):
    """Open a side's calibration folder in the OS file browser (Finder/Explorer/
    xdg-open). LOCAL, non-network action — spawns a GUI on the host machine only.
    The dir is created if missing so a fresh install opens an empty folder rather
    than failing. An unknown device_type is rejected with 400.
    """
    path = calibration_dir_for_device(request.device_type)
    if path is None:
        return JSONResponse(
            status_code=400,
            content={"opened": False, "message": "device_type must be 'teleop' or 'robot'"},
        )
    try:
        open_folder_in_file_browser(path)
    except Exception as e:
        logger.error(f"Failed to open calibration folder {path}: {e}")
        return JSONResponse(
            status_code=500,
            content={"opened": False, "message": f"Could not open folder: {e}", "path": path},
        )
    return {"opened": True, "path": path}


# ============================================================================
# PORT DETECTION ENDPOINTS
# ============================================================================


@app.get("/available-ports")
def get_available_ports():
    """Get all available serial ports"""
    try:
        ports = find_available_ports()
        return {"status": "success", "ports": ports}
    except Exception as e:
        logger.error(f"Error getting available ports: {e}")
        return {"status": "error", "message": str(e)}


class WiggleRequest(BaseModel):
    port: str


@app.post("/wiggle")
async def wiggle(request: WiggleRequest):
    """Wiggle the gripper on a port so the user can see which arm it is."""
    return await wiggle_gripper(request.port)


class IdentifyArmRequest(BaseModel):
    # Candidate ports to watch; empty/omitted = all detected arm ports.
    ports: list[str] | None = None


@app.post("/identify-arm")
async def identify_arm(request: IdentifyArmRequest):
    """The inverse of /wiggle: the user swings an arm's base (shoulder pan) by
    hand and we report which port saw the motion. Read-only — no motor writes."""
    return await identify_arm_by_motion(request.ports)


@app.get("/supply-voltage")
async def supply_voltage(port: str = ""):
    """One-shot, read-only supply-voltage reading (Present_Voltage) from the arm
    on `port`. Connects, reads, and releases the port immediately — never holds
    it — so calibration/teleoperation can grab the port right after."""
    return await read_supply_voltage(port)


# Runs in a fresh Python — see _avfoundation_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def _avfoundation_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. The real names let the
    frontend match each index to the browser's ``MediaDeviceInfo.label`` for the
    live preview. Falls back to generic names if pygrabber is unavailable.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
    except Exception as e:  # ImportError, or a COM/DirectShow failure
        logger.warning("pygrabber unavailable; using generic camera names: %s", e)
        import cv2

        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


# struct v4l2_capability from <linux/videodev2.h>, and the QUERYCAP ioctl
# request code (_IOR('V', 0, struct v4l2_capability), 104 bytes).
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001


class _V4l2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux capture devices via VIDIOC_QUERYCAP, without opening them.

    The previous cv2.VideoCapture probe claimed each device's format on open,
    which fails EBUSY while browser previews hold the cameras — V4L2 is
    single-streamer, so cameras vanished from the list whenever previews were
    live. QUERYCAP is a metadata query: it answers on busy devices (the Linux
    equivalent of the macOS AVFoundation name enumeration, which never opens
    devices either) and its device_caps distinguishes real capture nodes from
    the UVC metadata nodes (/dev/video1/3/5...) that cv2 could only rule out
    by failing to open them, one warning line each.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    cameras: list[dict[str, Any]] = []
    for i in range(10):
        try:
            fd = os.open(f"/dev/video{i}", os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            cap = _V4l2Capability()
            if libc.ioctl(fd, ctypes.c_ulong(_VIDIOC_QUERYCAP), ctypes.byref(cap)) != 0:
                continue
            # device_caps is per-node (capabilities is the union across the
            # whole device, so metadata nodes advertise VIDEO_CAPTURE there).
            caps = cap.device_caps or cap.capabilities
            if not caps & _V4L2_CAP_VIDEO_CAPTURE:
                continue
            name = cap.card.decode(errors="replace").strip() or _v4l2_camera_name(i) or f"Camera {i}"
            cameras.append({"index": i, "name": name, "available": True})
        finally:
            os.close(fd)
    return cameras


@app.get("/available-cameras")
def get_available_cameras():
    """List cameras with the same index ordering cv2 will use to record.

    Each platform enumerates in the order its cv2 backend indexes devices, and
    pairs each index with the device's real name so the frontend can match it to
    the browser's ``MediaDeviceInfo.label`` for the live preview:
      - macOS: AVFoundation ``localizedName`` (via a PyObjC subprocess);
      - Windows: DirectShow FriendlyName (via pygrabber; recording pinned DSHOW);
      - Linux: the VIDIOC_QUERYCAP card name (sysfs fallback); QUERYCAP works
        on busy devices, so live previews never hide cameras from the list.
    Without real names the frontend can't match a camera and shows "No browser
    match" with an empty device_id (issues #12, #16).
    """
    try:
        import platform

        system = platform.system()

        if system == "Darwin":
            cameras = _avfoundation_cameras_in_cv2_order()
            for cam in cameras:
                cam["available"] = True
            return {"status": "success", "cameras": cameras}
        if system == "Windows":
            return {"status": "success", "cameras": _windows_cameras()}
        if system == "Linux":
            return {"status": "success", "cameras": _linux_cameras()}

        import cv2

        return {"status": "success", "cameras": _generic_cv2_cameras(cv2.CAP_ANY)}
    except ImportError:
        logger.warning("OpenCV not available for camera detection")
        return {"status": "success", "cameras": []}
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {"status": "error", "message": str(e), "cameras": []}


@app.get("/camera-preview/{index}")
def camera_preview_stream(index: int, unique_id: str | None = None):
    """MJPEG preview stream of a camera attached to the *server* machine.

    The browser's getUserMedia only sees the *viewing* machine's cameras, and
    it identifies them by ``deviceId`` — which the frontend can only match to a
    cv2 index by localizedName. Two cameras of the same model share that name,
    so the match is a coin flip and the preview can show the wrong device (and
    swap between refreshes). Streaming from the backend by cv2 index removes
    the browser from the loop: the tile shows exactly what the recorder will
    open. It is also the only preview that works at all on a headless host.

    ``unique_id`` (AVFoundation uniqueID, from /available-cameras) re-anchors
    the index to the physical device before opening: cv2 resolves indices
    against this process's startup device snapshot, which diverges from the
    fresh-subprocess enumeration after a replug — without the re-anchor the
    stream can silently show a different camera (see makermodslab/camera_identity.py).

    Returns 409 while recording or inference is active (they own the cv2
    devices) and 503 when the camera can't be opened. Teleoperation drives the
    serial bus and opens no cv2 cameras, so a preview during teleop does not
    contend — it is allowed.
    """
    if record_state.recording_active:
        raise HTTPException(
            status_code=409,
            detail="Recording is active — the cameras are in use. Stop recording to preview them.",
        )
    if rollout_state.inference_active:
        raise HTTPException(
            status_code=409,
            detail="Inference is active — the cameras are in use. Stop the run to preview them.",
        )
    resolved = resolve_cv2_index(unique_id, index)
    if resolved is None:
        raise HTTPException(
            status_code=503,
            detail="Camera not visible to the server — it was plugged in after MakerMods Lab "
            "started. Restart MakerMods Lab to use it.",
        )
    try:
        stream = camera_preview_manager.open_stream(resolved)
    except CameraOpenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


RobotSideLiteral = Literal["leader", "follower"]


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: RobotSideLiteral):
    """Get the saved port for a robot type"""
    saved_port = get_saved_robot_port(robot_type)
    default_port = get_default_robot_port(robot_type)
    return {"status": "success", "saved_port": saved_port, "default_port": default_port}


# ============================================================================
# Robot config records (named robots)


def _record_with_clean(record: dict) -> dict:
    """Attach readiness flags to a record for API responses.

    `is_clean` folds every arm of the mode (gates teleop/record, which drive
    leaders AND followers); `follower_ready` scopes to the follower side so
    follower-only activities (inference, replay) aren't blocked by a leader arm
    they never touch."""
    return {
        **record,
        "is_clean": is_robot_record_clean(record),
        "follower_ready": is_robot_record_clean(record, arms="follower"),
    }


@app.get("/robots")
def get_robots():
    """List all saved robot records."""
    try:
        records = [_record_with_clean(r) for r in list_robot_records()]
        return {"status": "success", "robots": records}
    except Exception as e:
        logger.error(f"Error listing robots: {e}")
        return {"status": "error", "message": str(e), "robots": []}


@app.get("/robots/{name}")
def get_robot(name: str):
    """Get a single robot record by name."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    record = get_robot_record(name)
    if record is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})
    return {"status": "success", "robot": _record_with_clean(record)}


@app.post("/robots/{name}")
def upsert_robot(name: str, data: dict, create: bool = False):
    """
    Upsert a robot record.

    - `?create=true` is the "Add Robot" path: returns 409 if a record with that
      name already exists; otherwise creates with empty fields then merges body.
    - Without `?create=true` is the "patch" path (e.g., calibration write-back):
      merges body into existing record. If no record exists, no-ops and returns
      success — see deletion-during-calibration edge case in the spec.
    """
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})

    body = data or {}
    existing = get_robot_record(name) or {}

    # Mode is fixed at creation. A bimanual rig is a different machine (different
    # robot_type on datasets, forced _left/_right calibration naming, different
    # arms/cameras), and allowing a live toggle was a recurring stale-state bug
    # source. On the patch path (no ?create=true) reject any body `mode` that
    # differs from the stored value; a same-value echo stays a no-op. On create
    # the mode in the body is what establishes it.
    if (
        not create
        and existing
        and body.get("mode") in ("single", "bimanual")
        and body["mode"] != existing.get("mode", "single")
    ):
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "Mode is fixed at creation — create a new robot for a bimanual (or single-arm) setup.",
            },
        )

    # Effective mode for the slot/port conflict checks below. Because mode can't
    # change on an existing record, this is the stored mode for patches and the
    # body mode for creates (defaulting to single).
    effective_mode = (
        body["mode"]
        if create and body.get("mode") in ("single", "bimanual")
        else existing.get("mode", "single")
    )

    # Reject assigning the same calibration to both same-side arms of a bimanual
    # robot — that would point two physical arms at one calibration. Only checked
    # when the request actually touches a config slot, so unrelated edits
    # (cameras, ports) aren't blocked even on a pre-existing conflict.
    config_fields = ("leader_config", "follower_config", "right_leader_config", "right_follower_config")
    if any(f in body for f in config_fields):
        prospective = {"mode": effective_mode}
        for f in config_fields:
            prospective[f] = body[f] if isinstance(body.get(f), str) else existing.get(f, "")
        side = config_slot_conflict(prospective)
        if side:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"That {side} config is already assigned to the other {side} arm. "
                    "Each physical arm needs its own calibration — pick a different config.",
                },
            )

    # Reject assigning one serial port to more than one arm — each physical arm
    # is its own USB device. Checked when the request touches a port.
    port_field_names = ("leader_port", "follower_port", "right_leader_port", "right_follower_port")
    if any(f in body for f in port_field_names):
        prospective = {"mode": effective_mode}
        for f in port_field_names:
            prospective[f] = body[f] if isinstance(body.get(f), str) else existing.get(f, "")
        dup_port = port_slot_conflict(prospective)
        if dup_port:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"Port {dup_port} is already assigned to another arm of this robot. "
                    "Each arm needs its own serial port.",
                },
            )

    try:
        if create:
            if get_robot_record(name) is not None:
                return JSONResponse(
                    status_code=409,
                    content={"status": "error", "message": "A robot with this name already exists"},
                )
            save_robot_record(name, data or {}, allow_create=True)
        else:
            save_robot_record(name, data or {}, allow_create=False)
        record = get_robot_record(name)
        if record is None:
            return {"status": "success", "robot": None}
        return {"status": "success", "robot": _record_with_clean(record)}
    except Exception as e:
        logger.error(f"Error upserting robot {name}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/robots/{name}/rename")
def rename_robot(name: str, data: dict):
    """
    Rename a robot record. Body: {"new_name": "..."}. Calibration files are not
    affected (they're keyed by config name, not robot name).
    """
    new_name = (data or {}).get("new_name", "")
    if not isinstance(new_name, str):
        return JSONResponse(
            status_code=400, content={"status": "error", "message": "new_name must be a string"}
        )
    new_name = new_name.strip()

    ok, reason = rename_robot_record(name, new_name)
    if ok:
        record = get_robot_record(new_name)
        return {"status": "success", "robot": _record_with_clean(record) if record else None}

    status_code, message = {
        "invalid_name": (400, "Invalid robot name"),
        "not_found": (404, "Robot not found"),
        "name_taken": (409, "A robot with that name already exists"),
    }.get(reason, (500, "Rename failed"))
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


@app.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


@app.on_event("startup")
def startup_event():
    """One-time startup diagnostics surfaced in the server terminal."""
    warn_if_cuda_mismatch()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when FastAPI shuts down"""
    logger.info("🔄 FastAPI shutting down, cleaning up...")

    # Teleoperation and recording drive the follower(s) as background threads
    # INSIDE this process (teleoperation_thread / recording_thread); auto-
    # calibration and an in-flight inference run each drive real, independent
    # subprocess(es) (the vendored autocal script; `lerobot-rollout`). None of
    # this stops just because the process does — a plain `kill <pid>` or a
    # uvicorn `--reload` restart otherwise leaves a thread killed mid-loop
    # (no return-to-rest, no torque release) or a subprocess orphaned with the
    # arm potentially still energized, and nobody left to reach it from the
    # API. Each stop_and_wait()/handle_stop_inference() call is the same stop
    # its own Stop control uses, but blocks (bounded) until actually confirmed
    # stopped instead of just kicking it off — safe to wait here since this is
    # a one-time shutdown step, not a UI action waiting on a poll loop. All
    # five are mutually exclusive with each other in normal operation (see
    # CLAUDE.md's "State model & mutual exclusion"), so at most one of these
    # ever has real work — gathered concurrently anyway, both because it's
    # cheap and as a defensive measure if that invariant is ever violated,
    # rather than paying each stop's worst case one after another.
    results = await asyncio.gather(
        asyncio.to_thread(stop_teleoperation_and_wait),
        asyncio.to_thread(stop_recording_and_wait),
        asyncio.to_thread(auto_calibration_manager.stop_and_wait),
        asyncio.to_thread(auto_calibration_batch_manager.stop_and_wait),
        asyncio.to_thread(handle_stop_inference),
        return_exceptions=True,
    )
    labels = ("teleoperation", "recording", "auto-calibration", "auto-calibration batch", "inference")
    for label, result in zip(labels, results, strict=True):
        if isinstance(result, Exception):
            logger.exception(f"Failed to stop {label} during shutdown", exc_info=result)

    if manager:
        manager.stop_broadcast_thread()
    logger.info("✅ Cleanup completed")


def _accepts_html(accept: str) -> bool:
    """Whether an Accept header explicitly wants text/html (quality > 0).

    Browser navigations list `text/html` with a positive quality value, so
    they get the SPA shell. A `text/html;q=0` entry is an explicit refusal and
    must not count — a plain substring check would wrongly treat it as a yes.
    `*/*` (curl, XHR, API clients) is deliberately not treated as wanting HTML.
    """
    for part in accept.split(","):
        media_type, _, params = part.strip().partition(";")
        if media_type.strip().lower() != "text/html":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown client-side routes.

    The frontend is a single-page app: routes like /recording and /calibration
    exist only in the browser's router, not as files on disk. A hard reload or
    deep link to one of those URLs asks the server for a file that isn't there;
    plain StaticFiles answers 404 ({"detail":"Not Found"}), so the page breaks.

    Here we fall back to index.html on 404 so the SPA boots and its router
    renders the route. Only requests that accept HTML (i.e. browser navigations)
    get the fallback — API typos, XHR, and curl still receive a JSON 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _accepts_html(Headers(scope=scope).get("accept", "")):
                return await super().get_response("index.html", scope)
            raise


# Serve the built frontend at /. Must be mounted last so API routes win.
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; run `npm run build` in frontend/ or use `makermodslab --dev`."
    )
