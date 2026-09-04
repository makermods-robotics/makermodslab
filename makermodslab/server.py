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
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, Field, StringConstraints, ValidationError
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
    remote_host,
    remote_teleoperate,
    rollout as rollout_state,
    session_events,
    sfu,
)

# Import our custom calibration functionality
from .__version__ import __version__
from .api_errors import ApiError, ErrorCode, install_error_handlers
from .auto_calibrate import (
    AutoCalibrationBatchRequest,
    AutoCalibrationRequest,
    auto_calibration_batch_manager,
    auto_calibration_manager,
)
from .calibrate import CalibrationRequest, calibration_manager
from .camera_identity import identify_cv2_index, pump_avfoundation_runloop
from .camera_preview import CameraOpenError, camera_preview_manager
from .can_recovery import ReleaseCanTorqueRequest, handle_release_can_torque
from .dagger_protocol import (
    CMD_CANCEL,
    CMD_DROP_LAST,
    CMD_HANDBACK,
    CMD_HOLD,
    CMD_RECOVERED,
    CMD_RESET,
    CMD_RESUME,
    CMD_TAKEOVER,
)
from .identify import identify_arm_by_motion
from .jobs import (
    _KNOWN_FOUNDATION_BASE_REPO_IDS,
    CHECKPOINTS_STAGING_SUFFIX,
    DatasetHubCopyEmptyError,
    DatasetNotOnHubError,
    JobAlreadyContinuedError,
    JobHasChildrenError,
    JobNotFoundError,
    JobNotRunningError,
    JobPublishInProgressError,
    JobRemovalFailedError,
    JobSourceOfQueuedRunError,
    JobState,
    JobStateChangedError,
    JobTarget,
    QueueChangedError,
    _list_local_checkpoints,
    hub_ref_repo_id,
    hub_ref_step_label,
    job_registry,
    training_is_active,
)
from .maker_ports import identify_maker_arm_by_motion, probe_maker_ports
from .merge import MergeRequest, handle_merge_status, handle_start_merge
from .motor_power import read_supply_voltage
from .nodes import (
    NodeNotFoundError,
    NodeUnreachableError,
    handle_add_node,
    handle_delete_node_job,
    handle_get_node_job,
    handle_get_node_job_logs,
    handle_get_node_jobs,
    handle_get_node_policy_extra,
    handle_get_node_policy_extra_status,
    handle_get_node_queue,
    handle_install_node_policy_extra,
    handle_list_node_sources,
    handle_list_nodes,
    handle_remove_node,
    handle_restart_node,
    handle_stop_node_job,
)

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
from .replay import (
    ReplayRequest,
    handle_replay_status,
    handle_start_replay,
    handle_stop_replay,
    stop_and_wait as stop_replay_and_wait,
)
from .rollout import (
    InferenceRequest,
    handle_coaching_command,
    handle_inference_log,
    handle_inference_status,
    handle_next_episode,
    handle_start_inference,
    handle_stop_episode,
    handle_stop_inference,
)

# Response models for the typed /api/v1 surface (see makermodslab/schemas/).
from .schemas.datasets import (
    DatasetHubSettingsResponse,
    DatasetHubStatusResponse,
    DatasetInfoResponse,
    DatasetListItem,
    DatasetRenameResponse,
    DatasetTagsResponse,
    DatasetVisibilityResponse,
    DeleteDatasetResponse,
    DownloadStartResponse,
    DownloadStatusResponse,
    EpisodeJointSeriesResponse,
    EpisodeSummary,
    ExcludedEpisodesResponse,
    ImportResponse,
    MergeStartResponse,
    MergeStatusResponse,
    SetExcludedEpisodesResponse,
    SuccessRepoIdResponse,
    UploadStartResponse,
    UploadStatusResponse,
)
from .schemas.jobs import (
    CheckpointPolicyConfigResponse,
    HubJobDismissResponse,
    HubJobsResponse,
    HubModelDeleteResponse,
    JobCheckpointsResponse,
    JobListResponse,
    JobLogsResponse,
    JobMetricsHistoryResponse,
    JobQueueResponse,
    JobRecord,
    RunnersHardwareResponse,
)
from .schemas.models import (
    ModelDeleteResponse,
    ModelInfoResponse,
    ModelListItem,
    ModelPublishStartResponse,
    ModelPublishStatusResponse,
    ModelUploadResponse,
    RunCheckpointsResponse,
    SkillsResponse,
)
from .schemas.nodes import (
    NodeEntry,
    NodeListResponse,
    NodeRemoveResponse,
)
from .schemas.remote import (
    HostingStatusResponse,
    RemoteCommandResponse,
    RemoteTeleoperationStatusResponse,
    StationStatusResponse,
)
from .schemas.sessions import (
    CoachingCommandResponse,
    CurrentSessionResponse,
    SessionCoachingBody,
    SessionCoachingResponse,
    SessionHeartbeatBody,
    SessionHeartbeatResponse,
    SessionStartBody,
    SessionStartResponse,
    SessionStopResponse,
)
from .schemas.sfu import SfuTokenResponse
from .schemas.system import (
    AvailableCamerasResponse,
    AvailablePortsResponse,
    ExtraStatus,
    HealthResponse,
    HfAuthStatusResponse,
    HfLoginResponse,
    InstallStartResponse,
    InstallStatusResponse,
    MakerIdentifyArmResponse,
    MakerProbePortsResponse,
    PolicyExtraStatus,
    PolicyOptimizerDefaultsResponse,
    ReleaseCanTorqueResponse,
    RestartResponse,
    RobotPortResponse,
    SupplyVoltageResponse,
    UpdateResult,
    UpdateStatus,
)
from .sessions import (
    handle_coaching_command_for_session,
    handle_current_session,
    handle_heartbeat_session,
    handle_start_session,
    handle_stop_session,
    held_by,
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
    HOME_IS_OVERRIDDEN,
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
    get_excluded_episodes,
    get_instance_id,
    get_robot_record,
    get_saved_robot_port,
    is_robot_record_clean,
    is_valid_robot_name,
    list_robot_records,
    migrate_legacy_state,
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
    set_excluded_episodes,
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
    handle_get_remote_extra,
    handle_get_training_extra,
    handle_get_wandb_extra,
    handle_install_policy_extra,
    handle_install_policy_extra_status,
    handle_install_remote_extra,
    handle_install_remote_extra_status,
    handle_install_training_extra,
    handle_install_training_extra_status,
    handle_install_wandb_extra,
    handle_install_wandb_extra_status,
    install_in_progress,
    open_folder_in_file_browser,
    probe_gpu,
    restart_supported,
    schedule_restart,
    warn_if_cuda_mismatch,
)
from .wiggle import wiggle_gripper
from .zero_calibrate import zero_calibration_is_active, zero_calibration_manager

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


def _refuse_repeated_query_keys(request: Request) -> None:
    """Route dependency: 422 when any query key appears more than once.

    Guards `expect_state` on the stop/cancel routes. FastAPI resolves a
    repeated scalar key to its LAST value (starlette's multidict keeps the
    final duplicate), so `?expect_state=queued&expect_state=running` reached
    `JobRegistry.stop` as `running` — a Cancel-shaped URL with a stray
    duplicate (a retrying proxy that appends instead of replacing, a mangled
    copy-paste) walked past the optimistic-concurrency precondition and
    SIGTERMed a live run while the caller believed it cancelled a queued one.
    A repeated key is one request making two contradictory claims; refuse it
    as malformed rather than picking a winner.

    Declared as a plain-`Request` dependency so the parameter's OpenAPI schema
    stays the scalar it always was — this changes no contract, it just stops
    resolving an ambiguity that should never have been resolvable.
    """
    params = request.query_params
    repeated = sorted({key for key in params if len(params.getlist(key)) > 1})
    if repeated:
        names = ", ".join(repr(k) for k in repeated)
        raise ApiError(
            status_code=422,
            detail=f"Query parameter {names} was given more than once; pass each key at most once.",
            code=ErrorCode.REQUEST_VALIDATION,
        )


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

# Every endpoint registers on this router, which is mounted twice at the bottom
# of the module: once flat (the surface the shipped frontend was built against)
# and once under /api/v1 (the versioned surface SDK clients target). The two
# stay identical by construction; tests/test_api_contract.py asserts it.
router = APIRouter()

# NEW surface registers here instead: this router is mounted ONLY under
# /api/v1 (the flat mount is frozen — LEGACY_ROUTES is a shrink-only ratchet).
# Each addition is documented in tests/test_api_contract.py V1_ONLY_ROUTES.
v1_router = APIRouter()

# ApiError responses carry a machine-readable `code` beside the legacy string
# `detail` (see api_errors.py); plain HTTPException raises are untouched.
install_error_handlers(app)

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

    def notify_session_changed(self, event: dict) -> None:
        """Push a feature module's 'session_changed' hint to all WS clients.

        Wired into makermodslab/session_events.py below so the feature modules
        never import the manager. The event dict is built by the seam
        (type/session/timestamp); like notify_jobs_changed this is a droppable
        hint — skipped silently with no clients connected, and consumers
        refetch the relevant status endpoint rather than trusting the payload,
        so a missed broadcast is self-healing (every page already polls).
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(event)

    def notify_coaching_state(self, fields: dict[str, Any]) -> None:
        """Push the coaching block the instant it changes, ahead of the poll.

        Unlike `notify_jobs_changed` this carries the STATE rather than a
        "refetch me" nudge, because the thing it carries is safety-relevant and
        a refetch round-trip is most of the latency we are trying to remove:
        the operator has to know who is holding the arm now, not after another
        request. See rollout._on_coaching_state for the full argument.

        Dropped silently with no clients — the dialog polls once a second and
        reconciles itself, so a missed push costs a second, never correctness.
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(
                    {"type": "coaching_state", "timestamp": time.time(), **fields}
                )

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


def _on_jobs_changed() -> None:
    """Registry-change fan-out: drop the models listing cache, THEN announce.

    A run reaching a terminal state changes what `/models` lists — that is the
    moment it becomes a deployable skill — but no MODEL mutation ran, so until
    now nothing invalidated the listing cache. The picker stayed up to
    `_LISTING_CACHE_TTL_S` (45s) stale while the jobs-driven library, which
    reads the registry directly over a WS push, was already current. That gap
    is the transient half of "the two skill lists disagree". Registry renames
    had the same shape: `rename` fires this hook, but no route invalidated the
    listing, so the picker showed a run's old name for up to a TTL.

    Two placement details, both load-bearing:

      * Invalidation runs BEFORE the broadcast. Clients refetch on the event,
        so announcing first races a cache this call exists to drop.
      * Invalidation runs OUTSIDE `notify_jobs_changed`'s "are any clients
        connected?" guard. The broadcast is pointless with nobody listening;
        the cache drop is not — the next plain HTTP GET still wants the truth.
    """
    model_browser.invalidate_model_listing_cache()
    manager.notify_jobs_changed()


job_registry.set_on_change(_on_jobs_changed)
job_registry.set_on_progress(manager.notify_job_progress)
session_events.set_notifier(manager.notify_session_changed)
# Coaching phase changes reach the browser by push, not by poll — the banner
# names who is holding the arm, and a second of lag there is a second of the
# operator not knowing. See rollout._on_coaching_state.
rollout_state.set_on_coaching_state(manager.notify_coaching_state)


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


@router.get("/policy-optimizer-defaults", response_model=PolicyOptimizerDefaultsResponse, tags=["system"])
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


@router.post("/move-arm")
def teleoperate_arm(request: TeleoperateRequest):
    """Start teleoperation of the robot arm"""
    return handle_start_teleoperation(request, manager)


@router.post("/stop-teleoperation")
def stop_teleoperation():
    """Stop the current teleoperation session"""
    return handle_stop_teleoperation()


@router.get("/teleoperation-status")
def teleoperation_status():
    """Get the current teleoperation status"""
    return handle_teleoperation_status()


@router.post("/start-inference")
def start_inference(request: InferenceRequest):
    result = handle_start_inference(request)
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start inference"),
            code=result.get("code"),
        )
    return result


@router.post("/stop-inference")
def stop_inference():
    """Abort the whole session. In evaluation mode (eval_episodes > 1) this ends
    the run wherever it is and reports the partial tally with NO accuracy — the
    per-episode control is /inference-episode-stop."""
    result = handle_stop_inference()
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop inference"),
            code=result.get("code"),
        )
    return result


@router.post("/inference-episode-stop")
def inference_episode_stop():
    """Evaluation mode only: end the CURRENT episode early and score it a
    SUCCESS ("the robot did the task"). The session stays up and moves into its
    reset phase. 409 when no evaluation episode is running."""
    result = handle_stop_episode()
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop the episode"),
            code=result.get("code"),
        )
    return result


@router.post("/inference-next-episode")
def inference_next_episode():
    """Evaluation mode only: leave the reset phase and start the next episode.
    The reset is user-ended (no auto-timer). 409 unless an evaluation is parked
    waiting for a reset."""
    result = handle_next_episode()
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start the next episode"),
            code=result.get("code"),
        )
    return result


def _coaching_route(command: str):
    """Shared body for the eight flat coaching controls.

    They differ only in the verb they forward, so the route layer's job is
    entirely uniform: hand the verb to the orchestrator and translate a refusal
    into the right status code. Which transitions the verb is legal from is the
    RUNNER's call, not this layer's — see `handle_coaching_command`."""
    result = handle_coaching_command(command)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to send the coaching command"),
        )
    return result


@v1_router.post("/coaching-takeover", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_takeover():
    """Coaching mode only: take control from the policy and start recording.

    One press covers the whole handover — the policy pauses, an actuated leader
    glides toward the follower's pose so the operator picks up an arm roughly
    where the robot is, and only then does the correction begin recording. The
    glide is best-effort and nothing checks that it arrived: whatever gap is
    left is measured at the edge and cancelled out of every command, so the
    follower cannot jump however far apart the two arms were."""
    return _coaching_route(CMD_TAKEOVER)


@v1_router.post("/coaching-handback", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_handback():
    """Coaching mode only: end the correction, SAVE it, and resume the policy."""
    return _coaching_route(CMD_HANDBACK)


@v1_router.post("/coaching-cancel", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_cancel():
    """Coaching mode only: end the correction and DISCARD it.

    The fumbled-takeover escape. Upstream lerobot saves every correction
    unconditionally, which makes a botched takeover permanent training data;
    this drops the buffer instead.

    It then runs the ordinary reset behind the discard — the follower eases home
    and the session parks for a scene rearrangement, and the leader is released
    on the way. A discard means the last few seconds were a mess, and the scene
    almost always needs setting up again after one. Accepted from EVERY phase,
    not just mid-correction: with nothing in flight it is a plain reset, which
    is what still gives a wedged correction a way out."""
    return _coaching_route(CMD_CANCEL)


@v1_router.post("/coaching-hold", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_hold():
    """Coaching mode only: freeze the policy without taking over.

    The arm holds its pose and nothing is recorded — for when the operator needs
    a moment to decide, or to reposition the scene, without committing to a
    correction."""
    return _coaching_route(CMD_HOLD)


@v1_router.post("/coaching-resume", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_resume():
    """Coaching mode only: hand control back to the policy from a hold."""
    return _coaching_route(CMD_RESUME)


@v1_router.post("/coaching-reset", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_reset():
    """Coaching mode only: end this ATTEMPT at the task and reset for the next.

    Corrections-only DAgger has no task-episode concept — an "episode" there is
    one takeover — so this is what tells the session that the cube is finally in
    the tray. The policy stops, the follower eases back to the pose captured at
    connect, and the session parks so the scene can be rearranged. Nothing is
    written to the dataset: corrections are the only thing ever recorded.

    Valid mid-correction, where it SAVES the correction in flight before
    resetting: an operator who finishes the task while still driving has
    already decided those frames are the correction, and making them hand back
    and then reset as two presses let the policy briefly regain a finished
    scene in between. Use /coaching-cancel for the opposite — discard, then
    reset."""
    return _coaching_route(CMD_RESET)


@v1_router.post("/coaching-drop-last", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_drop_last():
    """Coaching mode only: un-record the correction from the attempt just ended.

    A real delete, and it can be one only because nothing has been deleted: the
    runner HOLDS a finished correction in memory rather than writing it at
    hand-back, and commits it when the operator takes over again or starts the
    next attempt. This says don't.

    That indirection is not an optimisation. `save_episode` interleaves an
    episode's frames into a shared per-chunk parquet file and appends its video
    into a shared per-chunk video file, and lerobot offers nothing that removes
    one episode from a dataset that is still open — `dataset_tools.delete_episodes`
    rebuilds a FINALIZED dataset into a new directory by copying and re-encoding
    everything that survives. See "The held correction" in dagger_protocol.

    So the window is narrow and the runner owns it: from the hand-back that
    ended the correction until the operator either takes over again or starts
    the next attempt — exactly the window they spend standing at a parked arm
    deciding. Clients must read
    `droppable_correction` off /inference-status rather than infer it from the
    phase.

    Refused mid-correction — there the operator means the take they are still
    recording, and /coaching-cancel is the control for that one.

    409 `coaching.nothing_to_drop` when the window is shut (mid-correction, or
    once the correction has been committed or already dropped). That is the one
    state check this surface makes, and it is not a phase guess: it reads
    `droppable_correction`, the runner's own published window. It used to answer
    200 "sent" and let the runner refuse in a log nobody reads."""
    return _coaching_route(CMD_DROP_LAST)


@v1_router.post("/coaching-recovered", response_model=CoachingCommandResponse, tags=["inference"])
def coaching_recovered():
    """Coaching mode only: mark the end of RECOVERY inside the correction.

    An intervention is two things wearing one name — first the operator rewinds
    the arm back to a state the policy has actually seen, then they demonstrate
    the behaviour that should follow. lerobot's own HIL guide names RaC
    (arXiv:2509.07953) as the protocol its DAgger strategy follows, and RaC's
    entire claim rests on that decomposition; the strategy nonetheless records
    both halves as one undifferentiated `intervention=True`.

    This records the boundary out of band (a sidecar beside the dataset — see
    dagger_protocol) because the dataset's feature dict is assembled inside
    lerobot with no hook to add a column. It requests no phase change: recovery
    and correction are the same control mode.

    Ignored outside a correction, and ignored a second time within one — the
    first mark is the one the operator meant."""
    return _coaching_route(CMD_RECOVERED)


@router.get("/inference-status")
def inference_status():
    return handle_inference_status()


@router.get("/inference-log")
def inference_log():
    """Tail of the active/most-recent rollout's log file (read-only, bounded).

    Returns {logs, log_path, belongs_to}; empty logs (not an error) when no run
    has produced output yet, so the frontend can poll unconditionally.

    `belongs_to` is "active" (the running session's own log), "last_run" (the
    most recent finished run of this server process) or null (nothing to show) —
    the caller must not present a "last_run" log as the live session's output."""
    return handle_inference_log()


@router.post("/start-replay")
def start_replay(request: ReplayRequest):
    result = handle_start_replay(request, manager)
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start replay"),
            code=result.get("code"),
        )
    return result


@router.post("/stop-replay")
def stop_replay():
    result = handle_stop_replay()
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop replay"),
            code=result.get("code"),
        )
    return result


@router.get("/replay-status")
def replay_status():
    return handle_replay_status()


# --- Sessions (v1-only surface; see v1_router note above) ---


@v1_router.post("/sessions", response_model=SessionStartResponse, status_code=201, tags=["sessions"])
def start_session(body: SessionStartBody):
    """Start a robot session by robot name; ports, configs, mode, right-arm
    fields and cameras resolve server-side from the saved robot record, and
    `options` carries only the kind-specific fields (see schemas/sessions.py).

    Startable kinds: teleoperation, recording, inference, replay,
    calibration, auto_calibration. Only wiggle still starts through its
    legacy flow endpoint (seconds of open-loop motion, no stop handler) —
    the identity tracker observes it all the same. Calibration's mid-session
    wizard controls (complete-calibration-step, the status polls) stay on
    their existing endpoints, like recording's pause/rerecord.

    201 returns the session identity plus optional `warnings` — warn-but-
    allow findings from the feature's start (teleoperation/replay
    arm-identity checks) that the legacy start responses used to carry; 409
    session.held (details name the holder) when any session already holds
    the hardware; 404 robot.not_found; 400 robot.not_ready (readiness is
    scoped to the arms the kind drives — inference/replay never open the
    leader bus; the calibration kinds need only a port per targeted slot);
    422 request.validation for options that don't fit the kind, an
    empty/oversized owner, or a lease_timeout_s outside 10–600. Other
    feature refusals pass through with their existing statuses and codes.

    `owner` attaches a lease: heartbeat within `lease_timeout_s` (default
    60s) or the session is safety-stopped. No owner, no lease, no
    timeout-stop — legacy-started and owner-less sessions are never killed."""
    return handle_start_session(body, manager)


@v1_router.get("/sessions/current", response_model=CurrentSessionResponse, tags=["sessions"])
def current_session():
    """Identity of the current session (or null), plus a summary of the last
    ended one. Identity only — kind-specific rich status stays on the feature
    status endpoints this phase. `robot`/`owner` are null for sessions started
    through the legacy endpoints (the tracker never guesses); `lease` is null
    unless the session was created with an owner. Reading NEVER renews the
    lease — renewal is the owner's deliberate act via the heartbeat endpoint."""
    return handle_current_session()


@v1_router.post(
    "/sessions/{session_id}/heartbeat", response_model=SessionHeartbeatResponse, tags=["sessions"]
)
def heartbeat_session(session_id: str, body: SessionHeartbeatBody):
    """Renew the current session's lease deadline — the owner's deliberate
    act (GET /sessions/current never renews).

    200 with the renewed identity when `session_id` names the current session
    and `owner` matches its lease; a current session with NO lease is a
    harmless no-op 200 (eases client rollout while leases are opt-in). 404
    session.not_found for an unknown or stale id — including a session the
    expiry watchdog already stopped and released; 409 session.lease_expired
    only in the window where the expiry stop is dispatched but the release
    hasn't landed; 409 session.not_owner on an owner mismatch."""
    return handle_heartbeat_session(session_id, body.owner)


@v1_router.post("/sessions/{session_id}/stop", response_model=SessionStopResponse, tags=["sessions"])
def stop_session(session_id: str):
    """Stop the current session by its own id — 404 session.not_found unless
    `session_id` names the session that is actually running, so a stale stop
    can never hit a session it didn't mean (the operation-identity guarantee).
    Returns the kind's stop-handler result verbatim beside the final
    identity."""
    return handle_stop_session(session_id)


@v1_router.post(
    "/sessions/{session_id}/coaching",
    response_model=SessionCoachingResponse,
    tags=["sessions"],
)
def coaching_command(session_id: str, body: SessionCoachingBody):
    """Send one coaching (DAgger) command to the current inference session.

    Session-scoped like /stop: 404 `session.not_found` unless `session_id`
    names the running session; a plain (non-coaching) inference session yields
    the runner's coded 409. The verb — takeover / handback / cancel / hold /
    resume / reset / recovered / drop_last — is forwarded to the coaching
    runner, which alone decides which phase it is legal from (a server-side
    phase copy is always one event stale). Never owner-gated: a physical arm
    must stay controllable by whoever can reach the API.

    THIS is the endpoint the browser uses. The flat `/coaching-*` verbs below
    remain for callers that only know "an inference run is active", exactly as
    `/stop-inference` does beside `/sessions/{id}/stop` — but anything holding a
    session id must come through here, or a dialog left open across a session
    change commands whichever session happens to be current instead of failing.
    """
    return handle_coaching_command_for_session(session_id, body.command)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check(request: Request):
    """Node identity + capability document.

    Doubles as the node-registry verify handshake: a discovered peer is
    confirmed by fetching this and reading version/instance_id/capabilities.
    `status`/`message` are the legacy reachability-probe fields — keep them.
    Capabilities grow additively (gpu, hardware inventory) as the registry
    needs them; absent key means "unknown/unsupported", never guess."""
    return {
        "status": "ok",
        "message": "FastAPI server is running",
        "version": __version__,
        "instance_id": get_instance_id(),
        "capabilities": {
            "serves_ui": ui_enabled(),
            "accepts_jobs": True,
            # Present only when the torch probe sees an accelerator — an
            # absent key means none/unknown, never guess (see HealthResponse).
            **({"gpu": gpu} if (gpu := probe_gpu()) else {}),
            # Present only when this process runs (or fronts) a LiveKit SFU
            # (--sfu): the signalling URL as reachable from the caller's
            # side. Absent = no SFU here; a peer wanting one asks another
            # node. Same absent-means-unknown rule as gpu.
            **(
                {"sfu": {"url": sfu.sfu_url(request.url.hostname or "localhost")}}
                if sfu.sfu_enabled()
                else {}
            ),
            # Present only while a hosting session is live — the robot this
            # station offers for remote teleoperation. A laptop's station
            # picker filters on it. Same absent-means-none rule.
            **_hosting_capability(),
        },
    }


def _hosting_capability() -> dict:
    descriptor = remote_host.current_descriptor
    if not remote_host.hosting_active or not descriptor:
        return {}
    return {
        "hosting": {
            "robot": descriptor["robot"],
            "arm_type": descriptor["arm_type"],
            "phase": remote_host.phase,
            "active_operator": remote_host.seat_holder(),
        }
    }


# --- SFU token broker (v1-only surface; see v1_router note above) ---


class SfuTokenBody(BaseModel):
    """Request for a LiveKit room token (sfu.py). Every field is optional:
    the server picks a unique identity and the station's default room, and
    `operator` is the role a laptop or a policy worker wants. `robot` is for
    the one participant that publishes cameras/state (normally this station
    itself, in a later phase); `viewer` subscribes only."""

    identity: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")] | None = None
    room: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")] | None = None
    role: Literal["robot", "operator", "viewer"] = "operator"
    ttl_seconds: int = Field(sfu.DEFAULT_TTL_SECONDS, ge=sfu.MIN_TTL_SECONDS, le=sfu.MAX_TTL_SECONDS)


@v1_router.post("/sfu/token", response_model=SfuTokenResponse, tags=["sfu"])
def issue_sfu_token(body: SfuTokenBody, request: Request):
    """Sign a short-lived, role-scoped LiveKit room token.

    The station is the only party holding the SFU secret, so participants
    (a laptop's makermodslab, a Modal worker, a browser) get their JWT here
    instead of carrying the secret. The URL is built from the host the
    caller reached THIS API on — the one address known to be routable from
    where they sit. 409 sfu.disabled when the launcher wasn't started with
    --sfu: the remedy is a restart with the flag, not a retry."""
    if not sfu.sfu_enabled():
        raise ApiError(
            409,
            "No LiveKit SFU is configured on this node. Start it with `makermodslab --sfu`.",
            code=ErrorCode.SFU_DISABLED,
        )
    api_key, api_secret = sfu.api_keys()
    identity = body.identity or sfu.default_identity(body.role)
    # Single seat: while a hosting session's seat is held, only its holder
    # (a reconnect) gets another operator token. The room cap is the SFU's
    # half of the same rule.
    if body.role == "operator":
        holder = remote_host.seat_holder()
        if holder is not None and holder != identity:
            raise ApiError(
                409,
                f"This station's operator seat is held by {holder!r}. Only one operator drives at a time.",
                code=ErrorCode.SFU_SEAT_TAKEN,
                details={"holder": holder},
            )
    room = body.room or sfu.default_room(get_instance_id())
    token, expires_at = sfu.mint_token(
        api_key=api_key,
        api_secret=api_secret,
        identity=identity,
        room=room,
        role=body.role,
        ttl_seconds=body.ttl_seconds,
    )
    return {
        "url": sfu.sfu_url(request.url.hostname or "localhost"),
        "token": token,
        "room": room,
        "identity": identity,
        "role": body.role,
        "expires_at": expires_at,
    }


# --- Remote teleoperation (v1-only surface; see v1_router note above) ---


@v1_router.get("/hosting", response_model=HostingStatusResponse, tags=["remote"])
def get_hosting_status(request: Request):
    """The station's hosting descriptor + status (remote_host.py). An
    operator node reads this (through its registry) to learn the room, the
    codec/fps, and the motor/camera schema before joining; the URL is
    derived from the host the caller reached this API on."""
    return remote_host.handle_hosting_status(request.url.hostname or "localhost")


@v1_router.get("/remote-teleoperation", response_model=RemoteTeleoperationStatusResponse, tags=["remote"])
def get_remote_teleoperation_status():
    """The operator side's status (remote_teleoperate.py): which station,
    which room, the remote cameras being re-streamed, Portal RTT metrics."""
    return remote_teleoperate.handle_remote_teleoperation_status()


@v1_router.get("/remote-teleoperation/camera/{name}", tags=["remote"])
def get_remote_teleoperation_camera(name: str):
    """MJPEG re-stream of one remote camera during a remote teleoperation
    session, from the frames Portal delivers — the existing camera tiles
    consume it unchanged. 404 when no session (or no such camera)."""
    if not remote_teleoperate.remote_teleoperation_active or name not in remote_teleoperate.current_cameras:
        raise ApiError(404, f"No remote camera named {name!r} is streaming.", code=ErrorCode.ROBOT_NOT_FOUND)
    return StreamingResponse(
        remote_teleoperate.camera_stream(name), media_type="multipart/x-mixed-replace; boundary=frame"
    )


class StationRobotBody(BaseModel):
    """PUT /api/v1/station/robot — the robot this station hosts; null clears
    the choice (hosting stops once idle and waits for a new pick)."""

    robot: str | None = None


@v1_router.get("/station", response_model=StationStatusResponse, tags=["remote"])
def get_station_status():
    """Station mode posture (remote_host.py): whether this machine was started
    with --host, which robot it hosts, which saved robots it could host."""
    return remote_host.handle_station_status()


@v1_router.put("/station/robot", response_model=StationStatusResponse, tags=["remote"])
def set_station_robot(body: StationRobotBody):
    """Choose (or clear) the hosted robot. Remembered across restarts; a
    parked, unseated hosting session of another robot yields and the
    supervisor re-hosts the new choice within seconds; an engaged one is
    refused with session.held."""
    return remote_host.set_station_robot(body.robot)


@v1_router.post("/remote-teleoperation/home", response_model=RemoteCommandResponse, tags=["remote"])
def remote_teleoperation_home():
    """Park the station's arm (return to rest, torque off) and hold it there
    until Engage. Forwarded to the station as a Portal RPC; the station
    honours it only from the seated operator."""
    return remote_teleoperate.handle_remote_home()


@v1_router.post("/remote-teleoperation/engage", response_model=RemoteCommandResponse, tags=["remote"])
def remote_teleoperation_engage():
    """Re-energize the station's arm after a Home, with a soft start."""
    return remote_teleoperate.handle_remote_engage()


@v1_router.get("/system/remote-extra", response_model=ExtraStatus, tags=["system"])
def get_remote_extra():
    """Whether the `remote` extra (LiveKit Portal's lerobot plugins) is importable."""
    return handle_get_remote_extra()


@v1_router.post("/system/remote-extra/install", response_model=InstallStartResponse, tags=["system"])
def install_remote_extra():
    """Spawn the Portal plugins' pip install as a background subprocess. No-op if already running."""
    return handle_install_remote_extra()


@v1_router.get("/system/remote-extra/install-status", response_model=InstallStatusResponse, tags=["system"])
def install_remote_extra_status():
    """Current install state plus any pending log lines (drained on read)."""
    return handle_install_remote_extra_status()


# --- Node registry (v1-only surface; see v1_router note above) ---


class AddNodeBody(BaseModel):
    url: str
    name: str | None = None


@v1_router.get("/nodes", response_model=NodeListResponse, tags=["nodes"])
def list_nodes(request: Request, force: bool = False):
    """All known nodes: this server first (is_self=true, built from the same
    health fields the handshake reads, so clients render one uniform list),
    then every registered peer. Peers whose last probe is older than the TTL
    are re-verified inline; a peer that fails re-verification is reported
    `unreachable` but kept until explicitly removed. `sources` names the
    registered discovery sources, so a client can tell "no peers" apart from
    "discovery is off". ?force=true is the manual-refresh contract: this one
    pass bypasses the TTL — discovery runs now and every known entry is
    probed now — so a refresh button answers with the world as it is, not as
    it was up to TTL seconds ago."""
    health = health_check(request)
    self_entry = {
        "url": None,  # a server doesn't know its own external address
        "instance_id": health["instance_id"],
        "name": None,
        "version": health["version"],
        "capabilities": health["capabilities"],
        "status": "ok",
        "last_verified_at": None,  # no handshake needed with ourselves
        "last_seen_at": None,
        "is_self": True,
        "source": "manual",  # intrinsic, like a hand-added peer — never discovered
    }
    return {
        "nodes": [self_entry, *handle_list_nodes(force=force)],
        "sources": handle_list_node_sources(),
    }


@v1_router.post("/nodes", response_model=NodeEntry, tags=["nodes"])
def add_node(body: AddNodeBody):
    """Verify-on-add: GET {url}/api/v1/health and register the peer's
    identity. 200 returns the entry (also when a known peer's URL is updated
    in place); 422 request.validation for a non-http(s) url; 409 node.self /
    node.duplicate; 502 node.unreachable when the handshake fails (dead host
    or a non-node answer) — an unreachable peer is an error, never a pending
    state."""
    return handle_add_node(body.url, name=body.name)


@v1_router.get("/nodes/{instance_id}/jobs", response_model=JobListResponse, tags=["nodes"])
def get_node_jobs(instance_id: str):
    """Server-to-server workload proxy: the peer's own typed GET /api/v1/jobs,
    returned verbatim (the browser talks to ITS server; only servers talk to
    peers). The response reuses JobListResponse because the peer runs this
    same code — and on version skew the stance is passthrough: a newer peer's
    additive fields are dropped by the model, never an error, so the proxy
    doesn't break the moment one machine updates first. 404 node.not_found
    for an unknown instance_id; 502 node.unreachable when the peer doesn't
    answer (short timeout — a peer that can't list its jobs promptly is as
    good as down for scheduling purposes)."""
    return handle_get_node_jobs(instance_id)


@v1_router.get("/nodes/{instance_id}/jobs/queue", response_model=JobQueueResponse, tags=["nodes"])
def get_node_queue(instance_id: str):
    """The peer's own typed GET /api/v1/jobs/queue, passed through — the EXACT
    queue. The sibling jobs proxy reads the peer's default jobs page, which is
    limited and can undercount queued runs on a busy peer; a client that shows
    a queued count reads this instead. Same passthrough/version-skew stance
    and error mapping as the jobs proxy."""
    return handle_get_node_queue(instance_id)


# The drill-in proxies below share the {job_id} segment with the queue proxy's
# literal "queue"; the queue route is declared first, so FastAPI's first-match
# routing keeps /jobs/queue answering as the queue (same note as the local
# /jobs/{job_id} family).
@v1_router.get("/nodes/{instance_id}/jobs/{job_id}", response_model=JobRecord, tags=["nodes"])
def get_node_job(instance_id: str, job_id: str):
    """Drill-in proxy: the peer's own GET /api/v1/jobs/{job_id}, passed
    through verbatim (same passthrough/version-skew stance as the jobs proxy —
    a newer peer's additive fields are dropped by the model, never an error).
    404 node.not_found for an unknown instance_id; 502 node.unreachable for
    ANY failure to read the peer, its own 404 for an unknown job included."""
    return handle_get_node_job(instance_id, job_id)


@v1_router.get("/nodes/{instance_id}/jobs/{job_id}/logs", response_model=JobLogsResponse, tags=["nodes"])
def get_node_job_logs(instance_id: str, job_id: str):
    """The peer's own GET /api/v1/jobs/{job_id}/logs, passed through. The peer
    drains its runner's live queue per call, so this proxy is inherently
    incremental — each call returns only the lines that arrived since the last
    one, whoever made it. Same error mapping as the record proxy above."""
    return handle_get_node_job_logs(instance_id, job_id)


@v1_router.post(
    "/nodes/{instance_id}/jobs/{job_id}/stop",
    response_model=JobRecord,
    tags=["nodes"],
    # A repeated ?expect_state= must not silently resolve to one of its two
    # contradictory values — see _refuse_repeated_query_keys.
    dependencies=[Depends(_refuse_repeated_query_keys)],
)
def stop_node_job(instance_id: str, job_id: str, expect_state: JobState | None = None):
    """Forward a stop/cancel to the peer, `expect_state` precondition included.

    Error stance — subtly different from the GET proxies, where any HTTP error
    counts as unreachable: a stop is a request the peer may REFUSE for its own
    reasons (409 job.state_changed / job.has_queued_dependents, 404
    job.not_found, …), and those coded refusals pass through with the PEER's
    status and body, never re-wrapped as 502. Only transport-level failure is
    502 node.unreachable; 404 node.not_found still names an unknown NODE."""
    return handle_stop_node_job(instance_id, job_id, expect_state=expect_state)


# 204 No Content, like the peer's own delete — no body to model, so the route
# sits in RESPONSE_MODEL_EXEMPT (tests/test_api_contract.py).
@v1_router.delete("/nodes/{instance_id}/jobs/{job_id}", status_code=204, tags=["nodes"])
def delete_node_job(instance_id: str, job_id: str):
    """Forward a delete to the peer (terminal runs only — the peer refuses the
    rest). Same passthrough stance as the stop above: the peer's coded
    refusals (409 job.has_children / job.has_queued_dependents, 404
    job.not_found, …) keep THEIR status and body; only transport-level failure
    is 502 node.unreachable, and 404 node.not_found names an unknown node."""
    handle_delete_node_job(instance_id, job_id)


@v1_router.get(
    "/nodes/{instance_id}/policy-extra/{policy_type}", response_model=PolicyExtraStatus, tags=["nodes"]
)
def get_node_policy_extra(instance_id: str, policy_type: str):
    """Environment proxy: the peer's own GET /api/v1/system/policy-extra/
    {policy_type}, passed through — whether the extra the policy needs is
    importable in THE PEER's environment, the one an offloaded run imports
    from (the local answer is irrelevant to it). Same error mapping as the
    other GET proxies: 404 node.not_found for an unknown instance_id, 502
    node.unreachable for ANY failure to read the peer."""
    return handle_get_node_policy_extra(instance_id, policy_type)


@v1_router.get(
    "/nodes/{instance_id}/policy-extra/{policy_type}/install-status",
    response_model=InstallStatusResponse,
    tags=["nodes"],
)
def get_node_policy_extra_status(instance_id: str, policy_type: str):
    """The peer's own install-status, passed through. The peer drains pending
    pip log lines per call, so this proxy is inherently incremental — like
    the job-log proxy. Same error mapping as the GET proxies."""
    return handle_get_node_policy_extra_status(instance_id, policy_type)


@v1_router.post(
    "/nodes/{instance_id}/policy-extra/{policy_type}/install",
    response_model=InstallStartResponse,
    tags=["nodes"],
)
def install_node_policy_extra(instance_id: str, policy_type: str):
    """Forward the install to the peer: `pip install lerobot[<extra>]` runs
    THERE, in the environment its training subprocesses import from. Mutation
    stance, like the stop/delete proxies: the peer's own refusals keep THEIR
    status and body; only transport-level failure is 502 node.unreachable,
    and 404 node.not_found names an unknown node."""
    return handle_install_node_policy_extra(instance_id, policy_type)


@v1_router.post("/nodes/{instance_id}/restart", response_model=RestartResponse, tags=["nodes"])
def restart_node(instance_id: str):
    """Forward a restart to the peer (its own POST /api/v1/system/restart).
    200 means the peer ANSWERED and scheduled its re-exec — expect it to flap
    unreachable for a few seconds; the registry's probes pick it back up. The
    peer's coded refusals (409 session.held / robot.busy.training /
    system.restart_unsupported — or a plain 404 from a peer too old to have
    the endpoint) pass through with THEIR status and body; only transport
    failure is 502 node.unreachable."""
    return handle_restart_node(instance_id)


@v1_router.delete("/nodes/{instance_id}", response_model=NodeRemoveResponse, tags=["nodes"])
def remove_node(instance_id: str):
    """Remove a registered peer. 404 node.not_found for an unknown
    instance_id (including a saved peer that has never completed a handshake
    this run — those carry a null instance_id until verified)."""
    return handle_remove_node(instance_id)


@router.get("/hf-auth-status", response_model=HfAuthStatusResponse, tags=["system"])
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


class HfLoginBody(BaseModel):
    token: str


@router.post("/hf-auth/login", response_model=HfLoginResponse, tags=["system"])
def hf_auth_login(body: HfLoginBody):
    """Persist a pasted HF token (validated against whoami) for this user."""
    try:
        return handle_hf_login(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


# exclude_unset: `saved_custom` exists only on pin-fold rows (absent, never
# null, elsewhere) while `last_modified` is legitimately null on pinned rows —
# unset-exclusion reproduces each producer's exact keys where None-exclusion
# would eat the legitimate nulls.
@router.get(
    "/datasets",
    response_model=list[DatasetListItem],
    response_model_exclude_unset=True,
    tags=["datasets"],
)
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()


@router.get("/datasets/info", response_model=DatasetInfoResponse, tags=["datasets"])
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


@router.get("/datasets/episodes", response_model=list[EpisodeSummary], tags=["datasets"])
def datasets_episodes(repo_id: str):
    """Per-episode index/length/duration/tasks for the dataset viewer window.
    404 when the dataset isn't local or predates the v3.0 parquet episode
    layout (meta/episodes/chunk-*/file-*.parquet) the viewer reads."""
    episodes = dataset_browser.list_episode_summaries(repo_id)
    if episodes is None:
        raise HTTPException(status_code=404, detail=f"No viewable episode list for '{repo_id}'")
    return episodes


@v1_router.get(
    "/datasets/excluded-episodes",
    response_model=ExcludedEpisodesResponse,
    tags=["datasets"],
)
def datasets_excluded_episodes(repo_id: str):
    """Episode indices the user excluded from training for this dataset
    (curation, not deletion — see set_excluded_episodes). Empty list for a
    dataset with no exclusions."""
    return {"repo_id": repo_id, "episode_indices": get_excluded_episodes(repo_id)}


class ExcludedEpisodesRequest(BaseModel):
    repo_id: str
    episode_indices: list[int]


@v1_router.put(
    "/datasets/excluded-episodes",
    response_model=SetExcludedEpisodesResponse,
    tags=["datasets"],
)
def datasets_set_excluded_episodes(request: ExcludedEpisodesRequest):
    """Replace the excluded-episode set for one dataset. NEVER deletes or
    mutates the dataset — the viewer computes the training subset from this
    and sends it as dataset_episodes when launching a run."""
    set_excluded_episodes(request.repo_id, request.episode_indices)
    return {
        "success": True,
        "repo_id": request.repo_id,
        "episode_indices": get_excluded_episodes(request.repo_id),
    }


@router.get("/datasets/episode-joints", response_model=EpisodeJointSeriesResponse, tags=["datasets"])
def datasets_episode_joints(repo_id: str, episode_index: int):
    """Per-frame timestamp + joint (observation.state) values for one episode,
    for the dataset viewer's joint-position chart."""
    series = dataset_browser.get_episode_joint_series(repo_id, episode_index)
    if series is None:
        raise HTTPException(
            status_code=404, detail=f"No joint data for episode {episode_index} of '{repo_id}'"
        )
    return series


@router.get("/datasets/episode-video")
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


@router.get("/datasets/hub-status", response_model=DatasetHubStatusResponse, tags=["datasets"])
def datasets_hub_status(repo_id: str):
    """Whether a dataset repo with this id exists on the Hub.

    Fetched lazily by the info card (separate from /datasets/info) so it never
    blocks the card render. Degrades to status "unknown" offline/unauthenticated
    — see get_hub_status. repo_id is a query param because repo ids contain '/'.
    """
    return dataset_browser.get_hub_status(repo_id)


@router.get("/datasets/hub-settings", response_model=DatasetHubSettingsResponse, tags=["datasets"])
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


@router.post("/datasets/visibility", response_model=DatasetVisibilityResponse, tags=["datasets"])
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


@router.post("/datasets/tags", response_model=DatasetTagsResponse, tags=["datasets"])
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


@router.post("/datasets/rename", response_model=DatasetRenameResponse, tags=["datasets"])
def datasets_rename(body: DatasetRenameBody):
    """Rename a locally-cached dataset by moving its directory, and its Hub
    copy (if any) to match.

    `new_name` is the NAME PART ONLY — the namespace prefix stays fixed, so
    `ns/old` renamed to `new` becomes `ns/new`. Refuses (409) if the dataset is
    being recorded, merged, or trained on locally, or if the new name is
    already taken (locally or on the Hub).

    Returns `{success, repo_id, hub}`, where `hub` is `"renamed"` (the Hub copy
    moved too), `"none"` (the Hub has no copy of this dataset), or `"skipped"`
    (the Hub step didn't run — offline, logged out, or someone else's
    namespace — so a Hub copy, if any, kept its old name). The caller needs
    that distinction to avoid claiming a Hub rename that didn't happen.
    """
    try:
        result = dataset_browser.rename_local_dataset(body.repo_id, body.new_name)
    except dataset_browser.DatasetRenameError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    return {"success": True, **result}


class CustomDatasetRequest(BaseModel):
    repo_id: str


# A Hub dataset id is namespace/name; allow word chars, dot, and dash in each.
_CUSTOM_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


@router.post("/datasets/custom", response_model=SuccessRepoIdResponse, tags=["datasets"])
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


@router.delete("/datasets/custom", response_model=SuccessRepoIdResponse, tags=["datasets"])
def datasets_remove_custom(request: CustomDatasetRequest):
    """Unpin a saved custom dataset (does NOT touch the Hub or any local copy)."""
    repo_id = request.repo_id.strip()
    removed = remove_saved_custom_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    return {"success": removed, "repo_id": repo_id}


@router.post("/datasets/hide", response_model=SuccessRepoIdResponse, tags=["datasets"])
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


@router.delete("/datasets/hide", response_model=SuccessRepoIdResponse, tags=["datasets"])
def datasets_unhide(request: CustomDatasetRequest):
    """Unhide a dataset so it reappears in the listing (does NOT touch the Hub)."""
    repo_id = request.repo_id.strip()
    removed = remove_hidden_dataset(repo_id)
    dataset_browser.invalidate_dataset_listing_cache()
    return {"success": removed, "repo_id": repo_id}


class DatasetDownloadRequest(BaseModel):
    repo_id: str


@router.post("/datasets/download", response_model=DownloadStartResponse, tags=["datasets"])
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


@router.get("/datasets/download-status", response_model=DownloadStatusResponse, tags=["datasets"])
def datasets_download_status():
    """Current download state (idle | running | done | error) + repo_id, message,
    and error once failed. Polled by the info card so a download survives
    navigation."""
    return dataset_browser.download_manager.get_status()


class DatasetImportRequest(BaseModel):
    path: str
    name: str | None = None


@router.post("/datasets/import", response_model=ImportResponse, tags=["datasets"])
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


@router.post("/datasets/merge", response_model=MergeStartResponse, tags=["datasets"])
def datasets_merge(request: MergeRequest):
    """Aggregate 2+ datasets into a new local dataset in the background."""
    return handle_start_merge(request)


@router.get("/datasets/merge/status", response_model=MergeStatusResponse, tags=["datasets"])
def datasets_merge_status():
    """Current merge state + drained log lines (idle | running | done | error)."""
    return handle_merge_status()


@router.websocket("/ws/joint-data")
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


@router.post("/start-recording")
def start_recording(request: RecordingRequest):
    """Start a dataset recording session.

    Refuses (409) if recording, teleoperation, inference, calibration,
    auto-calibration, or a gripper wiggle is already active on this robot;
    400 for a malformed dataset name."""
    result = handle_start_recording(request)
    if not result.get("success"):
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start recording"),
            code=result.get("code"),
        )
    return result


@router.post("/stop-recording")
def stop_recording(discard: bool = False):
    """End the current recording session.

    ``discard`` is a query parameter (not a body) so the browser's page-leave
    safety net can POST it via a keepalive/beacon "simple request" during unload
    without tripping a CORS preflight. ``discard=false`` keeps every saved
    episode (Done); ``discard=true`` throws the recording away (Quit / an
    unintentional page exit) — see handle_stop_recording."""
    return handle_stop_recording(discard=discard)


@router.get("/recording-status")
def recording_status():
    """Get the current recording status"""
    return handle_recording_status()


@router.get("/recording-log")
def recording_log():
    """Tail of the current/most-recent recording session's log (read-only,
    bounded ring buffer). Returns {logs}; empty (not an error) before a session
    has captured anything, so the frontend can poll unconditionally."""
    return handle_recording_log()


@router.post("/recording-exit-early")
def recording_exit_early():
    """Skip to next episode (replaces right arrow key)"""
    return handle_exit_early()


@router.post("/recording-rerecord-episode")
def recording_rerecord_episode():
    """Re-record current episode (replaces left arrow key)"""
    return handle_rerecord_episode()


@router.post("/recording-pause")
def recording_pause():
    """Pause the reset-phase gap between episodes (mouse-only, no keyboard
    shortcut). No-ops outside the reset phase — see handle_pause_recording."""
    return handle_pause_recording()


@router.post("/recording-resume")
def recording_resume():
    """Resume a paused reset-phase gap. No-ops if not currently paused —
    see handle_resume_recording."""
    return handle_resume_recording()


# Tagged "datasets": handled in record.py for historical reasons, but this is a
# dataset-library operation (push a recorded dataset to the Hub), not part of
# the recording session flow.
@router.post("/upload-dataset", response_model=UploadStartResponse, tags=["datasets"])
def upload_dataset(request: UploadRequest):
    """Start a background upload of a local dataset to the Hub.

    Returns immediately with {started, repo_id, message}; poll /upload-status
    for progress. 409 when an upload is already running (frontend maps it to a
    "an upload is already running" toast)."""
    result = handle_upload_dataset(request)
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Upload could not be started"))
    return result


# exclude_unset: `docs_url` is set only alongside an auth-failure message
# (absent otherwise, never null), while repo_id/message/dataset_url ARE null in
# the idle state — unset-exclusion keeps both behaviors byte-identical.
@router.get(
    "/upload-status",
    response_model=UploadStatusResponse,
    response_model_exclude_unset=True,
    tags=["datasets"],
)
def upload_status():
    """Current upload state + repo_id, message, and dataset_url once done."""
    return handle_upload_status()


@router.post("/delete-dataset", response_model=DeleteDatasetResponse, tags=["datasets"])
def delete_dataset(request: DatasetInfoRequest):
    """Remove a recorded dataset directory from local disk."""
    return handle_delete_dataset(request)


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================
# A datasets-style browser for trained policies. Local models are the final
# checkpoint of each completed local training run (read from the job registry);
# Hub models are the user's LeRobot policy repos. See makermodslab/models.py.


# exclude_unset: the listing merges four producers whose rows carry different
# key sets (repo_id/private/target_steps/state/saved_custom are absent — never
# null — outside their producer) while other keys are legitimately null; see
# ModelListItem. Unset-exclusion reproduces each producer's exact keys.
@router.get(
    "/models",
    response_model=list[ModelListItem],
    response_model_exclude_unset=True,
    tags=["models"],
)
def models_list():
    """List trained models available to the user — local runs + Hub repos.

    Each entry carries a `source` field: "local", "hub", or "both" (a local run
    that was also pushed to the Hub). Mirrors GET /datasets."""
    return model_browser.list_all_models()


# exclude_unset for the same reason as GET /models: the rows come from the same
# four producers, whose key sets differ (see SkillListItem).
@v1_router.get(
    "/skills",
    response_model=SkillsResponse,
    response_model_exclude_unset=True,
    tags=["models"],
)
def skills_list():
    """Every trained policy, each saying whether it can actually run.

    The deployable projection of the same merged build `/models` serves, so the
    deploy picker and the models library can no longer disagree about what a
    skill is — they were reading two different endpoints (`/models` and the
    `/jobs` registry) and filtering them on two different rules.

    Envelope, not a bare array: `{skills, hub}`. `hub` reports whether the Hub
    half was reachable, because "the Hub is down" and "you own no skills" used
    to render identically as an empty list. Each row carries `weights`
    (ready/unverified/none), `superseded_by`, `deployable`, `origin` and the
    `job_id` that deploys it."""
    return model_browser.list_skills()


# exclude_unset for the same reason as GET /models: the local/hub/probe
# branches carry different key sets (see ModelInfoResponse).
@router.get(
    "/models/info",
    response_model=ModelInfoResponse,
    response_model_exclude_unset=True,
    tags=["models"],
)
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


@router.post("/models/upload", response_model=ModelUploadResponse, tags=["models"])
def models_upload(body: ModelUploadBody):
    """Push a local run's final checkpoint to the Hub as a PUBLIC, MakerModsLab-tagged
    model repo. MUTATES the Hub (creates/updates the repo). 400 offline; 403 when
    the token can't write the namespace; 404 when the local model has no saved
    checkpoint; 502 on any other Hub failure. Returns {repo_id, url, tags}.

    The single-checkpoint synchronous push, frozen for SDK clients — including
    its ON-HUB SHAPE: files land at the repo root, loadable by a plain
    from_pretrained(repo_id) (root_layout=True). The training view's
    multi-checkpoint picker uses POST /api/v1/models/publish instead, which
    step-addresses under checkpoints/<step>/."""
    try:
        return model_browser.upload_local_model(body.id, body.repo_id, root_layout=True)
    except model_browser.ModelError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


@v1_router.get("/models/checkpoints", response_model=RunCheckpointsResponse, tags=["models"])
def models_checkpoints(id: str):
    """The publish picker's source of truth for one local run: every checkpoint
    it saved, which steps are already on the Hub, and the repo a publish would
    target. `id` is a run id (query param for symmetry with /models/info).
    404 when the run has no uploadable checkpoint."""
    try:
        return model_browser.list_run_checkpoints(id)
    except model_browser.ModelError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


class ModelPublishBody(BaseModel):
    id: str
    repo_id: str | None = None
    # Which checkpoints to publish. Omitted ⇒ the run's final checkpoint only.
    # Every step lands in the SAME repo under checkpoints/<step>/pretrained_model,
    # so a later call adds to the same model card instead of creating a second repo.
    steps: list[int] | None = None


@v1_router.post("/models/publish", response_model=ModelPublishStartResponse, tags=["models"])
def models_publish(body: ModelPublishBody):
    """START publishing a local run's checkpoints to the Hub as ONE PUBLIC,
    MakerModsLab-tagged model repo. MUTATES the Hub (creates/updates the repo).

    Returns immediately with {started, model_id, message} — the queue runs
    sequentially in a background thread (a run's worth of checkpoints is
    gigabytes, far past what an inline request should hold open) and
    GET /api/v1/models/publish-status reports progress. 409 when a publish is
    already running; the per-step failures (400 offline, 403 permission, 404
    unknown step, 502 Hub) surface through that status, not this call."""
    try:
        result = model_browser.model_upload_manager.start(body.id, body.repo_id, body.steps)
    except model_browser.ModelError as exc:
        # A worker that could not even be spawned — the manager has already
        # released the slot (state "error"), so this is a 500, not a 409.
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Publish busy"))
    return result


@v1_router.get("/models/publish-status", response_model=ModelPublishStatusResponse, tags=["models"])
def models_publish_status():
    """Poll the single background publish: state (idle/running/done/error),
    target repo + url, `done`/`total`/`current_step` for the queue position, and
    `done_steps` — the steps already on the Hub, which stay meaningful after an
    error because a failed queue keeps everything it published before it died."""
    return model_browser.model_upload_manager.get_status()


class ModelDeleteBody(BaseModel):
    id: str


@router.post("/models/delete", response_model=ModelDeleteResponse, tags=["models"])
def models_delete(body: ModelDeleteBody):
    """Delete a local model — its training run's output dir (strictly sandboxed
    under outputs/train/). Never touches the Hub. 400 unsafe/non-local; 404
    unknown; 409 when the run is still training or queued (job.not_terminal —
    only a terminal run has artifacts to delete; a queued run is cancelled on
    the jobs surface, never through here); 502 on a delete failure."""
    try:
        return model_browser.delete_local_model(body.id)
    except model_browser.ModelError as exc:
        # ApiError so a machine-readable `code` (when the refusal carries one)
        # rides beside the legacy string detail; the body shape is unchanged
        # for code-less refusals.
        raise ApiError(status_code=exc.status, detail=exc.message, code=exc.code) from exc


class CustomModelRequest(BaseModel):
    repo_id: str


@router.post("/models/custom", response_model=SuccessRepoIdResponse, tags=["models"])
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


@router.delete("/models/custom", response_model=SuccessRepoIdResponse, tags=["models"])
def models_remove_custom(request: CustomModelRequest):
    """Unpin a saved custom model (does NOT touch the Hub or any local copy)."""
    repo_id = request.repo_id.strip()
    removed = remove_saved_custom_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    return {"success": removed, "repo_id": repo_id}


@router.post("/models/hide", response_model=SuccessRepoIdResponse, tags=["models"])
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


@router.delete("/models/hide", response_model=SuccessRepoIdResponse, tags=["models"])
def models_unhide(request: CustomModelRequest):
    """Unhide a model so it reappears in the listing (does NOT touch the Hub)."""
    repo_id = request.repo_id.strip()
    removed = remove_hidden_model(repo_id)
    model_browser.invalidate_model_listing_cache()
    return {"success": removed, "repo_id": repo_id}


class ModelDownloadRequest(BaseModel):
    repo_id: str


@router.post("/models/download", response_model=DownloadStartResponse, tags=["models"])
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


@router.get("/models/download-status", response_model=DownloadStatusResponse, tags=["models"])
def models_download_status():
    """Current model-download state (idle | running | done | error) + repo_id,
    message, and error once failed. Polled by the model info card so a download
    survives navigation. Mirrors /datasets/download-status."""
    return model_browser.model_download_manager.get_status()


class ModelImportRequest(BaseModel):
    path: str
    name: str | None = None


@router.post("/models/import", response_model=ImportResponse, tags=["models"])
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


def _wire_job_record(record: JobRecord) -> JobRecord:
    """The wire view of a JobRecord: `output_dir` relative to the training
    output root.

    A run's output_dir is `<output_root>/<id>/run` — an absolute path into
    this machine's home directory, shipped verbatim in every /jobs response
    (and, under `--lan`, to everyone on the network; the delete route already
    scrubs the same path from its error bodies for exactly this reason). No
    consumer needs the prefix: the frontend uses output_dir for display and
    search only, and the LanNodeJobRunner discards it. An IMPORTED record's
    output_dir is the user's own import path — data, not a leak — and it
    lives outside the root, so the prefix test leaves it (and any legacy
    out-of-root record) untouched. Registry-internal callers keep the
    absolute form: this wraps route returns only, on the copies the registry
    read paths already hand out."""
    out = record.output_dir or ""
    root = str(job_registry._output_root)
    if out == root or out.startswith(root + os.sep):
        return record.model_copy(update={"output_dir": os.path.relpath(out, root)})
    return record


@router.post("/jobs/training", status_code=201, response_model=JobRecord, tags=["jobs"])
async def create_training_job(req: Request):
    # The body is parsed BY HAND (from_legacy accepts two shapes, which no
    # single response-model annotation can express), so the two failures
    # FastAPI normally absorbs — unparsable JSON, a body that fails pydantic
    # validation — surfaced here as uncaught exceptions, i.e. 500s that told
    # the caller nothing. Re-raise both as RequestValidationError so the
    # app-wide handler answers exactly what a declared body would have: 422,
    # FastAPI's error-list `detail` shape, `request.validation` beside it.
    try:
        raw = await req.json()
    except json.JSONDecodeError as exc:
        raise RequestValidationError(
            [{"type": "json_invalid", "loc": ("body", exc.pos), "msg": "JSON decode error", "input": {}}]
        ) from exc
    if not isinstance(raw, dict):
        # from_legacy assumes a JSON object (it probes raw["config"]); a valid
        # non-object body ("[]", "5") is the same caller mistake as a failed
        # field, not a crash.
        raise RequestValidationError(
            [
                {
                    "type": "model_attributes_type",
                    "loc": ("body",),
                    "msg": "Input should be an object",
                    "input": raw,
                }
            ]
        )
    try:
        body = StartTrainingBody.from_legacy(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    cfg = body.config
    # A lan_node target without a node is unroutable — refuse with the same
    # 422 + code a malformed body would get, before any slower preflight.
    # (JobRegistry.start re-checks as belt-and-braces, mirroring the flavor
    # guard; that copy surfaces as a plain 400 for non-HTTP callers.)
    if body.target is not None and body.target.runner == "lan_node" and not body.target.node_instance_id:
        raise ApiError(
            status_code=422,
            detail="target.node_instance_id is required when target.runner is 'lan_node'",
            code=ErrorCode.REQUEST_VALIDATION,
        )
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
            # 400, matching this endpoint's other preflight rejections (the
            # resume-steps guard above and the ValueError->400 below). It is a
            # malformed request for this server's configuration, not a conflict
            # with some other state — nothing is holding the dataset; it simply
            # isn't here and can't be fetched.
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
        # Off the event loop: start()'s remote preflight makes real network
        # calls (hub status, the emptiness probe, lan_node peer verification),
        # each worth a full round-trip timeout on the slow/flaky connections
        # this app is designed for — run inline they'd stall every other
        # request for the duration.
        record = await asyncio.to_thread(job_registry.start, body.config, body.target)
    except (DatasetNotOnHubError, DatasetHubCopyEmptyError) as exc:
        # Remote run on a dataset the remote side can't fetch: local-only
        # (upload it first — the browser flow does so automatically, so this
        # fires for non-UI callers), or a Hub repo that exists but is empty
        # (an interrupted upload) with no local copy the runner could refill
        # it from. 409 both ways: a conflict with Hub state the caller has to
        # resolve before the run can proceed, not a malformed request.
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
    except NodeNotFoundError as exc:
        # The request named a node this install has never registered — a bad
        # reference in the request, so 400 (not the DELETE route's 404: there
        # is no /nodes/{id} resource being addressed here).
        raise ApiError(status_code=400, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        # Same status the node routes use for a peer that didn't answer.
        # Raised by the pre-record resolve (no record) or by the runner's
        # submission (record already finalised `failed` by the registry).
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc
    except ValueError as exc:
        # e.g. "flavor is required when runner is hf_cloud"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _wire_job_record(record)


class ImportModelRequest(BaseModel):
    source: str
    name: str | None = None


@router.post("/jobs/import", status_code=201, response_model=JobRecord, tags=["jobs"])
def import_model(body: ImportModelRequest):
    """Register an external model (local dir or HF repo) as a pseudo-job.

    Importing an already-registered source is idempotent: the registry
    returns the EXISTING record (id and display alias preserved), and the
    response carries `already_imported: true` with a 200 (not 201) so the
    frontend can say "already imported" instead of pretending a new entry
    was created. That branch is a JSONResponse and passes through the
    declared response_model untouched — the model documents the 201."""
    try:
        existing = job_registry.find_imported(body.source)
        record = job_registry.register_imported(body.source, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if existing is not None and existing.id == record.id:
        payload = _wire_job_record(record).model_dump(mode="json")
        payload["already_imported"] = True
        return JSONResponse(status_code=200, content=payload)
    return _wire_job_record(record)


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
def list_jobs(limit: int = 10):
    return {"jobs": [_wire_job_record(r) for r in job_registry.list(limit=limit)]}


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


# The label keys a submitted job may carry, newest first. `makermodslab_run` is
# what hf_cloud._RUN_LABEL writes now; the dotted key is read but never written
# — the Hub now rejects a "key=value" tag containing a dot, so it was renamed,
# and every job submitted before that still carries the old one.
_HUB_RUN_LABELS = ("makermodslab_run", "makermodslab.run")


def _hub_job_argv(ji) -> list:
    """A Hub job's argv as one flat list, POSITIONS PRESERVED.

    `arguments` is where the Hub splits argv for some submission paths; ours
    rides entirely in `command`. Both are scanned so neither shape is missed.

    Non-string tokens are deliberately NOT dropped. Removing them would close
    the gap they leave and make two tokens adjacent that never were, so
    `--policy.type` followed by a non-string would read the token after it as
    its value. They are left in place and rejected by `_argv_value` instead.
    """
    return [*(getattr(ji, "command", None) or []), *(getattr(ji, "arguments", None) or [])]


def _argv_value(argv: list, flag: str) -> str | None:
    """The value of `--flag value` or `--flag=value` in argv; None if absent.

    Both spellings occur in the same command line: build_training_command
    (train.py) emits the space form for most flags but the '=' form for
    `--config_path` / `--policy.pretrained_path`, where lerobot's own
    pre-parser only accepts '='. Empty and whitespace-only values read as
    absent — an empty flag value carries no more information than no flag.

    A space-form value that is itself option-shaped is rejected. A dangling
    `--policy.pretrained_path` immediately before `--resume true` would
    otherwise yield the base model "--resume", and with it a confident
    "Fine-tune" chip on a run that is nothing of the kind. No value we look for
    can legitimately begin with "--": they are repo ids, policy names and step
    counts.
    """
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if not isinstance(tok, str):
            continue
        value = None
        if tok == flag and i + 1 < len(argv):
            nxt = argv[i + 1]
            if isinstance(nxt, str) and not nxt.startswith("--"):
                value = nxt
        elif tok.startswith(prefix):
            value = tok[len(prefix) :]
        if value is not None and value.strip():
            return value.strip()
    return None


def _hub_job_run_name(ji) -> str | None:
    """The training run's name for a Hub job, or None when it can't be derived.

    Every cloud run launches on the same image, so the frontend's
    docker_image fallback titles ALL of them "huggingface/lerobot-gpu:latest".
    A run launched from this machine is spared that because a local JobRecord
    carries its name — one launched from a teammate's machine has no such
    record, and the name has to come off the Hub instead.

    Two sources, preferred first:
    1. The run label hf_cloud stamps at submission (either spelling).
    2. `--policy.repo_id` in the job's own argv. Every cloud run publishes to
       "<user>/<run slug>", so this recovers a name for jobs submitted before
       labelling existed — the whole existing backlog.
    """
    labels = getattr(ji, "labels", None) or {}
    for key in _HUB_RUN_LABELS:
        labelled = labels.get(key)
        if isinstance(labelled, str) and labelled.strip():
            return labelled.strip()

    repo_id = _argv_value(_hub_job_argv(ji), "--policy.repo_id")
    # The slug after the namespace is the run id the library titles by.
    return repo_id.rsplit("/", 1)[-1] if repo_id else None


def _hub_job_provenance(ji) -> dict:
    """What a Hub job started FROM, read off its own argv.

    Four kinds, so a card can say what a run IS at a glance:

      * `finetune`   — fresh optimizer from a base checkpoint the user chose.
      * `foundation` — fresh optimizer from the public foundation checkpoint a
                       VLA policy defaults to. NOT a fine-tune in the sense the
                       user means: JobRegistry.start pins
                       `policy_pretrained_path` to lerobot/smolvla_base (and the
                       pi0 family's equivalents) for ANY such run that names no
                       starting point, so treating a bare `--policy.pretrained_path`
                       as a fine-tune would mislabel every from-scratch VLA run.
      * `resume`     — a continuation of an earlier run.
      * `scratch`    — random weights.

    Read from argv rather than from Hub labels because a label cannot carry a
    repo id at all: the Hub validates label keys and values under its `tags`
    rules (alphanumeric, '-', '_', '=' — see _RUN_LABEL in runners/hf_cloud.py),
    and every repo id contains a '/'. argv also covers the whole existing
    backlog, and is what actually ran.

    `--config_path` is deliberately NOT consulted. On a cloud continuation it
    holds a CONTAINER path ("/tmp/makermodslab/train/checkpoints/.../train_config.json",
    set in runners/hf_cloud.py) that names nothing the user could recognize. The
    real source rides in the wrapper's own `--resume-from=<repo>@checkpoints/<step>`
    directive, which is part of the submitted command and so visible here.

    Absent facts are omitted rather than guessed: build_training_command's
    resume branch emits neither `--dataset.repo_id` nor `--policy.type` (lerobot
    reconstructs both from the checkpoint config), so a continuation simply has
    no value for them.
    """
    argv = _hub_job_argv(ji)
    out: dict[str, object] = {
        "kind": "scratch",
        "base_ref": None,
        "base_repo": None,
        "base_step": None,
        "base_job_id": None,
        "dataset_repo_id": _argv_value(argv, "--dataset.repo_id"),
        "policy_type": _argv_value(argv, "--policy.type"),
        "steps": _argv_value(argv, "--steps"),
    }

    pretrained = _argv_value(argv, "--policy.pretrained_path")
    resume_from = _argv_value(argv, "--resume-from")

    if resume_from:
        out["kind"] = "resume"
        out["base_ref"] = resume_from
    elif pretrained:
        # A run whose base is one of the public foundation checkpoints was
        # defaulted there, not pointed there by the user.
        out["kind"] = "foundation" if pretrained in _KNOWN_FOUNDATION_BASE_REPO_IDS else "finetune"
        out["base_ref"] = pretrained
    elif (_argv_value(argv, "--resume") or "").lower() == "true":
        # Submitted before the wrapper carried --resume-from; we know it
        # continued something but not what.
        out["kind"] = "resume"

    base_ref = out["base_ref"]
    if isinstance(base_ref, str):
        # hub_ref_* fall back to the whole ref when it isn't step-suffixed, so a
        # plain repo id passes through as its own repo with no step.
        repo = hub_ref_repo_id(base_ref)
        step = hub_ref_step_label(base_ref)
        if step == base_ref and "@checkpoints/" in base_ref:
            # hub_ref_* only split a DIGIT step dir, but hf_cloud can emit
            # "<repo>@checkpoints/last". Without this the whole raw ref would
            # land in base_repo and be rendered at the user (R2).
            repo, _, step = base_ref.partition("@checkpoints/")
        out["base_repo"] = repo
        out["base_step"] = step if step != base_ref else None
        # A "<user>/<job id>_checkpoints" base is a STAGING repo holding a local
        # run's uploaded checkpoint (checkpoints_staging_repo_id in jobs.py).
        # The job id inside it is the thing a person recognizes; the repo id is
        # plumbing. Recovered here, next to the rule that mints it, rather than
        # sniffed for in the frontend.
        slug = repo.rsplit("/", 1)[-1]
        if slug.endswith(CHECKPOINTS_STAGING_SUFFIX):
            out["base_job_id"] = slug[: -len(CHECKPOINTS_STAGING_SUFFIX)]

    return out


# The trainer flags worth reading back off a Hub job, and the JSON key each one
# becomes on the listing row. Deliberately a small allowlist rather than "parse
# everything": these four are what an untracked row renders (title, policy chip,
# dataset/steps on the card), and every one of them is a field a tracked
# JobRecord already carries, so a foreign run reads like a local one.
_HUB_JOB_TRAINER_FLAGS = ("policy.type", "dataset.repo_id", "steps", "policy.repo_id")


def _hub_job_trainer_args(ji) -> dict[str, str]:
    """The allowlisted `--flag value` pairs out of a Hub job's own argv.

    A cloud run's whole trainer invocation is stored on the job, so what a
    foreign run trains — its policy, dataset, and step target — is already in
    the listing response we fetch, with no extra Hub call. This reads it back.

    The command we submit is
    ``python -c <wrapper source> <spec> [directives] -- <trainer argv>``, so
    parsing starts after the first BARE ``--`` sentinel where there is one: the
    wrapper source is a single argv token, but the wrapper-side directives
    before the sentinel (e.g. ``--resume-from=...``) are not ours to read as
    trainer flags. `arguments` is where the Hub splits argv for some submission
    paths; ours rides entirely in `command`, so both are scanned.

    Both spellings are accepted (``--flag value`` and ``--flag=value``) because
    build_training_command emits each in different places. A flag repeated wins
    on its first occurrence; an unparsable or valueless flag is simply absent
    rather than raising — this decorates a listing and must never be able to
    500 it.
    """
    argv = [*(getattr(ji, "command", None) or []), *(getattr(ji, "arguments", None) or [])]
    argv = [tok for tok in argv if isinstance(tok, str)]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    out: dict[str, str] = {}
    for i, tok in enumerate(argv):
        for flag in _HUB_JOB_TRAINER_FLAGS:
            if flag in out:
                continue
            value = None
            if tok == f"--{flag}" and i + 1 < len(argv):
                value = argv[i + 1]
            elif tok.startswith(f"--{flag}="):
                value = tok.split("=", 1)[1]
            # A following token that is itself a flag means this one was passed
            # without a value; leave it absent rather than recording "--next".
            if value and not value.startswith("--"):
                out[flag] = value.strip()
    return out


def _hub_job_identity(ji) -> dict[str, Any]:
    """The run-identity half of a `/jobs/hub` row: what this job trains.

    Every value is best-effort and independently nullable — a RESUMED cloud run
    passes `--config_path` instead of `--policy.type`/`--dataset.repo_id`
    (build_training_command reconstructs those from the checkpoint), so a
    continuation legitimately reports a repo and steps with no policy or
    dataset. The frontend reserves the columns and renders a blank, which is the
    honest answer; inventing one from the run name would be a guess.
    """
    args = _hub_job_trainer_args(ji)
    try:
        total_steps: int | None = int(args["steps"])
    except (KeyError, ValueError):
        total_steps = None
    return {
        "policy_type": args.get("policy.type"),
        "dataset": args.get("dataset.repo_id"),
        "total_steps": total_steps,
        "hf_repo_id": args.get("policy.repo_id"),
    }


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


@router.get(
    "/jobs/hub",
    response_model=HubJobsResponse,
    response_model_exclude_unset=True,
    tags=["jobs"],
)
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
                "name": _hub_job_run_name(ji),
                "created_at": ji.created_at.isoformat() if ji.created_at else None,
                "docker_image": ji.docker_image,
                "space_id": ji.space_id,
                "flavor": ji.flavor,
                "status": ({"stage": ji.status.stage, "message": ji.status.message} if ji.status else None),
                "owner": ji.owner.name if ji.owner else None,
                "url": ji.url,
                # What the run trains (policy/dataset/steps/repo), read back off
                # the job's own argv so a foreign run's card reads like a local
                # one.
                **_hub_job_identity(ji),
                # What the run started FROM (kind + base checkpoint), parsed off
                # the same argv. Every cloud run ships the same image and flavor,
                # so without this a card launched from another machine has almost
                # nothing on it that distinguishes one run from the next. Spread
                # last: its `policy_type` is computed identically to identity's.
                **_hub_job_provenance(ji),
            }
            for ji in jobs
        ],
        "models": models,
    }

    with _hub_jobs_cache_lock:
        _hub_jobs_cache = {"at": time.monotonic(), "value": response}
    return response


@router.delete("/jobs/hub/models/{repo_id:path}", response_model=HubModelDeleteResponse, tags=["jobs"])
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

    # A QUEUED local run may be holding a deferred ref to exactly this repo:
    # a fine-tune's base checkpoint (queued_hub_ref) or a cloud parent's
    # checkpoint a continuation downloads at promotion (queued_resume_ref) —
    # both frequently under the user's own namespace (staging repos, their
    # own cloud runs' output repos). Deleting the repo now fails that run
    # hours later with a download error nobody could tie to this click. Same
    # refusal family as every other queued-dependency guard.
    queued_readers = sorted(
        r.id
        for r in job_registry.list_queue()
        if repo_id in {hub_ref_repo_id(ref) for ref in (r.queued_hub_ref, r.queued_resume_ref) if ref}
    )
    if queued_readers:
        waiting = ", ".join(repr(qid) for qid in queued_readers[:10])
        raise ApiError(
            status_code=409,
            detail=(
                f"Repo {repo_id!r} holds the checkpoint queued run(s) {waiting} will train "
                "from. Cancel them first, or wait for them to finish."
            ),
            code=ErrorCode.JOB_HAS_QUEUED_DEPENDENTS,
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
    # repo doesn't linger until the TTL expires. The models/skills listing has
    # its own Hub cache and its own last-good fallback, so it needs telling
    # separately: without this the deleted repo survives the TTL AND, worse,
    # persists as a retained "stale" row every time a later fan-out degrades.
    model_browser.forget_hub_repo(repo_id)
    invalidate_hub_jobs_cache()
    return {"status": "success", "repo_id": repo_id}


@router.post("/jobs/hub/jobs/{job_id}/dismiss", response_model=HubJobDismissResponse, tags=["jobs"])
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


# NEW surface, so it lives on v1_router (never the flat mount). It still MUST
# match before GET /jobs/{job_id}: that route's single {job_id} segment happily
# matches the literal "queue", answering this request with a 404 for a job of
# that name — the same reason /jobs/hub sits above it. The two live on
# different routers, so the ordering is enforced where the routers are
# included: v1_router joins the /api/v1 mount BEFORE the shared router does.
# (The POST twin, /jobs/queue/reorder, is not at risk — every POST
# /jobs/{job_id}/… route ends in a literal segment.)
@v1_router.get("/jobs/queue", response_model=JobQueueResponse, tags=["jobs"])
def list_job_queue():
    """The whole local training queue, in the order it will run.

    Separate from `GET /jobs` because that is a capped, newest-first PAGE of
    history and this is a complete list. Deriving the queue from that page was
    wrong twice over: a queued record carries its SUBMIT time, so queued runs
    crowd the top of the page and pushed the actually-running job off it, and
    past the page size the queue itself was truncated — which silently dropped
    the runs at the HEAD of the line and made every reorder a 409, since
    `reorder_queue` requires the whole list.
    """
    return {"jobs": [_wire_job_record(r) for r in job_registry.list_queue()]}


@router.get("/jobs/{job_id}", response_model=JobRecord, tags=["jobs"])
def get_job(job_id: str):
    try:
        return _wire_job_record(job_registry.get(job_id))
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse, tags=["jobs"])
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    return {"logs": logs}


@router.get("/jobs/{job_id}/log-file", response_model=JobLogsResponse, tags=["jobs"])
def get_job_log_file(job_id: str):
    """Return the entire on-disk log file for a job. Drains the live queue too
    so the next /logs poll returns only lines that arrived after this call."""
    try:
        logs = job_registry.read_persisted_logs(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    # Best-effort drain so the frontend doesn't double-display.
    with contextlib.suppress(JobNotFoundError):
        job_registry.drain_logs(job_id)
    return {"logs": logs}


@router.get("/jobs/{job_id}/metrics-history", response_model=JobMetricsHistoryResponse, tags=["jobs"])
def get_job_metrics_history(job_id: str):
    """Return the per-step loss/lr/grad-norm series reconstructed from the
    job's log.jsonl. Used to seed the monitoring charts so curves persist
    across page reloads, navigation, and MakerMods Lab restarts."""
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    return {"points": points}


@router.get("/jobs/{job_id}/checkpoints", response_model=JobCheckpointsResponse, tags=["jobs"])
def get_job_checkpoints(job_id: str, lineage: bool = False):
    """List the checkpoints saved for this job, ascending by step.

    ``lineage=true`` widens that to the whole resume chain — this run plus the
    runs it resumed. Opt-in rather than the default so existing callers keep
    their exact semantics; the skill picker asks for it because a chain is one
    model and splitting its steps across rows is what this fixes.
    """
    try:
        if lineage:
            return {"checkpoints": job_registry.list_chain_checkpoints(job_id)}
        return {"checkpoints": job_registry.list_checkpoints(job_id)}
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc


@router.get(
    "/jobs/{job_id}/checkpoints/{step}/policy-config",
    response_model=CheckpointPolicyConfigResponse,
    tags=["jobs"],
)
def get_checkpoint_policy_config(job_id: str, step: int):
    """Return the UX-relevant slice of a checkpoint's pretrained_model config:
    policy_type, image_features (per-camera height/width), requires_task, the
    flat state_dim/action_dim (6 = single arm, 12 = bimanual) the inference
    modal uses to flag a single-arm/bimanual mismatch, and trained_on_robot_type
    (the arm the checkpoint was trained on, for the fine-tune panel's cross-arm
    warning; null when it can't be established)."""
    try:
        return job_registry.get_policy_config_summary(job_id, step)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/checkpoints/{step}/download")
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


@router.post("/jobs/{job_id}/rename", response_model=JobRecord, tags=["jobs"])
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
        return _wire_job_record(job_registry.rename(job_id, body.new_name))
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ReorderQueueRequest(BaseModel):
    # The WHOLE queue, first to run first. A partial list is refused rather
    # than merged — see JobRegistry.reorder_queue.
    #
    # Bounded because the queue is: it holds runs a user submitted by hand, one
    # at a time, and a machine with a single local training slot will not have
    # thousands waiting. Unbounded, a 20k-id body was validated INSIDE the
    # registry lock (freezing every /jobs* request behind the set math) and came
    # back as a 360 KB error detail echoing every bad id. 422 here costs neither.
    # Each ID is bounded too: capping only the count left 512 × multi-KB
    # strings building megabyte 400s out of echoed input (generated ids top out
    # around 150 chars — see jobs._NAMED_ID_MAX_CHARS, the render-side backstop).
    job_ids: list[Annotated[str, StringConstraints(max_length=200)]] = Field(max_length=512)


# NEW surface → v1_router (see the /jobs/queue GET above). Declared before
# /jobs/{job_id}/... so the intent is readable together with stop; FastAPI
# matches this path fine either way, since every {job_id} route ends in a
# literal segment ("rename", "stop", …) that "queue" isn't.
@v1_router.post("/jobs/queue/reorder", response_model=JobQueueResponse, tags=["jobs"])
def reorder_job_queue(body: ReorderQueueRequest):
    """Set the order of the local training queue.

    Local runs are one-at-a-time, so a second Start enqueues rather than
    failing; this is how the user changes their mind about what goes next.
    Only queued jobs can be reordered — the run already on the GPU is not in
    the list, and a job that started while the drag was in flight makes the
    request stale (409).
    """
    try:
        return {"jobs": [_wire_job_record(r) for r in job_registry.reorder_queue(body.job_ids)]}
    except ValueError as exc:
        # The request itself is wrong — an id that names no run at all, or one
        # listed twice. 400, not the 409 below: retrying it unchanged can never
        # succeed, and the detail names the offending ids so a non-UI caller can
        # fix them. An id that names a real run which has LEFT the queue is not
        # this case: that is the race below, and it retries successfully.
        raise ApiError(status_code=400, detail=str(exc), code=ErrorCode.REQUEST_VALIDATION) from exc
    except QueueChangedError as exc:
        # A well-formed list that lost its race. Retrying after a refetch is
        # exactly the right advice here, which is why the code appears only
        # here: job.queue_stale is the one refusal in this family a plain
        # refetch-and-retry clears.
        raise ApiError(
            status_code=409,
            detail=(
                "The training queue changed while you were reordering it — "
                "a job started, finished, or was cancelled. The list has been "
                "refreshed; try again."
            ),
            code=ErrorCode.JOB_QUEUE_STALE,
        ) from exc


@router.post(
    "/jobs/{job_id}/stop",
    response_model=JobRecord,
    tags=["jobs"],
    # A repeated ?expect_state= must not silently resolve to one of its two
    # contradictory values — see _refuse_repeated_query_keys.
    dependencies=[Depends(_refuse_repeated_query_keys)],
)
def stop_job(job_id: str, expect_state: JobState | None = None):
    """Stop a running job, or cancel a queued one.

    `expect_state` is optional and is the caller's precondition: pass the state
    the UI was showing when it drew the button. Cancel and kill are the same
    request here, so a Cancel drawn against a stale queue would otherwise
    SIGTERM a run the watchdog promoted in the meantime.

    Typed as `JobState`, not `str`, to match `JobRegistry.stop`: an unknown value
    used to reach the comparison, fail it, and come back as a 409 saying the job
    "changed while you were looking at it" — reporting a client's typo as a race,
    which no retry can ever clear. It is now a 422, and `/openapi.json`
    advertises the real member set instead of "any string".
    """
    try:
        return _wire_job_record(job_registry.stop(job_id, expect_state=expect_state))
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404, detail=f"Job {job_id!r} not found", code=ErrorCode.JOB_NOT_FOUND
        ) from exc
    except JobStateChangedError as exc:
        raise ApiError(
            status_code=409,
            detail=(
                f"Job {job_id!r} is {exc.actual!r}, not {exc.expected!r} — it changed while "
                "you were looking at it. Refresh and decide again."
            ),
            code=ErrorCode.JOB_STATE_CHANGED,
        ) from exc
    except JobNotRunningError as exc:
        raise ApiError(
            status_code=409,
            detail=f"Job {job_id!r} is neither running nor queued",
            code=ErrorCode.JOB_NOT_RUNNING,
        ) from exc
    # Cancelling a QUEUED run removes its record, so it carries the same two
    # refusals as DELETE. Stopping a running run does not — it leaves a record
    # behind — so these can only fire on the cancel path.
    except JobHasChildrenError as exc:
        continued_by = ", ".join(repr(cid) for cid in exc.child_ids)
        raise ApiError(
            status_code=409,
            detail=(
                f"Job {job_id!r} was continued by {continued_by}, which would be left "
                "pointing at a cancelled run. Cancel the continuation(s) first."
            ),
            code=ErrorCode.JOB_HAS_CHILDREN,
        ) from exc
    except JobSourceOfQueuedRunError as exc:
        waiting = ", ".join(repr(qid) for qid in exc.queued_ids)
        raise ApiError(
            status_code=409,
            detail=(
                f"Job {job_id!r} holds the checkpoint queued run(s) {waiting} will train "
                "from. Cancel those first."
            ),
            code=ErrorCode.JOB_HAS_QUEUED_DEPENDENTS,
        ) from exc
    except JobRemovalFailedError as exc:
        # 500, not 409: nothing about the request was wrong. Say plainly that
        # the run is untouched, because the alternative reading — "cancel
        # half-worked" — is what would make a user walk away from a run that is
        # still going to train.
        logger.exception("Could not cancel job %s", job_id)
        raise ApiError(
            status_code=500,
            # `strerror` only — see the delete twin below. The full OSError
            # carries the job directory's absolute path, and this body is
            # returned to the caller.
            detail=(
                f"Could not cancel job {job_id!r}: {exc.reason.strerror}. The run is still "
                "queued and will still start when the slot frees — nothing was removed, so "
                "it is safe to try again."
            ),
            code=ErrorCode.JOB_REMOVAL_FAILED,
        ) from exc


# 204 No Content — there is no body for a response_model to describe, so the
# route sits in RESPONSE_MODEL_EXEMPT (tests/test_api_contract.py) instead.
@router.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
def delete_job(job_id: str):
    try:
        record = job_registry.get(job_id)
        job_registry.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first") from exc
    except JobPublishInProgressError as exc:
        # The background Hub publish is reading this run's checkpoint dirs
        # right now; deleting them mid-upload kills the publish with an
        # opaque error. Same refusal POST /models/delete gives.
        raise ApiError(
            status_code=409,
            detail=(
                f"Job {job_id!r} is being published to the Hub — wait for the "
                "publish to finish before deleting it."
            ),
            code=ErrorCode.JOB_PUBLISH_IN_PROGRESS,
        ) from exc
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
    except JobSourceOfQueuedRunError as exc:
        # Same shape as the mid-chain refusal above, for the dependency
        # build_child_index does not model: a queued fine-tune froze this run's
        # checkpoint PATH at submit time and reads it when the slot frees, so
        # deleting the directory now fails that run hours from now with a
        # not-found traceback the user could not connect to this click.
        waiting = ", ".join(repr(qid) for qid in exc.queued_ids)
        raise ApiError(
            status_code=409,
            detail=(
                f"Job {job_id!r} holds the checkpoint queued run(s) {waiting} will train "
                "from. Cancel them first, or wait for them to finish."
            ),
            code=ErrorCode.JOB_HAS_QUEUED_DEPENDENTS,
        ) from exc
    except JobRemovalFailedError as exc:
        # 500, not 409: nothing about the request was wrong. Say that the run is
        # untouched, because the alternative reading — "delete half-worked" — is
        # what would leave a user surprised to see it again after a restart.
        # Where it is untouched depends on what it was: `delete` refuses only a
        # RUNNING run, so a queued one reaches here, and telling that user it is
        # "still in your history" describes the wrong place — it is still in the
        # queue and still going to train, which is the part they need to act on.
        logger.exception("Could not delete job %s", job_id)
        still = (
            "still queued and will still start when the slot frees"
            if record.state == "queued"
            else "untouched and still in your history"
        )
        raise ApiError(
            status_code=500,
            # `strerror` only: the full OSError carries the absolute path of the
            # job directory, and this body goes to whoever made the request —
            # including anyone on the LAN under `--lan`. The path is in the log.
            detail=(
                f"Could not delete job {job_id!r}: {exc.reason.strerror}. The run is "
                f"{still} — nothing was removed, so it is safe to try again."
            ),
            code=ErrorCode.JOB_REMOVAL_FAILED,
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


@router.get("/jobs/runners/hardware", response_model=RunnersHardwareResponse, tags=["jobs"])
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


@router.get("/system/training-extra", response_model=ExtraStatus, tags=["system"])
def get_training_extra():
    """Return whether the LeRobot training extra (accelerate) is importable."""
    return handle_get_training_extra()


@router.post("/system/training-extra/install", response_model=InstallStartResponse, tags=["system"])
def install_training_extra():
    """Spawn `pip install accelerate` as a background subprocess. No-op if already running."""
    return handle_install_training_extra()


@router.get("/system/training-extra/install-status", response_model=InstallStatusResponse, tags=["system"])
def install_training_extra_status():
    """Return current install state plus any pending log lines (drained on read)."""
    return handle_install_training_extra_status()


@router.get("/system/wandb-extra", response_model=ExtraStatus, tags=["system"])
def get_wandb_extra():
    """Return whether the `wandb` package is importable in this MakerMods Lab process."""
    return handle_get_wandb_extra()


@router.post("/system/wandb-extra/install", response_model=InstallStartResponse, tags=["system"])
def install_wandb_extra():
    """Spawn `pip install wandb` as a background subprocess. No-op if already running."""
    return handle_install_wandb_extra()


@router.get("/system/wandb-extra/install-status", response_model=InstallStatusResponse, tags=["system"])
def install_wandb_extra_status():
    """Return current wandb install state plus any pending log lines (drained on read)."""
    return handle_install_wandb_extra_status()


@router.get("/system/policy-extra/{policy_type}", response_model=PolicyExtraStatus, tags=["system"])
def get_policy_extra(policy_type: str):
    """Whether the optional LeRobot extra a policy needs (e.g. transformers for
    smolvla/pi0, diffusers for diffusion) is importable. Core policies report available."""
    return handle_get_policy_extra(policy_type)


@router.post(
    "/system/policy-extra/{policy_type}/install", response_model=InstallStartResponse, tags=["system"]
)
def install_policy_extra(policy_type: str):
    """Spawn `pip install lerobot[<extra>]` for the policy's extra in the background."""
    return handle_install_policy_extra(policy_type)


@router.get(
    "/system/policy-extra/{policy_type}/install-status",
    response_model=InstallStatusResponse,
    tags=["system"],
)
def install_policy_extra_status(policy_type: str):
    """Return the policy extra's install state plus any pending log lines (drained on read)."""
    return handle_install_policy_extra_status(policy_type)


# The busy matrix's discriminants, for the restart refusal: the same
# robot.busy.<feature> code the holder's own start-refusals use, so a client
# learns WHAT holds the machine from the code alone.
_HOLDER_BUSY_CODES: dict[str, ErrorCode] = {
    "recording": ErrorCode.ROBOT_BUSY_RECORDING,
    "teleoperation": ErrorCode.ROBOT_BUSY_TELEOPERATION,
    "inference": ErrorCode.ROBOT_BUSY_INFERENCE,
    "replay": ErrorCode.ROBOT_BUSY_REPLAY,
    "calibration": ErrorCode.ROBOT_BUSY_CALIBRATION,
    "auto_calibration": ErrorCode.ROBOT_BUSY_AUTO_CALIBRATION,
    "wiggle": ErrorCode.ROBOT_BUSY_WIGGLE,
}


@v1_router.post("/system/restart", response_model=RestartResponse, tags=["system"])
def restart_server():
    """Re-exec this server process in place (same argv/env/PID), so a remote
    operator — the node-proxy POST /api/v1/nodes/{id}/restart — can bounce a
    headless station without a shell on it. Answers FIRST, re-execs after a
    short grace delay so this response reaches the client.

    Refusals, all 409: a live robot flow (robot.busy.<feature> — killing the
    server mid-flow drops the hardware threads with it), a running or queued
    training run (robot.busy.training — the loader retires both on startup),
    and a process that cannot safely re-exec (system.restart_unsupported: a
    dev reload worker, a non-entry-point launch, or Windows)."""
    holder = held_by()
    if holder is not None:
        raise ApiError(
            status_code=409,
            detail=f"Cannot restart while {holder} is active — stop it first.",
            code=_HOLDER_BUSY_CODES.get(holder, ErrorCode.SESSION_HELD),
        )
    running = training_is_active()
    if running is not None:
        raise ApiError(
            status_code=409,
            detail=f"Cannot restart while a local training run ({running}) is active — stop it first.",
            code=ErrorCode.ROBOT_BUSY_TRAINING,
        )
    queued = job_registry.list_queue()
    if queued:
        raise ApiError(
            status_code=409,
            detail=(
                f"Cannot restart with {len(queued)} queued training run(s) — "
                "the restart would retire the queue. Cancel them first."
            ),
            code=ErrorCode.ROBOT_BUSY_TRAINING,
        )
    installing = install_in_progress()
    if installing is not None:
        raise ApiError(
            status_code=409,
            detail=f"Cannot restart while '{installing}' is installing — wait for it to finish.",
            code=ErrorCode.SYSTEM_INSTALL_IN_PROGRESS,
        )
    supported, why = restart_supported()
    if not supported:
        raise ApiError(status_code=409, detail=why, code=ErrorCode.SYSTEM_RESTART_UNSUPPORTED)
    schedule_restart()
    return {"restarting": True, "message": "Restarting — the server will be back in a few seconds."}


@router.get("/system/update-check", response_model=UpdateStatus, tags=["system"])
def update_check():
    """Report whether a newer MakerMods Lab commit exists on GitHub (cached, silent on failure)."""
    return handle_update_check()


@router.post("/system/update", response_model=UpdateResult, tags=["system"])
def run_update():
    """Run the pip upgrade in-process; the user must restart MakerMods Lab afterwards."""
    return handle_run_update()


# Replay is rendered by the embedded lerobot/visualize_dataset Space; no backend routes needed.


# ============================================================================
# Calibration endpoints
@router.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start calibration process.

    Legacy/external entry point: it takes a device + port directly rather than
    a robot name, so there is no record to read an arm type from and it always
    runs the SO-101 range sweep. The Maker arm's zero-pose flow is reached
    through the sessions surface (POST /api/v1/sessions, kind "calibration"),
    which resolves the arm type from the robot record.
    """
    return calibration_manager.start_calibration(request)


@router.post("/stop-calibration")
def stop_calibration():
    """Stop calibration process.

    Stops whichever calibration flow is live. Stopping is never owner-gated
    and the two managers are mutually exclusive, so trying the zero-pose flow
    first and falling through is unambiguous.
    """
    if zero_calibration_is_active():
        return zero_calibration_manager.stop()
    return calibration_manager.stop_calibration_process()


@router.get("/calibration-status")
def calibration_status():
    """Get current calibration status, from whichever flow is live.

    The two status dataclasses are field-compatible where they overlap, so one
    client shape reads both. `awaiting_pose` is present only on the zero-pose
    flow and defaults to False for the SO-101 sweep, which is what lets the
    frontend switch panels on it.
    """
    from dataclasses import asdict

    if zero_calibration_is_active():
        return asdict(zero_calibration_manager.get_status())
    payload = asdict(calibration_manager.get_status())
    payload.setdefault("awaiting_pose", False)
    return payload


@router.post("/complete-calibration-step")
def complete_calibration_step():
    """Complete the current calibration step (either flow)."""
    if zero_calibration_is_active():
        return zero_calibration_manager.complete_step()
    return calibration_manager.complete_step()


# --- Auto-calibration (drives the arm under torque; runs the vendored script) ---


@router.post("/start-auto-calibration")
def start_auto_calibration(request: AutoCalibrationRequest):
    """Start auto-calibration as a subprocess. The arm moves on its own."""
    return auto_calibration_manager.start(request)


@router.post("/stop-auto-calibration")
def stop_auto_calibration():
    """Stop a running auto-calibration."""
    return auto_calibration_manager.stop()


@router.get("/auto-calibration-status")
def auto_calibration_status():
    """Current auto-calibration state + streamed log lines."""
    return auto_calibration_manager.get_status()


@router.post("/start-auto-calibration-batch")
def start_auto_calibration_batch(request: AutoCalibrationBatchRequest):
    """Auto-calibrate a user-selected subset of arms CONCURRENTLY. Each arm runs
    its own subprocess on its own serial port with an independent outcome
    (partial success). Validated up front (1-4 arms, distinct ports, distinct
    same-side names, name-taken pre-check) before any hardware is touched."""
    return auto_calibration_batch_manager.start(request)


@router.post("/stop-auto-calibration-batch")
def stop_auto_calibration_batch():
    """Stop ALL running arms of a batch auto-calibration, releasing each arm's
    torque independently."""
    return auto_calibration_batch_manager.stop()


@router.get("/auto-calibration-batch-status")
def auto_calibration_batch_status():
    """Per-arm status + logs and overall counts for a batch auto-calibration."""
    return auto_calibration_batch_manager.get_status()


@router.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str, arm_type: str = "so101"):
    """Get all calibration config files for a specific device type"""
    try:
        config_path = calibration_dir_for_device(device_type, arm_type)
        if config_path is None:
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


@router.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str, arm_type: str = "so101"):
    """Delete a calibration config file"""
    try:
        config_path = calibration_dir_for_device(device_type, arm_type)
        if config_path is None:
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
        unassigned = clear_config_references(device_type, config_name, arm_type)
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


@router.get("/calibration-configs/{device_type}/{config_name}/download")
def download_calibration_config(device_type: str, config_name: str, arm_type: str = "so101"):
    """
    Download one arm's calibration as a raw lerobot calibration JSON file.

    The file IS lerobot's own calibration file (no MakerMods Lab wrapper), so it's
    drop-in: shareable, hand-copyable, and re-importable anywhere. The arm's
    side/name are supplied by the caller on re-import, not stored in the file.
    """
    config_path = calibration_dir_for_device(device_type, arm_type)
    if config_path is None:
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


@router.post("/calibration-configs/{device_type}/upload")
def upload_calibration_config(device_type: str, body: dict, arm_type: str = "so101"):
    """
    Import a calibration into a side's config dir. Body: {"name": "...",
    "data": {<raw lerobot calibration>}}. The data is shape-validated; an
    existing name is never overwritten (409 → caller renames).
    """
    name = (body or {}).get("name", "")
    data = (body or {}).get("data")
    if not isinstance(name, str):
        return JSONResponse(status_code=400, content={"success": False, "message": "name must be a string"})

    ok, reason, saved = save_imported_calibration(device_type, name, data, arm_type)
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


@router.post("/calibration-configs/{device_type}/{config_name}/rename")
def rename_calibration_config_endpoint(
    device_type: str, config_name: str, body: dict, arm_type: str = "so101"
):
    """
    Rename a calibration config file. Body: {"new_name": "..."}. Never
    overwrites; robot records referencing the old name are repointed.
    """
    new_name = (body or {}).get("new_name", "")
    if not isinstance(new_name, str):
        return JSONResponse(
            status_code=400, content={"success": False, "message": "new_name must be a string"}
        )

    ok, reason = rename_calibration_config(device_type, config_name, new_name, arm_type)
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
    # Which arm type's library to open — "so101" or "maker". The two live in
    # separate directories (so_leader/so_follower vs
    # rebot_102_leader/maker_follower).
    arm_type: str = "so101"


@router.post("/open-calibration-folder")
def open_calibration_folder(request: OpenCalibrationFolderRequest):
    """Open a side's calibration folder in the OS file browser (Finder/Explorer/
    xdg-open). LOCAL, non-network action — spawns a GUI on the host machine only.
    The dir is created if missing so a fresh install opens an empty folder rather
    than failing. An unknown device_type is rejected with 400.
    """
    path = calibration_dir_for_device(request.device_type, request.arm_type)
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


# exclude_none: success carries `ports`, failure carries `message` — the other
# key is absent, never null, so None-exclusion reproduces each branch exactly.
@router.get(
    "/available-ports",
    response_model=AvailablePortsResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
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


@router.post("/wiggle")
async def wiggle(request: WiggleRequest):
    """Wiggle the gripper on a port so the user can see which arm it is."""
    return await wiggle_gripper(request.port)


class IdentifyArmRequest(BaseModel):
    # Candidate ports to watch; empty/omitted = all detected arm ports.
    ports: list[str] | None = None


class MakerProbePortsRequest(BaseModel):
    # Candidate ports to probe; empty/omitted = every detected serial port.
    ports: list[str] | None = None
    # Which CAN family the follower probe should speak: "maker" (RobStride) or
    # "metal" (Damiao). The leader probe is identical either way (both
    # families use the Star Arm 102). Defaults to maker so a client that
    # predates the Metal arm is unchanged.
    arm_type: Literal["maker", "metal"] = "maker"


class MakerIdentifyArmRequest(BaseModel):
    # "robot" (the CAN follower) or "teleop" (the UART leader). Unlike the
    # SO-101, the two halves of a Maker rig need different bus drivers, so the
    # caller must say which side it is asking about.
    device_type: str
    ports: list[str] | None = None
    # See MakerProbePortsRequest. For "metal" the follower side is refused
    # (opening a Damiao bus energizes it mid-gesture); the leader side works.
    arm_type: Literal["maker", "metal"] = "maker"


@v1_router.post("/maker/probe-ports", response_model=MakerProbePortsResponse, tags=["system"])
async def probe_maker_arm_ports(request: MakerProbePortsRequest):
    """Find which ports carry a Maker follower and which carry its Star 102 leader.

    A CAN rig's two halves speak different protocols on different adapters
    (RobStride/Damiao over CAN vs FashionStar over UART), so unlike the SO-101
    this needs NO gesture from the user — asking each port which protocol
    answers is enough. The maker probe is strictly read-only; the METAL
    follower probe briefly enables the gravity-neutral base joint and disables
    it again (the Damiao handshake is the enable command — see
    maker_ports._open_metal_follower_bus).
    """
    return await probe_maker_ports(request.ports, request.arm_type)


# exclude_none: success carries `port`, failure omits it entirely (never null),
# so None-exclusion reproduces each branch exactly — same contract as
# /identify-arm above.
@v1_router.post(
    "/maker/identify-arm",
    response_model=MakerIdentifyArmResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
async def identify_maker_arm(request: MakerIdentifyArmRequest):
    """Tell one Maker arm from its twin by watching for a hand gesture.

    Only needed for a BIMANUAL Maker robot: both arms ship with identical CAN
    and servo ids, so probing alone cannot say which is left and which is
    right. The user swings one arm's base and we report the port that saw it.
    Read-only — no motor writes.
    """
    return await identify_maker_arm_by_motion(request.device_type, request.ports, request.arm_type)


@v1_router.post("/arms/release-torque", response_model=ReleaseCanTorqueResponse, tags=["system"])
async def release_can_torque(request: ReleaseCanTorqueRequest):
    """De-energize a CAN follower after a crash left it holding torque.

    A SIGKILL or power loss leaves Damiao motors rigid at their last command
    with no session and no device object to clean up through. This reopens
    the named bus WITHOUT the energizing handshake, broadcasts the disable,
    and closes. Refused (409 session.held) while any live session holds the
    hardware; not a session itself — no lease, no session events (see
    can_recovery.py).
    """
    return await asyncio.to_thread(handle_release_can_torque, request)


@router.post("/identify-arm")
async def identify_arm(request: IdentifyArmRequest):
    """The inverse of /wiggle: the user swings an arm's base (shoulder pan) by
    hand and we report which port saw the motion. Read-only — no motor writes."""
    return await identify_arm_by_motion(request.ports)


# exclude_none: success carries `voltage`, failure carries `message` — never
# both, never null (see read_supply_voltage), so None-exclusion is faithful.
@router.get(
    "/supply-voltage",
    response_model=SupplyVoltageResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
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


# exclude_none: `message` exists only on the error branch and `unique_id` only
# on macOS entries — both absent (never null) otherwise, so None-exclusion
# reproduces the platform-specific bodies exactly.
@router.get(
    "/available-cameras",
    response_model=AvailableCamerasResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
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


@router.get("/camera-preview/{index}")
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
    against this process's device snapshot, which diverges from the
    fresh-subprocess enumeration after a replug — without the re-anchor the
    stream can silently show a different camera (see makermodslab/camera_identity.py).
    The identity is also what the preview registry shares captures by, so the
    resolver hands back both: the index to open and the key to file it under.
    Keying by the bare index aliased two different cameras onto one handle
    whenever the device set renumbered mid-session.

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
    identified = identify_cv2_index(unique_id, index)
    if identified is None:
        raise HTTPException(
            status_code=503,
            # Two causes reach here and the server cannot tell them apart
            # without plumbing that would not change the remedy: the camera
            # was attached after startup and this process never saw it, or
            # macOS is denying MakerMods Lab camera access, in which case the
            # enumeration truthfully reports nothing and cv2 could not open
            # the device either. Naming both beats asserting the first and
            # sending the user to a restart that cannot help.
            detail="Camera not visible to the server — either it was plugged in after "
            "MakerMods Lab started, or macOS is not granting MakerMods Lab camera access "
            "(System Settings → Privacy & Security → Camera). Grant access if it is missing, "
            "then restart MakerMods Lab.",
        )
    resolved, key = identified
    try:
        stream = camera_preview_manager.open_stream(resolved, key)
    except CameraOpenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


RobotSideLiteral = Literal["leader", "follower"]


@router.get("/robot-port/{robot_type}", response_model=RobotPortResponse, tags=["system"])
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
    follower-only activities (inference, replay, hosting) aren't blocked by a
    leader arm they never touch; `leader_ready` is the mirror for remote
    teleoperation, which drives a STATION's follower with this node's leader.
    The record's `arms` layout says which of these the UI should even show."""
    return {
        **record,
        "is_clean": is_robot_record_clean(record),
        "follower_ready": is_robot_record_clean(record, arms="follower"),
        "leader_ready": is_robot_record_clean(record, arms="leader"),
    }


@router.get("/robots")
def get_robots():
    """List all saved robot records."""
    try:
        records = [_record_with_clean(r) for r in list_robot_records()]
        return {"status": "success", "robots": records}
    except Exception as e:
        logger.error(f"Error listing robots: {e}")
        return {"status": "error", "message": str(e), "robots": []}


@router.get("/robots/{name}")
def get_robot(name: str):
    """Get a single robot record by name."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    record = get_robot_record(name)
    if record is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})
    return {"status": "success", "robot": _record_with_clean(record)}


@router.post("/robots/{name}")
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


@router.post("/robots/{name}/rename")
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


@router.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


@app.on_event("startup")
def migrate_state_home():
    """Move pre-split state from lerobot's cache into MAKERMODSLAB_HOME.

    Registered FIRST so it runs before any other startup work; every reader
    of the moved entries is lazy (robot records, ports, the node registry's
    saved peers, the instance id), so startup is early enough. Skipped under
    a MAKERMODSLAB_HOME override — see utils/config.HOME_IS_OVERRIDDEN.
    """
    if HOME_IS_OVERRIDDEN:
        return
    migrate_legacy_state()


@app.on_event("startup")
def startup_event():
    """One-time startup diagnostics surfaced in the server terminal."""
    warn_if_cuda_mismatch()


@app.on_event("startup")
def start_station_mode():
    """`makermodslab --host <robot>`: keep that robot hosted for remote
    teleoperation (remote_host.start_station_mode) — parked from startup,
    re-armed after any local session, no browser required."""
    if os.environ.get(remote_host.STATION_ENV) == "1":
        remote_host.start_station_mode(
            os.environ.get(remote_host.STATION_ROBOT_ENV, "").strip() or None, manager
        )


# Strong reference so the loop's task set can't drop the pump mid-flight.
_avf_pump_task: asyncio.Task | None = None


@app.on_event("startup")
async def start_avfoundation_pump():
    """Keep the in-process camera list live on macOS (hotplug/replug).

    Async handler on purpose: it runs inside the event loop (main thread),
    which is where the pump must live — AVFoundation's device-cache updates
    only drain on the main thread's runloop (see camera_identity). A sync
    startup handler would run in the threadpool and couldn't schedule it.
    No-op off macOS.
    """
    global _avf_pump_task
    _avf_pump_task = asyncio.create_task(pump_avfoundation_runloop())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when FastAPI shuts down"""
    logger.info("🔄 FastAPI shutting down, cleaning up...")

    # FIRST, before anything below takes its (bounded but real) time: stop the
    # job watchdog, so the local training queue cannot promote a run while we
    # are shutting down.
    #
    # `_drain_queue` runs every second from a thread uvicorn does not manage, so
    # without this a queued run could still be promoted after uvicorn has
    # stopped accepting the HTTP requests that are the only other way to start
    # one. The run already training is then ended deliberately further down (see
    # `stop_local_for_shutdown`), because its stdout pipe dies with this process
    # regardless — the exit-status file + TailingJobRunner still cover a worker
    # reload that this same process survives.
    job_registry.shutdown()

    # Stop the AVFoundation pump first so its next tick can't interleave with
    # shutdown (and so --reload restarts don't log a destroyed-pending-task).
    if _avf_pump_task is not None:
        _avf_pump_task.cancel()

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
        asyncio.to_thread(stop_replay_and_wait),
        return_exceptions=True,
    )
    labels = (
        "teleoperation",
        "recording",
        "auto-calibration",
        "auto-calibration batch",
        "inference",
        "replay",
    )
    for label, result in zip(labels, results, strict=True):
        if isinstance(result, Exception):
            logger.exception(f"Failed to stop {label} during shutdown", exc_info=result)

    # Local training is not on the list above because it drives no hardware —
    # but it does die with this process regardless of what we do here. The
    # trainer's stdout is a pipe this process owns, so the moment we exit its
    # next write raises BrokenPipeError and it exits 1, with the traceback
    # going into the closed pipe: no log line, and a history entry reading
    # "Subprocess exited with code 1" that looks exactly like a broken model.
    # (`start_new_session=True` escapes the process group, not the pipe.) So we
    # end it deliberately instead, and file it as `interrupted` with a reason.
    # Cloud runs are untouched — they keep going on HF's GPUs.
    try:
        stopped = await asyncio.to_thread(job_registry.stop_local_for_shutdown)
        if stopped:
            logger.info(
                "Stopped %d local training job(s) on shutdown: %s",
                len(stopped),
                ", ".join(stopped),
            )
    except Exception:
        logger.exception("Failed to stop local training jobs during shutdown")

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


def _v1_operation_id(route: APIRoute) -> str:
    """v1 operation ids are the bare handler names — the method names an SDK
    generator emits — so handlers must be uniquely named (contract-tested)."""
    return route.name


# Flat mount first (default operation ids), then /api/v1 with clean ids.
# Both precede the SPA mount below: starlette matches in registration order,
# so anything registered after the "/" mount would be unreachable.
app.include_router(router)
# v1-only surface: included ONCE, versioned — never on the flat mount. It
# joins /api/v1 BEFORE the shared router does, and that order is load-bearing:
# starlette matches in registration order, and the shared router's
# GET /jobs/{job_id} would otherwise swallow the v1-only GET /jobs/queue
# (a single {job_id} segment happily matches the literal "queue").
app.include_router(v1_router, prefix="/api/v1", generate_unique_id_function=_v1_operation_id)
app.include_router(router, prefix="/api/v1", generate_unique_id_function=_v1_operation_id)


def ui_enabled() -> bool:
    """Whether this process serves the built frontend.

    MAKERMODSLAB_NO_UI=1 (the --no-ui flag) turns a node into a pure API
    server — same binary, headless role."""
    return FRONTEND_DIST.exists() and os.environ.get("MAKERMODSLAB_NO_UI") != "1"


# Serve the built frontend at /. Must be mounted last so API routes win.
if ui_enabled():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; run `npm run build` in frontend/ or use `makermodslab --dev`."
    )
