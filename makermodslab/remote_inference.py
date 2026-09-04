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

"""Remote inference: a policy on a remote GPU drives the local follower.

The eighth robot-driving feature and the eighth `robot.busy.*` discriminant
(see CLAUDE.md's "State model & mutual exclusion"). Shaped like `rollout.py` —
one global session, a module-level active flag under a per-feature lock, a
background startup worker, a subprocess with a stdout pump — and deliberately
NOT a variant of it: the two share only the middle of the ladder (`_prepare_robot`,
`_session_cameras`, `_arm_count_mismatch`), which is why those are imported from
`rollout` rather than reimplemented here.

What it owns, and what it does not
----------------------------------
This session owns the ROBOT side only: it spawns one of the two chunk players
(`_CHILD_MODULES`, picked by the request's `engine`), which opens the follower
bus, joins a LiveKit room and plays the action chunks a remote operator sends
back. The GPU side (`modal run makermodslab/drtc/modal_policy.py`, or
`modal_policy_rtc.py` for the rtc engine) is started by a human in another
terminal; the session VERIFIES it is there before it energizes anything (see
:func:`_probe_room`) and promises exactly what it can keep — "I own the arm; I
checked the other half first". Lifecycle option A of docs/drtc/SLICE3.md.

The SFU is no longer a third party a human runs. Since S3.6 the Lab hosts it
itself (`makermodslab --sfu`, `makermodslab/sfu.py`): when it is up, this
module takes the url, the room and the CHILD'S TOKEN from it in-process and
`livekit.env` is not read at all. That file is the LiveKit Cloud fallback for a
station with no local SFU, and it is now the ONLY credential file — the shell
scripts, their `livekit.local.env` override and the cwd dotenv rungs are all
retired. `_transport_source` names which of the two is in force.

The two engines are the same session in every respect this module can see: the
same preflight ladder, the same protocol, the same watchdogs, the same stop.
They differ only in which module is spawned and in one flag — see
`_robot_sync_args`. What the backend CANNOT do is check the engine against the
policy: it never reads the checkpoint, so it cannot tell a flow policy from an
ACT one. The UI is the gate (`deployGuards.ts`), and a caller driving this API
directly owns the choice.

Why a subprocess, and why stdin
-------------------------------
`livekit.portal` is an FFI dylib behind the optional `[drtc]` extra, and the
server must never load it — so the child is the only process that imports it,
and the two ends speak `makermodslab.drtc_protocol` over pipes. The child owns
the bus, so a stop CANNOT be a signal: `robot_sync`'s `finally:` would release
torque wherever the policy left the arm. `STOP` on its stdin makes it return to
the pose it captured at connect FIRST and release torque only after; a second
`STOP` cuts that return short (the same second-press semantics as replay and
teleop). See drtc_protocol's module docstring for the whole vocabulary.

The status dict (the S3.3 contract)
-----------------------------------
:func:`handle_remote_inference_status` and the terminal `_last_result` payload
are built by ONE function (:func:`_payload_locked`) and always carry EXACTLY
these keys, so `schemas/sessions.py` can put an exact `response_model` on them
with no `exclude_none`/`exclude_unset` (per the schema-fidelity rule: a model
that materializes absent optionals as `null` must describe a payload that
really always carries them):

    remote_inference_active : bool
    exited                  : bool
    exit_code               : int | None
    outcome                 : "ok" | "ran_with_warning" | "failed" | None
    error                   : str | None
    hint                    : str | None
    warning                 : str | None      (arm-identity warn-but-allow)
    phase                   : str | None      (the vocabulary below)
    policy_ref              : str | None
    engine                  : str | None      ("sync" / "rtc"; null before any run)
    started_at              : float | None    (unix seconds)
    elapsed_s               : float           (FROZEN at the exit on a terminal
                                               payload, not reset to 0)
    duration_s              : int | None
    log_path                : str | None
    returning_to_rest       : bool
    shutting_down           : bool
    transport               : {url, room, source, operator_present} | None
    stats                   : {every drtc_protocol.STATS_KEYS key} | None

`outcome`/`error`/`hint` reuse `rollout._classify_outcome` and
`utils.errors.friendly_hint` verbatim, so a terminal payload here means what it
means for local inference.

Phases (opaque strings, broadcast as `session_changed` hints):

    resolving        credentials + the optional extra
    transport_check  the room probe
    preflight        arm type / arm count / cameras / arm identity
    starting         the child is being spawned
    connecting       READY   — the child resolved its transport and is dialing
    warming_up       CONNECTED — in the room, no correlated chunk yet
    easing           EASING  — ramping into the first chunk's step-0 pose
    running          the first chunk has been counted
    stopping         a stop is in flight (INCLUDING the return to rest —
                     `returning_to_rest` says which half; a new phase name here
                     would fall outside sessions._WINDING_DOWN_PHASES and let an
                     expiry tick dispatch a SECOND stop into a live return)
    stopped / error  terminal
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel

from . import sfu
from .api_errors import ErrorCode
from .arm_capabilities import supports_remote_inference
from .arm_identity import ArmIdentityError
from .camera_preview import camera_preview_manager
from .drtc_protocol import (
    CMD_STOP,
    EVENT_ACTIVE,
    EVENT_BYE,
    EVENT_CONNECTED,
    EVENT_EASING,
    EVENT_ERROR,
    EVENT_READY,
    EVENT_RETURNING,
    EVENT_STATS,
    parse_event,
    parse_kv,
    parse_stats,
)
from .jobs import read_pretrained_config
from .rest_pose import RETURN_CEILING_S
from .rollout import (
    _HUB_ROOT_REF_RE,
    InferenceRequest,
    PolicyCameraDims,
    _arm_count_mismatch,
    _classify_outcome,
    _extract_error_from_log,
    _prepare_robot,
    _session_cameras,
    _terminate_tree,
)
from .session_events import notify_session_changed
from .utils.config import (
    DRTC_ENV_PATH,
    DRTC_LOG_DIR,
    LIVEKIT_KEY_FILE,
    CameraResolutionError,
    get_instance_id,
)
from .utils.errors import friendly_hint, transport_hint
from .utils.system import (
    MAX_EXTRA_IMAGE_ROLES,
    is_valid_image_role,
    policy_supports_extra_image_roles,
)

# The `[drtc]` extra is OPTIONAL, and this module is imported by the FastAPI
# server at boot (and by every peer feature's reciprocal guard), so none of its
# packages may be a hard import. Guarded at module TOP rather than inside the
# functions — an import inside a function body is the thing this project does
# not do — and read as "the extra is installed" at the one place that asks
# (`_extra_missing`). All three come from the same extra, so one guard covers
# them: a partial install is reported as a missing extra, which is the honest
# remedy either way.
#
# `livekit.api` is safe to import here: it is pure Python + aiohttp. The FFI
# dylib is `livekit.portal`, which this module NEVER imports — its presence is
# checked with importlib.util.find_spec and nothing else.
try:
    import aiohttp as _aiohttp
    from dotenv import dotenv_values as _dotenv_values
    from livekit import api as _livekit_api

    from .drtc._env import read_env as _read_env
except ImportError:  # pragma: no cover — exercised by monkeypatching the names
    _aiohttp = None
    _dotenv_values = None
    _livekit_api = None
    _read_env = None

logger = logging.getLogger(__name__)

# The session kind, the busy discriminant's tail, and the WS `session_changed`
# kind — one string, three contracts (session_events.SESSION_KINDS pins it).
KIND = "remote_inference"

PHASE_RESOLVING = "resolving"
PHASE_TRANSPORT_CHECK = "transport_check"
PHASE_PREFLIGHT = "preflight"
PHASE_STARTING = "starting"
PHASE_CONNECTING = "connecting"
PHASE_WARMING_UP = "warming_up"
PHASE_EASING = "easing"
PHASE_RUNNING = "running"
PHASE_STOPPING = "stopping"
PHASE_STOPPED = "stopped"
PHASE_ERROR = "error"

# The phases the two empty-room watchdogs below are armed in: the window
# between "in the room" and "the loop is actually executing chunks". Outside it
# there is either nothing to wait for yet (pre-CONNECTED) or nothing wrong
# (running), and a stop already in flight must never be re-triggered.
_WATCHED_PHASES = frozenset({PHASE_WARMING_UP, PHASE_EASING})

# No ACTIVE within this long after CONNECTED: the room is real and we are in
# it, but no operator ever joined — the GPU was never launched, its Modal
# secret names a different LIVEKIT_ROOM, or its tailnet auth key expired.
_ACTIVE_TIMEOUT_S = 15.0
# ACTIVE seen, but not one correlated chunk in this long: Portal's schema
# FINGERPRINT disagrees and it is dropping every packet in silence. This is the
# watchdog that earns its keep — the failure is invisible by construction and
# presents as a perfectly healthy session with 0 chunks.
_CHUNK_TIMEOUT_S = 10.0

# How long the room probe may take. livekit-api's own default is 10 s, which is
# an eternity behind a Start button; a reachable SFU answers in tens of ms.
_PROBE_TIMEOUT_S = 3.0

# Portal identities/roles that count as "a policy is in the room". `policy.py`
# connects as IDENTITY = "policy"; Portal itself self-sets `lk.portal.role` on
# connect (which is why the token grants can_update_own_metadata), so either is
# proof. Checking both means a renamed identity on the GPU side degrades to
# "found via the role attribute" rather than to a false empty-room refusal.
_POLICY_IDENTITY = "policy"
_PORTAL_ROLE_ATTRIBUTE = "lk.portal.role"
_OPERATOR_ROLE = "operator"

# The four credentials a child needs. LIVEKIT_ROOM is as load-bearing as the
# rest: it is the one value the GPU side takes ONLY from its Modal secret, so a
# mismatch is undetectable from here (see the transport.no_policy hint).
_REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_ROOM", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")

# Bound on the graceful stop: the child returns the arm to its captured start
# pose before releasing torque, and `rest_pose.RETURN_CEILING_S` is that
# return's own hard ceiling. +5 s covers the disconnect and process teardown
# behind it; past that the whole process group is terminated.
_STOP_WAIT_S = RETURN_CEILING_S + 5.0
# A stop that lands before the child was spawned, or on a child that is already
# gone, has nothing to wait for.
_PRE_SPAWN_TERMINATE_TIMEOUT_S = 1.0
# Same bounded-wait-and-report as rollout's second stop press: a startup worker
# inside `_prepare_robot` cannot be interrupted mid-call.
_STARTUP_STOP_JOIN_TIMEOUT_S = 5.0

# One log file per run, in its own subdirectory under the DRTC log root.
# Mirrors rollout's inference_logs.
_LOG_DIR = Path(DRTC_LOG_DIR) / "sessions"

# --- module state -----------------------------------------------------------
# Guarded by `_state_lock`, which is held only for the short critical sections
# in start/stop/status and in the pump's event handlers.
remote_inference_active: bool = False
_state_lock = threading.Lock()
_remote_proc: subprocess.Popen | None = None
_remote_started_at: float | None = None
# Wall-clock of the first counted chunk — the analogue of rollout's
# `_inference_rollout_started_at`, and what `_classify_outcome` reads as
# "did the run actually get going before it failed?".
_remote_running_started_at: float | None = None
_remote_meta: dict[str, Any] = {}
# The finished payload of the most recent run, kept until the NEXT start claims
# the slot. Terminal outcomes are idempotent, not consume-once: several
# surfaces poll the status concurrently and a report-once scheme lets whichever
# poll lands first swallow the error the user needed to see (see rollout's
# `_last_result` for the incident this convention comes from).
_last_result: dict[str, Any] | None = None
_remote_cancel: threading.Event | None = None
_startup_thread: threading.Thread | None = None
# Effective transport, as VERIFIED by the preflight: {url, room, source,
# operator_present}. None until the probe has run.
_transport: dict[str, Any] | None = None
# The most recent STATS sample (every STATS_KEYS key, null where unknown), or
# None before the first one.
_stats: dict[str, Any] | None = None
# Monotonic stamps the two watchdogs measure from, and the chunk counter they
# read. Monotonic (not wall clock) because they are durations.
_connected_at: float | None = None
_active_at: float | None = None
_chunks: int = 0
# True between RETURNING and the child's exit: the arm is being driven back to
# its start pose with torque still on. Exposed so the UI can say so during the
# `stopping` phase rather than inventing a phase name for it.
_returning_to_rest: bool = False
# Injected clock — the watchdogs' only time source, so tests drive them with
# tests/test_session_lease.py's FakeClock instead of sleeping.
_clock = time.monotonic


def remote_inference_is_active() -> bool:
    """Whether a remote-inference session currently holds the follower."""
    return remote_inference_active


class RemoteInferenceRequest(BaseModel, extra="forbid"):
    """One remote-inference run.

    Robot fields first (sessions.py resolves them from the robot record in
    S3.3, exactly as it does for inference), then the options. `extra="forbid"`
    because every field here is either a hardware address or half of a wire
    contract that must match the GPU side — a typo silently ignored is a run
    that connects, looks healthy and never receives a single chunk.
    """

    follower_port: str
    follower_config: str
    robot_name: str = ""
    arm_type: str = "so101"
    # Present so the refusal can be explicit rather than accidental: bimanual is
    # refused (see supports_remote_inference), and the arm-count guard needs it.
    mode: str = "single"

    # The Lab-side ref (/jobs/{id}/checkpoints). LOCAL METADATA ONLY in this
    # slice: the checkpoint is loaded by `from_pretrained` inside the Modal
    # container, so nothing here downloads it. It is what the launch UI already
    # has, and what yields checkpoint_state_dim / camera_dims / the task
    # prefill, which is why it is kept separate from policy_hub_id below.
    policy_ref: str = ""
    # "<owner>/<repo>" the GPU loads. Advisory in this slice (it becomes
    # `--policy-path` when the Lab launches Modal itself); recorded so the
    # status and the copy-able `modal run` line agree on one value.
    policy_hub_id: str = ""
    task: str = ""
    # {policy-expected camera name: robot-record camera name}. The ONLY way the
    # child's `--robot.cameras` may be built: Portal derives the robot's track
    # names from those keys and the policy's from the checkpoint's
    # observation.images.*, and a disagreement is the likeliest instance of the
    # silent-fingerprint failure. `bind_robot_cameras` is what makes them agree.
    camera_bindings: dict[str, str] = {}
    camera_dims: dict[str, PolicyCameraDims] = {}
    checkpoint_state_dim: int | None = None
    duration_s: int = 60
    # MUST match the GPU side. Not defaulted from the checkpoint here — the
    # other terminal's flags are the authority, and guessing would trade a
    # legible refusal for a silent zero-chunk run.
    horizon: int = 16
    fps: int = 30
    # A closed set, not free text: an unknown name reaches the child as
    # `getattr(VideoCodec, name)` and raises AFTER the bus is open and the arm
    # energized. H264 for >=480p or multi-camera (inter-frame, ~1 chunk per
    # frame); MJPEG only for <=~256px policies, where a frame fits one SCTP
    # chunk. Portal's PNG/RAW stay off the API — they are bench codecs.
    video_codec: Literal["H264", "MJPEG"] = "H264"
    # Which chunk player runs on the arm — and therefore which module is
    # spawned and which GPU server the other terminal must be running. See
    # `_CHILD_MODULES`, and `schemas/sessions.RemoteInferenceOptions.engine`
    # for why the BACKEND CANNOT VERIFY the choice against the checkpoint.
    engine: Literal["sync", "rtc"] = "sync"
    # Minimum execution budget in action steps; MUST match the GPU side's
    # `--s-min` on the rtc engine (the robot computes `overlap_end` from it and
    # the server trusts the field). Only sent for `engine="rtc"`.
    s_min: int = 4
    skip_identity_check: bool = False


# The child each engine spawns. `sync` plays every chunk to completion and
# adapts WHEN it prefetches; `rtc` re-plans continuously and ships the
# still-to-execute prefix so the server can guide denoising — which is what
# makes overlapping flow-policy chunks agree instead of fighting at every seam.
_CHILD_MODULES: dict[str, str] = {
    "sync": "makermodslab.drtc.robot_sync",
    "rtc": "makermodslab.drtc.robot_rtc",
}


@dataclass(frozen=True)
class RoomProbe:
    """What one `list_participants` call told us about the transport.

    Four independent facts, because the preflight's four rungs need to tell
    them apart: an SFU that is down, credentials that are wrong, a room that
    does not exist, and a room nobody has joined are four different remedies.
    """

    reachable: bool
    authorized: bool
    room_exists: bool
    operator_present: bool
    error: str | None = None


# --- pure helpers -----------------------------------------------------------


def _extra_missing() -> bool:
    """True when the `[drtc]` extra is not (fully) installed.

    `find_spec` and NEVER an import: `livekit.portal` is an FFI dylib, and
    loading it into the server process is precisely what the extra's split
    exists to avoid. The child imports it; we only ask whether it is there."""
    if _livekit_api is None or _read_env is None or _dotenv_values is None or _aiohttp is None:
        return True
    try:
        return importlib.util.find_spec("livekit.portal") is None
    except (ImportError, ValueError):
        # A half-installed namespace package can raise rather than return None.
        return True


def _transport_source(url: str) -> str:
    """Which layer supplied the EFFECTIVE url: sfu | cloud | process_env | none.

    Four values, and they are four different remedies:

    - `sfu` — this process runs the Lab's own LiveKit server
      (`makermodslab --sfu`). Nothing on disk is consulted at all: the url, the
      room and the child's token are all minted in-process from the 0600 key
      file, so there is no credential for anyone to get wrong.
    - `process_env` — the server's own environment exported LIVEKIT_URL, so
      editing `livekit.env` will not change anything. Telling an operator to
      edit a file that is being overridden is the worst of the three answers,
      which is why this is a value of its own.
    - `cloud` — `livekit.env`, the saved LiveKit Cloud credentials.
    - `none` — nothing anywhere set it.

    Membership, not value equality, decides among the last three, walked in
    `drtc._env.read_env`'s DESCENDING precedence: whichever layer SETS the key
    is by definition the one whose value survived the merge.

    Retired in S3.6 along with the shell SFU scripts: `local_override`
    (`livekit.local.env`) and `cwd` (a `.env`/`.env.local` beside wherever the
    Lab happened to be started). The Lab-owned SFU is what they were for.
    """
    if sfu.sfu_enabled():
        # Not conditioned on the url matching: when the SFU is up the session
        # uses it unconditionally, so the url IS the SFU's by construction.
        return "sfu"
    if not url:
        return "none"
    if os.environ.get("LIVEKIT_URL"):
        return "process_env"
    if _dotenv_values is None:  # pragma: no cover — extra guard
        return "cloud"
    try:
        saved = Path(DRTC_ENV_PATH)
        if saved.exists() and _dotenv_values(saved).get("LIVEKIT_URL"):
            return "cloud"
    except OSError:
        pass
    return "none"


def _missing_credentials(env: dict[str, str]) -> list[str]:
    """Which of the four LiveKit variables the resolved environment lacks."""
    return [name for name in _REQUIRED_ENV if not env.get(name)]


# --- extra camera views (S3.8g) ----------------------------------------------


def _checkpoint_camera_roles(policy_ref: str) -> tuple[set[str], str | None] | None:
    """``({role, …}, policy_type)`` off the checkpoint's own config.json.

    Roles are the SUFFIXES of the visual `input_features` keys — `cam0` for
    `observation.images.cam0` — because that is the vocabulary
    `camera_bindings` speaks: `/policy-config` strips the prefix the same way,
    and the panel builds its bindings from what that route returned.

    None means "not established", never "fine": an unreadable config (a private
    repo, no network, a path that is not there yet) must not refuse a run, on
    exactly the rule `modal_launcher.resolve_knobs` follows for the launch
    knobs. `@root` refs are handed over as the bare repo id, which is where that
    shape keeps its config.json — the same unwrap `rollout`'s own rtc guard does.
    """
    ref = (policy_ref or "").strip()
    if not ref:
        return None
    root = _HUB_ROOT_REF_RE.match(ref)
    try:
        cfg = read_pretrained_config(root.group("repo") if root else ref)
    except Exception:  # noqa: BLE001 — a camera-role check must never fail a start
        logger.info("Couldn't read the checkpoint config for %s; camera roles pass through", ref)
        cfg = None
    if cfg is None:
        return None
    features = cfg.get("input_features")
    if not isinstance(features, dict):
        return None
    roles = {
        str(key).split(".")[-1]
        for key, feat in features.items()
        if isinstance(feat, dict) and feat.get("type") == "VISUAL"
    }
    policy_type = cfg.get("type")
    return roles, policy_type if isinstance(policy_type, str) else None


def _extra_camera_role_refusal(request: RemoteInferenceRequest) -> str | None:
    """Why this run's camera bindings can't be honoured, or None (S3.8g).

    Binding a role the checkpoint never declared is now a REAL thing to do — the
    GPU side can declare extra views on a checkpoint before the weights load
    (`--extra-image-roles`, `drtc/_policy_views`) — but only for a family whose
    view count is a property of its lerobot wrapper rather than of its
    architecture. For everything else an unknown role is a camera the policy
    will never look at: the robot opens it, encodes it, publishes a track for
    it, and Portal's schema fingerprint no longer matches the policy's, which
    presents as a session that connects and receives nothing.

    So this runs BEFORE the arm is claimed and refuses in the panel, and it
    checks two separate things:

    * the role NAME, always and with no read at all. It becomes a
      `--robot.cameras` dict key inside a draccus-parsed argv and a Portal video
      TRACK name, and a comma, a brace or a space in it breaks one of those
      silently.
    * whether the checkpoint can TAKE an extra view, which needs the
      checkpoint's own config.json — a few KB, local for a local ref and a
      cached download for a Hub one, and read only when there are bindings to
      check.

    Fail-OPEN on an unreadable config, deliberately and on
    `modal_launcher.resolve_knobs`' rule: None from the reader is "not
    established", and an offline moment must not become a refusal. The GPU side
    still refuses the flag itself, before the weights load, which is what makes
    this an early diagnosis rather than the authority.
    """
    bindings = request.camera_bindings
    if not bindings:
        return None
    for role in bindings:
        if not is_valid_image_role(role):
            return (
                f"'{role}' isn't a usable camera role. Use lowercase letters, digits and "
                "underscores, starting with a letter, at most 32 characters (e.g. `cam2`) — "
                "the role becomes a video track name and a robot camera key, and anything "
                "else breaks the stream rather than raising."
            )

    known = _checkpoint_camera_roles(request.policy_ref)
    if known is None:
        return None
    roles, policy_type = known
    extra = sorted(set(bindings) - roles)
    if not extra:
        return None

    if not policy_supports_extra_image_roles(policy_type):
        declared = ", ".join(sorted(roles)) or "none"
        return (
            f"This checkpoint has no camera role called {', '.join(repr(r) for r in extra)}. "
            f"It declares: {declared}. A '{policy_type}' policy's image views are fixed by its "
            "architecture, so an extra camera would be published, encoded and never looked at — "
            "and the two halves' wire schemas would stop matching, which shows up as a session "
            "that connects and receives no chunks. Remove the binding."
        )
    if len(extra) > MAX_EXTRA_IMAGE_ROLES:
        return (
            f"{len(extra)} extra camera roles were bound ({', '.join(extra)}); at most "
            f"{MAX_EXTRA_IMAGE_ROLES} may be added to a checkpoint. Every extra view is another "
            "~196 image tokens through the policy's prefill, which is a latency ceiling rather "
            "than a model limit."
        )
    logger.info(
        "Remote inference: %s extra camera %s (%s) beyond the checkpoint's own (%s) — allowed, "
        "'%s' declares its views in its wrapper rather than its architecture. The GPU must be "
        "launched with --extra-image-roles=%s or the two wire schemas will not match.",
        len(extra),
        "role" if len(extra) == 1 else "roles",
        ", ".join(extra),
        ", ".join(sorted(roles)) or "none",
        policy_type,
        ",".join(extra),
    )
    return None


def _robot_request(request: RemoteInferenceRequest) -> InferenceRequest:
    """The rollout-shaped request `_prepare_robot`/`_session_cameras` consume.

    Those two are imported rather than reimplemented (they stage the
    calibration, run the arm-identity guard, reset the torque cap and resolve
    the record's cameras — all identical work), and they take an
    `InferenceRequest`. Building one explicitly is honest about that: the
    alternative, relying on the two models happening to share attribute names,
    breaks silently the day either gains a field the other lacks.

    `policy_ref` rides along because the model requires it; nothing on the
    robot-preparation path reads it. `coaching` and the eval fields keep their
    defaults — a remote run is neither.

    `camera_dims` is deliberately NOT carried. For a local rollout the
    checkpoint's image size must be the CAPTURE size, because nothing in
    lerobot's standard rollout resizes frames. A remote run is different on
    both ends: the robot child only encodes what the camera delivers, and
    both GPU servers (`policy.py` / `policy_rtc.py`, `_build_batch`) resize
    every decoded frame to the checkpoint's expected shape with
    `F.interpolate` before inference. So the honest capture size here is the
    robot record's own configured mode — a real mode the camera supports —
    and overlaying the checkpoint's size only manufactures failures:
    MolmoAct2 trains at 224×224, no UVC camera offers 224 wide, and lerobot's
    `OpenCVCamera` raised `failed to set capture_width=224 (actual_width=352)`
    on the bench (2026-09-03) for a run the GPU would have resized anyway.
    """
    return InferenceRequest(
        follower_port=request.follower_port,
        follower_config=request.follower_config,
        policy_ref=request.policy_ref,
        robot_name=request.robot_name,
        arm_type=request.arm_type,
        mode=request.mode,
        camera_bindings=dict(request.camera_bindings),
        camera_dims={},
        checkpoint_state_dim=request.checkpoint_state_dim,
        skip_identity_check=request.skip_identity_check,
        task=request.task,
    )


def _robot_sync_args(
    request: RemoteInferenceRequest,
    robot_args: list[str],
    *,
    url: str,
    room: str,
    token: str = "",
) -> list[str]:
    """The child's flags, without the interpreter/module prefix.

    Engine-aware, but only just: the two entrypoints deliberately share their
    whole session surface (`._session_glue`), so every flag below means the same
    thing on both and only `--s_min` is engine-specific. It is sent for `rtc`
    alone because there it is HALF A CONTRACT — the robot computes
    `overlap_end = H - max(s_min, d)` and `policy_rtc` trusts that field, so a
    disagreement silently guides the denoiser against the wrong mask. The sync
    engine's own `s_min` only tunes when its player declares itself degraded,
    and it is left at the child's (identical) default with the other scheduler
    knobs.

    `robot_args` is `_prepare_robot`'s output — the same `--robot.type/port/id/
    cameras` block `lerobot-rollout` gets, and the cameras in it are keyed by
    the POLICY-expected names (see `_session_cameras` / `bind_robot_cameras`).
    That keying is the whole mitigation for Portal's silent fingerprint
    mismatch, so `--robot.cameras` must be built no other way. An S3.8g extra
    role changes nothing about that: it is one more camera keyed by the name the
    policy will use for it, which is precisely why it needs no new flag here.

    The transport is PINNED to what the preflight actually verified rather than
    left to the child's own `_env` resolution, which is what closes the "parent
    probed room X, the child resolved room Y" class of failure. The child
    echoes the effective values back in READY and we compare.

    `token` is sent only in the SFU case, where the parent signed it in-process
    with the key file's secret: the child then needs no LiveKit credential of
    its own. Empty on the Cloud path, where the child mints its own from the
    environment (`_common.mint_token`). It is a short-lived, single-room,
    single-identity JWT rather than the API secret — which is why it may ride
    an argv at all, and the secret may not.

    draccus has NO `--no-<flag>` form (verified against the installed draccus,
    not assumed): a boolean is `--flag=false`. `--return_to_rest` and
    `--ease_in` already default true in `robot_sync`; they are passed
    explicitly anyway so the safety behaviour of a supervised run is stated in
    the argv a log records, not inherited from a default someone could flip.

    Everything else is left at the child's defaults on purpose (`adaptive`,
    `base_lead`, `align`, `action_delay`, `pacing`, `epsilon`, the JK constants,
    the rtc engine's low-pass, `video_quality`, `video_bitrate_kbps`,
    `reliable_state`): they are knobs whose wrong values present as "the arm
    freezes" or "the arm snaps at every boundary" rather than as an error, and
    `reliable_state` in particular auto-follows the codec — forcing it wrong
    head-of-line-blocks state behind H264 retransmits.
    """
    return [
        *robot_args,
        f"--fps={request.fps}",
        f"--horizon={request.horizon}",
        f"--duration_s={request.duration_s}",
        f"--video_codec={request.video_codec}",
        *([f"--s_min={request.s_min}"] if request.engine == "rtc" else []),
        f"--livekit_url={url}",
        f"--livekit_room={room}",
        *([f"--livekit_token={token}"] if token else []),
        "--return_to_rest=true",
        "--ease_in=true",
    ]


def _child_module(engine: str) -> str:
    """The `python -m` target for one engine.

    Falls back to the sync entrypoint for an engine the map does not know —
    unreachable through the API (the options model's Literal refuses anything
    else) and it would be a poor trade to raise here, in the worker thread,
    after the arm has already been claimed and preflighted."""
    return _CHILD_MODULES.get(engine, _CHILD_MODULES["sync"])


def _fingerprint(request: RemoteInferenceRequest) -> str:
    """The wire settings both ends must agree on, as one legible phrase.

    Named in the no-chunks watchdog's message because these — plus the camera
    names — are the whole content of Portal's schema fingerprint, and the
    operator's next action is to compare them against the other terminal's
    `modal run` line.

    `engine` leads it, because it decides WHICH `modal run` line that is: an
    rtc robot against `modal_policy.py` sends a state schema carrying five
    extra RTC fields, and Portal answers a fingerprint mismatch by silently
    dropping every packet — the healthiest-looking zero-chunk failure there
    is."""
    cameras = ", ".join(request.camera_bindings) or "none"
    return (
        f"engine={request.engine}, horizon={request.horizon}, fps={request.fps}, "
        f"video_codec={request.video_codec}, cameras={cameras}"
    )


def _stdin_seed() -> bytes:
    """One newline, to pre-answer lerobot's calibration prompt.

    `SOFollower.calibrate()` asks "Press ENTER to use the calibration file …" on
    stdin during `connect()`, and the child deliberately does not start reading
    commands until after that (see robot_sync). One arm, one newline; if the
    prompt never fires the newline is read by `pump_commands` as a blank line
    and ignored. Mirrors rollout's `_stdin_seed` for the single-arm case, which
    is the only case remote inference supports."""
    return b"\n"


# --- the Lab-owned SFU ------------------------------------------------------
#
# When this process was started with `makermodslab --sfu` (sfu.py), the whole
# credential question disappears: the SFU is on this machine, its API secret is
# in a 0600 file only the server reads, and the session mints the child's token
# itself. `livekit.env` is then never consulted — it is the LiveKit Cloud
# fallback for a station with no local SFU.

# Portal identities, and they are CONTRACTS, not defaults. The child connects
# as `robot` (robot_sync/robot_rtc's IDENTITY) and the GPU side as `policy`
# (policy.py's), which is what `_is_policy` looks for in the room probe.
# `sfu.default_identity` mints `<role>-<8 hex>` for a browser or a laptop and
# is deliberately NOT used here: a random identity would make the probe blind.
_ROBOT_IDENTITY = "robot"

# How long a child's token stays valid. The session's own `duration_s` ceiling
# is minutes, not hours; an hour covers a long run plus a reconnect, and the
# token dies with the room long before it could become a standing credential.
_TOKEN_TTL_S = 3600

# `tailscale ip -4` is how a Modal container learns an address that reaches
# this machine's signalling port. Read-only and bounded — it is called from a
# GET handler.
_TAILSCALE_TIMEOUT_S = 2.0
# The App Store build puts the CLI outside PATH; SLICE3.md §2C names this as
# the fallback every tailscale probe needs.
_TAILSCALE_APP_PATH = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"


@dataclass(frozen=True)
class SfuTransport:
    """What the local SFU gives one session: where to dial, which room, the
    credentials the PARENT probes with, and the token the CHILD joins with."""

    url: str
    room: str
    api_key: str
    api_secret: str
    token: str


def _sfu_transport() -> SfuTransport:
    """Resolve the local SFU's transport and sign the child's token.

    Raises `OSError`/`RuntimeError` when the key file is unreadable or holds no
    pair — the launcher wrote it before the app started, so that is a genuinely
    broken install rather than a configuration the user can fix in the panel.
    """
    api_key, api_secret = sfu.api_keys()
    room = sfu.default_room(get_instance_id())
    token, _expires_at = sfu.mint_token(
        api_key=api_key,
        api_secret=api_secret,
        identity=_ROBOT_IDENTITY,
        room=room,
        role="robot",
        ttl_seconds=_TOKEN_TTL_S,
    )
    return SfuTransport(url=sfu.local_url(), room=room, api_key=api_key, api_secret=api_secret, token=token)


def _sfu_key_file() -> str:
    """The key file path the launcher exported, or the default location."""
    return os.environ.get(sfu.ENV_KEY_FILE) or LIVEKIT_KEY_FILE


@dataclass(frozen=True)
class ResolvedTransport:
    """Where the two halves of a remote run meet, resolved ONCE.

    The SFU-first ladder of :func:`resolve_transport`, as a value. `missing` is
    non-empty only on the LiveKit Cloud path and then nothing else in here is
    meaningful; `child_token` is the parent-signed JWT the SFU path hands the
    child and is empty on the Cloud path, where the child mints its own.
    """

    url: str
    room: str
    api_key: str
    api_secret: str
    child_token: str
    source: str
    missing: tuple[str, ...]


def resolve_transport() -> ResolvedTransport:
    """THE credential resolution — the session's preflight, the transport
    endpoint and `modal_launcher` all call this and nothing else.

    Two mutually exclusive paths, and the SFU wins: when this process runs one,
    that is where the room is, and reading a LiveKit Cloud file to dial past
    our own server would be a config error waiting to happen. On the Cloud path
    we READ and never load — `_env.load_env` writes os.environ, and a server
    that has stamped a url into its own environment can never re-resolve it.

    Why one function rather than three call sites that happen to agree: the GPU
    side and the robot side meeting in DIFFERENT ROOMS is invisible by
    construction (Portal drops the mismatched stream in silence), so a second
    credential path is not a duplication smell, it is the failure mode. The one
    thing callers legitimately differ on is which URL a given peer can DIAL —
    see :func:`sfu_modal_url` for the container's.

    Raises `OSError`/`RuntimeError` from the SFU key file, as
    :func:`_sfu_transport` does: the launcher wrote that file before the app
    started, so an unreadable one is a broken install and not a configuration
    the caller can repair. Total otherwise — a Cloud path with nothing
    configured comes back with every variable in `missing`.
    """
    if sfu.sfu_enabled():
        transport = _sfu_transport()
        return ResolvedTransport(
            url=transport.url,
            room=transport.room,
            api_key=transport.api_key,
            api_secret=transport.api_secret,
            child_token=transport.token,
            source="sfu",
            missing=(),
        )
    # `_read_env` is None only when the `[drtc]` extra is absent. Both session
    # callers gate on `_extra_missing()` long before this, so the fallback is
    # for the launcher's benefit: "nothing is configured" is the honest answer
    # there, and it carries the same remedy.
    env = _read_env() if _read_env is not None else {}
    url = env.get("LIVEKIT_URL", "")
    return ResolvedTransport(
        url=url,
        room=env.get("LIVEKIT_ROOM", ""),
        api_key=env.get("LIVEKIT_API_KEY", ""),
        api_secret=env.get("LIVEKIT_API_SECRET", ""),
        child_token="",  # noqa: S106  # nosec B106 — an empty JWT: "the child mints its own"
        source=_transport_source(url),
        missing=tuple(_missing_credentials(env)),
    )


def _tailscale_ipv4() -> str | None:
    """This machine's tailnet IPv4, or None.

    None for every reason there is — no tailscale, not logged in, the command
    hung — because they all have the same consequence here: the panel cannot
    offer a Modal-reachable URL and says so instead of guessing one.
    """
    binary = shutil.which("tailscale")
    if binary is None and os.path.isfile(_TAILSCALE_APP_PATH):
        binary = _TAILSCALE_APP_PATH
    if binary is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell, bounded
            [binary, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=_TAILSCALE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first and first[0].strip() else None


def sfu_modal_url() -> str | None:
    """The signalling URL a Modal container should dial, or None.

    A container has no route to `127.0.0.1` and none to a LAN address either;
    what it can reach — with `--tailscale` and the `tailscale-auth` secret — is
    this machine's tailnet IP. Media is a separate problem and a separate flag
    (`--sfu-external-ip`; see sfu.render_config).

    Public because it has two consumers: the transport endpoint (which reports
    it so the panel can WRITE the `modal run` line) and `modal_launcher` (which
    RUNS that line). Deliberately not a field of :class:`ResolvedTransport` —
    it shells out to `tailscale ip -4`, and the session's own preflight has no
    business paying for that.
    """
    ip = _tailscale_ipv4()
    if not ip:
        return None
    port = os.environ.get(sfu.ENV_PORT) or str(sfu.SFU_HTTP_PORT)
    return f"ws://{ip}:{port}"


# --- the room probe ---------------------------------------------------------


def _probe_room(url: str, key: str, secret: str, room: str) -> RoomProbe:
    """Ask the SFU who is in `room`. The ONE seam tests monkeypatch.

    One call answers every question the preflight has: a connection error means
    the signaling endpoint is down, an auth error means the credentials are
    wrong, and the participant list says whether a policy is there to talk to.
    The twirp client normalizes `ws://` → `http://` itself, so a local SFU's
    `ws://127.0.0.1:7880` and a cloud `wss://…` both work unchanged.

    Synchronous by design — it runs on the request thread, bounded by
    `_PROBE_TIMEOUT_S`, before anything is energized. `asyncio.run` is safe
    here because the handler is a sync route (FastAPI runs those in a
    threadpool, with no loop of their own).

    Never raises: every failure is one of the four flags plus `error`. Tests
    patch this function rather than the network — livekit-api is aiohttp-based,
    so `httpx.MockTransport` does not apply and no new test dependency is
    warranted for a seam this thin.
    """
    if _livekit_api is None or _aiohttp is None:  # pragma: no cover — extra guard
        return RoomProbe(False, False, False, False, error="the drtc extra is not installed")

    async def _list() -> list[Any]:
        client = _livekit_api.LiveKitAPI(
            url,
            key,
            secret,
            timeout=_aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_S),
        )
        try:
            response = await client.room.list_participants(_livekit_api.ListParticipantsRequest(room=room))
            return list(response.participants)
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    try:
        participants = asyncio.run(_list())
    except Exception as exc:
        code = str(getattr(exc, "code", "") or "").lower()
        status = getattr(exc, "status", None)
        if code in ("unauthenticated", "permission_denied") or status in (401, 403):
            return RoomProbe(True, False, False, False, error=str(exc))
        if code == "not_found" or status == 404:
            # Reachable and authorized — the room simply is not there, which is
            # the empty-room case caught before the arm is energized.
            return RoomProbe(True, True, False, False, error=str(exc))
        return RoomProbe(False, False, False, False, error=str(exc))
    return RoomProbe(
        reachable=True,
        authorized=True,
        # A room with no participants and a room that does not exist are
        # indistinguishable here (LiveKit returns an empty list for both), and
        # they have the SAME remedy — so the distinction is deliberately not
        # invented; `operator_present` is the flag that decides.
        room_exists=True,
        operator_present=any(_is_policy(p) for p in participants),
    )


def _is_policy(participant: Any) -> bool:
    """Whether one participant is the remote policy (identity OR portal role)."""
    if getattr(participant, "identity", None) == _POLICY_IDENTITY:
        return True
    attributes = getattr(participant, "attributes", None) or {}
    try:
        return attributes.get(_PORTAL_ROLE_ATTRIBUTE) == _OPERATOR_ROLE
    except AttributeError:
        return False


# --- phase / state plumbing -------------------------------------------------


def _set_phase(phase: str) -> None:
    """Record a phase on the live meta and broadcast the refetch hint.

    A no-op when no session is active: a late stdout line arriving after
    teardown must not resurrect a phase on an empty meta (or broadcast a
    phantom one). The notify happens OUTSIDE the lock — it is a droppable queue
    put that consumers answer by refetching."""
    with _state_lock:
        if not _remote_meta:
            return
        _remote_meta["phase"] = phase
    notify_session_changed(KIND, True, phase=phase)


def _go_idle_locked() -> None:
    """Drop every per-session global back to idle. Caller holds `_state_lock`.

    Does NOT touch `_last_result` — whether a teardown leaves a terminal
    payload behind is the caller's decision."""
    global remote_inference_active, _remote_proc, _remote_started_at
    global _remote_running_started_at, _remote_meta, _remote_cancel
    global _transport, _stats, _connected_at, _active_at, _chunks, _returning_to_rest
    remote_inference_active = False
    _remote_proc = None
    _remote_started_at = None
    _remote_running_started_at = None
    _remote_meta = {}
    _remote_cancel = None
    _transport = None
    _stats = None
    _connected_at = None
    _active_at = None
    _chunks = 0
    _returning_to_rest = False


def _release_slot() -> None:
    """Undo a just-made claim after a pre-spawn guard refused."""
    with _state_lock:
        _go_idle_locked()
    notify_session_changed(KIND, False)


def _payload_locked(*, shutting_down: bool) -> dict[str, Any]:
    """The status dict, from the CURRENT globals. Caller holds `_state_lock`.

    The single builder for both the live payload and the terminal one (which is
    this, overridden with the exit fields) — see the module docstring's key
    list, which S3.3's response model must match exactly."""
    elapsed = (time.time() - _remote_started_at) if _remote_started_at else 0.0
    return {
        "remote_inference_active": remote_inference_active,
        "exited": False,
        "exit_code": None,
        "outcome": None,
        "error": _remote_meta.get("error"),
        "hint": _remote_meta.get("hint"),
        "warning": _remote_meta.get("warning"),
        "phase": _remote_meta.get("phase"),
        "policy_ref": _remote_meta.get("policy_ref"),
        "engine": _remote_meta.get("engine"),
        "started_at": _remote_started_at,
        "elapsed_s": elapsed,
        "duration_s": _remote_meta.get("duration_s"),
        "log_path": _remote_meta.get("log_path"),
        "returning_to_rest": _returning_to_rest,
        "shutting_down": shutting_down,
        "transport": dict(_transport) if _transport else None,
        "stats": dict(_stats) if _stats else None,
    }


def _terminal_payload_locked(
    *,
    exit_code: int | None,
    outcome: str,
    error: str | None,
    phase: str,
) -> dict[str, Any]:
    """The finished payload: the live shape with the exit fields filled in.

    Built BEFORE `_go_idle_locked` clears the globals it reads — which is also
    what FREEZES `elapsed_s` at the run's true length: this function runs once,
    at the exit (or at a pre-spawn failure), so `_payload_locked`'s
    `time.time() - _remote_started_at` is measured from `started_at` to that
    moment and then stored verbatim in `_last_result`. It is deliberately NOT
    zeroed. S3.4 zeroed it, and a finished run reporting "0s / 60" read as a run
    that never started — exactly the wrong thing to tell someone whose 40-second
    run just failed, and the one number that says whether it failed at once or
    ran most of its course first."""
    payload = _payload_locked(shutting_down=False)
    payload.update(
        {
            "remote_inference_active": False,
            "exited": True,
            "exit_code": exit_code,
            "outcome": outcome,
            "error": error,
            "hint": friendly_hint(error),
            "phase": phase,
            "returning_to_rest": False,
        }
    )
    return payload


def _fail_startup(error: str) -> None:
    """Record a pre-spawn failure as the terminal payload and go idle.

    A no-op once a stop has already torn the session down: the stop wins, and a
    preflight that raised while being abandoned must not resurrect a phantom
    failure. Mirrors rollout's `_fail_startup` so both inference modes surface
    a startup failure identically."""
    global _last_result
    with _state_lock:
        if not remote_inference_active:
            return
        _last_result = _terminal_payload_locked(
            exit_code=None, outcome="failed", error=error, phase=PHASE_ERROR
        )
        _go_idle_locked()
    notify_session_changed(KIND, False, phase=PHASE_ERROR)


def _finalise_exit_locked(rc: int | None) -> None:
    """Turn an observed child exit into the terminal payload. Holds the lock.

    A non-zero exit mines the real error out of the log (the child is a
    subprocess; the log is all the forensics there is), and `_classify_outcome`
    separates a run that worked but tripped a noisy shutdown warning from a
    real failure — the same distinction local inference makes, from the same
    function. An error recorded EARLIER (a watchdog, or the child's own ERROR
    event) wins over the log tail: it is the diagnosis, the exit is only its
    consequence."""
    global _last_result
    recorded = _remote_meta.get("error")
    error = recorded or (_extract_error_from_log(_remote_meta.get("log_path")) if rc else None)
    outcome = _classify_outcome(rc, _remote_running_started_at is not None, error)
    if recorded:
        # A recorded diagnosis means the session failed even if the child then
        # shut down cleanly at our request (rc 0) — the STOP was the remedy,
        # not the verdict.
        outcome = "failed"
    phase = PHASE_ERROR if (rc or error) else PHASE_STOPPED
    _last_result = _terminal_payload_locked(exit_code=rc, outcome=outcome, error=error, phase=phase)
    _go_idle_locked()
    # Final release. Under `_state_lock`; the notify is a lock-free droppable
    # queue put, so this cannot deadlock.
    notify_session_changed(KIND, False, phase=phase)


# --- watchdogs --------------------------------------------------------------


def _watchdog_failure_locked() -> str | None:
    """The empty-room verdict for right now, or None. Caller holds the lock.

    Split from the acting half so it can be tested against an injected clock
    with no process, no threads and no sleeps."""
    if not remote_inference_active:
        return None
    if _remote_meta.get("phase") not in _WATCHED_PHASES:
        return None
    now = _clock()
    if _active_at is None:
        if _connected_at is None or now - _connected_at < _ACTIVE_TIMEOUT_S:
            return None
        room = (_transport or {}).get("room", "the room")
        return (
            f"No policy joined room '{room}' within {_ACTIVE_TIMEOUT_S:.0f}s of connecting. "
            "Start the GPU side (modal run makermodslab/drtc/modal_policy.py), and check that "
            "its LiveKit-cloud secret names this same room."
        )
    if _chunks == 0 and now - _active_at >= _CHUNK_TIMEOUT_S:
        return (
            f"The policy is in the room but sent no action chunks in {_CHUNK_TIMEOUT_S:.0f}s. "
            "Portal drops every packet in silence when the two ends' schemas disagree — check that "
            f"the GPU side was launched with the same settings as this run ({_remote_meta.get('fingerprint', '')})."
        )
    return None


def _check_watchdogs() -> str | None:
    """Run both empty-room watchdogs; ask for a stop if either fired.

    Called from the stdout pump on every line (the child logs at 1 Hz for as
    long as its control loop runs, so both failure modes tick it) and from the
    status handler (so a poll is also a tick). Deliberately NOT a thread: there
    is nothing to watch that does not already wake one of those two.

    The stop is asked for by writing STOP and returning — never by calling the
    stop handler, which would block the pump that has to drain the child's
    stdout for the return to finish. The child returns the arm to rest, exits,
    and the pump's EOF path finalises with the recorded diagnosis."""
    with _state_lock:
        failure = _watchdog_failure_locked()
        if failure is None:
            return None
        _remote_meta["error"] = failure
        _remote_meta["hint"] = friendly_hint(failure)
        _remote_meta["phase"] = PHASE_STOPPING
        proc = _remote_proc
    logger.warning("Remote inference watchdog: %s", failure)
    _send_command(proc, CMD_STOP)
    notify_session_changed(KIND, True, phase=PHASE_STOPPING)
    return failure


# --- the child --------------------------------------------------------------


def _send_command(proc: subprocess.Popen | None, command: str) -> bool:
    """Write one command line to the child's stdin. True when it landed.

    Never raises: a dead child (BrokenPipeError) or a closed pipe is a state
    the caller handles, not an exception to unwind an HTTP handler with."""
    if proc is None or proc.stdin is None:
        return False
    try:
        proc.stdin.write(f"{command}\n".encode())
        proc.stdin.flush()
        return True
    except Exception as exc:
        logger.warning("Could not send %s to robot_sync: %s", command, exc)
        return False


def _spawn(cmd: list[str]) -> tuple[subprocess.Popen, IO[str], Path]:
    """Open a fresh log file and spawn the child with pipes on both ends.

    stdin STAYS OPEN — it is the command channel, and closing it is what would
    make a graceful stop impossible. `start_new_session=True` puts the child in
    its own process group so a teardown can reap the whole tree (see
    rollout's `_terminate_tree`, written after two orphaned runners survived
    SIGTERM by eight minutes)."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"{int(time.time())}.log"
    log_handle = log_path.open("w", buffering=1)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            log_handle.close()
        raise
    try:
        assert proc.stdin is not None
        proc.stdin.write(_stdin_seed())
        proc.stdin.flush()
    except Exception as exc:
        logger.warning("Failed to seed stdin for robot_sync: %s", exc)
    return proc, log_handle, log_path


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the child's output to the log and act on its protocol events.

    Lives for the whole session: the child does not exit between phases, so
    every transition arrives here as a line. EOF means the child is gone, which
    is where the run is finalised."""
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            try:
                log_handle.write(line)
                log_handle.flush()
            except Exception:
                pass
            try:
                _handle_line(line)
            except Exception:
                # One malformed event must not take the pump — and with it
                # every remaining transition — down with it.
                logger.exception("robot_sync event handling failed for %r", line.strip())
            try:
                _check_watchdogs()
            except Exception:
                logger.exception("Remote inference watchdog check failed")
    except Exception as exc:
        logger.exception("robot_sync stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()
        rc = _reap(proc)
        _handle_child_exit(proc, rc)


def _reap(proc: subprocess.Popen) -> int | None:
    """The child's exit code once stdout hit EOF; None if it will not be reaped.

    EOF means the process is already on its way out, so the wait is a
    formality — bounded only so a wedged exit cannot hang the pump thread."""
    try:
        return proc.wait(timeout=_STOP_WAIT_S)
    except Exception:
        logger.warning("robot_sync did not exit after its stdout closed; terminating")
        with contextlib.suppress(Exception):
            _terminate_tree(proc)
        return proc.poll()


def _handle_child_exit(proc: subprocess.Popen, rc: int | None) -> None:
    """Finalise the session on the pump's EOF path.

    A no-op when the stop handler (or a status poll) already finalised this
    child: whichever path arrives second finds `_remote_proc` cleared and
    leaves the recorded verdict alone."""
    with _state_lock:
        if not remote_inference_active or _remote_proc is not proc:
            return
        _finalise_exit_locked(rc)


def _handle_line(line: str) -> None:
    """Dispatch one line of the child's output."""
    parsed = parse_event(line)
    if parsed is None:
        return
    event, payload = parsed
    if event == EVENT_READY:
        _on_ready(parse_kv(payload))
    elif event == EVENT_CONNECTED:
        _on_connected()
    elif event == EVENT_EASING:
        _set_phase(PHASE_EASING)
    elif event == EVENT_ACTIVE:
        _on_active(parse_kv(payload))
    elif event == EVENT_STATS:
        _on_stats(parse_stats(payload))
    elif event == EVENT_RETURNING:
        _on_returning()
    elif event == EVENT_ERROR:
        _on_child_error(payload)
    elif event == EVENT_BYE:
        # Informational: the child is about to exit and the pump's EOF path is
        # a beat behind it. Nothing to do — finalising here would race that
        # path for the exit code it has and this does not.
        logger.info("robot_sync reported BYE")


def _on_ready(fields: dict[str, str]) -> None:
    """The child resolved its transport and is about to dial.

    READY echoes the EFFECTIVE url/room — what the child resolved, not what we
    believe we passed. A mismatch means the ground moved between the probe and
    the spawn (the local-SFU script restarted, a credential file rewritten), so
    it is a failure BEFORE the room is joined rather than a puzzle later."""
    global _transport
    with _state_lock:
        if not remote_inference_active:
            return
        expected = dict(_transport or {})
        url = fields.get("url", "")
        room = fields.get("room", "")
        mismatch = bool(expected) and (url != expected.get("url") or room != expected.get("room"))
        if _transport is not None:
            _transport = {**_transport, "url": url or _transport["url"], "room": room or _transport["room"]}
    if mismatch:
        _trigger_failure(
            f"The transport changed between the preflight and the run: verified "
            f"{expected.get('url')} room '{expected.get('room')}', the child resolved "
            f"{url} room '{room}'."
        )
        return
    _set_phase(PHASE_CONNECTING)


def _on_connected() -> None:
    """In the room. Arms the no-operator watchdog."""
    global _connected_at
    with _state_lock:
        if not remote_inference_active:
            return
        _connected_at = _clock()
    _set_phase(PHASE_WARMING_UP)


def _on_active(fields: dict[str, str]) -> None:
    """An operator claimed control. Arms the no-chunks watchdog."""
    global _active_at, _transport
    with _state_lock:
        if not remote_inference_active:
            return
        _active_at = _clock()
        if _transport is not None:
            _transport = {**_transport, "operator_present": True}
    logger.info("Remote inference operator active: %s", fields.get("operator"))


def _on_stats(sample: dict[str, Any] | None) -> None:
    """One 1 Hz telemetry sample; the first counted chunk means `running`.

    A malformed or truncated payload parses to None and is dropped — "no sample
    this second", never a half-populated status the UI would render as real."""
    global _stats, _chunks, _remote_running_started_at
    if sample is None:
        return
    with _state_lock:
        if not remote_inference_active:
            return
        _stats = sample
        chunks = sample.get("chunks")
        _chunks = chunks if isinstance(chunks, int) else _chunks
        started = _chunks > 0 and _remote_running_started_at is None
        if started:
            _remote_running_started_at = time.time()
        stopping = _remote_meta.get("phase") in (PHASE_STOPPING, PHASE_ERROR)
    if started and not stopping:
        _set_phase(PHASE_RUNNING)


def _on_returning() -> None:
    """The child is driving the arm back to its start pose, torque still on.

    Stays in the `stopping` phase deliberately (see the module docstring):
    a phase name of its own would fall outside
    `sessions._WINDING_DOWN_PHASES`, and an expiry tick landing during the
    return would dispatch a SECOND stop into it."""
    global _returning_to_rest
    with _state_lock:
        if not remote_inference_active:
            return
        _returning_to_rest = True
        _remote_meta["phase"] = PHASE_STOPPING
    notify_session_changed(KIND, True, phase=PHASE_STOPPING)


def _on_child_error(message: str) -> None:
    """The child reported a fatal error; it is already unwinding on its own."""
    with _state_lock:
        if not remote_inference_active:
            return
        _remote_meta.setdefault("error", message)
        _remote_meta.setdefault("hint", friendly_hint(message))
    logger.error("robot_sync error: %s", message)


def _trigger_failure(message: str) -> None:
    """Record a diagnosis and ask the child to stop, without blocking.

    Same discipline as `_check_watchdogs`: write STOP and let the child's own
    return-to-rest and exit carry the session to its terminal payload."""
    with _state_lock:
        if not remote_inference_active:
            return
        _remote_meta["error"] = message
        _remote_meta["hint"] = friendly_hint(message)
        _remote_meta["phase"] = PHASE_STOPPING
        proc = _remote_proc
    logger.error("Remote inference failing: %s", message)
    _send_command(proc, CMD_STOP)
    notify_session_changed(KIND, True, phase=PHASE_STOPPING)


# --- start ------------------------------------------------------------------


def _busy_refusal() -> dict[str, Any] | None:
    """The mutual-exclusion check: every peer that could hold the follower.

    All eight, plus this feature's own flag and the release grace — a new
    robot-driving feature must refuse against every existing one, and
    `tests/test_api_errors.py::test_busy_discriminants_cover_mutex_matrix` is
    an equality assertion, so it cannot be half-done. Caller holds
    `_state_lock`.

    Imported lazily for the reason every peer does it: those modules import
    this one back (their own reciprocal guard), and `jobs` imports it too."""
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        replay as _replay,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    if remote_inference_active:
        return {
            "success": False,
            "status_code": 409,
            "message": "Remote inference is already active. Stop it first.",
            "code": ErrorCode.ROBOT_BUSY_REMOTE_INFERENCE,
        }
    if _startup_thread is not None and _startup_thread.is_alive():
        # A previous session was stopped while its startup worker was inside
        # `_prepare_robot` (already touching hardware): the flag is False but
        # the worker still holds the port. Refuse until it is actually gone.
        return {
            "success": False,
            "status_code": 409,
            "message": "The previous session is still shutting down. Try again in a few seconds.",
            "code": ErrorCode.ROBOT_BUSY_RELEASING,
        }
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
    if _replay.replay_active:
        return {
            "success": False,
            "status_code": 409,
            "message": "Replay is currently active. Stop it first.",
            "code": ErrorCode.ROBOT_BUSY_REPLAY,
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
    # Lazy, because jobs imports this module back the same way. A remote run
    # wants only the USB bus, not the local GPU — refusing anyway is symmetric
    # with every other feature and cheap; "train locally while a Modal GPU
    # drives the arm" is a real capability, deliberately deferred.
    from . import jobs as _jobs

    if (training := _jobs.training_is_active()) is not None:
        return {
            "success": False,
            "status_code": 409,
            "message": f"Training run '{training}' is using this machine. Stop it first.",
            "code": ErrorCode.ROBOT_BUSY_TRAINING,
        }
    return None


def handle_start_remote_inference(request: RemoteInferenceRequest) -> dict[str, Any]:
    """Claim the follower, verify the transport, and hand the rest to a worker.

    Returns a dict the route/session layer turns into a response or an
    HTTPException. Everything in the ladder below is cheap and synchronous —
    including the room probe, which is bounded at 3 s — so a misconfigured
    launch is a 4xx in the panel rather than a session that energizes an arm
    and dies thirty seconds later. Only the arm-identity preflight and the
    spawn move off the request thread, exactly as local inference does.
    """
    global remote_inference_active, _remote_started_at, _remote_meta, _remote_cancel
    global _last_result, _startup_thread, _transport

    with _state_lock:
        refusal = _busy_refusal()
        if refusal is not None:
            return refusal
        # Claim the slot now so a concurrent caller losing the race sees us,
        # and seed the meta so the phase is visible from the first status poll.
        remote_inference_active = True
        _remote_started_at = time.time()
        _remote_cancel = threading.Event()
        cancel_event = _remote_cancel
        _remote_meta = {
            "phase": PHASE_RESOLVING,
            "policy_ref": request.policy_ref,
            "engine": request.engine,
            "duration_s": request.duration_s,
            "fingerprint": _fingerprint(request),
        }
        # A new run supersedes the previous run's terminal payload.
        _last_result = None
    notify_session_changed(KIND, True, phase=PHASE_RESOLVING)

    # 1. The optional extra. find_spec, never an import (FFI dylib).
    if _extra_missing():
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": "Remote inference needs the optional 'drtc' extra, which isn't installed. "
            + transport_hint(ErrorCode.TRANSPORT_EXTRA_MISSING),
            "code": ErrorCode.TRANSPORT_EXTRA_MISSING,
        }

    # 2. The transport — `resolve_transport()` and nothing else, so the
    #    preflight, the read-only transport endpoint and `modal_launcher` can
    #    never resolve three different rooms (see that function's docstring).
    try:
        resolved = resolve_transport()
    except (OSError, RuntimeError) as exc:
        _release_slot()
        logger.exception("The bundled SFU's key file could not be read")
        return {
            "success": False,
            "status_code": 400,
            "message": f"The Lab's SFU is running but its key file ({_sfu_key_file()}) "
            f"couldn't be read: {exc}",
            "code": ErrorCode.TRANSPORT_NOT_CONFIGURED,
        }
    if resolved.missing:
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"LiveKit credentials are incomplete (missing {', '.join(resolved.missing)}). "
            + transport_hint(ErrorCode.TRANSPORT_NOT_CONFIGURED),
            "code": ErrorCode.TRANSPORT_NOT_CONFIGURED,
        }
    url, room = resolved.url, resolved.room
    probe_key, probe_secret = resolved.api_key, resolved.api_secret
    child_token = resolved.child_token
    source = resolved.source

    # 3. The room probe: SFU reachable, credentials valid, a policy present.
    _set_phase(PHASE_TRANSPORT_CHECK)
    probe = _probe_room(url, room=room, key=probe_key, secret=probe_secret)
    if not probe.reachable:
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"Couldn't reach the LiveKit server at {url}. "
            + transport_hint(ErrorCode.TRANSPORT_UNREACHABLE, sfu=source == "sfu"),
            "code": ErrorCode.TRANSPORT_UNREACHABLE,
        }
    if not probe.authorized:
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"The LiveKit server at {url} rejected these credentials. "
            + transport_hint(ErrorCode.TRANSPORT_UNAUTHORIZED),
            "code": ErrorCode.TRANSPORT_UNAUTHORIZED,
        }
    if not probe.room_exists or not probe.operator_present:
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"No policy is in room '{room}'. "
            + transport_hint(ErrorCode.TRANSPORT_NO_POLICY, room=room),
            "code": ErrorCode.TRANSPORT_NO_POLICY,
        }
    with _state_lock:
        _transport = {
            "url": url,
            "room": room,
            "source": source,
            "operator_present": probe.operator_present,
        }

    # 4. Arm type and layout. Synchronous and PRE-SPAWN on purpose: the CAN
    #    followers are not registered with draccus in the child, so the failure
    #    it would otherwise be is a CLI-parse error inside a process the
    #    session has already claimed and preflighted an arm for.
    _set_phase(PHASE_PREFLIGHT)
    if not supports_remote_inference(request.arm_type, request.mode):
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": (
                "Remote inference runs on a single-arm SO-101 for now. The Maker and Metal "
                "followers aren't registered in the remote-inference entrypoint (a CAN arm "
                "would fail at start-up with the arm already claimed), and a bimanual robot "
                "has no first-action ease-in there, so its first move would be a full-speed "
                "snap to the policy's pose."
            ),
        }

    # 5. Arm count and cameras — reused verbatim from local inference.
    mismatch = _arm_count_mismatch(request.mode, request.checkpoint_state_dim, request.arm_type)
    if mismatch is not None:
        _release_slot()
        return {"success": False, "status_code": 409, "message": mismatch}
    # Before the record lookup, because this one needs no hardware and no robot
    # record at all: a role the CHECKPOINT does not have (and cannot be given)
    # is wrong however well the robot answers to it.
    role_refusal = _extra_camera_role_refusal(request)
    if role_refusal is not None:
        _release_slot()
        logger.warning("Rejected remote inference start: %s", role_refusal)
        return {"success": False, "status_code": 400, "message": role_refusal}

    robot_request = _robot_request(request)
    try:
        _session_cameras(robot_request)
    except CameraResolutionError as exc:
        _release_slot()
        logger.warning("Rejected remote inference start: %s", exc)
        return {"success": False, "status_code": 400, "message": str(exc)}

    # Backend camera previews hold the cv2 devices the child is about to open.
    camera_preview_manager.stop_all()

    worker = threading.Thread(
        target=_run_startup,
        args=(request, robot_request, url, room, child_token, cancel_event),
        name="remote-inference-startup",
        daemon=True,
    )
    _startup_thread = worker
    worker.start()
    return {"success": True, "message": "Remote inference starting"}


def _run_startup(
    request: RemoteInferenceRequest,
    robot_request: InferenceRequest,
    url: str,
    room: str,
    token: str,
    cancel_event: threading.Event,
) -> None:
    """Preflight the arm, then spawn the child. Runs off the request thread.

    Ordered preflight → spawn, and the cancel flag is re-checked at each
    boundary, so a stop pressed in this window never leaves a live child
    driving the arm. Nothing interrupts `_prepare_robot` mid-call (it is
    already touching the bus), which is why a start refuses while a previous
    worker is still alive."""
    global _remote_proc, _remote_meta

    try:
        robot_args, identity_warnings = _prepare_robot(robot_request)
    except ArmIdentityError as exc:
        # Already user-facing: the connected arm doesn't match its calibration.
        _fail_startup(str(exc))
        return
    except Exception as exc:
        logger.exception("Failed to prepare the robot for remote inference")
        _fail_startup(f"Failed to start remote inference: {exc}")
        return
    if cancel_event.is_set():
        logger.info("Remote inference startup abandoned after the preflight (stop requested)")
        return

    _set_phase(PHASE_STARTING)
    cmd = [
        sys.executable,
        "-m",
        _child_module(request.engine),
        *_robot_sync_args(request, robot_args, url=url, room=room, token=token),
    ]
    try:
        proc, log_handle, log_path = _spawn(cmd)
    except Exception as exc:
        logger.exception("Failed to spawn the %s remote-inference child", request.engine)
        _fail_startup(f"Failed to start remote inference: {exc}")
        return

    # Commit under the lock, re-checking the cancel flag: a stop that raced the
    # spawn must NOT leave a live child driving the arm.
    with _state_lock:
        abandoned = cancel_event.is_set() or not remote_inference_active
        if not abandoned:
            _remote_proc = proc
            _remote_meta["log_path"] = str(log_path)
            if identity_warnings:
                _remote_meta["warning"] = " ".join(identity_warnings)
    if abandoned:
        logger.info("Remote inference startup abandoned after the spawn; killing the child")
        _terminate_tree(proc, timeout=_PRE_SPAWN_TERMINATE_TIMEOUT_S)
        with contextlib.suppress(Exception):
            log_handle.close()
        return

    threading.Thread(
        target=_pump_stdout,
        args=(proc, log_handle),
        name="remote-inference-stdout-pump",
        daemon=True,
    ).start()
    logger.info("Remote inference started: pid=%s room=%s", proc.pid, room)


# --- stop / status ----------------------------------------------------------


def handle_stop_remote_inference() -> dict[str, Any]:
    """Stop the session: STOP on stdin, the arm home, then torque off.

    Never a signal. The child owns the bus, and a SIGTERM would run its
    `finally:` from wherever the policy left the arm — survivable on an SO-101,
    a fall on a CAN arm. `STOP` makes it return to the pose it captured at
    connect first; a SECOND stop while that return is in flight writes STOP
    again, which the child reads as "cut it short" and releases torque where
    the arm is — nearer rest than it started.

    The lease's expiry watchdog reaches this through `sessions._dispatch_stop`,
    so nothing extra is needed for an abandoned session."""
    with _state_lock:
        session_active = remote_inference_active
        orphaned_worker = _startup_thread if not session_active else None

    if not session_active:
        if orphaned_worker is None or not orphaned_worker.is_alive():
            return {"success": False, "status_code": 409, "message": "No remote inference is active"}
        # The "press Stop again" gesture: a previous stop already fired but its
        # startup worker is still inside `_prepare_robot`, with no way to be
        # interrupted mid-call. Joined outside the lock so a stuck worker can't
        # stall status polls for the whole timeout.
        orphaned_worker.join(timeout=_STARTUP_STOP_JOIN_TIMEOUT_S)
        if orphaned_worker.is_alive():
            return {
                "success": True,
                "shutting_down": True,
                "message": (
                    "The previous session is still shutting down "
                    f"(waited {_STARTUP_STOP_JOIN_TIMEOUT_S:.0f}s more). Try again shortly."
                ),
            }
        return {"success": True, "message": "The previous session has now finished shutting down."}

    with _state_lock:
        if _remote_cancel is not None:
            # The only way to abandon the pre-spawn window (arm preflight),
            # where there is no process to talk to.
            _remote_cancel.set()
        proc = _remote_proc
        second_press = _remote_meta.get("phase") == PHASE_STOPPING
        if _remote_meta:
            _remote_meta["phase"] = PHASE_STOPPING
        if proc is None:
            # Stopped before the child spawned: nothing has been energized by
            # the policy, and the orphaned worker bails at its next cancel
            # check. Nothing to terminate.
            _go_idle_locked()
            notify_session_changed(KIND, False, phase=PHASE_STOPPED)
            return {"success": True, "message": "Remote inference stopped"}
    notify_session_changed(KIND, True, phase=PHASE_STOPPING)

    if second_press:
        # A stop is already in flight and the first caller is inside the wait
        # below. This STOP is the abort: hand it over and return at once rather
        # than queueing a second bounded wait behind the first.
        _send_command(proc, CMD_STOP)
        return {"success": True, "message": "Cutting the return to rest short"}

    _stop_child_and_wait(proc)
    return {"success": True, "message": "Remote inference stopped"}


def _stop_child_and_wait(proc: subprocess.Popen) -> None:
    """STOP the child, wait out its return-to-rest, escalate, then finalise.

    Factored out of :func:`handle_stop_remote_inference` so
    :func:`stop_for_shutdown` drives the SAME sequence rather than a parallel
    one — a second implementation of "make the arm safe" is exactly the kind of
    duplication that ends with one of the two paths quietly missing the
    escalation."""
    if not _send_command(proc, CMD_STOP):
        # A child that cannot be talked to cannot return the arm either;
        # terminate the tree rather than wait out the ceiling for nothing.
        logger.warning("robot_sync did not accept STOP; terminating the process group")
        _terminate_tree(proc)
    else:
        _await_child_exit(proc)
    _finalise_if_ours(proc)


def _await_child_exit(proc: subprocess.Popen) -> None:
    """Block until the child is gone, bounded by `_STOP_WAIT_S`, then escalate.

    The bound is the return's own ceiling plus teardown; past it the whole
    process group is terminated, which is the escalation the second STOP press
    would otherwise have to be."""
    try:
        proc.wait(timeout=_STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        logger.warning(
            "robot_sync did not exit %.0fs after STOP; terminating the process group",
            _STOP_WAIT_S,
        )
        _terminate_tree(proc)
    except Exception as exc:
        logger.exception("Waiting for robot_sync to stop failed: %s", exc)


def _finalise_if_ours(proc: subprocess.Popen) -> None:
    """Record the terminal payload and emit the release — unless someone did.

    Re-reads under the lock: the pump's EOF path may have finalised this exit
    while the caller was waiting outside it, and whichever path arrives second
    must leave the recorded verdict alone."""
    with _state_lock:
        if remote_inference_active and _remote_proc is proc:
            _finalise_exit_locked(proc.poll())


def stop_for_shutdown() -> bool:
    """Stop a live session on the way out — synchronously, bounded.

    The shutdown twin of :func:`handle_stop_remote_inference`, and the reason it
    exists is that this is the ONE robot-driving flow whose arm is held by a
    child process that outlives us: `_spawn` gives it `start_new_session=True`,
    so a SIGTERM/SIGINT aimed at this worker never reaches it, and stdin EOF is
    ignored by design (`drtc_protocol.pump_commands`). Without this, a uvicorn
    `--reload` save, a Ctrl-C or a `makermodslab --stop` during a remote run
    left the child driving an energized arm with nobody able to reach it from
    the API — against this repo's contract that stopped means de-energized and
    that every flow returns the arm before releasing torque.

    Returns whether there was anything to stop. Never raises (it is called from
    a shutdown handler), and it emits the session_events release the normal
    stop does, because it IS the normal stop: the STOP goes on the child's
    stdin, the child returns the arm to its captured start pose, and only then
    is torque released and BYE written.

    Bounded by `_STOP_WAIT_S` for the return, plus at most
    `_STARTUP_STOP_JOIN_TIMEOUT_S` for a startup worker still inside
    `_prepare_robot` (it holds the serial bus, and it cannot be interrupted
    mid-call — the same bounded-wait-and-report the "press Stop again" gesture
    makes). Typical cost is the two or three seconds a real return-to-rest
    takes; zero when nothing is running.
    """
    with _state_lock:
        active = remote_inference_active
        proc = _remote_proc
        # A stop is already in flight on another thread (a Stop press, or a
        # watchdog). Pressing Stop again is the ABORT gesture — it would cut
        # that return short, which is the opposite of what a shutdown wants
        # while it can still afford to wait. So we wait it out instead.
        in_flight = _remote_meta.get("phase") == PHASE_STOPPING
        worker = _startup_thread

    if not active and (worker is None or not worker.is_alive()):
        return False

    logger.info("Stopping the remote-inference session before shutdown")
    try:
        if active and proc is not None and in_flight:
            _await_child_exit(proc)
            _finalise_if_ours(proc)
        elif active:
            handle_stop_remote_inference()
    except Exception:
        logger.exception("Failed to stop remote inference during shutdown")

    if worker is not None and worker.is_alive():
        # Not the child: the pre-spawn preflight, which has the follower's
        # serial port open. Leaving while it does is how the NEXT boot finds
        # the bus busy.
        worker.join(timeout=_STARTUP_STOP_JOIN_TIMEOUT_S)
    return True


def handle_remote_inference_status() -> dict[str, Any]:
    """The status dict — see the module docstring for the exact key list.

    Also a watchdog tick: a poll is one of the two things that can notice an
    empty room, and it costs a clock read."""
    _check_watchdogs()
    with _state_lock:
        # True only while idle: a stopped session's startup worker is still
        # alive, so a poller is not shown something indistinguishable from true
        # idle while that worker still holds the serial bus.
        shutting_down = (
            not remote_inference_active and _startup_thread is not None and _startup_thread.is_alive()
        )
        proc = _remote_proc
        if proc is None and not remote_inference_active and _last_result is not None:
            return {**_last_result, "shutting_down": shutting_down}
        if proc is not None and proc.poll() is not None:
            # Backstop for the pump, which normally notices EOF first.
            _finalise_exit_locked(proc.returncode)
            return {**_last_result, "shutting_down": shutting_down}
        return _payload_locked(shutting_down=shutting_down)


def handle_remote_inference_transport() -> dict[str, Any]:
    """What a child would resolve RIGHT NOW, and whether anything answers there.

    The read-only half of the transport surface (`GET
    /api/v1/remote-inference/transport`). Every key is ALWAYS present — the two
    probe-shaped ones are null when the probe DID NOT RUN, which is a third
    state distinct from "false": "we never asked" and "we asked and the SFU is
    down" have nothing in common as remedies.

    Two transports, reported through one shape. When this process runs the
    Lab's own SFU (`makermodslab --sfu`) the url, room and credentials are
    resolved in-process from the 0600 key file and `livekit.env` is not read at
    all; otherwise the LiveKit Cloud credentials are read afresh (never cached,
    never `load_env`) so an edited file takes effect on the next call.

    The `sfu_*` block is what the panel needs to write the OTHER terminal's
    `modal run` line: the key NAME (an identifier, not a secret), the file the
    secret is in — so a human can read it — and the tailnet URL a container can
    actually dial. **The API secret itself is never returned.** A status
    endpoint that hands out a signing key is a credential leak wearing a
    diagnostic hat.

    Synchronous on purpose. `_probe_room` runs `asyncio.run` internally, which
    RAISES inside a running loop — as a plain `def` this handler is dispatched
    to FastAPI's threadpool, which has no loop of its own, exactly the contract
    `_probe_room`'s docstring states. An `async def` here would turn every
    configured call into a 500.
    """
    enabled = sfu.sfu_enabled()
    payload: dict[str, Any] = {
        "extra_installed": True,
        "configured": False,
        "missing_vars": list(_REQUIRED_ENV),
        "url": "",
        "room": "",
        "source": "none",
        "sfu_enabled": enabled,
        "sfu_url": None,
        "sfu_modal_url": None,
        "sfu_external_ip": sfu.external_ip_enabled(),
        "sfu_key_id": None,
        "sfu_key_file": _sfu_key_file() if enabled else None,
        # Present ONLY when the binary is missing: an install hint beside a
        # working SFU is noise, and its absence is how the panel knows not to
        # offer the "start it with --sfu" line as if it were achievable.
        "sfu_install_hint": (None if sfu.find_livekit_server() else sfu.install_hint(platform.system())),
        "endpoint_reachable": None,
        "operator_present": None,
        "error_code": None,
        "message": None,
    }

    if _extra_missing():
        # Reported, never raised: the panel's job here is to tell the user what
        # to install — and the install command must name the PRIMARY checkout
        # (transport_hint owns that wording).
        payload["extra_installed"] = False
        payload["error_code"] = str(ErrorCode.TRANSPORT_EXTRA_MISSING)
        payload["message"] = "The optional 'drtc' extra isn't installed. " + transport_hint(
            ErrorCode.TRANSPORT_EXTRA_MISSING
        )
        return payload

    # The SAME resolution the session's preflight runs and the same one
    # `modal_launcher` builds its command from — one function, so this page can
    # never describe a transport a run would not actually use.
    try:
        resolved = resolve_transport()
    except (OSError, RuntimeError) as exc:
        payload["error_code"] = str(ErrorCode.TRANSPORT_NOT_CONFIGURED)
        payload["message"] = (
            f"The Lab's SFU is running but its key file ({_sfu_key_file()}) couldn't be read: {exc}"
        )
        return payload

    payload |= {
        "configured": not resolved.missing,
        "missing_vars": list(resolved.missing),
        "url": resolved.url,
        "room": resolved.room,
        "source": resolved.source,
    }
    if enabled:
        payload |= {
            "sfu_url": resolved.url,
            "sfu_modal_url": sfu_modal_url(),
            # The key ID, never the secret. It is the `--livekit-api-key` half
            # of the Modal line and it identifies rather than authorizes.
            "sfu_key_id": resolved.api_key,
        }
    if resolved.missing:
        payload["error_code"] = str(ErrorCode.TRANSPORT_NOT_CONFIGURED)
        payload["message"] = (
            f"LiveKit credentials are incomplete (missing {', '.join(resolved.missing)}). "
            + transport_hint(ErrorCode.TRANSPORT_NOT_CONFIGURED)
        )
        return payload
    key, secret = resolved.api_key, resolved.api_secret

    url, room, source = payload["url"], payload["room"], payload["source"]
    probe = _probe_room(url, room=room, key=key, secret=secret)
    payload["endpoint_reachable"] = probe.reachable
    payload["operator_present"] = probe.operator_present
    if not probe.reachable:
        payload["error_code"] = str(ErrorCode.TRANSPORT_UNREACHABLE)
        payload["message"] = f"Couldn't reach the LiveKit server at {url}. " + transport_hint(
            ErrorCode.TRANSPORT_UNREACHABLE, sfu=source == "sfu"
        )
    elif not probe.authorized:
        payload["error_code"] = str(ErrorCode.TRANSPORT_UNAUTHORIZED)
        payload["message"] = f"The LiveKit server at {url} rejected these credentials. " + transport_hint(
            ErrorCode.TRANSPORT_UNAUTHORIZED
        )
    elif not probe.operator_present:
        payload["error_code"] = str(ErrorCode.TRANSPORT_NO_POLICY)
        payload["message"] = f"No policy is in room '{room}'. " + transport_hint(
            ErrorCode.TRANSPORT_NO_POLICY, room=room
        )
    else:
        payload["message"] = _ready_message(url, room, source)
    return payload


def _ready_message(url: str, room: str, source: str) -> str:
    """The success line: what resolved, and which of the two transports it is.

    Naming the transport on the HAPPY path matters as much as on the
    unreachable one. Under the Lab's own SFU everything a peer needs is minted
    here and the GPU side has to be pointed at THIS machine explicitly; on
    LiveKit Cloud both ends read their own credentials and the room is the only
    thing that has to agree. An operator reading "it works" without knowing
    which of those they are in cannot tell what to change when it stops
    working."""
    if source == "sfu":
        return (
            f"A policy is in room '{room}' at {url} — the Lab's own SFU. The GPU side must be "
            "launched with the --livekit-url/--livekit-api-key/--livekit-api-secret from the "
            "command above."
        )
    return f"A policy is in room '{room}' at {url} (LiveKit Cloud credentials from livekit.env)."
