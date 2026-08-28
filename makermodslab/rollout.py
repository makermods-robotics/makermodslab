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

"""Inference mode: drives the SO-101 follower with a trained policy.

Mirrors `app/teleoperating.py` in shape — single global session, mutex
with teleoperation/recording (the follower's serial bus can only be
opened once), `lerobot.scripts.lerobot_rollout` running as a subprocess
for clean cancellation. Hub-checkpoint refs are resolved to a local dir
via huggingface_hub.snapshot_download before we spawn the subprocess.

Two subprocess shapes, chosen by `eval_episodes`:

  1 (the default) — one `lerobot-rollout`, exactly as it has always been.
      Everything below that mentions "the subprocess" means this one.
  >1 (EVAL mode)  — one `makermodslab.eval_runner` for the WHOLE session. It
      loads the policy and connects the bus and cameras once, then runs an
      episode per `EPISODE` line on its stdin. Spawning a rollout per episode
      instead re-paid a 15-40 s policy load and a full reconnect every time —
      5-10 minutes of dead time across a 20-episode eval — so this module
      keeps ONE process alive and talks to it (`makermodslab.eval_protocol`)
      rather than starting and killing N of them. Episode boundaries then
      arrive as stdout events rather than as process exits, which is why eval
      mode has its own pump (`_pump_runner_stdout`) and why a runner exit is
      read as a crash to contain, not as an episode that ended.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from .api_errors import ErrorCode
from .arm_identity import ArmIdentityError, ArmSlot, verify_devices
from .camera_preview import camera_preview_manager
from .dagger_protocol import (
    CANCEL_REASON_OPERATOR,
    CANCEL_REASON_TOO_SHORT,
    CMD_QUIT as DAGGER_CMD_QUIT,
    COMMANDS as DAGGER_COMMANDS,
    EVENT_ALIGN_REQUIRED,
    EVENT_ATTEMPT_RESET,
    EVENT_CORRECTION_CANCELLED,
    EVENT_CORRECTION_SAVED,
    EVENT_DATASET,
    EVENT_ERROR as DAGGER_EVENT_ERROR,
    EVENT_PHASE,
    EVENT_READY as DAGGER_EVENT_READY,
    PHASE_CORRECTING as DAGGER_PHASE_CORRECTING,
    PHASE_HANDING_OVER as DAGGER_PHASE_HANDING_OVER,
    PHASE_PAUSED as DAGGER_PHASE_PAUSED,
    PHASE_RESETTING as DAGGER_PHASE_RESETTING,
    PHASE_SAVING as DAGGER_PHASE_SAVING,
    PHASES as DAGGER_PHASES,
    parse_event as parse_dagger_event,
    parse_fields as parse_dagger_fields,
)
from .eval_protocol import (
    CMD_EPISODE,
    CMD_QUIT,
    CMD_STOP,
    EVENT_EPISODE_ENDED,
    EVENT_EPISODE_STARTED,
    EVENT_ERROR,
    EVENT_READY,
    REASON_STOPPED,
    parse_episode_end_reason,
    parse_event,
)
from .jobs import download_hub_checkpoint_ref, make_snapshot_progress_tqdm
from .models import (
    _downloaded_model_dir,
    _has_loadable_weights,
    _hub_cache_has_repo,
    _resolve_pretrained_dir,
)
from .motor_power import clear_goal_velocity, reset_torque_limit
from .record import _DEFAULT_FOURCC
from .session_events import notify_session_changed
from .utils.config import (
    LEADER_CONFIG_PATH,
    CameraResolutionError,
    bimanual_base_id,
    bind_robot_cameras,
    list_robot_records,
    setup_calibration_files,
    setup_follower_calibration_file,
    stage_bimanual_calibrations,
    stage_bimanual_follower_calibrations,
    validate_dataset_repo_id,
)
from .utils.errors import friendly_hint, is_cleanup_error

logger = logging.getLogger(__name__)

# Flat proprioceptive state width of a single SO-101 follower arm (one dim per
# joint). A bimanual checkpoint trains on two arms → twice this. The frontend
# forwards the checkpoint's state_dim (from /policy-config) so the server can
# reject an arm-count mismatch BEFORE spawning the rollout subprocess, instead
# of letting the shape mismatch crash deep inside it.
_SINGLE_ARM_STATE_DIM = 6


class PolicyCameraDims(BaseModel):
    """The frame size one of the checkpoint's cameras was trained on.

    Forwarded from /policy-config's `image_features` by the launch UI, in the
    same spirit as `checkpoint_state_dim` below: the client already has the
    checkpoint's metadata, and echoing it lets the server apply it
    authoritatively instead of guessing. See bind_robot_cameras for why capture
    resolution must come from the checkpoint rather than the robot record.
    """

    width: int
    height: int


class InferenceRequest(BaseModel):
    follower_port: str
    follower_config: str
    policy_ref: str  # opaque ref returned by /jobs/{id}/checkpoints
    task: str = ""
    # Which of the ROBOT RECORD's cameras plays each camera role the checkpoint
    # was trained with: {policy-expected camera name: robot-record camera name}.
    # Only the name pairing travels in the request — which device, and how it's
    # opened (index, unique_id, fps, fourcc, backend), is read server-side from
    # the record, so a run can never open a camera set the saved robot doesn't
    # have. (Capture resolution is the one exception: see `camera_dims` below.)
    # Empty ⇒ a camera-less policy. Replaces the former `cameras`
    # dict of full per-camera configs; pydantic ignores unknown fields, so an
    # older frontend's payload parses and its camera configs are ignored.
    camera_bindings: dict[str, str] = {}
    # Capture resolution per policy-expected camera name, from the checkpoint's
    # image_features. The one camera setting NOT taken from the robot record:
    # lerobot's standard rollout does not resize frames to the policy's input
    # shape, so capturing at the record's configured size would silently feed
    # the policy frames it was never trained on. Keyed to match camera_bindings;
    # a camera with no entry here (older client, or a checkpoint that doesn't
    # expose image dims) falls back to the record's own width/height.
    camera_dims: dict[str, PolicyCameraDims] = {}
    duration_s: int = 60
    # Bimanual: the follower_port/follower_config above is the LEFT arm; these
    # add the RIGHT arm.
    mode: str = "single"
    right_follower_port: str = ""
    right_follower_config: str = ""
    # LEADER arms. Blank for every non-coaching run: a plain rollout and an eval
    # drive the followers from the policy and have no use for a leader, which is
    # why inference carried no leader fields at all until coaching arrived. A
    # COACHING session is the exception — the whole point is that a human takes
    # over through the leader — so these are required when `coaching` is true
    # (and the right_* pair additionally when mode is bimanual).
    leader_port: str = ""
    leader_config: str = ""
    right_leader_port: str = ""
    right_leader_config: str = ""
    # Robot record name. Two jobs:
    #   1. It names the record `camera_bindings` resolve against (required
    #      whenever any binding is set — a missing record is a 400).
    #   2. Bimanual only: the BiSO staging base id — it decides the on-disk
    #      staging dir, not which calibration drives which arm. Blank/invalid
    #      falls back to DEFAULT_BIMANUAL_BASE.
    robot_name: str = ""
    # Flat state width of the selected checkpoint (6 = single SO-101 arm, 12 =
    # bimanual), forwarded from /policy-config so the server can reject an
    # arm-count mismatch pre-spawn. None when the checkpoint omits the feature —
    # the guard then defers to the rollout subprocess's own shape check.
    checkpoint_state_dim: int | None = None
    # Escape hatch for the arm-identity guard (see makermodslab/arm_identity.py):
    # when true, run even if the connected arm doesn't match its calibration.
    skip_identity_check: bool = False
    # Multi-episode EVALUATION mode. 1 (the default) is exactly the historical
    # single-rollout flow — no episode bookkeeping, no reset phase, no accuracy.
    # >1 walks N sequential rollout subprocesses inside ONE session (one model
    # download, one arm preflight, one camera handover), scoring each episode.
    # Clamped server-side to [1, MAX_EVAL_EPISODES] — see clamp_eval_episodes.
    eval_episodes: int = 1
    # Which lerobot inference engine drives the rollout (--inference.type).
    # "sync" is lerobot's own default and stays ours: one policy forward per
    # control tick, so a 50-action SmolVLA chunk replays for ~1.6s and the loop
    # then stalls ~430ms on MPS computing the next one. "rtc" (Real-Time
    # Chunking) moves that forward onto a background thread and blends the new
    # chunk onto the previous chunk's leftover prefix, removing the stall — but
    # it also routes flow-matching through the RTC processor, a DIFFERENT
    # action-generation path than the one a checkpoint was evaluated under.
    # Experimental on purpose: A/B it per run, don't assume equivalence.
    inference_engine: Literal["sync", "rtc"] = "sync"
    # ACT temporal ensembling (see _rollout_cli_args). None = off, which is
    # also lerobot's own default. Any positive coefficient turns it on; the
    # original ACT paper uses 0.01. Only ACT checkpoints have this field in
    # their config, so the UI offers it for `policy_type == "act"` alone.
    temporal_ensemble_coeff: float | None = None
    # COACHING (DAgger / HG-DAgger) mode — the third session shape, alongside
    # the single rollout and the multi-episode eval. The policy drives; the
    # operator takes over through the leader arm when it is about to fail; each
    # takeover is recorded as one episode of a new dataset. Mutually exclusive
    # with eval mode (`eval_episodes > 1`) — a coaching session has no episode
    # verdicts to tally, and an eval has no leader to take over with.
    coaching: bool = False
    # How many corrections to collect before the session ends on its own.
    # Clamped server-side to [1, MAX_COACHING_CORRECTIONS].
    target_corrections: int = 10
    # Dataset name for the corrections, WITHOUT the owner or the mandatory
    # `rollout_` prefix — both are applied server-side (see
    # `_coaching_dataset_repo_id`). lerobot then appends its own timestamp, so
    # the name on disk is discovered from the runner rather than predicted here.
    coaching_dataset_name: str = ""


inference_active: bool = False
_inference_proc: subprocess.Popen | None = None
_inference_started_at: float | None = None
_inference_rollout_started_at: float | None = None
_inference_meta: dict[str, Any] = {}
# The finished (exited) status payload of the most recent run, kept until the
# NEXT start claims the slot. Terminal outcomes must be idempotent, not
# consume-once: several surfaces poll /inference-status concurrently (the
# session dialog at 1 Hz, the Deploy panel at 0.5 Hz), and with a
# report-once-then-clear scheme whichever poll lands first after the subprocess
# dies swallows the outcome/error/hint — the dialog then sees a bare idle
# status and misreports a crash as a clean finish.
_last_result: dict[str, Any] | None = None
# Log file of the most recent run that actually SPAWNED a subprocess, kept until
# the next start claims the slot. Same lifecycle as `_last_result` above and for
# the same reason: a finished run's log must stay readable while the dialog is
# still showing its terminal state.
#
# It exists because the log used to be found by globbing the newest `*.log` out
# of the inference_logs dir whenever the active meta had no path — which is true
# in two ordinary windows: during a new session's pre-spawn phases (download /
# preflight), and after a run that FAILED before spawning (the startup error
# wipes the meta, so no path is ever committed). In both, the glob served a
# previous run's log as though it were this one, unlabelled. Observed live: a run
# that failed in `_prepare_robot` on a calibration error produced no log at all,
# and the user was shown a three-day-old RTC run's output — they reasonably
# concluded their sync run was executing RTC code.
#
# Binding log identity to the session lifecycle instead means the endpoint can
# only ever return THIS process's own runs, and can say which.
_last_log_path: str | None = None
# Set for the CURRENT session at claim time; the background startup worker
# captures its own reference and stop() sets it. It's the only way to abandon a
# start that's still in its pre-subprocess window (Hub download / arm preflight),
# where there's no process to terminate. A fresh Event per session means an
# orphaned worker from a stopped session sees its (set) event and bails, while a
# new session gets a clean one. None while idle.
_inference_cancel: threading.Event | None = None
# Handle to the background startup worker (_run_inference_startup) for the
# CURRENT/most-recently-started session. `_inference_cancel` only aborts the
# worker at coarse boundaries (before it opens the bus in _prepare_robot,
# and again after _prepare_robot returns) — nothing interrupts it WHILE
# _prepare_robot is actually touching hardware, so a stop can leave the
# worker alive and still driving the bus for a few more seconds. Tracking the
# thread (mirrors teleoperate.py's `teleoperation_thread`, added for the same
# reason in T3) lets handle_start_inference refuse a new session while that
# orphaned worker is still alive, instead of racing it for the same serial
# port. None once the worker has exited or before any session has started.
_inference_startup_thread: threading.Thread | None = None
# Multi-episode evaluation bookkeeping for the CURRENT session lives in
# `_eval_session`, declared just below the _EvalSession dataclass.
# Guards mutations to the globals above (and _eval_session); held only for the
# short critical sections in start/stop/status.
_state_lock = threading.Lock()
# Bound on how long a second stop-inference call waits for an orphaned startup
# worker (see _inference_startup_thread) to exit before giving up and
# reporting it's still alive. Mirrors teleoperate.py's second-stop join
# timeout; unlike that one, this can't force the worker out mid-call (no
# cooperative cancellation checkpoint inside _prepare_robot), so it's a
# bounded wait-and-report rather than a true force-release.
_STARTUP_STOP_JOIN_TIMEOUT_S = 5.0
# The two hub-ref shapes /jobs/{id}/checkpoints hands out. Kept here for the
# cheap no-network shape check below; the DOWNLOAD they imply lives once, in
# jobs.download_hub_checkpoint_ref, so inference and fine-tune resolve a ref
# identically.
_HUB_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")
# lerobot prints this once per run, the moment its main control loop is
# about to take over from the setup phase. We watch stdout for it so the
# UI can present a "rollout time" separate from the multi-second policy
# load + bus connect + camera connect setup overhead.
_ROLLOUT_START_MARKER = "Rollout setup complete"

# Structured "which substep am I in" for the startup sequence, surfaced in the
# /inference-status payload so the UI can name the wait ("Downloading model…",
# "Connecting to arm…") instead of a single opaque spinner. Ordered:
#   downloading_model — snapshot_download of a Hub checkpoint (server thread,
#       BEFORE the subprocess spawns). Skipped for a local checkpoint dir.
#   starting          — subprocess spawned, before any recognised setup line.
#   loading_policy    — lerobot's context.py "Loading policy from ..." emitted.
#   connecting        — lerobot's "Connecting robot ..." emitted (the bus- and
#       camera-connect window; both open inside robot.connect()).
#   running           — the rollout main loop has taken over (marker seen).
#   stopping/stopped/error — terminal, set by stop/status finalisation.
# There is no `downloading_dataset` phase: the base-strategy rollout command we
# build passes no --dataset, so build_rollout_context never sets up (or
# downloads) a dataset. We omit the phase rather than invent one that never
# fires.
#
# EVAL-ONLY phases (eval_episodes > 1). The setup phases above run ONCE per
# session, not once per episode: eval mode drives a single long-lived
# `makermodslab.eval_runner` subprocess that loads the policy and connects the
# robot one time and then runs episode after episode on command (they DO run
# again after a crash-respawn, which is a genuine second load). Episode two
# onwards therefore goes straight from `starting` to `running`. Plus:
#   resetting — an episode ended, the tally was updated, and the session is
#       parked waiting for the user to rearrange the scene and POST
#       /inference-next-episode. Also where a CRASHED episode parks, with
#       `error`/`hint` populated so the user can continue or abort.
#   finished  — every episode ran; terminal, carries `accuracy`.
#   aborted   — /stop-inference ended the session early; terminal, partial
#       tally, NO accuracy claimed.
PHASE_DOWNLOADING_MODEL = "downloading_model"
PHASE_STARTING = "starting"
PHASE_LOADING_POLICY = "loading_policy"
PHASE_CONNECTING = "connecting"
PHASE_RUNNING = "running"
PHASE_STOPPING = "stopping"
PHASE_STOPPED = "stopped"
PHASE_ERROR = "error"
PHASE_RESETTING = "resetting"
PHASE_FINISHED = "finished"
PHASE_ABORTED = "aborted"

# COACHING-ONLY phases. Like the eval phases above, the setup ladder
# (downloading_model → loading_policy → connecting) runs once per session; these
# three then replace `running` for the rest of it, because "running" is not a
# useful thing to tell an operator who needs to know, at a glance, whether the
# ROBOT or THEY are currently driving. They map 1:1 onto lerobot's DAggerPhase
# (autonomous / paused / correcting) and are translated in `_on_dagger_phase`.
#   watching   — the policy is driving, the operator is watching for a failure
#   holding    — frozen: the policy is paused and the arm holds its pose
#   correcting — the operator is driving through the leader; frames are recorded
# The arm is physically travelling into position for a handover — the one
# phase that is ours rather than lerobot's. See PHASE_HANDING_OVER in
# dagger_protocol for why a distinct state is required and not cosmetic.
PHASE_HANDING_OVER = "handing_over"
# The correction is being written to disk (parquet + video encode). Synchronous
# on the runner's control loop, so the arm is frozen and the policy has not
# resumed — the operator is waiting, and needs to be told why.
PHASE_SAVING = "saving"
# The operator declared this attempt at the task over; the follower is easing
# back to its start pose so the next attempt begins where the first did.
# Distinct from eval's `resetting`, which is a whole-episode boundary — here the
# dataset is untouched, only the scene and the arm are being put back.
PHASE_ATTEMPT_RESET = "attempt_reset"
PHASE_WATCHING = "watching"
PHASE_HOLDING = "holding"
PHASE_CORRECTING = "correcting"

# Per-episode verdicts, in the order the UI tallies them.
#   success — the user pressed "task succeeded" and we terminated the episode.
#   failure — the episode ran out its --duration without the user calling it.
#   error   — the episode crashed (a serial glitch, a camera drop, a policy
#       blow-up). NEITHER success nor failure: deliberately excluded from the
#       accuracy denominator so one hardware hiccup can't poison a 20-episode
#       number.
EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"
EPISODE_ERROR = "error"

# Upper bound on a single eval session. 200 episodes × a 60s duration is
# already a >3h bench session; anything past this is a typo, not a plan.
MAX_EVAL_EPISODES = 200

# Upper bound on one coaching session. A correction is a hands-on takeover —
# the operator is standing at the arm for every one of them — so the realistic
# ceiling is far lower than eval's. 100 is already a long, tiring session.
MAX_COACHING_CORRECTIONS = 100


def clamp_eval_episodes(value: int | None) -> int:
    """Coerce a requested episode count into [1, MAX_EVAL_EPISODES].

    Clamps rather than rejects: a nonsensical count (0, -5, 10_000) is a UI slip,
    and silently running one episode / the cap is friendlier than a 422 that
    loses the whole configured launch. A non-integer or None falls back to 1."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_EVAL_EPISODES, n))


def eval_accuracy(results: Sequence[str]) -> float | None:
    """successes / (successes + failures) over the recorded episode verdicts.

    Crashed episodes (EPISODE_ERROR) are excluded from BOTH numerator and
    denominator — see EPISODE_ERROR. Returns None when nothing scoreable has
    happened yet (no episodes, or every episode crashed), so the UI shows
    "no accuracy" instead of a misleading 0%."""
    scored = [r for r in results if r in (EPISODE_SUCCESS, EPISODE_FAILURE)]
    if not scored:
        return None
    return round(sum(1 for r in scored if r == EPISODE_SUCCESS) / len(scored), 4)


def classify_episode(
    rc: int | None,
    stop_requested: bool,
    rollout_started: bool,
    error_text: str | None,
) -> str:
    """Turn one episode's ending into a verdict.

    The single classifier for BOTH ways an episode can end under the eval
    runner, so the two paths can't drift into different semantics:
      - the runner reported `EPISODE_ENDED` — a clean end, passed rc=0;
      - the runner DIED mid-episode — passed its non-zero exit code.

    `stop_requested` (the user pressed "task succeeded — stop episode") wins
    outright: we asked for the ending, so how the episode terminated says
    nothing about the policy.

    Otherwise the episode ended on its own. Reuse `_classify_outcome`:
      ok / ran_with_warning → the episode ran its full --duration (a noisy
          torque-disable on teardown is not a failed episode) → FAILURE, i.e.
          the policy never got the task done in the time allowed.
      failed → the episode crashed → ERROR, excluded from the accuracy."""
    if stop_requested:
        return EPISODE_SUCCESS
    if _classify_outcome(rc, rollout_started, error_text) == "failed":
        return EPISODE_ERROR
    return EPISODE_FAILURE


@dataclass
class _EvalSession:
    """Bookkeeping for ONE multi-episode evaluation session.

    Lives for the whole session (across N subprocesses) — unlike
    `_inference_meta`, which is per-episode and cleared at each subprocess exit.
    Mutated only under `_state_lock`. None whenever the session is single-episode
    or idle, which is what every eval-only endpoint gates on."""

    request: InferenceRequest
    episodes_total: int
    # Resolved ONCE by the startup worker and reused verbatim for every episode:
    # the model is downloaded once and the arm preflight runs once per session.
    policy_path: str | None = None
    robot_args: list[str] = field(default_factory=list)
    # Verdicts in episode order; len(results) is how many episodes have finished,
    # so the CURRENT episode is 1-based index len(results) + 1.
    results: list[str] = field(default_factory=list)
    # Set by /inference-episode-stop just before it asks the runner to end the
    # episode, so the finalisation scores it a success instead of reading the
    # end as a plain timeout. Cleared as each episode is scored.
    stop_requested: bool = False
    # A crashed episode's mined error + plain-language hint, surfaced on the
    # resetting payload and cleared when the user continues.
    error: str | None = None
    hint: str | None = None

    # --- eval-runner bookkeeping (see makermodslab/eval_runner.py) --------------
    # True from the moment the runner reports EPISODE_STARTED until it reports
    # the episode's end. THIS, not `_inference_proc`, is what "an episode is in
    # flight" means now: the runner process spans the whole session, so a live
    # process no longer implies a live episode (it used to, when every episode
    # was its own subprocess).
    episode_running: bool = False
    # An EPISODE was asked for but hasn't started yet — either the runner is
    # still doing its one-time load/connect (the READY handler issues the
    # command when it lands) or the command is in flight. Keeps a READY from a
    # crash-respawn from starting an episode nobody asked for.
    episode_pending: bool = False
    # A QUIT has been written and the runner is winding down. Suppresses the
    # crash-containment path so an expected exit isn't scored as an error.
    quitting: bool = False
    # The runner's own ERROR line. Preferred over log-tail mining when present:
    # it's the exception message itself rather than a heuristic over a traceback.
    runner_error: str | None = None

    @property
    def episode_index(self) -> int:
        """1-based index of the episode currently running (or about to run).

        Clamped to episodes_total so the final payload reads "10 / 10" rather
        than "11 / 10"."""
        return min(len(self.results) + 1, self.episodes_total)


# Evaluation bookkeeping for the CURRENT session, or None when the session is a
# plain single rollout (eval_episodes <= 1) or nothing is running. Every
# eval-only endpoint gates on this being non-None, which is what keeps the
# single-episode flow bit-for-bit unchanged. Mutated under `_state_lock`.
_eval_session: _EvalSession | None = None


@dataclass
class _CoachSession:
    """Bookkeeping for ONE coaching (DAgger) session.

    The coaching counterpart of `_EvalSession`, and deliberately a separate type
    rather than more optional fields on that one: the two sessions share a
    subprocess shape and nothing else. An eval tallies verdicts over episodes
    the runner starts on command; a coaching session tallies corrections the
    OPERATOR starts, and has no notion of a per-episode verdict at all.

    Mutated only under `_state_lock`. None whenever the session is not a
    coaching one, which is what every coaching-only endpoint gates on."""

    request: InferenceRequest
    corrections_target: int
    # The runner's reported phase, or None until it reports one.
    #
    # None, NOT `paused`. Both real phases are claims about who holds the arm,
    # and there is a window between the runner reporting READY and its control
    # loop emitting the first PHASE where neither is known to be true. It used
    # to default to `paused`, which renders as "the arm is frozen" — a sentence
    # the UI would show at the exact moment the policy was starting to drive.
    # None renders as "Starting…", which is the only honest answer there.
    phase: str | None = None
    corrections_saved: int = 0
    # How many attempts at the TASK the operator has declared finished. Not an
    # episode count — corrections are the episodes — but the thing they are
    # actually counting while they work through a task.
    attempts: int = 0
    # True while the session is parked straight after a reset. The operator's
    # next move there is to START THE NEXT ATTEMPT, not to take over — and the
    # UI promotes the matching control, because pressing the usual primary
    # (take over) opened a correction nobody wanted. Cleared as soon as the
    # session moves on.
    awaiting_attempt: bool = False
    # Wall-clock seconds of recorded correction, summed across the session.
    # Reported alongside the count because ten one-second twitches and ten
    # ten-second recoveries are very different datasets.
    correction_seconds: float = 0.0
    # Resolved by the runner AFTER lerobot stamps its timestamp onto the
    # repo_id, and unknowable before that (see dagger_protocol's DATASET note).
    # None until the runner reports it.
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    # Set when a takeover was refused because the leader sits too far from the
    # follower; carries the joint deltas for the UI. Cleared on the next
    # successful phase change, so it reads as "your last attempt", not history.
    align_error: str | None = None
    # A QUIT has been written and the runner is winding down. Suppresses crash
    # containment so an expected exit isn't reported as a failure.
    quitting: bool = False
    # The runner's own ERROR line, preferred over log-tail mining when present.
    runner_error: str | None = None


# Coaching bookkeeping for the CURRENT session, or None when the session is a
# plain rollout / an eval / nothing. Mutated under `_state_lock`.
_coach_session: _CoachSession | None = None


# Pushed to the browser the instant coaching state changes, instead of waiting
# for its next poll. Wired to `ConnectionManager.notify_coaching_state` by
# server.py, exactly as `JobRegistry.set_on_change` is; None when nothing has
# wired it (tests, and any embedding that has no websocket).
#
# WHY THIS EXISTS. The coaching banner is the only thing in this app that tells
# a person whether they or a robot is holding an arm, and it was reaching them
# on a 1 Hz poll. The handover glide lasts about two seconds, so up to half of
# the window in which the banner reads "the arm is moving — don't fight it"
# could elapse before the operator could possibly have seen it. Every other
# phase has the same problem in miniature: the operator looks up at the moment
# they press the key, which is the moment the poll is most likely to be stale.
#
# The poll stays as the reconciler — a dropped or missed push heals within a
# second, and the payload is the same `_coach_fields` block either way, so the
# frontend has one shape to understand and no ordering to reason about.
_on_coaching_state: Callable[[dict[str, Any]], None] | None = None


def set_on_coaching_state(callback: Callable[[dict[str, Any]], None] | None) -> None:
    """Register the websocket push for coaching state. See `_on_coaching_state`."""
    global _on_coaching_state
    _on_coaching_state = callback


def _push_coaching_state() -> None:
    """Send the current coaching block to the browser now.

    Called at the end of every handler that mutates `_coach_session`. Runs on
    the runner's stdout pump thread, so it must not block: the callback only
    queues, and a failure here must never take the pump down — the poll would
    still carry the state, just a second later."""
    callback = _on_coaching_state
    if callback is None:
        return
    with _state_lock:
        fields = _coach_fields(_coach_session)
    with contextlib.suppress(Exception):
        callback(fields)


def clamp_coaching_corrections(value: int | None) -> int:
    """Coerce a requested correction target into [1, MAX_COACHING_CORRECTIONS].

    Clamps rather than rejects, exactly as `clamp_eval_episodes` does and for
    the same reason: a nonsensical target is a UI slip, and the session is
    stoppable at any moment anyway — the target is a stopping convenience, not a
    commitment."""
    try:
        target = int(value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(target, MAX_COACHING_CORRECTIONS))


def _coach_fields(cs: _CoachSession | None) -> dict[str, Any]:
    """The coaching block of an /inference-status payload.

    Emitted on EVERY payload so the shape is stable for the frontend, mirroring
    `_eval_fields`: a non-coaching run reports `coaching: False` with null
    companions rather than omitting the keys."""
    if cs is None:
        return {
            "coaching": False,
            "coaching_phase": None,
            "corrections_saved": None,
            "attempts": None,
            "awaiting_attempt": None,
            "corrections_target": None,
            "correction_seconds": None,
            "coaching_dataset": None,
            "align_error": None,
        }
    return {
        "coaching": True,
        "coaching_phase": cs.phase,
        "corrections_saved": cs.corrections_saved,
        "attempts": cs.attempts,
        "awaiting_attempt": cs.awaiting_attempt,
        "corrections_target": cs.corrections_target,
        "correction_seconds": round(cs.correction_seconds, 1),
        "coaching_dataset": cs.dataset_repo_id,
        "align_error": cs.align_error,
    }


def _eval_fields(
    ev: _EvalSession | None,
    *,
    accuracy: float | None = None,
) -> dict[str, Any]:
    """The eval block of an /inference-status payload.

    Emitted on EVERY payload so the shape is stable for the frontend: a
    single-episode run reports `eval_mode: False` with null/empty companions
    rather than omitting the keys. `accuracy` is passed in (not derived) because
    it is claimed ONLY on a session that ran to completion — an aborted session
    reports its partial tally with accuracy None."""
    if ev is None:
        return {
            "eval_mode": False,
            "episode_index": None,
            "episodes_total": None,
            "episode_results": None,
            "accuracy": None,
        }
    return {
        "eval_mode": True,
        "episode_index": ev.episode_index,
        "episodes_total": ev.episodes_total,
        "episode_results": list(ev.results),
        "accuracy": accuracy,
    }


# Stable lerobot setup log fragments (lerobot/rollout/context.py) that mark the
# transition into a finer sub-phase. Watched in _pump_stdout. These are plain
# logger.info messages, not a documented contract — if an upstream bump renames
# them the phase just stays at its previous (coarser but still correct) value,
# so a drift degrades gracefully rather than crashing.
_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Loading policy from", PHASE_LOADING_POLICY),
    ("Connecting robot", PHASE_CONNECTING),
)


def _set_phase(phase: str) -> None:
    """Record the current startup sub-phase on the shared inference meta.

    Guarded by _state_lock (short critical section). A no-op when no session is
    active — a late stdout line arriving after teardown can't resurrect a
    phase on an empty meta dict (and must not broadcast a phantom phase).

    Every recorded phase is also broadcast as a `session_changed` hint (see
    makermodslab/session_events.py) — outside the lock; the notify is a
    droppable queue put that consumers answer by refetching status."""
    with _state_lock:
        if not _inference_meta:
            return
        _inference_meta["phase"] = phase
    notify_session_changed("inference", True, phase=phase)


def _advance_setup_phase(line: str) -> bool:
    """Flip to a finer setup sub-phase when `line` is a recognised lerobot setup
    log. True when one matched. Cheap substring checks."""
    for fragment, phase in _PHASE_MARKERS:
        if fragment in line:
            _set_phase(phase)
            return True
    return False


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the subprocess's stdout to the log file, advance the startup
    sub-phase off recognised lerobot setup lines, and watch for the
    rollout-start marker.

    The SINGLE-episode path (`eval_episodes == 1`). Eval mode's long-lived
    runner is pumped by `_pump_runner_stdout` instead."""
    global _inference_rollout_started_at
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
            # Advance to a finer setup sub-phase on the first matching line.
            # Only fires before the rollout marker, so a later line mentioning
            # "Connecting robot" can't drag a running session backwards.
            if _inference_rollout_started_at is None:
                _advance_setup_phase(line)
            if _inference_rollout_started_at is None and _ROLLOUT_START_MARKER in line:
                _inference_rollout_started_at = time.time()
                _set_phase(PHASE_RUNNING)
                logger.info(
                    "Inference rollout main loop started after %.1fs of setup",
                    _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at),
                )
    except Exception as exc:
        logger.exception("Inference stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()


# How long the runner pump waits for the process to be reaped once its stdout
# hits EOF. EOF means the process is already on its way out, so this is a
# formality — it only exists so a wedged exit can't hang the pump thread.
_RUNNER_REAP_TIMEOUT_S = 5.0


def _pump_runner_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the eval runner's output to the log and act on its protocol events.

    The eval-mode counterpart of `_pump_stdout`, and a structurally different
    job: it lives for the whole SESSION, so it — not a `proc.poll()` in the
    status endpoint — is what observes episode boundaries. The runner does not
    exit between episodes, so there is no exit left for a status poll to notice;
    every episode start and end arrives here as a line on stdout.

    An exit therefore means something went wrong (or the session is over), which
    is why the EOF path hands off to the crash-containment handler."""
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
                _handle_runner_line(line)
            except Exception:
                # One malformed event must not take the pump — and with it every
                # remaining episode boundary — down with it.
                logger.exception("Eval runner event handling failed for %r", line.strip())
    except Exception as exc:
        logger.exception("Eval runner stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()
        rc: int | None = None
        with contextlib.suppress(Exception):
            rc = proc.wait(timeout=_RUNNER_REAP_TIMEOUT_S)
        _handle_runner_exit(proc, rc)


def _handle_runner_line(line: str) -> None:
    """Dispatch one line of runner output.

    Non-protocol lines are lerobot's own logging: during the runner's one-time
    load/connect those are exactly the fragments `_PHASE_MARKERS` recognises, so
    the UI still gets to name the wait ("Loading policy…", "Connecting to
    arm…"). They're only honoured while no episode is in flight — a mid-episode
    line that happens to mention "Connecting robot" must not drag a running
    session back to a setup phase."""
    parsed = parse_event(line)
    if parsed is None:
        with _state_lock:
            ev = _eval_session
            in_episode = ev is not None and ev.episode_running
        if not in_episode:
            _advance_setup_phase(line)
        return
    event, payload = parsed
    if event == EVENT_READY:
        _on_runner_ready()
    elif event == EVENT_EPISODE_STARTED:
        _on_episode_started()
    elif event == EVENT_EPISODE_ENDED:
        _on_episode_ended(parse_episode_end_reason(payload))
    elif event == EVENT_ERROR:
        _on_runner_error(payload)


def _on_runner_ready() -> None:
    """The runner finished its one-time load + connect: issue the pending episode.

    The runner never starts an episode on its own, so the session's first
    episode — and the first after a crash-respawn — is issued from here, once
    the expensive part is behind us. Gated on `episode_pending` so a READY that
    lands after an abort (or after the user parked without continuing) can't put
    the arm in motion."""
    with _state_lock:
        ev = _eval_session
        if not inference_active or ev is None or ev.quitting or not ev.episode_pending:
            logger.info("Eval runner is ready, but no episode is pending — staying idle")
            return
        proc = _inference_proc
        if _inference_meta:
            _inference_meta["phase"] = PHASE_STARTING
    if not _send_runner_command(proc, CMD_EPISODE):
        # The runner died between READY and here; the pump's EOF path scores it.
        logger.warning("Eval runner is ready but the EPISODE command could not be sent")


def _on_episode_started() -> None:
    """The runner's control loop has taken over for this episode."""
    global _inference_rollout_started_at
    with _state_lock:
        ev = _eval_session
        if ev is None:
            return
        ev.episode_pending = False
        ev.episode_running = True
        _inference_rollout_started_at = time.time()
        if _inference_meta:
            _inference_meta["phase"] = PHASE_RUNNING
        setup_s = _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at)
        episode_index = ev.episode_index
    logger.info("Eval episode %s rollout started after %.1fs of setup", episode_index, setup_s)


def _on_episode_ended(reason: str) -> None:
    """Score the finished episode, then park or finish the session.

    Runs on the pump thread — in eval mode this is the ONLY place an episode
    boundary is observed. Goes through the single scoring point with rc=0: the
    runner is alive and healthy, so `classify_episode` sees a clean ending and
    the verdict falls out of `stop_requested` (the user's success button) versus
    the episode simply running out its duration."""
    with _state_lock:
        ev = _eval_session
        if ev is None or not ev.episode_running:
            logger.warning("Eval runner reported an episode end with none in flight (reason=%r)", reason)
            return
        if reason == REASON_STOPPED:
            # The reason IS the STOP we sent, so honour it even if the flag were
            # somehow lost — the two can't disagree about what the user pressed.
            ev.stop_requested = True
        ev.episode_running = False
        # Captured before finalising: finishing the session clears the global.
        proc = _inference_proc
        _finalise_eval_episode_locked(0, ev, keep_runner=True)
        session_finished = _eval_session is None
    if session_finished:
        # That was the last episode — the slot is already released, so the runner
        # has nothing left to do. Ask it to go home and disconnect. Sent, not
        # waited on: this thread has to keep draining stdout or the runner's
        # teardown logging could fill the pipe and wedge its own shutdown.
        _send_runner_command(proc, CMD_QUIT)


def _on_runner_error(message: str) -> None:
    """Stash the runner's own exception text ahead of its exit.

    Not a verdict — the runner emits this and then dies, so this only makes the
    crash that follows legible. Preferred over mining the log tail because it is
    the exception message itself rather than a heuristic over a traceback."""
    with _state_lock:
        ev = _eval_session
        if ev is not None:
            ev.runner_error = message or None


def _pump_dagger_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the coaching runner's output to the log and act on its protocol events.

    Structurally the same job as `_pump_runner_stdout` — one pump for the whole
    session, phase changes arriving as lines rather than as process exits — but
    against the DAgger vocabulary. Kept separate rather than parameterised: the
    two protocols share no event beyond READY/ERROR, and a single pump switching
    on which session is live would be harder to read than two that each know
    exactly what they are reading."""
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
                _handle_dagger_line(line)
            except Exception:
                # One malformed event must not take the pump — and with it every
                # remaining phase change — down with it. A frozen phase display
                # over a moving arm is exactly the failure this guards against.
                logger.exception("Coaching runner event handling failed for %r", line.strip())
    except Exception as exc:
        logger.exception("Coaching runner stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()
        rc: int | None = None
        with contextlib.suppress(Exception):
            rc = proc.wait(timeout=_RUNNER_REAP_TIMEOUT_S)
        _handle_dagger_exit(proc, rc)


def _handle_dagger_line(line: str) -> None:
    """Dispatch one line of coaching-runner output.

    Non-protocol lines are lerobot's own logging; during the one-time
    load/connect those are the fragments `_PHASE_MARKERS` recognises, so the UI
    still names the wait. They're only honoured before the session has reported
    a phase — once the operator is watching a live policy, a log line that
    happens to mention "Connecting robot" must not drag the display back to a
    setup phase."""
    parsed = parse_dagger_event(line)
    if parsed is None:
        with _state_lock:
            cs = _coach_session
            started = cs is not None and cs.dataset_repo_id is not None
        if not started:
            _advance_setup_phase(line)
        return
    event, payload = parsed
    if event == DAGGER_EVENT_READY:
        _on_dagger_ready()
    elif event == EVENT_DATASET:
        _on_dagger_dataset(parse_dagger_fields(payload))
    elif event == EVENT_PHASE:
        _on_dagger_phase(parse_dagger_fields(payload).get("phase", ""))
    elif event == EVENT_CORRECTION_SAVED:
        _on_correction_saved(parse_dagger_fields(payload))
    elif event == EVENT_CORRECTION_CANCELLED:
        _on_correction_cancelled(parse_dagger_fields(payload))
    elif event == EVENT_ALIGN_REQUIRED:
        _on_align_required(parse_dagger_fields(payload))
    elif event == EVENT_ATTEMPT_RESET:
        _on_attempt_reset(parse_dagger_fields(payload))
    elif event == DAGGER_EVENT_ERROR:
        _on_dagger_error(payload)


def _on_dagger_ready() -> None:
    """The runner finished its one-time load + connect.

    Unlike eval's READY this issues no command: a coaching session starts
    driving the moment the control loop begins, and every transition after that
    is the operator's to request. All this does is retire the setup phases."""
    global _inference_rollout_started_at
    with _state_lock:
        if not inference_active or _coach_session is None:
            return
        _inference_rollout_started_at = time.time()
        setup_s = _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at)
    logger.info("Coaching session live after %.1fs of setup", setup_s)


def _on_dagger_dataset(fields: dict[str, str]) -> None:
    """Record the dataset name lerobot actually created.

    This is the only place the app learns it. `stamp_repo_id` appends a
    timestamp inside the subprocess, so the name the user typed is not the name
    on disk, and reconstructing it here would mean guessing the second the
    subprocess reached that line (see dagger_protocol)."""
    with _state_lock:
        cs = _coach_session
        if cs is None:
            return
        cs.dataset_repo_id = fields.get("repo_id") or None
        cs.dataset_root = fields.get("root") or None
    logger.info("Coaching dataset: %s", fields.get("repo_id"))


def _on_dagger_phase(phase: str) -> None:
    """Translate a lerobot DAggerPhase into the app's operator-facing phase.

    An unrecognised phase is ignored rather than passed through: the value ends
    up driving a banner that tells the operator whether the robot or they are in
    control, and showing an unknown string there is worse than showing a stale
    one they can still act on."""
    if phase not in DAGGER_PHASES:
        logger.warning("Ignoring unrecognised coaching phase %r", phase)
        return
    app_phase = {
        DAGGER_PHASE_CORRECTING: PHASE_CORRECTING,
        DAGGER_PHASE_PAUSED: PHASE_HOLDING,
        DAGGER_PHASE_HANDING_OVER: PHASE_HANDING_OVER,
        DAGGER_PHASE_SAVING: PHASE_SAVING,
        DAGGER_PHASE_RESETTING: PHASE_ATTEMPT_RESET,
    }.get(phase, PHASE_WATCHING)
    with _state_lock:
        cs = _coach_session
        if cs is None:
            return
        cs.phase = phase
        # A phase actually changed, so the last refused takeover is history.
        cs.align_error = None
        # Leaving the parked-after-reset state: any phase other than the two
        # the reset itself passes through means the operator has moved on.
        if phase not in (DAGGER_PHASE_PAUSED, DAGGER_PHASE_RESETTING):
            cs.awaiting_attempt = False
        if _inference_meta:
            _inference_meta["phase"] = app_phase
    # Outside the lock: the push builds its payload by taking the lock again,
    # and the callback runs arbitrary websocket code we do not want holding it.
    _push_coaching_state()


def _on_correction_saved(fields: dict[str, str]) -> None:
    """Tally one saved correction."""
    with _state_lock:
        cs = _coach_session
        if cs is None:
            return
        # Trust the runner's own count over incrementing our own: it is the side
        # that knows whether an episode was written, and a dropped event would
        # otherwise leave the two permanently out of step.
        try:
            cs.corrections_saved = int(fields.get("n", cs.corrections_saved + 1))
        except ValueError:
            cs.corrections_saved += 1
        with contextlib.suppress(ValueError):
            cs.correction_seconds += float(fields.get("seconds", 0.0))
        saved = cs.corrections_saved
        target = cs.corrections_target
    logger.info("Correction %d/%d saved", saved, target)
    _push_coaching_state()


def _on_correction_cancelled(fields: dict[str, str]) -> None:
    """A correction was discarded. Whether the operator hears about it depends
    entirely on WHO discarded it.

    An operator-pressed discard stays silent: they know, they asked, and the
    count not moving is the feedback. Telling them again would be nagging.

    A `too_short` discard is the opposite case and used to be indistinguishable
    from it — the operator took over, did something deliberate, handed back, and
    the runner binned it under `_MIN_CORRECTION_FRAMES` with nothing on screen
    to say so. They would only find out by counting episodes afterwards. That is
    the discard worth interrupting for, and the quick corrective nudge it eats
    is, per CR-DAgger (arXiv:2506.16685), among the most valuable data in the
    session — so the message names the floor rather than just apologising, and
    tells them what to do differently.

    Reuses `align_error`'s field rather than adding a second notice slot: both
    are "your last takeover produced nothing, here is why", they can never be
    true at the same moment, and one banner is one thing for the operator to
    look at. Cleared on the next phase change, like the other."""
    reason = fields.get("reason", CANCEL_REASON_OPERATOR)
    frames = fields.get("frames", "?")
    if reason != CANCEL_REASON_TOO_SHORT:
        logger.info("Correction discarded by the operator")
        _push_coaching_state()
        return
    seconds = fields.get("seconds", "?")
    minimum = fields.get("minimum")
    floor = f"the {minimum}-frame minimum" if minimum else "the minimum length"
    message = (
        f"That correction was discarded — {frames} frames ({seconds}s) is below {floor}, "
        "so saving it would have broken the dataset. Nothing was kept. Hold the "
        "takeover a moment longer next time."
    )
    with _state_lock:
        cs = _coach_session
        if cs is not None:
            cs.align_error = message
    logger.warning("Correction discarded as too short (%s frames)", frames)
    _push_coaching_state()


def _on_align_required(fields: dict[str, str]) -> None:
    """A takeover was refused: the leader sits too far from the follower.

    Turned into a plain-language sentence HERE rather than in the frontend
    because the joint list needs the same treatment as every other hardware
    hint in this module — the UI renders a message, it doesn't compose one."""
    joints = (fields.get("joints") or "").replace(",", ", ").replace(":", " ")
    detail = f" ({joints})" if joints else ""
    message = (
        "Takeover refused: the leader arms are too far from the robot's pose. "
        f"Move them closer and try again{detail}."
    )
    with _state_lock:
        cs = _coach_session
        if cs is not None:
            cs.align_error = message
    logger.warning(message)
    _push_coaching_state()


def _on_attempt_reset(fields: dict[str, str]) -> None:
    """One attempt at the task ended and the arm is back at its start pose."""
    with _state_lock:
        cs = _coach_session
        if cs is None:
            return
        with contextlib.suppress(ValueError, TypeError):
            cs.attempts = int(fields.get("n", cs.attempts + 1))
        cs.awaiting_attempt = True
        n = cs.attempts
    logger.info("Attempt %d reset — arm is home", n)
    _push_coaching_state()


def _on_dagger_error(message: str) -> None:
    """Stash the coaching runner's own exception text ahead of its exit."""
    with _state_lock:
        cs = _coach_session
        if cs is not None:
            cs.runner_error = message or None


def _handle_dagger_exit(proc: subprocess.Popen, rc: int | None) -> None:
    """Finalise a finished/dead coaching runner (called from the pump's EOF).

    Unlike eval, a coaching runner exiting is usually the HAPPY path: the
    session ends when the correction target is reached and the runner returns on
    its own. So this reports a terminal result rather than containing a crash,
    and only calls it an error when the exit code says so."""
    with _state_lock:
        cs = _coach_session
        if cs is None or _inference_proc is not proc:
            # Already finalised (a stop that ran to completion), or a stale pump.
            return
        # `quitting` means the operator pressed Stop and we asked the runner to
        # wind down. The runner then exits 0 — a clean exit — so without this the
        # session would be reported as having run to completion, and the summary
        # would congratulate the user on finishing a run they cut short.
        _finalise_coaching_locked(rc, cs, aborted=cs.quitting and not rc)


def _handle_runner_exit(proc: subprocess.Popen, rc: int | None) -> None:
    """Crash containment for a dead eval runner (called from the pump's EOF)."""
    with _state_lock:
        ev = _eval_session
        if ev is None or _inference_proc is not proc:
            # Either the session is already over (the expected QUIT after the
            # last episode, or an abort) or this is a stale pump whose runner we
            # have since replaced. Nothing to score either way.
            return
        _finalise_runner_exit_locked(rc, ev)


def _detect_device() -> str:
    """cuda → mps → cpu, picked once at start time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _report_download_progress(bytes_done: int, bytes_total: int | None) -> None:
    """Record Hub-download byte progress on the live inference meta.

    Fed from the snapshot_download tqdm hook (make_snapshot_progress_tqdm), may
    fire from any thread. A no-op once the meta is gone (a stopped/failed session
    cleared it) so a late tqdm callback can't resurrect a dead session. ``percent``
    is None while the total is still unknown → the UI shows an indeterminate bar."""
    with _state_lock:
        if not _inference_meta:
            return
        _inference_meta["download_bytes_done"] = bytes_done
        _inference_meta["download_bytes_total"] = bytes_total
        _inference_meta["download_percent"] = (
            round(bytes_done / bytes_total * 100, 1) if bytes_total else None
        )


def _policy_ref_is_valid(policy_ref: str) -> bool:
    """Cheap shape check for a policy ref (one is_dir stat, no network) so a
    malformed ref is rejected synchronously in the POST — surfacing in the modal
    as a 4xx — instead of failing later on the inference page."""
    return (
        bool(_HUB_REF_RE.match(policy_ref))
        or bool(_HUB_ROOT_REF_RE.match(policy_ref))
        or Path(policy_ref).is_dir()
    )


def _local_store_policy_path(repo_id: str, step_dir: str | None) -> str | None:
    """A ready-to-run pretrained_model dir for a Hub ref, taken from MakerMods
    Lab's own models store instead of the Hub — or None to go download.

    The Models page downloads with ``local_dir=<makermodslab_models>/<repo_id>``,
    and huggingface_hub's local_dir mode neither reads nor populates the shared
    hub cache. So a model the user already pulled on the Models page was, until
    this check, downloaded a SECOND time the first time inference ran on it
    (see design-debt F6). When the hub cache has no entry for the repo there is
    nothing for snapshot_download to dedupe against, so a usable local copy is
    strictly better: same bytes, zero network, no downloading_model phase.

    Deliberately does NOT pre-empt a populated hub cache: snapshot_download IS
    the hub-cache path, it is revision-aware and only fetches changed files, so
    once the repo is cached letting it run keeps the user on `main` rather than
    pinning them to a possibly stale local copy.

    `step_dir` is the zero-padded step of a ``@checkpoints/<step>`` ref, or None
    for a ``@root`` ref. Usability is judged with models.py's own helpers, so
    the tree that comes back is one the Models page would also call usable:
    ``_downloaded_model_dir`` for the traversal guard + a first usability probe,
    then the ref-shape-specific check. For ``@root`` the ROOT itself must be the
    pretrained dir (what a flat repo resolves to) — a local copy that resolves
    to a checkpoints sub-tree is not what the Hub path would have returned, so
    it falls through to the download rather than quietly substituting a
    different tree."""
    if _hub_cache_has_repo(repo_id):
        return None
    model_dir = _downloaded_model_dir(repo_id)
    if model_dir is None:
        return None
    resolved: Path | None = None
    if step_dir is not None:
        candidate = model_dir / "checkpoints" / step_dir / "pretrained_model"
        # config.json alone is NOT enough: an interrupted local_dir download
        # leaves the config and the processor safetensors behind but no policy
        # weights, and serving that turns a silently-partial store entry into a
        # FileNotFoundError deep inside lerobot. This branch addresses a specific
        # step directly (it never goes through _resolve_pretrained_dir), so it
        # has to ask the weights question itself.
        if (candidate / "config.json").is_file() and _has_loadable_weights(candidate):
            resolved = candidate
    elif _resolve_pretrained_dir(model_dir) == model_dir:
        resolved = model_dir
    if resolved is None:
        return None
    logger.info(
        "Using the local models store for %s (nothing cached for it in the HF hub cache): %s",
        repo_id,
        resolved,
    )
    return str(resolved)


def _resolve_policy_path(policy_ref: str, report: Callable[[int, int | None], None] | None = None) -> str:
    """Turn a checkpoints API ref into a local path that lerobot accepts.

    Local refs are already absolute paths to a pretrained_model dir.
    Hub refs look like 'user/repo@checkpoints/<step_dir>' (where <step_dir> is
    lerobot's zero-padded directory name, e.g. 000050) and resolve to that
    step's pretrained_model dir; a 'user/repo@root' ref means the whole repo IS
    the pretrained_model and resolves to its root.

    The download itself (which patterns each ref shape pulls, and what path it
    yields) lives in jobs.download_hub_checkpoint_ref, shared with the fine-tune
    path so a ref resolves to the same weights whoever asks. This wrapper owns
    only what is inference-specific: the local-dir short-circuit, the local
    models-store short-circuit, the downloading_model phase, and the progress
    hook.

    Before delegating, `_local_store_policy_path` gets a chance to serve the ref
    out of MakerMods Lab's own models store (what the Models page downloads
    into) — that store is invisible to the hub cache, so without the check a
    model already on disk is downloaded a second time (design-debt F6).

    That short-circuit deliberately lives HERE and not inside
    jobs.download_hub_checkpoint_ref, even though the duplicate-download problem
    is the same for every caller: the shared helper also feeds fine-tune/resume
    downloads, and the models store is written by `models._fetch_model_snapshot`,
    which strips ``training_state/`` (optimizer + scheduler state — dead weight
    for inference, often the bulk of a checkpoint). Serving the store from the
    shared helper would therefore hand a *resume* a checkpoint with no optimizer
    state to resume from. Inference only ever loads the policy weights, so it is
    the one caller for which the stripped tree is equivalent.

    When ``report`` is given, snapshot_download streams byte progress through it
    (see make_snapshot_progress_tqdm) so the inference page can show a real
    download bar. Local refs — on disk or in the models store — never download,
    so they never report and never flip the phase."""
    if Path(policy_ref).is_dir():
        # A local checkpoint — nothing to fetch, so no downloading_model phase.
        return policy_ref
    if not _policy_ref_is_valid(policy_ref):
        raise ValueError(f"Unrecognised policy ref: {policy_ref!r}")

    # A Hub ref: the download may pull hundreds of MB and take minutes.
    # Announce it (downloading_model phase) so the UI names the wait, and feed
    # byte progress through the tqdm hook when a reporter is supplied. Set only on
    # the download paths (not the local branch above), and only when a session is
    # live (_set_phase no-ops otherwise), so this helper stays safe to call from
    # the unit tests.
    # …but first: the ref may already be sitting in our own models store, in
    # which case there is nothing to announce. Match the ref shape here (rather
    # than inside the helper) because the two shapes ask a different question of
    # the store — a specific step's pretrained_model dir, or the repo root.
    m = _HUB_REF_RE.match(policy_ref)
    if m:
        local = _local_store_policy_path(m.group("repo"), m.group("step_dir"))
    else:
        m = _HUB_ROOT_REF_RE.match(policy_ref)
        local = _local_store_policy_path(m.group("repo"), None) if m else None
    if local is not None:
        # Already on disk in our own models store and nothing in the hub cache
        # to dedupe against — no fetch, so (like the local-ref branch above) no
        # downloading_model phase and no progress reporting.
        return local

    _set_phase(PHASE_DOWNLOADING_MODEL)
    tqdm_class = make_snapshot_progress_tqdm(report) if report is not None else None
    return download_hub_checkpoint_ref(policy_ref, tqdm_class=tqdm_class)


def _arm_count_mismatch(mode: str, checkpoint_state_dim: int | None) -> str | None:
    """Explain a checkpoint/robot arm-count mismatch, or None when they agree.

    An SO-101 follower has 6 state dims; a bimanual robot drives two arms (12
    dims). A checkpoint trained on one arm-count crashes on the other deep in
    the rollout subprocess (a raw shape mismatch, no explanation). Reject it
    up front with a legible message when the checkpoint exposes enough to tell.

    `checkpoint_state_dim` is None when the checkpoint omits observation.state
    (e.g. a vision-only policy) — then we can't tell cheaply, so return None and
    let the subprocess's own shape check speak (reported in the modal via the
    existing post-mortem path). A dim that's neither 6 nor a clean multiple is
    also left to the subprocess rather than guessed at here.
    """
    if checkpoint_state_dim is None:
        return None
    robot_is_bimanual = mode == "bimanual"
    # The checkpoint is bimanual iff its state is (a multiple of) two arms wide.
    if checkpoint_state_dim <= _SINGLE_ARM_STATE_DIM:
        checkpoint_is_bimanual = False
    elif checkpoint_state_dim % _SINGLE_ARM_STATE_DIM == 0:
        checkpoint_is_bimanual = checkpoint_state_dim // _SINGLE_ARM_STATE_DIM >= 2
    else:
        # An odd width we don't recognise — don't block on a guess.
        return None
    if robot_is_bimanual == checkpoint_is_bimanual:
        return None
    if checkpoint_is_bimanual:
        return (
            f"This checkpoint was trained on a bimanual robot "
            f"({checkpoint_state_dim}-dim state, 2 arms), but the selected robot is "
            "single-arm. Select a bimanual robot to run this policy."
        )
    return (
        f"This checkpoint was trained on a single-arm robot "
        f"({checkpoint_state_dim}-dim state), but the selected robot is bimanual. "
        "Select a single-arm robot to run this policy."
    )


def _counterpart_leader_slots(follower_id: str) -> list[ArmSlot]:
    """Leader config(s) paired with this follower config in saved robot records.

    Inference only connects the follower, so the guard can't derive the
    counterpart slot from the session itself (the way teleop/record do). Look
    it up: any robot record whose follower slot is `follower_id` names the
    leader config that belongs on the OTHER port — if the connected arm's
    EEPROM fingerprint matches that config, the ports are swapped (hard block
    instead of a generic warning)."""
    slots: list[ArmSlot] = []
    seen: set[tuple[str, str]] = set()
    for record in list_robot_records():
        for follower_field, leader_field, label in (
            ("follower_config", "leader_config", "leader"),
            ("right_follower_config", "right_leader_config", "right leader"),
        ):
            leader_name = record.get(leader_field) or ""
            if record.get(follower_field) == follower_id and leader_name and (label, leader_name) not in seen:
                seen.add((label, leader_name))
                slots.append(ArmSlot(label, "leader", leader_name))
    return slots


@contextmanager
def _open_follower(port: str, follower_id: str):
    """Open a bare follower bus on `port`, yield the connected robot, and
    release the port read-only on exit.

    Both rollout preflights connect one follower, do read-only work, then must
    free the port for the subprocess to reopen. Torque is never enabled here,
    so the release skips the torque-disable write (``disconnect(
    disable_torque=False)``) — a plain port close. The disconnect runs on any
    exit path (success or exception)."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id))
    robot.bus.connect()
    try:
        yield robot
    finally:
        robot.bus.disconnect(disable_torque=False)


def _preflight_arm_identity(port: str, follower_id: str, config_name: str | None = None) -> list[str]:
    """Read-only identity check of ONE follower arm before the rollout
    subprocess starts.

    The subprocess itself can't be guarded (its stdin is pre-seeded with a
    newline, which auto-confirms lerobot's "use the calibration file" prompt
    and stamps the file into EEPROM on mismatch), so the check happens here:
    connect the bare bus, verify, and release the port for the subprocess to
    reopen. Raises ArmIdentityError on a hard mismatch; returns the
    warn-but-allow messages otherwise.

    `follower_id` names the calibration the arm loads and is what identifies the
    slot by default. For a bimanual staging alias id ("<base>_left"), pass the
    real library stem as `config_name` so the guard compares against the library
    entry rather than the alias (mirrors verify_devices' config_names in
    record/teleop). Bimanual runs each follower bus through this separately —
    each opens and releases its own port — so the two are never open at once."""
    with _open_follower(port, follower_id) as robot:
        return verify_devices(
            ((robot, "follower"),),
            extra_slots=_counterpart_leader_slots(config_name or follower_id),
            config_names=[config_name] if config_name is not None else None,
        )


@contextmanager
def _open_leader(port: str, leader_id: str):
    """Open a bare leader bus on `port`, yield the connected teleop, and release
    the port read-only on exit.

    The leader-side twin of `_open_follower`, and used for the same reason: a
    coaching session's subprocess opens this port, so the identity check has to
    happen before it and hand the port back. Torque is never enabled here."""
    teleop = SO101Leader(SO101LeaderConfig(port=port, id=leader_id))
    teleop.bus.connect()
    try:
        yield teleop
    finally:
        teleop.bus.disconnect(disable_torque=False)


def _preflight_leader_identity(
    port: str, leader_id: str, follower_id: str, config_name: str | None = None
) -> list[str]:
    """Read-only identity check of ONE leader arm before a coaching session.

    The leader-side twin of `_preflight_arm_identity`, and needed only for
    coaching — it is the one inference flow that connects a leader at all. Not
    an optional nicety: during a pause the runner ENABLES TORQUE on this arm and
    drives it to the follower's pose, so an unrecognised arm on this port is
    every bit as capable of moving unexpectedly as the follower is.

    Verified sequentially against the follower rather than with both buses open
    at once, matching the invariant the bimanual follower preflight already
    keeps. The counterpart slot is passed explicitly (we know the follower's
    config for this very session) instead of looked up from the robot records
    the way `_counterpart_leader_slots` has to, so a port swap is caught against
    the pair the operator actually selected."""
    with _open_leader(port, leader_id) as teleop:
        return verify_devices(
            ((teleop, "leader"),),
            extra_slots=[ArmSlot("follower", "follower", follower_id)],
            config_names=[config_name] if config_name is not None else None,
        )


def _preflight_motor_registers(port: str, follower_id: str) -> list[str]:
    """Prime the follower's RAM motor registers before the rollout subprocess
    starts.

    The subprocess itself can't be instrumented, but Torque_Limit and
    Goal_Velocity are both RAM registers: they survive closing the serial port
    (only a power cycle resets them), and the subprocess's connect()/configure()
    never writes them — so setting them here and releasing the port is enough
    for the whole rollout. Two priming steps:
      - reset_torque_limit: restore stock torque (a previous auto-calibration's
        working torque would otherwise cap the whole rollout).
      - clear_goal_velocity: reset any leftover speed cap a previous
        arm-driving feature stamped (auto-cal fold/unfold=1000, rest-pose
        return=400), which would otherwise throttle the whole rollout.
    Never raises: a failure degrades to the previous register value (logged)
    and returns warning messages instead of aborting the start."""
    try:
        with _open_follower(port, follower_id) as robot:
            return reset_torque_limit(robot, "follower arm") + clear_goal_velocity(robot, "follower arm")
    except Exception as exc:
        message = (
            f"Could not reset the motor registers on {port}: {exc}. "
            "The arm runs at its previous torque/speed limits for this rollout."
        )
        logger.warning(message)
        return [message]


def _format_cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Convert {name: {type, camera_index, width, height, fps}} into
    lerobot's CLI dict syntax. `cameras` is the record-resolved session dict
    (see _session_cameras), keyed by the POLICY-expected camera names. The
    stored key `camera_index` is remapped to lerobot's `index_or_path`; the
    identity key `unique_id` is dropped — it is the robot record's own handle
    on the physical device (see makermodslab/camera_identity.py), and lerobot's
    OpenCVCameraConfig would reject it as an unknown field.

    Like recording (`record._build_camera_configs`), opencv cameras default to
    MJPG when the record doesn't pin a fourcc: without it, Linux/V4L2
    negotiates raw YUYV and a 3-camera rig exhausts the USB bus at STREAMON —
    the third camera fails during inference only, since recording already
    defaults to MJPG. An explicit fourcc from the record still wins.
    """
    parts = []
    for name, cfg in cameras.items():
        remapped = {
            ("index_or_path" if k == "camera_index" else k): v
            for k, v in cfg.items()
            if v is not None and k != "unique_id"
        }
        if cfg.get("type") == "opencv" and not cfg.get("fourcc"):
            remapped["fourcc"] = _DEFAULT_FOURCC
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


# Exception lines at the tail of a Python traceback look like
# "RuntimeError: ..." or "lerobot.errors.DeviceNotConnectedError: ...".
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Timeout|Failure)\b")


def _read_log_tail_lines(log_path: str | None) -> list[str] | None:
    """Decode the last ~64 KB of a log file into text lines (the window's oldest
    line first, newest last).

    Only the tail is read, so a multi-MB verbose log is never materialized in
    full — the shared basis for both the error-mining in _extract_error_from_log
    and the log-tail endpoint in handle_inference_log. Returns None for a missing
    path or an unreadable file (OSError); an empty list for an empty file."""
    if not log_path:
        return None
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            data = fh.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def _extract_error_from_log(log_path: str | None) -> str | None:
    """Pull the meaningful error out of a failed rollout's log so the UI can
    show it directly instead of telling the user to open a file in the cache.

    Subprocess forensics: we only have the log, so we mine the tail for the
    last traceback exception line + its message body. (Recording/teleop run
    in-process and will hand the caught exception's text straight to
    friendly_hint/is_cleanup_error instead — this step is rollout-only.)"""
    lines = _read_log_tail_lines(log_path)
    if lines is None:
        return None
    tail = lines[-50:]
    # Prefer the last exception line + everything after it (the message body).
    exc_idx = next((i for i in range(len(tail) - 1, -1, -1) if _EXC_LINE_RE.match(tail[i])), None)
    if exc_idx is not None:
        snippet = "\n".join(tail[exc_idx:]).strip()
    else:
        non_empty = [ln for ln in tail if ln.strip()]
        snippet = "\n".join(non_empty[-6:]).strip()
    snippet = re.sub(r"\n\s*\n+", "\n", snippet)
    if len(snippet) > 500:
        snippet = snippet[:500].rstrip() + "…"
    return snippet or None


def _classify_outcome(rc: int | None, rollout_started: bool, error_text: str | None) -> str:
    """ok | ran_with_warning | failed.

    A non-zero exit *after* the rollout main loop started, where the error is a
    torque-disable/overload on shutdown, means the skill ran but a motor (usually
    the loaded gripper) complained during cleanup — that's a warning, not a
    failure, so the UI shouldn't call a working run "failed". A mid-run
    disconnect (or a non-zero exit before the loop began) stays a real failure —
    is_cleanup_error deliberately excludes connection-loss markers."""
    if not rc:
        return "ok"
    if rollout_started and is_cleanup_error(error_text):
        return "ran_with_warning"
    return "failed"


def _rollout_cli_args(request: InferenceRequest, policy_path: str, robot_args: list[str]) -> list[str]:
    """The rollout flags, without the interpreter/module prefix.

    `robot_args` is the `--robot.*` block built per mode (single vs bimanual);
    everything else — strategy, policy, task, duration, and the teardown pin —
    is identical across modes and lives here so both paths stay in sync.

    Split out from `_build_rollout_cmd` because eval mode points the SAME flags
    at a different entry point (`makermodslab.eval_runner`, which speaks
    `lerobot-rollout`'s argv verbatim). One list, two front-ends, so a flag
    added for one is never missing from the other. Coaching mode is the third
    front-end (`makermodslab.dagger_runner`) and layers `--strategy.*` /
    `--dataset.*` / `--teleop.*` on top; `robot_args` carries the teleop block
    for it, built alongside the `--robot.*` block in `_prepare_robot`."""
    coaching = request.coaching
    args = [
        # Coaching replaces the strategy wholesale (see `_coaching_cli_args`);
        # every other run is a plain autonomous rollout.
        *([] if coaching else ["--strategy.type=base"]),
        # Emitted unconditionally, including for the "sync" default — same
        # reasoning as --strategy.type=base above and the teardown pin below:
        # `inference` is a draccus ChoiceRegistry field whose default lives
        # upstream (RolloutConfig.inference = SyncInferenceConfig), so naming it
        # makes the choice ours and keeps an upstream default flip from
        # silently changing which engine drives the arm.
        #
        # Coaching is pinned to sync regardless of what was requested, and the
        # request layer refuses rtc + coaching outright so the two can't
        # disagree. On this lerobot pin, resuming autonomous control after a
        # correction leaves the RTC engine holding the PRE-correction
        # observation, so it predicts its first chunk as though the arm were
        # still where the operator found it and snaps back toward that pose
        # (lerobot issue #3747; fix PR #4398 is unmerged). That is a physical
        # hazard at the exact moment the operator has just let go.
        f"--inference.type={'sync' if coaching else request.inference_engine}",
        f"--policy.path={policy_path}",
        f"--policy.device={_detect_device()}",
        *robot_args,
        f"--task={request.task}",
        # A coaching session has no clock. `duration` would end it mid-takeover
        # with the arm under the operator's hand, and the session already has
        # two honest endings: the correction target, and the Stop button. The
        # operator is standing at the robot by definition, so an unbounded
        # session is not an unattended one.
        f"--duration={0 if coaching else request.duration_s}",
        *([f"--fps={_COACHING_FPS}"] if coaching else []),
        *(_coaching_cli_args(request) if coaching else []),
        # Pin the teardown behaviour the stop dialog promises ("eases the
        # follower back to its start pose, then goes limp"). lerobot's
        # RolloutConfig.return_to_initial_position defaults to True today,
        # but relying on that default means an upstream flip would silently
        # break the promise — the arm would stay wherever the policy left
        # it. Set it explicitly so the contract is ours, not upstream's.
        "--return_to_initial_position=true",
    ]
    # ACT temporal ensembling. `--policy.*` flags are applied by lerobot as CLI
    # overrides ON TOP of the checkpoint's saved config (RolloutConfig.__post_init__
    # → PreTrainedConfig.from_pretrained(cli_overrides=...)), so this re-tunes a
    # trained ACT policy at inference time without retraining.
    #
    # n_action_steps=1 is NOT optional: ACTConfig.__post_init__ raises
    # NotImplementedError when temporal_ensemble_coeff is set alongside
    # n_action_steps > 1, and checkpoints ship the default 100. Ensembling needs
    # the policy queried every step to have overlapping chunks to average, so
    # the two flags always travel together.
    #
    # Lives here rather than in _build_rollout_cmd so eval mode's separate entry
    # point (_build_eval_runner_cmd) carries the flags too — the whole reason
    # this list was split out.
    if request.temporal_ensemble_coeff is not None:
        args += [
            f"--policy.temporal_ensemble_coeff={request.temporal_ensemble_coeff}",
            "--policy.n_action_steps=1",
        ]
    return args


# The control-loop rate a coaching session runs and records at. Pinned to
# lerobot's own `RolloutConfig.fps` default rather than left implicit, and
# emitted on BOTH `--fps` and `--dataset.fps`: the two must agree (the dataset's
# timestamps are derived from the loop's tick rate), and inheriting an upstream
# default on one of them is exactly how they would silently drift apart.
_COACHING_FPS = 30


def _coaching_dataset_repo_id(request: InferenceRequest) -> str:
    """`rollout_<name>` for a coaching session's correction dataset.

    Bare, with no owner — the same shape `RecordingRequest.dataset_repo_id`
    carries. The Hub namespace is applied at upload time, not creation time, so
    a logged-out operator can still coach and push later.

    Two upstream constraints, neither optional:

      * lerobot REFUSES a rollout dataset whose name doesn't start with
        `rollout_` ("Dataset names for rollout must start with 'rollout_'",
        lerobot/rollout/context.py). Applied here rather than asked of the user,
        who should not have to know that a deployment dataset is a different
        kind of thing from a recorded one.
      * lerobot appends its OWN `_YYYYmmdd_HHMMSS` inside the subprocess
        (`stamp_repo_id`, called unconditionally on the create path). So this is
        the name we ASK for, not the one that will exist; the real one comes
        back over the protocol. Deliberately NOT pre-stamped here the way
        record.py stamps its own — that would produce a double timestamp.
    """
    name = (request.coaching_dataset_name or "corrections").strip().strip("/")
    # Tolerate an operator who typed the prefix themselves rather than doubling it.
    return name if name.startswith("rollout_") else f"rollout_{name}"


def _teleop_args(request: InferenceRequest, leader_id: str, leader_staging: str | None) -> list[str]:
    """The `--teleop.*` block for a coaching session.

    The leader mirrors the follower's shape: an SO-101 leader for a single arm,
    a BiSO leader wrapping two sub-arms for bimanual. `leader_staging` is the
    per-session dir the two library calibrations were staged into under BiSO's
    `<base>_left/right.json` convention; None for single-arm, where the
    calibration file is addressed by id out of the shared leader dir.

    Built here rather than in utils/robot_factory.py for the same reason the
    `--robot.*` args are: the factory assembles config OBJECTS for the
    in-process flows (teleoperate, record), while everything in this module has
    to cross a subprocess boundary as argv."""
    if request.mode == "bimanual":
        return [
            "--teleop.type=bi_so_leader",
            f"--teleop.id={leader_id}",
            f"--teleop.calibration_dir={leader_staging}",
            f"--teleop.left_arm_config.port={request.leader_port}",
            f"--teleop.right_arm_config.port={request.right_leader_port}",
        ]
    return [
        "--teleop.type=so101_leader",
        f"--teleop.port={request.leader_port}",
        f"--teleop.id={leader_id}",
    ]


def _coaching_cli_args(request: InferenceRequest) -> list[str]:
    """The DAgger-strategy flags layered on top of the shared rollout args."""
    return [
        "--strategy.type=dagger",
        # Corrections-only. The runner refuses the continuous mode outright, and
        # merge.py's "drop the intervention column" shortcut is only lossless
        # because of this: with corrections-only, EVERY recorded frame is
        # intervention=True, so the column carries no information. Flipping this
        # to true invalidates that reasoning — see merge.py.
        "--strategy.record_autonomous=false",
        f"--strategy.num_episodes={clamp_coaching_corrections(request.target_corrections)}",
        f"--dataset.repo_id={_coaching_dataset_repo_id(request)}",
        f"--dataset.fps={_COACHING_FPS}",
        f"--dataset.single_task={request.task}",
        # Encode frames as they are captured rather than in one lump at save
        # time. MEASURED on the station, 132 frames x 2 cameras at 480x640:
        # save_episode takes 2.32s with lerobot's default (False) and 0.44s
        # with this on, producing identical output and no leftover PNGs.
        # `save_episode()` runs synchronously on the control loop at the
        # hand-back edge, so that difference is time the operator spends
        # waiting with the arm frozen.
        #
        # NOTE: deliberately NOT setting `rgb_encoder.vcodec=auto` the way
        # record.py does. On this station "auto" resolves to h264_nvenc, which
        # lerobot's own `detect_available_encoders` reports as available but
        # PyAV then FAILS to open ("avcodec_open2(h264_nvenc)", Errno 22) — so
        # pinning auto breaks encoding outright. The software default
        # (libsvtav1) encodes the same episode in 0.8s, which is not a
        # bottleneck worth risking that on.
        "--dataset.streaming_encoding=true",
        # NO `--dataset.root`, deliberately — the one place this flow diverges
        # from record.py, which pins its root explicitly. lerobot stamps the
        # timestamp onto repo_id INSIDE the subprocess and then derives the root
        # from the stamped name; a root computed out here, before the stamp,
        # would point at a directory whose name no longer matches the dataset
        # and the library would never find it again. Leaving it None puts the
        # dataset at HF_LEROBOT_HOME/<stamped id>, which is exactly where
        # datasets.py looks.
        # Push is the operator's decision, made afterwards from the library, not
        # a side effect of coaching. Uploading mid-session also competes with
        # the control loop for the machine.
        "--dataset.push_to_hub=false",
    ]


def _build_rollout_cmd(request: InferenceRequest, policy_path: str, robot_args: list[str]) -> list[str]:
    """The full `lerobot-rollout` argv — one rollout, one process.

    The single-episode path (and only it): `eval_episodes == 1` runs exactly the
    command it always has."""
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_rollout",
        *_rollout_cli_args(request, policy_path, robot_args),
    ]


def _build_eval_runner_cmd(request: InferenceRequest, policy_path: str, robot_args: list[str]) -> list[str]:
    """The full `makermodslab.eval_runner` argv — one process, N episodes.

    Identical flags to `_build_rollout_cmd`, different entry point: the runner
    parses `lerobot-rollout`'s config with lerobot's own parser and then serves
    episodes off stdin instead of running exactly one and exiting. `--duration`
    becomes the PER-EPISODE limit."""
    return [
        sys.executable,
        "-m",
        "makermodslab.eval_runner",
        *_rollout_cli_args(request, policy_path, robot_args),
    ]


def _build_dagger_runner_cmd(request: InferenceRequest, policy_path: str, robot_args: list[str]) -> list[str]:
    """The full `makermodslab.dagger_runner` argv — one process, one coaching session.

    Identical flags to `_build_rollout_cmd`, different entry point: the runner
    parses `lerobot-rollout`'s config with lerobot's own parser and then serves
    takeover/hand-back commands off stdin instead of driving the strategy from a
    keyboard listener it cannot reach from a browser."""
    return [
        sys.executable,
        "-m",
        "makermodslab.dagger_runner",
        *_rollout_cli_args(request, policy_path, robot_args),
    ]


def _session_cameras(request: InferenceRequest) -> dict[str, dict[str, Any]]:
    """This run's cameras, keyed by the policy-expected name.

    Camera identity and transport settings are resolved from the robot record
    every time they're needed rather than carried on the request: the record is
    the only place they live, and the lookup is one small JSON read. Capture
    resolution is overlaid from the checkpoint (see PolicyCameraDims /
    bind_robot_cameras). Raises CameraResolutionError, which
    handle_start_inference turns into a 400 before the session starts."""
    return bind_robot_cameras(
        request.robot_name,
        request.camera_bindings,
        dims={name: dims.model_dump() for name, dims in request.camera_dims.items()},
    )


def _single_robot_args(request: InferenceRequest, follower_id: str) -> list[str]:
    """`--robot.*` args for a single SO-101 follower."""
    args = [
        "--robot.type=so101_follower",
        f"--robot.port={request.follower_port}",
        f"--robot.id={follower_id}",
    ]
    cameras = _session_cameras(request)
    if cameras:
        args.append(f"--robot.cameras={_format_cameras_arg(cameras)}")
    return args


def _bimanual_robot_args(request: InferenceRequest, base: str, follower_staging: str) -> list[str]:
    """`--robot.*` args for a bimanual BiSO follower.

    lerobot's BiSOFollowerConfig wraps two SOFollowerConfig sub-arms
    (left_arm_config / right_arm_config) sharing ONE calibration_dir + base id,
    loading each sub-arm's calibration as "<base>_left.json"/"<base>_right.json".
    `follower_staging` is the per-session dir the two library calibrations were
    staged into under that convention (see stage_bimanual_follower_calibrations).
    Cameras
    go on the LEFT arm (BiSO re-exposes them prefixed "left_*"); the right arm is
    camera-free, matching the record/teleop bimanual shape."""
    args = [
        "--robot.type=bi_so_follower",
        f"--robot.id={base}",
        f"--robot.calibration_dir={follower_staging}",
        f"--robot.left_arm_config.port={request.follower_port}",
        f"--robot.right_arm_config.port={request.right_follower_port}",
    ]
    cameras = _session_cameras(request)
    if cameras:
        args.append(f"--robot.left_arm_config.cameras={_format_cameras_arg(cameras)}")
    return args


def _prepare_robot(request: InferenceRequest) -> tuple[list[str], list[str]]:
    """Stage calibrations, run the arm-identity + motor-power preflights, and
    build the `--robot.*` argv for the rollout subprocess.

    This is the robot-TOUCHING part of startup: it opens and releases the
    follower serial bus (read-only identity check + RAM torque-limit priming).
    It runs in the background startup worker AFTER the model download, so a stop
    pressed during the (long) download never reaches here — no bus is opened and
    no register is written. Raises ArmIdentityError on a hard arm mismatch;
    returns (robot_args, warn-but-allow messages)."""
    if request.coaching:
        return _prepare_coaching_robot(request)

    is_bimanual = request.mode == "bimanual"
    if is_bimanual:
        # BiSO loads each sub-arm's calibration as "<base>_left/right.json"
        # from one dir, with no way to point left/right at differently named
        # library files. Stage the two arbitrarily-named follower library
        # calibrations into that convention and point BiSO at the staging
        # dir. Inference has NO leader arms, so stage the follower side only
        # — staging the leader side would require leader library files that
        # this flow never uses (and usually don't exist under the follower's
        # names). The copy fails fast with a clear per-slot error if a
        # library file is missing.
        base = bimanual_base_id(request.robot_name)
        follower_staging, _ = stage_bimanual_follower_calibrations(
            base,
            request.follower_config,
            request.right_follower_config,
        )
        # Sub-arm ids are the BiSO staging aliases ("<base>_left/right"), so
        # the identity guard compares against the real library stems.
        left_id, right_id = f"{base}_left", f"{base}_right"

        identity_warnings: list[str] = []
        if request.skip_identity_check:
            logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
        else:
            # Each bus opens/verifies/releases sequentially — never both at
            # once — mirroring the single-arm preflight.
            identity_warnings += _preflight_arm_identity(
                request.follower_port, left_id, config_name=request.follower_config
            )
            identity_warnings += _preflight_arm_identity(
                request.right_follower_port, right_id, config_name=request.right_follower_config
            )
        # Register reset on both buses, sequentially (each opens its own port).
        identity_warnings += _preflight_motor_registers(request.follower_port, left_id)
        identity_warnings += _preflight_motor_registers(request.right_follower_port, right_id)

        return _bimanual_robot_args(request, base, follower_staging), identity_warnings

    # `setup_follower_calibration_file` returns the basename without the
    # .json extension. We need that stripped form for `--robot.id`,
    # because lerobot appends `.json` itself when constructing
    # `calibration_dir / f"{id}.json"`.
    follower_id = setup_follower_calibration_file(request.follower_config)

    # Arm-identity guard: refuse before the subprocess can move (or stamp
    # the wrong calibration into) an arm that doesn't match its file.
    identity_warnings = []
    if request.skip_identity_check:
        logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
    else:
        identity_warnings = _preflight_arm_identity(request.follower_port, follower_id)

    # Always reset so a previous auto-calibration's torque cap can't linger
    # when the arm was never power-cycled.
    identity_warnings += _preflight_motor_registers(request.follower_port, follower_id)

    return _single_robot_args(request, follower_id), identity_warnings


def _prepare_coaching_robot(request: InferenceRequest) -> tuple[list[str], list[str]]:
    """`_prepare_robot` for a coaching session: followers AND leaders.

    Split out rather than branched into the main body because it is the only
    inference flow with a leader side, and folding four extra staging/preflight
    steps into a function whose every comment says "inference has no leader
    arms" would make both paths harder to read.

    Returns `(argv, warnings)` where argv carries the `--robot.*` block followed
    by the `--teleop.*` one — `_rollout_cli_args` splices it in whole, so the
    leader travels the same path the follower already does.

    The leader gets the SAME arm-identity preflight the follower gets. It is not
    a passive device here: torque is enabled on it during the pause so it can be
    driven to the follower's pose, so an unrecognised arm on that port is just
    as capable of moving unexpectedly."""
    identity_warnings: list[str] = []

    if request.mode == "bimanual":
        # Both sides staged, unlike the follower-only inference path — a
        # coaching session genuinely drives the leaders, so their library
        # calibrations must exist and be staged under BiSO's naming convention.
        base = bimanual_base_id(request.robot_name)
        leader_staging, follower_staging, _ = stage_bimanual_calibrations(
            base,
            request.leader_config,
            request.right_leader_config,
            request.follower_config,
            request.right_follower_config,
        )
        left_id, right_id = f"{base}_left", f"{base}_right"

        if request.skip_identity_check:
            logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
        else:
            # Each bus opens/verifies/releases sequentially — never two at once.
            identity_warnings += _preflight_arm_identity(
                request.follower_port, left_id, config_name=request.follower_config
            )
            identity_warnings += _preflight_arm_identity(
                request.right_follower_port, right_id, config_name=request.right_follower_config
            )
            # The counterpart slot is the follower's LIBRARY stem, not the BiSO
            # staging alias — the identity library is keyed by library names
            # (same reason `config_name` is passed alongside the alias id).
            identity_warnings += _preflight_leader_identity(
                request.leader_port,
                left_id,
                request.follower_config,
                config_name=request.leader_config,
            )
            identity_warnings += _preflight_leader_identity(
                request.right_leader_port,
                right_id,
                request.right_follower_config,
                config_name=request.right_leader_config,
            )
        # Register reset on the FOLLOWER buses only. `reset_torque_limit` exists
        # to undo an autocal's torque cap on the arm that will be driven under
        # load; the leaders are back-driven by hand and a cap there is harmless.
        identity_warnings += _preflight_motor_registers(request.follower_port, left_id)
        identity_warnings += _preflight_motor_registers(request.right_follower_port, right_id)

        robot_args = _bimanual_robot_args(request, base, follower_staging)
        return robot_args + _teleop_args(request, base, leader_staging), identity_warnings

    # Single arm. `setup_calibration_files` is the shared helper teleoperation
    # and recording already use for exactly this pair; it returns both basenames
    # without their .json extension, which is the form `--robot.id`/`--teleop.id`
    # want (lerobot appends the extension itself).
    leader_id, follower_id = setup_calibration_files(request.leader_config, request.follower_config)

    if request.skip_identity_check:
        logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
    else:
        identity_warnings += _preflight_arm_identity(request.follower_port, follower_id)
        identity_warnings += _preflight_leader_identity(request.leader_port, leader_id, follower_id)

    identity_warnings += _preflight_motor_registers(request.follower_port, follower_id)

    robot_args = _single_robot_args(request, follower_id)
    return robot_args + _teleop_args(request, leader_id, None), identity_warnings


def _fail_startup(error: str) -> None:
    """Record a background-startup failure (download or preflight — before any
    subprocess exists) as the terminal `_last_result` payload, reusing the exact
    outcome/error/hint contract the subprocess-exit path already exposes so the
    inference page surfaces it the same way (and keeps surfacing it on every
    poll until the next run starts).

    A no-op when a stop already tore the session down (inference_active False):
    the stop wins, and a download that raised while being abandoned must not
    resurrect a phantom failure.

    Session-level, not episode-level: this is the download/preflight/first-spawn
    window, so in eval mode there is nothing to score yet — the eval session is
    dropped and the failure is reported the same way a single run's would be."""
    with _state_lock:
        _fail_startup_locked(error)


def _fail_startup_locked(error: str) -> None:
    """`_fail_startup`'s body, for callers that already hold `_state_lock`.

    Split out for the eval-runner crash path: a runner that dies before its
    FIRST episode ever started is a startup failure, not an episode to score,
    and that determination is made deep inside a locked section."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _last_result, _eval_session
    if not inference_active:
        return
    policy_ref = _inference_meta.get("policy_ref")
    finished_eval = _eval_session
    inference_active = False
    _inference_proc = None
    _inference_started_at = None
    _inference_rollout_started_at = None
    _inference_meta = {}
    _eval_session = None
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": None,
        "outcome": "failed",
        "error": error,
        "hint": friendly_hint(error),
        "phase": PHASE_ERROR,
        "policy_ref": policy_ref,
        "duration_s": None,
        "log_path": None,
        "started_at": None,
        "rollout_started_at": None,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(finished_eval),
    }
    # Final release (startup failure path). Caller holds _state_lock; the
    # notify is a lock-free droppable queue put, so this cannot deadlock.
    notify_session_changed("inference", False, phase=PHASE_ERROR)


def _spawn_rollout_process(
    cmd: list[str],
    stdin_seed: bytes,
    *,
    close_stdin: bool,
) -> tuple[subprocess.Popen, IO[str], Path]:
    """Open a fresh log file, spawn `cmd`, and seed its stdin.

    Shared by both entry points (one-shot `lerobot-rollout` and the persistent
    eval runner) — they differ only in argv and in whether stdin stays open.
    Returns (proc, log_handle, log_path); the caller owns committing them to the
    module state and starting the stdout pump. Raises on spawn failure (after
    closing the log handle) — the caller decides how to report it, since a
    first-episode failure fails the session while a later one fails just that
    episode."""
    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{int(time.time())}.log"
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
            # Own process group, so teardown can reap the WHOLE tree.
            #
            # LeRobotDataset spawns image-writer subprocesses of its own, and
            # they hold camera and file handles. `proc.kill()` only signals the
            # direct child, so a runner that had to be force-killed left those
            # writers behind — still holding /dev/video*, so the NEXT session
            # could not open the cameras. Observed on the station: two orphaned
            # dagger_runner processes surviving SIGTERM by eight minutes.
            #
            # posix-only; on Windows the flag is ignored by `_terminate_tree`,
            # which falls back to signalling the process alone.
            start_new_session=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            log_handle.close()
        raise
    try:
        assert proc.stdin is not None
        proc.stdin.write(stdin_seed)
        proc.stdin.flush()
        if close_stdin:
            proc.stdin.close()
    except Exception as exc:
        logger.warning("Failed to seed stdin for inference subprocess: %s", exc)
    return proc, log_handle, log_path


def _signal_group(proc: subprocess.Popen, signum: int) -> bool:
    """Signal the runner's whole process group. False when that isn't possible.

    Returns False (rather than raising) for every reason the group route can be
    unavailable — no `pid`, no process groups on this platform, the group
    already gone, or a test stand-in that isn't a real Popen — so the caller can
    fall back to signalling the single process.

    REFUSES to signal our own process group. `start_new_session=True` puts the
    runner in its own, so this should never trigger; if it ever did, the call
    would take down the FastAPI server along with the runner, and a wedged
    camera is a far better outcome than a dead app."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        return False
    try:
        if pgid == os.getpgid(0):
            logger.warning("Runner shares our process group; signalling it alone")
            return False
    except Exception:
        return False
    try:
        os.killpg(pgid, signum)
        return True
    except Exception as exc:
        logger.debug("killpg failed (%s); signalling the process alone", exc)
        return False


def _terminate_tree(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Signal a runner AND its children, escalating SIGTERM -> SIGKILL.

    `Popen.terminate()`/`kill()` reach only the direct child. The runner forks
    image writers (LeRobotDataset's `image_writer_processes`), and those keep
    the cameras open after their parent dies — which is what left the NEXT
    session unable to connect. Spawned with `start_new_session=True`, the runner
    leads its own process group, so one `killpg` takes the tree down together.

    Degrades to signalling the process alone whenever the group route is
    unavailable, so it stays correct on non-posix and under test doubles."""
    for signum, fallback in ((signal.SIGTERM, "terminate"), (signal.SIGKILL, "kill")):
        with contextlib.suppress(Exception):
            if proc.poll() is not None:
                return
        if not _signal_group(proc, signum):
            with contextlib.suppress(Exception):
                getattr(proc, fallback)()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            logger.warning("Runner did not exit %.0fs after %s", timeout, fallback)
        except Exception:
            return
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


def _stdin_seed(request: InferenceRequest) -> bytes:
    """Newlines to pre-answer lerobot's per-arm calibration prompt.

    Feed a newline into stdin PER follower arm so SOFollower.calibrate()'s
    `input("Press ENTER to use the calibration file ...")` returns "" and
    writes the existing calibration to the motors instead of hanging
    forever waiting for an interactive operator. A BiSO follower connects
    its two sub-arms sequentially (left then right), each of which can fire
    that prompt once — so seed two newlines for bimanual, one for single.
    Any prompt that doesn't fire just leaves an unread newline (harmless);
    for the one-shot path subsequent input() calls in the recalibration path get
    EOF and raise — fine, because we never want to enter that path from the UI.
    For the eval runner, whose stdin stays open as a command channel, an
    unconsumed newline surfaces as a blank command line and is ignored.

    A COACHING session connects LEADERS as well, and `SOLeader.calibrate()`
    fires the same prompt, so it needs one newline per arm on BOTH sides:
    two for single (follower + leader), four for bimanual."""
    arms = 2 if request.mode == "bimanual" else 1
    sides = 2 if request.coaching else 1
    return b"\n" * (arms * sides)


def _launch_rollout_subprocess(
    request: InferenceRequest,
    policy_path: str,
    robot_args: list[str],
) -> tuple[subprocess.Popen, IO[str], Path]:
    """Spawn ONE `lerobot-rollout` process — the single-episode path.

    stdin is closed right after the seed: this process is asked for nothing
    after it starts, and an open pipe would only be something to leak."""
    return _spawn_rollout_process(
        _build_rollout_cmd(request, policy_path, robot_args),
        _stdin_seed(request),
        close_stdin=True,
    )


def _launch_eval_runner(
    request: InferenceRequest,
    policy_path: str,
    robot_args: list[str],
) -> tuple[subprocess.Popen, IO[str], Path]:
    """Spawn the persistent eval runner — one process for the whole session.

    stdin STAYS OPEN: it is the command channel (`EPISODE` / `STOP` / `QUIT`)
    that replaces spawning-and-killing a process per episode."""
    return _spawn_rollout_process(
        _build_eval_runner_cmd(request, policy_path, robot_args),
        _stdin_seed(request),
        close_stdin=False,
    )


def _launch_dagger_runner(
    request: InferenceRequest,
    policy_path: str,
    robot_args: list[str],
) -> tuple[subprocess.Popen, IO[str], Path]:
    """Spawn the coaching runner — one process for the whole session.

    stdin STAYS OPEN: it is the command channel (`TAKEOVER` / `HANDBACK` /
    `CANCEL` / `HOLD` / `RESUME` / `QUIT`) that replaces the keyboard listener
    upstream's DAgger strategy installs, which a browser cannot reach."""
    return _spawn_rollout_process(
        _build_dagger_runner_cmd(request, policy_path, robot_args),
        _stdin_seed(request),
        close_stdin=False,
    )


# How long a QUIT gets to land before we escalate to SIGTERM. Has to cover
# lerobot's teardown: the 3 s ease-home interpolation plus the bus and camera
# disconnects. Escalating early would kill the arm mid-motion — the exact thing
# the clean-shutdown command exists to avoid.
_RUNNER_QUIT_TIMEOUT_S = 10.0


def _send_runner_command(proc: subprocess.Popen | None, command: str) -> bool:
    """Write one command line to the eval runner's stdin. True when it landed.

    Never raises: a dead runner (BrokenPipeError) or a closed pipe is a real
    state the caller has to handle — the crash-containment path — not an
    exception to unwind an HTTP handler with. Returning False lets the caller
    say so in a status payload instead."""
    if proc is None or proc.stdin is None:
        return False
    try:
        proc.stdin.write(f"{command}\n".encode())
        proc.stdin.flush()
        return True
    except Exception as exc:
        logger.warning("Could not send %s to the eval runner: %s", command, exc)
        return False


def _quit_runner(proc: subprocess.Popen) -> None:
    """End the eval runner cleanly, escalating only if it doesn't answer.

    QUIT is the happy path: the runner breaks out of any running episode, eases
    the follower home and disconnects the bus and cameras before exiting — the
    same teardown a one-shot rollout does at the end of its process. SIGTERM is
    the fallback for a runner that is wedged (and the runner installs lerobot's
    signal handler, so even that is asked-nicely first); SIGKILL is the last
    resort. Blocking, and called off the lock."""
    _send_runner_command(proc, CMD_QUIT)
    try:
        proc.wait(timeout=_RUNNER_QUIT_TIMEOUT_S)
        return
    except subprocess.TimeoutExpired:
        logger.warning("Runner did not exit %.0fs after QUIT; terminating", _RUNNER_QUIT_TIMEOUT_S)
    except Exception as exc:
        logger.exception("Waiting for the runner to quit failed: %s", exc)
        return
    # Whole tree, not just the runner — see `_terminate_tree`. A runner wedged
    # inside a slow `save_episode()` cannot answer QUIT, and leaving its image
    # writers behind is what blocks the next session's cameras.
    _terminate_tree(proc)


def _run_inference_startup(request: InferenceRequest, cancel_event: threading.Event) -> None:
    """Background startup sequence for one rollout: download the model (with byte
    progress), preflight the arm, then spawn the rollout subprocess.

    Runs off the request thread so POST /start-inference returns immediately and
    the UI lands on the inference page while the (possibly multi-minute) Hub
    download runs there with a progress bar. Ordered download → preflight → spawn
    so a stop pressed DURING the download never opens the serial bus or spawns a
    subprocess ("no robot touched"). snapshot_download can't be interrupted
    mid-flight, so a stop during the download abandons this worker: the download
    finishes into the HF cache (cached for next time) and the worker bails at the
    next cancel check without preflighting or spawning. Terminal download/
    preflight failures flow through _fail_startup into the shared outcome/error/
    hint status machinery."""
    global _inference_proc, _inference_rollout_started_at, _inference_meta, _last_log_path

    # 1. Resolve/download the policy. A Hub ref streams byte progress into the
    #    meta; a local dir returns instantly (no downloading_model phase, no
    #    robot touched yet).
    try:
        policy_path = _resolve_policy_path(request.policy_ref, report=_report_download_progress)
    except Exception as exc:
        logger.exception("Inference model download failed")
        _fail_startup(f"Failed to download the model: {exc}")
        return
    # Stop during the download → abandon (stop already set the state idle).
    if cancel_event.is_set():
        logger.info("Inference startup abandoned during model download (stop requested)")
        return

    # 2. Preflight + stage the arm (opens the serial bus). This is the first
    #    robot-touching step, deliberately AFTER the download.
    try:
        robot_args, identity_warnings = _prepare_robot(request)
    except ArmIdentityError as exc:
        # The connected arm doesn't match its assigned calibration; the message
        # is already user-facing.
        _fail_startup(str(exc))
        return
    except Exception as exc:
        logger.exception("Failed to prepare robot for inference")
        _fail_startup(f"Failed to start inference: {exc}")
        return
    if cancel_event.is_set():
        logger.info("Inference startup abandoned after preflight (stop requested)")
        return

    # 3. Spawn the subprocess. Single-episode runs get one `lerobot-rollout`, as
    #    they always have; eval mode gets ONE `makermodslab.eval_runner` that will
    #    serve every episode of the session, so the policy load and the bus and
    #    camera connect are paid once instead of N times. Cache what the later
    #    episodes reuse verbatim (the resolved path and the preflighted
    #    `--robot.*` args) so even a crash-respawn skips the download and the
    #    second arm-identity pass.
    with _state_lock:
        is_eval = _eval_session is not None
        is_coaching = _coach_session is not None
        if _eval_session is not None:
            _eval_session.policy_path = policy_path
            _eval_session.robot_args = list(robot_args)
            # Episode 1 is issued by the READY handler, once the runner has
            # finished loading — it is pending from this moment.
            _eval_session.episode_pending = True

    if is_coaching:
        launch = _launch_dagger_runner
    elif is_eval:
        launch = _launch_eval_runner
    else:
        launch = _launch_rollout_subprocess
    try:
        proc, log_handle, log_path = launch(request, policy_path, robot_args)
    except Exception as exc:
        logger.exception("Failed to spawn rollout subprocess")
        _fail_startup(f"Failed to start inference: {exc}")
        return

    # Commit the subprocess under the lock, re-checking the cancel flag: a stop
    # that raced the spawn must NOT leave a live subprocess driving the arm.
    with _state_lock:
        abandoned = cancel_event.is_set() or not inference_active
        if not abandoned:
            _inference_proc = proc
            _inference_rollout_started_at = None
            # Carry forward any phase the not-yet-started pump could set later;
            # the download phase is behind us, so `starting` is the floor.
            carried_phase = _inference_meta.get("phase") or PHASE_STARTING
            if carried_phase == PHASE_DOWNLOADING_MODEL:
                carried_phase = PHASE_STARTING
            meta: dict[str, Any] = {
                "policy_ref": request.policy_ref,
                # The RESOLVED local checkpoint dir (policy_ref can be a Hub ref,
                # fragile for path comparisons) — read by inference_in_use_path so
                # models.delete_local_model can refuse deleting it mid-run.
                "policy_path": policy_path,
                # 0 for coaching, and it must be: a coaching session runs
                # `--duration=0` (unbounded — it ends on the correction target
                # or the Stop button), and the dialog's hung-run safety net
                # fires on `rollout_elapsed_s > duration_s + 10`. Reporting the
                # request's nominal duration here would have that net stop a
                # perfectly healthy session ten seconds past a clock it is not
                # running against — quite possibly mid-takeover.
                "duration_s": 0 if request.coaching else request.duration_s,
                "log_path": str(log_path),
                "phase": carried_phase,
            }
            # Warn-but-allow arm-identity findings, surfaced once via the status
            # payload now that the POST returned before the preflight ran.
            if identity_warnings:
                meta["warning"] = " ".join(identity_warnings)
            _inference_meta = meta
            # This run now owns a log file. Recorded outside the meta too, so it
            # survives the meta being cleared at session end and the endpoint can
            # still serve THIS run's log (labelled last_run) afterwards.
            _last_log_path = str(log_path)

    if abandoned:
        # Stopped during/just after the spawn — kill the subprocess we just
        # started and leave the (already idle) state alone. The SIGTERM is
        # escalated because both entry points now handle it gracefully, which
        # means "finish loading the policy first" — a wait that can outlast the
        # 5 s. Leaving the orphan behind would keep the serial bus open and
        # block the next session.
        logger.info("Inference startup abandoned after spawn (stop requested); killing subprocess")
        _terminate_tree(proc)
        with contextlib.suppress(Exception):
            log_handle.close()
        return

    # Start the stdout pump only after committing, so it never advances the phase
    # of a subprocess we might have abandoned above.
    if is_coaching:
        pump = _pump_dagger_stdout
    elif is_eval:
        pump = _pump_runner_stdout
    else:
        pump = _pump_stdout
    threading.Thread(
        target=pump,
        args=(proc, log_handle),
        name="inference-stdout-pump",
        daemon=True,
    ).start()
    logger.info(
        "Inference started: pid=%s policy=%s eval=%s coaching=%s",
        proc.pid,
        policy_path,
        is_eval,
        is_coaching,
    )


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Validate the request cheaply and hand the heavy startup (model download →
    arm preflight → subprocess spawn) to a background worker, returning
    immediately.

    Returns a dict — the route layer turns it into a JSON response or
    HTTPException as appropriate. Only cheap, synchronous checks stay here
    (mutex, arm-count guard, policy-ref shape) so a 4xx still surfaces in the
    launch modal; the multi-minute Hub download moves off the request thread so
    the UI lands on the inference page and shows download progress there."""
    global inference_active, _inference_started_at, _inference_meta, _inference_cancel
    global _last_result, _last_log_path, _inference_startup_thread, _eval_session, _coach_session

    # Mutex with every other feature that drives the same serial bus (see
    # CLAUDE.md's "State model & mutual exclusion").
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        replay as _replay,
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
        if inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is already active. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_INFERENCE,
            }
        if _inference_startup_thread is not None and _inference_startup_thread.is_alive():
            # A previous session was stopped while its startup worker was
            # inside _prepare_robot (already touching hardware) or still
            # unwinding just after — inference_active is already False, but
            # the worker itself hasn't exited yet. Starting a new session now
            # would open the same serial port out from under it. Refuse until
            # it's actually gone.
            return {
                "success": False,
                "status_code": 409,
                "message": "The previous session is still shutting down. Try again in a few seconds.",
                "code": ErrorCode.ROBOT_BUSY_RELEASING,
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
        # Lazy, because jobs imports this module back the same way. Inference is
        # the worst pairing: both want several GB of VRAM, and whichever loses
        # takes a CUDA OOM — if that is the trainer, hours of work end as
        # "Subprocess exited with code 1" with nothing tying it to this click.
        from . import jobs as _jobs

        if (training := _jobs.training_is_active()) is not None:
            return {
                "success": False,
                "status_code": 409,
                "message": f"Training run '{training}' is using this machine. Stop it first.",
                "code": ErrorCode.ROBOT_BUSY_TRAINING,
            }
        # Claim the slot now so a concurrent caller losing the race sees us, and
        # seed the meta + timer so the phase is visible from the very first
        # status poll (the download runs on the inference page — the UI must be
        # able to name that wait before the subprocess even exists). A fresh
        # cancel Event lets stop() abandon the pre-subprocess window.
        inference_active = True
        _inference_started_at = time.time()
        _inference_cancel = threading.Event()
        cancel_event = _inference_cancel
        _inference_meta = {"phase": PHASE_STARTING, "policy_ref": request.policy_ref}
        # A new run supersedes the previous run's terminal payload — status
        # polls must reflect THIS session from the first tick.
        _last_result = None
        # ...and its log. Until THIS session opens its own log file, there is no
        # log to show: the pre-spawn phases (download / preflight) produce no
        # subprocess output, and inheriting the previous run's path here is
        # exactly how a stale log used to be presented as the current one.
        _last_log_path = None
        # Eval mode is decided once, here, and clamped: episodes > 1 seeds the
        # session bookkeeping so the very first status poll already reports
        # "episode 1 / N". A count of 1 leaves `_eval_session` None, which is
        # what keeps the historical single-rollout flow untouched.
        episodes = clamp_eval_episodes(request.eval_episodes)
        _eval_session = (
            _EvalSession(request=request, episodes_total=episodes)
            if episodes > 1 and not request.coaching
            else None
        )
        # Coaching is decided here too, and is exclusive with eval (the
        # validation above this call refuses the combination). Seeding it now
        # means the very first status poll already reports "0 / N corrections"
        # rather than a bare rollout the UI would render as a plain run.
        _coach_session = (
            _CoachSession(
                request=request,
                corrections_target=clamp_coaching_corrections(request.target_corrections),
            )
            if request.coaching
            else None
        )

    # The claim above is the real state transition — broadcast the hint so
    # every WS client (any page, any remote UI) refetches /inference-status.
    notify_session_changed("inference", True, phase=PHASE_STARTING)

    def _release_slot() -> None:
        global inference_active, _inference_started_at, _inference_cancel, _inference_meta
        global _eval_session, _coach_session
        with _state_lock:
            inference_active = False
            _inference_started_at = None
            _inference_cancel = None
            _inference_meta = {}
            _eval_session = None
            _coach_session = None
        # The pre-spawn guards below released the just-claimed slot: undo the
        # claim's active=True hint.
        notify_session_changed("inference", False)

    # --- Coaching-mode guards -------------------------------------------------
    # All cheap and synchronous, so a misconfigured coaching launch 4xxs in the
    # panel rather than failing minutes later, after a model download, as an
    # opaque subprocess crash.
    if request.coaching:
        if request.eval_episodes and clamp_eval_episodes(request.eval_episodes) > 1:
            _release_slot()
            return {
                "success": False,
                "status_code": 400,
                "message": (
                    "A coaching session can't also be an evaluation. "
                    "Evaluation scores episodes the policy runs alone; coaching records "
                    "the moments you take over."
                ),
            }
        if request.inference_engine == "rtc":
            # Refused server-side, not merely hidden in the UI: on this lerobot
            # pin the RTC engine keeps the pre-correction observation across a
            # hand-back and snaps the arm toward where it used to be
            # (lerobot #3747). That is a physical hazard at the exact instant
            # the operator has just let go of the leader.
            _release_slot()
            return {
                "success": False,
                "status_code": 400,
                "message": (
                    "Coaching runs on the standard (sync) inference engine. "
                    "Real-Time Chunking makes the arm jump back toward its pre-correction "
                    "pose when the policy resumes, which isn't safe with a hand nearby."
                ),
            }
        missing = []
        if not request.leader_port:
            missing.append("leader port")
        if not request.leader_config:
            missing.append("leader calibration")
        if request.mode == "bimanual":
            if not request.right_leader_port:
                missing.append("right leader port")
            if not request.right_leader_config:
                missing.append("right leader calibration")
        if missing:
            _release_slot()
            return {
                "success": False,
                "status_code": 400,
                "message": (
                    f"Coaching needs a leader arm to take over with — missing: {', '.join(missing)}."
                ),
            }
        # The name being non-empty is not the same as the calibration existing.
        # A robot record can name a leader calibration that was since deleted or
        # renamed (and a name like "None" is a perfectly legal file stem, so it
        # cannot be treated as unset). Checking the file here turns what would
        # be a failure deep inside the runner — after the model download and the
        # arm preflight — into a 400 in the launch panel.
        for label, name in (
            ("leader", request.leader_config),
            *((("right leader", request.right_leader_config),) if request.mode == "bimanual" else ()),
        ):
            stem = os.path.splitext(name)[0]
            if not os.path.isfile(os.path.join(LEADER_CONFIG_PATH, f"{stem}.json")):
                _release_slot()
                return {
                    "success": False,
                    "status_code": 400,
                    "message": (
                        f'The {label} arm\'s calibration "{stem}" no longer exists. '
                        "Re-assign or re-calibrate it in Robot settings."
                    ),
                }
        name_ok, name_reason = validate_dataset_repo_id(_coaching_dataset_repo_id(request))
        if not name_ok:
            _release_slot()
            return {"success": False, "status_code": 400, "message": name_reason}
        if not request.task.strip():
            # The task string is written into every recorded frame and is what a
            # language-conditioned policy is fine-tuned against. An empty one
            # silently produces a dataset that can't be used with SmolVLA/pi0.
            _release_slot()
            return {
                "success": False,
                "status_code": 400,
                "message": "Describe the task before coaching — it's saved with every correction.",
            }

    # Arm-count guard: reject a single-arm checkpoint on a bimanual robot (and
    # vice versa) BEFORE spawning the worker, where the shape mismatch would
    # otherwise crash unexplained. Cheap (no I/O) — defers to the subprocess when
    # the checkpoint doesn't expose observation.state.
    mismatch = _arm_count_mismatch(request.mode, request.checkpoint_state_dim)
    if mismatch is not None:
        _release_slot()
        return {"success": False, "status_code": 409, "message": mismatch}

    # Temporal-ensemble coefficient: lerobot builds the ensemble weights as
    # exp(-coeff * i), so a non-positive value is meaningless (0 weights every
    # step of the chunk equally; negative inverts the decay so the STALEST
    # prediction dominates). Reject here rather than letting draccus accept it
    # and the arm move under a nonsense action.
    if request.temporal_ensemble_coeff is not None and request.temporal_ensemble_coeff <= 0:
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": (
                f"temporal_ensemble_coeff must be greater than 0 (got {request.temporal_ensemble_coeff})."
            ),
        }

    # Cheap policy-ref shape check so a malformed ref 4xxs in the modal instead
    # of failing later on the inference page (one is_dir stat, no network).
    if not _policy_ref_is_valid(request.policy_ref):
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"Unrecognised policy ref: {request.policy_ref!r}",
        }

    # Resolve the camera bindings against the robot record now (one small JSON
    # read, no hardware). A binding that names a camera the record doesn't have
    # must 4xx in the panel; deferring it to the startup worker would surface
    # the same mistake as a mid-startup failure after the model download.
    try:
        _session_cameras(request)
    except CameraResolutionError as exc:
        _release_slot()
        logger.warning("Rejected inference start: %s", exc)
        return {"success": False, "status_code": 400, "message": str(exc)}

    # Backend camera previews hold the cv2 devices the rollout subprocess is about
    # to open. Released here — after the cheap guards above, so a rejected request
    # doesn't needlessly kill the modal's previews, and while `inference_active`
    # is already True so /camera-preview 409s instead of re-acquiring a device.
    camera_preview_manager.stop_all()

    # Everything heavy (download, preflight, spawn) runs off the request thread.
    # Tracked so a later start can tell whether a stopped session's worker is
    # still alive (see the is_alive() guard above) instead of racing it.
    worker = threading.Thread(
        target=_run_inference_startup,
        args=(request, cancel_event),
        name="inference-startup",
        daemon=True,
    )
    _inference_startup_thread = worker
    worker.start()
    return {"success": True, "message": "Inference starting"}


def inference_in_use_path() -> str | None:
    """The RESOLVED local policy path the running inference is reading, or None
    when no inference is active.

    The meta's ``policy_ref`` can be a Hub ref (``user/repo@root``), which is
    fragile for path comparisons — this is the local directory
    ``_resolve_policy_path`` returned, captured at start. Guarded by
    _state_lock (short critical section). Consumed by ``models._model_in_use``
    so deleting a checkpoint a live inference is reading is refused."""
    with _state_lock:
        if not inference_active:
            return None
        return _inference_meta.get("policy_path")


def _go_idle_locked() -> None:
    """Drop every per-session global back to the idle shape.

    Caller must hold `_state_lock`. Does NOT touch `_last_result` — whether a
    teardown leaves a terminal payload behind is the caller's decision."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _eval_session, _coach_session
    inference_active = False
    _inference_proc = None
    _inference_started_at = None
    _inference_rollout_started_at = None
    _inference_meta = {}
    _eval_session = None
    _coach_session = None
    # Final release. Caller holds _state_lock; the notify is a lock-free
    # droppable queue put, so this cannot deadlock.
    notify_session_changed("inference", False)


def _abort_eval_locked(ev: _EvalSession) -> None:
    """End an eval session early and leave the ABORTED terminal payload behind.

    Caller must hold `_state_lock`. Partial tally, and deliberately NO accuracy:
    a session the user cut short says nothing about the policy's success rate
    over N episodes, so claiming one would be a lie. The episode that was running
    when the abort landed is simply not scored."""
    global _last_result
    finished_meta = _inference_meta
    finished_started = _inference_started_at
    _go_idle_locked()
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": None,
        "outcome": "ok",
        "error": ev.error,
        "hint": ev.hint,
        "phase": PHASE_ABORTED,
        "policy_ref": finished_meta.get("policy_ref"),
        "duration_s": finished_meta.get("duration_s"),
        "log_path": finished_meta.get("log_path"),
        "started_at": finished_started,
        "rollout_started_at": None,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(ev),
        **_coach_fields(None),
    }


def _finalise_coaching_locked(rc: int | None, cs: _CoachSession, *, aborted: bool = False) -> None:
    """End a coaching session and leave its terminal payload behind.

    Caller must hold `_state_lock`. One function for all three endings, because
    they differ only in the phase they report and none of them scores anything:

      * the runner returned on its own, having collected the target (`finished`)
      * the user stopped it early (`aborted`, partial tally kept)
      * the runner died (`error`, with the cause mined out)

    Unlike an aborted EVAL — which must not claim an accuracy it did not
    measure — a stopped coaching session loses nothing by reporting its partial
    tally: every correction it saved is on disk and is exactly as useful as it
    would have been had the session run to target. The count is a description,
    not a claim.
    """
    global _last_result
    finished_meta = _inference_meta
    finished_started = _inference_started_at
    finished_rollout_started = _inference_rollout_started_at
    # The runner's own ERROR line beats mining the log tail; fall back to the
    # tail only when it died without saying why.
    error = None
    if rc:
        error = cs.runner_error or _extract_error_from_log(finished_meta.get("log_path"))
    if aborted:
        terminal_phase = PHASE_ABORTED
    elif rc:
        terminal_phase = PHASE_ERROR
    else:
        terminal_phase = PHASE_FINISHED
    outcome = "ok" if aborted else _classify_outcome(rc, finished_rollout_started is not None, error)
    _go_idle_locked()
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": rc,
        "outcome": outcome,
        "error": error,
        "hint": friendly_hint(error),
        "phase": terminal_phase,
        "policy_ref": finished_meta.get("policy_ref"),
        "duration_s": finished_meta.get("duration_s"),
        "log_path": finished_meta.get("log_path"),
        "started_at": finished_started,
        "rollout_started_at": finished_rollout_started,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(None),
        # The coaching block SURVIVES into the terminal payload, unlike the live
        # session it describes. It carries the dataset name and the tally — the
        # two things the follow-up card needs to offer "merge this" and
        # "fine-tune on it" — and those must outlive the session that produced
        # them or the operator is left to find the dataset by hand.
        **_coach_fields(cs),
    }
    logger.info(
        "Coaching session ended (%s): %d/%d corrections, dataset=%s",
        terminal_phase,
        cs.corrections_saved,
        cs.corrections_target,
        cs.dataset_repo_id,
    )


def handle_coaching_command(command: str) -> dict[str, Any]:
    """Forward one operator command to the coaching runner.

    The five verbs (`TAKEOVER` / `HANDBACK` / `CANCEL` / `HOLD` / `RESUME`) all
    land here; the runner is what interprets them against the current phase.
    Deliberately does NOT pre-check the phase server-side: the runner's phase is
    authoritative and this process only ever holds a copy that is one event
    stale, so refusing here would sometimes reject a command the arm was
    perfectly ready for. An invalid command is a no-op the runner logs.

    QUIT is not accepted — ending the session is `/stop-inference`, which also
    releases the slot and writes the terminal payload."""
    verb = (command or "").strip().upper()
    if verb == DAGGER_CMD_QUIT or verb not in DAGGER_COMMANDS:
        return {
            "success": False,
            "status_code": 400,
            "message": f"Unrecognised coaching command: {command!r}",
        }
    with _state_lock:
        cs = _coach_session
        proc = _inference_proc
        if not inference_active or cs is None:
            return {"success": False, "status_code": 409, "message": "No coaching session is active"}
        if cs.quitting:
            return {"success": False, "status_code": 409, "message": "The session is shutting down"}
        if proc is None:
            # Still in the pre-subprocess window (model download / arm
            # preflight). There is nothing to command yet, and saying so beats
            # a silent no-op the operator reads as a dead button.
            return {
                "success": False,
                "status_code": 409,
                "message": "The session is still starting up",
            }
    # Written outside the lock: the pipe can block if the runner is wedged, and
    # holding `_state_lock` through that would stall every status poll.
    if not _send_runner_command(proc, verb):
        return {
            "success": False,
            "status_code": 409,
            "message": "The coaching session is not responding",
        }
    return {"success": True, "message": f"{verb.capitalize()} sent"}


def handle_stop_inference() -> dict[str, Any]:
    """Abort the WHOLE session — single run or eval.

    In eval mode this is the session-level stop, not the per-episode one: it
    ends the run wherever it is (mid-episode or parked in a reset) and reports
    the partial tally under the `aborted` phase. Ending only the current episode
    while keeping the session alive is `handle_stop_episode`."""
    with _state_lock:
        session_active = inference_active
        orphaned_worker = _inference_startup_thread if not session_active else None

    if not session_active:
        if orphaned_worker is None or not orphaned_worker.is_alive():
            return {"success": False, "status_code": 409, "message": "No inference is active"}
        # A previous stop already fired (inference_active is False), but that
        # session's startup worker is still stuck inside _prepare_robot with no
        # way to be interrupted mid-call. This is the "press Stop again" gesture
        # (mirrors teleoperate.py's second stop): bounded-wait for it and report
        # honestly, instead of repeating a blanket "nothing to stop" that hides
        # a worker still touching the serial bus. Joined outside _state_lock so
        # a slow/stuck worker can't stall other requests (status polls, a
        # concurrent start) for the whole timeout.
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
        # Signal the background startup worker to abandon: this is the only way
        # to stop during the pre-subprocess window (Hub download / arm
        # preflight), where there's no process to terminate.
        if _inference_cancel is not None:
            _inference_cancel.set()
        proc = _inference_proc
        ev = _eval_session
        cs = _coach_session
        # Surface the stop as its own phase so a status poll racing the
        # terminate/wait below sees "stopping" rather than a stale "running".
        if _inference_meta:
            _inference_meta["phase"] = PHASE_STOPPING
        if cs is not None:
            # Same reason as eval's `quitting`: the runner is about to exit
            # because we asked it to, and the pump's EOF path must not report
            # that expected exit as a crash.
            cs.quitting = True
        if ev is not None:
            # Suppress crash containment: the runner is about to exit because we
            # asked it to, and an expected exit must not be scored an error.
            ev.quitting = True
            ev.episode_pending = False

        if proc is None:
            # Stop pressed with no live subprocess: either before the first one
            # spawned (during the model download / arm preflight — no policy has
            # driven the robot, and the orphaned startup worker bails at its next
            # cancel check), or, in eval mode, while parked in a reset after the
            # runner crashed. Either way there's nothing to terminate.
            if cs is not None:
                _finalise_coaching_locked(None, cs, aborted=True)
                return {"success": True, "message": "Coaching session stopped"}
            if ev is not None:
                _abort_eval_locked(ev)
                return {"success": True, "message": "Evaluation aborted"}
            _go_idle_locked()
            return {"success": True, "message": "Inference stopped"}

    if cs is not None:
        # QUIT rather than SIGTERM, for the same reason eval does: the runner
        # breaks out of the control loop, FINALIZES THE DATASET (encoding any
        # pending video) and eases the follower home. A signal here would risk
        # losing the corrections the operator just spent the session collecting.
        _quit_runner(proc)
    elif ev is not None:
        # Eval mode: QUIT rather than SIGTERM. The runner breaks out of the
        # running episode, eases the follower home and disconnects properly —
        # a signal would cut that short. The in-flight episode is deliberately
        # left unscored (see _abort_eval_locked), which is why the runner also
        # reports no episode end for it.
        _quit_runner(proc)
    else:
        try:
            _terminate_tree(proc)
        except Exception as exc:
            logger.exception("Stop inference: %s", exc)

    with _state_lock:
        # Re-read: a status poll could have finalised the exit (and, in eval
        # mode, even finished the session) while we were outside the lock.
        ev = _eval_session
        cs = _coach_session
        if cs is not None:
            _finalise_coaching_locked(None, cs, aborted=True)
            return {"success": True, "message": "Coaching session stopped"}
        if ev is not None:
            _abort_eval_locked(ev)
            return {"success": True, "message": "Evaluation aborted"}
        _go_idle_locked()
    return {"success": True, "message": "Inference stopped"}


def handle_stop_episode() -> dict[str, Any]:
    """End the CURRENT eval episode early and score it a SUCCESS.

    This is the "the robot did the task — next" button. It asks the runner to
    end the episode cleanly — no signal, no kill: the runner breaks out of the
    control loop and eases the follower back to its start pose, exactly what
    `--return_to_initial_position` did at each subprocess's teardown — and
    leaves the SESSION standing. The runner then reports the end on stdout, and
    the pump parks the session in the reset phase (or finishes it, if that was
    the last episode).

    Eval-only by design: a single-episode run has no tally to record a success
    into, so it gets a 409 rather than a silent no-op."""
    with _state_lock:
        ev = _eval_session
        if not inference_active or ev is None or not ev.episode_running:
            # No session, not an eval, still starting up, or parked in a reset —
            # in none of those is there a running episode to call a success.
            # `episode_running` (not a live process) is the test: the runner
            # outlives every episode, so it is alive between them too.
            return {
                "success": False,
                "status_code": 409,
                "message": "No evaluation episode is running",
            }
        proc = _inference_proc
        # Set BEFORE the command goes out, so the flag is always visible to the
        # pump thread that finalises the episode. If the episode hits its
        # duration in this exact window, the pump clears `episode_running` under
        # this same lock and the next call 409s — the two orders can't both
        # score the episode.
        ev.stop_requested = True

    if not _send_runner_command(proc, CMD_STOP):
        # The runner was already gone when we asked, so this is not a success
        # the user got to observe — take the flag back, or the crash-containment
        # path would read it as one ("stop_requested wins outright"). Guarded on
        # the episode still being in flight so we can't unset a flag that the
        # pump has meanwhile consumed to score a genuine success.
        with _state_lock:
            if _eval_session is ev and ev.episode_running:
                ev.stop_requested = False
        return {
            "success": False,
            "status_code": 409,
            "message": "The evaluation runner is no longer responding",
        }
    # The verdict is recorded in ONE place — `_finalise_eval_episode_locked`,
    # driven by the runner's episode-end line — so nothing here can double-score.
    return {"success": True, "message": "Episode recorded as a success"}


def handle_next_episode() -> dict[str, Any]:
    """Leave the reset phase and start the next eval episode.

    The reset between episodes is explicitly user-ended (no auto-timer, unlike
    recording's): rearranging a bench scene has no reason to be rushed.

    The normal case costs ONE line of stdin: the runner is still up with the
    policy resident and the bus and cameras open, so continuing is instant —
    that is the whole point of the runner. The exception is a runner that died
    (see `_finalise_runner_exit_locked`), where continuing respawns it and pays
    ONE reload; even then the model isn't re-downloaded and the arm isn't
    re-preflighted, because both results are cached on the session."""
    global _inference_proc, _inference_started_at, _inference_rollout_started_at, _inference_meta
    global _last_log_path

    with _state_lock:
        ev = _eval_session
        if not inference_active or ev is None:
            return {"success": False, "status_code": 409, "message": "No evaluation is active"}
        if _inference_meta.get("phase") != PHASE_RESETTING:
            return {
                "success": False,
                "status_code": 409,
                "message": "The evaluation is not waiting for a reset",
            }
        if ev.policy_path is None:
            # Only reachable if the startup worker never got far enough to cache
            # the resolved path, which also means no episode ever ran.
            return {
                "success": False,
                "status_code": 409,
                "message": "The evaluation session has no prepared policy to run",
            }
        request, policy_path, robot_args = ev.request, ev.policy_path, list(ev.robot_args)
        carried_ref = _inference_meta.get("policy_ref")
        carried_warning = _inference_meta.get("warning")
        respawn = _inference_proc is None
        proc = _inference_proc
        ev.episode_pending = True
        # Both timers restart per episode: `elapsed_s` is this episode's setup
        # time and `rollout_elapsed_s` its rollout time, so the dialog's clock
        # and the frontend's past-duration safety net both measure the EPISODE,
        # not the (much longer) session. On the live-runner path the setup time
        # is now ~0, which is the win.
        _inference_started_at = time.time()
        _inference_rollout_started_at = None
        # Clear the previous episode's crash banner — the user chose to continue.
        ev.error = None
        ev.hint = None
        ev.runner_error = None
        _inference_meta = {
            "policy_ref": carried_ref or request.policy_ref,
            "policy_path": policy_path,
            "duration_s": request.duration_s,
            # A respawn opens a fresh log; a live runner keeps writing the one
            # the session started with.
            "log_path": _inference_meta.get("log_path"),
            "phase": PHASE_STARTING,
            **({"warning": carried_warning} if carried_warning else {}),
        }
        episode_index = ev.episode_index

    if not respawn:
        if not _send_runner_command(proc, CMD_EPISODE):
            return {
                "success": False,
                "status_code": 409,
                "message": "The evaluation runner is no longer responding",
            }
        logger.info("Eval episode %s requested from the live runner", episode_index)
        return {"success": True, "message": f"Episode {episode_index} starting"}

    # The runner died during the previous episode (or during the reset). Respawn
    # it — one policy load — and let the READY handler issue the episode, the
    # same way the session's first episode is issued.
    try:
        proc, log_handle, log_path = _launch_eval_runner(request, policy_path, robot_args)
    except Exception as exc:
        logger.exception("Failed to respawn the eval runner")
        # Spawning is the cheap part; a failure here is a session-level problem
        # (a broken interpreter/env), not something the next reset can fix.
        _fail_startup(f"Failed to start the next episode: {exc}")
        return {"success": False, "status_code": 500, "message": f"Failed to start the next episode: {exc}"}

    with _state_lock:
        if not inference_active or _eval_session is None:
            # Aborted while we were spawning — kill what we just started rather
            # than leave a policy driving the arm for a dead session.
            logger.info("Eval runner respawn abandoned right after spawn; killing subprocess")
            _terminate_tree(proc)
            with contextlib.suppress(Exception):
                log_handle.close()
            return {"success": False, "status_code": 409, "message": "No evaluation is active"}
        _inference_proc = proc
        _inference_meta["log_path"] = str(log_path)
        # A respawn opens a FRESH log (see _open_log_and_spawn), so this is not a
        # set-once-per-session value: keep the two in step here as well as at the
        # session's first launch.
        _last_log_path = str(log_path)

    threading.Thread(
        target=_pump_runner_stdout,
        args=(proc, log_handle),
        name="inference-stdout-pump",
        daemon=True,
    ).start()
    logger.info("Eval runner respawned for episode %s: pid=%s", episode_index, proc.pid)
    return {"success": True, "message": f"Episode {episode_index} starting"}


# Tail cap for the inference-log endpoint: last N lines, bounded so a very long
# run's log can never be shipped to the browser in full.
_INFERENCE_LOG_MAX_LINES = 500


def _resolve_inference_log_path() -> tuple[Path | None, str | None]:
    """The log to show and WHOSE it is: (path, "active" | "last_run" | None).

    Two sources, in order, and nothing else:
      * the ACTIVE session's `_inference_meta["log_path"]` — this run's own log;
      * `_last_log_path` — the most recent run of THIS server process to have
        spawned a subprocess, so a finished run's log stays readable while the
        dialog shows its terminal state.

    There is deliberately NO directory fallback. Globbing the newest `*.log` out
    of inference_logs looks like a harmless convenience, but a log file on disk
    carries no evidence of which run produced it: during a new session's pre-spawn
    phases, and after a run that failed before spawning, the newest file belongs
    to some EARLIER run and was served as if it were the current one (see
    `_last_log_path`). Returning None is the honest answer.

    The cost is that after a server restart the endpoint reports no log even
    though files exist on disk. That is accepted: those files belong to runs this
    process never saw, and mislabelling them is worse than not showing them. The
    path is still printed in the status payload for anyone who wants to open it.
    """
    with _state_lock:
        meta_path = _inference_meta.get("log_path")
        active = inference_active
        last_path = _last_log_path
    if meta_path and active:
        p = Path(meta_path)
        if p.is_file():
            return p, "active"
    if last_path:
        p = Path(last_path)
        if p.is_file():
            return p, "last_run"
    return None, None


def handle_inference_log(max_lines: int = _INFERENCE_LOG_MAX_LINES) -> dict[str, Any]:
    """Return the tail of this session's (or the last run's) inference log.

    Read-only and bounded: at most `max_lines` trailing lines. Never raises —
    a missing/unreadable log yields empty text, so the route stays 200 even
    before a run has produced any output.

    `belongs_to` tells the caller what it is looking at, so the UI never has to
    guess: "active" is the running session's own log, "last_run" the most recent
    finished run of this process, and None means there is no log to show. A live
    session that reports anything other than "active" has not produced output
    yet — rendering `logs` in that case is how a stale run's output ends up
    labelled as the current one."""
    path, belongs_to = _resolve_inference_log_path()
    if path is None:
        return {"logs": "", "log_path": None, "belongs_to": None}
    # Bounded read: only the last ~64 KB is decoded (shared with the error-mining
    # path), which holds every line a rollout log this size produces. A
    # missing/unreadable file yields None -> empty text, keeping the route 200.
    lines = _read_log_tail_lines(str(path))
    if lines is None:
        return {"logs": "", "log_path": str(path), "belongs_to": belongs_to}
    tail = lines[-max_lines:] if max_lines > 0 else lines
    return {"logs": "\n".join(tail), "log_path": str(path), "belongs_to": belongs_to}


def _finalise_eval_episode_locked(
    rc: int | None,
    ev: _EvalSession,
    *,
    keep_runner: bool = False,
    runner_error: str | None = None,
) -> dict[str, Any]:
    """Score one finished eval episode and either park or finish the session.

    Caller must hold `_state_lock`. This is the SINGLE place an episode verdict
    is recorded, reached from both endings the eval runner can produce: a clean
    `EPISODE_ENDED` (rc=0, `keep_runner=True` — the runner is alive and will
    serve the next episode) and an unexpected runner death (its exit code,
    `keep_runner` left False so the next continue respawns). `handle_stop_episode`
    only sets a flag and asks the runner to end, so nothing can double-score.

    Not finishing the session means keeping `inference_active` True through the
    reset: the session still owns the inference slot (recording/teleop stay
    blocked) and still owns the cameras (previews stay 409'd), which is exactly
    what lets the next episode start straight into a ready rig."""
    global _inference_proc, _inference_rollout_started_at, _inference_meta, _last_result

    finished_meta = _inference_meta
    finished_started = _inference_started_at
    rollout_started = _inference_rollout_started_at is not None
    # Mine the error only when the ending was non-clean. `runner_error` is the
    # runner's own ERROR line when it managed to send one — the exception text
    # itself, so it beats the log-tail heuristic.
    error = (runner_error or _extract_error_from_log(finished_meta.get("log_path"))) if rc else None
    verdict = classify_episode(rc, ev.stop_requested, rollout_started, error)
    # Read the index BEFORE appending — the property is derived from the result
    # count, so afterwards it names the NEXT episode.
    logger.info(
        "Eval episode %s/%s exited rc=%s -> %s",
        ev.episode_index,
        ev.episodes_total,
        rc,
        verdict,
    )
    ev.results.append(verdict)
    ev.stop_requested = False
    ev.runner_error = None
    if verdict == EPISODE_ERROR:
        ev.error = error
        ev.hint = friendly_hint(error)
    else:
        ev.error = None
        ev.hint = None

    if not keep_runner:
        _inference_proc = None
    _inference_rollout_started_at = None

    if len(ev.results) < ev.episodes_total:
        # More to go: park in the reset phase and wait for the user to rearrange
        # the scene and POST /inference-next-episode. No auto-timer.
        _inference_meta = {**finished_meta, "phase": PHASE_RESETTING}
        return {
            **_eval_fields(ev),
            "inference_active": True,
            "started_at": finished_started,
            "rollout_started_at": None,
            "elapsed_s": 0,
            "rollout_elapsed_s": 0,
            "duration_s": finished_meta.get("duration_s"),
            "policy_ref": finished_meta.get("policy_ref"),
            "log_path": finished_meta.get("log_path"),
            "phase": PHASE_RESETTING,
            "download_bytes_done": None,
            "download_bytes_total": None,
            "download_percent": None,
            "warning": finished_meta.get("warning"),
            # Populated only for a CRASHED episode — the reset screen doubles as
            # the "this one broke, continue or abort?" screen.
            "error": ev.error,
            "hint": ev.hint,
        }

    # Last episode: the session is done. Claim the accuracy and release the slot.
    _go_idle_locked()
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": rc,
        "outcome": "ok",
        "error": ev.error,
        "hint": ev.hint,
        "phase": PHASE_FINISHED,
        "policy_ref": finished_meta.get("policy_ref"),
        "duration_s": finished_meta.get("duration_s"),
        "log_path": finished_meta.get("log_path"),
        "started_at": finished_started,
        "rollout_started_at": None,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(ev, accuracy=eval_accuracy(ev.results)),
    }
    logger.info("Evaluation finished: %s accuracy=%s", ev.results, _last_result["accuracy"])
    return dict(_last_result)


def _finalise_runner_exit_locked(rc: int | None, ev: _EvalSession) -> None:
    """Absorb an unexpected eval-runner death without losing the session.

    Caller must hold `_state_lock`. The runner is the session's ONE process, so
    losing it costs a policy reload — but not the run: the tally, the resolved
    policy path and the already-preflighted `--robot.*` args all live on the
    `_EvalSession`, so the user's next continue respawns and carries on from
    where the tally stood (`handle_next_episode`).

    An episode that was in flight is scored `error`, which is excluded from the
    accuracy denominator — a serial glitch on episode 7 must not be readable as
    the policy failing. The session then parks in `resetting` with the error on
    show, which is also the screen the abort button lives on."""
    global _inference_proc, _inference_meta

    if ev.quitting:
        # We asked for this exit (an abort is mid-flight). The in-flight episode
        # of an aborted session is deliberately left unscored, so returning here
        # is what keeps the abort's partial tally honest.
        logger.info("Eval runner exited after QUIT rc=%s", rc)
        _inference_proc = None
        return
    logger.warning("Eval runner exited unexpectedly rc=%s", rc)
    _inference_proc = None
    if not ev.results and not ev.episode_running:
        # The runner died before its very FIRST episode ever started — a bad
        # policy path, a camera that isn't there, a bus another process holds.
        # That is a startup failure, not an evaluation with a bad episode in it:
        # parking in a reset would offer a "continue" that can only fail the same
        # way. Report it exactly as a single run's startup failure is reported.
        error = ev.runner_error or _extract_error_from_log(_inference_meta.get("log_path"))
        _fail_startup_locked(error or f"The evaluation runner exited unexpectedly (code {rc}).")
        return
    ev.episode_pending = False
    if ev.episode_running:
        ev.episode_running = False
        # Any exit that interrupts an episode is a crash, whatever the code
        # says: a runner that finished an episode cleanly reports EPISODE_ENDED
        # and stays alive, so reaching here at all means it didn't.
        _finalise_eval_episode_locked(rc or 1, ev, runner_error=ev.runner_error)
        return

    # Died while parked between episodes: there is no episode to score, but the
    # user still needs to know why continuing will now cost a reload.
    error = ev.runner_error or _extract_error_from_log(_inference_meta.get("log_path"))
    ev.error = error
    ev.hint = friendly_hint(error)
    ev.runner_error = None
    if _inference_meta:
        _inference_meta["phase"] = PHASE_RESETTING


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _last_result

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        # True only while idle: a previous session's startup worker (see
        # _inference_startup_thread) is still alive after its stop already
        # fired, so a poller isn't looking at a status indistinguishable from
        # true idle while the worker still holds the serial bus.
        shutting_down = (
            not inference_active
            and _inference_startup_thread is not None
            and _inference_startup_thread.is_alive()
        )
        # Idle with a recorded terminal result (a subprocess exit finalised
        # below, or a download/preflight failure from _fail_startup): keep
        # returning that payload verbatim until the next start clears it.
        # Idempotence matters — several surfaces poll this endpoint
        # concurrently, and a consume-once payload lets one poller swallow the
        # error the user needed to see (see _last_result's declaration).
        if proc is None and not inference_active and _last_result is not None:
            return {**_last_result, "shutting_down": shutting_down}
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            if _coach_session is not None:
                # Coaching mode: phase changes arrive on the RUNNER's stdout, so
                # an exit observed here is the session ending — cleanly at its
                # target, or badly. Backstop for the pump (which normally
                # notices EOF first); `_finalise_coaching_locked` clears
                # `_inference_proc`, so whichever path arrives second finds
                # nothing left to finalise.
                _finalise_coaching_locked(rc, _coach_session)
                return {**_last_result, "shutting_down": shutting_down}
            if _eval_session is not None:
                # Eval mode: episode boundaries arrive on the RUNNER's stdout,
                # so an exit here is the runner dying — a crash, not an episode
                # end. Backstop for the pump (which normally notices EOF first);
                # `_finalise_runner_exit_locked` clears `_inference_proc`, so
                # whichever path gets here second finds nothing to finalise.
                _finalise_runner_exit_locked(rc, _eval_session)
                if not inference_active and _last_result is not None:
                    # The crash landed on the final episode: the session is over.
                    return {**_last_result, "shutting_down": shutting_down}
                # Otherwise fall through and report the parked (resetting) state.
            else:
                logger.info("Inference subprocess exited rc=%s", rc)
                finished_meta = _inference_meta
                finished_started = _inference_started_at
                finished_rollout_started = _inference_rollout_started_at
                # Terminal phase: a clean exit (rc 0, including a stop we asked
                # for) is `stopped`; any non-zero code is `error`. The prior
                # phase in `finished_meta` (e.g. "stopping" from a stop request)
                # is superseded — the subprocess has actually gone now.
                terminal_phase = PHASE_STOPPED if rc == 0 else PHASE_ERROR
                inference_active = False
                _inference_proc = None
                _inference_started_at = None
                _inference_rollout_started_at = None
                _inference_meta = {}
                # On a non-zero exit, mine the real error out of the log so the
                # UI can show it directly (hint + snippet) instead of sending the
                # user digging through the cache. `outcome` further distinguishes
                # a true failure from a run that worked but tripped a noisy
                # shutdown/cleanup warning (see _classify_outcome) so the
                # false-failure isn't reported as a hard error.
                error = _extract_error_from_log(finished_meta.get("log_path")) if rc else None
                outcome = _classify_outcome(rc, finished_rollout_started is not None, error)
                _last_result = {
                    "inference_active": False,
                    "exited": True,
                    "exit_code": rc,
                    "outcome": outcome,
                    "error": error,
                    "hint": friendly_hint(error),
                    "phase": terminal_phase,
                    "policy_ref": finished_meta.get("policy_ref"),
                    "duration_s": finished_meta.get("duration_s"),
                    "log_path": finished_meta.get("log_path"),
                    "started_at": finished_started,
                    "rollout_started_at": finished_rollout_started,
                    "rollout_elapsed_s": 0,
                    "elapsed_s": 0,
                    **_eval_fields(None),
                    **_coach_fields(None),
                }
                # Final release (lazy finalisation of a subprocess that died
                # on its own). Under _state_lock; the notify is a lock-free
                # droppable queue put, so this cannot deadlock.
                notify_session_changed("inference", False, phase=terminal_phase)
                return {**_last_result, "shutting_down": shutting_down}
        elapsed = (time.time() - _inference_started_at) if _inference_started_at else 0
        rollout_elapsed = time.time() - _inference_rollout_started_at if _inference_rollout_started_at else 0
        return {
            **_eval_fields(_eval_session),
            **_coach_fields(_coach_session),
            # A crashed episode parks the eval session in the reset phase with
            # its mined error still on show, so the user can decide to continue
            # or abort. Null on every other live payload.
            "error": _eval_session.error if _eval_session else None,
            "hint": _eval_session.hint if _eval_session else None,
            "inference_active": inference_active,
            "started_at": _inference_started_at,
            "rollout_started_at": _inference_rollout_started_at,
            "elapsed_s": elapsed,
            "rollout_elapsed_s": rollout_elapsed,
            "duration_s": _inference_meta.get("duration_s"),
            "policy_ref": _inference_meta.get("policy_ref"),
            "log_path": _inference_meta.get("log_path"),
            # None when idle (no session has seeded a meta yet); the frontend
            # treats an absent phase as "no active startup to narrate".
            "phase": _inference_meta.get("phase"),
            # Byte progress of the Hub model download, populated only during the
            # downloading_model phase (all None outside it / for a local
            # checkpoint). download_percent is None while the total is still
            # unknown → the UI shows an indeterminate bar.
            "download_bytes_done": _inference_meta.get("download_bytes_done"),
            "download_bytes_total": _inference_meta.get("download_bytes_total"),
            "download_percent": _inference_meta.get("download_percent"),
            # Warn-but-allow arm-identity finding, surfaced once the run is up
            # (the preflight now runs in the background, after the POST returned).
            "warning": _inference_meta.get("warning"),
            "shutting_down": shutting_down,
        }
