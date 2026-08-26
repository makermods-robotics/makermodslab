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

"""The /api/v1/sessions surface: session identity + server-side robot resolution.

Two things live here, deliberately together:

- :class:`SessionTracker` — gives every robot-driving session an identity
  (id / started_at / revision / phase) by OBSERVING the session_events seam.
  It does NOT own the mutex: the feature modules' active flags stay the single
  source of truth, and the tracker never initiates or blocks anything. Because
  the seam fires at the real transitions of every feature, identity attaches
  to sessions started through the LEGACY endpoints too (the un-migrated UI) —
  those just carry ``robot``/``owner`` of ``None``, since only the start
  wrapper below knows them.

- the ``handle_*`` functions the router calls — the resolution wrappers. A
  client names a saved robot and the kind-specific options; everything
  hardware-shaped (ports, configs, mode, right-arm fields, cameras) is
  resolved server-side from the robot record into the feature's existing
  request model, and the feature's existing ``handle_start_*`` does the actual
  work (and keeps enforcing the mutex).

No lease/heartbeat/timeout yet — ``owner`` is recorded but grants nothing;
``session.not_owner`` / ``session.lease_expired`` stay reserved for the lease
commit.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from pydantic import ValidationError

from . import session_events
from .api_errors import ApiError, ErrorCode
from .schemas.sessions import (
    InferenceOptions,
    RecordingOptions,
    ReplayOptions,
    SessionStartBody,
    TeleoperationOptions,
)
from .utils.config import get_robot_record, is_robot_record_clean, is_valid_robot_name

logger = logging.getLogger(__name__)

# Kinds a client can START through POST /api/v1/sessions. Calibration,
# auto-calibration and wiggle sessions are still started through their legacy
# wizard/flow endpoints this phase — the tracker observes them all the same.
STARTABLE_KINDS = ("teleoperation", "recording", "inference", "replay")

# Kinds that never open the leader bus, mirroring the frontend's robotSetupGap
# distinction: an unassigned leader port / missing leader calibration must not
# block them (bimanual = both followers, still no leaders).
_FOLLOWER_ONLY_KINDS = frozenset({"inference", "replay"})

_OPTIONS_MODELS = {
    "teleoperation": TeleoperationOptions,
    "recording": RecordingOptions,
    "inference": InferenceOptions,
    "replay": ReplayOptions,
}


class SessionTracker:
    """Identity for the one robot-driving session, maintained by observation.

    Fed every ``session_changed`` event through the seam's subscriber list:
    a claim (active=True with no current session of that kind) mints the
    identity, phase events bump ``revision``, the release clears it and keeps
    a small ``last_ended`` summary. All state lives behind one lock; readers
    get copies.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: dict[str, Any] | None = None
        self._last_ended: dict[str, Any] | None = None

    def observe(self, event: dict) -> None:
        """Consume one seam event. Runs inside notify_session_changed's
        per-subscriber try/except — must stay cheap and never block."""
        session = event.get("session") or {}
        kind = session.get("kind")
        phase = session.get("phase")
        with self._lock:
            if session.get("active"):
                if self._current is not None and self._current["kind"] == kind:
                    # A phase transition of the live session.
                    self._current["revision"] += 1
                    self._current["phase"] = phase
                    return
                if self._current is not None:
                    # The mutex makes this unreachable; if it ever fires, the
                    # new claim wins — flags are the truth, identity follows.
                    logger.warning(
                        f"session claim for {kind!r} while a {self._current['kind']!r} "
                        f"session was still tracked; replacing it"
                    )
                self._current = {
                    "id": uuid.uuid4().hex,
                    "kind": kind,
                    "robot": None,
                    "owner": None,
                    "started_at": time.time(),
                    "revision": 1,
                    "phase": phase,
                }
            elif self._current is not None and self._current["kind"] == kind:
                self._last_ended = {
                    "id": self._current["id"],
                    "kind": kind,
                    "ended_at": time.time(),
                    "phase": phase,
                }
                self._current = None
            # A release with no matching session (idle double-stop, or an
            # event for a kind we never saw claim) is ignored.

    def attribute(self, kind: str, robot: str | None = None, owner: str | None = None) -> dict | None:
        """Attach what only the start wrapper knows (robot, owner) to the
        current session, if it is of `kind`. Enrichment, not a transition —
        the revision does not bump. Returns a snapshot, or None when no
        session of that kind is current."""
        with self._lock:
            if self._current is None or self._current["kind"] != kind:
                return None
            if robot is not None:
                self._current["robot"] = robot
            if owner is not None:
                self._current["owner"] = owner
            return dict(self._current)

    def current(self) -> dict | None:
        with self._lock:
            return dict(self._current) if self._current is not None else None

    def last_ended(self) -> dict | None:
        with self._lock:
            return dict(self._last_ended) if self._last_ended is not None else None

    def reset(self) -> None:
        """Drop all tracked state (tests only — production identity only ever
        moves by observation)."""
        with self._lock:
            self._current = None
            self._last_ended = None


tracker = SessionTracker()
session_events.subscribe(tracker.observe)


# --- start: server-side robot resolution ------------------------------------


def _held_by() -> str | None:
    """Which feature's active flag currently holds the hardware, if any.

    The same reciprocal flags every ``handle_start_*`` checks — read here
    BEFORE robot resolution because exclusivity is a property of the node's
    one set of hardware, not of the robot named in the request (pinned by
    tests/test_api_errors.py::test_sessions_surface_uses_reserved_codes)."""
    from . import auto_calibrate, calibrate, record, replay, rollout, teleoperate, wiggle

    if teleoperate.teleoperation_active:
        return "teleoperation"
    if record.recording_active:
        return "recording"
    if rollout.inference_active:
        return "inference"
    if replay.replay_active:
        return "replay"
    if calibrate.calibration_is_active():
        return "calibration"
    if auto_calibrate.auto_calibration_is_active():
        return "auto_calibration"
    if wiggle.wiggle_active:
        return "wiggle"
    return None


def _raise_held(holder_kind: str | None, message: str) -> None:
    """409 session.held, with details naming the holder as precisely as the
    tracker can: its session id when the tracker saw the claim, else null
    (e.g. the flag was raised before this process's seam existed — or by a
    test)."""
    snapshot = tracker.current()
    holder_id = snapshot["id"] if snapshot and snapshot["kind"] == holder_kind else None
    raise ApiError(
        status_code=409,
        detail=message,
        code=ErrorCode.SESSION_HELD,
        details={"holder": {"kind": holder_kind, "session_id": holder_id}},
    )


def _build_teleoperation_request(record: dict, opts: TeleoperationOptions):
    from .teleoperate import TeleoperateRequest

    return TeleoperateRequest(
        leader_port=record["leader_port"],
        follower_port=record["follower_port"],
        leader_config=record["leader_config"],
        follower_config=record["follower_config"],
        mode=record["mode"],
        right_leader_port=record["right_leader_port"],
        right_follower_port=record["right_follower_port"],
        right_leader_config=record["right_leader_config"],
        right_follower_config=record["right_follower_config"],
        robot_name=record["name"],
        skip_identity_check=opts.skip_identity_check,
    )


def _build_recording_request(record: dict, opts: RecordingOptions):
    from .record import RecordingRequest

    # robot_name makes record.py resolve the session's cameras from this same
    # record server-side — the sessions surface adds no camera plumbing.
    return RecordingRequest(
        leader_port=record["leader_port"],
        follower_port=record["follower_port"],
        leader_config=record["leader_config"],
        follower_config=record["follower_config"],
        mode=record["mode"],
        right_leader_port=record["right_leader_port"],
        right_follower_port=record["right_follower_port"],
        right_leader_config=record["right_leader_config"],
        right_follower_config=record["right_follower_config"],
        robot_name=record["name"],
        dataset_repo_id=opts.dataset_repo_id,
        single_task=opts.single_task,
        num_episodes=opts.num_episodes,
        episode_time_s=opts.episode_time_s,
        reset_time_s=opts.reset_time_s,
        fps=opts.fps,
        video=opts.video,
        push_to_hub=opts.push_to_hub,
        tags=opts.tags,
        private=opts.private,
        resume=opts.resume,
        streaming_encoding=opts.streaming_encoding,
        skip_identity_check=opts.skip_identity_check,
    )


def _build_inference_request(record: dict, opts: InferenceOptions):
    from .rollout import InferenceRequest

    # Follower-only: inference never opens the leader bus, so only the
    # follower half of the record travels (right follower iff bimanual).
    return InferenceRequest(
        follower_port=record["follower_port"],
        follower_config=record["follower_config"],
        mode=record["mode"],
        right_follower_port=record["right_follower_port"],
        right_follower_config=record["right_follower_config"],
        robot_name=record["name"],
        policy_ref=opts.policy_ref,
        task=opts.task,
        camera_bindings=opts.camera_bindings,
        camera_dims=opts.camera_dims,
        duration_s=opts.duration_s,
        checkpoint_state_dim=opts.checkpoint_state_dim,
        eval_episodes=opts.eval_episodes,
        skip_identity_check=opts.skip_identity_check,
    )


def _build_replay_request(record: dict, opts: ReplayOptions):
    from .replay import ReplayRequest

    return ReplayRequest(
        repo_id=opts.repo_id,
        episode_index=opts.episode_index,
        follower_port=record["follower_port"],
        follower_config=record["follower_config"],
        robot_name=record["name"],
        skip_identity_check=opts.skip_identity_check,
    )


_REQUEST_BUILDERS = {
    "teleoperation": _build_teleoperation_request,
    "recording": _build_recording_request,
    "inference": _build_inference_request,
    "replay": _build_replay_request,
}


def _dispatch_start(kind: str, request, websocket_manager) -> dict[str, Any]:
    from . import record, replay, rollout, teleoperate

    if kind == "teleoperation":
        return teleoperate.handle_start_teleoperation(request, websocket_manager)
    if kind == "recording":
        return record.handle_start_recording(request)
    if kind == "inference":
        return rollout.handle_start_inference(request)
    return replay.handle_start_replay(request, websocket_manager)


def handle_start_session(body: SessionStartBody, websocket_manager=None) -> dict[str, Any]:
    """Start a session by robot name; the router returns the dict as a 201.

    Flow: hardware-hold gate (409 session.held) → robot record resolution
    (404 robot.not_found) → readiness with the arms the kind actually drives
    (400 robot.not_ready) → per-kind options validation (422
    request.validation) → build the feature's request model from the record →
    the feature's own ``handle_start_*``. A busy-coded refusal from the
    feature (the gate raced another start) maps to 409 session.held as well;
    any other refusal passes through with its own status/code.
    """
    kind = body.kind

    held = _held_by()
    if held is not None:
        _raise_held(
            held,
            f"The robot hardware is held by an active {held} session. Stop it first.",
        )

    record = get_robot_record(body.robot) if is_valid_robot_name(body.robot) else None
    if record is None:
        raise ApiError(
            status_code=404,
            detail=f"No robot named {body.robot!r}.",
            code=ErrorCode.ROBOT_NOT_FOUND,
        )

    arms = "follower" if kind in _FOLLOWER_ONLY_KINDS else "all"
    if not is_robot_record_clean(record, arms=arms):
        needs = "follower arm" if arms == "follower" else "arms"
        raise ApiError(
            status_code=400,
            detail=f"Robot {body.robot!r} is not fully set up for {kind}: "
            f"its {needs} need ports and existing calibrations.",
            code=ErrorCode.ROBOT_NOT_READY,
        )

    try:
        opts = _OPTIONS_MODELS[kind].model_validate(body.options)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'options'}: {err['msg']}" for err in exc.errors()
        )
        raise ApiError(
            status_code=422,
            detail=f"Invalid {kind} options: {problems}",
            code=ErrorCode.REQUEST_VALIDATION,
        ) from exc

    request = _REQUEST_BUILDERS[kind](record, opts)
    result = _dispatch_start(kind, request, websocket_manager)
    if not result.get("success", False):
        message = result.get("message", f"Failed to start {kind}")
        code = str(result.get("code") or "")
        if code.startswith("robot.busy."):
            # Raced another start past the gate above. The discriminant names
            # the holder except for `releasing`, where the tracker may still
            # know which session is winding down.
            discriminant = code.rsplit(".", 1)[-1]
            if discriminant in session_events.SESSION_KINDS:
                holder = discriminant
            else:
                snapshot = tracker.current()
                holder = snapshot["kind"] if snapshot else None
            _raise_held(holder, message)
        raise ApiError(
            status_code=result.get("status_code", 500),
            detail=message,
            code=result.get("code"),
        )

    # The claim event fired synchronously inside handle_start_* — the tracker
    # already minted the identity; attach what only this wrapper knows.
    session = tracker.attribute(kind, robot=body.robot, owner=body.owner)
    if session is None:
        logger.error(f"{kind} start reported success but the tracker saw no claim event")
        raise ApiError(
            status_code=500,
            detail=f"The {kind} session started but its identity could not be established.",
            code=ErrorCode.INTERNAL_UNEXPECTED,
        )
    return {"session": session}


# --- current / stop ----------------------------------------------------------


def handle_current_session() -> dict[str, Any]:
    return {"session": tracker.current(), "last_ended": tracker.last_ended()}


def _dispatch_stop(kind: str) -> dict[str, Any]:
    from . import auto_calibrate, calibrate, record, replay, rollout, teleoperate

    if kind == "teleoperation":
        return teleoperate.handle_stop_teleoperation()
    if kind == "recording":
        return record.handle_stop_recording()
    if kind == "inference":
        return rollout.handle_stop_inference()
    if kind == "replay":
        return replay.handle_stop_replay()
    if kind == "calibration":
        return calibrate.calibration_manager.stop_calibration_process()
    if kind == "auto_calibration":
        # The aggregate spans the single-arm manager and the batch manager —
        # stop whichever is live (a stop of the idle one reports failure).
        result = auto_calibrate.auto_calibration_manager.stop()
        if not result.get("success"):
            batch = auto_calibrate.auto_calibration_batch_manager.stop()
            if batch.get("success"):
                return batch
        return result
    # wiggle: a few seconds of open-loop gripper motion with no stop handler.
    raise ApiError(
        status_code=409,
        detail="A gripper wiggle finishes on its own within seconds and cannot be stopped.",
        code=ErrorCode.ROBOT_BUSY_WIGGLE,
    )


def handle_stop_session(session_id: str) -> dict[str, Any]:
    """Stop the current session, but only under its own id.

    The id-match is the operation-identity guarantee: a stop aimed at a
    session that has already ended (and possibly been replaced) is a 404
    session.not_found, never a stop of whatever runs now. Works for
    legacy-started sessions too — the observer gave them ids.

    The kind's stop result passes through verbatim beside the final identity;
    a kind whose stop is not immediate (teleoperation's release grace) may
    still show as current, in its releasing phase.
    """
    before = tracker.current()
    if before is None or before["id"] != session_id:
        raise ApiError(
            status_code=404,
            detail=f"No active session with id {session_id!r}.",
            code=ErrorCode.SESSION_NOT_FOUND,
        )
    result = _dispatch_stop(before["kind"])

    after = tracker.current()
    if after is not None and after["id"] == before["id"]:
        session = after
    else:
        # Released synchronously during the stop call — report the identity it
        # ended with (the release event's phase, when the tracker kept it).
        session = before
        ended = tracker.last_ended()
        if ended is not None and ended["id"] == before["id"]:
            session = dict(before, phase=ended["phase"])
    return {"session": session, "result": result}
