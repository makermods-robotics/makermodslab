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

The lease — server-authoritative ownership with a timeout fail-safe. This is
a HARDWARE-SAFETY mechanism: an abandoned session (client crashed, wifi died,
tab gone) must not leave an arm energized forever, so a leased session that
stops heartbeating is safety-stopped by the expiry watchdog. The attachment
rule is the compatibility linchpin: a session gets a lease ONLY when created
via POST /api/v1/sessions with an ``owner``. Owner-less POSTs and
legacy-endpoint starts get NO lease and are NEVER timeout-stopped — the
un-migrated UI polls the legacy status endpoints, heartbeats nothing, and
must not be killed under it. Enforcement becomes universal only once the UI
migrates (the next commit). Stopping is deliberately NEVER owner-gated (see
``handle_stop_session``): a physical arm must always be stoppable by whoever
can reach the API — ``session.not_owner`` guards heartbeat (and future
owner-gated mutations), never stop.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from . import session_events
from .api_errors import ApiError, ErrorCode
from .schemas.sessions import (
    LEASE_TIMEOUT_AUTO_CALIBRATION_S,
    LEASE_TIMEOUT_DEFAULT_S,
    LEASE_TIMEOUT_MAX_S,
    LEASE_TIMEOUT_MIN_S,
    OWNER_MAX_LENGTH,
    AutoCalibrationOptions,
    CalibrationOptions,
    InferenceOptions,
    RecordingOptions,
    ReplayOptions,
    SessionStartBody,
    TeleoperationOptions,
)
from .utils.config import get_robot_record, is_robot_record_clean, is_valid_robot_name

logger = logging.getLogger(__name__)

# Kinds a client can START through POST /api/v1/sessions. Only wiggle is left
# on its legacy flow endpoint (a few seconds of open-loop gripper motion — no
# stop handler, nothing to lease) — the tracker observes it all the same.
STARTABLE_KINDS = (
    "teleoperation",
    "recording",
    "inference",
    "replay",
    "calibration",
    "auto_calibration",
)

# Kinds that never open the leader bus, mirroring the frontend's robotSetupGap
# distinction: an unassigned leader port / missing leader calibration must not
# block them (bimanual = both followers, still no leaders).
_FOLLOWER_ONLY_KINDS = frozenset({"inference", "replay"})

# Setup kinds: calibration CREATES the record's calibrations (and writes the
# port back on success), so the record-clean readiness gate the driving kinds
# use would refuse exactly the robots these flows exist to fix. Their builders
# check the one thing they do need — a port for each targeted slot.
_SETUP_KINDS = frozenset({"calibration", "auto_calibration"})

_OPTIONS_MODELS = {
    "teleoperation": TeleoperationOptions,
    "recording": RecordingOptions,
    "inference": InferenceOptions,
    "replay": ReplayOptions,
    "calibration": CalibrationOptions,
    "auto_calibration": AutoCalibrationOptions,
}


class SessionTracker:
    """Identity for the one robot-driving session, maintained by observation.

    Fed every ``session_changed`` event through the seam's subscriber list:
    a claim (active=True with no current session of that kind) mints the
    identity, phase events bump ``revision``, the release clears it and keeps
    a small ``last_ended`` summary. All state lives behind one lock; readers
    get copies.

    The lease rides on the current session as an internal dict
    ``{owner, timeout_s, deadline, expired}`` — ``deadline`` is on the
    injected monotonic ``clock`` and never leaves the process (clients see a
    computed ``expires_in_s`` via :func:`_public_session`).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._current: dict[str, Any] | None = None
        self._last_ended: dict[str, Any] | None = None

    def _snapshot_locked(self) -> dict[str, Any]:
        """Copy of the current session, with the nested lease dict copied too
        (a shared lease dict would leak later mutations into old snapshots).
        Caller holds the lock."""
        snap = dict(self._current)
        if snap["lease"] is not None:
            snap["lease"] = dict(snap["lease"])
        return snap

    def observe(self, event: dict) -> None:
        """Consume one seam event. Runs inside notify_session_changed's
        per-subscriber try/except — must stay cheap and never block."""
        session = event.get("session") or {}
        kind = session.get("kind")
        phase = session.get("phase")
        released_lease = False
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
                    "lease": None,
                }
            elif self._current is not None and self._current["kind"] == kind:
                lease = self._current["lease"]
                self._last_ended = {
                    "id": self._current["id"],
                    "kind": kind,
                    "ended_at": time.time(),
                    "phase": phase,
                    # The reserved code string, so a client can tell a safety
                    # stop from a normal ending.
                    "reason": str(ErrorCode.SESSION_LEASE_EXPIRED)
                    if lease is not None and lease["expired"]
                    else None,
                }
                self._current = None
                released_lease = lease is not None
            # A release with no matching session (idle double-stop, or an
            # event for a kind we never saw claim) is ignored.
        if released_lease:
            # Outside the lock (never nest tracker → watchdog): the leased
            # session is gone, so the watchdog has nothing left to guard.
            _retire_watchdog()

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
            return self._snapshot_locked()

    def attach_lease(self, kind: str, owner: str, timeout_s: float) -> dict | None:
        """Put a lease on the current session, if it is of `kind`. Only the
        start wrapper calls this (the ONLY way a session gets a lease — see
        the module docstring). Returns a snapshot, or None when no session of
        that kind is current (it ended before the lease could attach)."""
        with self._lock:
            if self._current is None or self._current["kind"] != kind:
                return None
            self._current["lease"] = {
                "owner": owner,
                "timeout_s": timeout_s,
                "deadline": self._clock() + timeout_s,
                "expired": False,
            }
            return self._snapshot_locked()

    def renew_lease(self, session_id: str, owner: str) -> tuple[str, dict | None]:
        """Push the lease deadline out by its timeout — the heartbeat.

        Returns ``(status, snapshot)`` with status one of ``"renewed"``,
        ``"no_lease"`` (current session but nothing to renew — the caller's
        documented no-op), ``"expired"`` (the expiry stop is dispatched but
        the release hasn't landed), ``"not_owner"``, or ``"not_found"``
        (snapshot None). One method so check-and-renew is atomic."""
        with self._lock:
            if self._current is None or self._current["id"] != session_id:
                return "not_found", None
            lease = self._current["lease"]
            if lease is None:
                return "no_lease", self._snapshot_locked()
            if lease["expired"]:
                return "expired", self._snapshot_locked()
            if lease["owner"] != owner:
                return "not_owner", self._snapshot_locked()
            lease["deadline"] = self._clock() + lease["timeout_s"]
            return "renewed", self._snapshot_locked()

    def mark_lease_expired(self, session_id: str) -> dict | None:
        """Atomically flip the lease to expired, exactly once. Returns the
        snapshot iff `session_id` is still current, leased, and not already
        marked — the expiry dispatcher's claim ticket: a None means someone
        else (a natural release, or an earlier check) got there first."""
        with self._lock:
            cur = self._current
            if cur is None or cur["id"] != session_id or cur["lease"] is None or cur["lease"]["expired"]:
                return None
            cur["lease"]["expired"] = True
            return self._snapshot_locked()

    def current(self) -> dict | None:
        with self._lock:
            return self._snapshot_locked() if self._current is not None else None

    def last_ended(self) -> dict | None:
        with self._lock:
            return dict(self._last_ended) if self._last_ended is not None else None

    def reset(self) -> None:
        """Drop all tracked state (tests only — production identity only ever
        moves by observation). Retires the watchdog too, so no test leaves a
        live thread guarding nothing."""
        with self._lock:
            self._current = None
            self._last_ended = None
        _retire_watchdog()


tracker = SessionTracker()
session_events.subscribe(tracker.observe)


# --- the lease's timeout fail-safe: expiry check + watchdog ------------------

# Phases that mean a stop is already winding the session down (teleoperation /
# recording's release grace, replay / recording / inference's stopping) — the
# stop handler in flight already handles the hardware; expiry must not
# dispatch a second stop into it.
_WINDING_DOWN_PHASES = frozenset({"releasing", "stopping"})

_WATCHDOG_TICK_S = 1.0

_watchdog_lock = threading.Lock()
_watchdog_thread: threading.Thread | None = None


def _public_session(snap: dict[str, Any] | None) -> dict[str, Any] | None:
    """Externalize a tracker snapshot: the lease's monotonic ``deadline`` (and
    the ``expired`` marker) are internal — clients get
    ``{owner, timeout_s, expires_in_s}``, expires_in_s computed at read time
    and never negative."""
    if snap is None:
        return None
    out = {k: v for k, v in snap.items() if k != "lease"}
    lease = snap["lease"]
    out["lease"] = (
        {
            "owner": lease["owner"],
            "timeout_s": lease["timeout_s"],
            "expires_in_s": max(0.0, lease["deadline"] - tracker._clock()),
        }
        if lease is not None
        else None
    )
    return out


def check_expiry(now: float | None = None) -> dict[str, Any] | None:
    """Safety-stop the current session iff its lease deadline has passed.

    Pure decision logic over the tracker's state (the watchdog thread just
    calls it on a tick; tests call it directly with a fake ``now``):

    - no current session, no lease, or the deadline not yet reached → None.
      An UNLEASED session is never timeout-stopped, whatever ``now`` is —
      the compatibility linchpin in the module docstring.
    - a session already winding down (release-grace / stopping phases, or the
      expiry stop already dispatched — the lease's ``expired`` marker) → None:
      the stop in flight handles the hardware, never double-dispatch.
    - expired → mark the lease (mark_lease_expired is the atomic claim ticket,
      so a release racing this check makes it a no-op), then dispatch the SAME
      per-kind stop path the stop endpoint uses. The release event that stop
      produces lands in the tracker, which records ``last_ended`` with reason
      ``session.lease_expired``.

    Returns the stopped session's snapshot, or None when nothing was stopped.
    """
    snap = tracker.current()
    if snap is None:
        return None
    lease = snap["lease"]
    if lease is None or lease["expired"]:
        return None
    if (tracker._clock() if now is None else now) < lease["deadline"]:
        return None
    if snap["phase"] in _WINDING_DOWN_PHASES:
        return None
    marked = tracker.mark_lease_expired(snap["id"])
    if marked is None:
        # The session released (or another check claimed it) between the
        # snapshot and the mark — nothing left to stop.
        return None
    logger.error(
        f"SESSION LEASE EXPIRED: no heartbeat from owner {lease['owner']!r} within "
        f"{lease['timeout_s']:.0f}s — SAFETY-STOPPING the {snap['kind']} session {snap['id']} "
        f"to de-energize the arm"
    )
    _dispatch_stop(snap["kind"])
    return marked


def _ensure_watchdog() -> None:
    """Start the expiry watchdog if it isn't running — lazily, only when a
    lease attaches (unleased sessions need no guard, so legacy flows never
    pay for a thread). Mirrors ConnectionManager.start_broadcast_thread."""
    global _watchdog_thread
    with _watchdog_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return
        _watchdog_thread = threading.Thread(target=_watchdog_loop, name="session-lease-watchdog", daemon=True)
        _watchdog_thread.start()
        logger.info("Session-lease watchdog started")


def _retire_watchdog() -> None:
    """Signal the watchdog to exit. Never joins (ConnectionManager's
    stop_broadcast_thread discipline): the daemon notices the cleared slot
    within its tick and exits, and a rapid retire→ensure cycle is safe via
    the thread-identity check in the loop."""
    global _watchdog_thread
    with _watchdog_lock:
        _watchdog_thread = None


def _watchdog_loop() -> None:
    """~1 Hz expiry ticks while a leased session exists.

    Exits when retired/replaced (the identity check) or when it finds no
    leased session left — the latter re-checked under the watchdog lock, so a
    lease attaching concurrently serializes against _ensure_watchdog and gets
    a fresh thread rather than an exiting one. Lock nesting is one-way only
    (watchdog → tracker): the tracker's callers of _retire_watchdog release
    the tracker lock first. The loop must never die on a stop handler's
    exception — a safety net has to outlive the things it catches."""
    global _watchdog_thread
    me = threading.current_thread()
    while True:
        with _watchdog_lock:
            if _watchdog_thread is not me:
                return  # retired or replaced
            snap = tracker.current()
            if snap is None or snap["lease"] is None:
                _watchdog_thread = None
                logger.info("Session-lease watchdog stopped: no leased session remains")
                return
        try:
            check_expiry()
        except Exception:
            logger.exception(
                "Session-lease watchdog: expiry check failed; the watchdog stays up "
                "(the arm may still be energized — check the session manually)"
            )
        time.sleep(_WATCHDOG_TICK_S)


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
    # COACHING is the one exception — the operator takes over THROUGH the
    # leader — so its arms come off the same record, and only then. Sending
    # them unconditionally would hand every plain rollout a leader port it has
    # no business holding.
    leader = (
        {
            "leader_port": record["leader_port"],
            "leader_config": record["leader_config"],
            "right_leader_port": record["right_leader_port"],
            "right_leader_config": record["right_leader_config"],
        }
        if opts.coaching
        else {}
    )
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
        inference_engine=opts.inference_engine,
        temporal_ensemble_coeff=opts.temporal_ensemble_coeff,
        coaching=opts.coaching,
        target_corrections=opts.target_corrections,
        coaching_dataset_name=opts.coaching_dataset_name,
        **leader,
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


# --- the setup kinds: per-slot resolution --------------------------------------


def _slot_fields(device_type: str, arm: str) -> tuple[str, str]:
    """The robot record's (port_field, config_field) for one physical arm slot
    — the same mapping calibrate.py's and auto_calibrate.py's record
    write-backs use ("left" is also the single-arm pair)."""
    is_right = arm == "right"
    if device_type == "teleop":
        return (
            "right_leader_port" if is_right else "leader_port",
            "right_leader_config" if is_right else "leader_config",
        )
    return (
        "right_follower_port" if is_right else "follower_port",
        "right_follower_config" if is_right else "follower_config",
    )


def _resolve_slot(record: dict, device_type: str, arm: str, port: str | None, config_file: str | None):
    """Resolve one calibration target slot into (port, config_file).

    An explicit `port`/`config_file` in the options wins (calibration is the
    setup flow — see CalibrationOptions); otherwise the record's saved slot
    values, with the config falling back to the robot's default name for the
    slot (the UI's own default: "<robot>_<arm>" bimanual, "<robot>" single).
    No port anywhere → 400 robot.not_ready: you can't calibrate an arm whose
    bus we can't open."""
    port_field, config_field = _slot_fields(device_type, arm)
    resolved_port = port or record.get(port_field) or ""
    if not resolved_port:
        side = "leader" if device_type == "teleop" else "follower"
        raise ApiError(
            status_code=400,
            detail=f"Robot {record['name']!r} has no port assigned for its {arm} {side} arm; "
            "assign (or pass) a port before calibrating it.",
            code=ErrorCode.ROBOT_NOT_READY,
        )
    default_name = f"{record['name']}_{arm}" if record.get("mode") == "bimanual" else record["name"]
    resolved_config = config_file or record.get(config_field) or default_name
    return resolved_port, resolved_config


def _build_calibration_request(record: dict, opts: CalibrationOptions):
    from .calibrate import CalibrationRequest

    port, config_file = _resolve_slot(record, opts.device_type, opts.arm, opts.port, opts.config_file)
    return CalibrationRequest(
        device_type=opts.device_type,
        port=port,
        config_file=config_file,
        robot_name=record["name"],
        overwrite=opts.overwrite,
        arm=opts.arm,
    )


def _build_auto_calibration_request(record: dict, opts: AutoCalibrationOptions):
    """Always the BATCH request, even for one arm — the batch of one is
    exactly how the UI runs a single arm, and the aggregate auto_calibration
    session-event kind makes the whole batch one session (see
    AutoCalibrationOptions)."""
    from .auto_calibrate import AutoCalibrationBatchArm, AutoCalibrationBatchRequest

    arms = []
    for arm_opt in opts.arms:
        port, config_file = _resolve_slot(
            record, arm_opt.device_type, arm_opt.arm, arm_opt.port, arm_opt.config_file
        )
        arms.append(
            AutoCalibrationBatchArm(
                device_type=arm_opt.device_type, port=port, config_file=config_file, arm=arm_opt.arm
            )
        )
    return AutoCalibrationBatchRequest(
        arms=arms,
        robot_name=record["name"],
        overwrite=opts.overwrite,
        # The record's persisted per-robot torque cap is the default; an
        # explicit option (the UI's slider draft) overrides it.
        motor_power=opts.motor_power if opts.motor_power is not None else record.get("motor_power"),
    )


_REQUEST_BUILDERS = {
    "teleoperation": _build_teleoperation_request,
    "recording": _build_recording_request,
    "inference": _build_inference_request,
    "replay": _build_replay_request,
    "calibration": _build_calibration_request,
    "auto_calibration": _build_auto_calibration_request,
}


def _dispatch_start(kind: str, request, websocket_manager) -> dict[str, Any]:
    from . import auto_calibrate, calibrate, record, replay, rollout, teleoperate

    if kind == "teleoperation":
        return teleoperate.handle_start_teleoperation(request, websocket_manager)
    if kind == "recording":
        return record.handle_start_recording(request)
    if kind == "inference":
        return rollout.handle_start_inference(request)
    if kind == "calibration":
        return calibrate.calibration_manager.start_calibration(request)
    if kind == "auto_calibration":
        return auto_calibrate.auto_calibration_batch_manager.start(request)
    return replay.handle_start_replay(request, websocket_manager)


def handle_start_session(body: SessionStartBody, websocket_manager=None) -> dict[str, Any]:
    """Start a session by robot name; the router returns the dict as a 201.

    Flow: hardware-hold gate (409 session.held) → robot record resolution
    (404 robot.not_found) → readiness with the arms the kind actually drives
    (400 robot.not_ready; the setup kinds instead require only a port per
    targeted slot) → owner/lease-timeout and per-kind options validation
    (422 request.validation) → build the feature's request model from the
    record → the feature's own start handler. A busy-coded refusal from the
    feature (the gate raced another start) maps to 409 session.held as well;
    any other refusal passes through with its own status/code (name_taken
    defaults to 409).

    An ``owner`` attaches a lease (timeout ``lease_timeout_s``, default 60s,
    10–600 inclusive) and starts the expiry watchdog; no owner, no lease, no
    timeout-stop (see the module docstring).
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

    # The setup kinds skip the record-clean gate (they exist to make records
    # clean); their builders below still refuse a slot with no port.
    if kind not in _SETUP_KINDS:
        arms = "follower" if kind in _FOLLOWER_ONLY_KINDS else "all"
        if not is_robot_record_clean(record, arms=arms):
            needs = "follower arm" if arms == "follower" else "arms"
            raise ApiError(
                status_code=400,
                detail=f"Robot {body.robot!r} is not fully set up for {kind}: "
                f"its {needs} need ports and existing calibrations.",
                code=ErrorCode.ROBOT_NOT_READY,
            )

    # Owner / lease-timeout shape checks live here rather than as pydantic
    # Field constraints so the refusal carries the coded 422 shape, exactly
    # like the options validation below.
    if body.owner is not None and not (1 <= len(body.owner) <= OWNER_MAX_LENGTH):
        raise ApiError(
            status_code=422,
            detail=f"`owner` must be a non-empty string of at most {OWNER_MAX_LENGTH} characters.",
            code=ErrorCode.REQUEST_VALIDATION,
        )
    lease_timeout_s = body.lease_timeout_s
    if lease_timeout_s is None:
        lease_timeout_s = (
            LEASE_TIMEOUT_AUTO_CALIBRATION_S if body.kind == "auto_calibration" else LEASE_TIMEOUT_DEFAULT_S
        )
    if not (LEASE_TIMEOUT_MIN_S <= lease_timeout_s <= LEASE_TIMEOUT_MAX_S):
        raise ApiError(
            status_code=422,
            detail=f"`lease_timeout_s` must be between {LEASE_TIMEOUT_MIN_S:.0f} and "
            f"{LEASE_TIMEOUT_MAX_S:.0f} seconds.",
            code=ErrorCode.REQUEST_VALIDATION,
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
        # The calibration flows' name-collision refusal carries no status_code
        # of its own (legacy callers read it out of a 200 body) — it is a
        # conflict, not a server fault.
        default_status = 409 if code == "name_taken" else 500
        raise ApiError(
            status_code=result.get("status_code", default_status),
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
    if body.owner is not None:
        # The ONLY place a lease attaches (module docstring). None here means
        # the session already ended between attribute and now — nothing left
        # to guard, so no lease and no watchdog.
        leased = tracker.attach_lease(kind, owner=body.owner, timeout_s=lease_timeout_s)
        if leased is not None:
            session = leased
            _ensure_watchdog()
    # Warn-but-allow findings from the feature's start (teleoperation/replay
    # arm-identity checks) ride the 201 so the client can surface them — the
    # legacy start responses carried them and the sessions surface must not
    # drop them. The handlers join their findings into one `warning` string;
    # relay it verbatim as a single entry.
    warning = result.get("warning")
    return {"session": _public_session(session), "warnings": [warning] if warning else None}


# --- current / stop ----------------------------------------------------------


def handle_current_session() -> dict[str, Any]:
    """Identity of the current session, lease included — a pure read. It
    deliberately does NOT renew the lease: reads are for any observer, while
    renewal is the owner's deliberate act (the heartbeat endpoint)."""
    return {"session": _public_session(tracker.current()), "last_ended": tracker.last_ended()}


def handle_heartbeat_session(session_id: str, owner: str) -> dict[str, Any]:
    """Renew the current session's lease deadline — the owner's deliberate
    act; the router returns the dict as a 200.

    404 session.not_found unless `session_id` names the current session — a
    heartbeat for an expiry-stopped session whose release has already landed
    gets the same answer (the session is simply gone; there is no special
    path). 409 session.lease_expired only in the window where the expiry
    watchdog has dispatched the stop but the release event hasn't landed yet.
    409 session.not_owner when the lease belongs to someone else. A current
    session with NO lease makes this a no-op 200 — heartbeating an unleased
    session is harmless, which eases client rollout while lease attachment is
    still opt-in.
    """
    if not (1 <= len(owner) <= OWNER_MAX_LENGTH):
        raise ApiError(
            status_code=422,
            detail=f"`owner` must be a non-empty string of at most {OWNER_MAX_LENGTH} characters.",
            code=ErrorCode.REQUEST_VALIDATION,
        )
    status, snap = tracker.renew_lease(session_id, owner)
    if status == "not_found":
        raise ApiError(
            status_code=404,
            detail=f"No active session with id {session_id!r}.",
            code=ErrorCode.SESSION_NOT_FOUND,
        )
    if status == "expired":
        raise ApiError(
            status_code=409,
            detail="The session's lease expired and its safety stop is already in progress.",
            code=ErrorCode.SESSION_LEASE_EXPIRED,
        )
    if status == "not_owner":
        raise ApiError(
            status_code=409,
            detail="The session's lease belongs to a different owner.",
            code=ErrorCode.SESSION_NOT_OWNER,
        )
    # "renewed", or the documented "no_lease" no-op.
    return {"session": _public_session(snap)}


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

    Deliberately NEVER owner-gated: a physical arm must always be stoppable
    by whoever can reach the API — safety outranks ownership, so a leased
    session accepts a stop from anyone (session.not_owner guards heartbeat
    and future owner-gated mutations, not stop).

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
    return {"session": _public_session(session), "result": result}
