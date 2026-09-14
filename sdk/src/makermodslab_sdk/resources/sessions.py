"""The ``sessions`` namespace: THE front door for starting robot flows.

A session is the one live robot flow (teleoperation, recording, inference,
replay, calibration, auto-calibration) plus its identity. Starting takes a
saved robot record NAME and kind-specific options only — ports, configs,
arm layout and cameras all resolve server-side from the record.

The lease is the safety net: starting with an ``owner`` attaches a lease that
must be renewed via heartbeat, and the server's expiry watchdog safety-stops
a session whose owner goes silent (crashed process, dead network) so an arm
is never left energized. Stopping is deliberately never owner-gated — safety
outranks ownership, so anyone who can reach the API can stop the arm.

Response models mirror makermodslab/schemas/sessions.py. SDK models are
``extra="allow"`` everywhere — an older SDK against a newer server must keep
working, and the extra keys stay readable on the object.
"""

from __future__ import annotations

import os
import secrets
import socket
import threading
from typing import Any
from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.errors import ApiError, MakerModsError, NotFoundError
from makermodslab_sdk.resources._base import Resource, SdkModel

# Mirrors the server's lease/owner constraints (makermodslab/schemas/sessions.py).
# The SDK never imports the server package, so the numbers live here too.
OWNER_MAX_LENGTH = 128

# Heartbeats renew at a third of the lease timeout (three chances before the
# watchdog fires), floored so a tiny lease can't turn the SDK into a busy-loop.
MIN_HEARTBEAT_INTERVAL_S = 2.0


# --- response models (shape authority: makermodslab/schemas/sessions.py) -----


class SessionLeaseInfo(SdkModel):
    """The public face of a session's lease. ``expires_in_s`` is computed by
    the server at read time and never negative; only a heartbeat pushes it
    back up — reads don't renew."""

    owner: str
    timeout_s: float
    expires_in_s: float


class SessionInfo(SdkModel):
    """Identity of a session. ``robot`` and ``owner`` are null for sessions
    started through the legacy endpoints; ``lease`` is null for owner-less
    and legacy-started sessions — those are never timeout-stopped."""

    id: str
    kind: str
    robot: str | None
    owner: str | None
    started_at: float
    revision: int
    phase: str | None
    lease: SessionLeaseInfo | None


class EndedSessionInfo(SdkModel):
    """Summary of the most recently ended session. ``reason`` is
    ``"session.lease_expired"`` when the expiry watchdog safety-stopped it,
    null for every normal ending."""

    id: str
    kind: str
    ended_at: float
    phase: str | None
    reason: str | None


class StartedSession(SdkModel):
    """POST /api/v1/sessions response. ``warnings`` relays warn-but-allow
    findings from the feature's start (e.g. teleoperation/replay arm-identity
    checks: the session RUNS, but the servos' EEPROM disagrees with the saved
    calibration). Backend prose — surface it verbatim."""

    session: SessionInfo
    warnings: list[str] | None = None


class CurrentSession(SdkModel):
    """GET /api/v1/sessions/current response — a pure read (never renews the
    lease). ``session`` is null when the robot is idle."""

    session: SessionInfo | None
    last_ended: EndedSessionInfo | None


class StoppedSession(SdkModel):
    """POST /api/v1/sessions/{id}/stop response. ``result`` is the kind's
    stop handler response verbatim (teleoperation's ``releasing``/``warning``,
    etc.); ``session`` is the identity the session ended with — a kind whose
    stop is not immediate may still show as current, in a releasing phase."""

    session: SessionInfo
    result: dict[str, Any]


class SessionCoaching(SdkModel):
    """POST /api/v1/sessions/{id}/coaching response — ``result`` is the
    coaching runner's ``{success, message}`` answer verbatim."""

    result: dict[str, Any]


class _HeartbeatEnvelope(SdkModel):
    session: SessionInfo


def default_session_owner() -> str:
    """The lease owner the sugar methods use when none is given —
    ``sdk:<host>:<pid>:<token>``, unique per call so two SDK processes (or two
    clients in one process) never look like the same owner. Kept within the
    server's 128-char owner cap."""
    host = (socket.gethostname() or "unknown-host")[:96]
    return f"sdk:{host}:{os.getpid()}:{secrets.token_hex(2)}"


def _options(**kwargs: Any) -> dict[str, Any]:
    """A kind-options dict with the unset (None) knobs dropped — the server's
    options models are extra=\"forbid\" with their own defaults; never send a
    null it didn't ask for."""
    return {key: value for key, value in kwargs.items() if value is not None}


class SessionLostError(MakerModsError):
    """The lease on a managed session was lost mid-flight.

    Raised by ``ActiveSession.__exit__`` when a heartbeat discovered the
    session gone or unrenewable while the ``with`` body was still running —
    never when the body itself raised (your exception always wins), and never
    for a deliberate ``stop()``.

    ``reason`` is the coded cause the heartbeat hit: ``"session.not_found"``
    (the session ended under us — someone stopped it, or the watchdog's
    safety stop already completed), ``"session.not_owner"`` (another owner
    holds the lease now), or ``"session.lease_expired"`` (the safety stop was
    in flight). ``client.sessions.current().last_ended`` says how it ended.
    """

    def __init__(self, message: str, *, session_id: str, kind: str, reason: str) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.kind = kind
        self.reason = reason


class ActiveSession:
    """A started session plus the daemon heartbeat that keeps its lease alive.

    The flagship UX — hold the arm for exactly the body of a ``with``:

        >>> with client.sessions.teleoperate("bench") as s:
        ...     print(s.id, s.kind, s.warnings)
        ...     do_things_while_the_arm_runs()
        ...     s.alive  # still holding the lease?
        True

    While the block runs, a daemon thread renews the lease every
    ``max(lease_timeout / 3, 2)`` seconds; leaving the block always stops the
    heartbeat and then the session (also on exceptions — and a crashed
    process is what the server-side watchdog is for). ``stop()`` early is
    fine and idempotent. If the lease was lost mid-flight (the session
    stopped under us, the lease expired, or another owner took it),
    ``alive`` goes False and exiting the block raises :class:`SessionLostError`
    naming the reason — unless the body itself raised, in which case your
    exception propagates untouched and the cleanup still runs.
    """

    def __init__(
        self,
        sessions: SessionsResource,
        started: StartedSession,
        *,
        owner: str,
        auto_heartbeat: bool = True,
    ) -> None:
        self._sessions = sessions
        self._owner = owner
        self._lock = threading.Lock()
        self._info = started.session
        #: Warn-but-allow findings from the start (backend prose, verbatim).
        self.warnings: list[str] = list(started.warnings or [])
        self._lost_reason: str | None = None
        self._lost_detail: str | None = None
        self._stop_requested = False
        self._stop_result: StoppedSession | None = None
        self._stop_event = threading.Event()
        lease = started.session.lease
        #: Seconds between renewals; None when the session has no lease
        #: (nothing to renew — such a session is never timeout-stopped).
        self.heartbeat_interval_s: float | None = (
            max(lease.timeout_s / 3.0, MIN_HEARTBEAT_INTERVAL_S) if lease is not None else None
        )
        self._thread: threading.Thread | None = None
        if auto_heartbeat and self.heartbeat_interval_s is not None:
            self._start_heartbeat_thread()

    # --- identity ------------------------------------------------------------

    @property
    def id(self) -> str:
        """The session id (stable for the session's whole life)."""
        return self._info.id

    @property
    def kind(self) -> str:
        return self._info.kind

    @property
    def info(self) -> SessionInfo:
        """The latest session model seen (start, then each renewal, then stop)."""
        with self._lock:
            return self._info

    @property
    def alive(self) -> bool:
        """True while the session is believed live and the lease held —
        False after :meth:`stop` or once a heartbeat discovered a loss."""
        with self._lock:
            return self._lost_reason is None and not self._stop_requested

    @property
    def lost_reason(self) -> str | None:
        """The coded reason the lease was lost (see :class:`SessionLostError`),
        or None while nothing went wrong."""
        with self._lock:
            return self._lost_reason

    # --- the heartbeat: one TICK (testable) + the timing loop ----------------

    def _tick(self) -> str:
        """One renewal attempt; classification, no timing.

        Returns ``"renewed"``, ``"stopped"`` (a deliberate stop is in
        progress — nothing recorded), ``"gone"`` (the session no longer
        exists), ``"lost"`` (unrenewable: not owner / expiry in flight), or
        ``"transient"`` (network blip or unexpected server error — the
        session is NOT declared lost; the next tick retries).
        """
        with self._lock:
            if self._stop_requested:
                return "stopped"
            session_id, owner = self._info.id, self._owner
        try:
            renewed = self._sessions.heartbeat(session_id, owner)
        except NotFoundError as exc:
            return self._record_loss("session.not_found", exc.detail)
        except ApiError as exc:
            if exc.code in ("session.not_owner", "session.lease_expired"):
                return self._record_loss(exc.code, exc.detail)
            # Anything else (a 5xx hiccup, a proxy's stray page) is treated
            # as transient: a wrongly-declared loss would raise at exit for a
            # session that is in fact fine, and a truly dead lease resolves
            # itself — the watchdog stops the session and the next tick gets
            # the coded 404.
            return "transient"
        except MakerModsError:
            # ConnectionFailedError and kin: the server may be briefly
            # unreachable. Never kill the session over a network blip — the
            # server-side lease is the arbiter of expiry, not the client.
            return "transient"
        with self._lock:
            if not self._stop_requested:
                self._info = renewed
        return "renewed"

    def _record_loss(self, reason: str, detail: str | None) -> str:
        with self._lock:
            if self._stop_requested:
                # A late heartbeat racing our own stop() — the 404 is the
                # stop's success, not a loss.
                return "stopped"
            if self._lost_reason is None:
                self._lost_reason = reason
                self._lost_detail = detail
        return "gone" if reason == "session.not_found" else "lost"

    def _start_heartbeat_thread(self) -> None:
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"makermodslab-sdk-heartbeat-{self._info.id}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        # Event.wait(interval) so stop() interrupts a wait immediately —
        # wait-first, because the lease was just attached at full timeout.
        while not self._stop_event.wait(self.heartbeat_interval_s):
            if self._tick() in ("stopped", "gone", "lost"):
                return

    def coaching_command(self, command: str) -> SessionCoaching:
        """Send a coaching (DAgger) verb to THIS session — ``"takeover"``,
        ``"handback"``, ``"cancel"``, ``"hold"``, ``"resume"``, ``"reset"``,
        ``"recovered"``, ``"drop_last"``. Only meaningful for a run started
        with ``infer(..., coaching=True)``; on anything else the runner
        answers ``result["success"] == False`` with the reason.

        Example:
            >>> s.coaching_command("takeover").result["success"]
            True
        """
        return self._sessions.coaching_command(self.id, command)

    # --- lifecycle ------------------------------------------------------------

    def stop(self) -> StoppedSession | None:
        """Stop the heartbeat, then the session. Idempotent — the second call
        returns the first call's result without another request. Returns None
        when the session was already gone (for a stop, already-gone is
        success)."""
        with self._lock:
            if self._stop_requested:
                return self._stop_result
            self._stop_requested = True
        self._stop_event.set()
        if self._thread is not None:
            # The loop exits on the next wait; a tick already in flight ends
            # harmlessly (its outcome is discarded past _stop_requested).
            self._thread.join(timeout=5.0)
        try:
            result: StoppedSession | None = self._sessions.stop(self._info.id)
        except NotFoundError:
            result = None
        with self._lock:
            self._stop_result = result
            if result is not None:
                self._info = result.session
        return result

    def __enter__(self) -> ActiveSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._lock:
            lost, detail = self._lost_reason, self._lost_detail
        self.stop()
        if exc_type is None and lost is not None:
            raise SessionLostError(
                f"The {self.kind} session {self.id} was lost mid-flight ({lost}): "
                f"{detail or 'no server detail'}\n"
                "Next step: client.sessions.current().last_ended says how it ended; "
                "start a new session when the robot is free.",
                session_id=self.id,
                kind=self.kind,
                reason=lost,
            )

    def __repr__(self) -> str:
        state = "alive" if self.alive else (self.lost_reason or "stopped")
        return f"<ActiveSession {self.kind} {self.id} {state}>"


# Startable kind -> the sugar method covering it. The client half of the
# sessions parity tripwire: tests/test_sessions_parity.py equality-asserts
# this against the server's STARTABLE_KINDS and each method's kwargs against
# the kind's options model, so a new server kind (or option field) fails the
# build here, named, until the SDK grows the matching sugar.
SUGAR_BY_KIND: dict[str, str] = {
    "teleoperation": "teleoperate",
    "recording": "record",
    "inference": "infer",
    "replay": "replay",
    "calibration": "calibrate",
    "auto_calibration": "auto_calibrate",
}


class SessionsResource(Resource):
    """``client.sessions`` — start, watch, heartbeat and stop robot flows.

    The pythonic way is the per-kind sugar, one method per flow, each
    returning an :class:`ActiveSession` context manager:

        >>> with client.sessions.teleoperate("bench") as s:
        ...     print(s.id, s.warnings)
        ...     run_until_done()
        ... # leaving the block stops the session; a lost lease raises
        ... # SessionLostError here.

    The lease in two sentences: every sugar-started session carries a lease
    the ActiveSession's daemon thread renews by heartbeat, and if the
    heartbeats stop arriving (crashed process, dead network) the server's
    expiry watchdog safety-stops the session so the arm is never left
    energized. Stopping is deliberately never owner-gated — anyone who can
    reach the API can stop the arm (:meth:`stop_current` is the hammer).

    The raw operations (:meth:`start`, :meth:`current`, :meth:`heartbeat`,
    :meth:`stop`) are the same surface without the ergonomics — no default
    owner, no heartbeat thread, errors always raised.
    """

    @operation("start_session")
    def start(
        self,
        kind: str,
        robot: str,
        *,
        owner: str | None = None,
        options: dict[str, Any] | None = None,
        lease_timeout_s: float | None = None,
    ) -> StartedSession:
        """Start a session of ``kind`` on the saved robot record named ``robot``.

        ``kind`` is one of ``"teleoperation"``, ``"recording"``,
        ``"inference"``, ``"replay"``, ``"calibration"``,
        ``"auto_calibration"``. ``options`` is the kind-specific payload
        (see makermodslab/schemas/sessions.py); the server validates it
        ``extra="forbid"``, so a field under the wrong kind is a coded 422,
        never a silently ignored knob.

        ``owner`` (non-empty, at most 128 chars) attaches the lease;
        ``lease_timeout_s`` (10-600, server default 60, 90 for
        auto_calibration) is how long the server waits between heartbeats
        before safety-stopping. Surface ``.warnings`` to the user verbatim —
        the session runs, but the server had warn-but-allow findings.

        Example:
            >>> started = client.sessions.start(
            ...     "recording",
            ...     "bench",
            ...     owner="me",
            ...     options={"dataset_repo_id": "me/demo", "single_task": "pick"},
            ... )
            >>> started.session.kind, started.session.lease.timeout_s
            ('recording', 60.0)

        Raises SessionHeldError (409) when another session holds the
        hardware, NotFoundError (404 robot.not_found) for an unknown robot
        name, InvalidRequestError (422) for off-shape options.
        """
        body: dict[str, Any] = {"kind": kind, "robot": robot, "options": dict(options or {})}
        if owner is not None:
            body["owner"] = owner
        if lease_timeout_s is not None:
            body["lease_timeout_s"] = lease_timeout_s
        return StartedSession.model_validate(
            self._transport.request("POST", "/api/v1/sessions", json=body, action=f"Start {kind} session")
        )

    @operation("current_session")
    def current(self) -> CurrentSession:
        """The current session (or None) plus the last ended one — a pure read.

        Reading never renews the lease: renewal is the owner's deliberate act
        (:meth:`heartbeat`). ``last_ended.reason`` tells a safety stop
        (``"session.lease_expired"``) apart from a normal ending.

        Example:
            >>> now = client.sessions.current()
            >>> now.session.kind if now.session else "idle"
            'idle'
        """
        return CurrentSession.model_validate(
            self._transport.request("GET", "/api/v1/sessions/current", action="Get current session")
        )

    @operation("heartbeat_session")
    def heartbeat(self, session_id: str, owner: str) -> SessionInfo:
        """Renew the session's lease; returns the renewed identity.

        ``owner`` must match the lease's owner. Raises NotFoundError (404
        session.not_found) once the session is gone, ApiError 409 on owner
        mismatch (session.not_owner) or an expiry stop already in flight
        (session.lease_expired) — callers treating the lease as advisory
        classify those as "session lost", not as fatal (the ActiveSession
        context manager does this for you). Heartbeating a session with no
        lease is a documented no-op 200.

        Example:
            >>> client.sessions.heartbeat(started.session.id, "me").lease.expires_in_s
            60.0
        """
        return _HeartbeatEnvelope.model_validate(
            self._transport.request(
                "POST",
                f"/api/v1/sessions/{quote(session_id, safe='')}/heartbeat",
                json={"owner": owner},
                action="Heartbeat session",
            )
        ).session

    @operation("stop_session")
    def stop(self, session_id: str) -> StoppedSession:
        """Stop the session with this id.

        Deliberately never owner-gated server-side: a physical arm must
        always be stoppable by whoever can reach the API — safety outranks
        ownership. The id-match is the operation-identity guarantee: a stop
        aimed at an already-ended session is a 404 session.not_found
        (NotFoundError), never a stop of whatever runs now — for a stop that
        404 means "already gone, nothing left to do" (:meth:`stop_current`
        swallows it for you).

        Example:
            >>> stopped = client.sessions.stop(started.session.id)
            >>> stopped.result.get("success")
            True
        """
        return StoppedSession.model_validate(
            self._transport.request(
                "POST",
                f"/api/v1/sessions/{quote(session_id, safe='')}/stop",
                action="Stop session",
            )
        )

    @operation("coaching_command")
    def coaching_command(self, session_id: str, command: str) -> SessionCoaching:
        """One operator verb for a running coaching (DAgger) inference session.

        ``command`` is one of ``"takeover"`` (leader takes the arm, correction
        recording starts), ``"handback"`` (save the correction, policy
        resumes), ``"cancel"`` (abandon the correction unsaved), ``"hold"`` /
        ``"resume"`` (pause/unpause the policy), ``"reset"`` / ``"recovered"``
        (scene reset between attempts), ``"drop_last"`` (discard the last
        saved correction). The 404 session.not_found means the id no longer
        names the live session. ``result`` carries the runner's
        ``{success, message}`` verbatim — ``success=False`` with the reason
        in ``message`` (e.g. the session isn't a coaching run).

        Prefer the ActiveSession helper when you started the run yourself:
        ``s.coaching_command("takeover")``.

        Example:
            >>> client.sessions.coaching_command(s.id, "takeover").result["success"]
            True
        """
        return SessionCoaching.model_validate(
            self._transport.request(
                "POST",
                f"/api/v1/sessions/{quote(session_id, safe='')}/coaching",
                json={"command": command},
                action=f"Coaching command {command!r}",
            )
        )

    def stop_current(self) -> StoppedSession | None:
        """Stop whatever session is live right now; None when the robot is idle.

        The safety hammer: reads the current session and stops it by id,
        whoever owns it. A 404 from the stop (the session ended between the
        read and the stop) is swallowed — for a stop, already-gone is
        success.

        Example:
            >>> client.sessions.stop_current()  # idle robot
            >>> # -> None
        """
        now = self.current().session
        if now is None:
            return None
        try:
            return self.stop(now.id)
        except NotFoundError:
            return None

    # --- per-kind sugar: the pythonic API ------------------------------------
    #
    # Each method mirrors its kind's options model (makermodslab/schemas/
    # sessions.py; TypeScript twin frontend/src/lib/sessionApi.ts) kwarg for
    # kwarg, drops the knobs left at None so the server's defaults rule,
    # defaults the owner so the lease always attaches, and returns an
    # ActiveSession whose daemon heartbeat keeps that lease alive.

    def _start_managed(
        self,
        kind: str,
        robot: str,
        options: dict[str, Any],
        owner: str | None,
        lease_timeout_s: float | None,
    ) -> ActiveSession:
        owner = owner or default_session_owner()
        started = self.start(kind, robot, owner=owner, options=options, lease_timeout_s=lease_timeout_s)
        return ActiveSession(self, started, owner=owner)

    def teleoperate(
        self,
        robot: str,
        *,
        skip_identity_check: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Leader→follower teleoperation on the saved robot ``robot``.

        Example:
            >>> with client.sessions.teleoperate("bench") as s:
            ...     s.warnings  # arm-identity findings, surface verbatim
            []

        ``skip_identity_check=True`` proceeds past a servo-EEPROM /
        calibration mismatch (only when an intentional arm swap was already
        diagnosed — the mismatch error's text says when).
        """
        return self._start_managed(
            "teleoperation",
            robot,
            _options(skip_identity_check=skip_identity_check),
            owner,
            lease_timeout_s,
        )

    def record(
        self,
        robot: str,
        *,
        dataset_repo_id: str,
        single_task: str,
        num_episodes: int | None = None,
        episode_time_s: int | None = None,
        reset_time_s: int | None = None,
        fps: int | None = None,
        video: bool | None = None,
        push_to_hub: bool | None = None,
        tags: list[str] | None = None,
        private: bool | None = None,
        resume: bool | None = None,
        streaming_encoding: bool | None = None,
        skip_identity_check: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Record a teleoperated dataset into ``dataset_repo_id``.

        Only the dataset-shaped knobs live here; cameras resolve server-side
        from the robot record. Server defaults (unset knobs): 5 episodes,
        30s each, 10s reset, 30 fps, video on, local only (no Hub push).

        Example:
            >>> with client.sessions.record(
            ...     "bench", dataset_repo_id="me/demo", single_task="pick the cube"
            ... ) as s:
            ...     wait_for_operator()
        """
        return self._start_managed(
            "recording",
            robot,
            _options(
                dataset_repo_id=dataset_repo_id,
                single_task=single_task,
                num_episodes=num_episodes,
                episode_time_s=episode_time_s,
                reset_time_s=reset_time_s,
                fps=fps,
                video=video,
                push_to_hub=push_to_hub,
                tags=tags,
                private=private,
                resume=resume,
                streaming_encoding=streaming_encoding,
                skip_identity_check=skip_identity_check,
            ),
            owner,
            lease_timeout_s,
        )

    def infer(
        self,
        robot: str,
        *,
        policy_ref: str,
        task: str | None = None,
        camera_bindings: dict[str, str] | None = None,
        camera_dims: dict[str, dict[str, int]] | None = None,
        duration_s: int | None = None,
        checkpoint_state_dim: int | None = None,
        eval_episodes: int | None = None,
        inference_engine: str | None = None,
        temporal_ensemble_coeff: float | None = None,
        coaching: bool | None = None,
        target_corrections: int | None = None,
        coaching_dataset_name: str | None = None,
        skip_identity_check: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Run the trained policy ``policy_ref`` on the follower arm.

        ``camera_bindings`` maps policy-expected camera names to the robot
        record's camera names (the devices themselves come from the record);
        ``camera_dims`` values are ``{"width": ..., "height": ...}``.
        ``inference_engine`` is ``"sync"`` (server default) or ``"rtc"``.

        ``coaching=True`` starts a DAgger coaching run instead of a plain
        rollout: the LEADER arm stands armed for takeover (so unlike plain
        inference this is NOT follower-only), and each takeover→handback
        correction records an episode into ``coaching_dataset_name`` until
        ``target_corrections`` are collected. Drive the run with
        ``s.coaching_command("takeover")`` / ``"handback"`` / … while it runs.

        Example:
            >>> with client.sessions.infer("bench", policy_ref="me/act-pick", task="pick the cube") as s:
            ...     wait_for_rollout()
        """
        return self._start_managed(
            "inference",
            robot,
            _options(
                policy_ref=policy_ref,
                task=task,
                camera_bindings=camera_bindings,
                camera_dims=camera_dims,
                duration_s=duration_s,
                checkpoint_state_dim=checkpoint_state_dim,
                eval_episodes=eval_episodes,
                inference_engine=inference_engine,
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                coaching=coaching,
                target_corrections=target_corrections,
                coaching_dataset_name=coaching_dataset_name,
                skip_identity_check=skip_identity_check,
            ),
            owner,
            lease_timeout_s,
        )

    def replay(
        self,
        robot: str,
        *,
        repo_id: str,
        episode_index: int,
        skip_identity_check: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Replay one recorded episode on the follower arm.

        Example:
            >>> with client.sessions.replay("bench", repo_id="me/demo", episode_index=0) as s:
            ...     wait_for_replay()
        """
        return self._start_managed(
            "replay",
            robot,
            _options(repo_id=repo_id, episode_index=episode_index, skip_identity_check=skip_identity_check),
            owner,
            lease_timeout_s,
        )

    def calibrate(
        self,
        robot: str,
        *,
        device_type: str,
        arm: str | None = None,
        port: str | None = None,
        config_file: str | None = None,
        overwrite: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Manual step-by-step calibration of one arm slot.

        ``device_type`` is ``"robot"`` (follower) or ``"teleop"`` (leader);
        ``arm`` is ``"left"`` (server default; also the single-arm pair) or
        ``"right"``. Calibration is the setup flow, so ``port`` and
        ``config_file`` may ride here (a fresh robot has no saved port yet);
        omitted, they resolve from the record. ``overwrite=True`` is required
        to replace an existing config file of the same name.

        Example:
            >>> with client.sessions.calibrate("bench", device_type="robot") as s:
            ...     drive_calibration_steps()
        """
        return self._start_managed(
            "calibration",
            robot,
            _options(
                device_type=device_type,
                arm=arm,
                port=port,
                config_file=config_file,
                overwrite=overwrite,
            ),
            owner,
            lease_timeout_s,
        )

    def auto_calibrate(
        self,
        robot: str,
        *,
        arms: list[dict[str, Any]],
        motor_power: int | None = None,
        overwrite: bool | None = None,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
    ) -> ActiveSession:
        """Automatic calibration — DRIVES the arm under torque and WRITES
        servo EEPROM; make sure the workspace is clear.

        ``arms`` is 1-4 slot dicts run concurrently, each
        ``{"device_type": "robot"|"teleop", "arm": "left"|"right",
        "port": ..., "config_file": ...}`` with everything after
        ``device_type`` optional (a single arm is a batch of one).
        ``motor_power`` is the drive torque percent (10-100); omitted, the
        record's saved value. The server defaults this kind's lease to 90s —
        a run takes ~60s+ on real hardware.

        Example:
            >>> with client.sessions.auto_calibrate("bench", arms=[{"device_type": "robot"}]) as s:
            ...     wait_for_completion()
        """
        return self._start_managed(
            "auto_calibration",
            robot,
            _options(arms=arms, motor_power=motor_power, overwrite=overwrite),
            owner,
            lease_timeout_s,
        )
