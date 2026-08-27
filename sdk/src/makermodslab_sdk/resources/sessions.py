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


class _HeartbeatEnvelope(SdkModel):
    session: SessionInfo


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

    def __exit__(self, exc_type, exc, tb) -> bool:
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
        return False

    def __repr__(self) -> str:
        state = "alive" if self.alive else (self.lost_reason or "stopped")
        return f"<ActiveSession {self.kind} {self.id} {state}>"


class SessionsResource(Resource):
    """``client.sessions`` — start, watch, heartbeat and stop robot flows.

    Example:
        >>> started = client.sessions.start("teleoperation", "bench", owner="me", options={})
        >>> started.session.id
        'e3b0c44298fc1c149afbf4c8996fb924'
        >>> client.sessions.stop(started.session.id).result["success"]
        True

    Starting with an ``owner`` attaches a lease: renew it with
    :meth:`heartbeat` before ``lease.expires_in_s`` runs out, or the server's
    expiry watchdog safety-stops the session. Without an owner there is no
    lease and no timeout-stop.
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
