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

from typing import Any
from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.errors import NotFoundError
from makermodslab_sdk.resources._base import Resource, SdkModel

# Mirrors the server's lease/owner constraints (makermodslab/schemas/sessions.py).
# The SDK never imports the server package, so the numbers live here too.
OWNER_MAX_LENGTH = 128


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
