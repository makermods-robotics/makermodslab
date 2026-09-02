"""Typed errors with remediation — the SDK's primary UI for agents.

An agent recovers from a failure by reading the exception, so every error
carries: the HTTP status, the machine-readable ``code``
(``<domain>.<condition>[.<detail>]`` — makermodslab/api_errors.py is the
authority), the server's human-readable ``detail``, structured ``details``
where the code defines them, and the agent-first part: a ``suggestion`` — the
literal next call to make. ``str(err)`` includes all of it.

Catchable hierarchy (everything derives from MakerModsError):

    ConnectionFailedError   server unreachable (no HTTP response at all)
    ApiError                any non-2xx response
      NotFoundError         ``*.not_found`` codes
      InvalidRequestError   ``request.*`` codes / HTTP 422
      RobotBusyError        ``robot.busy.*`` (``.busy_with`` names the holder)
      SessionHeldError      ``session.held`` (``.holder`` names the session)

Branch on ``isinstance`` or on ``.code`` — never on the prose, which the
server is free to reword.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ApiError",
    "ConnectionFailedError",
    "InvalidRequestError",
    "MakerModsError",
    "NotFoundError",
    "RobotBusyError",
    "SessionHeldError",
    "build_api_error",
]

# Exact-code remediations. Keys must be real codes from the server's ErrorCode
# enum — tests/test_errors.py cross-checks. Wording is written for an agent
# mid-task: name the next concrete call, not general advice. Methods referenced
# here that arrive in later stages (sessions/jobs namespaces) are contract:
# those tracks implement exactly these names.
REMEDIATIONS: dict[str, str] = {
    "request.validation": (
        "The request body didn't match the endpoint schema — each entry in .detail names the offending "
        "field. Fix that field and retry."
    ),
    "request.invalid_name": (
        "The server refuses this name — use letters, digits, '-' and '_' only, then retry."
    ),
    "robot.not_found": (
        "No saved robot record has that name. Robot names are exact record names — list them in the web "
        "UI (or GET /api/v1/robots) and retry with one of those."
    ),
    "robot.busy.releasing": (
        "The arm is still returning to rest from the previous session — wait a few seconds and retry."
    ),
    "robot.busy.training": (
        "A local training run holds the machine (GPU and the arms' USB bus). Wait for it or stop it "
        "with client.jobs.stop(job_id); note a training SUBMIT never hits this — it queues instead "
        "(client.jobs.queue())."
    ),
    "job.queue_stale": (
        "The reorder named a set of runs that is no longer the queue — refetch client.jobs.queue() "
        "and retry with the current ids. This is the one 409 in the queue family a retry can clear."
    ),
    "job.state_changed": (
        "The job's state changed since you looked (your expect_state no longer holds) — refetch "
        "client.jobs.get(job_id) and re-decide."
    ),
    "job.has_queued_dependents": (
        "A QUEUED run will train from this job's checkpoint — cancel that queued run first "
        "(client.jobs.queue() shows it), then retry."
    ),
    "job.removal_failed": (
        "The record could not be unlinked; nothing was removed — safe to retry client.jobs.delete(job_id)."
    ),
    "hardware.port_unavailable": (
        "The serial port could not be opened — check the USB cable is plugged in and that no other "
        "process holds the port, then retry."
    ),
    "hardware.connect_failed": (
        "The servo bus did not answer — check the arm's power supply and USB cable, then retry."
    ),
    "hardware.identity_mismatch": (
        "The arm on this port doesn't match the robot record's servo fingerprint — arms were likely "
        "swapped between ports. Replug them to match the record, or pass skip_identity_check=True if "
        "the swap is intentional."
    ),
    "hub.unauthenticated": (
        "The server has no Hugging Face token. Log in with client.system.hf_login(token=...) or run "
        "`hf auth login` on the server machine, then retry."
    ),
    "hub.offline": (
        "The server can't reach the Hugging Face Hub right now — local listings still work; retry the "
        "Hub-touching call when the machine is back online."
    ),
    "hub.repo_not_found": (
        "No Hub repo has that id — check the owner/name spelling and that the repo is visible to the "
        "server's Hub account."
    ),
    "job.not_found": "No job has that id — client.jobs.list() shows the known ids.",
    "dataset.not_found": (
        "No dataset has that repo id — client.datasets.list() shows what the server can see."
    ),
    "model.not_found": "No model has that repo id — client.models.list() shows what the server can see.",
    "session.held": (
        "Another session holds the robot hardware (.holder names it). Stop is deliberately never "
        "owner-gated: call client.sessions.stop_current(), then retry the start."
    ),
    "session.not_found": (
        "That id no longer names the live session. For a stop this means the session is already gone "
        "(nothing left to do); for a heartbeat the session ended — start a new one."
    ),
    "session.not_owner": (
        "The lease belongs to a different owner and only the owner may heartbeat. Reads and stop are "
        "not owner-gated — client.sessions.current() shows who holds it."
    ),
    "session.lease_expired": (
        "The lease lapsed and the watchdog safety-stopped the session. Heartbeat within the lease "
        "timeout next time (the session context manager does this for you) — start a new session."
    ),
    "node.unreachable": (
        "The peer didn't answer /api/v1/health with a node identity document — check the URL, that the "
        "peer is running, and that this machine can reach it."
    ),
    "node.duplicate": "That peer is already registered — client.nodes.list() shows it.",
    "node.self": "That URL points back at this server itself — register other machines, not this one.",
    "internal.unexpected": (
        "Server-side bug — the .detail carries the exception text; check the server logs for the traceback."
    ),
}

# Family fallbacks, keyed by code PREFIX, used when no exact entry matches
# (e.g. robot.busy.teleoperation → the "robot.busy" family).
FAMILY_REMEDIATIONS: dict[str, str] = {
    "robot.busy": (
        "Another flow holds the robot hardware. See what it is with client.sessions.current(), stop it "
        "with client.sessions.stop_current(), then retry."
    ),
}


def suggestion_for(code: str | None) -> str | None:
    """The remediation for a code: exact entry first, then its family."""
    if code is None:
        return None
    if code in REMEDIATIONS:
        return REMEDIATIONS[code]
    parts = code.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        family = ".".join(parts[:cut])
        if family in FAMILY_REMEDIATIONS:
            return FAMILY_REMEDIATIONS[family]
    return None


class MakerModsError(Exception):
    """Base for every error this SDK raises on purpose."""


class ConnectionFailedError(MakerModsError):
    """No HTTP response at all — the server was unreachable."""

    def __init__(self, message: str, *, base_url: str) -> None:
        super().__init__(message)
        self.base_url = base_url


class ApiError(MakerModsError):
    """A non-2xx response, decoded. The Python twin of the frontend's ApiError."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        detail: str | None = None,
        code: str | None = None,
        details: Any = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.code = code
        self.details = details
        self.suggestion = suggestion


class NotFoundError(ApiError):
    """A ``*.not_found`` code — the named resource doesn't exist (anymore)."""


class InvalidRequestError(ApiError):
    """The request itself was malformed (``request.*`` codes, HTTP 422)."""


class RobotBusyError(ApiError):
    """A ``robot.busy.*`` mutual-exclusion refusal."""

    @property
    def busy_with(self) -> str | None:
        """What holds the robot: "teleoperation", "recording", "releasing", …"""
        parts = (self.code or "").split(".", 2)
        return parts[2] if len(parts) == 3 else None


class SessionHeldError(ApiError):
    """``session.held`` — a live session holds the hardware."""

    @property
    def holder(self) -> dict[str, Any] | None:
        """The holding session, ``{"kind": ..., "session_id": ...}``, when the server named it."""
        if isinstance(self.details, dict):
            holder = self.details.get("holder")
            if isinstance(holder, dict):
                return holder
        return None


def _normalize_detail(raw: Any) -> str | None:
    """The frontend's normalization, verbatim: FastAPI 422s put a list of
    ``{loc,msg,type}`` in ``detail``, most errors a plain string — never let a
    non-string render as its repr soup."""
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        import json

        return "; ".join(
            d.get("msg") if isinstance(d, dict) and d.get("msg") is not None else json.dumps(d) for d in raw
        )
    import json

    return json.dumps(raw)


def _class_for(status: int, code: str | None) -> type[ApiError]:
    if code == "session.held":
        return SessionHeldError
    if code is not None and code.startswith("robot.busy."):
        return RobotBusyError
    if code is not None and code.endswith(".not_found"):
        return NotFoundError
    if (code is not None and code.startswith("request.")) or status == 422:
        return InvalidRequestError
    return ApiError


def build_api_error(status: int, body: Any, action: str) -> ApiError:
    """Decode a non-2xx body (parsed JSON, or None when it wasn't JSON) into
    the right exception, message written for the agent that will read it."""
    detail: str | None = None
    code: str | None = None
    details: Any = None
    if isinstance(body, dict):
        detail = _normalize_detail(body.get("detail", body.get("message")))
        if isinstance(body.get("code"), str):
            code = body["code"]
        if body.get("details") is not None:
            details = body["details"]

    suggestion = suggestion_for(code)
    headline = f"{action} failed ({status}{', ' + code if code else ''}): {detail or 'no error detail'}"
    message = headline if suggestion is None else f"{headline}\nNext step: {suggestion}"
    cls = _class_for(status, code)
    return cls(message, status=status, detail=detail, code=code, details=details, suggestion=suggestion)
