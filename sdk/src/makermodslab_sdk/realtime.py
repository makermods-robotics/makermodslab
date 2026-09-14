"""Realtime events from the server's shared WebSocket (the ``[realtime]`` extra).

One socket — ``/api/v1/ws/joint-data`` — carries everything the server pushes:

* **Joint telemetry**: :class:`JointData` frames at ~20 FPS while
  teleoperation (``type: "joint_update"``) or replay
  (``type: "replay_joint_update"``) is streaming. Nothing streams while no
  hardware flow is running.
* **Control events**: :class:`JobsChanged`, :class:`JobProgress`,
  :class:`SessionChanged`.

THE HINT CONTRACT (matches the frontend and CLAUDE.md "WebSocket broadcast"):
control events are droppable *refetch hints*, never state. On
``session_changed`` refetch the session status endpoint; on ``jobs_changed`` /
``job_progress`` refetch the jobs API. Never make a decision from the hint's
payload — the server may drop, coalesce, or outrun any of them, and a missed
event self-heals on your next fetch. Joint frames are telemetry samples, fine
to read directly, but the same caveat applies: frames are droppable, not a
lossless recording.

Unknown message types parse to :class:`UnknownEvent` (raw payload preserved)
so an older SDK keeps working against a newer server; malformed bodies of a
known type are downgraded the same way rather than raising.

The demux (:func:`parse_message`, :func:`events_from`,
:func:`collect_joint_frames`) is pure and needs no extra dependency; opening a
socket needs the ``websockets`` package — ``pip install
"makermodslab-sdk[realtime]"``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Union

from pydantic import ValidationError

from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.resources._base import SdkModel

WS_PATH = "/api/v1/ws/joint-data"

_INSTALL_HINT = (
    "Realtime streaming needs the websockets package (the SDK's realtime extra). "
    'Next step: pip install "makermodslab-sdk[realtime]", then retry.'
)


class JointData(SdkModel):
    """One joint-telemetry frame (~20 FPS while an arm is being driven).

    ``type`` is ``"joint_update"`` (teleoperation; leader-side positions) or
    ``"replay_joint_update"`` (replay; follower observation). ``joints`` maps
    joint name to position. Bimanual teleoperation adds ``joints_right``, and
    frames may piggyback ``follower_currents_ma`` (~1 Hz-fresh follower
    current draw, mA) — both absent otherwise.
    """

    type: str
    joints: dict[str, float]
    timestamp: float
    joints_right: dict[str, float] | None = None
    follower_currents_ma: dict[str, float] | None = None


class JobsChanged(SdkModel):
    """Hint: the set/state of training jobs changed — refetch the jobs list."""

    type: str
    timestamp: float


class JobProgress(SdkModel):
    """Hint: per-running-job metric snapshots. Refetch the job for real state;
    ``jobs`` (id/state/metrics/wandb_run_url/checkpoint_count dicts) is a
    display convenience, not state."""

    type: str
    jobs: list[dict[str, Any]]
    timestamp: float


class SessionState(SdkModel):
    """The transition named by a session_changed hint."""

    kind: str
    active: bool
    phase: str | None = None


class SessionChanged(SdkModel):
    """Hint: a robot session claimed/changed phase/released — refetch the
    session status endpoint; never act on ``session`` alone."""

    type: str
    session: SessionState
    timestamp: float


class UnknownEvent(SdkModel):
    """Forward-compatible catch-all: an unrecognized or malformed message,
    raw payload preserved. Safe to ignore."""

    raw: Any


RealtimeEvent = Union[JointData, JobsChanged, JobProgress, SessionChanged, UnknownEvent]  # noqa: UP007

_EVENT_TYPES: dict[str, type[SdkModel]] = {
    "joint_update": JointData,
    "replay_joint_update": JointData,
    "jobs_changed": JobsChanged,
    "job_progress": JobProgress,
    "session_changed": SessionChanged,
}


def parse_message(raw: dict[str, Any]) -> RealtimeEvent:
    """Demux one decoded WS message into its typed event.

    Pure and total: every message the server currently sends carries a
    ``type`` key; anything unrecognized — unknown type, missing type, a known
    type whose body doesn't validate, even a non-dict — comes back as
    :class:`UnknownEvent` with the raw payload attached. Never raises.
    """
    if isinstance(raw, dict):
        cls = _EVENT_TYPES.get(raw.get("type"))  # type: ignore[arg-type]
        if cls is not None:
            try:
                return cls.model_validate(raw)  # type: ignore[return-value]
            except ValidationError:
                pass
    return UnknownEvent(raw=raw)


Kinds = Union[type, tuple[type, ...], Iterable[type], None]  # noqa: UP007


def _normalize_kinds(kinds: Kinds) -> tuple[type, ...] | None:
    if kinds is None:
        return None
    if isinstance(kinds, type):
        return (kinds,)
    return tuple(kinds)


def events_from(messages: Iterable[dict[str, Any]], *, kinds: Kinds = None) -> Iterator[RealtimeEvent]:
    """Parse an iterable of decoded messages into typed events.

    ``kinds`` optionally filters to the given event class(es) —
    ``events_from(msgs, kinds=JointData)`` or ``kinds=(JobsChanged,
    SessionChanged)``.
    """
    wanted = _normalize_kinds(kinds)
    for raw in messages:
        event = parse_message(raw)
        if wanted is None or isinstance(event, wanted):
            yield event


def ws_url(base_url: str) -> str:
    """The WebSocket URL for a server base URL (http→ws, https→wss)."""
    base = base_url.rstrip("/")
    for http_scheme, ws_scheme in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http_scheme):
            return ws_scheme + base[len(http_scheme) :] + WS_PATH
    if base.startswith(("ws://", "wss://")):
        return base + WS_PATH
    raise MakerModsError(
        f"Cannot derive a WebSocket URL from base_url {base_url!r} — expected an http:// or https:// URL."
    )


def require_websockets() -> None:
    """Raise the install hint if the ``websockets`` package is missing.

    Called eagerly by the Client hooks so a missing extra surfaces at call
    time, not on the first ``next()`` of a generator.
    """
    try:
        import websockets.sync.client  # noqa: F401
    except ImportError as exc:
        raise MakerModsError(_INSTALL_HINT) from exc


def _connect(base_url: str):
    """Open the sync WebSocket client against a server base URL."""
    require_websockets()
    from websockets.sync.client import connect

    return connect(ws_url(base_url))


def _decode(message: Any) -> dict[str, Any] | None:
    """A wire frame → decoded dict, or None for non-JSON/non-dict frames."""
    if isinstance(message, dict):
        return message
    try:
        data = json.loads(message)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def events(base_url: str, *, kinds: Kinds = None) -> Iterator[RealtimeEvent]:
    """Connect and yield typed events until the server closes the socket.

    This is the open-ended stream (the human/dashboard variant) — it blocks
    between messages and can run forever. Agents wanting a bounded read
    should use :func:`sample_joints` instead. Non-JSON frames are skipped.
    """
    wanted = _normalize_kinds(kinds)
    with _connect(base_url) as ws:
        for message in ws:
            raw = _decode(message)
            if raw is None:
                continue
            event = parse_message(raw)
            if wanted is None or isinstance(event, wanted):
                yield event


def collect_joint_frames(
    recv: Callable[[float], Any],
    duration_s: float,
    *,
    max_frames: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[JointData]:
    """The pure bounded-sampling loop behind :func:`sample_joints`.

    ``recv(timeout)`` returns one wire frame (JSON str or decoded dict) or
    raises ``TimeoutError`` when nothing arrives in time — either ends the
    collection. Stops at the ``duration_s`` deadline (measured by ``clock``)
    or after ``max_frames`` joint frames, whichever comes first; control
    events and undecodable frames pass through without counting.
    """
    frames: list[JointData] = []
    deadline = clock() + duration_s
    while max_frames is None or len(frames) < max_frames:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            message = recv(remaining)
        except TimeoutError:
            break
        raw = _decode(message)
        if raw is None:
            continue
        event = parse_message(raw)
        if isinstance(event, JointData):
            frames.append(event)
    return frames


def sample_joints(
    base_url: str,
    duration_s: float = 2.0,
    *,
    max_frames: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[JointData]:
    """Collect joint frames for up to ``duration_s`` seconds; always returns.

    Returns an empty list when no hardware flow is streaming — that is an
    answer ("nothing is moving"), not an error. ``clock`` is an injection
    seam for tests; leave it defaulted in real use.
    """
    with _connect(base_url) as ws:

        def recv(timeout: float) -> Any:
            try:
                return ws.recv(timeout=timeout)
            except TimeoutError:
                raise
            except Exception as exc:  # ConnectionClosed etc: stream ended early
                raise TimeoutError(str(exc)) from exc

        return collect_joint_frames(recv, duration_s, max_frames=max_frames, clock=clock)
