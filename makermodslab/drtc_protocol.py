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

"""Line protocol spoken between the remote-inference parent and `robot_sync`.

Its own module, deliberately free of any heavy import, for the same reason
`makermodslab.eval_protocol` is: BOTH ends need it and they live on opposite
sides of a dependency wall.

  - the parent (the future `makermodslab.remote_inference`, imported by the
    FastAPI server at boot) must NEVER import `livekit.portal` — it is an FFI
    dylib behind an optional extra, and loading it into the server process is
    exactly what the `[drtc]` split exists to avoid;
  - the child (`makermodslab.drtc.robot_sync`) is precisely the process that
    does.

Restating the strings in both modules would be a silent-drift bug the first
time one side renames a marker — the parent would simply stop recognising
transitions, and a run that had already energized the arm would look like it
never started.

Commands — parent -> child stdin, one bare word per line:
    STOP    leave the control loop, return the arm to its captured start pose,
            disconnect, exit 0
    STOP    (a SECOND one, while the return is in flight) set the abort event
            so the return is cut short and torque releases where the arm is —
            nearer rest than it started. Mirrors the second-stop-press
            semantics of `replay._replay_worker` and
            `teleoperate._return_followers_to_rest`.
    QUIT    immediate: no return-to-rest, straight to disconnect

Events — child -> stdout, one line each, carrying a grep-able prefix so the
parent's log pump can pick them out of lerobot's own INFO chatter (the child's
stderr is merged into the same pipe):
    MAKERMODSLAB-DRTC READY url=wss://x.livekit.cloud room=portal-lerobot-inference
    MAKERMODSLAB-DRTC EASING
    MAKERMODSLAB-DRTC CONNECTED
    MAKERMODSLAB-DRTC ACTIVE operator=policy
    MAKERMODSLAB-DRTC STATS {"t":1,"chunks":3,...}
    MAKERMODSLAB-DRTC RETURNING
    MAKERMODSLAB-DRTC ERROR <message, whitespace collapsed to one line>
    MAKERMODSLAB-DRTC BYE

`READY` echoes the EFFECTIVE url/room — the ones the child resolved from its
flags and `_env`, not the ones the parent believes it passed. The parent
compares and errors on a mismatch, which is what catches "the SFU script was
restarted between preflight and spawn" and "the parent verified room X, the
child's `.env.local` said room Y".

`STATS` is emitted once a second alongside the human `[robot]` log line (which
stays: it is the artifact that made the first live runs diagnosable). Every key
in :data:`STATS_KEYS` is ALWAYS present, `None` where unknown, so a response
model can describe the parent's status dict exactly — per the schema-fidelity
rule in the root CLAUDE.md, a model that materializes absent optionals as
`null` must be describing a payload that really always carries them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable

# --- Commands (stdin) -------------------------------------------------------
CMD_STOP = "STOP"
CMD_QUIT = "QUIT"

# --- Events (stdout) --------------------------------------------------------
# Prefix chosen to be unmistakable in a log full of lerobot INFO lines and
# unlikely to appear in a traceback. Distinct from eval_protocol's
# MAKERMODSLAB-EVAL so the two protocols can never be cross-parsed.
EVENT_PREFIX = "MAKERMODSLAB-DRTC"

EVENT_READY = "READY"
EVENT_EASING = "EASING"
EVENT_CONNECTED = "CONNECTED"
EVENT_ACTIVE = "ACTIVE"
EVENT_STATS = "STATS"
EVENT_RETURNING = "RETURNING"
EVENT_ERROR = "ERROR"
EVENT_BYE = "BYE"

# Every key a STATS payload carries, in emission order. The contract is
# "always present, null where unknown" — see the module docstring. S3.2/S3.3
# consume this set verbatim; adding a key is an API change, so change it here
# and nowhere else.
STATS_KEYS: tuple[str, ...] = (
    "t",  # int   — whole seconds since the control loop started
    "chunks",  # int   — action chunks received
    "reqs",  # int   — observations emitted (one per chunk request)
    "sched",  # int   — runway: steps still queued in the playing chunk
    "lead",  # int   — the adaptive prefetch lead, in steps
    "s_min",  # int   — configured minimum execution budget
    "horizon",  # int   — configured chunk horizon
    "lat_steps",  # int   — the JK latency estimate, in steps
    "lat_ms",  # float — the same estimate in milliseconds
    "holds",  # int   — ticks the player held the last action (starvation)
    "degrade",  # bool  — the player is in play-to-drain-and-hold
    "chunk_age_ms",  # float|None — age of the playing chunk, None before the first
    "active",  # str|None   — the active operator identity, None until one joins
    "e2e_p50_us",  # int|None
    "e2e_p95_us",  # int|None
    "rtt_us",  # int|None
    "uncorr",  # int   — chunks that arrived with no matching observation
)


def format_event(event: str, payload: str = "") -> str:
    """Render one protocol line.

    The payload's whitespace is collapsed so a multi-line exception message can
    never split one event across several lines — the reader is line-oriented and
    a wrapped traceback would otherwise be read as several unknown events.

    The collapse is why STATS must be serialized COMPACTLY
    (`json.dumps(..., separators=(",", ":"))`, which :func:`format_stats` does):
    collapsing runs of whitespace to one space is lossless for compact JSON and
    would mangle a pretty-printed document into something that still parses but
    no longer round-trips byte-for-byte."""
    body = " ".join(str(payload).split())
    return f"{EVENT_PREFIX} {event} {body}".rstrip()


def parse_event(line: str) -> tuple[str, str] | None:
    """`(event, payload)` for a protocol line, or None when the line isn't one.

    Matches the prefix anywhere in the line rather than only at the start: the
    child's logging handler writes to the same merged pipe, and a log record
    flushed without its trailing newline would otherwise swallow the event that
    follows it on the wire. A line that merely mentions the prefix can't be
    produced by the child (it only ever writes it via `format_event`)."""
    idx = line.find(EVENT_PREFIX)
    if idx < 0:
        return None
    rest = line[idx + len(EVENT_PREFIX) :].strip()
    if not rest:
        return None
    event, _, payload = rest.partition(" ")
    return event, payload.strip()


def format_ready(url: str, room: str) -> str:
    """The READY payload: the EFFECTIVE transport the child actually resolved."""
    return f"url={url} room={room}"


def parse_kv(payload: str) -> dict[str, str]:
    """`key=value` tokens of a payload, e.g. READY's url/room or ACTIVE's operator.

    Tokens without an `=` are skipped rather than guessed at — a renamed or
    absent key must read as "not reported", never as a wrong value silently
    accepted (the same discipline `eval_protocol.parse_episode_end_reason`
    applies to its one key)."""
    out: dict[str, str] = {}
    for token in payload.split():
        key, sep, value = token.partition("=")
        if sep and key:
            out[key] = value
    return out


def format_stats(values: dict[str, object]) -> str:
    """Serialize one STATS payload: every key in STATS_KEYS, compact JSON.

    Missing keys are emitted as `null` rather than dropped, and unknown keys
    raise — the always-present contract is what lets the parent's response
    model be exact instead of `exclude_none`."""
    unknown = set(values) - set(STATS_KEYS)
    if unknown:
        raise ValueError(f"unknown STATS keys: {sorted(unknown)}")
    ordered = {key: values.get(key) for key in STATS_KEYS}
    # Compact separators are load-bearing, not cosmetic: format_event collapses
    # whitespace in the payload (see its docstring).
    return json.dumps(ordered, separators=(",", ":"))


def parse_stats(payload: str) -> dict[str, object] | None:
    """Decode a STATS payload, or None when it isn't usable.

    Returns None for malformed or truncated JSON and for a document that isn't
    an object: a dropped or interleaved line must degrade to "no sample this
    second", never to a half-populated status the UI would render as real.
    Missing keys are filled with None so the caller always sees the full set."""
    try:
        decoded = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return {key: decoded.get(key) for key in STATS_KEYS}


def apply_command(
    line: str,
    stop_event: threading.Event,
    abort_event: threading.Event,
    quit_event: threading.Event,
) -> str | None:
    """Act on one command line; return the command recognised, or None.

    The two-STOP rule lives here rather than in the child's loop because it is
    protocol semantics, not loop mechanics: the FIRST STOP asks for a graceful
    return, and any LATER STOP means "I already asked, cut it short" — so it
    sets `abort_event`, which `rest_pose.return_to_rest_pose` polls and reports
    back as `cut-short`. QUIT is the immediate path: it sets all three, so the
    loop breaks, no return is attempted, and any in-flight return unwinds at
    once.

    Blank lines are ignored (the parent may seed a newline), and an
    unrecognised word returns None rather than raising — an unknown command on
    a pipe must never take down a process that is holding an energized arm."""
    command = line.strip().upper()
    if not command:
        return None
    if command == CMD_STOP:
        if stop_event.is_set():
            abort_event.set()
        else:
            stop_event.set()
        return CMD_STOP
    if command == CMD_QUIT:
        quit_event.set()
        stop_event.set()
        abort_event.set()
        return CMD_QUIT
    return None


def pump_commands(
    stream: Iterable[str],
    stop_event: threading.Event,
    abort_event: threading.Event,
    quit_event: threading.Event,
) -> None:
    """Drive :func:`apply_command` over every line of `stream` until EOF.

    Blocks on the stream, so the child runs it on a daemon thread; EOF (the
    parent died or closed the pipe) falls through and the thread ends. It does
    NOT stop the loop on EOF: an abandoned session is the server-side lease
    watchdog's job, and a child that killed itself on a closed pipe would drop
    an energized arm the moment a log pump hiccuped."""
    for raw in stream:
        apply_command(raw, stop_event, abort_event, quit_event)
        if quit_event.is_set():
            return
