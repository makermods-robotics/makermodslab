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

"""The Lab launches the GPU half of a remote-inference run (`modal run`).

Until S3.8 the operator ran `modal run makermodslab/drtc/modal_policy.py …` in
another terminal and the session merely VERIFIED the result (`_probe_room`).
This module runs that command itself. It is deliberately **not** part of the
session:

- A **Lab-level resource**, like a training job, not a session field. The GPU's
  cold start is 60–180 s during which nothing touches the arm; folding it into
  `handle_start_remote_inference` would set `remote_inference_active` — and
  therefore refuse teleop, record, replay and both calibrations — for minutes
  while the follower sits completely free, and would put a phase with no child
  and no arm inside `sessions._WINDING_DOWN_PHASES`' assumptions.
- Shaped after `utils/system.InstallManager`: a module-level singleton with a
  state, a `Popen`, a log and a lock, plus routes that start / stop / poll it.
- Its exit does **not** stop a session, and `_dispatch_stop` does not stop it.
  A lease expiry is a SAFETY stop whose one job is de-energizing an arm; adding
  a network call to that path is exactly wrong, and the session's own two
  watchdogs already diagnose an empty or silent room far better than "the local
  log stream ended" ever could.

Readiness is a HINT here, never an authority. `state == "ready"` is derived
from the container's own stdout; the only gate on energizing the arm remains
`remote_inference._probe_room`, which observes the ROOM. The worst case of a
reformatted log line is a misleading panel, not a wrong energization.

Attached, never `--detach`. The local `modal run` process is the app's
lifeline, so killing its process group stops the app — the cost-safety property
this whole slice wants, and it composes with `rollout._terminate_tree`'s
SIGTERM→SIGKILL-over-the-group discipline for free. The trade, stated plainly:
the GPU dies with the Lab. That is acceptable because the robot session dies
with it too, and the child's own `finally:` returns the arm to rest first.

The secret never reaches argv
-----------------------------
`modal run … --livekit-api-secret <secret>` would put a signing key in `ps` on
this machine. Both wrappers' `main()` therefore fall back to `LIVEKIT_API_KEY`
/ `LIVEKIT_API_SECRET` from the environment (a `local_entrypoint` body runs
HERE, on the user's machine), and :func:`build_argv` passes neither flag —
they ride the child env instead. `tests/test_modal_launcher.py` asserts both
halves of that, because it is the one regression nothing else would catch.

What is deliberately NOT tested
-------------------------------
The real `modal run` subprocess, Modal authentication, cold-start timing, the
tailscale relay, Modal's log-stream reconnection, and the wrapper env fallback
itself — `modal_policy*.py` import `modal` at module top and the Lab venv has
no `modal`, so those six lines are not importable from the test suite. They are
covered indirectly (the argv test asserts the Lab does not pass the flag) plus
a bench run. Everything else here is pure or clock-injected.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from . import remote_inference
from .api_errors import ApiError, ErrorCode
from .rollout import _terminate_tree
from .utils.config import DRTC_LOG_DIR

logger = logging.getLogger(__name__)

# --- the binary --------------------------------------------------------------

BINARY_NAME = "modal"
# Same override contract as `sfu.ENV_BIN`: it must point at an existing file, so
# a typo REFUSES rather than silently running a different build than asked for.
ENV_MODAL_BIN = "MAKERMODSLAB_MODAL_BIN"
# The CLI is normally a uv tool with its own interpreter (which is why nothing
# here imports `modal`), and uv tools land outside a login shell's PATH when the
# server was started from a launchd/Finder context.
_FALLBACK_BINS = (
    os.path.expanduser("~/.local/bin/modal"),
    "/opt/homebrew/bin/modal",
    "/usr/local/bin/modal",
)
# `uv pip install modal` into the Lab venv would be the WRONG fix: it drags
# Modal's dependency tree into the app's environment to no benefit, and the
# wrappers still could not import `makermodslab` from it.
INSTALL_HINT = "Install the Modal CLI: `uv tool install modal`"

# The two GPU servers, addressed BY PATH so nothing here imports `drtc/` (that
# package is only importable with the optional `[drtc]` extra, and this module
# is imported by the server at boot).
_DRTC_DIR = Path(__file__).parent / "drtc"
WRAPPERS: dict[str, Path] = {
    "sync": _DRTC_DIR / "modal_policy.py",
    "rtc": _DRTC_DIR / "modal_policy_rtc.py",
}

# --- timings -----------------------------------------------------------------

# No `connected as` within this long after the spawn. 60–180 s is the realistic
# band, but a cold `hf-cache` volume plus a first-ever VLA `from_pretrained` can
# exceed 180, and a false failure at that exact moment is maximally annoying.
_COLD_START_TIMEOUT_S = 300.0
# An A100 at Modal is roughly $2-4/hr, so a forgotten `ready` GPU is real money.
# Ten minutes is longer than a realistic gap between two runs (re-arm the form,
# reposition the block, re-grip) and costs at most ~$0.5 of waste. Measured from
# whichever is LATER: reaching `ready`, or the end of the last remote session.
_GPU_IDLE_STOP_S = 600.0
# How long a terminate may take before the group is SIGKILLed (rollout's own
# default, restated so the number is visible next to the ones above).
_TERMINATE_TIMEOUT_S = 5.0
# And how long we then wait for the PUMP to notice. `_terminate_tree` already
# escalated to SIGKILL, so the process is gone; what can still hang is the
# stdout pipe, whose write end an un-reaped grandchild may hold open — leaving
# `readline` blocked forever and the launcher stuck in `stopping` with no way
# out. Past this bound the terminal state is forced and the pump is ORPHANED
# (see `_handle_line`/`_handle_exit`'s `_proc is proc` guard), which is what
# stops a zombie thread from writing into the NEXT launch's state.
_STOP_DRAIN_TIMEOUT_S = 10.0

# One log file per launch, a SIBLING of the session logs — a GPU log can never
# be mistaken for a session's (same argument `remote_inference._LOG_DIR` makes).
_LOG_DIR = Path(DRTC_LOG_DIR) / "gpu"
# How much of the tail `classify_failure` reads. Modal prints its traceback and
# its own error banner at the very end; forty lines covers both without holding
# a whole cold start's chatter in memory.
_TAIL_LINES = 40

# --- log line -> phase -------------------------------------------------------

PHASE_TAILSCALE_UP = "tailscale_up"
PHASE_LOADING = "loading"
PHASE_WARMUP = "warmup"
PHASE_CONNECTING = "connecting"
PHASE_CONNECTED = "connected"
PHASE_CLAIMED = "claimed"

# Matched ANYWHERE in the line, `drtc_protocol`'s rule verbatim: a record
# flushed without its newline — or wearing Modal's own line decoration — would
# otherwise swallow the event. Ordered by progress; a line matching more than
# one resolves to the furthest along.
_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("[tailscale] relay", PHASE_TAILSCALE_UP),
    ("[policy] loading '", PHASE_LOADING),
    ("warming up model", PHASE_WARMUP),
    ("connecting to", PHASE_CONNECTING),
    ("connected as", PHASE_CONNECTED),
    ("claimed control as", PHASE_CLAIMED),
)

# The line that means the policy is in the room. NOT `claimed`: policy.py
# claims control in a BACKGROUND task and the claim is non-fatal, so a
# perfectly healthy run may never print it — treating it as readiness would
# time out runs that are working.
READY_PHASE = PHASE_CONNECTED
_READY_PHASES = frozenset({PHASE_CONNECTED, PHASE_CLAIMED})

STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_FAILED = "failed"
STATE_STOPPING = "stopping"


def find_modal(
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    is_file: Callable[[str], bool] = os.path.isfile,
) -> str | None:
    """Path to the `modal` CLI, or None.

    `MAKERMODSLAB_MODAL_BIN` wins when set and must point at an existing file —
    a typo'd override must not silently fall through to PATH and run a different
    build than the user asked for. Then PATH, then the places a uv tool lands.
    Injected `which`/`is_file` so a test can drive the whole ladder with no
    filesystem (mirrors `sfu.find_livekit_server`).
    """
    env = os.environ if env is None else env
    override = env.get(ENV_MODAL_BIN)
    if override:
        return override if is_file(override) else None
    found = which(BINARY_NAME)
    if found:
        return found
    for candidate in _FALLBACK_BINS:
        if is_file(candidate):
            return candidate
    return None


# --- the transport plan ------------------------------------------------------


@dataclass(frozen=True)
class TransportPlan:
    """Everything the GPU side needs to end up in the SAME room as the arm.

    Produced by :func:`resolve_transport_plan` and consumed by the argv builder
    and the child env. One producer on purpose: the two halves meeting in
    different rooms is the failure Portal turns into a silently dropped stream,
    so a second credential path is not a duplication smell, it is the bug.
    """

    #: What a MODAL CONTAINER dials — not what a local child dials. Empty when
    #: the SFU is up but this machine has no tailnet address to offer.
    url: str
    room: str
    api_key: str
    api_secret: str
    #: Whether the container has to join the tailnet to reach `url` at all.
    #: The SEAM for the open question `--sfu-external-ip` raises: if the Lab's
    #: SFU ever becomes directly reachable, this goes false in ONE place and
    #: the argv, the child env and the tests all follow.
    needs_tailscale: bool
    source: str


def resolve_transport_plan() -> TransportPlan:
    """The launcher's view of `remote_inference.resolve_transport()`.

    Calls THE resolver — never a second credential path — and translates it
    into what a container needs: on the Lab's own SFU that means the TAILNET
    url (`sfu_modal_url`) rather than the loopback one a local child dials, and
    the tailscale flags to reach it; on LiveKit Cloud the url is reachable as-is
    and the container's own `LiveKit-cloud` secret would even carry the
    credentials (we pass ours anyway, so the two halves cannot drift).

    Propagates `OSError`/`RuntimeError` from an unreadable SFU key file, like
    every other caller of the resolver.
    """
    resolved = remote_inference.resolve_transport()
    if resolved.source == "sfu":
        return TransportPlan(
            url=remote_inference.sfu_modal_url() or "",
            room=resolved.room,
            api_key=resolved.api_key,
            api_secret=resolved.api_secret,
            needs_tailscale=True,
            source=resolved.source,
        )
    return TransportPlan(
        url=resolved.url,
        room=resolved.room,
        api_key=resolved.api_key,
        api_secret=resolved.api_secret,
        needs_tailscale=False,
        source=resolved.source,
    )


def build_argv(
    plan: TransportPlan,
    *,
    engine: str,
    policy_hub_id: str,
    task: str,
    horizon: int,
    fps: int,
    video_codec: str,
    s_min: int,
    modal_bin: str,
) -> list[str]:
    """The `modal run` command, as a LIST — never a string, never a shell.

    Mirrors `frontend/.../modalCommand.ts` flag for flag, deliberately: that
    line stays in the panel as the manual fallback and as the ground truth an
    operator compares against when the fingerprint watchdog fires, so the two
    must not be able to disagree.

    Three rules earn their comments:

    - `--task` is OMITTED when empty rather than passed as `""`, matching the
      generated line so the two remain comparable.
    - `--s-min` is rtc-ONLY: `modal_policy.py` has no such flag, so emitting it
      there is a Click usage error, not a defaulted run.
    - **the key and the secret are not here.** They ride the child env
      (`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`), which both wrappers' `main()`
      falls back to. argv is world-readable in `ps` on this machine.

    `--duration` is left at the wrapper's 0.0: the GPU is a Lab-level resource
    with its own idle stop, and the run's length is the ROBOT side's ceiling.
    """
    wrapper = WRAPPERS.get(engine, WRAPPERS["sync"])
    return [
        modal_bin,
        "run",
        str(wrapper),
        "--policy-path",
        policy_hub_id,
        *(["--task", task] if task else []),
        "--horizon",
        str(horizon),
        "--fps",
        str(fps),
        *(["--s-min", str(s_min)] if engine == "rtc" else []),
        "--video-codec",
        video_codec,
        # The room is what makes the two sides meet. Without it the GPU takes
        # the room from its own Modal secret, which the Lab cannot read and
        # therefore cannot check — the one mismatch that is invisible by
        # construction.
        "--livekit-room",
        plan.room,
        *(["--tailscale", "--livekit-url", plan.url] if plan.needs_tailscale else []),
    ]


def child_env(plan: TransportPlan, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the `modal run` child inherits.

    Unbuffered (so the phase lines arrive as they are printed rather than in
    4 KB gulps at the end) plus the two credentials that must never be argv.
    """
    base = dict(os.environ if env is None else env)
    base["PYTHONUNBUFFERED"] = "1"
    base["LIVEKIT_API_KEY"] = plan.api_key
    base["LIVEKIT_API_SECRET"] = plan.api_secret
    return base


def parse_phase(line: str) -> str | None:
    """One log line → a phase, or None for a line that says nothing new.

    A matcher, NOT a state machine: ordering is the caller's business, so a
    `connected as` arriving without a preceding warmup still reports
    `connected`. Returns the FURTHEST-along marker present in the line.
    """
    phase: str | None = None
    for marker, name in _PHASE_MARKERS:
        if marker in line:
            phase = name
    return phase


# --- failure classification --------------------------------------------------

# Substrings, lowercased, that make a failure diagnosable. Kept as data so the
# three interesting classes read as a table rather than as a chain of ifs.
_AUTH_MARKERS = ("modal token new", "not authenticated", "authenticationerror", "token id", "token_id")
_TAILSCALE_MARKERS = ("ts_authkey", "authkey", "auth key", "tailscale up", "[tailscale]")


def classify_failure(
    output_tail: str,
    exit_code: int | None,
    *,
    phase: str | None = None,
    timed_out: bool = False,
) -> tuple[ErrorCode, str, str | None]:
    """(code, message, hint) for a launch that ended badly.

    `timed_out` is the cold-start overrun — the one failure with no exit code
    of its own, because WE ended it. It names the last phase reached, which is
    the whole diagnosis: "stuck at `loading`" (a slow or wrong checkpoint) and
    "stuck at `tailscale_up`" (the tailnet never came up) have nothing in
    common.

    The two text classifiers below are the reason this exists at all. Modal's
    own auth error and an expired tailnet auth key BOTH currently reach the
    operator as `transport.no_policy`, whose hint lists three possible causes;
    the launcher can now say which one.
    """
    tail = output_tail.lower()
    if timed_out:
        where = f" It was still at `{phase}`." if phase else " It never reported a phase."
        return (
            ErrorCode.GPU_LAUNCH_FAILED,
            f"The GPU didn't reach the room within {int(_COLD_START_TIMEOUT_S)}s.{where}",
            "A first-ever load of a large checkpoint can be slow — try again, and check the log below "
            "for what the container was doing. A run stuck before `loading` is usually the tailnet, "
            "not the policy.",
        )
    if any(marker in tail for marker in _AUTH_MARKERS):
        return (
            ErrorCode.GPU_UNAUTHENTICATED,
            "Modal rejected this machine's credentials.",
            "Run `modal token new` in a terminal on this machine, then try again. "
            "The Lab never reads or writes ~/.modal.toml — the CLI owns it.",
        )
    if any(marker in tail for marker in _TAILSCALE_MARKERS):
        return (
            ErrorCode.GPU_LAUNCH_FAILED,
            "The GPU couldn't join the tailnet, so it had no route back to this machine's SFU.",
            "The `tailscale-auth` Modal secret's TS_AUTHKEY has most likely expired. Mint a new "
            "REUSABLE + EPHEMERAL key and run: "
            "`modal secret create tailscale-auth TS_AUTHKEY=tskey-... --force`",
        )
    code = "" if exit_code is None else f" (exit code {exit_code})"
    return (
        ErrorCode.GPU_LAUNCH_FAILED,
        f"`modal run` ended without the policy reaching the room{code}.",
        "The container's own output is in the log below — a bad --policy-path shows up there as a "
        "from_pretrained traceback.",
    )


# --- module state ------------------------------------------------------------
# The InstallManager shape (utils/system.py): one singleton, one lock, one
# subprocess. Everything below is read and written only under `_state_lock`,
# except the two blocking calls (`_terminate_tree`, the pump's readline) that
# must never hold it.

_state_lock = threading.Lock()
_state: str = STATE_IDLE
_proc: subprocess.Popen | None = None
_phase: str | None = None
_engine: str | None = None
_policy_hub_id: str | None = None
_room: str | None = None
_log_path: str | None = None
_started_at: float | None = None  # wall clock, for display
_started_mono: float | None = None  # monotonic, for elapsed + the cold-start bound
_message: str | None = None
_hint: str | None = None
# The classified `gpu.*` code behind a terminal failure, as a plain string;
# None in every non-failed state. It rides the status body so an SDK can
# dispatch on the FAILURE (auth vs tailnet vs generic) the same way it
# dispatches on a coded refusal, instead of matching the prose.
_code: str | None = None
_last_line: str | None = None
# Monotonic stamp the idle auto-stop measures from: set when the GPU reaches
# `ready` and re-set every time a remote session ENDS. None while a session is
# actually running — a busy GPU is not idle, and the countdown must not tick.
_idle_since: float | None = None
# Which terminal state the CURRENT `stopping` resolves to when the process is
# reaped. A stop the operator (or the idle timer) asked for lands in `idle`; a
# cold-start overrun is a FAILURE we happen to implement by killing the group,
# and it must keep its diagnosis on screen rather than looking like a clean
# stop. Read only inside `_handle_exit`.
_stop_outcome: str = STATE_IDLE
# Monotonic instant past which a `stopping` gives up on its pump. Set by the
# terminate thread once the kill has actually returned, cleared on every
# transition out of `stopping`. None means "not waiting on a drain".
_drain_deadline: float | None = None
_tail: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)

# Injected clock, exactly as `remote_inference._clock` is: the cold-start bound
# and the idle stop are DURATIONS, and a test drives them with a FakeClock
# rather than sleeping for five minutes.
_clock: Callable[[], float] = time.monotonic
# Injected too, and a callable rather than an import, so the idle-stop test
# needs no session at all. This is the only thing the launcher asks about the
# session, and it asks in one direction only.
_remote_inference_is_active: Callable[[], bool] = remote_inference.remote_inference_is_active


# --- lifecycle ---------------------------------------------------------------


def _go_idle_locked() -> None:
    """Clear the run's state, keeping nothing but the log path.

    The log path survives an idle transition on purpose: the panel's "the log
    is here" line is the most useful thing left after a run that failed, and
    the next start replaces it.
    """
    global _state, _proc, _phase, _engine, _policy_hub_id, _room
    global _started_at, _started_mono, _idle_since, _last_line, _drain_deadline
    _state = STATE_IDLE
    _proc = None
    _phase = None
    _engine = None
    _policy_hub_id = None
    _room = None
    _started_at = None
    _started_mono = None
    _idle_since = None
    _last_line = None
    _drain_deadline = None


def _open_log() -> tuple[IO[str], Path]:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOG_DIR / f"{int(time.time())}.log"
    return path.open("w", buffering=1), path


def _popen(argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Spawn `modal run` in its own process group.

    `stdin=DEVNULL` deliberately, unlike `remote_inference._spawn`: there is no
    command protocol here, and leaving stdin open on a process we may need to
    kill buys nothing. `start_new_session=True` is what lets `_terminate_tree`
    take the whole tree down with one killpg.
    """
    return subprocess.Popen(  # noqa: S603 — a fixed argv, no shell
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
        bufsize=1,
    )


def _start_pump(proc: subprocess.Popen, log_handle: IO[str]) -> None:
    """Run the stdout pump on its own thread. Its own function so a test can
    replace it and drive `_pump` synchronously."""
    threading.Thread(
        target=_pump,
        args=(proc, log_handle),
        name="modal-launcher-pump",
        daemon=True,
    ).start()


def _pump(proc: subprocess.Popen, log_handle: IO[str]) -> None:
    """Tee the child's output to the log, track the phase, finalize on EOF."""
    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            if not line:
                break
            with contextlib.suppress(Exception):
                log_handle.write(line)
                log_handle.flush()
            try:
                _handle_line(proc, line)
            except Exception:
                # One odd line must never take the pump — and with it every
                # remaining transition — down with it.
                logger.exception("GPU launcher line handling failed for %r", line.strip())
    except Exception:
        logger.exception("GPU launcher stdout pump failed")
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()
        rc = _reap(proc)
        _handle_exit(proc, rc)


def _reap(proc: subprocess.Popen) -> int | None:
    """The exit code once stdout hit EOF. Bounded, so a wedged exit cannot
    hold the pump thread forever."""
    try:
        return proc.wait(timeout=_TERMINATE_TIMEOUT_S)
    except Exception:
        with contextlib.suppress(Exception):
            _terminate_tree(proc, timeout=_TERMINATE_TIMEOUT_S)
        return proc.poll()


def _handle_line(proc: subprocess.Popen, line: str) -> None:
    """Record one line, and advance the state when it is a phase.

    `proc` is the ORPHAN GUARD, not decoration: a pump whose stdout never
    closed can outlive its own launch (see `_STOP_DRAIN_TIMEOUT_S`), and
    without this check that zombie thread would write phases into the NEXT
    run's state. `_proc` is cleared the moment a launch stops owning the slot,
    so a stale pump matches nothing.
    """
    global _phase, _state, _last_line, _idle_since
    text = line.rstrip()
    phase = parse_phase(line)
    with _state_lock:
        if _proc is not proc:
            return
        if _state not in (STATE_STARTING, STATE_READY):
            return
        _tail.append(text)
        if text:
            _last_line = text
        if phase is not None:
            _phase = phase
            if phase in _READY_PHASES and _state == STATE_STARTING:
                _state = STATE_READY
                _idle_since = _clock()
                logger.info("GPU policy server is in room %s", _room)
    # Both bounded checks the plan puts on the two things that already wake:
    # this pump, and a status poll. Deliberately not a thread — there is
    # nothing to watch that does not already wake one of those two.
    _check_deadlines()


def _handle_exit(proc: subprocess.Popen, rc: int | None) -> None:
    """Finalize on the pump's EOF path.

    A stop we asked for lands as `idle` (keeping the idle timer's explanation,
    if that is what asked); a cold-start overrun lands as `failed` with its
    diagnosis intact; anything else is a `failed` the panel keeps until the
    next start. **It never stops a remote-inference session** — the session's
    own watchdogs own that, and they measure the room rather than a proxy for
    it (see the module docstring).
    """
    global _state, _message, _hint, _code, _proc
    with _state_lock:
        if _proc is not proc:
            # A newer launch owns the slot, or this pump was orphaned by the
            # drain bound. Either way its verdict is stale — drop it.
            return
        if _state == STATE_STOPPING:
            if _stop_outcome == STATE_FAILED:
                # The deadline already wrote the message and the hint, and the
                # phase it died at is part of the diagnosis — keep both.
                _proc = None
                _state = STATE_FAILED
            else:
                _go_idle_locked()
            return
        was_ready = _state == STATE_READY
        _proc = None
        _state = STATE_FAILED
        if was_ready:
            # It got to the room and then went away — a real diagnosis, but not
            # one the classifiers can improve on.
            _code = str(ErrorCode.GPU_LAUNCH_FAILED)
            _message = f"The GPU policy server exited{'' if rc is None else f' (exit code {rc})'}."
            _hint = "Start it again, or check the log below for why the container ended."
        else:
            code, _message, _hint = classify_failure("\n".join(_tail), rc, phase=_phase)
            _code = str(code)
    if not was_ready:
        logger.warning("GPU launch failed (rc=%s): %s", rc, _message)


# What a forced terminal state says. The kill DID work — `_terminate_tree`
# escalates to SIGKILL — so this is about the pipe, not about the process, and
# saying so is the difference between "it is still running somewhere" (it is
# not) and "we stopped listening" (we did).
_ABANDONED_NOTE = (
    "Its process group was killed, but the log stream never closed, so the launcher stopped waiting for it."
)


def _terminate_and_watch(proc: subprocess.Popen) -> None:
    """Kill the group, then start the clock on the pump's EOF.

    `_terminate_tree` returns only once the process is gone (SIGTERM, then
    SIGKILL). What can STILL hang after that is the stdout pipe — an un-reaped
    grandchild holding its write end leaves `readline` blocked forever — so the
    pump may never run its finalizer and `stopping` would be permanent. Arming
    the drain deadline here rather than at the stop REQUEST is deliberate: the
    bound is on the drain, not on the kill, which has a ceiling of its own.
    """
    global _drain_deadline
    try:
        _terminate_tree(proc, timeout=_TERMINATE_TIMEOUT_S)
    finally:
        with _state_lock:
            # Only if this stop is still the live one: a pump that finalized
            # while we were killing has already moved the state on.
            if _proc is proc and _state == STATE_STOPPING:
                _drain_deadline = _clock() + _STOP_DRAIN_TIMEOUT_S


def _terminate_async(proc: subprocess.Popen) -> None:
    """Take the process group down OFF the caller's thread.

    Both callers need this. From the pump, a synchronous `_terminate_tree`
    would wait for an exit that cannot happen while the pump is the thing not
    reading the pipe; from a status poll it would put a 5 s kill inside a GET.
    Its own function so a test can make it synchronous.
    """
    threading.Thread(
        target=_terminate_and_watch,
        args=(proc,),
        name="modal-launcher-stop",
        daemon=True,
    ).start()


def _check_deadlines() -> None:
    """The cold-start bound and the idle auto-stop, evaluated once.

    Called from the pump and from every status poll — the two things that
    already wake — so neither deadline needs a thread of its own.
    """
    global _state, _message, _hint, _code, _idle_since, _stop_outcome
    global _proc, _drain_deadline
    doomed: subprocess.Popen | None = None
    with _state_lock:
        now = _clock()
        if _state == STATE_STOPPING:
            # The stop's own bound. Forcing the terminal state ALSO clears
            # `_proc`, which orphans the wedged pump: from here its lines and
            # its eventual verdict are dropped rather than applied to whatever
            # runs next.
            if _drain_deadline is not None and now >= _drain_deadline:
                logger.warning(
                    "GPU launcher gave up on its log stream %.0fs after the kill",
                    _STOP_DRAIN_TIMEOUT_S,
                )
                if _stop_outcome == STATE_FAILED:
                    # Keep the diagnosis AND the phase it names — the same
                    # fields `_handle_exit`'s failure branch leaves standing.
                    _proc = None
                    _drain_deadline = None
                    _state = STATE_FAILED
                    _message = f"{_message} {_ABANDONED_NOTE}" if _message else _ABANDONED_NOTE
                else:
                    _go_idle_locked()  # clears _proc and _drain_deadline too
                    _message = _ABANDONED_NOTE
                    _hint = None
                    _code = None
        elif _state == STATE_STARTING and _started_mono is not None:
            if now - _started_mono >= _COLD_START_TIMEOUT_S:
                code, _message, _hint = classify_failure("\n".join(_tail), None, phase=_phase, timed_out=True)
                _code = str(code)
                _stop_outcome = STATE_FAILED
                _state = STATE_STOPPING
                doomed = _proc
        elif _state == STATE_READY:
            if _remote_inference_is_active():
                # A running session is the opposite of idle; the countdown does
                # not merely pause, it restarts when the session ends.
                _idle_since = None
            else:
                if _idle_since is None:
                    _idle_since = now
                elif now - _idle_since >= _GPU_IDLE_STOP_S:
                    _message = (
                        f"Stopped automatically after {int(_GPU_IDLE_STOP_S / 60)} minutes with no "
                        "remote run — a GPU nobody is using is still billing."
                    )
                    _hint = None
                    _stop_outcome = STATE_IDLE
                    _state = STATE_STOPPING
                    doomed = _proc
    if doomed is not None:
        _terminate_async(doomed)


# --- the API surface ---------------------------------------------------------


def start(
    *,
    engine: str,
    policy_hub_id: str,
    task: str = "",
    horizon: int = 16,
    fps: int = 30,
    video_codec: str = "H264",
    s_min: int = 4,
) -> dict[str, Any]:
    """Launch the GPU policy server. Returns `{started, message, gpu}`.

    Every refusal is synchronous and BEFORE the spawn, because the alternative
    is a Click usage error ninety seconds into a log the user is watching for
    cold-start progress. Deliberately NOT refused while a local training run
    holds the machine: a Modal A100 is not this machine's GPU, and the existing
    remote-inference↔training refusal is already flagged in CLAUDE.md as the one
    asymmetry in the matrix — a second, weaker instance would entrench a rule we
    already suspect.
    """
    global _state, _proc, _phase, _engine, _policy_hub_id, _room
    global _started_at, _started_mono, _log_path, _message, _hint, _last_line, _idle_since
    global _stop_outcome, _code, _drain_deadline

    with _state_lock:
        if _state in (STATE_STARTING, STATE_READY, STATE_STOPPING):
            raise ApiError(
                409,
                f"A GPU policy server is already {_state}. Stop it first.",
                code=ErrorCode.GPU_ALREADY_RUNNING,
            )

    hub_id = policy_hub_id.strip()
    if not hub_id:
        # `--policy-path` is required by both entrypoints and has no default.
        raise ApiError(
            400,
            "The GPU needs a Hub policy id to load — fill in the Hub policy id field.",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        )

    modal_bin = find_modal()
    if modal_bin is None:
        raise ApiError(
            400,
            f"The `modal` command isn't on this machine's PATH. {INSTALL_HINT}",
            code=ErrorCode.GPU_CLI_MISSING,
        )

    try:
        plan = resolve_transport_plan()
    except (OSError, RuntimeError) as exc:
        logger.exception("The bundled SFU's key file could not be read")
        raise ApiError(
            400,
            f"The Lab's SFU is running but its key file couldn't be read: {exc}",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        ) from exc
    if not plan.room or not plan.api_key or not plan.api_secret:
        raise ApiError(
            400,
            "No LiveKit transport is configured, so there is no room to launch the GPU into. "
            "Check the transport section below.",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        )
    if plan.needs_tailscale and not plan.url:
        raise ApiError(
            400,
            "This machine has no tailnet address, so a Modal container has no way to reach its "
            "LiveKit server. Sign in to Tailscale here and re-check the transport.",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        )

    argv = build_argv(
        plan,
        engine=engine,
        policy_hub_id=hub_id,
        task=task,
        horizon=horizon,
        fps=fps,
        video_codec=video_codec,
        s_min=s_min,
        modal_bin=modal_bin,
    )
    log_handle, path = _open_log()
    try:
        proc = _popen(argv, child_env(plan))
    except Exception as exc:
        with contextlib.suppress(Exception):
            log_handle.close()
        logger.exception("Failed to spawn `modal run`")
        raise ApiError(
            400,
            f"Couldn't run `{modal_bin}`: {exc}",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        ) from exc

    with _state_lock:
        _tail.clear()
        _proc = proc
        _state = STATE_STARTING
        _phase = None
        _engine = engine
        _policy_hub_id = hub_id
        _room = plan.room
        _log_path = str(path)
        _started_at = time.time()
        _started_mono = _clock()
        _idle_since = None
        _stop_outcome = STATE_IDLE
        _drain_deadline = None
        _message = None
        _hint = None
        _code = None
        _last_line = None
        payload = _status_locked()
    # The command, WITHOUT the credentials — they are in the env by design and
    # this line is what an operator compares against the panel's manual one.
    logger.info("Launching the GPU policy server: %s", " ".join(argv))
    _start_pump(proc, log_handle)
    return {
        "started": True,
        "message": f"Starting the {engine} policy server on Modal. Cold start is usually 1-3 minutes.",
        "gpu": payload,
    }


def stop() -> dict[str, Any]:
    """Stop the GPU policy server. Returns the status after the request.

    Returns while the process group is still going down (`stopping`); the
    pump's EOF path is what lands it in `idle`. A `failed` launcher has nothing
    left to stop — its process is already gone — so it answers `gpu.not_running`
    like an idle one and is cleared by the next start.
    """
    global _state, _message, _hint, _code, _stop_outcome, _drain_deadline
    with _state_lock:
        if _state in (STATE_IDLE, STATE_FAILED) or _proc is None:
            raise ApiError(
                409,
                "No GPU policy server is running.",
                code=ErrorCode.GPU_NOT_RUNNING,
            )
        doomed = _proc
        _stop_outcome = STATE_IDLE
        # Armed by the terminate thread once the kill returns, not here: the
        # bound is on the pump's drain, not on the kill.
        _drain_deadline = None
        _state = STATE_STOPPING
        # Nothing to say afterwards: the operator asked, and a stale
        # "Stopping…" left on an idle panel would read as a stuck request.
        _message = None
        _hint = None
        payload = _status_locked()
    _terminate_async(doomed)
    return payload


def status() -> dict[str, Any]:
    """The launcher's state — see `schemas/sessions.GpuStatusResponse`.

    A poll is also one of the two things that can notice a cold start that
    overran or a GPU nobody is using any more, and it costs a clock read.
    """
    _check_deadlines()
    with _state_lock:
        return _status_locked()


def _status_locked() -> dict[str, Any]:
    """EVERY key, always — the schema-fidelity rule (`response_model` with no
    exclusion mode materializes absent optionals as null, so the payload has to
    really carry them). Caller holds `_state_lock`."""
    now = _clock()
    elapsed = 0.0 if _started_mono is None else max(0.0, now - _started_mono)
    idle_in: float | None = None
    if _state == STATE_READY and _idle_since is not None:
        idle_in = max(0.0, _GPU_IDLE_STOP_S - (now - _idle_since))
    return {
        "state": _state,
        "phase": _phase,
        "engine": _engine,
        "policy_hub_id": _policy_hub_id,
        "room": _room,
        "log_path": _log_path,
        "started_at": _started_at,
        "elapsed_s": elapsed,
        "message": _message,
        "hint": _hint,
        # The failure's `gpu.*` code, or null. An SDK dispatches on THIS —
        # `gpu.unauthenticated` (run `modal token new`) and a tailnet auth-key
        # expiry have nothing in common as remedies, and the prose beside it is
        # free to improve.
        "code": _code,
        "last_line": _last_line,
        "idle_stop_in_s": idle_in,
    }
