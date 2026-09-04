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
lifeline. The trade, stated plainly: the GPU dies with the Lab. That is
acceptable because the robot session dies with it too, and the child's own
`finally:` returns the arm to rest first.

Why the stop is SIGINT first (S3.8c)
------------------------------------
"Killing the client stops the app" was the assumption S3.8 shipped on, and it
is FALSE. In the `modal` client (`modal/runner.py::_run_app`) the app-stop RPC
— `AppClientDisconnect`, which terminates an ephemeral app's tasks — is sent
from `except KeyboardInterrupt` and from the exception paths. There is no
SIGTERM handler. So a client killed with SIGTERM/SIGKILL, which is what
`rollout._terminate_tree` does, dies WITHOUT telling Modal anything, and the
container keeps running (and billing) until Modal's client-heartbeat timeout
reaps it — measured at 5-7 minutes, three times on 2026-09-03.

Hence :func:`_graceful_terminate`: SIGINT to the child's process group first
(what a terminal's Ctrl-C would deliver), a bounded wait, and only then the
existing SIGTERM→SIGKILL escalation. And hence the app-id record: when the
client did NOT get to make that call, `modal app stop --yes <app id>` is the
only remaining way to stop the app, and it needs an id that a dead process
cannot tell us — so it is parsed from the log and persisted the moment it
appears.

The credential is a token, and it never reaches argv
----------------------------------------------------
The GPU side joins the room with an OPERATOR-role JWT the station signs for
identity `policy` (`remote_inference.resolve_transport().policy_token`) — never
with the SFU's API key and secret, which stay in the station's 0600 key file.
Even so, `modal run … --livekit-token <jwt>` would put a live credential in
`ps` on this machine, so both wrappers' `main()` fall back to `LIVEKIT_TOKEN`
from the environment (a `local_entrypoint` body runs HERE, on the user's
machine) and :func:`build_argv` passes no such flag — it rides the child env
instead. `tests/test_modal_launcher.py` asserts both halves of that, because
it is the one regression nothing else would catch.

Which workspace it bills (S3.8b)
--------------------------------
A machine can hold many Modal profiles, one of them active, and each profile's
workspace holds one or more environments. The launch takes both PER LAUNCH,
through the two mechanisms the CLI itself documents, and through no other:

- the PROFILE as ``MODAL_PROFILE`` in the CHILD ENV, next to the credentials
  that already travel there. Never ``modal profile activate`` — that rewrites
  ``~/.modal.toml`` and would silently re-point every other terminal on this
  machine, which is a side effect a web request has no business having.
- the ENVIRONMENT as ``modal run --env <name>``, a `modal run` OPTION, so it
  goes before the wrapper path; after it, Click hands it to the wrapper's own
  `local_entrypoint` and the run dies on an unknown flag.

Empty for either means the CLI's own resolution (``MODAL_ENVIRONMENT``, then
the active profile, then the workspace default), which is exactly what S3.8
did — so an API client that never sends them sees no change at all.

``~/.modal.toml`` holds both halves of every profile's token and is NEVER read
here. The two listing commands (`modal profile list --json`,
`modal environment list --json`) carry names and workspaces only, and are the
launcher's one and only view of what this machine can bill.

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
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from . import remote_inference
from .api_errors import ApiError, ErrorCode
from .rollout import _signal_group, _terminate_tree
from .utils.config import DRTC_GPU_APP_FILE, DRTC_LOG_DIR, _atomic_write_text

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
# package is only importable with the optional `[remote]` extra, and this module
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
# How long the `modal run` client gets to shut its APP down after SIGINT,
# before the SIGTERM→SIGKILL escalation takes over. The client sends
# `AppClientDisconnect` immediately (observed: it prints "Stopping app" and is
# gone within ~2 s) and only then waits on the final log stream, which Modal
# bounds itself at its own `logs_timeout` (10 s by default). So this bound has
# to cover the RPC, not the log drain: 5 s is 2.5x the observed exit and still
# leaves the whole stop inside the panel's poll interval. Escalating past it is
# safe rather than lossy — `_settle_app` then stops the app over the API.
_SIGINT_GRACE_S = 5.0
# The bound on the `modal app stop --yes <app id>` fallback. It is one API call
# with no log stream behind it; anything slower is a Modal outage, and the
# right answer there is a logged line, not a stop that never returns.
_APP_STOP_TIMEOUT_S = 15.0
# The two listings behind the profile / environment pickers. They sit inside a
# GET the panel calls on open, so the bound is short: an unreachable Modal must
# cost the operator a coded line, not a hung panel. `environment list` talks to
# the API (it is the one that can be slow); `profile list` is local.
_TARGETS_TIMEOUT_S = 8.0
# How much of a failed listing's own output is quoted back. One line, capped:
# enough to name the cause, short enough that a CLI that ever decides to dump
# something long cannot turn a status line into a wall.
_TARGETS_DETAIL_CHARS = 240
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
# Where the running app's id is remembered ACROSS PROCESSES, so a restart can
# still stop what the last one launched. A module-level Path rather than the
# raw constant so a test can redirect it in one line (nothing here may ever
# write into the developer's real cache).
_APP_RECORD_FILE = Path(DRTC_GPU_APP_FILE)
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


# --- what this machine can bill ----------------------------------------------
# Everything below reads the `modal` CLI's own listings and nothing else. It
# never opens ~/.modal.toml (which holds token_id AND token_secret per profile),
# never activates a profile, and never runs a subcommand with a side effect.


def _as_bool(value: Any) -> bool:
    """Modal's two listings disagree about how to say "active".

    `profile list --json` emits a JSON boolean; `environment list --json` emits
    the STRING "True" (its table renderer's text, serialized as-is). Parsing
    defensively here is cheaper than depending on either staying put.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def parse_profiles(payload: Any) -> list[dict[str, Any]]:
    """`modal profile list --json` → `[{name, workspace, active}]`.

    Pure, and forgiving: a row without a usable name is DROPPED rather than
    offered as a blank option, and a missing workspace renders as empty. A
    listing this build cannot understand must degrade to "no choices", never to
    a picker whose entries would launch against something unnamed.
    """
    rows = payload if isinstance(payload, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "workspace": str(row.get("workspace") or "").strip(),
                "active": _as_bool(row.get("active")),
            }
        )
    return out


def parse_environments(payload: Any) -> list[dict[str, Any]]:
    """`modal environment list --json` → `[{name, active}]`.

    Same forgiveness as `parse_profiles`. Note the shape it reads: the CLI's
    rows are `{"name": ..., "web suffix": ..., "active": "True"}` — a key with
    a SPACE in it and a string boolean, neither of which this parser depends on
    beyond the two fields it takes.
    """
    rows = payload if isinstance(payload, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append({"name": name, "active": _as_bool(row.get("active"))})
    return out


def _listing_error(text: str, exit_code: int | None) -> tuple[str, str]:
    """(code, message) for a listing that came back badly.

    Modal's unauthenticated failure is the one worth naming — its remedy
    (`modal token new`) has nothing in common with any other, and it is exactly
    the state a machine with seven profiles ends up in when one of them expires.
    Everything else is quoted back, one capped line of it: the CLI's own words
    beat any sentence written here about a condition we did not anticipate.
    """
    if any(marker in text.lower() for marker in _AUTH_MARKERS):
        return (
            str(ErrorCode.GPU_UNAUTHENTICATED),
            "Modal rejected this machine's credentials, so its profiles couldn't be listed. "
            "Run `modal token new` in a terminal on this machine. "
            "The Lab never reads or writes ~/.modal.toml — the CLI owns it.",
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    detail = f" {lines[-1][:_TARGETS_DETAIL_CHARS]}" if lines else ""
    code = "" if exit_code is None else f" (exit code {exit_code})"
    return (
        str(ErrorCode.GPU_TARGETS_UNAVAILABLE),
        f"The `modal` CLI couldn't list this machine's Modal targets{code}.{detail}",
    )


def _modal_json(argv: list[str], *, profile: str = "") -> tuple[Any, tuple[str, str] | None]:
    """Run one read-only `modal … --json` listing. `(payload, None)` or `(None, (code, message))`.

    `profile` selects WHICH profile the listing describes, through the env var
    the CLI documents for exactly this — never `modal profile activate`, which
    would rewrite a file every other terminal on this machine shares.

    Bounded and non-raising by construction: this sits inside a GET that a panel
    calls on open, and a listing failure is a line of text, never a 500 and
    never a blocked launch.
    """
    env = dict(os.environ)
    if profile:
        env["MODAL_PROFILE"] = profile
    try:
        proc = subprocess.run(  # noqa: S603 — a fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=_TARGETS_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, (
            str(ErrorCode.GPU_TARGETS_UNAVAILABLE),
            f"`modal {' '.join(argv[1:])}` didn't answer within {int(_TARGETS_TIMEOUT_S)}s.",
        )
    except OSError as exc:
        return None, (str(ErrorCode.GPU_TARGETS_UNAVAILABLE), f"Couldn't run `{argv[0]}`: {exc}")
    if proc.returncode != 0:
        return None, _listing_error(f"{proc.stderr or ''}\n{proc.stdout or ''}", proc.returncode)
    try:
        return json.loads(proc.stdout or ""), None
    except ValueError:
        return None, (
            str(ErrorCode.GPU_TARGETS_UNAVAILABLE),
            "The `modal` CLI's listing wasn't the JSON this build knows how to read. "
            "The profile and environment pickers are unavailable; the launch still uses the CLI's "
            "own active profile.",
        )


def _cli_missing_error() -> dict[str, str]:
    return {
        "code": str(ErrorCode.GPU_CLI_MISSING),
        "message": f"The `modal` command isn't on this machine's PATH. {INSTALL_HINT}",
    }


def list_targets(profile: str | None = None) -> dict[str, Any]:
    """What this machine can bill: its Modal profiles, and one profile's environments.

    `profile` picks whose environments to list; empty means the active one.
    Returns `{profiles, environments, profile, error}` where `profile` names the
    profile the environments belong to (null when none could be listed) and
    `error` is `{code, message}` or null.

    NEVER raises and never 500s. A missing CLI, an expired token, a slow API and
    a listing this build cannot parse all come back as a coded body, because the
    launch does not depend on any of them: with no selection at all the CLI
    resolves the profile and environment itself, exactly as it did before S3.8b.
    """
    modal_bin = find_modal()
    if modal_bin is None:
        return {"profiles": [], "environments": [], "profile": None, "error": _cli_missing_error()}

    wanted = (profile or "").strip()
    payload, err = _modal_json([modal_bin, "profile", "list", "--json"])
    if err is not None:
        return {
            "profiles": [],
            "environments": [],
            "profile": None,
            "error": {"code": err[0], "message": err[1]},
        }
    profiles = parse_profiles(payload)
    active = next((p["name"] for p in profiles if p["active"]), None)

    if wanted and not any(p["name"] == wanted for p in profiles):
        # Naming a profile this machine does not have is the one case where
        # falling back to the active one would be actively harmful: the panel
        # would then show a DIFFERENT workspace's environments under the label
        # the operator picked.
        return {
            "profiles": profiles,
            "environments": [],
            "profile": None,
            "error": {
                "code": str(ErrorCode.GPU_TARGETS_UNAVAILABLE),
                "message": f"This machine has no Modal profile called `{wanted}`.",
            },
        }

    listed_for = wanted or active
    payload, err = _modal_json([modal_bin, "environment", "list", "--json"], profile=wanted)
    if err is not None:
        # The profiles are still worth returning: picking one is what re-runs
        # this listing, and a workspace that is merely slow to answer should not
        # cost the operator the profile picker too.
        return {
            "profiles": profiles,
            "environments": [],
            "profile": None,
            "error": {"code": err[0], "message": err[1]},
        }
    return {
        "profiles": profiles,
        "environments": parse_environments(payload),
        "profile": listed_for,
        "error": None,
    }


def check_target(profile: str, environment: str) -> None:
    """Refuse a launch aimed at a target this machine cannot confirm.

    A no-op when BOTH are empty, which is the whole S3.8 behaviour and the
    default: the CLI resolves the profile and environment itself, and there is
    nothing to check.

    When either is named, the listings are the authority, and a listing that
    did not answer is a REFUSAL rather than a shrug. The asymmetry is
    deliberate and it is about money: an unconfirmable target may be a typo
    (loud, ninety seconds later, in a log nobody is reading yet) or a real
    workspace belonging to someone else (silent, and billed). Refusing costs a
    retry; guessing costs an A100-hour on the wrong account.

    Raises `ApiError(400)` with a `gpu.*` code; the caller does nothing but let
    it out.
    """
    wanted_profile = profile.strip()
    wanted_env = environment.strip()
    if not wanted_profile and not wanted_env:
        return

    targets = list_targets(wanted_profile)
    error = targets["error"]
    if error is not None:
        raise ApiError(
            400,
            f"{error['message']} Clear the profile and environment to let the `modal` CLI choose, "
            "or fix the CLI and try again.",
            code=error["code"],
        )
    if wanted_env and not any(e["name"] == wanted_env for e in targets["environments"]):
        known = ", ".join(f"`{e['name']}`" for e in targets["environments"]) or "none"
        where = f"profile `{wanted_profile}`" if wanted_profile else "the active profile"
        raise ApiError(
            400,
            f"{where} has no Modal environment called `{wanted_env}`. It has: {known}.",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        )


# --- the transport plan ------------------------------------------------------


@dataclass(frozen=True)
class TransportPlan:
    """What the container needs to reach the room, as the launcher sees it.

    Produced by :func:`resolve_transport_plan` and consumed by the argv builder
    and the child env. `url` is the address a CONTAINER dials — the station's
    tailnet address, never the loopback one a local child uses — and is empty
    when the SFU is up but this machine has no tailnet address to offer.
    `token` is the operator-role JWT the station signed for the GPU side's
    identity; the API secret is not in here and never was the container's.
    """

    url: str
    room: str
    token: str
    #: Whether the container has to join the tailnet to reach `url` at all.
    #: Always True while the SFU is the station's own; the seam stays so that
    #: if the SFU ever becomes directly reachable this goes false in ONE place
    #: and the argv, the child env and the tests all follow.
    needs_tailscale: bool
    source: str
    #: False when the Lab runs no SFU: there is no room to launch into.
    configured: bool = True


def resolve_transport_plan() -> TransportPlan:
    """The launcher's view of `remote_inference.resolve_transport()`.

    Calls THE resolver — never a second credential path — and translates it
    into what a container needs: the TAILNET url (`sfu_modal_url`) rather than
    the loopback one a local child dials, the tailscale flags to reach it, and
    the policy token the station signed. An unconfigured transport (no `--sfu`)
    comes back as an unconfigured plan, which `start()` refuses.

    Propagates `OSError`/`RuntimeError` from an unreadable SFU key file, like
    every other caller of the resolver.
    """
    resolved = remote_inference.resolve_transport()
    if not resolved.configured:
        return TransportPlan(
            url="",
            room="",
            token="",  # noqa: S106  # nosec B106 — no SFU, nothing was signed
            needs_tailscale=False,
            source=resolved.source,
            configured=False,
        )
    return TransportPlan(
        url=remote_inference.sfu_modal_url() or "",
        room=resolved.room,
        token=resolved.policy_token,
        needs_tailscale=True,
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
    environment: str = "",
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
    - `--env` is a `modal run` OPTION and therefore goes BEFORE the wrapper
      path — `modal run [OPTIONS] FUNC_REF`. After it, Click hands it to the
      wrapper's own `local_entrypoint`, which has no such parameter, and the
      run dies on an unknown flag instead of billing the named environment.
      Omitted when empty, so the CLI's own resolution (`MODAL_ENVIRONMENT`,
      the active profile, the workspace default) stands untouched.
    - the PROFILE is deliberately NOT here: it travels as `MODAL_PROFILE` in
      the child env (`child_env`), because the CLI takes it that way and the
      alternative — `modal profile activate` — would rewrite a file every
      other terminal on this machine shares.
    - **the token is not here.** It rides the child env (`LIVEKIT_TOKEN`),
      which both wrappers' `main()` falls back to. argv is world-readable in
      `ps` on this machine.

    `--duration` is left at the wrapper's 0.0: the GPU is a Lab-level resource
    with its own idle stop, and the run's length is the ROBOT side's ceiling.
    """
    wrapper = WRAPPERS.get(engine, WRAPPERS["sync"])
    return [
        modal_bin,
        "run",
        *(["--env", environment] if environment else []),
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


def child_env(
    plan: TransportPlan,
    env: Mapping[str, str] | None = None,
    *,
    profile: str = "",
) -> dict[str, str]:
    """The environment the `modal run` child inherits.

    Unbuffered (so the phase lines arrive as they are printed rather than in
    4 KB gulps at the end) plus the one credential that must never be argv:
    the station-signed operator token.

    `profile` is the third thing that travels here, for a related reason: the
    CLI reads `MODAL_PROFILE` per process, so ONE launch can bill a different
    workspace without touching the machine-wide active profile that every other
    terminal on this machine shares. Empty leaves the variable alone (rather
    than setting it empty, which the CLI would read as a profile named ""),
    so the CLI's own resolution stands.
    """
    base = dict(os.environ if env is None else env)
    base["PYTHONUNBUFFERED"] = "1"
    base["LIVEKIT_TOKEN"] = plan.token
    if profile:
        base["MODAL_PROFILE"] = profile
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


# --- the app id: parsing it, keeping it, using it ----------------------------

# The client prints its app page as one of the very first lines:
#
#     ✓ Initialized. View run at
#     https://modal.com/apps/makermods/main/ap-QfTK2AxcfbJnnY1kLS7Y22
#
# Note the WRAP: Rich breaks that sentence, so the url lands on a line of its
# own and "View run at" is NOT on it. Anchoring on the url is therefore the
# only anchor that works — and it is the stricter one anyway, because it can
# never mistake an `ap-…`-looking token in a traceback for an app id.
_APP_ID_RE = re.compile(r"modal\.com/apps/\S*?(ap-[A-Za-z0-9]+)")


def parse_app_id(line: str) -> str | None:
    """The Modal app id in one log line, or None.

    Pure, like `parse_phase`, and matched ANYWHERE in the line for the same
    reason: Modal decorates and wraps its output freely.
    """
    match = _APP_ID_RE.search(line)
    return match.group(1) if match else None


def _write_app_record(app_id: str, profile: str | None, started_at: float | None) -> None:
    """Remember the running app across processes. Best-effort, never raises.

    A failed write costs an orphan reap after a hard kill, which is strictly
    better than a launch that refuses because a cache file is unwritable.
    """
    try:
        _atomic_write_text(
            str(_APP_RECORD_FILE),
            json.dumps({"app_id": app_id, "profile": profile or "", "started_at": started_at}, indent=2),
        )
    except Exception:
        logger.warning("Couldn't record the GPU app id at %s", _APP_RECORD_FILE, exc_info=True)


def read_app_record() -> dict[str, Any] | None:
    """The persisted `{app_id, profile, started_at}`, or None.

    Forgiving by design: a missing file, unreadable JSON or a row without a
    usable app id all read as "nothing to reap". Nothing downstream may treat
    this file as authoritative — it is a HINT that an app may still be up.
    """
    try:
        payload = json.loads(_APP_RECORD_FILE.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    app_id = str(payload.get("app_id") or "").strip()
    if not app_id:
        return None
    started = payload.get("started_at")
    return {
        "app_id": app_id,
        "profile": str(payload.get("profile") or ""),
        "started_at": float(started) if isinstance(started, (int, float)) else None,
    }


def _clear_app_record(app_id: str) -> None:
    """Forget the app, but ONLY if the file still names this one.

    The guard is the race with a relaunch: a stop's settle runs on its own
    thread and a new launch may already have written its own id by the time it
    gets here. Deleting that would leave the NEW app unreapable.
    """
    current = read_app_record()
    if current is None or current["app_id"] != app_id:
        return
    with contextlib.suppress(OSError):
        _APP_RECORD_FILE.unlink()


def stop_app(app_id: str, profile: str = "") -> tuple[bool, str]:
    """`modal app stop --yes <app id>` — the only way to stop an app whose
    client is already gone. Returns `(stopped, detail)`.

    `--yes` is not optional: without it the CLI prompts, and a server has no
    terminal, so it aborts with "no interactive terminal detected". `profile`
    rides `MODAL_PROFILE` exactly as a launch and a listing do — an app id
    belongs to ONE workspace, so stopping it from the wrong profile is a 404,
    not a no-op.

    Bounded and non-raising: every caller is a best-effort cleanup path.
    "App is already stopped." is SUCCESS — it is the outcome we wanted.
    """
    modal_bin = find_modal()
    if modal_bin is None:
        return False, f"the `modal` command isn't on this machine's PATH ({INSTALL_HINT})"
    env = dict(os.environ)
    if profile:
        env["MODAL_PROFILE"] = profile
    try:
        proc = subprocess.run(  # noqa: S603 — a fixed argv, no shell
            [modal_bin, "app", "stop", "--yes", app_id],
            capture_output=True,
            text=True,
            timeout=_APP_STOP_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"`modal app stop` didn't answer within {int(_APP_STOP_TIMEOUT_S)}s"
    except OSError as exc:
        return False, f"couldn't run `{modal_bin}`: {exc}"
    if proc.returncode == 0:
        return True, "stopped"
    text = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    detail = lines[-1][:_TARGETS_DETAIL_CHARS] if lines else f"exit code {proc.returncode}"
    return False, detail


# --- failure classification --------------------------------------------------

# Substrings, lowercased, that make a failure diagnosable. Kept as data so the
# three interesting classes read as a table rather than as a chain of ifs.
_AUTH_MARKERS = ("modal token new", "not authenticated", "authenticationerror", "token id", "token_id")
# Auth-key failure markers ONLY. `[tailscale]` and `tailscale up` used to be in
# this list, and they appear in EVERY launch log (the container narrates its
# tailnet join), so any failure early enough for the join block to still be
# inside the 40-line tail was blamed on the auth key — on 2026-09-03 a policy
# server that refused to start over an empty --task was reported as "The GPU
# couldn't join the tailnet" with a key-minting hint, while the log's own
# "joined tailnet as modal-policy" sat twenty lines above the real error.
_TAILSCALE_MARKERS = ("ts_authkey", "authkey", "auth key", "needslogin", "login required")
# The line the container prints once the tailnet join succeeded; its presence
# rules the tailnet OUT as the cause of whatever came after.
_TAILNET_JOINED_MARKER = "joined tailnet as"
# The container's own last words. Modal prints the remote exception on one
# line; our servers exit via SystemExit with a full sentence. Either is the
# backend's (well, the GPU's) own prose and is surfaced verbatim.
_REMOTE_EXCEPTION_MARKERS = (
    "uncaught exception raised in remote container:",
    "systemexit:",
)


def _last_remote_exception_line(output_tail: str) -> str | None:
    """The container's exit message, verbatim, or None.

    Modal relays a remote exception as one line beginning
    `Stopping app - uncaught exception raised in remote container: <repr>`;
    our servers refuse to start with a `SystemExit: <sentence>` line in the
    remote traceback. The LAST such line wins (the traceback repeats it),
    and the repr wrapper `SystemExit('…')` is unwrapped so the operator reads
    the sentence, not Python.
    """
    found: str | None = None
    for raw in output_tail.splitlines():
        line = raw.strip()
        low = line.lower()
        for marker in _REMOTE_EXCEPTION_MARKERS:
            idx = low.find(marker)
            if idx < 0:
                continue
            text = line[idx + len(marker) :].strip()
            # SystemExit('msg') / SystemExit("msg") → msg
            for prefix in ("SystemExit('", 'SystemExit("'):
                if text.startswith(prefix):
                    text = text[len(prefix) :].rstrip(")").rstrip("'\"")
            # Modal's line is a Python repr, so quotes inside the sentence
            # arrive escaped (\'molmoact2\'); the operator reads a sentence.
            text = text.replace("\\'", "'").replace('\\"', '"')
            if text:
                found = text
    return found


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
    # The policy server's own refusal (an empty --task for a language-
    # conditioned policy, a checkpoint that cannot be loaded, a wrong flag):
    # the most specific diagnosis there is, and it is already a sentence.
    # Checked BEFORE the tailnet markers, and independently of them: the join
    # block is always in the tail of an early failure.
    remote_line = _last_remote_exception_line(output_tail)
    if remote_line is not None:
        return (
            ErrorCode.GPU_LAUNCH_FAILED,
            f"The policy server stopped itself: {remote_line}",
            "That is the container's own message. Fix what it names and start the GPU again; "
            "the full traceback is in the log below.",
        )
    if any(marker in tail for marker in _TAILSCALE_MARKERS) and _TAILNET_JOINED_MARKER not in tail:
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
# The Modal target this launch was billed to, AS LAUNCHED — null when the
# operator left the choice to the CLI. Echoed in the status so a running GPU
# can say which workspace is paying for it, which is the whole point of letting
# it be chosen at all.
_profile: str | None = None
_environment: str | None = None
# The Modal app id, as printed by the client on its second line. The ONE piece
# of state that outlives this process (see `_APP_RECORD_FILE`), because it is
# the only handle on an app whose client is gone.
_app_id: str | None = None
# The transport tuple AS LAUNCHED. Echoed in the status so the drift warning
# compares the form against the SERVER's record rather than the tab's memory —
# which is what makes it fire after a reload, and for a GPU another tab
# started. Half of it (horizon/fps/codec/s_min) is the Portal wire schema, so a
# disagreement is a stream dropped in silence, not an error; `task` steers the
# policy itself.
_task: str | None = None
_horizon: int | None = None
_fps: int | None = None
_video_codec: str | None = None
_s_min: int | None = None
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
    global _profile, _environment, _app_id
    global _task, _horizon, _fps, _video_codec, _s_min
    _state = STATE_IDLE
    _proc = None
    _phase = None
    _engine = None
    _policy_hub_id = None
    _room = None
    _profile = None
    _environment = None
    _app_id = None
    _task = None
    _horizon = None
    _fps = None
    _video_codec = None
    _s_min = None
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
    global _phase, _state, _last_line, _idle_since, _app_id
    text = line.rstrip()
    phase = parse_phase(line)
    app_id = parse_app_id(line)
    record: tuple[str, str | None, float | None] | None = None
    with _state_lock:
        if _proc is not proc:
            return
        if _state not in (STATE_STARTING, STATE_READY):
            return
        _tail.append(text)
        if text:
            _last_line = text
        if app_id is not None and app_id != _app_id:
            # The first line of the run that matters after this process dies.
            _app_id = app_id
            record = (app_id, _profile, _started_at)
        if phase is not None:
            _phase = phase
            if phase in _READY_PHASES and _state == STATE_STARTING:
                _state = STATE_READY
                _idle_since = _clock()
                logger.info("GPU policy server is in room %s", _room)
    if record is not None:
        # Outside the lock: a disk write has no business inside it, and the
        # record is a hint whose freshness is measured in minutes.
        logger.info("GPU app id: %s", record[0])
        _write_app_record(*record)
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

    It also decides, from the exit code alone, whether the app record can be
    forgotten. A client that exited NORMALLY (rc >= 0 — completion, or an
    uncaught exception) ran its own disconnect, so its app is going down and
    the record would only make the next boot claim it reaped an orphan it
    never had. A client killed by a SIGNAL this launcher did not send (rc < 0,
    someone's `kill -9`) made no such call, so the record is exactly what the
    next boot needs. A stop we asked for is not decided here at all —
    `_settle_app` owns that record, and clearing it from both places would
    race the API stop that may still need it.
    """
    global _state, _message, _hint, _code, _proc
    clear_record: str | None = None
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
        if rc is not None and rc >= 0:
            clear_record = _app_id
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
    if clear_record:
        _clear_app_record(clear_record)
    if not was_ready:
        logger.warning("GPU launch failed (rc=%s): %s", rc, _message)


# What a forced terminal state says. The kill DID work — `_terminate_tree`
# escalates to SIGKILL — so this is about the pipe, not about the process, and
# saying so is the difference between "it is still running somewhere" (it is
# not) and "we stopped listening" (we did).
_ABANDONED_NOTE = (
    "Its process group was killed, but the log stream never closed, so the launcher stopped waiting for it."
)


def _graceful_terminate(proc: subprocess.Popen) -> bool:
    """SIGINT the client's group, wait a bounded moment, THEN escalate.

    Returns True only when the client exited on the SIGINT alone — i.e. when
    it had the chance to run its `except KeyboardInterrupt` and tell Modal to
    stop the app. Every other outcome is False and means `_settle_app` has to
    stop the app over the API instead.

    Three cases, and they are all the same three the old SIGTERM-only path had
    — it just never distinguished them:

    - **already gone.** Nothing to signal, and no way to know whether the
      client made its call; treated as NOT clean, because `modal app stop` on
      an already-stopped app is free and a leaked A100 is not.
    - **exits within `_SIGINT_GRACE_S`.** The good path. The client printed
      "Stopping app", sent `AppClientDisconnect`, and the app is going down.
    - **still there.** Escalate through `rollout._terminate_tree` exactly as
      before (SIGTERM → SIGKILL over the group). Deliberately layered in FRONT
      of that function rather than inside it: other runners depend on its
      current behaviour, and none of them has an app on a remote machine.

    SIGINT goes to the GROUP, not the leader, because that is what a terminal
    delivers on Ctrl-C — the exact signal path Modal's own client is written
    against. `_signal_group` refuses to signal our own group, so a stand-in
    that isn't a real Popen falls through to the escalation rather than
    SIGINTing the server.
    """
    with contextlib.suppress(Exception):
        if proc.poll() is not None:
            return False
    if _signal_group(proc, signal.SIGINT):
        try:
            proc.wait(timeout=_SIGINT_GRACE_S)
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                "The `modal run` client did not exit %.0fs after SIGINT — escalating, "
                "and its app will be stopped over the API",
                _SIGINT_GRACE_S,
            )
        except Exception:
            pass
    _terminate_tree(proc, timeout=_TERMINATE_TIMEOUT_S)
    return False


def _settle_app(app_id: str | None, profile: str | None, *, client_exited_cleanly: bool) -> None:
    """Make sure the Modal APP is stopped, not just the local client.

    The clean case is nothing at all: the client's own disconnect is the
    authority, and a second `modal app stop` would only add a subprocess to
    the stop path. Every other case runs it, because the alternative is an
    A100 billing for the 5-7 minutes Modal's heartbeat timeout takes to notice
    nobody is attached.

    Never raises, and never blocks a state transition: the caller has already
    armed the drain deadline by the time this runs.
    """
    if not app_id:
        if not client_exited_cleanly:
            logger.warning(
                "The GPU client was killed before it named its Modal app, so the app could not "
                "be stopped from here — check `modal app list` if a run is still billing."
            )
        return
    if client_exited_cleanly:
        _clear_app_record(app_id)
        return
    stopped, detail = stop_app(app_id, profile or "")
    if stopped:
        logger.info("Stopped Modal app %s over the API (its client never got to)", app_id)
        _clear_app_record(app_id)
    else:
        logger.error(
            "Could not stop Modal app %s: %s. It may still be billing — run "
            "`modal app stop %s` in a terminal.",
            app_id,
            detail,
            app_id,
        )


def _terminate_and_watch(proc: subprocess.Popen) -> None:
    """Stop the client, start the clock on the pump's EOF, then settle the app.

    `_graceful_terminate` returns only once the process is gone (SIGINT, then
    SIGTERM, then SIGKILL). What can STILL hang after that is the stdout pipe —
    an un-reaped grandchild holding its write end leaves `readline` blocked
    forever — so the pump may never run its finalizer and `stopping` would be
    permanent. Arming the drain deadline here rather than at the stop REQUEST
    is deliberate: the bound is on the drain, not on the kill, which has a
    ceiling of its own.

    `_settle_app` comes AFTER the arming, deliberately: its subprocess must not
    lengthen the window in which the launcher reports `stopping`.
    """
    global _drain_deadline
    with _state_lock:
        app_id, profile = _app_id, _profile
    clean = False
    try:
        clean = _graceful_terminate(proc)
    finally:
        with _state_lock:
            # Only if this stop is still the live one: a pump that finalized
            # while we were killing has already moved the state on.
            if _proc is proc and _state == STATE_STOPPING:
                _drain_deadline = _clock() + _STOP_DRAIN_TIMEOUT_S
    _settle_app(app_id, profile, client_exited_cleanly=clean)


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
    profile: str = "",
    environment: str = "",
) -> dict[str, Any]:
    """Launch the GPU policy server. Returns `{started, message, gpu}`.

    Every refusal is synchronous and BEFORE the spawn, because the alternative
    is a Click usage error ninety seconds into a log the user is watching for
    cold-start progress. Deliberately NOT refused while a local training run
    holds the machine: a Modal A100 is not this machine's GPU, and the existing
    remote-inference↔training refusal is already flagged in CLAUDE.md as the one
    asymmetry in the matrix — a second, weaker instance would entrench a rule we
    already suspect.

    `profile` / `environment` say WHICH WORKSPACE PAYS. Both empty is S3.8's
    behaviour byte for byte — the CLI resolves them — so nothing changes for a
    client that never sends them. A non-empty one is checked against this
    machine's own listings first: an A100-hour billed to the wrong workspace is
    not recoverable, so "we could not confirm the target" refuses in a
    millisecond rather than spending ninety seconds finding out.
    """
    global _state, _proc, _phase, _engine, _policy_hub_id, _room
    global _started_at, _started_mono, _log_path, _message, _hint, _last_line, _idle_since
    global _stop_outcome, _code, _drain_deadline, _profile, _environment, _app_id
    global _task, _horizon, _fps, _video_codec, _s_min

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

    # Before anything is spawned, and before the transport is even resolved: a
    # target the listings cannot confirm is the one pre-spawn refusal whose
    # cost is money rather than a retry.
    want_profile = profile.strip()
    want_environment = environment.strip()
    check_target(want_profile, want_environment)

    try:
        plan = resolve_transport_plan()
    except (OSError, RuntimeError) as exc:
        logger.exception("The bundled SFU's key file could not be read")
        raise ApiError(
            400,
            f"The Lab's SFU is running but its key file couldn't be read: {exc}",
            code=ErrorCode.GPU_LAUNCH_FAILED,
        ) from exc
    if not plan.configured or not plan.room or not plan.token:
        raise ApiError(
            400,
            "This Lab isn't running its LiveKit SFU, so there is no room to launch the GPU into. "
            "Start it with `makermodslab --sfu --sfu-external-ip` and re-check the transport.",
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
        environment=want_environment,
    )
    log_handle, path = _open_log()
    try:
        proc = _popen(argv, child_env(plan, profile=want_profile))
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
        # Null, not "", when the CLI chose: the status says "we did not pick",
        # which is a different fact from "we picked the empty one".
        _profile = want_profile or None
        _environment = want_environment or None
        # Not known until the client prints its app page (a second or two in),
        # so a stop this early has nothing to stop over the API — which is
        # exactly what `_settle_app` says out loud when it happens.
        _app_id = None
        # The transport tuple AS LAUNCHED, so the panel's drift warning can
        # compare against what this SERVER started rather than what some tab
        # remembers starting.
        _task = task
        _horizon = horizon
        _fps = fps
        _video_codec = video_codec
        _s_min = s_min
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


def stop_for_shutdown() -> bool:
    """Stop the GPU on the way down — SYNCHRONOUSLY, and SIGINT first.

    The shutdown twin of `stop()`, and the reason it exists is the ordering:
    every other stop path can return while the kill runs on its own thread,
    because this process will still be here to finish it. On the way out it
    will not be, and a `modal run` client that outlives its parent by a
    millisecond is a client that never gets its SIGINT — which is precisely
    how a `--reload` save left an A100 running.

    Returns whether there was anything to stop. Never raises (it is called from
    a shutdown handler) and never returns an ApiError: "nothing running" is not
    an exceptional condition here, it is the normal one.

    Bounded: `_SIGINT_GRACE_S` for the client's own teardown, then at most two
    `_TERMINATE_TIMEOUT_S` escalations, then at most `_APP_STOP_TIMEOUT_S` for
    the API fallback — and the typical cost is the ~2 s the client takes to
    disconnect. The drain deadline is not waited for at all: the pump is a
    daemon thread and this process is leaving.
    """
    global _state, _stop_outcome, _drain_deadline, _message, _hint
    with _state_lock:
        proc = _proc
        if proc is None:
            return False
        _stop_outcome = STATE_IDLE
        _state = STATE_STOPPING
        _drain_deadline = None
        _message = None
        _hint = None
    logger.info("Stopping the GPU policy server before shutdown")
    try:
        _terminate_and_watch(proc)
    except Exception:
        logger.exception("Failed to stop the GPU policy server during shutdown")
    return True


# --- the orphan reaper -------------------------------------------------------
# The other half of "a Modal app outlives its client": when this process is not
# the one that lost it. A record left on disk with no client here means the
# last run's app may still be up, and only an explicit `modal app stop` can end
# it now.

# Once per PROCESS. Not a lock-free bool by accident — it is read and written
# under `_state_lock` with the decision it guards.
_reaped = False


def _reap_orphan_app() -> None:
    """Stop an app the last process left behind, and say so. Best-effort.

    Runs on its own thread (see :func:`reap_orphan_app_async`) so a Modal API
    call can never sit inside a request or a boot. Three ways it declines, all
    silent: no record on disk, a launch already owns the slot (the record is
    THAT launch's), or it has already run in this process.

    A failure is reported the same way a success is — in the idle status's
    `message` — because "an app may still be billing and I could not stop it"
    is exactly the sentence an operator needs to see. It never blocks or
    refuses a launch: the launcher's state is untouched except for that line.
    """
    global _message
    record = read_app_record()
    if record is None:
        return
    with _state_lock:
        if _proc is not None:
            # A live launch owns the slot, so this record describes IT.
            return
    app_id = record["app_id"]
    when = record["started_at"]
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when)) if when else "an earlier run"
    logger.info("Reaping a GPU app left behind by an earlier run: %s", app_id)
    stopped, detail = stop_app(app_id, record["profile"])
    if stopped:
        _clear_app_record(app_id)
        note = f"Stopped an orphaned GPU app {app_id} from {stamp} — it outlived the Lab that started it."
    else:
        note = (
            f"A GPU app from {stamp} ({app_id}) may still be running, and stopping it from here "
            f"failed: {detail}. Run `modal app stop {app_id}` in a terminal."
        )
        logger.error(note)
    with _state_lock:
        # Only onto an idle panel: a launch that started while we were talking
        # to Modal owns the message line, and its own prose matters more.
        if _state == STATE_IDLE:
            _message = note


def reap_orphan_app_async() -> None:
    """Kick :func:`_reap_orphan_app` once per process, off the caller's thread.

    Called from the server's startup event rather than at module import or on
    the first status poll: import-time subprocesses would fire in tests, SDK
    clients and `makermodslab --stop`, and a GET must not carry a Modal API
    call. Startup is also early enough that the message is already in the very
    first status the panel reads, which is the point of surfacing it.
    """
    global _reaped
    with _state_lock:
        if _reaped:
            return
        _reaped = True
    threading.Thread(target=_reap_orphan_app, name="modal-launcher-reap", daemon=True).start()


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
        # The Modal target AS LAUNCHED, null when the CLI resolved it. The
        # panel's ready line says which workspace is billing, which is the
        # reason the choice exists.
        "profile": _profile,
        "environment": _environment,
        # The Modal app this run created, null until its client prints the app
        # page (and while idle). It is what `modal app stop` takes, so an
        # operator whose Lab died mid-run can stop the app by hand from it.
        "app_id": _app_id,
        # The transport tuple AS LAUNCHED, null while idle. The panel compares
        # the form against THESE — a server-side record survives a page reload
        # and describes a GPU another tab started, neither of which a tab's own
        # memory of its last start can do.
        "task": _task,
        "horizon": _horizon,
        "fps": _fps,
        "video_codec": _video_codec,
        # Echoed for both engines, as launched. It only reaches the wire for
        # `rtc` (`build_argv` omits the flag otherwise), so a `sync` run's
        # value is what a switch to rtc WOULD have used — the panel compares it
        # only when the engine is rtc, for the same reason.
        "s_min": _s_min,
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
