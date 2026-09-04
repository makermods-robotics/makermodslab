"""LiveKit CLOUD credential loading for remote inference — no LiveKit dependency.

Split out of `_common` so the precedence rules can be unit-tested without the
`drtc` extra installed (`_common` imports `livekit.api` for token minting).

This is the FALLBACK path. When the Lab runs its own SFU (`makermodslab --sfu`,
see `makermodslab/sfu.py`) the session resolves url, room and token in-process
and passes them to the child on the command line, so nothing here is consulted
at all. What is left is the LiveKit Cloud case, and it has exactly two rungs:

  1. `DRTC_ENV_PATH` (`~/.cache/huggingface/lerobot/livekit.env`) — the saved
     credentials, beside the rest of the Lab's persistent state.
  2. The process environment, which wins (dotenv's own `override=False` rule).

Three rungs were RETIRED in S3.6 along with the `tools/drtc/local_sfu*.sh`
scripts they existed for: a cwd `.env`, a cwd `.env.local`, and
`livekit.local.env`. All three were `override=True` or cwd-relative, and both
properties were bugs in a long-lived server — a cwd-relative source makes the
answer depend on where the Lab was started (two "connection refused" false
starts on 2026-09-02), and an override the process cannot un-set is a transport
that survives the deletion of the file naming it. The Lab-owned SFU replaces
what they were for.

Two entry points, ONE precedence implementation:

  * :func:`read_env` RESOLVES the sources over a copy of the process
    environment and hands the result back. Nothing is mutated, so it is safe to
    call repeatedly from the long-lived FastAPI process.
  * :func:`load_env` is `read_env` plus a write into `os.environ`, for the CLI
    entrypoints whose downstream code (`_common.mint_token`, `policy.py`) reads
    credentials from the environment.

With no override rung left the two can no longer disagree about a deleted file,
but the split stays: `read_env` is still the only correct call from a server
that must re-resolve on every request, and `load_env` is still what a child
process needs before `required_env`.
"""

from __future__ import annotations

import os
import pathlib

from ..utils.config import DRTC_ENV_PATH

_EXTRA_HINT = "remote inference needs the optional 'drtc' extra: uv pip install -e '.[remote]'"

try:
    from dotenv import dotenv_values
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"python-dotenv is missing; {_EXTRA_HINT}") from exc


def _apply(env: dict[str, str], path: pathlib.Path, override: bool) -> None:
    """Merge one dotenv file into `env` with dotenv's own `override` semantics.

    Mirrors `DotEnv.set_as_environment_variables`: a `None` value (a bare `KEY`
    line with no `=`) is skipped rather than written as an empty string, and
    with `override=False` an existing key is left alone.

    One deliberate narrowing versus `load_dotenv`: `${VAR}` interpolation
    inside the file resolves against the PROCESS environment only. It is a
    four-line credential file; nothing in the shipped `livekit.env.example`
    interpolates."""
    for key, value in dotenv_values(path).items():
        if value is None:
            continue
        if not override and key in env:
            continue
        env[key] = value


def read_env(search_from: pathlib.Path | None = None) -> dict[str, str]:
    """Resolve the saved credentials over the process environment, WITHOUT
    mutating it.

    Returns the environment a freshly-started child would see: a copy of
    `os.environ` with `livekit.env` layered UNDER it. Callers want
    `result["LIVEKIT_URL"]` and friends; the rest of the environment rides
    along because "what `load_env` would have produced" is the only definition
    that cannot drift from `load_env` itself.

    `search_from` is accepted and ignored. It named the cwd the two dotenv
    rungs were resolved against, and both are gone; the parameter stays so the
    two entrypoints' `load_env()` call sites and the tests keep one signature
    while the Cloud path is still reachable.
    """
    env = dict(os.environ)
    saved = pathlib.Path(DRTC_ENV_PATH)
    if saved.exists():
        _apply(env, saved, override=False)
    return env


def load_env(search_from: pathlib.Path | None = None) -> None:
    """Load LiveKit credentials into `os.environ`; precedence per the docstring.

    For the CLI entrypoints only — see the module docstring for why a
    long-lived process should call :func:`read_env` instead. A child the Lab
    spawned under its own SFU has its url, room and token pinned on the command
    line and does not depend on this having found anything."""
    for key, value in read_env(search_from).items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} must be set (see docs/drtc/livekit.env.example)")
    return v


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default
