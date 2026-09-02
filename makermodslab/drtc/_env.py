"""LiveKit credential loading for remote inference — no LiveKit dependency.

Split out of `_common` so the precedence rules can be unit-tested without the
`drtc` extra installed (`_common` imports `livekit.api` for token minting).

Precedence, lowest first; a later source overrides an earlier one only where
marked `override`:

  1. `DRTC_ENV_PATH` (`~/.cache/huggingface/lerobot/livekit.env`) — the saved
     credentials, beside the rest of the Lab's persistent state.
  2. `.env` in the current directory.
  3. `DRTC_LOCAL_ENV_PATH` (`~/.cache/huggingface/lerobot/livekit.local.env`),
     override — written by `tools/drtc/local_sfu*.sh` while a local SFU runs.
  4. `.env.local` in the current directory, override — the source repo's
     convention, kept so an existing `livekit-drtc` checkout still works as a
     working directory.

Process environment wins over 1 and 2 (dotenv's own `override=False` rule) and
loses to 3 and 4, exactly as it lost to `.env.local` in the source repo.

Two entry points, ONE precedence implementation:

  * :func:`read_env` RESOLVES the four sources over a copy of the process
    environment and hands the result back. Nothing is mutated, so it is safe to
    call repeatedly from the long-lived FastAPI process.
  * :func:`load_env` is `read_env` plus a write into `os.environ`, for the CLI
    entrypoints whose downstream code (`_common.mint_token`, `policy.py`) reads
    credentials from the environment.

The split exists because `load_env`'s `override=True` on sources 3 and 4 is a
latent bug in a long-lived process: once the server has loaded a local-SFU
override, DELETING `livekit.local.env` can never un-set what it stamped into
`os.environ`, so the server keeps dialing a dead `ws://127.0.0.1:7880` until it
is restarted. A status endpoint that wants "what would a child resolve right
now?" must ask `read_env`, never `load_env`.
"""

from __future__ import annotations

import os
import pathlib

from ..utils.config import DRTC_ENV_PATH, DRTC_LOCAL_ENV_PATH

_EXTRA_HINT = "remote inference needs the optional 'drtc' extra: uv pip install -e '.[drtc]'"

try:
    from dotenv import dotenv_values
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"python-dotenv is missing; {_EXTRA_HINT}") from exc


def _apply(env: dict[str, str], path: pathlib.Path, override: bool) -> None:
    """Merge one dotenv file into `env` with dotenv's own `override` semantics.

    Mirrors `DotEnv.set_as_environment_variables`: a `None` value (a bare `KEY`
    line with no `=`) is skipped rather than written as an empty string, and
    with `override=False` an existing key is left alone.

    One deliberate narrowing versus chaining `load_dotenv` calls: `${VAR}`
    interpolation inside these files resolves against the PROCESS environment
    only, not against values a previous source in this chain contributed. These
    are four-line credential files; nothing in the shipped
    `livekit.env.example` or the local-SFU scripts interpolates."""
    for key, value in dotenv_values(path).items():
        if value is None:
            continue
        if not override and key in env:
            continue
        env[key] = value


def read_env(search_from: pathlib.Path | None = None) -> dict[str, str]:
    """Resolve the four sources over the process environment, WITHOUT mutating it.

    Returns the environment a freshly-started child would see: a copy of
    `os.environ` with the four sources layered on in the order the module
    docstring describes. Callers want `result["LIVEKIT_URL"]` and friends; the
    rest of the environment rides along because "what `load_env` would have
    produced" is the only definition that cannot drift from `load_env` itself.
    """
    env = dict(os.environ)
    start = search_from or pathlib.Path.cwd()

    saved = pathlib.Path(DRTC_ENV_PATH)
    if saved.exists():
        _apply(env, saved, override=False)
    dotenv = start / ".env"
    if dotenv.exists():
        _apply(env, dotenv, override=False)

    local = pathlib.Path(DRTC_LOCAL_ENV_PATH)
    if local.exists():
        _apply(env, local, override=True)
    env_local = start / ".env.local"
    if env_local.exists():
        _apply(env, env_local, override=True)

    return env


def load_env(search_from: pathlib.Path | None = None) -> None:
    """Load LiveKit credentials into `os.environ`; precedence per the docstring.

    For the CLI entrypoints only — see the module docstring for why a
    long-lived process should call :func:`read_env` instead."""
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
