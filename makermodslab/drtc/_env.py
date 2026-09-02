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
"""

from __future__ import annotations

import os
import pathlib

from ..utils.config import DRTC_ENV_PATH, DRTC_LOCAL_ENV_PATH

_EXTRA_HINT = "remote inference needs the optional 'drtc' extra: uv pip install -e '.[drtc]'"

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"python-dotenv is missing; {_EXTRA_HINT}") from exc


def load_env(search_from: pathlib.Path | None = None) -> None:
    """Load LiveKit credentials; see the module docstring for precedence."""
    saved = pathlib.Path(DRTC_ENV_PATH)
    if saved.exists():
        load_dotenv(saved, override=False)

    start = search_from or pathlib.Path.cwd()
    env = start / ".env"
    if env.exists():
        load_dotenv(env, override=False)

    local = pathlib.Path(DRTC_LOCAL_ENV_PATH)
    if local.exists():
        load_dotenv(local, override=True)
    env_local = start / ".env.local"
    if env_local.exists():
        load_dotenv(env_local, override=True)


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
