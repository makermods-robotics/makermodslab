"""Shared helpers for remote inference: LiveKit env loading + token minting.

Ported from the `livekit-drtc` repo. The one behavioural change is WHERE the
credentials come from: the standalone repo searched the script's own directory
and its parent, which inside an installed wheel would mean site-packages. Here
the search is (lowest precedence first):

  1. `DRTC_ENV_PATH` — `~/.cache/huggingface/lerobot/livekit.env`, beside the
     rest of the Lab's persistent state, so a wheel install and a source
     checkout read the same saved credentials.
  2. `.env` in the current directory.
  3. `.env.local` in the current directory — highest precedence, and the file
     the local-SFU scripts generate to point the robot at `ws://127.0.0.1:7880`.
     Delete it to fall back to whatever `.env` / the saved creds say.

Process environment always wins over all three (`override=False` on the first
two mirrors dotenv's own rule); `.env.local` is the deliberate exception, as
in the source repo.
"""

from __future__ import annotations

import datetime
import os
import pathlib

from ..utils.config import DRTC_ENV_PATH

# The standalone repo printed a hint and called SystemExit here. A module in
# the Lab must not kill the interpreter on import — a caller that imports this
# package defensively has to be able to catch the failure — so a missing extra
# raises ImportError naming the extra that supplies it.
_EXTRA_HINT = "remote inference needs the optional 'drtc' extra: uv pip install -e '.[drtc]'"

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"python-dotenv is missing; {_EXTRA_HINT}") from exc

try:
    from livekit import api
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"livekit-api is missing; {_EXTRA_HINT}") from exc


def load_env(search_from: pathlib.Path | None = None) -> None:
    """Load LiveKit credentials; see the module docstring for precedence."""
    saved = pathlib.Path(DRTC_ENV_PATH)
    if saved.exists():
        load_dotenv(saved, override=False)

    start = search_from or pathlib.Path.cwd()
    env = start / ".env"
    if env.exists():
        load_dotenv(env, override=False)
    env_local = start / ".env.local"
    if env_local.exists():
        load_dotenv(env_local, override=True)


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set (see .env.example)")
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        # `Robot` and `Operator` self-set the `lk.portal.role` attribute on
        # connect so participants can discover one another. The grant must
        # permit it, otherwise connect fails with "does not have permission
        # to update own metadata".
        can_update_own_metadata=True,
    )
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(datetime.timedelta(hours=ttl_hours))
        .to_jwt()
    )


def required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return v


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def fmt_us(value) -> str:
    """Format µs as `NNNus` or `N.NNms`, or `-` for None / 0."""
    if value is None or value == 0:
        return "-"
    if value < 1000:
        return f"{value}us"
    return f"{value / 1000:.2f}ms"
