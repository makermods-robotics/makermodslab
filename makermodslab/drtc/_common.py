"""Shared helpers for remote inference: token minting + log formatting.

Ported from the `livekit-drtc` repo. Credential loading lives in `_env` (so it
is importable without LiveKit) and is re-exported here to keep the call sites
`from ._common import load_env, mint_token, ...` unchanged.
"""

from __future__ import annotations

import datetime
import os

from ._env import (  # noqa: F401  (re-exported)
    _EXTRA_HINT,
    env_float,
    env_int,
    env_str,
    load_env,
    required_env,
)

try:
    from livekit import api
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"livekit-api is missing; {_EXTRA_HINT}") from exc


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set (see docs/drtc/livekit.env.example)"
        )
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


def fmt_us(value) -> str:
    """Format µs as `NNNus` or `N.NNms`, or `-` for None / 0."""
    if value is None or value == 0:
        return "-"
    if value < 1000:
        return f"{value}us"
    return f"{value / 1000:.2f}ms"
