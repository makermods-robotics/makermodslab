"""The ``sfu`` namespace: the Lab's bundled LiveKit SFU (``makermodslab
--sfu``) — the transport remote teleoperation and remote inference ride.

``POST /api/v1/sfu/token`` is the ONLY signer: participants get short-lived,
role-scoped JWTs here instead of ever holding the SFU secret. Refuses with
``sfu.disabled`` when the server wasn't started with ``--sfu``, and with
``sfu.seat_taken`` for a second operator token while the single seat is held.
"""

from __future__ import annotations

from typing import Any

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class SfuToken(SdkModel):
    """A signed join token: connect to ``url``, room ``room``, as
    ``identity`` with ``role``; expires at ``expires_at`` (epoch seconds)."""

    url: str
    token: str
    room: str
    identity: str
    role: str
    expires_at: int


class SfuResource(Resource):
    """``client.sfu`` — join tokens for the bundled LiveKit SFU.

    Example:
        >>> t = client.sfu.token(role="operator", room="node-ab12")
        >>> t.url, t.expires_at
        ('ws://station:7880', 1758400000)
    """

    @operation("issue_sfu_token")
    def token(
        self,
        *,
        role: str | None = None,
        room: str | None = None,
        identity: str | None = None,
        ttl_seconds: int | None = None,
    ) -> SfuToken:
        """Issue a short-lived, role-scoped SFU join token (the server picks
        sane defaults for anything omitted; the URL is derived from the host
        you reached the API on).

        Example:
            >>> client.sfu.token(role="operator").token[:16]
            'eyJhbGciOiJIUzI1'
        """
        body: dict[str, Any] = {}
        if role is not None:
            body["role"] = role
        if room is not None:
            body["room"] = room
        if identity is not None:
            body["identity"] = identity
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        return SfuToken.model_validate(
            self._transport.request("POST", "/api/v1/sfu/token", json=body, action="Issue SFU token")
        )
