from __future__ import annotations

from makermodslab_sdk._transport import Transport


class Resource:
    """Base for the namespace clients (one per API tag); holds the shared Transport."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
