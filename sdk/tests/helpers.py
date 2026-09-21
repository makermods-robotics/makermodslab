from __future__ import annotations

from collections.abc import Callable

import httpx
from makermodslab_sdk import Client


def mock_client(
    handler: Callable[[httpx.Request], httpx.Response], *, check_compatibility: bool = False
) -> Client:
    """A Client wired to an httpx.MockTransport handler — no server, no network."""
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    return Client("http://mock", http_client=http, check_compatibility=check_compatibility)
