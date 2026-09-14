"""Layer 1: one HTTP request in, parsed JSON or a typed error out.

Ports to other languages re-implement this file and errors.py idiomatically;
everything above them stays declarative. ``http_client`` injection is how the
test suite runs the SDK against the real FastAPI app in-process
(fastapi.testclient.TestClient IS an httpx.Client) — zero network, zero mocks
that can drift from the server.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .errors import ConnectionFailedError, build_api_error

DEFAULT_TIMEOUT = 30.0


class Transport:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
        on_first_request: Callable[[], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._on_first_request = on_first_request

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        action: str | None = None,
    ) -> Any:
        """Perform a request; return the parsed JSON body (None on 204/empty).

        ``action`` is the human-readable label errors lead with ("Start
        teleoperation session") — the first thing the reader of a failure
        sees, so callers should always pass one.
        """
        if self._on_first_request is not None:
            # Cleared BEFORE running so the hook's own SDK calls don't recurse.
            hook, self._on_first_request = self._on_first_request, None
            hook()
        label = action or f"{method} {path}"
        try:
            response = self._client.request(method, path, json=json, params=params)
        except httpx.TransportError as exc:
            raise ConnectionFailedError(
                f"{label} failed: could not reach the MakerMods Lab server at {self.base_url} "
                f"({type(exc).__name__}: {exc})\nNext step: check that the server is running and the "
                f"URL is right — the app serves on http://<host>:8000 by default.",
                base_url=self.base_url,
            ) from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except Exception:
                body = None
            raise build_api_error(response.status_code, body, label)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def close(self) -> None:
        self._client.close()
