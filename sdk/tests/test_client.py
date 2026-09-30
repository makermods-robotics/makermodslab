"""Client-level behavior: the lazy compatibility handshake and lifecycle."""

from __future__ import annotations

import warnings

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import Client, CompatibilityWarning
from makermodslab_sdk.client import _parse_version


def health_body(version: str) -> dict:
    return {
        "status": "ok",
        "message": "ok",
        "version": version,
        "instance_id": "ab" * 16,
        "capabilities": {"serves_ui": True, "accepts_jobs": True},
    }


def counting_handler(version="0.1.0", health_status=200):
    calls = {"health": 0, "other": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/health":
            calls["health"] += 1
            if health_status != 200:
                return httpx.Response(health_status, json={"detail": "Not Found"})
            return httpx.Response(200, json=health_body(version))
        calls["other"] += 1
        return httpx.Response(200, json={"ok": True})

    return handler, calls


def test_old_server_warns_once_and_health_fetched_once():
    handler, calls = counting_handler(version="0.0.9")
    client = mock_client(handler, check_compatibility=True)
    with pytest.warns(CompatibilityWarning, match="older than the minimum"):
        client._transport.request("GET", "/api/v1/whatever")
    # Later requests: no second handshake, no second warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client._transport.request("GET", "/api/v1/whatever")
    assert calls["health"] == 1
    assert calls["other"] == 2
    client.close()


def test_current_server_passes_silently():
    handler, calls = counting_handler(version="0.1.0")
    client = mock_client(handler, check_compatibility=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client._transport.request("GET", "/api/v1/whatever")
    assert calls["health"] == 1
    client.close()


def test_missing_health_endpoint_warns_but_request_succeeds():
    handler, calls = counting_handler(health_status=404)
    client = mock_client(handler, check_compatibility=True)
    with pytest.warns(CompatibilityWarning, match="did not answer"):
        result = client._transport.request("GET", "/api/v1/whatever")
    assert result == {"ok": True}
    client.close()


def test_check_compatibility_false_skips_handshake():
    handler, calls = counting_handler()
    client = mock_client(handler, check_compatibility=False)
    client._transport.request("GET", "/api/v1/whatever")
    assert calls["health"] == 0
    client.close()


def test_context_manager_and_repr():
    handler, _ = counting_handler()
    with mock_client(handler) as client:
        assert isinstance(client, Client)
        assert "http://mock" in repr(client)
        assert "system" in repr(client)


def test_parse_version_tolerance():
    assert _parse_version("0.1.0") == (0, 1, 0)
    assert _parse_version("1.2.3rc1") == (1, 2, 3)
    assert _parse_version("weird") is None
