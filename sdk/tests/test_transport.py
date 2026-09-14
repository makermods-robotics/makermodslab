"""Layer-1 behavior: request/response passthrough and error decoding,
exercised through the public Client against httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import (
    ApiError,
    ConnectionFailedError,
    InvalidRequestError,
    NotFoundError,
    RobotBusyError,
    SessionHeldError,
)


def request_via(handler, method="GET", path="/api/v1/anything", **kwargs):
    client = mock_client(handler)
    try:
        return client._transport.request(method, path, **kwargs)
    finally:
        client.close()


def raise_via(handler, expected, **kwargs):
    with pytest.raises(expected) as excinfo:
        request_via(handler, **kwargs)
    return excinfo.value


def test_json_body_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hello": "world"})

    assert request_via(handler) == {"hello": "world"}


def test_204_and_empty_bodies_return_none():
    assert request_via(lambda r: httpx.Response(204)) is None
    assert request_via(lambda r: httpx.Response(200)) is None


def test_json_request_body_and_params_forwarded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    request_via(handler, method="POST", json={"a": 1}, params={"q": "x"})
    assert b'"a"' in seen["body"] and b"1" in seen["body"]
    assert "q=x" in seen["url"]


def test_coded_error_decodes_all_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"detail": "Hub is unreachable", "code": "hub.offline", "details": {"k": 1}}
        )

    err = raise_via(handler, ApiError, action="List datasets")
    assert err.status == 503
    assert err.code == "hub.offline"
    assert err.detail == "Hub is unreachable"
    assert err.details == {"k": 1}
    assert err.suggestion is not None
    text = str(err)
    assert "List datasets failed (503, hub.offline): Hub is unreachable" in text
    assert "Next step:" in text


def test_422_array_detail_normalized_and_typed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "fps"], "msg": "too big", "type": "x"},
                    {"loc": ["body", "tags"], "msg": "not a list", "type": "y"},
                ],
                "code": "request.validation",
            },
        )

    err = raise_via(handler, InvalidRequestError)
    assert err.detail == "too big; not a list"
    assert err.code == "request.validation"


def test_non_json_error_body_still_raises_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<html>boom</html>")

    err = raise_via(handler, ApiError)
    assert err.status == 500
    assert err.detail is None
    assert "500" in str(err)


def test_session_held_maps_to_typed_error_with_holder():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "detail": "robot is busy",
                "code": "session.held",
                "details": {"holder": {"kind": "teleoperation", "session_id": "abc123"}},
            },
        )

    err = raise_via(handler, SessionHeldError)
    assert err.holder == {"kind": "teleoperation", "session_id": "abc123"}
    assert "stop_current" in str(err)


def test_robot_busy_family_maps_with_discriminant():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "busy", "code": "robot.busy.teleoperation"})

    err = raise_via(handler, RobotBusyError)
    assert err.busy_with == "teleoperation"


def test_not_found_family_maps_to_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no such job", "code": "job.not_found"})

    err = raise_via(handler, NotFoundError)
    assert err.code == "job.not_found"


def test_unknown_code_stays_plain_api_error_without_suggestion():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "odd", "code": "hub.upload_failed"})

    err = raise_via(handler, ApiError)
    assert type(err) is ApiError
    assert err.suggestion is None
    assert "Next step" not in str(err)


def test_unreachable_server_raises_connection_failed_with_guidance():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    err = raise_via(handler, ConnectionFailedError)
    assert err.base_url == "http://mock"
    assert "http://mock" in str(err)
    assert "Next step:" in str(err)
