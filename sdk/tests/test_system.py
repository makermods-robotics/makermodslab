"""End-to-end against the real app: the tracer bullet's proof that every layer
lines up — SDK → httpx → FastAPI routing → handler → response model → SDK model."""

from __future__ import annotations

import pytest
from makermodslab_sdk import ApiError, NotFoundError
from makermodslab_sdk.resources.system import Health


def test_health_end_to_end(sdk_client):
    health = sdk_client.system.health()
    assert isinstance(health, Health)
    assert health.status == "ok"
    assert health.version
    assert len(health.instance_id) == 32
    int(health.instance_id, 16)  # 32-hex node identity
    assert isinstance(health.capabilities.serves_ui, bool)
    assert isinstance(health.capabilities.accepts_jobs, bool)


def test_coded_error_end_to_end(sdk_client):
    """A real server refusal arrives as the typed, coded, remediated exception.

    A session start naming a robot record that doesn't exist refuses with
    robot.not_found BEFORE any hardware is touched — the safe coded-error
    path to exercise against the real app.
    """
    with pytest.raises(NotFoundError) as excinfo:
        sdk_client._transport.request(
            "POST",
            "/api/v1/sessions",
            json={"kind": "teleoperation", "robot": "__sdk_test_missing__", "options": {}},
            action="Start teleoperation session",
        )
    err = excinfo.value
    assert err.status == 404
    assert err.code == "robot.not_found"
    assert err.suggestion is not None
    assert "Next step:" in str(err)


def test_validation_error_end_to_end(sdk_client):
    """A real FastAPI 422 arrives typed, with the field list flattened into
    readable prose. (At this snapshot the server doesn't yet stamp
    request.validation on 422 bodies — classification is by status, and the
    SDK accepts the code when newer servers add it.)"""
    from makermodslab_sdk import InvalidRequestError

    with pytest.raises(InvalidRequestError) as excinfo:
        sdk_client._transport.request(
            "POST", "/api/v1/sessions", json={"kind": "no-such-kind"}, action="Start session"
        )
    err = excinfo.value
    assert err.status == 422
    assert err.detail  # normalized, human-readable, not a repr of the list
    assert "[{" not in err.detail


def test_unknown_route_is_plain_api_error(sdk_client):
    with pytest.raises(ApiError) as excinfo:
        sdk_client._transport.request("GET", "/api/v1/definitely/not/a/route")
    assert excinfo.value.status == 404
