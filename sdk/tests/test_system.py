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


# --- the wider system namespace ---------------------------------------------


def test_hf_login_sends_token():
    import httpx
    from helpers import mock_client

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "authenticated": True,
                "username": "someuser",
                "orgs": ["makermods"],
                "login_command": "hf auth login",
            },
        )

    with mock_client(handler) as client:
        result = client.system.hf_login(token="hf_secret")
    assert seen["path"] == "/api/v1/hf-auth/login"
    assert b"hf_secret" in seen["body"]
    assert result.authenticated is True
    assert result.orgs == ["makermods"]


def test_supply_voltage_param_only_when_given():
    import httpx
    from helpers import mock_client

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"success": True, "voltage": 12.1, "message": None})

    with mock_client(handler) as client:
        assert client.system.supply_voltage(port="/dev/ttyUSB0").voltage == 12.1
        assert "port=%2Fdev%2FttyUSB0" in seen["url"]
        client.system.supply_voltage()
        assert "port=" not in seen["url"]


def test_policy_extra_paths_carry_policy_type():
    import httpx
    from helpers import mock_client

    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/install"):
            return httpx.Response(200, json={"started": True, "message": "installing"})
        if request.url.path.endswith("/install-status"):
            return httpx.Response(200, json={"state": "running", "error": None, "logs": []})
        return httpx.Response(
            200,
            json={
                "policy_type": "pi0",
                "needs_extra": True,
                "available": False,
                "package": "lerobot[pi0]",
                "install_target": "lerobot[pi0]",
                "install_hint": "install it",
            },
        )

    with mock_client(handler) as client:
        assert client.system.policy_extra("pi0").needs_extra is True
        assert client.system.install_policy_extra("pi0").started is True
        assert client.system.policy_extra_install_status("pi0").state == "running"
    assert paths == [
        "/api/v1/system/policy-extra/pi0",
        "/api/v1/system/policy-extra/pi0/install",
        "/api/v1/system/policy-extra/pi0/install-status",
    ]


def test_policy_optimizer_defaults_end_to_end(sdk_client):
    defaults = sdk_client.system.policy_optimizer_defaults()
    assert isinstance(defaults.defaults, dict) and defaults.defaults
    assert isinstance(defaults.available, dict)


def test_training_extra_end_to_end(sdk_client):
    status = sdk_client.system.training_extra()
    assert isinstance(status.available, bool)
    assert isinstance(status.install_hint, str)


def test_robot_port_end_to_end(sdk_client):
    port = sdk_client.system.robot_port("follower")
    assert port.status == "success"
    assert isinstance(port.default_port, str)
