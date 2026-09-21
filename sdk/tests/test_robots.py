"""client.robots — the provisional record-CRUD namespace. Mutations are
MockTransport-only (records live in the user's real robot store); e2e is
confined to the read paths."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import ApiError
from makermodslab_sdk.resources.robots import Robot

RECORD = {
    "name": "bench",
    "mode": "single",
    "leader_port": "/dev/tty.usbmodemA",
    "follower_port": "/dev/tty.usbmodemB",
    "leader_config": "bench_leader",
    "follower_config": "bench_follower",
    "motor_power": 50,
}


def test_create_sends_create_flag_and_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"status": "success", "robot": RECORD})

    with mock_client(handler) as client:
        robot = client.robots.create("bench", mode="single", leader_port="/dev/tty.usbmodemA")
    assert "create=true" in seen["url"]
    assert seen["body"] == {"mode": "single", "leader_port": "/dev/tty.usbmodemA"}
    assert isinstance(robot, Robot)
    assert robot.mode == "single"


def test_update_patches_without_create_flag():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"status": "success", "robot": {**RECORD, "motor_power": 60}})

    with mock_client(handler) as client:
        robot = client.robots.update("bench", motor_power=60)
    assert "create" not in seen["url"]
    assert seen["body"] == {"motor_power": 60}
    assert robot is not None and robot.motor_power == 60


def test_update_of_missing_record_is_none_not_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "robot": None})

    with mock_client(handler) as client:
        assert client.robots.update("ghost", motor_power=60) is None


def test_legacy_conflict_body_surfaces_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"status": "error", "message": "Mode is fixed at creation — create a new robot."},
        )

    with mock_client(handler) as client, pytest.raises(ApiError) as excinfo:
        client.robots.update("bench", mode="bimanual")
    err = excinfo.value
    assert err.status == 409
    assert err.code is None  # legacy uncoded route, by design at this snapshot
    assert "Mode is fixed at creation" in str(err)


def test_rename_and_delete_paths():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        # raw_path, not url.path: httpx decodes the latter, hiding the quoting.
        calls.append((request.method, request.url.raw_path.decode()))
        if request.url.path.endswith("/rename"):
            return httpx.Response(200, json={"status": "success", "robot": RECORD})
        return httpx.Response(200, json={"status": "success"})

    with mock_client(handler) as client:
        client.robots.rename("old bench", "bench")
        client.robots.delete("bench")
    assert calls == [
        ("POST", "/api/v1/robots/old%20bench/rename"),
        ("DELETE", "/api/v1/robots/bench"),
    ]


def test_list_end_to_end(sdk_client):
    listing = sdk_client.robots.list()
    assert listing.status == "success"
    assert isinstance(listing.robots, list)


def test_get_missing_end_to_end(sdk_client):
    with pytest.raises(ApiError) as excinfo:
        sdk_client.robots.get("sdk-test-no-such-robot")
    assert excinfo.value.status == 404
