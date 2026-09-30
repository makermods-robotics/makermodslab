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
    records = sdk_client.robots.list()
    assert isinstance(records, list)
    # A plain list of typed records: the obvious loop must work, because an
    # agent writes it without reading anything.
    assert all(isinstance(record, Robot) for record in records)


def test_get_missing_end_to_end(sdk_client):
    with pytest.raises(ApiError) as excinfo:
        sdk_client.robots.get("sdk-test-no-such-robot")
    assert excinfo.value.status == 404


def test_record_exposes_the_servers_readiness_flags():
    """Readiness is what setup turns on; the server computes it on every read
    (server.py _record_with_clean) so the model must declare it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "success",
                "robot": {
                    **RECORD,
                    "arm_type": "metal",
                    "leader_kind": "star",
                    "arms": "both",
                    "is_clean": False,
                    "follower_ready": True,
                    "leader_ready": False,
                    "arm_available": True,
                },
            },
        )

    with mock_client(handler) as client:
        robot = client.robots.get("bench")
    assert robot.is_clean is False
    assert robot.follower_ready is True
    assert robot.leader_ready is False
    assert robot.arm_available is True
    assert robot.arm_type == "metal" and robot.leader_kind == "star" and robot.arms == "both"


def test_readiness_is_none_when_an_older_server_omits_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "robot": RECORD})

    with mock_client(handler) as client:
        robot = client.robots.get("bench")
    assert robot.is_clean is None and robot.arm_available is None


def test_readiness_flags_survive_end_to_end(sdk_client):
    """The real app computes these; if the key names ever drift, this fails."""
    for record in sdk_client.robots.list():
        assert record.is_clean is not None
        assert record.arm_available is not None
        assert record.name
