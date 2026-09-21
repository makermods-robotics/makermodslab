"""The remote / sfu / recording namespaces and the remote session kinds —
MockTransport for everything that would need an SFU, Portal, or hardware;
e2e only for the read-only status routes."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import ApiError


def test_remote_status_and_controls_paths():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v1/hosting":
            return httpx.Response(
                200,
                json={
                    "hosting_active": True,
                    "hosting": {"room": "node-ab12", "phase": "parked"},
                    "releasing": False,
                    "last_cleanup_error": None,
                    "outcome": None,
                    "error": None,
                    "hint": None,
                    "message": "hosting",
                },
            )
        if request.url.path == "/api/v1/station":
            return httpx.Response(
                200,
                json={
                    "station_mode": True,
                    "robot": "bench",
                    "hostable": ["bench"],
                    "hosting_active": True,
                    "phase": "parked",
                },
            )
        if request.url.path == "/api/v1/station/robot":
            assert json.loads(request.read()) == {"robot": "bench2"}
            return httpx.Response(
                200,
                json={
                    "station_mode": True,
                    "robot": "bench2",
                    "hostable": ["bench", "bench2"],
                    "hosting_active": False,
                    "phase": None,
                },
            )
        return httpx.Response(200, json={"success": True, "message": "ok"})

    with mock_client(handler) as client:
        hosting = client.remote.hosting_status()
        assert hosting.hosting_active and hosting.hosting["room"] == "node-ab12"
        assert client.remote.station().robot == "bench"
        assert client.remote.set_station_robot("bench2").robot == "bench2"
        assert client.remote.home().success is True
        assert client.remote.engage().success is True
        # camera_url is a URL, not a request — MJPEG streams are unbounded.
        url = client.remote.camera_url("front cam")
        assert url == "http://mock/api/v1/remote-teleoperation/camera/front%20cam"
    paths = [path for _, path in calls]
    assert "/api/v1/remote-teleoperation/home" in paths
    assert "/api/v1/remote-teleoperation/engage" in paths
    assert not any("camera" in path for path in paths)


def test_sfu_token_sends_only_set_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "url": "ws://mock:7880",
                "token": "jwt",
                "room": "node-ab12",
                "identity": "op-1",
                "role": "operator",
                "expires_at": 1758400000,
            },
        )

    with mock_client(handler) as client:
        token = client.sfu.token(role="operator", room="node-ab12")
        assert seen["body"] == {"role": "operator", "room": "node-ab12"}
        assert token.url.startswith("ws://")
        client.sfu.token()
        assert seen["body"] == {}


def test_recording_namespace():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "message": "task set"})

    with mock_client(handler) as client:
        assert client.recording.set_episode_task("pick the red cube").success is True
        assert seen["path"] == "/api/v1/recording-episode-task"
        assert seen["body"] == {"task": "pick the red cube"}
        url = client.recording.camera_preview_url("front")
        assert url == "http://mock/api/v1/recording-preview/front"


def test_remote_kind_sugars_send_kind_and_options():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stop"):  # body-less; not under test
            return httpx.Response(
                200,
                json={
                    "session": {
                        "id": "s1",
                        "kind": "hosting",
                        "robot": "bench",
                        "owner": None,
                        "started_at": 1.0,
                        "revision": 2,
                        "phase": None,
                        "lease": None,
                    },
                    "result": {},
                },
            )
        body = json.loads(request.read())
        bodies.append(body)
        return httpx.Response(
            201,
            json={
                "session": {
                    "id": "s1",
                    "kind": body["kind"],
                    "robot": body["robot"],
                    "owner": body.get("owner"),
                    "started_at": 1.0,
                    "revision": 1,
                    "phase": None,
                    "lease": None,
                },
                "warnings": None,
            },
        )

    with mock_client(handler) as client:
        client.sessions.host("bench", video_codec="MJPEG").stop()
        client.sessions.remote_teleoperate("laptop-leader", station="cd" * 16).stop()
        client.sessions.remote_infer("bench", policy_ref="me/act", engine="rtc", latency_k=2.0).stop()
    hosting, teleop, infer = bodies  # stops are filtered out by the handler
    assert hosting["kind"] == "hosting" and hosting["options"] == {"video_codec": "MJPEG"}
    assert teleop["kind"] == "remote_teleoperation"
    assert teleop["options"] == {"station": "cd" * 16}
    assert infer["kind"] == "remote_inference"
    assert infer["options"] == {"policy_ref": "me/act", "engine": "rtc", "latency_k": 2.0}


def test_gpu_start_body_passthrough():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={"started": True, "message": "launching", "gpu": {"state": "starting", "elapsed_s": 0.0}},
        )

    with mock_client(handler) as client:
        launch = client.sessions.gpu_start(policy_hub_id="me/act-pick", gpu="A10G")
    assert seen["path"] == "/api/v1/remote-inference/gpu/start"
    assert seen["body"] == {"policy_hub_id": "me/act-pick", "gpu": "A10G"}
    assert launch.gpu.state == "starting"


def test_remote_status_routes_end_to_end(sdk_client):
    assert sdk_client.remote.station().station_mode is False
    assert sdk_client.remote.teleoperation_status().remote_teleoperation_active is False
    assert sdk_client.sessions.remote_inference_status().remote_inference_active is False
    transport = sdk_client.sessions.remote_inference_transport()
    assert transport.configured is False  # no --sfu in the test app
    assert isinstance(sdk_client.system.arms().arms[0]["id"], str)


def test_sfu_token_refuses_coded_without_sfu_end_to_end(sdk_client):
    with pytest.raises(ApiError) as excinfo:
        sdk_client.sfu.token()
    assert excinfo.value.code == "sfu.disabled"
    assert "Next step" in str(excinfo.value)
