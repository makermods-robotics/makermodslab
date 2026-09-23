# Copyright 2026 MakerMods. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Remote-inference flow state transitions through MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.flows_remote_inference import (
    RemoteInferenceFlowError,
    RemoteInferenceStartupTimeout,
)

LAUNCH_ID = "launch-001"
SESSION_ID = "remote-001"


def transport(*, configured: bool = True, error_code: str | None = "transport.no_policy") -> dict:
    return {
        "extra_installed": True,
        "configured": configured,
        "url": "ws://127.0.0.1:7880" if configured else "",
        "room": "lab-room" if configured else "",
        "source": "sfu" if configured else "none",
        "sfu_enabled": configured,
        "sfu_url": "ws://127.0.0.1:7880" if configured else None,
        "sfu_modal_url": "ws://100.64.0.1:7880" if configured else None,
        "sfu_external_ip": True,
        "sfu_install_hint": None,
        "policy_token": "token" if configured else None,
        "endpoint_reachable": True if configured else None,
        "operator_present": False if configured else None,
        "error_code": error_code,
        "message": "No policy peer yet" if configured else "Start with --sfu",
    }


def gpu(state: str, *, launch_id: str | None = LAUNCH_ID, phase: str | None = None) -> dict:
    return {
        "launch_id": launch_id,
        "state": state,
        "phase": phase,
        "engine": "rtc",
        "policy_hub_id": "maker/act-pick",
        "room": "lab-room",
        "gpu": "A10G",
        "elapsed_s": 1.0,
        "message": "warming" if state == "starting" else None,
        "code": "gpu.launch_failed" if state == "failed" else None,
        "hint": "Inspect the GPU log" if state == "failed" else None,
    }


def session() -> dict:
    return {
        "id": SESSION_ID,
        "kind": "remote_inference",
        "robot": "bench",
        "owner": "sdk:test",
        "started_at": 100.0,
        "revision": 1,
        "phase": "running",
        "lease": None,
    }


class Script:
    def __init__(self, routes):
        self.routes = {key: list(value) for key, value in routes.items()}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        assert key in self.routes and self.routes[key], f"unscripted {key}"
        code, body = self.routes[key].pop(0)
        return httpx.Response(code, json=body)


def success_routes(*statuses: dict) -> dict:
    return {
        ("GET", "/api/v1/remote-inference/transport"): [(200, transport())],
        ("POST", "/api/v1/remote-inference/gpu/start"): [
            (
                200,
                {
                    "started": True,
                    "launch_id": LAUNCH_ID,
                    "message": "started",
                    "gpu": gpu("starting"),
                },
            )
        ],
        ("GET", "/api/v1/remote-inference/gpu"): [(200, item) for item in statuses],
        ("POST", "/api/v1/sessions"): [(201, {"session": session(), "warnings": ["camera warning"]})],
        ("POST", f"/api/v1/sessions/{SESSION_ID}/stop"): [
            (200, {"session": session(), "result": {"success": True}})
        ],
        ("POST", "/api/v1/remote-inference/gpu/stop"): [(200, gpu("stopping"))],
    }


def paths(script: Script) -> list[str]:
    return [request.url.path for request in script.requests]


def test_context_launches_matching_halves_and_cleans_up_session_before_gpu():
    script = Script(success_routes(gpu("starting", phase="loading"), gpu("ready", phase="connected")))
    slept: list[float] = []

    with (
        mock_client(script) as client,
        client.flows.remote_inference(
            "bench",
            policy_ref="maker/act-pick@checkpoints/002000",
            policy_hub_id="maker/act-pick",
            gpu="A10G",
            engine="rtc",
            task="pick the cube",
            horizon=24,
            fps=20,
            video_codec="MJPEG",
            s_min=6,
            duration_s=120,
            startup_timeout=10,
            poll_interval=2,
            sleep_fn=slept.append,
        ) as run,
    ):
        assert run.launch_id == LAUNCH_ID
        assert run.session_id == SESSION_ID
        assert run.ready.phase == "connected"
        assert run.start_warnings == ("camera warning",)

    assert slept == [2]
    assert paths(script) == [
        "/api/v1/remote-inference/transport",
        "/api/v1/remote-inference/gpu/start",
        "/api/v1/remote-inference/gpu",
        "/api/v1/remote-inference/gpu",
        "/api/v1/sessions",
        f"/api/v1/sessions/{SESSION_ID}/stop",
        "/api/v1/remote-inference/gpu/stop",
    ]
    gpu_start = json.loads(script.requests[1].content)
    session_start = json.loads(script.requests[4].content)
    for key, value in {
        "policy_hub_id": "maker/act-pick",
        "engine": "rtc",
        "task": "pick the cube",
        "horizon": 24,
        "fps": 20,
        "video_codec": "MJPEG",
        "s_min": 6,
    }.items():
        assert gpu_start[key] == value
        assert session_start["options"][key] == value
    assert session_start["options"]["policy_ref"] == "maker/act-pick@checkpoints/002000"
    assert session_start["options"]["duration_s"] == 120
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID


def test_unconfigured_preflight_starts_nothing():
    script = Script(
        {
            ("GET", "/api/v1/remote-inference/transport"): [
                (200, transport(configured=False, error_code="transport.not_configured"))
            ]
        }
    )
    with (
        mock_client(script) as client,
        pytest.raises(RemoteInferenceFlowError) as error,
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        pass
    assert error.value.launch_id is None and error.value.session_id is None
    assert error.value.last_state == "transport.not_configured"
    assert "remote_inference_transport" in error.value.recovery_call
    assert paths(script) == ["/api/v1/remote-inference/transport"]


def test_timeout_stops_only_the_launch_and_exposes_recovery_state():
    script = Script(success_routes(gpu("starting", phase="loading")))
    with (
        mock_client(script) as client,
        pytest.raises(RemoteInferenceStartupTimeout) as error,
        client.flows.remote_inference(
            "bench",
            policy_ref="maker/act-pick",
            startup_timeout=0,
            sleep_fn=lambda _: None,
            clock=lambda: 0.0,
        ),
    ):
        pass
    assert error.value.launch_id == LAUNCH_ID
    assert error.value.session_id is None
    assert error.value.last_state == "starting"
    assert error.value.last_phase == "loading"
    assert error.value.waited == 0
    assert "gpu_status" in error.value.recovery_call
    assert paths(script)[-1] == "/api/v1/remote-inference/gpu/stop"
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID
    assert "/api/v1/sessions" not in paths(script)


def test_replaced_gpu_is_never_used_and_cleanup_is_conditional():
    routes = success_routes(gpu("ready", launch_id="launch-002", phase="connected"))
    routes[("POST", "/api/v1/remote-inference/gpu/stop")] = [
        (
            409,
            {
                "detail": "launch changed",
                "code": "gpu.launch_replaced",
                "details": {"expected_launch_id": LAUNCH_ID, "current_launch_id": "launch-002"},
            },
        )
    ]
    script = Script(routes)
    with (
        mock_client(script) as client,
        pytest.raises(RemoteInferenceFlowError, match="launch-002") as error,
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        pass
    assert error.value.launch_id == LAUNCH_ID
    assert error.value.last_state == "ready"
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID
    assert "/api/v1/sessions" not in paths(script)


def test_failed_gpu_is_stopped_and_reports_server_diagnosis():
    script = Script(success_routes(gpu("failed", phase="loading")))
    with (
        mock_client(script) as client,
        pytest.raises(RemoteInferenceFlowError, match="Inspect the GPU log") as error,
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        pass
    assert error.value.last_state == "failed"
    assert error.value.last_phase == "loading"
    assert paths(script)[-1] == "/api/v1/remote-inference/gpu/stop"


def test_session_start_failure_stops_gpu_without_an_unscoped_stop():
    routes = success_routes(gpu("ready", phase="connected"))
    routes[("POST", "/api/v1/sessions")] = [
        (409, {"detail": "busy", "code": "session.held", "details": {"holder": {"kind": "recording"}}})
    ]
    script = Script(routes)
    with (
        mock_client(script) as client,
        pytest.raises(RemoteInferenceFlowError) as error,
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        pass
    assert error.value.launch_id == LAUNCH_ID
    assert error.value.session_id is None
    assert error.value.last_state == "ready"
    assert paths(script)[-1] == "/api/v1/remote-inference/gpu/stop"
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID


def test_body_error_is_preserved_after_ordered_cleanup():
    script = Script(success_routes(gpu("ready", phase="connected")))
    with (
        mock_client(script) as client,
        pytest.raises(RuntimeError, match="operator stopped"),
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        raise RuntimeError("operator stopped")
    assert paths(script)[-2:] == [
        f"/api/v1/sessions/{SESSION_ID}/stop",
        "/api/v1/remote-inference/gpu/stop",
    ]


def test_body_error_is_not_masked_when_replacement_refuses_gpu_cleanup():
    routes = success_routes(gpu("ready", phase="connected"))
    routes[("POST", "/api/v1/remote-inference/gpu/stop")] = [
        (409, {"detail": "replacement is current", "code": "gpu.launch_replaced"})
    ]
    script = Script(routes)
    with (
        mock_client(script) as client,
        pytest.raises(RuntimeError, match="operator stopped"),
        client.flows.remote_inference("bench", policy_ref="maker/act-pick") as flow,
    ):
        raise RuntimeError("operator stopped")
    assert flow is not None and flow.cleanup_error is not None
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID


def test_gpu_cleanup_still_runs_when_session_stop_fails():
    routes = success_routes(gpu("ready", phase="connected"))
    routes[("POST", f"/api/v1/sessions/{SESSION_ID}/stop")] = [
        (500, {"detail": "stop failed", "code": "internal.unexpected"})
    ]
    script = Script(routes)
    with (
        mock_client(script) as client,
        pytest.raises(MakerModsError, match="stop failed"),
        client.flows.remote_inference("bench", policy_ref="maker/act-pick"),
    ):
        pass
    assert paths(script)[-2:] == [
        f"/api/v1/sessions/{SESSION_ID}/stop",
        "/api/v1/remote-inference/gpu/stop",
    ]
    assert script.requests[-1].url.params["launch_id"] == LAUNCH_ID


def test_constructing_context_starts_nothing():
    def no_requests(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request}")

    with mock_client(no_requests) as client:
        run = client.flows.remote_inference("bench", policy_ref="maker/act-pick")
    assert run.launch_id is None and run.session_id is None


@pytest.mark.parametrize(
    ("startup_timeout", "poll_interval"),
    [(-1, 1), (1, 0)],
)
def test_invalid_timing_starts_nothing(startup_timeout: float, poll_interval: float):
    def no_requests(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request}")

    with mock_client(no_requests) as client, pytest.raises(ValueError):
        client.flows.remote_inference(
            "bench",
            policy_ref="maker/act-pick",
            startup_timeout=startup_timeout,
            poll_interval=poll_interval,
        )
