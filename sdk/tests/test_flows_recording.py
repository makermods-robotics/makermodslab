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
"""Recording flow state transitions through MockTransport; no hardware or sleeps."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import ServerTooOldError
from makermodslab_sdk.flows_recording import RecordingFlowError, RecordingFlowTimeout


def session(kind="recording"):
    return {
        "id": "rec-1",
        "kind": kind,
        "robot": "bench",
        "owner": "sdk:test",
        "started_at": 100.0,
        "revision": 1,
        "phase": "preparing",
        "lease": None,
    }


def terminal(*, phase="completed", reason=None):
    return {
        "session": None,
        "last_ended": {
            "id": "rec-1",
            "kind": "recording",
            "ended_at": 200.0,
            "phase": phase,
            "reason": reason,
        },
    }


def status(phase, saved, *, episode=None, ended=False, outcome=None, error=None):
    return {
        "recording_active": not ended,
        "current_phase": phase,
        "session_ended": ended,
        "saved_episodes": saved,
        "current_episode": episode,
        "outcome": outcome,
        "error": error,
        "dataset_repo_id": "me/dataset_20260922_120000",
        "discarded_empty": False if ended else None,
    }


class Script:
    def __init__(self, routes):
        self.routes = {key: list(value) for key, value in routes.items()}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        key = (request.method, request.url.path)
        assert key in self.routes and self.routes[key], f"unscripted {key}"
        code, body = self.routes[key].pop(0)
        return httpx.Response(code, json=body)


def routes(*statuses, end_phase="completed"):
    naming_reads = [
        (200, {"session": session(), "last_ended": None})
        for item in statuses
        if item["current_phase"] == "naming"
    ]
    return {
        ("POST", "/api/v1/sessions"): [(201, {"session": session(), "warnings": None})],
        ("GET", "/api/v1/sessions/rec-1/recording/status"): [(200, item) for item in statuses],
        ("GET", "/api/v1/sessions/current"): [*naming_reads, (200, terminal(phase=end_phase))],
        ("POST", "/api/v1/sessions/rec-1/stop"): [(404, {"detail": "gone", "code": "session.not_found"})],
    }


def test_episode_prompts_progress_and_completion():
    script_routes = routes(
        status("naming", 0, episode=1),
        status("recording", 0, episode=1),
        status("naming", 0, episode=2),
        status("completed", 2, ended=True, outcome="ok"),
    )
    script_routes[("POST", "/api/v1/sessions/rec-1/recording/episode-task")] = [
        (200, {"success": True, "message": "ok"}),
        (200, {"success": True, "message": "ok"}),
    ]
    script = Script(script_routes)
    pauses = []
    progress = []
    with mock_client(script) as client:
        result = client.flows.record_episodes(
            "bench",
            "me/dataset",
            task=["red cube", "blue cube"],
            timeout=10,
            poll_interval=2,
            sleep_fn=pauses.append,
            on_progress=lambda item: progress.append((item.current_phase, item.saved_episodes)),
        )
    assert result.session.id == "rec-1" and result.saved_episodes == 2
    assert result.dataset_repo_id == "me/dataset_20260922_120000"
    assert result.outcome == "ok"
    assert pauses == [2, 2, 2]
    assert progress == [("naming", 0), ("recording", 0), ("naming", 0), ("completed", 2)]
    start = next(r for r in script.requests if r.url.path == "/api/v1/sessions")
    assert json.loads(start.content)["options"] == {
        "dataset_repo_id": "me/dataset",
        "single_task": "red cube",
        "num_episodes": 2,
        "per_episode_task": True,
    }
    prompts = [
        json.loads(r.content)["task"]
        for r in script.requests
        if r.url.path.endswith("/recording/episode-task")
    ]
    assert prompts == ["red cube", "blue cube"]


def test_fixed_task_does_not_submit_prompts():
    script = Script(routes(status("recording", 0), status("completed", 3, ended=True, outcome="ok")))
    with mock_client(script) as client:
        result = client.flows.record_episodes(
            "bench",
            "me/dataset",
            task="pick",
            episodes=3,
            timeout=3,
            poll_interval=1,
            sleep_fn=lambda _: None,
        )
    assert result.saved_episodes == 3
    assert not any(r.url.path.endswith("/recording/episode-task") for r in script.requests)


def test_timeout_stops_owned_recording():
    script_routes = routes(status("recording", 1))
    script_routes[("POST", "/api/v1/sessions/rec-1/stop")] = [
        (200, {"session": session(), "result": {"success": True}}),
    ]
    script = Script(script_routes)
    with mock_client(script) as client, pytest.raises(RecordingFlowTimeout) as error:
        client.flows.record_episodes("bench", "me/dataset", task="pick", episodes=3, timeout=0)
    assert error.value.saved_episodes == 1
    assert any(r.url.path.endswith("/stop") for r in script.requests)


def test_zero_timeout_never_submits_episode_prompt():
    script_routes = routes(status("naming", 0, episode=1))
    script_routes[("POST", "/api/v1/sessions/rec-1/stop")] = [
        (200, {"session": session(), "result": {"success": True}}),
    ]
    script = Script(script_routes)
    with mock_client(script) as client, pytest.raises(RecordingFlowTimeout):
        client.flows.record_episodes("bench", "me/dataset", task=["pick"], timeout=0)
    assert not any(r.url.path.endswith("/recording/episode-task") for r in script.requests)


def test_timeout_counts_slow_status_request():
    script_routes = routes(status("recording", 1))
    script_routes[("POST", "/api/v1/sessions/rec-1/stop")] = [
        (200, {"session": session(), "result": {"success": True}}),
    ]
    script = Script(script_routes)
    now = [0.0]
    sleeps = []

    def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/sessions/rec-1/recording/status":
            now[0] += 6.0
        return script(request)

    with mock_client(slow_handler) as client, pytest.raises(RecordingFlowTimeout) as error:
        client.flows.record_episodes(
            "bench",
            "me/dataset",
            task="pick",
            episodes=3,
            timeout=5,
            poll_interval=2,
            sleep_fn=sleeps.append,
            clock=lambda: now[0],
        )
    assert error.value.saved_episodes == 1
    assert sleeps == []
    assert any(r.url.path.endswith("/stop") for r in script.requests)


def test_terminal_error_is_not_reported_as_success():
    script = Script(
        routes(status("error", 1, ended=True, outcome="failed", error="camera lost"), end_phase="error")
    )
    with mock_client(script) as client, pytest.raises(RecordingFlowError, match="camera lost"):
        client.flows.record_episodes("bench", "me/dataset", task="pick", episodes=3)


def test_completed_with_warning_preserves_server_detail():
    terminal_status = status(
        "completed", 1, ended=True, outcome="ran_with_warning", error="Torque release failed"
    )
    terminal_status["hint"] = "Check motor power"
    terminal_status["warning"] = "Identity was not verified"
    script_routes = routes(terminal_status)
    script_routes[("POST", "/api/v1/sessions")] = [
        (201, {"session": session(), "warnings": ["Camera mapping uncertain"]}),
    ]
    script = Script(script_routes)
    with mock_client(script) as client:
        result = client.flows.record_episodes("bench", "me/dataset", task="pick", episodes=1)
    assert result.status.error == "Torque release failed"
    assert result.status.hint == "Check motor power"
    assert result.status.warning == "Identity was not verified"
    assert result.start_warnings == ("Camera mapping uncertain",)


def test_normal_early_end_is_partial_result_error():
    script = Script(routes(status("completed", 1, ended=True, outcome="ok")))
    with mock_client(script) as client, pytest.raises(RecordingFlowError, match="1 of 3") as error:
        client.flows.record_episodes("bench", "me/dataset", task="pick", episodes=3)
    assert error.value.dataset_repo_id == "me/dataset_20260922_120000"
    assert error.value.saved_episodes == 1


def test_stale_session_cannot_submit_prompt():
    script_routes = routes(status("naming", 0, episode=1))
    script_routes[("GET", "/api/v1/sessions/current")] = [
        (200, {"session": session(kind="teleoperation") | {"id": "other"}, "last_ended": None}),
    ]
    script = Script(script_routes)
    with mock_client(script) as client, pytest.raises(RecordingFlowError, match="no longer owns"):
        client.flows.record_episodes("bench", "me/dataset", task=["pick"], timeout=1)
    assert not any(r.url.path.endswith("/recording/episode-task") for r in script.requests)


def test_server_predating_session_recording_routes_stops_the_session_and_says_so():
    """An older server starts the recording, then has no session-scoped status
    route: the flow must surface ServerTooOldError (not "session not found")
    and stop the session it started on the way out."""
    script_routes = routes()
    script_routes[("GET", "/api/v1/sessions/rec-1/recording/status")] = [(404, {"detail": "Not Found"})]
    script_routes[("POST", "/api/v1/sessions/rec-1/stop")] = [
        (200, {"session": session(), "result": {"success": True}}),
    ]
    script = Script(script_routes)
    with mock_client(script) as client, pytest.raises(ServerTooOldError) as error:
        client.flows.record_episodes("bench", "me/dataset", task="pick", episodes=3, timeout=10)
    assert "predates" in str(error.value)
    assert [r.url.path for r in script.requests if r.url.path.endswith("/stop")] == [
        "/api/v1/sessions/rec-1/stop"
    ]
