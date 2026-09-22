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
"""Training flow contract against MockTransport: no network or real sleeps."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk.flows_training import (
    PublishWaitTimeout,
    TrainingFlowError,
    TrainingFlows,
)
from makermodslab_sdk.resources.jobs import JobWaitTimeout

JOB_ID = "act_pick_001"


def job_body(state: str = "running", *, runner: str = "local", error_message: str | None = None) -> dict:
    return {
        "id": JOB_ID,
        "name": JOB_ID,
        "state": state,
        "config": {"dataset_repo_id": "maker/pick"},
        "output_dir": f"outputs/train/{JOB_ID}",
        "started_at": 1.0,
        "runner": runner,
        "error_message": error_message,
    }


def checkpoint_body(steps: tuple[int, ...] = (1000, 2000)) -> dict:
    return {
        "id": JOB_ID,
        "default_repo_id": "maker/act-pick",
        "hf_repo_id": None,
        "legacy_root_checkpoint": False,
        "hub_readable": True,
        "checkpoints": [
            {"step": step, "path": f"outputs/train/{JOB_ID}/checkpoints/{step}", "published": False}
            for step in steps
        ],
    }


def status(state: str, *, model_id: str | None = JOB_ID, error: str | None = None) -> dict:
    return {
        "state": state,
        "model_id": model_id,
        "repo_id": "maker/act-pick",
        "url": "https://huggingface.co/maker/act-pick" if state == "done" else None,
        "message": None,
        "error": error,
        "total": 1,
        "done": 1 if state == "done" else 0,
        "current_step": None,
        "done_steps": [2000] if state == "done" else [],
    }


def scripted_handler(
    *,
    jobs: tuple[dict, ...] = (job_body(), job_body("done")),
    checkpoints: dict | None = None,
    statuses: tuple[dict, ...] = (status("running"), status("done")),
    publish_start: dict | None = None,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    requests: list[httpx.Request] = []
    job_queue = list(jobs)
    status_queue = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/api/v1/jobs/training":
            return httpx.Response(201, json=job_body())
        if path == f"/api/v1/jobs/{JOB_ID}":
            return httpx.Response(200, json=job_queue.pop(0))
        if path == "/api/v1/models/checkpoints":
            return httpx.Response(200, json=checkpoints if checkpoints is not None else checkpoint_body())
        if path == "/api/v1/models/publish":
            return httpx.Response(
                200,
                json=publish_start
                if publish_start is not None
                else {"started": True, "model_id": JOB_ID, "message": "Publish started"},
            )
        if path == "/api/v1/models/publish-status":
            return httpx.Response(200, json=status_queue.pop(0))
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    return handler, requests


def paths(requests: list[httpx.Request]) -> list[str]:
    return [request.url.path for request in requests]


def test_train_wait_publish_success_with_distinct_training_and_publish_steps():
    handler, requests = scripted_handler()
    slept: list[float] = []
    with mock_client(handler) as client:
        result = TrainingFlows(client).train_and_publish(
            "maker/pick",
            repo_id="maker/act-pick",
            steps=4000,
            publish_steps=[2000],
            save_freq=1000,
            train_timeout=10,
            publish_timeout=10,
            poll_interval=2,
            sleep_fn=slept.append,
        )

    assert result.job.id == JOB_ID and result.job.state == "done"
    assert result.publish.state == "done"
    assert result.publish.url == "https://huggingface.co/maker/act-pick"
    assert slept == [2, 2]
    assert paths(requests) == [
        "/api/v1/jobs/training",
        f"/api/v1/jobs/{JOB_ID}",
        f"/api/v1/jobs/{JOB_ID}",
        "/api/v1/models/checkpoints",
        "/api/v1/models/publish",
        "/api/v1/models/publish-status",
        "/api/v1/models/publish-status",
    ]
    training = json.loads(requests[0].content)
    assert training == {
        "config": {"dataset_repo_id": "maker/pick", "steps": 4000, "save_freq": 1000},
        "target": {"runner": "local"},
    }
    assert requests[3].url.params["id"] == JOB_ID
    assert json.loads(requests[4].content) == {
        "id": JOB_ID,
        "repo_id": "maker/act-pick",
        "steps": [2000],
    }


@pytest.mark.parametrize("state", ["failed", "interrupted"])
def test_failed_or_interrupted_job_never_publishes(state: str):
    handler, requests = scripted_handler(jobs=(job_body(state, error_message="trainer stopped"),))
    with mock_client(handler) as client, pytest.raises(TrainingFlowError, match="trainer stopped") as exc:
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert exc.value.job_id == JOB_ID
    assert paths(requests) == ["/api/v1/jobs/training", f"/api/v1/jobs/{JOB_ID}"]


def test_no_checkpoints_never_publishes():
    handler, requests = scripted_handler(jobs=(job_body("done"),), checkpoints=checkpoint_body(()))
    with (
        mock_client(handler) as client,
        pytest.raises(TrainingFlowError, match="without publishable checkpoints"),
    ):
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert paths(requests)[-1] == "/api/v1/models/checkpoints"


def test_missing_selected_step_never_publishes():
    handler, requests = scripted_handler(jobs=(job_body("done"),))
    with mock_client(handler) as client, pytest.raises(TrainingFlowError, match="requested steps"):
        TrainingFlows(client).train_and_publish(
            "maker/pick", publish_steps=[999], train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert paths(requests)[-1] == "/api/v1/models/checkpoints"


def test_train_timeout_leaves_job_running_and_exposes_id():
    handler, requests = scripted_handler(jobs=(job_body(),))
    with mock_client(handler) as client, pytest.raises(JobWaitTimeout) as exc:
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=0, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert exc.value.job_id == JOB_ID
    assert "client.jobs.wait" in str(exc.value)
    assert paths(requests) == ["/api/v1/jobs/training", f"/api/v1/jobs/{JOB_ID}"]


def test_publish_timeout_leaves_publish_running_and_exposes_id():
    handler, requests = scripted_handler(jobs=(job_body("done"),), statuses=(status("running"),))
    with mock_client(handler) as client, pytest.raises(PublishWaitTimeout) as exc:
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=0, sleep_fn=lambda _: None
        )
    assert exc.value.job_id == JOB_ID
    assert "client.models.publish_status()" in str(exc.value)
    assert paths(requests)[-1] == "/api/v1/models/publish-status"


def test_publish_timeout_counts_slow_status_request():
    handler, requests = scripted_handler(jobs=(job_body("done"),), statuses=(status("running"),))
    now = [0.0]
    sleeps = []

    def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models/publish-status":
            now[0] += 6.0
        return handler(request)

    with mock_client(slow_handler) as client, pytest.raises(PublishWaitTimeout) as error:
        client.flows.train_and_publish(
            "maker/pick",
            train_timeout=5,
            publish_timeout=5,
            poll_interval=2,
            sleep_fn=sleeps.append,
            clock=lambda: now[0],
        )
    assert error.value.waited == 6.0
    assert sleeps == []
    assert paths(requests)[-1] == "/api/v1/models/publish-status"


def test_publish_error_reports_server_error_and_job_id():
    handler, _ = scripted_handler(jobs=(job_body("done"),), statuses=(status("error", error="Hub denied"),))
    with mock_client(handler) as client, pytest.raises(TrainingFlowError, match="Hub denied") as exc:
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert exc.value.job_id == JOB_ID


def test_publish_slot_identity_must_match_even_on_done():
    handler, _ = scripted_handler(jobs=(job_body("done"),), statuses=(status("done", model_id="other"),))
    with mock_client(handler) as client, pytest.raises(TrainingFlowError, match="belongs to 'other'"):
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )


def test_publish_rejected_by_busy_slot_is_not_reported_as_success():
    handler, requests = scripted_handler(
        jobs=(job_body("done"),),
        publish_start={"started": False, "model_id": "other", "message": "Already publishing"},
    )
    with mock_client(handler) as client, pytest.raises(TrainingFlowError, match="not accepted"):
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, sleep_fn=lambda _: None
        )
    assert paths(requests)[-1] == "/api/v1/models/publish"


def test_invalid_timing_is_rejected_before_job_creation():
    def no_requests(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request}")

    with mock_client(no_requests) as client, pytest.raises(ValueError, match="poll_interval"):
        TrainingFlows(client).train_and_publish(
            "maker/pick", train_timeout=5, publish_timeout=5, poll_interval=0
        )
