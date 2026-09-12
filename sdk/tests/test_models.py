"""The models namespace against httpx.MockTransport: every operation sends
the documented request (path, method, params/body) and parses the documented
response shape — house style: never sleeps, never network."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk.resources._waiting import OperationFailedError
from makermodslab_sdk.resources.datasets import DownloadStart, DownloadStatus, ImportResult, SuccessRepoId
from makermodslab_sdk.resources.models import (
    ModelDeleteResult,
    ModelInfo,
    ModelListItem,
    ModelUploadResult,
)

MODEL_ID = "act_pick_place"
REPO = "maker/act-pick-place"


def call_one(response_json, invoke):
    """Run one SDK call against a recording MockTransport; return (request, result)."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=response_json)

    with mock_client(handler) as client:
        result = invoke(client)
    assert len(calls) == 1
    return calls[0], result


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


def local_run_row() -> dict:
    """A local training-run row (_local_model_summary's key set)."""
    return {
        "id": MODEL_ID,
        "name": "act_pick_place",
        "policy_type": "act",
        "dataset": "maker/pick-place",
        "steps": 20000,
        "path": "outputs/train/act_pick_place",
        "last_modified": "2026-08-20T09:00:00Z",
        "hf_repo_id": None,
        "source": "local",
        "target_steps": 20000,
        "state": "done",
    }


def hub_row() -> dict:
    """A Hub-seeded row (repo_id + private, null training fields)."""
    return {
        "id": REPO,
        "name": "act-pick-place",
        "policy_type": "act",
        "dataset": None,
        "steps": None,
        "path": None,
        "last_modified": "2026-08-21T10:00:00Z",
        "hf_repo_id": REPO,
        "source": "hub",
        "repo_id": REPO,
        "private": True,
        "target_steps": None,
        "state": None,
        "saved_custom": True,
    }


def test_list_parses_heterogeneous_producer_rows():
    request, rows = call_one([local_run_row(), hub_row()], lambda c: c.models.list())
    assert (request.method, request.url.path) == ("GET", "/api/v1/models")
    assert all(isinstance(r, ModelListItem) for r in rows)
    local, hub = rows
    assert local.state == "done" and local.repo_id is None and local.saved_custom is None
    assert hub.private is True and hub.saved_custom is True
    assert hub.hf_repo_id == REPO


def test_info_sends_id_query_and_parses():
    body = {
        "id": MODEL_ID,
        "name": "act_pick_place",
        "policy_type": "act",
        "dataset": "maker/pick-place",
        "steps": 20000,
        "path": "outputs/train/act_pick_place",
        "hf_repo_id": None,
        "size_bytes": 210_000_000,
        "source": "local",
        "target_steps": 20000,
        "state": "done",
    }
    request, info = call_one(body, lambda c: c.models.info(MODEL_ID))
    assert (request.method, request.url.path) == ("GET", "/api/v1/models/info")
    assert request.url.params["id"] == MODEL_ID
    assert isinstance(info, ModelInfo)
    assert info.size_bytes == 210_000_000
    assert info.last_modified is None  # absent on this producer, defaulted
    assert info.source == "local"


def test_download_and_download_status_share_dataset_shapes():
    request, start = call_one(
        {"started": True, "repo_id": REPO, "message": "Download started"}, lambda c: c.models.download(REPO)
    )
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/download")
    assert body_of(request) == {"repo_id": REPO}
    assert isinstance(start, DownloadStart)

    request, status = call_one(
        {"state": "done", "repo_id": REPO, "message": "Downloaded", "error": None},
        lambda c: c.models.download_status(),
    )
    assert (request.method, request.url.path) == ("GET", "/api/v1/models/download-status")
    assert isinstance(status, DownloadStatus)
    assert status.state == "done"


def test_upload_with_and_without_target_repo():
    body = {"repo_id": f"{REPO}", "url": f"https://huggingface.co/{REPO}", "tags": ["lerobot", "act"]}
    request, result = call_one(body, lambda c: c.models.upload(MODEL_ID))
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/upload")
    assert body_of(request) == {"id": MODEL_ID, "repo_id": None}
    assert isinstance(result, ModelUploadResult)
    assert result.tags == ["lerobot", "act"]

    request, _ = call_one(body, lambda c: c.models.upload(MODEL_ID, repo_id=REPO))
    assert body_of(request) == {"id": MODEL_ID, "repo_id": REPO}


def test_delete():
    request, result = call_one({"deleted": True, "id": MODEL_ID}, lambda c: c.models.delete(MODEL_ID))
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/delete")
    assert body_of(request) == {"id": MODEL_ID}
    assert isinstance(result, ModelDeleteResult)
    assert result.deleted is True


def test_import_local_with_and_without_name():
    request, result = call_one({"repo_id": "local/ckpt"}, lambda c: c.models.import_local("/data/ckpt"))
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/import")
    assert body_of(request) == {"path": "/data/ckpt", "name": None}
    assert isinstance(result, ImportResult)

    request, _ = call_one(
        {"repo_id": "local/renamed"}, lambda c: c.models.import_local("/data/ckpt", name="renamed")
    )
    assert body_of(request) == {"path": "/data/ckpt", "name": "renamed"}


def test_save_and_remove_custom():
    request, result = call_one({"success": True, "repo_id": REPO}, lambda c: c.models.save_custom(REPO))
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/custom")
    assert body_of(request) == {"repo_id": REPO}
    assert isinstance(result, SuccessRepoId)

    request, result = call_one({"success": False, "repo_id": REPO}, lambda c: c.models.remove_custom(REPO))
    assert (request.method, request.url.path) == ("DELETE", "/api/v1/models/custom")
    assert body_of(request) == {"repo_id": REPO}
    assert result.success is False


def test_hide_and_unhide():
    request, _ = call_one({"success": True, "repo_id": REPO}, lambda c: c.models.hide(REPO))
    assert (request.method, request.url.path) == ("POST", "/api/v1/models/hide")
    request, _ = call_one({"success": True, "repo_id": REPO}, lambda c: c.models.unhide(REPO))
    assert (request.method, request.url.path) == ("DELETE", "/api/v1/models/hide")
    assert body_of(request) == {"repo_id": REPO}


# ---------------------------------------------------------------- waiters


def test_wait_for_download_polls_the_models_slot():
    bodies = [
        {"state": "running", "repo_id": REPO, "message": "Downloading…", "error": None},
        {"state": "done", "repo_id": REPO, "message": "Downloaded", "error": None},
    ]
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(request.url.path)
        return httpx.Response(200, json=bodies[min(len(served) - 1, len(bodies) - 1)])

    slept: list[float] = []
    with mock_client(handler) as client:
        status = client.models.wait_for_download(REPO, poll_interval=3.0, sleep_fn=slept.append)
    assert status.state == "done"
    assert served == ["/api/v1/models/download-status"] * 2  # the models slot, not the datasets one
    assert slept == [3.0]


def test_wait_for_download_failure_raises_with_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"state": "error", "repo_id": REPO, "message": "…", "error": "repo gated"}
        )

    with mock_client(handler) as client, pytest.raises(OperationFailedError, match="repo gated"):
        client.models.wait_for_download(REPO, sleep_fn=lambda s: None)


# ---------------------------------------------------------------- e2e (read-only, offline-safe)


def test_download_status_end_to_end(sdk_client):
    """The poll endpoint is pure local state — safe to hit for real."""
    assert sdk_client.models.download_status().state in {"idle", "running", "done", "error"}


def test_skills_listing_shapes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/skills"
        return httpx.Response(
            200,
            json={
                "skills": [
                    {
                        "id": "job-1",
                        "name": "pick-place v2",
                        "policy_type": "act",
                        "dataset": "u/pick",
                        "steps": 20000,
                        "path": "outputs/train/job-1",
                        "last_modified": "2026-09-01T10:00:00",
                        "hf_repo_id": None,
                        "source": "local",
                        "origin": "trained",
                        "weights": "local",
                        "superseded_by": None,
                        "deployable": True,
                        "job_id": "job-1",
                    }
                ],
                "hub": {"ok": True, "authenticated": True, "degraded": False, "stale_rows": False},
            },
        )

    with mock_client(handler) as client:
        skills = client.models.skills()
    assert skills.hub.ok is True
    assert skills.skills[0].deployable is True
    assert skills.skills[0].name == "pick-place v2"
