"""The datasets namespace against httpx.MockTransport: every operation sends
the documented request (path, method, params/body) and parses the documented
response shape; waiter tests drive scripted status sequences with a recording
fake sleep — house style: never sleeps, never network."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk.resources._waiting import OperationFailedError, WaitTimeoutError
from makermodslab_sdk.resources.datasets import (
    DatasetHubSettings,
    DatasetHubStatus,
    DatasetInfo,
    DatasetListItem,
    DatasetRenameResult,
    DatasetTags,
    DatasetVisibility,
    DeleteDatasetResult,
    DownloadStart,
    DownloadStatus,
    EpisodeJointSeries,
    EpisodeSummary,
    ImportResult,
    MergeStart,
    MergeStatus,
    SuccessRepoId,
    UploadStart,
    UploadStatus,
)

REPO = "maker/pick-place"


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


# ---------------------------------------------------------------- listings


def test_list_parses_heterogeneous_rows():
    rows = [
        {"repo_id": REPO, "last_modified": "2026-08-01T12:00:00Z", "private": False, "source": "local"},
        {
            "repo_id": "hub/other",
            "last_modified": None,
            "private": True,
            "source": "hub",
            "saved_custom": True,
        },
    ]
    request, result = call_one(rows, lambda c: c.datasets.list())
    assert request.method == "GET"
    assert request.url.path == "/api/v1/datasets"
    assert [type(r) for r in result] == [DatasetListItem, DatasetListItem]
    assert result[0].repo_id == REPO
    assert result[0].saved_custom is None  # absent key, defaulted
    assert result[1].saved_custom is True
    assert result[1].last_modified is None


def test_info_sends_repo_id_query_and_parses():
    body = {
        "repo_id": REPO,
        "total_episodes": 12,
        "total_frames": 4800,
        "fps": 30,
        "robot_type": "so101_follower",
        "cameras": ["front", "wrist"],
        "tasks": [{"task": "pick the cube", "num_episodes": 12}],
        "size_bytes": 123456,
        "source": "local",
    }
    request, info = call_one(body, lambda c: c.datasets.info(REPO))
    assert request.url.path == "/api/v1/datasets/info"
    assert request.url.params["repo_id"] == REPO
    assert isinstance(info, DatasetInfo)
    assert info.fps == 30
    assert info.tasks[0].task == "pick the cube"
    assert info.source == "local"


def test_episodes_and_episode_joints():
    rows = [
        {
            "episode_index": 0,
            "length": 400,
            "duration": 13.3,
            "tasks": ["pick"],
            "video_offsets": {"front": {"from": 0.0, "to": 13.3}},
        }
    ]
    request, episodes = call_one(rows, lambda c: c.datasets.episodes(REPO))
    assert request.url.path == "/api/v1/datasets/episodes"
    assert request.url.params["repo_id"] == REPO
    assert isinstance(episodes[0], EpisodeSummary)
    assert episodes[0].video_offsets["front"]["to"] == 13.3

    series_body = {"joint_names": ["shoulder_pan"], "timestamps": [0.0, 0.033], "values": [[1.0], [1.1]]}
    request, series = call_one(series_body, lambda c: c.datasets.episode_joints(REPO, 0))
    assert request.url.path == "/api/v1/datasets/episode-joints"
    assert request.url.params["repo_id"] == REPO
    assert request.url.params["episode_index"] == "0"
    assert isinstance(series, EpisodeJointSeries)
    assert series.values[1] == [1.1]


# ---------------------------------------------------------------- Hub edits


def test_hub_status_and_hub_settings():
    request, status = call_one(
        {"repo_id": REPO, "status": "on_hub", "url": f"https://huggingface.co/datasets/{REPO}"},
        lambda c: c.datasets.hub_status(REPO),
    )
    assert request.url.path == "/api/v1/datasets/hub-status"
    assert request.url.params["repo_id"] == REPO
    assert isinstance(status, DatasetHubStatus)
    assert status.status == "on_hub"

    request, settings = call_one(
        {"repo_id": REPO, "private": False, "tags": ["lerobot", "so101"]},
        lambda c: c.datasets.hub_settings(REPO),
    )
    assert request.url.path == "/api/v1/datasets/hub-settings"
    assert isinstance(settings, DatasetHubSettings)
    assert settings.tags == ["lerobot", "so101"]


def test_visibility_posts_private_flag():
    request, result = call_one(
        {"repo_id": REPO, "private": True}, lambda c: c.datasets.visibility(REPO, private=True)
    )
    assert request.method == "POST"
    assert request.url.path == "/api/v1/datasets/visibility"
    assert body_of(request) == {"repo_id": REPO, "private": True}
    assert isinstance(result, DatasetVisibility)
    assert result.private is True


def test_tags_posts_list_and_returns_final_tags():
    request, result = call_one(
        {"repo_id": REPO, "tags": ["lerobot", "custom"]}, lambda c: c.datasets.tags(REPO, ["custom"])
    )
    assert request.url.path == "/api/v1/datasets/tags"
    assert body_of(request) == {"repo_id": REPO, "tags": ["custom"]}
    assert isinstance(result, DatasetTags)
    assert result.tags == ["lerobot", "custom"]  # the list actually written, not the input


def test_rename():
    request, result = call_one(
        {"success": True, "repo_id": "maker/pick-place-v2", "hub": "renamed"},
        lambda c: c.datasets.rename(REPO, "pick-place-v2"),
    )
    assert request.url.path == "/api/v1/datasets/rename"
    assert body_of(request) == {"repo_id": REPO, "new_name": "pick-place-v2"}
    assert isinstance(result, DatasetRenameResult)
    assert result.hub == "renamed"


# ---------------------------------------------------------------- pins & hides


def test_save_and_remove_custom():
    request, result = call_one({"success": True, "repo_id": REPO}, lambda c: c.datasets.save_custom(REPO))
    assert (request.method, request.url.path) == ("POST", "/api/v1/datasets/custom")
    assert body_of(request) == {"repo_id": REPO}
    assert isinstance(result, SuccessRepoId)

    request, result = call_one({"success": False, "repo_id": REPO}, lambda c: c.datasets.remove_custom(REPO))
    assert (request.method, request.url.path) == ("DELETE", "/api/v1/datasets/custom")
    assert body_of(request) == {"repo_id": REPO}
    assert result.success is False  # nothing to remove is still a 200


def test_hide_and_unhide():
    request, _ = call_one({"success": True, "repo_id": REPO}, lambda c: c.datasets.hide(REPO))
    assert (request.method, request.url.path) == ("POST", "/api/v1/datasets/hide")
    request, _ = call_one({"success": True, "repo_id": REPO}, lambda c: c.datasets.unhide(REPO))
    assert (request.method, request.url.path) == ("DELETE", "/api/v1/datasets/hide")
    assert body_of(request) == {"repo_id": REPO}


# ---------------------------------------------------------------- transfers


def test_download_and_download_status():
    request, start = call_one(
        {"started": True, "repo_id": REPO, "message": "Download started"},
        lambda c: c.datasets.download(REPO),
    )
    assert (request.method, request.url.path) == ("POST", "/api/v1/datasets/download")
    assert body_of(request) == {"repo_id": REPO}
    assert isinstance(start, DownloadStart)

    request, status = call_one(
        {"state": "running", "repo_id": REPO, "message": "Downloading…", "error": None},
        lambda c: c.datasets.download_status(),
    )
    assert (request.method, request.url.path) == ("GET", "/api/v1/datasets/download-status")
    assert isinstance(status, DownloadStatus)
    assert status.state == "running"
    assert status.error is None


def test_upload_defaults_and_upload_status_idle():
    request, start = call_one(
        {"started": True, "repo_id": REPO, "message": "Upload started"}, lambda c: c.datasets.upload(REPO)
    )
    assert (request.method, request.url.path) == ("POST", "/api/v1/upload-dataset")
    assert body_of(request) == {"dataset_repo_id": REPO, "private": False, "tags": []}
    assert isinstance(start, UploadStart)

    # Idle body: nullable fields present as null, docs_url genuinely absent.
    request, status = call_one(
        {"state": "idle", "repo_id": None, "message": None, "dataset_url": None},
        lambda c: c.datasets.upload_status(),
    )
    assert (request.method, request.url.path) == ("GET", "/api/v1/upload-status")
    assert isinstance(status, UploadStatus)
    assert status.docs_url is None


def test_upload_forwards_private_and_tags():
    request, _ = call_one(
        {"started": True, "repo_id": REPO, "message": "Upload started"},
        lambda c: c.datasets.upload(REPO, private=True, tags=["so101"]),
    )
    assert body_of(request) == {"dataset_repo_id": REPO, "private": True, "tags": ["so101"]}


def test_import_local_with_and_without_name():
    request, result = call_one({"repo_id": "local/imported"}, lambda c: c.datasets.import_local("/data/raw"))
    assert (request.method, request.url.path) == ("POST", "/api/v1/datasets/import")
    assert body_of(request) == {"path": "/data/raw", "name": None}
    assert isinstance(result, ImportResult)

    request, _ = call_one(
        {"repo_id": "local/renamed"}, lambda c: c.datasets.import_local("/data/raw", name="renamed")
    )
    assert body_of(request) == {"path": "/data/raw", "name": "renamed"}


def test_merge_and_merge_status():
    request, start = call_one(
        {"started": True, "message": "Merge started"},
        lambda c: c.datasets.merge(["maker/a", "maker/b"], "maker/merged"),
    )
    assert (request.method, request.url.path) == ("POST", "/api/v1/datasets/merge")
    assert body_of(request) == {"source_repo_ids": ["maker/a", "maker/b"], "output_repo_id": "maker/merged"}
    assert isinstance(start, MergeStart)
    assert start.started is True

    request, status = call_one(
        {
            "state": "done",
            "error": None,
            "output_repo_id": "maker/merged",
            "log_path": "/tmp/merge.log",
            "logs": [{"timestamp": 1723.5, "message": "aggregated 2 datasets"}],
        },
        lambda c: c.datasets.merge_status(),
    )
    assert (request.method, request.url.path) == ("GET", "/api/v1/datasets/merge/status")
    assert isinstance(status, MergeStatus)
    assert status.logs[0].message == "aggregated 2 datasets"


def test_delete():
    request, result = call_one(
        {"success": True, "message": f"Deleted {REPO}"}, lambda c: c.datasets.delete(REPO)
    )
    assert (request.method, request.url.path) == ("POST", "/api/v1/delete-dataset")
    assert body_of(request) == {"dataset_repo_id": REPO}
    assert isinstance(result, DeleteDatasetResult)
    assert result.success is True


# ---------------------------------------------------------------- waiters


def scripted_status_client(path: str, bodies: list[dict]):
    """A client whose GET ``path`` replays ``bodies`` (last one repeats)."""
    served: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == path
        index = min(len(served), len(bodies) - 1)
        served.append(index)
        return httpx.Response(200, json=bodies[index])

    return mock_client(handler), served


def download_body(state, repo_id=REPO, error=None):
    return {"state": state, "repo_id": repo_id, "message": "…", "error": error}


def test_wait_for_download_polls_until_done_without_real_sleep():
    client, served = scripted_status_client(
        "/api/v1/datasets/download-status",
        [download_body("running"), download_body("running"), download_body("done")],
    )
    slept: list[float] = []
    with client:
        status = client.datasets.wait_for_download(REPO, poll_interval=2.0, sleep_fn=slept.append)
    assert status.state == "done"
    assert status.repo_id == REPO
    assert len(served) == 3
    assert slept == [2.0, 2.0]  # one sleep between each poll, never a real one


def test_wait_for_download_error_state_raises_with_server_text():
    client, _ = scripted_status_client(
        "/api/v1/datasets/download-status",
        [download_body("running"), download_body("error", error="401 unauthorized")],
    )
    with client, pytest.raises(OperationFailedError, match="401 unauthorized"):
        client.datasets.wait_for_download(REPO, sleep_fn=lambda s: None)


def test_wait_for_download_times_out_in_virtual_time():
    client, served = scripted_status_client("/api/v1/datasets/download-status", [download_body("running")])
    slept: list[float] = []
    with client, pytest.raises(WaitTimeoutError, match="still running"):
        client.datasets.wait_for_download(REPO, timeout=5.0, poll_interval=2.0, sleep_fn=slept.append)
    assert slept == [2.0, 2.0, 2.0]  # 6s of virtual time crosses the 5s budget
    assert len(served) == 4


def test_wait_for_download_idle_means_nothing_running():
    client, _ = scripted_status_client(
        "/api/v1/datasets/download-status",
        [{"state": "idle", "repo_id": None, "message": None, "error": None}],
    )
    with client, pytest.raises(OperationFailedError, match="idle"):
        client.datasets.wait_for_download(REPO, sleep_fn=lambda s: None)


def test_wait_for_download_other_repo_in_slot_raises():
    client, _ = scripted_status_client(
        "/api/v1/datasets/download-status", [download_body("running", repo_id="someone/else")]
    )
    with client, pytest.raises(OperationFailedError, match="someone/else"):
        client.datasets.wait_for_download(REPO, sleep_fn=lambda s: None)


def test_wait_for_upload_polls_upload_status():
    bodies = [
        {"state": "running", "repo_id": REPO, "message": "Uploading…", "dataset_url": None},
        {
            "state": "done",
            "repo_id": REPO,
            "message": "Uploaded",
            "dataset_url": f"https://huggingface.co/datasets/{REPO}",
        },
    ]
    client, served = scripted_status_client("/api/v1/upload-status", bodies)
    slept: list[float] = []
    with client:
        status = client.datasets.wait_for_upload(REPO, poll_interval=1.5, sleep_fn=slept.append)
    assert status.state == "done"
    assert status.dataset_url.endswith(REPO)
    assert slept == [1.5]
    assert len(served) == 2


def test_wait_for_upload_error_carries_message():
    bodies = [
        {
            "state": "error",
            "repo_id": REPO,
            "message": "Upload failed: token has no write scope",
            "dataset_url": None,
            "docs_url": "https://huggingface.co/docs/hub/security-tokens",
        }
    ]
    client, _ = scripted_status_client("/api/v1/upload-status", bodies)
    with client, pytest.raises(OperationFailedError, match="write scope"):
        client.datasets.wait_for_upload(REPO, sleep_fn=lambda s: None)


# ---------------------------------------------------------------- e2e (read-only, offline-safe)


def test_status_ops_end_to_end(sdk_client):
    """The three poll endpoints are pure local state — safe to hit for real."""
    assert sdk_client.datasets.download_status().state in {"idle", "running", "done", "error"}
    assert sdk_client.datasets.upload_status().state in {"idle", "running", "done", "error"}
    assert sdk_client.datasets.merge_status().state in {"idle", "running", "done", "error"}
