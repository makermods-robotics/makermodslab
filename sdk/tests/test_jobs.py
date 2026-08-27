"""The ``jobs`` namespace: request shapes, response models, and the wait()
flagship — mocked bodies copied from makermodslab/schemas/jobs.py shapes, plus
read-only end-to-end checks against the real app. Never sleeps, never network."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import ApiError, NotFoundError
from makermodslab_sdk.resources.jobs import (
    TERMINAL_STATES,
    Checkpoint,
    HubJob,
    Job,
    JobList,
    LogLine,
    MetricsPoint,
    RunnerFlavor,
)


def job_body(**overrides) -> dict:
    """A realistic JobRecord wire body (makermodslab/jobs.py JobRecord):
    uniform-with-nulls, every key present, runner-specific fields null."""
    body = {
        "id": "act_so101_pick_2026-08-27_10-00-00",
        "job_number": 46,
        "name": "act_so101_pick_2026-08-27_10-00-00",
        "display_name": None,
        "state": "running",
        "config": {
            "dataset_repo_id": "user/so101-pick",
            "policy_type": "act",
            "steps": 10000,
            "batch_size": 8,
            "log_freq": 50,
            "save_freq": 1000,
            "output_dir": "outputs/train",
        },
        "output_dir": "outputs/train/act_so101_pick_2026-08-27_10-00-00",
        "started_at": 1_777_000_000.0,
        "ended_at": None,
        "exit_code": None,
        "error_message": None,
        "metrics": {
            "current_step": 120,
            "total_steps": 10000,
            "current_loss": 0.42,
            "current_lr": 1e-05,
            "grad_norm": 2.1,
            "eta_seconds": 5400.0,
        },
        "runner": "local",
        "process_pid": 4242,
        "node_instance_id": None,
        "node_url": None,
        "remote_job_id": None,
        "hf_job_id": None,
        "hf_flavor": None,
        "hf_repo_id": None,
        "hf_job_url": None,
        "checkpoint_count": 0,
        "checkpoints_hub_repo_id": None,
        "wandb_run_url": None,
        "checkpoints_hub_steps": [],
        "child_ids": [],
        "ancestor_ids": [],
    }
    body.update(overrides)
    return body


def call_via(handler, fn):
    client = mock_client(handler)
    try:
        return fn(client)
    finally:
        client.close()


# ---------------------------------------------------------------- registry ops


def test_list_jobs_sends_limit_and_types_records():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json={"jobs": [job_body()]})

    result = call_via(handler, lambda c: c.jobs.list(limit=25))
    assert seen["path"] == "/api/v1/jobs"
    assert seen["limit"] == "25"
    assert isinstance(result, JobList)
    job = result.jobs[0]
    assert isinstance(job, Job)
    assert job.job_number == 46
    assert job.metrics.current_loss == 0.42
    assert job.config.dataset_repo_id == "user/so101-pick"
    assert job.config.steps == 10000


def test_get_job_hits_id_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/act_so101_pick_2026-08-27_10-00-00"
        assert request.method == "GET"
        return httpx.Response(200, json=job_body(state="done", exit_code=0))

    job = call_via(handler, lambda c: c.jobs.get("act_so101_pick_2026-08-27_10-00-00"))
    assert job.state == "done"
    assert job.exit_code == 0


def test_stop_job_posts_and_returns_final_record():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/jobs/j1/stop"
        return httpx.Response(
            200,
            json=job_body(
                id="j1",
                state="interrupted",
                error_message="Stopped at your request, not by a training error.",
            ),
        )

    job = call_via(handler, lambda c: c.jobs.stop("j1"))
    assert job.state == "interrupted"


def test_delete_job_returns_none_on_204():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/jobs/j1"
        return httpx.Response(204)

    assert call_via(handler, lambda c: c.jobs.delete("j1")) is None


def test_rename_job_sends_new_name_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/jobs/j1/rename"
        assert json.loads(request.content) == {"new_name": "pick v2"}
        return httpx.Response(200, json=job_body(display_name="pick v2"))

    job = call_via(handler, lambda c: c.jobs.rename("j1", "pick v2"))
    assert job.display_name == "pick v2"


def test_logs_and_log_file_share_the_logline_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path in ("/api/v1/jobs/j1/logs", "/api/v1/jobs/j1/log-file")
        return httpx.Response(
            200, json={"logs": [{"timestamp": 1_777_000_100.5, "message": "step:100 loss:0.5"}]}
        )

    logs = call_via(handler, lambda c: c.jobs.logs("j1"))
    assert isinstance(logs.logs[0], LogLine)
    assert logs.logs[0].message == "step:100 loss:0.5"
    whole = call_via(handler, lambda c: c.jobs.log_file("j1"))
    assert whole.logs[0].timestamp == 1_777_000_100.5


def test_metrics_history_points():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/j1/metrics-history"
        return httpx.Response(
            200,
            json={
                "points": [
                    {"step": 50, "loss": 0.9, "lr": 1e-05, "grad_norm": 3.0},
                    {"step": 100, "loss": 0.5, "lr": 1e-05, "grad_norm": None},
                ]
            },
        )

    history = call_via(handler, lambda c: c.jobs.metrics_history("j1"))
    assert [p.step for p in history.points] == [50, 100]
    assert isinstance(history.points[0], MetricsPoint)
    assert history.points[1].grad_norm is None


def test_checkpoints_ascending_by_step():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/j1/checkpoints"
        return httpx.Response(
            200,
            json={
                "checkpoints": [
                    {
                        "step": 1000,
                        "source": "local",
                        "ref": "outputs/train/j1/checkpoints/001000/pretrained_model",
                    },
                    {"step": 2000, "source": "hub", "ref": "user/j1-repo/checkpoints/002000"},
                ]
            },
        )

    result = call_via(handler, lambda c: c.jobs.checkpoints("j1"))
    assert isinstance(result.checkpoints[0], Checkpoint)
    assert result.checkpoints[1].source == "hub"


def test_checkpoint_policy_config_path_and_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/j1/checkpoints/2000/policy-config"
        return httpx.Response(
            200,
            json={
                "policy_type": "act",
                "image_features": {"observation.images.front": {"height": 480, "width": 640}},
                "requires_task": False,
                "state_dim": 6,
                "action_dim": 6,
            },
        )

    cfg = call_via(handler, lambda c: c.jobs.checkpoint_policy_config("j1", 2000))
    assert cfg.policy_type == "act"
    assert cfg.image_features["observation.images.front"].width == 640
    assert cfg.state_dim == 6


# ------------------------------------------------------------------- creation


def test_create_training_sends_config_target_envelope():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=job_body())

    job = call_via(
        handler,
        lambda c: c.jobs.create_training("user/so101-pick", steps=20000, config={"save_freq": 2000}),
    )
    assert seen["path"] == "/api/v1/jobs/training"
    assert seen["body"]["config"]["dataset_repo_id"] == "user/so101-pick"
    assert seen["body"]["config"]["policy_type"] == "act"
    assert seen["body"]["config"]["steps"] == 20000
    assert seen["body"]["config"]["save_freq"] == 2000  # extra config merged in
    assert seen["body"]["target"] == {"runner": "local"}
    assert isinstance(job, Job)


def test_create_training_cloud_target_carries_flavor():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=job_body(runner="hf_cloud", hf_flavor="a10g-small"))

    job = call_via(
        handler,
        lambda c: c.jobs.create_training(
            "user/so101-pick", runner="hf_cloud", flavor="a10g-small", job_name="cloud-run"
        ),
    )
    assert seen["body"]["target"] == {"runner": "hf_cloud", "flavor": "a10g-small"}
    assert seen["body"]["config"]["job_name"] == "cloud-run"
    assert job.runner == "hf_cloud"


def test_import_model_posts_source_and_reads_already_imported_flag():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/v1/jobs/import"
        assert body == {"source": "user/act-so101", "name": "imported act"}
        payload = job_body(runner="imported")
        payload["already_imported"] = True  # 200 re-import branch's extra key
        return httpx.Response(200, json=payload)

    job = call_via(handler, lambda c: c.jobs.import_model("user/act-so101", name="imported act"))
    assert job.runner == "imported"
    # extra="allow" keeps the branch marker readable on the model.
    assert getattr(job, "already_imported", False) is True


# ------------------------------------------------------------------------ hub


def test_list_hub_authenticated_branch():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/hub"
        return httpx.Response(
            200,
            json={
                "authenticated": True,
                "jobs_permission": True,
                "jobs": [
                    {
                        "id": "hubjob1",
                        "name": "act_user_so101-pick_2026-08-27_10-00-00",
                        "created_at": "2026-08-27T10:00:00Z",
                        "docker_image": "ghcr.io/makermods/train:latest",
                        "space_id": None,
                        "flavor": "a10g-small",
                        "status": {"stage": "RUNNING", "message": None},
                        "owner": "user",
                        "url": "https://huggingface.co/jobs/user/hubjob1",
                    }
                ],
                "models": [{"repo_id": "user/act-so101", "last_modified": None, "private": True}],
            },
        )

    hub = call_via(handler, lambda c: c.jobs.list_hub())
    assert hub.authenticated is True
    assert hub.jobs_permission is True
    assert isinstance(hub.jobs[0], HubJob)
    assert hub.jobs[0].status.stage == "RUNNING"
    assert hub.models[0].private is True


def test_list_hub_unauthenticated_branch_omits_jobs_permission():
    def handler(request: httpx.Request) -> httpx.Response:
        # exclude_unset on the server: no jobs_permission key at all here.
        return httpx.Response(200, json={"authenticated": False, "jobs": [], "models": []})

    hub = call_via(handler, lambda c: c.jobs.list_hub())
    assert hub.authenticated is False
    assert hub.jobs_permission is None


def test_dismiss_hub_posts_to_dismiss_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/jobs/hub/jobs/hubjob1/dismiss"
        return httpx.Response(200, json={"status": "success", "job_id": "hubjob1"})

    result = call_via(handler, lambda c: c.jobs.dismiss_hub("hubjob1"))
    assert result.status == "success"
    assert result.job_id == "hubjob1"


def test_delete_hub_model_keeps_repo_id_slash_in_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        # The route is {repo_id:path} — the namespace slash must survive.
        assert request.url.path == "/api/v1/jobs/hub/models/user/act-so101"
        return httpx.Response(200, json={"status": "success", "repo_id": "user/act-so101"})

    result = call_via(handler, lambda c: c.jobs.delete_hub_model("user/act-so101"))
    assert result.repo_id == "user/act-so101"


def test_runners_hardware_flavor_catalog():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/runners/hardware"
        return httpx.Response(
            200,
            json={
                "authenticated": True,
                "username": "user",
                "flavors": [
                    {
                        "name": "cpu-basic",
                        "pretty_name": "CPU basic",
                        "cpu": "2 vCPU",
                        "ram": "16 GB",
                        "accelerator": None,
                        "vram": None,
                        "unit_cost_usd": 0,
                        "unit_label": "hour",
                    },
                    {
                        "name": "a10g-small",
                        "pretty_name": "A10G small",
                        "cpu": "4 vCPU",
                        "ram": "15 GB",
                        "accelerator": "Nvidia A10G",
                        "vram": "24 GB",
                        "unit_cost_usd": 1.05,
                        "unit_label": "hour",
                    },
                ],
                "offline": False,
            },
        )

    hardware = call_via(handler, lambda c: c.jobs.runners_hardware())
    assert hardware.username == "user"
    assert isinstance(hardware.flavors[0], RunnerFlavor)
    assert hardware.flavors[0].accelerator is None
    assert hardware.flavors[1].unit_cost_usd == 1.05


# -------------------------------------------------------------- wait flagship


# --------------------------------------------------- end-to-end (read-only)


def test_list_end_to_end(sdk_client):
    result = sdk_client.jobs.list()
    assert isinstance(result, JobList)
    for job in result.jobs:
        assert isinstance(job, Job)
        assert job.state in TERMINAL_STATES | {"running"}


def test_get_missing_job_end_to_end_is_uncoded_404(sdk_client):
    """SERVER FACT at this snapshot: get_job raises a plain HTTPException — the
    404 body has a string detail and NO `code`, so it decodes as a plain
    ApiError (code None), not NotFoundError."""
    with pytest.raises(ApiError) as excinfo:
        sdk_client.jobs.get("__sdk_test_missing__")
    err = excinfo.value
    assert type(err) is ApiError
    assert not isinstance(err, NotFoundError)
    assert err.status == 404
    assert err.code is None
    assert "__sdk_test_missing__" in (err.detail or "")
