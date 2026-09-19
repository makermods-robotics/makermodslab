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


"""W&B credential preflight, independent of the retired Hub-offline tests."""

import pytest
from fastapi.testclient import TestClient

import makermodslab.server as server_mod
from makermodslab.jobs import JobRecord

DATASET_ID = "alice/hub_only_dataset"


def _record(config, job_id="job-123"):
    return JobRecord(
        id=job_id, name="ACT", state="running", config=config, output_dir="/tmp/run", started_at=1.0
    )


def _post_local_training(client):
    return client.post("/api/v1/jobs/training", json={"dataset_repo_id": DATASET_ID, "steps": 100})


def test_cloud_wandb_without_a_key_rejects_before_the_dataset_is_touched(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key is forwarded to the pod as a job secret, so a missing one makes
    the run impossible. Refuse at the endpoint — before job_registry.start, and
    therefore long before HfCloudJobRunner._ensure_dataset_on_hub could push a
    local-only dataset to the Hub for a job that will never run."""
    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", lambda: None)

    def _fail_if_called(*_a, **_k):  # pragma: no cover - must not run
        raise AssertionError("job_registry.start must not be reached when the key is missing")

    monkeypatch.setattr(server_mod.job_registry, "start", _fail_if_called)

    resp = client.post(
        "/api/v1/jobs/training",
        json={
            "config": {"dataset_repo_id": DATASET_ID, "steps": 100, "wandb_enable": True},
            "target": {"runner": "hf_cloud", "flavor": "a10g-small"},
        },
    )

    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_local_wandb_without_a_key_is_rejected_too(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W&B runs locally as well as on the cloud, and the guard applies to both.
    A local trainer is a non-tty subprocess: `wandb.init` cannot prompt for a
    login, so it fails after the record already says `running` — the same
    useless failure as the cloud case, just on cheaper hardware."""
    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", lambda: None)

    def _fail_if_called(*_a, **_k):  # pragma: no cover - must not run
        raise AssertionError("job_registry.start must not be reached when the key is missing")

    monkeypatch.setattr(server_mod.job_registry, "start", _fail_if_called)

    resp = client.post(
        "/api/v1/jobs/training",
        json={"dataset_repo_id": DATASET_ID, "steps": 100, "wandb_enable": True},
    )

    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_wandb_with_a_key_is_not_blocked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is about the key, not about W&B: with one resolvable the
    request proceeds into job_registry.start (stubbed)."""
    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", lambda: "a-key")

    started = {"called": False}

    def _fake_start(config, target):
        started["called"] = True
        return _record(config)

    monkeypatch.setattr(server_mod.job_registry, "start", _fake_start)

    resp = client.post(
        "/api/v1/jobs/training",
        json={"dataset_repo_id": DATASET_ID, "steps": 100, "wandb_enable": True},
    )

    assert started["called"] is True
    assert resp.status_code == 201


def test_a_run_with_wandb_off_never_probes_for_a_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard keys on W&B, not on the runner: an ordinary run that isn't
    logging must not be made to depend on a credential it will never use."""

    def _fail_if_checked():  # pragma: no cover - must not run
        raise AssertionError("the W&B key must not be probed when W&B is off")

    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", _fail_if_checked)
    monkeypatch.setattr(
        server_mod.job_registry,
        "start",
        lambda config, target: _record(config, "job-1"),
    )

    resp = _post_local_training(client)
    assert resp.status_code == 201


def test_wandb_credentials_endpoint_reports_only_a_boolean(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe behind GET /system/wandb-credentials must never leak the key
    itself — the UI only needs to know whether a launch would be refused."""
    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", lambda: "super-secret-key")
    resp = client.get("/api/v1/system/wandb-credentials")
    assert resp.status_code == 200
    assert resp.json()["available"] is True
    assert "super-secret-key" not in resp.text

    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", lambda: None)
    assert client.get("/api/v1/system/wandb-credentials").json()["available"] is False


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_local_non_online_mode_never_requires_a_key(client, monkeypatch, tmp_path, mode):
    from unittest.mock import MagicMock

    from makermodslab import jobs
    from makermodslab.runners import hf_cloud

    def no_key_probe():
        raise AssertionError("non-online logging must not inspect credentials")

    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", no_key_probe)
    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", no_key_probe)
    runner = MagicMock()
    runner.wandb_run_url.return_value = None
    monkeypatch.setattr(jobs, "LocalJobRunner", lambda *args, **kwargs: runner)
    registry = jobs.JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(server_mod, "job_registry", registry)
    response = client.post(
        "/api/v1/jobs/training",
        json={"dataset_repo_id": DATASET_ID, "wandb_enable": True, "wandb_mode": mode},
    )
    assert response.status_code == 201, response.text
    assert response.json()["config"]["wandb_mode"] == mode
    runner.start.assert_called_once()


def test_lan_origin_defers_credentials_to_peer(client, monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from makermodslab import datasets, jobs, nodes
    from makermodslab.runners import hf_cloud, lan_node

    def origin_has_no_key():
        raise AssertionError("the origin must not inspect its own W&B key for a peer run")

    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", origin_has_no_key)
    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", origin_has_no_key)
    monkeypatch.setattr(nodes.node_registry, "resolve", lambda node_id: MagicMock())
    monkeypatch.setattr(datasets, "get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr(datasets, "hub_copy_has_data", lambda repo_id: True)
    peer = MagicMock()
    peer.wandb_run_url.return_value = None
    peer.node_url.return_value = "http://peer:8000"
    peer.remote_job_id.return_value = "peer-job"
    monkeypatch.setattr(lan_node, "LanNodeJobRunner", lambda *args, **kwargs: peer)
    registry = jobs.JobRegistry(tmp_path / "origin-jobs")
    monkeypatch.setattr(server_mod, "job_registry", registry)
    response = client.post(
        "/api/v1/jobs/training",
        json={
            "config": {"dataset_repo_id": DATASET_ID, "wandb_enable": True, "wandb_mode": "online"},
            "target": {"runner": "lan_node", "node_instance_id": "bb" * 16},
        },
    )
    assert response.status_code == 201, response.text
    peer.start.assert_called_once()
    forwarded = peer.start.call_args.args[1]
    assert forwarded.wandb_enable is True
    assert forwarded.wandb_mode == "online"

    # On the execution host this is a local request, so its own key authorizes it.
    monkeypatch.setattr(server_mod, "resolve_wandb_api_key", lambda: "peer-only-key")
    monkeypatch.setattr(server_mod.job_registry, "start", lambda config, target: _record(config))
    response = client.post("/api/v1/jobs/training", json=forwarded.model_dump())
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_resume_uses_checkpoint_owners_mode_for_credentials(monkeypatch, tmp_path, mode):
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget

    from .test_jobs import _wandb_parent, _wandb_resume_request

    registry = JobRegistry(tmp_path / "jobs")
    _wandb_parent(registry, wandb_enable=True)
    registry._records["P"].config.wandb_mode = mode
    runner = MagicMock()
    runner.hf_job_id.return_value = "peer-job"
    runner.hf_job_url.return_value = None
    runner.wandb_run_url.return_value = None

    def no_key_probe():
        raise AssertionError("the inherited non-online mode needs no key")

    with (
        patch("makermodslab.jobs._resolve_cloud_resume", return_value=("user/P", "004000")),
        patch("makermodslab.runners.hf_cloud.resolve_wandb_api_key", no_key_probe),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: runner),
        patch("makermodslab.datasets.get_hub_status", return_value={"status": "on_hub"}),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
    ):
        record = registry.start(
            _wandb_resume_request(wandb_mode="online"), JobTarget(runner="hf_cloud", flavor="t4-small")
        )
    assert record.config.wandb_mode == mode
    assert runner.start.call_args.args[1].wandb_mode == mode


@pytest.mark.parametrize("form_mode", ["offline", "disabled"])
def test_online_resume_cannot_bypass_key_check_with_form_mode(monkeypatch, tmp_path, form_mode):
    from unittest.mock import patch

    from makermodslab.jobs import JobRegistry, JobTarget

    from .test_jobs import _wandb_parent, _wandb_resume_request

    registry = JobRegistry(tmp_path / "jobs")
    _wandb_parent(registry, wandb_enable=True)
    with (
        patch("makermodslab.jobs._resolve_cloud_resume", return_value=("user/P", "004000")),
        patch("makermodslab.runners.hf_cloud.resolve_wandb_api_key", return_value=None),
        patch("makermodslab.datasets.get_hub_status", return_value={"status": "on_hub"}),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
        pytest.raises(ValueError, match="Weights & Biases API key"),
    ):
        registry.start(
            _wandb_resume_request(wandb_mode=form_mode), JobTarget(runner="hf_cloud", flavor="t4-small")
        )
    assert set(registry._records) == {"P"}
