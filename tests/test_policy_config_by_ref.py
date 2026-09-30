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

"""GET /api/v1/policy-config?policy_ref=… — a checkpoint's config summary
addressed by the same opaque ref inference starts from, not by a Lab job id.

The SDK checks `camera_bindings` coverage against this before starting
inference. Every Hub read is faked (both the module-level `hf_hub_download` in
jobs.py and huggingface_hub's own, which the train_config reader imports
lazily) — never the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ACT_CONFIG = {
    "type": "act",
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [7]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 240, 320]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    "n_action_steps": 100,
    "chunk_size": 100,
}


def _write_model(root: Path, config: dict, train_config: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    if train_config is not None:
        (root / "train_config.json").write_text(json.dumps(train_config))
    return root


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A fake Hub: `files[(repo_id, filename)] = dict` serves that JSON; any
    other file raises like a missing one. Records every request."""
    files: dict[tuple[str, str], dict] = {}
    calls: list[tuple[str, str]] = []
    served = tmp_path / "hub"
    served.mkdir()

    def fake_download(repo_id, filename, repo_type="model", **_kw):
        calls.append((repo_id, filename))
        if (repo_id, filename) not in files:
            raise FileNotFoundError(f"{repo_id}/{filename}")
        out = served / f"{len(calls)}.json"
        out.write_text(json.dumps(files[(repo_id, filename)]))
        return str(out)

    import huggingface_hub

    monkeypatch.setattr("makermodslab.jobs.hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return files, calls


def _get(client, ref: str):
    return client.get("/api/v1/policy-config", params={"policy_ref": ref})


def test_local_ref_summarizes_the_config(client, tmp_path, tmp_lerobot_home, fake_hub) -> None:
    model = _write_model(tmp_path / "pretrained_model", ACT_CONFIG)
    resp = _get(client, str(model))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["policy_type"] == "act"
    assert body["image_features"] == {
        "top": {"height": 480, "width": 640},
        "wrist": {"height": 240, "width": 320},
    }
    assert body["state_dim"] == 7
    assert body["action_dim"] == 7
    assert body["n_action_steps"] == 100
    assert body["dataset_repo_id"] is None
    # A local ref never touches the Hub.
    assert fake_hub[1] == []


def test_hub_step_ref_reads_only_that_steps_config(client, tmp_lerobot_home, fake_hub) -> None:
    files, calls = fake_hub
    files[("user/policy", "checkpoints/005000/pretrained_model/config.json")] = ACT_CONFIG
    files[("user/policy", "checkpoints/005000/pretrained_model/train_config.json")] = {
        "dataset": {"repo_id": "user/pick"}
    }
    resp = _get(client, "user/policy@checkpoints/005000")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["image_features"]) == {"top", "wrist"}
    assert body["dataset_repo_id"] == "user/pick"
    # Configs only — never a weights file.
    assert all(name.endswith(".json") for _repo, name in calls)
    assert ("user/policy", "checkpoints/005000/pretrained_model/config.json") in calls


def test_hub_root_ref_reads_the_repo_root_config(client, tmp_lerobot_home, fake_hub) -> None:
    files, calls = fake_hub
    files[("user/flat", "config.json")] = {**ACT_CONFIG, "type": "smolvla"}
    resp = _get(client, "user/flat@root")
    assert resp.status_code == 200, resp.text
    assert resp.json()["policy_type"] == "smolvla"
    # The `@root` suffix is unwrapped to the bare repo id, exactly as the
    # inference start does before reading the config.
    assert ("user/flat", "config.json") in calls
    assert all(repo == "user/flat" for repo, _name in calls)


def test_unreadable_config_is_a_coded_404(client, tmp_lerobot_home, fake_hub) -> None:
    resp = _get(client, "user/private@checkpoints/000100")
    assert resp.status_code == 404
    assert resp.json()["code"] == "checkpoint.config_unreadable"


def test_local_dir_without_config_is_a_coded_404(client, tmp_path, tmp_lerobot_home, fake_hub) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    resp = _get(client, str(empty))
    assert resp.status_code == 404
    assert resp.json()["code"] == "checkpoint.config_unreadable"


@pytest.mark.parametrize("ref", ["user/repo", "user/repo@main", "/no/such/dir", ""])
def test_malformed_ref_is_a_coded_400(client, tmp_lerobot_home, fake_hub, ref) -> None:
    """Exactly the refs inference would refuse as unrecognised — a bare repo id
    included, since the inference start does not accept one either."""
    resp = _get(client, ref)
    assert resp.status_code == 400
    assert resp.json()["code"] == "checkpoint.invalid_ref"
    assert fake_hub[1] == []


def test_both_routes_answer_the_same_body(client, tmp_path, tmp_lerobot_home, fake_hub, monkeypatch) -> None:
    """Parity: the job-checkpoint route and the by-ref route compute from one
    helper, so the same checkpoint answers the same body on both."""
    from makermodslab import server
    from makermodslab.jobs import JobRegistry

    ds_meta = tmp_lerobot_home / "user" / "pick" / "meta"
    ds_meta.mkdir(parents=True)
    (ds_meta / "info.json").write_text(json.dumps({"robot_type": "maker_follower", "features": {}}))
    model = _write_model(tmp_path / "model", ACT_CONFIG, {"dataset": {"repo_id": "user/pick"}})

    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))
    monkeypatch.setattr(server, "job_registry", reg)

    by_job = client.get(f"/api/v1/jobs/{rec.id}/checkpoints/0/policy-config")
    by_ref = _get(client, str(model))
    assert by_job.status_code == 200, by_job.text
    assert by_ref.status_code == 200, by_ref.text
    assert by_ref.json() == by_job.json()
    assert by_ref.json()["trained_on_robot_type"] == "maker_follower"
