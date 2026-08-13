# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Offline pre-flight guard for LOCAL training runs.

Covers the belt-and-braces guard in POST /jobs/training (server.py): a local
run on a dataset that isn't available locally, while the Hub is offline, must
be rejected up front instead of letting lerobot try (and hang) to download it.
Also unit-tests the availability helper's two-cache-layout logic in datasets.py.

Everything is mocked — no real HF API, no real training subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import makermodslab.datasets as datasets_mod
import makermodslab.server as server_mod

DATASET_ID = "alice/hub_only_dataset"


def _post_local_training(client: TestClient, repo_id: str = DATASET_ID):
    """Start a LOCAL training run (no target => runner defaults to 'local')."""
    return client.post("/jobs/training", json={"dataset_repo_id": repo_id, "steps": 100})


# ---------------------------------------------------------------------------
# The guard, exercised through the real endpoint.
# ---------------------------------------------------------------------------


def test_offline_and_not_local_rejects_before_start(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline + dataset NOT locally available => 400 with an actionable message,
    and job_registry.start is never called."""
    monkeypatch.setattr(server_mod, "hf_hub_offline", lambda: True)
    monkeypatch.setattr(datasets_mod, "is_dataset_available_locally", lambda _repo_id: False)

    def _fail_if_called(*_a, **_k):  # pragma: no cover - must not run
        raise AssertionError("job_registry.start must not be reached when the guard rejects")

    monkeypatch.setattr(server_mod.job_registry, "start", _fail_if_called)

    resp = _post_local_training(client)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert DATASET_ID in detail
    assert "offline" in detail.lower()


def test_offline_but_available_locally_not_rejected_by_guard(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline + dataset available locally => guard does NOT block; the request
    proceeds into job_registry.start (which we stub)."""
    monkeypatch.setattr(server_mod, "hf_hub_offline", lambda: True)
    monkeypatch.setattr(datasets_mod, "is_dataset_available_locally", lambda _repo_id: True)

    started = {"called": False}

    def _fake_start(config, target):
        started["called"] = True
        return {"id": "job-123", "name": "ACT", "state": "running"}

    monkeypatch.setattr(server_mod.job_registry, "start", _fake_start)

    resp = _post_local_training(client)

    assert started["called"] is True
    assert resp.status_code == 201


def test_online_hub_only_not_rejected_by_guard(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Online (not offline) + hub-only dataset => guard is skipped entirely;
    is_dataset_available_locally is never even consulted."""
    monkeypatch.setattr(server_mod, "hf_hub_offline", lambda: False)

    def _fail_if_checked(_repo_id):  # pragma: no cover - must not run
        raise AssertionError("availability check must be skipped when online")

    monkeypatch.setattr(datasets_mod, "is_dataset_available_locally", _fail_if_checked)

    started = {"called": False}

    def _fake_start(config, target):
        started["called"] = True
        return {"id": "job-123", "name": "ACT", "state": "running"}

    monkeypatch.setattr(server_mod.job_registry, "start", _fake_start)

    resp = _post_local_training(client)

    assert started["called"] is True
    assert resp.status_code == 201


def test_cloud_run_unaffected_by_offline_local_guard(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud (hf_cloud) run must NOT be blocked by the local offline guard —
    it has its own dataset-upload path. The availability check is local-only."""
    monkeypatch.setattr(server_mod, "hf_hub_offline", lambda: True)

    def _fail_if_checked(_repo_id):  # pragma: no cover - must not run
        raise AssertionError("local availability check must not run for a cloud run")

    monkeypatch.setattr(datasets_mod, "is_dataset_available_locally", _fail_if_checked)

    def _fake_start(config, target):
        return {"id": "job-cloud", "name": "ACT", "state": "running"}

    monkeypatch.setattr(server_mod.job_registry, "start", _fake_start)

    resp = client.post(
        "/jobs/training",
        json={
            "config": {"dataset_repo_id": DATASET_ID, "steps": 100},
            "target": {"runner": "hf_cloud", "flavor": "a10g-small"},
        },
    )

    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# The availability helper's two-layout logic (datasets.py).
# ---------------------------------------------------------------------------


def _make_flat_dataset(root: Path, repo_id: str) -> None:
    """Materialize the flat-layout dataset dir _is_dataset_dir recognizes."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))


def test_availability_true_for_flat_layout(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locally recorded (flat-layout) dataset counts as available without any
    hub-cache probe."""

    def _fail_cache(*_a, **_k):  # pragma: no cover - flat hit short-circuits first
        raise AssertionError("flat-layout hit should short-circuit the cache probe")

    monkeypatch.setattr(datasets_mod, "try_to_load_from_cache", _fail_cache)
    _make_flat_dataset(tmp_lerobot_home, DATASET_ID)

    assert datasets_mod.is_dataset_available_locally(DATASET_ID) is True


def test_availability_true_for_hub_snapshot_cache(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset present only in a hub snapshot cache (not the flat layout) still
    counts as available — verified via huggingface_hub's cache lookup."""
    # No flat dataset created; simulate a cache hit (real path string returned).
    monkeypatch.setattr(datasets_mod, "try_to_load_from_cache", lambda *a, **k: "/cache/.../meta/info.json")

    assert datasets_mod.is_dataset_available_locally(DATASET_ID) is True


def test_availability_false_when_absent_from_both_layouts(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent from the flat layout AND from every hub cache (None / sentinel,
    i.e. non-str) => not available."""
    monkeypatch.setattr(datasets_mod, "try_to_load_from_cache", lambda *a, **k: None)

    assert datasets_mod.is_dataset_available_locally(DATASET_ID) is False


def test_availability_conservative_on_cache_probe_error(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache-probe exception is not evidence of absence: degrade to 'assume
    present' so an internal error never wrongly blocks a run."""

    def _boom(*_a, **_k):
        raise RuntimeError("cache probe blew up")

    monkeypatch.setattr(datasets_mod, "try_to_load_from_cache", _boom)

    assert datasets_mod.is_dataset_available_locally(DATASET_ID) is True
