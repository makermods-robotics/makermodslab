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
"""Tests for makermodslab.datasets — local cache walk and merge logic."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient


def _make_dataset(root: Path, repo_id: str, episodes: int = 1) -> None:
    """Create the minimal layout `_is_dataset_dir` recognizes. `episodes`
    defaults to 1 so the dataset isn't filtered out as empty."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))


def test_list_local_datasets_empty_when_root_missing(
    tmp_lerobot_home: Path,
) -> None:
    # tmp_lerobot_home creates the cache; remove it so the function sees the
    # "missing root" branch.
    import shutil

    from makermodslab.datasets import list_local_datasets

    shutil.rmtree(tmp_lerobot_home)
    assert list_local_datasets() == []


def test_list_local_datasets_finds_top_level_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "pusht" in repo_ids


def test_list_local_datasets_finds_nested_user_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "alice/pusht" in repo_ids


def test_list_local_datasets_skips_non_dataset_dirs(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import list_local_datasets

    (tmp_lerobot_home / "calibration").mkdir(exist_ok=True)
    (tmp_lerobot_home / "ports").mkdir(exist_ok=True)
    _make_dataset(tmp_lerobot_home, "real_dataset")

    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "real_dataset" in repo_ids
    assert "calibration" not in repo_ids
    assert "ports" not in repo_ids


def test_list_local_datasets_hides_empty_dataset(
    tmp_lerobot_home: Path,
) -> None:
    """A 0-episode dataset (aborted recording) is hidden so it can't be picked
    for merging/training, where it only errors out."""
    from makermodslab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "has_eps", episodes=3)
    _make_dataset(tmp_lerobot_home, "empty_ds", episodes=0)

    repo_ids = [d["repo_id"] for d in list_local_datasets()]
    assert "has_eps" in repo_ids
    assert "empty_ds" not in repo_ids


def test_list_local_datasets_skips_dot_prefixed_dirs(
    tmp_lerobot_home: Path,
) -> None:
    """Dot-prefixed directories (scratch/hidden by convention, including
    temporary episode-delete swap dirs) are skipped even if they contain
    valid meta/info.json and episodes."""
    from makermodslab.datasets import list_local_datasets

    # Create a normal dataset that should appear.
    _make_dataset(tmp_lerobot_home, "normal", episodes=3)

    # Create a dot-prefixed dataset that should NOT appear.
    dot_dataset = tmp_lerobot_home / ".hidden"
    dot_dataset.mkdir(parents=True)
    (dot_dataset / "meta").mkdir()
    (dot_dataset / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3}))

    repo_ids = [d["repo_id"] for d in list_local_datasets()]
    assert "normal" in repo_ids
    # The dot-prefixed dir name doesn't appear even though it has valid structure.
    assert ".hidden" not in repo_ids
    assert not any(".hidden" in rid for rid in repo_ids)


def test_list_user_datasets_returns_empty_when_not_logged_in(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import list_user_datasets

    with patch("makermodslab.datasets.cached_whoami", return_value=None):
        assert list_user_datasets() == []


def test_list_all_datasets_merges_hub_and_local(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import list_all_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")

    with patch(
        "makermodslab.datasets.list_user_datasets",
        return_value=[
            {"repo_id": "alice/pusht", "last_modified": "2026-01-01T00:00:00Z", "private": False},
            {"repo_id": "alice/aloha", "last_modified": "2026-02-01T00:00:00Z", "private": True},
        ],
    ):
        result = list_all_datasets()

    by_id = {d["repo_id"]: d for d in result}
    assert by_id["alice/pusht"]["source"] == "both"
    assert by_id["alice/aloha"]["source"] == "hub"


def _write_info(root: Path, repo_id: str, info: dict[str, Any]) -> Path:
    """Write a dataset dir with the given meta/info.json; returns the dir."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps(info))
    return d


def test_get_local_dataset_info_returns_full_details(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "alice/pick",
        {
            "total_episodes": 20,
            "total_frames": 16723,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {
                "action": {"dtype": "float32"},
                "observation.state": {"dtype": "float32"},
                "observation.images.wrist": {"dtype": "video"},
                "observation.images.front": {"dtype": "video"},
            },
        },
    )
    # v3.0 task metadata: tasks.parquet with task_index + task columns,
    # deliberately written out of index order to check the sort.
    pq.write_table(
        pa.table({"task_index": [1, 0], "task": ["second task", "first task"]}),
        d / "meta" / "tasks.parquet",
    )
    # v3.0 episode metadata: per-episode `tasks` column split across chunked
    # parquet files — 18 episodes of "first task", 2 of "second task".
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": list(range(15)), "tasks": [["first task"]] * 15}),
        episodes_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(15, 20)),
                "tasks": [["first task"]] * 3 + [["second task"]] * 2,
            }
        ),
        episodes_dir / "file-001.parquet",
    )
    (d / "data").mkdir()
    (d / "data" / "file-000.parquet").write_bytes(b"x" * 1234)

    result = get_local_dataset_info("alice/pick")
    assert result is not None
    assert result["total_episodes"] == 20
    assert result["total_frames"] == 16723
    assert result["fps"] == 30
    assert result["robot_type"] == "so_follower"
    assert result["cameras"] == ["wrist", "front"]
    assert result["tasks"] == [
        {"task": "first task", "num_episodes": 18},
        {"task": "second task", "num_episodes": 2},
    ]
    # Directory walk covers data + meta files, so at least the data blob.
    assert result["size_bytes"] >= 1234


def test_get_local_dataset_info_reads_v2_tasks_jsonl(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "old_format",
        {"total_episodes": 2, "total_frames": 100, "fps": 30, "features": {}},
    )
    lines = [
        json.dumps({"task_index": 1, "task": "beta"}),
        json.dumps({"task_index": 0, "task": "alpha"}),
    ]
    (d / "meta" / "tasks.jsonl").write_text("\n".join(lines))
    # v2.x episode metadata: episodes.jsonl with per-episode `tasks` lists.
    ep_lines = [
        json.dumps({"episode_index": 0, "tasks": ["alpha"], "length": 50}),
        json.dumps({"episode_index": 1, "tasks": ["beta"], "length": 50}),
        json.dumps({"episode_index": 2, "tasks": ["beta"], "length": 50}),
    ]
    (d / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines))

    result = get_local_dataset_info("old_format")
    assert result is not None
    assert result["tasks"] == [
        {"task": "alpha", "num_episodes": 1},
        {"task": "beta", "num_episodes": 2},
    ]


def test_get_local_dataset_info_single_task_missing_episode_metadata(
    tmp_lerobot_home: Path,
) -> None:
    """Task strings without episode metadata still render — counts degrade to 0."""
    from makermodslab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "alice/solo",
        {"total_episodes": 5, "total_frames": 500, "fps": 30, "features": {}},
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": ["only task"]}),
        d / "meta" / "tasks.parquet",
    )

    result = get_local_dataset_info("alice/solo")
    assert result is not None
    assert result["tasks"] == [{"task": "only task", "num_episodes": 0}]


def test_get_local_dataset_info_zero_episodes_and_no_cameras(
    tmp_lerobot_home: Path,
) -> None:
    """A 0-episode dataset is hidden from the listing but must still resolve
    here, so the frontend can render its warning badges."""
    from makermodslab.datasets import get_local_dataset_info

    _write_info(
        tmp_lerobot_home,
        "alice/aborted",
        {
            "total_episodes": 0,
            "total_frames": 0,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {"action": {"dtype": "float32"}},
        },
    )

    result = get_local_dataset_info("alice/aborted")
    assert result is not None
    assert result["total_episodes"] == 0
    assert result["cameras"] == []
    assert result["tasks"] == []


def test_get_local_dataset_info_missing_dataset_returns_none(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import get_local_dataset_info

    assert get_local_dataset_info("nobody/nothing") is None


def test_get_local_dataset_info_rejects_path_traversal(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.datasets import get_local_dataset_info

    # A dataset-shaped dir OUTSIDE the cache root must not be reachable.
    outside = tmp_lerobot_home.parent / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))

    assert get_local_dataset_info("../outside") is None
    assert get_local_dataset_info("..") is None
    assert get_local_dataset_info(".") is None


def test_datasets_info_endpoint(client: TestClient, tmp_lerobot_home: Path) -> None:
    _write_info(
        tmp_lerobot_home,
        "alice/pick",
        {
            "total_episodes": 3,
            "total_frames": 900,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {"observation.images.front": {"dtype": "video"}},
        },
    )

    ok = client.get("/datasets/info", params={"repo_id": "alice/pick"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["total_episodes"] == 3
    assert body["cameras"] == ["front"]
    assert body["size_bytes"] > 0

    missing = client.get("/datasets/info", params={"repo_id": "alice/ghost"})
    assert missing.status_code == 404


def test_video_camera_names_filters_by_dtype() -> None:
    from makermodslab.datasets import _video_camera_names

    features = {
        "observation.images.front": {"dtype": "video"},
        "observation.images.raw": {"dtype": "image"},
        "observation.images.wrist": {"dtype": "video"},
        "observation.state": {"dtype": "float32"},
        "action": {"dtype": "float32"},
    }
    assert _video_camera_names(features) == ["front", "wrist"]


# --- Hub sync status --------------------------------------------------------


def _clear_hub_status_cache() -> None:
    from makermodslab import datasets as ds

    with ds._HUB_STATUS_LOCK:
        ds._HUB_STATUS_CACHE.clear()


def test_get_hub_status_reports_on_hub_when_repo_exists() -> None:
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makermodslab.datasets.shared_hf_api", return_value=fake_api):
        result = ds.get_hub_status("alice/pick")

    assert result["status"] == "on_hub"
    assert result["url"] == "https://huggingface.co/datasets/alice/pick"
    fake_api.repo_exists.assert_called_once_with("alice/pick", repo_type="dataset")


def test_get_hub_status_reports_local_only_when_missing_from_hub_but_local() -> None:
    """Not on the Hub, but a usable local copy exists → "local_only" (offer
    upload)."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.is_dataset_available_locally", return_value=True),
    ):
        result = ds.get_hub_status("alice/pick")

    assert result["status"] == "local_only"
    assert result["url"] is None


def test_get_hub_status_reports_absent_when_neither_hub_nor_local() -> None:
    """Neither on the Hub nor in the local cache → "absent", NOT "local_only".

    This is the BUG-3 root cause: a stale pin (e.g. a merge output that was
    deleted/renamed) used to report "local_only", which the info card read as
    "you have it locally" and rendered the contradictory "not downloaded
    locally" + "Local only / Upload" pair. "absent" is also NOT cached (a later
    record/merge can make it appear locally), so a second call re-checks."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.is_dataset_available_locally", return_value=False),
    ):
        result = ds.get_hub_status("makermods/sock")
        assert result["status"] == "absent"
        assert result["url"] is None
        # Not cached: a second call re-invokes repo_exists.
        ds.get_hub_status("makermods/sock")
    assert fake_api.repo_exists.call_count == 2


def test_get_hub_status_degrades_to_unknown_offline() -> None:
    """A transport error (offline / rate-limited) degrades to "unknown" and is
    NOT cached, so the next check re-tries once connectivity returns."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.side_effect = OSError("no network")
    with patch("makermodslab.datasets.shared_hf_api", return_value=fake_api):
        result = ds.get_hub_status("alice/pick")
        assert result["status"] == "unknown"
        assert result["url"] is None
        # Not cached: a second call re-invokes repo_exists.
        ds.get_hub_status("alice/pick")
    assert fake_api.repo_exists.call_count == 2


def test_get_hub_status_caches_definitive_answer() -> None:
    """A definitive answer is memoized: repo_exists runs once across calls."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makermodslab.datasets.shared_hf_api", return_value=fake_api):
        ds.get_hub_status("alice/pick")
        ds.get_hub_status("alice/pick")
    assert fake_api.repo_exists.call_count == 1


def test_invalidate_hub_status_forces_recheck() -> None:
    """After invalidation (called on successful upload), the next check
    re-queries the Hub — so a "local_only" answer can flip to "on_hub"."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.is_dataset_available_locally", return_value=True),
    ):
        assert ds.get_hub_status("alice/pick")["status"] == "local_only"
        # Simulate a successful upload: repo now exists, cache invalidated.
        ds.invalidate_hub_status("alice/pick")
        fake_api.repo_exists.return_value = True
        assert ds.get_hub_status("alice/pick")["status"] == "on_hub"
    assert fake_api.repo_exists.call_count == 2


def test_hub_status_endpoint(client: TestClient) -> None:
    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makermodslab.datasets.shared_hf_api", return_value=fake_api):
        resp = client.get("/datasets/hub-status", params={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["repo_id"] == "alice/pick"
    assert body["status"] == "on_hub"
    assert body["url"] == "https://huggingface.co/datasets/alice/pick"


# --- Rename -----------------------------------------------------------------


def test_rename_local_dataset_moves_directory(tmp_lerobot_home: Path) -> None:
    """Happy path: the directory moves, only the name segment changes, and the
    returned repo id carries the fixed namespace prefix."""
    from makermodslab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/old_name", episodes=3)

    new_id = rename_local_dataset("makermods/old_name", "new_name")

    assert new_id == "makermods/new_name"
    assert not (tmp_lerobot_home / "makermods" / "old_name").exists()
    assert (tmp_lerobot_home / "makermods" / "new_name" / "meta" / "info.json").is_file()


def test_rename_endpoint_old_id_404s_new_id_resolves(client: TestClient, tmp_lerobot_home: Path) -> None:
    """End-to-end through the route: after a rename the old id 404s on
    /datasets/info and the new id resolves."""
    _write_info(
        tmp_lerobot_home,
        "makermods/pick",
        {"total_episodes": 3, "total_frames": 900, "fps": 30, "features": {}},
    )

    resp = client.post(
        "/datasets/rename",
        json={"repo_id": "makermods/pick", "new_name": "place"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "makermods/place"}

    old = client.get("/datasets/info", params={"repo_id": "makermods/pick"})
    assert old.status_code == 404
    new = client.get("/datasets/info", params={"repo_id": "makermods/place"})
    assert new.status_code == 200
    assert new.json()["total_episodes"] == 3


def test_rename_bare_dataset_keeps_no_namespace(tmp_lerobot_home: Path) -> None:
    """A dataset with no namespace renames to a bare name (no prefix invented)."""
    from makermodslab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "solo", episodes=1)
    assert rename_local_dataset("solo", "solo2") == "solo2"
    assert (tmp_lerobot_home / "solo2" / "meta" / "info.json").is_file()


def test_rename_same_name_is_noop(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/keep", episodes=1)
    assert rename_local_dataset("makermods/keep", "keep") == "makermods/keep"


def test_rename_rejects_invalid_name(tmp_lerobot_home: Path) -> None:
    """new_name is validated with the same rules as recording — a slash is a
    name segment, not a namespace, so it's rejected."""
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/src", episodes=1)

    for bad in ["with/slash", "..", " leading", ""]:
        with pytest.raises(DatasetRenameError) as exc:
            rename_local_dataset("makermods/src", bad)
        assert exc.value.status == 400
    # The source was never moved by a rejected rename.
    assert (tmp_lerobot_home / "makermods" / "src").exists()


def test_rename_missing_source_404s(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    with pytest.raises(DatasetRenameError) as exc:
        rename_local_dataset("makermods/ghost", "new")
    assert exc.value.status == 404


def test_rename_target_exists_409s(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/src", episodes=1)
    _make_dataset(tmp_lerobot_home, "makermods/taken", episodes=1)

    with pytest.raises(DatasetRenameError) as exc:
        rename_local_dataset("makermods/src", "taken")
    assert exc.value.status == 409
    # Neither directory was touched.
    assert (tmp_lerobot_home / "makermods" / "src").exists()
    assert (tmp_lerobot_home / "makermods" / "taken").exists()


def test_rename_rejects_path_traversal(tmp_lerobot_home: Path) -> None:
    """A source id escaping the cache root is refused before any move."""
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    outside = tmp_lerobot_home.parent / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))

    for bad in ["../outside", "..", "."]:
        with pytest.raises(DatasetRenameError):
            rename_local_dataset(bad, "new")


def test_rename_busy_guard_recording(tmp_lerobot_home: Path) -> None:
    """A rename is refused (409) while a recording session writes to the id —
    matching either the stamped id or a rename of the still-writing base."""
    from makermodslab import record as rec
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/live", episodes=1)

    fake_cfg = MagicMock()
    # Recording stamps a timestamp: name -> name_<ts>.
    fake_cfg.dataset_repo_id = "makermods/live_20260101"
    with (
        patch.object(rec, "recording_active", True),
        patch.object(rec, "recording_config", fake_cfg),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/live", "renamed")
    assert exc.value.status == 409
    assert (tmp_lerobot_home / "makermods" / "live").exists()


def test_rename_busy_guard_merge(tmp_lerobot_home: Path) -> None:
    """A rename is refused while a merge is producing the target id."""
    from makermodslab import merge
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/out", episodes=1)

    with (
        patch.object(merge.merge_manager, "state", "running"),
        patch.object(merge.merge_manager, "output_repo_id", "makermods/out"),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/out", "renamed")
    assert exc.value.status == 409


def test_rename_busy_guard_upload(tmp_lerobot_home: Path) -> None:
    """A rename is refused (409) while the dataset is being pushed to the Hub."""
    from makermodslab import record as rec
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/uploading", episodes=1)

    with (
        patch.object(rec.upload_manager, "state", "running"),
        patch.object(rec.upload_manager, "repo_id", "makermods/uploading"),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/uploading", "renamed")
    assert exc.value.status == 409
    assert (tmp_lerobot_home / "makermods" / "uploading").exists()


def test_rename_busy_guard_local_training(tmp_lerobot_home: Path) -> None:
    """A rename is refused while a running local job trains on the id."""
    from makermodslab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/train_ds", episodes=1)

    # _dataset_in_use imports job_registry from .jobs lazily (datasets<->record
    # cycle), so patch it at its source module.
    from makermodslab import jobs

    job = MagicMock()
    job.state = "running"
    job.runner = "local"
    job.config.dataset_repo_id = "makermods/train_ds"
    with (
        patch.object(jobs.job_registry, "list", return_value=[job]),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/train_ds", "renamed")
    assert exc.value.status == 409


def test_rename_invalidates_hub_status_for_both_ids(tmp_lerobot_home: Path) -> None:
    """The cached Hub-existence answer is dropped for BOTH the old and new id,
    so the info card re-checks each after the move."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "makermods/before", episodes=1)

    with patch("makermodslab.datasets.invalidate_hub_status") as inval:
        ds.rename_local_dataset("makermods/before", "after")

    called = {c.args[0] for c in inval.call_args_list}
    assert called == {"makermods/before", "makermods/after"}


# --- Episode delete ----------------------------------------------------------


def test_delete_episode_missing_dataset_404s(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    with pytest.raises(DatasetEpisodeDeleteError) as exc:
        delete_local_episode("makermods/ghost", 0)
    assert exc.value.status == 404


def test_delete_episode_rejects_path_traversal(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    for bad in ["../outside", "..", "."]:
        with pytest.raises(DatasetEpisodeDeleteError) as exc:
            delete_local_episode(bad, 0)
        assert exc.value.status == 404


def test_delete_episode_busy_guard_recording(tmp_lerobot_home: Path) -> None:
    """Refused (409) while a recording session writes to the id — before ever
    touching lerobot's delete_episodes."""
    from makermodslab import record as rec
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/live", episodes=3)

    fake_cfg = MagicMock()
    fake_cfg.dataset_repo_id = "makermods/live"
    with (
        patch.object(rec, "recording_active", True),
        patch.object(rec, "recording_config", fake_cfg),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/live", 0)
    assert exc.value.status == 409
    assert (tmp_lerobot_home / "makermods" / "live").exists()


def test_delete_episode_busy_guard_merge(tmp_lerobot_home: Path) -> None:
    from makermodslab import merge
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/out", episodes=3)

    with (
        patch.object(merge.merge_manager, "state", "running"),
        patch.object(merge.merge_manager, "output_repo_id", "makermods/out"),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/out", 0)
    assert exc.value.status == 409


def test_delete_episode_busy_guard_concurrent_episode_delete(tmp_lerobot_home: Path) -> None:
    """A second delete_local_episode call for the same repo_id, while the
    first is still mid-swap, is refused (409) instead of racing it."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "makermods/busy", episodes=3)

    ds._episode_deletes_in_progress.add("makermods/busy")
    try:
        with pytest.raises(ds.DatasetEpisodeDeleteError) as exc:
            ds.delete_local_episode("makermods/busy", 0)
        assert exc.value.status == 409
    finally:
        ds._episode_deletes_in_progress.discard("makermods/busy")


def test_dataset_in_use_reports_episode_delete_in_progress() -> None:
    """_dataset_in_use (used by rename, whole-dataset delete, and
    upload-start) refuses a dataset that's mid-episode-delete, so those
    operations can't race delete_local_episode's swap."""
    from makermodslab import datasets as ds

    ds._episode_deletes_in_progress.add("makermods/mid-swap")
    try:
        assert ds._dataset_in_use("makermods/mid-swap") is not None
    finally:
        ds._episode_deletes_in_progress.discard("makermods/mid-swap")


def test_delete_episode_and_upload_start_cannot_interleave(tmp_lerobot_home: Path) -> None:
    """TOCTOU regression test (PR #54 review, C2): delete_local_episode's busy
    CHECK (_dataset_in_use) and its CLAIM (_episode_deletes_in_progress.add)
    must be atomic with respect to UploadManager.start's own check-and-claim.
    An upload-start landing in the gap between delete's check and its claim
    must not slip through and begin pushing a directory that's about to be
    rewritten out from under it.

    Reproduced deterministically (no real timing race): a patched
    _dataset_in_use pauses on its FIRST call only (delete_local_episode's),
    holding whatever synchronization the real code provides, while a second
    thread attempts UploadManager.start for the same repo_id. If the two
    don't share a lock, the upload's own call to _dataset_in_use runs
    unimpeded and sees nothing claimed yet — the actual bug."""
    from makermodslab import datasets as ds
    from makermodslab import record as rec
    from makermodslab.record import UploadRequest

    repo_id = "makermods/three"
    _make_dataset(tmp_lerobot_home, repo_id, episodes=3)

    checked = threading.Event()
    proceed = threading.Event()
    real_dataset_in_use = ds._dataset_in_use
    first_call_lock = threading.Lock()
    first_call_done = False

    def delayed_check(rid: str):
        nonlocal first_call_done
        with first_call_lock:
            is_first = not first_call_done
            first_call_done = True
        result = real_dataset_in_use(rid)
        if is_first:
            checked.set()
            assert proceed.wait(timeout=5), "test deadlocked waiting for main thread"
        return result

    mgr = rec.UploadManager()
    # If the race isn't closed, start() succeeds and spawns a real worker
    # thread that would hit the network (LeRobotDataset load + push_to_hub).
    # Neuter it — only start()'s own check-and-claim is under test here.
    mgr._worker = lambda request: None

    delete_result: dict[str, Any] = {}

    def run_delete() -> None:
        with (
            patch("makermodslab.datasets._dataset_in_use", delayed_check),
            patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
            patch(
                "makermodslab.datasets.delete_episodes",
                side_effect=_stub_delete_episodes_success(2),
            ),
            # The real job_registry is process-global and, on a dev machine
            # with real HF Cloud job history, listing it hits the network
            # (checkpoint lookups) — unrelated to this test, so keep
            # _dataset_in_use's job-registry check fast and deterministic.
            patch("makermodslab.jobs.job_registry.list", return_value=[]),
        ):
            delete_result["value"] = ds.delete_local_episode(repo_id, 1)

    deleter = threading.Thread(target=run_delete, daemon=True)
    deleter.start()
    assert checked.wait(timeout=5), "delete_local_episode never reached its busy check"

    upload_result: dict[str, Any] = {}
    upload_attempted = threading.Event()

    def run_upload() -> None:
        upload_attempted.set()
        upload_result["value"] = mgr.start(UploadRequest(dataset_repo_id=repo_id))

    uploader = threading.Thread(target=run_upload, daemon=True)
    uploader.start()
    assert upload_attempted.wait(timeout=5)
    uploader.join(timeout=0.3)
    assert uploader.is_alive(), (
        "upload-start returned while delete_local_episode was still mid busy-check — "
        "the two don't share a lock, so the upload slipped through the TOCTOU gap"
    )

    proceed.set()
    deleter.join(timeout=10)
    uploader.join(timeout=10)

    assert delete_result["value"]["success"] is True
    assert upload_result["value"]["started"] is False


def test_delete_episode_load_failure_400s(tmp_lerobot_home: Path) -> None:
    """A dataset dir that passes the cheap `_is_dataset_dir` check (has
    meta/info.json) but isn't a real loadable LeRobotDataset (e.g. corrupt,
    or predates v3.0) 400s instead of raising an unhandled exception."""
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/corrupt", episodes=3)

    with (
        patch("lerobot.datasets.LeRobotDataset", side_effect=ValueError("bad dataset")),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/corrupt", 0)
    assert exc.value.status == 400


def _fake_loaded_dataset(total_episodes: int) -> MagicMock:
    ds = MagicMock()
    ds.meta.total_episodes = total_episodes
    return ds


def test_delete_episode_out_of_range_400s(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/three", 3)
    assert exc.value.status == 400


def test_delete_episode_last_remaining_400s(tmp_lerobot_home: Path) -> None:
    """The only episode in the dataset — refused; the frontend is expected to
    route this through whole-dataset delete instead, but the backend must not
    silently misbehave if it's hit directly."""
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/solo", episodes=1)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(1)),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/solo", 0)
    assert exc.value.status == 400
    assert "only episode" in exc.value.message


def _stub_delete_episodes_success(new_total: int, *, marker: bytes = b"new-data"):
    """Build a fake `delete_episodes` that mimics lerobot's real contract:
    write something into `output_dir` and return an object with the new
    dataset's metadata."""

    def _fake(dataset, episode_indices, output_dir=None, repo_id=None):
        output_dir = Path(output_dir)
        (output_dir / "meta").mkdir(parents=True)
        (output_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": new_total}))
        (output_dir / "marker.bin").write_bytes(marker)
        result = MagicMock()
        result.meta.total_episodes = new_total
        return result

    return _fake


def test_delete_episode_happy_path_swaps_directory(tmp_lerobot_home: Path) -> None:
    """The live directory ends up as whatever lerobot's delete_episodes wrote
    into output_dir, the temp/backup dirs are cleaned up, and the listing
    cache is invalidated."""
    from makermodslab.datasets import delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch(
            "makermodslab.datasets.delete_episodes",
            side_effect=_stub_delete_episodes_success(2),
        ),
        patch("makermodslab.datasets.invalidate_dataset_listing_cache") as inval,
    ):
        result = delete_local_episode("makermods/three", 1)

    assert result == {
        "success": True,
        "repo_id": "makermods/three",
        "deleted_episode": 1,
        "total_episodes": 2,
    }
    live = tmp_lerobot_home / "makermods" / "three"
    assert (live / "marker.bin").read_bytes() == b"new-data"
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 2
    # No leftover temp/backup siblings.
    leftovers = [p.name for p in (tmp_lerobot_home / "makermods").iterdir() if p.name != "three"]
    assert leftovers == []
    inval.assert_called_once()


def test_delete_episode_never_pushes_to_hub(tmp_lerobot_home: Path) -> None:
    """Deleting an episode must never trigger a Hub push — a local edit does
    not touch an already-published Hub copy. (This replaces the old
    resync-on-delete feature, which used to push the rewritten dataset back
    to the Hub automatically.)"""
    from makermodslab import record
    from makermodslab.datasets import delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch(
            "makermodslab.datasets.delete_episodes",
            side_effect=_stub_delete_episodes_success(2),
        ),
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "makermods/three", "status": "on_hub", "url": "x"},
        ),
        patch.object(record.upload_manager, "start") as start_mock,
    ):
        result = delete_local_episode("makermods/three", 1)

    assert result == {
        "success": True,
        "repo_id": "makermods/three",
        "deleted_episode": 1,
        "total_episodes": 2,
    }
    start_mock.assert_not_called()


def test_delete_episode_rewrite_failure_leaves_original_intact(tmp_lerobot_home: Path) -> None:
    """If lerobot's delete_episodes itself raises, the original dataset
    directory is untouched and no temp dir is left behind."""
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    def _boom(dataset, episode_indices, output_dir=None, repo_id=None):
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "partial").write_bytes(b"x")
        raise RuntimeError("re-encode blew up")

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch("makermodslab.datasets.delete_episodes", side_effect=_boom),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/three", 1)

    assert exc.value.status == 500
    live = tmp_lerobot_home / "makermods" / "three"
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 3
    leftovers = [p.name for p in (tmp_lerobot_home / "makermods").iterdir() if p.name != "three"]
    assert leftovers == []


def test_delete_episode_swap_failure_rolls_back(tmp_lerobot_home: Path) -> None:
    """If the final tmp_dir -> target rename fails, the original directory is
    restored from the backup rather than left missing."""
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)
    live = tmp_lerobot_home / "makermods" / "three"

    real_rename = os.rename
    call_count = {"n": 0}

    def _flaky_rename(src, dst):
        call_count["n"] += 1
        # First rename is target -> backup (let it through); second is
        # tmp_dir -> target (fail it) to exercise the rollback path.
        if call_count["n"] == 2:
            raise OSError("disk full")
        real_rename(src, dst)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch(
            "makermodslab.datasets.delete_episodes",
            side_effect=_stub_delete_episodes_success(2),
        ),
        patch("makermodslab.datasets.os.rename", side_effect=_flaky_rename),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/three", 1)

    assert exc.value.status == 500
    # Rolled back: the original 3-episode dataset is back at the live path.
    assert live.exists()
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 3
    leftovers = [p.name for p in (tmp_lerobot_home / "makermods").iterdir() if p.name != "three"]
    assert leftovers == []


def test_delete_episode_swap_and_rollback_both_fail(tmp_lerobot_home: Path) -> None:
    """If both the tmp_dir -> target swap AND the backup_dir -> target rollback
    fail, tmp_dir must be cleaned up (backup_dir is the safety net and stays)."""
    from makermodslab.datasets import DatasetEpisodeDeleteError, delete_local_episode

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    real_rename = os.rename
    call_count = {"n": 0}

    def _flaky_rename(src, dst):
        call_count["n"] += 1
        # First rename is target -> backup (let it through); second is
        # tmp_dir -> target (fail it); third is backup_dir -> target
        # (also fail it) to exercise the double-failure rollback cleanup path.
        if call_count["n"] in (2, 3):
            raise OSError("disk full")
        real_rename(src, dst)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch(
            "makermodslab.datasets.delete_episodes",
            side_effect=_stub_delete_episodes_success(2),
        ),
        patch("makermodslab.datasets.os.rename", side_effect=_flaky_rename),
        pytest.raises(DatasetEpisodeDeleteError) as exc,
    ):
        delete_local_episode("makermods/three", 1)

    assert exc.value.status == 500
    # backup_dir exists (safety net with original data).
    backups = [p.name for p in (tmp_lerobot_home / "makermods").iterdir() if ".pre-delete-" in p.name]
    assert len(backups) == 1
    # tmp_dir is cleaned up (not left as orphan).
    tmp_dirs = [p.name for p in (tmp_lerobot_home / "makermods").iterdir() if ".delete-tmp-" in p.name]
    assert tmp_dirs == []


# --- Orphaned episode-delete dir recovery ---


def _make_dot_dir(root: Path, repo_id: str, kind: str, episodes: int = 1) -> Path:
    """Create a dot-prefixed sibling of `repo_id`'s dataset dir, named like
    delete_local_episode's own tmp_dir/backup_dir (".<name>.<kind>-<hex8>"),
    with a real dataset layout inside so recovery can be verified by content."""
    namespace, _, name = repo_id.rpartition("/")
    parent = (root / namespace) if namespace else root
    parent.mkdir(parents=True, exist_ok=True)
    dot_dir = parent / f".{name}.{kind}-aaaaaaaa"
    (dot_dir / "meta").mkdir(parents=True)
    (dot_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    return dot_dir


def test_recover_orphaned_dirs_removes_stale_tmp_dir(tmp_lerobot_home: Path) -> None:
    """A leftover .delete-tmp-* dir (crash mid re-encode) is always disposable."""
    from makermodslab.datasets import recover_orphaned_episode_delete_dirs

    tmp_dir = _make_dot_dir(tmp_lerobot_home, "makermods/three", "delete-tmp")

    recover_orphaned_episode_delete_dirs(tmp_lerobot_home)

    assert not tmp_dir.exists()


def test_recover_orphaned_dirs_removes_backup_when_swap_already_completed(
    tmp_lerobot_home: Path,
) -> None:
    """If the live dataset already exists, the swap finished before the crash
    and only the backup's own cleanup was interrupted — remove the backup."""
    from makermodslab.datasets import recover_orphaned_episode_delete_dirs

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=2)
    backup_dir = _make_dot_dir(tmp_lerobot_home, "makermods/three", "pre-delete", episodes=3)

    recover_orphaned_episode_delete_dirs(tmp_lerobot_home)

    assert not backup_dir.exists()
    live = tmp_lerobot_home / "makermods" / "three"
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 2


def test_recover_orphaned_dirs_restores_backup_when_live_dir_missing(
    tmp_lerobot_home: Path,
) -> None:
    """If the live dataset is missing, the crash landed mid-swap and the
    backup is the only surviving copy — restore it rather than delete it."""
    from makermodslab.datasets import recover_orphaned_episode_delete_dirs

    backup_dir = _make_dot_dir(tmp_lerobot_home, "makermods/three", "pre-delete", episodes=3)

    recover_orphaned_episode_delete_dirs(tmp_lerobot_home)

    assert not backup_dir.exists()
    live = tmp_lerobot_home / "makermods" / "three"
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 3


def test_recover_orphaned_dirs_handles_top_level_repo_id(tmp_lerobot_home: Path) -> None:
    """The same recovery works for a bare (non-namespaced) repo id, not just
    a namespace/name one."""
    from makermodslab.datasets import recover_orphaned_episode_delete_dirs

    backup_dir = _make_dot_dir(tmp_lerobot_home, "pusht", "pre-delete", episodes=5)

    recover_orphaned_episode_delete_dirs(tmp_lerobot_home)

    assert not backup_dir.exists()
    live = tmp_lerobot_home / "pusht"
    assert json.loads((live / "meta" / "info.json").read_text())["total_episodes"] == 5


def test_recover_orphaned_dirs_ignores_unrelated_dot_dirs(tmp_lerobot_home: Path) -> None:
    """A dot-dir that doesn't match the tmp/backup naming (e.g. some other
    hidden dir) is left alone."""
    from makermodslab.datasets import recover_orphaned_episode_delete_dirs

    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=2)
    unrelated = tmp_lerobot_home / "makermods" / ".hidden"
    unrelated.mkdir(parents=True)
    (unrelated / "marker").write_bytes(b"x")

    recover_orphaned_episode_delete_dirs(tmp_lerobot_home)

    assert unrelated.exists()


def test_list_local_datasets_recovers_orphan_and_lists_it(tmp_lerobot_home: Path) -> None:
    """list_local_datasets sweeps for orphaned episode-delete dirs (once per
    cache root) so a dataset stranded by a crash reappears without a restart."""
    from makermodslab.datasets import list_local_datasets

    _make_dot_dir(tmp_lerobot_home, "makermods/recovered", "pre-delete", episodes=4)

    repo_ids = [d["repo_id"] for d in list_local_datasets()]

    assert "makermods/recovered" in repo_ids


# --- Episode delete endpoint ---

def test_delete_episode_endpoint_success(client: TestClient, tmp_lerobot_home: Path) -> None:
    _make_dataset(tmp_lerobot_home, "makermods/three", episodes=3)

    with (
        patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(3)),
        patch(
            "makermodslab.datasets.delete_episodes",
            side_effect=_stub_delete_episodes_success(2),
        ),
    ):
        resp = client.post(
            "/datasets/episode-delete",
            json={"repo_id": "makermods/three", "episode_index": 1},
        )
    assert resp.status_code == 200
    assert resp.json() == {
        "success": True,
        "repo_id": "makermods/three",
        "deleted_episode": 1,
        "total_episodes": 2,
    }


def test_delete_episode_endpoint_404_missing(client: TestClient, tmp_lerobot_home: Path) -> None:
    resp = client.post(
        "/datasets/episode-delete",
        json={"repo_id": "makermods/ghost", "episode_index": 0},
    )
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_delete_episode_endpoint_last_episode_400s(client: TestClient, tmp_lerobot_home: Path) -> None:
    _make_dataset(tmp_lerobot_home, "makermods/solo", episodes=1)

    with patch("lerobot.datasets.LeRobotDataset", return_value=_fake_loaded_dataset(1)):
        resp = client.post(
            "/datasets/episode-delete",
            json={"repo_id": "makermods/solo", "episode_index": 0},
        )
    assert resp.status_code == 400


# --- Hub visibility / tags editing (post-upload) ----------------------------


def test_set_dataset_visibility_calls_hfapi_with_repo_type() -> None:
    """set_dataset_visibility drives HfApi.update_repo_settings with the
    requested private flag and repo_type="dataset"; result echoes the flag."""
    from makermodslab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.invalidate_hub_status") as inval,
    ):
        result = ds.set_dataset_visibility("alice/pick", private=True)

    fake_api.update_repo_settings.assert_called_once_with("alice/pick", private=True, repo_type="dataset")
    assert result == {"repo_id": "alice/pick", "private": True}
    inval.assert_called_once_with("alice/pick")


def test_set_dataset_visibility_public_passes_false() -> None:
    from makermodslab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.invalidate_hub_status"),
    ):
        result = ds.set_dataset_visibility("alice/pick", private=False)

    fake_api.update_repo_settings.assert_called_once_with("alice/pick", private=False, repo_type="dataset")
    assert result["private"] is False


def test_set_dataset_visibility_rejected_offline() -> None:
    """Offline: no HfApi call, a 400 DatasetHubEditError instead."""
    from makermodslab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=True),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_visibility("alice/pick", private=True)

    assert exc.value.status == 400
    fake_api.update_repo_settings.assert_not_called()


def test_set_dataset_visibility_maps_permission_error() -> None:
    """A 403/forbidden Hub failure becomes a 403 DatasetHubEditError."""
    from makermodslab import datasets as ds

    fake_api = MagicMock()
    fake_api.update_repo_settings.side_effect = Exception("403 Forbidden: no write access")
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_visibility("alice/pick", private=True)

    assert exc.value.status == 403


def test_set_dataset_tags_runs_through_with_makermodslab_tag_before_update() -> None:
    """User tags are funnelled through with_makermodslab_tag (so makermods/openbooth/
    MakerMods Lab survive) BEFORE metadata_update, which is called with overwrite=True
    and repo_type="dataset". The returned tag list is what was written."""
    from makermodslab import datasets as ds
    from makermodslab.utils.config import REQUIRED_HUB_TAGS

    _clear_hub_status_cache()
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.metadata_update") as meta,
        patch("makermodslab.datasets.invalidate_hub_status") as inval,
    ):
        result = ds.set_dataset_tags("alice/pick", ["robotics", "so101"])

    meta.assert_called_once()
    args, kwargs = meta.call_args
    assert args[0] == "alice/pick"
    written = args[1]["tags"]
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["overwrite"] is True
    # User tags come first, org tags are appended and never dropped.
    assert written[:2] == ["robotics", "so101"]
    for required in REQUIRED_HUB_TAGS:
        assert required in written
    assert result["tags"] == written
    inval.assert_called_once_with("alice/pick")


def test_set_dataset_tags_preserves_org_tags_when_user_omits_them() -> None:
    """Even an empty user tag list still writes the required org tags — an edit
    can never strip makermods/openbooth/MakerModsLab off the card."""
    from makermodslab import datasets as ds
    from makermodslab.utils.config import REQUIRED_HUB_TAGS

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.metadata_update") as meta,
        patch("makermodslab.datasets.invalidate_hub_status"),
    ):
        result = ds.set_dataset_tags("alice/pick", [])

    written = meta.call_args.args[1]["tags"]
    assert set(REQUIRED_HUB_TAGS).issubset(set(written))
    assert result["tags"] == written


def test_set_dataset_tags_rejected_offline() -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=True),
        patch("makermodslab.datasets.metadata_update") as meta,
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_tags("alice/pick", ["robotics"])

    assert exc.value.status == 400
    meta.assert_not_called()


def test_set_dataset_tags_maps_auth_error() -> None:
    """A 401/auth Hub failure maps to a 403 DatasetHubEditError with docs_url."""
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch(
            "makermodslab.datasets.metadata_update",
            side_effect=Exception("401 you must be authenticated"),
        ),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_tags("alice/pick", ["robotics"])

    assert exc.value.status == 403
    assert exc.value.docs_url is not None


def test_get_hub_settings_returns_private_and_tags() -> None:
    from makermodslab import datasets as ds

    fake_info = MagicMock()
    fake_info.private = True
    fake_info.tags = ["robotics", "makermods"]
    fake_api = MagicMock()
    fake_api.dataset_info.return_value = fake_info
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
    ):
        result = ds.get_hub_settings("alice/pick")

    fake_api.dataset_info.assert_called_once_with("alice/pick")
    assert result == {"repo_id": "alice/pick", "private": True, "tags": ["robotics", "makermods"]}


def test_get_hub_settings_rejected_offline() -> None:
    from makermodslab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=True),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.get_hub_settings("alice/pick")

    assert exc.value.status == 400
    fake_api.dataset_info.assert_not_called()


def test_visibility_endpoint(client: TestClient) -> None:
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=MagicMock()),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.invalidate_hub_status"),
    ):
        resp = client.post("/datasets/visibility", json={"repo_id": "alice/pick", "private": True})
    assert resp.status_code == 200
    assert resp.json() == {"repo_id": "alice/pick", "private": True}


def test_visibility_endpoint_offline_400(client: TestClient) -> None:
    with patch("makermodslab.datasets.hf_hub_offline", return_value=True):
        resp = client.post("/datasets/visibility", json={"repo_id": "alice/pick", "private": True})
    assert resp.status_code == 400


def test_tags_endpoint_writes_and_preserves_org_tags(client: TestClient) -> None:
    from makermodslab.utils.config import REQUIRED_HUB_TAGS

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.metadata_update") as meta,
        patch("makermodslab.datasets.invalidate_hub_status"),
    ):
        resp = client.post("/datasets/tags", json={"repo_id": "alice/pick", "tags": ["robotics"]})
    assert resp.status_code == 200
    written = meta.call_args.args[1]["tags"]
    for required in REQUIRED_HUB_TAGS:
        assert required in written
    assert resp.json()["tags"] == written


def test_hub_settings_endpoint(client: TestClient) -> None:
    fake_info = MagicMock()
    fake_info.private = False
    fake_info.tags = ["robotics"]
    fake_api = MagicMock()
    fake_api.dataset_info.return_value = fake_info
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
    ):
        resp = client.get("/datasets/hub-settings", params={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["private"] is False
    assert body["tags"] == ["robotics"]


# ---------------------------------------------------------------------------
# DownloadManager — background Hub-dataset download (start → running → done|error).
# The fetch runs in a worker thread; tests mock snapshot_download so no real Hub
# call happens, then join the thread before asserting on the final state.
# ---------------------------------------------------------------------------


def _join_download(mgr, timeout: float = 5.0) -> None:
    thread = mgr._thread
    if thread is not None:
        thread.join(timeout=timeout)


def _dataset_download_manager():
    """A fresh DownloadManager wired with the dataset fetch/cleanup callables —
    the same wiring as the module singleton, but with clean state per test."""
    from makermodslab import datasets as ds

    return ds.DownloadManager(ds._fetch_dataset_snapshot, ds._cleanup_partial_dataset)


def test_download_manager_idle_shape() -> None:
    status = _dataset_download_manager().get_status()
    assert status["state"] == "idle"
    assert status["repo_id"] is None
    assert status["message"] is None
    assert status["error"] is None


def test_download_manager_start_runs_and_completes(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start fetches in a worker thread into the FLAT cache layout and lands in
    state "done", invalidating the hub status + listing caches so the source
    flips to "both"."""
    from makermodslab import datasets as ds

    def _fake_snapshot(repo_id, repo_type, local_dir):  # noqa: ARG001
        # Materialize the flat layout list_local_datasets / is_dataset_available
        # _locally recognize.
        d = Path(local_dir)
        (d / "meta").mkdir(parents=True)
        (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2}))

    monkeypatch.setattr(ds, "snapshot_download", _fake_snapshot)
    invalidated: list[str] = []
    monkeypatch.setattr(ds, "invalidate_hub_status", invalidated.append)

    mgr = _dataset_download_manager()
    result = mgr.start("alice/pick")
    assert result == {"started": True, "repo_id": "alice/pick", "message": "Download started"}

    _join_download(mgr)
    status = mgr.get_status()
    assert status["state"] == "done"
    assert status["repo_id"] == "alice/pick"
    assert status["error"] is None
    assert invalidated == ["alice/pick"]
    # The dataset now lives in the flat layout, so it's available locally.
    assert ds.is_dataset_available_locally("alice/pick")


def test_download_manager_error_surfaces_message(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fetch lands in state "error" with the message and error set, and
    leaves no half-written dataset dir behind."""
    from makermodslab import datasets as ds

    def _boom(repo_id, repo_type, local_dir):  # noqa: ARG001
        raise RuntimeError("network exploded")

    monkeypatch.setattr(ds, "snapshot_download", _boom)

    mgr = _dataset_download_manager()
    mgr.start("alice/pick")
    _join_download(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "network exploded" in status["message"]
    assert status["error"] == "network exploded"
    assert not (tmp_lerobot_home / "alice" / "pick").exists()


def test_download_manager_rejects_concurrent_start() -> None:
    """A second start while one is running is refused (409-mapped by the route),
    naming the repo already downloading; the running download is untouched."""
    mgr = _dataset_download_manager()
    mgr.state = "running"
    mgr.repo_id = "alice/first"

    result = mgr.start("bob/second")
    assert result["started"] is False
    assert "already running" in result["message"]
    assert "alice/first" in result["message"]
    assert mgr.repo_id == "alice/first"


def test_download_endpoint_rejects_bad_repo_id(client: TestClient) -> None:
    resp = client.post("/datasets/download", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


def test_download_endpoint_409_when_running(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.datasets as ds

    monkeypatch.setattr(ds.download_manager, "state", "running")
    monkeypatch.setattr(ds.download_manager, "repo_id", "alice/busy")
    resp = client.post("/datasets/download", json={"repo_id": "bob/other"})
    assert resp.status_code == 409
    assert "alice/busy" in resp.json()["detail"]


def test_download_status_endpoint_idle(client: TestClient) -> None:
    resp = client.get("/datasets/download-status")
    assert resp.status_code == 200
    assert resp.json()["state"] in {"idle", "running", "done", "error"}


# ---------------------------------------------------------------------------
# import_local_dataset — copy a local LeRobot dataset folder into the cache.
# ---------------------------------------------------------------------------


def _make_source_dataset(root: Path, name: str, episodes: int = 2) -> Path:
    """A LeRobot dataset dir OUTSIDE the cache, to import FROM."""
    d = root / name
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    (d / "data").mkdir()
    (d / "data" / "chunk.parquet").write_bytes(b"payload")
    return d


def test_import_local_dataset_copies_into_cache(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "my_ds")
    result = import_local_dataset(str(src))
    assert result == {"repo_id": "my_ds"}

    dst = tmp_lerobot_home / "my_ds"
    assert (dst / "meta" / "info.json").is_file()
    assert (dst / "data" / "chunk.parquet").read_bytes() == b"payload"
    # COPY, not move — the source is left intact.
    assert (src / "meta" / "info.json").is_file()


def test_import_local_dataset_honors_explicit_namespaced_name(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "raw")
    result = import_local_dataset(str(src), name="team/renamed")
    assert result == {"repo_id": "team/renamed"}
    assert (tmp_lerobot_home / "team" / "renamed" / "meta" / "info.json").is_file()


def test_import_local_dataset_404_missing_folder(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import DatasetImportError, import_local_dataset

    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset("/definitely/not/here")
    assert ei.value.status == 404


def test_import_local_dataset_400_not_a_dataset(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import DatasetImportError, import_local_dataset

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(plain))
    assert ei.value.status == 400


def test_import_local_dataset_400_empty_dataset(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import DatasetImportError, import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "empty", episodes=0)
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src))
    assert ei.value.status == 400


def test_import_local_dataset_400_bad_name(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import DatasetImportError, import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "raw")
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src), name="a/b/c")  # too many slashes
    assert ei.value.status == 400


def test_import_local_dataset_409_target_exists(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.datasets import DatasetImportError, import_local_dataset

    _make_dataset(tmp_lerobot_home, "taken", episodes=1)  # already in the cache
    src = _make_source_dataset(tmp_path / "external", "src")
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src), name="taken")
    assert ei.value.status == 409


def test_import_endpoint_success(client: TestClient, tmp_lerobot_home: Path, tmp_path: Path) -> None:
    src = _make_source_dataset(tmp_path / "external", "endpoint_ds")
    resp = client.post("/datasets/import", json={"path": str(src)})
    assert resp.status_code == 200
    assert resp.json() == {"repo_id": "endpoint_ds"}
    assert (tmp_lerobot_home / "endpoint_ds" / "meta" / "info.json").is_file()


def test_import_endpoint_404_missing(client: TestClient, tmp_lerobot_home: Path) -> None:
    resp = client.post("/datasets/import", json={"path": "/no/such/folder"})
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


# ---------------------------------------------------------------------------
# Hidden datasets — persistent "remove from list" for hub rows.
# ---------------------------------------------------------------------------


@pytest.fixture
def hidden_datasets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SAVED_HIDDEN_DATASETS_FILE into a tmp file so hide tests never
    touch the developer's real ~/.cache."""
    from makermodslab.utils import config as cfg

    path = tmp_path / "hidden_datasets.json"
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_DATASETS_FILE", str(path))
    return path


def test_hidden_datasets_round_trip(hidden_datasets_file: Path) -> None:
    from makermodslab.utils.config import (
        add_hidden_dataset,
        get_hidden_datasets,
        remove_hidden_dataset,
    )

    assert get_hidden_datasets() == set()
    assert add_hidden_dataset("alice/pick")
    assert add_hidden_dataset("alice/pick")  # idempotent re-hide
    assert add_hidden_dataset("bob/place")
    assert get_hidden_datasets() == {"alice/pick", "bob/place"}

    assert remove_hidden_dataset("alice/pick")
    assert not remove_hidden_dataset("alice/pick")  # already unhidden
    assert get_hidden_datasets() == {"bob/place"}
    assert not add_hidden_dataset("")  # blank refused


def test_hidden_datasets_corrupt_file_degrades_to_empty(hidden_datasets_file: Path) -> None:
    from makermodslab.utils.config import get_hidden_datasets

    hidden_datasets_file.write_text("{not json")
    assert get_hidden_datasets() == set()
    hidden_datasets_file.write_text(json.dumps({"not": "a list"}))
    assert get_hidden_datasets() == set()


def test_listing_filters_hidden_hub_row(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import list_all_datasets

    hub_rows = [{"repo_id": "alice/pick", "last_modified": None, "private": False}]
    with (
        patch("makermodslab.datasets.list_user_datasets", return_value=hub_rows),
        patch("makermodslab.datasets.get_saved_custom_datasets", return_value=[]),
        patch("makermodslab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_listing_hidden_filter_runs_after_pin_fold(tmp_lerobot_home: Path) -> None:
    """A hidden id can't resurface via a pin — the filter runs AFTER the pin
    fold, so hidden+pinned stays hidden (until the pin ROUTE auto-unhides)."""
    from makermodslab.datasets import list_all_datasets

    with (
        patch("makermodslab.datasets.list_user_datasets", return_value=[]),
        patch("makermodslab.datasets.get_saved_custom_datasets", return_value=["alice/pick"]),
        patch("makermodslab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_listing_hidden_filter_covers_local_copy(tmp_lerobot_home: Path) -> None:
    """A hidden id with a local (downloaded) copy stays hidden — the filter
    runs after the hub/local merge too."""
    from makermodslab.datasets import list_all_datasets

    _make_dataset(tmp_lerobot_home, "alice/pick", episodes=2)
    with (
        patch("makermodslab.datasets.list_user_datasets", return_value=[]),
        patch("makermodslab.datasets.get_saved_custom_datasets", return_value=[]),
        patch("makermodslab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_hide_endpoint_rejects_bad_repo_id(client: TestClient, hidden_datasets_file: Path) -> None:
    resp = client.post("/datasets/hide", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


def test_hide_unhide_endpoints_round_trip(client: TestClient, hidden_datasets_file: Path) -> None:
    from makermodslab.utils.config import get_hidden_datasets

    resp = client.post("/datasets/hide", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "alice/pick"}
    assert get_hidden_datasets() == {"alice/pick"}

    resp = client.request("DELETE", "/datasets/hide", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert get_hidden_datasets() == set()


def test_hide_endpoint_invalidates_listing_cache(
    client: TestClient, tmp_lerobot_home: Path, hidden_datasets_file: Path
) -> None:
    """Hiding must drop the cached listing so the row vanishes immediately
    instead of after the TTL."""
    hub_rows = [{"repo_id": "alice/pick", "last_modified": None, "private": False}]
    with (
        patch("makermodslab.datasets.list_user_datasets", return_value=hub_rows),
        patch("makermodslab.datasets.get_saved_custom_datasets", return_value=[]),
    ):
        first = client.get("/datasets").json()
        assert [d["repo_id"] for d in first] == ["alice/pick"]

        client.post("/datasets/hide", json={"repo_id": "alice/pick"})
        second = client.get("/datasets").json()
    assert second == []


def test_pin_route_auto_unhides(
    client: TestClient,
    tmp_lerobot_home: Path,
    hidden_datasets_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-adding a hidden dataset via POST /datasets/custom removes it from the
    hidden set — otherwise the pin would land behind the filter and never show."""
    from makermodslab.utils import config as cfg
    from makermodslab.utils.config import add_hidden_dataset, get_hidden_datasets

    # Keep the pin write in tmp too.
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_DATASETS_FILE", str(tmp_path / "pins.json"))

    add_hidden_dataset("alice/pick")
    assert get_hidden_datasets() == {"alice/pick"}

    resp = client.post("/datasets/custom", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert get_hidden_datasets() == set()


# ---------------------------------------------------------------------------
# Hub dataset summary — the /datasets/info hub fallback (meta/info.json only).
# ---------------------------------------------------------------------------


def _clear_hub_dataset_info_cache() -> None:
    from makermodslab import datasets as ds

    with ds._HUB_DATASET_INFO_LOCK:
        ds._HUB_DATASET_INFO_CACHE.clear()


def _write_hub_meta(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "info.json"
    p.write_text(json.dumps(payload))
    return p


def test_get_hub_dataset_info_maps_meta(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(
        tmp_path,
        {
            "total_episodes": 12,
            "total_frames": 3600,
            "fps": 30,
            "robot_type": "so101_follower",
            "features": {
                "observation.images.front": {"dtype": "video"},
                "observation.images.wrist": {"dtype": "video"},
                "observation.state": {"dtype": "float32"},
            },
        },
    )
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        row = ds.get_hub_dataset_info("alice/pick")

    dl.assert_called_once_with("alice/pick", filename="meta/info.json", repo_type="dataset")
    assert row == {
        "repo_id": "alice/pick",
        "total_episodes": 12,
        "total_frames": 3600,
        "fps": 30,
        "robot_type": "so101_follower",
        "cameras": ["front", "wrist"],
        "tasks": [],
        "size_bytes": None,
        "source": "hub",
    }


def test_get_hub_dataset_info_excludes_non_video_camera_features(tmp_path: Path) -> None:
    """A camera-prefixed feature that isn't dtype == "video" (e.g. raw stored
    images) has no mp4 chunk for this app's video pipeline to serve, so it's
    excluded from `cameras` — the same field the Hub listing filter and viewer
    gate both key off."""
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(
        tmp_path,
        {
            "total_episodes": 4,
            "total_frames": 100,
            "fps": 30,
            "features": {
                "observation.images.front": {"dtype": "image"},
                "observation.state": {"dtype": "float32"},
            },
        },
    )
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)),
    ):
        row = ds.get_hub_dataset_info("alice/image_only")

    assert row is not None
    assert row["cameras"] == []


def test_get_hub_dataset_info_offline_returns_none() -> None:
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    with patch("makermodslab.datasets.hf_hub_offline", return_value=True):
        assert ds.get_hub_dataset_info("alice/pick") is None


def test_get_hub_dataset_info_error_degrades_and_is_not_cached() -> None:
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", side_effect=RuntimeError("hub down")) as dl,
    ):
        assert ds.get_hub_dataset_info("alice/pick") is None
        assert ds.get_hub_dataset_info("alice/pick") is None
    assert dl.call_count == 2  # the degrade is never cached


def test_get_hub_dataset_info_caches_success(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(tmp_path, {"total_episodes": 1, "total_frames": 30, "fps": 30})
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        ds.get_hub_dataset_info("alice/cached")
        ds.get_hub_dataset_info("alice/cached")
        assert dl.call_count == 1
        ds.invalidate_hub_dataset_info("alice/cached")
        ds.get_hub_dataset_info("alice/cached")
        assert dl.call_count == 2


def test_datasets_info_endpoint_hub_fallback(
    client: TestClient, tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """A dataset with no local copy gets the hub summary (source: 'hub')
    instead of a 404; a repo with neither still 404s."""
    from makermodslab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(tmp_path, {"total_episodes": 5, "total_frames": 150, "fps": 30})
    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)),
    ):
        resp = client.get("/datasets/info", params={"repo_id": "alice/hub_only"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "hub"
    assert body["total_episodes"] == 5
    assert body["size_bytes"] is None

    _clear_hub_dataset_info_cache()
    with patch("makermodslab.datasets.hf_hub_offline", return_value=True):
        resp = client.get("/datasets/info", params={"repo_id": "alice/nowhere"})
    assert resp.status_code == 404
    assert ds is not None  # keep the import referenced


def test_get_local_dataset_info_marks_source_local(tmp_lerobot_home: Path) -> None:
    from makermodslab.datasets import get_local_dataset_info

    _make_dataset(tmp_lerobot_home, "alice/local_ds", episodes=2)
    info = get_local_dataset_info("alice/local_ds")
    assert info is not None
    assert info["source"] == "local"


# ---------------------------------------------------------------------------
# On-demand Hub episode-metadata fetch — the foundation the viewer's Hub
# fallback (Tasks 3-5) builds on. Downloads meta/info.json + the small
# meta/episodes/**/*.parquet chunks (never video/data chunks themselves) into
# huggingface_hub's own cache, so the existing local-path parsing code can run
# against that snapshot dir unmodified.
# ---------------------------------------------------------------------------


def test_hub_dataset_has_video_true_and_false() -> None:
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets.get_hub_dataset_info", return_value={"cameras": ["front"]}):
        assert ds._hub_dataset_has_video("alice/pick") is True
    with patch("makermodslab.datasets.get_hub_dataset_info", return_value={"cameras": []}):
        assert ds._hub_dataset_has_video("alice/pick") is False
    with patch("makermodslab.datasets.get_hub_dataset_info", return_value=None):
        assert ds._hub_dataset_has_video("alice/pick") is False


def test_ensure_hub_episodes_root_returns_none_offline() -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=True),
        patch("makermodslab.datasets.get_hub_dataset_info") as info,
    ):
        assert ds._ensure_hub_episodes_root("alice/pick") is None
    info.assert_not_called()


def test_ensure_hub_episodes_root_returns_none_when_no_video() -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.get_hub_dataset_info", return_value={"cameras": []}),
        patch("makermodslab.datasets.hf_hub_download") as dl,
    ):
        assert ds._ensure_hub_episodes_root("alice/no_video") is None
    dl.assert_not_called()


def test_ensure_hub_episodes_root_downloads_info_and_episode_chunks(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text("{}")

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [
        "meta/info.json",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-001.parquet",
        "meta/stats.json",  # not an episodes parquet — must be skipped
        "videos/observation.images.front/chunk-000/file-000.mp4",  # not fetched here
    ]
    downloaded: list[str] = []

    def _fake_download(repo_id, filename, repo_type):  # noqa: ARG001
        downloaded.append(filename)
        return str(snapshot / filename)

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.get_hub_dataset_info", return_value={"cameras": ["front"]}),
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_download", side_effect=_fake_download),
    ):
        root = ds._ensure_hub_episodes_root("alice/pick")

    assert root == snapshot
    assert downloaded == [
        "meta/info.json",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-001.parquet",
    ]


def test_ensure_hub_episodes_root_degrades_on_fetch_failure() -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.get_hub_dataset_info", return_value={"cameras": ["front"]}),
        patch("makermodslab.datasets.hf_hub_download", side_effect=RuntimeError("hub down")),
    ):
        assert ds._ensure_hub_episodes_root("alice/pick") is None


# ---------------------------------------------------------------------------
# list_episode_summaries — Hub fallback for episode metadata (Task 3).
# ---------------------------------------------------------------------------


def test_list_episode_summaries_local_unchanged(tmp_lerobot_home: Path) -> None:
    """Existing local behavior is untouched: a local dataset never touches the
    Hub fallback."""
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"fps": 30, "features": {"observation.images.front": {"dtype": "video"}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": [["task"]],
                "length": [30],
                "videos/observation.images.front/from_timestamp": [0.0],
                "videos/observation.images.front/to_timestamp": [1.0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    with patch("makermodslab.datasets._ensure_hub_episodes_root") as hub_fetch:
        result = ds.list_episode_summaries("alice/local")

    hub_fetch.assert_not_called()
    assert result is not None
    assert result[0]["episode_index"] == 0


def test_list_episode_summaries_hub_fallback(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    """No local copy: falls back to reading the Hub-fetched snapshot root."""
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(
        json.dumps({"fps": 30, "features": {"observation.images.front": {"dtype": "video"}}})
    )
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "tasks": [["task"], ["task"]],
                "length": [30, 60],
                "videos/observation.images.front/from_timestamp": [0.0, 1.0],
                "videos/observation.images.front/to_timestamp": [1.0, 3.0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    with patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot) as hub_fetch:
        result = ds.list_episode_summaries("alice/hub_only")

    hub_fetch.assert_called_once_with("alice/hub_only")
    assert result is not None
    assert [e["episode_index"] for e in result] == [0, 1]
    assert result[1]["duration"] == 2.0  # 60 frames / 30 fps


def test_list_episode_summaries_returns_none_when_hub_fetch_fails(tmp_lerobot_home: Path) -> None:
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=None):
        assert ds.list_episode_summaries("alice/no_video") is None


def test_list_episode_summaries_hub_fallback_filters_non_video_cameras(
    tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """Hub fallback: dataset with both dtype=='video' and dtype=='image' cameras
    correctly reads only the video camera's columns, not the image camera's."""
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    # Dataset has one video camera and one image camera
    (snapshot / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.images.front": {"dtype": "video"},
                    "observation.images.raw": {"dtype": "image"},
                },
            }
        )
    )
    # Parquet contains only the video camera's columns, not the image camera's
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "tasks": [["task"], ["task"]],
                "length": [30, 60],
                "videos/observation.images.front/from_timestamp": [0.0, 1.0],
                "videos/observation.images.front/to_timestamp": [1.0, 3.0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    with patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot) as hub_fetch:
        result = ds.list_episode_summaries("alice/mixed_cameras")

    hub_fetch.assert_called_once_with("alice/mixed_cameras")
    assert result is not None
    assert len(result) == 2
    assert [e["episode_index"] for e in result] == [0, 1]
    # Verify video_offsets only contains the video camera, not the image camera
    assert result[0]["video_offsets"] == {
        "front": {"from": 0.0, "to": 1.0},
    }
    assert result[1]["video_offsets"] == {
        "front": {"from": 1.0, "to": 3.0},
    }


def test_list_episode_summaries_skips_malformed_episode_row(tmp_lerobot_home: Path) -> None:
    """A malformed episode_index (e.g. a NaN from a third-party Hub dataset
    with corrupt metadata) is skipped, not a raised exception that 500s the
    whole endpoint — the good rows still come back."""
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"fps": 30, "features": {"observation.images.front": {"dtype": "video"}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [float("nan"), 1.0],
                "tasks": [["task"], ["task"]],
                "length": [30, 60],
                "videos/observation.images.front/from_timestamp": [0.0, 1.0],
                "videos/observation.images.front/to_timestamp": [1.0, 3.0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    result = ds.list_episode_summaries("alice/local")

    assert result is not None
    assert [e["episode_index"] for e in result] == [1]


def test_get_episode_video_path_local_unchanged(tmp_lerobot_home: Path) -> None:
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"features": {"observation.images.front": {"dtype": "video"}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "videos/observation.images.front/chunk_index": [0],
                "videos/observation.images.front/file_index": [0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    video_dir = d / "videos" / "observation.images.front" / "chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").write_bytes(b"fake mp4")

    with patch("makermodslab.datasets._ensure_hub_episodes_root") as hub_fetch:
        result = ds.get_episode_video_path("alice/local", 0, "front")

    hub_fetch.assert_not_called()
    assert result == video_dir / "file-000.mp4"


def test_get_episode_video_path_returns_none_on_malformed_chunk_index(tmp_lerobot_home: Path) -> None:
    """A malformed chunk_index/file_index (e.g. a NaN from a corrupt or
    adversarial Hub dataset) 404s gracefully instead of raising."""
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"features": {"observation.images.front": {"dtype": "video"}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "videos/observation.images.front/chunk_index": [float("nan")],
                "videos/observation.images.front/file_index": [0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    assert ds.get_episode_video_path("alice/local", 0, "front") is None


def test_get_episode_video_path_hub_fallback_downloads_one_chunk(
    tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """No local copy: fetches ONLY the one video chunk file the episode/camera
    needs, via hf_hub_download — not the whole dataset."""
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.images.front": {"dtype": "video"}}})
    )
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "videos/observation.images.front/chunk_index": [0],
                "videos/observation.images.front/file_index": [0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    downloaded_video = snapshot / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    downloaded_video.parent.mkdir(parents=True)
    downloaded_video.write_bytes(b"fake mp4")

    with (
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(downloaded_video)) as dl,
    ):
        result = ds.get_episode_video_path("alice/hub_only", 0, "front")

    dl.assert_called_once_with(
        "alice/hub_only",
        filename="videos/observation.images.front/chunk-000/file-000.mp4",
        repo_type="dataset",
    )
    assert result == downloaded_video


def test_get_episode_video_path_hub_fallback_degrades_on_download_failure(
    tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.images.front": {"dtype": "video"}}})
    )
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "videos/observation.images.front/chunk_index": [0],
                "videos/observation.images.front/file_index": [0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    with (
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot),
        patch("makermodslab.datasets.hf_hub_download", side_effect=RuntimeError("network")),
    ):
        assert ds.get_episode_video_path("alice/hub_only", 0, "front") is None


# ---------------------------------------------------------------------------
# _fan_out_hub_authors — the OVERALL fan-out deadline actually bounds a hung
# author. The shared HfApi httpx client has timeout=None, so this budget is the
# ONLY timeout in the stack: a blackholed connection must be abandoned (and
# named in a warning) rather than stalling the caller.
# ---------------------------------------------------------------------------


def test_fan_out_hub_authors_bounds_a_hung_author(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the budget shrunk to 0.2s, a fast author returns while a hung author
    (blocked on an Event never set during the call) is abandoned by the deadline:
    the call returns fast, carries ONLY the fast author's result, and logs a
    warning naming the hung author."""
    from makermodslab import datasets as ds

    monkeypatch.setattr(ds, "_HUB_FANOUT_TIMEOUT_S", 0.2)

    # Never set DURING the call; released in the finally so the leaked worker
    # thread exits and pytest terminates cleanly.
    release = threading.Event()

    def call(author: str) -> str:
        if author == "fast":
            return f"result-for-{author}"
        release.wait(timeout=30)  # the hung author
        return "late"

    try:
        start = time.monotonic()
        with caplog.at_level(logging.WARNING):
            result = ds._fan_out_hub_authors(["fast", "hung"], call)
        elapsed = time.monotonic() - start

        # Bounded by the 0.2s budget, not the 30s the hung worker would take.
        assert elapsed < 3.0
        # Only the finished author's result survives; the hung one contributes nothing.
        assert result == ["result-for-fast"]
        # The timeout warning names the author that didn't finish.
        timeout_logs = [r.getMessage() for r in caplog.records if "exceeded" in r.getMessage()]
        assert timeout_logs, "expected a fan-out timeout warning"
        assert any("hung" in msg for msg in timeout_logs)
    finally:
        release.set()


def test_fan_out_hub_authors_no_timeout_when_all_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline is a ceiling, not a floor: when every author finishes well
    inside the budget, all results are returned in author order."""
    from makermodslab import datasets as ds

    monkeypatch.setattr(ds, "_HUB_FANOUT_TIMEOUT_S", 0.5)
    result = ds._fan_out_hub_authors(["a", "b", "c"], lambda author: author.upper())
    assert result == ["A", "B", "C"]


def test_get_episode_joint_series_local_unchanged(tmp_lerobot_home: Path) -> None:
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"features": {"observation.state": {"names": ["shoulder", "elbow"]}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0]}),
        episodes_dir / "file-000.parquet",
    )
    data_dir = d / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0],
                "timestamp": [0.0, 0.033],
                "observation.state": [[0.1, 0.2], [0.15, 0.25]],
            }
        ),
        data_dir / "file-000.parquet",
    )

    with patch("makermodslab.datasets._ensure_hub_episodes_root") as hub_fetch:
        result = ds.get_episode_joint_series("alice/local", 0)

    hub_fetch.assert_not_called()
    assert result is not None
    assert result["joint_names"] == ["shoulder", "elbow"]
    assert result["timestamps"] == [0.0, 0.033]


def test_get_episode_joint_series_returns_none_on_malformed_chunk_index(tmp_lerobot_home: Path) -> None:
    """A malformed data/chunk_index (e.g. a NaN from a corrupt or adversarial
    Hub dataset) 404s gracefully instead of raising."""
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"features": {"observation.state": {"names": ["shoulder"]}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": [float("nan")], "data/file_index": [0]}),
        episodes_dir / "file-000.parquet",
    )

    assert ds.get_episode_joint_series("alice/local", 0) is None


def test_get_episode_joint_series_skips_malformed_data_row(tmp_lerobot_home: Path) -> None:
    """A malformed episode_index in the *data* parquet chunk itself (as
    opposed to the meta/episodes row already covered above) is skipped, not a
    raised exception that 500s the endpoint — the well-formed frames for the
    requested episode still come back."""
    from makermodslab import datasets as ds

    d = _write_info(
        tmp_lerobot_home,
        "alice/local",
        {"features": {"observation.state": {"names": ["shoulder", "elbow"]}}},
    )
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0]}),
        episodes_dir / "file-000.parquet",
    )
    data_dir = d / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [float("nan"), 0.0],
                "timestamp": [0.0, 0.033],
                "observation.state": [[0.1, 0.2], [0.15, 0.25]],
            }
        ),
        data_dir / "file-000.parquet",
    )

    result = ds.get_episode_joint_series("alice/local", 0)

    assert result is not None
    assert result["timestamps"] == [0.033]
    assert result["values"] == [[0.15, 0.25]]


def test_get_episode_joint_series_hub_fallback_downloads_one_chunk(
    tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.state": {"names": ["shoulder"]}}})
    )
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0]}),
        episodes_dir / "file-000.parquet",
    )
    downloaded_data = snapshot / "data" / "chunk-000" / "file-000.parquet"
    downloaded_data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "timestamp": [0.0], "observation.state": [[0.5]]}),
        downloaded_data,
    )

    with (
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(downloaded_data)) as dl,
    ):
        result = ds.get_episode_joint_series("alice/hub_only", 0)

    dl.assert_called_once_with(
        "alice/hub_only", filename="data/chunk-000/file-000.parquet", repo_type="dataset"
    )
    assert result is not None
    assert result["joint_names"] == ["shoulder"]
    assert result["values"] == [[0.5]]


def test_get_episode_joint_series_hub_fallback_degrades_on_download_failure(
    tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    from makermodslab import datasets as ds

    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(json.dumps({"features": {}}))
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0]}),
        episodes_dir / "file-000.parquet",
    )

    with (
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=snapshot),
        patch("makermodslab.datasets.hf_hub_download", side_effect=RuntimeError("network")),
    ):
        assert ds.get_episode_joint_series("alice/hub_only", 0) is None


# ---------------------------------------------------------------------------
# End-to-end route coverage for Hub dataset viewer — Tasks 1–5 integrated
# through the real FastAPI routes without production code changes.
# ---------------------------------------------------------------------------


def test_hub_dataset_viewer_endpoints_end_to_end(
    client: TestClient, tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """A Hub-only dataset with video is viewable through all three viewer
    routes without ever being downloaded locally."""
    _clear_hub_dataset_info_cache()
    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 2,
        "features": {
            "observation.images.front": {"dtype": "video"},
            "observation.state": {"names": ["shoulder"]},
        },
    }
    (snapshot / "meta" / "info.json").write_text(json.dumps(info))
    episodes_dir = snapshot / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": [["pick"]],
                "length": [2],
                "videos/observation.images.front/chunk_index": [0],
                "videos/observation.images.front/file_index": [0],
                "videos/observation.images.front/from_timestamp": [0.0],
                "videos/observation.images.front/to_timestamp": [0.066],
                "data/chunk_index": [0],
                "data/file_index": [0],
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    video_file = snapshot / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video_file.parent.mkdir(parents=True)
    video_file.write_bytes(b"fake mp4 bytes")
    data_file = snapshot / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 0], "timestamp": [0.0, 0.033], "observation.state": [[0.1], [0.2]]}),
        data_file,
    )

    def _fake_download(repo_id, filename, repo_type):  # noqa: ARG001
        return str(snapshot / filename)

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [
        "meta/info.json",
        "meta/episodes/chunk-000/file-000.parquet",
    ]

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=fake_api),
        patch("makermodslab.datasets.hf_hub_download", side_effect=_fake_download),
    ):
        episodes = client.get("/datasets/episodes", params={"repo_id": "alice/hub_only"})
        assert episodes.status_code == 200
        assert episodes.json()[0]["episode_index"] == 0

        joints = client.get(
            "/datasets/episode-joints", params={"repo_id": "alice/hub_only", "episode_index": 0}
        )
        assert joints.status_code == 200
        assert joints.json()["joint_names"] == ["shoulder"]

        video = client.get(
            "/datasets/episode-video",
            params={"repo_id": "alice/hub_only", "episode_index": 0, "camera": "front"},
        )
        assert video.status_code == 200
        assert video.content == b"fake mp4 bytes"


def test_hub_dataset_viewer_endpoints_404_without_video(
    client: TestClient, tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """A Hub-only dataset with no dtype=="video" feature never triggers a chunk
    fetch — /datasets/episodes 404s after only the cheap meta/info.json probe."""
    _clear_hub_dataset_info_cache()
    meta = tmp_path / "info.json"
    meta.write_text(json.dumps({"features": {"observation.images.front": {"dtype": "image"}}}))

    with (
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        resp = client.get("/datasets/episodes", params={"repo_id": "alice/no_video"})

    assert resp.status_code == 404
    dl.assert_called_once()  # only the meta/info.json probe inside get_hub_dataset_info
