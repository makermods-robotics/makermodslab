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
"""Tests for makermodslab.merge — the request guards that run before any subprocess."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_info(
    cache: Path,
    repo_id: str,
    *,
    fps: int = 30,
    cameras: tuple[str, ...] = ("front", "wrist"),
    action_shape: tuple[int, ...] = (6,),
) -> None:
    """Write a minimal ``<cache>/<repo_id>/meta/info.json`` for the helper to read."""
    features: dict = {
        "action": {"dtype": "float32", "shape": list(action_shape)},
        "observation.state": {"dtype": "float32", "shape": list(action_shape)},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
        }
    meta_dir = cache / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"fps": fps, "robot_type": "so101_follower", "features": features})
    )


def test_merge_incompatibility_identical_returns_none(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist"))
    _write_info(tmp_lerobot_home, "a/two", cameras=("front", "wrist"))
    assert _merge_incompatibility(["a/one", "a/two"]) is None


def test_merge_incompatibility_different_cameras(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist"))
    _write_info(tmp_lerobot_home, "a/two", cameras=("front", "wrist", "side"))
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "side" in msg
    assert "camera" in msg.lower()


def test_merge_incompatibility_different_fps(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", fps=30)
    _write_info(tmp_lerobot_home, "a/two", fps=50)
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "30" in msg and "50" in msg
    assert "fps" in msg.lower()


def test_merge_incompatibility_different_feature_shape(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", action_shape=(6,))
    _write_info(tmp_lerobot_home, "a/two", action_shape=(7,))
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "action" in msg


def test_merge_incompatibility_skips_when_not_local(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_incompatibility

    # Only one source is present locally → can't compare → don't block.
    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist", "side"))
    assert _merge_incompatibility(["a/one", "a/hub-only"]) is None


def test_merge_rejects_fewer_than_two_sources() -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one"], output_repo_id="a/merged"))
    assert res["started"] is False
    assert "two" in res["message"].lower()
    assert mgr.state == "idle"  # no subprocess spawned


def test_merge_rejects_output_matching_a_source() -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="a/one"))
    assert res["started"] is False
    assert mgr.state == "idle"


def test_merge_rejects_blank_output() -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="  "))
    assert res["started"] is False
    assert mgr.state == "idle"


def test_merge_status_shape_when_idle() -> None:
    from makermodslab.merge import MergeManager

    status = MergeManager().get_status()
    assert status["state"] == "idle"
    assert status["output_repo_id"] is None
    assert status["log_path"] is None
    assert status["logs"] == []


def test_merge_rejects_existing_output(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    # The retry crash: a residue from an earlier failed merge already sits at
    # <cache>/<output>, so start() must refuse before spawning anything.
    (tmp_lerobot_home / "makermods" / "socks").mkdir(parents=True)

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="makermods/socks"))
    assert res["started"] is False
    assert "already exists" in res["message"]
    assert "makermods/socks" in res["message"]
    assert mgr.state == "idle"  # no subprocess spawned


def test_cli_friendly_error_maps_file_exists(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _cli_friendly_error

    # Belt-and-suspenders: a subprocess-side FileExistsError (race) becomes a
    # friendly line rather than a raw `[Errno 17] File exists` traceback.
    exc = FileExistsError(17, "File exists", str(tmp_lerobot_home / "makermods" / "socks"))
    msg = _cli_friendly_error(exc, ["a/one", "a/two"], tmp_lerobot_home)
    assert "already exists" in msg
    assert "Errno 17" not in msg


def test_merge_log_file_written(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import MergeManager

    mgr = MergeManager()
    mgr._open_log()
    try:
        assert mgr.log_path is not None
        assert "merge_logs" in mgr.log_path
        # Additive status field carries the path while a run is live.
        assert mgr.get_status()["log_path"] == mgr.log_path
        mgr._enqueue("Merging 2 datasets -> makermods/socks")
        mgr._enqueue("Cleaned up partial output (119MB).")
    finally:
        mgr._close_log()

    content = Path(mgr.log_path).read_text()
    assert "Merging 2 datasets -> makermods/socks" in content
    assert "Cleaned up partial output (119MB)." in content
    # The log lives under the redirected cache, not the real one.
    assert str(tmp_lerobot_home) in mgr.log_path


def _fake_aggregate_partial(output_root: Path):
    """A monkeypatched ``aggregate_datasets`` that writes the observed 14:13
    residue shape (meta/info.json + videos/, no completed episodes) then dies
    mid-aggregation."""

    def _aggregate(repo_ids, aggr_repo_id, roots):  # noqa: ARG001
        meta = output_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text("{}")
        (output_root / "videos").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("boom mid-aggregation")

    return _aggregate


def test_run_cli_cleans_up_partial_output_on_failure(tmp_lerobot_home: Path, monkeypatch) -> None:
    from makermodslab import merge

    output = "makermods/socks"
    output_root = tmp_lerobot_home / "makermods" / "socks"

    # No network: sources resolve to their (never-read) local roots.
    monkeypatch.setattr(merge, "_ensure_local_source", lambda repo_id, cache_root: cache_root / repo_id)
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate_partial(output_root))

    rc = merge._run_cli([output, "a/one", "a/two"])
    assert rc == 1
    # The residue this run created is gone.
    assert not output_root.exists()


def test_run_cli_leaves_preexisting_output_on_failure(tmp_lerobot_home: Path, monkeypatch) -> None:
    from makermodslab import merge

    output = "makermods/socks"
    output_root = tmp_lerobot_home / "makermods" / "socks"

    # A dir that existed BEFORE the merge — must never be removed.
    output_root.mkdir(parents=True)
    (output_root / "sentinel.txt").write_text("keep me")

    monkeypatch.setattr(merge, "_ensure_local_source", lambda repo_id, cache_root: cache_root / repo_id)
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate_partial(output_root))

    rc = merge._run_cli([output, "a/one", "a/two"])
    assert rc == 1
    # Pre-existing output is untouched.
    assert output_root.exists()
    assert (output_root / "sentinel.txt").read_text() == "keep me"


def _write_dataset_tree(cache: Path, repo_id: str, *, total_episodes: int = 1) -> None:
    """Write a fully-populated local dataset (info.json + the required files)."""
    _write_info(cache, repo_id)
    root = cache / repo_id
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "robot_type": "so101_follower",
                "total_episodes": total_episodes,
                "features": {
                    "action": {"dtype": "float32", "shape": [6]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
                    "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
                },
            }
        )
    )
    (root / "meta" / "tasks.parquet").write_text("")
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("")
    ep_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_file.parent.mkdir(parents=True, exist_ok=True)
    ep_file.write_text("")


def test_merge_source_problem_valid_returns_none(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    assert _merge_source_problem(["a/one", "a/two"]) is None


def test_merge_source_problem_missing_tasks_parquet(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    # Corrupt: remove the tasks parquet the metadata references.
    (tmp_lerobot_home / "a/two" / "meta" / "tasks.parquet").unlink()
    msg = _merge_source_problem(["a/one", "a/two"])
    assert msg is not None
    assert "a/two" in msg
    assert "tasks.parquet" in msg
    assert "incomplete" in msg.lower() or "corrupt" in msg.lower()


def test_merge_source_problem_missing_data_parquet(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    # Corrupt: total_episodes>0 but no data parquet on disk.
    (tmp_lerobot_home / "a/two" / "data" / "chunk-000" / "file-000.parquet").unlink()
    msg = _merge_source_problem(["a/one", "a/two"])
    assert msg is not None
    assert "a/two" in msg
    assert "incomplete" in msg.lower() or "corrupt" in msg.lower()


def test_merge_source_problem_not_found_on_hub(tmp_lerobot_home: Path, monkeypatch) -> None:
    """A confirmed 404 blocks the merge.

    The existence check now lives in datasets.hub_repo_exists (shared with the
    info card), so the seam patched here is the HfApi call that helper makes,
    not a merge-private one.
    """
    from unittest.mock import MagicMock

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    from makermodslab import merge

    _write_dataset_tree(tmp_lerobot_home, "a/one")

    def _raise_not_found(self, repo_id, **kwargs):
        raise RepositoryNotFoundError(f"404 for {repo_id}", response=MagicMock())

    # repo_exists() maps RepositoryNotFoundError to False itself; patch the
    # repo_info underneath it rather than stubbing repo_exists, so the taxonomy
    # under test is the real one. The shared client is an HfApi instance, so
    # patching the class reaches it.
    monkeypatch.setattr(HfApi, "repo_info", _raise_not_found)
    msg = merge._merge_source_problem(["a/one", "a/hub-only"])
    assert msg is not None
    assert "a/hub-only" in msg
    assert "found" in msg.lower()
    assert "hub" in msg.lower()


def test_merge_source_problem_empty_hub_repo_blocks_with_the_real_cause(
    tmp_lerobot_home: Path, monkeypatch
) -> None:
    """A repo left behind by a half-finished upload exists but holds no
    dataset. Letting it through fails deep in the merge subprocess with a
    misleading "incomplete or corrupt — re-record it"; the preflight must name
    the actual state instead."""
    from huggingface_hub import HfApi

    from makermodslab import merge

    _write_dataset_tree(tmp_lerobot_home, "a/one")

    monkeypatch.setattr(HfApi, "repo_info", lambda self, repo_id, **kwargs: object())
    # The emptiness probe: meta/info.json is not among the repo's paths.
    monkeypatch.setattr(HfApi, "get_paths_info", lambda self, repo_id, paths, **kwargs: [])

    msg = merge._merge_source_problem(["a/one", "a/empty-upload"])
    assert msg is not None
    assert "a/empty-upload" in msg
    assert "no data" in msg.lower()
    assert "re-upload" in msg.lower()


def test_merge_source_problem_offline_does_not_block(tmp_lerobot_home: Path, monkeypatch) -> None:
    from huggingface_hub import HfApi

    from makermodslab import merge

    _write_dataset_tree(tmp_lerobot_home, "a/one")

    def _raise_network(self, repo_id, **kwargs):
        # Simulate offline / transient connection failure.
        raise OSError("connection failed")

    monkeypatch.setattr(HfApi, "repo_info", _raise_network)
    # Offline / transient error → hub_repo_exists returns None ("no claim"),
    # which must NOT block the merge.
    assert merge._merge_source_problem(["a/one", "a/hub-only"]) is None


def test_expand_weighted_repeats_by_weight() -> None:
    from makermodslab.merge import _expand_weighted

    assert _expand_weighted(["a", "b"], [1, 3]) == ["a", "b", "b", "b"]


def test_expand_weighted_none_is_passthrough() -> None:
    from makermodslab.merge import _expand_weighted

    assert _expand_weighted(["a", "b"], None) == ["a", "b"]


def test_weights_problem_none_is_valid() -> None:
    from makermodslab.merge import _weights_problem

    assert _weights_problem(2, None) is None


def test_weights_problem_matching_lengths_is_valid() -> None:
    from makermodslab.merge import _weights_problem

    assert _weights_problem(2, [1, 1]) is None


def test_weights_problem_length_mismatch() -> None:
    from makermodslab.merge import _weights_problem

    msg = _weights_problem(2, [1, 1, 1])
    assert msg is not None
    assert "3" in msg and "2" in msg


def test_weights_problem_weight_below_one() -> None:
    from makermodslab.merge import _weights_problem

    msg = _weights_problem(2, [1, 0])
    assert msg is not None


def test_weights_problem_weight_above_max() -> None:
    from makermodslab.merge import MAX_SOURCE_WEIGHT, _weights_problem

    msg = _weights_problem(2, [1, MAX_SOURCE_WEIGHT + 1])
    assert msg is not None
    assert str(MAX_SOURCE_WEIGHT) in msg


def test_describe_weights_includes_repeat_counts() -> None:
    from makermodslab.merge import _describe_weights

    desc = _describe_weights(["a/one", "a/two"], [1, 3])
    assert "a/two x3" in desc


def test_merge_rejects_mismatched_weights_length() -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(
        MergeRequest(
            source_repo_ids=["a/one", "a/two"],
            output_repo_id="a/merged",
            source_weights=[1, 2, 3],
        )
    )
    assert res["started"] is False
    assert "3" in res["message"] and "2" in res["message"]
    assert mgr.state == "idle"  # rejected before any subprocess spawned


def test_merge_rejects_duplicate_source_uses_weight_instead() -> None:
    from makermodslab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/one"], output_repo_id="a/merged"))
    assert res["started"] is False
    assert "weight" in res["message"].lower()
    assert mgr.state == "idle"  # rejected before any subprocess spawned


def test_weights_all_ones_matches_unweighted_at_validation_layer() -> None:
    from makermodslab.merge import _expand_weighted, _weights_problem

    sources = ["a/one", "a/two"]
    assert _weights_problem(2, [1, 1]) is None
    assert _expand_weighted(sources, [1, 1]) == sources


# ---------------------------------------------------------------------------
# Sampling weights written at merge time (step 1.1).
# ---------------------------------------------------------------------------


def _write_source(cache: Path, repo_id: str, n_episodes: int) -> Path:
    """A source dataset root whose meta/info.json declares `n_episodes`."""
    _write_info(cache, repo_id)
    info_path = cache / repo_id / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = n_episodes
    info_path.write_text(json.dumps(info))
    return cache / repo_id


def _write_output_episodes(cache: Path, repo_id: str, chunks: list[list[int]]) -> Path:
    """An aggregated output whose episode rows are spread over `chunks`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = cache / repo_id
    for i, episode_indices in enumerate(chunks):
        chunk_dir = root / "meta" / "episodes" / f"chunk-{i:03d}"
        chunk_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"episode_index": episode_indices, "length": [30] * len(episode_indices)}),
            chunk_dir / "file-000.parquet",
        )
    return root


def _read_weights(root: Path) -> list[float]:
    import pyarrow.parquet as pq

    rows: list[tuple[int, float]] = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        rows.extend(
            zip(
                table.column("episode_index").to_pylist(),
                table.column("sampling_weight").to_pylist(),
                strict=True,
            )
        )
    return [weight for _, weight in sorted(rows)]


def _column_names(root: Path) -> list[str]:
    import pyarrow.parquet as pq

    path = sorted((root / "meta" / "episodes").glob("**/*.parquet"))[0]
    return list(pq.read_schema(path).names)


def test_weight_per_episode_flattens_source_weights() -> None:
    from makermodslab.merge import _weight_per_episode

    assert _weight_per_episode([2, 3], [1, 3]) == [1.0, 1.0, 3.0, 3.0, 3.0]


def test_source_episode_counts_read_each_sources_own_info(tmp_lerobot_home: Path) -> None:
    """Boundaries are read, not guessed: they decide which source's weight lands
    on which output episode."""
    from makermodslab.merge import _source_episode_counts

    roots = [
        _write_source(tmp_lerobot_home, "a/one", 2),
        _write_source(tmp_lerobot_home, "a/two", 7),
    ]
    assert _source_episode_counts(roots) == [2, 7]


def test_stamp_sampling_weights_keys_by_episode_index(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _stamp_sampling_weights

    # Episode rows deliberately split across two chunks, out of file order.
    root = _write_output_episodes(tmp_lerobot_home, "a/out", [[2, 3, 4], [0, 1]])
    stamped = _stamp_sampling_weights(root, [1.0, 1.0, 3.0, 3.0, 3.0])

    assert stamped == 5
    assert _read_weights(root) == [1.0, 1.0, 3.0, 3.0, 3.0]


def test_stamp_sampling_weights_replaces_an_existing_column(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _stamp_sampling_weights

    root = _write_output_episodes(tmp_lerobot_home, "a/out", [[0, 1]])
    _stamp_sampling_weights(root, [2.0, 2.0])
    _stamp_sampling_weights(root, [1.0, 4.0])

    assert _read_weights(root) == [1.0, 4.0]
    assert _column_names(root).count("sampling_weight") == 1


def test_stamp_sampling_weights_refuses_a_count_mismatch(tmp_lerobot_home: Path) -> None:
    """Mis-stamped weights would oversample the wrong episodes forever, so a
    layout that doesn't line up must fail the merge instead."""
    from makermodslab.merge import _stamp_sampling_weights

    # Sources account for 3 episodes but the merge only produced 2.
    root = _write_output_episodes(tmp_lerobot_home, "a/out", [[0, 1]])
    with pytest.raises(RuntimeError, match="episode rows"):
        _stamp_sampling_weights(root, [1.0, 3.0, 3.0])

    assert "sampling_weight" not in _column_names(root)  # nothing written


def test_stamp_sampling_weights_refuses_an_out_of_range_index(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import _stamp_sampling_weights

    root = _write_output_episodes(tmp_lerobot_home, "a/out", [[0, 9]])
    with pytest.raises(RuntimeError, match="outside"):
        _stamp_sampling_weights(root, [1.0, 3.0])


def _fake_aggregate(cache: Path, calls: list[list[str]]):
    """Stand-in for lerobot's aggregate_datasets: records the repo_ids it was
    handed and writes one contiguous episodes table for them."""

    def _aggregate(repo_ids, aggr_repo_id, roots):  # noqa: ARG001
        calls.append(list(repo_ids))
        total = 0
        for repo_id in repo_ids:
            info = json.loads((cache / repo_id / "meta" / "info.json").read_text())
            total += int(info["total_episodes"])
        _write_output_episodes(cache, aggr_repo_id, [list(range(total))])
        (cache / aggr_repo_id / "meta" / "info.json").write_text(json.dumps({"total_episodes": total}))

    return _aggregate


def test_run_cli_passes_each_source_once_and_stamps_the_weights(tmp_lerobot_home: Path, monkeypatch) -> None:
    """The whole point: weight 3 costs one copy on disk, not three."""
    from makermodslab import merge

    _write_source(tmp_lerobot_home, "a/one", 2)
    _write_source(tmp_lerobot_home, "a/two", 3)
    calls: list[list[str]] = []
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate(tmp_lerobot_home, calls))

    assert merge._run_cli(["a/out", "a/one", "a/two", "--weights", "1", "3"]) == 0

    assert calls == [["a/one", "a/two"]]
    assert _read_weights(tmp_lerobot_home / "a/out") == [1.0, 1.0, 3.0, 3.0, 3.0]


def test_run_cli_writes_no_weight_column_when_every_weight_is_one(
    tmp_lerobot_home: Path, monkeypatch
) -> None:
    """R2: all-1 weights are indistinguishable from no weights, so the output
    must be exactly what an unweighted merge has always produced."""
    from makermodslab import merge

    _write_source(tmp_lerobot_home, "a/one", 2)
    _write_source(tmp_lerobot_home, "a/two", 2)
    calls: list[list[str]] = []
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate(tmp_lerobot_home, calls))

    assert merge._run_cli(["a/out", "a/one", "a/two", "--weights", "1", "1"]) == 0

    assert calls == [["a/one", "a/two"]]
    assert "sampling_weight" not in _column_names(tmp_lerobot_home / "a/out")


def test_run_cli_duplicate_flag_restores_physical_expansion(tmp_lerobot_home: Path, monkeypatch) -> None:
    """The escape hatch still duplicates on disk — and writes no weights, since
    the ratio is baked into the data."""
    from makermodslab import merge

    _write_source(tmp_lerobot_home, "a/one", 2)
    _write_source(tmp_lerobot_home, "a/two", 1)
    calls: list[list[str]] = []
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate(tmp_lerobot_home, calls))

    assert merge._run_cli(["a/out", "a/one", "a/two", "--weights", "1", "3", "--duplicate"]) == 0

    assert calls == [["a/one", "a/two", "a/two", "a/two"]]
    assert "sampling_weight" not in _column_names(tmp_lerobot_home / "a/out")


def _fake_aggregate_carrying_weights(cache: Path):
    """Like _fake_aggregate, but the merged output carries a `sampling_weight`
    column — as it does when a source was itself a weighted merge and
    aggregate_datasets copies its episode rows straight through."""

    def _aggregate(repo_ids, aggr_repo_id, roots):  # noqa: ARG001
        import pyarrow as pa
        import pyarrow.parquet as pq

        total = sum(
            int(json.loads((cache / r / "meta" / "info.json").read_text())["total_episodes"])
            for r in repo_ids
        )
        _write_output_episodes(cache, aggr_repo_id, [list(range(total))])
        path = sorted((cache / aggr_repo_id / "meta" / "episodes").glob("**/*.parquet"))[0]
        table = pq.read_table(path).append_column(
            "sampling_weight", pa.array([3.0] + [1.0] * (total - 1), type=pa.float64())
        )
        pq.write_table(table, path)
        (cache / aggr_repo_id / "meta" / "info.json").write_text(json.dumps({"total_episodes": total}))

    return _aggregate


def test_run_cli_strips_sampling_weights_inherited_from_a_weighted_source(
    tmp_lerobot_home: Path, monkeypatch
) -> None:
    """M4: merging a weighted dataset without --weights must not silently produce
    a weighted output — dataset_is_weighted would then launch the weighted
    trainer for a merge the UI called plain."""
    from makermodslab import merge

    _write_source(tmp_lerobot_home, "a/one", 2)
    _write_source(tmp_lerobot_home, "a/two", 3)
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate_carrying_weights(tmp_lerobot_home))

    assert merge._run_cli(["a/out", "a/one", "a/two"]) == 0
    assert "sampling_weight" not in _column_names(tmp_lerobot_home / "a/out")


def test_run_cli_removes_the_output_when_weights_cannot_be_stored(
    tmp_lerobot_home: Path, monkeypatch
) -> None:
    """R6: handing back a merged-but-unweighted dataset would let it train on the
    wrong mix while looking correct, so the merge fails and cleans up instead."""
    from makermodslab import merge

    _write_source(tmp_lerobot_home, "a/one", 2)
    _write_source(tmp_lerobot_home, "a/two", 3)
    monkeypatch.setattr(merge, "aggregate_datasets", _fake_aggregate(tmp_lerobot_home, []))
    monkeypatch.setattr(
        merge,
        "_stamp_sampling_weights",
        lambda root, weights: (_ for _ in ()).throw(RuntimeError("disk full")),  # noqa: ARG005
    )

    assert merge._run_cli(["a/out", "a/one", "a/two", "--weights", "1", "3"]) == 1
    assert not (tmp_lerobot_home / "a/out").exists()
