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
    extra_features: tuple[str, ...] = (),
    codec: str | None = None,
    crf: int = 30,
    width: int = 640,
) -> None:
    """Write a minimal ``<cache>/<repo_id>/meta/info.json`` for the helper to read.

    ``extra_features`` adds scalar columns beyond the common set — used to model
    a coaching dataset, which carries an ``intervention`` bool that a recorded
    dataset does not.

    ``codec`` populates the per-camera ``info`` block the way a real v3.0
    dataset carries it. Left None the block is absent entirely, which models
    both a pre-3.0 dataset and a 0-episode one — and is what every test written
    before the video-format check used, so they keep exercising the paths they
    were written for."""
    features: dict = {
        "action": {"dtype": "float32", "shape": list(action_shape)},
        "observation.state": {"dtype": "float32", "shape": list(action_shape)},
    }
    for name in extra_features:
        features[name] = {"dtype": "bool", "shape": [1]}
    for cam in cameras:
        spec: dict = {"dtype": "video", "shape": [480, width, 3]}
        if codec is not None:
            spec["info"] = {
                "is_depth_map": False,
                "video.height": 480,
                "video.width": width,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
                # Encoder TUNING — lerobot ignores these when merging.
                "video.g": 2,
                "video.crf": crf,
                "video.preset": None,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {},
            }
        features[f"observation.images.{cam}"] = spec
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

# ---------------------------------------------------------------------------
# Droppable features
#
# A coaching (DAgger) dataset carries an `intervention` bool that the recorded
# demonstrations it was collected against do not. Merging the two — the whole
# point of a coaching session — therefore trips lerobot's exact-feature-set
# requirement. These cover the offer-to-drop path that resolves it.
# ---------------------------------------------------------------------------


def test_droppable_features_finds_a_column_only_one_source_has(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import merge_droppable_features

    _write_info(tmp_lerobot_home, "a/demos")
    _write_info(tmp_lerobot_home, "a/corrections", extra_features=("intervention",))
    assert merge_droppable_features(["a/demos", "a/corrections"]) == ["intervention"]


def test_droppable_features_empty_when_sources_already_agree(tmp_lerobot_home: Path) -> None:
    from makermodslab.merge import merge_droppable_features

    _write_info(tmp_lerobot_home, "a/one", extra_features=("intervention",))
    _write_info(tmp_lerobot_home, "a/two", extra_features=("intervention",))
    assert merge_droppable_features(["a/one", "a/two"]) == []


def test_droppable_features_ignores_columns_not_on_the_allowlist(tmp_lerobot_home: Path) -> None:
    """The allowlist is closed on purpose. Offering to drop any scalar the
    sources happen to disagree about would let a genuinely meaningful column be
    discarded on a shrug."""
    from makermodslab.merge import merge_droppable_features

    _write_info(tmp_lerobot_home, "a/one")
    _write_info(tmp_lerobot_home, "a/two", extra_features=("next.reward",))
    assert merge_droppable_features(["a/one", "a/two"]) == []


def test_merge_incompatibility_flags_the_intervention_column_by_default(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/demos")
    _write_info(tmp_lerobot_home, "a/corrections", extra_features=("intervention",))
    message = _merge_incompatibility(["a/demos", "a/corrections"])
    assert message is not None
    assert "intervention" in message


def test_merge_incompatibility_passes_once_the_column_is_dropped(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/demos")
    _write_info(tmp_lerobot_home, "a/corrections", extra_features=("intervention",))
    assert _merge_incompatibility(["a/demos", "a/corrections"], ["intervention"]) is None


def test_dropping_a_column_does_not_excuse_a_real_mismatch(tmp_lerobot_home: Path) -> None:
    """Dropping `intervention` must not become a way to merge datasets that
    disagree about something that matters."""
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/demos", cameras=("front",))
    _write_info(
        tmp_lerobot_home,
        "a/corrections",
        cameras=("front", "wrist"),
        extra_features=("intervention",),
    )
    message = _merge_incompatibility(["a/demos", "a/corrections"], ["intervention"])
    assert message is not None
    assert "cameras" in message


def test_merge_start_offers_to_drop_rather_than_refusing(tmp_lerobot_home: Path) -> None:
    """The ordinary case when merging coaching corrections back into their
    demos. It must read as a question, not a wall."""
    from makermodslab.merge import MergeManager, MergeRequest

    _write_dataset_tree(tmp_lerobot_home, "a/demos")
    _write_dataset_tree(tmp_lerobot_home, "a/corrections")
    _write_info(tmp_lerobot_home, "a/demos")
    _write_info(tmp_lerobot_home, "a/corrections", extra_features=("intervention",))

    result = MergeManager().start(
        MergeRequest(source_repo_ids=["a/demos", "a/corrections"], output_repo_id="a/merged")
    )
    assert result["started"] is False
    assert result["droppable_features"] == ["intervention"]
    assert "intervention" in result["message"]


def test_merge_start_still_refuses_an_unresolvable_mismatch(tmp_lerobot_home: Path) -> None:
    """No `droppable_features` key means "this is a wall, not a question" — the
    dialog renders it as a plain error."""
    from makermodslab.merge import MergeManager, MergeRequest

    _write_dataset_tree(tmp_lerobot_home, "a/demos")
    _write_dataset_tree(tmp_lerobot_home, "a/corrections")
    _write_info(tmp_lerobot_home, "a/demos", fps=30)
    _write_info(tmp_lerobot_home, "a/corrections", fps=50)

    result = MergeManager().start(
        MergeRequest(source_repo_ids=["a/demos", "a/corrections"], output_repo_id="a/merged")
    )
    assert result["started"] is False
    assert "droppable_features" not in result
    assert "frame rates" in result["message"]


def test_merge_request_defaults_to_dropping_nothing() -> None:
    """A caller that doesn't know about the field must never have a column
    silently removed from their data."""
    from makermodslab.merge import MergeRequest

    request = MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="a/out")
    assert request.drop_features == []


def test_merge_start_ignores_an_unacknowledgeable_drop_request(tmp_lerobot_home: Path) -> None:
    """Echoing back a name that isn't droppable — or isn't actually in
    disagreement — must not remove a column. Only the intersection of "the
    caller agreed" and "it is genuinely mismatched and allowlisted" is stripped."""
    from makermodslab.merge import _merge_incompatibility, merge_droppable_features

    _write_info(tmp_lerobot_home, "a/one", cameras=("front",))
    _write_info(tmp_lerobot_home, "a/two", cameras=("front", "wrist"))

    droppable = merge_droppable_features(["a/one", "a/two"])
    assert droppable == []
    # A caller asking to drop a camera gets nowhere: it isn't allowlisted, so it
    # never reaches the drop list and the camera mismatch still refuses.
    drop = [n for n in droppable if n in {"observation.images.wrist"}]
    assert drop == []
    assert _merge_incompatibility(["a/one", "a/two"], drop) is not None


# --- The droppable-column prompt claims only what it can support --------------


def test_the_prompt_claims_losslessness_only_for_our_own_coaching_datasets() -> None:
    """`intervention` is constant only because OUR runner refuses
    `record_autonomous=true`. A dataset from anywhere else can carry real
    provenance in that column, and dropping it relabels every autonomous frame
    as a human demonstration."""
    from makermodslab.merge import _droppable_prompt

    ours = _droppable_prompt(["intervention"], ["user/demos", "user/rollout_fixes_20260819"])
    assert "losslessly" in ours

    # The attribution is named, so a wrong guess is visible to the operator.
    assert "user/rollout_fixes_20260819" in ours

    foreign = _droppable_prompt(["intervention"], ["user/demos", "someone/hil_dataset"])
    assert "losslessly" not in foreign
    assert "indistinguishable" in foreign


def test_a_bare_rollout_name_still_counts_as_ours() -> None:
    """Coaching datasets are created without a namespace — see rollout.py."""
    from makermodslab.merge import _looks_like_our_coaching_dataset

    assert _looks_like_our_coaching_dataset("rollout_fixes_20260819_120000") is True
    assert _looks_like_our_coaching_dataset("user/rollout_fixes_20260819_120000") is True
    assert _looks_like_our_coaching_dataset("user/my_demos") is False


# -- video stream format ---------------------------------------------------
#
# The real pairing these model: `makermods/200ep_blue_cube_orange_box` (h264,
# recorded) against `rollout_correction_smolvla_200ep_..._194018` (av1, coached).
# On the operator's own machine those two are compatible on fps, cameras, shapes
# and — once `intervention` is dropped — feature keys, and lerobot still refuses
# them solely on `video.codec`.


def test_merge_incompatibility_flags_a_codec_mismatch(tmp_lerobot_home: Path) -> None:
    """The failure that made coaching corrections unmergeable in practice.

    Recording pins `vcodec="auto"` and gets hardware H.264; coaching takes
    lerobot's software default and gets AV1. `video.codec` is NOT one of the
    encoder keys `features_equal_for_merge` forgives, so the merge dies — and
    before this check it died in the subprocess, after the drop prompt."""
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/demos", codec="h264")
    _write_info(tmp_lerobot_home, "a/corrections", codec="av1")
    message = _merge_incompatibility(["a/demos", "a/corrections"])
    assert message is not None
    assert "h264" in message and "av1" in message
    assert "re-encoded" in message


def test_merge_incompatibility_ignores_encoder_tuning_differences(
    tmp_lerobot_home: Path,
) -> None:
    """The other half of the check, and the half that would make it a nuisance.

    crf/preset/g and friends differ freely between two honest recordings, and
    lerobot strips exactly those before comparing. If this check ever starts
    reporting them it will block merges that lerobot would have accepted."""
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", codec="h264", crf=30)
    _write_info(tmp_lerobot_home, "a/two", codec="h264", crf=23)
    assert _merge_incompatibility(["a/one", "a/two"]) is None


def test_merge_incompatibility_flags_a_resolution_mismatch(tmp_lerobot_home: Path) -> None:
    """Same block, non-codec key: `video.width` is compared too, and the message
    names the property rather than claiming a codec problem."""
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", codec="h264", width=640)
    _write_info(tmp_lerobot_home, "a/two", codec="h264", width=1280)
    message = _merge_incompatibility(["a/one", "a/two"])
    assert message is not None
    assert "width" in message
    assert "codec" not in message


def test_merge_incompatibility_skips_video_check_when_a_stream_is_unreadable(
    tmp_lerobot_home: Path,
) -> None:
    """A dataset that has never encoded a frame carries no `video.codec`.

    Guessing at a mismatch from an absent block would refuse a merge on no
    evidence, so this stays silent and lets the subprocess be the authority —
    the same treatment a Hub-only source already gets."""
    from makermodslab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/encoded", codec="av1")
    _write_info(tmp_lerobot_home, "a/empty")  # no info block at all
    assert _merge_incompatibility(["a/encoded", "a/empty"]) is None


def test_a_codec_mismatch_is_reported_instead_of_a_pointless_drop_prompt(
    tmp_lerobot_home: Path,
) -> None:
    """THE point of checking the codec here rather than in the subprocess.

    A coaching dataset differs from its demos twice: the `intervention` column
    (droppable) and the codec (not). `start()` only offers the drop when
    dropping actually makes the sources mergeable, so with the codec checked the
    operator is told the real reason — instead of being walked through
    permanently dropping a column and only then hitting a subprocess crash."""
    from makermodslab.merge import _merge_incompatibility, merge_droppable_features

    _write_info(tmp_lerobot_home, "a/demos", codec="h264")
    _write_info(tmp_lerobot_home, "a/corrections", codec="av1", extra_features=("intervention",))
    sources = ["a/demos", "a/corrections"]

    # The column is still droppable in principle...
    assert merge_droppable_features(sources) == ["intervention"]
    # ...but dropping it does NOT clear the way, which is the condition
    # `start()` requires before it will prompt.
    assert _merge_incompatibility(sources, ["intervention"]) is not None
    # And the message the operator sees names the codec, not the column.
    message = _merge_incompatibility(sources)
    assert message is not None
    assert "codecs" in message
    assert "intervention" not in message
