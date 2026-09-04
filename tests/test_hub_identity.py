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
"""How a dataset's local repo_id maps onto the Hub — the whole contract, once.

A locally-recorded dataset's repo_id is BARE (no "namespace/" prefix): its id
is its directory name under the cache root, and local directories aren't
namespaced. The Hub SDK splits down the middle on a bare id — `create_repo`
and `delete_repo` resolve it to the caller's namespace, while `repo_exists`,
`dataset_info`, `update_repo_settings`, `metadata_update`, `hf_hub_download`
and `snapshot_download` do a literal lookup and 404 on it. Every call therefore
has to agree, up front, on ONE id.

`resolve_hub_dataset_id` is the single place that decides. This module pins:

  1. the primitive's answer for every (id shape x auth state) combination,
  2. that `resolve_hub_repo_id` is exactly the primitive's `repo_id`, so the
     read path can never drift from the write path,
  3. that EVERY call site — each dataset function that talks to the Hub, in
     datasets.py, merge.py, record.py and runners/hf_cloud.py — hands the Hub
     the resolved id and not the bare one,
  4. that every HTTP endpoint reaching one of those does the same end-to-end,
  5. a static backstop (`test_no_unresolved_hub_call_sites`) that fails when
     NEW code adds a Hub call taking an unresolved id.

(3) and (5) overlap deliberately. The runtime tests prove the current call
sites behave; the static one catches the next `get_episode_action_series` —
a function hand-copied from a resolved sibling, merging clean, passing every
existing test, and quietly keeping the bug.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

# --- whoami payloads --------------------------------------------------------
#
# The shapes `cached_whoami()` returns. `roleInGroup` drives writability:
# `writable_namespaces` keeps the user's own name plus every org whose role is
# in WRITE_ROLES.

ALICE: dict[str, Any] = {"name": "alice", "orgs": []}
ALICE_IN_MYORG: dict[str, Any] = {
    "name": "alice",
    "orgs": [{"name": "myorg", "roleInGroup": "write"}],
}
ALICE_READONLY_ORG: dict[str, Any] = {
    "name": "alice",
    "orgs": [{"name": "readonly-org", "roleInGroup": "read"}],
}


def _make_dataset(root: Path, repo_id: str, episodes: int = 1) -> Path:
    """The minimal layout `_is_dataset_dir` recognizes."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    return d


# ===========================================================================
# 1. The primitive
# ===========================================================================
# resolve_hub_dataset_id is a PURE function of (repo_id, whoami payload) — no
# network, no cache, no module state — so the full matrix is cheap to pin and
# every caller's behaviour follows from it.


def test_primitive_qualifies_a_bare_id_with_the_users_namespace() -> None:
    """The core case: a locally-recorded dataset. Without this the literal-
    lookup Hub calls 404 while create_repo silently creates the repo."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("pick_place", ALICE)

    assert got.repo_id == "alice/pick_place"
    assert got.namespace == "alice"
    assert got.writable is True


def test_primitive_leaves_a_canonical_owned_id_untouched() -> None:
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("alice/pick_place", ALICE)

    assert got.repo_id == "alice/pick_place"
    assert got.namespace == "alice"
    assert got.writable is True


def test_primitive_canonicalises_casing_of_an_owned_namespace() -> None:
    """A locally-recorded "Alice/foo" must reach the Hub as the canonical
    "alice/foo" the account actually owns — a mismatched-casing call risks a
    404/502 against a namespace the user does in fact own."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("ALICE/pick_place", ALICE)

    assert got.repo_id == "alice/pick_place"
    assert got.namespace == "alice"
    assert got.writable is True


def test_primitive_canonicalises_casing_of_a_writable_org() -> None:
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("MyOrg/pick_place", ALICE_IN_MYORG)

    assert got.repo_id == "myorg/pick_place"
    assert got.namespace == "myorg"
    assert got.writable is True


def test_primitive_reports_a_third_party_id_unwritable_but_addressable() -> None:
    """A downloaded third-party dataset is ALREADY the right id — ownership is
    irrelevant to reading it. The two answers diverge here, which is the whole
    reason the primitive returns both: address it as-is, don't try to write."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("lerobot/pusht", ALICE)

    assert got.repo_id == "lerobot/pusht"
    assert got.namespace == "lerobot"
    assert got.writable is False


def test_primitive_reports_a_read_only_org_unwritable() -> None:
    """Membership isn't write access. The id still addresses fine for reads."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("readonly-org/pick", ALICE_READONLY_ORG)

    assert got.repo_id == "readonly-org/pick"
    assert got.namespace == "readonly-org"
    assert got.writable is False


def test_primitive_degrades_when_unauthenticated_bare_id() -> None:
    """No token → nothing to resolve against. Read callers degrade rather than
    raise; write callers see writable=False and skip the Hub."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("pick_place", None)

    assert got.repo_id == "pick_place"
    assert got.namespace is None
    assert got.writable is False


def test_primitive_degrades_when_unauthenticated_namespaced_id() -> None:
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("lerobot/pusht", None)

    assert got.repo_id == "lerobot/pusht"
    assert got.writable is False


def test_primitive_splits_on_the_first_slash_only() -> None:
    """`validate_dataset_repo_id` rejects a multi-slash id before any Hub call,
    but the primitive must not mangle one if it ever gets there: the namespace
    is the FIRST segment, the name is everything after it."""
    from makermodslab import datasets as ds

    got = ds.resolve_hub_dataset_id("alice/a/b", ALICE)

    assert got.repo_id == "alice/a/b"
    assert got.namespace == "alice"


def test_primitive_is_idempotent() -> None:
    """Resolving an already-resolved id is a no-op. hf_cloud's
    `_ensure_dataset_on_hub` relies on this — it is handed a resolved id and
    passes it to `hub_repo_exists`, which resolves again."""
    from makermodslab import datasets as ds

    once = ds.resolve_hub_dataset_id("pick_place", ALICE)
    twice = ds.resolve_hub_dataset_id(once.repo_id, ALICE)

    assert twice == once


def test_primitive_does_not_hit_the_network() -> None:
    """It takes the whoami payload as an argument precisely so it can't. A
    caller that already failed closed on whoami (rename) must not have a second,
    silently-degrading lookup happen underneath it."""
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets.cached_whoami") as spy:
        ds.resolve_hub_dataset_id("pick_place", ALICE)

    spy.assert_not_called()


# ===========================================================================
# 2. resolve_hub_repo_id is the primitive, not a second implementation
# ===========================================================================


@pytest.mark.parametrize(
    "repo_id,who",
    [
        ("pick_place", ALICE),
        ("alice/pick_place", ALICE),
        ("ALICE/pick_place", ALICE),
        ("MyOrg/pick_place", ALICE_IN_MYORG),
        ("lerobot/pusht", ALICE),
        ("readonly-org/pick", ALICE_READONLY_ORG),
        ("pick_place", None),
        ("lerobot/pusht", None),
    ],
)
def test_resolve_hub_repo_id_is_the_primitives_repo_id(repo_id: str, who: dict | None) -> None:
    """The anti-drift test. `resolve_hub_repo_id` must stay a thin projection of
    the primitive — the moment it grows its own rules, the read path and the
    write path can disagree about the same dataset in the same process, which
    is exactly the class of bug this whole mechanism exists to remove."""
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets.cached_whoami", return_value=who):
        assert ds.resolve_hub_repo_id(repo_id) == ds.resolve_hub_dataset_id(repo_id, who).repo_id


def test_resolve_hub_repo_id_reads_the_live_auth_state() -> None:
    """It resolves against whoever is logged in NOW. Pinned deliberately: it is
    a known limitation (a bare id can resolve to the wrong account after a
    switch), not an accident, and the caches are keyed by the RESOLVED id so
    that an answer never outlives the auth state it came from."""
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets.cached_whoami", return_value=ALICE):
        assert ds.resolve_hub_repo_id("pick") == "alice/pick"
    with patch("makermodslab.datasets.cached_whoami", return_value={"name": "bob", "orgs": []}):
        assert ds.resolve_hub_repo_id("pick") == "bob/pick"


# ===========================================================================
# 3. Call sites — every dataset function that talks to the Hub
# ===========================================================================
# One test per site, each asserting the id the Hub SDK actually RECEIVED. A
# site that regresses to the bare id fails here with the id it used.


def test_push_dataset_to_hub_pushes_under_the_resolved_id() -> None:
    """`push_to_hub` takes no repo_id argument — it reads `dataset.repo_id`.
    Inside one call, create_repo resolves a bare id and the upload_folder right
    after it doesn't, which is what strands an empty repo."""
    from makermodslab import datasets as ds

    fake_dataset = MagicMock(num_episodes=3)
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("lerobot.datasets.LeRobotDataset", return_value=fake_dataset) as ctor,
    ):
        landed = ds.push_dataset_to_hub("pick_place", tags=["t"], private=False)

    # The LOCAL cache is addressed by the bare id; the Hub by the resolved one.
    # Positional args only: the ctor also takes a `video_backend` whose value is
    # this HOST's (whether torchcodec's dylibs load here), which is nothing to
    # do with repo identity and must not make this test host-dependent.
    assert ctor.call_count == 1
    assert ctor.call_args.args == ("pick_place",)
    assert fake_dataset.repo_id == "alice/pick_place"
    assert landed == "alice/pick_place"


def test_push_dataset_to_hub_refuses_when_it_cannot_resolve() -> None:
    """An upload cannot degrade: with no token there is nowhere to push, and a
    bare create_repo would fail anyway with a far less legible error."""
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=None),
        patch("lerobot.datasets.LeRobotDataset") as ctor,
        pytest.raises(RuntimeError, match="You must be authenticated"),
    ):
        ds.push_dataset_to_hub("pick_place", tags=None, private=False)

    # Refused BEFORE loading the dataset — no 100+MB read for a doomed push.
    ctor.assert_not_called()


def test_push_dataset_to_hub_allows_a_writable_org() -> None:
    from makermodslab import datasets as ds

    fake_dataset = MagicMock(num_episodes=1)
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE_IN_MYORG),
        patch("lerobot.datasets.LeRobotDataset", return_value=fake_dataset),
    ):
        assert ds.push_dataset_to_hub("MyOrg/pick", tags=None, private=True) == "myorg/pick"


def test_hub_repo_exists_checks_the_resolved_id() -> None:
    from makermodslab import datasets as ds

    api = MagicMock()
    api.repo_exists.return_value = True
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        assert ds.hub_repo_exists("pick_place") is True

    api.repo_exists.assert_called_once_with("alice/pick_place", repo_type="dataset")


def test_get_hub_status_checks_and_links_the_resolved_id() -> None:
    from makermodslab import datasets as ds

    api = MagicMock()
    api.repo_exists.return_value = True
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.get_hub_status("pick_place")

    api.repo_exists.assert_called_once_with("alice/pick_place", repo_type="dataset")
    # The public contract is unchanged: repo_id echoes back exactly as passed,
    # while the url — which a human clicks — points at the resolved repo.
    assert result["repo_id"] == "pick_place"
    assert result["url"] == "https://huggingface.co/datasets/alice/pick_place"


def test_get_hub_settings_reads_the_resolved_id() -> None:
    from makermodslab import datasets as ds

    api = MagicMock()
    api.dataset_info.return_value = MagicMock(private=True, tags=["makermods"])
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.get_hub_settings("pick_place")

    assert api.dataset_info.call_args.args[0] == "alice/pick_place"
    assert result["repo_id"] == "pick_place"


def test_set_dataset_visibility_writes_to_the_resolved_id() -> None:
    from makermodslab import datasets as ds

    api = MagicMock()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        ds.set_dataset_visibility("pick_place", True)

    assert api.update_repo_settings.call_args.args[0] == "alice/pick_place"


def test_set_dataset_tags_writes_to_the_resolved_id() -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.metadata_update") as update,
    ):
        ds.set_dataset_tags("pick_place", ["mine"])

    assert update.call_args.args[0] == "alice/pick_place"


def test_get_hub_dataset_info_downloads_the_resolved_id(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    meta = tmp_path / "info.json"
    meta.write_text(json.dumps({"total_episodes": 1, "total_frames": 10, "fps": 30, "features": {}}))
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        ds.get_hub_dataset_info("pick_place")

    assert dl.call_args.args[0] == "alice/pick_place"


def test_read_dataset_features_downloads_the_resolved_id(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    meta = tmp_path / "info.json"
    meta.write_text(json.dumps({"features": {"action": {"names": ["a"]}}}))
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        assert ds.read_dataset_features("pick_place") == {"action": {"names": ["a"]}}

    assert dl.call_args.args[0] == "alice/pick_place"


def test_ensure_hub_episodes_root_fetches_the_resolved_id(tmp_path: Path) -> None:
    """The episode-metadata prefetch behind the Hub dataset viewer: one
    hf_hub_download for meta/info.json, a list_repo_files, then one download per
    episode chunk. All three must agree on the id."""
    from makermodslab import datasets as ds

    root = tmp_path / "snap"
    (root / "meta").mkdir(parents=True)
    info = root / "meta" / "info.json"
    info.write_text("{}")
    api = MagicMock()
    api.list_repo_files.return_value = ["meta/episodes/chunk-000/file-000.parquet"]
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets._hub_dataset_has_video", return_value=True),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(info)) as dl,
    ):
        assert ds._ensure_hub_episodes_root("pick_place") is not None

    assert api.list_repo_files.call_args.args[0] == "alice/pick_place"
    for c in dl.call_args_list:
        assert c.args[0] == "alice/pick_place", f"unresolved id in {c}"


def test_fetch_dataset_snapshot_downloads_the_resolved_id(tmp_lerobot_home: Path) -> None:
    """Download addresses the Hub by the resolved id while the local directory
    keeps the id the caller passed — the flat cache layout the listing walks."""
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.snapshot_download") as snap,
    ):
        ds._fetch_dataset_snapshot("pick_place")

    assert snap.call_args.args[0] == "alice/pick_place"
    assert snap.call_args.kwargs["local_dir"] == str(tmp_lerobot_home / "pick_place")


# --- Episode readers --------------------------------------------------------
#
# Three near-identical functions read one episode's chunk off the Hub. They are
# hand-copies of each other, which is exactly why each gets its own test: a
# fourth copy is the likeliest way this bug comes back.


def _hub_episode_dataset(tmp_path: Path, column: str) -> Path:
    """A snapshot root laid out like a v3.0 dataset with one episode, so the
    episode readers reach their Hub-download branch with real data to parse."""
    root = tmp_path / "snap"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    column: {"names": ["shoulder.pos", "gripper.pos"]},
                    # A video-dtype camera, so get_episode_video_path's camera
                    # check passes and it reaches its Hub-download branch.
                    "observation.images.front": {"dtype": "video"},
                }
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "videos/observation.images.front/chunk_index": [0],
                "videos/observation.images.front/file_index": [0],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table({"episode_index": [0, 0], "timestamp": [0.0, 0.1], column: [[1.0, 2.0], [3.0, 4.0]]}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    return root


def test_get_episode_joint_series_downloads_the_resolved_id(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    root = _hub_episode_dataset(tmp_path, "observation.state")
    chunk = root / "data" / "chunk-000" / "file-000.parquet"
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=root),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(chunk)) as dl,
    ):
        assert ds.get_episode_joint_series("pick_place", 0) is not None

    assert dl.call_args.args[0] == "alice/pick_place"


def test_get_episode_action_series_downloads_the_resolved_id(tmp_path: Path) -> None:
    """The hardware-replay reader (makermodslab/replay.py). It was hand-copied
    from get_episode_joint_series BEFORE that function was fixed, so it kept the
    bare id: replaying a Hub-only dataset 404s while its sibling works."""
    from makermodslab import datasets as ds

    root = _hub_episode_dataset(tmp_path, "action")
    chunk = root / "data" / "chunk-000" / "file-000.parquet"
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=root),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(chunk)) as dl,
    ):
        series = ds.get_episode_action_series("pick_place", 0)

    assert series is not None
    assert dl.call_args.args[0] == "alice/pick_place"


def test_get_episode_video_path_downloads_the_resolved_id(tmp_path: Path) -> None:
    from makermodslab import datasets as ds

    root = _hub_episode_dataset(tmp_path, "observation.state")
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00")
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=root),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(mp4)) as dl,
    ):
        got = ds.get_episode_video_path("pick_place", 0, "front")

    assert got is not None
    assert dl.call_args.args[0] == "alice/pick_place"


# --- Call sites outside datasets.py ----------------------------------------


def test_merge_source_snapshot_downloads_the_resolved_id(tmp_path: Path) -> None:
    """Merge pulls a Hub-only source into the local cache before aggregating."""
    from makermodslab import merge

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.merge.snapshot_download") as snap,
    ):
        merge._ensure_local_source("pick_place", tmp_path)

    assert snap.call_args.kwargs["repo_id"] == "alice/pick_place"


def test_merge_preflight_finds_a_locally_recorded_dataset_on_the_hub(tmp_path: Path) -> None:
    """The preflight goes through the shared hub_repo_exists, so a
    locally-recorded dataset that IS on the Hub stops being reported as
    "wasn't found" — the bug a bare-id existence check produced."""
    from makermodslab import merge

    api = MagicMock()
    api.repo_exists.return_value = True
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
        # Not in the local cache, so the preflight falls through to the Hub.
        patch("makermodslab.merge._load_info", return_value=None),
    ):
        assert merge._merge_source_problem(["pick_place"]) is None

    assert api.repo_exists.call_args.args[0] == "alice/pick_place"


def test_cloud_runner_pins_the_resolved_id_into_the_job_config() -> None:
    """The pod has no local cache and no namespace to guess from, so the id
    persisted in JobRecord.config — "what actually ran" — must be resolved."""
    from makermodslab import datasets as ds

    with patch("makermodslab.datasets.cached_whoami", return_value=ALICE):
        assert ds.resolve_hub_repo_id("pick_place") == "alice/pick_place"


def test_upload_manager_uploads_through_the_shared_push(tmp_lerobot_home: Path) -> None:
    """The UI's upload path must report the id the dataset LANDED under — a
    success toast linking the bare id sends the user to a 404."""
    from makermodslab import record
    from makermodslab.record import UploadManager, UploadRequest

    _make_dataset(tmp_lerobot_home, "pick_place")
    mgr = UploadManager()
    with patch.object(record, "push_dataset_to_hub", return_value="alice/pick_place") as push:
        mgr._worker(UploadRequest(dataset_repo_id="pick_place", tags=[], private=False))

    assert push.call_args.args[0] == "pick_place"
    assert mgr.state == "done", mgr.message
    assert mgr.dataset_url == "https://huggingface.co/datasets/alice/pick_place"
    assert "alice/pick_place" in (mgr.message or "")


# ===========================================================================
# 4. Rename — the second resolution path, unified
# ===========================================================================
# Rename needs a different ANSWER from the read paths (it must skip the Hub on
# a namespace it can't write to, rather than address it as-is), but it must not
# compute that answer from a second implementation. It asks the same primitive
# and reads `.writable` instead of ignoring it.


def _rename_api(exists: bool = True) -> MagicMock:
    api = MagicMock()
    api.repo_exists.side_effect = lambda rid, **_: exists and not rid.endswith("/new_name")
    return api


def test_rename_resolves_through_the_shared_primitive(tmp_lerobot_home: Path) -> None:
    """The unification test. Rename must not reimplement namespace resolution —
    it must be the primitive's answer for the same id and the same whoami."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "pick_place")
    api = _rename_api()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
        patch("makermodslab.datasets.resolve_hub_dataset_id", wraps=ds.resolve_hub_dataset_id) as spy,
    ):
        result = ds.rename_local_dataset("pick_place", "new_name")

    spy.assert_called_once_with("pick_place", ALICE)
    assert result["hub"] == "renamed"
    api.move_repo.assert_called_once_with("alice/pick_place", "alice/new_name", repo_type="dataset")


def test_rename_moves_a_bare_id_under_the_users_namespace(tmp_lerobot_home: Path) -> None:
    """A recorded-then-uploaded dataset is the common case: the local id is
    bare, the Hub copy lives under the account. Both ends must move."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "pick_place")
    api = _rename_api()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.rename_local_dataset("pick_place", "new_name")

    # The LOCAL id and the RETURNED id stay bare; qualifying is Hub-side only.
    assert result == {"repo_id": "new_name", "hub": "renamed"}
    assert (tmp_lerobot_home / "new_name").is_dir()
    assert not (tmp_lerobot_home / "pick_place").exists()


def test_rename_canonicalises_org_casing_for_both_hub_ids(tmp_lerobot_home: Path) -> None:
    """Both the source AND the target id must use whoami's spelling — a target
    built from the local directory's casing lands the repo in the wrong place."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "MyOrg/pick_place")
    api = _rename_api()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE_IN_MYORG),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.rename_local_dataset("MyOrg/pick_place", "new_name")

    api.move_repo.assert_called_once_with("myorg/pick_place", "myorg/new_name", repo_type="dataset")
    # The local path keeps its own casing — the directory is the user's own.
    assert result["repo_id"] == "MyOrg/new_name"


def test_rename_skips_the_hub_for_a_third_party_namespace(tmp_lerobot_home: Path) -> None:
    """Where the two answers must differ: a read would address `lerobot/pusht`
    as-is, but move_repo there can never succeed. The local rename still runs —
    the directory is the user's own and moving it needs no Hub permission."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "lerobot/pusht")
    api = MagicMock()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.rename_local_dataset("lerobot/pusht", "my_pusht")

    assert result == {"repo_id": "lerobot/my_pusht", "hub": "skipped"}
    api.repo_exists.assert_not_called()
    api.move_repo.assert_not_called()
    assert (tmp_lerobot_home / "lerobot" / "my_pusht").is_dir()


def test_rename_skips_the_hub_for_a_read_only_org(tmp_lerobot_home: Path) -> None:
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "readonly-org/pick")
    api = MagicMock()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE_READONLY_ORG),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.rename_local_dataset("readonly-org/pick", "mine")

    assert result["hub"] == "skipped"
    api.move_repo.assert_not_called()


def test_rename_skips_the_hub_when_unauthenticated(tmp_lerobot_home: Path) -> None:
    """Failing shut would break renaming a never-uploaded dataset while logged
    out — the case least deserving of a Hub error."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "pick_place")
    api = MagicMock()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=None),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.rename_local_dataset("pick_place", "new_name")

    assert result == {"repo_id": "new_name", "hub": "skipped"}
    api.repo_exists.assert_not_called()


def test_rename_fails_closed_when_whoami_fails_with_a_token(tmp_lerobot_home: Path) -> None:
    """A transient whoami failure must not silently fall through to a local-only
    rename that leaves a stale Hub copy under the old name. Unification must not
    quietly swap this for the read path's degrade-to-None."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "pick_place")
    with (
        patch("makermodslab.datasets.cached_whoami", side_effect=OSError("boom")),
        patch("makermodslab.datasets.shared_hf_api", return_value=MagicMock()),
        pytest.raises(ds.DatasetRenameError) as exc,
    ):
        ds.rename_local_dataset("pick_place", "new_name")

    assert exc.value.status == 502
    # Nothing moved: the local directory is untouched.
    assert (tmp_lerobot_home / "pick_place").is_dir()


def test_rename_asks_whoami_to_fail_on_error(tmp_lerobot_home: Path) -> None:
    """Pins the mechanism behind the test above: rename opts INTO the raising
    variant, unlike every read path."""
    from makermodslab import datasets as ds

    _make_dataset(tmp_lerobot_home, "pick_place")
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=None) as who,
        patch("makermodslab.datasets.shared_hf_api", return_value=MagicMock()),
    ):
        ds.rename_local_dataset("pick_place", "new_name")

    who.assert_called_once_with(fail_on_error=True)


# ===========================================================================
# 5. Endpoints — the same resolution, end to end over HTTP
# ===========================================================================
# Each endpoint whose handler reaches a resolving function. These assert the id
# that reached the Hub SDK, not just the response body: a handler that returns
# a plausible 200 while having asked the Hub about the wrong repo is precisely
# the failure the unit tests above cannot see from the outside.


def test_hub_status_endpoint_checks_the_resolved_id(client: TestClient) -> None:
    api = MagicMock()
    api.repo_exists.return_value = True
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        r = client.get("/datasets/hub-status", params={"repo_id": "pick_place"})

    assert r.status_code == 200
    assert api.repo_exists.call_args.args[0] == "alice/pick_place"
    assert r.json()["url"] == "https://huggingface.co/datasets/alice/pick_place"


def test_hub_settings_endpoint_reads_the_resolved_id(client: TestClient) -> None:
    api = MagicMock()
    api.dataset_info.return_value = MagicMock(private=False, tags=["makermods"])
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        r = client.get("/datasets/hub-settings", params={"repo_id": "pick_place"})

    assert r.status_code == 200
    assert api.dataset_info.call_args.args[0] == "alice/pick_place"


def test_visibility_endpoint_writes_to_the_resolved_id(client: TestClient) -> None:
    api = MagicMock()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        r = client.post("/datasets/visibility", json={"repo_id": "pick_place", "private": True})

    assert r.status_code == 200
    assert api.update_repo_settings.call_args.args[0] == "alice/pick_place"


def test_tags_endpoint_writes_to_the_resolved_id(client: TestClient) -> None:
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.metadata_update") as update,
    ):
        r = client.post("/datasets/tags", json={"repo_id": "pick_place", "tags": ["mine"]})

    assert r.status_code == 200
    assert update.call_args.args[0] == "alice/pick_place"


def test_datasets_info_endpoint_falls_back_to_the_resolved_hub_summary(
    client: TestClient, tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """No local copy → the card falls back to a Hub summary. A bare-id read
    404s there, so the card would claim the dataset doesn't exist at all."""
    meta = tmp_path / "info.json"
    meta.write_text(json.dumps({"total_episodes": 2, "total_frames": 20, "fps": 30, "features": {}}))
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        r = client.get("/datasets/info", params={"repo_id": "pick_place"})

    assert r.status_code == 200
    assert dl.call_args.args[0] == "alice/pick_place"


def test_episode_joints_endpoint_downloads_the_resolved_id(client: TestClient, tmp_path: Path) -> None:
    root = _hub_episode_dataset(tmp_path, "observation.state")
    chunk = root / "data" / "chunk-000" / "file-000.parquet"
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=root),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(chunk)) as dl,
    ):
        r = client.get("/datasets/episode-joints", params={"repo_id": "pick_place", "episode_index": 0})

    assert r.status_code == 200
    assert dl.call_args.args[0] == "alice/pick_place"


def test_episode_video_endpoint_downloads_the_resolved_id(client: TestClient, tmp_path: Path) -> None:
    root = _hub_episode_dataset(tmp_path, "observation.state")
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00")
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets._resolve_local_dataset_path", return_value=None),
        patch("makermodslab.datasets._ensure_hub_episodes_root", return_value=root),
        patch("makermodslab.datasets.hf_hub_download", return_value=str(mp4)) as dl,
    ):
        r = client.get(
            "/datasets/episode-video",
            params={"repo_id": "pick_place", "episode_index": 0, "camera": "front"},
        )

    assert r.status_code == 200
    assert dl.call_args.args[0] == "alice/pick_place"


def test_download_endpoint_snapshots_the_resolved_id(client: TestClient, tmp_lerobot_home: Path) -> None:
    from makermodslab import datasets as ds

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.snapshot_download") as snap,
    ):
        r = client.post("/datasets/download", json={"repo_id": "alice/pick_place"})
        assert r.status_code == 200
        # The download runs on the manager's background thread; wait it out so
        # the assertion below sees the call it made.
        assert ds.download_manager._thread is not None
        ds.download_manager._thread.join(timeout=10)

    assert snap.call_args.args[0] == "alice/pick_place"


def test_rename_endpoint_moves_the_resolved_hub_id(client: TestClient, tmp_lerobot_home: Path) -> None:
    _make_dataset(tmp_lerobot_home, "pick_place")
    api = _rename_api()
    with (
        patch("makermodslab.datasets.cached_whoami", return_value=ALICE),
        patch("makermodslab.datasets.hf_hub_offline", return_value=False),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        r = client.post("/datasets/rename", json={"repo_id": "pick_place", "new_name": "new_name"})

    assert r.status_code == 200
    assert r.json() == {"success": True, "repo_id": "new_name", "hub": "renamed"}
    api.move_repo.assert_called_once_with("alice/pick_place", "alice/new_name", repo_type="dataset")


def test_upload_endpoint_pushes_under_the_resolved_id(client: TestClient, tmp_lerobot_home: Path) -> None:
    from makermodslab import record

    _make_dataset(tmp_lerobot_home, "pick_place")
    with patch.object(record, "push_dataset_to_hub", return_value="alice/pick_place") as push:
        r = client.post("/upload-dataset", json={"dataset_repo_id": "pick_place", "tags": []})
        assert r.status_code == 200
        assert record.upload_manager._thread is not None
        record.upload_manager._thread.join(timeout=10)

    assert push.call_args.args[0] == "pick_place"
    status = client.get("/upload-status").json()
    assert status["dataset_url"] == "https://huggingface.co/datasets/alice/pick_place"


# ===========================================================================
# 6. Static backstop — no NEW unresolved call sites
# ===========================================================================


# Hub SDK entry points that take a repo id as their first positional argument
# (or as `repo_id=`) and do a LITERAL lookup — the ones that 404 on a bare id.
_HUB_CALLS = frozenset(
    {
        "hf_hub_download",
        "snapshot_download",
        "dataset_info",
        "repo_exists",
        "update_repo_settings",
        "metadata_update",
        "move_repo",
        "list_repo_files",
    }
)

# Parameter/local names that hold an id straight from a caller — bare, as the
# app received it. Passing one of these to a _HUB_CALLS function is the bug.
_UNRESOLVED_NAMES = frozenset(
    {
        "repo_id",
        "local_repo_id",
        "dataset_repo_id",
        "new_repo_id",
        "source_repo_id",
        "local_dataset_repo_id",
    }
)

# The convention that makes this checkable: an id destined for the Hub is
# either `resolve_hub_repo_id(...)` inline, or lives in a `hub_`-prefixed local.
_RESOLVED_PREFIX = "hub_"

_SCANNED = ("makermodslab/datasets.py", "makermodslab/merge.py")


def _repo_id_argument(call: ast.Call) -> ast.expr | None:
    """The expression a Hub call receives as its repo id: first positional, or
    the `repo_id=` keyword."""
    for kw in call.keywords:
        if kw.arg == "repo_id":
            return kw.value
    return call.args[0] if call.args else None


def test_no_unresolved_hub_call_sites() -> None:
    """No dataset-scoped Hub call may receive a bare, caller-supplied id.

    The backstop for the runtime tests above. `get_episode_action_series` was
    hand-copied from `get_episode_joint_series` after that function was fixed,
    kept the bare id, merged clean and passed every existing test — because no
    test knew the new function existed. This one does not need to know: it reads
    the source and fails on the SHAPE of the mistake.

    If this fails on a legitimate new call site, the fix is to resolve the id
    (or bind it to a `hub_`-prefixed local), not to widen the allowlist.
    """
    offenders: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent

    for rel in _SCANNED:
        path = repo_root / rel
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _HUB_CALLS:
                continue
            arg = _repo_id_argument(node)
            if isinstance(arg, ast.Name) and arg.id in _UNRESOLVED_NAMES:
                offenders.append(f"{rel}:{node.lineno}: {name}({arg.id}) — id was never resolved")
            elif isinstance(arg, ast.Name) and not arg.id.startswith(_RESOLVED_PREFIX):
                offenders.append(
                    f"{rel}:{node.lineno}: {name}({arg.id}) — not resolve_hub_repo_id(...) "
                    f"and not a '{_RESOLVED_PREFIX}*' local"
                )

    assert not offenders, "Hub calls addressing an unresolved dataset id:\n  " + "\n  ".join(offenders)


def test_push_dataset_to_hub_is_the_only_place_that_calls_push_to_hub() -> None:
    """`push_dataset_to_hub` documents itself as "the one place that works
    around LeRobotDataset.push_to_hub's split behaviour on a bare repo_id".
    That claim is only true while it IS the one place.

    It was not: `record_with_web_events` ended a recording with a direct
    `dataset.push_to_hub(...)` on a bare id, stranding an empty repo exactly
    the way the UploadManager path used to. Reaching for `.push_to_hub` in a
    second place is the shape of that bug, so the shape is what's banned —
    route it through `push_dataset_to_hub` instead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []

    for rel in ("makermodslab/datasets.py", "makermodslab/record.py", "makermodslab/merge.py"):
        tree = ast.parse((repo_root / rel).read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name == "push_dataset_to_hub":
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "push_to_hub"
                ):
                    offenders.append(f"{rel}:{node.lineno}: {fn.name}() calls .push_to_hub directly")

    assert not offenders, "dataset pushes bypassing push_dataset_to_hub:\n  " + "\n  ".join(offenders)


def test_the_static_backstop_actually_catches_the_bug() -> None:
    """Guard the guard. A scanner that silently matches nothing would pass
    forever, so prove it flags the exact shape it exists to flag."""
    tree = ast.parse("hf_hub_download(repo_id, filename='meta/info.json', repo_type='dataset')")
    call = tree.body[0].value  # type: ignore[attr-defined]
    arg = _repo_id_argument(call)

    assert isinstance(arg, ast.Name)
    assert arg.id in _UNRESOLVED_NAMES


def test_the_static_backstop_accepts_a_resolved_call() -> None:
    tree = ast.parse("hf_hub_download(resolve_hub_repo_id(repo_id), repo_type='dataset')")
    call = tree.body[0].value  # type: ignore[attr-defined]

    assert isinstance(_repo_id_argument(call), ast.Call)
