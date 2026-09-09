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

"""Tests for makermodslab.runners._dataset pure helpers."""

from __future__ import annotations

import makermodslab.merge as merge
import makermodslab.runners._dataset as rd
from makermodslab.datasets import _lerobot_cache_root
from makermodslab.merge_manifest import build_merge_manifest, write_merge_manifest


def test_upload_plan_private_for_temporary_merge(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    write_merge_manifest(
        cache / "ns/mix",
        build_merge_manifest(["ns/a", "ns/b"], [1, 2], [3, 4], temporary=True, created_at=1.0),
    )
    monkeypatch.setattr(rd, "_local_dataset_dir", lambda rid: cache / rid)
    private, is_temp = rd._dataset_upload_plan("ns/mix", "user/mix")
    assert private is True and is_temp is True


def test_upload_plan_public_for_plain_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rd, "_local_dataset_dir", lambda rid: tmp_path / "nope" / rid)
    private, is_temp = rd._dataset_upload_plan("ns/plain", "user/plain")
    assert private is False and is_temp is False


def test_local_dataset_dir_resolves_where_the_merge_wrote_the_sidecar(tmp_path, monkeypatch) -> None:
    # The merge subprocess writes meta/makermodslab_merge.json under its own
    # cache root (makermodslab.merge._lerobot_cache_root, which reads
    # $HF_LEROBOT_HOME per call). _dataset_upload_plan must read it back from
    # the same place, or a temporary merge is uploaded public.
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    assert rd._local_dataset_dir("ns/mix") == merge._lerobot_cache_root() / "ns/mix"
    assert rd._local_dataset_dir("ns/mix") == _lerobot_cache_root() / "ns/mix"


def test_upload_plan_reads_a_real_sidecar_under_the_cache_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    write_merge_manifest(
        _lerobot_cache_root() / "ns/mix",
        build_merge_manifest(["ns/a", "ns/b"], [1, 1], [3, 4], temporary=True, created_at=1.0),
    )
    private, is_temp = rd._dataset_upload_plan("ns/mix", "user/mix")
    assert private is True and is_temp is True
