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

"""Tests for makermodslab.runners._dataset pure helpers."""

from __future__ import annotations

import makermodslab.runners._dataset as rd
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
