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

"""Tests for makermodslab.merge_manifest — the merge sidecar's pure helpers."""

from __future__ import annotations

import json
from pathlib import Path

from makermodslab.merge_manifest import (
    MERGE_MANIFEST_FILENAME,
    MERGE_MANIFEST_SCHEMA,
    build_merge_manifest,
    parse_merge_manifest,
    read_merge_manifest,
    write_merge_manifest,
)


def test_build_records_sources_weights_and_counts() -> None:
    m = build_merge_manifest(
        ["ns/a", "ns/b", "ns/c"], [1, 3, 2], [200, 30, 100], temporary=True, created_at=123.0
    )
    assert m.schema_version == MERGE_MANIFEST_SCHEMA
    assert m.created_at == 123.0
    assert m.created_by == "makermodslab.merge"
    assert m.temporary is True
    assert m.weighted is True
    assert m.hub_repo is None
    assert [(s.repo_id, s.weight, s.episodes) for s in m.sources] == [
        ("ns/a", 1, 200),
        ("ns/b", 3, 30),
        ("ns/c", 2, 100),
    ]


def test_build_marks_unweighted_when_every_weight_is_one() -> None:
    m = build_merge_manifest(["ns/a", "ns/b"], [1, 1], [10, 20], temporary=False)
    assert m.weighted is False
    assert m.temporary is False


def test_roundtrip_through_disk(tmp_path: Path) -> None:
    m = build_merge_manifest(["ns/a", "ns/b"], [1, 2], [10, 20], temporary=True, created_at=1.0)
    write_merge_manifest(tmp_path, m)
    on_disk = json.loads((tmp_path / "meta" / MERGE_MANIFEST_FILENAME).read_text())
    assert on_disk["schema"] == MERGE_MANIFEST_SCHEMA  # serialized key is "schema", not "schema_version"
    assert read_merge_manifest(tmp_path) == m


def test_read_missing_is_none(tmp_path: Path) -> None:
    assert read_merge_manifest(tmp_path) is None


def test_parse_rejects_non_dict_and_wrong_schema() -> None:
    assert parse_merge_manifest([1, 2, 3]) is None
    assert parse_merge_manifest({"schema": 999, "sources": []}) is None
    assert parse_merge_manifest({"sources": []}) is None


def test_read_corrupt_file_is_none(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / MERGE_MANIFEST_FILENAME).write_text("{not json")
    assert read_merge_manifest(tmp_path) is None
