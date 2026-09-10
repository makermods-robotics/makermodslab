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

"""The merge sidecar: `meta/makermodslab_merge.json`.

Written into every dataset produced by `makermodslab.merge` (the subprocess),
read by the dataset browser and the job registry. One module so the writer and
every reader share a filename, a schema number, and a shape — the same reason
`SAMPLING_WEIGHT_COLUMN` lives with its sampler.

Stdlib + pydantic only: `merge.py` imports this at FastAPI boot and must not pay
for the dataset-writing stack to do it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

MERGE_MANIFEST_FILENAME = "makermodslab_merge.json"
MERGE_MANIFEST_SCHEMA = 1
_CREATED_BY = "makermodslab.merge"


class MergeManifestSource(BaseModel):
    repo_id: str
    weight: int
    episodes: int | None = None


class MergeManifest(BaseModel):
    # Serialized as "schema" on disk (see model_dump below); the Python
    # attribute avoids shadowing pydantic's own `schema`.
    schema_version: int = Field(default=MERGE_MANIFEST_SCHEMA)
    created_at: float
    created_by: str = _CREATED_BY
    weighted: bool
    temporary: bool
    hub_repo: str | None = None
    sources: list[MergeManifestSource]

    def to_disk_dict(self) -> dict:
        d = self.model_dump()
        d["schema"] = d.pop("schema_version")
        return d


def build_merge_manifest(
    sources: list[str],
    weights: list[int],
    episode_counts: list[int],
    *,
    temporary: bool,
    created_at: float | None = None,
) -> MergeManifest:
    """Assemble a manifest from the recipe the merge subprocess already has.

    `weights` and `episode_counts` are positionally aligned with `sources`
    (same contract as merge.py's own helpers). `episode_counts` may be shorter
    or padded with None if a source count could not be read — zip stops at the
    shortest, and a missing tail is recorded as episodes=None.
    """
    counts = list(episode_counts) + [None] * (len(sources) - len(episode_counts))
    return MergeManifest(
        created_at=time.time() if created_at is None else created_at,
        weighted=any(w != 1 for w in weights),
        temporary=temporary,
        hub_repo=None,
        sources=[
            MergeManifestSource(repo_id=r, weight=w, episodes=c)
            for r, w, c in zip(sources, weights, counts, strict=False)
        ],
    )


def parse_merge_manifest(raw: object) -> MergeManifest | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") != MERGE_MANIFEST_SCHEMA:
        return None
    data = dict(raw)
    data["schema_version"] = data.pop("schema")
    try:
        return MergeManifest.model_validate(data)
    except ValidationError as exc:
        logger.info("Ignoring unparsable merge manifest: %s", exc)
        return None


def _manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "meta" / MERGE_MANIFEST_FILENAME


def read_merge_manifest(dataset_dir: Path) -> MergeManifest | None:
    path = _manifest_path(dataset_dir)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("Could not read merge manifest at %s: %s", path, exc)
        return None
    return parse_merge_manifest(raw)


def write_merge_manifest(dataset_dir: Path, manifest: MergeManifest) -> None:
    path = _manifest_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest.to_disk_dict(), indent=2))
    os.replace(tmp, path)
