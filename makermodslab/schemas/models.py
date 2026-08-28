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

"""Response models for the "models" route group (the trained-model browser:
listing, info card, upload/delete, pin/hide, download, import). See the
package docstring for the fidelity rules; the shape authority is always the
handler in makermodslab/models.py, named next to each model.

The download, pin/hide and import shapes are re-exported from
schemas/datasets.py: the two browsers share DownloadManager (models.py
instantiates the very class datasets.py defines) and the custom/hide/import
routes are deliberate mirrors, so one model serves both groups.

The listing and info rows are built by several distinct producers (local
training runs, downloaded/imported checkpoints, Hub repos, pins) that carry
different key sets — absent keys, never null ones — while other keys on the
same rows are legitimately null. Their routes therefore serialize with
``response_model_exclude_unset=True`` (see schemas/datasets.py for why
exclude_none can't do this job).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Shared with the datasets group — see the module docstring.
from makermodslab.schemas.datasets import (
    DownloadStartResponse,
    DownloadStatusResponse,
    ImportResponse,
    SuccessRepoIdResponse,
)

__all__ = [
    "DownloadStartResponse",
    "DownloadStatusResponse",
    "ImportResponse",
    "ModelDeleteResponse",
    "ModelInfoResponse",
    "ModelListItem",
    "ModelUploadResponse",
    "SuccessRepoIdResponse",
]


class ModelListItem(BaseModel):
    """One row of GET /models (models.py list_all_models).

    Four producers, four key sets — every one carries id/name/policy_type/
    dataset/steps/path/last_modified/hf_repo_id/source, and the rest are
    genuinely absent (never null) outside their producer:

    * Hub-seeded rows (and the pin fold) add ``repo_id`` + ``private``;
    * local training-run rows (_local_model_summary) add ``target_steps`` +
      ``state`` ("done"/"interrupted"), which the hub seed also carries as
      nulls;
    * downloaded/imported checkpoint rows (_downloaded_model_summary) carry
      neither of those pairs;
    * ``saved_custom`` exists only on rows the pin fold touched.

    The route serializes with exclude_unset so each producer's exact keys
    survive — see the module docstring.
    """

    id: str
    name: str
    policy_type: str | None
    dataset: str | None
    steps: int | None
    path: str | None
    last_modified: str | None
    hf_repo_id: str | None
    source: Literal["local", "hub", "both"]
    repo_id: str | None = None
    target_steps: int | None = None
    state: str | None = None
    private: bool | None = None
    saved_custom: bool | None = None


class ModelInfoResponse(BaseModel):
    """GET /models/info (models.py get_model_info): a listing row plus
    ``size_bytes`` (null for a hub repo probed without usedStorage).

    Same absent-key heterogeneity as the listing (the route also serializes
    with exclude_unset): a local run adds target_steps/state, the single-call
    hub branch (_hub_model_info) adds private + last_modified, and its
    file-tree fallback (_hub_model_probe) carries neither. ``repo_id`` and
    ``saved_custom`` never appear here — no producer sets them."""

    id: str
    name: str
    policy_type: str | None
    dataset: str | None
    steps: int | None
    path: str | None
    hf_repo_id: str | None
    size_bytes: int | None
    # Info never says "both": each branch reports the copy it actually read
    # (a local checkpoint, or the Hub repo) — unlike the merged listing.
    source: Literal["local", "hub"]
    last_modified: str | None = None
    target_steps: int | None = None
    state: str | None = None
    private: bool | None = None


class ModelUploadResponse(BaseModel):
    """models.py upload_local_model (success path only; failures raise) —
    the Hub repo the checkpoint landed in, and the final tag list written."""

    repo_id: str
    url: str
    tags: list[str]


class ModelDeleteResponse(BaseModel):
    """models.py delete_local_model (success path only; failures raise)."""

    deleted: bool
    id: str
