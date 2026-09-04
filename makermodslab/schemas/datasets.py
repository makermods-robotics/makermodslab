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

"""Response models for the "datasets" route group (the dataset library:
listing, info cards, episode viewer, Hub edits, download/upload/import/merge,
plus the record.py-owned library ops upload-dataset / upload-status /
delete-dataset). See the package docstring for the fidelity rules; the shape
authority is always the handler, named next to each model.

Some shapes here are shared with the models group on purpose —
DownloadManager (datasets.py) is instantiated by BOTH browsers, and the
custom/hide/import routes are deliberate mirrors of each other — so their
models live here once and schemas/models.py re-exports them.

Two routes in this group serve rows where a key is genuinely ABSENT on some
rows (never null) while ANOTHER key on the same row is legitimately null:
the listing's ``saved_custom`` vs its null ``last_modified``, and
upload-status's ``docs_url`` vs its null idle-state fields. exclude_none
would eat the legitimate nulls, so those routes use
``response_model_exclude_unset=True`` instead: a key present in the
handler's dict (null included) is serialized, a key the handler didn't set
is omitted — exactly the wire bytes the dict produced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = [
    "DatasetHubSettingsResponse",
    "DatasetHubStatusResponse",
    "DatasetInfoResponse",
    "DatasetListItem",
    "DatasetRenameResponse",
    "DatasetTagsResponse",
    "DatasetTaskCount",
    "DatasetVisibilityResponse",
    "DeleteDatasetResponse",
    "DownloadStartResponse",
    "DownloadStatusResponse",
    "EpisodeJointSeriesResponse",
    "EpisodeSummary",
    "ExcludedEpisodesResponse",
    "ImportResponse",
    "MergeLogEntry",
    "MergeStartResponse",
    "MergeStatusResponse",
    "SetExcludedEpisodesResponse",
    "SuccessRepoIdResponse",
    "UploadStartResponse",
    "UploadStatusResponse",
]


class DatasetListItem(BaseModel):
    """One row of GET /datasets (datasets.py list_all_datasets).

    Heterogeneous on purpose: ``saved_custom`` exists only on rows the pin
    fold touched (a pinned hub row, or a local row a pin flipped to "both") —
    absent everywhere else, never null. ``last_modified`` is null (present)
    for a pinned row with no Hub timestamp, so the route serializes with
    exclude_unset, not exclude_none.
    """

    repo_id: str
    last_modified: str | None
    private: bool
    source: Literal["local", "hub", "both"]
    saved_custom: bool | None = None


class DatasetTaskCount(BaseModel):
    """One entry of /datasets/info `tasks` (datasets.py get_local_dataset_info
    / get_hub_dataset_info).

    num_episodes is null when the count is UNKNOWN — episode metadata that
    could not be read, or a Hub summary where the per-episode files were not
    fetched. It is 0 only for a task that really is used by no episode. The two
    were previously indistinguishable, which let an unreadable file silently
    decide which task a client ranked first; clients must not sort on null."""

    task: str
    num_episodes: int | None


class DatasetInfoResponse(BaseModel):
    """datasets.py get_local_dataset_info / get_hub_dataset_info — both
    branches carry every key. The hub summary carries task STRINGS (one small
    file next to meta/info.json) with null counts, and degrades size_bytes to
    null (the repo isn't on disk). fps is `int | float` because
    it passes through from meta/info.json — a whole-number fps must stay the
    integer the file holds, not become 30.0 on the wire."""

    repo_id: str
    total_episodes: int
    total_frames: int
    fps: int | float | None
    robot_type: str | None
    cameras: list[str]
    tasks: list[DatasetTaskCount]
    size_bytes: int | None
    source: Literal["local", "hub"]


class EpisodeSummary(BaseModel):
    """One row of GET /datasets/episodes (datasets.py list_episode_summaries).
    video_offsets maps camera name -> {"from": s, "to": s} slice bounds within
    the (possibly shared) mp4; kept a plain mapping because "from" is a Python
    keyword and the inner keys are exactly those two."""

    episode_index: int
    length: int
    duration: float
    tasks: list[str]
    video_offsets: dict[str, dict[str, float]]


class EpisodeJointSeriesResponse(BaseModel):
    """datasets.py get_episode_joint_series — per-frame timestamps + one
    observation.state vector per frame."""

    joint_names: list[str]
    timestamps: list[float]
    values: list[list[float]]


class ExcludedEpisodesResponse(BaseModel):
    """server.py datasets_excluded_episodes — the episode indices left OUT of
    the training subset for one dataset. Empty list when nothing is excluded;
    the dataset itself is untouched (curation, not deletion)."""

    repo_id: str
    episode_indices: list[int]


class SetExcludedEpisodesResponse(BaseModel):
    """server.py datasets_set_excluded_episodes — echoes the set actually
    persisted (re-read from disk), not the caller's input."""

    success: bool
    repo_id: str
    episode_indices: list[int]


class DatasetHubStatusResponse(BaseModel):
    """datasets.py get_hub_status — url is null (not absent) for every status
    except on_hub, so None must NOT be excluded on this route."""

    repo_id: str
    status: Literal["on_hub", "local_only", "absent", "unknown"]
    url: str | None
    # Qualifies "on_hub": False when the repo exists but holds no dataset (a
    # half-finished upload left the empty repo behind — see
    # datasets.hub_copy_has_data); None = no claim. The response model FILTERS
    # undeclared fields, so leaving this out silently strips the emptiness
    # warning the info card and cache dialog render from it.
    hub_has_data: bool | None


class DatasetHubSettingsResponse(BaseModel):
    """datasets.py get_hub_settings (success path only; failures raise)."""

    repo_id: str
    private: bool
    tags: list[str]


class DatasetVisibilityResponse(BaseModel):
    """datasets.py set_dataset_visibility (success path only)."""

    repo_id: str
    private: bool


class DatasetTagsResponse(BaseModel):
    """datasets.py set_dataset_tags — tags is the final list actually written
    (org tags re-added), not the caller's input."""

    repo_id: str
    tags: list[str]


class DatasetRenameResponse(BaseModel):
    """server.py datasets_rename: {"success": True, **rename_local_dataset()}.
    `hub` reports what happened to the Hub copy — "skipped" means a Hub copy,
    if any, kept its old name (see rename_local_dataset)."""

    success: bool
    repo_id: str
    hub: Literal["renamed", "none", "skipped"]


class SuccessRepoIdResponse(BaseModel):
    """The pin/hide mutations' shared shape (server.py datasets_save_custom /
    datasets_remove_custom / datasets_hide / datasets_unhide and their models
    mirrors) — success is False when a remove/unhide had nothing to remove."""

    success: bool
    repo_id: str


class DownloadStartResponse(BaseModel):
    """datasets.py DownloadManager.start success branch, shared by the dataset
    and model download routes (both raise on the started=False branch, so only
    this shape reaches the wire)."""

    started: bool
    repo_id: str
    message: str


class DownloadStatusResponse(BaseModel):
    """datasets.py DownloadManager.get_status, shared by both download-status
    routes. Every field but state is null (not absent) until a download has
    run/failed, so None must NOT be excluded on these routes."""

    state: Literal["idle", "running", "done", "error"]
    repo_id: str | None
    message: str | None
    error: str | None


class ImportResponse(BaseModel):
    """datasets.py import_local_dataset / models.py import_local_model
    (success path only) — the repo id the copy landed under."""

    repo_id: str


class MergeStartResponse(BaseModel):
    """merge.py MergeManager.start — unlike the download routes, the
    started=False refusals return 200 with the reason in `message`.

    `warnings` is populated only when a merge is refused pending confirmation
    (the sources span more than one arm family): the client shows them and
    re-submits with `acknowledge_warnings=true`. Empty on every other outcome.
    """

    started: bool
    message: str
    warnings: list[str] = []


class MergeLogEntry(BaseModel):
    """One drained line of the merge subprocess log (merge.py _enqueue)."""

    timestamp: float
    message: str


class MergeStatusResponse(BaseModel):
    """merge.py MergeManager.get_status — error/output_repo_id/log_path are
    null (not absent) outside their states, so None must NOT be excluded."""

    state: Literal["idle", "running", "done", "error"]
    error: str | None
    output_repo_id: str | None
    log_path: str | None
    logs: list[MergeLogEntry]


class UploadStartResponse(BaseModel):
    """record.py UploadManager.start success branch (the route 409s on
    started=False, so only this shape reaches the wire)."""

    started: bool
    repo_id: str
    message: str


class UploadStatusResponse(BaseModel):
    """record.py UploadManager.get_status. docs_url is set only when an auth
    failure produced one (absent otherwise, never null), while the other
    nullable fields ARE null in the idle state — so the route serializes with
    exclude_unset, not exclude_none."""

    state: Literal["idle", "running", "done", "error"]
    repo_id: str | None
    message: str | None
    dataset_url: str | None
    docs_url: str | None = None


class DeleteDatasetResponse(BaseModel):
    """record.py handle_delete_dataset — success False carries the refusal
    reason in `message` (still HTTP 200)."""

    success: bool
    message: str
