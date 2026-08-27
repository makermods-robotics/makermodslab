"""The ``datasets`` namespace: the dataset library — listing, info cards, the
episode viewer, Hub edits, download/upload/import/merge/delete.

Response models mirror makermodslab/schemas/datasets.py (shared shapes like
DownloadStatus are defined here once and reused by the models namespace, the
same way the server shares them). SDK models are ``extra="allow"`` — an older
SDK against a newer server keeps working, extra keys stay readable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel
from makermodslab_sdk.resources._waiting import wait_for_repo_operation


class DatasetListItem(SdkModel):
    """One row of the library listing. ``source`` says where the copy lives
    ("local", "hub", or "both"); ``saved_custom`` appears only on pinned rows."""

    repo_id: str
    last_modified: str | None
    private: bool
    source: Literal["local", "hub", "both"]
    saved_custom: bool | None = None


class DatasetTaskCount(SdkModel):
    """One task of a dataset; num_episodes is 0 when the per-episode count is unavailable."""

    task: str
    num_episodes: int


class DatasetInfo(SdkModel):
    """A dataset's info card. A hub-only dataset degrades ``tasks`` to [] and
    ``size_bytes`` to None (the repo isn't on disk)."""

    repo_id: str
    total_episodes: int
    total_frames: int
    fps: int | float | None
    robot_type: str | None
    cameras: list[str]
    tasks: list[DatasetTaskCount]
    size_bytes: int | None
    source: Literal["local", "hub"]


class EpisodeSummary(SdkModel):
    """One episode of a local dataset. ``video_offsets`` maps camera name ->
    ``{"from": seconds, "to": seconds}`` slice bounds within the mp4."""

    episode_index: int
    length: int
    duration: float
    tasks: list[str]
    video_offsets: dict[str, dict[str, float]]


class EpisodeJointSeries(SdkModel):
    """Per-frame timestamps plus one observation.state vector per frame."""

    joint_names: list[str]
    timestamps: list[float]
    values: list[list[float]]


class DatasetHubStatus(SdkModel):
    """Where a dataset lives relative to the Hub; ``url`` is set only for on_hub."""

    repo_id: str
    status: Literal["on_hub", "local_only", "absent", "unknown"]
    url: str | None


class DatasetHubSettings(SdkModel):
    """The Hub repo's current visibility and tags."""

    repo_id: str
    private: bool
    tags: list[str]


class DatasetVisibility(SdkModel):
    """The visibility actually set on the Hub repo."""

    repo_id: str
    private: bool


class DatasetTags(SdkModel):
    """``tags`` is the final list actually written (org tags re-added), not the input."""

    repo_id: str
    tags: list[str]


class DatasetRenameResult(SdkModel):
    """``hub`` reports what happened to the Hub copy — "skipped" means a Hub
    copy, if any, kept its old name."""

    success: bool
    repo_id: str
    hub: Literal["renamed", "none", "skipped"]


class SuccessRepoId(SdkModel):
    """The pin/hide mutations' shared result — ``success`` is False when a
    remove/unhide had nothing to remove."""

    success: bool
    repo_id: str


class DownloadStart(SdkModel):
    """A Hub download accepted into the background slot (refusals raise instead)."""

    started: bool
    repo_id: str
    message: str


class DownloadStatus(SdkModel):
    """The single download slot: every field but ``state`` is None until a
    download has run or failed."""

    state: Literal["idle", "running", "done", "error"]
    repo_id: str | None
    message: str | None
    error: str | None


class ImportResult(SdkModel):
    """The repo id the imported copy landed under."""

    repo_id: str


class MergeStart(SdkModel):
    """Unlike downloads, a refused merge returns ``started=False`` with the
    reason in ``message`` (still HTTP 200) — check ``started``."""

    started: bool
    message: str


class MergeLogEntry(SdkModel):
    """One drained line of the merge subprocess log."""

    timestamp: float
    message: str


class MergeStatus(SdkModel):
    """The single merge slot; ``error``/``output_repo_id``/``log_path`` are
    None outside their states."""

    state: Literal["idle", "running", "done", "error"]
    error: str | None
    output_repo_id: str | None
    log_path: str | None
    logs: list[MergeLogEntry]


class UploadStart(SdkModel):
    """A Hub upload accepted into the background slot (refusals 409 instead)."""

    started: bool
    repo_id: str
    message: str


class UploadStatus(SdkModel):
    """The single upload slot. On an auth failure ``docs_url`` points at the
    Hub token docs; failure text is in ``message`` (there is no ``error`` field)."""

    state: Literal["idle", "running", "done", "error"]
    repo_id: str | None
    message: str | None
    dataset_url: str | None
    docs_url: str | None = None


class DeleteDatasetResult(SdkModel):
    """``success=False`` carries the refusal reason in ``message`` (still HTTP 200)."""

    success: bool
    message: str


class DatasetsResource(Resource):
    """``client.datasets`` — the dataset library.

    Example:
        >>> [d.repo_id for d in client.datasets.list() if d.source == "local"]
        ['maker/pick-place']
    """

    # ------------------------------------------------------------- listings

    @operation("datasets_list")
    def list(self) -> list[DatasetListItem]:
        """Every dataset the server can see — local cache and Hub account merged.

        Example:
            >>> rows = client.datasets.list()
            >>> rows[0].repo_id, rows[0].source
            ('maker/pick-place', 'both')
        """
        data = self._transport.request("GET", "/api/v1/datasets", action="List datasets")
        return [DatasetListItem.model_validate(row) for row in data]

    @operation("datasets_info")
    def info(self, repo_id: str) -> DatasetInfo:
        """The info card for one dataset (local copy preferred, Hub fallback).

        Example:
            >>> info = client.datasets.info("maker/pick-place")
            >>> info.total_episodes, info.fps, [t.task for t in info.tasks]
            (12, 30, ['pick the cube'])
        """
        return DatasetInfo.model_validate(
            self._transport.request(
                "GET", "/api/v1/datasets/info", params={"repo_id": repo_id}, action="Get dataset info"
            )
        )

    @operation("datasets_episodes")
    def episodes(self, repo_id: str) -> list[EpisodeSummary]:
        """Per-episode summaries of a local dataset (lengths, tasks, video slices).

        Example:
            >>> eps = client.datasets.episodes("maker/pick-place")
            >>> eps[0].episode_index, eps[0].duration
            (0, 13.3)
        """
        data = self._transport.request(
            "GET", "/api/v1/datasets/episodes", params={"repo_id": repo_id}, action="List dataset episodes"
        )
        return [EpisodeSummary.model_validate(row) for row in data]

    @operation("datasets_episode_joints")
    def episode_joints(self, repo_id: str, episode_index: int) -> EpisodeJointSeries:
        """One episode's joint trajectory: per-frame timestamps + state vectors.

        Example:
            >>> series = client.datasets.episode_joints("maker/pick-place", 0)
            >>> series.joint_names[0], len(series.timestamps)
            ('shoulder_pan', 400)
        """
        return EpisodeJointSeries.model_validate(
            self._transport.request(
                "GET",
                "/api/v1/datasets/episode-joints",
                params={"repo_id": repo_id, "episode_index": episode_index},
                action="Get episode joint series",
            )
        )

    # ------------------------------------------------------------- Hub edits

    @operation("datasets_hub_status")
    def hub_status(self, repo_id: str) -> DatasetHubStatus:
        """Whether a dataset exists on the Hub, only locally, or nowhere.

        Example:
            >>> client.datasets.hub_status("maker/pick-place").status
            'on_hub'
        """
        return DatasetHubStatus.model_validate(
            self._transport.request(
                "GET",
                "/api/v1/datasets/hub-status",
                params={"repo_id": repo_id},
                action="Get dataset hub status",
            )
        )

    @operation("datasets_hub_settings")
    def hub_settings(self, repo_id: str) -> DatasetHubSettings:
        """The Hub repo's current visibility and tag list.

        Example:
            >>> s = client.datasets.hub_settings("maker/pick-place")
            >>> s.private, s.tags
            (False, ['lerobot', 'so101'])
        """
        return DatasetHubSettings.model_validate(
            self._transport.request(
                "GET",
                "/api/v1/datasets/hub-settings",
                params={"repo_id": repo_id},
                action="Get dataset hub settings",
            )
        )

    @operation("datasets_visibility")
    def visibility(self, repo_id: str, *, private: bool) -> DatasetVisibility:
        """Make the Hub dataset repo private or public.

        Example:
            >>> client.datasets.visibility("maker/pick-place", private=True).private
            True
        """
        return DatasetVisibility.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/datasets/visibility",
                json={"repo_id": repo_id, "private": private},
                action="Set dataset visibility",
            )
        )

    @operation("datasets_tags")
    def tags(self, repo_id: str, tags: list[str]) -> DatasetTags:
        """Replace the Hub dataset repo's tags; returns the list actually
        written (protected org tags are re-added server-side).

        Example:
            >>> client.datasets.tags("maker/pick-place", ["so101", "demo"]).tags
            ['lerobot', 'so101', 'demo']
        """
        return DatasetTags.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/datasets/tags",
                json={"repo_id": repo_id, "tags": tags},
                action="Set dataset tags",
            )
        )

    @operation("datasets_rename")
    def rename(self, repo_id: str, new_name: str) -> DatasetRenameResult:
        """Rename a local dataset (``new_name`` is the bare name, no owner);
        ``hub`` in the result says what happened to any Hub copy.

        Example:
            >>> client.datasets.rename("maker/pick-place", "pick-place-v2").repo_id
            'maker/pick-place-v2'
        """
        return DatasetRenameResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/datasets/rename",
                json={"repo_id": repo_id, "new_name": new_name},
                action="Rename dataset",
            )
        )

    # ------------------------------------------------------------- pins & hides

    @operation("datasets_save_custom")
    def save_custom(self, repo_id: str) -> SuccessRepoId:
        """Pin any Hub dataset into the library listing (a bookmark, no download).

        Example:
            >>> client.datasets.save_custom("lerobot/svla_so101_pickplace").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "POST", "/api/v1/datasets/custom", json={"repo_id": repo_id}, action="Pin custom dataset"
            )
        )

    @operation("datasets_remove_custom")
    def remove_custom(self, repo_id: str) -> SuccessRepoId:
        """Remove a pinned dataset from the listing (``success=False`` when it
        wasn't pinned).

        Example:
            >>> client.datasets.remove_custom("lerobot/svla_so101_pickplace").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "DELETE",
                "/api/v1/datasets/custom",
                json={"repo_id": repo_id},
                action="Unpin custom dataset",
            )
        )

    @operation("datasets_hide")
    def hide(self, repo_id: str) -> SuccessRepoId:
        """Hide a dataset from the library listing (UI-level, nothing is deleted).

        Example:
            >>> client.datasets.hide("maker/old-attempt").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "POST", "/api/v1/datasets/hide", json={"repo_id": repo_id}, action="Hide dataset"
            )
        )

    @operation("datasets_unhide")
    def unhide(self, repo_id: str) -> SuccessRepoId:
        """Bring a hidden dataset back into the listing (``success=False`` when
        it wasn't hidden).

        Example:
            >>> client.datasets.unhide("maker/old-attempt").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "DELETE", "/api/v1/datasets/hide", json={"repo_id": repo_id}, action="Unhide dataset"
            )
        )

    # ------------------------------------------------------------- transfers

    @operation("datasets_download")
    def download(self, repo_id: str) -> DownloadStart:
        """Start downloading a Hub dataset into the local cache (background;
        one download at a time). Poll ``download_status()`` or block with
        ``wait_for_download()``.

        Example:
            >>> client.datasets.download("lerobot/svla_so101_pickplace").started
            True
        """
        return DownloadStart.model_validate(
            self._transport.request(
                "POST", "/api/v1/datasets/download", json={"repo_id": repo_id}, action="Download dataset"
            )
        )

    @operation("datasets_download_status")
    def download_status(self) -> DownloadStatus:
        """The single dataset-download slot's state (idle/running/done/error).

        Example:
            >>> s = client.datasets.download_status()
            >>> s.state, s.repo_id
            ('running', 'lerobot/svla_so101_pickplace')
        """
        return DownloadStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/datasets/download-status", action="Get dataset download status"
            )
        )

    @operation("upload_dataset")
    def upload(self, repo_id: str, *, private: bool = False, tags: list[str] | None = None) -> UploadStart:
        """Start pushing a local dataset to the Hub (background; one upload at
        a time). Poll ``upload_status()`` or block with ``wait_for_upload()``.

        Example:
            >>> client.datasets.upload("maker/pick-place", private=True, tags=["so101"]).started
            True
        """
        return UploadStart.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/upload-dataset",
                json={"dataset_repo_id": repo_id, "private": private, "tags": tags or []},
                action="Upload dataset",
            )
        )

    @operation("upload_status")
    def upload_status(self) -> UploadStatus:
        """The single dataset-upload slot's state; on failure the reason is in
        ``message`` (and ``docs_url`` for auth problems).

        Example:
            >>> s = client.datasets.upload_status()
            >>> s.state, s.dataset_url
            ('done', 'https://huggingface.co/datasets/maker/pick-place')
        """
        return UploadStatus.model_validate(
            self._transport.request("GET", "/api/v1/upload-status", action="Get dataset upload status")
        )

    @operation("datasets_import")
    def import_local(self, path: str, *, name: str | None = None) -> ImportResult:
        """Copy a LeRobot dataset folder from a path on the server's disk into
        the library (``name`` overrides the folder name).

        Example:
            >>> client.datasets.import_local("/data/exported/pick", name="pick-v1").repo_id
            'local/pick-v1'
        """
        return ImportResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/datasets/import",
                json={"path": path, "name": name},
                action="Import dataset from path",
            )
        )

    @operation("datasets_merge")
    def merge(self, source_repo_ids: list[str], output_repo_id: str) -> MergeStart:
        """Start merging local datasets into a new one (background subprocess).
        A refusal comes back ``started=False`` with the reason in ``message`` —
        check ``started``; poll ``merge_status()`` for progress.

        Example:
            >>> client.datasets.merge(["maker/a", "maker/b"], "maker/merged").started
            True
        """
        return MergeStart.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/datasets/merge",
                json={"source_repo_ids": source_repo_ids, "output_repo_id": output_repo_id},
                action="Merge datasets",
            )
        )

    @operation("datasets_merge_status")
    def merge_status(self) -> MergeStatus:
        """The single merge slot's state, with the drained subprocess log.

        Example:
            >>> s = client.datasets.merge_status()
            >>> s.state, s.output_repo_id
            ('done', 'maker/merged')
        """
        return MergeStatus.model_validate(
            self._transport.request("GET", "/api/v1/datasets/merge/status", action="Get merge status")
        )

    @operation("delete_dataset")
    def delete(self, repo_id: str) -> DeleteDatasetResult:
        """Delete a dataset's local copy (the Hub copy, if any, is untouched).
        A refusal comes back ``success=False`` with the reason in ``message``.

        Example:
            >>> client.datasets.delete("maker/old-attempt").success
            True
        """
        return DeleteDatasetResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/delete-dataset",
                json={"dataset_repo_id": repo_id},
                action="Delete local dataset",
            )
        )

    # ---------------------------------------------------- waiters (ergonomics, not operations)

    def wait_for_download(
        self,
        repo_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> DownloadStatus:
        """Block until the Hub download of ``repo_id`` finishes; return the
        final status. Raises OperationFailedError if it failed (or was never
        the one running), WaitTimeoutError if still running after ``timeout``
        seconds — the download itself keeps going server-side.

        Example:
            >>> client.datasets.download("lerobot/svla_so101_pickplace")
            >>> client.datasets.wait_for_download("lerobot/svla_so101_pickplace").state
            'done'
        """
        return wait_for_repo_operation(
            self.download_status,
            repo_id=repo_id,
            describe="dataset download",
            status_call="client.datasets.download_status()",
            timeout=timeout,
            poll_interval=poll_interval,
            sleep_fn=sleep_fn,
        )

    def wait_for_upload(
        self,
        repo_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> UploadStatus:
        """Block until the Hub upload of ``repo_id`` finishes; return the final
        status (``dataset_url`` points at the pushed repo). Raises
        OperationFailedError on failure, WaitTimeoutError if still running
        after ``timeout`` seconds — the upload keeps going server-side.

        Example:
            >>> client.datasets.upload("maker/pick-place")
            >>> client.datasets.wait_for_upload("maker/pick-place").dataset_url
            'https://huggingface.co/datasets/maker/pick-place'
        """
        return wait_for_repo_operation(
            self.upload_status,
            repo_id=repo_id,
            describe="dataset upload",
            status_call="client.datasets.upload_status()",
            timeout=timeout,
            poll_interval=poll_interval,
            sleep_fn=sleep_fn,
        )
