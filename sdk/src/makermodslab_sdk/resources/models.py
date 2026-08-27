"""The ``models`` namespace: the trained-model browser — listing, info card,
Hub download/upload, import, pin/hide, delete.

Response models mirror makermodslab/schemas/models.py; the download, pin/hide
and import shapes are imported from the datasets module the same way the
server shares them (both browsers run the very same DownloadManager class).
SDK models are ``extra="allow"`` — an older SDK against a newer server keeps
working, extra keys stay readable.
"""

from __future__ import annotations

from typing import Literal

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel
from makermodslab_sdk.resources.datasets import (
    DownloadStart,
    DownloadStatus,
    ImportResult,
    SuccessRepoId,
)

__all__ = [
    "DownloadStart",
    "DownloadStatus",
    "ImportResult",
    "ModelDeleteResult",
    "ModelInfo",
    "ModelListItem",
    "ModelUploadResult",
    "ModelsResource",
    "SuccessRepoId",
]


class ModelListItem(SdkModel):
    """One row of the model browser. Rows come from four producers with
    different key sets (defaults cover the genuinely-absent keys):

    * Hub-seeded rows add ``repo_id`` + ``private``;
    * local training-run rows add ``target_steps`` + ``state``
      ("done"/"interrupted");
    * downloaded/imported checkpoint rows carry neither pair;
    * ``saved_custom`` appears only on pinned rows.
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


class ModelInfo(SdkModel):
    """A model's info card: a listing row plus ``size_bytes`` (None for a Hub
    repo probed without storage info). ``source`` never says "both" here —
    each branch reports the copy it actually read."""

    id: str
    name: str
    policy_type: str | None
    dataset: str | None
    steps: int | None
    path: str | None
    hf_repo_id: str | None
    size_bytes: int | None
    source: Literal["local", "hub"]
    last_modified: str | None = None
    target_steps: int | None = None
    state: str | None = None
    private: bool | None = None


class ModelUploadResult(SdkModel):
    """The Hub repo the checkpoint landed in, and the final tag list written."""

    repo_id: str
    url: str
    tags: list[str]


class ModelDeleteResult(SdkModel):
    """A local checkpoint deletion (failures raise instead)."""

    deleted: bool
    id: str


class ModelsResource(Resource):
    """``client.models`` — the trained-model browser.

    Example:
        >>> [m.id for m in client.models.list() if m.policy_type == "act"]
        ['act_pick_place']
    """

    @operation("models_list")
    def list(self) -> list[ModelListItem]:
        """Every model the server can see — local runs, downloaded checkpoints,
        and Hub repos merged.

        Example:
            >>> rows = client.models.list()
            >>> rows[0].id, rows[0].source, rows[0].state
            ('act_pick_place', 'local', 'done')
        """
        data = self._transport.request("GET", "/api/v1/models", action="List models")
        return [ModelListItem.model_validate(row) for row in data]

    @operation("models_info")
    def info(self, id: str) -> ModelInfo:  # noqa: A002 - mirrors the API's query param name
        """The info card for one model; ``id`` is a listing row's ``id`` (a
        local run name, or a Hub repo id).

        Example:
            >>> info = client.models.info("act_pick_place")
            >>> info.policy_type, info.steps, info.size_bytes
            ('act', 20000, 210000000)
        """
        return ModelInfo.model_validate(
            self._transport.request("GET", "/api/v1/models/info", params={"id": id}, action="Get model info")
        )

    @operation("models_download")
    def download(self, repo_id: str) -> DownloadStart:
        """Start downloading a Hub policy checkpoint (background; one download
        at a time). Poll ``download_status()`` or block with
        ``wait_for_download()``.

        Example:
            >>> client.models.download("maker/act-pick-place").started
            True
        """
        return DownloadStart.model_validate(
            self._transport.request(
                "POST", "/api/v1/models/download", json={"repo_id": repo_id}, action="Download model"
            )
        )

    @operation("models_download_status")
    def download_status(self) -> DownloadStatus:
        """The single model-download slot's state (idle/running/done/error).

        Example:
            >>> s = client.models.download_status()
            >>> s.state, s.repo_id
            ('running', 'maker/act-pick-place')
        """
        return DownloadStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/models/download-status", action="Get model download status"
            )
        )

    @operation("models_upload")
    def upload(self, id: str, *, repo_id: str | None = None) -> ModelUploadResult:  # noqa: A002
        """Push a local checkpoint to the Hub, synchronously (unlike dataset
        uploads there is no background slot — the call blocks until pushed).
        ``repo_id`` overrides the default ``<user>/<model name>`` target.

        Example:
            >>> client.models.upload("act_pick_place").url
            'https://huggingface.co/maker/act_pick_place'
        """
        return ModelUploadResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/models/upload",
                json={"id": id, "repo_id": repo_id},
                action="Upload model",
            )
        )

    @operation("models_delete")
    def delete(self, id: str) -> ModelDeleteResult:  # noqa: A002
        """Delete a local checkpoint (the Hub copy, if any, is untouched).

        Example:
            >>> client.models.delete("act_old_attempt").deleted
            True
        """
        return ModelDeleteResult.model_validate(
            self._transport.request(
                "POST", "/api/v1/models/delete", json={"id": id}, action="Delete local model"
            )
        )

    @operation("models_import")
    def import_local(self, path: str, *, name: str | None = None) -> ImportResult:
        """Copy a policy checkpoint folder from a path on the server's disk
        into the browser (``name`` overrides the folder name).

        Example:
            >>> client.models.import_local("/data/checkpoints/act", name="act-v1").repo_id
            'local/act-v1'
        """
        return ImportResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/models/import",
                json={"path": path, "name": name},
                action="Import model from path",
            )
        )

    @operation("models_save_custom")
    def save_custom(self, repo_id: str) -> SuccessRepoId:
        """Pin any Hub model into the browser listing (a bookmark, no download).

        Example:
            >>> client.models.save_custom("lerobot/act_base").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "POST", "/api/v1/models/custom", json={"repo_id": repo_id}, action="Pin custom model"
            )
        )

    @operation("models_remove_custom")
    def remove_custom(self, repo_id: str) -> SuccessRepoId:
        """Remove a pinned model from the listing (``success=False`` when it
        wasn't pinned).

        Example:
            >>> client.models.remove_custom("lerobot/act_base").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "DELETE", "/api/v1/models/custom", json={"repo_id": repo_id}, action="Unpin custom model"
            )
        )

    @operation("models_hide")
    def hide(self, repo_id: str) -> SuccessRepoId:
        """Hide a model from the browser listing (UI-level, nothing is deleted).

        Example:
            >>> client.models.hide("maker/act-old").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "POST", "/api/v1/models/hide", json={"repo_id": repo_id}, action="Hide model"
            )
        )

    @operation("models_unhide")
    def unhide(self, repo_id: str) -> SuccessRepoId:
        """Bring a hidden model back into the listing (``success=False`` when
        it wasn't hidden).

        Example:
            >>> client.models.unhide("maker/act-old").success
            True
        """
        return SuccessRepoId.model_validate(
            self._transport.request(
                "DELETE", "/api/v1/models/hide", json={"repo_id": repo_id}, action="Unhide model"
            )
        )
