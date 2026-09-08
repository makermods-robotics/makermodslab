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

"""Dataset transport shared by the remote runners.

Every remote runner resolves the training dataset by repo_id from the Hub: an
HF Jobs pod and a LAN peer alike have no view of this machine's
``~/.cache/huggingface/lerobot``. Datasets therefore travel via the Hub — a
deliberate design decision for this phase — and this module is the single
implementation of "push it there first when it only exists here", so the two
runners cannot drift on when a push happens or what it looks like.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..datasets import (
    hub_copy_has_data,
    hub_repo_exists,
    local_pushable_copy_exists,
    push_dataset_to_hub,
)
from ..utils.config import with_makermodslab_tag


def _local_dataset_dir(local_repo_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / local_repo_id


def _dataset_upload_plan(local_repo_id: str, hub_repo_id: str) -> tuple[bool, bool]:
    """(`private`, `is_temporary_merge`) for an implicit pre-run upload.

    A temporary merge is throwaway scratch, so it goes up PRIVATE — a narrow,
    deliberate exception to "implicit uploads are public" (spec §6.6). Every
    other dataset keeps the public default.
    """
    from ..merge_manifest import read_merge_manifest

    manifest = read_merge_manifest(_local_dataset_dir(local_repo_id))
    is_temp = bool(manifest and manifest.temporary)
    return (is_temp, is_temp)


def ensure_dataset_on_hub(local_repo_id: str, hub_repo_id: str, log: Callable[[str], None]) -> None:
    """If the dataset is local-only, push it to the Hub.

    The remote side resolves the dataset by repo_id; it can't see the
    host's `~/.cache/huggingface/lerobot`. We push synchronously and
    let any failure bubble up — JobRegistry.start marks the record
    as failed with the exception message.

    `local_repo_id` addresses the host's cache (a locally-recorded
    dataset's directory — and so its id — is bare); `hub_repo_id` is that
    id resolved against the caller's namespace, which is what every Hub
    call here must use. `log` receives the runner's progress lines (each
    runner tees them into its job's log file).

    Existence goes through the shared hub_repo_exists — not
    get_hub_status, whose process-lifetime memo is wrong for a caller about
    to WRITE. Only a confirmed absence pushes: on None
    (offline / rate-limited / any transport error) we leave the Hub alone,
    because pushing into a repo we could not verify is worse than a remote
    job that fails resolving a dataset.

    An EXISTING but EMPTY repo counts as absent. A half-finished upload
    leaves behind the empty repo its first call created; "the repo exists"
    was enough to skip the push, so the remote job would then die resolving
    a dataset with no files in it. Refilling it is the whole remedy and
    needs nothing from the user, so it happens silently rather than as a
    refusal they'd have to act on. The emptiness read is ``fresh=True`` for
    the same reason existence is uncached: a memo is wrong for a caller
    about to decide whether to WRITE.
    """
    exists = hub_repo_exists(hub_repo_id)
    if exists is None:
        return
    if exists and hub_copy_has_data(hub_repo_id, fresh=True) is not False:
        return

    if not local_pushable_copy_exists(local_repo_id):
        # Neither local nor usable on the Hub. Let the trainer surface the
        # error — same behaviour as before — but say why in the job log:
        # an empty repo was positively diagnosed, and silence here would
        # leave the doomed run unexplained.
        if exists:
            log(
                f"[upload] dataset {hub_repo_id} exists on the Hub but holds no data,"
                " and there is no local copy to push."
            )
        return

    # Public by default: MakerMods Lab's global policy is that datasets it pushes
    # to the Hub are public and carry the required org/product tags (see
    # with_makermodslab_tag / REQUIRED_HUB_TAGS) so all MakerMods Lab-produced
    # datasets are discoverable. The lone exception is a temporary merge — a
    # throwaway scratch dataset — which goes up private.
    private, is_temp = _dataset_upload_plan(local_repo_id, hub_repo_id)
    visibility = "private" if private else "public"
    reason = (
        "exists on the Hub but holds no data (an earlier upload didn't finish)" if exists else "not on Hub"
    )
    log(f"[upload] dataset {hub_repo_id} {reason}; pushing local copy ({visibility})...")
    try:
        push_dataset_to_hub(local_repo_id, tags=with_makermodslab_tag(None), private=private)
    except Exception as exc:
        msg = f"Failed to upload local dataset {local_repo_id} to Hub: {exc}"
        log(f"[upload] {msg}")
        raise RuntimeError(msg) from exc
    log(f"[upload] dataset {hub_repo_id} uploaded.")

    if is_temp:
        # Record where it landed so "clean up temporary merges" can delete the
        # Hub copy too — the ONLY signal that authorises a Hub delete. A missing
        # back-ref only costs us that later cleanup, so never let it raise.
        try:
            from ..merge_manifest import read_merge_manifest, write_merge_manifest

            d = _local_dataset_dir(local_repo_id)
            m = read_merge_manifest(d)
            if m is not None and m.hub_repo != hub_repo_id:
                m.hub_repo = hub_repo_id
                write_merge_manifest(d, m)
        except Exception as exc:
            log(f"[upload] could not record the Hub repo on the merge sidecar: {exc}")
