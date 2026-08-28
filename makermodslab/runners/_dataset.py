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

import os
from collections.abc import Callable
from pathlib import Path

from ..datasets import hub_copy_has_data, hub_repo_exists, push_dataset_to_hub
from ..utils.config import with_makermodslab_tag


def local_pushable_copy_exists(local_repo_id: str) -> bool:
    """Is there a flat-layout copy of `local_repo_id` in the host's lerobot
    cache — the only form ``push_dataset_to_hub`` can push?

    A snapshot-cache download (a Hub dataset fetched for local use) is
    deliberately NOT pushable and returns False: the runner cannot re-push
    what it never had in pushable form. Shared with the jobs preflight, which
    must ask the exact same question — "can the runner refill an empty repo
    itself?" — so the two cannot drift on what counts as a local copy.
    """
    cache_root = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    return (cache_root / local_repo_id / "meta" / "info.json").is_file()


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

    reason = (
        "exists on the Hub but holds no data (an earlier upload didn't finish)" if exists else "not on Hub"
    )
    log(f"[upload] dataset {hub_repo_id} {reason}; pushing local copy (public)...")
    try:
        # Public by default: MakerMods Lab's global policy is that datasets it pushes
        # to the Hub are public and carry the required org/product tags (see
        # with_makermodslab_tag / REQUIRED_HUB_TAGS). This implicit pre-run upload
        # follows that same default so all MakerMods Lab-produced datasets are
        # discoverable. (This intentionally reverses the earlier private
        # default — an implicit upload of a local-only dataset is now public.)
        push_dataset_to_hub(local_repo_id, tags=with_makermodslab_tag(None), private=False)
    except Exception as exc:
        msg = f"Failed to upload local dataset {local_repo_id} to Hub: {exc}"
        log(f"[upload] {msg}")
        raise RuntimeError(msg) from exc
    log(f"[upload] dataset {hub_repo_id} uploaded.")
