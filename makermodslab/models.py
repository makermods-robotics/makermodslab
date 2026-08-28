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

"""Trained-model browser: the datasets.py of policies.

Mirrors `makermodslab/datasets.py` for MODELS instead of datasets:

  * LOCAL models are the final checkpoint of each COMPLETED local training run,
    read straight from the job registry (`makermodslab/jobs.py`) — we never re-track
    jobs here, only read them. A run's final checkpoint is the highest-step
    `checkpoints/<step>/pretrained_model/` dir; its `train_config.json` supplies
    the policy type, base dataset repo_id, and step target.
  * HUB models are the user's LeRobot policy repos on the Hub. Listed with the
    same per-author fan-out + resilience the datasets listing uses (a GFW-killed
    TLS handshake / timeout for one author degrades to the others rather than
    500ing), reusing `datasets._fan_out_hub_authors`.

`list_all_models` merges the two into one listing with a `source` of
"local" / "hub" / "both", exactly like `list_all_datasets`. Upload pushes a
local checkpoint to the Hub as a public, MakerModsLab-tagged model repo; delete removes
a local run's output dir (strictly sandboxed under outputs/train/).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants, metadata_update, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import filter_repo_objects

from .datasets import (
    DownloadManager,
    _dir_mtime_iso,
    _fan_out_hub_authors,
    _lerobot_cache_root,
)
from .jobs import (
    JobHasChildrenError,
    JobRecord,
    _list_local_checkpoints,
    _read_checkpoint_config,
    job_registry,
)
from .utils.config import (
    get_hidden_models,
    get_saved_custom_models,
    validate_dataset_repo_id,
    with_makermodslab_tag,
)
from .utils.hf_auth import cached_whoami, get_token, hf_hub_offline, shared_hf_api
from .utils.naming import (
    KNOWN_POLICY_TYPES,
    POLICY_TYPES_BY_LENGTH as _POLICY_TYPES_BY_LENGTH,
    RUN_REPO_TIMESTAMP_RE,
    dedupe_display_names,
    imported_name_suffixes,
    iso_time_suffixes,
)

logger = logging.getLogger(__name__)

# Cap on the local-run scan. The registry keeps full history; a browser never
# needs thousands of entries, and this bounds the per-run checkpoint stat work.
_LOCAL_MODEL_SCAN_LIMIT = 500

# A repo qualifies as a MakerMods Lab-relevant model if it carries the `lerobot` library
# tag (what push_to_hub stamps) OR its name matches the MakerMods Lab run-repo pattern
# (a "_<timestamp>" suffix). Same union as server.py's _list_author_models.
# The pattern itself now lives in utils/naming.py, which also derives titles from
# it — one definition, so a repo that reads as generated to one reads so to both.
_RUN_REPO_RE = RUN_REPO_TIMESTAMP_RE


def _hub_policy_type(tags: list[str] | None, repo_name: str) -> str | None:
    """Infer a Hub model repo's policy type from data the listing already has —
    zero extra network.

    Two signals, in order:
      * TAGS — lerobot-native pushes stamp the type as a tag
        (lerobot/policies/pretrained.py push_model_to_hub → generate_model_card
        with ``tags={"robotics", "lerobot", model_type}``);
      * NAME PREFIX — MakerMods Lab UI uploads historically don't tag the type (fixed
        by upload_local_model's stamping going forward) but embed it as the
        repo name's prefix (jobs._generate_job_id → ``{policy_type}_{dataset}_
        {ts}``). Matched longest-first so pi0 never shadows pi0_fast.

    Returns None when neither signal matches (the UI simply omits the label)."""
    for tag in tags or []:
        if tag in KNOWN_POLICY_TYPES:
            return tag
    for policy_type in _POLICY_TYPES_BY_LENGTH:
        if repo_name.startswith(policy_type + "_"):
            return policy_type
    return None


# The listing is cached in TWO halves, because its two halves cost completely
# different things and go stale for completely different reasons.
#
#   * The HUB half is a per-author network fan-out: slow, rate-limited, flaky.
#     It is worth caching for a long time (_HUB_LISTING_TTL_S) and it does not
#     go stale because anything happened on this machine.
#   * The LOCAL half is a registry scan plus directory walks. It is cheap, and
#     it goes stale the instant a run finishes, is renamed, imported or deleted.
#
# One 45s cache over both meant a finished run stayed invisible in the picker
# for up to 45 seconds while the jobs-driven library already showed it — the
# transient half of "the two skill lists disagree". Splitting lets the merged
# result carry a short TTL (a burst guard: four consumers mount at once on
# startup and must not each redo the filesystem work) while the expensive Hub
# half keeps the long one. Registry mutations drop the merged cache outright via
# server._on_jobs_changed, so the short TTL is only ever protecting against
# bursts, never hiding a change.
#
# TTLs are measured with time.monotonic() (app runtime, immune to clock jumps).
_HUB_LISTING_TTL_S = 45.0
_LISTING_CACHE_TTL_S = 5.0
_listing_cache_lock = threading.Lock()
_listing_cache: dict[str, Any] | None = None  # {"at": monotonic, "value": [...]}
_hub_listing_cache: dict[str, Any] | None = None  # {"at": monotonic, "rows", "status"}
# The last Hub listing that came back COMPLETE. Serves rows a degraded fan-out
# would otherwise drop silently — see _hub_listing. Deliberately NOT cleared by
# invalidate_model_listing_cache: it is the fallback for an outage, and registry
# mutations (which are frequent) must not wipe the safety net.
_hub_last_good: list[dict[str, Any]] | None = None
# The credentials _hub_last_good was listed under, so the fallback can refuse to
# serve one account's repos to another.
_hub_last_good_auth: str = ""
# The author set (user + orgs) that listing covered, when it was known. Checked
# alongside the credentials because org membership can change under one token.
_hub_last_good_authors: tuple[str, ...] = ()
# Repos the user deleted from the Hub: lower-cased id -> the monotonic time of
# the delete. A delete clears the caches, but a fan-out that STARTED before it
# finishes afterwards holding data from before the deletion, and would write the
# deleted repo straight back. The timestamp tells those apart: a listing that
# began before the delete cannot be trusted about that repo, while one that
# began after it and still returns the repo proves it exists again (recreated),
# which retires the tombstone.
_forgotten_hub_repos: dict[str, float] = {}


def invalidate_model_listing_cache() -> None:
    """Drop the cached /models listing so the next call re-fetches.

    Called after any mutation that changes the listing — a model upload/delete,
    and now every job-registry change (server._on_jobs_changed) — so a change
    the user just made shows up immediately instead of after the TTL. Mirrors
    datasets.invalidate_dataset_listing_cache.

    Drops the Hub half too. A repo the user just pushed or deleted is a Hub
    change as much as a local one, and leaving the 45s half warm would show the
    old repo set for up to 45s after the mutation that changed it."""
    global _listing_cache, _hub_listing_cache
    with _listing_cache_lock:
        _listing_cache = None
        _hub_listing_cache = None


def forget_hub_repo(repo_id: str) -> None:
    """Drop every trace of a Hub repo the user just deleted.

    `invalidate_model_listing_cache` alone is not enough here. It clears the
    caches, but the deleted repo also sits in `_hub_last_good` — the fallback a
    degraded fan-out serves from — so the next Hub blip would resurrect a repo
    that no longer exists, and keep resurrecting it. Retention is a safety net
    for rows we merely failed to SEE; a row we know is gone must leave it."""
    global _hub_last_good
    wanted = repo_id.lower()
    with _listing_cache_lock:
        # Recorded as well as removed: a Hub fetch already in flight will
        # repopulate the fallback after this returns, and would otherwise
        # resurrect the repo the user just deleted.
        _forgotten_hub_repos[wanted] = time.monotonic()
        if _hub_last_good is not None:
            _hub_last_good = [r for r in _hub_last_good if r["repo_id"].lower() != wanted]
    invalidate_model_listing_cache()
    invalidate_model_hub_info(repo_id)


class ModelError(Exception):
    """Raised when a model mutation (upload/delete) can't proceed. `status` is
    the HTTP status the route should return (400 offline/invalid, 403 no write
    permission, 404 not found, 409 busy, 502 other Hub failure); `message` is
    the user-facing reason; `docs_url` (optional) links auth docs for a login
    failure."""

    def __init__(self, status: int, message: str, docs_url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.docs_url = docs_url


# ---------------------------------------------------------------------------
# Local models — the final checkpoint of each completed local training run.
# ---------------------------------------------------------------------------


def _final_checkpoint_dir(record: JobRecord) -> Path | None:
    """The pretrained_model dir of a completed local run's final (highest-step)
    checkpoint, or None if the run saved no checkpoint.

    Reuses jobs._list_local_checkpoints (which validates each
    checkpoints/<step>/pretrained_model/config.json), so a half-saved checkpoint
    dir is never returned. The list is step-sorted; the last entry is the final
    checkpoint. `ref` is the absolute pretrained_model dir."""
    checkpoints = _list_local_checkpoints(record.output_dir)
    if not checkpoints:
        return None
    return Path(checkpoints[-1].ref)


def _read_train_config(pretrained_dir: Path) -> dict[str, Any]:
    """Load a checkpoint's train_config.json (policy type / dataset / steps).

    lerobot writes this alongside config.json inside pretrained_model/. Absent or
    unreadable metadata degrades to an empty dict — the caller renders without
    it rather than dropping the model from the listing."""
    path = pretrained_dir / "train_config.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s: %s", path, exc)
        return {}


def _local_model_summary(record: JobRecord, pretrained_dir: Path) -> dict[str, Any]:
    """Build the listing row for one completed local run's final checkpoint.

    policy_type / dataset come from train_config.json where present, and fall
    back to the registry record (config.policy_type, config.dataset_repo_id)
    so a checkpoint missing its train_config still renders a usable row.

    `steps` and `target_steps` are deliberately kept separate: `steps` is the
    checkpoint's actual step (its dir name) — the authoritative count for what
    was actually saved — while `target_steps` is the run's configured step
    count (train_config.json's `steps`, falling back to the registry record's).
    An "interrupted" run that stopped short of its target has `steps <
    target_steps`; without both fields such a run is indistinguishable from
    one that finished exactly at that checkpoint. `state` ("done" or
    "interrupted", per the `list_local_models` gate) is carried for the same
    reason — callers should not assume every row here trained to completion."""
    train_config = _read_train_config(pretrained_dir)

    policy = train_config.get("policy") or {}
    policy_type = policy.get("type") or record.config.policy_type

    dataset = train_config.get("dataset") or {}
    dataset_repo_id = dataset.get("repo_id") or record.config.dataset_repo_id or None

    raw_target_steps = train_config.get("steps")
    target_steps = raw_target_steps if isinstance(raw_target_steps, int) else record.config.steps

    # Prefer the checkpoint's real step (its dir name) over the config target.
    try:
        steps: int | None = int(pretrained_dir.parent.name)
    except (ValueError, TypeError):
        steps = target_steps

    return {
        "id": record.id,
        "name": record.display_name or record.name,
        "policy_type": policy_type,
        "dataset": dataset_repo_id,
        "steps": steps,
        "target_steps": target_steps,
        "state": record.state,
        "path": str(pretrained_dir),
        "last_modified": _epoch_iso(record.ended_at or record.started_at),
        "hf_repo_id": record.hf_repo_id or None,
        "source": "local",
        # Where the weights came from, as opposed to where they live now
        # (`source`). The skills listing filters and labels on this.
        "origin": "trained-local",
        # The registry record that can be deployed/resumed for this row. Stamped
        # HERE, server-side, so the frontend stops re-deriving it: it was
        # re-implementing this repo's own `_job_outranks` ranking in TypeScript
        # over a separately-fetched page of /jobs.
        "job_id": record.id,
    }


def _epoch_iso(ts: float | None) -> str | None:
    from datetime import UTC, datetime

    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def list_local_models() -> list[dict[str, Any]]:
    """Every local training run that produced a usable final checkpoint.

    Reads the job registry (never mutates it). A run qualifies iff it is a
    `local` runner in state "done" or "interrupted" AND has at least one
    valid on-disk checkpoint. A "done" run with no checkpoint (crashed before
    its first save) is skipped. An "interrupted" run (e.g. its exit couldn't
    be confirmed after a server restart) whose checkpoint is on disk still
    counts — that state is only a claim about how we found out, not about
    whether the weights exist. A "failed" run is excluded even with a
    checkpoint present: unlike "interrupted", its exit code is a confirmed
    non-zero result, so listing it next to real models — indistinguishable,
    selectable for inference or a Hub push — would misrepresent a known
    failure as a usable one. Each row carries the policy type, base dataset
    repo_id, step count, and the local checkpoint path, read from the final
    checkpoint's train_config.json.

    Newest first (by the run's end/start time)."""
    out: list[dict[str, Any]] = []
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT, with_checkpoints=False):
        if record.runner != "local" or record.state not in ("done", "interrupted"):
            continue
        # A run that something resumed is not its own skill: the chain is ONE
        # trained model that happens to have been produced across several runs.
        # Listing every link put the same skill in the picker N times, split so
        # that each row offered only the steps that link happened to save (the
        # parent 500-1500, the child 2000-5000). The successor's row stands for
        # the whole chain and offers all of them — see list_chain_checkpoints.
        if record.child_ids:
            continue
        pretrained_dir = _final_checkpoint_dir(record)
        if pretrained_dir is None:
            continue
        out.append(_local_model_summary(record, pretrained_dir))

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


def _find_local_record(model_id: str) -> JobRecord | None:
    """The usable local-run registry record for `model_id`, or None.

    None when the id is unknown, isn't a local run, isn't "done" or
    "interrupted", was superseded by a run that resumed it, or left no
    checkpoint — the same four gates `list_local_models` applies, so callers
    agree with the listing about what a "local model" is.

    The superseded gate was missing here while the docstring already claimed
    parity. The consequence was small but real and pointed the wrong way: a
    mid-chain parent the listing deliberately hides was still reachable by id,
    so `upload_local_model` would publish a Hub repo for a run the UI says is
    not a skill — a side effect for a model the user cannot see, name, or
    manage afterwards.

    `delete_local_model` does NOT go through here (it calls `job_registry.get`
    itself), so tightening this does not swallow its 409 "delete the
    continuation(s) first" — that message stays the one a mid-chain delete
    gets."""
    from .jobs import JobNotFoundError

    try:
        record = job_registry.get(model_id)
    except JobNotFoundError:
        return None
    if record.runner != "local" or record.state not in ("done", "interrupted"):
        return None
    if record.child_ids:
        return None
    if _final_checkpoint_dir(record) is None:
        return None
    return record


def _superseded_by(model_id: str) -> str | None:
    """The run(s) that resumed `model_id`, formatted for a message, or None.

    Lets a caller tell "this id is not a model" apart from "this id is a link in
    a chain another run represents" — two different answers that
    `_find_local_record` collapses into one None."""
    from .jobs import JobNotFoundError

    try:
        record = job_registry.get(model_id)
    except JobNotFoundError:
        return None
    if not record.child_ids:
        return None
    return ", ".join(repr(child_id) for child_id in record.child_ids)


# ---------------------------------------------------------------------------
# Downloaded / imported local models — checkpoints in the local models dir.
# ---------------------------------------------------------------------------
#
# The second kind of "local" model, alongside completed training runs: a
# checkpoint COPIED onto this machine, either downloaded from the Hub
# (model_download_manager) or imported from a folder on disk
# (import_local_model). They live under a stable local models dir so the
# /models listing can flip a Hub repo to "both" and inference can run offline —
# the models mirror of the datasets' FLAT download layout.


def _local_models_root() -> Path:
    """The stable directory downloaded/imported model checkpoints land in:
    ``<lerobot_home>/makermodslab_models/<repo_id>/``. Kept under the same home as the
    datasets' flat layout (which uses the home root directly — models get a
    dedicated subdir so checkpoint dirs are never confused with dataset dirs).

    Existing installs used ``lelab_models``. Move that directory on first use
    so locally downloaded checkpoints remain available after the package
    rebrand. If the move is not possible (for example, a read-only cache), keep
    using the legacy directory rather than forcing a new Hub download.
    """
    cache_root = _lerobot_cache_root()
    root = cache_root / "makermodslab_models"
    legacy_root = cache_root / "lelab_models"
    if not root.exists() and legacy_root.exists():
        try:
            legacy_root.rename(root)
        except OSError as exc:
            logger.warning("Could not migrate local model cache from %s to %s: %s", legacy_root, root, exc)
            return legacy_root
    return root


# The weights file lerobot's loader actually opens, and the one alternative
# layout it accepts. Derived from the PINNED lerobot (v0.6.0), not guessed:
#
#   * Standard policy — `PreTrainedPolicy.from_pretrained` on a LOCAL dir does
#     exactly `os.path.join(model_id, SAFETENSORS_SINGLE_FILE)` and hands it
#     straight to `_load_as_safetensor` (policies/pretrained.py:195-198).
#     `SAFETENSORS_SINGLE_FILE` is huggingface_hub's "model.safetensors".
#   * SHARDED weights are deliberately NOT accepted. That local-dir branch has no
#     index-file handling at all, and the save side pins `max_shard_size` above
#     the total size specifically "so the output stays a single
#     `model.safetensors`" (pretrained.py:157-159). Treating
#     `model-00001-of-...` as usable would recreate this very bug — a tree we
#     call loadable that lerobot then fails to open.
#   * ADAPTER (PEFT/LoRA) repos are accepted, because `make_policy` has a real
#     branch for them: with `use_peft`, it reads `PeftConfig.from_pretrained(dir)`
#     and calls `PeftModel.from_pretrained(policy, dir)` (policies/factory.py:
#     616-638), loading the BASE weights from the adapter config's
#     `base_model_name_or_path`. So an adapter dir is complete without
#     `model.safetensors`; what it must have is PEFT's own pair.
#
# NOT sufficient, and the trap this exists to close: "some *.safetensors is
# present". A policy checkpoint also ships pre/post-processor safetensors, so an
# interrupted download can carry several .safetensors files and still have no
# policy weights at all.
_POLICY_WEIGHTS_FILE = "model.safetensors"
_ADAPTER_WEIGHTS_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _has_loadable_weights(pretrained_dir: Path) -> bool:
    """True when `pretrained_dir` holds weights lerobot can actually load.

    A `config.json` alone proves only that a download STARTED. An interrupted
    `local_dir` download leaves a tree that looks complete to a config-only check
    — config.json, train_config.json, both processor safetensors — and then dies
    with FileNotFoundError on `model.safetensors` the moment inference runs. That
    happened live: a 68 KB store entry served by the F6 short-circuit turned a
    silently-partial download into a hard crash, where the pre-F6 path would have
    re-downloaded and worked.
    """
    try:
        if (pretrained_dir / _POLICY_WEIGHTS_FILE).is_file():
            return True
        return all((pretrained_dir / name).is_file() for name in _ADAPTER_WEIGHTS_FILES)
    except OSError:
        return False


def _resolve_pretrained_dir(path: Path) -> Path | None:
    """The pretrained_model dir inside a downloaded/imported checkpoint dir, or
    None when `path` isn't a usable policy checkpoint.

    Recognizes the same two layouts jobs._list_imported_local does (and in the
    same order): a ``checkpoints/<step>/pretrained_model`` tree (the final,
    highest-step checkpoint wins), else the dir itself when it has a root
    ``config.json`` (the shape ``upload_local_model`` pushes). The returned dir
    is EXACTLY what rollout._resolve_policy_path consumes verbatim for a local
    ref, so `path` rows built from it are directly usable for inference.

    "Usable" requires the POLICY WEIGHTS, not just a config (see
    `_has_loadable_weights`). This is the single definition every consumer shares
    — the /models listing, `is_model_available_locally`, `_fetch_model_snapshot`'s
    post-download validation, `_cleanup_partial_model`, `import_local_model` and
    the F6 local-store short-circuit in rollout all ask this one question — so a
    weights-less tree is uniformly invisible rather than usable-here-and-broken-
    there. A partial download therefore drops out of the listing instead of being
    offered and then crashing mid-run.

    In the checkpoints layout the scan walks from the highest step DOWN and takes
    the first checkpoint with weights: a checkpoint still being written by a live
    training run must not hide the last complete one. The returned path names its
    own step, so there is no risk of the UI labelling one step's weights as
    another's.
    """
    try:
        checkpoints = _list_local_checkpoints(str(path))
        for checkpoint in reversed(checkpoints):
            candidate = Path(checkpoint.ref)
            if _has_loadable_weights(candidate):
                return candidate
        if checkpoints:
            return None
        if (path / "config.json").is_file() and _has_loadable_weights(path):
            return path
    except OSError:
        return None
    return None


def _downloaded_model_dir(repo_id: str) -> Path | None:
    """The models-root dir for `repo_id` IF it holds a usable checkpoint, else
    None. Traversal-guarded: a repo_id that resolves outside the models root
    (e.g. '../evil') is refused."""
    root = _local_models_root().resolve()
    try:
        path = (root / repo_id).resolve()
    except OSError:
        return None
    if path == root or root not in path.parents:
        return None
    if not path.is_dir() or _resolve_pretrained_dir(path) is None:
        return None
    return path


def _hub_cache_has_repo(repo_id: str) -> bool:
    """True when the shared HF hub cache already holds a snapshot of `repo_id`.

    The hub-cache twin of `_downloaded_model_dir` above: that one answers "is it
    in OUR models store", this one "is it in the cache huggingface_hub itself
    manages". Both callers of this are about not downloading the same bytes
    twice — `_fetch_model_snapshot` below (serve a Models-page download from the
    cache) and `rollout._local_store_policy_path` (the mirror image: serve
    inference from our store when the cache is empty). It lives HERE, and
    rollout imports it, so the two directions can never disagree about what
    "already cached" means; models.py must not import from rollout (cycle).

    huggingface_hub 1.21.0 has no public "is this repo cached at all?" call:
    ``try_to_load_from_cache`` answers per FILE and per revision (so it says
    "no" for a repo cached under a different revision), and ``scan_cache_dir``
    walks the ENTIRE cache — every dataset too — to answer one question about
    one repo. So probe the documented on-disk layout directly, which is exactly
    the question we want: ``<HF_HUB_CACHE>/models--<user>--<repo>/snapshots/``.
    A repo dir with no snapshot in it (an interrupted or wiped entry) counts as
    absent. HF_HUB_CACHE is read off the module at call time, not bound at
    import, so an env override (or a test) is honoured.

    A malformed repo id answers "not cached" rather than raising: `policy_ref`
    is user input and the hub-ref regex accepts any ``[^@]+``, so
    `repo_folder_name`'s validation can reject it (HFValidationError is a
    ValueError). Reporting False just routes it to snapshot_download, which
    produces the canonical, already-handled error message for a bad repo id."""
    try:
        folder = Path(hf_constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")
        snapshots = folder / "snapshots"
        return snapshots.is_dir() and any(snapshots.iterdir())
    except (OSError, ValueError):
        return False


def is_model_available_locally(repo_id: str) -> bool:
    """True when `repo_id` has a usable checkpoint in the local models dir, so
    inference could run on it without a Hub download. Filesystem-only (no
    network) — safe to call offline. The datasets mirror is
    is_dataset_available_locally."""
    return _downloaded_model_dir(repo_id) is not None


def _downloaded_model_summary(repo_id: str, model_dir: Path) -> dict[str, Any]:
    """Listing row for one downloaded/imported model checkpoint.

    Shaped like a local-run row (id/name = repo_id since there's no run record):
    policy type from the checkpoint's config.json, dataset/steps from its
    train_config.json where present (upload_folder pushes the whole
    pretrained_model dir, so downloaded MakerMods Lab-made models carry it; a tree
    checkpoint's step dir is authoritative for steps). `path` is the resolved
    pretrained dir — directly consumable by rollout._resolve_policy_path.
    hf_repo_id stays None here: the scan can't know whether the dir came from
    the Hub or a disk import; the listing merge / pin fold fills it in when the
    repo is confirmed on the Hub."""
    pretrained = _resolve_pretrained_dir(model_dir)
    assert pretrained is not None  # callers guarantee it (dir already probed)

    policy_type = None
    try:
        policy_type = json.loads((pretrained / "config.json").read_text()).get("type")
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s: %s", pretrained / "config.json", exc)

    train_config = _read_train_config(pretrained)
    dataset = (train_config.get("dataset") or {}).get("repo_id")

    steps: int | None = None
    if pretrained != model_dir:
        # Tree layout: checkpoints/<step>/pretrained_model — the dir name is the
        # actual step.
        try:
            steps = int(pretrained.parent.name)
        except (TypeError, ValueError):
            steps = None
    if steps is None:
        raw_steps = train_config.get("steps")
        steps = int(raw_steps) if isinstance(raw_steps, int) else None

    return {
        "id": repo_id,
        "name": repo_id,
        "policy_type": policy_type,
        "dataset": dataset,
        "steps": steps,
        "path": str(pretrained),
        "last_modified": _dir_mtime_iso(model_dir),
        "hf_repo_id": None,
        "source": "local",
        "origin": "downloaded",
        # A scanned directory has no run behind it; the deploy path lazy-imports
        # one on pick, exactly as before.
        "job_id": None,
    }


def list_downloaded_models() -> list[dict[str, Any]]:
    """Scan the local models dir for downloaded/imported checkpoints.

    Walks one level deep, mirroring datasets.list_local_datasets: a top-level
    dir that is itself a checkpoint is recorded as "<name>"; otherwise each
    checkpoint subdir is recorded as "<top>/<sub>" (namespace layout). Unusable
    dirs (no config.json / checkpoints tree — e.g. a foreign download) are
    skipped."""
    root = _local_models_root()
    if not root.is_dir():
        return []

    try:
        top_entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("Could not read local models root %s: %s", root, exc)
        return []

    out: list[dict[str, Any]] = []
    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _resolve_pretrained_dir(top) is not None:
            out.append(_downloaded_model_summary(top.name, top))
            continue

        # Not a checkpoint itself — treat as a namespace dir, descend one level.
        try:
            sub_entries = list(top.iterdir())
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if not sub.is_dir():
                    continue
            except OSError:
                continue
            if _resolve_pretrained_dir(sub) is not None:
                out.append(_downloaded_model_summary(f"{top.name}/{sub.name}", sub))

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Registered imports — pointer records the other two local sources cannot see.
# ---------------------------------------------------------------------------


def _imported_model_summary(record: JobRecord) -> dict[str, Any] | None:
    """Listing row for one `runner == "imported"` record, or None if unusable.

    Two shapes, matching the two things `register_imported` stores:

      * HUB POINTER (`hf_repo_id` set, `output_dir` empty) — the weights live
        on the Hub and nothing was copied here, so the row is `source="hub"`
        with a null `path`. Built from the record's own fields: this function
        performs no repo listing, no tree scan and no I/O of any kind. That
        restraint is deliberate — see `list_imported_models`.
      * DISK POINTER (`output_dir` set) — an arbitrary directory on this
        machine. Reuses `_downloaded_model_summary`, so the resolved
        `pretrained_model` dir, policy type and train-config detail are read by
        exactly the same code the downloaded-checkpoint scan uses, and a
        weights-less tree drops out here as uniformly as it does there.

    The record's own config is NOT trusted for detail: an import stores
    `dataset_repo_id="(imported)"` and, when the checkpoint config could not be
    read, `policy_type="model"` (jobs.register_imported). Both are placeholders
    that would render as if they were facts, so each degrades to None and the
    disk shape prefers the values read off the checkpoint itself.
    """
    name = record.display_name or record.name
    last_modified = _epoch_iso(record.ended_at or record.started_at)

    if record.hf_repo_id:
        repo_id = record.hf_repo_id
        # "model" is register_imported's fallback when the checkpoint config was
        # unreadable, not a policy type — infer from the repo name instead, the
        # same way a pinned repo the Hub listing missed is inferred above.
        policy_type = record.config.policy_type
        if not policy_type or policy_type == "model":
            policy_type = _hub_policy_type(None, repo_id.split("/", 1)[-1])
        return {
            "id": repo_id,
            "name": name,
            "repo_id": repo_id,
            "policy_type": policy_type,
            "dataset": None,
            "steps": None,
            "target_steps": None,
            "state": record.state,
            "path": None,
            "last_modified": last_modified,
            "hf_repo_id": repo_id,
            "private": False,
            "source": "hub",
            "origin": "imported",
            "job_id": record.id,
        }

    if not record.output_dir:
        return None
    model_dir = Path(record.output_dir)
    if _resolve_pretrained_dir(model_dir) is None:
        # The import was registered against a directory that has since been
        # moved, deleted, or emptied. Drop it rather than offer a path that
        # would fail at load — the same rule the downloaded scan applies.
        return None

    row = _downloaded_model_summary(record.id, model_dir)
    # The scan names a row after the directory it found; a registered import has
    # a real identity, so the run id routes it and the record's title names it.
    row["name"] = name
    row["origin"] = "imported"
    row["job_id"] = record.id
    row["state"] = record.state
    row["target_steps"] = None
    row["last_modified"] = last_modified or row["last_modified"]
    return row


def list_imported_models() -> list[dict[str, Any]]:
    """Registered imports as listing rows — the third local source.

    Without this a POINTER import is registered, listed by `/jobs`, shown in the
    models library, and INVISIBLE to `/models`: it is not a `local` run (so
    `list_local_models` skips it on the runner gate), it holds no repo the user
    owns (so a foreign-namespace import never reaches `list_hub_models`), and
    it was never copied into `_local_models_root()` (so the downloaded scan
    never walks it). The picker's own Import button would refetch `/models` and
    still not find what it had just imported.

    That is not a staleness bug and no cache TTL ever healed it — it is the
    listing disagreeing with itself about what an import is.

    Rows are built WITHOUT verifying the weights. A Hub import is stamped from
    the record; whether its repo is still fetchable is answered at selection, by
    the download the deploy path runs anyway — loudly, and once, for the one
    repo the user actually picked. Confirming every row here instead would put a
    serial, un-deadlined Hub round-trip per import inside a listing that already
    spends a fan-out per author.

    The registry scan below opts out of checkpoint counting
    (`with_checkpoints=False`) for the same reason. That count reaches the Hub
    for every imported-hub record, serially and without a deadline, and nothing
    here reads it. The scan limit matches `list_local_models`'s so the listing
    cannot end up knowing about more runs than imports, or the reverse.

    Superseded records are skipped on the same rule `list_local_models` uses:
    one skill per resume chain, represented by its tip.

    Newest first, matching every other source here."""
    out: list[dict[str, Any]] = []
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT, with_checkpoints=False):
        if record.runner != "imported" or record.child_ids:
            continue
        row = _imported_model_summary(record)
        if row is not None:
            out.append(row)

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Cloud runs — stamped from the registry, not gated on the Hub listing.
# ---------------------------------------------------------------------------


def list_cloud_models() -> list[dict[str, Any]]:
    """Terminal cloud runs as listing rows, built from the registry alone.

    A cloud run publishes its policy to a Hub repo, so it used to enter the
    listing ONLY through `list_hub_models` — meaning a run that had finished,
    been recorded as done, and had its repo pushed still did not exist as a
    skill until the user's per-author fan-out happened to return that repo. Two
    separate delays sat in that gap (the push landing, and the next fan-out),
    and neither was visible to the user: the run simply was not there, with the
    Train panel already showing it as finished.

    The registry knows the run finished and knows which repo it published to.
    That is enough to list it. Whether the weights are actually fetchable is
    left `unverified` — the same claim an imported Hub pointer makes, settled by
    the download the deploy path performs anyway. So the Hub listing now only
    ever ADDS detail (privacy flag, real last-modified) to a row whose existence
    it no longer gates.

    One record per repo, chosen by `_jobs_by_hub_repo`'s ranking: a cloud resume
    chain republishes into run #1's repo, so several records name one repo and
    only the one that best identifies it should stand for it.

    Only `done` and `interrupted` — the same gate `list_local_models` applies.
    `failed` is deliberately NOT stamped from the registry here, and an earlier
    draft of this that claimed parity with the local `failed` case was simply
    wrong: a failed LOCAL run never enters /models at all, and reaches /skills
    only after its weights are verified on disk. Nothing equivalent can be
    verified for a cloud run without the Hub call this function exists to avoid,
    and the repo's existence proves nothing — `hf_repo_id` is stamped at submit
    and the cloud wrapper creates the repo BEFORE training starts. So a run that
    died at step 0 would otherwise be offered as deployable against an empty
    repo, failing only when the user tried to run it. A failed cloud run that
    did push still appears the moment the Hub listing returns its repo; it just
    does not get the registry shortcut.

    `interrupted` is gated on the run having logged a step, for the same reason:
    a run stopped before its first step pushed nothing.

    Superseded records are skipped — one skill per chain, named by its tip."""
    out: list[dict[str, Any]] = []
    for record in _jobs_by_hub_repo().values():
        repo_id = record.hf_repo_id or ""
        if record.runner != "hf_cloud" or record.state not in ("done", "interrupted"):
            continue
        if record.child_ids:
            continue
        # An interrupted run that never logged a step never pushed one either.
        if record.state == "interrupted" and not record.metrics.current_step:
            continue
        out.append(
            {
                "id": repo_id,
                "name": record.display_name or _run_identity_name(record),
                "repo_id": repo_id,
                "policy_type": record.config.policy_type or None,
                "dataset": record.config.dataset_repo_id or None,
                "steps": _job_run_steps(record),
                "target_steps": record.config.steps,
                "state": record.state,
                "path": None,
                "last_modified": _epoch_iso(record.ended_at or record.started_at),
                "hf_repo_id": repo_id,
                # Unknown without a Hub call, and this row exists precisely to
                # avoid one. The Hub listing corrects it when it returns.
                "private": False,
                "source": "hub",
                "origin": "trained-cloud",
                "job_id": record.id,
            }
        )

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Hub models — the user's LeRobot policy repos on the Hub.
# ---------------------------------------------------------------------------


def _list_author_models(api, author: str) -> list[dict[str, Any]]:
    """All MakerMods Lab-relevant model repos for one author, materialized.

    ONE unfiltered `list_models(author=...)` call, filtered client-side to the
    union of the `lerobot` library tag and the MakerMods Lab run-repo naming — the same
    set (and the same single-call shape) as server.py's `/jobs/hub` listing. The
    generator is materialized HERE, inside the fan-out worker, so the network I/O
    (and any GFW-killed connection) happens under the per-author timeout budget
    rather than lazily later while we iterate."""
    rows: list[dict[str, Any]] = []
    for m in api.list_models(author=author, limit=200, expand=["lastModified", "private", "tags"]):
        tags = getattr(m, "tags", None) or []
        name = m.id.split("/", 1)[-1]
        if "lerobot" in tags or _RUN_REPO_RE.search(name):
            rows.append(
                {
                    "repo_id": m.id,
                    "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                    "private": bool(getattr(m, "private", False)),
                    # The expand already fetched the tags — recognize the policy
                    # type from them (or the name prefix) for free, instead of
                    # discarding them.
                    "policy_type": _hub_policy_type(tags, name),
                }
            )
    return rows


def list_hub_models() -> list[dict[str, Any]]:
    """The user's (and their orgs') LeRobot policy repos on the Hub.

    Mirrors datasets.list_user_datasets: fan out per author concurrently via the
    shared `_fan_out_hub_authors` helper, so one blocked/slow/GFW-killed author
    degrades to "the others' results" instead of a 500. Empty when no token is
    configured. Deduped by repo_id, newest first.

    Records how the attempt went in `_hub_outcome` on the way out, so the
    listing can say "the Hub was unreachable" instead of silently returning
    fewer rows. See `_hub_listing` for why that is module state rather than a
    second return value."""
    info = cached_whoami()
    if info is None:
        _record_hub_outcome(authenticated=False, authors=(), answered=0)
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    batches = _fan_out_hub_authors(authors, lambda author: _list_author_models(api, author))
    for rows in batches:
        for row in rows:
            if row["repo_id"] in seen:
                continue
            seen.add(row["repo_id"])
            out.append(row)

    # `_fan_out_hub_authors` drops an author that errored or blew the deadline,
    # so a short result list IS the failure signal — there is no other one.
    _record_hub_outcome(authenticated=True, authors=tuple(authors), answered=len(batches))

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# How the most recent REAL Hub listing went. Module state rather than a second
# return value from list_hub_models because that function is the seam ~20 tests
# substitute to stay off the network; widening its contract would rewrite all of
# them to assert on a shape none of them care about. When it IS patched, nothing
# writes here and the defaults below stand — which is correct, because a patched
# listing did not talk to the Hub and has no outcome to report.
_hub_outcome_lock = threading.Lock()
_hub_outcome: dict[str, Any] = {"authenticated": False, "authors": (), "answered": 0}


def _record_hub_outcome(*, authenticated: bool, authors: tuple[str, ...], answered: int) -> None:
    global _hub_outcome
    with _hub_outcome_lock:
        _hub_outcome = {
            "authenticated": authenticated,
            "authors": authors,
            "answered": answered,
        }


# Held across a Hub fan-out so only one runs at a time — see _hub_listing.
_hub_fetch_lock = threading.Lock()


def _auth_fingerprint() -> str:
    """A stable, non-reversible id for the credentials in play, "" when signed out.

    This is the identity key for the Hub cache and its last-good fallback: an
    account switch IS a token change, so the token answers the question
    directly.

    Deliberately NOT `cached_whoami`. That reaches the network on a cold cache,
    and the cache-HIT path here runs on every listing call — putting a round
    trip there makes the cache cost the very thing it exists to save. It also
    made the test suite talk to the Hub (patching `list_hub_models` no longer
    shielded it), which is how this was caught.

    Hashed, never stored or logged raw: this value sits in module state beside
    rows that get serialized, and a token does not belong there."""
    token = get_token()
    if not token:
        return ""
    return hashlib.sha256(token.encode()).hexdigest()[:32]


def _hub_listing() -> tuple[list[dict[str, Any]], dict[str, Any]]:  # noqa: C901
    """The Hub half of the listing, cached, plus an honest status for it.

    Two jobs beyond caching:

      * SAY when the Hub could not be reached. `list_hub_models` never raises,
        so before this an outage, an expired token or a GFW-killed fan-out all
        arrived at the UI as a shorter list — indistinguishable from the user
        having deleted those models. Silent degradation is the worst failure
        mode this listing has, because the honest reading of the screen is
        wrong.
      * KEEP the rows a degraded fan-out dropped. On a partial failure the repos
        that did not come back this time are served from the last complete
        listing, flagged `stale`, instead of vanishing. A blip must not delete
        a user's skills from the screen.

    Returns (rows, status). Status carries `ok` / `authenticated` / `degraded` /
    `stale_rows`, which the envelope hands to the frontend verbatim."""
    global _hub_listing_cache, _hub_last_good, _hub_last_good_auth, _hub_last_good_authors

    identity = _auth_fingerprint()
    now = time.monotonic()
    with _listing_cache_lock:
        cached = _hub_listing_cache
        # The identity check belongs on the HIT path too, not only on the
        # retention path: a cached listing is one account's repos, and signing
        # out of Alice and into Bob must not show Bob a warm 45s window of
        # Alice's models. A changed identity is a miss, not a stale hit.
        if cached is not None and cached["auth"] == identity and (now - cached["at"]) < _HUB_LISTING_TTL_S:
            return cached["rows"], cached["status"]

    # One fan-out at a time. `list_hub_models` reports how it went through
    # module state, so two concurrent callers could interleave and let one read
    # the other's outcome — publishing a partial listing as complete, which is
    # exactly the silent degradation this function exists to prevent. Serializing
    # also collapses a thundering herd of cold-cache callers into one fan-out.
    with _hub_fetch_lock:
        # Re-check: a caller that queued behind the lock usually wants the
        # listing the holder just produced, not a second fan-out.
        now = time.monotonic()
        with _listing_cache_lock:
            cached = _hub_listing_cache
            if (
                cached is not None
                and cached["auth"] == identity
                and (now - cached["at"]) < _HUB_LISTING_TTL_S
            ):
                return cached["rows"], cached["status"]

        fetch_started = time.monotonic()
        rows = list_hub_models()
        # Re-read AFTER the fetch. Capturing the identity before it and storing
        # the result under that label is a race: sign out of Alice and into Bob
        # while the fan-out is in flight and Bob's repos get filed under Alice's
        # key, to be served back to Alice on her next visit. The credentials in
        # play when the rows were produced are the honest label for them.
        # (A switch DURING the fan-out can still mix two accounts' authors into
        # one result; that is bounded by the 5s fan-out deadline and cannot be
        # closed without holding auth still across the call.)
        identity = _auth_fingerprint()
        with _hub_outcome_lock:
            outcome = dict(_hub_outcome)

        authors = outcome["authors"]
        if identity and not outcome["authenticated"]:
            # Credentials are present but the listing came back unauthenticated.
            # `cached_whoami` swallows a transport failure and returns None, so
            # "signed out" and "the identity check just failed" arrive here
            # looking identical — and /whoami-v2 is heavily rate-limited, so the
            # failure is not rare. The TOKEN tells them apart: holding one means
            # this is an outage, not a state.
            #
            # Treating it as "not signed in" was strictly worse than the bug
            # this function exists to fix: every Hub row vanished, the status
            # claimed `ok`, AND the else-branch below overwrote the last-good
            # fallback with the empty result — destroying the safety net at the
            # exact moment it was needed.
            degraded = True
        else:
            # No authors means genuinely signed out (or patched in a test): not
            # a failure, just nothing to fan out to.
            degraded = bool(authors) and outcome["answered"] < len(authors)

        # Drop anything this fetch reports that was deleted WHILE it ran: those
        # rows predate the deletion, so the listing is simply wrong about them.
        with _listing_cache_lock:
            raced = {repo for repo, deleted_at in _forgotten_hub_repos.items() if deleted_at >= fetch_started}
            # A listing that BEGAN after a delete and still returns the repo is
            # evidence it exists again; retire those tombstones.
            if not degraded:
                returned = {row["repo_id"].lower() for row in rows}
                for repo in list(_forgotten_hub_repos):
                    if repo in returned and _forgotten_hub_repos[repo] < fetch_started:
                        del _forgotten_hub_repos[repo]
        if raced:
            rows = [row for row in rows if row["repo_id"].lower() not in raced]

        stale_rows = False
        if degraded:
            present = {row["repo_id"] for row in rows}
            with _listing_cache_lock:
                # Retention is keyed to the identity the rows were listed FOR:
                # sign out of one account and into another and the fallback
                # would otherwise show the previous account's repos on the new
                # one's screen. A stale listing is a tolerable degradation,
                # someone else's listing is not.
                #
                # The author set is checked too, but only when this attempt
                # actually learned one. The token identifies the CREDENTIALS
                # while the listing is really "user + orgs", so joining or
                # leaving an org changes the right row set under an unchanged
                # token — and `_hub_last_good` has no TTL, so a departed org's
                # repos could otherwise be resurrected as `stale` forever.
                # Skipped when `authors` is empty, because the whoami-failure
                # case above has no author set to compare and is precisely when
                # retention matters most.
                same_identity = _hub_last_good_auth == identity
                same_authors = not authors or _hub_last_good_authors == authors
                last_good = _hub_last_good if (same_identity and same_authors) else None
                retained = [
                    r
                    for r in (last_good or [])
                    if r["repo_id"] not in present and r["repo_id"].lower() not in _forgotten_hub_repos
                ]
            if retained:
                rows = rows + [{**r, "stale": True} for r in retained]
                stale_rows = True
        else:
            with _listing_cache_lock:
                _hub_last_good = list(rows)
                _hub_last_good_auth = identity
                _hub_last_good_authors = authors
                # `rows` has already had raced deletions stripped above, so the
                # fallback can never be seeded with a repo that is gone.

        status = {
            "ok": not degraded,
            "authenticated": outcome["authenticated"],
            "degraded": degraded,
            "stale_rows": stale_rows,
        }
        with _listing_cache_lock:
            _hub_listing_cache = {
                "at": time.monotonic(),
                "auth": identity,
                "rows": rows,
                "status": status,
            }
        return rows, status


# ---------------------------------------------------------------------------
# Merged listing.
# ---------------------------------------------------------------------------


def _job_run_steps(record: JobRecord) -> int | None:
    """How many steps the run behind a Hub repo actually trained.

    A run that reached "done" trained to its configured target, and
    `config.steps` is that target in GLOBAL steps even for a resumed run —
    prefer it, because the metrics tail can stop short of the end (the last log
    lines are parsed at log_freq granularity, and a terminal transition can drop
    the final one). Otherwise the run stopped early, so the last step its metrics
    observed is the honest number. None when neither is known: the row keeps the
    listing's null rather than invent one.
    """
    if record.state == "done" and record.config.steps:
        return record.config.steps
    if record.metrics.current_step:
        return record.metrics.current_step
    return None


def _job_outranks(candidate: JobRecord, incumbent: JobRecord) -> bool:
    """Ranking used to pick which run identifies a shared output repo.

    A LEAF (nothing resumed it) beats a non-leaf, because the tip of a chain is
    the run that stands for it. Then a run that reached "done" beats one that
    did not (it published the final policy at the repo root), and among equals
    the later-started run wins (the deepest link).

    Leaf-first is a guard, not a common path: `JobRegistry.start` refuses to
    resume a "done" run at all, so within a chain the done record IS the tip and
    the two rules agree. It matters for a registry written by an older version
    or edited by hand, where a done parent with children would otherwise be
    elected and then dropped by the leaf filter in `list_cloud_models` — taking
    the whole chain's row with it."""
    candidate_leaf = not candidate.child_ids
    incumbent_leaf = not incumbent.child_ids
    if candidate_leaf != incumbent_leaf:
        return candidate_leaf
    candidate_done = candidate.state == "done"
    incumbent_done = incumbent.state == "done"
    if candidate_done != incumbent_done:
        return candidate_done
    return candidate.started_at > incumbent.started_at


def _jobs_by_hub_repo() -> dict[str, JobRecord]:
    """Tracked runs keyed by the Hub repo they publish to, one run per repo.

    A cloud resume REUSES its parent's output repo (hf_cloud sets
    `policy_repo_id = resume_from_hub_repo`), so an N-deep resume chain shares a
    single repo named after run #1 — and the Hub listing, which keys on repo_id,
    then has no row under the name of the run that actually finished. Collapse
    each repo to the run that best identifies it (see _job_outranks).

    IMPORTED records are skipped: an import REGISTERS an existing repo rather
    than producing it, so its config is a placeholder (dataset_repo_id
    "(imported)", the default step count) and its auto-generated name says less
    than the repo id does. They are also always "done" and usually the newest
    record for their repo, so ranking them would let a placeholder outrank the
    run that actually trained the weights.

    Reads the registry, never mutates it. The scan limit matches
    list_local_models's so every pass over the registry sees the same set of
    runs. Like the others it opts out of checkpoint counting: ranking needs only
    state and start time, and counting would put a Hub round-trip behind each
    cloud record on a listing path that never reads the number."""
    best: dict[str, JobRecord] = {}
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT, with_checkpoints=False):
        repo_id = record.hf_repo_id
        if not repo_id or record.runner == "imported":
            continue
        # Keyed case-INSENSITIVELY: HF repo ids are unique across casing, so two
        # records spelling one repo differently are two records for ONE repo and
        # must be ranked against each other. Lower-casing only at lookup time
        # left them as separate entries here, and whichever the later pass
        # happened to overwrite won — registry iteration order deciding what
        # `_job_outranks` exists to decide. Callers read the ORIGINAL casing off
        # the elected record's own `hf_repo_id`, never off this key.
        key = repo_id.lower()
        incumbent = best.get(key)
        if incumbent is None or _job_outranks(record, incumbent):
            best[key] = record
    return best


def _run_identity_name(record: JobRecord) -> str:
    """A run's human name peeled down to the TASK it learned.

    jobs.start auto-generates an unnamed run's name as
    "{POLICY} · {dataset_repo_id}" — "SMOLVLA · makermods/eraser_place". Both
    halves of that are stated elsewhere on every surface that renders it: the
    row carries `policy_type` in its own field and `dataset` (the full repo id,
    namespace included) in another, so the title line — the widest and the first
    to truncate — spends its pixels on facts printed twice. Peel both:

      "SMOLVLA · makermods/eraser_place"  ->  "eraser_place"

    which converges model-card naming on one philosophy: the task identity,
    policy-free and namespace-free. Imported titles already read this way
    (utils.naming.derive_imported_title), and the retired "Imported · " prefix
    (jobs._LEGACY_IMPORT_NAME_PREFIX) went for the same reason.

    Only the GENERATED shape is peeled, and only when the head is a policy type
    we recognize — a user-typed job_name that happens to contain " · " keeps
    every word, and so does a name whose head isn't a policy. The namespace peel
    is gated behind that same check, so it can never eat a slash out of a name a
    human wrote. A rename (`display_name`) never reaches here; the caller
    prefers it outright.

    Two datasets that share a task name across namespaces (a/pick and b/pick)
    now render one label twice — that is what _dedupe_listing_names exists for,
    and its date suffix separates them.
    """
    head, separator, tail = record.name.partition(" · ")
    if not (separator and tail and head.lower() in KNOWN_POLICY_TYPES):
        return record.name
    # The tail is a dataset repo id: keep the segment after the namespace. A
    # bare name with no slash is already the task.
    _namespace, slash, task = tail.partition("/")
    return task if slash and task else tail


def _enrich_repo_rows_from_jobs(merged: dict[str, dict[str, Any]]) -> None:
    """Give each repo-keyed row the identity of the run that produced it.

    Applies only to rows still named after their repo id — i.e. rows no local
    checkpoint has claimed (a "both" collapse from a local run already put the
    run's own name and detail there, and local always wins). That covers the
    hub-only rows, hub rows flipped to "both" by a downloaded copy, and pinned
    rows the Hub listing didn't return.

    Identity fields are NOT touched: `id`, `repo_id` and `hf_repo_id` route
    every request (info / download / pin / hide / deploy) and a resume chain's
    repo keeps the name of run #1 on the Hub. Only the human-facing `name` is
    replaced, plus the detail fields the repo listing could not know — filled in
    only where they are still null, so a checkpoint-derived value always wins.
    """
    by_repo = _jobs_by_hub_repo()
    if not by_repo:
        return
    # Matched case-INSENSITIVELY. The registry stores whatever repo id the run
    # was configured with, while the Hub listing returns its canonical spelling,
    # and HF repo ids are unique across casing (the Hub redirects between them).
    # An exact compare here silently fails to connect a row to the run that
    # produced it — which used to cost only a nicer name, and now costs the
    # `job_id` that deploys it.
    for row in merged.values():
        repo_id = row.get("hf_repo_id")
        if not repo_id:
            continue
        record = by_repo.get(repo_id.lower())
        if record is None:
            continue
        # Ownership facts are stamped on EVERY row a run owns, not only the
        # unnamed ones: a "both" row already carries the local run's name, and
        # still needs to say which run deploys it.
        if row.get("job_id") is None:
            row["job_id"] = record.id
        if row.get("origin") == "hub-untracked":
            row["origin"] = "trained-cloud" if record.runner == "hf_cloud" else "trained-local"
        # A repo whose owning run was continued is a chain LINK. This is
        # reachable: a CLOUD run continued by a LOCAL one leaves the repo owned
        # by the parent (the local child has no repo of its own), so once the
        # Hub listing returns that repo it arrives beside the local tip's own
        # row — two picker entries for one trained model, with nothing saying
        # which. Marked here so list_skills makes it non-deployable, exactly as
        # it does for a superseded local run.
        if record.child_ids and row.get("superseded_by") is None:
            row["superseded_by"] = record.child_ids[0]
        # Naming, by contrast, applies only to rows still called after their
        # repo id — a local checkpoint that claimed the row already named it,
        # and local always wins.
        if row.get("name") != repo_id:
            continue
        row["name"] = record.display_name or _run_identity_name(record)
        if row.get("policy_type") is None:
            row["policy_type"] = record.config.policy_type or None
        if row.get("dataset") is None:
            row["dataset"] = record.config.dataset_repo_id or None
        if row.get("steps") is None:
            row["steps"] = _job_run_steps(record)


def _row_dedupe_suffixes(row: dict[str, Any]) -> list[str]:
    """The disambiguator ladder for one listing row, least → most specific.

    Prefers the run timestamp embedded in the row's own NAME — every repo a
    MakerMods Lab run publishes to carries the `_<date>_<time>` tail its run id
    was generated with — over the Hub's `last_modified`. The two usually agree,
    and where they disagree the name is the one the user means: last_modified
    moves whenever anything is pushed to the repo (a re-push of the same
    weights, a README edit, a later checkpoint upload), so a row could acquire a
    date months after the run it labels and read as simply wrong next to its
    sibling. The name's stamp is when the run STARTED and never moves.

    Falls back to last_modified for a row whose name carries no stamp — a
    community repo, a hand-named upload — which is the only signal there is.
    Both ladders come from utils.naming, so a name-derived suffix and a
    time-derived one are formatted identically and can disambiguate each other.
    """
    source = str(row.get("hf_repo_id") or row.get("id") or "")
    if source and _RUN_REPO_RE.search(source.rsplit("/", 1)[-1]):
        return imported_name_suffixes(source)
    return iso_time_suffixes(row.get("last_modified"))


def _dedupe_listing_names(merged: dict[str, dict[str, Any]]) -> None:
    """Make the listing's display names unique, in place.

    Run names are auto-generated from policy + dataset (jobs.start →
    "SMOLVLA · ns/task", which the enrichment peels down to "task" — see
    _run_identity_name), so retraining one task — a longer run, a different
    seed — produces a SECOND row with a byte-identical name. Naming rows after
    their run (see _enrich_repo_rows_from_jobs) makes that collision visible in
    the picker, where two rows then read as duplicates of each other with
    nothing on either to say which is which.

    Disambiguate least-specific first (see _row_dedupe_suffixes for where the
    date comes from, and dedupe_display_names, which escalates to date + time
    and falls back to an ordinal if even that ties). Runs over EVERY row, not
    just the enriched ones — two local runs of one policy on one dataset collide
    the same way and never went near the enrichment.

    Rows are grouped by (name, policy_type), so an ACT and a SmolVLA of one task
    are NOT suffixed: the Policy row on the card already separates them, and a
    date there would imply the difference between the two is when they ran.
    Identity fields are left alone, on the same rule as the enrichment: only
    `name` is display.
    """
    rows = list(merged.values())
    resolved = dedupe_display_names(
        [(str(row.get("name") or ""), _row_dedupe_suffixes(row)) for row in rows],
        group_keys=[row.get("policy_type") for row in rows],
    )
    for row, name in zip(rows, resolved, strict=True):
        row["name"] = name


def _build_listing() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merged listing: local completed runs + Hub policy repos, with a `source`.

    A local run that was also pushed to the Hub (its `hf_repo_id` matches a Hub
    repo) collapses to ONE entry with source="both"; a Hub-only repo is "hub"; a
    local-only run is "local". Mirrors list_all_datasets's merge/dedup/sort and
    its resilience: the Hub half is best-effort (list_hub_models never raises),
    so a Hub outage degrades to local-only rather than crashing.

    A row still named after its repo id is then named after the tracked run that
    produced it (see _enrich_repo_rows_from_jobs) — a repo id names a run only
    by accident, and never names the right one for a cloud resume chain, which
    publishes every link into the FIRST run's repo.

    UNCACHED, and returns the Hub half's status alongside the rows. Both
    `list_all_models` (the /models array) and `list_skills` (the /skills
    envelope) are projections of this one build — the whole point of the change
    is that there is exactly one merge, so two surfaces cannot disagree about
    what exists by each doing their own."""
    local = list_local_models()
    hub, hub_status = _hub_listing()

    merged: dict[str, dict[str, Any]] = {}
    # Seed with hub repos, keyed by repo_id. `list_hub_models` returns the
    # /jobs/hub row shape (repo_id / last_modified / private), so map each into
    # the full ModelItem shape the frontend expects — a hub-only model has no
    # local checkpoint, so its `id`/`name`/`hf_repo_id` are all the repo id and
    # the checkpoint-derived detail fields (policy_type/dataset/steps/path) are
    # null. Without this mapping hub rows reach the UI with no id/name and crash
    # the picker.
    for item in hub:
        repo_id = item["repo_id"]
        merged[repo_id] = {
            "id": repo_id,
            "name": repo_id,
            # Kept alongside hf_repo_id for symmetry with the datasets listing
            # rows (which key on repo_id) — hub-derived rows carry both.
            "repo_id": repo_id,
            # Recognized from the repo's tags / name by the hub listing (see
            # _hub_policy_type) — no extra network. A local checkpoint's
            # config.json value still wins on a "both" collapse below.
            "policy_type": item.get("policy_type"),
            "dataset": None,
            "steps": None,
            "target_steps": None,
            "state": None,
            "path": None,
            "last_modified": item.get("last_modified"),
            "hf_repo_id": repo_id,
            # The hub listing already fetches the private flag — surface it so
            # the picker can badge private repos (mirrors DatasetItem.private).
            "private": bool(item.get("private", False)),
            "source": "hub",
            # Nothing in the registry claims this repo yet. _enrich_repo_rows_from_jobs
            # upgrades it to trained-local/trained-cloud if a run turns out to own it.
            "origin": "hub-untracked",
            "job_id": None,
        }

    # Local rows key on their run id, EXCEPT when the run's hub_repo_id matches a
    # hub repo already present — then it's the same model on both, collapsed to
    # "both" under the hub repo_id (so the row isn't duplicated).
    # Several local runs can name one output repo (a resume chain publishes
    # every link into the first run's repo). `local` is newest-first, and the
    # collapse below overwrites unconditionally, so without this the LAST one
    # processed — the OLDEST run — would end up owning the row, and `job_id`
    # would point at a run that is not the one the listing means. First claim
    # wins, which is newest-wins, matching _job_outranks' own tie-break.
    claimed_repos: set[str] = set()
    for item in local:
        hub_repo = item.get("hf_repo_id")
        if hub_repo and hub_repo in merged and hub_repo not in claimed_repos:
            claimed_repos.add(hub_repo)
            existing = merged[hub_repo]
            existing["source"] = "both"
            # The local checkpoint is authoritative for detail: override the
            # hub row's placeholder id/name and fill in the checkpoint fields it
            # couldn't know (per the contract, `id`/`name` follow the local run
            # for a "both" model).
            for key in (
                "id",
                "name",
                "policy_type",
                "dataset",
                "steps",
                "target_steps",
                "state",
                "path",
                "origin",
                "job_id",
            ):
                if item.get(key) is not None:
                    existing[key] = item[key]
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[item["id"]] = {**item, "source": "local"}

    # Downloaded/imported checkpoints (the local models dir), keyed by repo_id.
    # One that matches a hub row flips it to "both" and fills in the checkpoint
    # detail the hub row couldn't know (path/policy_type/steps) — that's the
    # "download to local" listing flip. A checkpoint the Hub doesn't list (a
    # disk import, or a foreign repo when offline) stays a "local" row.
    for item in list_downloaded_models():
        rid = item["id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # The on-disk checkpoint is authoritative for detail: its
            # config.json-derived values override the hub row's tag/name-derived
            # ones (same local-wins rule as the run collapse above).
            for key in ("policy_type", "dataset", "steps", "path", "origin", "job_id"):
                if item.get(key) is not None:
                    existing[key] = item[key]
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = item

    # Read once, used twice: the imported fold below consults it to avoid
    # resurrecting a hidden repo under a different casing, and the filter at the
    # end applies it for real.
    hidden = get_hidden_models()

    # Fold in the user's pinned custom Hub models (the "Add model" chooser's
    # add-from-HF flow), mirroring list_all_datasets' pin fold. A pin already
    # covered by a hub/both row is redundant and skipped. A pinned repo whose
    # checkpoint was downloaded (the scan added a "local" row) is a Hub model
    # with a local copy: flip to "both", stamp the hf_repo_id the scan couldn't
    # know, and keep saved_custom so "remove from list" (unpin) stays available.
    for repo_id in get_saved_custom_models():
        existing = merged.get(repo_id)
        if existing is None:
            merged[repo_id] = {
                "id": repo_id,
                "name": repo_id,
                # A pinned repo the Hub listing didn't return (private, GFW-dropped,
                # or untagged) still has an inferable type from its name prefix
                # (act_… / smolvla_…) — infer it here instead of dropping to None,
                # so the picker shows the policy label. Mirrors the tag/name
                # inference the hub listing does via _list_author_models.
                "policy_type": _hub_policy_type(None, repo_id.split("/", 1)[-1]),
                "dataset": None,
                "steps": None,
                "target_steps": None,
                "state": None,
                "path": None,
                "last_modified": None,
                "hf_repo_id": repo_id,
                "private": False,
                "source": "hub",
                "origin": "hub-untracked",
                "job_id": None,
                "saved_custom": True,
            }
        elif existing["source"] == "local":
            existing["source"] = "both"
            existing["hf_repo_id"] = repo_id
            existing["saved_custom"] = True

    # Registered imports (see list_imported_models) — the source that closes the
    # "imported here, invisible there" hole, folded in only where nothing above
    # could produce the row.
    #
    # AFTER the pin fold, not before, and the ordering is load-bearing. An
    # import stores whatever casing the user pasted, while a pin is keyed by the
    # casing the pin was saved under. The pin fold matches exactly
    # (`merged.get(repo_id)`), so an import folded in first as "Acme/Policy"
    # leaves the pin for "acme/policy" unmatched and the listing shows one repo
    # twice. Folding imports last means every case-insensitive comparison below
    # is made against a `merged` that already holds the pins.
    #
    # Identity is matched four ways, because an import can arrive wearing any of
    # them:
    #   * its own key — the repo id or the run id the row is filed under;
    #   * the REPO it points at, compared case-insensitively (the rule
    #     jobs.find_imported dedups on: HF repo ids are practically unique
    #     across casing and the Hub redirects between them, so an exact compare
    #     reintroduces the very duplicate this exists to prevent);
    #   * the DIRECTORY it points at — nothing stops an import being registered
    #     against a completed run's own output dir or against the downloaded
    #     model store, and then ONE checkpoint is offered twice under two ids
    #     that both work. Two rows resolving the same pretrained dir are the
    #     same weights by definition;
    #   * the HIDDEN list, again case-insensitively. The final filter below is
    #     an exact-key match, so without this a repo hidden as "Acme/Policy"
    #     walks back into the listing the moment "acme/policy" is imported —
    #     "removed from list" silently undone by an unrelated action.
    #
    # A matched row is NOT renamed after the import. `_enrich_repo_rows_from_jobs`
    # already refuses imported records for naming, because an import's title is
    # derived from the very repo id the row already shows, while the run that
    # trained the weights says something the repo id cannot.
    merged_repos = {(row.get("hf_repo_id") or "").lower() for row in merged.values()}
    merged_repos.discard("")
    merged_paths = {row.get("path") for row in merged.values() if row.get("path")}
    hidden_repos = {h.lower() for h in hidden}
    # An import whose ROW is deduped away still contributes its identity: it is
    # the registry record that tracks those weights, and a surface keying on
    # `job_id` (the models library does) would otherwise lose a card it used to
    # show for every import of a repo the user already owns. Applied after the
    # run enrichment below, so a real training run still wins the slot.
    # Cloud runs FIRST, so a repo a real run published owns its row before an
    # import of the same repo can claim it — a run that trained the weights is a
    # better identity than a record that merely registered a pointer to them.
    # Only rows the Hub listing did not already return are added; when it did,
    # _enrich_repo_rows_from_jobs stamps the same job_id and origin onto it.
    for item in list_cloud_models():
        repo = item["hf_repo_id"].lower()
        if item["id"] in merged or repo in merged_repos or repo in hidden_repos:
            continue
        merged_repos.add(repo)
        merged[item["id"]] = item

    import_job_by_repo: dict[str, str] = {}
    import_job_by_path: dict[str, str] = {}
    for item in list_imported_models():
        rid = item["id"]
        repo = (item.get("hf_repo_id") or "").lower()
        job_id = item.get("job_id")
        if repo and job_id:
            import_job_by_repo.setdefault(repo, job_id)
        # Same rule for the other way a row gets deduped away: an import may
        # point AT the downloaded-model store, or at a run's own output dir, and
        # then the path check drops its row. The record still tracks those
        # weights and still has to be reachable.
        if item.get("path") and job_id:
            import_job_by_path.setdefault(item["path"], job_id)
        if rid in merged:
            continue
        if repo and (repo in merged_repos or repo in hidden_repos):
            continue
        path = item.get("path")
        if path and path in merged_paths:
            continue
        if repo:
            merged_repos.add(repo)
        if path:
            merged_paths.add(path)
        merged[rid] = item

    # Name the repo-keyed rows after the run that produced them, now that every
    # source has merged in. A tracked run whose output repo is shared with its
    # resume chain (or simply never collapsed into a local row) would otherwise
    # reach the picker as a bare repo id with null steps/dataset — under the
    # name of the FIRST run in the chain, never the one that finished. Runs
    # after the merge and before the hidden filter so it can't resurrect a
    # hidden row, and so local-checkpoint detail always wins (see the helper).
    _enrich_repo_rows_from_jobs(merged)

    # Then the imports, for the rows no run claimed. Deliberately second: a run
    # that TRAINED the weights is a better identity than a record that merely
    # registered a pointer to them (its config is a placeholder — see
    # _jobs_by_hub_repo, which skips imports for the same reason).
    if import_job_by_repo or import_job_by_path:
        for row in merged.values():
            if row.get("job_id") is not None:
                continue
            job_id = import_job_by_repo.get((row.get("hf_repo_id") or "").lower())
            if job_id is None and row.get("path"):
                job_id = import_job_by_path.get(row["path"])
            if job_id:
                row["job_id"] = job_id

    # Then separate the rows the naming above left indistinguishable: two runs
    # of one policy on one dataset carry the same auto-generated name, so the
    # picker would show the same label twice. Immediately after the enrichment
    # so it sees the final names, and still before the hidden filter.
    _dedupe_listing_names(merged)

    # Hidden models ("removed from list") are filtered LAST — after the
    # hub/local/downloaded merge and the pin fold — so a hidden id can't
    # resurface via a pin or a local copy. Re-pinning auto-unhides (see
    # /models/custom). Mirrors list_all_datasets.
    if hidden:
        # The repo comparison is case-INSENSITIVE. Hiding a repo stores whatever
        # casing the UI had, while the Hub listing returns the canonical one, so
        # an exact-only filter let a hidden repo walk back into the listing
        # under a different spelling — "removed from list" silently undone. The
        # id comparison stays exact: a run id is not a repo id, and lower-casing
        # it could match something it should not.
        hidden_repos = {h.lower() for h in hidden}
        merged = {
            rid: row
            for rid, row in merged.items()
            if rid not in hidden and (row.get("hf_repo_id") or "").lower() not in hidden_repos
        }

    out = list(merged.values())
    out.sort(key=lambda m: m.get("last_modified") or "", reverse=True)
    return out, hub_status


def _cached_build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """`_build_listing` behind the short merged-listing TTL.

    ONE cache, two projections: /models reads the rows and /skills reads the
    rows plus the Hub status. Caching the build rather than each projection is
    what makes it impossible for the two endpoints to answer from different
    snapshots — which is the failure they exist to stop reproducing."""
    global _listing_cache

    now = time.monotonic()
    with _listing_cache_lock:
        cached = _listing_cache
        if cached is not None and (now - cached["at"]) < _LISTING_CACHE_TTL_S:
            return cached["value"], cached["hub"]

    out, hub_status = _build_listing()

    with _listing_cache_lock:
        _listing_cache = {"at": time.monotonic(), "value": out, "hub": hub_status}
    return out, hub_status


def list_all_models() -> list[dict[str, Any]]:
    """The /models array: model ARTIFACTS, for the fine-tune base picker.

    Shape unchanged — a bare array — because that is what the base picker and
    the existing clients consume, and because its population is deliberately
    wider than /skills': a foundation checkpoint is a legitimate fine-tune base
    and not a deployable skill."""
    return _cached_build()[0]


# ---------------------------------------------------------------------------
# Skills — the deployable projection.
# ---------------------------------------------------------------------------
#
# /models answers "what can I fine-tune FROM"; /skills answers "what can I run
# ON THE ROBOT". They were one endpoint feeding two surfaces that then filtered
# it differently — the deploy picker by usable-checkpoint, the models library by
# runner type — which is why the two lists disagreed about what a skill is.
# There is now one definition, here, and both surfaces read it.


def _weights_state(row: dict[str, Any]) -> str:
    """Whether this row's weights can be loaded: ready | unverified | none.

    "unverified" is the deliberate one. A Hub-backed row is NOT probed to
    confirm the push landed — doing that per row is a serial, un-deadlined
    round-trip each, on a listing that already fans out per author. The claim
    is made from the record and settled at selection, by the download the deploy
    path performs anyway: loudly, once, and only for the row the user picked."""
    path = row.get("path")
    if path:
        # A PATH is not weights. `_final_checkpoint_dir` accepts a checkpoint dir
        # on the strength of its config.json alone (that is the registry's
        # "a checkpoint exists" question), while loading needs
        # model.safetensors or an adapter pair. A run killed mid-save leaves
        # exactly the first without the second, and calling that "ready" would
        # offer a skill that cannot start.
        return "ready" if _has_loadable_weights(Path(path)) else "none"
    if row.get("hf_repo_id"):
        return "unverified"
    return "none"


def _explained_rows(
    known_ids: set[str], known_paths: set[str], known_repos: set[str]
) -> list[dict[str, Any]]:
    """Runs the /models listing omits, returned WITH the reason they were omitted.

    Two kinds, and the picker was silent about both:

      * SUPERSEDED — a run another run resumed. The chain collapses to its tip,
        which is right, but "my run finished and it is not in the list" deserves
        "it is represented by <tip>", not an absence.
      * FAILED WITH WEIGHTS — excluded from /models on the principle that a
        confirmed non-zero exit should not sit next to real models. The Train
        panel's own card has always ignored that principle (its Run row gates on
        having a checkpoint, not on state), so the rule was enforced in one
        panel and contradicted in the next. An OOM at step 90k with a good 80k
        checkpoint is a usable skill; it is listed, carrying `state: "failed"`
        so the UI can badge it rather than pretend it succeeded.

      * BROKEN IMPORT — an import is a POINTER, so the directory it names can be
        moved, emptied or deleted afterwards. The listing drops it (offering a
        path that fails at load is worse than omitting it), but dropping it from
        the LIBRARY too would leave a registered record the user can no longer
        see, and therefore can no longer delete. It is listed with
        `weights: "none"`, which keeps it out of the picker and visible where it
        can be cleaned up.

    Rows already produced by the merge are skipped THREE ways — by id, by
    resolved path, and by repo id. The last matters because a failed local run
    that was pushed to the Hub already reaches the merge as a repo-keyed row: its
    id there is the repo, its path is null, so neither of the other two checks
    can see it and the run would be listed twice."""
    out: list[dict[str, Any]] = []
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT, with_checkpoints=False):
        if record.runner == "imported":
            # Only the ones list_imported_models could not build a row for; the
            # healthy ones are already in the merge.
            if record.id in known_ids or (record.hf_repo_id or "").lower() in known_repos:
                continue
            if _imported_model_summary(record) is not None:
                continue
            out.append(
                {
                    "id": record.id,
                    "name": record.display_name or record.name,
                    "policy_type": None,
                    "dataset": None,
                    "steps": None,
                    "target_steps": None,
                    "state": record.state,
                    "path": None,
                    "last_modified": _epoch_iso(record.ended_at or record.started_at),
                    "hf_repo_id": None,
                    "source": "local",
                    "origin": "imported",
                    "job_id": record.id,
                    "superseded_by": None,
                }
            )
            continue

        if record.runner != "local" or record.state not in ("done", "interrupted", "failed"):
            continue
        # A leaf that reached a clean terminal state is already in the merge.
        if not record.child_ids and record.state != "failed":
            continue
        pretrained_dir = _final_checkpoint_dir(record)
        if pretrained_dir is None:
            continue
        row = _local_model_summary(record, pretrained_dir)
        if row["id"] in known_ids or row["path"] in known_paths:
            continue
        if (record.hf_repo_id or "").lower() in known_repos:
            continue
        row["superseded_by"] = record.child_ids[0] if record.child_ids else None
        out.append(row)
    return out


def list_skills() -> dict[str, Any]:
    """The /skills envelope: every trained policy, each saying whether it can run.

    A projection of the same merged build /models uses, plus the rows that build
    deliberately drops (see _explained_rows), annotated with:

      * `weights`      — ready | unverified | none
      * `superseded_by` — the run that represents this one's chain, or None
      * `deployable`   — weights are loadable AND nothing supersedes it

    `deployable` is DERIVED, never stored: a checkpoint dir can be deleted and a
    resume can land at any time, so a persisted flag would rot with no writer
    nearby. That is the same reason `checkpoint_count` and `child_ids` are
    computed on read in the registry.

    The envelope carries the Hub half's status so the UI can tell "the Hub was
    unreachable" from "you own nothing" — the two states that used to render
    identically, as an empty list."""
    rows, hub_status = _cached_build()

    known_ids = {row["id"] for row in rows}
    known_paths = {row["path"] for row in rows if row.get("path")}
    known_repos = {(row.get("hf_repo_id") or "").lower() for row in rows}
    known_repos.discard("")
    hidden = get_hidden_models()
    extra = [row for row in _explained_rows(known_ids, known_paths, known_repos) if row["id"] not in hidden]

    skills: list[dict[str, Any]] = []
    for row in [*rows, *extra]:
        weights = _weights_state(row)
        superseded_by = row.get("superseded_by")
        skills.append(
            {
                **row,
                "weights": weights,
                "superseded_by": superseded_by,
                "deployable": weights != "none" and superseded_by is None,
            }
        )

    skills.sort(key=lambda m: m.get("last_modified") or "", reverse=True)
    return {"skills": skills, "hub": hub_status}


# ---------------------------------------------------------------------------
# Per-model info card.
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under `path`. Unreadable files are skipped."""
    import os

    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def get_model_info(id_or_repo: str) -> dict[str, Any] | None:
    """Detail view of one model: policy type, dataset, steps, size, path/repo.

    Resolves `id_or_repo` as a local run id first (a completed local run with a
    checkpoint); if that misses, treats it as a Hub repo id and reads the
    checkpoint config from the Hub. Returns None when neither resolves.

    Local: reads train_config.json + config.json from the final checkpoint and
    walks the dir for its on-disk size. Hub: reads the root config.json via the
    Hub (the network call is the caller's — the info card fetches it lazily) and
    reports no size (the repo isn't on disk)."""
    record = _find_local_record(id_or_repo)
    if record is not None:
        pretrained_dir = _final_checkpoint_dir(record)
        assert pretrained_dir is not None  # _find_local_record guarantees it
        summary = _local_model_summary(record, pretrained_dir)
        summary["size_bytes"] = _dir_size_bytes(pretrained_dir)
        return summary

    # A downloaded/imported checkpoint in the local models dir — filesystem
    # only, so it works offline (that's the point of downloading a model).
    model_dir = _downloaded_model_dir(id_or_repo)
    if model_dir is not None:
        summary = _downloaded_model_summary(id_or_repo, model_dir)
        summary["size_bytes"] = _dir_size_bytes(model_dir)
        return summary

    # Not local at all — try the Hub. Offline ⇒ can't read a hub-only model.
    if hf_hub_offline():
        return None
    return _hub_model_info(id_or_repo)


# In-process cache of per-repo Hub model metadata (the /models/info hub
# branch), mirroring datasets._HUB_STATUS_CACHE conventions: successful answers
# are memoized for the process lifetime; the offline/error degrade is NEVER
# cached, so connectivity returning is picked up on the next check. Invalidated
# alongside the listing cache on the mutations that change a repo (upload,
# download-complete, delete, hide) — see invalidate_model_hub_info.
_MODEL_HUB_INFO_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_HUB_INFO_LOCK = threading.Lock()


def invalidate_model_hub_info(repo_id: str) -> None:
    """Drop the cached Hub metadata for `repo_id`, so the next /models/info
    re-reads it (e.g. after an upload changed the repo's tags/size)."""
    with _MODEL_HUB_INFO_LOCK:
        _MODEL_HUB_INFO_CACHE.pop(repo_id, None)


def _hub_model_probe(repo_id: str) -> dict[str, Any] | None:
    """The pre-existing file-tree probe for a Hub model repo: reads the
    checkpoint config from the Hub (root config.json, or the latest
    checkpoints/<step>/pretrained_model tree) for the policy type + step. Two
    Hub calls (list_repo_files + hf_hub_download), so it's the FALLBACK when the
    single model_info call fails or leaves the policy type unknown. Returns None
    if the repo has no usable model config (or can't be read)."""
    from .jobs import _list_imported_hub

    api = shared_hf_api()
    checkpoints = _list_imported_hub(api, repo_id)
    if not checkpoints:
        return None
    final = checkpoints[-1]
    policy_type = None
    steps: int | None = final.step or None
    try:
        cfg = _read_checkpoint_config(final)
        policy_type = cfg.get("type")
    except Exception as exc:
        logger.info("Could not read hub model config for %s: %s", repo_id, exc)

    return {
        "id": repo_id,
        "name": repo_id.split("/", 1)[-1],
        "policy_type": policy_type,
        "dataset": None,
        "steps": steps,
        "path": None,
        "hf_repo_id": repo_id,
        "size_bytes": None,
        "source": "hub",
    }


def _hub_model_info(repo_id: str) -> dict[str, Any] | None:
    """Detail for a Hub-only model repo, built around ONE
    ``HfApi.model_info(expand=...)`` call: policy type (card model_name → tags →
    name prefix, all against KNOWN_POLICY_TYPES), base dataset (card
    ``datasets``), size on the Hub (``usedStorage``), private flag, and last
    modified. Only when the cheap signals leave the policy type unknown does it
    fall back to the old two-call file-tree probe (_hub_model_probe), which also
    recovers the checkpoint step.

    Degrade-not-crash: if model_info itself fails this falls back to the probe
    (itself best-effort) rather than raising, and only successful answers are
    cached (see _MODEL_HUB_INFO_CACHE) so transient failures self-heal."""
    with _MODEL_HUB_INFO_LOCK:
        cached = _MODEL_HUB_INFO_CACHE.get(repo_id)
    if cached is not None:
        return dict(cached)

    api = shared_hf_api()
    try:
        info = api.model_info(repo_id, expand=["cardData", "tags", "lastModified", "private", "usedStorage"])
    except Exception as exc:
        # Transient / not-found / auth — degrade to the old probe, uncached.
        logger.info("model_info(%s) failed: %s", repo_id, exc)
        return _hub_model_probe(repo_id)

    name = repo_id.split("/", 1)[-1]
    tags = list(getattr(info, "tags", None) or [])
    card = getattr(info, "card_data", None)

    # lerobot's generate_model_card sets model_name to the policy type — the
    # most explicit signal; then the tag / name-prefix inference.
    policy_type = None
    card_model_name = getattr(card, "model_name", None) if card is not None else None
    if card_model_name in KNOWN_POLICY_TYPES:
        policy_type = card_model_name
    if policy_type is None:
        policy_type = _hub_policy_type(tags, name)

    # Base dataset(s) from the card metadata; a list takes its first entry.
    dataset = getattr(card, "datasets", None) if card is not None else None
    if isinstance(dataset, list):
        dataset = dataset[0] if dataset else None
    if not isinstance(dataset, str):
        dataset = None

    last_modified = getattr(info, "last_modified", None)
    used_storage = getattr(info, "used_storage", None)

    row: dict[str, Any] = {
        "id": repo_id,
        "name": name,
        "policy_type": policy_type,
        "dataset": dataset,
        "steps": None,
        "path": None,
        "last_modified": last_modified.isoformat() if last_modified else None,
        "hf_repo_id": repo_id,
        "private": bool(getattr(info, "private", False)),
        "size_bytes": int(used_storage) if isinstance(used_storage, int) else None,
        "source": "hub",
    }

    if row["policy_type"] is None:
        # Last resort: the old config.json download path (also recovers steps).
        probe = _hub_model_probe(repo_id)
        if probe is not None:
            row["policy_type"] = probe["policy_type"]
            row["steps"] = probe["steps"]

    with _MODEL_HUB_INFO_LOCK:
        _MODEL_HUB_INFO_CACHE[repo_id] = dict(row)
    return row


# ---------------------------------------------------------------------------
# Upload a local checkpoint to the Hub as a model repo.
# ---------------------------------------------------------------------------


def _upload_model_error(exc: Exception) -> ModelError:
    """Map a huggingface_hub exception from create_repo/upload_folder to a
    ModelError with a legible message. A 401/auth failure or a 403 permission
    failure becomes a clear "you can't push this" message; anything else
    degrades to a generic Hub-failure 502. Reuses record._upload_auth_error so
    the 401 maps identically to the dataset upload path."""
    from .record import _upload_auth_error

    auth = _upload_auth_error(exc)
    if auth is not None:
        return ModelError(403, auth["message"], docs_url=auth.get("docs_url"))

    err_text = str(exc).lower()
    if "403" in err_text or "forbidden" in err_text or "permission" in err_text:
        return ModelError(
            403,
            "You don't have permission to push this model to the Hub. You can "
            "only upload models to a namespace you can write to.",
        )
    return ModelError(502, f"The Hub rejected the upload: {exc}")


def _default_model_repo_id(record: JobRecord) -> str:
    """The Hub repo id a local run uploads to when the caller names none.

    Namespaced to the logged-in user (models can't push to a bare name), using
    the run id as the repo name so it's stable and legible. cached_whoami is the
    same auth source the listing uses."""
    info = cached_whoami()
    namespace = info["name"] if info else None
    return f"{namespace}/{record.id}" if namespace else record.id


def upload_local_model(model_id: str, repo_id: str | None = None) -> dict[str, Any]:
    """Push a completed local run's final checkpoint to the Hub as a MODEL repo.

    PUBLIC by default and tagged makermods / openbooth / MakerModsLab via with_makermodslab_tag
    (the same funnel datasets/policies use). Steps:
      1. resolve the local run + its final pretrained_model dir (404 if missing);
      2. create_repo(repo_id, repo_type="model", private=False, exist_ok=True);
      3. upload_folder(folder_path=<checkpoint>, repo_id, repo_type="model");
      4. metadata_update(repo_id, {"tags": with_makermodslab_tag(None)}, repo_type=
         "model", overwrite=True).

    Refuses offline (can't mutate the Hub) with a clear error. Auth/permission
    failures map like the dataset upload path. Invalidates the model-listing
    cache so the freshly-pushed repo appears immediately. Returns
    {repo_id, url, tags}.

    NOTE: this runs synchronously (a small policy checkpoint, unlike a
    multi-GB dataset) — the route calls it inline."""
    if hf_hub_offline():
        raise ModelError(400, "The Hub is offline — you can't upload a model right now.")

    record = _find_local_record(model_id)
    if record is None:
        # A mid-chain parent is a real run holding real weights, so "not found"
        # would be both wrong and unhelpful. Name the run that represents the
        # chain instead — that is the one whose upload carries all of it.
        continued_by = _superseded_by(model_id)
        if continued_by:
            raise ModelError(
                409,
                f"Model {model_id!r} was continued by {continued_by}, so it is not a model "
                "on its own. Upload the continuation instead — it carries the whole chain.",
            )
        raise ModelError(
            404,
            f"No completed local model with a saved checkpoint found for {model_id!r}.",
        )
    pretrained_dir = _final_checkpoint_dir(record)
    assert pretrained_dir is not None  # _find_local_record guarantees it

    target_repo_id = repo_id or record.hf_repo_id or _default_model_repo_id(record)
    api = shared_hf_api()
    try:
        api.create_repo(target_repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=str(pretrained_dir),
            repo_id=target_repo_id,
            repo_type="model",
        )
    except Exception as exc:
        logger.info("Upload of local model %s -> %s failed: %s", model_id, target_repo_id, exc)
        raise _upload_model_error(exc) from exc

    # Stamp the policy type as a tag alongside the org tags, so future MakerMods Lab
    # uploads are self-describing on the Hub (the same tag lerobot-native
    # pushes stamp; _hub_policy_type's name-prefix fallback covers old repos).
    policy_tag = None
    try:
        policy_tag = json.loads((pretrained_dir / "config.json").read_text()).get("type")
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s for tag stamping: %s", pretrained_dir / "config.json", exc)
    final_tags = with_makermodslab_tag([policy_tag] if policy_tag in KNOWN_POLICY_TYPES else None)
    try:
        metadata_update(target_repo_id, {"tags": final_tags}, repo_type="model", overwrite=True)
    except Exception as exc:
        # The weights are already pushed; a tag write failing is non-fatal to the
        # upload, but surface it so the user knows the tags didn't land.
        logger.info("Tagging model %s failed after upload: %s", target_repo_id, exc)
        raise _upload_model_error(exc) from exc

    invalidate_model_listing_cache()
    # The repo's tags/size just changed — drop its cached hub metadata too.
    invalidate_model_hub_info(target_repo_id)
    logger.info("Uploaded local model %s to %s (tags=%s)", model_id, target_repo_id, final_tags)
    return {
        "repo_id": target_repo_id,
        "url": f"https://huggingface.co/{target_repo_id}",
        "tags": final_tags,
    }


# ---------------------------------------------------------------------------
# Delete a local model (its training-run output dir).
# ---------------------------------------------------------------------------


def _model_in_use(target_dir: Path) -> str | None:
    """If a running inference is reading a checkpoint under `target_dir`,
    return a legible reason to refuse deleting it; else None. Shaped like
    datasets._dataset_in_use (a false "in use" is safer than yanking a
    directory a live subprocess is reading).

    Containment on RESOLVED paths: the inference path (the pretrained_model
    dir rollout captured at start) equals the target, or lives anywhere under
    it — a run/downloaded dir contains its checkpoints/<step>/pretrained_model.

    Lazy import: rollout pulls heavy lerobot modules at import time, and the
    models browser must stay cheap to import (no cycle — rollout never imports
    models)."""
    from . import rollout as _rollout

    in_use = _rollout.inference_in_use_path()
    if not in_use:
        return None
    try:
        active = Path(in_use).resolve()
        target = target_dir.resolve()
    except OSError:
        return None
    if active == target or target in active.parents:
        return "This model is being used by a running inference. Stop it first."
    return None


def delete_local_model(model_id: str) -> dict[str, Any]:
    """Delete a local model's LOCAL files — a training run's output dir, or a
    downloaded/imported checkpoint's dir in the local models dir.

    SAFETY: a run dir is resolved strictly under the job registry's
    outputs/train/ root, a downloaded/imported dir strictly under the local
    models root (both traversal-guarded); anything escaping is refused. Never
    touches the Hub — only local files, so deleting the local copy of a "both"
    model leaves the Hub row listed.

    Training runs coordinate with the registry: a RUNNING job's dir is never
    deleted (reuses JobRegistry.delete, which raises for a running job), and
    deleting through the registry also drops the record + persisted job.json.
    Ids that aren't registry records fall through to the downloaded/imported
    probe (_downloaded_model_dir). Invalidates the model-listing cache.

    Returns {deleted: True, id}. Raises ModelError (404 unknown, 409 running,
    400 unsafe path, 502 delete failure)."""
    from .jobs import JobNotFoundError, JobNotRunningError

    try:
        record = job_registry.get(model_id)
    except JobNotFoundError as exc:
        # Not a training run — a downloaded/imported checkpoint in the local
        # models dir? _downloaded_model_dir is traversal-guarded, so anything it
        # returns is strictly inside the models root and safe to remove.
        model_dir = _downloaded_model_dir(model_id)
        if model_dir is None:
            raise ModelError(404, f"Model {model_id!r} not found.") from exc
        in_use = _model_in_use(model_dir)
        if in_use is not None:
            raise ModelError(409, in_use) from None
        try:
            shutil.rmtree(model_dir)
        except OSError as rm_exc:
            logger.error("Failed to delete downloaded model %s: %s", model_id, rm_exc)
            raise ModelError(502, f"Failed to delete model: {rm_exc}") from rm_exc
        invalidate_model_listing_cache()
        # The local copy is gone; the card falls back to the hub view — drop the
        # cached hub metadata so it re-reads fresh.
        invalidate_model_hub_info(model_id)
        logger.info("Deleted downloaded/imported model %s (dir %s)", model_id, model_dir)
        return {"deleted": True, "id": model_id}

    if record.runner != "local":
        raise ModelError(
            400,
            f"Model {model_id!r} is not a local training run, so there's nothing local to delete.",
        )

    # Resolve the run dir and refuse anything outside outputs/train/. The
    # registry owns the (resolved) root; the run dir is <root>/<id>. Guarding on
    # the resolved paths catches a traversal id or a symlinked output_dir.
    root = job_registry._output_root  # resolved in JobRegistry.__init__
    run_dir = (root / record.id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ModelError(
            400,
            f"Refusing to delete {model_id!r}: its directory resolves outside the training-output root.",
        ) from exc
    if run_dir == root:
        raise ModelError(400, f"Refusing to delete {model_id!r}: resolves to the output root itself.")

    # A COMPLETED run's checkpoint can still be an active inference target —
    # the registry's running-job guard below doesn't cover that.
    in_use = _model_in_use(run_dir)
    if in_use is not None:
        raise ModelError(409, in_use)

    try:
        job_registry.delete(model_id)
    except JobNotRunningError as exc:
        # JobRegistry.delete raises JobNotRunningError when the job IS running
        # (its guard refuses to delete a live run's dir).
        raise ModelError(
            409,
            f"Model {model_id!r} is still training — stop the run before deleting it.",
        ) from exc
    except JobHasChildrenError as exc:
        # Mid-chain delete, reached from the MODEL library rather than the jobs
        # one: another run resumed out of this run's checkpoint dir, which the
        # delete would take with it. Same refusal and same shape the /jobs
        # route returns (409, naming the continuations) — without this it fell
        # into the catch-all below and surfaced as a 502 "Failed to delete".
        continued_by = ", ".join(repr(cid) for cid in exc.child_ids)
        raise ModelError(
            409,
            f"Model {model_id!r} was continued by {continued_by}, which would be left "
            "pointing at a deleted run. Delete the continuation(s) first.",
        ) from exc
    except JobNotFoundError as exc:
        raise ModelError(404, f"Model {model_id!r} not found.") from exc
    except Exception as exc:
        logger.error("Failed to delete local model %s: %s", model_id, exc)
        raise ModelError(502, f"Failed to delete model: {exc}") from exc

    invalidate_model_listing_cache()
    if record.hf_repo_id:
        # A pushed run's hub row survives the local delete — drop its cached
        # hub metadata so the card re-reads fresh.
        invalidate_model_hub_info(record.hf_repo_id)
    logger.info("Deleted local model %s (run dir %s)", model_id, run_dir)
    return {"deleted": True, "id": model_id}


# ---------------------------------------------------------------------------
# Download a Hub model into the local models dir (background, pollable).
# ---------------------------------------------------------------------------


# Optimizer/scheduler state, excluded from everything the Models page pulls.
# Two patterns because huggingface_hub filters with fnmatch, NOT path-aware
# globbing: `training_state/**` catches the root one a flat push leaves, and
# `*/training_state/**` catches the per-step ones (fnmatch's `*` spans `/`, so
# it covers `checkpoints/<step>/training_state/...` at any depth).
_TRAINING_STATE_IGNORE = ["training_state/**", "*/training_state/**"]


def _copy_snapshot_into_store(snapshot: Path, target: Path) -> None:
    """Copy a hub-cache snapshot dir into the local models store.

    Two things this must NOT do naively:

    * **Symlinks.** A cache snapshot is a farm of symlinks into the sibling
      ``blobs/`` dir; copying them as links would leave the store pointing at
      files `huggingface_hub` may garbage-collect (``delete_revisions``) or that
      simply vanish if the user clears their hub cache. Every file is
      dereferenced (``shutil.copyfile`` copies CONTENT through a symlink), so
      the store stays the self-contained directory the rest of models.py
      assumes. It also leaves dest mtimes at "now", which is what the listing's
      `_dir_mtime_iso` should report for a just-downloaded model — a blob's own
      mtime could be months old.
    * **training_state.** A cache entry can predate this exclusion (inference's
      @root fetch, or a plain `hf_hub_download`, may have populated it), so the
      snapshot on disk can hold optimizer state even though we asked
      snapshot_download to skip it. Filtering is delegated to huggingface_hub's
      OWN `filter_repo_objects` — the same function snapshot_download filters
      with — so the copied tree is byte-for-byte the tree the local_dir download
      would have produced, rather than a hand-rolled guess at fnmatch semantics.

    Copies IN PLACE into `target` rather than staging in a temp dir and swapping:
    it matches what the local_dir download does (write over whatever is there),
    keeps a re-download of an already-present model from ever having a window
    where the model is missing, and a half-finished copy is handled the same way
    a half-finished download is — the caller falls back to the network path,
    which rewrites these very paths, and `_cleanup_partial_model` removes the
    dir if the result still doesn't resolve."""
    rel_paths = [p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()]
    for rel in filter_repo_objects(rel_paths, ignore_patterns=_TRAINING_STATE_IGNORE):
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / rel, dest)


def _fetch_via_hub_cache(repo_id: str, target: Path) -> bool:
    """Materialize `repo_id` into `target` from the shared hub cache, or return
    False to say "not applicable — do the normal download".

    Closes the second half of design-debt F6. `_fetch_model_snapshot` downloads
    with ``local_dir=``, and huggingface_hub 1.21.0's local_dir mode neither
    reads nor populates the shared cache (it is the deliberate "give me a plain
    directory, no cache machinery" mode). So a repo inference had already pulled
    into the cache was downloaded a SECOND time, in full, the moment the user
    clicked Download on the Models page. (`rollout._local_store_policy_path`
    covers the opposite order: Models page first, inference second.)

    Deliberately routed through a cache-mode ``snapshot_download`` — no
    ``local_dir`` — rather than copying the newest snapshot dir straight out of
    the cache. That call dedupes against the cache and pulls only what is
    missing or changed, which means:

      * a PARTIAL cache entry (someone `hf_hub_download`-ed a single file, so
        `_hub_cache_has_repo` says yes but the snapshot is one symlink) gets
        COMPLETED rather than copied as a broken tree, and
      * the result is revision-correct — we get `main` as it is now, not
        whatever stale revision happens to sit in the cache.

    It is an optimization and must never become a new failure mode, so ANY
    failure (HF error, a cache entry mutated out from under us, a copy that runs
    out of disk) returns False and lets the caller do the plain download. The
    exception is logged, not swallowed silently — a cache path that always fails
    would otherwise just look like "downloads are slow"."""
    if not _hub_cache_has_repo(repo_id):
        return False
    try:
        snapshot = snapshot_download(
            repo_id,
            repo_type="model",
            ignore_patterns=_TRAINING_STATE_IGNORE,
        )
        _copy_snapshot_into_store(Path(snapshot), target)
    except Exception as exc:  # noqa: BLE001 - optimization only; the caller re-downloads
        logger.warning(
            "Could not serve %s from the HF hub cache (%s) — falling back to a full download",
            repo_id,
            exc,
        )
        return False
    logger.info("Served %s from the HF hub cache instead of re-downloading it", repo_id)
    return True


def _fetch_model_snapshot(repo_id: str) -> None:
    """Snapshot a Hub model repo into ``<local models root>/<repo_id>/``.

    After the fetch the dir must resolve to a usable checkpoint
    (_resolve_pretrained_dir) — a repo with no config.json / checkpoints tree is
    not a policy and is rejected (the manager's cleanup then removes it).
    Invalidates the /models listing cache so the source flips to "both"
    immediately.

    Optimizer/scheduler state is excluded. A training repo carries a
    ``training_state/`` dir next to every ``pretrained_model/`` (and one at the
    repo root for a flat push), which exists only to RESUME training — nothing
    that reads a downloaded model here (inference, the checkpoint browser,
    _resolve_pretrained_dir) ever opens it, and it is hundreds of MB per step.
    Observed: a 5.6 GB local model dir of which ~3.5 GB was training_state.
    ``rollout._resolve_policy_path``'s @root fetch already dropped exactly these
    for exactly this reason; the two call sites now agree. The
    ``checkpoints/<step>/pretrained_model`` trees are deliberately KEPT — the
    checkpoint browser lists and inference selects individual steps from them.

    When the shared hub cache ALREADY holds the repo, the bytes are taken from
    there instead of off the network — see `_fetch_via_hub_cache`, which is a
    pure optimization and falls back to the download below on any failure.
    """
    target = _local_models_root() / repo_id
    if not _fetch_via_hub_cache(repo_id, target):
        snapshot_download(
            repo_id,
            repo_type="model",
            local_dir=str(target),
            ignore_patterns=_TRAINING_STATE_IGNORE,
        )
    if _resolve_pretrained_dir(target) is None:
        raise ModelError(
            400,
            f"'{repo_id}' doesn't look like a policy checkpoint — no config.json "
            "or checkpoints/<step>/pretrained_model tree in the repo.",
        )
    invalidate_model_listing_cache()
    # The card flips from the hub view to the local one — drop the cached hub
    # metadata alongside the listing.
    invalidate_model_hub_info(repo_id)


def _cleanup_partial_model(repo_id: str) -> None:
    """Remove a failed download's partial/unusable dir so it isn't mistaken for
    a complete local checkpoint by is_model_available_locally."""
    target = _local_models_root() / repo_id
    if target.exists() and _resolve_pretrained_dir(target) is None:
        shutil.rmtree(target, ignore_errors=True)


# The models twin of datasets.download_manager — same generic state machine
# (one download at a time, start/poll, survives navigation), model-specific
# fetch/cleanup.
model_download_manager = DownloadManager(_fetch_model_snapshot, _cleanup_partial_model)


# ---------------------------------------------------------------------------
# Import a model checkpoint folder already on disk into the local models dir.
# ---------------------------------------------------------------------------


def import_local_model(source_path: str, name: str | None = None) -> dict[str, Any]:
    """Copy a policy checkpoint folder already on the server machine into the
    local models dir, so it appears under "Local" (the models mirror of
    datasets.import_local_dataset — a COPY, never a move or a pointer-register).

    `source_path` points at a checkpoint dir in either shape
    _resolve_pretrained_dir recognizes (root config.json, or a
    checkpoints/<step>/pretrained_model tree). `name` is the target id — a bare
    name or ``namespace/name``, validated with the same segment rules dataset
    names use; defaults to the source folder's basename.

    Raises ModelError on: a missing source (404), a source that isn't a
    checkpoint (400), a bad target name (400), a target escaping the models
    root or overlapping the source (400), or a target that already exists
    (409). Invalidates the listing cache on success. Returns {"repo_id": ...}.

    NOTE: the copy runs SYNCHRONOUSLY (route blocks, frontend shows a spinner),
    same documented tradeoff as the dataset import — checkpoints are typically
    far smaller than datasets, so inline is even more comfortably acceptable.
    """
    try:
        src = Path(source_path).expanduser().resolve()
    except OSError:
        raise ModelError(400, "Invalid source path.") from None
    if not src.is_dir():
        raise ModelError(404, f"No folder found at '{source_path}'.")
    if _resolve_pretrained_dir(src) is None:
        raise ModelError(
            400,
            "That folder isn't a policy checkpoint (no config.json or "
            "checkpoints/<step>/pretrained_model tree inside it).",
        )

    raw = (name or "").strip() or src.name
    ok, reason = validate_dataset_repo_id(raw)
    if not ok:
        # The segment rules are shared with dataset names; reword for models.
        raise ModelError(400, reason.replace("Dataset name", "Model name"))

    root = _local_models_root().resolve()
    dst = (root / raw).resolve()
    # Reject a target that escapes the models root (traversal).
    if dst == root or root not in dst.parents:
        raise ModelError(400, "Invalid target model name.")
    # Refuse overlapping source/target (e.g. importing a dir into itself).
    if dst == src or src in dst.parents or dst in src.parents:
        raise ModelError(400, "The source folder and the import target overlap.")
    if dst.exists():
        raise ModelError(409, f"A model named '{raw}' already exists locally.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        logger.error("Failed to import model %s -> %s: %s", src, dst, exc)
        # Remove a partial copy so a failed import leaves no half-written dir.
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        raise ModelError(500, f"Failed to copy the model: {exc}") from exc

    invalidate_model_listing_cache()
    logger.info("Imported model %s -> %s", src, dst)
    return {"repo_id": raw}
