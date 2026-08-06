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

import concurrent.futures
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from huggingface_hub import (
    hf_hub_download,
    metadata_update,
    snapshot_download,
    try_to_load_from_cache,
)
from huggingface_hub.errors import HfHubHTTPError
from lerobot.datasets.dataset_tools import delete_episodes

from .utils.config import (
    get_hidden_datasets,
    get_saved_custom_datasets,
    validate_dataset_name,
    validate_dataset_repo_id,
    with_makermodslab_tag,
)
from .utils.hf_auth import cached_whoami, hf_hub_offline, shared_hf_api

logger = logging.getLogger(__name__)

CAMERA_FEATURE_PREFIX = "observation.images."


def _safe_int(value: Any) -> int | None:
    """`int(value)`, or None on a value that can't convert (e.g. a malformed
    episode_index/chunk_index/file_index in a Hub dataset's own parquet
    metadata — content this app doesn't control once the dataset isn't ours).
    Callers treat None as "this row/episode isn't usable," matching the
    graceful-404 pattern used everywhere else in the episode viewer, instead
    of letting a bad third-party dataset raise an unhandled ValueError."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _video_camera_names(features: dict[str, Any]) -> list[str]:
    """Camera feature names actually backed by an mp4 (dtype == "video"), not
    just any observation.images.* key. A raw-image (dtype == "image") camera
    feature has no video chunk file for this app's video-serving pipeline to
    point a <video> tag at, so it doesn't count as "viewable"."""
    return [
        key[len(CAMERA_FEATURE_PREFIX) :]
        for key, spec in features.items()
        if key.startswith(CAMERA_FEATURE_PREFIX) and isinstance(spec, dict) and spec.get("dtype") == "video"
    ]


# Errors a per-author / per-listing Hub call may raise that must NOT bubble up
# and 500 the endpoint. HfHubHTTPError covers HTTP-status failures; httpx.HTTPError
# is the base of ConnectError / TimeoutException / TransportError, which is what a
# GFW-killed TLS connection raises ([SSL: UNEXPECTED_EOF_WHILE_READING]); OSError
# covers lower-level socket failures. Any of these degrades a listing to
# "whatever other authors returned" rather than crashing.
_HUB_LISTING_ERRORS = (HfHubHTTPError, httpx.HTTPError, OSError)

# Cap on the concurrent per-author Hub fan-out. Small: a handful of authors
# (user + their orgs), and we don't want to hammer the Hub / open dozens of TLS
# handshakes behind a flaky link.
_HUB_FANOUT_MAX_WORKERS = 8

# OVERALL fan-out budget: the single deadline the whole per-author batch must
# finish within (authors run concurrently, so overall ≈ per-author). This is
# the ONLY timeout in the stack — the shared HfApi httpx client is built with
# timeout=None, so a blackholed connection would otherwise stall the listing
# until the OS TCP layer gives up. 5s is generous enough that a merely-slow-
# but-working Hub still succeeds, short enough that a hung author is abandoned
# fast and degrades to "whatever the finished authors returned".
_HUB_FANOUT_TIMEOUT_S = 5.0

# In-process cache of Hub existence checks, keyed by repo_id. /whoami-v2 and
# repo-existence lookups hit the network, so the info card fetches this lazily
# and we memoize the "on Hub" answer for the process lifetime. A successful
# upload invalidates the entry (see invalidate_hub_status), so the card can
# flip Local only -> On Hub without waiting for a cache expiry. "unknown" (the
# offline/unauthenticated/error degrade) is never cached, so connectivity
# returning is picked up on the next check.
_HUB_STATUS_CACHE: dict[str, str] = {}
_HUB_STATUS_LOCK = threading.Lock()


def invalidate_hub_status(repo_id: str) -> None:
    """Drop the cached Hub-existence answer for `repo_id`. Called after a
    successful upload so the next /datasets/hub-status re-checks (and sees
    the freshly pushed repo)."""
    with _HUB_STATUS_LOCK:
        _HUB_STATUS_CACHE.pop(repo_id, None)


# Short-TTL cache of the merged /datasets listing. Startup + navigation re-hit
# this endpoint in quick succession; without a cache each load re-fans-out to the
# Hub (slow/flaky behind the GFW). A <=TTL-stale listing is fine; a mutation the
# user just performed invalidates the cache (see invalidate_dataset_listing_cache)
# so it reflects immediately. TTL is measured with time.monotonic() — this is
# app runtime, so a monotonic clock (immune to wall-clock jumps) is the right tool.
_LISTING_CACHE_TTL_S = 45.0
_listing_cache_lock = threading.Lock()
_listing_cache: dict[str, Any] | None = None  # {"at": monotonic, "value": [...]}


def invalidate_dataset_listing_cache() -> None:
    """Drop the cached /datasets listing so the next call re-fetches from the
    Hub. Called after any mutation that changes the listing — dataset upload,
    delete, rename, visibility flip, or tag edit — so a change the user just made
    shows up immediately instead of after the TTL. Mirrors invalidate_hub_status."""
    global _listing_cache
    with _listing_cache_lock:
        _listing_cache = None


def _fan_out_hub_authors(authors: list[str], call: Callable[[str], Any]) -> list[Any]:
    """Run `call(author)` for each author concurrently, gathering the results.

    Each author's call runs in a bounded ThreadPoolExecutor and is guarded so a
    Hub failure for one author (a GFW-killed TLS connection, any transport
    error) is logged and swallowed — it contributes nothing rather than sinking
    the whole batch. The whole batch runs under ONE overall deadline
    (_HUB_FANOUT_TIMEOUT_S, via `as_completed(timeout=...)`): authors that
    haven't finished by then are abandoned and logged by name, and the finished
    authors' results are returned. That deadline is load-bearing — the shared
    HfApi httpx client has timeout=None, so a hung socket would otherwise stall
    the caller until the OS TCP timeout. `call` must return the
    (already-materialized) result for one author; returns the list of
    successful results in author order.
    """
    if not authors:
        return []

    results: list[Any] = [None] * len(authors)
    max_workers = min(_HUB_FANOUT_MAX_WORKERS, len(authors))
    # Deliberately NOT `with ThreadPoolExecutor(...)`: the context-manager exit
    # JOINS the worker threads, so a hung author would stall us at the `with`
    # exit even after the as_completed deadline fired. Instead shut down with
    # wait=False + cancel_futures=True in the finally: queued-not-started work
    # is cancelled, while an already-running hung thread is left to die with
    # its socket — a bounded leak (the OS TCP timeout eventually reaps it),
    # which beats blocking the endpoint on it.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_to_idx = {pool.submit(call, author): i for i, author in enumerate(authors)}
    try:
        for future in concurrent.futures.as_completed(future_to_idx, timeout=_HUB_FANOUT_TIMEOUT_S):
            idx = future_to_idx[future]
            author = authors[idx]
            try:
                results[idx] = future.result()
            except _HUB_LISTING_ERRORS as exc:
                logger.warning("Hub listing for author %s failed: %s", author, exc)
            except Exception as exc:  # noqa: BLE001 - listings are best-effort; never 500
                logger.warning("Hub listing for author %s failed unexpectedly: %s", author, exc)
    except concurrent.futures.TimeoutError:
        unfinished = [authors[i] for f, i in future_to_idx.items() if not f.done()]
        logger.warning(
            "Hub listing fan-out exceeded %ss; giving up on authors: %s",
            _HUB_FANOUT_TIMEOUT_S,
            ", ".join(unfinished) or "(none)",
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return [r for r in results if r is not None]


def get_hub_status(repo_id: str) -> dict[str, Any]:
    """Where a dataset repo with this id lives.

    Returns ``{"repo_id": ..., "status": "on_hub" | "local_only" | "absent" |
    "unknown", "url": <hub url> | None}``:

    * ``on_hub``     — the repo exists on the Hub.
    * ``local_only`` — NOT on the Hub, but a usable local copy exists (a
      recorded/merged dataset the user hasn't uploaded yet) — the card offers
      "Upload to Hub".
    * ``absent``     — neither on the Hub NOR in the local cache: a stale
      selection (a pin to a dataset that was deleted/renamed, or a merge output
      that was never materialized). The old code returned ``local_only`` here,
      which the info card read as "you have it locally" and rendered the
      contradictory "not downloaded locally" + "Local only / Upload" pair; the
      distinct status lets the card say "not found" instead.
    * ``unknown``    — offline / unauthenticated / any transport error.

    Never raises. Definitive Hub answers (``on_hub`` / ``local_only``) are
    memoized per repo_id for the process lifetime; ``"unknown"`` and ``"absent"``
    are NOT cached (a later record/merge/download can make an ``absent`` dataset
    appear locally without a hub-status invalidation, and a transient failure
    should self-heal) so both re-check on the next call.
    """
    url = f"https://huggingface.co/datasets/{repo_id}"

    with _HUB_STATUS_LOCK:
        cached = _HUB_STATUS_CACHE.get(repo_id)
    if cached is not None:
        return {"repo_id": repo_id, "status": cached, "url": url if cached == "on_hub" else None}

    api = shared_hf_api()
    try:
        exists = api.repo_exists(repo_id, repo_type="dataset")
    except Exception as exc:
        # Offline / rate-limited / any other transport error: degrade to
        # "unknown" without caching so it re-checks once connectivity returns.
        logger.info("hub-status repo_exists(%s) failed: %s", repo_id, exc)
        return {"repo_id": repo_id, "status": "unknown", "url": None}

    if exists:
        status = "on_hub"
    elif is_dataset_available_locally(repo_id):
        # Not on the Hub, but a usable local copy exists — genuinely local-only.
        status = "local_only"
    else:
        # Neither on the Hub nor local: don't mislabel this "local_only".
        status = "absent"

    with _HUB_STATUS_LOCK:
        # Cache only the definitive, stable answers; "absent" can flip to local
        # without a hub-status invalidation, so leave it uncached (like "unknown").
        if status in ("on_hub", "local_only"):
            _HUB_STATUS_CACHE[repo_id] = status
    return {"repo_id": repo_id, "status": status, "url": url if exists else None}


class DatasetHubEditError(Exception):
    """Raised when a Hub visibility/tags edit can't proceed. `status` is the
    HTTP status the route should return (400 offline/invalid, 403 no write
    permission, 502 other Hub failure); `message` is the user-facing reason;
    `docs_url` (optional) links auth docs for a login failure."""

    def __init__(self, status: int, message: str, docs_url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.docs_url = docs_url


def _hub_edit_error(exc: Exception) -> DatasetHubEditError:
    """Map a huggingface_hub exception raised by a visibility/tags mutation to a
    DatasetHubEditError with a legible message. A 401/auth failure or a 403
    permission failure becomes a clear "you can't edit this" message; anything
    else degrades to a generic Hub-failure 502."""
    from .record import _upload_auth_error

    auth = _upload_auth_error(exc)
    if auth is not None:
        return DatasetHubEditError(403, auth["message"], docs_url=auth.get("docs_url"))

    err_text = str(exc).lower()
    if "403" in err_text or "forbidden" in err_text or "permission" in err_text:
        return DatasetHubEditError(
            403,
            "You don't have permission to change this dataset on the Hub. "
            "You can only edit datasets in a namespace you can write to.",
        )
    return DatasetHubEditError(502, f"The Hub rejected the change: {exc}")


def get_hub_settings(repo_id: str) -> dict[str, Any]:
    """Current Hub-side visibility + tags for a dataset, for pre-filling the
    editor. Returns ``{"repo_id": ..., "private": bool, "tags": [str, ...]}``.

    Reads ``HfApi().dataset_info(repo_id)`` — the network call is the caller's
    (the info card fetches it lazily). Raises DatasetHubEditError offline (can't
    read reliably) or on a Hub failure so the route can surface a clear error.
    Tags come from the dataset card metadata (``dataset_info(...).tags``); the
    REQUIRED_HUB_TAGS are not stripped here — the card shows exactly what's live.
    """
    if hf_hub_offline():
        raise DatasetHubEditError(400, "The Hub is offline — dataset settings can't be read right now.")
    api = shared_hf_api()
    try:
        info = api.dataset_info(repo_id)
    except Exception as exc:
        logger.info("dataset_info(%s) failed: %s", repo_id, exc)
        raise _hub_edit_error(exc) from exc
    return {
        "repo_id": repo_id,
        "private": bool(getattr(info, "private", False)),
        "tags": list(getattr(info, "tags", None) or []),
    }


def set_dataset_visibility(repo_id: str, private: bool) -> dict[str, Any]:
    """Flip a Hub dataset's visibility (public <-> private).

    Wraps ``HfApi().update_repo_settings(repo_id, private=..., repo_type="dataset")``
    (this huggingface_hub version has no ``update_repo_visibility``). Refuses
    offline (can't mutate). Maps auth/permission failures to a clear message.
    Invalidates the cached Hub-existence answer so the card re-reads settings.
    """
    if hf_hub_offline():
        raise DatasetHubEditError(
            400, "The Hub is offline — you can't change a dataset's visibility right now."
        )
    api = shared_hf_api()
    try:
        api.update_repo_settings(repo_id, private=private, repo_type="dataset")
    except Exception as exc:
        logger.info("update_repo_settings(%s, private=%s) failed: %s", repo_id, private, exc)
        raise _hub_edit_error(exc) from exc

    invalidate_hub_status(repo_id)
    invalidate_dataset_listing_cache()
    logger.info("Set dataset %s visibility private=%s", repo_id, private)
    return {"repo_id": repo_id, "private": private}


def set_dataset_tags(repo_id: str, tags: list[str]) -> dict[str, Any]:
    """Replace a Hub dataset card's ``tags:`` metadata.

    User-supplied `tags` are funnelled through ``with_makermodslab_tag`` FIRST, so the
    required org/product tags (makermods / openbooth / MakerModsLab) are never dropped
    by an edit, then written with ``metadata_update(..., overwrite=True)``.
    Refuses offline. Maps auth/permission failures. Invalidates the cached
    Hub-existence answer. Returns the final tag list actually written.
    """
    if hf_hub_offline():
        raise DatasetHubEditError(400, "The Hub is offline — you can't edit a dataset's tags right now.")
    final_tags = with_makermodslab_tag(tags)
    try:
        metadata_update(repo_id, {"tags": final_tags}, repo_type="dataset", overwrite=True)
    except Exception as exc:
        logger.info("metadata_update(%s, tags=%s) failed: %s", repo_id, final_tags, exc)
        raise _hub_edit_error(exc) from exc

    invalidate_hub_status(repo_id)
    invalidate_dataset_listing_cache()
    logger.info("Set dataset %s tags -> %s", repo_id, final_tags)
    return {"repo_id": repo_id, "tags": final_tags}


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _is_dataset_dir(path: Path) -> bool:
    """A directory is a LeRobot dataset iff <dir>/meta/info.json exists."""
    try:
        return (path / "meta" / "info.json").is_file()
    except OSError:
        return False


def is_dataset_available_locally(repo_id: str) -> bool:
    """True if lerobot could train on `repo_id` WITHOUT a Hub download.

    Filesystem-only (no network), so it's safe to call when the Hub is offline.
    We are deliberately conservative — only return False when the dataset is
    absent from BOTH cache layouts — because a false "not available" would
    wrongly block a run on a dataset that's actually present.

    Two layouts have to be covered, both confirmed live:

    * FLAT layout — locally recorded/materialized datasets live directly at
      ``<lerobot_home>/<repo_id>/`` (recognized by ``meta/info.json``). This is
      where recording, merging, and `list_local_datasets` put things.
    * HF HUB SNAPSHOT cache — a dataset downloaded from the Hub lands as
      ``datasets--<namespace>--<name>/snapshots/<rev>/`` under a ``hub/`` cache,
      NOT in the flat layout. lerobot's ``LeRobotDataset._download`` (no
      ``--dataset.root``) snapshots into ``$HF_LEROBOT_HOME/hub`` — i.e.
      ``<lerobot_home>/hub`` — while a plain ``huggingface_hub`` download lands
      in the default ``~/.cache/huggingface/hub``. Both have been observed in
      the wild, so we probe BOTH cache dirs. lerobot resolves this via
      huggingface_hub's own cache, so we ask huggingface_hub whether
      ``meta/info.json`` is already cached. ``try_to_load_from_cache`` returns a
      real path (str) when the file is cached, and ``None`` / the
      ``_CACHED_NO_EXIST`` sentinel (a non-str object) otherwise — both of the
      latter mean "not usable offline" here.
    """
    # FLAT layout: locally recorded / materialized dataset.
    if _is_dataset_dir(_lerobot_cache_root() / repo_id):
        return True

    # HF hub snapshot cache: a previously downloaded Hub dataset. Purely a
    # cache lookup — no network even when a token is present. Probe both the
    # lerobot hub cache (where MakerMods Lab's own local runs auto-download) and the
    # default hub cache (where a manual `huggingface-cli download` would land).
    # `None` (default) tells try_to_load_from_cache to use the default cache.
    lerobot_hub_cache = _lerobot_cache_root() / "hub"
    for cache_dir in (str(lerobot_hub_cache), None):
        try:
            cached = try_to_load_from_cache(
                repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            # A cache-probe failure is not evidence of absence; degrade to
            # "assume present" so we never wrongly block a run on an internal
            # error (conservative: a false "not available" is the bad outcome).
            logger.info("try_to_load_from_cache(%s) failed: %s", repo_id, exc)
            return True
        if isinstance(cached, str):
            return True

    return False


def _dataset_has_episodes(path: Path) -> bool:
    """True if the dataset recorded at least one episode. An empty dataset
    (0 episodes — e.g. a recording aborted before saving) has no task/data
    files and only breaks downstream steps like training and merging, so we
    hide it from the listing rather than let it be selected."""
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(info.get("total_episodes"))


def _dir_mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Scan the LeRobot cache for local datasets (dirs containing meta/info.json).

    Walks one level deep: a top-level dataset dir is recorded as "<name>"; if a
    top-level dir is not itself a dataset, each subdir that is a dataset is
    recorded as "<top>/<sub>". Does not descend further.
    """
    root = _lerobot_cache_root()
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    try:
        top_entries = list(root.iterdir())
    except OSError as e:
        logger.warning(f"Could not read LeRobot cache root {root}: {e}")
        return []

    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if top.name.startswith("."):
            continue

        if _is_dataset_dir(top):
            # It IS a dataset (empty or not) — record it only if non-empty, but
            # don't descend into its subdirs either way.
            if _dataset_has_episodes(top):
                out.append(
                    {
                        "repo_id": top.name,
                        "last_modified": _dir_mtime_iso(top),
                        "private": False,
                    }
                )
            continue

        # Not a dataset itself — descend one level.
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

            if sub.name.startswith("."):
                continue

            if _is_dataset_dir(sub) and _dataset_has_episodes(sub):
                out.append(
                    {
                        "repo_id": f"{top.name}/{sub.name}",
                        "last_modified": _dir_mtime_iso(sub),
                        "private": False,
                    }
                )

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def _read_task_strings(meta_dir: Path) -> list[str]:
    """Task strings for a dataset, ordered by task_index.

    v3.0 datasets keep them in ``meta/tasks.parquet`` (columns ``task_index``
    and ``task``; pandas stores ``task`` as the frame index, but pyarrow
    surfaces both as plain columns). Older v2.x datasets use
    ``meta/tasks.jsonl`` with one ``{"task_index": …, "task": …}`` object per
    line. Unreadable/absent task metadata degrades to an empty list — the info
    card can render without it.
    """
    parquet_path = meta_dir / "tasks.parquet"
    if parquet_path.is_file():
        try:
            table = pq.read_table(parquet_path).to_pydict()
            rows = sorted(zip(table.get("task_index", []), table.get("task", []), strict=True))
            return [str(task) for _, task in rows]
        except Exception as e:
            logger.warning(f"Could not read {parquet_path}: {e}")
            return []

    jsonl_path = meta_dir / "tasks.jsonl"
    if jsonl_path.is_file():
        try:
            rows = []
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((obj.get("task_index", 0), str(obj.get("task", ""))))
            rows.sort()
            return [task for _, task in rows if task]
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {jsonl_path}: {e}")

    return []


def _count_task_episodes(meta_dir: Path) -> dict[str, int]:
    """Episodes per task string, from the per-episode ``tasks`` column.

    Each episode lists the task strings it uses, and an episode counts once
    per distinct task. Read directly from the metadata files —
    v3.0 keeps episode rows in ``meta/episodes/chunk-*/file-*.parquet`` (only
    the ``tasks`` column is loaded, not the wide per-episode stats), v2.x in
    ``meta/episodes.jsonl`` — so the endpoint stays a cheap file read instead
    of a full ``LeRobotDataset`` load. Unreadable/absent episode metadata
    degrades to an empty dict (counts render as 0).
    """
    counts: dict[str, int] = {}

    episodes_dir = meta_dir / "episodes"
    if episodes_dir.is_dir():
        for parquet_path in sorted(episodes_dir.glob("**/*.parquet")):
            try:
                table = pq.read_table(parquet_path, columns=["tasks"])
            except Exception as e:
                logger.warning(f"Could not read {parquet_path}: {e}")
                continue
            for episode_tasks in table.column("tasks").to_pylist():
                for task in set(episode_tasks or []):
                    counts[str(task)] = counts.get(str(task), 0) + 1
        return counts

    jsonl_path = meta_dir / "episodes.jsonl"
    if jsonl_path.is_file():
        try:
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for task in set(obj.get("tasks") or []):
                    counts[str(task)] = counts.get(str(task), 0) + 1
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {jsonl_path}: {e}")

    return counts


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under `path`. Unreadable files are skipped."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _resolve_local_dataset_path(repo_id: str) -> Path | None:
    """Resolve `repo_id` to its local cache directory. None if it escapes the
    cache root (path traversal) or isn't a local dataset dir (e.g. it only
    exists on the Hub). Shared by every local-only dataset reader below."""
    root = _lerobot_cache_root().resolve()
    try:
        path = (root / repo_id).resolve()
    except OSError:
        return None
    # Reject path traversal: the dataset dir must stay strictly inside the cache.
    if path == root or root not in path.parents:
        return None
    if not _is_dataset_dir(path):
        return None
    return path


def get_local_dataset_info(repo_id: str) -> dict[str, Any] | None:
    """Detail view of one locally-cached dataset, for the selection info card.

    Reads ``meta/info.json`` + task metadata and walks the directory for its
    size on disk — per-dataset on demand, so the cheap ``/datasets`` listing
    stays cheap. Returns None if `repo_id` escapes the cache root or isn't a
    local dataset (e.g. it only exists on the Hub).
    """
    path = _resolve_local_dataset_path(repo_id)
    if path is None:
        return None

    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None

    features = info.get("features") or {}
    cameras = [key[len(CAMERA_FEATURE_PREFIX) :] for key in features if key.startswith(CAMERA_FEATURE_PREFIX)]

    task_counts = _count_task_episodes(path / "meta")
    tasks = [
        {"task": task, "num_episodes": task_counts.get(task, 0)} for task in _read_task_strings(path / "meta")
    ]

    return {
        "repo_id": repo_id,
        "total_episodes": int(info.get("total_episodes") or 0),
        "total_frames": int(info.get("total_frames") or 0),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "cameras": cameras,
        "tasks": tasks,
        "size_bytes": _dir_size_bytes(path),
        # ADDITIVE /datasets/info contract: "local" (full detail from the local
        # cache) vs "hub" (the meta/info.json summary of a not-yet-downloaded
        # Hub dataset — see get_hub_dataset_info). The card gates its local-only
        # affordances (rename, size, task counts) on this.
        "source": "local",
    }


def _read_episode_rows(meta_dir: Path, columns: list[str] | None = None) -> list[dict[str, Any]] | None:
    """Every row of ``meta/episodes/chunk-*/file-*.parquet``, column-pruned.

    v3.0-only: older v2.x datasets keep episodes in ``meta/episodes.jsonl``,
    which has no per-camera video chunk/file-index columns, so the dataset
    viewer (episode list, video, joint chart) isn't offered for them — callers
    treat a None return as "not viewable", not an error. Returns None if the
    directory is absent or nothing could be read.
    """
    episodes_dir = meta_dir / "episodes"
    if not episodes_dir.is_dir():
        return None
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(episodes_dir.glob("**/*.parquet")):
        try:
            table = pq.read_table(parquet_path, columns=columns)
        except Exception as e:
            logger.warning(f"Could not read {parquet_path}: {e}")
            continue
        rows.extend(table.to_pylist())
    return rows or None


def _hub_dataset_has_video(repo_id: str) -> bool:
    """Whether `repo_id` (assumed to exist on the Hub) has at least one
    dtype == "video" camera feature, via the same cached summary /datasets/info
    already uses — no extra network call beyond what get_hub_dataset_info
    itself needs."""
    info = get_hub_dataset_info(repo_id)
    return bool(info and info["cameras"])


def _ensure_hub_episodes_root(repo_id: str) -> Path | None:
    """Download meta/info.json + every meta/episodes/**/*.parquet chunk for a
    Hub dataset confirmed to have video, into huggingface_hub's own on-disk
    cache (~/.cache/huggingface/hub by default) — NOT MakerMods Lab's
    ~/.cache/huggingface/lerobot dataset cache, and NOT a full dataset
    snapshot. Returns the snapshot root directory so the existing local-path
    reading code (_read_episode_rows, etc.) can run against it exactly like a
    local dataset dir; None if the dataset isn't viewable this way (offline,
    no video, or the fetch failed).

    The episode-metadata parquet files are small regardless of how large the
    dataset's actual video is — this never pulls video/data chunks themselves;
    those are fetched one at a time by get_episode_video_path /
    get_episode_joint_series only when a specific episode is actually opened.
    No caching layer beyond hf_hub_download's own: a repeat call re-lists repo
    files and re-touches already-cached files, which is fast (etag-checked
    cache hits), not a re-download.
    """
    if hf_hub_offline():
        return None
    if not _hub_dataset_has_video(repo_id):
        return None
    try:
        info_path = hf_hub_download(repo_id, filename="meta/info.json", repo_type="dataset")
        root = Path(info_path).parents[1]  # strip "meta/info.json"'s 2 path parts
        files = shared_hf_api().list_repo_files(repo_id, repo_type="dataset")
        for f in files:
            if f.startswith("meta/episodes/") and f.endswith(".parquet"):
                hf_hub_download(repo_id, filename=f, repo_type="dataset")
    except Exception as exc:
        logger.info("hub episode metadata fetch for %s failed: %s", repo_id, exc)
        return None
    return root


def list_episode_summaries(repo_id: str) -> list[dict[str, Any]] | None:
    """Per-episode index/length/duration/tasks/video_offsets for the dataset
    viewer's episode list. None if `repo_id` isn't a local dataset in the
    v3.0 parquet layout.

    ``video_offsets`` matters because v3.0 packs MULTIPLE consecutive
    episodes into the same physical mp4 per camera (confirmed on real data:
    episode 0 and 1 of a 2-episode recording share ``chunk-000/file-000.mp4``,
    distinguished only by ``from_timestamp``/``to_timestamp``) — a naive
    "serve the episode's video file" playing it start-to-finish runs straight
    into whatever episode comes next in that file. The viewer needs each
    camera's slice boundaries to seek to the right start and stop at the
    right end within the shared file.

    A dataset with no local copy falls back to fetching just its episode metadata from the Hub (see _ensure_hub_episodes_root) — None only when neither resolves, or the dataset has no video.
    """
    path = _resolve_local_dataset_path(repo_id)
    if path is None:
        path = _ensure_hub_episodes_root(repo_id)
    if path is None:
        return None
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None
    fps = info.get("fps") or 1
    features = info.get("features") or {}
    cameras = _video_camera_names(features)

    video_cols: list[str] = []
    col_to_camera: dict[str, tuple[str, str]] = {}
    for camera in cameras:
        video_key = f"{CAMERA_FEATURE_PREFIX}{camera}"
        from_col, to_col = f"videos/{video_key}/from_timestamp", f"videos/{video_key}/to_timestamp"
        video_cols += [from_col, to_col]
        col_to_camera[from_col] = (camera, "from")
        col_to_camera[to_col] = (camera, "to")

    rows = _read_episode_rows(path / "meta", columns=["episode_index", "tasks", "length", *video_cols])
    if rows is None:
        return None
    out = []
    for row in rows:
        episode_index = _safe_int(row.get("episode_index"))
        length = _safe_int(row.get("length"))
        if episode_index is None or length is None:
            logger.warning("Skipping episode row with malformed episode_index/length for %s", repo_id)
            continue
        video_offsets: dict[str, dict[str, float]] = {}
        for col, (camera, which) in col_to_camera.items():
            value = row.get(col)
            if value is None:
                continue
            try:
                video_offsets.setdefault(camera, {})[which] = float(value)
            except (TypeError, ValueError):
                continue
        out.append(
            {
                "episode_index": episode_index,
                "length": length,
                "duration": round(length / fps, 3),
                "tasks": [str(t) for t in (row.get("tasks") or [])],
                "video_offsets": video_offsets,
            }
        )
    out.sort(key=lambda e: e["episode_index"])
    return out


def get_episode_video_path(repo_id: str, episode_index: int, camera: str) -> Path | None:
    """The mp4 file backing one camera's footage for one episode.

    None if `repo_id`/`episode_index`/`camera` doesn't resolve, the dataset
    isn't the v3.0 parquet layout, or the file is missing/unfetchable.
    `camera` is checked against the dataset's own camera list (from
    meta/info.json) before it's used to build a path, so it can never point
    outside the videos dir. A repo with no local copy falls back to
    downloading just this one video chunk from the Hub (see
    _ensure_hub_episodes_root) when the dataset is confirmed to have video.
    """
    path = _resolve_local_dataset_path(repo_id)
    is_hub = path is None
    if is_hub:
        path = _ensure_hub_episodes_root(repo_id)
    if path is None:
        return None
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None
    features = info.get("features") or {}
    cameras = _video_camera_names(features)
    if camera not in cameras:
        return None

    video_key = f"{CAMERA_FEATURE_PREFIX}{camera}"
    chunk_col, file_col = f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"
    rows = _read_episode_rows(path / "meta", columns=["episode_index", chunk_col, file_col])
    if rows is None:
        return None
    row = next((r for r in rows if _safe_int(r.get("episode_index")) == episode_index), None)
    if row is None or row.get(chunk_col) is None or row.get(file_col) is None:
        return None
    chunk_index, file_index = _safe_int(row[chunk_col]), _safe_int(row[file_col])
    if chunk_index is None or file_index is None:
        logger.warning(
            "Malformed chunk/file index for %s episode %d camera %s", repo_id, episode_index, camera
        )
        return None

    rel_video_path = Path("videos") / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if is_hub:
        try:
            return Path(hf_hub_download(repo_id, filename=str(rel_video_path), repo_type="dataset"))
        except Exception as exc:
            logger.info("hub video chunk fetch for %s failed: %s", repo_id, exc)
            return None

    video_path = path / rel_video_path
    return video_path if video_path.is_file() else None


def get_episode_joint_series(repo_id: str, episode_index: int) -> dict[str, Any] | None:
    """Per-frame timestamp + ``observation.state`` for one episode, for the
    dataset viewer's joint-position chart. None if it can't be resolved/read
    (not local, not the v3.0 parquet layout, or the episode doesn't exist). A
    repo with no local copy falls back to downloading just this one data chunk
    from the Hub (see _ensure_hub_episodes_root) when the dataset is confirmed
    to have video.
    """
    path = _resolve_local_dataset_path(repo_id)
    is_hub = path is None
    if is_hub:
        path = _ensure_hub_episodes_root(repo_id)
    if path is None:
        return None
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None
    joint_names = ((info.get("features") or {}).get("observation.state") or {}).get("names") or []

    episode_rows = _read_episode_rows(
        path / "meta", columns=["episode_index", "data/chunk_index", "data/file_index"]
    )
    if episode_rows is None:
        return None
    row = next((r for r in episode_rows if _safe_int(r.get("episode_index")) == episode_index), None)
    if row is None or row.get("data/chunk_index") is None or row.get("data/file_index") is None:
        return None
    chunk_index, file_index = _safe_int(row["data/chunk_index"]), _safe_int(row["data/file_index"])
    if chunk_index is None or file_index is None:
        logger.warning("Malformed data chunk/file index for %s episode %d", repo_id, episode_index)
        return None

    rel_data_path = Path("data") / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    if is_hub:
        try:
            data_path = Path(hf_hub_download(repo_id, filename=str(rel_data_path), repo_type="dataset"))
        except Exception as exc:
            logger.info("hub data chunk fetch for %s failed: %s", repo_id, exc)
            return None
    else:
        data_path = path / rel_data_path
        if not data_path.is_file():
            return None
    try:
        table = pq.read_table(data_path, columns=["episode_index", "timestamp", "observation.state"])
    except Exception as e:
        logger.warning(f"Could not read {data_path}: {e}")
        return None

    frames = sorted(
        (
            (float(ts), [float(v) for v in state])
            for ep, ts, state in zip(
                table.column("episode_index").to_pylist(),
                table.column("timestamp").to_pylist(),
                table.column("observation.state").to_pylist(),
                strict=True,
            )
            if _safe_int(ep) == episode_index
        ),
        key=lambda pair: pair[0],
    )
    if not frames:
        return None
    return {
        "joint_names": [str(n) for n in joint_names],
        "timestamps": [t for t, _ in frames],
        "values": [v for _, v in frames],
    }


# In-process cache of per-repo Hub dataset summaries (the /datasets/info hub
# fallback), mirroring _HUB_STATUS_CACHE conventions: successful answers are
# memoized for the process lifetime; the offline/error degrade is NEVER cached,
# so connectivity returning is picked up on the next check. Invalidated when
# the repo's content changes (upload / download-complete) or the row is hidden.
_HUB_DATASET_INFO_CACHE: dict[str, dict[str, Any]] = {}
_HUB_DATASET_INFO_LOCK = threading.Lock()


def invalidate_hub_dataset_info(repo_id: str) -> None:
    """Drop the cached Hub summary for `repo_id`, so the next /datasets/info
    re-fetches its meta/info.json (e.g. after an upload changed it)."""
    with _HUB_DATASET_INFO_LOCK:
        _HUB_DATASET_INFO_CACHE.pop(repo_id, None)


def get_hub_dataset_info(repo_id: str) -> dict[str, Any] | None:
    """Summary of a Hub dataset that has NO local copy, for the info card's
    hub fallback (the /datasets/info route tries get_local_dataset_info first).

    Fetches just ``meta/info.json`` via hf_hub_download — a tiny file — for the
    episode/frame counts, fps, robot type, and camera keys (from ``features``).
    Task strings and size-on-disk need the full dataset, so they degrade to
    empty/None; ``source: "hub"`` tells the card which contract it got. This is
    a LAZY per-card fetch, deliberately not part of the /datasets listing.

    Degrade-not-crash: returns None offline or on any fetch/parse failure (the
    card then falls back to the sparse "not downloaded" view); only successful
    answers are cached (see _HUB_DATASET_INFO_CACHE).
    """
    if hf_hub_offline():
        return None

    with _HUB_DATASET_INFO_LOCK:
        cached = _HUB_DATASET_INFO_CACHE.get(repo_id)
    if cached is not None:
        return dict(cached)

    try:
        path = hf_hub_download(repo_id, filename="meta/info.json", repo_type="dataset")
        info = json.loads(Path(path).read_text())
    except Exception as exc:
        logger.info("hub dataset info fetch for %s failed: %s", repo_id, exc)
        return None

    features = info.get("features") or {}
    cameras = _video_camera_names(features)

    row: dict[str, Any] = {
        "repo_id": repo_id,
        "total_episodes": int(info.get("total_episodes") or 0),
        "total_frames": int(info.get("total_frames") or 0),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "cameras": cameras,
        "tasks": [],
        "size_bytes": None,
        "source": "hub",
    }

    with _HUB_DATASET_INFO_LOCK:
        _HUB_DATASET_INFO_CACHE[repo_id] = dict(row)
    return row


def read_dataset_features(repo_id: str) -> dict[str, Any] | None:
    """The RAW ``features`` map from a dataset's ``meta/info.json``.

    The other readers above summarise info.json for a UI card (camera names,
    episode counts); this one hands back the feature specs untouched —
    ``dtype``/``shape``/``names`` per key — because the fine-tune preflight in
    jobs.py compares them dimension-for-dimension against a checkpoint's own
    ``input_features``/``output_features``. Local first (a plain file read, no
    network), falling back to fetching just ``meta/info.json`` from the Hub for
    a dataset with no local copy — the same tiny file get_hub_dataset_info
    uses.

    Returns None when it can't be read — not local, offline, absent/private
    repo, malformed JSON, or no ``features`` map. None means "not established",
    never "fine": a caller must treat it as a reason to stay silent rather than
    as a clean bill of health.
    """
    path = _resolve_local_dataset_path(repo_id)
    if path is not None:
        try:
            info = json.loads((path / "meta" / "info.json").read_text())
        except (OSError, ValueError):
            return None
    elif hf_hub_offline():
        return None
    else:
        try:
            local = hf_hub_download(repo_id, filename="meta/info.json", repo_type="dataset")
            info = json.loads(Path(local).read_text())
        except Exception as exc:
            logger.info("dataset features fetch for %s failed: %s", repo_id, exc)
            return None

    features = info.get("features") if isinstance(info, dict) else None
    return features if isinstance(features, dict) else None


class DatasetRenameError(Exception):
    """Raised by rename_local_dataset when the rename can't proceed. `status`
    is the HTTP status the route should return (400 invalid, 404 not found,
    409 conflict/busy); `message` is the user-facing reason."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _dataset_in_use(repo_id: str) -> str | None:
    """If `repo_id`'s directory is in use by a running operation, return a
    legible reason to refuse a rename; else None.

    Checks the four ways a dataset dir can be actively read/written:
      * recording — the active session's (timestamp-stamped) repo id, OR the
        base name the user typed (recording stamps ``name`` → ``name_<ts>``,
        so a rename of the base while a session writes ``name_<ts>`` would
        pull the directory out from under it);
      * merge — the output dataset currently being aggregated;
      * upload — the dataset currently being pushed to the Hub (renaming or
        deleting the directory mid-push would corrupt the upload);
      * local training — any running local job whose config trains on it.

    Read-only imports; each module owns its own state. Kept deliberately
    simple: a false "in use" is safer than yanking a directory mid-write.
    """
    # Recording: record.py owns recording_active + recording_config.
    from . import record as _record

    if _record.recording_active and _record.recording_config is not None:
        active_id = getattr(_record.recording_config, "dataset_repo_id", None)
        # The session stamps a timestamp onto the base name (name -> name_<ts>),
        # so match either the stamped id or a rename of the still-writing base.
        if active_id and (active_id == repo_id or active_id.startswith(f"{repo_id}_")):
            return "A recording session is writing to this dataset. Stop it first."

    # Upload: record.py owns an UploadManager singleton (state + repo_id). Same
    # lazy import (datasets<->record cycle) as recording above.
    if _record.upload_manager.state == "running" and _record.upload_manager.repo_id == repo_id:
        return "This dataset is being uploaded to the Hub right now. Wait for it to finish."

    # Merge: merge.py exposes a MergeManager singleton with state + output id.
    from . import merge as _merge

    mgr = _merge.merge_manager
    if mgr.state == "running" and mgr.output_repo_id == repo_id:
        return "A merge is producing this dataset right now. Wait for it to finish first."

    # Local training: a running local job whose config trains on this dataset.
    from .jobs import job_registry

    for record in job_registry.list(limit=200):
        if (
            record.state == "running"
            and record.runner == "local"
            and record.config.dataset_repo_id == repo_id
        ):
            return "A local training run is using this dataset. Stop it first."

    return None


def rename_local_dataset(repo_id: str, new_name: str) -> str:
    """Rename a locally-cached dataset by moving its directory.

    A dataset's repo id *is* its path under the cache root, so a rename is a
    directory move. `new_name` is the NAME PART ONLY — the namespace prefix is
    fixed, so ``ns/old`` renamed to ``new`` becomes ``ns/new`` and a bare
    ``old`` becomes ``new``. Returns the new repo id.

    Raises DatasetRenameError (with an HTTP status + message) on: a bad
    new_name, a source that isn't a local dataset, a target that already
    exists, or the dataset being actively used (recording / merge / local
    training). Invalidates the cached Hub-existence answer for BOTH ids so the
    info card re-checks after the move.
    """
    ok, reason = validate_dataset_name(new_name)
    if not ok:
        raise DatasetRenameError(400, reason)

    root = _lerobot_cache_root().resolve()
    try:
        src = (root / repo_id).resolve()
    except OSError:
        raise DatasetRenameError(400, "Invalid dataset path") from None
    # Reject path traversal: the source must stay strictly inside the cache.
    if src == root or root not in src.parents:
        raise DatasetRenameError(400, "Invalid dataset path")
    if not _is_dataset_dir(src):
        raise DatasetRenameError(404, f"Dataset '{repo_id}' not found in the local cache")

    # The namespace prefix is fixed — swap only the final path segment.
    namespace = repo_id.rsplit("/", 1)[0] if "/" in repo_id else None
    new_repo_id = f"{namespace}/{new_name}" if namespace else new_name
    if new_repo_id == repo_id:
        return repo_id  # no-op

    dst = src.parent / new_name
    if dst.exists():
        raise DatasetRenameError(409, f"A dataset named '{new_repo_id}' already exists.")

    in_use = _dataset_in_use(repo_id)
    if in_use is not None:
        raise DatasetRenameError(409, in_use)

    try:
        os.rename(src, dst)
    except OSError as exc:
        logger.error("Failed to rename dataset %s -> %s: %s", repo_id, new_repo_id, exc)
        raise DatasetRenameError(500, f"Failed to rename dataset: {exc}") from exc

    # The old id no longer exists and the new id now does — drop both cached
    # Hub-existence answers so the next hub-status check re-queries.
    invalidate_hub_status(repo_id)
    invalidate_hub_status(new_repo_id)
    invalidate_dataset_listing_cache()

    logger.info("Renamed dataset directory %s -> %s", src, dst)
    return new_repo_id


class DatasetEpisodeDeleteError(Exception):
    """Raised by delete_local_episode when the delete can't proceed. `status`
    is the HTTP status the route should return (400 invalid/last-episode,
    404 not found, 409 busy, 500 rewrite/swap failure); `message` is the
    user-facing reason."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _resync_episode_delete_to_hub(repo_id: str) -> tuple[str, str | None]:
    """After a local episode delete, push the rewritten dataset to the Hub if
    a Hub copy exists, so the two don't silently disagree on episode count.

    Reuses the existing UploadManager background job ("Upload to Hub") rather
    than a new push mechanism, preserving the repo's CURRENT Hub visibility
    and tags (never guesses defaults that could flip a private repo public or
    drop existing tags).

    Returns (hub_sync, hub_sync_message):
      * ("not_on_hub", None) — no Hub copy, nothing to sync.
      * ("started", None) — a background resync push was kicked off; poll
        the existing /upload-status for progress/errors, same as any other
        Hub upload.
      * ("skipped", <reason>) — a Hub copy exists but the resync couldn't be
        started (current Hub settings unreadable, or the single global
        upload slot is busy with another dataset).

    Never raises — a resync failure must not undo an already-committed local
    delete; the caller has already swapped the local directory in by the
    time this runs.
    """
    if get_hub_status(repo_id)["status"] != "on_hub":
        return "not_on_hub", None

    try:
        settings = get_hub_settings(repo_id)
    except Exception as exc:
        logger.info(
            "Could not read Hub settings for %s before episode-delete resync: %s", repo_id, exc
        )
        return "skipped", "Couldn't read current Hub settings — use Upload to Hub to sync manually."

    # Lazy import: record.py already imports from datasets.py at call time to
    # avoid a datasets<->record import cycle (see _dataset_in_use); this is
    # the same trick in the other direction.
    from . import record as _record

    result = _record.upload_manager.start(
        _record.UploadRequest(
            dataset_repo_id=repo_id,
            tags=settings["tags"],
            private=settings["private"],
        )
    )
    if not result.get("started"):
        return "skipped", result.get("message") or "Could not start the Hub resync."
    return "started", None


def delete_local_episode(repo_id: str, episode_index: int) -> dict[str, Any]:
    """Delete one episode from a locally-cached dataset.

    Episode deletion isn't a row removal: lerobot's v3.0 layout packs
    consecutive episodes into shared per-camera video files, so removing one
    episode can require re-encoding whatever physical file it shares with a
    kept episode. We reuse lerobot's own ``delete_episodes``, which builds an
    entirely new dataset directory rather than mutating in place — into a
    temp sibling directory here, then atomically swapped in for the original.

    If the dataset also exists on the Hub, this kicks off a background push
    (reusing the same UploadManager "Upload to Hub" already uses) to bring
    the Hub copy back in sync — see ``_resync_episode_delete_to_hub``. The
    local swap always happens first and is never rolled back if that resync
    fails to start or later fails; poll /upload-status the same way
    "Upload to Hub" is polled.

    Raises DatasetEpisodeDeleteError (status + message) when: the dataset
    isn't found locally (404), it's busy — recording / merge / upload /
    local training (409), the dataset can't be loaded, the index is out of
    range, or it's the dataset's only episode (400), or the rewrite/swap
    itself fails (500, rolling back to the original directory when
    possible).
    """
    target = _resolve_local_dataset_path(repo_id)
    if target is None:
        raise DatasetEpisodeDeleteError(404, f"Dataset '{repo_id}' not found in the local cache")

    in_use = _dataset_in_use(repo_id)
    if in_use is not None:
        raise DatasetEpisodeDeleteError(409, in_use)

    from lerobot.datasets import LeRobotDataset

    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=target)
    except Exception as exc:
        logger.error("Failed to load dataset %s for episode delete: %s", repo_id, exc)
        raise DatasetEpisodeDeleteError(400, f"Could not read dataset: {exc}") from exc

    total_episodes = dataset.meta.total_episodes
    if episode_index < 0 or episode_index >= total_episodes:
        raise DatasetEpisodeDeleteError(
            400,
            f"Episode {episode_index} does not exist in '{repo_id}' ({total_episodes} episode(s))",
        )
    if total_episodes <= 1:
        raise DatasetEpisodeDeleteError(
            400, "This is the dataset's only episode — delete the whole dataset instead."
        )

    tmp_dir = target.parent / f".{target.name}.delete-tmp-{uuid.uuid4().hex[:8]}"
    try:
        delete_episodes(dataset, [episode_index], output_dir=tmp_dir, repo_id=repo_id)
    except ValueError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise DatasetEpisodeDeleteError(400, str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Failed to delete episode %s from %s: %s", episode_index, repo_id, exc)
        raise DatasetEpisodeDeleteError(500, f"Failed to delete episode: {exc}") from exc

    backup_dir = target.parent / f".{target.name}.pre-delete-{uuid.uuid4().hex[:8]}"
    try:
        os.rename(target, backup_dir)
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Failed to stage %s aside for episode delete: %s", target, exc)
        raise DatasetEpisodeDeleteError(500, f"Failed to delete episode: {exc}") from exc

    try:
        os.rename(tmp_dir, target)
    except OSError as exc:
        try:
            os.rename(backup_dir, target)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.error(
                "Failed to roll back %s after a failed episode-delete swap — dataset directory "
                "may be missing; original data is at %s",
                target,
                backup_dir,
            )
            raise DatasetEpisodeDeleteError(
                500, f"Failed to delete episode and could not restore the original dataset: {exc}"
            ) from exc
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Failed to swap in the edited dataset for %s: %s", target, exc)
        raise DatasetEpisodeDeleteError(500, f"Failed to delete episode: {exc}") from exc

    shutil.rmtree(backup_dir, ignore_errors=True)
    invalidate_dataset_listing_cache()

    hub_sync, hub_sync_message = _resync_episode_delete_to_hub(repo_id)

    new_total = total_episodes - 1
    logger.info("Deleted episode %s from %s (%s episode(s) remain)", episode_index, repo_id, new_total)
    return {
        "success": True,
        "repo_id": repo_id,
        "deleted_episode": episode_index,
        "total_episodes": new_total,
        "hub_sync": hub_sync,
        "hub_sync_message": hub_sync_message,
    }


def list_user_datasets() -> list[dict[str, Any]]:
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()

    def _one_author(author: str) -> list[dict[str, Any]]:
        # Materialize the lazy generator HERE, inside the worker, so the network
        # I/O (and any GFW-killed connection) happens under the fan-out's per-call
        # timeout budget rather than lazily later while we iterate.
        # No tag filter: list EVERY dataset the account/org owns. Datasets
        # uploaded outside lerobot's push_to_hub (e.g. a raw upload_folder)
        # carry no LeRobot tag, and filter="LeRobot" made them invisible.
        rows: list[dict[str, Any]] = []
        for ds in api.list_datasets(author=author, limit=200):
            rows.append(
                {
                    "repo_id": ds.id,
                    "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                    "private": bool(getattr(ds, "private", False)),
                }
            )
        return rows

    # Fan out per-author concurrently; each author's errors are guarded so one
    # blocked/slow author degrades to "the others' results" instead of a 500.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rows in _fan_out_hub_authors(authors, _one_author):
        for row in rows:
            if row["repo_id"] in seen:
                continue
            seen.add(row["repo_id"])
            out.append(row)

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_all_datasets() -> list[dict[str, Any]]:
    """Merged listing: Hub datasets + local cache, with `source` field.

    A repo_id present in both lists is collapsed to one entry with
    source="both" and last_modified set to the more recent of the two.

    Result is cached for up to _LISTING_CACHE_TTL_S so repeated startup/nav loads
    reuse a recent listing instead of re-fanning-out to the (slow/flaky) Hub. A
    mutation (upload/delete/rename/visibility/tags) invalidates the cache so it
    reflects immediately — see invalidate_dataset_listing_cache.
    """
    global _listing_cache

    now = time.monotonic()
    with _listing_cache_lock:
        if _listing_cache is not None and (now - _listing_cache["at"]) < _LISTING_CACHE_TTL_S:
            return _listing_cache["value"]

    hub = list_user_datasets()
    local = list_local_datasets()

    merged: dict[str, dict[str, Any]] = {}
    for item in hub:
        merged[item["repo_id"]] = {**item, "source": "hub"}
    for item in local:
        rid = item["repo_id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # Keep the newer timestamp; ISO strings sort lexically.
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = {**item, "source": "local"}

    # Fold in the user's pinned custom Hub datasets (typed into the picker). Any
    # that already surfaced as their own hub row are skipped — the pin is
    # redundant then. A pinned dataset that ALSO has a local copy (the flat scan
    # found it — e.g. it was downloaded after pinning) is a Hub dataset with a
    # local copy: flip its source to "both" and keep the saved_custom flag so
    # "remove from list" (unpin) stays available. The rest join as hub rows
    # flagged saved_custom=True; private flag / timestamp / episode counts fill
    # in lazily via /datasets/hub-status and /datasets/info.
    for repo_id in get_saved_custom_datasets():
        existing = merged.get(repo_id)
        if existing is None:
            merged[repo_id] = {
                "repo_id": repo_id,
                "last_modified": None,
                "private": False,
                "source": "hub",
                "saved_custom": True,
            }
        elif existing["source"] == "local":
            existing["source"] = "both"
            existing["saved_custom"] = True

    # Hidden datasets ("removed from list") are filtered LAST — after the
    # hub/local merge and the pin fold — so a hidden id can't resurface via a
    # pin or a local copy. Re-pinning auto-unhides (see /datasets/custom), which
    # is the intended way back in.
    hidden = get_hidden_datasets()
    if hidden:
        merged = {rid: row for rid, row in merged.items() if rid not in hidden}

    out = list(merged.values())
    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)

    with _listing_cache_lock:
        _listing_cache = {"at": time.monotonic(), "value": out}
    return out


# ---------------------------------------------------------------------------
# Download a Hub dataset into the local cache (background, pollable).
# ---------------------------------------------------------------------------


class DownloadManager:
    """Runs one Hub snapshot download at a time in a background thread.

    ``snapshot_download`` of a LeRobot dataset (or a policy checkpoint) pulls
    100+ MB over the network and takes minutes, so it runs off the request
    thread (same start/poll shape as record.UploadManager) rather than block the
    browser on a multi-minute HTTP request a navigation-away would abort
    mid-fetch. One download at a time: a second concurrent start for any repo is
    refused (409-mapped by the route). The per-repo status lets the info card /
    picker poll "is *my* repo downloading?" and survive navigation.

    The state machine is repo-type agnostic — the datasets and models browsers
    share it by instantiating with their own callables:

    * ``fetch(repo_id)`` performs the actual download into the right local
      layout AND any success-side cache invalidation; it raises on failure.
    * ``cleanup(repo_id)`` (optional) removes any partial/unusable artifact a
      failed fetch left behind, so a half-download is never mistaken for a
      complete local copy.
    """

    def __init__(
        self,
        fetch: Callable[[str], None],
        cleanup: Callable[[str], None] | None = None,
    ) -> None:
        self._fetch = fetch
        self._cleanup = cleanup
        self.state: str = "idle"  # "idle" | "running" | "done" | "error"
        self.repo_id: str | None = None
        self.message: str | None = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, repo_id: str) -> dict[str, Any]:
        with self._lock:
            if self.state == "running":
                return {
                    "started": False,
                    "repo_id": self.repo_id,
                    "message": f"A download is already running for {self.repo_id}",
                }
            self.state = "running"
            self.repo_id = repo_id
            self.message = f"Downloading {repo_id} from the Hub…"
            self.error = None

        self._thread = threading.Thread(
            target=self._worker, args=(repo_id,), name="hub-download-worker", daemon=True
        )
        self._thread.start()
        return {"started": True, "repo_id": repo_id, "message": "Download started"}

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "repo_id": self.repo_id,
                "message": self.message,
                "error": self.error,
            }

    def _worker(self, repo_id: str) -> None:
        try:
            logger.info("Downloading %s from the Hub", repo_id)
            self._fetch(repo_id)
            logger.info("Downloaded %s", repo_id)
            with self._lock:
                self.state = "done"
                self.message = f"Downloaded {repo_id} to the local cache"
                self.error = None
        except Exception as exc:  # noqa: BLE001 - surface any failure to the poller
            logger.error("Error downloading %s: %s", repo_id, exc)
            if self._cleanup is not None:
                self._cleanup(repo_id)
            with self._lock:
                self.state = "error"
                self.error = str(exc)
                self.message = f"Failed to download {repo_id}: {exc}"


def _fetch_dataset_snapshot(repo_id: str) -> None:
    """Snapshot a Hub dataset into the FLAT cache layout
    ``<lerobot_home>/<repo_id>/`` (via ``local_dir``), the same layout
    recording/merge produce. That layout is recognized by BOTH
    ``is_dataset_available_locally`` (its first probe) AND ``list_local_datasets``
    (the merged-listing scan), so on completion the listing source flips from
    "hub" to "both" — which downloading only into the hub snapshot cache would
    NOT achieve (that cache isn't walked by the listing). Invalidates the
    hub-status + listing caches so the flip shows immediately."""
    target = _lerobot_cache_root() / repo_id
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(target))
    invalidate_hub_status(repo_id)
    invalidate_dataset_listing_cache()
    # The card flips from the hub summary to full local detail — drop the
    # cached hub summary alongside the listing.
    invalidate_hub_dataset_info(repo_id)


def _cleanup_partial_dataset(repo_id: str) -> None:
    """Remove a failed download's partial dir so it isn't mistaken for a
    complete local copy by is_dataset_available_locally."""
    target = _lerobot_cache_root() / repo_id
    if target.exists() and not _is_dataset_dir(target):
        shutil.rmtree(target, ignore_errors=True)


download_manager = DownloadManager(_fetch_dataset_snapshot, _cleanup_partial_dataset)


# ---------------------------------------------------------------------------
# Import a LeRobot dataset folder already on disk into the local cache.
# ---------------------------------------------------------------------------


class DatasetImportError(Exception):
    """Raised by import_local_dataset when the import can't proceed. `status` is
    the HTTP status the route should return (400 invalid source/name, 404 no
    such folder, 409 target already exists); `message` is the user-facing
    reason."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def import_local_dataset(source_path: str, name: str | None = None) -> dict[str, Any]:
    """Copy a LeRobot dataset folder already on the server machine into the flat
    cache layout, so it appears under "Local".

    `source_path` points at an existing LeRobot dataset dir (recognized by
    meta/info.json). `name` is the target repo id — a bare name or
    ``namespace/name`` — validated with the same rules the recorder uses
    (validate_dataset_repo_id → validate_dataset_name per segment); when omitted
    it defaults to the source folder's basename. The dataset is COPIED (never
    moved) into ``<lerobot_home>/<name>/`` so the user's source folder is left
    intact.

    Raises DatasetImportError (with an HTTP status + message) on: a source that
    isn't a folder / isn't a LeRobot dataset / has no episodes, a bad target
    name, a target that escapes the cache root, an overlapping source/target, or
    a target that already exists. Invalidates the listing cache on success so the
    imported dataset shows under "Local" immediately. Returns {"repo_id": ...}.

    NOTE: the copy runs SYNCHRONOUSLY (the route blocks on it, frontend shows a
    spinner). A large multi-GB dataset makes this a slow request; a background
    manager (like DownloadManager) would be the follow-up if that becomes a pain
    point, but a local disk copy is far faster than a network fetch, so inline is
    an acceptable tradeoff for now.
    """
    try:
        src = Path(source_path).expanduser().resolve()
    except OSError:
        raise DatasetImportError(400, "Invalid source path.") from None
    if not src.is_dir():
        raise DatasetImportError(404, f"No folder found at '{source_path}'.")
    if not _is_dataset_dir(src):
        raise DatasetImportError(400, "That folder isn't a LeRobot dataset (no meta/info.json inside it).")
    if not _dataset_has_episodes(src):
        raise DatasetImportError(400, "That dataset has no recorded episodes — nothing to import.")

    raw = (name or "").strip() or src.name
    ok, reason = validate_dataset_repo_id(raw)
    if not ok:
        raise DatasetImportError(400, reason)

    root = _lerobot_cache_root().resolve()
    dst = (root / raw).resolve()
    # Reject a target that escapes the cache root (traversal) — the imported
    # dataset must land strictly inside it.
    if dst == root or root not in dst.parents:
        raise DatasetImportError(400, "Invalid target dataset name.")
    # Refuse a source/target that overlap (e.g. importing a dir into itself),
    # which would corrupt the copy.
    if dst == src or src in dst.parents or dst in src.parents:
        raise DatasetImportError(400, "The source folder and the import target overlap.")
    if dst.exists():
        raise DatasetImportError(409, f"A dataset named '{raw}' already exists locally.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        logger.error("Failed to import dataset %s -> %s: %s", src, dst, exc)
        # Remove a partial copy so a failed import leaves no half-written dir.
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        raise DatasetImportError(500, f"Failed to copy the dataset: {exc}") from exc

    invalidate_hub_status(raw)
    invalidate_dataset_listing_cache()
    logger.info("Imported dataset %s -> %s", src, dst)
    return {"repo_id": raw}
