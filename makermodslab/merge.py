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

"""Dataset merging: wrap lerobot's ``aggregate_datasets`` as a background job.

Aggregation copies every episode's parquet + video files and recomputes stats,
so it can take minutes for large datasets. We run it in a subprocess (same
shape as training/pip-install) and stream its stdout for a live progress log,
rather than blocking a server thread on CPU-bound work.

The subprocess entry is ``python -m makermodslab.merge <output_repo_id> <src> <src>…``.
"""

import argparse
import contextlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from pydantic import BaseModel

from lerobot.datasets.aggregate import aggregate_datasets

from .utils.config import validate_dataset_repo_id


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _merge_logs_dir() -> Path:
    """Sibling of lerobot's ``inference_logs/`` (see rollout.py) — where each
    merge subprocess's teed stdout is persisted so a failure's cause survives
    the in-memory log queue."""
    return _lerobot_cache_root() / "merge_logs"


def _dir_size(path: Path) -> int:
    """Total size in bytes of every file under ``path`` (best-effort; skips
    entries that vanish or can't be stat'd)."""
    total = 0
    for entry in path.rglob("*"):
        with contextlib.suppress(OSError):
            if entry.is_file():
                total += entry.stat().st_size
    return total


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _cleanup_partial_output(output_root: Path) -> None:
    """Best-effort remove a partial merge output the current run created, logging
    what was removed and its size. Never called for a pre-existing directory —
    the caller checks that first."""
    try:
        size = _dir_size(output_root)
    except OSError:
        size = 0
    try:
        shutil.rmtree(output_root)
        print(
            f"Cleaned up partial output {output_root} ({_human_size(size)}).",
            flush=True,
        )
    except OSError as exc:
        print(
            f"Warning: could not remove partial output {output_root}: {exc}",
            flush=True,
        )


logger = logging.getLogger(__name__)


def _load_info(repo_id: str) -> dict[str, Any] | None:
    """Load ``meta/info.json`` for a locally cached dataset, or None if it
    isn't present locally / can't be read (hub-only, corrupt, etc.)."""
    info_path = _lerobot_cache_root() / repo_id / "meta" / "info.json"
    try:
        with info_path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _camera_names(features: dict[str, Any]) -> set[str]:
    """Camera feature names: dtype == "video" or the name contains "image"."""
    return {
        name
        for name, spec in features.items()
        if (isinstance(spec, dict) and spec.get("dtype") == "video") or "image" in name
    }


def _short_cam(name: str) -> str:
    """Last dotted segment of a camera feature name, e.g.
    ``observation.images.front`` -> ``front``."""
    return name.rsplit(".", 1)[-1]


def _missing_local_file(repo_id: str, info: dict[str, Any]) -> str | None:
    """For a locally cached source (``info.json`` present), return the relative
    path of an obviously-required file that's missing on disk, or None.

    Pragmatic, not exhaustive — the ``_run_cli`` backstop is the real safety net
    for corruption the preflight can't predict. We check ``meta/tasks.parquet``
    and, when the dataset has episodes, that at least one ``data/**/*.parquet``
    file exists and any listed ``meta/episodes/`` file is present.
    """
    root = _lerobot_cache_root() / repo_id

    if not (root / "meta" / "tasks.parquet").exists():
        return "meta/tasks.parquet"

    total_episodes = info.get("total_episodes")
    if isinstance(total_episodes, int) and total_episodes > 0:
        data_dir = root / "data"
        if not any(data_dir.glob("**/*.parquet")):
            return "data/**/*.parquet"

        episodes_dir = root / "meta" / "episodes"
        # info.json doesn't inline per-episode filenames, but the aggregator
        # reads meta/episodes/**/*.parquet — if that tree exists yet is empty
        # (a half-recorded / interrupted dataset) it's corrupt.
        if episodes_dir.exists() and not any(episodes_dir.glob("**/*.parquet")):
            return "meta/episodes/chunk-000/file-000.parquet"

    return None


def _merge_source_problem(repo_ids: list[str]) -> str | None:
    """Return a friendly message for the first source that is not-found (Type A)
    or corrupt/incomplete (Type B), or None if every source is retrievable.

    Ordered independently of :func:`_merge_incompatibility`: a corrupt or
    missing source can't be feature-compared, so we surface it first.
    """
    for repo_id in repo_ids:
        info = _load_info(repo_id)

        if info is None:
            # Not in the local cache — only a *confirmed* not-found blocks here.
            # A source that exists on the Hub (or that we can't check because
            # we're offline) is allowed through: the merge subprocess downloads
            # it into the cache first (see _ensure_local_source), which also
            # sidesteps lerobot's broken in-merge Hub version resolution
            # (lerobot 0.5.2 raises RevisionNotFoundError positionally, which
            # huggingface_hub >=1.x rejects → a cryptic `response`-arg TypeError).
            # Lazy import: this module also runs as a subprocess CLI (_run_cli),
            # and datasets pulls in httpx/pyarrow the CLI path never needs.
            from .datasets import hub_repo_exists

            # Only a *confirmed* absence blocks; None ("couldn't tell") falls
            # through, as before. Unlike the old merge-private check, the id is
            # resolved first, so a locally-recorded dataset that IS on the Hub
            # stops being reported as missing.
            if hub_repo_exists(repo_id) is False:
                return (
                    f"Dataset \"{repo_id}\" wasn't found — it isn't in your local "
                    "cache or on the Hugging Face Hub. Check the name (or log in "
                    "if it's a private dataset)."
                )
            continue

        # Local source — verify the files its metadata references exist.
        rel = _missing_local_file(repo_id, info)
        if rel is not None:
            return (
                f'Dataset "{repo_id}" looks incomplete or corrupt — a file it '
                f"references is missing ({rel}). Re-record it, or remove it from "
                "the merge."
            )

    return None


def _merge_incompatibility(repo_ids: list[str]) -> str | None:
    """Return a friendly one-line message describing the first incompatibility
    between the source datasets, or None if they're compatible (or can't be
    checked because their metadata isn't available locally).

    Hub-only sources with no local ``info.json`` are skipped — the subprocess
    backstop covers those. Compares every readable source against the first
    readable one on fps, camera set, and feature keys/shapes.
    """
    infos: list[tuple[str, dict[str, Any]]] = []
    for repo_id in repo_ids:
        info = _load_info(repo_id)
        if info is not None:
            infos.append((repo_id, info))

    if len(infos) < 2:
        return None  # nothing (or not enough) to compare locally

    base_id, base = infos[0]
    base_features = base.get("features") or {}
    base_cams = _camera_names(base_features)

    for other_id, other in infos[1:]:
        # fps mismatch
        base_fps, other_fps = base.get("fps"), other.get("fps")
        if base_fps is not None and other_fps is not None and base_fps != other_fps:
            return (
                f"Datasets have different frame rates: `{base_id}` is {base_fps} fps, "
                f"`{other_id}` is {other_fps} fps. All datasets must share the same "
                "fps to merge."
            )

        other_features = other.get("features") or {}
        other_cams = _camera_names(other_features)

        # camera-set mismatch
        if base_cams != other_cams:
            added = sorted(_short_cam(c) for c in other_cams - base_cams)
            removed = sorted(_short_cam(c) for c in base_cams - other_cams)
            diff_parts = []
            if added:
                diff_parts.append(f"`{other_id}` adds: {', '.join(added)}")
            if removed:
                diff_parts.append(f"`{other_id}` is missing: {', '.join(removed)}")
            base_list = ", ".join(sorted(_short_cam(c) for c in base_cams))
            other_list = ", ".join(sorted(_short_cam(c) for c in other_cams))
            return (
                f"Datasets have different cameras: `{base_id}` has "
                f"[{base_list}], `{other_id}` has [{other_list}]. "
                f"{'; '.join(diff_parts)}. "
                "All datasets must share the same cameras to merge."
            )

        # non-camera feature keys or per-feature shape mismatch
        differing: list[str] = []
        for key in sorted(set(base_features) | set(other_features)):
            if key in base_cams or key in other_cams:
                continue  # camera differences handled above
            base_spec = base_features.get(key)
            other_spec = other_features.get(key)
            if base_spec is None or other_spec is None:  # noqa: SIM114 — missing feature spec vs shape mismatch are distinct cases; merging the branches would obscure that
                differing.append(key)
            elif (
                isinstance(base_spec, dict)
                and isinstance(other_spec, dict)
                and base_spec.get("shape") != other_spec.get("shape")
            ):
                differing.append(key)
        if differing:
            return (
                f"Datasets have different features: `{base_id}` vs `{other_id}` "
                f"differ in {', '.join(differing)}. All datasets must share "
                "identical features to merge."
            )

    return None


class MergeRequest(BaseModel):
    source_repo_ids: list[str]
    output_repo_id: str


class MergeManager:
    """Runs one dataset merge at a time as a tracked subprocess."""

    def __init__(self) -> None:
        self.state: str = "idle"  # "idle" | "running" | "done" | "error"
        self.error: str | None = None
        self.output_repo_id: str | None = None
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.log_path: str | None = None
        self._log_handle: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, request: MergeRequest) -> dict[str, Any]:
        sources = [s for s in request.source_repo_ids if s.strip()]
        output = request.output_repo_id.strip()
        with self._lock:
            if self.state == "running":
                return {"started": False, "message": "A merge is already in progress"}
            if len(sources) < 2:
                return {"started": False, "message": "Select at least two datasets to merge"}
            if not output:
                return {"started": False, "message": "An output dataset name is required"}
            name_ok, name_reason = validate_dataset_repo_id(output)
            if not name_ok:
                logger.warning("Rejected merge: invalid output name %r (%s)", output, name_reason)
                return {"started": False, "message": name_reason}
            if output in sources:
                return {"started": False, "message": "Output name must differ from the sources"}
            if (_lerobot_cache_root() / output).exists():
                logger.warning("Rejected merge: output %r already exists locally", output)
                return {
                    "started": False,
                    "message": (
                        f'A dataset named "{output}" already exists locally. '
                        "Choose a new name, or delete the existing dataset first."
                    ),
                }
            problem = _merge_source_problem(sources)
            if problem is not None:
                logger.warning("Rejected merge: unusable source %s (%s)", sources, problem)
                return {"started": False, "message": problem}
            incompat = _merge_incompatibility(sources)
            if incompat is not None:
                logger.warning("Rejected merge: incompatible sources %s (%s)", sources, incompat)
                return {"started": False, "message": incompat}
            self.state = "running"
            self.error = None
            self.output_repo_id = output
            self._drain_queue()
            self._close_log()
            self.log_path = None

        self._open_log()

        cmd = [sys.executable, "-m", "makermodslab.merge", output, *sources]
        logger.info("Starting dataset merge: %s", " ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.exception("Failed to spawn merge subprocess")
            with self._lock:
                self.state = "error"
                self.error = f"Failed to spawn merge: {exc}"
            return {"started": False, "message": str(exc)}

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return {"started": True, "message": "Merge started"}

    def get_status(self) -> dict[str, Any]:
        logs: list[dict[str, Any]] = []
        with contextlib.suppress(queue.Empty):
            while True:
                logs.append(self.log_queue.get_nowait())
        return {
            "state": self.state,
            "error": self.error,
            "output_repo_id": self.output_repo_id,
            "log_path": self.log_path,
            "logs": logs,
        }

    def _monitor(self) -> None:
        assert self.process is not None
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self._enqueue(line.rstrip())
        except Exception as exc:  # pragma: no cover — best-effort streaming
            logger.exception("Error reading merge output")
            self._enqueue(f"[merge] error reading output: {exc}")
        self.process.wait()
        return_code = self.process.returncode
        self._close_log()
        with self._lock:
            if return_code == 0:
                self.state = "done"
                self.error = None
            else:
                self.state = "error"
                self.error = f"Merge exited with code {return_code}"

    def _enqueue(self, message: str) -> None:
        # Tee to the persistent log file first (best-effort) so a failure's
        # cause survives even after the in-memory queue is drained/capped.
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.write(message + "\n")
                self._log_handle.flush()
        # Cap the queue so a chatty subprocess can't grow memory unbounded.
        if self.log_queue.qsize() >= 1000:
            with contextlib.suppress(queue.Empty):
                self.log_queue.get_nowait()
        self.log_queue.put({"timestamp": time.time(), "message": message})

    def _open_log(self) -> None:
        """Create ``merge_logs/<ts>.log`` and open it for the current run.
        Best-effort: a failure to create the log must never abort the merge."""
        try:
            log_dir = _merge_logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{int(time.time())}.log"
            self._log_handle = path.open("w", buffering=1)
            self.log_path = str(path)
        except OSError as exc:
            logger.warning("Could not open merge log file: %s", exc)
            self._log_handle = None
            self.log_path = None

    def _close_log(self) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.flush()
                self._log_handle.close()
            self._log_handle = None

    def _drain_queue(self) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                self.log_queue.get_nowait()


merge_manager = MergeManager()


def handle_start_merge(request: MergeRequest) -> dict[str, Any]:
    return merge_manager.start(request)


def handle_merge_status() -> dict[str, Any]:
    return merge_manager.get_status()


def _source_for_path(text: str, source_repo_ids: list[str], cache_root: Path) -> tuple[str, str] | None:
    """If ``text`` mentions a path under one of the sources' cache dirs, return
    ``(repo_id, relative_path)``, else None. Used to name the culprit dataset
    and the missing file in backstop messages.
    """
    for repo_id in source_repo_ids:
        prefix = str(cache_root / repo_id)
        idx = text.find(prefix)
        if idx != -1:
            tail = text[idx + len(prefix) :].lstrip("/").split()[0].rstrip("'\"),.")
            return repo_id, tail or "(unknown file)"
    return None


def _cli_friendly_error(exc: Exception, source_repo_ids: list[str], cache_root: Path) -> str:
    """Turn a raw aggregation exception into a one/two-sentence message.

    Reliable net for corruption / not-found that the in-process preflight can't
    fully predict (interrupted downloads, hub-only sources, mid-merge deletes).
    """
    text = str(exc)

    # Output already exists — normally caught by the start() preflight, but a
    # race (or a residue the cleanup couldn't remove) can still surface it here.
    if isinstance(exc, FileExistsError) or "File exists" in text:
        return (
            "The output dataset already exists locally. Choose a new name, or "
            "delete the existing dataset first."
        )

    # Type B — a referenced file is missing on disk.
    if isinstance(exc, FileNotFoundError) or "No such file or directory" in text:
        hit = _source_for_path(text, source_repo_ids, cache_root)
        if hit is not None:
            repo_id, rel = hit
            return (
                f'Dataset "{repo_id}" looks incomplete or corrupt — a file it '
                f"references is missing ({rel}). Re-record it, or remove it from "
                "the merge."
            )

    # Type A — a source doesn't exist on the Hub (and wasn't local).
    if isinstance(exc, RepositoryNotFoundError) or "404" in text or "tasks.parquet" in text:
        hit = _source_for_path(text, source_repo_ids, cache_root)
        if hit is not None:
            repo_id = hit[0]
            return (
                f"Dataset \"{repo_id}\" wasn't found — it isn't in your local "
                "cache or on the Hugging Face Hub. Check the name (or log in if "
                "it's a private dataset)."
            )
        # No path to pin to a source, but still a not-found signature.
        if isinstance(exc, RepositoryNotFoundError) or "404" in text:
            return (
                "A source dataset wasn't found — it isn't in your local cache or "
                "on the Hugging Face Hub. Check the names (or log in for private "
                "datasets)."
            )

    # Type C — lerobot resolved a source's version against the Hub and it went
    # wrong. Two shapes in this environment: a genuine RevisionNotFoundError
    # (the repo has no codebase-version tag), or a TypeError, because lerobot
    # 0.5.2 raises that error positionally while huggingface_hub >=1.x requires
    # `response=` — so constructing the friendly error itself throws. Both mean
    # the source couldn't be loaded from the Hub and isn't available locally.
    # (The preflight blocks not-downloaded sources before we get here; this is
    # the backstop for a source that vanished or lost its cache mid-merge.)
    if (
        isinstance(exc, RevisionNotFoundError)
        or "must be tagged with a codebase version" in text
        or ("HfHubHTTPError" in text and "response" in text)
    ):
        hit = _source_for_path(text, source_repo_ids, cache_root)
        who = f'Dataset "{hit[0]}"' if hit else "A source dataset"
        return (
            f"{who} couldn't be loaded from the Hugging Face Hub and isn't "
            "downloaded locally. Download it first (open or replay it), then "
            "merge. If you're offline or behind a network block, the Hub may be "
            "unreachable."
        )

    # Feature incompatibility — reuse the metadata-derived message when possible.
    friendly = _merge_incompatibility(source_repo_ids)
    if friendly is None and "Same features is expected" in text:
        friendly = (
            "Datasets have incompatible features (different cameras or "
            "signals). They must share identical features to merge."
        )
    if friendly is None:
        friendly = f"Merge failed: {type(exc).__name__}: {exc}"
    return friendly


def _download_failed_message(repo_id: str, exc: Exception) -> str:
    """One-line, actionable message for a source that couldn't be downloaded."""
    text = str(exc)
    if isinstance(exc, RepositoryNotFoundError) or "404" in text:
        return (
            f'Dataset "{repo_id}" wasn\'t found on the Hugging Face Hub. Check '
            "the name (or log in if it's a private dataset)."
        )
    return (
        f'Couldn\'t download "{repo_id}" from the Hugging Face Hub '
        f"({type(exc).__name__}). Check your internet connection or proxy and "
        "try again."
    )


def _ensure_local_source(repo_id: str, cache_root: Path) -> Path:
    """Return the local root for ``repo_id``, downloading it from the Hub into
    the lerobot cache first if it isn't already present.

    Downloading here (via huggingface_hub's own ``snapshot_download``) rather
    than letting ``aggregate_datasets`` fetch it means the source is a plain
    local dataset by the time lerobot loads it — so lerobot takes its
    cache-load path and never runs the Hub version resolution that crashes
    under huggingface_hub >=1.x (see _cli_friendly_error). Raises on failure.
    """
    root = cache_root / repo_id
    if (root / "meta" / "info.json").exists():
        return root
    print(f"Downloading {repo_id} from the Hugging Face Hub…", flush=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(root))
    print(f"Downloaded {repo_id}.", flush=True)
    return root


def _run_cli(argv: list[str] | None = None) -> int:
    """Subprocess entry: aggregate the source datasets into the output repo."""
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets")
    parser.add_argument("output_repo_id")
    parser.add_argument("source_repo_ids", nargs="+")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    print(f"Merging {len(args.source_repo_ids)} datasets → {args.output_repo_id}", flush=True)

    # Make every source local first (downloading any that aren't), then pass its
    # root so lerobot loads from cache instead of resolving a version against the
    # Hub — the latter 404s for never-pushed datasets and, under
    # huggingface_hub >=1.x, crashes outright. A download failure is reported
    # per-source and aborts before aggregation.
    cache_root = _lerobot_cache_root()
    roots: list[Path | None] = []
    for repo_id in args.source_repo_ids:
        try:
            roots.append(_ensure_local_source(repo_id, cache_root))
        except Exception as exc:
            print(_download_failed_message(repo_id, exc), flush=True)
            return 1

    # If aggregation dies mid-copy it leaves a partial output (e.g. meta/info.json
    # + videos/ with no completed episodes) that then makes the retry crash with a
    # raw FileExistsError. Remember whether the output existed BEFORE we started so
    # we only ever remove residue this run created — never a pre-existing dataset.
    output_root = cache_root / args.output_repo_id
    output_pre_existed = output_root.exists()

    try:
        aggregate_datasets(
            repo_ids=args.source_repo_ids,
            aggr_repo_id=args.output_repo_id,
            roots=roots,
        )
    except Exception as exc:  # condense lerobot's giant feature-dict dumps
        friendly = _cli_friendly_error(exc, args.source_repo_ids, cache_root)
        print(friendly, flush=True)
        if not output_pre_existed and output_root.exists():
            _cleanup_partial_output(output_root)
        return 1

    print(f"Done. Created {args.output_repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
