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

The subprocess entry is ``python -m makermodslab.merge <output_repo_id> <src> <src>…``,
optionally with ``--weights <n> <n>…`` (one per source).

**Weights are sampling weights, not copies.** Every source is handed to
``aggregate_datasets`` exactly once; the weight is then stamped into the
output's ``meta/episodes/**/*.parquet`` as a ``sampling_weight`` column, one row
per episode. A weight-3 source therefore costs 1x disk and is oversampled at
*training* time by ``makermodslab.sampling.WeightedEpisodeAwareSampler`` (see
``makermodslab/train_weighted.py``). Retuning a ratio is an edit to a number,
not a re-merge.

Stamping is skipped entirely when every weight is 1, so an unweighted merge
writes byte-for-byte what it wrote before this feature existed and the output
carries no new column at all — except that if a SOURCE was itself a weighted
merge, ``aggregate_datasets`` copies its ``sampling_weight`` column straight
through; ``_strip_sampling_weights`` removes that inherited column on any merge
that is not deliberately stamping weights, so "weighted" is always a choice this
run made rather than one a source smuggled in.

``--duplicate`` restores the old physical-duplication behaviour: each source is
passed to ``aggregate_datasets`` ``weight`` times (``_expand_weighted``) and no
``sampling_weight`` is written. It is an escape hatch for the CLI only — the UI
never sends it — kept until weighted sampling has trained a model end to end.
Repeating a source is safe by construction: ``aggregate_datasets`` offsets each
source's ``episode_index`` by the destination's running ``total_episodes`` and
re-keys its ``src_to_dst`` video map per source, so copy N never collides with
copy N-1. The cost is honest and on-disk: weight 3 stores 3 copies.
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from pydantic import BaseModel

from lerobot.configs import VIDEO_ENCODER_INFO_KEYS
from lerobot.datasets.aggregate import aggregate_datasets

# The column name lives with the sampler that consumes it, so the writer here
# and the reader in datasets.py can never drift apart.
from .sampling import SAMPLING_WEIGHT_COLUMN
from .utils.config import validate_dataset_repo_id
from .utils.system import torchcodec_loads


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
            from .datasets import hub_copy_has_data, hub_repo_exists

            # Only a *confirmed* absence blocks; None ("couldn't tell") falls
            # through, as before. Unlike the old merge-private check, the id is
            # resolved first, so a locally-recorded dataset that IS on the Hub
            # stops being reported as missing.
            exists = hub_repo_exists(repo_id)
            if exists is False:
                return (
                    f"Dataset \"{repo_id}\" wasn't found — it isn't in your local "
                    "cache or on the Hugging Face Hub. Check the name (or log in "
                    "if it's a private dataset)."
                )
            # "Exists" is not "retrievable": a repo left behind by a
            # half-finished upload holds no dataset, and letting it through
            # would fail deep in the merge subprocess with a misleading
            # "incomplete or corrupt — re-record it" pointing at the wrong
            # cause. Asked only on a confirmed-existing repo (a None existence
            # would just fail the same way again), and only a *confirmed*
            # empty blocks; None falls through.
            if exists and hub_copy_has_data(repo_id) is False:
                return (
                    f'Dataset "{repo_id}" exists on the Hub but holds no data — '
                    "an earlier upload didn't finish. Re-upload it (or remove it "
                    "from the merge)."
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


# Features that may be DROPPED to make otherwise-identical datasets mergeable,
# rather than blocking the merge.
#
# Exactly one entry, and the allowlist is deliberately closed rather than "any
# scalar the sources disagree about". A coaching (DAgger) dataset carries an
# `intervention` bool that a recorded dataset does not, so merging corrections
# back into the demos they were collected against — the entire point of a
# coaching session — hits `features_equal_for_merge`'s `set(a) != set(b)` and is
# refused. Dropping it is lossless HERE because coaching runs in lerobot's
# corrections-only mode, where every recorded frame is a human correction and
# the column is therefore constant True: it distinguishes nothing.
#
# That reasoning is load-bearing. If continuous recording
# (`--strategy.record_autonomous=true`) is ever enabled, autonomous frames enter
# the same dataset with `intervention=False`, the column starts carrying real
# provenance, and dropping it silently discards the signal a weighted fine-tune
# would need (lerobot PR #4222). Revisit this before flipping that flag —
# see makermodslab/dagger_runner.py, which refuses the mode outright today.
DROPPABLE_FEATURES = frozenset({"intervention"})


def _looks_like_our_coaching_dataset(repo_id: str) -> bool:
    """True when the `rollout_` prefix says WE produced this dataset.

    That prefix is the only evidence available at this point, and it is the only
    thing that makes the losslessness argument hold: our coaching runner refuses
    `record_autonomous=true` (`dagger_runner.WebDAggerStrategy.run`) and writes
    `intervention=True` on every frame it records, so the column really is
    constant. It says nothing about a dataset from anywhere else."""
    return repo_id.split("/", 1)[-1].startswith("rollout_")


def _droppable_prompt(features: list[str], sources: list[str]) -> str:
    """Ask to drop a column, claiming only what we can actually support.

    The losslessness claim is TRUE for datasets this app produced and unfounded
    for anything else. Upstream lerobot's DAgger has a `record_autonomous=true`
    mode we refuse but do not own — a dataset recorded that way, or pulled from
    the Hub, carries real provenance in `intervention`, and dropping it
    relabels every autonomous frame as a human demonstration. Nothing here
    opens a parquet, so the column's values are genuinely unknown; the honest
    move is to say which case we are in rather than assert the reassuring one.
    """
    names = ", ".join(f"`{f}`" for f in features)
    base = f"These datasets are identical apart from the {names} column, which only one of them has."
    # Which source the column came from is not knowable here — nothing opens a
    # parquet — so attribute it to the coaching dataset among the sources and
    # NAME that attribution, so an operator merging something else can see the
    # guess and catch it.
    ours = [s for s in sources if _looks_like_our_coaching_dataset(s)]
    if ours:
        return (
            f"{base} It comes from {', '.join(ours)}, a coaching session, where every frame "
            "is a human correction — so the column is the same on all of them and drops "
            "losslessly. Drop it and merge?"
        )
    return (
        f"{base} None of these came from a coaching session here, so the column's values "
        "have not been checked. If those frames are a mix of autonomous and human control, "
        "dropping it makes them indistinguishable in the merged dataset. Your originals are "
        "not modified. Drop it and merge?"
    )


def merge_droppable_features(repo_ids: list[str]) -> list[str]:
    """Allowlisted features present in SOME but not all sources.

    These are the only feature differences the merge will offer to resolve
    instead of refusing. Returned sorted so the message and the client's
    acknowledgement agree on the order."""
    infos = [info for repo_id in repo_ids if (info := _load_info(repo_id)) is not None]
    if len(infos) < 2:
        return []
    feature_sets = [set(info.get("features") or {}) for info in infos]
    in_all = set.intersection(*feature_sets)
    not_in_all = set().union(*feature_sets) - in_all
    return sorted(not_in_all & DROPPABLE_FEATURES)


def _comparable_video_info(feature: Any) -> dict[str, Any]:
    """The stream properties lerobot actually COMPARES for a video feature, or
    ``{}`` when they cannot be read.

    `features_equal_for_merge` ignores the six encoder-TUNING keys in
    `VIDEO_ENCODER_INFO_KEYS` (crf, preset, g, fast_decode, extra_options,
    video_backend) — those legitimately differ between two recordings of the
    same thing — and compares every other key in the block. The constant is
    imported rather than restated so this tracks the lerobot pin instead of
    drifting away from it.

    Returns ``{}`` when there is no `video.codec`, which means either a dataset
    old enough to predate the stream block or one that has never encoded a frame
    (a 0-episode coaching dataset carries only `is_depth_map`). Neither can be
    compared, so they take the same route as a Hub-only source: silence here,
    and the subprocess backstop covers it.
    """
    if not isinstance(feature, dict):
        return {}
    info = feature.get("info")
    if not isinstance(info, dict) or "video.codec" not in info:
        return {}
    return {key: value for key, value in info.items() if key not in VIDEO_ENCODER_INFO_KEYS}


def _merge_incompatibility(repo_ids: list[str], drop_features: Sequence[str] = ()) -> str | None:
    """Return a friendly one-line message describing the first incompatibility
    between the source datasets, or None if they're compatible (or can't be
    checked because their metadata isn't available locally).

    Hub-only sources with no local ``info.json`` are skipped — the subprocess
    backstop covers those. Compares every readable source against the first
    readable one on fps, camera set, video stream format, and feature
    keys/shapes.

    ``drop_features`` names features the caller has agreed to strip before
    aggregating (see :data:`DROPPABLE_FEATURES`); they are excluded from the
    comparison because they will not exist by the time lerobot sees the data.
    """
    dropped = set(drop_features)
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

        # video stream format — compared by lerobot, invisible to the checks above
        #
        # Placed BEFORE the feature-key comparison on purpose. A coaching
        # dataset differs from the demonstrations twice over: it carries an
        # `intervention` column (droppable, prompted for) and it is encoded with
        # a different codec (not droppable, not fixable here). Reporting the
        # column first would walk the operator through agreeing to drop a column
        # permanently and only THEN fail — `start()` withholds the drop prompt
        # unless dropping actually makes the sources mergeable, so surfacing the
        # codec here is what makes that guard fire.
        #
        # `video.codec` is derived from the encoded stream, not from the
        # requested setting (video_utils stamps `codec.canonical_name`), so the
        # two flows disagree by construction: record.py asks for `vcodec="auto"`
        # and gets hardware H.264, while rollout.py deliberately takes the
        # software default and gets AV1 (see its comment — "auto" resolves to
        # h264_nvenc on the station and PyAV then fails to open it). Without
        # this, merging corrections into the demos they were collected against
        # — the entire point of a coaching session — dies inside
        # `validate_all_metadata` as a subprocess crash whose message is both
        # features dicts dumped as JSON.
        for cam in sorted(base_cams):
            base_stream = _comparable_video_info(base_features.get(cam))
            other_stream = _comparable_video_info(other_features.get(cam))
            if not base_stream or not other_stream:
                continue  # unencoded or too old to say; the subprocess still backstops it
            differing_props = sorted(
                key
                for key in set(base_stream) | set(other_stream)
                if base_stream.get(key) != other_stream.get(key)
            )
            if not differing_props:
                continue
            if "video.codec" in differing_props:
                return (
                    f"Datasets were encoded with different video codecs: `{base_id}` is "
                    f"{base_stream['video.codec']}, `{other_id}` is "
                    f"{other_stream['video.codec']} (camera {_short_cam(cam)}). lerobot "
                    "refuses to merge video streams whose codecs differ. Coaching "
                    "sessions encode with the software AV1 encoder while recordings use "
                    "hardware H.264, so one side has to be re-encoded before these can "
                    "be merged — dropping a column will not help."
                )
            pretty = ", ".join(key.removeprefix("video.") for key in differing_props)
            return (
                f"Datasets have different video formats: `{base_id}` and `{other_id}` "
                f"differ in {pretty} (camera {_short_cam(cam)}). All datasets must share "
                "the same video format to merge."
            )

        # non-camera feature keys or per-feature shape mismatch
        differing: list[str] = []
        for key in sorted(set(base_features) | set(other_features)):
            if key in base_cams or key in other_cams:
                continue  # camera differences handled above
            if key in dropped:
                continue  # stripped from every source before aggregation
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
            # The arm note only makes sense when it's the proprioceptive vectors
            # that clash — that's the difference two different-DOF robots cause.
            arm_note = (
                _arm_shape_note(base, other)
                if {"observation.state", "action"}.intersection(differing)
                else ""
            )
            return (
                f"Datasets have different features: `{base_id}` vs `{other_id}` "
                f"differ in {', '.join(differing)}. All datasets must share "
                "identical features to merge." + arm_note
            )

    return None


def _arm_shape_note(base_info: dict[str, Any], other_info: dict[str, Any]) -> str:
    """A trailing sentence naming the arm-type difference when two sources'
    ``robot_type`` strings resolve to different arm families — appended to the
    feature-mismatch message so an SO-101 (6-DOF) vs Maker/Metal (7-DOF) merge
    rejection reads as "different robots" rather than a bare "differ in action,
    observation.state"."""
    from .arm_capabilities import ARM_TYPE_LABEL, arm_type_from_robot_type

    base_arm = arm_type_from_robot_type(base_info.get("robot_type"))
    other_arm = arm_type_from_robot_type(other_info.get("robot_type"))
    if base_arm and other_arm and base_arm != other_arm:
        return (
            f" These datasets were recorded on different robots — {ARM_TYPE_LABEL[base_arm]} "
            f"and {ARM_TYPE_LABEL[other_arm]} — which have different joint counts, so "
            "their state and action vectors can't be combined."
        )
    return ""


def _dataset_arm_types(repo_ids: list[str]) -> dict[str, str]:
    """Normalised arm type for each source whose arm can be established — a local
    ``meta/info.json`` first, then the Hub's ``meta/info.json`` summary for a
    source not in the cache. Sources whose ``robot_type`` is missing or
    unrecognized are simply absent from the result (a warning must be provable,
    not a guess).
    """
    from .arm_capabilities import arm_type_from_robot_type

    arms: dict[str, str] = {}
    for repo_id in repo_ids:
        info = _load_info(repo_id)
        if info is not None:
            # A local source: trust its info.json even if it has no robot_type
            # (an imported/community dataset) — don't then go to the Hub for it.
            robot_type = info.get("robot_type")
        else:
            # Not in the cache: one small (cached) meta/info.json GET, the same
            # shape of Hub probe _merge_source_problem already does per hub-only
            # source.
            from .datasets import get_hub_dataset_info

            try:
                hub_info = get_hub_dataset_info(repo_id)
            except Exception:  # pragma: no cover — network/offline best-effort
                hub_info = None
            robot_type = hub_info.get("robot_type") if hub_info else None
        arm_type = arm_type_from_robot_type(robot_type)
        if arm_type is not None:
            arms[repo_id] = arm_type
    return arms


def _arm_mismatch_warning(repo_ids: list[str]) -> str | None:
    """A warning if the sources span more than one arm family, else None.

    Advisory, not a hard block: the two CAN families (Maker, Metal) are both
    7-DOF, so such a merge aggregates cleanly at the file level and the user may
    know the joint semantics line up. A cross-DOF pair (SO-101 vs a CAN arm) is
    already refused by ``_merge_incompatibility`` on the state/action shape —
    this only names the reason. Datasets whose arm can't be established don't
    count either way.
    """
    from .arm_capabilities import ARM_TYPE_LABEL

    arms = _dataset_arm_types(repo_ids)
    distinct = sorted(set(arms.values()))
    if len(distinct) < 2:
        return None
    groups = "; ".join(
        f"{', '.join(f'`{r}`' for r in repo_ids if arms.get(r) == arm)} on {ARM_TYPE_LABEL[arm]}"
        for arm in distinct
    )
    return (
        f"These datasets were recorded on different robot arms — {groups}. Merging them "
        "makes one dataset whose episodes don't share joint semantics, so a policy trained "
        "on it learns an average of two robots. Merge anyway only if you know the arms are "
        "equivalent."
    )


#: Largest weight the UI/API will accept for one source. A fat-fingered "300"
#: no longer fills the drive (weights are metadata now), but it would still make
#: that source ~99% of every epoch — and, under ``--duplicate``, write hundreds
#: of GB into the cache.
MAX_SOURCE_WEIGHT = 20


class MergeRequest(BaseModel):
    source_repo_ids: list[str]
    output_repo_id: str
    #: Per-source sampling weight, positionally aligned with ``source_repo_ids``.
    #: Stored per episode in the merged dataset and applied at training time — it
    #: costs no extra disk. ``None`` means "all 1" (the pre-weights behaviour).
    source_weights: list[int] | None = None
    # Features the caller has agreed to strip so the sources become mergeable
    # (see DROPPABLE_FEATURES). Empty on a first attempt: a merge that needs a
    # drop is refused with `droppable_features` in the payload, and the client
    # re-submits echoing them back. An explicit acknowledgement rather than a
    # silent strip — dropping a column from a dataset is not something to do on
    # the user's behalf without telling them.
    drop_features: list[str] = []
    #: Set once the user has seen and accepted an advisory warning (today: the
    #: sources were recorded on different arm families). A merge that raises such
    #: a warning refuses with ``started=False`` + ``warnings`` until this is
    #: true; hard incompatibilities (fps, cameras, feature shape) ignore it.
    acknowledge_warnings: bool = False


def _weights_problem(n_sources: int, weights: list[int] | None) -> str | None:
    """Return a friendly message if ``weights`` can't be applied to
    ``n_sources`` sources, else None. ``None`` weights are always valid."""
    if weights is None:
        return None
    if len(weights) != n_sources:
        return (
            f"Got {len(weights)} weights for {n_sources} datasets — there must be "
            "exactly one weight per selected dataset."
        )
    for weight in weights:
        if weight < 1:
            return "Each dataset's weight must be at least 1."
        if weight > MAX_SOURCE_WEIGHT:
            return (
                f"A weight of {weight} is too high — the maximum is "
                f"{MAX_SOURCE_WEIGHT}. A weight multiplies how often a source is "
                "sampled per epoch, so a large one drowns out everything else."
            )
    return None


def _source_episode_counts(roots: list[Path]) -> list[int]:
    """Each source's ``total_episodes``, read from its own ``meta/info.json``.

    Read rather than assumed: ``aggregate_datasets`` lays the sources out
    back-to-back and renumbers ``episode_index`` contiguously from 0, so these
    counts ARE the source boundaries in the output. Guessing them (e.g. from
    equal splits) would stamp the wrong source's weight onto an episode.
    """
    counts: list[int] = []
    for root in roots:
        info = json.loads((root / "meta" / "info.json").read_text())
        counts.append(int(info["total_episodes"]))
    return counts


def _weight_per_episode(episode_counts: list[int], weights: list[int]) -> list[float]:
    """Flatten per-SOURCE weights into one weight per OUTPUT episode index.

    ``([2, 3], [1, 3])`` -> ``[1.0, 1.0, 3.0, 3.0, 3.0]``.
    """
    return [
        float(weight) for count, weight in zip(episode_counts, weights, strict=True) for _ in range(count)
    ]


def _strip_sampling_weights(output_root: Path) -> int:
    """Drop any inherited ``sampling_weight`` column from the merged dataset's
    ``meta/episodes/**/*.parquet``. Returns the number of chunk files rewritten.

    ``aggregate_datasets`` copies each source's episode rows verbatim, so merging
    a dataset that ALREADY carries weights leaves a partial column on the output
    (the weighted source's rows keep their values, everyone else's read back as
    1.0). ``dataset_is_weighted`` would then return True and the weighted trainer
    would launch for a merge the UI called unweighted. An unweighted merge must
    produce an unweighted dataset — so unless this run is deliberately stamping
    weights, the column is removed here.
    """
    episodes_dir = output_root / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return 0
    rewritten = 0
    for parquet_path in sorted(episodes_dir.glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        if SAMPLING_WEIGHT_COLUMN not in table.column_names:
            continue
        table = table.drop_columns([SAMPLING_WEIGHT_COLUMN])
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, parquet_path)
        rewritten += 1
    return rewritten


def _stamp_sampling_weights(output_root: Path, weight_by_episode: list[float]) -> int:
    """Write ``sampling_weight`` into every ``meta/episodes/**/*.parquet`` of the
    merged dataset, keyed by each row's own ``episode_index``. Returns the number
    of episode rows stamped.

    Keyed by ``episode_index`` rather than by row order because the episode rows
    are chunked across several parquet files and only the column is authoritative
    about which output episode a row is.

    Raises if the episode rows don't line up with ``weight_by_episode`` — a
    mis-stamped weight is worse than a failed merge, because it would silently
    oversample the wrong episodes.
    """
    episodes_dir = output_root / "meta" / "episodes"
    if not episodes_dir.is_dir():
        raise RuntimeError(
            f"{output_root} has no meta/episodes directory, so per-episode sampling "
            "weights cannot be written. The merge did not produce a v3.0 dataset."
        )

    # Pass 1: read only the episode_index column (cheap — the rest of an episode
    # row is per-feature stats) and validate the whole layout BEFORE writing
    # anything. Half a merged dataset carrying weights and half not is worse than
    # a merge that refuses.
    paths = sorted(episodes_dir.glob("**/*.parquet"))
    per_file: list[tuple[Path, list[float]]] = []
    total_rows = 0
    for parquet_path in paths:
        indices = pq.read_table(parquet_path, columns=["episode_index"]).column(0).to_pylist()
        try:
            weights = [weight_by_episode[int(idx)] for idx in indices]
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{parquet_path} references an episode_index outside "
                f"0..{len(weight_by_episode) - 1}, so its sampling weight is unknown: {exc}"
            ) from exc
        per_file.append((parquet_path, weights))
        total_rows += len(indices)

    if total_rows != len(weight_by_episode):
        raise RuntimeError(
            f"Merged dataset has {total_rows} episode rows but the sources account for "
            f"{len(weight_by_episode)}. Refusing to write sampling weights that would "
            "not line up with the episodes."
        )

    # Pass 2: rewrite each chunk with the column attached.
    for parquet_path, weights in per_file:
        table = pq.read_table(parquet_path)
        column = pa.array(weights, type=pa.float64())
        if SAMPLING_WEIGHT_COLUMN in table.column_names:
            table = table.set_column(
                table.column_names.index(SAMPLING_WEIGHT_COLUMN), SAMPLING_WEIGHT_COLUMN, column
            )
        else:
            table = table.append_column(SAMPLING_WEIGHT_COLUMN, column)
        # Write beside the target and rename: a crash mid-write must not leave a
        # truncated episodes chunk, which would make the whole dataset unreadable.
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, parquet_path)

    return total_rows


def _expand_weighted(sources: list[str], weights: list[int] | None) -> list[str]:
    """Repeat each source according to its weight, preserving order.

    ``(["a", "b"], [1, 3])`` -> ``["a", "b", "b", "b"]``.

    Only the ``--duplicate`` escape hatch passes real weights here; the default
    path passes ``None`` (one copy per source) and stores the weights as metadata
    instead — see ``_stamp_sampling_weights``. Every guard and preflight still
    runs against the *unique* sources so errors name a dataset once.
    """
    if weights is None:
        return list(sources)
    return [repo_id for repo_id, weight in zip(sources, weights, strict=True) for _ in range(weight)]


def _describe_weights(sources: list[str], weights: list[int] | None) -> str:
    """One-line ``a/base x1, a/corrections x3`` summary for the run log."""
    if weights is None:
        return ", ".join(sources)
    return ", ".join(f"{repo_id} x{weight}" for repo_id, weight in zip(sources, weights, strict=True))


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
        # Validate the weights against the RAW source list, before blanks are
        # dropped — weights are positional, so checking length after filtering
        # would silently accept a mismatched pairing.
        weights_problem = _weights_problem(len(request.source_repo_ids), request.source_weights)
        if weights_problem is not None:
            logger.warning("Rejected merge: %s", weights_problem)
            return {"started": False, "message": weights_problem}

        # Pair each source with its weight, then drop blanks as a unit so the
        # two lists can never drift apart.
        raw_weights = request.source_weights or [1] * len(request.source_repo_ids)
        pairs = [
            (repo_id.strip(), weight)
            for repo_id, weight in zip(request.source_repo_ids, raw_weights, strict=True)
            if repo_id.strip()
        ]
        sources = [repo_id for repo_id, _ in pairs]
        weights = [weight for _, weight in pairs]
        weighted = any(weight != 1 for weight in weights)

        output = request.output_repo_id.strip()
        with self._lock:
            if self.state == "running":
                return {"started": False, "message": "A merge is already in progress"}
            if len(sources) < 2:
                return {"started": False, "message": "Select at least two datasets to merge"}
            if len(set(sources)) != len(sources):
                # Selecting the same dataset twice is what weights replace; two
                # identical rows would double-count against its weight.
                return {
                    "started": False,
                    "message": (
                        "The same dataset is selected more than once. Use its weight "
                        "to include it multiple times instead."
                    ),
                }
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
            # Only strip what the caller actually acknowledged AND what is
            # genuinely mismatched — echoing back a name that isn't droppable,
            # or isn't in disagreement, must not quietly remove a column.
            droppable = merge_droppable_features(sources)
            drop = [name for name in droppable if name in set(request.drop_features)]
            incompat = _merge_incompatibility(sources, drop)
            if incompat is not None:
                unacknowledged = [name for name in droppable if name not in drop]
                if unacknowledged and _merge_incompatibility(sources, droppable) is None:
                    # The ONLY thing standing between these datasets is a
                    # droppable column. Ask rather than refuse — this is the
                    # ordinary case when merging coaching corrections back into
                    # the demonstrations they were collected against.
                    logger.info("Merge needs a dropped feature %s for %s", unacknowledged, sources)
                    return {
                        "started": False,
                        "droppable_features": unacknowledged,
                        "message": _droppable_prompt(unacknowledged, sources),
                    }
                logger.warning("Rejected merge: incompatible sources %s (%s)", sources, incompat)
                return {"started": False, "message": incompat}
            # Advisory (not a hard incompatibility): the sources span more than
            # one arm family. Refuse once with the warning attached, then let the
            # same request through when it comes back acknowledged.
            arm_warning = _arm_mismatch_warning(sources)
            if arm_warning is not None and not request.acknowledge_warnings:
                logger.warning("Merge needs confirmation: %s", arm_warning)
                return {"started": False, "message": arm_warning, "warnings": [arm_warning]}
            if arm_warning is not None:
                logger.warning("Proceeding with cross-arm merge (acknowledged): %s", arm_warning)
            self.state = "running"
            self.error = None
            self.output_repo_id = output
            self._drain_queue()
            self._close_log()
            self.log_path = None

        self._open_log()

        cmd = [sys.executable, "-m", "makermodslab.merge", output, *sources]
        if weighted:
            cmd.extend(["--weights", *(str(weight) for weight in weights)])
        for name in drop:
            cmd += ["--drop-feature", name]
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

    The Hub is addressed by the RESOLVED id (snapshot_download is a literal
    lookup, so a bare locally-recorded id 404s) while the local directory keeps
    the id the caller passed — the flat layout every other local path here
    uses, and the one aggregate_datasets is handed below.
    """
    from .datasets import resolve_hub_repo_id

    root = cache_root / repo_id
    if (root / "meta" / "info.json").exists():
        return root
    print(f"Downloading {repo_id} from the Hugging Face Hub…", flush=True)
    snapshot_download(repo_id=resolve_hub_repo_id(repo_id), repo_type="dataset", local_dir=str(root))
    print(f"Downloaded {repo_id}.", flush=True)
    return root


# Where a stripped working copy of a source lives while a merge runs. Prefixed
# so it is recognisable as machine-made residue if a crash ever leaves one
# behind, and kept inside the lerobot cache root so it shares the filesystem
# with the source it was copied from.
_STRIP_PREFIX = "_makermodslab_merge_tmp"


def _strip_features(
    repo_id: str, root: Path | None, drop: list[str], cache_root: Path, index: int
) -> tuple[str, Path] | None:
    """Write a copy of `repo_id` without `drop`, or None when it has none of them.

    A COPY, never in place. The source is the operator's own coaching dataset,
    and the `intervention` column is the only record of which frames were human
    corrections — stripping it from the original to satisfy a merge would
    destroy that provenance permanently, for a merge the operator might well
    discard afterwards.

    Correction datasets are small (a handful of short takeovers), so the copy is
    cheap; the temporary is deleted once aggregation is done.

    The two lerobot imports are local because only the SUBPROCESS ever reaches
    this function, while the module itself is imported by the FastAPI server at
    boot — `dataset_tools` pulls in the dataset-writing stack, and paying for it
    on every server start to serve a code path that runs in another process
    entirely would be pure cost."""
    from lerobot.datasets.dataset_tools import remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Same pyav fallback the training path takes (jobs.py): torchcodec is
    # lerobot's default decoder and its dylibs do not load on a host without
    # FFmpeg — `dlopen … libavutil.56.dylib` — so the first frame this dataset
    # decodes raises and the merge dies. pyav ships its own bundled FFmpeg.
    # `None` means "leave lerobot's default alone" on a host where torchcodec
    # is fine.
    video_backend = None if torchcodec_loads() else "pyav"
    dataset = LeRobotDataset(repo_id, root=root, video_backend=video_backend)
    present = [name for name in drop if name in dataset.meta.features]
    if not present:
        return None
    tmp_repo_id = f"{_STRIP_PREFIX}_{index}"
    tmp_root = cache_root / tmp_repo_id
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    print(f"Removing {', '.join(present)} from a working copy of {repo_id}…", flush=True)
    remove_feature(dataset, present, output_dir=str(tmp_root), repo_id=tmp_repo_id)
    return tmp_repo_id, tmp_root


def _run_cli(argv: list[str] | None = None) -> int:
    """Subprocess entry: aggregate the source datasets into the output repo."""
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets")
    parser.add_argument("output_repo_id")
    parser.add_argument("source_repo_ids", nargs="+")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Sampling weight per source (same order and count as the sources). "
            "Stored as a per-episode `sampling_weight` in the output's metadata "
            "and honoured at training time — it does not copy anything on disk."
        ),
    )
    parser.add_argument(
        "--duplicate",
        action="store_true",
        help=(
            "Escape hatch: apply --weights by physically repeating each source "
            "that many times instead of writing sampling weights. Costs N x disk "
            "and cannot be retuned without re-merging. Not reachable from the UI."
        ),
    )
    parser.add_argument(
        "--drop-feature",
        action="append",
        default=[],
        dest="drop_features",
        help="Feature to strip from a working copy of any source that has it, before merging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    weights_problem = _weights_problem(len(args.source_repo_ids), args.weights)
    if weights_problem is not None:
        print(weights_problem, flush=True)
        return 1

    # Weights are applied as metadata unless --duplicate asks for the old
    # physical expansion. All-1 weights are indistinguishable from no weights, so
    # neither path does anything special for them and the output is byte-for-byte
    # what an unweighted merge has always produced (R2).
    stamp_weights = bool(args.weights) and not args.duplicate and any(w != 1 for w in args.weights)
    expanded = _expand_weighted(args.source_repo_ids, args.weights if args.duplicate else None)
    if args.weights is None:
        print(f"Merging {len(args.source_repo_ids)} datasets → {args.output_repo_id}", flush=True)
    else:
        print(
            f"Merging {len(args.source_repo_ids)} datasets "
            f"({_describe_weights(args.source_repo_ids, args.weights)}) "
            f"→ {args.output_repo_id}",
            flush=True,
        )
        if args.duplicate:
            print(
                f"Weighted merge (--duplicate): {len(expanded)} dataset copies will be written.",
                flush=True,
            )
        elif stamp_weights:
            print(
                "Weighted merge: one copy of each source, weights stored per episode "
                "as `sampling_weight` and applied at training time.",
                flush=True,
            )

    # Make every source local first (downloading any that aren't), then pass its
    # root so lerobot loads from cache instead of resolving a version against the
    # Hub — the latter 404s for never-pushed datasets and, under
    # huggingface_hub >=1.x, crashes outright. A download failure is reported
    # per-source and aborts before aggregation.
    # Download once per UNIQUE source, then fan the resolved root back out over
    # the expanded list — a weight-3 source must not be fetched three times.
    cache_root = _lerobot_cache_root()
    root_by_repo: dict[str, Path] = {}
    for repo_id in args.source_repo_ids:
        try:
            root_by_repo[repo_id] = _ensure_local_source(repo_id, cache_root)
        except Exception as exc:
            print(_download_failed_message(repo_id, exc), flush=True)
            return 1
    roots: list[Path | None] = [root_by_repo[repo_id] for repo_id in expanded]

    # If aggregation dies mid-copy it leaves a partial output (e.g. meta/info.json
    # + videos/ with no completed episodes) that then makes the retry crash with a
    # raw FileExistsError. Remember whether the output existed BEFORE we started so
    # we only ever remove residue this run created — never a pre-existing dataset.
    output_root = cache_root / args.output_repo_id
    output_pre_existed = output_root.exists()

    # Swap in stripped working copies for any source carrying a dropped feature.
    # Done AFTER every source is local (the strip reads the dataset) and BEFORE
    # aggregation, so lerobot only ever sees sources whose feature sets already
    # agree. Stripped once per UNIQUE source and then fanned out over `expanded`,
    # for the same reason downloads are: under --duplicate a weight-3 source must
    # not be copied three times. `root_by_repo` deliberately keeps pointing at the
    # ORIGINALS — the temporaries are gone by the time the weights are stamped.
    repo_ids = list(expanded)
    temporaries: list[Path] = []
    if args.drop_features:
        try:
            stripped_by_repo: dict[str, tuple[str, Path]] = {}
            for index, repo_id in enumerate(args.source_repo_ids):
                stripped = _strip_features(
                    repo_id, root_by_repo[repo_id], args.drop_features, cache_root, index
                )
                if stripped is not None:
                    stripped_by_repo[repo_id] = stripped
                    temporaries.append(stripped[1])
            repo_ids = [stripped_by_repo.get(repo_id, (repo_id, None))[0] for repo_id in expanded]
            roots = [
                stripped_by_repo[repo_id][1] if repo_id in stripped_by_repo else root_by_repo[repo_id]
                for repo_id in expanded
            ]
        except Exception as exc:
            print(f"Could not remove {', '.join(args.drop_features)}: {exc}", flush=True)
            for tmp_root in temporaries:
                _cleanup_partial_output(tmp_root)
            return 1

    try:
        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=args.output_repo_id,
            roots=roots,
        )
    except Exception as exc:  # condense lerobot's giant feature-dict dumps
        friendly = _cli_friendly_error(exc, args.source_repo_ids, cache_root)
        print(friendly, flush=True)
        if not output_pre_existed and output_root.exists():
            _cleanup_partial_output(output_root)
        return 1
    finally:
        # Best-effort either way: a stranded temporary would show up in the
        # dataset library as a mystery entry, which is worse than the disk it
        # occupies.
        for tmp_root in temporaries:
            _cleanup_partial_output(tmp_root)

    # Stamp the weights only after aggregation has written the metadata it owns.
    # A failure here leaves a correct-but-unweighted dataset, which R6 forbids
    # training on silently — so the output is removed and the merge reported as
    # failed rather than handing back a dataset whose weights were dropped.
    if stamp_weights:
        try:
            counts = _source_episode_counts([root_by_repo[r] for r in args.source_repo_ids])
            stamped = _stamp_sampling_weights(output_root, _weight_per_episode(counts, args.weights))
        except Exception as exc:
            print(
                f"Merged the datasets but could not store their sampling weights: {exc}\n"
                "The output has been removed — training on it would have silently "
                "ignored the weights.",
                flush=True,
            )
            if not output_pre_existed and output_root.exists():
                _cleanup_partial_output(output_root)
            return 1
        print(f"Stored sampling weights on {stamped} episodes.", flush=True)
    else:
        # A source that was itself a weighted merge carries a `sampling_weight`
        # column that aggregate_datasets copies straight through. Left in place it
        # would make this output read as weighted when the caller asked for a
        # plain merge. Drop it so "weighted" stays a deliberate choice.
        try:
            cleared = _strip_sampling_weights(output_root)
        except Exception as exc:
            print(
                f"Merged the datasets but could not clear inherited sampling weights: {exc}\n"
                "The output has been removed — training on it would have honoured "
                "weights this merge never asked for.",
                flush=True,
            )
            if not output_pre_existed and output_root.exists():
                _cleanup_partial_output(output_root)
            return 1
        if cleared:
            print(
                f"Cleared sampling weights inherited from a weighted source ({cleared} chunk(s)).",
                flush=True,
            )

    print(f"Done. Created {args.output_repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
