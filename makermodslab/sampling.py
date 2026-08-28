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

"""Per-episode weighted sampling, without duplicating a single byte on disk.

A merge weight (``merge.py``) is stored as a per-episode ``sampling_weight``
float in ``meta/episodes/**/*.parquet``. This module turns that column into
oversampling at *training* time: :class:`WeightedEpisodeAwareSampler` scales how
many sampling **positions** an episode contributes to an epoch by its weight and
wraps each position back into the episode's real frames.

Only the length table changes, which is what makes this cheap and safe: the
parent class's per-epoch ``(seed, epoch)`` shuffle, its ``state_dict`` /
``load_state_dict`` sample-exact resume, and its ``drop_n_first_frames`` /
``drop_n_last_frames`` handling all keep working untouched. With every weight at
1 the sampler yields the exact same sequence as the stock sampler.

**Why the module-level pending-weights handoff.** lerobot's train script
constructs the sampler itself, from index columns only — it never passes a
dataset, and draccus rejects CLI flags it does not know about. So
``train_weighted.py`` reads the weights off the dataset lerobot just loaded and
parks them here for the sampler that is about to be built. It is process-global
because there is exactly one training run per process and exactly one
``EpisodeAwareSampler`` in it (the eval dataloader uses none).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from lerobot.datasets import EpisodeAwareSampler

logger = logging.getLogger(__name__)

#: Per-episode float column carrying a merge weight. Absent means 1.0 for every
#: episode (R3) — every dataset recorded before this feature trains unchanged.
SAMPLING_WEIGHT_COLUMN = "sampling_weight"

_pending_weights: np.ndarray | None = None
_weights_required: bool = False


def require_pending_weights() -> None:
    """Declare that this process must not train unweighted (R6).

    Once armed, a sampler built with no explicit weights and nothing parked for
    it raises instead of silently defaulting to all-1. The launcher only routes a
    run through the shim when the dataset really does carry non-unit weights, so
    "no weights found" there means the handoff broke — and a run that quietly
    ignored the weights would look exactly like a successful one.
    """
    global _weights_required
    _weights_required = True


def set_pending_weights(weights: Sequence[float] | np.ndarray | None) -> None:
    """Park the weights for the next sampler built in this process."""
    global _pending_weights
    _pending_weights = None if weights is None else np.asarray(weights, dtype=np.float64)


def clear_pending_weights() -> None:
    """Forget any parked weights and drop the R6 requirement (tests, teardown)."""
    global _pending_weights, _weights_required
    _pending_weights = None
    _weights_required = False


def _take_pending_weights(n_episodes: int) -> np.ndarray | None:
    """The parked weights, if they fit ``n_episodes``; None when unweighted.

    Raises when weights are required (see :func:`require_pending_weights`) but
    missing or the wrong length.
    """
    if _pending_weights is None:
        if _weights_required:
            raise RuntimeError(
                "This run was launched to honour per-episode sampling weights, but none "
                "reached the sampler. Refusing to train unweighted — the result would be "
                "indistinguishable from a correct run. Check that the dataset still "
                f"carries a `{SAMPLING_WEIGHT_COLUMN}` column in meta/episodes."
            )
        return None
    if _pending_weights.shape != (n_episodes,):
        raise RuntimeError(
            f"Sampling weights cover {_pending_weights.shape[0]} episodes but the sampler "
            f"was built over {n_episodes}. Refusing to train: applying them positionally "
            "would oversample the wrong episodes."
        )
    return _pending_weights


def episode_weights_from_dataset(dataset: object) -> np.ndarray | None:
    """Per-episode weights read off a loaded ``LeRobotDataset``, or None.

    Reads ``dataset.meta.episodes`` — the full episode table, in episode-index
    order, which is the same table lerobot's train script slices the sampler's
    ``dataset_from_index`` / ``dataset_to_index`` columns out of, so the arrays
    line up positionally even when ``--dataset.episodes`` selects a subset.

    A dataset with no such column is all-1 (R3). None means "could not read it at
    all" — e.g. a ``MultiLeRobotDataset``, which has no single episode table.
    """
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is None:
        return None
    try:
        n_episodes = len(episodes)
        names = list(getattr(episodes, "column_names", None) or [])
        if SAMPLING_WEIGHT_COLUMN not in names:
            return np.ones(n_episodes, dtype=np.float64)
        raw = episodes[SAMPLING_WEIGHT_COLUMN]
    except Exception as exc:
        logger.warning("Could not read `%s` from dataset metadata: %s", SAMPLING_WEIGHT_COLUMN, exc)
        return None
    # A null in the column is a missing weight, which is 1.0 (R3) — not a zero,
    # which would silently drop the episode from training.
    return np.asarray([1.0 if value is None else float(value) for value in raw], dtype=np.float64)


class WeightedEpisodeAwareSampler(EpisodeAwareSampler):
    """``EpisodeAwareSampler`` that oversamples episodes by ``sampling_weight``.

    Per kept episode ``e``, with ``real_length`` the parent's DROP-ADJUSTED
    length (``to - drop_n_last - (from + drop_n_first)``)::

        weight == 0                      -> episode excluded outright
        effective_length = max(1, round(real_length * weight))
        position_in_episode %= real_length

    ``_cum_lengths`` is built from ``effective_length`` so an epoch draws
    ``weight x`` as many positions from that episode, while ``_starts`` still
    points at its real first (post-drop) frame. ``len(self)`` therefore grows
    with the total weight — exactly as a physically duplicated dataset's length
    did, which is what keeps steps-per-epoch accounting comparable. lerobot's
    train script feeds ``len(sampler)`` to ``compute_sampler_state``, so resume
    follows the inflated length on its own.

    Weights default to whatever ``train_weighted.py`` parked in this module; pass
    ``sampling_weights`` explicitly to bypass that (tests, direct use). All-1
    weights reproduce the parent's yielded sequence exactly (R2).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
        sampling_weights: Sequence[float] | np.ndarray | None = None,
    ):
        super().__init__(
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=shuffle,
            seed=seed,
            absolute_to_relative_idx=absolute_to_relative_idx,
        )

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True
        used &= lengths > 0

        # The selection above mirrors the parent's. If a lerobot bump changes how
        # episodes are dropped, the mirror goes stale and every weight would land
        # on the wrong episode — so check it against what the parent actually
        # built and refuse rather than mis-weight (R5).
        if not np.array_equal(starts[used], self._starts):
            raise RuntimeError(
                "The installed lerobot's EpisodeAwareSampler selects episodes differently "
                "than makermodslab.sampling expects, so per-episode weights cannot be "
                "aligned to it. Refusing to train: the weights would be applied to the "
                "wrong episodes. Update makermodslab.sampling for this lerobot version."
            )

        weights = sampling_weights
        if weights is None:
            weights = _take_pending_weights(len(from_indices))
        if weights is None:
            weights = np.ones(len(from_indices), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(from_indices),):
            raise ValueError(
                f"sampling_weights must have one entry per episode "
                f"({len(from_indices)}), got {weights.shape[0]}"
            )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sampling_weights must all be finite and >= 0")

        # Zero-weight episodes are dropped BEFORE the max(1, ...) clamp below:
        # clamping first would round every excluded episode back up to one frame.
        keep = used & (weights > 0)
        if not keep.any():
            raise ValueError(
                "Every episode has a sampling weight of 0 (or was dropped by "
                "drop_n_first_frames/drop_n_last_frames), so there is nothing to sample."
            )
        excluded = int(np.count_nonzero(used & ~keep))
        if excluded:
            logger.info("Excluding %d episode(s) with sampling_weight 0.", excluded)

        real_lengths = lengths[keep]
        effective = np.maximum(1, np.rint(real_lengths * weights[keep])).astype(np.int64)

        self._starts = starts[keep]
        #: Drop-adjusted real frame count per kept episode. The wrap modulus —
        #: NOT the raw episode length, which would resurrect dropped frames.
        self._real_lengths = real_lengths
        self._cum_lengths = np.cumsum(effective)
        self._num_frames = int(self._cum_lengths[-1])

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        # Wrap FIRST: before the start offset is added and before the
        # absolute->relative lookup, or an oversampled position runs past the
        # episode's last frame and off the end of the mapping.
        position_in_episode %= int(self._real_lengths[episode])
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx
