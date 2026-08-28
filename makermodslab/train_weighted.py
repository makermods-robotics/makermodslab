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

"""Run lerobot's trainer with per-episode sampling weights honoured.

Drop-in replacement for ``python -m lerobot.scripts.lerobot_train``: same argv,
same config parsing, same everything — except the sampler oversamples episodes
by their ``sampling_weight`` (see :mod:`makermodslab.sampling`).
``makermodslab/train.py`` selects this module only when the dataset actually
carries a non-unit weight; unweighted runs invoke lerobot untouched (R2).

**Runtime injection, never a fork (R1).** lerobot is pinned to a release tag and
hardcodes ``EpisodeAwareSampler`` in its train script with no CLI flag or config
field to swap it. So this module rebinds two names inside the already-imported
``lerobot.scripts.lerobot_train`` module and calls its ``main()``:

- ``make_train_eval_datasets`` — wrapped, to read the weights off the dataset
  lerobot just loaded and park them for the sampler.
- ``EpisodeAwareSampler`` — replaced by the weighted subclass.

**Loud on drift (R5).** Both symbols, and the sampler's constructor keyword
names, are checked before anything is rebound; a mismatch raises and names the
installed lerobot version. Falling back to unweighted training is the one
outcome worth failing hard to avoid, because it looks exactly like success.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

from .sampling import (
    WeightedEpisodeAwareSampler,
    episode_weights_from_dataset,
    require_pending_weights,
    set_pending_weights,
)

#: Constructor keywords the weighted subclass forwards to the parent. If lerobot
#: renames one of these, the subclass's ``super().__init__`` breaks — better to
#: say so up front than to fail deep inside a training run.
_REQUIRED_SAMPLER_KWARGS = (
    "dataset_from_indices",
    "dataset_to_indices",
    "shuffle",
    "seed",
)


def _lerobot_version() -> str:
    try:
        import lerobot

        return str(getattr(lerobot, "__version__", "unknown"))
    except Exception:  # pragma: no cover - lerobot is a hard dependency
        return "unknown"


def assert_lerobot_compatible(lt: Any) -> None:
    """Raise unless ``lerobot.scripts.lerobot_train`` still has what we patch."""
    version = _lerobot_version()
    for name in ("EpisodeAwareSampler", "make_train_eval_datasets", "main"):
        if not hasattr(lt, name):
            raise RuntimeError(
                f"lerobot {version}'s train script has no `{name}`, so MakerMods Lab "
                "cannot inject weighted sampling into it. Refusing to run: training "
                "would silently ignore this dataset's sampling weights. Update "
                "makermodslab/train_weighted.py for this lerobot version."
            )

    try:
        params = inspect.signature(lt.EpisodeAwareSampler.__init__).parameters
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Could not inspect lerobot {version}'s EpisodeAwareSampler constructor "
            f"({exc}), so weighted sampling cannot be injected safely."
        ) from exc

    missing = [name for name in _REQUIRED_SAMPLER_KWARGS if name not in params]
    if missing:
        raise RuntimeError(
            f"lerobot {version}'s EpisodeAwareSampler no longer accepts "
            f"{', '.join(missing)}, so MakerMods Lab's weighted sampler cannot subclass "
            "it. Refusing to run: training would silently ignore this dataset's "
            "sampling weights. Update makermodslab/sampling.py for this lerobot version."
        )


def _install(lt: Any) -> None:
    """Wrap dataset creation and swap in the weighted sampler, in that order."""
    assert_lerobot_compatible(lt)

    original_make_datasets = lt.make_train_eval_datasets

    def _make_train_eval_datasets(*args: Any, **kwargs: Any) -> Any:
        result = original_make_datasets(*args, **kwargs)
        train_dataset = result[0] if isinstance(result, tuple) else result
        set_pending_weights(episode_weights_from_dataset(train_dataset))
        return result

    lt.make_train_eval_datasets = _make_train_eval_datasets
    lt.EpisodeAwareSampler = WeightedEpisodeAwareSampler
    # Armed only now that the patch is in place: from here on, a sampler built
    # without weights is a broken handoff, not an unweighted run (R6).
    require_pending_weights()


def main() -> None:
    """Patch lerobot's train script in-process, then hand argv straight to it."""
    import lerobot.scripts.lerobot_train as lt

    _install(lt)
    lt.main()


if __name__ == "__main__":
    sys.exit(main())
