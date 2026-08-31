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
"""Tests for weighted sampling: the sampler's pure logic, the injection shim's
drift guards, and the launch-time routing that keeps a weighted dataset from ever
training unweighted."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

SEED = 1234


@pytest.fixture(autouse=True)
def _clean_pending_weights():
    """The pending-weights handoff is process-global (lerobot builds the sampler
    itself), so every test starts and ends with it empty."""
    from makermodslab.sampling import clear_pending_weights

    clear_pending_weights()
    yield
    clear_pending_weights()


def _sampler(
    lengths: list[int],
    weights: list[float] | None = None,
    *,
    shuffle: bool = True,
    **kwargs,
):
    """A WeightedEpisodeAwareSampler over back-to-back episodes of `lengths`."""
    from makermodslab.sampling import WeightedEpisodeAwareSampler

    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    return WeightedEpisodeAwareSampler(
        bounds[:-1],
        bounds[1:],
        shuffle=shuffle,
        seed=SEED,
        sampling_weights=weights,
        **kwargs,
    )


def _episode_of(index: int, lengths: list[int]) -> int:
    bound = 0
    for episode, length in enumerate(lengths):
        bound += length
        if index < bound:
            return episode
    raise AssertionError(f"index {index} past the end of {lengths}")


# ---------------------------------------------------------------------------
# The sampler (step 2.1)
# ---------------------------------------------------------------------------


def test_weight_inflates_epoch_length() -> None:
    assert len(_sampler([10, 10], [1, 3])) == 40


def test_draws_are_split_by_weight() -> None:
    lengths = [10, 10]
    drawn = Counter(_episode_of(i, lengths) for i in _sampler(lengths, [1, 3]))
    assert drawn == Counter({0: 10, 1: 30})


def test_oversampled_episode_repeats_its_own_frames_evenly() -> None:
    """A weight-3 episode's 30 draws are its 10 real frames, 3 times each — the
    wrap must not spill into the next episode or resample unevenly."""
    counts = Counter(i for i in _sampler([10, 10], [1, 3]) if i >= 10)
    assert sorted(counts) == list(range(10, 20))
    assert set(counts.values()) == {3}


def test_all_weights_one_is_identical_to_stock_sampler() -> None:
    """R2: an unweighted dataset must sample exactly as it does today."""
    from lerobot.datasets import EpisodeAwareSampler

    stock = EpisodeAwareSampler([0, 10], [10, 20], shuffle=True, seed=SEED)
    assert list(_sampler([10, 10], [1, 1])) == list(stock)


def test_no_weights_at_all_is_identical_to_stock_sampler() -> None:
    """Same, via the "nothing was parked for me" path rather than explicit 1s."""
    from lerobot.datasets import EpisodeAwareSampler

    stock = EpisodeAwareSampler([0, 10], [10, 20], shuffle=True, seed=SEED)
    assert list(_sampler([10, 10], None)) == list(stock)


def test_weight_zero_excludes_the_episode() -> None:
    sampler = _sampler([10, 10], [0, 1])
    drawn = list(sampler)
    assert len(sampler) == 10
    assert len(drawn) == 10
    assert all(index >= 10 for index in drawn)


def test_all_weights_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="sampling weight of 0"):
        _sampler([10, 10], [0, 0])


def test_fractional_weights_round_per_episode() -> None:
    lengths = [10, 10]
    sampler = _sampler(lengths, [1, 1.5])
    assert len(sampler) == 25
    drawn = Counter(_episode_of(i, lengths) for i in sampler)
    # Ideal shares of 25 draws at 1 : 1.5 are 10 and 15 — realised exactly, so
    # well inside the "within 1 frame per episode" bound.
    assert abs(drawn[0] - 10) <= 1
    assert abs(drawn[1] - 15) <= 1
    assert drawn[0] + drawn[1] == 25


def test_weight_below_one_frame_still_keeps_one_position() -> None:
    """max(1, ...) floor: a tiny weight thins an episode out but never silently
    deletes it — only an explicit 0 does that."""
    sampler = _sampler([10, 10], [0.01, 1])
    assert len(sampler) == 11


def test_state_dict_resumes_mid_epoch() -> None:
    sampler = _sampler([10, 10], [1, 3])
    full = list(sampler)
    assert len(full) == 40

    sampler.load_state_dict({"epoch": 0, "start_index": 17})
    assert sampler.state_dict() == {"epoch": 0, "start_index": 17}
    assert list(sampler) == full[17:]


def test_fractional_weights_resume_through_state_dict() -> None:
    """The rounding path must not shift the epoch's order, or a resumed run would
    replay or skip frames."""
    sampler = _sampler([10, 10], [1, 1.5])
    full = list(sampler)
    assert len(full) == 25

    sampler.load_state_dict({"epoch": 0, "start_index": 9})
    assert list(sampler) == full[9:]


def test_dropped_frames_are_never_drawn_from_a_weighted_episode() -> None:
    lengths = [20, 20]
    sampler = _sampler(lengths, [1, 3], drop_n_first_frames=2, drop_n_last_frames=2)
    # 16 real frames per episode after dropping; episode 1 is oversampled 3x.
    assert len(sampler) == 16 + 48
    dropped = {0, 1, 18, 19, 20, 21, 38, 39}
    assert dropped.isdisjoint(set(sampler))


def test_absolute_to_relative_map_is_never_indexed_past_an_episode() -> None:
    """With an episode subset, lerobot hands the sampler a map covering only the
    selected frames. An oversampled position that wrapped after the start offset
    was added would run off the end of it."""
    from makermodslab.sampling import WeightedEpisodeAwareSampler

    # Three episodes of 10 frames; only 1 and 2 selected, so the map covers 10..29.
    mapping = {absolute: relative for relative, absolute in enumerate(range(10, 30))}
    sampler = WeightedEpisodeAwareSampler(
        [0, 10, 20],
        [10, 20, 30],
        episode_indices_to_use=[1, 2],
        shuffle=True,
        seed=SEED,
        absolute_to_relative_idx=mapping,
        sampling_weights=[1, 3, 1],
    )
    assert len(sampler) == 40
    drawn = list(sampler)  # a KeyError here is the bug this test exists for
    assert set(drawn) == set(range(20))
    # Episode 1 maps to relative 0..9, episode 2 to 10..19: neither may bleed past
    # its own last frame.
    assert max(i for i in drawn if i < 10) == 9
    assert max(drawn) == 19


def test_weights_must_cover_every_episode() -> None:
    with pytest.raises(ValueError, match="one entry per episode"):
        _sampler([10, 10], [1])


def test_negative_weights_are_refused() -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        _sampler([10, 10], [1, -1])


# ---------------------------------------------------------------------------
# The pending-weights handoff (R6)
# ---------------------------------------------------------------------------


def test_parked_weights_are_picked_up_without_being_passed() -> None:
    from makermodslab.sampling import set_pending_weights

    set_pending_weights([1, 3])
    assert len(_sampler([10, 10])) == 40


def test_required_weights_that_never_arrive_refuse_to_train() -> None:
    """R6: a run launched to honour weights must fail rather than quietly train
    on the unweighted mix, which would be indistinguishable from success."""
    from makermodslab.sampling import require_pending_weights

    require_pending_weights()
    with pytest.raises(RuntimeError, match="Refusing to train unweighted"):
        _sampler([10, 10])


def test_mismatched_parked_weights_refuse_to_train() -> None:
    from makermodslab.sampling import set_pending_weights

    set_pending_weights([1, 3, 3])
    with pytest.raises(RuntimeError, match="oversample the wrong episodes"):
        _sampler([10, 10])


def test_episode_weights_from_dataset_defaults_to_ones() -> None:
    """R3: no column means weight 1 everywhere, never an error."""
    from makermodslab.sampling import episode_weights_from_dataset

    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=_FakeEpisodes({"length": [10, 10, 10]})))
    weights = episode_weights_from_dataset(dataset)
    assert list(weights) == [1.0, 1.0, 1.0]


def test_episode_weights_from_dataset_reads_the_column() -> None:
    from makermodslab.sampling import episode_weights_from_dataset

    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes=_FakeEpisodes({"sampling_weight": [1.0, None, 3.0]}))
    )
    # A null cell is a missing weight, so 1.0 — not 0.0, which would drop it.
    assert list(episode_weights_from_dataset(dataset)) == [1.0, 1.0, 3.0]


def test_episode_weights_from_dataset_without_metadata_is_unknown() -> None:
    from makermodslab.sampling import episode_weights_from_dataset

    assert episode_weights_from_dataset(SimpleNamespace()) is None


class _FakeEpisodes:
    """The slice of `datasets.Dataset` that episode_weights_from_dataset uses."""

    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = columns

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def __getitem__(self, key: str) -> list:
        return self._columns[key]


# ---------------------------------------------------------------------------
# The injection shim (step 2.2)
# ---------------------------------------------------------------------------


def test_shim_accepts_the_installed_lerobot() -> None:
    import lerobot.scripts.lerobot_train as lt
    from makermodslab.train_weighted import assert_lerobot_compatible

    assert_lerobot_compatible(lt)  # no raise on the pinned version


def test_shim_refuses_when_the_sampler_symbol_is_gone() -> None:
    """R5: a lerobot bump that moves the symbol must stop the run, not fall back
    to unweighted training."""
    from makermodslab.train_weighted import assert_lerobot_compatible

    fake = SimpleNamespace(make_train_eval_datasets=lambda cfg: None, main=lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        assert_lerobot_compatible(fake)
    assert "EpisodeAwareSampler" in str(excinfo.value)
    assert _installed_lerobot_version() in str(excinfo.value)


def test_shim_refuses_when_the_constructor_signature_drifts() -> None:
    from makermodslab.train_weighted import assert_lerobot_compatible

    class Renamed:
        def __init__(self, from_indices, to_indices, shuffle=False, seed=0):  # noqa: ARG002
            pass

    fake = SimpleNamespace(
        EpisodeAwareSampler=Renamed,
        make_train_eval_datasets=lambda cfg: None,
        main=lambda: None,
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_lerobot_compatible(fake)
    message = str(excinfo.value)
    assert "dataset_from_indices" in message
    assert _installed_lerobot_version() in message


def test_shim_parks_the_datasets_weights_and_swaps_the_sampler() -> None:
    from lerobot.datasets import EpisodeAwareSampler
    from makermodslab import sampling
    from makermodslab.train_weighted import _install

    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=_FakeEpisodes({"sampling_weight": [1.0, 3.0]})))
    fake = SimpleNamespace(
        EpisodeAwareSampler=EpisodeAwareSampler,
        make_train_eval_datasets=lambda cfg: (dataset, None),  # noqa: ARG005
        main=lambda: None,
    )
    _install(fake)

    assert fake.EpisodeAwareSampler is sampling.WeightedEpisodeAwareSampler
    train, eval_ds = fake.make_train_eval_datasets("cfg")
    assert (train, eval_ds) == (dataset, None)
    assert len(_sampler([10, 10])) == 40


def _installed_lerobot_version() -> str:
    import lerobot

    return str(lerobot.__version__)


# ---------------------------------------------------------------------------
# Launch-time routing (step 2.3)
# ---------------------------------------------------------------------------


def test_unweighted_run_command_is_unchanged() -> None:
    """R2: byte-identical argv to what shipped before this feature."""
    from makermodslab.train import TrainingRequest, build_training_command

    request = TrainingRequest(dataset_repo_id="a/plain")
    baseline = build_training_command(request, "/tmp/out", "/usr/bin/python")
    assert baseline[:3] == ["/usr/bin/python", "-m", "lerobot.scripts.lerobot_train"]
    assert build_training_command(request, "/tmp/out", "/usr/bin/python", weighted=False) == baseline


def test_weighted_run_routes_through_the_shim() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    request = TrainingRequest(dataset_repo_id="a/weighted")
    plain = build_training_command(request, "/tmp/out", "/usr/bin/python")
    weighted = build_training_command(request, "/tmp/out", "/usr/bin/python", weighted=True)
    assert weighted[:3] == ["/usr/bin/python", "-m", "makermodslab.train_weighted"]
    # Nothing else about the run changes — same flags, same order.
    assert weighted[3:] == plain[3:]


def test_weighted_resume_also_routes_through_the_shim() -> None:
    """The resume branch returns early, so the module choice has to happen before
    it — otherwise a resumed weighted run would drop its weights."""
    from makermodslab.train import TrainingRequest, build_training_command

    request = TrainingRequest(
        dataset_repo_id="a/weighted", resume=True, config_path="/ckpt/train_config.json"
    )
    cmd = build_training_command(request, "/tmp/out", "/usr/bin/python", weighted=True)
    assert cmd[:3] == ["/usr/bin/python", "-m", "makermodslab.train_weighted"]
    assert "--config_path=/ckpt/train_config.json" in cmd


def test_cloud_wrapper_carries_the_sampler_modules() -> None:
    """R7 is gone: the wrapper materializes the sampler pod-side, so a weighted
    dataset trains on cloud instead of being refused.

    The sources are read from the LIVE modules at import, so this also pins that
    there is no hand-copied second sampler to drift.
    """
    import ast
    import base64
    import re

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    assert "__SAMPLING_B64__" not in WRAPPER_SOURCE
    assert "__TRAIN_WEIGHTED_B64__" not in WRAPPER_SOURCE

    embedded = dict(re.findall(r'\("(\w+\.py)", "([A-Za-z0-9+/=]+)"\)', WRAPPER_SOURCE))
    assert set(embedded) == {"sampling.py", "train_weighted.py"}

    sampling_src = base64.b64decode(embedded["sampling.py"]).decode("utf-8")
    assert "class WeightedEpisodeAwareSampler" in sampling_src
    ast.parse(sampling_src)  # what the pod writes must be importable
    ast.parse(base64.b64decode(embedded["train_weighted.py"]).decode("utf-8"))
    ast.parse(WRAPPER_SOURCE)  # and the wrapper itself must still parse


def test_cloud_run_launches_the_weighted_trainer(monkeypatch) -> None:
    """A weighted dataset bound for cloud must emit OUR trainer module — the
    wrapper keys off that name to put the sampler on PYTHONPATH."""
    from makermodslab.train import TrainingRequest, build_training_command

    argv = build_training_command(
        TrainingRequest(dataset_repo_id="a/weighted"), "/out", "python", weighted=True
    )
    assert argv[:3] == ["python", "-m", "makermodslab.train_weighted"]


def test_cloud_run_allows_an_unweighted_dataset_past_the_weight_guard(tmp_path, monkeypatch) -> None:
    """An ordinary cloud run reaches the launch path unobstructed.

    This once guarded a weighted-dataset refusal; that refusal is gone (the
    wrapper materializes the sampler pod-side), so what it pins now is simply
    that weightedness resolution does not block or crash a plain cloud launch.
    """
    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "ok"})  # noqa: ARG005
    monkeypatch.setattr("makermodslab.datasets.dataset_is_weighted", lambda repo_id: False)  # noqa: ARG005

    # Stop the launch at record creation so the test never reaches the Hub.
    # (`_assert_no_running_local` was the stop point until the local queue
    # replaced the one-run mutex; id minting is the equivalent choke point on
    # the cloud path, and it is reached only after every earlier guard passed.)
    def _stop(self, policy_type, dataset_repo_id):  # noqa: ARG001
        raise RuntimeError("reached the mutex check")

    monkeypatch.setattr(JobRegistry, "_unique_job_id", _stop)

    registry = JobRegistry(tmp_path / "root")
    with pytest.raises(RuntimeError, match="reached the mutex check"):
        registry.start(
            TrainingRequest(dataset_repo_id="a/plain"),
            JobTarget(runner="hf_cloud", flavor="a10g-small"),
        )
