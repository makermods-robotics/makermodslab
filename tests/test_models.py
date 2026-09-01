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
"""Tests for makermodslab.models — the trained-model browser.

HF and the filesystem are MOCKED throughout: no test hits the real Hub, creates
or deletes a real repo, or removes a real file outside its tmp dir. Local runs
are seeded into a temp outputs/train via a fresh JobRegistry."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Clear the module-global /models listing cache before and after each test
    so a cached result from one test never leaks into another (the conftest
    autouse fixture resets the datasets/jobs caches but not this one)."""
    import makermodslab.models as m

    m.invalidate_model_listing_cache()
    m._published_state_cache.clear()
    yield
    m.invalidate_model_listing_cache()
    m._published_state_cache.clear()


@pytest.fixture
def registry(tmp_path: Path):
    """A JobRegistry rooted at a temp outputs/train, patched in as the module
    singleton `makermodslab.models.job_registry` reads. Watchdog is stopped so no
    background thread runs during the test."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "outputs" / "train")
    reg.shutdown()  # stop the watchdog thread; we drive state directly
    with patch("makermodslab.models.job_registry", reg), patch("makermodslab.jobs.job_registry", reg):
        yield reg


def _seed_run(
    registry,
    job_id: str,
    *,
    state: str = "done",
    runner: str = "local",
    policy_type: str = "act",
    dataset: str = "user/pick",
    steps: int = 100,
    with_checkpoint: bool = True,
    hf_repo_id: str | None = None,
    ended_at: float = 1000.0,
) -> Path:
    """Register a JobRecord directly and lay out its final checkpoint on disk.

    Returns the pretrained_model dir. When with_checkpoint is False, no
    checkpoint is written (simulating a run that died before its first save)."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    run_dir = registry._output_root / job_id / "run"
    record = JobRecord(
        id=job_id,
        name=f"run {job_id}",
        state=state,
        config=TrainingRequest(dataset_repo_id=dataset, policy_type=policy_type),
        output_dir=str(run_dir),
        started_at=1.0,
        ended_at=ended_at,
        runner=runner,
        hf_repo_id=hf_repo_id,
    )
    registry._records[job_id] = record

    pretrained = run_dir / "checkpoints" / str(steps) / "pretrained_model"
    if with_checkpoint:
        pretrained.mkdir(parents=True)
        # _list_local_checkpoints requires pretrained_model/config.json, and
        # _resolve_pretrained_dir additionally requires the policy weights lerobot
        # actually loads (model.safetensors) — a config-only tree is a partial
        # download, not a checkpoint.
        (pretrained / "config.json").write_text(json.dumps({"type": policy_type}))
        (pretrained / "model.safetensors").write_text("weights")
        (pretrained / "train_config.json").write_text(
            json.dumps(
                {
                    "policy": {"type": policy_type},
                    "dataset": {"repo_id": dataset},
                    "steps": steps,
                }
            )
        )
    return pretrained


def _add_checkpoint(registry, job_id: str, step: int, *, policy_type: str = "act") -> Path:
    """Lay down one MORE checkpoint for an already-seeded run.

    _seed_run writes a single (final) checkpoint; the publish path is the one
    caller that works over the whole set, so its tests need runs with several.
    Returns the new pretrained_model dir."""
    record = registry._records[job_id]
    pretrained = Path(record.output_dir) / "checkpoints" / str(step) / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text(json.dumps({"type": policy_type}))
    return pretrained


@pytest.fixture
def quiet_hub_reads():
    """Neutralize the two best-effort Hub READS the publish path makes AFTER the
    weights land — the published-step probe and the model-card refresh — so a
    test asserting on upload_folder isn't also asserting on network behaviour.
    Both are failure-tolerant in production (see their docstrings); this only
    makes them quiet and deterministic. The probe reports an EMPTY but READABLE
    repo, since `readable=False` deliberately suppresses the card refresh.
    Yields the model-card mock for the tests that care what the card was asked
    to contain."""
    from makermodslab.models import PublishedRepoState

    with (
        patch(
            "makermodslab.models._published_repo_state",
            return_value=PublishedRepoState({}, False, True),
        ),
        patch("makermodslab.models._sync_model_card") as card,
    ):
        yield card


# ---------------------------------------------------------------------------
# list_local_models — enumeration from the registry + train_config parsing.
# ---------------------------------------------------------------------------


def test_list_local_models_enumerates_completed_run(registry) -> None:
    from makermodslab.models import list_local_models

    pretrained = _seed_run(registry, "act_pick_2026", policy_type="act", dataset="user/pick", steps=250)

    models = list_local_models()
    assert len(models) == 1
    m = models[0]
    assert m["id"] == "act_pick_2026"
    assert m["policy_type"] == "act"
    assert m["dataset"] == "user/pick"
    assert m["steps"] == 250
    assert m["path"] == str(pretrained)
    assert m["source"] == "local"


def test_list_local_models_reads_train_config_over_record(registry) -> None:
    """policy_type / dataset come from train_config.json, not just the record."""
    from makermodslab.models import list_local_models

    pretrained = _seed_run(registry, "run_a", policy_type="act", dataset="rec/ds", steps=100)
    # Rewrite train_config.json with DIFFERENT values than the record carries.
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "policy": {"type": "smolvla"},
                "dataset": {"repo_id": "cfg/other"},
                "steps": 100,
            }
        )
    )

    m = list_local_models()[0]
    assert m["policy_type"] == "smolvla"
    assert m["dataset"] == "cfg/other"


def test_list_local_models_gates_dataset_episodes_same_as_get_model_info(registry) -> None:
    """The listing (GET /models) must never carry an ungated dataset_episodes
    for a private-dataset run — the privacy rule has to hold everywhere the
    field is emitted, not only on the per-model detail view."""
    from makermodslab.models import list_local_models

    _seed_run_with_episodes(registry, "curated_run", dataset="user/pick", episodes=[0, 1])
    with patch("makermodslab.models.is_dataset_private", return_value=True):
        assert list_local_models()[0]["dataset_episodes"] is None
    with patch("makermodslab.models.is_dataset_private", return_value=False):
        assert list_local_models()[0]["dataset_episodes"] == [0, 1]


def test_list_local_models_skips_running_but_keeps_checkpointed_interrupted(registry) -> None:
    """A run counts as usable once it's "done" or "interrupted" (not
    "running") and has a real checkpoint on disk (MT10). An "interrupted" run
    that still saved a valid final checkpoint (e.g. an unconfirmed exit after
    a server restart) must stay visible rather than vanish from the library."""
    from makermodslab.models import list_local_models

    _seed_run(registry, "done_run", state="done")
    _seed_run(registry, "running_run", state="running")
    _seed_run(registry, "interrupted_run", state="interrupted")

    ids = {m["id"] for m in list_local_models()}
    assert ids == {"done_run", "interrupted_run"}


def test_list_local_models_skips_failed_run_even_with_checkpoint(registry) -> None:
    """A "failed" run's exit code is a confirmed non-zero result, unlike
    "interrupted"'s unconfirmed one — so it must not appear in the library
    looking like a usable model, even if an earlier checkpoint saved before
    the crash (e.g. an OOM mid-training, or a manual stop that finalizes as
    "failed")."""
    from makermodslab.models import list_local_models

    _seed_run(registry, "failed_run", state="failed")
    assert list_local_models() == []


def test_list_local_models_skips_checkpointless_run(registry) -> None:
    """A run that died before its first save has no checkpoint and is hidden
    (nothing to browse / serve) regardless of its terminal state."""
    from makermodslab.models import list_local_models

    _seed_run(registry, "no_ckpt_done", state="done", with_checkpoint=False)
    _seed_run(registry, "no_ckpt_interrupted", state="interrupted", with_checkpoint=False)
    assert list_local_models() == []


def test_list_local_models_skips_non_local_runner(registry) -> None:
    from makermodslab.models import list_local_models

    _seed_run(registry, "cloud_run", state="done", runner="hf_cloud")
    assert list_local_models() == []


def test_list_local_models_exposes_state_and_target_steps_for_interrupted_run(registry) -> None:
    """An interrupted run kept in the library must be tellable apart from one
    that finished normally: `steps` is the checkpoint's actual step, distinct
    from `target_steps` (the run's configured target), and `state` carries
    the terminal state. Without this a run interrupted at step 5000 of a
    configured 10000 looks identical to one configured for (and that finished
    at) exactly 5000 steps."""
    from makermodslab.models import list_local_models

    pretrained = _seed_run(registry, "interrupted_run", state="interrupted", steps=5000)
    (pretrained / "train_config.json").write_text(
        json.dumps({"policy": {"type": "act"}, "dataset": {"repo_id": "user/pick"}, "steps": 10000})
    )

    m = list_local_models()[0]
    assert m["steps"] == 5000
    assert m["target_steps"] == 10000
    assert m["state"] == "interrupted"


def test_list_local_models_target_steps_matches_steps_for_completed_run(registry) -> None:
    """A run that trained to its configured target has steps == target_steps,
    state == "done" — the case a bare `steps` count can't be told apart from
    an interrupted run without the fields above."""
    from makermodslab.models import list_local_models

    _seed_run(registry, "done_run", state="done", steps=10000)

    m = list_local_models()[0]
    assert m["steps"] == 10000
    assert m["target_steps"] == 10000
    assert m["state"] == "done"


# ---------------------------------------------------------------------------
# list_all_models — hub/local merge + source badges.
# ---------------------------------------------------------------------------


def test_list_all_models_merges_local_and_hub(registry) -> None:
    from makermodslab.models import list_all_models

    _seed_run(registry, "local_only_run", state="done", dataset="user/pick", ended_at=1000.0)

    hub_rows = [
        {"repo_id": "user/hub_model", "last_modified": "2026-02-01T00:00:00+00:00", "private": False},
    ]
    with patch("makermodslab.models.list_hub_models", return_value=hub_rows):
        result = list_all_models()

    by_key = {m.get("id", m.get("repo_id")): m for m in result}
    assert by_key["local_only_run"]["source"] == "local"
    assert by_key["user/hub_model"]["source"] == "hub"


def test_list_all_models_collapses_pushed_run_to_both(registry, tmp_lerobot_home) -> None:
    """A local run whose hf_repo_id matches a Hub repo → one 'both' entry."""
    from makermodslab.models import list_all_models

    _seed_run(
        registry,
        "pushed_run",
        state="done",
        dataset="user/pick",
        hf_repo_id="user/hub_model",
        ended_at=5000.0,
    )
    hub_rows = [
        {"repo_id": "user/hub_model", "last_modified": "2026-01-01T00:00:00+00:00", "private": False},
    ]
    with patch("makermodslab.models.list_hub_models", return_value=hub_rows):
        result = list_all_models()

    # Collapsed: exactly one row, keyed on the hub repo id, source "both", and
    # carrying the local-only detail fields (dataset / path).
    assert len(result) == 1
    row = result[0]
    assert row["repo_id"] == "user/hub_model"
    assert row["source"] == "both"
    assert row["dataset"] == "user/pick"
    assert row["id"] == "pushed_run"


def test_list_all_models_degrades_to_local_when_hub_empty(registry, tmp_lerobot_home) -> None:
    """The hub half is best-effort; an empty/failed hub listing degrades to
    local-only rather than crashing."""
    from makermodslab.models import list_all_models

    _seed_run(registry, "local_run", state="done")
    with patch("makermodslab.models.list_hub_models", return_value=[]):
        result = list_all_models()
    assert [m["id"] for m in result] == ["local_run"]


def test_list_hub_models_empty_when_not_logged_in() -> None:
    from makermodslab.models import list_hub_models

    with patch("makermodslab.models.cached_whoami", return_value=None):
        assert list_hub_models() == []


def test_list_hub_models_filters_and_dedupes() -> None:
    """Only repos with the `lerobot` tag or a run-repo timestamp suffix qualify;
    fan-out over authors is deduped by repo_id."""
    from makermodslab.models import list_hub_models

    m_tagged = MagicMock()
    m_tagged.id = "user/act_model"
    m_tagged.tags = ["lerobot"]
    m_tagged.last_modified = None
    m_tagged.private = False

    m_run = MagicMock()
    m_run.id = "user/smolvla_ds_2026-01-01_10-00-00"
    m_run.tags = []
    m_run.last_modified = None
    m_run.private = False

    m_other = MagicMock()  # neither tag nor run-repo naming → excluded
    m_other.id = "user/random_repo"
    m_other.tags = ["some-other-tag"]
    m_other.last_modified = None
    m_other.private = False

    fake_api = MagicMock()
    fake_api.list_models.return_value = [m_tagged, m_run, m_other]

    with (
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
    ):
        rows = list_hub_models()

    ids = {r["repo_id"] for r in rows}
    assert ids == {"user/act_model", "user/smolvla_ds_2026-01-01_10-00-00"}


def test_list_all_models_surfaces_policy_type_from_name_only_tags(registry, tmp_lerobot_home) -> None:
    """BUG 2 regression: a hub repo named ``act_<stuff>`` carrying only the
    org tags (makermods / MakerModsLab), with NO ``lerobot``/policy-type tag, must
    surface policy_type "act" end-to-end through list_all_models — via the
    name-prefix fallback in _hub_policy_type. This is the exact shape whose
    policy label went missing in the picker."""
    from makermodslab.models import list_all_models

    m_named = MagicMock()
    m_named.id = "makermods/act_makermods_pick_up_red_cube_10_2026-07-04_17-09-13"
    m_named.tags = ["makermods", "MakerModsLab"]  # org tags only — no policy-type tag
    m_named.last_modified = None
    m_named.private = False

    fake_api = MagicMock()
    fake_api.list_models.return_value = [m_named]

    with (
        patch("makermodslab.models.cached_whoami", return_value={"name": "makermods", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == m_named.id)
    assert row["source"] == "hub"
    assert row["policy_type"] == "act"


def test_list_all_models_infers_pinned_model_policy_type_from_name(registry) -> None:
    """A pinned custom model the Hub listing didn't return still gets its policy
    type inferred from the repo name (act_… / smolvla_…) rather than dropping to
    None — so the picker shows the label even for a pin-only row."""
    from makermodslab.models import list_all_models

    with (
        patch("makermodslab.models.list_hub_models", return_value=[]),
        patch(
            "makermodslab.models.get_saved_custom_models",
            return_value=["makermods/smolvla_makermods_sock_2026-07-08_01-47-15"],
        ),
    ):
        result = list_all_models()

    row = next(r for r in result if r.get("id") == "makermods/smolvla_makermods_sock_2026-07-08_01-47-15")
    assert row["source"] == "hub"
    assert row["hf_repo_id"] == "makermods/smolvla_makermods_sock_2026-07-08_01-47-15"
    assert row["saved_custom"] is True
    assert row["policy_type"] == "smolvla"


# ---------------------------------------------------------------------------
# list_all_models — naming a repo-keyed row after the run that produced it.
# ---------------------------------------------------------------------------


def _seed_cloud_run(
    registry,
    job_id: str,
    *,
    repo_id: str,
    state: str = "done",
    started_at: float = 1.0,
    policy_type: str = "smolvla",
    dataset: str = "makermods/eraser",
    steps: int = 20000,
    display_name: str | None = None,
) -> None:
    """Register a cloud JobRecord publishing to `repo_id`. No local checkpoint:
    a cloud run's artifacts live on the Hub, so it never appears in
    list_local_models — only as the identity behind a Hub-keyed row."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    registry._records[job_id] = JobRecord(
        id=job_id,
        name=job_id,
        display_name=display_name,
        state=state,
        config=TrainingRequest(dataset_repo_id=dataset, policy_type=policy_type, steps=steps),
        output_dir="",
        started_at=started_at,
        ended_at=started_at + 1.0,
        runner="hf_cloud",
        hf_repo_id=repo_id,
    )


class _NoHubFiles:
    """HfApi stand-in for the registry's per-record checkpoint count: an empty
    repo listing, so seeding cloud records costs no network."""

    def list_repo_files(self, repo_id, repo_type):
        return []


def _sandboxed_listing(hub_rows: list[dict[str, Any]]):
    """The patches every list_all_models test needs to stay off the network and
    off the developer's real pinned/hidden-model files."""
    return (
        patch("makermodslab.models.list_hub_models", return_value=hub_rows),
        patch("makermodslab.models.get_saved_custom_models", return_value=[]),
        patch("makermodslab.models.get_hidden_models", return_value=set()),
        patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()),
    )


_SHARED_REPO = "makermods/smolvla_eraser_2026-07-31_17-35-54"


def test_list_all_models_names_repo_row_after_the_run_that_finished(registry, tmp_lerobot_home) -> None:
    """MT12's user-facing symptom: a cloud resume reuses its PARENT's output
    repo, so a resume chain shares one repo named after run #1. /models keys Hub
    entries by repo_id, so the run that actually finished had no entry under its
    own name — the only row was the parent's, with null steps/dataset. The row
    now carries the finishing run's identity while its routing keys (id /
    repo_id / hf_repo_id) stay the repo id."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "run_17-35-54", repo_id=_SHARED_REPO, state="failed", started_at=100.0)
    _seed_cloud_run(registry, "run_20-31-48", repo_id=_SHARED_REPO, state="failed", started_at=200.0)
    # The one that reached its 20k target — newer AND done, so it names the repo.
    _seed_cloud_run(registry, "run_22-40-15", repo_id=_SHARED_REPO, state="done", started_at=300.0)

    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == _SHARED_REPO)
    assert row["name"] == "run_22-40-15"
    # Identity is untouched — every request (info / download / deploy) routes on it.
    assert row["id"] == _SHARED_REPO
    assert row["hf_repo_id"] == _SHARED_REPO
    assert row["source"] == "hub"
    # Detail the Hub listing had no way to know.
    assert row["steps"] == 20000
    assert row["dataset"] == "makermods/eraser"
    assert row["policy_type"] == "smolvla"


def test_list_all_models_repo_row_prefers_done_over_newer_unfinished(registry, tmp_lerobot_home) -> None:
    """A later resume attempt that failed does not get to name the repo: the run
    that reached "done" published the policy sitting at the repo root."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "finished", repo_id=_SHARED_REPO, state="done", started_at=100.0)
    _seed_cloud_run(registry, "later_crash", repo_id=_SHARED_REPO, state="failed", started_at=999.0)

    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    assert next(r for r in result if r["repo_id"] == _SHARED_REPO)["name"] == "finished"


def test_list_all_models_repo_row_uses_display_name_when_renamed(registry, tmp_lerobot_home) -> None:
    """A renamed run shows its alias — the same display_name/name precedence the
    local rows and the job cards use."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(
        registry, "raw_run_id", repo_id=_SHARED_REPO, state="done", display_name="Eraser placing v3"
    )
    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    assert next(r for r in result if r["repo_id"] == _SHARED_REPO)["name"] == "Eraser placing v3"


def test_list_all_models_repo_row_falls_back_to_newest_when_none_done(registry, tmp_lerobot_home) -> None:
    """No run in the chain finished (all failed/interrupted): the newest one
    still names the repo — it is the last thing that wrote to it."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "older", repo_id=_SHARED_REPO, state="failed", started_at=100.0)
    _seed_cloud_run(registry, "newest", repo_id=_SHARED_REPO, state="interrupted", started_at=400.0)

    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == _SHARED_REPO)
    assert row["name"] == "newest"
    # Never finished and never reported a step ⇒ no step count invented.
    assert row["steps"] is None


def test_list_all_models_local_checkpoint_detail_wins_over_job_identity(registry, tmp_lerobot_home) -> None:
    """A local run collapsed into its Hub row already owns the row's name and
    checkpoint-derived detail; the run-identity pass must not overwrite it."""
    from makermodslab.models import list_all_models

    _seed_run(
        registry,
        "pushed_run",
        state="done",
        dataset="user/pick",
        steps=250,
        hf_repo_id="user/hub_model",
    )
    hub_rows = [{"repo_id": "user/hub_model", "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == "user/hub_model")
    assert row["source"] == "both"
    assert row["name"] == "run pushed_run"  # the local row's name, not re-derived
    assert row["steps"] == 250  # the checkpoint's real step, not the config target
    assert row["id"] == "pushed_run"


def test_list_all_models_repo_row_ignores_imported_records(registry, tmp_lerobot_home) -> None:
    """Re-importing a repo registers a POINTER to it, whose config is a
    placeholder (dataset "(imported)", the default 10000 steps) and which is
    always done + newest. It must not outrank the run that trained the weights,
    or the row would advertise a step count and dataset nobody trained on."""
    from makermodslab.jobs import JobRecord
    from makermodslab.models import list_all_models
    from makermodslab.train import TrainingRequest

    _seed_cloud_run(registry, "real_run", repo_id=_SHARED_REPO, state="done", started_at=100.0)
    registry._records["smolvla_imported_x"] = JobRecord(
        id="smolvla_imported_x",
        name="smolvla_imported_x",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="smolvla", steps=10000),
        output_dir="",
        started_at=900.0,  # newest, and done — would win without the guard
        runner="imported",
        hf_repo_id=_SHARED_REPO,
    )

    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == _SHARED_REPO)
    assert row["name"] == "real_run"
    assert row["steps"] == 20000
    assert row["dataset"] == "makermods/eraser"


def test_list_all_models_reduces_a_generated_run_name_to_the_task(registry, tmp_lerobot_home) -> None:
    """An auto-generated run name is "{POLICY} · {dataset}" (jobs.start). Both
    halves are printed elsewhere on the row — policy_type and dataset each have
    their own field — so the title line keeps only the task: policy prefix and
    dataset namespace both peeled."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "generated", repo_id=_SHARED_REPO, state="done")
    registry._records["generated"].name = "SMOLVLA · makermods/eraser_place"
    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == _SHARED_REPO)
    assert row["name"] == "eraser_place"
    # Neither fact is lost, just moved to where each is rendered once.
    assert row["policy_type"] == "smolvla"
    assert row["dataset"] == "makermods/eraser"


def test_list_all_models_keeps_a_generated_name_whose_dataset_has_no_namespace(
    registry, tmp_lerobot_home
) -> None:
    """A dataset id with no "/" is already the task — nothing to peel off it."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "bare", repo_id=_SHARED_REPO, state="done")
    registry._records["bare"].name = "ACT · eraser_place"
    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    assert next(r for r in result if r["repo_id"] == _SHARED_REPO)["name"] == "eraser_place"


def test_list_all_models_keeps_a_user_name_that_contains_the_separator(registry, tmp_lerobot_home) -> None:
    """Only the GENERATED shape is peeled. A job_name the user typed keeps every
    word, even when it contains " · " — the head isn't a policy type."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "typed", repo_id=_SHARED_REPO, state="done")
    registry._records["typed"].name = "Monday · eraser retrain"
    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    assert next(r for r in result if r["repo_id"] == _SHARED_REPO)["name"] == ("Monday · eraser retrain")


def test_list_all_models_leaves_untracked_repo_row_alone(registry, tmp_lerobot_home) -> None:
    """A Hub repo no tracked run publishes to keeps the repo id as its name —
    the enrichment is a fill-in, never a rewrite of unknown rows."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "other_run", repo_id="makermods/some_other_repo", state="done")
    hub_rows = [{"repo_id": "user/untracked", "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["repo_id"] == "user/untracked")
    assert row["name"] == "user/untracked"
    assert row["steps"] is None


def test_list_all_models_separates_two_runs_that_share_a_name(registry, tmp_lerobot_home) -> None:
    """The reported case: retraining one task publishes a SECOND repo, and both
    rows take the same auto-generated run name ("SMOLVLA · ns/task", from
    jobs.start, peeled to "ns/task" by the enrichment) — the picker then showed
    one label twice with nothing on either row to say which is which. The rows'
    last-modified dates break the tie; the routing keys stay untouched."""
    from makermodslab.models import list_all_models

    shared_name = "SMOLVLA · makermods/eraser_place_unblurry_real"
    # What the enrichment renders: the task alone — policy prefix and dataset
    # namespace both dropped (each has its own field on the row).
    shown = "eraser_place_unblurry_real"
    long_repo = "makermods/smolvla_makermods_eraser_place_unblurry_real_2026-07-31_17-35-54"
    short_repo = "makermods/smolvla_makermods_eraser_place_unblurry_real_2026-08-02_12-22-54"
    _seed_cloud_run(registry, "run_long", repo_id=long_repo, state="done", steps=20000)
    _seed_cloud_run(registry, "run_short", repo_id=short_repo, state="done", steps=5500)
    registry._records["run_long"].name = shared_name
    registry._records["run_short"].name = shared_name

    hub_rows = [
        {"repo_id": long_repo, "last_modified": "2026-07-31T17:35:54Z", "private": False},
        {"repo_id": short_repo, "last_modified": "2026-08-02T12:22:54Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[long_repo]["name"] == f"{shown} (2026-07-31)"
    assert by_repo[short_repo]["name"] == f"{shown} (2026-08-02)"
    assert by_repo[long_repo]["hf_repo_id"] == long_repo
    assert by_repo[short_repo]["hf_repo_id"] == short_repo


def test_list_all_models_same_day_collision_escalates_to_the_time(registry, tmp_lerobot_home) -> None:
    """Two runs of one task on one day: the date alone doesn't separate them, so
    the next rung of the ladder is used for BOTH rows."""
    from makermodslab.models import list_all_models

    shared_name = "SMOLVLA · makermods/eraser_place_unblurry_real"
    shown = "eraser_place_unblurry_real"
    a_repo = "makermods/smolvla_a_2026-07-31_17-35-54"
    b_repo = "makermods/smolvla_b_2026-07-31_12-22-54"
    _seed_cloud_run(registry, "run_a", repo_id=a_repo, state="done")
    _seed_cloud_run(registry, "run_b", repo_id=b_repo, state="done")
    registry._records["run_a"].name = shared_name
    registry._records["run_b"].name = shared_name

    hub_rows = [
        {"repo_id": a_repo, "last_modified": "2026-07-31T17:35:54Z", "private": False},
        {"repo_id": b_repo, "last_modified": "2026-07-31T12:22:54Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[a_repo]["name"] == f"{shown} (2026-07-31 17:35)"
    assert by_repo[b_repo]["name"] == f"{shown} (2026-07-31 12:22)"


def test_list_all_models_does_not_separate_two_policies_of_one_task(registry, tmp_lerobot_home) -> None:
    """An ACT and a SmolVLA of one task both enrich to the same title, but the
    row already carries `policy_type` in its own field — the card's Policy row
    separates them. Suffixing would spend the title line restating that, and
    would suggest the pair differs by when it ran rather than by what it is."""
    from makermodslab.models import list_all_models

    act_repo = "makermods/act_makermods_eraser_place_2026-07-31_17-35-54"
    smolvla_repo = "makermods/smolvla_makermods_eraser_place_2026-08-02_12-22-54"
    _seed_cloud_run(registry, "run_act", repo_id=act_repo, state="done", policy_type="act")
    _seed_cloud_run(registry, "run_smolvla", repo_id=smolvla_repo, state="done", policy_type="smolvla")
    registry._records["run_act"].name = "ACT · makermods/eraser_place"
    registry._records["run_smolvla"].name = "SMOLVLA · makermods/eraser_place"

    hub_rows = [
        {"repo_id": act_repo, "last_modified": "2026-07-31T17:35:54Z", "private": False},
        {"repo_id": smolvla_repo, "last_modified": "2026-08-02T12:22:54Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[act_repo]["name"] == "eraser_place"
    assert by_repo[smolvla_repo]["name"] == "eraser_place"
    # The fact that separates them is rendered where it belongs.
    assert by_repo[act_repo]["policy_type"] == "act"
    assert by_repo[smolvla_repo]["policy_type"] == "smolvla"


def test_list_all_models_still_separates_two_runs_of_one_policy(registry, tmp_lerobot_home) -> None:
    """The policy key narrows collisions rather than abolishing them: two
    SmolVLA runs of one task are still two rows nothing else tells apart."""
    from makermodslab.models import list_all_models

    a_repo = "makermods/smolvla_makermods_eraser_place_2026-07-31_17-35-54"
    b_repo = "makermods/smolvla_makermods_eraser_place_2026-08-02_12-22-54"
    _seed_cloud_run(registry, "run_a", repo_id=a_repo, state="done", policy_type="smolvla")
    _seed_cloud_run(registry, "run_b", repo_id=b_repo, state="done", policy_type="smolvla")
    registry._records["run_a"].name = "SMOLVLA · makermods/eraser_place"
    registry._records["run_b"].name = "SMOLVLA · makermods/eraser_place"

    hub_rows = [
        {"repo_id": a_repo, "last_modified": "2026-07-31T17:35:54Z", "private": False},
        {"repo_id": b_repo, "last_modified": "2026-08-02T12:22:54Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[a_repo]["name"] == "eraser_place (2026-07-31)"
    assert by_repo[b_repo]["name"] == "eraser_place (2026-08-02)"


def test_list_all_models_suffix_prefers_the_name_stamp_over_last_modified(registry, tmp_lerobot_home) -> None:
    """The disambiguator is WHEN THE RUN RAN, and the repo name carries that
    verbatim. `last_modified` does not: it moves on any push to the repo — a
    re-push of the same weights, a README edit, a later checkpoint upload — so
    two runs weeks apart can both report a date in September and read, next to
    each other, as simply wrong. The name's stamp never moves."""
    from makermodslab.models import list_all_models

    july_repo = "makermods/smolvla_makermods_eraser_place_2026-07-31_17-35-54"
    august_repo = "makermods/smolvla_makermods_eraser_place_2026-08-02_12-22-54"
    _seed_cloud_run(registry, "run_july", repo_id=july_repo, state="done")
    _seed_cloud_run(registry, "run_august", repo_id=august_repo, state="done")
    registry._records["run_july"].name = "SMOLVLA · makermods/eraser_place"
    registry._records["run_august"].name = "SMOLVLA · makermods/eraser_place"

    # Both repos re-pushed on the same later day: last_modified would date both
    # rows to September, and the date rung wouldn't even separate them.
    hub_rows = [
        {"repo_id": july_repo, "last_modified": "2026-09-20T10:00:00Z", "private": False},
        {"repo_id": august_repo, "last_modified": "2026-09-20T11:00:00Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[july_repo]["name"] == "eraser_place (2026-07-31)"
    assert by_repo[august_repo]["name"] == "eraser_place (2026-08-02)"


def test_list_all_models_suffix_falls_back_to_last_modified_without_a_stamp(
    registry, tmp_lerobot_home
) -> None:
    """A repo whose name carries no run stamp — a hand-named upload, a community
    repo — has only last_modified to offer, so that is what it uses."""
    from makermodslab.models import list_all_models

    v1_repo = "makermods/eraser_place_v1"
    v2_repo = "makermods/eraser_place_v2"
    _seed_cloud_run(registry, "run_v1", repo_id=v1_repo, state="done")
    _seed_cloud_run(registry, "run_v2", repo_id=v2_repo, state="done")
    registry._records["run_v1"].name = "SMOLVLA · makermods/eraser_place"
    registry._records["run_v2"].name = "SMOLVLA · makermods/eraser_place"

    hub_rows = [
        {"repo_id": v1_repo, "last_modified": "2026-07-31T17:35:54Z", "private": False},
        {"repo_id": v2_repo, "last_modified": "2026-08-02T12:22:54Z", "private": False},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    by_repo = {r["repo_id"]: r for r in result}
    assert by_repo[v1_repo]["name"] == "eraser_place (2026-07-31)"
    assert by_repo[v2_repo]["name"] == "eraser_place (2026-08-02)"


def test_list_all_models_leaves_unique_names_alone(registry, tmp_lerobot_home) -> None:
    """The collision pass is a no-op on a listing with no duplicates — a row
    never acquires a date it doesn't need to be distinguishable."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "solo_run", repo_id=_SHARED_REPO, state="done")
    hub_rows = [{"repo_id": _SHARED_REPO, "last_modified": "2026-07-31T17:35:54Z", "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    assert next(r for r in result if r["repo_id"] == _SHARED_REPO)["name"] == "solo_run"


# ---------------------------------------------------------------------------
# get_model_info.
# ---------------------------------------------------------------------------


def test_get_model_info_local(registry) -> None:
    from makermodslab.models import get_model_info

    pretrained = _seed_run(registry, "info_run", policy_type="act", dataset="user/pick", steps=100)
    (pretrained / "extra.bin").write_bytes(b"x" * 42)

    info = get_model_info("info_run")
    assert info is not None
    assert info["policy_type"] == "act"
    assert info["dataset"] == "user/pick"
    assert info["path"] == str(pretrained)
    assert info["size_bytes"] > 0  # walked the dir


def test_get_model_info_unknown_returns_none(registry) -> None:
    from makermodslab.models import get_model_info

    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        assert get_model_info("nope") is None


# ---------------------------------------------------------------------------
# dataset_episodes — training-episode provenance, gated on the source
# dataset's Hub privacy (see models._gate_dataset_episodes).
# ---------------------------------------------------------------------------


def _seed_run_with_episodes(registry, job_id: str, *, dataset: str, episodes: list[int]) -> Path:
    """_seed_run, then rewrite train_config.json with a dataset.episodes
    subset — mirrors test_list_local_models_reads_train_config_over_record's
    pattern of overwriting the file _seed_run already wrote."""
    pretrained = _seed_run(registry, job_id, dataset=dataset)
    (pretrained / "train_config.json").write_text(
        json.dumps({"policy": {"type": "act"}, "dataset": {"repo_id": dataset, "episodes": episodes}})
    )
    return pretrained


def test_get_model_info_local_shows_episodes_for_public_dataset(registry) -> None:
    from makermodslab.models import get_model_info

    _seed_run_with_episodes(registry, "curated_run", dataset="user/pick", episodes=[3, 1, 1, 2])
    with patch("makermodslab.models.is_dataset_private", return_value=False) as gate:
        info = get_model_info("curated_run")
    assert info["dataset_episodes"] == [1, 2, 3]  # deduped, sorted
    gate.assert_called_once_with("user/pick")


def test_get_model_info_local_hides_episodes_for_private_dataset(registry) -> None:
    from makermodslab.models import get_model_info

    _seed_run_with_episodes(registry, "curated_run", dataset="user/pick", episodes=[0, 1])
    with patch("makermodslab.models.is_dataset_private", return_value=True):
        info = get_model_info("curated_run")
    assert info["dataset_episodes"] is None


def test_get_model_info_local_hides_episodes_when_dataset_unresolvable(registry) -> None:
    """Fail closed: a dataset that can't be resolved (never pushed to the Hub,
    deleted, offline) is treated the same as an explicit private=True."""
    from makermodslab.models import get_model_info

    _seed_run_with_episodes(registry, "curated_run", dataset="user/pick", episodes=[0, 1])
    with patch("makermodslab.models.is_dataset_private", return_value=None):
        info = get_model_info("curated_run")
    assert info["dataset_episodes"] is None


def test_get_model_info_local_no_subset_skips_privacy_check(registry) -> None:
    """A run trained on every episode has nothing to gate, so the Hub privacy
    lookup — a real network call for an arbitrary dataset — never fires."""
    from makermodslab.models import get_model_info

    _seed_run(registry, "full_run", dataset="user/pick")
    with patch("makermodslab.models.is_dataset_private") as gate:
        info = get_model_info("full_run")
    assert info["dataset_episodes"] is None
    gate.assert_not_called()


# ---------------------------------------------------------------------------
# upload_local_model — tags via with_makermodslab_tag, create_repo/upload_folder mocked.
# ---------------------------------------------------------------------------


def test_upload_local_model_calls_hub_public_and_tagged(registry) -> None:
    from makermodslab.models import upload_local_model

    pretrained = _seed_run(registry, "up_run", policy_type="act", dataset="user/pick", steps=100)

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update") as mock_meta,
    ):
        result = upload_local_model("up_run")

    # create_repo: model repo, PUBLIC (private=False), exist_ok.
    _, ckw = fake_api.create_repo.call_args
    assert ckw["repo_type"] == "model"
    assert ckw["private"] is False
    assert ckw["exist_ok"] is True

    # upload_folder: the resolved final checkpoint dir, as a model repo.
    _, ukw = fake_api.upload_folder.call_args
    assert ukw["folder_path"] == str(pretrained)
    assert ukw["repo_type"] == "model"

    # tags run through with_makermodslab_tag (makermods / openbooth / MakerModsLab present).
    _, mkw = mock_meta.call_args
    assert mkw["repo_type"] == "model"
    assert mkw["overwrite"] is True
    tags = mock_meta.call_args.args[1]["tags"]
    assert {"makermods", "openbooth", "MakerModsLab"}.issubset(set(tags))
    assert set(result["tags"]) == set(tags)


def test_upload_local_model_rejects_offline(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "off_run", state="done")
    with patch("makermodslab.models.hf_hub_offline", return_value=True), pytest.raises(ModelError) as ei:
        upload_local_model("off_run")
    assert ei.value.status == 400


def test_upload_local_model_404_when_no_checkpoint(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "empty_run", state="done", with_checkpoint=False)
    with patch("makermodslab.models.hf_hub_offline", return_value=False), pytest.raises(ModelError) as ei:
        upload_local_model("empty_run")
    assert ei.value.status == 404


def test_upload_local_model_maps_auth_error(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "auth_run", state="done")
    fake_api = MagicMock()
    fake_api.create_repo.side_effect = Exception("401 Client Error: You must be authenticated")
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("auth_run")
    assert ei.value.status == 403


def test_upload_local_model_404_for_failed_run_with_checkpoint(registry) -> None:
    """A "failed" run is never publishable, even with a checkpoint on disk — its
    non-zero exit is a confirmed failure, so `_find_local_record` refuses it the
    same way `list_local_models` leaves it out of the listing."""
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "failed_run", state="failed", steps=50)
    with patch("makermodslab.models.hf_hub_offline", return_value=False), pytest.raises(ModelError) as ei:
        upload_local_model("failed_run")
    assert ei.value.status == 404


def test_upload_local_model_succeeds_for_interrupted_run_with_checkpoint(registry) -> None:
    from makermodslab.models import upload_local_model

    pretrained = _seed_run(registry, "interrupted_run", state="interrupted", steps=50)

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
    ):
        result = upload_local_model("interrupted_run")

    _, ukw = fake_api.upload_folder.call_args
    assert ukw["folder_path"] == str(pretrained)
    assert result["repo_id"]


def test_upload_local_model_404_when_still_running(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "running_run", state="running")
    with patch("makermodslab.models.hf_hub_offline", return_value=False), pytest.raises(ModelError) as ei:
        upload_local_model("running_run")
    assert ei.value.status == 404


def test_upload_local_model_404_for_failed_run_without_checkpoint(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "dead_run", state="failed", with_checkpoint=False)
    with patch("makermodslab.models.hf_hub_offline", return_value=False), pytest.raises(ModelError) as ei:
        upload_local_model("dead_run")
    assert ei.value.status == 404


def test_get_model_info_still_none_for_failed_run(registry) -> None:
    """A failed run is not a usable model on any path — info included."""
    from makermodslab.models import get_model_info

    _seed_run(registry, "failed_info_run", state="failed", steps=50)
    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        assert get_model_info("failed_info_run") is None


# ---------------------------------------------------------------------------
# upload_local_model — multi-checkpoint publishing into ONE repo.
# ---------------------------------------------------------------------------


def _publish(model_id: str, **kwargs):
    """Run a publish with the Hub fully mocked. Returns (result, fake_api)."""
    from makermodslab.models import upload_local_model

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
    ):
        return upload_local_model(model_id, **kwargs), fake_api


def _uploaded_paths(fake_api) -> list[str]:
    return [c.kwargs["path_in_repo"] for c in fake_api.upload_folder.call_args_list]


def test_legacy_root_layout_uploads_to_the_repo_root(registry) -> None:
    """POST /models/upload is frozen for SDK clients INCLUDING its on-Hub
    shape: the final checkpoint's files land at the repo root (loadable by a
    plain from_pretrained), no repo pin, and the return is exactly the
    pre-multi-checkpoint {repo_id, url, tags}."""
    from makermodslab.models import upload_local_model

    _seed_run(registry, "legacy_run", steps=300)
    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
    ):
        result = upload_local_model("legacy_run", root_layout=True)

    (call,) = fake_api.upload_folder.call_args_list
    assert "path_in_repo" not in call.kwargs
    assert sorted(result) == ["repo_id", "tags", "url"]
    # The legacy push never pins — the old route didn't, and a pin would flip
    # the training dialog to "published" for a legacy-shaped repo.
    assert registry.get("legacy_run").hf_repo_id is None


def test_published_repo_state_is_cached_for_the_picker(registry) -> None:
    """The picker probes the target repo on every dialog open; within the TTL
    that must cost ONE Hub roundtrip, and `fresh=True` (the post-upload reader)
    must bypass and refresh the cache."""
    from makermodslab.models import _published_repo_state

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = ["checkpoints/000300/pretrained_model/config.json"]
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        first = _published_repo_state("user/cached")
        second = _published_repo_state("user/cached")
        assert fake_api.list_repo_files.call_count == 1
        assert second == first
        _published_repo_state("user/cached", fresh=True)
        assert fake_api.list_repo_files.call_count == 2


def test_published_repo_state_does_not_cache_failures(registry) -> None:
    """An unreadable probe ("couldn't check") must retry on the next read
    rather than pinning the warning for the whole TTL."""
    from makermodslab.models import _published_repo_state

    fake_api = MagicMock()
    fake_api.list_repo_files.side_effect = RuntimeError("hub down")
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        assert _published_repo_state("user/flaky").readable is False
        assert _published_repo_state("user/flaky").readable is False
        assert fake_api.list_repo_files.call_count == 2


def test_publish_start_spawn_failure_lands_on_error_not_running(monkeypatch) -> None:
    """The wedge: state flipped to "running" before the thread spawned, so a
    spawn failure left the manager claiming a publish forever — every later
    publish 409ed (and the delete guard refused deletes) until restart. A spawn
    failure must free the slot as a visible "error"."""
    import makermodslab.models as m
    from makermodslab.models import ModelError, ModelUploadManager

    manager = ModelUploadManager()

    def _no_thread(*args, **kwargs):
        raise RuntimeError("no threads left")

    monkeypatch.setattr(m.threading, "Thread", _no_thread)
    with pytest.raises(ModelError) as ei:
        manager.start("run_w")
    assert ei.value.status == 500
    status = manager.get_status()
    assert status["state"] == "error"
    assert "no threads left" in (status["error"] or "")


def test_upload_defaults_to_final_checkpoint_only(registry, quiet_hub_reads) -> None:
    """No `steps` ⇒ the run's newest checkpoint, and nothing else — the
    pre-multi-checkpoint default an API caller can still rely on."""
    _seed_run(registry, "run_a", steps=300)
    _add_checkpoint(registry, "run_a", 100)
    _add_checkpoint(registry, "run_a", 200)

    result, fake_api = _publish("run_a")

    assert result["steps"] == [300]
    assert _uploaded_paths(fake_api) == ["checkpoints/300/pretrained_model"]


def test_upload_publishes_every_selected_step_into_one_repo(registry, quiet_hub_reads) -> None:
    """The headline behaviour: many checkpoints, ONE repo, one commit stream —
    each step step-addressed so none overwrites another."""
    _seed_run(registry, "run_b", steps=300)
    _add_checkpoint(registry, "run_b", 100)
    _add_checkpoint(registry, "run_b", 200)

    result, fake_api = _publish("run_b", steps=[100, 200, 300])

    assert result["steps"] == [100, 200, 300]
    assert _uploaded_paths(fake_api) == [
        "checkpoints/100/pretrained_model",
        "checkpoints/200/pretrained_model",
        "checkpoints/300/pretrained_model",
    ]
    # One repo, created once, for every step.
    assert fake_api.create_repo.call_count == 1
    targets = {c.kwargs["repo_id"] for c in fake_api.upload_folder.call_args_list}
    assert targets == {result["repo_id"]}


def test_upload_runs_steps_oldest_first_whatever_order_was_asked(registry, quiet_hub_reads) -> None:
    """Queue order is step order, so a queue that dies part-way leaves a
    contiguous published prefix rather than a hole in the middle."""
    _seed_run(registry, "run_c", steps=300)
    _add_checkpoint(registry, "run_c", 100)
    _add_checkpoint(registry, "run_c", 200)

    result, fake_api = _publish("run_c", steps=[300, 100, 200, 100])

    assert result["steps"] == [100, 200, 300]
    assert _uploaded_paths(fake_api) == [
        "checkpoints/100/pretrained_model",
        "checkpoints/200/pretrained_model",
        "checkpoints/300/pretrained_model",
    ]


def test_upload_preserves_zero_padded_checkpoint_dir_names(registry, quiet_hub_reads) -> None:
    """lerobot zero-pads checkpoint dirs and jobs._hub_checkpoints_from_files
    round-trips that padding into the ref it downloads by — so the repo path has
    to be the on-disk dir name, not str(step)."""
    _seed_run(registry, "run_pad", steps=100)
    record = registry._records["run_pad"]
    padded = Path(record.output_dir) / "checkpoints" / "000050" / "pretrained_model"
    padded.mkdir(parents=True)
    (padded / "config.json").write_text(json.dumps({"type": "act"}))

    _, fake_api = _publish("run_pad", steps=[50])

    assert _uploaded_paths(fake_api) == ["checkpoints/000050/pretrained_model"]


def test_upload_pins_the_repo_so_a_later_publish_reuses_it(registry, quiet_hub_reads) -> None:
    """A second publish must land in the SAME repo — the record is what carries
    that across calls, so the first upload writes it."""
    _seed_run(registry, "run_d", steps=300)
    _add_checkpoint(registry, "run_d", 100)

    first, _ = _publish("run_d", steps=[100], repo_id="user/pinned")
    assert registry._records["run_d"].hf_repo_id == "user/pinned"

    # No repo_id this time — it still goes to the pinned repo, not a new one.
    second, fake_api = _publish("run_d", steps=[300])
    assert second["repo_id"] == first["repo_id"] == "user/pinned"
    assert {c.kwargs["repo_id"] for c in fake_api.upload_folder.call_args_list} == {"user/pinned"}


def test_upload_reports_steps_already_on_the_hub(registry) -> None:
    """`published_steps` unions what was just pushed with what the repo already
    held, so the caller can render "5 of 8 published" after an incremental add."""
    from makermodslab.models import PublishedRepoState, upload_local_model

    _seed_run(registry, "run_e", steps=300)
    _add_checkpoint(registry, "run_e", 100)

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
        patch(
            "makermodslab.models._published_repo_state",
            return_value=PublishedRepoState({100: "100"}, False, True),
        ),
        patch("makermodslab.models._sync_model_card"),
    ):
        result = upload_local_model("run_e", steps=[300])

    assert result["steps"] == [300]
    assert result["published_steps"] == [100, 300]


def test_upload_skips_the_card_when_the_repo_cant_be_read(registry) -> None:
    """An unreadable probe leaves `published` holding only THIS call's steps.
    Rewriting the index from that would delete the rows of steps that ARE
    published, so the card is left stale instead."""
    from makermodslab.models import PublishedRepoState, upload_local_model

    _seed_run(registry, "run_unreadable", steps=300)
    _add_checkpoint(registry, "run_unreadable", 100)

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
        patch(
            "makermodslab.models._published_repo_state",
            return_value=PublishedRepoState({}, False, False),
        ),
        patch("makermodslab.models._sync_model_card") as card,
    ):
        upload_local_model("run_unreadable", steps=[300])

    card.assert_not_called()


def test_upload_reports_progress_per_step(registry, quiet_hub_reads) -> None:
    """on_progress fires BEFORE each step, so `done` is the count already on the
    Hub — the invariant the manager's queue position and its partial-failure
    `done_steps` slice both rest on."""
    from makermodslab.models import upload_local_model

    _seed_run(registry, "run_f", steps=300)
    _add_checkpoint(registry, "run_f", 100)
    _add_checkpoint(registry, "run_f", 200)

    seen: list[tuple[int, int, int | None]] = []
    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
    ):
        upload_local_model(
            "run_f",
            steps=[100, 200, 300],
            on_progress=lambda done, total, step: seen.append((done, total, step)),
        )

    # The trailing (total, total, None) fires once every upload has landed and
    # BEFORE the card/tag work, so a failure there can't be read as a lost step.
    assert seen == [(0, 3, 100), (1, 3, 200), (2, 3, 300), (3, 3, None)]


def test_upload_signals_all_landed_before_the_tagging_that_fails(registry, quiet_hub_reads) -> None:
    """Regression: metadata_update raising AFTER every upload succeeded must not
    look like the last checkpoint never made it."""
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "run_tagfail", steps=300)
    _add_checkpoint(registry, "run_tagfail", 100)

    seen: list[tuple[int, int, int | None]] = []
    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update", side_effect=RuntimeError("hub 503")),
        pytest.raises(ModelError),
    ):
        upload_local_model(
            "run_tagfail",
            steps=[100, 300],
            on_progress=lambda done, total, step: seen.append((done, total, step)),
        )

    assert seen[-1] == (2, 2, None)


def test_publish_manager_keeps_every_landed_step_when_the_tagging_fails(registry, quiet_hub_reads) -> None:
    """The manager's partial-failure account, end to end: a failure after the
    last upload reports ALL steps as published, not one fewer. Getting this
    wrong sends the user to re-upload weights that are already on the Hub — and
    for a single-checkpoint publish it claimed nothing had been published."""
    import makermodslab.models as m

    _seed_run(registry, "run_mgr", steps=300)
    _add_checkpoint(registry, "run_mgr", 100)

    manager = m.ModelUploadManager()
    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update", side_effect=RuntimeError("hub 503")),
    ):
        manager.start("run_mgr", None, [100, 300])
        manager._thread.join(timeout=10)

    status = manager.get_status()
    assert status["state"] == "error"
    assert status["done_steps"] == [100, 300]
    assert status["done"] == 2


def test_publish_manager_drops_the_step_that_failed_mid_queue(registry, quiet_hub_reads) -> None:
    """The other half of the contract: a failure DURING an upload reports only
    the steps before it, since the one in flight never landed."""
    import makermodslab.models as m

    _seed_run(registry, "run_mid", steps=300)
    _add_checkpoint(registry, "run_mid", 100)
    _add_checkpoint(registry, "run_mid", 200)

    manager = m.ModelUploadManager()
    fake_api = MagicMock()
    # Succeed for step 100, blow up on 200.
    fake_api.upload_folder.side_effect = [None, RuntimeError("connection reset")]
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update"),
    ):
        manager.start("run_mid", None, [100, 200, 300])
        manager._thread.join(timeout=10)

    status = manager.get_status()
    assert status["state"] == "error"
    assert status["done_steps"] == [100]


def test_upload_rejects_a_step_the_run_never_saved(registry) -> None:
    """Silently dropping an unknown step would report success for an upload that
    never happened."""
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "run_g", steps=300)
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("run_g", steps=[999])
    assert ei.value.status == 404
    assert "999" in ei.value.message


def test_upload_rejects_an_empty_selection(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_run(registry, "run_h", steps=300)
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("run_h", steps=[])
    assert ei.value.status == 400


# ---------------------------------------------------------------------------
# list_run_checkpoints — what the publish picker reads.
# ---------------------------------------------------------------------------


def test_list_run_checkpoints_marks_the_steps_already_published(registry) -> None:
    from makermodslab.models import PublishedRepoState, list_run_checkpoints

    _seed_run(registry, "run_i", steps=300, hf_repo_id="user/pinned")
    _add_checkpoint(registry, "run_i", 100)
    _add_checkpoint(registry, "run_i", 200)

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch(
            "makermodslab.models._published_repo_state",
            return_value=PublishedRepoState({100: "100", 200: "200"}, False, True),
        ),
    ):
        out = list_run_checkpoints("run_i")

    assert out["hf_repo_id"] == "user/pinned"
    assert out["default_repo_id"] == "user/pinned"
    assert out["hub_readable"] is True
    assert [(c["step"], c["published"]) for c in out["checkpoints"]] == [
        (100, True),
        (200, True),
        (300, False),
    ]


def test_list_run_checkpoints_skips_the_hub_probe_when_offline(registry) -> None:
    """Offline is a normal state for this app — the picker still renders, it
    just can't know what's published, and false-when-unknown only ever costs a
    redundant re-upload."""
    from makermodslab.models import list_run_checkpoints

    _seed_run(registry, "run_j", steps=300)
    probe = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=True),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models._published_repo_state", probe),
    ):
        out = list_run_checkpoints("run_j")

    probe.assert_not_called()
    assert [(c["step"], c["published"]) for c in out["checkpoints"]] == [(300, False)]
    # The UI must be able to tell "not published" from "couldn't check".
    assert out["hub_readable"] is False


def test_published_run_row_still_reports_its_local_side_as_a_run(registry) -> None:
    """Regression: pinning hf_repo_id collapses a published run onto its own Hub
    repo as source="both". The row must keep local_kind="run", because the
    frontend's delete confirmation reads it to decide whether removing the local
    files is recoverable — for a run it isn't: only the published subset of its
    checkpoints exists anywhere else."""
    from makermodslab.models import list_all_models

    _seed_run(registry, "run_p", steps=300, hf_repo_id="user/run_p")

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        # list_hub_models returns the /jobs hub-row shape, keyed on repo_id.
        patch(
            "makermodslab.models.list_hub_models",
            return_value=[{"repo_id": "user/run_p", "last_modified": None}],
        ),
        patch("makermodslab.models.list_downloaded_models", return_value=[]),
    ):
        rows = list_all_models()

    row = next(r for r in rows if r["id"] == "run_p")
    assert row["source"] == "both"
    assert row["local_kind"] == "run"


def test_run_collapse_matches_hub_repo_case_insensitively(registry) -> None:
    """A user-typed repo id differing in case from the Hub's canonical listing
    must still collapse to ONE "both" row — the same fold every jobs-side dedup
    (find_imported, the frontend's trackedRepoIds) applies. Byte-exact matching
    left the run and its own repo as two rows."""
    from makermodslab.models import list_all_models

    _seed_run(registry, "run_c", steps=300, hf_repo_id="User/Run_C")

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch(
            "makermodslab.models.list_hub_models",
            return_value=[{"repo_id": "user/run_c", "last_modified": None}],
        ),
        patch("makermodslab.models.list_downloaded_models", return_value=[]),
    ):
        rows = list_all_models()

    matches = [r for r in rows if r["id"] == "run_c" or r.get("hf_repo_id", "").lower() == "user/run_c"]
    assert len(matches) == 1
    assert matches[0]["source"] == "both"
    assert matches[0]["local_kind"] == "run"


def test_run_collapse_keeps_dataset_episodes(registry) -> None:
    """The both-collapse copies the checkpoint-derived fields onto the
    hub-seeded row; dataset_episodes was missing from that tuple, so publishing
    a run silently dropped its episode subset from the listing."""
    from makermodslab.models import list_all_models

    _seed_run_with_episodes(registry, "run_ep", dataset="user/pick", episodes=[2, 1])
    registry._records["run_ep"].hf_repo_id = "user/run_ep"

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.is_dataset_private", return_value=False),
        patch(
            "makermodslab.models.list_hub_models",
            return_value=[{"repo_id": "user/run_ep", "last_modified": None}],
        ),
        patch("makermodslab.models.list_downloaded_models", return_value=[]),
    ):
        rows = list_all_models()

    row = next(r for r in rows if r["id"] == "run_ep")
    assert row["source"] == "both"
    assert row["dataset_episodes"] == [1, 2]


def test_downloaded_model_row_reports_its_local_side_as_a_copy(tmp_lerobot_home: Path) -> None:
    """The other side of the same signal: a downloaded copy IS replaceable from
    the Hub, so its row keeps the two-press "remove local copy" semantics."""
    from makermodslab.models import _local_models_root, list_downloaded_models

    _make_model_checkpoint(_local_models_root(), "user/policy")
    rows = list_downloaded_models()

    assert [r["local_kind"] for r in rows] == ["downloaded"]


def test_list_run_checkpoints_404_without_a_checkpoint(registry) -> None:
    from makermodslab.models import ModelError, list_run_checkpoints

    _seed_run(registry, "run_k", state="failed", with_checkpoint=False)
    with pytest.raises(ModelError) as ei:
        list_run_checkpoints("run_k")
    assert ei.value.status == 404


# ---------------------------------------------------------------------------
# The model card's checkpoint index.
# ---------------------------------------------------------------------------


def test_model_card_index_lists_every_published_step(registry) -> None:
    from makermodslab.models import _render_checkpoint_index

    _seed_run(registry, "run_l", steps=300)
    body = _render_checkpoint_index(registry._records["run_l"], {100: "100", 200: "200", 300: "300"})

    for step in (100, 200, 300):
        assert f"`checkpoints/{step}/pretrained_model`" in body


def test_model_card_index_prints_the_real_zero_padded_paths(registry) -> None:
    """Regression: the index exists to be copied, and lerobot writes
    checkpoints/000050 — a row saying checkpoints/50 is a 404 for anyone who
    trusts it."""
    from makermodslab.models import _render_checkpoint_index

    _seed_run(registry, "run_padcard", steps=300)
    body = _render_checkpoint_index(registry._records["run_padcard"], {50: "000050"})

    assert "`checkpoints/000050/pretrained_model`" in body
    assert "checkpoints/50/" not in body
    # The step column stays human-readable.
    assert "| 50 |" in body


def test_published_repo_state_carries_the_padded_dir_names(registry) -> None:
    """The padding comes back off the Hub in the checkpoint ref, and has to
    survive into the card index."""
    from makermodslab.models import _published_repo_state

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [
        "checkpoints/000050/pretrained_model/config.json",
        "checkpoints/000300/pretrained_model/config.json",
    ]
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        state = _published_repo_state("user/repo")

    assert state.steps == {50: "000050", 300: "000300"}
    assert state.readable is True
    assert state.has_legacy_root is False


def test_published_repo_state_reports_unreadable_rather_than_empty(registry) -> None:
    from makermodslab.models import _published_repo_state

    fake_api = MagicMock()
    fake_api.list_repo_files.side_effect = RuntimeError("network down")
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        state = _published_repo_state("user/repo")

    assert state.steps == {}
    assert state.readable is False


def test_model_card_refresh_preserves_the_rest_of_the_card(registry) -> None:
    """Only the marked block is rewritten, so a hand-edited card (and its YAML
    frontmatter) survives every re-publish."""
    from makermodslab.models import _CARD_MARK_END, _CARD_MARK_START, _sync_model_card

    _seed_run(registry, "run_m", steps=300)
    existing = (
        "---\ntags:\n- lerobot\n---\n\n"
        "# My model\n\nHand-written notes.\n\n"
        f"{_CARD_MARK_START}\nstale index\n{_CARD_MARK_END}\n\n"
        "## Citation\n\nCite me.\n"
    )

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.Path") as fake_path,
        patch("huggingface_hub.hf_hub_download", return_value="/tmp/README.md"),
    ):
        fake_path.return_value.read_text.return_value = existing
        _sync_model_card(fake_api, "user/repo", registry._records["run_m"], {100: "100", 300: "300"})

    body = fake_api.upload_file.call_args.kwargs["path_or_fileobj"].decode()
    assert "tags:\n- lerobot" in body
    assert "Hand-written notes." in body
    assert "Cite me." in body
    assert "stale index" not in body
    assert "`checkpoints/300/pretrained_model`" in body


def test_model_card_refresh_leaves_an_unreadable_card_alone(registry) -> None:
    """A network blip must not be mistaken for "no card yet" — that would
    overwrite the user's prose with a bare index, which is exactly what the
    marker mechanism exists to prevent."""
    from makermodslab.models import _sync_model_card

    _seed_run(registry, "run_n", steps=300)
    fake_api = MagicMock()
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("hub 500")):
        _sync_model_card(fake_api, "user/repo", registry._records["run_n"], {300: "300"})

    fake_api.upload_file.assert_not_called()


def test_model_card_is_written_from_scratch_when_absent(registry) -> None:
    """The first-publish case: a genuinely missing README is the one failure
    that DOES mean "write one"."""
    from huggingface_hub.errors import EntryNotFoundError

    from makermodslab.models import _sync_model_card

    _seed_run(registry, "run_o", steps=300)
    fake_api = MagicMock()
    with patch("huggingface_hub.hf_hub_download", side_effect=EntryNotFoundError("nope")):
        _sync_model_card(fake_api, "user/repo", registry._records["run_o"], {300: "300"})

    body = fake_api.upload_file.call_args.kwargs["path_or_fileobj"].decode()
    assert "`checkpoints/300/pretrained_model`" in body


# ---------------------------------------------------------------------------
# delete_local_model — sandboxed under outputs/train/.
# ---------------------------------------------------------------------------


def test_delete_local_model_removes_run_dir(registry) -> None:
    from makermodslab.models import delete_local_model

    _seed_run(registry, "del_run", state="done")
    run_root = registry._output_root / "del_run"
    assert run_root.exists()

    result = delete_local_model("del_run")
    assert result["deleted"] is True
    assert not run_root.exists()
    assert "del_run" not in registry._records


def test_delete_local_model_404_unknown(registry) -> None:
    from makermodslab.models import ModelError, delete_local_model

    with pytest.raises(ModelError) as ei:
        delete_local_model("ghost")
    assert ei.value.status == 404


def test_delete_local_model_409_when_running(registry) -> None:
    from makermodslab.models import ModelError, delete_local_model

    _seed_run(registry, "live_run", state="running")
    with pytest.raises(ModelError) as ei:
        delete_local_model("live_run")
    assert ei.value.status == 409
    # The dir must still be there — a running job is never deleted.
    assert (registry._output_root / "live_run").exists()


def test_delete_local_model_409_when_resumed_by_another_run(registry) -> None:
    """A MID-CHAIN delete reached from the model library is refused the same way
    the /jobs route refuses it: 409, naming the continuation. Without its own
    handler JobHasChildrenError fell into delete_local_model's catch-all and
    surfaced as a 502 "Failed to delete model", which reads as a bug in the
    app rather than as the deliberate guard it is."""
    from makermodslab.jobs import JobRecord
    from makermodslab.models import ModelError, delete_local_model
    from makermodslab.train import TrainingRequest

    _seed_run(registry, "parent_run", state="interrupted")
    registry._records["child_run"] = JobRecord(
        id="child_run",
        name="child of parent_run",
        state="done",
        config=TrainingRequest(
            dataset_repo_id="user/pick",
            resume=True,
            resume_from_job_id="parent_run",
        ),
        output_dir=str(registry._output_root / "child_run" / "run"),
        started_at=2.0,
        ended_at=3.0,
        runner="local",
    )

    with pytest.raises(ModelError) as ei:
        delete_local_model("parent_run")
    assert ei.value.status == 409
    assert "child_run" in ei.value.message
    # The parent's dir — including the checkpoint the child resumed out of —
    # must survive the refusal.
    assert (registry._output_root / "parent_run").exists()
    assert "parent_run" in registry._records


def test_delete_local_model_refuses_path_outside_output_root(registry) -> None:
    """A record whose id resolves OUTSIDE outputs/train (traversal) is refused;
    no rmtree runs, so nothing outside the sandbox is touched."""
    from makermodslab.jobs import JobRecord
    from makermodslab.models import ModelError, delete_local_model
    from makermodslab.train import TrainingRequest

    # An id containing '..' would resolve <root>/../evil, escaping the root.
    evil_id = "../evil"
    registry._records[evil_id] = JobRecord(
        id=evil_id,
        name="evil",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/x"),
        output_dir=str(registry._output_root / evil_id / "run"),
        started_at=1.0,
        ended_at=2.0,
        runner="local",
    )

    with patch("makermodslab.jobs.shutil.rmtree") as mock_rmtree, pytest.raises(ModelError) as ei:
        delete_local_model(evil_id)
    assert ei.value.status == 400
    mock_rmtree.assert_not_called()  # nothing was deleted


def test_delete_local_model_400_non_local(registry) -> None:
    from makermodslab.models import ModelError, delete_local_model

    _seed_run(registry, "cloud_del", state="done", runner="hf_cloud")
    with pytest.raises(ModelError) as ei:
        delete_local_model("cloud_del")
    assert ei.value.status == 400


# ---------------------------------------------------------------------------
# Endpoint wiring (server routes) — HF + registry mocked.
# ---------------------------------------------------------------------------


def test_models_endpoint_returns_listing(client, registry) -> None:
    with patch("makermodslab.models.list_hub_models", return_value=[]):
        _seed_run(registry, "ep_run", state="done", dataset="user/pick")
        resp = client.get("/models")
    assert resp.status_code == 200
    ids = {m.get("id") for m in resp.json()}
    assert "ep_run" in ids


def test_models_info_404(client, registry) -> None:
    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        resp = client.get("/models/info", params={"id": "missing"})
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_models_delete_endpoint(client, registry) -> None:
    _seed_run(registry, "ep_del", state="done")
    resp = client.post("/models/delete", json={"id": "ep_del"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Downloaded / imported local models — the local models dir scan + probe.
# ---------------------------------------------------------------------------


def _make_model_checkpoint(
    root: Path, repo_id: str, shape: str = "root", step: int = 500, policy_type: str = "act"
) -> Path:
    """Fabricate a checkpoint dir in one of the two recognized shapes: a root
    config.json ("root", the flat layout uploads used before they became
    step-addressed, and what a foreign Hub repo usually looks like) or a
    checkpoints/<step>/pretrained_model tree ("tree", what upload_local_model
    pushes now).

    Both shapes carry `model.safetensors`: `_resolve_pretrained_dir` requires the
    weights lerobot actually loads, so a config-only tree is deliberately NOT a
    usable checkpoint (it is what an interrupted download leaves behind)."""
    d = root / repo_id
    if shape == "root":
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"type": policy_type}))
        (d / "model.safetensors").write_text("weights")
    else:
        p = d / "checkpoints" / str(step) / "pretrained_model"
        p.mkdir(parents=True)
        (p / "config.json").write_text(json.dumps({"type": policy_type}))
        (p / "model.safetensors").write_text("weights")
    return d


def test_local_models_root_migrates_pre_rebrand_cache(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import _local_models_root

    legacy_root = tmp_lerobot_home / "lelab_models"
    checkpoint = _make_model_checkpoint(legacy_root, "user/policy")

    root = _local_models_root()

    assert root == tmp_lerobot_home / "makermodslab_models"
    assert not legacy_root.exists()
    assert (root / checkpoint.relative_to(legacy_root)).is_dir()


def test_list_downloaded_models_root_shape(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import list_downloaded_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    rows = list_downloaded_models()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "user/policy"
    assert row["policy_type"] == "act"
    assert row["source"] == "local"
    # Root shape: the dir itself is the pretrained_model.
    assert row["path"] == str((tmp_lerobot_home / "makermodslab_models" / "user" / "policy").resolve())


def test_list_downloaded_models_tree_shape_reports_final_step(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import list_downloaded_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "runrepo", shape="tree", step=750)
    rows = list_downloaded_models()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "runrepo"
    assert row["steps"] == 750
    assert row["path"].endswith("checkpoints/750/pretrained_model")


def test_list_downloaded_models_skips_non_checkpoint_dirs(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import list_downloaded_models

    (tmp_lerobot_home / "makermodslab_models" / "junk" / "not_a_model").mkdir(parents=True)
    assert list_downloaded_models() == []


def test_is_model_available_locally(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import is_model_available_locally

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    assert is_model_available_locally("user/policy")
    assert not is_model_available_locally("user/other")


def test_is_model_available_locally_rejects_traversal(tmp_lerobot_home: Path) -> None:
    """A repo_id escaping the models root (e.g. a dataset dir one level up) is
    refused even if the target exists."""
    from makermodslab.models import is_model_available_locally

    outside = tmp_lerobot_home / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    assert not is_model_available_locally("../outside")


def test_list_all_models_downloaded_flips_hub_to_both(registry, tmp_lerobot_home: Path) -> None:
    """A hub repo whose checkpoint was downloaded into the local models dir is
    collapsed to one 'both' row carrying the local path — the listing flip that
    makes 'download to local' visible."""
    from makermodslab.models import list_all_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    hub_rows = [
        {"repo_id": "user/policy", "last_modified": "2026-02-01T00:00:00+00:00", "private": False},
    ]
    with patch("makermodslab.models.list_hub_models", return_value=hub_rows):
        result = list_all_models()

    assert len(result) == 1
    row = result[0]
    assert row["source"] == "both"
    assert row["hf_repo_id"] == "user/policy"
    assert row["path"] is not None  # local checkpoint detail filled in


def test_list_all_models_downloaded_only_is_local(registry, tmp_lerobot_home: Path) -> None:
    from makermodslab.models import list_all_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "imported_policy")
    with patch("makermodslab.models.list_hub_models", return_value=[]):
        result = list_all_models()
    assert [m["id"] for m in result] == ["imported_policy"]
    assert result[0]["source"] == "local"


def test_get_model_info_downloaded_checkpoint(registry, tmp_lerobot_home: Path) -> None:
    """A downloaded/imported checkpoint resolves in get_model_info without the
    Hub (works offline) and reports its on-disk size."""
    from makermodslab.models import get_model_info

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        info = get_model_info("user/policy")
    assert info is not None
    assert info["policy_type"] == "act"
    assert info["size_bytes"] > 0
    assert info["source"] == "local"


# ---------------------------------------------------------------------------
# Saved custom models — pin/unpin persistence + listing fold + routes.
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SAVED_CUSTOM_MODELS_FILE into a tmp file so pin tests never
    touch the developer's real ~/.cache."""
    from makermodslab.utils import config as cfg

    path = tmp_path / "saved_custom_models.json"
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_MODELS_FILE", str(path))
    return path


def test_saved_custom_models_round_trip(custom_models_file: Path) -> None:
    from makermodslab.utils.config import (
        add_saved_custom_model,
        get_saved_custom_models,
        remove_saved_custom_model,
    )

    assert get_saved_custom_models() == []
    assert add_saved_custom_model("lerobot/smolvla_base")
    assert add_saved_custom_model("user/act_model")
    # Re-saving moves it to the front (most-recently-used first).
    assert add_saved_custom_model("lerobot/smolvla_base")
    assert get_saved_custom_models() == ["lerobot/smolvla_base", "user/act_model"]

    assert remove_saved_custom_model("user/act_model")
    assert not remove_saved_custom_model("user/act_model")  # already gone
    assert get_saved_custom_models() == ["lerobot/smolvla_base"]
    assert not add_saved_custom_model("")  # blank refused


def test_list_all_models_includes_pinned_custom(registry, tmp_lerobot_home) -> None:
    from makermodslab.models import list_all_models

    with (
        patch("makermodslab.models.list_hub_models", return_value=[]),
        patch("makermodslab.models.get_saved_custom_models", return_value=["lerobot/smolvla_base"]),
    ):
        result = list_all_models()

    assert len(result) == 1
    row = result[0]
    assert row["id"] == "lerobot/smolvla_base"
    assert row["source"] == "hub"
    assert row["saved_custom"] is True
    assert row["hf_repo_id"] == "lerobot/smolvla_base"


def test_list_all_models_pinned_and_downloaded_is_both(registry, tmp_lerobot_home: Path) -> None:
    """A pinned foreign repo whose checkpoint was downloaded flips to 'both'
    (Hub + local copy) and keeps saved_custom so unpin stays available."""
    from makermodslab.models import list_all_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "lerobot/smolvla_base")
    with (
        patch("makermodslab.models.list_hub_models", return_value=[]),
        patch("makermodslab.models.get_saved_custom_models", return_value=["lerobot/smolvla_base"]),
    ):
        result = list_all_models()

    assert len(result) == 1
    row = result[0]
    assert row["source"] == "both"
    assert row["saved_custom"] is True
    assert row["hf_repo_id"] == "lerobot/smolvla_base"
    assert row["path"] is not None


def test_models_custom_endpoints_round_trip(client, custom_models_file: Path) -> None:
    resp = client.post("/models/custom", json={"repo_id": "lerobot/smolvla_base"})
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "lerobot/smolvla_base"}

    from makermodslab.utils.config import get_saved_custom_models

    assert get_saved_custom_models() == ["lerobot/smolvla_base"]

    resp = client.request("DELETE", "/models/custom", json={"repo_id": "lerobot/smolvla_base"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert get_saved_custom_models() == []


def test_models_custom_endpoint_rejects_bad_repo_id(client, custom_models_file: Path) -> None:
    resp = client.post("/models/custom", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


# ---------------------------------------------------------------------------
# Model download — the models twin of the dataset DownloadManager.
# ---------------------------------------------------------------------------


def _model_download_manager():
    """A fresh DownloadManager wired with the model fetch/cleanup callables —
    same wiring as the module singleton, clean state per test."""
    import makermodslab.models as m
    from makermodslab.datasets import DownloadManager

    return DownloadManager(m._fetch_model_snapshot, m._cleanup_partial_model)


def _join_download(mgr, timeout: float = 5.0) -> None:
    thread = mgr._thread
    if thread is not None:
        thread.join(timeout=timeout)


def test_model_download_manager_completes_and_lands_locally(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import makermodslab.models as m

    def _fake_snapshot(repo_id, repo_type, local_dir, ignore_patterns=None):  # noqa: ARG001
        d = Path(local_dir)
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"type": "act"}))
        (d / "model.safetensors").write_text("weights")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    result = mgr.start("user/policy")
    assert result["started"] is True

    _join_download(mgr)
    status = mgr.get_status()
    assert status["state"] == "done"
    assert status["error"] is None
    assert m.is_model_available_locally("user/policy")


def test_model_download_manager_rejects_non_policy_repo(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo that downloads fine but has no config.json / checkpoints tree is
    not a policy — the fetch errors and the partial dir is cleaned up."""
    import makermodslab.models as m

    def _fake_snapshot(repo_id, repo_type, local_dir, ignore_patterns=None):  # noqa: ARG001
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "README.md").write_text("not a model")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/notapolicy")
    _join_download(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "doesn't look like a policy checkpoint" in status["message"]
    assert not (tmp_lerobot_home / "makermodslab_models" / "user" / "notapolicy").exists()


def test_model_download_skips_training_state_but_keeps_checkpoints(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Models-page download must not pull optimizer/scheduler state.

    `training_state/` exists only to RESUME training — nothing that reads a
    downloaded model here ever opens it, and it is hundreds of MB per checkpoint
    step (a real 5.6 GB local model dir was ~3.5 GB of it). The
    `checkpoints/<step>/pretrained_model` trees must survive: the checkpoint
    browser lists them and inference selects individual steps from them.

    The fake snapshot runs the captured ignore_patterns through huggingface_hub's
    OWN filter, so this asserts the real matching semantics (fnmatch, not
    path-aware globbing) rather than a hand-rolled guess — and it does it with no
    network."""
    from huggingface_hub.utils import filter_repo_objects

    import makermodslab.models as m

    repo_files = [
        "README.md",
        "config.json",
        "model.safetensors",
        "train_config.json",
        "training_state/optimizer_state.safetensors",
        "checkpoints/000050/pretrained_model/config.json",
        "checkpoints/000050/pretrained_model/model.safetensors",
        "checkpoints/000050/training_state/optimizer_state.safetensors",
        "checkpoints/last/training_state/scheduler_state.json",
    ]
    seen: dict = {}

    def _fake_snapshot(repo_id, repo_type, local_dir, ignore_patterns=None):  # noqa: ARG001
        seen["ignore_patterns"] = ignore_patterns
        for rel in filter_repo_objects(repo_files, ignore_patterns=ignore_patterns):
            path = Path(local_dir) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"type": "act"}) if rel.endswith(".json") else "weights")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/policy")
    _join_download(mgr)
    assert mgr.get_status()["state"] == "done"

    assert seen["ignore_patterns"] == ["training_state/**", "*/training_state/**"]
    target = tmp_lerobot_home / "makermodslab_models" / "user" / "policy"
    # Both the root-level and the per-checkpoint optimizer state are gone...
    assert not (target / "training_state").exists()
    assert not (target / "checkpoints" / "000050" / "training_state").exists()
    assert not (target / "checkpoints" / "last").exists()
    # ...while everything a downloaded model is actually read for survives.
    assert (target / "config.json").is_file()
    assert (target / "model.safetensors").is_file()
    assert (target / "checkpoints" / "000050" / "pretrained_model" / "model.safetensors").is_file()
    # And the trimmed tree still resolves the way the listing/inference expect.
    assert (
        m._resolve_pretrained_dir(target)
        == (target / "checkpoints" / "000050" / "pretrained_model").resolve()
    )
    assert m.is_model_available_locally("user/policy")


# ---------------------------------------------------------------------------
# Models-page download → served from the shared HF hub cache (design-debt F6,
# the other direction).
#
# `_fetch_model_snapshot` downloads with local_dir=, and huggingface_hub 1.21.0's
# local_dir mode neither reads nor populates the shared cache — so a repo
# inference had already cached was pulled over the network a SECOND time. The
# fetch now goes through the cache when it holds the repo. No network anywhere:
# snapshot_download is monkeypatched, and the "cache" is a tmp dir built to the
# real on-disk layout (blobs/ + a snapshots/<rev>/ symlink farm).
# ---------------------------------------------------------------------------


def _seed_hub_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cached_repo: str | None = None,
    with_snapshot: bool = True,
) -> Path:
    """Redirect HF_HUB_CACHE at a tmp dir, optionally holding one model repo.

    Returns the cache root. Deliberately redirected in every test here rather
    than relying on the developer's real cache being empty of the fake repo id."""
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.file_download import repo_folder_name

    cache = tmp_path / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    if cached_repo is not None:
        snapshots = cache / repo_folder_name(repo_id=cached_repo, repo_type="model") / "snapshots"
        snapshots.mkdir(parents=True)
        if with_snapshot:
            (snapshots / "deadbeef").mkdir()
    return cache


def _build_cache_snapshot(cache: Path, repo_id: str) -> Path:
    """Write a realistic cache snapshot for `repo_id`: a blobs/ dir plus a
    snapshots/<rev>/ tree whose entries are SYMLINKS into it, which is what
    huggingface_hub actually leaves on disk. Includes a training_state file
    (a cache entry can predate our exclusion) and a nested checkpoint tree."""
    from huggingface_hub.file_download import repo_folder_name

    repo_root = cache / repo_folder_name(repo_id=repo_id, repo_type="model")
    blobs = repo_root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot = repo_root / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True, exist_ok=True)

    contents = {
        "config.json": json.dumps({"type": "act"}),
        "model.safetensors": "root-weights",
        "training_state/optimizer_state.safetensors": "optimizer-junk",
        "checkpoints/000050/pretrained_model/config.json": json.dumps({"type": "act"}),
        "checkpoints/000050/pretrained_model/model.safetensors": "step-weights",
        "checkpoints/000050/training_state/optimizer_state.safetensors": "optimizer-junk",
    }
    for i, (rel, text) in enumerate(contents.items()):
        blob = blobs / f"blob{i}"
        blob.write_text(text)
        link = snapshot / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        # Relative link, exactly as huggingface_hub writes them.
        link.symlink_to(Path(os.path.relpath(blob, link.parent)))
    return snapshot


def test_model_download_serves_a_cached_repo_without_re_downloading(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo is already in the shared hub cache, so the Models-page download
    must NOT pull it over the network a second time (design-debt F6).

    Asserts the whole contract of the cache path at once: no local_dir download
    happens, the symlink farm lands DEREFERENCED (the store must not point into
    blobs/, which huggingface_hub may garbage-collect), training_state is still
    excluded even though the cache entry has it, and the download manager still
    reaches "done" normally."""
    import makermodslab.models as m

    cache = _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/policy")
    snapshot = _build_cache_snapshot(cache, "user/policy")
    calls: list[dict] = []

    def _fake_snapshot(repo_id, repo_type=None, local_dir=None, ignore_patterns=None):
        calls.append({"repo_id": repo_id, "local_dir": local_dir, "ignore_patterns": ignore_patterns})
        if local_dir is not None:
            raise AssertionError("a cached repo must not be re-downloaded with local_dir=")
        return str(snapshot)

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/policy")
    _join_download(mgr)

    assert mgr.get_status()["state"] == "done"
    # Cache mode: no local_dir, and the exclusion is still requested so missing
    # files aren't fetched as optimizer state.
    assert len(calls) == 1
    assert calls[0]["local_dir"] is None
    assert calls[0]["ignore_patterns"] == ["training_state/**", "*/training_state/**"]

    target = tmp_lerobot_home / "makermodslab_models" / "user" / "policy"
    weights = target / "model.safetensors"
    assert weights.is_file()
    assert not weights.is_symlink()
    assert weights.read_text() == "root-weights"
    nested = target / "checkpoints" / "000050" / "pretrained_model" / "model.safetensors"
    assert nested.is_file()
    assert not nested.is_symlink()
    assert nested.read_text() == "step-weights"
    # A cache entry predating the exclusion still must not leak optimizer state.
    assert not (target / "training_state").exists()
    assert not (target / "checkpoints" / "000050" / "training_state").exists()
    # And the copied tree passes the same validation the download path does.
    assert (
        m._resolve_pretrained_dir(target)
        == (target / "checkpoints" / "000050" / "pretrained_model").resolve()
    )
    assert m.is_model_available_locally("user/policy")


def test_model_download_uses_the_network_when_the_cache_is_empty(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing cached for the repo → the plain local_dir download, unchanged.
    The cache path is an optimization, not a precondition."""
    import makermodslab.models as m

    _seed_hub_cache(monkeypatch, tmp_path)
    calls: list[dict] = []

    def _fake_snapshot(repo_id, repo_type=None, local_dir=None, ignore_patterns=None):
        calls.append({"local_dir": local_dir})
        assert local_dir is not None, "an uncached repo has nothing to serve from the cache"
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"type": "act"}))
        (d / "model.safetensors").write_text("weights")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/policy")
    _join_download(mgr)

    assert mgr.get_status()["state"] == "done"
    assert len(calls) == 1 and calls[0]["local_dir"] is not None
    assert m.is_model_available_locally("user/policy")


def test_hub_cache_has_repo_requires_an_actual_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both sides of the rule: a repo dir whose snapshots/ is empty (an
    interrupted or wiped entry) has nothing to dedupe against and counts as
    ABSENT, so the caller downloads instead of copying a broken tree."""
    import makermodslab.models as m

    _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/policy")
    assert m._hub_cache_has_repo("user/policy") is True
    assert m._hub_cache_has_repo("user/other") is False

    _seed_hub_cache(monkeypatch, tmp_path / "b", cached_repo="user/policy", with_snapshot=False)
    assert m._hub_cache_has_repo("user/policy") is False
    # A repo id the hub's own validation rejects answers "not cached" rather
    # than raising — snapshot_download then produces the canonical error.
    assert m._hub_cache_has_repo("../../etc") is False


def test_model_download_falls_back_when_the_cache_fetch_errors(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache path must never become a NEW failure mode: an HF error on the
    cache-mode call falls back to the plain local_dir download and the download
    still succeeds."""
    import makermodslab.models as m

    _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/policy")
    calls: list[dict] = []

    def _fake_snapshot(repo_id, repo_type=None, local_dir=None, ignore_patterns=None):
        calls.append({"local_dir": local_dir})
        if local_dir is None:
            raise OSError("cache entry vanished mid-fetch")
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"type": "act"}))
        (d / "model.safetensors").write_text("downloaded")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/policy")
    _join_download(mgr)

    assert mgr.get_status()["state"] == "done"
    # Tried the cache first, then fell back to the network.
    assert [c["local_dir"] is None for c in calls] == [True, False]
    target = tmp_lerobot_home / "makermodslab_models" / "user" / "policy"
    assert (target / "model.safetensors").read_text() == "downloaded"
    assert m.is_model_available_locally("user/policy")


def test_model_download_falls_back_when_the_snapshot_copy_errors(
    tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee for the other half of the cache path — a copy that dies
    (out of disk, a blob yanked out from under us) also falls back."""
    import makermodslab.models as m

    cache = _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/policy")
    snapshot = _build_cache_snapshot(cache, "user/policy")
    calls: list[dict] = []

    def _fake_snapshot(repo_id, repo_type=None, local_dir=None, ignore_patterns=None):
        calls.append({"local_dir": local_dir})
        if local_dir is None:
            return str(snapshot)
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"type": "act"}))
        (d / "model.safetensors").write_text("downloaded")

    def _boom(_snapshot, _target):
        raise OSError("No space left on device")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)
    monkeypatch.setattr(m, "_copy_snapshot_into_store", _boom)

    mgr = _model_download_manager()
    mgr.start("user/policy")
    _join_download(mgr)

    assert mgr.get_status()["state"] == "done"
    assert [c["local_dir"] is None for c in calls] == [True, False]
    assert (
        tmp_lerobot_home / "makermodslab_models" / "user" / "policy" / "model.safetensors"
    ).read_text() == ("downloaded")


def test_models_download_endpoint_rejects_bad_repo_id(client) -> None:
    resp = client.post("/models/download", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400


def test_models_download_endpoint_409_when_running(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.models as m

    monkeypatch.setattr(m.model_download_manager, "state", "running")
    monkeypatch.setattr(m.model_download_manager, "repo_id", "user/busy")
    resp = client.post("/models/download", json={"repo_id": "user/other"})
    assert resp.status_code == 409
    assert "user/busy" in resp.json()["detail"]


def test_models_download_status_endpoint(client) -> None:
    resp = client.get("/models/download-status")
    assert resp.status_code == 200
    assert resp.json()["state"] in {"idle", "running", "done", "error"}


def test_models_publish_endpoint_starts_a_queue(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route hands the selection to the manager and returns immediately —
    a run's worth of checkpoints is far past what an inline request should
    hold open."""
    import makermodslab.models as m

    started = MagicMock(return_value={"started": True, "model_id": "run_x", "message": "ok"})
    monkeypatch.setattr(m.model_upload_manager, "start", started)

    resp = client.post("/api/v1/models/publish", json={"id": "run_x", "steps": [100, 200]})
    assert resp.status_code == 200
    assert resp.json()["started"] is True
    started.assert_called_once_with("run_x", None, [100, 200])


def test_models_publish_endpoint_409_when_a_publish_is_running(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    import makermodslab.models as m

    monkeypatch.setattr(m.model_upload_manager, "state", "running")
    monkeypatch.setattr(m.model_upload_manager, "model_id", "run_busy")
    resp = client.post("/api/v1/models/publish", json={"id": "run_other"})
    assert resp.status_code == 409
    assert "run_busy" in resp.json()["detail"]


def test_models_publish_status_endpoint(client) -> None:
    resp = client.get("/api/v1/models/publish-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] in {"idle", "running", "done", "error"}
    assert "done_steps" in body and "total" in body


def test_models_checkpoints_endpoint_404_for_unknown_run(client) -> None:
    resp = client.get("/api/v1/models/checkpoints?id=nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# import_local_model — copy a checkpoint folder into the local models dir.
# ---------------------------------------------------------------------------


def test_import_local_model_copies_root_shape(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.models import import_local_model, is_model_available_locally

    src = _make_model_checkpoint(tmp_path / "external", "my_policy")
    result = import_local_model(str(src))
    assert result == {"repo_id": "my_policy"}
    assert is_model_available_locally("my_policy")
    # COPY, not move — the source is left intact.
    assert (src / "config.json").is_file()


def test_import_local_model_copies_tree_shape(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.models import get_model_info, import_local_model

    src = _make_model_checkpoint(tmp_path / "external", "run_out", shape="tree", step=300)
    result = import_local_model(str(src), name="team/imported")
    assert result == {"repo_id": "team/imported"}

    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        info = get_model_info("team/imported")
    assert info is not None
    assert info["steps"] == 300
    assert info["path"].endswith("checkpoints/300/pretrained_model")


def test_import_local_model_404_missing_folder(tmp_lerobot_home: Path) -> None:
    from makermodslab.models import ModelError, import_local_model

    with pytest.raises(ModelError) as ei:
        import_local_model("/definitely/not/here")
    assert ei.value.status == 404


def test_import_local_model_400_not_a_checkpoint(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.models import ModelError, import_local_model

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ModelError) as ei:
        import_local_model(str(plain))
    assert ei.value.status == 400


def test_import_local_model_400_bad_name_reworded(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.models import ModelError, import_local_model

    src = _make_model_checkpoint(tmp_path / "external", "raw")
    with pytest.raises(ModelError) as ei:
        import_local_model(str(src), name="a/b/c")  # too many slashes
    assert ei.value.status == 400
    assert "Model name" in ei.value.message  # dataset wording is replaced


def test_import_local_model_409_target_exists(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makermodslab.models import ModelError, import_local_model

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "taken")
    src = _make_model_checkpoint(tmp_path / "external", "src")
    with pytest.raises(ModelError) as ei:
        import_local_model(str(src), name="taken")
    assert ei.value.status == 409


def test_models_import_endpoint_success(client, tmp_lerobot_home: Path, tmp_path: Path) -> None:
    src = _make_model_checkpoint(tmp_path / "external", "endpoint_model")
    resp = client.post("/models/import", json={"path": str(src)})
    assert resp.status_code == 200
    assert resp.json() == {"repo_id": "endpoint_model"}


def test_models_import_endpoint_404_missing(client, tmp_lerobot_home: Path) -> None:
    resp = client.post("/models/import", json={"path": "/no/such/folder"})
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


# ---------------------------------------------------------------------------
# Hidden models — persistent "remove from list" for hub rows (mirror of the
# hidden datasets).
# ---------------------------------------------------------------------------


@pytest.fixture
def hidden_models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SAVED_HIDDEN_MODELS_FILE into a tmp file so hide tests never
    touch the developer's real ~/.cache."""
    from makermodslab.utils import config as cfg

    path = tmp_path / "hidden_models.json"
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_MODELS_FILE", str(path))
    return path


def test_hidden_models_round_trip(hidden_models_file: Path) -> None:
    from makermodslab.utils.config import add_hidden_model, get_hidden_models, remove_hidden_model

    assert get_hidden_models() == set()
    assert add_hidden_model("user/policy")
    assert add_hidden_model("user/policy")  # idempotent re-hide
    assert get_hidden_models() == {"user/policy"}

    assert remove_hidden_model("user/policy")
    assert not remove_hidden_model("user/policy")  # already unhidden
    assert get_hidden_models() == set()
    assert not add_hidden_model("")  # blank refused


def test_hidden_models_corrupt_file_degrades_to_empty(hidden_models_file: Path) -> None:
    from makermodslab.utils.config import get_hidden_models

    hidden_models_file.write_text("{not json")
    assert get_hidden_models() == set()


def test_models_listing_filters_hidden_hub_row(registry, tmp_lerobot_home) -> None:
    from makermodslab.models import list_all_models

    hub_rows = [{"repo_id": "user/policy", "last_modified": None, "private": False}]
    with (
        patch("makermodslab.models.list_hub_models", return_value=hub_rows),
        patch("makermodslab.models.get_hidden_models", return_value={"user/policy"}),
    ):
        result = list_all_models()
    assert result == []


def test_models_hidden_filter_runs_after_pin_fold(registry, tmp_lerobot_home) -> None:
    """Hidden+pinned stays hidden — the filter runs AFTER the pin fold."""
    from makermodslab.models import list_all_models

    with (
        patch("makermodslab.models.list_hub_models", return_value=[]),
        patch("makermodslab.models.get_saved_custom_models", return_value=["user/policy"]),
        patch("makermodslab.models.get_hidden_models", return_value={"user/policy"}),
    ):
        result = list_all_models()
    assert result == []


def test_models_hidden_filter_covers_downloaded_copy(registry, tmp_lerobot_home: Path) -> None:
    """Hidden+downloaded stays hidden — the filter runs after the downloaded
    merge too."""
    from makermodslab.models import list_all_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    with (
        patch("makermodslab.models.list_hub_models", return_value=[]),
        patch("makermodslab.models.get_hidden_models", return_value={"user/policy"}),
    ):
        result = list_all_models()
    assert result == []


def test_models_hide_endpoint_rejects_bad_repo_id(client, hidden_models_file: Path) -> None:
    resp = client.post("/models/hide", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400


def test_models_hide_unhide_endpoints_round_trip(client, hidden_models_file: Path) -> None:
    from makermodslab.utils.config import get_hidden_models

    resp = client.post("/models/hide", json={"repo_id": "user/policy"})
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "user/policy"}
    assert get_hidden_models() == {"user/policy"}

    resp = client.request("DELETE", "/models/hide", json={"repo_id": "user/policy"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert get_hidden_models() == set()


def test_models_pin_route_auto_unhides(client, hidden_models_file: Path, custom_models_file: Path) -> None:
    """Re-adding a hidden model via POST /models/custom removes it from the
    hidden set (mirrors the dataset pin route)."""
    from makermodslab.utils.config import add_hidden_model, get_hidden_models

    add_hidden_model("user/policy")
    resp = client.post("/models/custom", json={"repo_id": "user/policy"})
    assert resp.status_code == 200
    assert get_hidden_models() == set()


# ---------------------------------------------------------------------------
# delete_local_model — downloaded/imported checkpoints (the local models dir).
# ---------------------------------------------------------------------------


def test_delete_local_model_removes_downloaded_checkpoint(registry, tmp_lerobot_home: Path) -> None:
    """A downloaded/imported checkpoint (no registry record) is deleted from
    the local models dir — the 'both' first-press local-copy removal."""
    from makermodslab.models import delete_local_model, is_model_available_locally

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    assert is_model_available_locally("user/policy")

    result = delete_local_model("user/policy")
    assert result == {"deleted": True, "id": "user/policy"}
    assert not is_model_available_locally("user/policy")
    assert not (tmp_lerobot_home / "makermodslab_models" / "user" / "policy").exists()


def test_delete_local_model_unknown_still_404(registry, tmp_lerobot_home: Path) -> None:
    """An id that is neither a registry record nor a downloaded checkpoint
    still 404s (and a traversal id resolves to None, so it 404s too)."""
    from makermodslab.models import ModelError, delete_local_model

    with pytest.raises(ModelError) as ei:
        delete_local_model("ghost/none")
    assert ei.value.status == 404

    outside = tmp_lerobot_home / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    with pytest.raises(ModelError) as ei:
        delete_local_model("../outside")
    assert ei.value.status == 404
    assert outside.exists()  # nothing outside the models root is ever touched


# ---------------------------------------------------------------------------
# Hub-side metadata — policy-type inference + the rich hub info card.
# ---------------------------------------------------------------------------


def test_hub_policy_type_tag_wins() -> None:
    from makermodslab.models import _hub_policy_type

    assert _hub_policy_type(["robotics", "lerobot", "act"], "whatever_name") == "act"


def test_hub_policy_type_longest_prefix_wins() -> None:
    """pi0_fast_... must resolve to pi0_fast, never be shadowed by pi0."""
    from makermodslab.models import _hub_policy_type

    assert _hub_policy_type([], "pi0_fast_sock_2026-01-01_10-00-00") == "pi0_fast"
    assert _hub_policy_type([], "pi0_sock_2026-01-01_10-00-00") == "pi0"


def test_hub_policy_type_makermodslab_name_prefix() -> None:
    from makermodslab.models import _hub_policy_type

    assert _hub_policy_type(["lerobot"], "smolvla_makermods_sock_2026-07-01_10-00-00") == "smolvla"


def test_hub_policy_type_unknown_returns_none() -> None:
    from makermodslab.models import _hub_policy_type

    assert _hub_policy_type(["robotics"], "some_random_repo") is None
    assert _hub_policy_type(None, "actual_name") is None  # "actual" != "act_" prefix


def _fake_model_info(*, tags=None, model_name=None, datasets=None, private=False, used_storage=12345):
    from datetime import UTC, datetime

    info = MagicMock()
    info.tags = tags or []
    info.private = private
    info.used_storage = used_storage
    info.last_modified = datetime(2026, 7, 1, tzinfo=UTC)
    card = MagicMock()
    card.model_name = model_name
    card.datasets = datasets
    info.card_data = card
    return info


def _clear_model_hub_info_cache() -> None:
    import makermodslab.models as m

    with m._MODEL_HUB_INFO_LOCK:
        m._MODEL_HUB_INFO_CACHE.clear()


def test_hub_model_info_maps_expanded_fields() -> None:
    """ONE model_info call yields policy type (card model_name), dataset (card
    datasets), size (usedStorage), private, and last_modified — no file-tree
    probe when the type is already known."""
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(
        tags=["lerobot"], model_name="act", datasets=["user/pick"], private=True
    )
    with (
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models._hub_model_probe") as probe,
    ):
        row = m._hub_model_info("user/policy")

    probe.assert_not_called()  # cheap signals sufficed — no extra Hub calls
    assert row["policy_type"] == "act"
    assert row["dataset"] == "user/pick"
    assert row["size_bytes"] == 12345
    assert row["private"] is True
    assert row["last_modified"] is not None
    assert row["source"] == "hub"
    fake_api.model_info.assert_called_once()


def test_hub_model_info_reads_episodes_from_final_checkpoint_train_config() -> None:
    """dataset_episodes for a Hub skill comes from its final checkpoint's
    train_config.json — a file read separate from the card-metadata dataset
    name, since the Hub card only ever carries the repo id, not episode
    granularity."""
    import makermodslab.models as m
    from makermodslab.jobs import JobCheckpoint

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(model_name="act", datasets=["user/pick"])
    final_ckpt = JobCheckpoint(step=1000, source="hub", ref="user/policy@checkpoints/001000")
    with (
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.jobs._list_imported_hub", return_value=[final_ckpt]),
        patch(
            "makermodslab.jobs.read_checkpoint_train_config",
            return_value={"dataset": {"repo_id": "user/pick", "episodes": [2, 0]}},
        ),
    ):
        row = m._hub_model_info("user/policy")
    assert row["dataset_episodes"] == [0, 2]


def test_hub_model_info_no_dataset_skips_episode_lookup() -> None:
    """No dataset from the card ⇒ no point fetching a checkpoint just to look
    for its episodes."""
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(model_name="act", datasets=None)
    with (
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.jobs._list_imported_hub") as list_ckpts,
    ):
        row = m._hub_model_info("user/policy")
    assert row["dataset_episodes"] is None
    list_ckpts.assert_not_called()


def test_hub_model_info_falls_back_to_probe_on_error() -> None:
    """model_info raising degrades to the old probe (never propagates)."""
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.side_effect = RuntimeError("hub down")
    probe_row = {"id": "user/policy", "policy_type": "act", "steps": 500}
    with (
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models._hub_model_probe", return_value=probe_row) as probe,
    ):
        row = m._hub_model_info("user/policy")
    probe.assert_called_once()
    assert row == probe_row


def test_hub_model_info_probe_recovers_unknown_type() -> None:
    """When the cheap signals leave the type unknown, the probe supplies the
    type + step from the checkpoint config."""
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(tags=["robotics"], model_name=None)
    probe_row = {"policy_type": "vqbet", "steps": 700}
    with (
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models._hub_model_probe", return_value=probe_row),
    ):
        row = m._hub_model_info("user/mystery_repo")
    assert row["policy_type"] == "vqbet"
    assert row["steps"] == 700


def test_hub_model_info_caches_success_not_failure() -> None:
    """A successful answer is memoized (one model_info across two calls); a
    failed one is NOT cached, so the next call retries."""
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(model_name="act")
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        m._hub_model_info("user/cached")
        m._hub_model_info("user/cached")
    assert fake_api.model_info.call_count == 1

    _clear_model_hub_info_cache()
    failing_api = MagicMock()
    failing_api.model_info.side_effect = RuntimeError("down")
    with (
        patch("makermodslab.models.shared_hf_api", return_value=failing_api),
        patch("makermodslab.models._hub_model_probe", return_value=None),
    ):
        assert m._hub_model_info("user/flaky") is None
        assert m._hub_model_info("user/flaky") is None
    assert failing_api.model_info.call_count == 2  # degrade never cached


def test_invalidate_model_hub_info_forces_refetch() -> None:
    import makermodslab.models as m

    _clear_model_hub_info_cache()
    fake_api = MagicMock()
    fake_api.model_info.return_value = _fake_model_info(model_name="act")
    with patch("makermodslab.models.shared_hf_api", return_value=fake_api):
        m._hub_model_info("user/inval")
        m.invalidate_model_hub_info("user/inval")
        m._hub_model_info("user/inval")
    assert fake_api.model_info.call_count == 2


def test_list_all_models_hub_rows_carry_policy_type(registry) -> None:
    from makermodslab.models import list_all_models

    hub_rows = [
        {
            "repo_id": "user/act_sock_2026-01-01_10-00-00",
            "last_modified": None,
            "private": False,
            "policy_type": "act",
        },
    ]
    with patch("makermodslab.models.list_hub_models", return_value=hub_rows):
        result = list_all_models()
    assert result[0]["policy_type"] == "act"


def test_list_all_models_local_type_wins_on_both_collapse(registry, tmp_lerobot_home: Path) -> None:
    """The on-disk checkpoint's config.json type overrides the hub row's
    tag/name-derived one when a downloaded copy collapses to 'both'."""
    from makermodslab.models import list_all_models

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy", policy_type="smolvla")
    hub_rows = [
        {"repo_id": "user/policy", "last_modified": None, "private": False, "policy_type": "act"},
    ]
    with patch("makermodslab.models.list_hub_models", return_value=hub_rows):
        result = list_all_models()
    assert len(result) == 1
    assert result[0]["source"] == "both"
    assert result[0]["policy_type"] == "smolvla"  # local config.json wins


def test_upload_local_model_stamps_policy_tag(registry) -> None:
    """The uploaded tag set includes the checkpoint's policy type alongside the
    org tags, so MakerMods Lab uploads are self-describing on the Hub."""
    from makermodslab.models import upload_local_model

    _seed_run(registry, "tag_run", policy_type="act", dataset="user/pick", steps=100)

    fake_api = MagicMock()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=fake_api),
        patch("makermodslab.models.metadata_update") as mock_meta,
    ):
        result = upload_local_model("tag_run")

    tags = mock_meta.call_args.args[1]["tags"]
    assert "act" in tags
    assert {"makermods", "openbooth", "MakerModsLab"}.issubset(set(tags))
    assert "act" in result["tags"]


# ---------------------------------------------------------------------------
# Inference-delete guard — a checkpoint a live inference reads can't be deleted.
# ---------------------------------------------------------------------------


def _set_running_inference(monkeypatch: pytest.MonkeyPatch, policy_path: str) -> None:
    """Simulate an active inference reading `policy_path` (the resolved local
    checkpoint dir rollout captures at start)."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"policy_path": policy_path})


def test_model_in_use_containment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact-dir match AND parent-of-pretrained_model both count as in use;
    an unrelated sibling does not."""
    from makermodslab.models import _model_in_use

    target = tmp_path / "models" / "user" / "policy"
    pretrained = target / "checkpoints" / "500" / "pretrained_model"
    pretrained.mkdir(parents=True)

    _set_running_inference(monkeypatch, str(pretrained))
    assert _model_in_use(pretrained) is not None  # exact dir
    assert _model_in_use(target) is not None  # ancestor of the active path
    other = tmp_path / "models" / "user" / "other"
    other.mkdir(parents=True)
    assert _model_in_use(other) is None  # unrelated dir


def test_delete_downloaded_model_409_when_inference_reads_it(
    registry, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab.models import ModelError, delete_local_model

    model_dir = _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/live_policy")
    _set_running_inference(monkeypatch, str(model_dir))

    with pytest.raises(ModelError) as ei:
        delete_local_model("user/live_policy")
    assert ei.value.status == 409
    assert "running inference" in ei.value.message
    assert model_dir.exists()  # nothing was deleted


def test_delete_run_model_409_when_inference_reads_its_checkpoint(
    registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A COMPLETED run's final checkpoint being an active inference target
    blocks the run-dir delete (the registry's running-guard doesn't cover it)."""
    from makermodslab.models import ModelError, delete_local_model

    pretrained = _seed_run(registry, "live_run", state="done", steps=100)
    _set_running_inference(monkeypatch, str(pretrained))

    with pytest.raises(ModelError) as ei:
        delete_local_model("live_run")
    assert ei.value.status == 409
    assert "running inference" in ei.value.message
    assert (registry._output_root / "live_run").exists()


def test_delete_succeeds_when_inference_reads_other_path(
    registry, tmp_lerobot_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab.models import delete_local_model

    _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/idle_policy")
    elsewhere = tmp_path / "elsewhere" / "pretrained_model"
    elsewhere.mkdir(parents=True)
    _set_running_inference(monkeypatch, str(elsewhere))

    result = delete_local_model("user/idle_policy")
    assert result["deleted"] is True


def test_model_download_rejects_a_fetch_that_lands_without_weights(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that ends weights-less is a FAILURE, not a usable model.

    Reproduces the live incident at the download boundary: the fetch lands
    config.json, train_config.json and both processor safetensors but no
    model.safetensors — the 68 KB interrupted-download shape. The post-download
    validation must catch it so the partial is cleaned up and the user is told,
    rather than the entry sitting in the library until inference dies on
    FileNotFoundError deep inside lerobot.
    """
    import makermodslab.models as m

    def _fake_snapshot(repo_id, repo_type, local_dir, ignore_patterns=None):  # noqa: ARG001
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"type": "act"}))
        (d / "train_config.json").write_text(json.dumps({"type": "act"}))
        # Present, and deliberately NOT policy weights.
        (d / "preprocessor.safetensors").write_text("processor")
        (d / "postprocessor.safetensors").write_text("processor")

    monkeypatch.setattr(m, "snapshot_download", _fake_snapshot)

    mgr = _model_download_manager()
    mgr.start("user/partial")
    _join_download(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert not m.is_model_available_locally("user/partial")
    # The partial dir is cleaned up, so it can't be mistaken for a complete copy.
    assert not (tmp_lerobot_home / "makermodslab_models" / "user" / "partial").exists()


def test_resolve_pretrained_dir_skips_a_half_written_newest_checkpoint(tmp_lerobot_home: Path) -> None:
    """A checkpoint still being written must not hide the last complete one.

    Training writes checkpoints/<step>/pretrained_model incrementally, so the
    highest step can exist with a config and no weights for a while. The scan
    walks down to the newest COMPLETE checkpoint instead of reporting the run
    unusable.
    """
    import makermodslab.models as m

    root = tmp_lerobot_home / "makermodslab_models" / "user" / "run"
    good = root / "checkpoints" / "000100" / "pretrained_model"
    good.mkdir(parents=True)
    (good / "config.json").write_text(json.dumps({"type": "act"}))
    (good / "model.safetensors").write_text("weights")
    partial = root / "checkpoints" / "000200" / "pretrained_model"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text(json.dumps({"type": "act"}))

    assert m._resolve_pretrained_dir(root) == good.resolve()


# ---------------------------------------------------------------------------
# Review follow-ups (PR #83 queue): /models/delete vs the training queue.
# ---------------------------------------------------------------------------


def test_delete_local_model_refuses_a_queued_run_instead_of_cancelling_it(registry) -> None:
    """POST /models/delete with a QUEUED run's id used to silently cancel the
    queued run and answer {deleted: true}: the registry's delete() guard only
    refuses `running`, so a queued record sailed through `job_registry.delete`
    and left the queue — a cancel the user never asked for, reported as a
    model deletion. Only terminal runs (done / interrupted / failed) hold
    deletable artifacts; everything else is refused with a coded 409 naming
    the queue as the place to act."""
    from makermodslab.models import ModelError, delete_local_model

    _seed_run(registry, "queued_run", state="queued", with_checkpoint=False)
    registry._records["queued_run"].queue_seq = 10
    # A real queued record always has its job dir (start() claims it at submit).
    (registry._output_root / "queued_run").mkdir(parents=True)

    with pytest.raises(ModelError) as ei:
        delete_local_model("queued_run")

    assert ei.value.status == 409
    assert ei.value.code == "job.not_terminal"
    assert "queue" in ei.value.message.lower()
    # The run is still queued — nothing was cancelled or removed.
    assert registry._records["queued_run"].state == "queued"
    assert (registry._output_root / "queued_run").exists()


def test_delete_local_model_running_refusal_carries_the_same_code(registry) -> None:
    """The running-run refusal is the same condition (not terminal yet), so it
    speaks the same code. Message unchanged — codes are additive."""
    from makermodslab.models import ModelError, delete_local_model

    _seed_run(registry, "live_run2", state="running")
    with pytest.raises(ModelError) as ei:
        delete_local_model("live_run2")
    assert ei.value.status == 409
    assert ei.value.code == "job.not_terminal"


def test_models_delete_route_forwards_the_refusal_code(client, monkeypatch, tmp_path) -> None:
    """The HTTP layer must not drop the code: the 409 body carries `code`
    beside the legacy string `detail` (the ApiError shape every coded refusal
    uses)."""
    from makermodslab.jobs import JobRecord, JobRegistry, job_registry
    from makermodslab.train import TrainingRequest

    monkeypatch.setattr(JobRegistry, "_drain_queue", lambda self: None)
    record = JobRecord(
        id="queued-on-wire",
        name="queued-on-wire",
        state="queued",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act"),
        output_dir=str(tmp_path / "queued-on-wire" / "run"),
        started_at=0.0,
        runner="local",
        queue_seq=10,
    )
    original = dict(job_registry._records)
    try:
        job_registry._records["queued-on-wire"] = record
        resp = client.post("/models/delete", json={"id": "queued-on-wire"})
        assert resp.status_code == 409, resp.text
        body = resp.json()
        assert body["code"] == "job.not_terminal"
        assert isinstance(body["detail"], str)
        assert job_registry._records["queued-on-wire"].state == "queued"
    finally:
        job_registry._records.clear()
        job_registry._records.update(original)


def test_delete_local_model_409_while_its_publish_is_running(registry, monkeypatch) -> None:
    """The background publish reads a run's checkpoint dirs for minutes; a
    delete that passes every terminal-state guard would rmtree them out from
    under upload_folder mid-read. The registry guard refuses it with the same
    coded 409 shape every other delete refusal uses."""
    import makermodslab.models as m
    from makermodslab.models import ModelError, delete_local_model

    _seed_run(registry, "pub_run", state="done")
    monkeypatch.setattr(m.model_upload_manager, "state", "running")
    monkeypatch.setattr(m.model_upload_manager, "model_id", "pub_run")
    with pytest.raises(ModelError) as ei:
        delete_local_model("pub_run")
    assert ei.value.status == 409
    assert ei.value.code == "job.publish_in_progress"
    # The dir must still be there — the publish is reading it.
    assert (registry._output_root / "pub_run").exists()


def test_delete_local_model_allowed_while_another_runs_publish_is_running(registry, monkeypatch) -> None:
    """The guard is per-run: a publish of run A must not lock run B's delete —
    the single-slot manager would otherwise make every delete 409 for minutes."""
    import makermodslab.models as m
    from makermodslab.models import delete_local_model

    _seed_run(registry, "other_run", state="done")
    monkeypatch.setattr(m.model_upload_manager, "state", "running")
    monkeypatch.setattr(m.model_upload_manager, "model_id", "some_other_publish")
    assert delete_local_model("other_run")["deleted"] is True


def test_jobs_delete_route_409_while_publish_is_running(client, monkeypatch, tmp_path) -> None:
    """The jobs surface reuses the registry guard: DELETE /jobs/{id} during
    that run's publish refuses with the same code instead of racing the
    upload."""
    import makermodslab.models as m
    from makermodslab.jobs import JobRecord, job_registry
    from makermodslab.train import TrainingRequest

    monkeypatch.setattr(m.model_upload_manager, "state", "running")
    monkeypatch.setattr(m.model_upload_manager, "model_id", "pub-on-wire")
    record = JobRecord(
        id="pub-on-wire",
        name="pub-on-wire",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act"),
        output_dir=str(tmp_path / "pub-on-wire" / "run"),
        started_at=0.0,
        runner="local",
    )
    original = dict(job_registry._records)
    try:
        job_registry._records["pub-on-wire"] = record
        resp = client.delete("/api/v1/jobs/pub-on-wire")
        assert resp.status_code == 409, resp.text
        body = resp.json()
        assert body["code"] == "job.publish_in_progress"
        assert isinstance(body["detail"], str)
        # The record survives the refusal.
        assert "pub-on-wire" in job_registry._records
    finally:
        job_registry._records.clear()
        job_registry._records.update(original)


def test_models_routes_keep_local_kind(client, monkeypatch) -> None:
    """Regression: `local_kind` was emitted by the producers but undeclared on
    ModelListItem/ModelInfoResponse, so response_model silently FILTERED it and
    the frontend's delete semantics never saw it. Declared now — the routes
    must pass it through, and exclude_unset must keep it absent (not null) on
    rows whose producer never set it."""
    import makermodslab.models as m

    run_row = {
        "id": "run_lk",
        "name": "run lk",
        "policy_type": "act",
        "dataset": "user/pick",
        "steps": 100,
        "path": "/tmp/x",
        "last_modified": None,
        "hf_repo_id": None,
        "source": "local",
        "local_kind": "run",
    }
    hub_row = {
        "id": "user/repo",
        "name": "repo",
        "policy_type": None,
        "dataset": None,
        "steps": None,
        "path": None,
        "last_modified": None,
        "hf_repo_id": "user/repo",
        "source": "hub",
    }
    monkeypatch.setattr(m, "list_all_models", lambda: [run_row, hub_row])
    rows = client.get("/api/v1/models").json()
    assert rows[0]["local_kind"] == "run"
    assert "local_kind" not in rows[1]

    monkeypatch.setattr(m, "get_model_info", lambda _id: {**run_row, "size_bytes": 1})
    info = client.get("/api/v1/models/info", params={"id": "run_lk"}).json()
    assert info["local_kind"] == "run"


def test_delete_local_model_names_the_queued_finetune_instead_of_502ing(registry) -> None:
    """JobRegistry.delete raises JobSourceOfQueuedRunError when a QUEUED run
    froze this run's checkpoint path at submit time. models.py didn't catch
    it, so the refusal fell into the catch-all and surfaced as a 502 'Failed
    to delete model' — a deliberate guard reported as an infrastructure
    failure. Same 409 + job.has_queued_dependents the /jobs route emits."""
    from makermodslab.jobs import JobRecord
    from makermodslab.models import ModelError, delete_local_model
    from makermodslab.train import TrainingRequest

    _seed_run(registry, "src_run", state="done")
    registry._records["queued_ft"] = JobRecord(
        id="queued_ft",
        name="queued_ft",
        state="queued",
        config=TrainingRequest(dataset_repo_id="user/pick", finetune_from_job_id="src_run"),
        output_dir=str(registry._output_root / "queued_ft" / "run"),
        started_at=2.0,
        runner="local",
        queue_seq=10,
    )

    with pytest.raises(ModelError) as ei:
        delete_local_model("src_run")

    assert ei.value.status == 409
    assert ei.value.code == "job.has_queued_dependents"
    assert "queued_ft" in ei.value.message
    # The source run — and the checkpoint the queued fine-tune will read — is
    # untouched.
    assert "src_run" in registry._records
    assert (registry._output_root / "src_run").exists()


def test_delete_downloaded_model_refuses_when_a_queued_run_reads_it(registry, tmp_lerobot_home: Path) -> None:
    """The downloaded/imported branch of delete_local_model checked only the
    live-inference guard (_model_in_use) — a QUEUED run whose frozen
    policy_pretrained_path points inside the store dir sailed past it, and the
    rmtree pulled the base out from under a run that fails at launch, hours
    later, with a path nobody could tie to this click. Same guard, same 409 +
    job.has_queued_dependents as every other queued-dependency refusal."""
    from makermodslab.jobs import JobRecord
    from makermodslab.models import ModelError, delete_local_model
    from makermodslab.train import TrainingRequest

    model_dir = _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/base")
    registry._records["queued_ft"] = JobRecord(
        id="queued_ft",
        name="queued_ft",
        state="queued",
        config=TrainingRequest(
            dataset_repo_id="user/pick",
            policy_pretrained_path=str(model_dir.resolve()),
        ),
        output_dir=str(registry._output_root / "queued_ft" / "run"),
        started_at=1.0,
        runner="local",
        queue_seq=10,
    )

    with pytest.raises(ModelError) as ei:
        delete_local_model("user/base")

    assert ei.value.status == 409
    assert ei.value.code == "job.has_queued_dependents"
    assert "queued_ft" in ei.value.message
    assert model_dir.exists(), "the refusal must leave the checkpoint on disk"

    # Without the dependent, the delete works exactly as before.
    del registry._records["queued_ft"]
    result = delete_local_model("user/base")
    assert result["deleted"] is True
    assert not model_dir.exists()


# ---------------------------------------------------------------------------
# Multi-checkpoint publish — adversarial edge cases (folded in from the former
# test_pr61_adversarial.py: zero-padded vs bare checkpoint dirs, mid-queue Hub
# failures, a run downloaded back from its own repo, and the delete semantics
# of a partly published run).
# ---------------------------------------------------------------------------


def _seed_multi(
    registry,
    job_id: str,
    step_dirs: list[str],
    *,
    state: str = "done",
    policy_type: str = "act",
    dataset: str = "user/pick",
    hf_repo_id: str | None = None,
) -> Path:
    """Register a run and lay out one checkpoint per entry of `step_dirs`.

    `step_dirs` are the literal on-disk directory names, so a test can use
    lerobot's real zero-padded form ("000100") or a bare one ("100")."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    run_dir = registry._output_root / job_id / "run"
    registry._records[job_id] = JobRecord(
        id=job_id,
        name=f"run {job_id}",
        state=state,
        config=TrainingRequest(dataset_repo_id=dataset, policy_type=policy_type),
        output_dir=str(run_dir),
        started_at=1.0,
        ended_at=1000.0,
        runner="local",
        hf_repo_id=hf_repo_id,
    )
    for name in step_dirs:
        p = run_dir / "checkpoints" / name / "pretrained_model"
        p.mkdir(parents=True)
        (p / "config.json").write_text(json.dumps({"type": policy_type}))
        (p / "train_config.json").write_text(
            json.dumps(
                {
                    "policy": {"type": policy_type},
                    "dataset": {"repo_id": dataset},
                    "steps": int(name),
                }
            )
        )
    return run_dir


def _hub_api(files: list[str] | None = None) -> MagicMock:
    api = MagicMock()
    api.list_repo_files.return_value = list(files or [])
    return api


def _no_card():
    """Patch the card sync's downloader so no test reaches the network."""
    from huggingface_hub.errors import EntryNotFoundError

    return patch("huggingface_hub.hf_hub_download", side_effect=EntryNotFoundError("no card"))


# ---------------------------------------------------------------------------
# 1. Zero-padded checkpoint dirs must round-trip into the Hub path.
# ---------------------------------------------------------------------------


def test_padded_checkpoint_dirs_upload_to_padded_hub_paths(registry) -> None:
    from makermodslab.models import upload_local_model

    _seed_multi(registry, "pad_run", ["000100", "000200", "001000"])
    api = _hub_api()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        result = upload_local_model("pad_run", steps=[100, 1000])

    paths = [c.kwargs["path_in_repo"] for c in api.upload_folder.call_args_list]
    assert paths == [
        "checkpoints/000100/pretrained_model",
        "checkpoints/001000/pretrained_model",
    ]
    assert result["steps"] == [100, 1000]


# ---------------------------------------------------------------------------
# 2. Step selection errors.
# ---------------------------------------------------------------------------


def test_invalid_user_typed_repo_name_gets_a_useful_error(registry) -> None:
    """PublishToHubRow's repo-name Input is free text with no validation. A name
    the Hub can't accept should come back as an actionable 400, not a generic
    'the Hub rejected the upload' 502."""
    from huggingface_hub.errors import HFValidationError

    from makermodslab.models import ModelError, upload_local_model

    _seed_multi(registry, "bad_name", ["100"])
    api = _hub_api()
    api.create_repo.side_effect = HFValidationError(
        "Repo id must use alphanumeric chars or '-', '_', '.'; got 'My Run!!'"
    )
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("bad_name", repo_id="My Run!!", steps=[100])
    assert ei.value.status == 400


def test_empty_step_list_is_400(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_multi(registry, "e_run", ["100"])
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("e_run", steps=[])
    assert ei.value.status == 400


def test_unknown_step_is_404_and_names_the_saved_steps(registry) -> None:
    from makermodslab.models import ModelError, upload_local_model

    _seed_multi(registry, "u_run", ["100", "200"])
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("u_run", steps=[100, 999])
    assert ei.value.status == 404
    assert "999" in ei.value.message and "100, 200" in ei.value.message


def test_duplicate_steps_upload_once(registry) -> None:
    from makermodslab.models import upload_local_model

    _seed_multi(registry, "d_run", ["100", "200"])
    api = _hub_api()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        result = upload_local_model("d_run", steps=[200, 100, 200])

    assert api.upload_folder.call_count == 2
    assert result["steps"] == [100, 200]


# ---------------------------------------------------------------------------
# 3. Partial failure: what survived, and where the run thinks it published to.
# ---------------------------------------------------------------------------


def test_partial_failure_reports_only_the_steps_that_landed(registry) -> None:
    from makermodslab.models import ModelUploadManager

    _seed_multi(registry, "p_run", ["100", "200", "300"])
    api = _hub_api()
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("hub exploded")

    api.upload_folder.side_effect = flaky

    mgr = ModelUploadManager()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        mgr.start("p_run", None, [100, 200, 300])
        mgr._thread.join(timeout=10)

    st = mgr.get_status()
    assert st["state"] == "error"
    assert st["done_steps"] == [100, 200]


def test_partial_failure_still_pins_the_repo_it_published_into(registry) -> None:
    """A queue that dies mid-way has already put weights in the target repo.

    The record must remember that repo, or the retry (and the picker's
    `published` badges) go looking at a different one."""
    from makermodslab.models import ModelUploadManager

    _seed_multi(registry, "pin_run", ["100", "200"])
    api = _hub_api()
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("hub exploded")

    api.upload_folder.side_effect = flaky

    mgr = ModelUploadManager()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        mgr.start("pin_run", "myorg/custom-name", [100, 200])
        mgr._thread.join(timeout=10)

    assert mgr.get_status()["done_steps"] == [100]
    # Step 100 is on the Hub in myorg/custom-name.
    assert registry._records["pin_run"].hf_repo_id == "myorg/custom-name"


def test_retry_after_partial_failure_targets_the_same_repo(registry) -> None:
    """The failure toast says 'retry the rest'. The retry must not fork the run
    into a second repo."""
    from makermodslab.models import ModelUploadManager, list_run_checkpoints

    _seed_multi(registry, "retry_run", ["100", "200"])
    api = _hub_api()
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("hub exploded")

    api.upload_folder.side_effect = flaky

    mgr = ModelUploadManager()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        mgr.start("retry_run", "myorg/custom-name", [100, 200])
        mgr._thread.join(timeout=10)

        # What the reopened picker would target on retry.
        api.list_repo_files.return_value = ["checkpoints/100/pretrained_model/config.json"]
        picker = list_run_checkpoints("retry_run")

    assert picker["default_repo_id"] == "myorg/custom-name"


# ---------------------------------------------------------------------------
# 4. hub_readable must distinguish "new repo" from "Hub unreachable".
# ---------------------------------------------------------------------------


def test_first_publish_does_not_claim_the_hub_was_unreachable(registry) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    from makermodslab.models import list_run_checkpoints

    _seed_multi(registry, "new_run", ["100"])
    resp = MagicMock()
    resp.status_code = 404
    api = MagicMock()
    api.list_repo_files.side_effect = RepositoryNotFoundError("404 — repo does not exist", response=resp)

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
    ):
        data = list_run_checkpoints("new_run")

    # The Hub answered — it said "no such repo". That is not an outage, and the
    # picker renders `hub_readable: false` as "Couldn't reach the Hub".
    assert data["hub_readable"] is True
    assert data["checkpoints"] == [{"step": 100, "path": data["checkpoints"][0]["path"], "published": False}]


# ---------------------------------------------------------------------------
# 5. A published run that is ALSO downloaded back to the local models dir.
# ---------------------------------------------------------------------------


def test_published_run_downloaded_back_still_reads_as_a_run(registry, tmp_lerobot_home: Path) -> None:
    """`local_kind` decides whether the delete dialog is the safe two-press
    'remove local copy' or the destructive run delete. A run that was published
    and then downloaded must stay 'run' — its unpublished checkpoints exist
    nowhere else."""
    from makermodslab.models import _local_models_root, list_all_models

    _seed_multi(registry, "both_run", ["100", "200"], hf_repo_id="user/both_run")

    # The user downloaded their own published repo back to the models dir.
    dl = _local_models_root() / "user" / "both_run"
    (dl / "checkpoints" / "100" / "pretrained_model").mkdir(parents=True)
    (dl / "checkpoints" / "100" / "pretrained_model" / "config.json").write_text(json.dumps({"type": "act"}))

    hub_rows = [{"repo_id": "user/both_run", "last_modified": "2026-01-01T00:00:00Z", "private": False}]
    with (
        patch("makermodslab.models.list_hub_models", return_value=hub_rows),
        patch("makermodslab.models.get_saved_custom_models", return_value=[]),
    ):
        rows = list_all_models()

    row = next(r for r in rows if r.get("hf_repo_id") == "user/both_run")
    assert row["source"] == "both"
    # The row's id is the RUN id, so a delete from this row deletes the run dir.
    assert row["id"] == "both_run"
    # ...therefore it must not be described as a replaceable downloaded copy.
    assert row["local_kind"] == "run"


# ---------------------------------------------------------------------------
# 6. Concurrency: one publish at a time.
# ---------------------------------------------------------------------------


def test_second_publish_is_refused_while_one_is_running(registry) -> None:
    import threading

    from makermodslab.models import ModelUploadManager

    _seed_multi(registry, "c1", ["100"])
    _seed_multi(registry, "c2", ["100"])

    gate = threading.Event()
    api = _hub_api()
    api.upload_folder.side_effect = lambda **kw: gate.wait(timeout=10)

    mgr = ModelUploadManager()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        assert mgr.start("c1", None, [100])["started"] is True
        second = mgr.start("c2", None, [100])
        assert second["started"] is False
        gate.set()
        mgr._thread.join(timeout=10)

    assert mgr.get_status()["model_id"] == "c1"


# ---------------------------------------------------------------------------
# 7. Card index vs. what is actually on the Hub.
# ---------------------------------------------------------------------------


def test_card_index_merges_previously_published_steps(registry) -> None:
    from makermodslab.models import upload_local_model

    _seed_multi(registry, "card_run", ["000100", "000200"], hf_repo_id="user/card_run")
    # 000100 is already up there from an earlier publish.
    api = _hub_api(["checkpoints/000100/pretrained_model/config.json"])

    captured = {}

    def fake_sync(_api, repo_id, record, steps):
        captured["steps"] = dict(steps)

    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        patch("makermodslab.models._sync_model_card", side_effect=fake_sync),
    ):
        result = upload_local_model("card_run", steps=[200])

    assert captured["steps"] == {100: "000100", 200: "000200"}
    assert result["published_steps"] == [100, 200]


def test_delete_from_the_clobbered_row_destroys_unpublished_checkpoints(
    registry, tmp_lerobot_home: Path
) -> None:
    """Regression guard for the local_kind clobber.

    The row's id is the RUN id, so a delete of the row removes the whole
    training run — unpublished checkpoints included. The row must therefore
    keep local_kind "run": the library UI no longer offers destructive model
    deletes at all, but the routes remain for a future management surface, and
    resolveDeleteAction (which that surface will reuse) reads local_kind to
    pick the honest destructive dialog over the reassuring "the Hub copy
    stays" one.

    (Deleting really does destroy the unpublished steps; that is correct and
    irreversible. What must never happen is that outcome behind the two-press
    'remove local copy' wording.)"""
    from makermodslab.models import _local_models_root, delete_local_model, list_all_models

    run_dir = _seed_multi(registry, "victim", ["100", "200"], hf_repo_id="user/victim")
    unpublished = run_dir / "checkpoints" / "200" / "pretrained_model" / "config.json"
    assert unpublished.is_file()

    dl = _local_models_root() / "user" / "victim"
    (dl / "checkpoints" / "100" / "pretrained_model").mkdir(parents=True)
    (dl / "checkpoints" / "100" / "pretrained_model" / "config.json").write_text(json.dumps({"type": "act"}))

    hub_rows = [{"repo_id": "user/victim", "last_modified": "2026-01-01T00:00:00Z", "private": False}]
    with (
        patch("makermodslab.models.list_hub_models", return_value=hub_rows),
        patch("makermodslab.models.get_saved_custom_models", return_value=[]),
    ):
        row = next(r for r in list_all_models() if r.get("hf_repo_id") == "user/victim")

    # "run" + source "both" is the one combination resolveDeleteAction routes to
    # the destructive "Delete" dialog rather than "Remove local copy of".
    assert row["local_kind"] == "run"
    assert row["id"] == "victim"

    with patch("makermodslab.models._model_in_use", return_value=None):
        delete_local_model(row["id"])

    # "The Hub copy stays" was true. "Only the local copy is removed" was not.
    assert not unpublished.exists(), "unpublished step 200 survived"
    assert not run_dir.parent.exists(), "run dir survived"
