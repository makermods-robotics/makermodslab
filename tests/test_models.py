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
    """Clear the module-global /models listing state before and after each test
    so one test's result never leaks into another (the conftest autouse fixture
    resets the datasets/jobs caches but not this one).

    Wider than the public invalidation on purpose. `invalidate_model_listing_cache`
    deliberately KEEPS the last-good Hub rows and the last fan-out outcome —
    they are the fallback for a Hub outage, and registry mutations (which are
    frequent) must not wipe the safety net. Across tests that same persistence
    is contamination, so the fixture reaches past the public call and resets
    them too."""
    import makermodslab.models as m

    def _reset() -> None:
        m.invalidate_model_listing_cache()
        m._hub_last_good = None
        m._hub_last_good_auth = ""
        m._hub_last_good_authors = ()
        m._forgotten_hub_repos = {}
        m._hub_outcome = {"authenticated": False, "authors": (), "answered": 0}

    _reset()
    # Signed OUT by default. The listing's identity key is the HF token, so
    # without this every assertion about Hub reachability would depend on
    # whether the person running the suite happens to have a token in their
    # environment — the same machine-state coupling that made
    # test_list_all_models_hub_rows_carry_policy_type flaky. A test that needs
    # an identity patches `get_token` itself; the inner patch wins.
    with patch("makermodslab.models.get_token", return_value=""):
        yield
    _reset()


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
# list_imported_models — registered pointer imports as a listing source.
# ---------------------------------------------------------------------------


def _seed_import(
    registry,
    job_id: str,
    *,
    output_dir: Any = "",
    hf_repo_id: str | None = None,
    name: str | None = None,
    started_at: float = 900.0,
):
    """Register an `imported` pointer record the way jobs.register_imported does:
    a resolved local dir in output_dir, OR a repo id in hf_repo_id, never both,
    and a placeholder config ("(imported)" dataset, "model" policy type)."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id=job_id,
        name=name or job_id,
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="model"),
        output_dir=str(output_dir),
        started_at=started_at,
        ended_at=started_at,
        runner="imported",
        hf_repo_id=hf_repo_id,
    )
    registry._records[job_id] = record
    return record


def _listing(hub_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from makermodslab.models import list_all_models

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows or []):
            stack.enter_context(cm)
        return list_all_models()


def test_list_all_models_lists_a_disk_pointer_import(registry, tmp_lerobot_home, tmp_path) -> None:
    """An import stores a POINTER, not a copy: an arbitrary output_dir and no
    repo. Such a record is not a `local` run (so list_local_models skips it on
    the runner gate), owns no Hub repo (so list_hub_models never sees it), and
    was never copied into the local models store (so the downloaded scan never
    walks it). It fell through all three, and /models did not list it at all
    while /jobs did — the models library and the deploy picker disagreeing
    permanently, with no TTL involved."""
    ckpt = _make_model_checkpoint(tmp_path / "elsewhere", "my_policy", shape="tree")
    _seed_import(registry, "act_imported", output_dir=ckpt, name="Sock folding")

    row = next(r for r in _listing() if r["id"] == "act_imported")
    assert row["name"] == "Sock folding"
    assert row["source"] == "local"
    # Detail is read off the checkpoint, never off the placeholder config.
    assert row["policy_type"] == "act"
    assert row["dataset"] is None
    assert row["path"].endswith("pretrained_model")


def test_list_all_models_lists_a_foreign_hub_pointer_import(registry, tmp_lerobot_home) -> None:
    """The other pointer shape: a repo in someone else's namespace. list_hub_models
    fans out over the USER's own authors, so a foreign repo never appears there
    however healthy the Hub is. Stamped from the record — the row exists without
    a single Hub call, which is what keeps this source off the /models latency
    path."""
    _seed_import(registry, "smolvla_imported", hf_repo_id="someone/smolvla_dishes")

    row = next(r for r in _listing() if r["id"] == "someone/smolvla_dishes")
    assert row["source"] == "hub"
    assert row["hf_repo_id"] == "someone/smolvla_dishes"
    assert row["path"] is None
    # "model" is register_imported's fallback, not a policy — inferred from the
    # repo name instead of surfaced as if it were a fact.
    assert row["policy_type"] == "smolvla"


def test_list_all_models_import_does_not_duplicate_its_own_hub_row(registry, tmp_lerobot_home) -> None:
    """A repo the user owns is already listed by the Hub half. Re-adding it under
    the import's key would offer one set of weights twice, under two ids that
    both work. Matched case-insensitively, the rule jobs.find_imported dedups
    on — HF repo ids are practically unique across casing."""
    _seed_import(registry, "act_imported", hf_repo_id="Makermods/ACT_Pick")

    hub_rows = [{"repo_id": "makermods/act_pick", "last_modified": None, "private": False}]
    ids = [r["id"] for r in _listing(hub_rows)]
    assert ids.count("makermods/act_pick") == 1
    assert "Makermods/ACT_Pick" not in ids


def test_list_all_models_import_of_an_owned_repo_keeps_its_job_id(registry, tmp_lerobot_home) -> None:
    """Importing a repo the user already owns dedupes the ROW away — correctly,
    it is one set of weights — but the import is still the registry record that
    tracks them. Without carrying its id onto the surviving row, every surface
    keying on `job_id` (the models library does) silently lost a card it used to
    show for exactly this shape of import."""
    from makermodslab.models import list_all_models

    _seed_import(registry, "smolvla_imported", hf_repo_id="me/act_pick")
    hub_rows = [{"repo_id": "me/act_pick", "last_modified": None, "private": False}]

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    rows = [r for r in result if (r.get("hf_repo_id") or "") == "me/act_pick"]
    assert len(rows) == 1
    assert rows[0]["job_id"] == "smolvla_imported"


def test_list_all_models_a_training_run_outranks_an_import_for_the_job_id(registry, tmp_lerobot_home) -> None:
    """When both a real run and an import point at one repo, the run wins the
    slot: it trained the weights, while the import's config is a placeholder."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "real_run", repo_id="me/act_pick", state="done")
    _seed_import(registry, "smolvla_imported", hf_repo_id="me/act_pick")
    hub_rows = [{"repo_id": "me/act_pick", "last_modified": None, "private": False}]

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["id"] == "me/act_pick")
    assert row["job_id"] == "real_run"
    assert row["origin"] == "trained-cloud"


def test_list_all_models_import_does_not_duplicate_a_pin_across_casing(registry, tmp_lerobot_home) -> None:
    """An import stores whatever casing the user pasted; a pin is keyed by the
    casing it was saved under. The pin fold matches EXACTLY, so folding imports
    in first left the pin unmatched and the listing showed one repo twice —
    which is why imports fold in after the pins, compared case-insensitively."""
    from makermodslab.models import list_all_models

    _seed_import(registry, "act_imported", hf_repo_id="Acme/Policy")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("makermodslab.models.list_hub_models", return_value=[]))
        stack.enter_context(
            patch("makermodslab.models.get_saved_custom_models", return_value=["acme/policy"])
        )
        stack.enter_context(patch("makermodslab.models.get_hidden_models", return_value=set()))
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()))
        result = list_all_models()

    repos = [(r.get("hf_repo_id") or "").lower() for r in result]
    assert repos.count("acme/policy") == 1


def test_list_all_models_import_does_not_resurrect_a_hidden_repo_across_casing(
    registry, tmp_lerobot_home
) -> None:
    """ "Removed from list" is filtered by exact key. Without a case-insensitive
    check in the imported fold, hiding "Acme/Policy" and then importing
    "acme/policy" walks the hidden repo straight back into the picker — the
    user's removal silently undone by an unrelated action."""
    from makermodslab.models import list_all_models

    _seed_import(registry, "act_imported", hf_repo_id="acme/policy")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("makermodslab.models.list_hub_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_saved_custom_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_hidden_models", return_value={"Acme/Policy"}))
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()))
        result = list_all_models()

    assert [r for r in result if (r.get("hf_repo_id") or "").lower() == "acme/policy"] == []


def test_list_all_models_import_of_a_run_output_dir_is_not_a_second_row(registry, tmp_lerobot_home) -> None:
    """Nothing stops an import being registered against a completed run's own
    output dir. Both sources then resolve the SAME pretrained_model dir, so
    without a path check the listing offers one checkpoint twice under two
    working ids — the precise duplication this whole change exists to remove."""
    _seed_run(registry, "act_pick_2026", policy_type="act", dataset="user/pick", steps=250)
    run_dir = registry._output_root / "act_pick_2026" / "run"
    _seed_import(registry, "act_imported", output_dir=run_dir)

    result = _listing()
    paths = [r["path"] for r in result if r["path"]]
    assert len(paths) == len(set(paths))
    assert "act_imported" not in [r["id"] for r in result]


def test_list_all_models_drops_an_import_whose_directory_lost_its_weights(
    registry, tmp_lerobot_home, tmp_path
) -> None:
    """The import is a pointer, so the directory it names can be moved, emptied
    or deleted afterwards. Offering a path that fails at load is worse than
    omitting it — the same rule the downloaded scan applies to a partial
    download."""
    gone = tmp_path / "moved_away"
    gone.mkdir()
    _seed_import(registry, "act_imported", output_dir=gone)

    assert "act_imported" not in [r["id"] for r in _listing()]


def test_list_all_models_skips_a_superseded_import(registry, tmp_lerobot_home, tmp_path) -> None:
    """One skill per resume chain, represented by its tip — the same rule
    list_local_models applies, applied here so a chain cannot re-enter the
    listing through its imported root."""
    ckpt = _make_model_checkpoint(tmp_path / "elsewhere", "base_policy", shape="tree")
    _seed_import(registry, "act_imported", output_dir=ckpt)
    _seed_run(registry, "act_continued", policy_type="act")
    registry._records["act_continued"].config.resume_from_job_id = "act_imported"

    assert "act_imported" not in [r["id"] for r in _listing()]


class _CountingHubApi:
    """HfApi stand-in that RECORDS repo listings instead of performing them.

    Counting, not raising: `_list_hub_checkpoints` wraps its `list_repo_files`
    call in `except Exception: return []` (a repo may legitimately not exist
    yet, mid-training), so an exploding stub is swallowed and proves exactly
    nothing — a test built on one passes whether or not the Hub was reached.
    The call log is the only honest evidence."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_repo_files(self, repo_id, repo_type):
        self.calls.append(repo_id)
        return []


def test_list_all_models_never_probes_the_hub_for_checkpoint_counts(registry, tmp_lerobot_home) -> None:
    """The listing scans the registry three times (runs, imports, repo naming).
    `JobRegistry.list` stamps `checkpoint_count` on every record it returns, and
    for a cloud or imported-hub record that count is a Hub round-trip — serial,
    and with no deadline of its own. So a user with N cloud runs was paying up
    to N blocking round-trips on any /models call that arrived with a cold 30s
    cache, for a field the listing never reads.

    All three scans now opt out, so a cold cache costs the listing nothing. The
    seeded records cover both shapes that used to probe: a cloud run and an
    imported Hub pointer."""
    from makermodslab.models import list_all_models

    _seed_cloud_run(registry, "cloud_run", repo_id="user/act_cloud_2026-01-01_10-00-00")
    _seed_import(registry, "smolvla_imported", hf_repo_id="someone/smolvla_dishes")
    _seed_run(registry, "act_local", policy_type="act")

    api = _CountingHubApi()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("makermodslab.models.list_hub_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_saved_custom_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_hidden_models", return_value=set()))
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=api))
        result = list_all_models()

    assert api.calls == []
    # ...and the rows those scans exist to produce are still here, so this is a
    # test about not paying for an unread field, not about not scanning.
    ids = {r["id"] for r in result}
    assert "act_local" in ids
    assert "someone/smolvla_dishes" in ids


# ---------------------------------------------------------------------------
# _hub_listing — degraded-Hub status and last-good retention.
# ---------------------------------------------------------------------------


def _hub_call(
    rows: list[dict[str, Any]],
    *,
    authors: tuple[str, ...],
    answered: int,
    authenticated: bool = True,
):
    """A list_hub_models stand-in that also reports how the fan-out went.

    The real function records its outcome on the way out; a patched one has to
    do the same or _hub_listing cannot tell a complete listing from a degraded
    one."""
    import makermodslab.models as m

    def _call():
        m._record_hub_outcome(authenticated=authenticated, authors=authors, answered=answered)
        return rows

    return _call


def _row(repo_id: str) -> dict[str, Any]:
    return {"repo_id": repo_id, "last_modified": None, "private": False}


def test_hub_listing_retains_rows_a_degraded_fan_out_dropped() -> None:
    """A Hub blip used to delete the user's models from the screen: the fan-out
    swallowed the failure, returned fewer rows, and the UI rendered that as the
    truth. The rows are kept, flagged stale, and the status says it is degraded."""
    import makermodslab.models as m

    with patch(
        "makermodslab.models.list_hub_models",
        side_effect=_hub_call([_row("me/act_a")], authors=("me",), answered=1),
    ):
        rows, status = m._hub_listing()
    assert [r["repo_id"] for r in rows] == ["me/act_a"]
    assert status["ok"] is True

    m.invalidate_model_listing_cache()
    with patch("makermodslab.models.list_hub_models", side_effect=_hub_call([], authors=("me",), answered=0)):
        rows, status = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["me/act_a"]
    assert rows[0]["stale"] is True
    assert status["ok"] is False
    assert status["degraded"] is True
    assert status["stale_rows"] is True


def test_hub_listing_does_not_serve_one_accounts_repos_to_another() -> None:
    """The fallback is keyed to the identity its rows were listed for. A stale
    listing is a tolerable degradation; someone else's listing is not."""
    import makermodslab.models as m

    with (
        patch("makermodslab.models.get_token", return_value="alice-token"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("alice/act_a")], authors=("alice",), answered=1),
        ),
    ):
        m._hub_listing()

    m.invalidate_model_listing_cache()
    with (
        patch("makermodslab.models.get_token", return_value="bob-token"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([], authors=("bob",), answered=0),
        ),
    ):
        rows, status = m._hub_listing()

    assert rows == []
    assert status["degraded"] is True
    assert status["stale_rows"] is False


# ---------------------------------------------------------------------------
# list_cloud_models — cloud runs are not gated on the Hub listing.
# ---------------------------------------------------------------------------


def test_list_all_models_lists_a_finished_cloud_run_before_the_hub_does(registry, tmp_lerobot_home) -> None:
    """The cloud gap. A cloud run entered the listing ONLY via list_hub_models,
    so a run that had finished, been recorded done, and had its repo pushed did
    not exist as a skill until the per-author fan-out happened to return that
    repo — while the Train panel already showed it finished. The registry knows
    the run ended and which repo it published to; that is enough to list it."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_done", repo_id=repo, state="done")

    # The Hub returns NOTHING — the push may still be settling, or the fan-out
    # may have been cut short.
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["id"] == repo)
    assert row["origin"] == "trained-cloud"
    assert row["job_id"] == "cloud_done"
    assert row["dataset"] == "makermods/eraser"


def test_list_skills_marks_a_registry_stamped_cloud_run_unverified(registry, tmp_lerobot_home) -> None:
    """Listing it is a claim about the RUN, not about the bytes. Nothing was
    probed, so the row says so — and is still offered, because the download the
    deploy path runs anyway is what settles it, loudly and once."""
    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_done", repo_id=repo, state="done")

    row = next(r for r in _skills()["skills"] if r["id"] == repo)
    assert row["weights"] == "unverified"
    assert row["deployable"] is True


def test_list_all_models_cloud_row_does_not_duplicate_its_hub_row(registry, tmp_lerobot_home) -> None:
    """Once the Hub DOES return the repo, the registry-stamped row must step
    aside — the Hub listing adds detail (privacy, real last-modified) to a row
    whose existence it no longer gates, rather than producing a second one."""
    from makermodslab.models import list_all_models

    # Deliberately different casing on the two sides: the runner stores whatever
    # repo id it was configured with, the Hub returns its canonical spelling, and
    # HF repo ids are unique case-insensitively. An exact compare here lists one
    # repo twice.
    repo = "Me/ACT_Cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_done", repo_id=repo, state="done")
    hub_rows = [{"repo_id": "me/act_cloud_2026-01-01_10-00-00", "last_modified": None, "private": True}]

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    rows = [r for r in result if (r.get("hf_repo_id") or "").lower() == repo.lower()]
    assert len(rows) == 1
    assert rows[0]["private"] is True
    assert rows[0]["job_id"] == "cloud_done"
    assert rows[0]["origin"] == "trained-cloud"


def test_list_all_models_skips_a_cloud_run_that_is_still_training(registry, tmp_lerobot_home) -> None:
    """Only terminal runs. A running job's repo may hold nothing yet, and the
    Train panel is where a run in progress belongs."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_running", repo_id=repo, state="running")

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        assert repo not in {r["id"] for r in list_all_models()}


def test_list_all_models_skips_a_superseded_cloud_run(registry, tmp_lerobot_home) -> None:
    """One skill per resume chain. A cloud resume republishes into run #1's
    repo, so without this the chain would list under a link rather than its
    tip."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_parent", repo_id=repo, state="done", started_at=100.0)
    _seed_cloud_run(registry, "cloud_tip", repo_id=repo, state="running", started_at=200.0)
    registry._records["cloud_tip"].config.resume_from_job_id = "cloud_parent"

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        assert repo not in {r["id"] for r in list_all_models()}


def test_list_all_models_does_not_resurrect_a_hidden_cloud_repo(registry, tmp_lerobot_home) -> None:
    """ "Removed from list" survives the new source: the fold consults the hidden
    set itself, because it runs before the listing's own exact-key filter."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_done", repo_id=repo, state="done")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("makermodslab.models.list_hub_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_saved_custom_models", return_value=[]))
        # Hidden under the casing the user hid it with, published under another.
        stack.enter_context(
            patch("makermodslab.models.get_hidden_models", return_value={"Me/ACT_Cloud_2026-01-01_10-00-00"})
        )
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()))
        result = list_all_models()

    assert repo not in {r["id"] for r in result}


def _cloud_ids(hub_rows: list[dict[str, Any]] | None = None) -> set[str]:
    from makermodslab.models import invalidate_model_listing_cache, list_all_models

    # Each call is a fresh build: the merged listing is cached, and these tests
    # deliberately ask the same question twice with different Hub answers.
    invalidate_model_listing_cache()
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows or []):
            stack.enter_context(cm)
        return {r["id"] for r in list_all_models()}


def test_list_all_models_lists_an_interrupted_cloud_run_that_trained(registry, tmp_lerobot_home) -> None:
    """Pins the STATE TUPLE, which nothing else did. Both reviewers found the
    same escape: mutate the gate to `state == "done"` and every other cloud test
    still passed while interrupted tips silently vanished."""
    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_stopped", repo_id=repo, state="interrupted")
    registry._records["cloud_stopped"].metrics.current_step = 4000

    assert repo in _cloud_ids()


def test_list_all_models_skips_an_interrupted_cloud_run_that_never_stepped(
    registry, tmp_lerobot_home
) -> None:
    """The evidence gate. `hf_repo_id` is stamped at SUBMIT and the cloud wrapper
    creates the repo before training starts, so the repo existing proves nothing
    — a run stopped before its first step pushed nothing, and offering it would
    be a picker entry that fails only when the user tries to run it."""
    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_stillborn", repo_id=repo, state="interrupted")
    registry._records["cloud_stillborn"].metrics.current_step = 0

    assert repo not in _cloud_ids()


def test_list_all_models_does_not_registry_stamp_a_failed_cloud_run(registry, tmp_lerobot_home) -> None:
    """A failed LOCAL run never enters /models at all, and reaches /skills only
    after its weights are verified on disk. Nothing equivalent is checkable for
    a cloud run without the Hub call this source exists to avoid, so `failed`
    gets no registry shortcut — it still appears the moment the Hub listing
    returns its repo."""
    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_failed", repo_id=repo, state="failed")
    registry._records["cloud_failed"].metrics.current_step = 9000

    assert repo not in _cloud_ids()
    # ...but the Hub returning it is still enough.
    assert repo in _cloud_ids([{"repo_id": repo, "last_modified": None, "private": False}])


def test_list_all_models_cloud_chain_row_carries_the_tips_job_id(registry, tmp_lerobot_home) -> None:
    """A realistic chain through the fold, which no test covered. A cloud resume
    republishes into run #1's repo, so both links name one repo and the ranking
    has to elect the tip — otherwise the row deploys an earlier link than the
    one that finished."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "link_1", repo_id=repo, state="interrupted", started_at=100.0)
    registry._records["link_1"].metrics.current_step = 1500
    _seed_cloud_run(registry, "link_2_tip", repo_id=repo, state="done", started_at=200.0)
    registry._records["link_2_tip"].config.resume_from_job_id = "link_1"

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        result = list_all_models()

    rows = [r for r in result if (r.get("hf_repo_id") or "") == repo]
    assert len(rows) == 1
    assert rows[0]["job_id"] == "link_2_tip"


def test_list_all_models_elects_the_tip_when_a_done_parent_has_a_child(registry, tmp_lerobot_home) -> None:
    """Leaf-first ranking, which nothing else exercises.

    `JobRegistry.start` refuses to resume a "done" run, so within a chain the
    done record IS the tip and the done/leaf rules agree — this state is only
    reachable from a registry written by an older version or edited by hand.
    Without leaf-first the done parent is elected, then dropped by the leaf
    filter in list_cloud_models, and the chain vanishes from the listing
    entirely rather than falling through to its tip."""
    from makermodslab.models import list_all_models

    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "done_parent", repo_id=repo, state="done", started_at=100.0)
    _seed_cloud_run(registry, "real_tip", repo_id=repo, state="interrupted", started_at=200.0)
    registry._records["real_tip"].metrics.current_step = 7000
    registry._records["real_tip"].config.resume_from_job_id = "done_parent"

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if (r.get("hf_repo_id") or "") == repo)
    assert row["job_id"] == "real_tip"


def test_hub_listing_labels_rows_with_the_identity_they_were_fetched_under() -> None:
    """Capturing the identity BEFORE the fan-out and filing the result under it
    is a race: sign out of Alice and into Bob while the fetch is in flight and
    Bob's repos get stored under Alice's key, to be handed back to Alice on her
    next visit. The credentials in play when the rows were produced are the
    honest label."""
    import makermodslab.models as m

    # "alice" for the pre-fetch read, "bob" for every read after: the switch
    # lands while the fan-out is running.
    reads = {"n": 0}

    def switching_token():
        reads["n"] += 1
        return "alice-token" if reads["n"] == 1 else "bob-token"

    with (
        patch("makermodslab.models.get_token", side_effect=switching_token),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("bob/act_b")], authors=("bob",), answered=1),
        ),
    ):
        rows, _ = m._hub_listing()
    assert [r["repo_id"] for r in rows] == ["bob/act_b"]

    # Alice comes back. The cache is warm and inside its TTL, but it belongs to
    # Bob — she must get her own listing, not his.
    with (
        patch("makermodslab.models.get_token", return_value="alice-token"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("alice/act_a")], authors=("alice",), answered=1),
        ),
    ):
        rows, _ = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["alice/act_a"]


def test_list_skills_marks_a_cloud_repo_continued_by_a_local_run_superseded(
    registry, tmp_lerobot_home
) -> None:
    """The reachable superseded case, which the earlier test missed by building
    an API-impossible one (resuming a `done` run is refused outright).

    A CLOUD run continued by a LOCAL one keeps the repo on the parent — the
    local child has none of its own. So once the Hub returns that repo it lands
    beside the local tip's row: two picker entries for one trained model. The
    repo row has to say it is a chain link."""
    repo = "me/act_cloud_2026-01-01_10-00-00"
    _seed_cloud_run(registry, "cloud_parent", repo_id=repo, state="interrupted", started_at=100.0)
    registry._records["cloud_parent"].metrics.current_step = 1500
    _seed_run(registry, "local_tip", policy_type="act", ended_at=900.0)
    registry._records["local_tip"].config.resume_from_job_id = "cloud_parent"

    skills = _skills([{"repo_id": repo, "last_modified": None, "private": False}])["skills"]

    repo_row = next(r for r in skills if r["id"] == repo)
    assert repo_row["superseded_by"] == "local_tip"
    assert repo_row["deployable"] is False
    # The local tip is the one that runs.
    assert next(r for r in skills if r["id"] == "local_tip")["deployable"] is True


def test_hub_listing_survives_a_whoami_blip_without_wiping_the_fallback() -> None:
    """`cached_whoami` swallows transport failures and returns None, so a token
    holder whose identity check blips looked exactly like a signed-out user.
    That was worse than the bug this fallback exists to fix: every Hub row
    vanished, the status still claimed `ok`, and the last-good listing was
    overwritten with the empty result — destroying the safety net precisely when
    it was needed."""
    import makermodslab.models as m

    with (
        patch("makermodslab.models.get_token", return_value="tok"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("me/act_a")], authors=("me",), answered=1),
        ),
    ):
        m._hub_listing()

    m.invalidate_model_listing_cache()
    # Token still present; whoami failed, so the listing reports unauthenticated
    # with no authors and returns nothing.
    with (
        patch("makermodslab.models.get_token", return_value="tok"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([], authenticated=False, authors=(), answered=0),
        ),
    ):
        rows, status = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["me/act_a"]
    assert rows[0]["stale"] is True
    assert status["ok"] is False
    assert m._hub_last_good, "the fallback must survive the blip that needed it"


def test_forget_hub_repo_is_not_undone_by_a_fetch_already_in_flight() -> None:
    """The delete clears the fallback, but a fan-out that STARTED before it
    finishes afterwards holding pre-deletion data — and would write the deleted
    repo straight back in. Modelled exactly: the delete lands mid-fetch, and the
    listing still reports the repo because it read the Hub before it happened."""
    import makermodslab.models as m

    def listing_that_is_overtaken_by_a_delete():
        # Happens while the fan-out is in flight.
        m.forget_hub_repo("me/act_gone")
        m._record_hub_outcome(authenticated=True, authors=("me",), answered=1)
        return [_row("me/act_gone"), _row("me/act_kept")]

    with (
        patch("makermodslab.models.get_token", return_value="tok"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=listing_that_is_overtaken_by_a_delete,
        ),
    ):
        rows, _ = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["me/act_kept"]
    assert [r["repo_id"] for r in (m._hub_last_good or [])] == ["me/act_kept"]


def test_forget_hub_repo_is_retired_when_the_repo_comes_back() -> None:
    """The tombstone is not permanent. A listing that BEGAN after the delete and
    still returns the repo is evidence it exists again — recreated, or the Hub
    was eventually consistent — so the row is honoured rather than suppressed
    forever."""
    import makermodslab.models as m

    m.forget_hub_repo("me/act_gone")

    with (
        patch("makermodslab.models.get_token", return_value="tok"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("me/act_gone")], authors=("me",), answered=1),
        ),
    ):
        rows, _ = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["me/act_gone"]
    assert "me/act_gone" not in m._forgotten_hub_repos


def test_list_all_models_hidden_repo_does_not_return_under_another_casing(registry, tmp_lerobot_home) -> None:
    """Hiding stores whatever casing the UI had; the Hub returns its canonical
    one. An exact-only final filter let the repo walk straight back in."""
    from makermodslab.models import list_all_models

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "makermodslab.models.list_hub_models",
                return_value=[{"repo_id": "me/act_pick", "last_modified": None, "private": False}],
            )
        )
        stack.enter_context(patch("makermodslab.models.get_saved_custom_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_hidden_models", return_value={"Me/ACT_Pick"}))
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()))
        result = list_all_models()

    assert [r for r in result if (r.get("hf_repo_id") or "").lower() == "me/act_pick"] == []


# ---------------------------------------------------------------------------
# list_skills — the deployable projection.
# ---------------------------------------------------------------------------


def _skills(hub_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from makermodslab.models import list_skills

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows or []):
            stack.enter_context(cm)
        return list_skills()


def test_list_skills_marks_a_local_run_deployable(registry, tmp_lerobot_home) -> None:
    """The baseline: a completed local run with weights on disk is deployable,
    carries the job id that deploys it, and says where it came from."""
    _seed_run(registry, "act_pick_2026", policy_type="act", dataset="user/pick", steps=250)

    row = next(r for r in _skills()["skills"] if r["id"] == "act_pick_2026")
    assert row["deployable"] is True
    assert row["weights"] == "ready"
    assert row["origin"] == "trained-local"
    assert row["job_id"] == "act_pick_2026"
    assert row["superseded_by"] is None


def test_list_skills_lists_a_failed_run_that_saved_weights(registry, tmp_lerobot_home) -> None:
    """/models excludes a failed run on principle. The Train panel's card has
    always run one anyway (its Run row gates on having a checkpoint, not on
    state), so the principle was enforced in one panel and contradicted in the
    next. An OOM at step 90k with a good 80k checkpoint is a usable skill: it is
    listed and deployable, and keeps `state: "failed"` so the UI can badge it."""
    from makermodslab.models import list_all_models

    _seed_run(registry, "act_oom", policy_type="act", state="failed")

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        assert "act_oom" not in {r["id"] for r in list_all_models()}

    row = next(r for r in _skills()["skills"] if r["id"] == "act_oom")
    assert row["state"] == "failed"
    assert row["deployable"] is True
    assert row["weights"] == "ready"


def test_list_skills_explains_a_superseded_run_instead_of_hiding_it(registry, tmp_lerobot_home) -> None:
    """A resumed run collapses into its chain's tip, which is right — but the
    picker simply omitted it, so "my run finished and it is not in the list" had
    no answer. It is listed, not deployable, and names the run that represents
    it."""
    _seed_run(registry, "act_parent", policy_type="act")
    _seed_run(registry, "act_tip", policy_type="act")
    registry._records["act_tip"].config.resume_from_job_id = "act_parent"

    skills = _skills()["skills"]
    parent = next(r for r in skills if r["id"] == "act_parent")
    assert parent["superseded_by"] == "act_tip"
    assert parent["deployable"] is False
    # The tip still stands for the chain and is the one that runs.
    assert next(r for r in skills if r["id"] == "act_tip")["deployable"] is True


def test_list_skills_keeps_a_broken_import_visible_but_unrunnable(
    registry, tmp_lerobot_home, tmp_path
) -> None:
    """An import is a pointer, so its directory can be moved or emptied later.
    The listing drops such a row rather than offer a path that fails at load —
    but dropping it from the LIBRARY too would leave a registered record the
    user can see nowhere and therefore delete nowhere. It is listed with no
    weights: out of the picker, still cleanable."""
    gone = tmp_path / "moved_away"
    gone.mkdir()
    _seed_import(registry, "act_imported", output_dir=gone, name="Old import")

    row = next(r for r in _skills()["skills"] if r["id"] == "act_imported")
    assert row["weights"] == "none"
    assert row["deployable"] is False
    assert row["origin"] == "imported"
    assert row["name"] == "Old import"


def test_list_skills_marks_a_hub_only_row_unverified_not_ready(registry, tmp_lerobot_home) -> None:
    """A Hub repo is never probed at listing time to confirm its push landed —
    that is a serial round-trip per row. It claims `unverified`, and the
    download the deploy path runs anyway settles it."""
    hub_rows = [{"repo_id": "user/act_sock_2026-01-01_10-00-00", "last_modified": None, "private": False}]
    row = next(r for r in _skills(hub_rows)["skills"] if r["id"] == hub_rows[0]["repo_id"])
    assert row["weights"] == "unverified"
    assert row["deployable"] is True
    assert row["origin"] == "hub-untracked"


def test_list_skills_reports_hub_reachability(registry, tmp_lerobot_home) -> None:
    """The envelope's whole reason to exist: "the Hub was unreachable" and "you
    own no skills" used to render identically, as an empty list."""
    assert _skills()["hub"]["ok"] is True

    from makermodslab.models import invalidate_model_listing_cache, list_skills

    # The status rides the merged cache with the rows it describes — one build,
    # one snapshot, so a row and the reachability claim about it can never come
    # from different moments. Drop it to observe a second, different build.
    invalidate_model_listing_cache()

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        stack.enter_context(
            patch(
                "makermodslab.models._hub_listing",
                return_value=(
                    [],
                    {"ok": False, "authenticated": True, "degraded": True, "stale_rows": False},
                ),
            )
        )
        envelope = list_skills()

    assert envelope["hub"]["ok"] is False
    assert envelope["hub"]["degraded"] is True


def test_list_skills_does_not_resurrect_a_hidden_superseded_run(registry, tmp_lerobot_home) -> None:
    """The explained rows are folded in after the merge, so they bypass the
    listing's own hidden filter and have to honour it themselves."""
    _seed_run(registry, "act_parent", policy_type="act")
    _seed_run(registry, "act_tip", policy_type="act")
    registry._records["act_tip"].config.resume_from_job_id = "act_parent"

    from makermodslab.models import list_skills

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("makermodslab.models.list_hub_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_saved_custom_models", return_value=[]))
        stack.enter_context(patch("makermodslab.models.get_hidden_models", return_value={"act_parent"}))
        stack.enter_context(patch("makermodslab.jobs.shared_hf_api", return_value=_NoHubFiles()))
        skills = list_skills()["skills"]

    assert "act_parent" not in {r["id"] for r in skills}


def test_hub_listing_cache_hit_does_not_serve_another_accounts_rows() -> None:
    """The identity check has to sit on the cache-HIT path too. Guarding only
    the last-good fallback still left a 45s window in which signing out of one
    account and into another showed the previous account's repos."""
    import makermodslab.models as m

    with (
        patch("makermodslab.models.get_token", return_value="alice-token"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("alice/act_a")], authors=("alice",), answered=1),
        ),
    ):
        rows, _ = m._hub_listing()
    assert [r["repo_id"] for r in rows] == ["alice/act_a"]

    # No invalidation: the cache is warm and well inside its TTL. Only the
    # credentials changed, which is what an account switch IS.
    with (
        patch("makermodslab.models.get_token", return_value="bob-token"),
        patch(
            "makermodslab.models.list_hub_models",
            side_effect=_hub_call([_row("bob/act_b")], authors=("bob",), answered=1),
        ),
    ):
        rows, _ = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["bob/act_b"]


def test_forget_hub_repo_drops_it_from_the_last_good_fallback() -> None:
    """Retention is a net for rows we failed to SEE. A repo the user deleted is
    not one of those: leaving it in the fallback resurrects it on every later
    degraded fan-out, forever."""
    import makermodslab.models as m

    with patch(
        "makermodslab.models.list_hub_models",
        side_effect=_hub_call([_row("me/act_a"), _row("me/act_b")], authors=("me",), answered=1),
    ):
        m._hub_listing()

    m.forget_hub_repo("me/act_a")

    with patch(
        "makermodslab.models.list_hub_models",
        side_effect=_hub_call([], authors=("me",), answered=0),
    ):
        rows, status = m._hub_listing()

    assert [r["repo_id"] for r in rows] == ["me/act_b"]
    assert status["degraded"] is True


def test_list_all_models_newest_run_owns_a_shared_repo_row(registry, tmp_lerobot_home) -> None:
    """Several local runs can publish to one repo. The collapse overwrites
    unconditionally and `local` is newest-first, so without a first-claim guard
    the OLDEST run processed last would end up owning the row — and `job_id`
    would name a run the listing does not mean."""
    from makermodslab.models import list_all_models

    repo = "me/act_shared"
    _seed_run(registry, "act_old", policy_type="act", hf_repo_id=repo, ended_at=100.0)
    _seed_run(registry, "act_new", policy_type="act", hf_repo_id=repo, ended_at=900.0)

    hub_rows = [{"repo_id": repo, "last_modified": None, "private": False}]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r.get("hf_repo_id") == repo and r["source"] == "both")
    assert row["job_id"] == "act_new"


def test_list_skills_will_not_call_a_config_only_checkpoint_ready(registry, tmp_lerobot_home) -> None:
    """A path is not weights. The registry accepts a checkpoint dir on its
    config.json alone, but loading needs model.safetensors — a run killed
    mid-save leaves exactly the first without the second, and calling that
    deployable offers a skill that cannot start."""
    pretrained = _seed_run(registry, "act_halfsaved", policy_type="act", state="failed")
    (pretrained / "model.safetensors").unlink()

    row = next(r for r in _skills()["skills"] if r["id"] == "act_halfsaved")
    assert row["weights"] == "none"
    assert row["deployable"] is False


def test_list_skills_does_not_list_a_pushed_failed_run_twice(registry, tmp_lerobot_home) -> None:
    """A failed run that was pushed already reaches the merge as a repo-keyed
    row — id is the repo, path is null — so neither the id nor the path check
    can see it, and it came back a second time as an explained row."""
    repo = "me/act_failed"
    _seed_run(registry, "act_failed", policy_type="act", state="failed", hf_repo_id=repo)

    hub_rows = [{"repo_id": repo, "last_modified": None, "private": False}]
    skills = _skills(hub_rows)["skills"]

    assert len([r for r in skills if r.get("hf_repo_id") == repo]) == 1
    assert "act_failed" not in {r["id"] for r in skills}


def test_list_all_models_import_deduped_by_path_keeps_its_job_id(registry, tmp_lerobot_home) -> None:
    """The other way an import's row is deduped away: it points AT a checkpoint
    the downloaded-model scan already found, so the path check drops it. That
    scanned row carries no job_id of its own, so unless the import's id is
    stamped onto it the record becomes unreachable — and the library, which
    keys on job_id, loses a card it used to show."""
    from makermodslab.models import list_all_models

    ckpt = _make_model_checkpoint(tmp_lerobot_home / "makermodslab_models", "user/policy")
    _seed_import(registry, "act_imported", output_dir=ckpt)

    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing([]):
            stack.enter_context(cm)
        result = list_all_models()

    rows = [r for r in result if r.get("path") == str(ckpt)]
    assert len(rows) == 1, "one checkpoint, one row"
    assert rows[0]["job_id"] == "act_imported"


# ---------------------------------------------------------------------------
# Downloaded / imported local models — the local models dir scan + probe.
# ---------------------------------------------------------------------------


def _make_model_checkpoint(
    root: Path, repo_id: str, shape: str = "root", step: int = 500, policy_type: str = "act"
) -> Path:
    """Fabricate a checkpoint dir in one of the two recognized shapes: a root
    config.json ("root", what upload_local_model pushes) or a
    checkpoints/<step>/pretrained_model tree ("tree").

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


def test_list_all_models_hub_rows_carry_policy_type(registry, tmp_lerobot_home) -> None:
    """A hub row's policy type survives the merge into the listing.

    Sandboxed like every other list_all_models test: patching only
    list_hub_models left the pin/hidden files and the downloaded-model scan
    pointed at the DEVELOPER'S real ~/.cache, so the assertion passed or failed
    according to what the person running it happened to have pinned. It also
    asserted on result[0], which is a claim about sort order, not about policy
    types — the row is now selected by the repo it belongs to."""
    from makermodslab.models import list_all_models

    repo_id = "user/act_sock_2026-01-01_10-00-00"
    hub_rows = [
        {
            "repo_id": repo_id,
            "last_modified": None,
            "private": False,
            "policy_type": "act",
        },
    ]
    with contextlib.ExitStack() as stack:
        for cm in _sandboxed_listing(hub_rows):
            stack.enter_context(cm)
        result = list_all_models()

    row = next(r for r in result if r["id"] == repo_id)
    assert row["policy_type"] == "act"


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
