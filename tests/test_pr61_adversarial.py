"""Adversarial edge-case suite for PR 61 (multi-checkpoint publish to one Hub repo).

Fabricates local runs in shapes the PR's own tests don't cover: zero-padded and
unpadded checkpoint dirs, many checkpoints, no checkpoints, a run that is also
downloaded back from the Hub, and mid-queue Hub failures.

Nothing here touches the real Hub or the real filesystem outside tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_model_cache():
    import makermodslab.models as m

    m.invalidate_model_listing_cache()
    yield
    m.invalidate_model_listing_cache()


@pytest.fixture
def registry(tmp_path: Path):
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "outputs" / "train")
    reg.shutdown()
    with patch("makermodslab.models.job_registry", reg), patch("makermodslab.jobs.job_registry", reg):
        yield reg


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

    The row's id is the RUN id, so BOTH delete actions send the same request —
    deleteModel(row.id) — which removes the whole training run. The row must
    therefore keep local_kind "run", so resolveDeleteAction picks the
    destructive dialog instead of the reassuring "the Hub copy stays" one.

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

    with (
        patch("makermodslab.models.rollout")
        if False
        else patch("makermodslab.models._model_in_use", return_value=None)
    ):
        delete_local_model(row["id"])

    # "The Hub copy stays" was true. "Only the local copy is removed" was not.
    assert not unpublished.exists(), "unpublished step 200 survived"
    assert not run_dir.parent.exists(), "run dir survived"
