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
"""Tests for makermodslab.jobs — parsers and Pydantic models. Does not exercise
LocalJobRunner.start() (see plan, "Discovered issue")."""

from __future__ import annotations

import json as _json
import os
import threading
from pathlib import Path

import pytest


def _make_checkpoint(
    output_dir: Path,
    step: int,
    *,
    with_state: bool = True,
    with_optimizer: bool = True,
) -> None:
    """Lay out a lerobot-style checkpoint under <output_dir>/checkpoints/<step>.

    `with_state=False` is the weights-only shape (an imported model);
    `with_optimizer=False` is the interrupted-save shape the cloud uploader
    used to publish — training_state/ exists but the big optimizer file that
    lerobot writes last never landed.
    """
    ck = output_dir / "checkpoints" / str(step)
    pm = ck / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text("{}")  # required by _list_local_checkpoints
    (pm / "train_config.json").write_text("{}")
    (pm / "model.safetensors").write_bytes(b"weights")
    if with_state:
        ts = ck / "training_state"
        ts.mkdir()
        (ts / "training_step.json").write_text("{}")
        (ts / "rng_state.safetensors").write_bytes(b"rng")
        if with_optimizer:
            (ts / "optimizer_state.safetensors").write_bytes(b"optim")


def _record(output_dir: Path, runner: str = "local"):
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    return JobRecord(
        id="job-1",
        name="run",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/ds"),
        output_dir=str(output_dir),
        started_at=0.0,
        runner=runner,
    )


def test_resolve_resume_config_path_returns_train_config(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 5000)
    path = _resolve_resume_config_path(_record(out), 5000)
    assert path.endswith("checkpoints/5000/pretrained_model/train_config.json")


def test_resolve_resume_config_path_defaults_to_latest(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 1000)
    _make_checkpoint(out, 3000)
    path = _resolve_resume_config_path(_record(out), None)  # None ⇒ latest
    assert "checkpoints/3000/" in path


def test_resolve_resume_config_path_rejects_missing_training_state(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_state=False)  # weights-only (e.g. imported)
    with pytest.raises(ValueError, match="training_state"):
        _resolve_resume_config_path(_record(out), 2000)


def test_resolve_resume_config_path_rejects_interrupted_save(tmp_path) -> None:
    """training_state/ exists but the optimizer file lerobot writes last never
    landed — the shape the cloud uploader used to publish. It must be refused
    at the API with the remedy named, not accepted and crashed on inside the
    trainer."""
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_optimizer=False)
    with pytest.raises(ValueError, match="incomplete") as excinfo:
        _resolve_resume_config_path(_record(out), 2000)
    assert "optimizer_state.safetensors" in str(excinfo.value)
    assert "fine-tune from its weights" in str(excinfo.value)


def test_resolve_resume_config_path_rejects_non_local(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="local"):
        _resolve_resume_config_path(_record(out, runner="hf_cloud"), 2000)


def test_resolve_resume_config_path_rejects_unknown_step(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_resume_config_path(_record(out), 9999)


def _cloud_record(repo_id: str | None = "user/act_ds_2026", state: str = "failed"):
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    return JobRecord(
        id="cloud-1",
        name="run",
        state=state,
        config=TrainingRequest(dataset_repo_id="user/ds", steps=10000),
        output_dir="",
        started_at=0.0,
        runner="hf_cloud",
        hf_repo_id=repo_id,
    )


class _FakeHubApi:
    """Minimal HfApi stand-in: returns a fixed repo file listing."""

    def __init__(self, files: list[str]) -> None:
        self._files = files

    def list_repo_files(self, repo_id, repo_type):
        return self._files


def _hub_checkpoint_files(step_dir: str, *, with_optimizer: bool = True) -> list[str]:
    """The repo paths a COMPLETE cloud checkpoint publishes (or, without the
    optimizer file, the partial tree a mid-save upload used to seal)."""
    files = [
        f"checkpoints/{step_dir}/pretrained_model/config.json",
        f"checkpoints/{step_dir}/pretrained_model/model.safetensors",
        f"checkpoints/{step_dir}/pretrained_model/train_config.json",
        f"checkpoints/{step_dir}/training_state/training_step.json",
        f"checkpoints/{step_dir}/training_state/rng_state.safetensors",
    ]
    if with_optimizer:
        files.append(f"checkpoints/{step_dir}/training_state/optimizer_state.safetensors")
    return files


def _hub_pretrained_files(step_dir: str) -> list[str]:
    """The repo paths a WEIGHTS-ONLY staging upload publishes — the fine-tune
    base half of the tree above, with no training_state/ at all."""
    return [
        f"checkpoints/{step_dir}/pretrained_model/config.json",
        f"checkpoints/{step_dir}/pretrained_model/model.safetensors",
    ]


def test_resolve_cloud_resume_returns_repo_and_step_dir(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000"))
    )
    repo_id, step_dir = _resolve_cloud_resume(_cloud_record(), 5000)
    assert repo_id == "user/act_ds_2026"
    assert step_dir == "005000"  # zero-padded dir name preserved


def test_resolve_cloud_resume_defaults_to_latest(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("003000")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    _repo, step_dir = _resolve_cloud_resume(_cloud_record(), None)  # None ⇒ latest
    assert step_dir == "003000"


def test_resolve_cloud_resume_rejects_partial_hub_checkpoint(monkeypatch) -> None:
    """The NEW-17 shape: everything on the Hub except the optimizer file the
    uploader raced. `training_state/training_step.json` alone used to pass this
    guard, so the run died inside the trainer on a FileNotFoundError instead of
    at the API with something the user can act on."""
    from makermodslab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("005000", with_optimizer=False)
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="incomplete on the Hub") as excinfo:
        _resolve_cloud_resume(_cloud_record(), 5000)
    message = str(excinfo.value)
    assert "uploader race" in message
    assert "training_state/optimizer_state.safetensors" in message
    assert "fine-tune from its weights" in message  # the named remedy


def test_resolve_cloud_resume_ignores_other_steps_when_checking_completeness(
    monkeypatch,
) -> None:
    """Completeness is judged per step: a complete 001000 must not vouch for a
    partial 003000 (the file listing is repo-wide and flat)."""
    from makermodslab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("003000", with_optimizer=False)
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="incomplete on the Hub"):
        _resolve_cloud_resume(_cloud_record(), 3000)


def test_resolve_cloud_resume_rejects_no_checkpoints(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(["README.md"]))
    with pytest.raises(ValueError, match="died before its first save"):
        _resolve_cloud_resume(_cloud_record(), None)


def test_resolve_cloud_resume_rejects_missing_training_state(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    # Weights present but no training_state/ on the Hub ⇒ not resumable.
    files = ["checkpoints/005000/pretrained_model/config.json"]
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="training_state"):
        _resolve_cloud_resume(_cloud_record(), 5000)


def test_resolve_cloud_resume_rejects_unknown_step(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000"))
    )
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_cloud_resume(_cloud_record(), 9999)


def test_resolve_cloud_resume_rejects_non_cloud(tmp_path) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="cloud"):
        _resolve_cloud_resume(_record(tmp_path, runner="local"), None)


def test_resolve_cloud_resume_rejects_missing_repo() -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="no output repo"):
        _resolve_cloud_resume(_cloud_record(repo_id=None), None)


# ---------------------------------------------------------------------------
# Checkpoint completeness — the single readiness rule shared by both resume
# guards above and (inlined verbatim) by the in-container cloud uploader.
# ---------------------------------------------------------------------------


def _complete_names() -> set[str]:
    """Every file a COMPLETE, resumable checkpoint tree holds."""
    return {
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json",
        "training_state/training_step.json",
        "training_state/rng_state.safetensors",
        "training_state/optimizer_state.safetensors",
    }


def test_missing_checkpoint_files_accepts_a_complete_tree() -> None:
    from makermodslab.jobs import missing_checkpoint_files

    assert missing_checkpoint_files(_complete_names()) == []


def test_missing_checkpoint_files_does_not_require_a_scheduler() -> None:
    """save_training_state writes scheduler_state.json only `if scheduler is not
    None`, so requiring it would permanently block scheduler-less presets."""
    from makermodslab.jobs import missing_checkpoint_files

    assert "training_state/scheduler_state.json" not in _complete_names()
    assert missing_checkpoint_files(_complete_names()) == []


def test_missing_checkpoint_files_flags_a_mid_save_snapshot() -> None:
    """config.json is the FIRST artifact lerobot writes — on its own it means a
    save just started, not a checkpoint."""
    from makermodslab.jobs import missing_checkpoint_files

    missing = missing_checkpoint_files({"pretrained_model/config.json"})
    assert "pretrained_model/*.safetensors" in missing
    assert "training_state/training_step.json" in missing
    assert "training_state/optimizer_state.safetensors" in missing


def test_missing_checkpoint_files_flags_the_optimizer_file_alone() -> None:
    from makermodslab.jobs import missing_checkpoint_files

    names = _complete_names() - {"training_state/optimizer_state.safetensors"}
    assert missing_checkpoint_files(names) == ["training_state/optimizer_state.safetensors"]


def test_missing_checkpoint_files_accepts_nested_multi_optimizer_state() -> None:
    """A MultiAdam policy writes training_state/<name>/optimizer_state.safetensors,
    so the optimizer probe must match at any depth or such runs would never be
    considered ready."""
    from makermodslab.jobs import missing_checkpoint_files

    names = (_complete_names() - {"training_state/optimizer_state.safetensors"}) | {
        "training_state/actor/optimizer_state.safetensors",
        "training_state/critic/optimizer_state.safetensors",
    }
    assert missing_checkpoint_files(names) == []


def test_missing_checkpoint_files_accepts_a_peft_adapter_as_weights() -> None:
    from makermodslab.jobs import missing_checkpoint_files

    names = (_complete_names() - {"pretrained_model/model.safetensors"}) | {
        "pretrained_model/adapter_model.safetensors"
    }
    assert missing_checkpoint_files(names) == []


def test_scan_checkpoint_dir_reports_relative_names_and_a_change_sensitive_fingerprint(
    tmp_path,
) -> None:
    from makermodslab.jobs import missing_checkpoint_files, scan_checkpoint_dir

    _make_checkpoint(tmp_path, 1000)
    ck = tmp_path / "checkpoints" / "1000"

    names, fingerprint = scan_checkpoint_dir(ck)
    assert "training_state/optimizer_state.safetensors" in names  # posix, relative
    assert missing_checkpoint_files(names) == []
    assert scan_checkpoint_dir(ck)[1] == fingerprint  # stable while nothing writes

    (ck / "training_state" / "optimizer_state.safetensors").write_bytes(b"grown-larger")
    assert scan_checkpoint_dir(ck)[1] != fingerprint  # a byte written moves it


# ── the weights-only completeness rule, for a fine-tune base (F7's fourth
# quadrant) ─────────────────────────────────────────────────────────────────
# A fine-tune reads pretrained_model/ and nothing else, so the staged copy of a
# local base is weights-only and needs its own completeness rule. Judging it by
# the resume rule would declare every staging upload broken.


def test_missing_pretrained_files_accepts_the_weights_half_alone() -> None:
    from makermodslab.jobs import missing_pretrained_files

    names = {n for n in _complete_names() if n.startswith("pretrained_model/")}
    assert missing_pretrained_files(names) == []


def test_missing_pretrained_files_does_not_require_train_config() -> None:
    """A flat Hub-imported base (laid out by push_to_hub, not by a checkpoint
    save) legitimately has no train_config.json — requiring it would refuse the
    canonical SmolVLA-style base."""
    from makermodslab.jobs import missing_pretrained_files

    assert (
        missing_pretrained_files({"pretrained_model/config.json", "pretrained_model/model.safetensors"}) == []
    )


def test_missing_pretrained_files_flags_a_missing_config() -> None:
    """config.json is what the in-container wrapper gates its download on."""
    from makermodslab.jobs import missing_pretrained_files

    assert missing_pretrained_files({"pretrained_model/model.safetensors"}) == [
        "pretrained_model/config.json"
    ]


def test_missing_pretrained_files_flags_missing_weights() -> None:
    from makermodslab.jobs import missing_pretrained_files

    assert missing_pretrained_files({"pretrained_model/config.json"}) == ["pretrained_model/*.safetensors"]


def test_missing_pretrained_files_ignores_a_missing_training_state() -> None:
    """The whole point: the optimizer half is deliberately never staged, so its
    absence must not read as an incomplete upload."""
    from makermodslab.jobs import missing_pretrained_files

    names = {n for n in _complete_names() if n.startswith("pretrained_model/")}
    assert not any(n.startswith("training_state/") for n in names)
    assert missing_pretrained_files(names) == []


def test_parse_duration_handles_mm_ss_and_hh_mm_ss() -> None:
    from makermodslab.jobs import _parse_duration

    assert _parse_duration("01:30") == 90
    assert _parse_duration("01:00:00") == 3600
    assert _parse_duration("?") is None
    assert _parse_duration("garbage") is None


def test_parse_metrics_into_extracts_loss_and_step() -> None:
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    line = "INFO ... step:42 smpl:336 loss:0.0123 grdn:1.5 lr:0.0001 ..."
    parse_metrics_into(line, m)

    assert m.current_step == 42
    assert m.current_loss == pytest.approx(0.0123)
    assert m.current_lr == pytest.approx(0.0001)
    assert m.grad_norm == pytest.approx(1.5)


def test_parse_metrics_into_keeps_tqdm_step_when_log_line_step_is_abbreviated() -> None:
    """At >=1000 steps lerobot formats the log-line step with format_big_number
    ("1K"), which int() can't parse. Feeding a tqdm line (exact step) then the
    abbreviated loss line into the same metrics object must retain the exact
    step and still extract the loss — this is what read_metrics_history relies
    on so it doesn't drop every point past step 1000.
    """
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  10%|██░| 1000/10000 [00:30<04:30, 3.2it/s]", m)
    parse_metrics_into("INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001 ...", m)

    assert m.current_step == 1000  # kept from tqdm, not zeroed by "1K"
    assert m.current_loss == pytest.approx(0.0077)
    assert m.current_lr == pytest.approx(0.0001)


def _tqdm_burst(first: int, last: int, total: int, eta: str = "6:26:18") -> str:
    """One log line carrying every tqdm redraw from `first` to `last`.

    tqdm separates redraws with \\r; a transport that doesn't split on \\r (HF
    Jobs' SSE log stream) delivers the whole burst as a single line with the
    trailing 'INFO ... step:N ...' appended to the LAST frame.
    """
    return "\r".join(
        f"Training:  39%|███▊      | {s}/{total} [2:31:07<{eta},  2.12s/step]" for s in range(first, last + 1)
    )


@pytest.mark.parametrize(
    ("burst", "info", "resume_total", "expect_step", "expect_total"),
    [
        # The real shape of a resumed cloud run: 50 frames of the remaining-window
        # bar + an abbreviated 'step:4K' that int() can't use. Last frame 50 of
        # 11000 remaining, on a 15000-step target → global step 4050.
        (
            _tqdm_burst(1, 50, 11000),
            "INFO 2026-07-29 02:11:59 train.py:606 step:4K smpl:259K ep:878 "
            "epch:43.90 loss:0.040 grdn:0.919 lr:8.4e-05",
            15000,
            4050,
            15000,
        ),
        # Same batching on a fresh run: the bar is already global, and the
        # 'step:1K' token is still unusable, so the last frame must stand.
        (
            _tqdm_burst(951, 1000, 10000),
            "INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001",
            None,
            1000,
            10000,
        ),
        # Below 1000 the log line's step is a plain int and wins outright —
        # which is also the only reason the first-frame bug stayed invisible
        # under step 1000.
        (
            _tqdm_burst(901, 950, 10000),
            "INFO ... step:950 smpl:7K loss:0.0077 grdn:0.9 lr:0.0001",
            None,
            950,
            10000,
        ),
    ],
    ids=["resumed-cloud-burst", "fresh-burst-abbreviated", "fresh-burst-exact"],
)
def test_parse_metrics_into_uses_the_last_tqdm_frame_of_a_batched_line(
    burst: str, info: str, resume_total: int | None, expect_step: int, expect_total: int
) -> None:
    """A batched line's LAST tqdm frame is the one the appended INFO line belongs
    to. Taking the first understated every step above 1000 by log_freq−1 (a real
    run charted 8201 where the true step was 8250)."""
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into(f"{burst}{info}", m, resume_total)

    assert m.current_step == expect_step
    assert m.total_steps == expect_total
    assert m.current_loss is not None
    # ETA comes from the same (last) frame.
    assert m.eta_seconds == 6 * 3600 + 26 * 60 + 18


def test_read_metrics_history_of_a_batched_resumed_log(tmp_path) -> None:
    """End-to-end on the shape a resumed cloud run actually writes: batched tqdm
    bursts + abbreviated step tokens land on the true global steps (multiples of
    log_freq), not log_freq−1 below them."""
    from makermodslab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path)
    root = reg._output_root
    msgs = [
        _tqdm_burst(first, first + 49, 11000) + f"INFO ... step:4K loss:0.04{i} grdn:0.9 lr:8.4e-05"
        for i, first in enumerate((1, 51, 101))
    ]
    p = _job_log_path(root, "R")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for msg in msgs:
            f.write(LogLine(timestamp=0.0, message=msg).model_dump_json() + "\n")
    reg._records["R"] = JobRecord(
        id="R",
        name="r",
        state="done",
        config=TrainingRequest(dataset_repo_id="d", resume=True, steps=15000),
        output_dir=str(root / "R" / "run"),
        started_at=0.0,
    )

    assert [pt.step for pt in reg.read_metrics_history("R")] == [4050, 4100, 4150]


def test_parse_metrics_into_extracts_tqdm_progress() -> None:
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    # tqdm format: "Training:  10%|...| 100/1000 [00:30<04:30, ..."
    line = "Training:  10%|██░|  100/1000 [00:30<04:30, 3.21it/s]"
    parse_metrics_into(line, m)

    assert m.current_step == 100
    assert m.total_steps == 1000
    assert m.eta_seconds == 270  # 4 min 30 s


def test_extract_wandb_run_url_finds_canonical_url() -> None:
    from makermodslab.jobs import extract_wandb_run_url

    line = "wandb: \U0001f680 View run at https://wandb.ai/me/myproj/runs/abc123 trailing text"
    assert extract_wandb_run_url(line) == "https://wandb.ai/me/myproj/runs/abc123"


def test_extract_wandb_run_url_returns_none_when_absent() -> None:
    from makermodslab.jobs import extract_wandb_run_url

    assert extract_wandb_run_url("nothing here") is None


def test_parse_metrics_into_rebases_resumed_tqdm_to_global_step() -> None:
    """On resume lerobot's bar counts only the remaining window (0 → steps−ckpt),
    so a raw 55/100 is really global step 155 of 200. With resume_total set, the
    parser must rebase so the UI shows 155/200, not 55/100."""
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  55%|█████| 55/100 [00:30<01:00, 2.0s/step]", m, resume_total=200)
    assert m.current_step == 155  # 200 - 100 + 55
    assert m.total_steps == 200


def test_parse_metrics_into_fresh_run_ignores_resume_rebase() -> None:
    """A fresh run passes resume_total=None; its bar is already the global step."""
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  30%|███| 30/100 [00:30<01:00, 2.0s/step]", m)
    assert m.current_step == 30
    assert m.total_steps == 100


def test_resume_start_step_prefers_the_requested_step() -> None:
    """The request's own answer wins when it has one."""
    from makermodslab.jobs import _resume_start_step
    from makermodslab.train import TrainingRequest

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_step=10,
        config_path="/out/checkpoints/000010/pretrained_model/train_config.json",
        steps=60,
    )
    assert _resume_start_step(cfg) == 10


def test_resume_start_step_reads_the_resolved_checkpoint_when_latest_was_asked_for() -> None:
    """Resuming "the latest checkpoint" leaves resume_from_step None — the step
    then lives only in what the resolvers wrote back onto the config, which is a
    zero-padded checkpoint dir on either runner."""
    from makermodslab.jobs import _resume_start_step
    from makermodslab.train import TrainingRequest

    local = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        config_path="/out/checkpoints/000010/pretrained_model/train_config.json",
        steps=60,
    )
    assert _resume_start_step(local) == 10

    cloud = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_hub_repo="user/out",
        resume_from_hub_step="004000",
        steps=15000,
    )
    assert _resume_start_step(cloud) == 4000


def test_resume_start_step_is_none_when_nothing_names_the_checkpoint() -> None:
    """A fresh run has no floor, and neither does a resume driven by a
    hand-supplied config_path that isn't a checkpoint tree — both must return
    None rather than invent a step."""
    from makermodslab.jobs import _resume_start_step
    from makermodslab.train import TrainingRequest

    fresh = TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=60)
    assert _resume_start_step(fresh) is None

    odd = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        config_path="/somewhere/custom/train_config.json",
        steps=60,
    )
    assert _resume_start_step(odd) is None


def test_initial_metrics_seeds_a_resumed_run_at_its_checkpoint_floor() -> None:
    """Before lerobot's first tqdm frame a resumed run has no parsed metrics, and
    reporting 0/0 there made every progress readout show a confident
    "0 / 60 · 0.0%" for the whole (multi-second to multi-minute) startup window.
    The floor is known from the request, so start there."""
    from makermodslab.jobs import _initial_metrics
    from makermodslab.train import TrainingRequest

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_step=10,
        steps=60,
    )
    m = _initial_metrics(cfg)
    assert (m.current_step, m.total_steps) == (10, 60)
    # A floor only — nothing is claimed about loss or ETA.
    assert m.current_loss is None
    assert m.eta_seconds is None


def test_initial_metrics_leaves_a_fresh_run_at_zero() -> None:
    """A fresh run really does start at 0, and total_steps == 0 is what the UI
    reads as "Training starting…". Unchanged."""
    from makermodslab.jobs import _initial_metrics
    from makermodslab.train import TrainingRequest

    cfg = TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=60)
    m = _initial_metrics(cfg)
    assert (m.current_step, m.total_steps) == (0, 0)


def test_initial_metrics_needs_a_step_target_to_seed() -> None:
    """No configured target ⇒ no honest percentage, so don't half-seed."""
    from makermodslab.jobs import _initial_metrics
    from makermodslab.train import TrainingRequest

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_step=10,
        steps=0,
    )
    assert _initial_metrics(cfg).current_step == 0


def test_read_metrics_history_stitches_resume_lineage(tmp_path) -> None:
    """A resumed run's curve is continuous across the whole lineage: the source
    run's points (0→100) are prepended to the resumed run's (150→200)."""
    from makermodslab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path)
    root = reg._output_root

    def write_log(job_id: str, msgs: list[str]) -> None:
        p = _job_log_path(root, job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for m in msgs:
                f.write(LogLine(timestamp=0.0, message=m).model_dump_json() + "\n")

    write_log("A", ["INFO step:50 loss:1.5 grdn:1 lr:0.001", "INFO step:100 loss:1.2 grdn:1 lr:0.001"])
    write_log("B", ["INFO step:150 loss:1.1 grdn:1 lr:5e-4", "INFO step:200 loss:1.0 grdn:1 lr:2e-4"])
    reg._records["A"] = JobRecord(
        id="A",
        name="a",
        state="done",
        config=TrainingRequest(dataset_repo_id="d"),
        output_dir=str(root / "A" / "run"),
        started_at=0.0,
    )
    reg._records["B"] = JobRecord(
        id="B",
        name="b",
        state="done",
        config=TrainingRequest(dataset_repo_id="d", resume=True, resume_from_job_id="A", steps=200),
        output_dir=str(root / "B" / "run"),
        started_at=0.0,
    )

    assert [p.step for p in reg.read_metrics_history("B")] == [50, 100, 150, 200]
    # The source run on its own is unchanged (no lineage to prepend).
    assert [p.step for p in reg.read_metrics_history("A")] == [50, 100]


def test_parse_metrics_into_ignores_unrelated_lines() -> None:
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("just a log line with no metrics", m)
    assert m.current_step == 0 or m.current_step is None  # accept either default


def test_log_line_round_trips_to_json() -> None:
    from makermodslab.jobs import LogLine

    line = LogLine(timestamp=1.5, message="hello")
    payload = line.model_dump_json()
    parsed = LogLine.model_validate_json(payload)
    assert parsed.timestamp == 1.5
    assert parsed.message == "hello"


def test_pid_alive_returns_false_for_unlikely_pid() -> None:
    from makermodslab.jobs import _pid_alive

    # DISCOVERED: os.kill(-1, 0) on macOS sends to process group and succeeds
    # (returns True), so we use a large PID that certainly does not exist.
    assert _pid_alive(999999999) is False


def _write_log_lines(path: Path, messages: list[str]) -> None:
    from makermodslab.jobs import LogLine

    lines = (LogLine(timestamp=float(i), message=m).model_dump_json() + "\n" for i, m in enumerate(messages))
    path.write_text("".join(lines))


def test_tailing_runner_returncode_confirms_done_when_exit_status_zero(tmp_path) -> None:
    """MT10: a reattached job (TailingJobRunner) whose pid has disappeared is
    only confirmed 'done' when the wrapper LocalJobRunner.start() launched
    actually wrote a real exit status of 0 to disk — the trainer's own log
    output is not usable evidence, because after a server restart nothing
    appends to that log ever again (see LocalJobRunner._pump_stdout, which is
    the log's only writer and lives in the process that just died)."""
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    log_path = tmp_path / "log.jsonl"
    _write_log_lines(log_path, ["Training:  99%|##########| 999/1000 [01:00<00:01,  1.0step/s]"])
    status_path = tmp_path / "exit_status"
    status_path.write_text("0")

    # A pid guaranteed not to exist (see test_pid_alive_returns_false_for_unlikely_pid)
    # stands in for "the process is gone by the time we look."
    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=999999999, status_path=status_path)
    runner.start_tailing()
    runner._tail_thread.join(timeout=5)
    assert not runner._tail_thread.is_alive()

    assert runner.returncode() == 0


def test_tailing_runner_returncode_reports_real_nonzero_exit(tmp_path) -> None:
    """A trainer that crashed after reattach records its own real exit code
    via the wrapper — returncode() must surface that code (not clamp it to
    None or 0) so JobRegistry._tick() can finalise 'failed' with a real
    exit_code, same as it would for a LocalJobRunner that never detached."""
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    log_path = tmp_path / "log.jsonl"
    _write_log_lines(log_path, ["Training:  42%|####      | 420/1000 [00:30<00:41, 14.0step/s]"])
    status_path = tmp_path / "exit_status"
    status_path.write_text("1")

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=999999999, status_path=status_path)
    runner.start_tailing()
    runner._tail_thread.join(timeout=5)
    assert not runner._tail_thread.is_alive()

    assert runner.returncode() == 1


def test_tailing_runner_returncode_unconfirmed_when_status_file_absent(tmp_path) -> None:
    """MT10 regression: this is the actual shape of a reattached job whose
    trainer keeps running (or crashes) for the rest of its life after a
    server restart — log.jsonl is frozen (nothing appends to it once the
    owning process is gone) and no exit_status file has been written yet.
    Before the fix, returncode() unconditionally returned 0 once the pid
    vanished, silently recording a crash as a successful run. It must now
    report None so JobRegistry._tick() marks the record 'interrupted' rather
    than asserting a 'done' or 'failed' it can't back up."""
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    log_path = tmp_path / "log.jsonl"
    _write_log_lines(log_path, ["Training:  42%|####      | 420/1000 [00:30<00:41, 14.0step/s]"])
    status_path = tmp_path / "exit_status"  # never written

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=999999999, status_path=status_path)
    runner.start_tailing()
    runner._tail_thread.join(timeout=5)
    assert not runner._tail_thread.is_alive()

    assert runner.returncode() is None


def test_tailing_runner_returncode_unconfirmed_when_status_file_malformed(tmp_path) -> None:
    """A status file that isn't a clean integer (e.g. a torn read caught
    mid-write, though start()'s tmp+rename should prevent that in practice)
    must degrade to 'unconfirmed', not raise or silently pick a wrong code."""
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    log_path = tmp_path / "log.jsonl"
    _write_log_lines(log_path, ["Training:  42%|####      | 420/1000 [00:30<00:41, 14.0step/s]"])
    status_path = tmp_path / "exit_status"
    status_path.write_text("not-a-number")

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=999999999, status_path=status_path)
    runner.start_tailing()
    runner._tail_thread.join(timeout=5)
    assert not runner._tail_thread.is_alive()

    assert runner.returncode() is None


def test_tailing_runner_stop_signalled_after_term_reaches_the_group(tmp_path, monkeypatch) -> None:
    """stop_signalled() is JobRegistry._tick()'s way of telling a
    user-requested stop apart from an unconfirmed crash/restart when
    returncode() comes back None (see test_tick_uses_stop_message_when_stop_was_signalled).
    False before stop(), and True once killpg has actually put a SIGTERM into
    the run's process group."""
    import signal

    from makermodslab import jobs
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    delivered: list[tuple[int, int]] = []
    monkeypatch.setattr(jobs.os, "killpg", lambda pid, sig: delivered.append((pid, sig)))

    log_path = tmp_path / "log.jsonl"
    status_path = tmp_path / "exit_status"

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=4242, status_path=status_path)
    assert runner.stop_signalled() is False

    runner.stop()

    assert delivered == [(4242, signal.SIGTERM)]
    assert runner.stop_signalled() is True


def test_tailing_runner_stop_signalled_false_when_group_already_gone(tmp_path, monkeypatch) -> None:
    """The fact stop_signalled() reports is "we delivered a SIGTERM to a live
    process group", not "someone called stop()". A run that crashed (or died
    to a restart) before the user clicked Stop reaches killpg with nothing
    left to signal — ProcessLookupError — and must stay False, or _tick()
    would launder that crash into "stopped at your request" and tell the user
    we stopped a run that had already ended on its own.

    stop()'s own bookkeeping still runs: _stop_event winds the tail loop
    down either way."""
    from makermodslab import jobs
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    def _already_gone(pid: int, sig: int) -> None:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(jobs.os, "killpg", _already_gone)

    log_path = tmp_path / "log.jsonl"
    status_path = tmp_path / "exit_status"

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=4242, status_path=status_path)
    runner.stop()

    assert runner.stop_signalled() is False
    assert runner._stop_event.is_set()


def test_tailing_runner_returncode_unconfirmed_while_pid_alive(tmp_path) -> None:
    """Even with a written exit_status on disk (e.g. left over from state we
    shouldn't trust yet), a live pid always means 'still running' — returncode()
    must not report a stale status file's contents while the process itself
    is confirmed alive."""
    from makermodslab.jobs import TailingJobRunner, TrainingMetrics

    log_path = tmp_path / "log.jsonl"
    status_path = tmp_path / "exit_status"
    status_path.write_text("0")

    runner = TailingJobRunner(TrainingMetrics(), log_path, pid=os.getpid(), status_path=status_path)

    assert runner.returncode() is None


class _FixedRcRunner:
    """Minimal JobRunner stand-in for exercising JobRegistry._tick()'s
    finalisation branch without a real subprocess."""

    def __init__(self, rc: int | None) -> None:
        self._rc = rc

    def is_running(self) -> bool:
        return False

    def returncode(self) -> int | None:
        return self._rc

    def wandb_run_url(self) -> str | None:
        return None


class _FakeStopSignalledRunner(_FixedRcRunner):
    """Like _FixedRcRunner, but also reports stop_signalled() == True — the
    shape of a TailingJobRunner whose stop() group-TERMed the wrapper before
    it could write an exit status."""

    def stop_signalled(self) -> bool:
        return True


def _inject_running_job(reg, tmp_path: Path, rc: int | None, runner=None):
    """Stop the registry's own watchdog thread (so our manual _tick() call
    below is deterministic, not racing a background tick), then splice a
    'running' record backed by `runner` (default: _FixedRcRunner(rc)) straight
    into the registry's internal maps — the same shape _load_from_disk /
    start() would produce."""
    reg.shutdown()
    if reg._watchdog_thread is not None:
        reg._watchdog_thread.join(timeout=2)

    record = _record(tmp_path / "job-1")
    record.state = "running"
    with reg._lock:
        reg._records[record.id] = record
        reg._runners[record.id] = runner if runner is not None else _FixedRcRunner(rc)
    return record


def test_tick_marks_interrupted_when_runner_cannot_confirm_exit(tmp_path) -> None:
    """MT10: JobRegistry._tick() must not treat "runner says not running,
    returncode() is None" as a failure (or a success) — that combination
    means the runner has no evidence either way (TailingJobRunner's signal
    for "pid died on our watch, no exit_status file written"). It should
    finalise as 'interrupted', the same honest state already used when a
    reattach finds an already-dead pid at boot, with no exit_code but an
    explanatory error_message — not 'done', and not 'failed' either. The
    message matters because 'interrupted' no longer implies the run actually
    failed; a real checkpoint may still be sitting on disk (see
    models.list_local_models, which no longer deletes it from the library)."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _inject_running_job(reg, tmp_path, rc=None)

    reg._tick()

    finalized = reg._records[record.id]
    assert finalized.state == "interrupted"
    assert finalized.exit_code is None
    assert finalized.error_message is not None
    assert "restarted" in finalized.error_message


def test_tick_uses_stop_message_when_stop_was_signalled(tmp_path) -> None:
    """A user-requested stop of a reattached run must never be blamed on a
    restart that never happened.

    stop() group-TERMs the wrapper before it can write an exit status, so the
    pid disappears with no evidence on disk — the same shape as an unconfirmed
    crash. What separates the two is the pair of signals this test sets up:
    the registry's recorded intent (`_stop_requested`, what JobRegistry.stop()
    writes under the lock) and the runner's own stop_signalled() confirming it
    reached a live process. TailingJobRunner turns that pair into a
    synthesised -SIGTERM, so the run classifies as `interrupted` and gets the
    deliberate-stop wording rather than the unconfirmed one."""
    import signal

    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, UNCONFIRMED_OUTCOME_MESSAGE, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _inject_running_job(reg, tmp_path, rc=None, runner=_FakeStopSignalledRunner(-signal.SIGTERM))
    with reg._lock:
        reg._stop_requested.add(record.id)

    reg._tick()

    finalized = reg._records[record.id]
    assert finalized.state == "interrupted"
    assert finalized.exit_code == -signal.SIGTERM
    assert finalized.error_message == STOPPED_BY_REQUEST_MESSAGE
    assert finalized.error_message != UNCONFIRMED_OUTCOME_MESSAGE


def test_tick_does_not_claim_a_stop_that_never_reached_the_run(tmp_path) -> None:
    """End-to-end counterpart to the test above, with the real
    TailingJobRunner rather than a fake: the run is already dead (crash, or a
    restart that killed it) when the user's Stop arrives, so stop()'s killpg
    finds nothing to signal. _tick() must fall back to the restart message —
    telling the user we stopped a run that had already ended on its own is the
    same false story in the opposite direction."""
    from makermodslab.jobs import JobRegistry, TailingJobRunner, TrainingMetrics

    runner = TailingJobRunner(
        TrainingMetrics(),
        tmp_path / "log.jsonl",
        pid=999999999,  # long dead; killpg raises ProcessLookupError
        status_path=tmp_path / "exit_status",  # never written
    )
    runner.stop()

    reg = JobRegistry(tmp_path / "root")
    record = _inject_running_job(reg, tmp_path, rc=None, runner=runner)

    reg._tick()

    finalized = reg._records[record.id]
    assert finalized.state == "interrupted"
    assert finalized.exit_code is None
    assert finalized.error_message is not None
    assert "restarted" in finalized.error_message
    assert "stopped at your request" not in finalized.error_message


def test_tick_marks_done_when_runner_confirms_zero_exit(tmp_path) -> None:
    """Sanity check for the untouched happy path this fix must not regress:
    when a runner confirms rc == 0 (LocalJobRunner completing normally, or
    TailingJobRunner after observing "End of training"), the watchdog still
    finalises promptly as 'done'."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _inject_running_job(reg, tmp_path, rc=0)

    reg._tick()

    finalized = reg._records[record.id]
    assert finalized.state == "done"
    assert finalized.exit_code == 0


def test_hub_checkpoints_from_files_parses_tree() -> None:
    from makermodslab.jobs import _hub_checkpoints_from_files

    files = [
        "README.md",
        "checkpoints/000010/pretrained_model/config.json",
        "checkpoints/000020/pretrained_model/config.json",
        "checkpoints/000020/pretrained_model/model.safetensors",
    ]
    out = _hub_checkpoints_from_files(files, "user/repo")
    assert [c.step for c in out] == [10, 20]
    assert out[1].source == "hub"
    assert out[1].ref == "user/repo@checkpoints/000020"


def _make_pretrained(dir_path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(_json.dumps({"type": "act"}))


def test_list_imported_local_single_model(tmp_path) -> None:
    from makermodslab.jobs import _list_imported_local

    _make_pretrained(tmp_path)  # config.json at the root
    out = _list_imported_local(str(tmp_path))
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "local"
    assert out[0].ref == str(tmp_path.resolve())


def test_list_imported_local_checkpoints_tree(tmp_path) -> None:
    from makermodslab.jobs import _list_imported_local

    _make_pretrained(tmp_path / "checkpoints" / "000010" / "pretrained_model")
    out = _list_imported_local(str(tmp_path))
    assert [c.step for c in out] == [10]
    assert out[0].source == "local"
    assert out[0].ref.endswith("/checkpoints/000010/pretrained_model")


def test_list_imported_local_empty_when_no_model(tmp_path) -> None:
    from makermodslab.jobs import _list_imported_local

    assert _list_imported_local(str(tmp_path)) == []


def test_list_imported_hub_single_model() -> None:
    from makermodslab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors", "README.md"]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "hub"
    assert out[0].ref == "user/repo@root"


def test_list_imported_hub_prefers_checkpoints_tree() -> None:
    from makermodslab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return [
                "config.json",  # also present, but the tree wins
                "checkpoints/000050/pretrained_model/config.json",
            ]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert [c.step for c in out] == [50]
    assert out[0].ref == "user/repo@checkpoints/000050"


def test_list_imported_hub_empty_when_no_model() -> None:
    from makermodslab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["README.md"]

    assert _list_imported_hub(FakeApi(), "user/repo") == []


def test_list_hub_checkpoints_falls_back_to_root_policy() -> None:
    """MT3 residual: a tracked run whose repo holds a root policy but NO
    checkpoints/ tree (checkpoint saving off) used to list zero checkpoints, so
    its job card said "no checkpoints" while a loadable model sat in the repo.
    The cloud listing now falls back to the same '@root' entry the imported
    listing has always returned."""
    from makermodslab.jobs import _list_hub_checkpoints

    out = _list_hub_checkpoints(_FakeHubApi(["config.json", "model.safetensors"]), "user/repo")
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "hub"
    # The ref shape rollout._resolve_policy_path already downloads and runs, so
    # the entry is deployable and not merely listed.
    assert out[0].ref == "user/repo@root"


def test_list_hub_checkpoints_prefers_tree_over_root() -> None:
    """The root push is byte-identical to the final checkpoint, so a repo with
    a tree must NOT also offer a root entry — that would be a duplicate of the
    highest step under a second name."""
    from makermodslab.jobs import _list_hub_checkpoints

    files = ["config.json", "model.safetensors", *_hub_checkpoint_files("005000")]
    out = _list_hub_checkpoints(_FakeHubApi(files), "user/repo")
    assert [c.ref for c in out] == ["user/repo@checkpoints/005000"]


def test_list_hub_checkpoints_empty_without_root_config() -> None:
    from makermodslab.jobs import _list_hub_checkpoints

    assert _list_hub_checkpoints(_FakeHubApi(["README.md", "model.safetensors"]), "user/repo") == []


def test_resolve_cloud_resume_rejects_root_only_repo(monkeypatch) -> None:
    """The root fallback makes a checkpoint-less repo listable and runnable, but
    root weights carry no training_state/ — resume must refuse it in plain
    language rather than tripping over the ref shape."""
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(["config.json", "model.safetensors"]),
    )
    with pytest.raises(ValueError, match="saved no checkpoints") as excinfo:
        _resolve_cloud_resume(_cloud_record(), None)
    assert "Fine-tune from its weights" in str(excinfo.value)


def test_read_checkpoint_config_local_reads_config_json(tmp_path) -> None:
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

    (tmp_path / "config.json").write_text(_json.dumps({"type": "act"}))
    ckpt = JobCheckpoint(step=0, source="local", ref=str(tmp_path))
    assert _read_checkpoint_config(ckpt) == {"type": "act"}


def test_read_checkpoint_config_hub_root(monkeypatch, tmp_path) -> None:
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"type": "smolvla"}))
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ckpt = JobCheckpoint(step=0, source="hub", ref="user/repo@root")
    assert _read_checkpoint_config(ckpt) == {"type": "smolvla"}
    assert seen["repo_id"] == "user/repo"
    assert seen["filename"] == "config.json"


def test_read_checkpoint_config_hub_tree(monkeypatch, tmp_path) -> None:
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"type": "act"}))
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ckpt = JobCheckpoint(step=50, source="hub", ref="user/repo@checkpoints/000050")
    assert _read_checkpoint_config(ckpt) == {"type": "act"}
    assert seen["repo_id"] == "user/repo"
    assert seen["filename"] == "checkpoints/000050/pretrained_model/config.json"


def test_register_imported_local_dir(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)  # config.json at root
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))

    assert rec.runner == "imported"
    assert rec.state == "done"
    assert rec.output_dir == str(model.resolve())
    assert rec.hf_repo_id is None
    cks = reg.list_checkpoints(rec.id)
    assert [c.step for c in cks] == [0]
    # Persisted as a pointer job.json, reloadable.
    reg2 = JobRegistry(tmp_path / "root")
    assert reg2.get(rec.id).runner == "imported"


def test_register_imported_rejects_unusable_source(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    empty = tmp_path / "empty"
    empty.mkdir()
    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(ValueError, match="No usable model"):
        reg.register_imported(str(empty))


def test_rename_sets_display_name_and_persists(tmp_path) -> None:
    """Rename is a metadata-only alias: trimmed, persisted to job.json, and the
    immutable identity (id / name / output_dir) is untouched."""
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))
    assert rec.display_name is None

    renamed = reg.rename(rec.id, "  pick-and-place v2  ")
    assert renamed.display_name == "pick-and-place v2"  # trimmed
    assert renamed.id == rec.id
    assert renamed.name == rec.name
    assert renamed.output_dir == str(model.resolve())

    # Round-trips through job.json on a fresh registry.
    reg2 = JobRegistry(tmp_path / "root")
    assert reg2.get(rec.id).display_name == "pick-and-place v2"


def test_rename_rejects_empty_and_path_characters(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))

    with pytest.raises(ValueError, match="empty"):
        reg.rename(rec.id, "   ")
    with pytest.raises(ValueError, match="Invalid"):
        reg.rename(rec.id, "evil/../name")
    assert reg.get(rec.id).display_name is None  # nothing persisted


def test_rename_unknown_job_raises(tmp_path) -> None:
    from makermodslab.jobs import JobNotFoundError, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(JobNotFoundError):
        reg.rename("nope", "anything")


def test_rename_allows_duplicate_aliases(tmp_path) -> None:
    """Aliases are display-only (not file keys like calibration/robot names),
    so uniqueness is deliberately NOT enforced."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    for jid in ("A", "B"):
        reg._records[jid] = JobRecord(
            id=jid,
            name=jid,
            state="done",
            config=TrainingRequest(dataset_repo_id="d"),
            output_dir=str(reg._output_root / jid / "run"),
            started_at=0.0,
        )
    reg.rename("A", "same alias")
    reg.rename("B", "same alias")
    assert reg.get("A").display_name == "same alias"
    assert reg.get("B").display_name == "same alias"


def test_job_json_without_display_name_loads_with_none(tmp_path) -> None:
    """Registry files written before the alias field existed load fine, and a
    subsequent rename persists the new field alongside the old ones."""
    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    job_dir = root / "old-job"
    job_dir.mkdir(parents=True)
    meta = {
        "id": "old-job",
        "name": "ACT · user/ds",
        "state": "done",
        "config": {"dataset_repo_id": "user/ds", "policy_type": "act"},
        "output_dir": str(job_dir / "run"),
        "started_at": 1.0,
    }
    (job_dir / "job.json").write_text(_json.dumps(meta))

    reg = JobRegistry(root)
    assert reg.get("old-job").display_name is None

    reg.rename("old-job", "legacy run")
    data = _json.loads((job_dir / "job.json").read_text())
    assert data["display_name"] == "legacy run"


def test_register_imported_hub_repo(monkeypatch, tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    # Patch the symbol where jobs.py binds it (`from .utils.hf_auth import
    # shared_hf_api`) — patching it in its home module has no effect on the
    # already-bound name and the test would hit the network.
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported("user/some-model")

    assert rec.runner == "imported"
    assert rec.hf_repo_id == "user/some-model"
    assert rec.output_dir == ""
    cks = reg.list_checkpoints(rec.id)
    assert [c.ref for c in cks] == ["user/some-model@root"]


def test_register_imported_local_dir_is_idempotent(tmp_path) -> None:
    """Importing the same local dir twice returns the EXISTING record — same
    id, display alias untouched, no second registry entry."""
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported(str(model))
    reg.rename(first.id, "my import")

    again = reg.register_imported(str(model), name="ignored on duplicate")
    assert again.id == first.id
    assert again.display_name == "my import"
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_register_imported_hub_repo_is_idempotent(monkeypatch, tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("user/some-model")
    again = reg.register_imported("user/some-model")
    assert again.id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_find_imported_hub_id_compare_is_case_insensitive(monkeypatch, tmp_path) -> None:
    """REVERSAL of the earlier exact-match choice, prompted by a real duplicate
    that slipped through on a case-only difference: HF repo ids are practically
    unique case-insensitively (the Hub redirects across casings), and the
    failure mode of exact matching is silent duplicate cards."""
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("user/some-model")
    assert reg.find_imported("user/some-model") is not None
    assert reg.find_imported("User/Some-Model") is not None
    assert reg.register_imported("USER/SOME-MODEL").id == first.id


def test_register_imported_hub_url_normalizes_to_repo_id(monkeypatch, tmp_path) -> None:
    """A pasted model-page URL is normalized to the bare repo id at the boundary
    — both for storage (so checkpoint listing works) and for dedup."""
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "user/some-model"  # bare id, never the pasted URL
            return ["config.json"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("https://huggingface.co/user/some-model/")
    assert first.hf_repo_id == "user/some-model"
    assert reg.register_imported("user/some-model").id == first.id
    assert reg.register_imported("  https://hf.co/user/some-model ").id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def _case_variant_dir(path: Path) -> Path | None:
    """A differently-cased spelling of `path` that still resolves to the same
    directory — only possible on a case-insensitive filesystem (macOS default,
    where the real bug happened). None on case-sensitive filesystems."""
    variant = path.parent / path.name.swapcase()
    try:
        if str(variant) != str(path) and variant.is_dir() and os.path.samefile(variant, path):
            return variant
    except OSError:
        pass
    return None


def test_find_imported_local_matches_case_variant_spelling(tmp_path) -> None:
    """Regression from the real duplicate pair: the same directory imported as
    '/Users/mokuroh54/…/smolvla_real_5k/pretrained_model' and
    '/Users/Mokuroh54/…' (case-insensitive macOS filesystem; Path.resolve()
    preserves the typed case) produced two cards, because identity was an
    exact string compare. Identity is now filesystem identity (samefile)."""
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "so101-real" / "smolvla_real_5k" / "pretrained_model"
    _make_pretrained(model)
    variant = _case_variant_dir(model)
    if variant is None:
        pytest.skip("requires a case-insensitive filesystem (the real bug's environment)")

    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported(str(model))
    again = reg.register_imported(str(variant))
    assert again.id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_boot_sweep_collapses_real_case_variant_duplicate_pair(tmp_path) -> None:
    """Fixture mirrors the real pair found in the live registry:
      smolvla_imported_2026-06-27_16-19-02  name='smolvla 5k'
        output_dir '…/mokuroh54/…/smolvla_real_5k/pretrained_model'
      smolvla_imported_2026-07-02_14-24-15  name='Imported · pretrained_model'
        output_dir '…/Mokuroh54/…' (same directory, different case)
    The sweep groups local imports by device:inode, so the pair collapses to
    the oldest record and the newer job.json-only dir is removed."""
    from makermodslab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "so101-real" / "smolvla_real_5k" / "pretrained_model"
    _make_pretrained(model)
    variant = _case_variant_dir(model)
    if variant is None:
        pytest.skip("requires a case-insensitive filesystem (the real bug's environment)")

    root = tmp_path / "root"
    _write_imported_pointer(
        root, "smolvla_imported_2026-06-27_16-19-02", str(model), started_at=1782548342.584353
    )
    _write_imported_pointer(
        root, "smolvla_imported_2026-07-02_14-24-15", str(variant), started_at=1782973455.742018
    )

    reg = JobRegistry(root)
    kept = reg.get("smolvla_imported_2026-06-27_16-19-02")
    assert kept.output_dir == str(model)
    with pytest.raises(JobNotFoundError):
        reg.get("smolvla_imported_2026-07-02_14-24-15")
    assert not (root / "smolvla_imported_2026-07-02_14-24-15").exists()


def test_unique_job_id_suffixes_on_same_second_collision(tmp_path, monkeypatch) -> None:
    """_generate_job_id has second-granularity timestamps; two different models
    imported within the same second must not overwrite each other."""
    from makermodslab import jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_generate_job_id", lambda p, d: "act_imported_T")
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_pretrained(a)
    _make_pretrained(b)
    reg = jobs_mod.JobRegistry(tmp_path / "root")
    r1 = reg.register_imported(str(a))
    r2 = reg.register_imported(str(b))
    assert r1.id == "act_imported_T"
    assert r2.id == "act_imported_T-2"
    assert {r.id for r in reg.list(limit=100)} == {r1.id, r2.id}


def _write_imported_pointer(
    root: Path, job_id: str, output_dir: str, started_at: float, display_name: str | None = None
) -> Path:
    """Lay out an on-disk imported pseudo-job dir (job.json only), the way
    older MakerMods Lab versions left duplicates behind before dedup-at-registration."""
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    meta = {
        "id": job_id,
        "name": f"Imported · {job_id}",
        "display_name": display_name,
        "state": "done",
        "config": {"dataset_repo_id": "(imported)", "policy_type": "act"},
        "output_dir": output_dir,
        "started_at": started_at,
        "ended_at": started_at,
        "runner": "imported",
    }
    (job_dir / "job.json").write_text(_json.dumps(meta))
    return job_dir


def test_boot_sweep_collapses_duplicate_imports_keeping_oldest(tmp_path) -> None:
    """Pre-existing duplicate pointers collapse on load: oldest kept, the
    newest duplicate's alias migrated onto it, duplicate job.json-only dirs
    removed."""
    from makermodslab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    root = tmp_path / "root"
    _write_imported_pointer(root, "A", str(model.resolve()), started_at=1.0)
    _write_imported_pointer(root, "B", str(model.resolve()), started_at=2.0, display_name="nice name")

    reg = JobRegistry(root)
    kept = reg.get("A")
    assert kept.display_name == "nice name"  # migrated from the newer dup
    with pytest.raises(JobNotFoundError):
        reg.get("B")
    assert not (root / "B").exists()  # contained only job.json → removed
    # The migrated alias is persisted on the keeper.
    assert _json.loads((root / "A" / "job.json").read_text())["display_name"] == "nice name"
    # Idempotent: a fresh load sees one record and nothing left to collapse.
    reg2 = JobRegistry(root)
    assert reg2.get("A").display_name == "nice name"


def test_boot_sweep_never_deletes_dirs_with_extra_content(tmp_path) -> None:
    """A duplicate whose dir holds more than job.json is only dropped from the
    in-memory map — its files stay on disk."""
    from makermodslab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    root = tmp_path / "root"
    _write_imported_pointer(root, "A", str(model.resolve()), started_at=1.0, display_name="keeper alias")
    dup_dir = _write_imported_pointer(
        root, "B", str(model.resolve()), started_at=2.0, display_name="dup alias"
    )
    (dup_dir / "extra.safetensors").write_text("")  # anything beyond job.json

    reg = JobRegistry(root)
    kept = reg.get("A")
    assert kept.display_name == "keeper alias"  # keeper's own alias wins
    with pytest.raises(JobNotFoundError):
        reg.get("B")
    assert (dup_dir / "job.json").exists()  # nothing deleted
    assert (dup_dir / "extra.safetensors").exists()


def test_flat_feature_dim_reads_single_arm_and_bimanual_state() -> None:
    """observation.state / action are 1-D: [6] for one SO-101 arm, [12] for a
    bimanual (two-arm) checkpoint. The inference modal keys the single-arm vs
    bimanual mismatch off this."""
    from makermodslab.jobs import _flat_feature_dim

    assert _flat_feature_dim({"type": "STATE", "shape": [6]}) == 6
    assert _flat_feature_dim({"type": "STATE", "shape": [12]}) == 12
    assert _flat_feature_dim({"type": "ACTION", "shape": (12,)}) == 12


def test_flat_feature_dim_returns_none_for_missing_or_non_1d() -> None:
    from makermodslab.jobs import _flat_feature_dim

    assert _flat_feature_dim(None) is None
    assert _flat_feature_dim({}) is None
    assert _flat_feature_dim({"shape": [3, 480, 640]}) is None  # a VISUAL feature
    assert _flat_feature_dim({"shape": []}) is None
    assert _flat_feature_dim({"shape": "nope"}) is None


def test_cloud_start_rejects_local_only_dataset(tmp_path) -> None:
    """A cloud (hf_cloud) run on a dataset that's only local raises
    DatasetNotOnHubError before any record/runner is created — HF Jobs pods
    resolve the dataset from the Hub, so a local-only one would fail remotely."""
    from unittest.mock import patch

    from makermodslab.jobs import DatasetNotOnHubError, JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/local_only", "status": "local_only", "url": None},
        ),
        pytest.raises(DatasetNotOnHubError) as exc,
    ):
        reg.start(cfg, target)

    assert exc.value.repo_id == "user/local_only"
    assert "not on the Hugging Face Hub" in str(exc.value)
    # Nothing was registered — the guard fires before the record is created.
    assert reg.list(limit=10) == []


def test_cloud_start_allows_hub_dataset(tmp_path) -> None:
    """When the dataset is on the Hub, the preflight passes and the runner is
    started (stubbed here — we assert the guard doesn't block, not a real
    submission)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/on_hub", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = "https://hf.co/jobs/job-xyz"

    def _fake_runner_factory(*_args, **_kwargs):
        return fake_runner

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", _fake_runner_factory),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"
    fake_runner.start.assert_called_once()


def test_cloud_start_rejects_empty_hub_copy(tmp_path) -> None:
    """A cloud run on a dataset whose Hub repo exists but has no data (an
    interrupted upload left the empty repo behind) raises
    DatasetHubCopyEmptyError before any record/runner is created — the pod
    trains on the HUB copy, so an empty one would fail remotely instead of
    here with an actionable message."""
    from unittest.mock import patch

    from makermodslab.jobs import DatasetHubCopyEmptyError, JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/empty_upload", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/empty_upload", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=False),
        pytest.raises(DatasetHubCopyEmptyError) as exc,
    ):
        reg.start(cfg, target)

    assert exc.value.repo_id == "user/empty_upload"
    assert "no data in it" in str(exc.value)
    # Nothing was registered — the guard fires before the record is created.
    assert reg.list(limit=10) == []


def test_cloud_start_allows_hub_copy_with_unknown_data_status(tmp_path) -> None:
    """hub_copy_has_data returning None (offline / transport error) does NOT
    block the run — same "only a definitive answer blocks" rule as the
    local_only guard. A network blip must not wrongly refuse a real dataset."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/on_hub", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=None),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"


def test_cloud_start_allows_unknown_status_dataset(tmp_path) -> None:
    """An "unknown" hub status (offline / transient transport error) does NOT
    block the run — a network blip must not wrongly refuse a real Hub dataset;
    the existing _ensure_dataset_on_hub fallback handles a genuinely-missing
    one. The guard only rejects a definitive "local_only"."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/maybe", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/maybe", "status": "unknown", "url": None},
        ),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"


def test_cloud_start_passes_resume_total_to_the_runner(tmp_path) -> None:
    """A resumed cloud run must hand the runner its full step target, or the log
    parser can't rebase the remaining-window tqdm bar and the UI reports
    resume-relative progress (observed: 4,251/11,000 instead of 8,251/15,000)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(
        dataset_repo_id="user/on_hub",
        policy_type="act",
        resume=True,
        # Stands in for a resume selection; the runner (which is what turns this
        # into a Hub download for a cloud job) is stubbed out below.
        config_path="/somewhere/checkpoints/004000/pretrained_model/train_config.json",
        steps=15000,
    )
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    seen: list[tuple] = []
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    def _factory(*args, **kwargs):
        seen.append(args)
        return fake_runner

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", _factory),
    ):
        reg.start(cfg, target)

    assert seen and seen[0][-1] == 15000


def test_start_seeds_a_resumed_records_progress_at_the_checkpoint_step(tmp_path) -> None:
    """The record a resume starts must already read 4,000/15,000 — not 0/0.

    resume_total only helps once lerobot's tqdm bar exists, and nothing fills
    the gap before it: on a real local resume that window was 12s of a 69s run,
    during which every progress readout in the app said step 0. The seed has to
    be on the RECORD (not just the runner) so the persisted job.json, the /jobs
    payload and the ~1Hz progress broadcast all carry it from the first tick."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(
        dataset_repo_id="user/on_hub",
        policy_type="act",
        resume=True,
        config_path="/somewhere/checkpoints/004000/pretrained_model/train_config.json",
        steps=15000,
    )
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
        patch(
            "makermodslab.runners.hf_cloud.HfCloudJobRunner",
            lambda *a, **k: fake_runner,
        ),
    ):
        record = reg.start(cfg, JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert (record.metrics.current_step, record.metrics.total_steps) == (4000, 15000)
    # And it survives to disk, which is what a reattach after a restart reloads.
    persisted = _json.loads((tmp_path / "root" / record.id / "job.json").read_text())
    assert persisted["metrics"]["current_step"] == 4000


def test_start_leaves_a_fresh_records_progress_at_zero(tmp_path) -> None:
    """The non-resumed path is untouched: 0/0 is correct there, and total_steps
    == 0 is the signal the UI renders as "Training starting…"."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/on_hub", policy_type="act", steps=15000)
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.datasets.hub_copy_has_data", return_value=True),
        patch(
            "makermodslab.runners.hf_cloud.HfCloudJobRunner",
            lambda *a, **k: fake_runner,
        ),
    ):
        record = reg.start(cfg, JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert (record.metrics.current_step, record.metrics.total_steps) == (0, 0)


def test_cloud_reattach_passes_resume_total_to_the_runner(monkeypatch, tmp_path) -> None:
    """Re-attaching to a running cloud job after a restart must carry the resume
    target too — otherwise the progress readout silently rebases itself on the
    remaining window mid-run."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    job_dir = root / "cloud-job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        _json.dumps(
            {
                "id": "cloud-job",
                "name": "SMOLVLA · user/ds",
                "state": "running",
                "config": {
                    "dataset_repo_id": "user/ds",
                    "policy_type": "smolvla",
                    "resume": True,
                    "steps": 15000,
                },
                "output_dir": str(job_dir / "run"),
                "started_at": 1.0,
                "runner": "hf_cloud",
                "hf_job_id": "hf-job-1",
                "hf_flavor": "a10g-small",
            }
        )
    )

    seen: list[tuple] = []

    def _factory(*args, **kwargs):
        seen.append(args)
        return MagicMock()

    # No watchdog: this test is about what _load_from_disk hands the runner, and
    # the tick would poll the (stubbed) runner and the Hub for checkpoints.
    monkeypatch.setattr(JobRegistry, "_start_watchdog", lambda self: None)
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", _factory):
        JobRegistry(root)

    assert seen and seen[0][-1] == 15000


def test_local_start_skips_hub_preflight(tmp_path) -> None:
    """A local run on a local-only dataset is fine — no Hub involved — so the
    preflight must not fire (get_hub_status is never consulted)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")

    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242

    with (
        patch("makermodslab.datasets.get_hub_status") as get_status,
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, JobTarget(runner="local"))

    get_status.assert_not_called()
    assert record.runner == "local"


# ── Imported-model card titles ───────────────────────────────────────────────
# The name an imported record is born with, and how the registry keeps two of
# them apart. The derivation rules themselves live in tests/test_naming.py.


def _hub_reg(monkeypatch, tmp_path, root="root"):
    """A registry whose Hub probe always reports a root-level checkpoint, so a
    hub import succeeds offline."""
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    return JobRegistry(tmp_path / root)


def test_imported_name_is_the_derived_task_not_the_repo_id(monkeypatch, tmp_path) -> None:
    """No "Imported ·" prefix (the card's provenance chip already says it) and
    no namespace or policy token (the policy row says that) — just the task, so
    the title's pixels go to the one thing nothing else on the card states."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")

    assert rec.name == "orange_box"
    assert "Imported" not in rec.name


def test_imported_name_from_a_local_dir(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "smolvla_me_orange_box_2026-08-03_12-53-30"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    # No namespace on a path, so only the policy token and the timestamp go.
    assert reg.register_imported(str(model)).name == "me_orange_box"


def test_imported_name_keeps_a_community_repo_name(monkeypatch, tmp_path) -> None:
    """A repo with no generated timestamp was named by a human, so nothing but
    the namespace is dropped — the card falls back to a middle-ellipsized
    basename rather than guessing at structure that isn't there."""
    reg = _hub_reg(monkeypatch, tmp_path)
    assert reg.register_imported("lerobot/smolvla_base").name == "smolvla_base"


def test_explicit_import_name_wins_over_the_derivation(monkeypatch, tmp_path) -> None:
    """POST /jobs/import may carry a name. It's the user's, so neither the
    derivation nor the collision pass may overwrite it."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported(
        "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
        name="Orange picker v2",
    )
    assert rec.name == "Orange picker v2"

    reg2 = _hub_reg(monkeypatch, tmp_path)  # reload from disk
    assert reg2.get(rec.id).name == "Orange picker v2"


def test_rename_alias_survives_the_collision_pass(monkeypatch, tmp_path) -> None:
    """A user-set display_name always wins on the card; re-deriving `name`
    underneath it must not disturb the alias."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    reg.rename(rec.id, "Orange picker")

    reg2 = _hub_reg(monkeypatch, tmp_path)
    reloaded = reg2.get(rec.id)
    assert reloaded.display_name == "Orange picker"
    assert reloaded.name == "orange_box"


def test_two_imports_of_one_task_are_disambiguated(monkeypatch, tmp_path) -> None:
    """Both titles derive to "orange_box". BOTH cards get the timestamp back as
    a suffix — leaving the first bare would make it the ambiguous one."""
    reg = _hub_reg(monkeypatch, tmp_path)
    a = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    b = reg.register_imported("makermods/act_makermods_orange_box_2026-08-05_09-00-00")

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03)"
    assert names[b.id] == "orange_box (2026-08-05)"


def test_same_day_imports_escalate_to_the_time(monkeypatch, tmp_path) -> None:
    reg = _hub_reg(monkeypatch, tmp_path)
    a = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    b = reg.register_imported("makermods/act_makermods_orange_box_2026-08-03_18-04-00")

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03 12:53)"
    assert names[b.id] == "orange_box (2026-08-03 18:04)"


def test_collision_suffixes_are_recomputed_not_accumulated(monkeypatch, tmp_path) -> None:
    """The pass is idempotent: it re-derives from the source every time, so a
    reload (or a third import) never stacks "(date) (date)" onto a title."""
    reg = _hub_reg(monkeypatch, tmp_path)
    reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    reg.register_imported("makermods/act_makermods_orange_box_2026-08-05_09-00-00")

    reg2 = _hub_reg(monkeypatch, tmp_path)
    reg3 = _hub_reg(monkeypatch, tmp_path)
    assert sorted(r.name for r in reg3.list(limit=100)) == sorted(r.name for r in reg2.list(limit=100))
    assert all(r.name.count("(") <= 1 for r in reg3.list(limit=100))


def test_legacy_imported_name_is_re_derived_on_load(monkeypatch, tmp_path) -> None:
    """Records written before titles were derived carry the old
    "Imported · <repo id>" name. Boot upgrades them in place, so the fix reaches
    the cards already in the user's library and not only the next import."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    root = tmp_path / "root"
    job_dir = root / "smolvla_imported_2026-08-03_12-53-30"
    job_dir.mkdir(parents=True)
    legacy = JobRecord(
        id=job_dir.name,
        name="Imported · makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="smolvla"),
        output_dir="",
        started_at=1.0,
        ended_at=1.0,
        runner="imported",
        hf_repo_id="makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
    )
    (job_dir / "job.json").write_text(legacy.model_dump_json(indent=2))

    reg = JobRegistry(root)
    assert reg.get(legacy.id).name == "orange_box"
    # Persisted, not just fixed in memory.
    assert _json.loads((job_dir / "job.json").read_text())["name"] == "orange_box"


def _typed_hub_reg(monkeypatch, tmp_path, policy_by_repo, root="root"):
    """`_hub_reg` plus a readable checkpoint config, so each import lands with a
    real `config.policy_type` — the field the card's Policy row renders, and the
    one the collision pass groups on."""
    reg = _hub_reg(monkeypatch, tmp_path, root=root)
    monkeypatch.setattr(
        "makermodslab.jobs._read_checkpoint_config",
        # A hub checkpoint's ref is "<repo_id>@root" (see _list_imported_hub).
        lambda ckpt: {"type": policy_by_repo[ckpt.ref.split("@", 1)[0]]},
    )
    return reg


def test_two_policies_of_one_task_are_not_disambiguated(monkeypatch, tmp_path) -> None:
    """Both derive to "orange_box", but one card says ACT and the other SmolVLA
    a line below the title — they are already told apart, so a date suffix would
    only crowd out the task name it is meant to clarify."""
    act_repo = "makermods/act_makermods_orange_box_2026-08-05_09-00-00"
    smolvla_repo = "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30"
    reg = _typed_hub_reg(monkeypatch, tmp_path, {act_repo: "act", smolvla_repo: "smolvla"})
    a = reg.register_imported(smolvla_repo)
    b = reg.register_imported(act_repo)

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box"
    assert names[b.id] == "orange_box"


# ── Resume is only for a run that stopped short ──────────────────────────────
# A completed run's LR schedule is spent (SmolVLA's preset cosine-decays to a
# 2.5e-6 floor over a fixed 30k-step horizon), so a continuation trains at floor
# LR and the flat loss curve reads as convergence. The UI hides the button; this
# is the backend half, which also catches a direct API call. Blanket by
# decision — no per-policy exceptions.


def _resumable_source(tmp_path, state: str, *, job_id: str = "src", steps: int = 200):
    """A local run record with one real, fully-saved checkpoint at step 100."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    run_dir = tmp_path / job_id / "run"
    run_dir.mkdir(parents=True)
    _make_checkpoint(run_dir, 100)
    reg = JobRegistry(tmp_path / "root")
    reg._records[job_id] = JobRecord(
        id=job_id,
        name="run",
        state=state,
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=steps),
        output_dir=str(run_dir),
        started_at=0.0,
        runner="local",
    )
    return reg


def _assert_nothing_was_created(reg) -> None:
    """A synchronous refusal must leave NO trace on disk either.

    Cheap guards stayed in front of the record when the base-checkpoint download
    moved behind it (see the deferred-materialization section), and a job
    directory or a log file appearing here would be the first sign that one of
    them slipped past. `_resumable_source` keeps its source run outside the
    registry root, so the root is empty until a record is created."""
    assert list(reg._output_root.iterdir()) == []


def _resume_request(job_id: str = "src"):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_job_id=job_id,
    )


def test_start_refuses_to_resume_a_completed_run(tmp_path) -> None:
    """The point of the gate: a `done` source is refused with a 400-shaped
    ValueError that names fine-tuning as the way forward. No record created."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    assert [r.id for r in reg.list(limit=10)] == ["src"]
    _assert_nothing_was_created(reg)


def test_start_refuses_to_resume_a_completed_cloud_run(tmp_path) -> None:
    """The refusal is on the source's STATE, not its runner, so it lands before
    the local/cloud branch and covers both."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    reg._records["src"].runner = "hf_cloud"
    reg._records["src"].hf_repo_id = "user/some-model"
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))


@pytest.mark.parametrize("state", ["interrupted", "failed"])
def test_start_still_resumes_a_run_that_stopped_short(tmp_path, state) -> None:
    """The gate must not swallow the case resume exists for: a run that ended
    below its target still resumes, and still resolves its --config_path."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, state)
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))

    assert record.config.resume is True
    assert record.config.config_path.endswith("train_config.json")
    assert "checkpoints/100" in record.config.config_path


# ── …and only toward a target it can actually reach ─────────────────────────
# lerobot trains `range(resumed_step, steps)`, so a target at or below the
# checkpoint is an empty range: the run does nothing, exits 0, and the registry
# gets a `done` phantom claiming a target it never trained toward. The endpoint
# pre-flight in server.py catches the case where the REQUEST names its step;
# these cover the registry's own guard, which runs after the step is resolved
# and so also covers "latest checkpoint" (resume_from_step=None) requests.


def _resume_request_at(steps: int, *, step: int | None = None):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_job_id="src",
        resume_from_step=step,
        steps=steps,
    )


@pytest.mark.parametrize("steps", [0, 50, 100])
def test_start_refuses_a_resume_target_at_or_below_the_checkpoint(tmp_path, steps) -> None:
    """The boundary is strict: equal to the checkpoint step trains nothing, and
    so does anything below it. `steps=0` is refused with the rest — a request's
    own target is a required field, so 0 means "train nothing", not "unset"."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")  # checkpoint at step 100
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="would train nothing"),
    ):
        reg.start(_resume_request_at(steps, step=100), JobTarget(runner="local"))

    assert [r.id for r in reg.list(limit=10)] == ["src"]
    _assert_nothing_was_created(reg)


def test_start_allows_a_resume_target_one_step_above_the_checkpoint(tmp_path) -> None:
    """Just past the boundary is a real (if short) continuation, and must not be
    swept up by the guard."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request_at(101, step=100), JobTarget(runner="local"))

    assert record.config.steps == 101


def test_start_refuses_a_latest_checkpoint_resume_below_its_target(tmp_path) -> None:
    """The hole the endpoint's pre-flight can't see: the request leaves the step
    to the registry ("latest checkpoint"), so `resume_from_step` is None when
    the endpoint looks. The registry re-checks once it has resolved step 100."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    request = _resume_request_at(100, step=None)
    assert request.resume_from_step is None  # the pre-flight's blind spot

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="checkpoint step 100"),
    ):
        reg.start(request, JobTarget(runner="local"))

    _assert_nothing_was_created(reg)


def test_start_step_target_guard_leaves_fresh_runs_alone(tmp_path) -> None:
    """It is a RESUME guard. A fresh run has no checkpoint step to be above, and
    `_resume_start_step` returns None for it, so nothing is refused."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )

    assert record.config.resume is False


# ── …and only ONCE: sticks, not forks ───────────────────────────────────────
# User decision 2026-08-07. A run may be continued once, so a resume whose
# source already has a child is refused at CREATION time. Legacy forks on disk
# are untouched by this — nothing here runs at load or list time (the lineage
# section at the end of this file covers that half).


def _child_of(reg, parent_id: str, *, job_id: str = "child", state: str = "interrupted"):
    """Register a run that continues `parent_id`, i.e. gives it a child."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    reg._records[job_id] = JobRecord(
        id=job_id,
        name="continuation",
        state=state,
        config=TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            resume=True,
            resume_from_job_id=parent_id,
        ),
        output_dir=f"/nonexistent/{job_id}",
        started_at=1.0,
        runner="local",
    )
    return reg._records[job_id]


# ── …and only when the request actually ASKS to resume ──────────────────────
# A resume source with `resume` left false used to launch as an ordinary fresh
# run: it skipped every guard below (they all sit under `if config.resume`) yet
# still persisted resume_from_job_id, which build_child_index reads as a
# lineage edge. That produced a run that trained from scratch while counting as
# a continuation — superseding a parent it never continued, and blocking that
# parent from being resumed for real.


def test_start_refuses_a_resume_source_without_the_resume_flag(tmp_path) -> None:
    """The combination the live repro produced: a spurious extra child of an
    existing lineage, created by a request that never said 'resume'."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    request = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=False,  # the hole
        resume_from_job_id="src",
    )
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="resume_from_job_id was given but 'resume' is false"),
    ):
        reg.start(request, JobTarget(runner="local"))

    # No record, so no phantom lineage edge either.
    assert [r.id for r in reg.list(limit=10)] == ["src"]
    assert reg.get("src").child_ids == []
    _assert_nothing_was_created(reg)


def test_start_refuses_a_resume_step_without_the_resume_flag(tmp_path) -> None:
    """Same contract, named by the field actually supplied."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="resume_from_step was given but 'resume' is false"),
    ):
        reg.start(
            TrainingRequest(
                dataset_repo_id="user/ds",
                policy_type="act",
                resume=False,
                resume_from_step=100,
            ),
            JobTarget(runner="local"),
        )


def test_start_still_allows_a_plain_fresh_run(tmp_path) -> None:
    """The other direction: neither field set, so nothing is refused. Without
    this the guard could pass by rejecting every fresh run ever launched."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )

    assert record.config.resume is False
    assert record.config.resume_from_job_id is None
    # ...and it is not anybody's child.
    assert reg.get("src").child_ids == []


def test_start_refuses_a_finetune_step_without_a_finetune_source(tmp_path) -> None:
    """The analogous fine-tune inconsistency. There is no `finetune` boolean —
    the id IS the mode — so the mismatch is a step with nothing to take it
    from, which used to be dropped silently and trained from scratch."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="finetune_from_step was given without"),
    ):
        reg.start(
            TrainingRequest(
                dataset_repo_id="user/ds",
                policy_type="act",
                finetune_from_step=100,
            ),
            JobTarget(runner="local"),
        )


def test_start_refuses_to_resume_an_already_continued_run(tmp_path) -> None:
    """The sticks rule: one continuation per run. A second would fork the
    lineage, so it is refused — naming the child, which is what lets the HTTP
    layer tell the user which run to delete first. No record created."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobAlreadyContinuedError, JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    _child_of(reg, "src")

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(JobAlreadyContinuedError) as excinfo,
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    assert excinfo.value.job_id == "src"
    assert excinfo.value.child_ids == ["child"]
    assert set(reg._records) == {"src", "child"}
    _assert_nothing_was_created(reg)


def test_start_refusal_ignores_a_finetune_child(tmp_path) -> None:
    """A fine-tune is not a resume edge anywhere else (the child index, the
    delete guard), and it isn't one here either: its source keeps its own
    identity and stays continuable."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRecord, JobTarget
    from makermodslab.train import TrainingRequest

    reg = _resumable_source(tmp_path, "interrupted")
    reg._records["ft"] = JobRecord(
        id="ft",
        name="finetune",
        # `done`, not `running` — a live local run would trip the one-at-a-time
        # mutex and mask the thing under test.
        state="done",
        config=TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            finetune_from_job_id="src",
        ),
        output_dir="/nonexistent/ft",
        started_at=1.0,
        runner="local",
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))

    assert record.config.resume is True


def test_start_refusal_defers_to_the_completed_source_refusal(tmp_path) -> None:
    """Both refusals can be true of one legacy source. The `done` one wins, and
    must: telling the user to delete a child to unlock a resume that would then
    be refused for a spent LR schedule is worse guidance than "fine-tune"."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    _child_of(reg, "src")

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))


def test_deleting_the_tip_frees_its_parent_to_be_resumed(tmp_path) -> None:
    """The recovery the refusal's message promises, end to end at the registry
    level: the fork is blocked, deleting the existing continuation makes its
    parent a leaf again, and the same resume then succeeds."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobAlreadyContinuedError, JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    _child_of(reg, "src")

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(JobAlreadyContinuedError),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    reg.delete("child")

    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))

    assert record.config.resume is True
    assert "checkpoints/100" in record.config.config_path
    # ...and the new continuation is now the one child `src` is allowed.
    assert reg.get("src").child_ids == [record.id]


def test_start_refuses_a_second_continuation_of_a_cloud_source(tmp_path) -> None:
    """The refusal is on the LINEAGE, not the runner, so it lands before the
    local/cloud branch that moves the checkpoint — same placement as the
    completed-source refusal above."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobAlreadyContinuedError, JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    reg._records["src"].runner = "hf_cloud"
    reg._records["src"].hf_repo_id = "user/some-model"
    _child_of(reg, "src")

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(JobAlreadyContinuedError),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))


# ── …and on either runner, once the checkpoint can get there (F7) ───────────
# A continuation may cross runners in both directions now. What each direction
# has to do first is move the parent's checkpoint to wherever the trainer will
# run: DOWN from the Hub for a cloud parent continued locally (MT4 — the old
# code pointed --config_path at a pod path that never existed here), and UP to
# the Hub for a local parent continued on the cloud (MT42 — the old code let the
# pod start a fresh run at step 0 while the record called itself a resume).
# Both transfers happen off-request in the preparing thread, so these join it
# the way the fine-tune materialization tests do.
#
# Reuses the section's helpers above; a cloud parent is the same record with its
# runner/repo flipped, exactly as the done-source pair does it.


def _cloud_parent(reg, *, job_id: str = "src"):
    """Turn a `_resumable_source` record into a cloud run in place."""
    reg._records[job_id].runner = "hf_cloud"
    reg._records[job_id].hf_repo_id = "user/some-model"
    return reg


def _local_to_cloud_request(*, consent: bool = True, job_id: str = "src"):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_job_id=job_id,
        upload_resume_checkpoint=consent,
    )


@pytest.fixture
def cloud_preflight(monkeypatch):
    """Keep the cloud dataset preflight off the network — it sits ahead of the
    resume block, so without this it, not the code under test, is what fails."""
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr("makermodslab.datasets.hub_copy_has_data", lambda repo_id: True)


# ── cloud parent → Local ─────────────────────────────────────────────────────


def test_cloud_parent_resumed_locally_downloads_the_chosen_step(tmp_path, monkeypatch) -> None:
    """The parent's Hub checkpoint is materialized HERE — whole, including
    training_state/ — and config_path points into it, exactly as a local→local
    resume would. The step is the one the resolver chose, not lerobot's
    latest-only Hub rule."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100")),
    )
    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_resume_snapshot(tmp_path, seen))
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))
        _join_prepare(reg, record.id)

    # The WHOLE step tree, not just pretrained_model/ — a resume without the
    # optimizer state is a fine-tune wearing a resume label.
    assert seen["allow_patterns"] == ["checkpoints/000100/*"]
    record = reg._records[record.id]
    assert record.config.config_path.endswith("checkpoints/000100/pretrained_model/train_config.json")
    assert Path(record.config.config_path).is_file()
    assert record.state == "running"
    assert fake_runner.start.called


# ── local parent → Cloud ─────────────────────────────────────────────────────


def test_local_parent_resumed_on_the_cloud_uploads_then_submits(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The checkpoint goes up first — whole tree, PRIVATE repo — and only then
    is the cloud job submitted, pointed at what was just uploaded."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    assert api.created == [
        {"repo_id": "alice/src_checkpoints", "repo_type": "model", "private": True, "exist_ok": True}
    ]
    assert api.uploaded[0]["path_in_repo"] == "checkpoints/100"
    assert api.uploaded[0]["folder_path"].endswith("checkpoints/100")
    record = reg._records[record.id]
    assert record.config.resume_from_hub_repo == "alice/src_checkpoints"
    assert record.config.resume_from_hub_step == "100"
    assert record.config.resume_from_uploaded_checkpoint is True
    # Never a host path: that is what the pod cannot resolve.
    assert record.config.config_path is None
    assert record.metrics.current_step == 100
    assert fake_runner.start.called


def test_local_parent_resumed_on_the_cloud_records_the_upload_on_the_parent(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Where the bytes went is remembered on the PARENT, and survives a reload —
    that record is what stops the next continuation of the same step from
    pushing the same GBs again."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()):
        record = reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    parent = reg._records["src"]
    assert parent.checkpoints_hub_repo_id == "alice/src_checkpoints"
    assert parent.checkpoints_hub_steps == ["100"]
    # The parent record is persisted, so a fresh registry over the same root
    # reads the same answer.
    reloaded = JobRegistry(reg._output_root)._records["src"]
    assert reloaded.checkpoints_hub_repo_id == "alice/src_checkpoints"
    assert reloaded.checkpoints_hub_steps == ["100"]


def test_local_parent_resumed_on_the_cloud_reuses_an_earlier_upload(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Second continuation of the same step: the checkpoint is already on the
    Hub and confirmed there, so nothing is uploaded — and no consent is asked
    for, because nothing new is disclosed."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    reg._records["src"].checkpoints_hub_repo_id = "alice/src_checkpoints"
    reg._records["src"].checkpoints_hub_steps = ["100"]
    api = _FakeUploadApi(_hub_checkpoint_files("100"))
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(
            _local_to_cloud_request(consent=False),
            JobTarget(runner="hf_cloud", flavor="t4-small"),
        )

    assert api.uploaded == []
    assert record.config.resume_from_hub_repo == "alice/src_checkpoints"
    assert record.config.resume_from_hub_step == "100"
    # Nothing had to move, so the job went straight to the runner — no
    # preparing thread at all.
    assert record.id not in reg._prepare_threads
    assert fake_runner.start.called


def test_local_parent_resumed_on_the_cloud_re_uploads_when_the_hub_lost_it(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The record is a hint, not the truth: a staging repo that has since been
    deleted (or was half-pushed) produces a fresh upload rather than a job that
    dies looking for bytes."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    reg._records["src"].checkpoints_hub_repo_id = "alice/src_checkpoints"
    reg._records["src"].checkpoints_hub_steps = ["100"]
    api = _FakeUploadApi(_hub_checkpoint_files("100", with_optimizer=False))
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()):
        record = reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    assert [u["path_in_repo"] for u in api.uploaded] == ["checkpoints/100"]


def test_local_parent_resumed_on_the_cloud_refuses_without_consent(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """An upload is a disclosure, so it never happens as a side effect of
    Continue: without the form's explicit consent the launch is refused, nothing
    is uploaded, and no record is left behind."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="only on this machine"),
    ):
        reg.start(
            _local_to_cloud_request(consent=False),
            JobTarget(runner="hf_cloud", flavor="t4-small"),
        )

    assert api.created == [] and api.uploaded == []
    assert list(reg._records) == ["src"]
    _assert_nothing_was_created(reg)


def test_local_parent_resumed_on_the_cloud_refuses_without_hf_auth(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """No Hub identity ⇒ no namespace to upload into. Refused with the login
    instruction rather than failing later inside the runner."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: None)
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="signed in"),
    ):
        reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert api.uploaded == []
    _assert_nothing_was_created(reg)


def test_local_parent_resumed_on_the_cloud_refuses_when_offline(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Offline mode disables every Hub write, so the upload this continuation
    depends on is impossible — say so instead of trying."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeUploadApi())
    monkeypatch.setattr("makermodslab.jobs.hf_hub_offline", lambda: True)
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Offline mode"),
    ):
        reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    _assert_nothing_was_created(reg)


def test_local_parent_resumed_on_the_cloud_never_starts_a_fresh_run(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """MT42, as an invariant: when the checkpoint cannot be put where the pod
    will look for it, the job FAILS — it does not quietly become a fresh run at
    step 0 on rented hardware while the record calls itself a continuation."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    api = _FakeUploadApi()
    api.upload_error = RuntimeError("413 payload too large")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    failed = reg._records[record.id]
    assert failed.state == "failed"
    assert "413 payload too large" in failed.error_message
    # The one assertion that matters: no cloud job was ever submitted.
    assert not fake_runner.start.called


def test_cross_runner_resume_still_refuses_a_completed_parent(tmp_path, monkeypatch, cloud_preflight) -> None:
    """Only the runner-mismatch refusal went away. A parent that spent its LR
    schedule is still unresumable — on either runner, in either direction."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
    _assert_nothing_was_created(reg)


# ── local base → Cloud FINE-TUNE (F7's fourth quadrant) ──────────────────────
# The same crossing as the resume above, for the other mode: a base checkpoint
# that exists only on this machine, fine-tuned on rented hardware. What moves is
# the WEIGHTS ONLY — a fine-tune starts a fresh optimizer at step 0 and never
# reads training_state/, which is the bigger half — into the same private
# per-source staging repo a cloud resume uses. The request is then rewritten to
# the 'repo@checkpoints/<step>' ref the pod already knows how to materialize, so
# the record describes the run that actually happens rather than a host path the
# container could never resolve.


def _local_finetune_request(*, consent: bool = True, job_id: str = "src", step: int | None = None):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        finetune_from_job_id=job_id,
        finetune_from_step=step,
        upload_finetune_checkpoint=consent,
    )


def test_local_base_finetuned_on_the_cloud_uploads_weights_then_submits(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The base's weights go up first — pretrained_model/ only, PRIVATE repo —
    and only then is the cloud job submitted, pointed at what was just staged."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")  # a finished local run IS a base
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    assert api.created == [
        {"repo_id": "alice/src_checkpoints", "repo_type": "model", "private": True, "exist_ok": True}
    ]
    # ONE upload, and it is the weights subtree — never the whole checkpoint.
    assert [u["path_in_repo"] for u in api.uploaded] == ["checkpoints/100/pretrained_model"]
    assert api.uploaded[0]["folder_path"].endswith("checkpoints/100/pretrained_model")
    record = reg._records[record.id]
    assert record.config.policy_pretrained_path == "alice/src_checkpoints@checkpoints/100"
    # A fine-tune stays a FRESH run: nothing about it is a continuation.
    assert record.config.resume is False and record.config.resume_from_hub_repo is None
    assert fake_runner.start.called


def test_local_base_finetuned_on_the_cloud_records_the_ref_before_the_upload(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The rewrite happens at record creation, not after the bytes land: the
    record handed back from `start` — while the upload is still deferred —
    already names the Hub ref the pod will read, so no persisted config ever
    claims the run trains from a path only this machine has."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    fake_runner.hf_job_url.return_value = "https://hf.co/jobs/hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        assert record.config.policy_pretrained_path == "alice/src_checkpoints@checkpoints/100"
        _join_prepare(reg, record.id)

    reloaded = JobRegistry(reg._output_root)._records[record.id]
    assert reloaded.config.policy_pretrained_path == "alice/src_checkpoints@checkpoints/100"


def test_local_base_finetuned_on_the_cloud_records_the_upload_on_the_source(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Where the weights went is remembered on the SOURCE — keyed off the
    child's finetune_from_job_id, since a fine-tune has no resume edge — and
    survives a reload, so the next fine-tune of the same step reuses it."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()):
        record = reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    source = reg._records["src"]
    assert source.checkpoints_hub_repo_id == "alice/src_checkpoints"
    assert source.checkpoints_hub_steps == ["100"]
    reloaded = JobRegistry(reg._output_root)._records["src"]
    assert reloaded.checkpoints_hub_repo_id == "alice/src_checkpoints"
    assert reloaded.checkpoints_hub_steps == ["100"]


def test_local_base_finetuned_on_the_cloud_reuses_an_earlier_staging(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Second fine-tune of the same base: the weights are already staged and
    confirmed there, so nothing is uploaded — and no consent is asked for,
    because nothing new is disclosed. The confirmation uses the WEIGHTS-ONLY
    rule, so the training_state/ a staging upload never pushed doesn't read as
    a broken repo."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    reg._records["src"].checkpoints_hub_repo_id = "alice/src_checkpoints"
    reg._records["src"].checkpoints_hub_steps = ["100"]
    api = _FakeUploadApi(_hub_pretrained_files("100"))
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(
            _local_finetune_request(consent=False),
            JobTarget(runner="hf_cloud", flavor="t4-small"),
        )

    assert api.uploaded == []
    assert record.config.policy_pretrained_path == "alice/src_checkpoints@checkpoints/100"
    # Nothing had to move, so the job went straight to the runner.
    assert record.id not in reg._prepare_threads
    assert fake_runner.start.called


def test_local_base_finetuned_on_the_cloud_refuses_without_consent(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """An upload is a disclosure, so it never happens as a side effect of
    picking a base model: without the form's explicit consent the launch is
    refused, nothing is uploaded, and no record is left behind."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Confirm the upload in the training form"),
    ):
        reg.start(
            _local_finetune_request(consent=False),
            JobTarget(runner="hf_cloud", flavor="t4-small"),
        )

    assert api.created == [] and api.uploaded == []
    assert list(reg._records) == ["src"]
    _assert_nothing_was_created(reg)


def test_local_base_finetuned_on_the_cloud_refuses_when_offline(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """Offline mode disables every Hub write, so the staging this fine-tune
    depends on is impossible — say so instead of trying."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeUploadApi())
    monkeypatch.setattr("makermodslab.jobs.hf_hub_offline", lambda: True)
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Offline mode"),
    ):
        reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    _assert_nothing_was_created(reg)


def test_local_base_finetuned_on_the_cloud_refuses_without_hf_auth(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """No Hub identity ⇒ no namespace to stage into. Refused with the login
    instruction rather than failing later inside the runner."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: None)
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="signed in"),
    ):
        reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert api.uploaded == []
    _assert_nothing_was_created(reg)


def test_local_base_finetuned_on_the_cloud_stages_a_flat_import_at_step_zero(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The other local base shape: a flat imported directory that IS the
    pretrained_model, with no checkpoints/<step>/ around it to read the step
    from. Its listing offers exactly one checkpoint, at step 0, so the staged
    step dir is the zero-padded 0 — and the ref names it, rather than the repo
    root, so the pod materializes the weights that were actually uploaded."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget

    src = tmp_path / "flat_base"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "act"}))
    (src / "model.safetensors").write_bytes(b"weights")

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()):
        record = reg.start(
            _local_finetune_request(job_id=source.id),
            JobTarget(runner="hf_cloud", flavor="t4-small"),
        )
        _join_prepare(reg, record.id)

    assert [u["path_in_repo"] for u in api.uploaded] == ["checkpoints/000000/pretrained_model"]
    assert api.uploaded[0]["folder_path"] == str(src.resolve())
    assert (
        reg._records[record.id].config.policy_pretrained_path
        == f"alice/{source.id}_checkpoints@checkpoints/000000"
    )


def test_local_base_finetuned_on_the_cloud_never_starts_from_scratch(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """The fine-tune reading of MT42: when the weights can't be put where the
    pod will look for them, the job FAILS — it does not quietly become a
    from-scratch run on rented hardware while the record calls itself a
    fine-tune of somebody's checkpoint."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    api.upload_error = RuntimeError("413 payload too large")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_finetune_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    failed = reg._records[record.id]
    assert failed.state == "failed"
    assert "413 payload too large" in failed.error_message
    assert not fake_runner.start.called


def test_local_base_finetuned_locally_stages_nothing(tmp_path, monkeypatch) -> None:
    """The staging is a property of the CROSSING, not of the base: the same
    local base fine-tuned on this machine keeps the host path and touches the
    Hub not at all."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    api = _FakeUploadApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_finetune_request(consent=False), JobTarget(runner="local"))

    assert api.created == [] and api.uploaded == []
    assert record.config.policy_pretrained_path.endswith("checkpoints/100/pretrained_model")
    assert fake_runner.start.called


# ---------------------------------------------------------------------------
# MT2 — fine-tuning a selected Hub step must actually train from THAT step.
# The resolver used to return `ref.split("@")[0]`, so a picked step silently
# became repo-ROOT weights. One rule now: a step-suffixed ref names the
# checkpoint, and whoever will run the trainer materializes it — host-side for a
# local run, in the container for a cloud one.
# ---------------------------------------------------------------------------


def _fake_snapshot(tmp_path: Path, seen: dict):
    """A snapshot_download stand-in that records its kwargs and lays down the
    tree the real one would (so the caller's path arithmetic is exercised)."""

    def _download(**kwargs):
        seen.update(kwargs)
        root = tmp_path / "snapshot"
        for pattern in kwargs.get("allow_patterns") or []:
            (root / pattern.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    return _download


def test_download_hub_checkpoint_ref_scopes_to_the_selected_step(monkeypatch, tmp_path) -> None:
    """Only that step's pretrained_model/ is pulled, and the returned path is
    that directory — the whole point being that lerobot cannot address a Hub
    sub-path itself."""
    from makermodslab.jobs import download_hub_checkpoint_ref

    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    out = download_hub_checkpoint_ref("user/repo@checkpoints/000500")

    assert seen["repo_id"] == "user/repo"
    assert seen["allow_patterns"] == ["checkpoints/000500/pretrained_model/*"]
    assert out.endswith("/checkpoints/000500/pretrained_model")


def test_download_hub_checkpoint_ref_root_skips_the_heavy_trees(monkeypatch, tmp_path) -> None:
    """A '@root' ref is the whole repo, minus the per-step snapshots and
    optimizer state — neither is needed to load the policy and both can be GBs."""
    from makermodslab.jobs import download_hub_checkpoint_ref

    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    out = download_hub_checkpoint_ref("user/repo@root")

    assert seen["repo_id"] == "user/repo"
    assert seen["ignore_patterns"] == ["checkpoints/**", "training_state/**"]
    assert out == str(tmp_path / "snapshot")


def test_download_hub_checkpoint_ref_rejects_a_non_ref() -> None:
    from makermodslab.jobs import download_hub_checkpoint_ref

    with pytest.raises(ValueError, match="Unrecognised policy ref"):
        download_hub_checkpoint_ref("just-a-repo-id")


def test_download_hub_checkpoint_ref_widens_to_the_whole_step_for_a_resume(monkeypatch, tmp_path) -> None:
    """`with_training_state` is the one difference between "load these weights"
    and "continue this run": the optimizer state comes along. Opt-in because it
    is ~394 MB per step that a deploy or fine-tune never reads."""
    from makermodslab.jobs import download_hub_checkpoint_ref

    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    out = download_hub_checkpoint_ref("user/repo@checkpoints/000500", with_training_state=True)

    assert seen["allow_patterns"] == ["checkpoints/000500/*"]
    # The return value is still the pretrained_model dir, so the two modes agree
    # about what a ref resolves TO; its parent is the checkpoint dir.
    assert out.endswith("/checkpoints/000500/pretrained_model")


def test_download_hub_resume_checkpoint_returns_the_train_config(monkeypatch, tmp_path) -> None:
    """The value lerobot's --config_path wants, inside a tree whose parent
    directory is the checkpoint (which is how lerobot finds training_state/)."""
    from makermodslab.jobs import download_hub_resume_checkpoint

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_resume_snapshot(tmp_path, {}))

    out = Path(download_hub_resume_checkpoint("user/repo@checkpoints/000100"))

    assert out.name == "train_config.json" and out.is_file()
    assert (out.parent.parent / "training_state" / "optimizer_state.safetensors").is_file()


def test_download_hub_resume_checkpoint_refuses_an_incomplete_download(monkeypatch, tmp_path) -> None:
    """Checked on the bytes that actually landed, not just on the repo listing:
    an interrupted upload passes the listing check and then dies inside the
    trainer, which is precisely MT4."""
    from makermodslab.jobs import download_hub_resume_checkpoint

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_resume_snapshot(tmp_path, {}, complete=False)
    )

    with pytest.raises(ValueError, match="optimizer_state.safetensors"):
        download_hub_resume_checkpoint("user/repo@checkpoints/000100")


def test_localize_pretrained_path_passes_through_non_step_values(monkeypatch, tmp_path) -> None:
    """A local dir and a bare repo id are already loadable — lerobot resolves a
    repo root itself, so materializing one here would only duplicate its fetch."""
    from makermodslab.jobs import localize_pretrained_path

    def _no_downloads(**kwargs):
        raise AssertionError("nothing to materialize for a non-step value")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)
    assert localize_pretrained_path("user/some-model") == "user/some-model"
    assert localize_pretrained_path(str(tmp_path)) == str(tmp_path)


def test_localize_pretrained_path_reports_a_failed_download(monkeypatch) -> None:
    """A Hub failure becomes a 400-shaped ValueError naming the checkpoint,
    rather than a raw Hub exception surfacing as a 500."""
    from makermodslab.jobs import localize_pretrained_path

    def _boom(**kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)
    with pytest.raises(ValueError, match="Could not download the base checkpoint") as excinfo:
        localize_pretrained_path("user/repo@checkpoints/003000")
    assert "hub down" in str(excinfo.value)


def test_resolve_finetune_pretrained_path_keeps_the_selected_step(monkeypatch) -> None:
    """MT2 core: the step the user picked survives resolution. It used to be
    truncated to the repo id here, which is repo-ROOT weights."""
    from makermodslab.jobs import _resolve_finetune_pretrained_path

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("005000")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))

    assert _resolve_finetune_pretrained_path(_cloud_record(), 1000) == "user/act_ds_2026@checkpoints/001000"
    # step=None still means "the latest", now equally un-truncated.
    assert _resolve_finetune_pretrained_path(_cloud_record(), None) == "user/act_ds_2026@checkpoints/005000"


def test_resolve_finetune_pretrained_path_root_ref_stays_a_repo_id(tmp_path) -> None:
    """An imported repo whose ROOT is the model needs no materialization —
    lerobot loads a repo id directly, so handing it the bare id avoids a
    pointless download."""
    from unittest.mock import patch

    from makermodslab.jobs import JobRecord, _resolve_finetune_pretrained_path
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id="imported-src",
        name="imported",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="act"),
        output_dir="",
        started_at=0.0,
        runner="imported",
        hf_repo_id="user/flat_model",
    )
    with patch(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(["config.json", "model.safetensors"]),
    ):
        assert _resolve_finetune_pretrained_path(record, None) == "user/flat_model"


def _cloud_finetune_source(reg, repo_id: str = "user/act_ds_2026"):
    """A tracked cloud run in `reg` whose weights live on the Hub."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id="cloud-src",
        name="cloud src",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=10000),
        output_dir="",
        started_at=0.0,
        runner="hf_cloud",
        hf_repo_id=repo_id,
    )
    reg._records[record.id] = record
    return record


def _join_prepare(reg, job_id: str, timeout: float = 10.0) -> None:
    """Wait for the thread that materializes a fine-tune's base checkpoint.

    Joining the thread — rather than polling for its effect — is what keeps
    these tests off sleeps and out of the flaky column."""
    thread = reg._prepare_threads[job_id]
    thread.join(timeout)
    assert not thread.is_alive(), "the preparing thread never finished"


def test_finetune_start_local_materializes_the_selected_step(monkeypatch, tmp_path) -> None:
    """LOCAL target: the ref becomes a real directory on this machine before the
    trainer starts, and that directory is the SELECTED step.

    The download now happens off-request (see the deferred-materialization
    section below), so join the preparing thread before asserting on its
    effect."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    seen: dict = {}
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("003000"))
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    monkeypatch.setattr("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner)

    record = reg.start(
        TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            finetune_from_job_id=source.id,
            finetune_from_step=3000,
        ),
        JobTarget(runner="local"),
    )
    _join_prepare(reg, record.id)

    resolved = record.config.policy_pretrained_path
    assert "@" not in resolved  # materialized, not a ref
    assert resolved.endswith("/checkpoints/003000/pretrained_model")
    assert seen["allow_patterns"] == ["checkpoints/003000/pretrained_model/*"]
    # The trainer really was handed the materialized directory, not the ref.
    assert fake_runner.start.call_args.args[1].policy_pretrained_path == resolved


def test_finetune_start_cloud_keeps_the_step_ref(monkeypatch, tmp_path) -> None:
    """CLOUD target: the ref is passed through untouched — a host path means
    nothing on the pod, so the container materializes the same ref itself. The
    host must NOT download the weights for a run that happens elsewhere."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    def _no_downloads(**kwargs):
        raise AssertionError("the host must not download a cloud run's base weights")

    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("003000"))
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr("makermodslab.datasets.hub_copy_has_data", lambda repo_id: True)
    monkeypatch.setattr("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock())

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)

    record = reg.start(
        TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            finetune_from_job_id=source.id,
            finetune_from_step=3000,
        ),
        JobTarget(runner="hf_cloud", flavor="a10g-small"),
    )

    assert record.config.policy_pretrained_path == "user/act_ds_2026@checkpoints/003000"


# ---------------------------------------------------------------------------
# The base-checkpoint download happens AFTER the record exists.
#
# Materializing a Hub checkpoint for a local fine-tune is minutes and gigabytes.
# Doing it inside POST /jobs/training meant the request hung with nothing on
# screen and no job to look at. The record + log file are now created first and
# the download runs in a background thread, so the monitor opens immediately and
# tails the progress lines. Everything that can refuse the request cheaply still
# refuses it synchronously, before any record exists.
# ---------------------------------------------------------------------------


def _patch_hub_for_finetune(monkeypatch, tmp_path, policy_type: str = "act"):
    """Stand-ins for the CHEAP Hub reads a fine-tune start makes synchronously:
    the source's checkpoint listing, and the checkpoint's own config.json."""
    cfg_file = tmp_path / "base_config.json"
    cfg_file.write_text(_json.dumps({"type": policy_type}))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("003000"))
    )
    monkeypatch.setattr("makermodslab.jobs.hf_hub_download", lambda **kw: str(cfg_file))


def _hub_finetune_request(source_id: str, policy_type: str = "act"):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type=policy_type,
        finetune_from_job_id=source_id,
        finetune_from_step=3000,
    )


def _gated_snapshot(tmp_path, *, started: threading.Event, release: threading.Event):
    """_fake_snapshot, but it parks in the middle so a test can observe the
    world WHILE the download is in flight."""
    seen: dict = {}
    inner = _fake_snapshot(tmp_path, seen)

    def _download(**kwargs):
        started.set()
        assert release.wait(timeout=10), "the gated download was never released"
        return inner(**kwargs)

    return _download


def _fake_local_runner(monkeypatch):
    from unittest.mock import MagicMock

    runner = MagicMock()
    runner.pid.return_value = 4242
    monkeypatch.setattr("makermodslab.jobs.LocalJobRunner", lambda *a, **k: runner)
    return runner


def test_local_finetune_start_returns_before_the_download_finishes(monkeypatch, tmp_path) -> None:
    """The whole point: the caller gets a job id (and a log file with something
    in it) while the gigabytes are still moving."""
    from makermodslab.jobs import JobRegistry, JobTarget

    started, release = threading.Event(), threading.Event()
    _patch_hub_for_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _gated_snapshot(tmp_path, started=started, release=release),
    )

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    fake_runner = _fake_local_runner(monkeypatch)

    record = reg.start(_hub_finetune_request(source.id), JobTarget(runner="local"))
    assert started.wait(timeout=10), "the download never started"

    # Mid-download: the record is real, visible, and running — and no trainer
    # has been spawned yet.
    assert record.state == "running"
    assert reg.get(record.id).state == "running"
    assert fake_runner.start.call_count == 0

    # The monitor's two log sources both have the opening line: the file it
    # seeds from on mount, and the live drain it polls every second.
    log_path = reg._output_root / record.id / "log.jsonl"
    assert log_path.exists()
    assert "Preparing fine-tune" in log_path.read_text()
    assert any("Preparing fine-tune" in line.message for line in reg.drain_logs(record.id))
    assert "003000" in reg.read_persisted_logs(record.id)[0].message

    release.set()
    _join_prepare(reg, record.id)

    final = reg.get(record.id)
    assert final.state == "running"
    assert final.process_pid == 4242
    assert fake_runner.start.call_count == 1
    assert "@" not in final.config.policy_pretrained_path


def test_local_finetune_download_failure_fails_the_record(monkeypatch, tmp_path) -> None:
    """The failure that used to be an HTTP 400 now lands ON the record, with the
    same wording — no orphan record, no job stuck at 'running'."""
    from makermodslab.jobs import JobRegistry, JobTarget

    def _boom(**kwargs):
        raise RuntimeError("hub down")

    _patch_hub_for_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    fake_runner = _fake_local_runner(monkeypatch)

    record = reg.start(_hub_finetune_request(source.id), JobTarget(runner="local"))
    _join_prepare(reg, record.id)

    final = reg.get(record.id)
    assert final.state == "failed"
    assert "Could not download the base checkpoint" in final.error_message
    assert "hub down" in final.error_message
    # No process ever existed, so no synthetic exit code is invented for one.
    assert final.exit_code is None
    assert final.ended_at is not None
    assert fake_runner.start.call_count == 0
    # The stand-in runner is gone, so the watchdog has nothing left to finalise.
    assert record.id not in reg._runners
    # Persisted, not just fixed in memory.
    meta = _json.loads((reg._output_root / record.id / "job.json").read_text())
    assert meta["state"] == "failed"


def test_stop_during_the_download_is_interrupted(monkeypatch, tmp_path) -> None:
    """Stop must work while the base checkpoint is downloading. huggingface_hub
    can't be aborted mid-flight, so the cancel takes effect when the download
    returns — before the trainer is spawned — and reads as a deliberate stop."""
    from makermodslab.jobs import _PREPARE_STOPPED_MESSAGE, JobRegistry, JobTarget

    started, release = threading.Event(), threading.Event()
    _patch_hub_for_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _gated_snapshot(tmp_path, started=started, release=release),
    )

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    fake_runner = _fake_local_runner(monkeypatch)

    record = reg.start(_hub_finetune_request(source.id), JobTarget(runner="local"))
    assert started.wait(timeout=10), "the download never started"

    # Stop lands while the bytes are still moving; the record can only settle
    # once the download returns, so this call reports the job as still running.
    assert reg.stop(record.id).state == "running"
    release.set()
    _join_prepare(reg, record.id)

    final = reg.get(record.id)
    assert final.state == "interrupted"
    assert final.error_message == _PREPARE_STOPPED_MESSAGE
    assert "exited with code" not in (final.error_message or "")
    assert final.exit_code is None
    # The stop is honoured by NOT starting the trainer we were about to start.
    assert fake_runner.start.call_count == 0
    assert record.id not in reg._runners


def test_a_download_bound_job_refuses_a_second_local_run(monkeypatch, tmp_path) -> None:
    """The local mutex covers the download window too — the machine is spoken
    for from the moment the record exists, not from the trainer's first step."""
    from makermodslab.jobs import JobAlreadyRunningError, JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    started, release = threading.Event(), threading.Event()
    _patch_hub_for_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _gated_snapshot(tmp_path, started=started, release=release),
    )

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    _fake_local_runner(monkeypatch)

    record = reg.start(_hub_finetune_request(source.id), JobTarget(runner="local"))
    assert started.wait(timeout=10), "the download never started"

    with pytest.raises(JobAlreadyRunningError):
        reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act"),
            JobTarget(runner="local"),
        )

    release.set()
    _join_prepare(reg, record.id)


def test_download_progress_logs_are_readable_not_per_chunk() -> None:
    """The progress hook has to be worth reading: snapshot_download calls back
    per CHUNK, and one log line per chunk would bury the run's own output. Drive
    the tqdm class the way huggingface_hub does — one shared bytes bar plus a
    file-count bar — and check the cadence and the wording."""
    from makermodslab.jobs import _DownloadProgressLogger, make_snapshot_progress_tqdm

    lines: list[str] = []
    tqdm_class = make_snapshot_progress_tqdm(_DownloadProgressLogger(lines.append, "012000"))
    bytes_bar = tqdm_class(unit="B", unit_scale=True, total=0)
    file_bar = tqdm_class(total=4)  # not the bytes bar; must contribute nothing

    bytes_bar.total = 1_288_490_188  # ~1.2 GB, discovered as metadata arrives
    bytes_bar.refresh()
    for _ in range(200):  # ~0.5% per chunk
        bytes_bar.update(6_442_450)
    file_bar.update(1)

    # ~5% apart, so a 1.2 GB download is tens of lines, not thousands.
    assert 10 <= len(lines) <= 40
    assert lines[1] == "Downloading base checkpoint 012000 — 6% (74 MB / 1.2 GB)"
    assert "1.2 GB / 1.2 GB" in lines[-1]


def test_an_unknown_finetune_source_still_refuses_before_any_record(monkeypatch, tmp_path) -> None:
    """The cheap guards did NOT move behind the record. A request that can be
    refused without touching the Hub still fails fast, with no record, no job
    directory, and no download."""
    from makermodslab.jobs import JobRegistry, JobTarget

    def _no_downloads(**kwargs):
        raise AssertionError("a refused request must not download anything")

    _patch_hub_for_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)

    reg = JobRegistry(tmp_path / "root")
    _fake_local_runner(monkeypatch)

    with pytest.raises(ValueError, match="not found"):
        reg.start(_hub_finetune_request("no-such-run"), JobTarget(runner="local"))

    assert list(reg._records) == []
    assert list(reg._output_root.iterdir()) == []
    assert reg._prepare_threads == {}


# ---------------------------------------------------------------------------
# Feature-space guard (MT44). A matching architecture is not enough: this
# launch path is `--policy.type` + `--policy.pretrained_path`, so lerobot sizes
# the policy from the DATASET and loads the checkpoint's weights strict=False.
# A dof mismatch is loud-but-late for ACT and SILENT for SmolVLA/pi0 (they pad
# to 32 dims), and renamed or disjoint cameras are silent for every policy.
# Phase 1 refuses those; a changed camera COUNT that still overlaps, or a
# changed resolution, is legitimate and only warns.
#
# Every Hub read is patched out: the checkpoint side through
# jobs.hf_hub_download, the dataset side through jobs.read_dataset_features.
# ---------------------------------------------------------------------------


_DEFAULT_CAMERAS = ("front", "wrist")


# ---------------------------------------------------------------------------
# Fine-tune policy-type guard: --policy.type must match the source checkpoint's
# architecture, because lerobot loads pretrained weights non-strictly and would
# otherwise train a fresh policy that only looks like a fine-tune.
# ---------------------------------------------------------------------------


def _finetune_source(policy_type: str, runner: str = "imported"):
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    return JobRecord(
        id="src-1",
        name="Imported · lerobot/smolvla_base",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type=policy_type),
        output_dir="",
        started_at=0.0,
        runner=runner,
    )


def test_check_finetune_policy_type_rejects_mismatch() -> None:
    from makermodslab.jobs import _check_finetune_policy_type

    with pytest.raises(ValueError, match="smolvla") as exc:
        _check_finetune_policy_type(_finetune_source("smolvla"), "act")
    # Both sides named, so the toast tells the user what to switch.
    assert "'act'" in str(exc.value)


def test_check_finetune_policy_type_accepts_match() -> None:
    from makermodslab.jobs import _check_finetune_policy_type

    _check_finetune_policy_type(_finetune_source("smolvla"), "smolvla")


def test_check_finetune_policy_type_ignores_unknown_source_type() -> None:
    """register_imported records the "model" placeholder when a checkpoint's
    config.json can't be read — that says nothing about the weights, so it must
    not block a fine-tune."""
    from makermodslab.jobs import _check_finetune_policy_type

    _check_finetune_policy_type(_finetune_source("model"), "act")


def test_finetune_start_rejects_contradicting_policy_type(tmp_path) -> None:
    """End to end through JobRegistry.start: a smolvla base + an "act" request
    (the old silent default) fails with a 400-shaped ValueError instead of
    launching an ACT run from smolvla weights. No record is created."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    # A flat imported checkpoint dir whose config.json names the architecture.
    src = tmp_path / "smolvla_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))
    assert source.config.policy_type == "smolvla"

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",  # what the form sends when the type never propagates
        finetune_from_job_id=source.id,
    )
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))

    assert [r.id for r in reg.list(limit=10)] == [source.id]


def test_finetune_start_accepts_matching_policy_type(tmp_path) -> None:
    """The same fine-tune with the propagated type launches, and resolves the
    source checkpoint into --policy.pretrained_path."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    src = tmp_path / "smolvla_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="smolvla",
        finetune_from_job_id=source.id,
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(cfg, JobTarget(runner="local"))

    assert record.config.policy_type == "smolvla"
    assert record.config.policy_pretrained_path == str(src.resolve())


def test_read_pretrained_policy_type_reads_a_step_ref(monkeypatch, tmp_path) -> None:
    """The pre-download policy-type guard must look inside the step it names,
    not at the repo root — otherwise it would validate different weights than
    the ones the run trains from."""
    from makermodslab.jobs import read_pretrained_policy_type

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"type": "smolvla"}))
    seen: dict = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    # This one reads through jobs' module-level binding (unlike
    # _read_checkpoint_config, which re-imports), so patch it there.
    monkeypatch.setattr("makermodslab.jobs.hf_hub_download", fake_download)

    assert read_pretrained_policy_type("user/repo@checkpoints/000500") == "smolvla"
    assert seen["filename"] == "checkpoints/000500/pretrained_model/config.json"


def test_contradicting_policy_type_still_refuses_before_any_record(monkeypatch, tmp_path) -> None:
    """The cheap guards did NOT move behind the record. A request that can be
    refused from the checkpoint's config.json alone still fails fast, with no
    record, no job directory, and no download."""
    from makermodslab.jobs import JobRegistry, JobTarget

    def _no_downloads(**kwargs):
        raise AssertionError("a refused request must not download anything")

    # The base checkpoint is smolvla; the request below asks for act.
    _patch_hub_for_finetune(monkeypatch, tmp_path, policy_type="smolvla")
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    _fake_local_runner(monkeypatch)

    with pytest.raises(ValueError, match="smolvla"):
        reg.start(_hub_finetune_request(source.id), JobTarget(runner="local"))

    assert list(reg._records) == [source.id]
    assert list(reg._output_root.iterdir()) == []
    assert reg._prepare_threads == {}


# ---------------------------------------------------------------------------
# Checkpoint-level policy-type guard. _check_finetune_policy_type compares the
# source JobRecord's *recorded* type, so it is blind to a request that supplies
# policy_pretrained_path directly (a public TrainingRequest field, no
# finetune_from_job_id needed) and it opts out entirely when the record carries
# register_imported's "model" placeholder. These cover the checkpoint's own
# config.json being consulted instead.
# ---------------------------------------------------------------------------


def _flat_ckpt(tmp_path: Path, name: str, policy_type: str) -> Path:
    """A flat pretrained_model-shaped dir whose config.json names an architecture."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(_json.dumps({"type": policy_type}))
    return d


def test_read_pretrained_policy_type_reads_local_config(tmp_path) -> None:
    from makermodslab.jobs import read_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    assert read_pretrained_policy_type(str(ckpt)) == "smolvla"


def test_read_pretrained_policy_type_none_when_unreadable(tmp_path) -> None:
    """Missing config.json, blank type, and a bad Hub ref all yield None —
    "not established", which callers must not treat as a clean result."""
    from unittest.mock import patch

    from makermodslab.jobs import read_pretrained_policy_type

    bare = tmp_path / "no_config"
    bare.mkdir()
    assert read_pretrained_policy_type(str(bare)) is None

    blank = tmp_path / "blank"
    blank.mkdir()
    (blank / "config.json").write_text(_json.dumps({"type": "   "}))
    assert read_pretrained_policy_type(str(blank)) is None

    # Not a directory ⇒ treated as a Hub repo id; a failed download is silent.
    with patch("makermodslab.jobs.hf_hub_download", side_effect=OSError("offline")):
        assert read_pretrained_policy_type("someone/nope") is None


def test_check_pretrained_policy_type_rejects_mismatch(tmp_path) -> None:
    from makermodslab.jobs import _check_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    with pytest.raises(ValueError, match="smolvla") as exc:
        _check_pretrained_policy_type(str(ckpt), "act")
    assert "'act'" in str(exc.value)


def test_check_pretrained_policy_type_silent_when_matching_or_unknown(tmp_path) -> None:
    """A match passes, and so does an unverifiable checkpoint — an unreadable
    source must not block a launch, only an actual contradiction may."""
    from makermodslab.jobs import _check_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "act_ckpt", "act")
    _check_pretrained_policy_type(str(ckpt), "act")

    bare = tmp_path / "unknown"
    bare.mkdir()
    _check_pretrained_policy_type(str(bare), "act")


def test_start_rejects_direct_pretrained_path_mismatch(tmp_path) -> None:
    """The hole _check_finetune_policy_type leaves open: policy_pretrained_path
    set directly, with no finetune_from_job_id, so the record-based guard never
    runs. The checkpoint's own config.json must still stop it, and no record may
    be created."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        policy_pretrained_path=str(ckpt),
    )
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))

    assert reg.list(limit=10) == []


def test_start_rejects_finetune_when_record_type_is_placeholder(tmp_path) -> None:
    """register_imported stores the "model" placeholder when it can't read a
    checkpoint's config.json, and _check_finetune_policy_type deliberately skips
    that case. If the config becomes readable by launch time, the checkpoint
    check must still catch the mismatch."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    src = tmp_path / "mystery_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))
    # Simulate the import having failed to read the architecture.
    source.config.policy_type = "model"

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        finetune_from_job_id=source.id,
    )
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))


def test_start_allows_resume_without_checkpoint_type_check(tmp_path) -> None:
    """Resume passes --config_path and never emits --policy.pretrained_path, so
    the pair can't contradict and the guard must not fire on it (a resumed
    smolvla run whose request still carries the "act" default would otherwise be
    refused)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        policy_pretrained_path=str(ckpt),
        resume=True,
        config_path=str(tmp_path / "train_config.json"),
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 99
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(cfg, JobTarget(runner="local"))
    assert record.state == "running"


# ---------------------------------------------------------------------------
# Feature-space guard (MT44). Matching architectures are not enough: this
# launch path is `--policy.type` + `--policy.pretrained_path`, so lerobot sizes
# the policy from the DATASET and loads the checkpoint's weights strict=False.
# A dof mismatch is loud-but-late for ACT and SILENT for SmolVLA/pi0 (they pad
# to 32 dims), and renamed cameras are silent for every policy. Phase 1 refuses
# those two; a changed camera COUNT or resolution is legitimate and only warns.
#
# Every Hub read is patched out: the checkpoint side through
# jobs.hf_hub_download, the dataset side through jobs.read_dataset_features.
# ---------------------------------------------------------------------------


_DEFAULT_CAMERAS = ("front", "wrist")


def _feature_ckpt(
    tmp_path: Path,
    name: str,
    *,
    policy_type: str = "act",
    state_dim: int = 6,
    action_dim: int = 6,
    cameras: tuple[str, ...] = _DEFAULT_CAMERAS,
    height: int = 480,
    width: int = 640,
) -> Path:
    """A pretrained_model dir whose config.json carries a real feature space.
    Image shapes are CHW, as a policy checkpoint writes them."""
    d = tmp_path / name
    d.mkdir()
    inputs: dict = {"observation.state": {"type": "STATE", "shape": [state_dim]}}
    for cam in cameras:
        inputs[f"observation.images.{cam}"] = {"type": "VISUAL", "shape": [3, height, width]}
    (d / "config.json").write_text(
        _json.dumps(
            {
                "type": policy_type,
                "input_features": inputs,
                "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
            }
        )
    )
    return d


def _dataset_features(
    *,
    state_dim: int = 6,
    action_dim: int = 6,
    cameras: tuple[str, ...] = _DEFAULT_CAMERAS,
    height: int = 480,
    width: int = 640,
) -> dict:
    """A dataset meta/info.json `features` map. Image shapes are HWC with named
    axes — the opposite convention from the checkpoint above, which is exactly
    what the guard has to normalise."""
    features: dict = {
        "action": {"dtype": "float32", "shape": [action_dim]},
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        }
    return features


def _patch_dataset_features(features):
    """Pin what the guard sees for the selected dataset (no Hub, no cache)."""
    from unittest.mock import patch

    return patch("makermodslab.jobs.read_dataset_features", lambda repo_id: features)


def test_check_feature_space_rejects_single_arm_checkpoint_on_bimanual_dataset(tmp_path) -> None:
    """The SILENT case MT44 is really about: SmolVLA pads dofs to 32, so a
    6-dof checkpoint loads cleanly into a 12-dof run and trains garbage that is
    recorded as a fine-tune. Both widths and both sources must be named."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "single_arm", policy_type="smolvla")
    with (
        _patch_dataset_features(_dataset_features(state_dim=12, action_dim=12)),
        pytest.raises(ValueError) as exc,
    ):
        _check_pretrained_feature_space(str(ckpt), "user/bimanual_ds")
    message = str(exc.value)
    assert "6-dim robot state" in message
    assert "12-dim robot state" in message
    assert "user/bimanual_ds" in message
    assert str(ckpt) in message


def test_check_feature_space_rejects_bimanual_checkpoint_on_single_arm_dataset(tmp_path) -> None:
    """The same refusal in the other direction — neither side is privileged."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "bimanual", state_dim=12, action_dim=12)
    with (
        _patch_dataset_features(_dataset_features(state_dim=6, action_dim=6)),
        pytest.raises(ValueError) as exc,
    ):
        _check_pretrained_feature_space(str(ckpt), "user/single_ds")
    message = str(exc.value)
    assert "12-dim robot state" in message
    assert "6-dim robot state" in message


def test_check_feature_space_rejects_action_width_mismatch(tmp_path) -> None:
    """State can agree while the action head doesn't (a checkpoint whose output
    space was changed); the action side is checked on its own terms."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "wide_action", action_dim=12)
    with (
        _patch_dataset_features(_dataset_features()),
        pytest.raises(ValueError, match="action"),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/ds")


def test_check_feature_space_rejects_renamed_cameras(tmp_path) -> None:
    """Equal camera COUNT, different keys — the bimanual `left_` prefix case.
    This is the refusal the whole rule exists for, and the generic-base
    exemption below must never swallow it: `front`/`wrist` are real mounts."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "named_ckpt", cameras=("front", "wrist"))
    with (
        _patch_dataset_features(_dataset_features(cameras=("left_front", "left_wrist"))),
        pytest.raises(ValueError) as exc,
    ):
        _check_pretrained_feature_space(str(ckpt), "user/renamed_ds")
    message = str(exc.value)
    assert "front, wrist" in message
    assert "left_front, left_wrist" in message
    # The two ways out the message must offer.
    assert "base model" in message
    assert "from scratch" in message


def test_check_feature_space_exempts_a_generic_base_from_the_rename_rule(tmp_path, caplog) -> None:
    """lerobot/smolvla_base ships camera1/camera2/camera3 — placeholders, not a
    rig. Binding those to a named 3-camera dataset is THE canonical SmolVLA
    fine-tune, so it warns instead of refusing."""
    import logging

    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(
        tmp_path, "smolvla_base", policy_type="smolvla", cameras=("camera1", "camera2", "camera3")
    )
    with (
        caplog.at_level(logging.WARNING, logger="makermodslab.jobs"),
        _patch_dataset_features(_dataset_features(cameras=("front", "wrist", "top"))),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/named_rig_ds")
    assert "placeholder camera names" in caplog.text
    assert "front, top, wrist" in caplog.text


def test_check_feature_space_rejects_disjoint_cameras_at_different_counts(tmp_path) -> None:
    """Zero overlap is the rename mistake at an unequal count, and it slipped
    through live: a 1-camera `left` dataset against a wrist/front checkpoint
    fell past the rename rule (counts differ) into the benign count-change
    branch. None of the checkpoint's cameras survive, so it is refused."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "two_cam_ckpt", cameras=("front", "wrist"))
    with (
        _patch_dataset_features(_dataset_features(cameras=("left",))),
        pytest.raises(ValueError) as exc,
    ):
        _check_pretrained_feature_space(str(ckpt), "user/left_only_ds")
    message = str(exc.value)
    assert "no camera in common" in message
    assert "front, wrist" in message
    assert "left" in message
    # The two ways out the message must offer.
    assert "base model" in message
    assert "from scratch" in message


def test_check_feature_space_exempts_a_generic_base_from_the_disjoint_rule(tmp_path, caplog) -> None:
    """The canonical smolvla_base fine-tune is disjoint AND unequal in count
    (camera1/2/3 vs a real 2-camera rig), so the generic-base exemption has to
    cover the new rule too or it would refuse the commonest fine-tune there is."""
    import logging

    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(
        tmp_path, "generic_base", policy_type="smolvla", cameras=("camera1", "camera2", "camera3")
    )
    with (
        caplog.at_level(logging.WARNING, logger="makermodslab.jobs"),
        _patch_dataset_features(_dataset_features(cameras=("front", "wrist"))),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/two_cam_ds")
    assert "placeholder camera names" in caplog.text


def test_check_feature_space_exemption_is_all_or_nothing(tmp_path) -> None:
    """A checkpoint mixing a placeholder with a real mount (camera1 + wrist) is
    not a generic base — it came off some rig — so the rename refusal stands."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "half_generic", cameras=("camera1", "wrist"))
    with (
        _patch_dataset_features(_dataset_features(cameras=("front", "top"))),
        pytest.raises(ValueError, match="different names"),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/other_rig_ds")


def test_check_feature_space_accepts_matching_features(tmp_path) -> None:
    """The ordinary fine-tune: same robot, same cameras, same resolution."""
    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "matched")
    with _patch_dataset_features(_dataset_features()):
        _check_pretrained_feature_space(str(ckpt), "user/ds")


def test_check_feature_space_allows_camera_count_change_with_a_warning(tmp_path, caplog) -> None:
    """A dropped camera is a real sensor-suite change but a legitimate one
    (ACT's backbone is shared), so phase 1 records it instead of refusing. The
    warn-and-confirm UI is phase 2."""
    import logging

    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "two_cams", cameras=("front", "wrist"))
    with (
        caplog.at_level(logging.WARNING, logger="makermodslab.jobs"),
        _patch_dataset_features(_dataset_features(cameras=("front",))),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/one_cam_ds")
    assert "wrist" in caplog.text

    # ...and the same for a camera the checkpoint never saw.
    caplog.clear()
    with (
        caplog.at_level(logging.WARNING, logger="makermodslab.jobs"),
        _patch_dataset_features(_dataset_features(cameras=("front", "wrist", "top"))),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/three_cam_ds")
    assert "top" in caplog.text


def test_check_feature_space_allows_resolution_change_with_a_warning(tmp_path, caplog) -> None:
    """Resolution only moves ACT's token count and VRAM — allowed, but noted.
    Also the one case that proves CHW (checkpoint) and HWC (dataset) shapes are
    normalised before comparison rather than compared axis-for-axis."""
    import logging

    from makermodslab.jobs import _check_pretrained_feature_space

    ckpt = _feature_ckpt(tmp_path, "hi_res", height=480, width=640)
    with (
        caplog.at_level(logging.WARNING, logger="makermodslab.jobs"),
        _patch_dataset_features(_dataset_features(height=240, width=320)),
    ):
        _check_pretrained_feature_space(str(ckpt), "user/small_ds")
    assert "480x640 -> 240x320" in caplog.text


def test_check_feature_space_silent_when_either_side_is_unreadable(tmp_path) -> None:
    """The discipline the neighbouring guards keep: an unverifiable pair must
    not block a launch. None means "not established", never "fine"."""
    from makermodslab.jobs import _check_pretrained_feature_space

    bare = tmp_path / "no_config"
    bare.mkdir()
    with _patch_dataset_features(_dataset_features(state_dim=12, action_dim=12)):
        _check_pretrained_feature_space(str(bare), "user/bimanual_ds")

    # A config.json with no feature maps at all says nothing either.
    typed_only = _flat_ckpt(tmp_path, "type_only", "act")
    with _patch_dataset_features(_dataset_features(state_dim=12, action_dim=12)):
        _check_pretrained_feature_space(str(typed_only), "user/bimanual_ds")

    # Unreadable dataset meta (offline, or a repo we can't see).
    ckpt = _feature_ckpt(tmp_path, "readable", state_dim=6)
    with _patch_dataset_features(None):
        _check_pretrained_feature_space(str(ckpt), "user/unknown_ds")


def test_read_pretrained_feature_space_reads_a_step_ref(monkeypatch, tmp_path) -> None:
    """Like the policy-type read, this must look inside the step it names —
    and it must come out of the SAME config.json fetch, not a second one."""
    from makermodslab.jobs import read_pretrained_feature_space

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        _json.dumps(
            {
                "type": "smolvla",
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            }
        )
    )
    seen: dict = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    monkeypatch.setattr("makermodslab.jobs.hf_hub_download", fake_download)

    inputs, outputs = read_pretrained_feature_space("user/repo@checkpoints/000500")
    assert inputs["observation.state"]["shape"] == [6]
    assert outputs["action"]["shape"] == [6]
    assert seen["filename"] == "checkpoints/000500/pretrained_model/config.json"


def test_start_rejects_feature_space_mismatch_and_leaves_no_record(tmp_path) -> None:
    """End to end: the refusal is synchronous, is a 400-shaped ValueError, and
    happens before anything is materialized — no record, no output dir, no
    prepare thread."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    ckpt = _feature_ckpt(tmp_path, "single_arm", policy_type="smolvla")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/bimanual_ds",
        policy_type="smolvla",
        policy_pretrained_path=str(ckpt),
    )
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        _patch_dataset_features(_dataset_features(state_dim=12, action_dim=12)),
        pytest.raises(ValueError, match="12-dim robot state"),
    ):
        reg.start(cfg, JobTarget(runner="local"))

    assert reg.list(limit=10) == []
    assert list(reg._output_root.iterdir()) == []
    assert reg._prepare_threads == {}


def test_start_allows_matching_feature_space(tmp_path) -> None:
    """The guard must not become a new way for an ordinary fine-tune to fail:
    a matching checkpoint/dataset pair still reaches the normal start path."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    ckpt = _feature_ckpt(tmp_path, "matched")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        policy_pretrained_path=str(ckpt),
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner),
        _patch_dataset_features(_dataset_features()),
    ):
        record = reg.start(cfg, JobTarget(runner="local"))
    assert record.state == "running"


# --- Deliberate stop vs genuine failure -------------------------------------
#
# Regression cover for the defect where every press of Stop landed in run
# history as `failed` + "Subprocess exited with code 1", indistinguishable
# from a crash: JobRegistry.stop() recorded no intent and the watchdog had
# only the exit code to go on. The state machine already had `interrupted`,
# reachable only by startup reconciliation of a stranded record.


class _FakeRunner:
    """Minimal JobRunner. Deliberately does NOT expose the optional hooks —
    subclasses add them, mirroring runners that can and can't answer."""

    def __init__(self, *, code=None, on_stop_code=None, stage=None, on_stop_stage=None):
        self._code = code  # None + no stage => still running
        self._on_stop_code = on_stop_code
        self._stage = stage
        self._on_stop_stage = on_stop_stage
        self.stopped = False

    def start(self, job_id, config, output_dir) -> None:
        # No subprocess: liveness is driven by the fields above so the
        # watchdog's exit-detection can be stepped deterministically.
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self._on_stop_code is not None:
            self._code = self._on_stop_code
        # Idempotent like HfCloudJobRunner._set_terminal: a stage the platform
        # already reported survives our cancel.
        if self._on_stop_stage is not None and self._stage is None:
            self._stage = self._on_stop_stage

    def is_running(self) -> bool:
        return self._code is None and self._stage is None

    def returncode(self):
        if self._stage is not None:
            return 0 if self._stage == "COMPLETED" else 1
        return self._code

    def stream_log_lines(self):
        return []

    def wandb_run_url(self):
        return None

    def pid(self):
        return 4242


class _FakeSignallingRunner(_FakeRunner):
    """A local-shaped runner: reports whether it actually signalled."""

    def __init__(self, *, signals=True, **kw):
        super().__init__(**kw)
        self._signals = signals

    def stop(self) -> None:
        if self._signals:
            super().stop()
        else:
            # Process was already gone; stop() short-circuits and claims
            # nothing, exactly like LocalJobRunner's poll() guard.
            self.stopped = True

    def stop_signalled(self) -> bool:
        return self._signals and self.stopped


class _FakeStagedRunner(_FakeRunner):
    """A cloud-shaped runner: reports a platform terminal stage + message."""

    def __init__(self, *, message=None, **kw):
        super().__init__(**kw)
        self._message = message

    def terminal_stage(self):
        return self._stage

    def terminal_message(self):
        return self._message


def _start_with(reg, runner, **cfg_kw):
    """Start a job whose runner is `runner`, via the real JobRegistry.start."""
    from unittest.mock import patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    cfg = TrainingRequest(dataset_repo_id="user/ds", **cfg_kw)
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: runner):
        return reg.start(cfg, JobTarget(runner="local"))


def _stop_and_finalise(reg, job_id):
    """Stop, then force a watchdog tick so the assertion doesn't race the
    1Hz background thread. _tick is a no-op if that thread got there first."""
    reg.stop(job_id)
    reg._tick()
    return reg.get(job_id)


# -- the pure classifier ----------------------------------------------------


@pytest.mark.parametrize(
    ("rc", "stop_requested", "stage", "expected"),
    [
        # Local: a clean exit is `done` no matter what else is true.
        (0, False, None, "done"),
        (0, True, None, "done"),
        # Local: nonzero without a stop is a real failure (unchanged).
        (1, False, None, "failed"),
        (-15, False, None, "failed"),
        # Local: nonzero after a stop we signalled is deliberate.
        (1, True, None, "interrupted"),
        (-15, True, None, "interrupted"),
        # No code at all: no evidence, stays a failure (unchanged).
        (None, False, None, "failed"),
        (None, True, None, "failed"),
        # Cloud: the platform stage wins over the collapsed exit code.
        (0, False, "COMPLETED", "done"),
        (0, True, "COMPLETED", "done"),
        (1, True, "CANCELED", "interrupted"),
        (1, False, "CANCELED", "failed"),
        (1, True, "ERROR", "failed"),
        (1, False, "ERROR", "failed"),
        (1, True, "DELETED", "failed"),
        # Stage matching is case-insensitive (HF returns an enum we str()).
        (1, True, "canceled", "interrupted"),
    ],
)
def test_classify_terminal_state_table(rc, stop_requested, stage, expected) -> None:
    from makermodslab.jobs import classify_terminal_state

    assert (
        classify_terminal_state(returncode=rc, stop_requested=stop_requested, terminal_stage=stage)
        == expected
    )


# -- registry: local runner -------------------------------------------------


def test_stop_records_intent_before_signalling(tmp_path) -> None:
    """The intent must be on the registry before the signal leaves, or the
    watchdog can finalise a stop it never heard about."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    seen: list[bool] = []

    class _Probe(_FakeSignallingRunner):
        def stop(self):
            # Observed from inside stop(), i.e. before any signal lands.
            seen.append(record.id in reg._stop_requested)
            super().stop()

    runner = _Probe(on_stop_code=-15)
    record = _start_with(reg, runner)
    reg.stop(record.id)

    assert seen == [True]


def test_local_stop_is_interrupted_not_failed(tmp_path) -> None:
    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "interrupted"
    assert final.error_message == STOPPED_BY_REQUEST_MESSAGE
    assert "exited with code" not in (final.error_message or "")
    # The real code is still recorded for anyone debugging.
    assert final.exit_code == -15
    assert final.ended_at is not None


def test_local_stop_of_trainer_that_catches_sigterm_is_still_interrupted(tmp_path) -> None:
    """A trainer with its own SIGTERM handler exits 1, not -15. Narrowing
    `interrupted` to signal-shaped codes would leave the bug unfixed here."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=1))

    assert _stop_and_finalise(reg, record.id).state == "interrupted"


def test_crash_without_a_stop_stays_failed(tmp_path) -> None:
    """The unchanged path: nothing asked this to stop, so it failed."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner()
    record = _start_with(reg, runner)

    runner._code = 1  # crashed on its own
    reg._tick()

    final = reg.get(record.id)
    assert final.state == "failed"
    assert final.error_message == "Subprocess exited with code 1"


def test_clean_finish_racing_a_stop_stays_done(tmp_path) -> None:
    """rc == 0 means the trainer ran its own shutdown to completion; a stop
    that arrived too late must not relabel it."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner(on_stop_code=0)
    record = _start_with(reg, runner)

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "done"
    assert final.error_message is None


def test_crash_before_the_signal_landed_is_not_laundered(tmp_path) -> None:
    """The process died on its own between the intent and the signal, so
    LocalJobRunner.stop() short-circuits and reports it signalled nothing.
    The nonzero code is the process's own: still a failure."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner(signals=False)
    record = _start_with(reg, runner)

    runner._code = 1  # crashed in the window
    final = _stop_and_finalise(reg, record.id)

    assert runner.stopped is True  # we did ask
    assert final.state == "failed"
    assert final.error_message == "Subprocess exited with code 1"


def test_runner_without_the_hook_still_gets_interrupted(tmp_path) -> None:
    """A runner that can't say whether it signalled abstains rather than
    vetoing — recorded intent alone is enough."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeRunner(on_stop_code=1)
    assert not hasattr(runner, "stop_signalled")
    record = _start_with(reg, runner)

    assert _stop_and_finalise(reg, record.id).state == "interrupted"


def test_stop_intent_is_dropped_after_finalisation(tmp_path) -> None:
    """No stale intent may linger to mislabel anything later."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))
    _stop_and_finalise(reg, record.id)

    assert record.id not in reg._stop_requested


def test_interrupted_state_survives_a_restart(tmp_path) -> None:
    """The classification is persisted, not just in-memory — the user's
    history has to still read `interrupted` on the next launch."""
    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    root = tmp_path / "root"
    reg = JobRegistry(root)
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))
    _stop_and_finalise(reg, record.id)
    reg.shutdown()

    reloaded = JobRegistry(root).get(record.id)
    assert reloaded.state == "interrupted"
    assert reloaded.error_message == STOPPED_BY_REQUEST_MESSAGE


def test_stop_rejects_an_already_finished_job_without_recording_intent(tmp_path) -> None:
    from makermodslab.jobs import JobNotRunningError, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner()
    record = _start_with(reg, runner)

    runner._code = 0
    reg._tick()
    assert reg.get(record.id).state == "done"

    with pytest.raises(JobNotRunningError):
        reg.stop(record.id)
    assert record.id not in reg._stop_requested


# -- registry: cloud-shaped runner (classified on terminal_stage) -----------


def test_cloud_cancel_is_interrupted(tmp_path) -> None:
    """The reported case: a stopped HF Jobs run. returncode() collapses every
    non-COMPLETED stage to 1, so before this it read `failed` + "Subprocess
    exited with code 1" and looked like a broken model."""
    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeStagedRunner(on_stop_stage="CANCELED"))

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "interrupted"
    assert final.error_message == STOPPED_BY_REQUEST_MESSAGE


def test_cloud_job_that_completed_before_the_cancel_stays_done(tmp_path) -> None:
    """The poller saw COMPLETED first; _set_terminal is idempotent so our
    cancel doesn't overwrite it, and the run keeps its success."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(on_stop_stage="CANCELED")
    record = _start_with(reg, runner)

    runner._stage = "COMPLETED"  # observed by the status poller
    final = _stop_and_finalise(reg, record.id)

    assert final.state == "done"
    assert final.error_message is None


def test_cloud_job_that_errored_before_the_cancel_stays_failed(tmp_path) -> None:
    """A real crash that merely coincided with the stop must not be laundered
    into `interrupted` — that would hide a genuine failure."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(on_stop_stage="CANCELED", message="boom")
    record = _start_with(reg, runner)

    runner._stage = "ERROR"
    final = _stop_and_finalise(reg, record.id)

    assert final.state == "failed"
    assert final.error_message == "boom"


def test_cloud_timeout_stays_failed_and_keeps_its_platform_message(tmp_path) -> None:
    """HF Jobs' 'Job timeout' arrives as an ERROR stage with a message. It is
    a failure, not a user stop, and the message must still reach the UI."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(message="Job timeout")
    record = _start_with(reg, runner)

    runner._stage = "ERROR"
    reg._tick()

    final = reg.get(record.id)
    assert final.state == "failed"
    assert final.error_message == "Job timeout"


def test_cloud_cancel_from_outside_makermodslab_stays_failed(tmp_path) -> None:
    """A CANCELED we never asked for (HF web UI, platform-side kill). HF's
    stage doesn't say who asked, so this is left alone rather than guessed
    into `interrupted`. Documented limitation, asserted so it's a choice."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner()
    record = _start_with(reg, runner)

    runner._stage = "CANCELED"
    reg._tick()

    assert reg.get(record.id).state == "failed"


# -- TailingJobRunner: no Popen to reap, so the code is synthesised ---------


def _tailing_runner(pid, monkeypatch, *, alive=True, status_path=None):
    """A TailingJobRunner over a fake pid; os.kill (liveness probes) and
    os.killpg (the stop signal) are both stubbed so no real process or group
    is signalled. `status_path` defaults to a path that cannot exist, i.e. the
    "wrapper left no exit status" case."""
    from makermodslab import jobs as jobs_mod

    state = {"alive": alive}

    def fake_kill(target_pid, sig):
        assert target_pid == pid
        if not state["alive"]:
            raise ProcessLookupError(pid)
        if sig != 0:
            state["alive"] = False  # SIGTERM landed

    monkeypatch.setattr(jobs_mod.os, "kill", fake_kill)
    monkeypatch.setattr(jobs_mod.os, "killpg", fake_kill)
    runner = jobs_mod.TailingJobRunner(
        jobs_mod.TrainingMetrics(),
        Path("/nonexistent"),
        pid,
        status_path if status_path is not None else Path("/nonexistent/exit_status"),
    )
    return runner, state


def test_tailing_runner_reports_sigterm_after_a_delivered_stop(monkeypatch) -> None:
    """With no exit status on disk — the normal shape of a stop, since the
    group TERM kills the wrapper before it can write one — a bare "the pid is
    gone" would file a deliberate stop as `done`. Once we know we signalled a
    live pid, naming the signal is the more honest synthetic answer, and it is
    what lets classify_terminal_state reach `interrupted`."""
    import signal as signal_mod

    runner, _ = _tailing_runner(31337, monkeypatch)
    assert runner.returncode() is None  # still alive

    runner.stop()
    assert runner.stop_signalled() is True
    assert runner.returncode() == -signal_mod.SIGTERM


def test_tailing_runner_prefers_the_real_exit_code_over_the_synthesised_signal(tmp_path, monkeypatch) -> None:
    """The synthesised SIGTERM above is a fallback, never a preference. When
    the wrapper did manage to write its status file (the trainer installed its
    own handler, shut down cleanly and exited 0 before the group TERM reached
    the wrapper), that REAL code wins — otherwise a run that genuinely
    finished would be filed as `interrupted` on the strength of a signal that
    changed nothing."""
    status_path = tmp_path / "exit_status"
    status_path.write_text("0")

    runner, _ = _tailing_runner(31339, monkeypatch, status_path=status_path)
    runner.stop()

    assert runner.stop_signalled() is True
    assert runner.returncode() == 0


def test_tailing_runner_reports_unconfirmed_when_pid_was_already_gone(monkeypatch) -> None:
    """Nothing was signalled, so the pid's absence isn't ours to claim — and
    with no exit status on disk either, nothing else knows how it ended.

    This used to synthesise an optimistic 0 (finalising as `done`), which MT10
    removed: an unconfirmed disappearance is reported as None here and
    finalised as `interrupted` by JobRegistry._tick(), never as a success we
    can't back up."""
    runner, _ = _tailing_runner(31338, monkeypatch, alive=False)

    runner.stop()
    assert runner.stop_signalled() is False
    assert runner.returncode() is None


def test_two_imports_of_one_task_and_policy_are_still_disambiguated(monkeypatch, tmp_path) -> None:
    """Same task AND same policy: nothing on either card separates them, so the
    timestamp the title dropped comes back on both."""
    early = "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30"
    late = "makermods/smolvla_makermods_orange_box_2026-08-05_09-00-00"
    reg = _typed_hub_reg(monkeypatch, tmp_path, {early: "smolvla", late: "smolvla"})
    a = reg.register_imported(early)
    b = reg.register_imported(late)

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03)"
    assert names[b.id] == "orange_box (2026-08-05)"


def _fake_resume_snapshot(tmp_path, seen: dict, *, complete: bool = True):
    """A snapshot_download stand-in that lays down a real checkpoint tree.

    Mirrors what the Hub returns for `allow_patterns=['checkpoints/<step>/*']`:
    a snapshot root holding the whole step directory. `complete=False` is the
    interrupted-upload shape — weights but no optimizer state — which a resume
    must refuse rather than hand to the trainer."""

    def _download(**kwargs):
        seen.update(kwargs)
        root = tmp_path / "snapshot"
        _make_checkpoint(root, 100, with_optimizer=complete)
        # The Hub's zero-padded dir name, which _make_checkpoint doesn't use.
        (root / "checkpoints" / "100").rename(root / "checkpoints" / "000100")
        return str(root)

    return _download


class _FakeUploadApi:
    """HfApi stand-in for the upload path: records the calls, moves no bytes.

    `list_repo_files` answers from whatever has been "uploaded" so far, so the
    post-upload verification in _upload_resume_then_start exercises the real
    completeness rule instead of a stub that always says yes. A weights-only
    push (the fine-tune staging path, whose path_in_repo ends in
    /pretrained_model) publishes only that half, so its verification meets the
    same tree the real one would — not a full checkpoint it never uploaded."""

    def __init__(self, files: list[str] | None = None) -> None:
        self._files = list(files or [])
        self.created: list[dict] = []
        self.uploaded: list[dict] = []
        self.upload_error: Exception | None = None

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        if self.upload_error is not None:
            raise self.upload_error
        self.uploaded.append(kwargs)
        parts = kwargs["path_in_repo"].split("/")
        if parts[-1] == "pretrained_model":
            self._files.extend(_hub_pretrained_files(parts[-2]))
        else:
            self._files.extend(_hub_checkpoint_files(parts[-1]))

    def list_repo_files(self, repo_id, repo_type):
        return self._files


def test_cloud_parent_resumed_locally_seeds_progress_from_the_inherited_step(tmp_path, monkeypatch) -> None:
    """The record's metrics start at the checkpoint's step, not at 0 — and they
    do so from the moment it is created, i.e. before the (minutes-long) download
    finishes. A 0 there is what wipes the seeded loss chart (MT16's local twin)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100")),
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_resume_snapshot(tmp_path, {}))
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()):
        # `resume_from_step` left unset: "the latest checkpoint", which the
        # resolver has to pin to a real step for the seeding to work at all.
        record = reg.start(_resume_request(), JobTarget(runner="local"))
        assert record.metrics.current_step == 100
        assert record.metrics.total_steps == record.config.steps
        assert record.config.resume_from_step == 100
        _join_prepare(reg, record.id)


def test_cloud_parent_resumed_locally_refuses_an_incomplete_hub_checkpoint(tmp_path, monkeypatch) -> None:
    """Refused synchronously, from the repo's file listing, before a record or a
    single byte exists — the completeness gate is the same one cloud→cloud uses."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100", with_optimizer=False)),
    )

    def _no_downloads(**kwargs):
        raise AssertionError("an incomplete checkpoint must be refused before downloading")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="incomplete"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    assert list(reg._records) == ["src"]
    _assert_nothing_was_created(reg)


def test_cloud_parent_resumed_locally_fails_the_job_on_an_incomplete_download(tmp_path, monkeypatch) -> None:
    """MT4's failure mode, closed: if the bytes that land are short of a
    resumable checkpoint, the job fails with a message naming it and NO trainer
    is spawned — rather than lerobot dying on a missing optimizer file minutes
    into startup."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    # The listing says complete; the bytes that arrive are not (the uploader
    # race). Only the on-disk check can catch that.
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100")),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_resume_snapshot(tmp_path, {}, complete=False)
    )
    fake_runner = MagicMock()
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))
        _join_prepare(reg, record.id)

    failed = reg._records[record.id]
    assert failed.state == "failed"
    assert "optimizer_state.safetensors" in failed.error_message
    assert not fake_runner.start.called


def test_local_parent_resumed_on_the_cloud_fails_when_the_upload_cannot_be_confirmed(
    tmp_path, monkeypatch, cloud_preflight
) -> None:
    """An upload that reports success but leaves the repo short of a resumable
    checkpoint is the same failure as one that raised — verified from the Hub's
    own listing, before anything is submitted."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")

    class _SilentlyPartialApi(_FakeUploadApi):
        def upload_folder(self, **kwargs):
            self.uploaded.append(kwargs)
            self._files.extend(_hub_checkpoint_files("100", with_optimizer=False))

    api = _SilentlyPartialApi()
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: api)
    monkeypatch.setattr("makermodslab.jobs.cached_whoami", lambda: {"name": "alice"})
    fake_runner = MagicMock()
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_local_to_cloud_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))
        _join_prepare(reg, record.id)

    failed = reg._records[record.id]
    assert failed.state == "failed"
    assert "optimizer_state.safetensors" in failed.error_message
    assert not fake_runner.start.called
    assert reg._records["src"].checkpoints_hub_steps == []


def test_start_still_resumes_a_cloud_run_on_the_cloud(tmp_path, monkeypatch) -> None:
    """The other half of the gate: a same-runner cloud resume is untouched and
    still resolves the parent's Hub checkpoint. (local→local is covered by
    test_start_still_resumes_a_run_that_stopped_short above.)"""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100")),
    )
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr("makermodslab.datasets.hub_copy_has_data", lambda repo_id: True)
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert record.config.resume is True
    assert record.config.resume_from_hub_repo == "user/some-model"
    assert record.config.resume_from_hub_step == "000100"


def _write_running_job_json(job_dir: Path, output_dir: Path) -> None:
    """Lay out an on-disk 'running' job.json the way a crash mid-training
    would leave it: state still 'running', a process_pid that (the caller
    arranges to) no longer exists by the time the registry boots and reads
    it back — the exact shape _load_from_disk() sees on a full server
    restart, as opposed to _tick()'s in-memory finalisation of a job that
    died while the server stayed up."""
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "job-1",
        "name": "run",
        "state": "running",
        "config": {"dataset_repo_id": "user/ds"},
        "output_dir": str(output_dir),
        "started_at": 0.0,
        "runner": "local",
        "process_pid": 999999999,  # long dead, see test_pid_alive_returns_false_for_unlikely_pid
    }
    (job_dir / "job.json").write_text(_json.dumps(meta))


def test_boot_reattach_reads_exit_status_when_pid_already_dead_and_failed(tmp_path) -> None:
    """IsaacSinn's PR #34 follow-up: a run that crashed while the server was
    down (server killed/crashed, trainer keeps going per the whole point of
    the wrapper, then dies and writes its real nonzero exit code to
    <output_dir>/exit_status) must NOT be silently reported as 'interrupted'
    once the server comes back — that status file is exactly the evidence
    TailingJobRunner.returncode() already trusts when the server stays up
    (see test_tick_marks_interrupted_when_runner_cannot_confirm_exit and
    friends). _load_from_disk() must consult the same file before giving up
    and asserting 'interrupted' for a pid that's merely gone."""
    from makermodslab.jobs import _EXIT_STATUS_FILENAME, JobRegistry

    root = tmp_path / "root"
    job_dir = root / "job-1"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / _EXIT_STATUS_FILENAME).write_text("1")
    _write_running_job_json(job_dir, output_dir)

    reg = JobRegistry(root)

    record = reg.get("job-1")
    assert record.state == "failed"
    assert record.exit_code == 1
    assert record.error_message is not None
    assert "exited with code 1" in record.error_message
    # Persisted, not just fixed in memory.
    meta = _json.loads((job_dir / "job.json").read_text())
    assert meta["state"] == "failed"


def test_boot_reattach_reads_exit_status_when_pid_already_dead_and_done(tmp_path) -> None:
    """Mirror of the failed case above: a run that actually finished
    successfully while the server was down must be recognised as 'done', not
    downgraded to 'interrupted' just because nobody was watching when it
    exited."""
    from makermodslab.jobs import _EXIT_STATUS_FILENAME, JobRegistry

    root = tmp_path / "root"
    job_dir = root / "job-1"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / _EXIT_STATUS_FILENAME).write_text("0")
    _write_running_job_json(job_dir, output_dir)

    reg = JobRegistry(root)

    record = reg.get("job-1")
    assert record.state == "done"
    assert record.exit_code == 0
    meta = _json.loads((job_dir / "job.json").read_text())
    assert meta["state"] == "done"


def test_boot_reattach_stays_interrupted_when_no_exit_status_file(tmp_path) -> None:
    """Regression guard for the existing, still-correct case: a pid that's
    dead AND left no exit_status file at all (SIGKILL, a reboot that cut off
    the wrapper before it could write) is genuinely unconfirmed and must stay
    'interrupted', same as before this fix."""
    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    job_dir = root / "job-1"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)  # no exit_status written
    _write_running_job_json(job_dir, output_dir)

    reg = JobRegistry(root)

    record = reg.get("job-1")
    assert record.state == "interrupted"
    assert record.exit_code is None


def _write_log(path: Path, messages: list[str]) -> Path:
    """Write messages in the log.jsonl shape both runners produce."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_json.dumps({"timestamp": 1.0, "message": m}) for m in messages) + "\n")
    return path


def test_oom_failure_reason_names_the_gpu_oom(tmp_path) -> None:
    """The whole point: a run that died on CUDA OOM must finalise with a reason
    the user can act on, not "Subprocess exited with code 1"."""
    from makermodslab.jobs import _oom_failure_reason

    log = _write_log(
        tmp_path / "log.jsonl",
        [
            "INFO 2026-08-07 10:31:02 train.py:243 step:1 loss:2.104",
            'File "/app/lerobot/policies/pi05/modeling_pi05.py", line 612, in forward',
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.20 GiB. GPU 0 has a total "
            "capacity of 79.14 GiB of which 1.88 GiB is free.",
            "[wrapper] trainer exited with rc=1",
        ],
    )
    reason = _oom_failure_reason(log, 1)
    assert reason is not None
    assert "memory" in reason.lower()
    assert "batch size" in reason.lower()


def test_oom_failure_reason_reads_past_the_last_line(tmp_path) -> None:
    """torch prints the OOM body BELOW the exception line and the trainer keeps
    logging on its way down, so matching only the final line would miss it."""
    from makermodslab.jobs import _oom_failure_reason

    log = _write_log(
        tmp_path / "log.jsonl",
        ["torch.OutOfMemoryError: CUDA out of memory."]
        + [f"[wrapper] scanning checkpoints {i}" for i in range(30)],
    )
    assert _oom_failure_reason(log, 1) is not None


def test_oom_failure_reason_is_silent_on_an_ordinary_failure(tmp_path) -> None:
    """No OOM evidence ⇒ None, so the caller keeps its existing message rather
    than mislabelling every failure as out of memory."""
    from makermodslab.jobs import _oom_failure_reason

    log = _write_log(tmp_path / "log.jsonl", ["ValueError: expected 6-dim action, got 12"])
    assert _oom_failure_reason(log, 1) is None
    assert _oom_failure_reason(tmp_path / "missing.jsonl", 1) is None


def test_oom_failure_reason_recognises_a_sigkill_with_an_empty_log(tmp_path) -> None:
    """The host OOM killer sends SIGKILL and the process prints nothing, so the
    exit code is the only evidence there is."""
    from makermodslab.jobs import _oom_failure_reason

    log = _write_log(tmp_path / "log.jsonl", ["INFO step:120 loss:0.8"])
    for rc in (-9, 137):
        reason = _oom_failure_reason(log, rc)
        assert reason is not None and "ram" in reason.lower()
    assert _oom_failure_reason(log, 1) is None


def test_read_log_tail_messages_survives_a_mid_line_seek(tmp_path) -> None:
    """The reader seeks to a fixed byte offset, which lands inside a record;
    the fragment must be dropped, not fed to json.loads as a whole line."""
    from makermodslab.jobs import _LOG_TAIL_BYTES, _read_log_tail_messages

    filler = ["x" * 200 for _ in range(_LOG_TAIL_BYTES // 200 + 40)]
    log = _write_log(tmp_path / "log.jsonl", [*filler, "torch.OutOfMemoryError: CUDA out of memory."])
    messages = _read_log_tail_messages(log)
    assert messages  # the fragment didn't take the whole window with it
    assert messages[-1] == "torch.OutOfMemoryError: CUDA out of memory."


def test_read_log_tail_messages_skips_malformed_lines(tmp_path) -> None:
    from makermodslab.jobs import _read_log_tail_messages

    path = tmp_path / "log.jsonl"
    path.write_text('{"timestamp": 1.0, "message": "ok"}\nnot json at all\n{"timestamp": 2.0}\n')
    assert _read_log_tail_messages(path) == ["ok"]


# ---------------------------------------------------------------------------
# Resume lineage: the child index, the ancestor walk, and the delete guard that
# reads them. All pure registry state — no runner is started here.


def _lineage_record(
    job_id: str,
    *,
    parent: str | None = None,
    started_at: float = 0.0,
    finetune_parent: str | None = None,
    state: str = "failed",
):
    """A bare record whose only interesting property is who it continues from."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    return JobRecord(
        id=job_id,
        name=job_id,
        state=state,
        config=TrainingRequest(
            dataset_repo_id="user/ds",
            resume=parent is not None,
            resume_from_job_id=parent,
            finetune_from_job_id=finetune_parent,
        ),
        output_dir=f"/nonexistent/{job_id}",
        started_at=started_at,
    )


def test_build_child_index_maps_parents_to_children_newest_first() -> None:
    from makermodslab.jobs import build_child_index

    index = build_child_index(
        [
            _lineage_record("A"),
            _lineage_record("B", parent="A", started_at=10.0),
            _lineage_record("C", parent="B", started_at=20.0),
        ]
    )

    assert index == {"A": ["B"], "B": ["C"]}


def test_build_child_index_keeps_every_child_of_a_fork_newest_first() -> None:
    """LEGACY DATA. `start` now refuses a second resume off one parent (sticks
    only, user decision 2026-08-07), so no new fork can appear — but registries
    written before that rule hold real ones, and the index they load through
    stays forest-capable: both children indexed, newest first. Rolling this
    back to "one child" would silently drop a leaf from the list."""
    from makermodslab.jobs import build_child_index

    index = build_child_index(
        [
            _lineage_record("A"),
            _lineage_record("older", parent="A", started_at=10.0),
            _lineage_record("newer", parent="A", started_at=20.0),
        ]
    )

    assert index["A"] == ["newer", "older"]


def test_build_child_index_ignores_finetune_edges() -> None:
    """A fine-tune starts a fresh schedule from a checkpoint's weights: a new
    model, not a continuation, so it must NOT supersede (hide) its source."""
    from makermodslab.jobs import build_child_index

    index = build_child_index(
        [
            _lineage_record("A"),
            _lineage_record("F", finetune_parent="A", started_at=10.0),
        ]
    )

    assert index == {}


def test_build_child_index_drops_a_self_edge() -> None:
    from makermodslab.jobs import build_child_index

    assert build_child_index([_lineage_record("A", parent="A")]) == {}


def test_ancestor_ids_walk_nearest_parent_first() -> None:
    from makermodslab.jobs import ancestor_ids_of

    records = {
        r.id: r
        for r in [
            _lineage_record("A"),
            _lineage_record("B", parent="A"),
            _lineage_record("C", parent="B"),
        ]
    }

    assert ancestor_ids_of(records, "C") == ["B", "A"]
    assert ancestor_ids_of(records, "A") == []


def test_ancestor_ids_of_forked_siblings_share_the_trunk() -> None:
    """LEGACY DATA, same as the fork index above: the walk is per-record and
    upward, so a trunk with two leaves hanging off it reads correctly from
    either leaf. Unchanged by the sticks rule, which only refuses new forks."""
    from makermodslab.jobs import ancestor_ids_of

    records = {
        r.id: r
        for r in [
            _lineage_record("A"),
            _lineage_record("B", parent="A"),
            _lineage_record("fork1", parent="B"),
            _lineage_record("fork2", parent="B"),
        ]
    }

    assert ancestor_ids_of(records, "fork1") == ["B", "A"]
    assert ancestor_ids_of(records, "fork2") == ["B", "A"]


def test_ancestor_ids_truncate_at_a_deleted_ancestor() -> None:
    """A source run that no longer exists ends the walk — the lineage just
    starts later, exactly as read_metrics_history's curve does."""
    from makermodslab.jobs import ancestor_ids_of

    records = {
        r.id: r
        for r in [
            _lineage_record("B", parent="gone"),
            _lineage_record("C", parent="B"),
        ]
    }

    assert ancestor_ids_of(records, "C") == ["B"]


def test_ancestor_ids_survive_a_cycle() -> None:
    """Corrupt data that points a chain back at itself must terminate, not spin."""
    from makermodslab.jobs import ancestor_ids_of

    records = {
        r.id: r
        for r in [
            _lineage_record("A", parent="B"),
            _lineage_record("B", parent="A"),
        ]
    }

    assert ancestor_ids_of(records, "A") == ["B"]


def test_list_annotates_lineage_and_marks_leaves(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    for rec in [
        _lineage_record("A", started_at=1.0),
        _lineage_record("B", parent="A", started_at=2.0),
        _lineage_record("C", parent="B", started_at=3.0),
    ]:
        reg._records[rec.id] = rec

    by_id = {r.id: r for r in reg.list(limit=10)}

    assert by_id["A"].child_ids == ["B"] and by_id["A"].ancestor_ids == []
    assert by_id["B"].child_ids == ["C"] and by_id["B"].ancestor_ids == ["A"]
    # The tip of the chain is the leaf: no children, whole trunk behind it.
    assert by_id["C"].child_ids == [] and by_id["C"].ancestor_ids == ["B", "A"]


def test_list_annotates_a_legacy_fork_unchanged(tmp_path) -> None:
    """A registry that already holds a fork keeps listing exactly as it did:
    the trunk names BOTH children (so it is superseded and hidden), and each
    leaf carries the shared trunk behind it, so the UI renders one row per leaf.

    This is the half of the sticks decision that is deliberately NOT enforced.
    The refusal lives at creation time only — there is no migration, and no
    load- or list-time rejection of data that predates it."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    for rec in [
        _lineage_record("trunk", started_at=1.0),
        _lineage_record("older", parent="trunk", started_at=2.0),
        _lineage_record("newer", parent="trunk", started_at=3.0),
    ]:
        reg._records[rec.id] = rec

    by_id = {r.id: r for r in reg.list(limit=10)}

    assert by_id["trunk"].child_ids == ["newer", "older"]
    assert by_id["older"].child_ids == [] and by_id["older"].ancestor_ids == ["trunk"]
    assert by_id["newer"].child_ids == [] and by_id["newer"].ancestor_ids == ["trunk"]


def test_list_sees_a_child_that_fell_off_the_page(tmp_path) -> None:
    """The child index is built over the whole registry, so a parent is still
    known to be superseded when its successor is past the listing's limit —
    the hole in the old client-side approximation."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    reg._records["A"] = _lineage_record("A", started_at=1.0)
    reg._records["B"] = _lineage_record("B", parent="A", started_at=2.0)

    # limit=1 returns only the newest (B); A is off the page entirely.
    page = reg.list(limit=1)
    assert [r.id for r in page] == ["B"]
    # ...and asking for A alone still reports its successor.
    assert reg.get("A").child_ids == ["B"]


def test_get_annotates_lineage(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    reg._records["A"] = _lineage_record("A", started_at=1.0)
    reg._records["B"] = _lineage_record("B", parent="A", started_at=2.0)

    record = reg.get("B")
    assert record.child_ids == []
    assert record.ancestor_ids == ["A"]


def test_delete_refuses_a_run_that_was_continued(tmp_path) -> None:
    """Deleting mid-chain would orphan the subtree (and wipe the local
    checkpoint dir its children resumed out of), so it is refused."""
    from makermodslab.jobs import JobHasChildrenError, JobRegistry

    reg = JobRegistry(tmp_path)
    reg._records["A"] = _lineage_record("A", started_at=1.0)
    reg._records["B"] = _lineage_record("B", parent="A", started_at=2.0)

    with pytest.raises(JobHasChildrenError) as excinfo:
        reg.delete("A")
    assert excinfo.value.child_ids == ["B"]
    # Nothing was removed.
    assert set(reg._records) == {"A", "B"}


def test_delete_allows_a_leaf_then_its_freed_parent(tmp_path) -> None:
    """Deleting from the tip inwards works: once the child is gone the parent
    is itself a leaf."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    reg._records["A"] = _lineage_record("A", started_at=1.0)
    reg._records["B"] = _lineage_record("B", parent="A", started_at=2.0)

    reg.delete("B")
    reg.delete("A")

    assert reg._records == {}


def test_delete_is_unaffected_by_a_finetune_child(tmp_path) -> None:
    """A fine-tune is not a lineage edge, so its source stays deletable."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path)
    reg._records["A"] = _lineage_record("A", started_at=1.0)
    reg._records["F"] = _lineage_record("F", finetune_parent="A", started_at=2.0)

    reg.delete("A")

    assert set(reg._records) == {"F"}


# ---------------------------------------------------------------------------
# The HTTP half of the second-resume refusal: POST /jobs/training turns
# JobAlreadyContinuedError into a 409 whose message teaches the way out. The
# registry is stubbed — what is under test is the routing and the wording, not
# the rule (covered above).


def _post_resume(client, source_id: str = "src"):
    return client.post(
        "/jobs/training",
        json={
            "dataset_repo_id": "user/ds",
            "steps": 200,
            "resume": True,
            "resume_from_job_id": source_id,
        },
    )


def test_endpoint_409s_a_second_continuation_and_names_the_way_out(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """409, not 400: a conflict with existing state, routed exactly like the
    mid-chain delete refusal it is the mirror of. The message must name the run
    to delete AND the run that frees up, because neither is guessable from
    "resume refused"."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)

    resp = _post_resume(client)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "src" in detail and "kid" in detail
    assert "already continued" in detail
    assert "delete" in detail.lower()


def test_endpoint_409_labels_the_runs_by_their_display_name(client, monkeypatch) -> None:
    """Telling the user to delete X is only actionable if X is findable in the
    list, and the list shows the display alias — so the message resolves ids to
    names when the registry still holds them."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobNotFoundError, JobRecord
    from makermodslab.train import TrainingRequest

    names = {"src": "overnight act", "kid": "overnight act v2"}

    def _fake_get(job_id: str):
        if job_id not in names:
            raise JobNotFoundError(job_id)
        return JobRecord(
            id=job_id,
            name="auto-generated",
            display_name=names[job_id],
            state="interrupted",
            config=TrainingRequest(dataset_repo_id="user/ds"),
            output_dir=f"/nonexistent/{job_id}",
            started_at=0.0,
        )

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)

    detail = _post_resume(client).json()["detail"]

    assert "overnight act" in detail and "overnight act v2" in detail


def test_endpoint_409_falls_back_to_the_id_when_a_run_is_gone(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label lookup must not turn a refusal into a 500 when the registry
    can't resolve an id (a record deleted between the failure and the message)."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobNotFoundError

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    def _fake_get(job_id: str):
        raise JobNotFoundError(job_id)

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)

    resp = _post_resume(client)

    assert resp.status_code == 409
    assert "src" in resp.json()["detail"]


# The 409's REMEDY is child-aware: "delete the continuation" is sound advice
# only for the single-unfinished-child lineage the sticks rule creates. On a
# legacy fork, or against a continuation that ran to completion, the same
# sentence tells the user to throw away finished training.


def _stub_children(monkeypatch, source_id: str, children: dict[str, str]):
    """Raise JobAlreadyContinuedError(source_id, children) from start, and make
    the registry resolve each child id to a record in the given state."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobNotFoundError, JobRecord
    from makermodslab.train import TrainingRequest

    states = {source_id: "interrupted", **children}

    def _fake_get(job_id: str):
        if job_id not in states:
            raise JobNotFoundError(job_id)
        return JobRecord(
            id=job_id,
            name=job_id,
            state=states[job_id],
            config=TrainingRequest(dataset_repo_id="user/ds"),
            output_dir=f"/nonexistent/{job_id}",
            started_at=0.0,
        )

    def _raise(config, target):
        raise JobAlreadyContinuedError(source_id, list(children))

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)


def test_endpoint_409_keeps_delete_first_for_one_unfinished_child(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The linear case — the only shape sticks can produce — still gets the
    cheap, correct two-step remedy."""
    _stub_children(monkeypatch, "src", {"kid": "interrupted"})

    detail = _post_resume(client).json()["detail"]

    assert "delete" in detail.lower()
    assert "fine-tune" not in detail.lower()


def test_endpoint_409_recommends_finetune_on_a_legacy_fork(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two children: freeing the parent means deleting BOTH, so deletion stops
    being reasonable advice and fine-tune (unrestricted by sticks) takes over."""
    _stub_children(monkeypatch, "src", {"kid": "interrupted", "other": "failed"})

    detail = _post_resume(client).json()["detail"]

    assert "fine-tune" in detail.lower()
    assert "kid" in detail and "other" in detail


def test_endpoint_409_will_not_advise_deleting_a_finished_run(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user's actual case: a lone continuation that ran to completion. The
    advice must not be "delete it" — that is the run holding the finished
    training — and the message says which run it is protecting."""
    _stub_children(monkeypatch, "src", {"kid": "done"})

    detail = _post_resume(client).json()["detail"]

    assert "fine-tune" in detail.lower()
    assert "finished run" in detail
    assert "kid" in detail


def test_endpoint_409_survives_an_unresolvable_child(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The state lookup must not turn the refusal into a 500 when a child id no
    longer resolves; it just can't claim the run is finished."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobNotFoundError

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    def _fake_get(job_id: str):
        raise JobNotFoundError(job_id)

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)

    resp = _post_resume(client)

    assert resp.status_code == 409
    assert "finished run" not in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Run numbers. A persisted, monotonic counter — the point is that a number is
# never handed out twice, which is exactly what deriving max(existing)+1 at
# render time cannot promise.


def _numbered_registry(tmp_path):
    from makermodslab.jobs import JobRegistry

    return JobRegistry(tmp_path / "root")


def test_start_numbers_runs_from_one_upward(tmp_path) -> None:
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget
    from makermodslab.train import TrainingRequest

    reg = _numbered_registry(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    numbers = []
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        for _ in range(3):
            rec = reg.start(
                TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
                JobTarget(runner="local"),
            )
            reg._records[rec.id].state = "done"  # free the one-local-run mutex
            numbers.append(rec.job_number)

    assert numbers == [1, 2, 3]


def test_job_numbers_survive_a_registry_reload(tmp_path) -> None:
    """The counter is on disk, so a restart continues the sequence instead of
    starting over and colliding with the runs already numbered."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = _numbered_registry(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        first = reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )
        reg._records[first.id].state = "done"
        reg._persist(reg._records[first.id], force=True)

    reopened = JobRegistry(tmp_path / "root")
    assert reopened.get(first.id).job_number == first.job_number

    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        second = reopened.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )

    assert second.job_number == first.job_number + 1


def test_deleting_the_highest_numbered_run_does_not_free_its_number(tmp_path) -> None:
    """THE reason the counter is persisted rather than derived. max(existing)+1
    would reissue #1 here, so two different runs would have worn the same
    number — across a restart too, since the counter file outlives the record."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = _numbered_registry(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        first = reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )
    reg._records[first.id].state = "done"
    reg._persist(reg._records[first.id], force=True)
    assert first.job_number == 1

    reg.delete(first.id)
    # ...and the registry is empty, so a derived number would restart at 1.
    assert reg._records == {}

    reopened = JobRegistry(tmp_path / "root")
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        second = reopened.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )

    assert second.job_number == 2


def test_backfill_numbers_legacy_records_oldest_first(tmp_path) -> None:
    """Records written before the field existed get numbers in the order they
    happened, so the sequence agrees with the history the user remembers."""
    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    for job_id, started in (("late", 300.0), ("early", 100.0), ("middle", 200.0)):
        rec = _lineage_record(job_id, started_at=started)
        d = root / job_id
        d.mkdir(parents=True)
        # Written WITHOUT job_number, the shape a pre-existing registry holds.
        data = rec.model_dump(mode="json")
        data.pop("job_number")
        (d / "job.json").write_text(_json.dumps(data))

    reg = JobRegistry(root)

    assert reg.get("early").job_number == 1
    assert reg.get("middle").job_number == 2
    assert reg.get("late").job_number == 3


def test_backfill_breaks_started_at_ties_deterministically(tmp_path) -> None:
    """Legacy timestamps are second-granular, so ties are real. Without a
    tie-break two boots could order the same pair differently and silently
    renumber history."""
    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    for job_id in ("bbb", "aaa"):
        rec = _lineage_record(job_id, started_at=100.0)
        d = root / job_id
        d.mkdir(parents=True)
        data = rec.model_dump(mode="json")
        data.pop("job_number")
        (d / "job.json").write_text(_json.dumps(data))

    reg = JobRegistry(root)

    assert reg.get("aaa").job_number == 1
    assert reg.get("bbb").job_number == 2


def test_backfill_is_idempotent_across_restarts(tmp_path) -> None:
    """A second boot must find everything numbered and change nothing — the
    numbers are persisted back to each job.json, not recomputed per process."""
    from makermodslab.jobs import JobRegistry

    root = tmp_path / "root"
    for job_id, started in (("a", 100.0), ("b", 200.0)):
        rec = _lineage_record(job_id, started_at=started)
        d = root / job_id
        d.mkdir(parents=True)
        data = rec.model_dump(mode="json")
        data.pop("job_number")
        (d / "job.json").write_text(_json.dumps(data))

    first = {r.id: r.job_number for r in JobRegistry(root).list(limit=10)}
    second = {r.id: r.job_number for r in JobRegistry(root).list(limit=10)}

    assert first == {"a": 1, "b": 2}
    assert first == second


def test_a_lost_counter_file_still_will_not_reissue_a_live_number(tmp_path) -> None:
    """Degraded case: the counter is gone but the records are not. The floor is
    recomputed from the records, and persisted, so the next boot keeps it even
    after the highest-numbered run is deleted."""
    from makermodslab.jobs import JobRegistry, _job_counter_path

    root = tmp_path / "root"
    for job_id, number in (("a", 1), ("b", 7)):
        rec = _lineage_record(job_id, started_at=float(number))
        rec.job_number = number
        d = root / job_id
        d.mkdir(parents=True)
        (d / "job.json").write_text(rec.model_dump_json())

    reg = JobRegistry(root)
    assert _job_counter_path(root).exists()
    assert reg._next_job_number == 8

    reg.delete("b")
    assert JobRegistry(root)._next_job_number == 8


def test_a_corrupt_counter_file_is_ignored_not_fatal(tmp_path) -> None:
    """A half-written or hand-edited counter must not take the registry down,
    and must not read as a number below the records in use."""
    from makermodslab.jobs import JobRegistry, _job_counter_path

    root = tmp_path / "root"
    rec = _lineage_record("a", started_at=1.0)
    rec.job_number = 4
    (root / "a").mkdir(parents=True)
    (root / "a" / "job.json").write_text(rec.model_dump_json())
    _job_counter_path(root).write_text("{not json")

    assert JobRegistry(root)._next_job_number == 5


def test_the_counter_file_is_not_mistaken_for_a_job(tmp_path) -> None:
    """It lives in the registry root beside the job dirs, so the loader has to
    keep ignoring it (it globs directories only)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = _numbered_registry(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        rec = reg.start(
            TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=100),
            JobTarget(runner="local"),
        )
    reg._records[rec.id].state = "done"
    reg._persist(reg._records[rec.id], force=True)

    assert [r.id for r in JobRegistry(tmp_path / "root").list(limit=10)] == [rec.id]


def test_imported_records_take_a_number_from_the_same_sequence(tmp_path) -> None:
    """Imports share the libraries with runs, so a library where some rows have
    a number and some don't is the thing to avoid."""
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")

    assert reg.register_imported(str(model)).job_number == 1


def test_endpoint_409_label_leads_with_the_run_number(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The number is what the UI shows, so the API's refusals have to speak it
    too — while keeping the id, which is what survives a rename."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobNotFoundError, JobRecord
    from makermodslab.train import TrainingRequest

    numbers = {"src": 46, "kid": 47}

    def _fake_get(job_id: str):
        if job_id not in numbers:
            raise JobNotFoundError(job_id)
        return JobRecord(
            id=job_id,
            job_number=numbers[job_id],
            name="overnight act",
            state="interrupted",
            config=TrainingRequest(dataset_repo_id="user/ds"),
            output_dir=f"/nonexistent/{job_id}",
            started_at=0.0,
        )

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)

    detail = _post_resume(client).json()["detail"]

    assert "#46" in detail and "#47" in detail
    assert "src" in detail and "kid" in detail  # ids still present


def test_endpoint_409_label_omits_the_number_when_unassigned(client, monkeypatch) -> None:
    """A record that predates the field must not be labelled "#0"."""
    import makermodslab.server as server_mod
    from makermodslab.jobs import JobAlreadyContinuedError, JobRecord
    from makermodslab.train import TrainingRequest

    def _fake_get(job_id: str):
        return JobRecord(
            id=job_id,
            name="overnight act",
            state="interrupted",
            config=TrainingRequest(dataset_repo_id="user/ds"),
            output_dir=f"/nonexistent/{job_id}",
            started_at=0.0,
        )

    def _raise(config, target):
        raise JobAlreadyContinuedError("src", ["kid"])

    monkeypatch.setattr(server_mod.job_registry, "start", _raise)
    monkeypatch.setattr(server_mod.job_registry, "get", _fake_get)

    assert "#0" not in _post_resume(client).json()["detail"]


# ---------------------------------------------------------------------------
# CHAIN REWIND. A resume continues the LEAF from ANY checkpoint on its lineage:
# the edge points at the leaf (so chains stay linear), while the bytes are read
# from whichever ancestor owns the chosen checkpoint. The empty-handed tip — a
# run that died before saving anything — is the case this exists for.


def _rewind_chain(tmp_path):
    """A two-run chain: `trunk` with real checkpoints at 100 and 200, and
    `tip` continuing it with none of its own (it died before its first save)."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    trunk_dir = tmp_path / "trunk" / "run"
    trunk_dir.mkdir(parents=True)
    _make_checkpoint(trunk_dir, 100)
    _make_checkpoint(trunk_dir, 200)
    tip_dir = tmp_path / "tip" / "run"
    tip_dir.mkdir(parents=True)

    reg = JobRegistry(tmp_path / "root")
    reg._records["trunk"] = JobRecord(
        id="trunk",
        name="run",
        state="interrupted",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=1000),
        output_dir=str(trunk_dir),
        started_at=0.0,
        runner="local",
    )
    reg._records["tip"] = JobRecord(
        id="tip",
        name="run",
        state="interrupted",
        config=TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            steps=1000,
            resume=True,
            resume_from_job_id="trunk",
            resume_from_step=200,
        ),
        output_dir=str(tip_dir),
        started_at=1.0,
        runner="local",
    )
    return reg


def _rewind_request(*, leaf: str, owner: str | None, step: int | None, steps: int = 1000):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        steps=steps,
        resume=True,
        resume_from_job_id=leaf,
        resume_from_step=step,
        resume_from_checkpoint_job_id=owner,
    )


def test_rewind_resumes_an_empty_tip_from_its_ancestors_checkpoint(tmp_path) -> None:
    """THE case the redirect exists for. The tip saved nothing, so before rewind
    it was a dead row; now it continues itself from the trunk's checkpoint —
    and the new run is a child of the TIP, so the chain stays linear."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        record = reg.start(
            _rewind_request(leaf="tip", owner="trunk", step=200),
            JobTarget(runner="local"),
        )

    # The EDGE names the tip...
    assert record.config.resume_from_job_id == "tip"
    assert reg.get("tip").child_ids == [record.id]
    # ...and the trunk gains no second child, which is what keeps it linear.
    assert reg.get("trunk").child_ids == ["tip"]
    # The BYTES come from the trunk's checkpoint dir.
    assert "trunk" in record.config.config_path
    assert "checkpoints/200" in record.config.config_path


def test_rewind_can_reach_an_older_checkpoint_of_the_ancestor(tmp_path) -> None:
    """Any checkpoint on the lineage, not just the newest — rewinding past a
    bad stretch is the point."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        record = reg.start(
            _rewind_request(leaf="tip", owner="trunk", step=100),
            JobTarget(runner="local"),
        )

    assert "checkpoints/100" in record.config.config_path
    assert record.config.resume_from_job_id == "tip"


def test_a_plain_tip_resume_still_needs_no_owner(tmp_path) -> None:
    """Backward compatibility: omitting the owner means the leaf owns the
    checkpoint, which is every request written before rewind existed."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")  # checkpoint at step 100
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        record = reg.start(
            _rewind_request(leaf="src", owner=None, step=100, steps=200),
            JobTarget(runner="local"),
        )

    assert record.config.resume_from_checkpoint_job_id is None
    assert "checkpoints/100" in record.config.config_path


def test_rewind_naming_the_leaf_as_its_own_owner_is_accepted(tmp_path) -> None:
    """The explicit spelling of the same thing — the UI omits it, but a caller
    that sends it must not be refused for agreeing with the default."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    fake = MagicMock()
    fake.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake):
        record = reg.start(
            _rewind_request(leaf="src", owner="src", step=100, steps=200),
            JobTarget(runner="local"),
        )

    assert "checkpoints/100" in record.config.config_path


def test_rewind_refuses_an_owner_off_the_leaf_lineage(tmp_path) -> None:
    """The wrong-weights guard. A run that is not an ancestor has no business
    seeding this chain — its bytes would enter a history claiming continuity
    with them."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRecord, JobTarget
    from makermodslab.train import TrainingRequest

    reg = _rewind_chain(tmp_path)
    stranger_dir = tmp_path / "stranger" / "run"
    stranger_dir.mkdir(parents=True)
    _make_checkpoint(stranger_dir, 100)
    reg._records["stranger"] = JobRecord(
        id="stranger",
        name="unrelated",
        state="interrupted",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=1000),
        output_dir=str(stranger_dir),
        started_at=2.0,
        runner="local",
    )

    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="is not on 'tip''s lineage"),
    ):
        reg.start(
            _rewind_request(leaf="tip", owner="stranger", step=100),
            JobTarget(runner="local"),
        )


def test_rewind_refuses_an_owner_that_lacks_the_named_step(tmp_path) -> None:
    """Naming a real ancestor is not enough — it must actually hold that
    checkpoint, or the resolver's 'latest' fallback would silently substitute
    different weights."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="no checkpoint at step 999"),
    ):
        reg.start(
            _rewind_request(leaf="tip", owner="trunk", step=999),
            JobTarget(runner="local"),
        )


def test_rewind_refuses_an_unknown_owner(tmp_path) -> None:
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Resume checkpoint owner 'ghost' not found"),
    ):
        reg.start(
            _rewind_request(leaf="tip", owner="ghost", step=100),
            JobTarget(runner="local"),
        )


def test_rewind_requires_an_explicit_step(tmp_path) -> None:
    """'Latest' has no meaning once an owner is named: a rewound lineage can
    hold several checkpoints at one step, so the pair must be exact."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="resume_from_step is required"),
    ):
        reg.start(
            _rewind_request(leaf="tip", owner="trunk", step=None),
            JobTarget(runner="local"),
        )


def test_rewind_still_refuses_a_second_continuation_of_the_leaf(tmp_path) -> None:
    """The sticks 409 survives the redirect and is now pure API integrity: no
    legitimate caller names a non-leaf as the edge, because the UI always seeds
    the leaf. A leaf that already has a child is not a leaf."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobAlreadyContinuedError, JobTarget

    reg = _rewind_chain(tmp_path)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(JobAlreadyContinuedError),
    ):
        # `trunk` is mid-chain — `tip` already continues it.
        reg.start(
            _rewind_request(leaf="trunk", owner="trunk", step=200),
            JobTarget(runner="local"),
        )


def test_rewind_steps_guard_reads_the_chosen_checkpoint(tmp_path) -> None:
    """The target must beat the step actually being resumed from, which after a
    rewind is the ANCESTOR's step, not the leaf's progress."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _rewind_chain(tmp_path)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="would train nothing"),
    ):
        reg.start(
            _rewind_request(leaf="tip", owner="trunk", step=200, steps=200),
            JobTarget(runner="local"),
        )
