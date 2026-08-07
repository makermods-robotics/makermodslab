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


def test_resolve_cloud_resume_returns_repo_and_step_dir(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000")))
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

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000")))
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
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", _fake_runner_factory),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"
    fake_runner.start.assert_called_once()


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


def _tailing_runner(pid, monkeypatch, *, alive=True):
    """A TailingJobRunner over a fake pid; os.kill is stubbed so no real
    process is signalled."""
    from makermodslab import jobs as jobs_mod

    state = {"alive": alive}

    def fake_kill(target_pid, sig):
        assert target_pid == pid
        if not state["alive"]:
            raise ProcessLookupError(pid)
        if sig != 0:
            state["alive"] = False  # SIGTERM landed

    monkeypatch.setattr(jobs_mod.os, "kill", fake_kill)
    runner = jobs_mod.TailingJobRunner(jobs_mod.TrainingMetrics(), Path("/nonexistent"), pid)
    return runner, state


def test_tailing_runner_reports_sigterm_after_a_delivered_stop(monkeypatch) -> None:
    """Its returncode() synthesises 0 when the pid is gone, which would file a
    deliberate stop as `done`. Once we know we signalled a live pid, naming
    the signal is the more honest synthetic answer."""
    import signal as signal_mod

    runner, _ = _tailing_runner(31337, monkeypatch)
    assert runner.returncode() is None  # still alive

    runner.stop()
    assert runner.stop_signalled() is True
    assert runner.returncode() == -signal_mod.SIGTERM


def test_tailing_runner_keeps_optimistic_zero_when_pid_was_already_gone(monkeypatch) -> None:
    """Nothing was signalled, so the pid's absence isn't ours to claim: the
    detached run that finished normally still finalises as `done`."""
    runner, _ = _tailing_runner(31338, monkeypatch, alive=False)

    runner.stop()
    assert runner.stop_signalled() is False
    assert runner.returncode() == 0


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
    completeness rule instead of a stub that always says yes."""

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
        step_dir = kwargs["path_in_repo"].rsplit("/", 1)[-1]
        self._files.extend(_hub_checkpoint_files(step_dir))

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
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert record.config.resume is True
    assert record.config.resume_from_hub_repo == "user/some-model"
    assert record.config.resume_from_hub_step == "000100"


# ---------------------------------------------------------------------------
# Human model names: a run started from the UI is named by the user, and that
# name — not the policy/dataset slug — becomes the run's id and therefore the
# Hub repo it publishes to. The machine shape stays as the fallback for callers
# that name nothing (the bare API, imported records).
# ---------------------------------------------------------------------------


def _named_request(job_name: str | None):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", policy_type="act", job_name=job_name)


def _start_local(reg, cfg):
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        return reg.start(cfg, JobTarget(runner="local"))


def test_start_builds_the_job_id_from_the_model_name(tmp_path) -> None:
    """The human stem replaces the machine prefix: `{name}_{timestamp}`. The
    timestamp keeps the shape utils.naming.RUN_REPO_TIMESTAMP_RE peels off for
    display titles, so a named run still reads as a MakerMods Lab run — and the
    record's display name stays the verbatim name the user typed."""
    from makermodslab.jobs import JobRegistry
    from makermodslab.utils.naming import RUN_REPO_TIMESTAMP_RE

    reg = JobRegistry(tmp_path / "root")
    record = _start_local(reg, _named_request("eraser_stack"))

    match = RUN_REPO_TIMESTAMP_RE.search(record.id)
    assert match is not None
    assert record.id[: match.start()] == "eraser_stack"
    assert record.name == "eraser_stack"
    # The id is the job directory's name, so the whole run's storage follows it.
    assert (reg._output_root / record.id).is_dir()


def test_start_falls_back_to_machine_naming_without_a_model_name(tmp_path) -> None:
    """`job_name` stays optional on the API — the requirement lives in the form,
    the same division of labour recording uses. A bare caller gets the legacy
    `{policy}_{dataset_slug}_{timestamp}` id, unchanged."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_local(reg, _named_request(None))

    assert record.id.startswith("act_user_ds_")
    assert record.name == "ACT · user/ds"

    # Whitespace-only is "no name", not a name made of spaces.
    reg._records[record.id].state = "done"  # the local-run mutex allows only one
    blank = _start_local(reg, _named_request("   "))
    assert blank.id.startswith("act_user_ds_")


@pytest.mark.parametrize(
    "job_name",
    [
        "my model",  # spaces
        "user/model",  # a slash would smuggle in a namespace
        "-leading-dash",
        "model!",
        "..",
        "x" * 97,
    ],
)
def test_start_refuses_a_model_name_that_isnt_a_repo_segment(tmp_path, job_name) -> None:
    """The name becomes one path segment of the run's Hub repo id, so a name
    that can't be one is REFUSED (ValueError → 400) rather than slugified into
    something the user never typed. The refusal is synchronous and leaves no
    record, directory or log behind."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(ValueError, match="Model name"):
        _start_local(reg, _named_request(job_name))

    assert reg.list(limit=10) == []
    _assert_nothing_was_created(reg)


def test_two_runs_of_the_same_model_name_get_unique_ids(tmp_path, monkeypatch) -> None:
    """Two runs sharing a NAME is the ordinary case for the human shape (train
    the same skill twice), so the collision guard must cover it too — a second
    run within the same second gets `-2`, not the first run's directory."""
    from makermodslab import jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_job_id_timestamp", lambda: "2026-08-04_11-02-19")
    reg = jobs_mod.JobRegistry(tmp_path / "root")
    first = _start_local(reg, _named_request("eraser_stack"))
    reg._records[first.id].state = "done"  # the local-run mutex allows only one
    second = _start_local(reg, _named_request("eraser_stack"))

    assert first.id == "eraser_stack_2026-08-04_11-02-19"
    assert second.id == "eraser_stack_2026-08-04_11-02-19-2"
    assert {r.id for r in reg.list(limit=10)} == {first.id, second.id}


def test_resume_is_unaffected_by_model_naming(tmp_path) -> None:
    """A continuation keeps the parent run's identity — the form locks the name
    field, so no name is sent and the id falls back to the machine shape. The
    resume resolution itself is untouched."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))

    assert record.config.resume is True
    assert record.id.startswith("act_user_ds_")


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
    """Nothing stops two runs resuming one parent, so the lineage is a forest,
    not a set of chains — both forks must be indexed, newest first."""
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
