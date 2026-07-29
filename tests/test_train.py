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
"""Tests for makermodslab.train — request schema and CLI builder."""

from __future__ import annotations

import pytest


def _arg_value(cmd: list[str], flag: str) -> str:
    """Return the value passed to `--flag`. Fails the test if absent."""
    assert flag in cmd, f"{flag} missing from {cmd}"
    return cmd[cmd.index(flag) + 1]


def test_minimal_request_yields_well_formed_argv() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert _arg_value(cmd, "--dataset.repo_id") == "lerobot/pusht"
    assert _arg_value(cmd, "--policy.type") == "act"
    assert _arg_value(cmd, "--steps") == "10000"
    assert _arg_value(cmd, "--output_dir") == "/tmp/out"


def test_resume_request_emits_minimal_argv() -> None:
    """On resume, lerobot reconstructs the run from config_path, so the builder
    must NOT re-pass --dataset.* / --policy.type (they'd fight the loaded
    config) and must pass the resume essentials plus the overridable knobs."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        resume=True,
        config_path="/runs/abc/checkpoints/5000/pretrained_model/train_config.json",
        steps=20000,
        num_workers=12,
        batch_size=64,
        seed=7,
    )
    cmd = build_training_command(req, output_dir="/tmp/new")

    # config_path MUST be the "--config_path=<path>" form: lerobot's own
    # pre-parser ignores the space-separated form.
    cfg_args = [a for a in cmd if a.startswith("--config_path=")]
    assert cfg_args == ["--config_path=/runs/abc/checkpoints/5000/pretrained_model/train_config.json"]
    assert "--config_path" not in cmd  # not the two-token form
    assert _arg_value(cmd, "--resume") == "true"
    assert _arg_value(cmd, "--output_dir") == "/tmp/new"
    assert _arg_value(cmd, "--steps") == "20000"
    assert _arg_value(cmd, "--log_freq") == str(req.log_freq)
    assert _arg_value(cmd, "--save_freq") == str(req.save_freq)
    # num_workers is a HOST-capacity knob, not an experiment property (the
    # resumed run's flavor can differ from the parent's), so it stays editable
    # on a continuation and must actually reach the CLI.
    assert _arg_value(cmd, "--num_workers") == "12"
    # Inherited from the checkpoint — must not be re-specified on the CLI.
    assert "--dataset.repo_id" not in cmd
    assert "--policy.type" not in cmd
    assert "--batch_size" not in cmd
    assert "--seed" not in cmd
    assert "--policy.device" not in cmd
    assert "--policy.use_amp" not in cmd
    assert "--optimizer.type" not in cmd
    # Optimizer state comes from the checkpoint on a resume — see
    # test_resume_emits_no_optimizer_flags for the full assertion.
    assert not any(tok.startswith("--policy.optimizer_") for tok in cmd)


def test_optional_dataset_fields_only_present_when_set() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, "/tmp/out")
    assert "--dataset.revision" not in cmd
    assert "--dataset.root" not in cmd
    assert "--dataset.episodes" not in cmd

    req2 = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        dataset_revision="v2",
        dataset_root="/data",
        dataset_episodes=[0, 1, 2],
    )
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--dataset.revision") == "v2"
    assert _arg_value(cmd2, "--dataset.root") == "/data"
    # `--dataset.episodes` is followed by 3 string-encoded ints.
    idx = cmd2.index("--dataset.episodes")
    assert cmd2[idx + 1 : idx + 4] == ["0", "1", "2"]


def test_wandb_block_only_serialized_when_enabled() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    off = build_training_command(TrainingRequest(dataset_repo_id="x", wandb_enable=False), "/tmp/out")
    assert _arg_value(off, "--wandb.enable") == "false"
    assert "--wandb.project" not in off

    on = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            wandb_enable=True,
            wandb_project="proj",
            wandb_entity="me",
            wandb_run_id="abc",
        ),
        "/tmp/out",
    )
    assert _arg_value(on, "--wandb.enable") == "true"
    assert _arg_value(on, "--wandb.project") == "proj"
    assert _arg_value(on, "--wandb.entity") == "me"
    assert _arg_value(on, "--wandb.run_id") == "abc"


def test_push_to_hub_emits_repo_id_only_when_enabled() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    off = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=False, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(off, "--policy.push_to_hub") == "false"
    assert "--policy.repo_id" not in off

    on = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(on, "--policy.push_to_hub") == "true"
    assert _arg_value(on, "--policy.repo_id") == "me/x"
    # A pushed policy is public and carries the required Hub tags.
    assert _arg_value(on, "--policy.private") == "false"
    assert _arg_value(on, "--policy.tags") == "[makermods,openbooth,MakerModsLab]"
    # When not pushing, no privacy/tags flags are emitted.
    assert "--policy.private" not in off
    assert "--policy.tags" not in off


def test_resume_push_to_hub_emits_public_but_never_tags() -> None:
    """The resume branch makes a pushed policy public but must NOT re-pass tags.

    Regression test for MT24. With --config_path present, lerobot parses through
    draccus.parse(cls, config_file, args=cli_args), which merges each CLI
    override into the config dict as a RAW STRING before decoding it against the
    field type — so `--policy.tags '[a,b,c]'` reaches list[str] decoding as the
    literal string and raises DecodingError, killing the run at startup. The
    checkpoint's own train_config.json already carries policy.tags, so dropping
    the flag loses nothing. The fresh-run branch still emits it (argparse splits
    the bracket form there) — covered by the test above.
    """
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="x",
        resume=True,
        config_path="/runs/abc/checkpoints/5000/pretrained_model/train_config.json",
        policy_push_to_hub=True,
        policy_repo_id="me/x",
    )
    cmd = build_training_command(req, "/tmp/new")

    assert _arg_value(cmd, "--policy.push_to_hub") == "true"
    assert _arg_value(cmd, "--policy.private") == "false"
    assert "--policy.tags" not in cmd


def test_seed_omitted_when_none() -> None:
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", seed=None)
    cmd = build_training_command(req, "/tmp/out")
    assert "--seed" not in cmd

    req2 = TrainingRequest(dataset_repo_id="x", seed=42)
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--seed") == "42"


def test_explicit_device_passes_through() -> None:
    """A concrete device (persisted by an older config) passes through
    unchanged for backward compatibility."""
    from makermodslab.train import TrainingRequest, build_training_command

    cmd = build_training_command(TrainingRequest(dataset_repo_id="x", policy_device="cuda"), "/tmp/out")
    assert _arg_value(cmd, "--policy.device") == "cuda"

    cmd_cpu = build_training_command(TrainingRequest(dataset_repo_id="x", policy_device="cpu"), "/tmp/out")
    assert _arg_value(cmd_cpu, "--policy.device") == "cpu"


def test_auto_device_resolves_to_concrete_backend(monkeypatch) -> None:
    """The default "auto" resolves to a real backend so the logged config is
    truthful. Resolution is made deterministic here via monkeypatch."""
    import torch

    from makermodslab.train import TrainingRequest, build_training_command

    # No GPU available -> cpu.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cmd = build_training_command(TrainingRequest(dataset_repo_id="x", policy_device="auto"), "/tmp/out")
    assert _arg_value(cmd, "--policy.device") == "cpu"

    # CUDA available -> cuda.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cmd_cuda = build_training_command(TrainingRequest(dataset_repo_id="x", policy_device="auto"), "/tmp/out")
    assert _arg_value(cmd_cuda, "--policy.device") == "cuda"


def test_default_device_is_auto_and_resolved(monkeypatch) -> None:
    """The request default is "auto" (not "cuda"); build resolves it to a
    concrete backend rather than emitting "auto"."""
    import torch

    from makermodslab.train import TrainingRequest, build_training_command

    assert TrainingRequest(dataset_repo_id="x").policy_device == "auto"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cmd = build_training_command(TrainingRequest(dataset_repo_id="x"), "/tmp/out")
    assert _arg_value(cmd, "--policy.device") == "mps"


def test_training_request_validates_required_field() -> None:
    from pydantic import ValidationError

    from makermodslab.train import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest()  # dataset_repo_id is required


# ---------------------------------------------------------------------------
# Optimizer knobs. The form always runs with use_policy_training_preset true,
# and lerobot then REPLACES the optimizer with the policy's preset
# (TrainPipelineConfig.validate, configs/train.py:249-253), so `--optimizer.*`
# is inert. The knobs must ride on the POLICY config the preset is built from,
# and only where that policy actually declares them (draccus fails at CLI parse
# on an unknown --policy.<field>). See MT43.
# ---------------------------------------------------------------------------


def test_optimizer_knobs_ride_on_policy_config() -> None:
    """lr + weight_decay reach argv as --policy.optimizer_* for a policy (act)
    that declares both."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="x",
        policy_type="act",
        optimizer_lr=1e-4,
        optimizer_weight_decay=0.01,
    )
    cmd = build_training_command(req, "/tmp/out")

    assert _arg_value(cmd, "--policy.optimizer_lr") == "0.0001"
    assert _arg_value(cmd, "--policy.optimizer_weight_decay") == "0.01"


def test_optimizer_knobs_absent_when_unset() -> None:
    """A knob the user left blank must not be forced onto argv — the policy
    preset's own default has to win."""
    from makermodslab.train import TrainingRequest, build_training_command

    cmd = build_training_command(TrainingRequest(dataset_repo_id="x", policy_type="act"), "/tmp/out")

    assert "--policy.optimizer_lr" not in cmd
    assert "--policy.optimizer_weight_decay" not in cmd
    assert "--policy.optimizer_grad_clip_norm" not in cmd


def test_grad_clip_gated_on_policy_support() -> None:
    """smolvla declares optimizer_grad_clip_norm; act does NOT. Passing it to
    act would make draccus die at parse time, so it must be dropped even though
    the user set a value."""
    from makermodslab.train import TrainingRequest, build_training_command

    smolvla = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_type="smolvla", optimizer_grad_clip_norm=5.0),
        "/tmp/out",
    )
    assert _arg_value(smolvla, "--policy.optimizer_grad_clip_norm") == "5.0"

    act = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_type="act", optimizer_grad_clip_norm=5.0),
        "/tmp/out",
    )
    assert "--policy.optimizer_grad_clip_norm" not in act
    # ...and dropping it must not disturb the knobs act DOES support.
    assert "--policy.type" in act


def test_weight_decay_gated_for_tdmpc() -> None:
    """tdmpc declares only optimizer_lr — weight_decay must be dropped."""
    from makermodslab.train import TrainingRequest, build_training_command

    cmd = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            policy_type="tdmpc",
            optimizer_lr=3e-4,
            optimizer_weight_decay=0.01,
            optimizer_grad_clip_norm=5.0,
        ),
        "/tmp/out",
    )

    assert _arg_value(cmd, "--policy.optimizer_lr") == "0.0003"
    assert "--policy.optimizer_weight_decay" not in cmd
    assert "--policy.optimizer_grad_clip_norm" not in cmd


def test_gaussian_actor_takes_no_optimizer_knobs() -> None:
    """gaussian_actor's preset is a MultiAdamConfig built from per-group
    settings; its config declares none of the three scalar knobs."""
    from makermodslab.train import TrainingRequest, build_training_command

    cmd = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            policy_type="gaussian_actor",
            optimizer_lr=1e-4,
            optimizer_weight_decay=0.01,
            optimizer_grad_clip_norm=5.0,
        ),
        "/tmp/out",
    )

    assert not any(tok.startswith("--policy.optimizer_") for tok in cmd)


def test_unknown_policy_type_falls_back_to_lr_only() -> None:
    """A policy type absent from the capability table (form addition ahead of
    the table, or an old persisted config) gets the conservative subset."""
    from makermodslab.train import TrainingRequest, build_training_command

    cmd = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            policy_type="some_future_policy",
            optimizer_lr=1e-4,
            optimizer_weight_decay=0.01,
            optimizer_grad_clip_norm=5.0,
        ),
        "/tmp/out",
    )

    assert _arg_value(cmd, "--policy.optimizer_lr") == "0.0001"
    assert "--policy.optimizer_weight_decay" not in cmd
    assert "--policy.optimizer_grad_clip_norm" not in cmd


@pytest.mark.parametrize("policy_type", ["act", "diffusion", "pi0", "smolvla", "tdmpc", "vqbet", "pi0_fast"])
def test_no_optimizer_namespace_flag_is_ever_emitted(policy_type: str) -> None:
    """The `--optimizer.*` namespace is dead under the training preset. Nothing
    the builder emits may land there — including the optimizer TYPE, which the
    policy preset fixes and the CLI cannot override."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="x",
        policy_type=policy_type,
        optimizer_type="sgd",
        optimizer_lr=1e-4,
        optimizer_weight_decay=0.01,
        optimizer_grad_clip_norm=5.0,
    )
    cmd = build_training_command(req, "/tmp/out")

    assert not any(tok.startswith("--optimizer.") for tok in cmd)
    assert "sgd" not in cmd
    # The preset is what makes --optimizer.* inert; assert it's actually on.
    assert _arg_value(cmd, "--use_policy_training_preset") == "true"


def test_resume_emits_no_optimizer_flags() -> None:
    """On resume lerobot SKIPS the preset overwrite (the `not self.resume`
    guard) and restores optimizer state from the checkpoint. Re-specifying the
    knobs would fight that, so neither namespace may appear."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="x",
        policy_type="smolvla",
        resume=True,
        config_path="/runs/abc/checkpoints/5000/pretrained_model/train_config.json",
        optimizer_type="sgd",
        optimizer_lr=1e-4,
        optimizer_weight_decay=0.01,
        optimizer_grad_clip_norm=5.0,
    )
    cmd = build_training_command(req, "/tmp/new")

    assert not any(tok.startswith("--optimizer.") for tok in cmd)
    assert not any(tok.startswith("--policy.optimizer_") for tok in cmd)


# ---------------------------------------------------------------------------
# HF Jobs timeout: parse helper + request-level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_seconds",
    [
        ("2h", 7200),
        ("45m", 2700),
        ("90s", 90),
        ("1d", 86400),
        ("3h30m", 12600),
        ("1.5h", 5400),
        ("  2H  ", 7200),  # trimmed + case-insensitive
    ],
)
def test_parse_hf_duration_accepts_valid_forms(value: str, expected_seconds: int) -> None:
    from makermodslab.train import parse_hf_duration

    assert parse_hf_duration(value) == expected_seconds


@pytest.mark.parametrize("value", ["", "   ", "2h30", "2x", "abc", "-1h", "0h", "0s", "h"])
def test_parse_hf_duration_rejects_bad_forms(value: str) -> None:
    from makermodslab.train import parse_hf_duration

    with pytest.raises(ValueError):
        parse_hf_duration(value)


def test_hf_job_timeout_defaults_to_none_and_round_trips() -> None:
    """Optional field: absent in old persisted config JSON loads as None, and
    a valid value survives a JSON round-trip."""
    from makermodslab.train import TrainingRequest

    assert TrainingRequest(dataset_repo_id="x").hf_job_timeout is None

    # Old JobRecord.config JSON (pre-field) has no hf_job_timeout key.
    legacy = TrainingRequest.model_validate({"dataset_repo_id": "x", "policy_type": "act"})
    assert legacy.hf_job_timeout is None

    req = TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")
    assert TrainingRequest.model_validate_json(req.model_dump_json()).hf_job_timeout == "3h30m"


@pytest.mark.parametrize(
    "value,stored",
    [("2h", "2h"), ("3h30m", "3h30m"), ("  45m  ", "45m"), (None, None), ("", None), ("   ", None)],
)
def test_hf_job_timeout_validator_accepts_and_normalises(value, stored) -> None:
    """Valid (or blank/None) inputs pass; the friendly form is kept (whitespace
    trimmed), NOT converted to seconds — the runner does that conversion."""
    from makermodslab.train import TrainingRequest

    req = TrainingRequest(dataset_repo_id="x", hf_job_timeout=value)
    assert req.hf_job_timeout == stored


@pytest.mark.parametrize("value", ["2h30", "2x", "banana", "0h", "-5m"])
def test_hf_job_timeout_validator_rejects_bad_forms(value: str) -> None:
    from pydantic import ValidationError

    from makermodslab.train import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest(dataset_repo_id="x", hf_job_timeout=value)


def test_hf_job_timeout_never_leaks_into_training_argv() -> None:
    """The timeout is a runner/platform concern; build_training_command must
    not emit it as a lerobot CLI flag (local runs ignore the field entirely)."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert not any("timeout" in tok.lower() for tok in cmd)
    assert "3h30m" not in cmd
