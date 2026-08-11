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

"""Training-specific helpers: the request schema and the LeRobot CLI builder.

The actual job lifecycle (subprocess management, registry, log streaming)
lives in app/jobs.py.
"""

import re

import torch
from pydantic import BaseModel, field_validator

from makermodslab.utils.config import REQUIRED_HUB_TAGS

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# HF-Jobs duration strings (the `timeout` HfApi.run_job accepts). The Hub API
# itself only understands a SINGLE unit suffix (e.g. "2h", "30m") or a bare
# integer of seconds — see huggingface_hub._jobs_api._create_job_spec, which
# does `float(timeout[:-1]) * factor[timeout[-1]]`. We accept a friendlier
# superset here (compound forms like "3h30m") and normalise to an int of
# seconds in parse_hf_duration, so the runner always hands run_job a plain
# seconds int rather than relying on the Hub's single-unit parser.
_DURATION_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# One-or-more <number><unit> segments and nothing else. Number may be integer
# or decimal (e.g. "1.5h"); units are s/m/h/d. Anchored so trailing junk like
# "2h30" (number without a unit) or "2x" is rejected.
_DURATION_FULL_RE = re.compile(r"^(?:\d+(?:\.\d+)?[smhd])+$")
_DURATION_SEGMENT_RE = re.compile(r"(\d+(?:\.\d+)?)([smhd])")


def parse_hf_duration(value: str) -> int:
    """Parse an HF-Jobs timeout string into a positive integer of seconds.

    Accepts single- or multi-segment durations: "2h", "45m", "3h30m", "90s",
    "1d", "1.5h". Raises ValueError with a user-facing message for anything
    that isn't a well-formed duration or resolves to <= 0 seconds. Kept
    separate from the pydantic model so the cloud runner can reuse the exact
    same normalisation when converting the request's value for run_job.
    """
    text = value.strip().lower()
    if not text or not _DURATION_FULL_RE.match(text):
        raise ValueError(
            f"Invalid job timeout {value!r}. Use a duration like '2h', '45m', "
            "'3h30m', '90s', or '1d' (units: s, m, h, d)."
        )
    total = 0.0
    for num, unit in _DURATION_SEGMENT_RE.findall(text):
        total += float(num) * _DURATION_UNIT_SECONDS[unit]
    seconds = int(round(total))
    if seconds <= 0:
        raise ValueError(f"Job timeout {value!r} must be greater than zero.")
    return seconds


def _policy_hub_flags() -> list[str]:
    """CLI flags making a pushed policy PUBLIC and carrying the required Hub tags.

    LeRobot exposes `--policy.private` and `--policy.tags` (draccus fields on
    PreTrainedConfig). `private false` maps to `create_repo(private=False)`;
    `--policy.tags '[...]'` is merged into the model-card metadata (unioned with
    lerobot's own robotics/lerobot/<model-type> defaults). Emitted only when the
    caller is actually pushing the policy. The tag list is passed as a single
    draccus-parsable argv token (no spaces).
    """
    tag_list = "[" + ",".join(REQUIRED_HUB_TAGS) + "]"
    return ["--policy.private", "false", "--policy.tags", tag_list]


# Which optimizer knobs each policy type actually exposes on ITS OWN config.
#
# Why this exists: the form always runs with `use_policy_training_preset true`,
# and lerobot's TrainPipelineConfig.validate() then REPLACES the whole optimizer
# with `active_cfg.get_optimizer_preset()` (configs/train.py:249-253) on every
# fresh run. So `--optimizer.*` flags are silently discarded — the only way to
# influence the optimizer is to set the POLICY fields the preset is built from
# (`ACTConfig.get_optimizer_preset()` -> `AdamWConfig(lr=self.optimizer_lr,
# weight_decay=self.optimizer_weight_decay)`), i.e. `--policy.optimizer_lr`.
#
# Gating is load-bearing, not cosmetic: draccus fails at CLI PARSE time when
# given a `--policy.<field>` the selected policy config doesn't declare, so
# emitting an unsupported knob kills the run before training starts. And the
# /policy-optimizer-defaults payload can NOT be used to derive this — both
# AdamWConfig and AdamConfig carry a `grad_clip_norm` field regardless of
# whether the policy exposes an `optimizer_grad_clip_norm` knob.
#
# Derived by dataclass introspection of the pinned lerobot (see the `lerobot`
# commit pin in pyproject.toml): for each policy in the form's
# POLICY_TYPE_OPTIONS, `dataclasses.fields(make_policy_config(<type>))`.
# Re-verify after a pin bump. Unknown/absent policy types fall back to lr-only.
#
# Notable: `gaussian_actor` exposes NONE of the three (its preset is a
# MultiAdamConfig built from per-group settings), so it gets an empty set.
_POLICY_OPTIMIZER_FIELDS: dict[str, frozenset[str]] = {
    "act": frozenset({"lr", "weight_decay"}),
    "diffusion": frozenset({"lr", "weight_decay"}),
    "pi0": frozenset({"lr", "weight_decay", "grad_clip_norm"}),
    "smolvla": frozenset({"lr", "weight_decay", "grad_clip_norm"}),
    "tdmpc": frozenset({"lr"}),
    "vqbet": frozenset({"lr", "weight_decay"}),
    "pi0_fast": frozenset({"lr", "weight_decay", "grad_clip_norm"}),
    # Same PreTrainedConfig shape as pi0 (verified by dataclass introspection
    # against the pinned lerobot: identical optimizer_lr/betas/eps/
    # weight_decay/grad_clip_norm/scheduler_* fields and defaults).
    "pi05": frozenset({"lr", "weight_decay", "grad_clip_norm"}),
    "gaussian_actor": frozenset(),
}

# Conservative fallback for a policy type not in the table (e.g. one added to
# the form ahead of this table, or an old persisted config). `optimizer_lr` is
# the one knob every action policy in the pin declares.
_DEFAULT_POLICY_OPTIMIZER_FIELDS = frozenset({"lr"})


def _resolve_device(device: str | None) -> str:
    """Resolve the requested training device to a concrete backend.

    lerobot's trainer runs under HuggingFace Accelerate, which auto-detects the
    hardware and ignores `policy.device` — except that `device == "cpu"` forces
    CPU. So functionally the UI choice is binary: auto-GPU vs force-CPU. We keep
    the logged config truthful by resolving "auto"/None to the platform's real
    device here.

    Explicit "cuda"/"mps"/"cpu" pass through unchanged (backward-compat with
    configs that persisted a concrete device before this collapse to auto/cpu).
    """
    if device in ("cuda", "mps", "cpu"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TrainingRequest(BaseModel):
    # Dataset configuration
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None

    # Policy configuration
    policy_type: str = "act"

    # Core training parameters
    steps: int = 10000
    batch_size: int = 8
    seed: int | None = 1000
    num_workers: int = 4

    # Logging and checkpointing
    # log_freq drives how often lerobot prints loss/lr (and thus the chart's
    # resolution — one point per log line). Lower = smoother curves but noisier
    # per-window averages and more log volume.
    log_freq: int = 50
    save_freq: int = 1000
    # lerobot 0.6.0 renamed the training CLI flag --eval_freq -> --env_eval_freq
    # (lerobot_train's argparse rejects --eval_freq with rc=2). Frontend never
    # sends this field, so the request contract is unchanged for clients.
    env_eval_freq: int = 0
    save_checkpoint: bool = True

    # Output configuration
    output_dir: str = "outputs/train"
    resume: bool = False
    # Set by the "Continue training" flow: the source run + checkpoint step to
    # resume from. The JobRegistry resolves these into `config_path` (lerobot
    # needs the checkpoint's train_config.json to reconstruct the run).
    resume_from_job_id: str | None = None
    resume_from_step: int | None = None
    # Set by the "Fine-tune" flow: start a FRESH run (fresh optimizer, step 0)
    # whose weights are initialized from an imported/existing checkpoint. Unlike
    # resume, this needs no optimizer/step state — weights-only is exactly the
    # point. The UI picks a source + step; JobRegistry resolves them into
    # `policy_pretrained_path` (the checkpoint's pretrained_model dir or Hub ref
    # that lerobot's --policy.pretrained_path accepts).
    finetune_from_job_id: str | None = None
    finetune_from_step: int | None = None
    policy_pretrained_path: str | None = None
    job_name: str | None = None
    # Cloud resume only. An HF Job is immutable once ended, so "resume a cloud
    # run" means launching a NEW job that continues from a prior run's Hub
    # checkpoint. Local resume points config_path at a host-local
    # train_config.json; that file doesn't exist in the container, so cloud
    # resume instead names the parent's Hub output repo + zero-padded step dir.
    # The HfCloudJobRunner tells its in-container wrapper to download
    # checkpoints/<step_dir>/ from this repo and reconstruct lerobot's output-dir
    # layout, then resumes with --config_path pointing at the container path.
    # Set by JobRegistry.start from resume_from_job_id/resume_from_step when the
    # resume source is a cloud run; never set for local runs.
    resume_from_hub_repo: str | None = None
    resume_from_hub_step: str | None = None
    # Cross-runner resume, local parent → cloud target (F7). The parent's
    # checkpoint is on THIS machine, so it has to be uploaded to the Hub before
    # a pod can read it. That upload is a deliberate act, not a side effect of
    # clicking Continue: the client must say yes here, and the registry refuses
    # the launch otherwise (naming the control that grants it). Ignored on every
    # other path — nothing is uploaded when the checkpoint is already on the Hub,
    # including a re-resume of a step a previous continuation already pushed.
    upload_resume_checkpoint: bool = False
    # Set by the registry (never by a client) when `resume_from_hub_repo` is the
    # staging repo above rather than a cloud parent's own output repo. It tells
    # the cloud runner not to adopt the source repo as this run's OUTPUT repo:
    # a cloud→cloud continuation shares its parent's repo on purpose (one
    # lineage, one place), but a local→cloud one must not publish into a staging
    # repo that exists only to carry the parent's bytes. Defaults False so
    # configs persisted before F7 — all of them cloud→cloud — keep their meaning.
    resume_from_uploaded_checkpoint: bool = False

    # Weights & Biases
    wandb_enable: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = "online"
    wandb_disable_artifact: bool = False

    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int = 10
    eval_batch_size: int = 50
    eval_use_async_envs: bool = False

    # Policy-specific
    policy_device: str | None = "auto"
    policy_use_amp: bool = False
    # Hub upload (set by HfCloudJobRunner; not exposed in the form)
    policy_push_to_hub: bool = False
    policy_repo_id: str | None = None

    # Optimizer
    optimizer_type: str | None = "adam"
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None

    # Advanced
    use_policy_training_preset: bool = True
    config_path: str | None = None

    # HF Jobs runner only. When set, overrides the platform-default job timeout
    # for CLOUD training (local runs ignore this field entirely — it never
    # reaches build_training_command's argv). Duration string in HF-Jobs format
    # ("2h", "45m", "3h30m", "1.5h"); validated below at request time and
    # normalised to an int of seconds by the cloud runner when it calls
    # run_job. None ⇒ the runner falls back to its HF_JOB_TIMEOUT default.
    # Optional with a default so JobRecord.config JSON persisted before this
    # field existed round-trips unchanged.
    hf_job_timeout: str | None = None

    @field_validator("hf_job_timeout")
    @classmethod
    def _validate_hf_job_timeout(cls, value: str | None) -> str | None:
        """Reject malformed timeout strings at request time with a clear
        message (pydantic wraps the ValueError into a 422 whose detail carries
        the guidance text). The friendly form the user typed (e.g. "3h30m") is
        kept on the model — the cloud runner calls parse_hf_duration to convert
        it to the seconds int run_job wants. None/blank passes through untouched
        so the runner applies its own default."""
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        # Raises ValueError with a user-facing message when malformed or <= 0.
        parse_hf_duration(text)
        return text


def _policy_optimizer_flags(request: "TrainingRequest") -> list[str]:
    """`--policy.optimizer_*` flags for the knobs this policy actually exposes.

    Fresh runs only — see `_POLICY_OPTIMIZER_FIELDS` above for why these replace
    the inert `--optimizer.*` flags, and why unsupported knobs must be dropped
    rather than passed through. A value the policy can't accept is silently
    skipped: the user's stored config keeps it (so switching back to a policy
    that does support it restores the value) but it never reaches argv.
    """
    supported = _POLICY_OPTIMIZER_FIELDS.get(request.policy_type, _DEFAULT_POLICY_OPTIMIZER_FIELDS)
    flags: list[str] = []
    for name, value in (
        ("lr", request.optimizer_lr),
        ("weight_decay", request.optimizer_weight_decay),
        ("grad_clip_norm", request.optimizer_grad_clip_norm),
    ):
        if value is not None and name in supported:
            flags.extend([f"--policy.optimizer_{name}", str(value)])
    return flags


def build_training_command(
    request: TrainingRequest, output_dir: str, python_executable: str = "python"
) -> list[str]:
    """Build the argv list to invoke `<python_executable> -m lerobot.scripts.lerobot_train`.

    `output_dir` is supplied separately from the request so the caller (the
    JobRegistry) can pin it to the per-job directory rather than relying on
    request.output_dir, which the frontend doesn't even send in the new world.

    `python_executable` defaults to "python" for the cloud runner (whose
    container has lerobot on PATH); the local runner must pass sys.executable
    so the subprocess uses the same interpreter as MakerMods Lab itself — otherwise
    PATH lookup picks up a different env (uv tool venv, miniforge3 base, etc.)
    that lacks lerobot.
    """
    cmd: list[str] = [python_executable, "-m", "lerobot.scripts.lerobot_train"]

    # Resume: lerobot reconstructs the whole run (policy, dataset, optimizer,
    # batch size, …) from the checkpoint's train_config.json, so we pass ONLY
    # the resume essentials plus the few top-level overrides that make sense to
    # change on continuation. Passing --policy.type / --dataset.* here would
    # fight the loaded config — lerobot's resume path expects those to come
    # from config_path, not the CLI. --steps must be raised above the resumed
    # step for the loop to do any work; --output_dir points new checkpoints at
    # this job's own dir so tracking stays consistent (state still loads from
    # the source checkpoint). --num_workers rides along too: it is a
    # host-capacity knob, not an experiment property, and a resume can land on a
    # different flavor than the parent run, so inheriting the checkpoint's
    # worker count would oversubscribe or starve the new host. (A bare int
    # decodes fine through the config_path parse path.)
    if request.resume and request.config_path:
        # lerobot pre-parses config_path with its own parser that ONLY accepts
        # the "--config_path=<path>" form (space-separated is silently ignored,
        # yielding "A config_path is expected when resuming a run").
        cmd.append(f"--config_path={request.config_path}")
        cmd.extend(["--resume", "true"])
        cmd.extend(["--output_dir", output_dir])
        cmd.extend(["--steps", str(request.steps)])
        cmd.extend(["--num_workers", str(request.num_workers)])
        cmd.extend(["--log_freq", str(request.log_freq)])
        cmd.extend(["--save_freq", str(request.save_freq)])
        cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])
        # Cloud resume keeps pushing into the SAME output repo as the parent run
        # so the whole lineage lives in one place. Local resume leaves these off
        # (push_to_hub defaults false), so the resume flags stay minimal for it.
        cmd.extend(["--policy.push_to_hub", "true" if request.policy_push_to_hub else "false"])
        if request.policy_push_to_hub and request.policy_repo_id:
            cmd.extend(["--policy.repo_id", request.policy_repo_id])
        if request.policy_push_to_hub:
            # Visibility only — NOT _policy_hub_flags(). On the resume path
            # lerobot parses via TrainPipelineConfig.from_pretrained(config_path,
            # cli_args=...) -> draccus.parse(cls, config_file, args=...), which
            # merges each CLI override into the config dict as a RAW STRING and
            # then decodes it against the field type. `--policy.tags
            # '[a,b,c]'` therefore reaches list[str] decoding as the literal
            # string "[a,b,c]" and dies with DecodingError. (The fresh-run
            # branch survives the same token because it goes through argparse,
            # which splits the bracket form itself.) Re-passing tags is also
            # redundant: the checkpoint's train_config.json already carries
            # policy.tags/private/repo_id/push_to_hub. See MT24.
            cmd.extend(["--policy.private", "false"])
        if request.job_name:
            cmd.extend(["--job_name", request.job_name])
        return cmd

    # Dataset
    cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
    if request.dataset_revision:
        cmd.extend(["--dataset.revision", request.dataset_revision])
    if request.dataset_root:
        cmd.extend(["--dataset.root", request.dataset_root])
    if request.dataset_episodes:
        cmd.extend(["--dataset.episodes"] + [str(ep) for ep in request.dataset_episodes])

    # Policy
    cmd.extend(["--policy.type", request.policy_type])
    # Fine-tune: initialize weights (+ processors) from an existing checkpoint.
    # This is a normal FRESH run (--resume false below) — lerobot loads only the
    # weights via pretrained_path, starting a new optimizer at step 0. Equals
    # form (like config_path) to be safe against value parsing. Never emitted on
    # the resume branch above.
    if request.policy_pretrained_path:
        cmd.append(f"--policy.pretrained_path={request.policy_pretrained_path}")

    # Core training params
    cmd.extend(["--steps", str(request.steps)])
    cmd.extend(["--batch_size", str(request.batch_size)])
    cmd.extend(["--num_workers", str(request.num_workers)])
    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    # Policy device / AMP / hub
    resolved = _resolve_device(request.policy_device)
    cmd.extend(["--policy.device", resolved])
    cmd.extend(["--policy.use_amp", "true" if request.policy_use_amp else "false"])
    # LeRobot defaults push_to_hub=True and demands --policy.repo_id when so.
    # Local jobs keep it off; HF Cloud jobs flip it on via the runner.
    cmd.extend(["--policy.push_to_hub", "true" if request.policy_push_to_hub else "false"])
    if request.policy_push_to_hub and request.policy_repo_id:
        cmd.extend(["--policy.repo_id", request.policy_repo_id])
    if request.policy_push_to_hub:
        # Public + required Hub tags, same global default as datasets.
        cmd.extend(_policy_hub_flags())

    # Logging / checkpointing
    cmd.extend(["--log_freq", str(request.log_freq)])
    cmd.extend(["--save_freq", str(request.save_freq)])
    cmd.extend(["--env_eval_freq", str(request.env_eval_freq)])
    cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

    # Output
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--resume", "true" if request.resume else "false"])
    if request.job_name:
        cmd.extend(["--job_name", request.job_name])

    # W&B
    cmd.extend(["--wandb.enable", "true" if request.wandb_enable else "false"])
    if request.wandb_enable:
        if request.wandb_project:
            cmd.extend(["--wandb.project", request.wandb_project])
        if request.wandb_entity:
            cmd.extend(["--wandb.entity", request.wandb_entity])
        if request.wandb_notes:
            cmd.extend(["--wandb.notes", request.wandb_notes])
        if request.wandb_run_id:
            cmd.extend(["--wandb.run_id", request.wandb_run_id])
        if request.wandb_mode:
            cmd.extend(["--wandb.mode", request.wandb_mode])
        cmd.extend(["--wandb.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

    # Env
    if request.env_type:
        cmd.extend(["--env.type", request.env_type])
    if request.env_task:
        cmd.extend(["--env.task", request.env_task])

    # Eval
    cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
    cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
    cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

    # Optimizer. Routed through the POLICY config, never `--optimizer.*`: with
    # the training preset on (always, below) lerobot overwrites the whole
    # optimizer from `active_cfg.get_optimizer_preset()`, so every `--optimizer.*`
    # flag is discarded before the first step. `request.optimizer_type` is
    # deliberately NOT emitted — the optimizer CLASS is fixed by the policy's
    # preset (AdamW for act/smolvla/pi0/pi0_fast, Adam for diffusion/vqbet/tdmpc)
    # and is not overridable; the field survives on the request only because
    # persisted job configs and the resume prefill read it. See MT43.
    cmd.extend(_policy_optimizer_flags(request))

    # Advanced
    cmd.extend(["--use_policy_training_preset", "true" if request.use_policy_training_preset else "false"])
    if request.config_path:
        # Equals form required — see the resume branch above.
        cmd.append(f"--config_path={request.config_path}")

    return cmd
