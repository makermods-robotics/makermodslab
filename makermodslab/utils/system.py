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

import contextlib
import importlib.util
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def open_folder_in_file_browser(path: str) -> None:
    """Open `path` in the OS file browser (Finder/Explorer/xdg-open).

    Creates the directory (parents included) first so a fresh install with no
    calibration files yet still opens a real, empty folder instead of failing.
    Does NOT block on the launched process — the file browser runs detached.
    This is a LOCAL action only (spawns a GUI on the host); it never touches the
    network. Raises on an unsupported platform or a spawn failure so the caller
    can report a clear error.
    """
    os.makedirs(path, exist_ok=True)
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", path])
    elif system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]  # Windows-only  # nosec B606 — opens a folder we just created, no shell, no user input
    elif system == "Linux":
        subprocess.Popen(["xdg-open", path])
    else:
        raise OSError(f"Opening a folder is not supported on this platform: {system!r}")


# accelerate and wandb are only ever consumed by training SUBPROCESSES (fresh
# processes that see new installs immediately) — nothing needs them imported
# into this server process. So availability is probed live per request via
# ``_extra_available`` (mirroring ``handle_get_policy_extra``), never cached at
# import. A just-installed package becomes available without a server restart.
TRAINING_PROBE_MODULE: str = "accelerate"
TRAINING_INSTALL_HINT: str = "pip install accelerate"

WANDB_PROBE_MODULE: str = "wandb"
WANDB_INSTALL_HINT: str = "pip install wandb"


def _extra_available(module: str) -> bool:
    """Whether ``module`` is importable right now (probed live, not cached)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Where the standard installers put uv when it is NOT on this process's PATH —
# a headless server started over ssh/nohup gets a minimal PATH without
# ~/.local/bin, and the whole point of the uv branch below is that a uv venv
# has no pip to fall back to (field-debugged on a remote node whose installs
# all died with "pip exited with code 1"). Same disease, same cure as the
# macOS tailscale CLI lookup in node_sources.py.
_UV_FALLBACK_PATHS = (
    os.path.expanduser("~/.local/bin/uv"),  # the uv installer's default target
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
)


def _find_uv() -> str | None:
    """The uv binary to run: PATH first, then the standard install locations."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in _UV_FALLBACK_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _build_install_cmd(package: str) -> list[str]:
    """Pick the best installer for the running Python.

    Venvs created with `uv venv` don't ship pip, so `python -m pip` fails with
    `No module named pip`. Find uv (PATH, then the standard install
    locations) and use it with --python pinned to sys.executable so the
    install lands in this Python's site-packages. Otherwise fall back to
    `python -m pip`.
    """
    uv = _find_uv()
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, package]
    return [sys.executable, "-m", "pip", "install", package]


class ExtraStatus(BaseModel):
    available: bool
    install_hint: str


class CudaStatus(BaseModel):
    gpu_present: bool
    cuda_available: bool
    mismatch: bool
    torch_version: str | None = None
    install_hint: str
    docs_url: str


class InstallStartResponse(BaseModel):
    started: bool
    message: str


class InstallStatusResponse(BaseModel):
    state: str  # "idle" | "installing" | "done" | "error"
    error: str | None = None
    logs: list[dict[str, Any]] = []


class InstallManager:
    def __init__(self, package: str) -> None:
        self.package = package
        self.state: str = "idle"
        self.error: str | None = None
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.state == "installing":
                return {"started": False, "message": "Install already in progress"}
            # Reset for a fresh attempt (covers retry from done/error/idle).
            self.state = "installing"
            self.error = None
            self._drain_queue()

        try:
            self.process = subprocess.Popen(
                _build_install_cmd(self.package),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.exception("Failed to spawn pip subprocess")
            with self._lock:
                self.state = "error"
                self.error = f"Failed to spawn pip: {exc}"
            return {"started": False, "message": str(exc)}

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return {"started": True, "message": "Install started"}

    def get_status(self) -> dict[str, Any]:
        logs: list[dict[str, Any]] = []
        try:
            while not self.log_queue.empty():
                logs.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return {"state": self.state, "error": self.error, "logs": logs}

    def _monitor(self) -> None:
        assert self.process is not None
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self._enqueue(line.rstrip())
        except Exception as exc:
            logger.exception("Error reading pip output")
            self._enqueue(f"[install-monitor] error reading output: {exc}")

        self.process.wait()
        return_code = self.process.returncode
        if return_code == 0:
            # The subprocess just wrote a new entry into this process's
            # site-packages. Python's import system caches directory listings
            # (FileFinder), so a long-running server can miss the newly created
            # package on the next find_spec. Invalidate those caches so the very
            # next availability poll (handle_get_*_extra) finds it — no restart.
            importlib.invalidate_caches()
        with self._lock:
            if return_code == 0:
                self.state = "done"
                self.error = None
            else:
                self.state = "error"
                self.error = f"pip exited with code {return_code}"

    def _enqueue(self, message: str) -> None:
        # Cap queue size so a chatty pip can't grow memory unbounded.
        if self.log_queue.qsize() >= 1000:
            with contextlib.suppress(queue.Empty):
                self.log_queue.get_nowait()
        self.log_queue.put({"timestamp": time.time(), "message": message})

    def _drain_queue(self) -> None:
        try:
            while not self.log_queue.empty():
                self.log_queue.get_nowait()
        except queue.Empty:
            pass


training_install_manager = InstallManager("accelerate")
wandb_install_manager = InstallManager("wandb")


def handle_get_training_extra() -> dict[str, Any]:
    return {
        "available": _extra_available(TRAINING_PROBE_MODULE),
        "install_hint": TRAINING_INSTALL_HINT,
    }


def handle_install_training_extra() -> dict[str, Any]:
    return training_install_manager.start()


def handle_install_training_extra_status() -> dict[str, Any]:
    return training_install_manager.get_status()


def handle_get_wandb_extra() -> dict[str, Any]:
    return {
        "available": _extra_available(WANDB_PROBE_MODULE),
        "install_hint": WANDB_INSTALL_HINT,
    }


def handle_install_wandb_extra() -> dict[str, Any]:
    return wandb_install_manager.start()


def handle_install_wandb_extra_status() -> dict[str, Any]:
    return wandb_install_manager.get_status()


# --------------------------------------------------------------------------- #
# Policy extras
# --------------------------------------------------------------------------- #
# Some LeRobot policies import an optional extra at construction time; training
# (or inference) otherwise dies with a buried ImportError once the subprocess is
# already running. Map each such policy to the module we probe and the
# ``pip install lerobot[extra]`` target. Policies not listed (act, vqbet, tdmpc,
# gaussian_actor, reward_classifier) need nothing extra.
POLICY_EXTRAS: dict[str, tuple[str, str]] = {
    # policy_type: (probe_module, install_target)
    "smolvla": ("transformers", "lerobot[smolvla]"),
    "pi0": ("transformers", "lerobot[pi]"),
    "pi0_fast": ("transformers", "lerobot[pi]"),
    "pi05": ("transformers", "lerobot[pi]"),
    "diffusion": ("diffusers", "lerobot[diffusion]"),
    # MolmoAct2's `lerobot[molmoact2]` extra is transformers + peft + scipy, but
    # only TRANSFORMERS is a construction-time requirement for a rollout on this
    # pin, so that is what we probe:
    #   * peft is reached only from `_apply_lora_adapters`, i.e. when
    #     `enable_lora_vlm` is set — a training-side option;
    #   * scipy is the discrete action tokenizer, required by
    #     `MolmoAct2PackInputsProcessorStep.__post_init__` only when the
    #     checkpoint's `action_mode` is "discrete" or "both". The released
    #     `lerobot/MolmoAct2-*-LeRobot` checkpoints save `action_mode:
    #     "continuous"`, which is the one value that skips it.
    # Probing either would report needs_extra && !available for a checkpoint
    # that runs fine, and DeployPanel REFUSES to launch on that — a false
    # blocker is worse than a buried ImportError. Revisit when MolmoAct2
    # training lands, or if a "both"/"discrete" checkpoint ships.
    "molmoact2": ("transformers", "lerobot[molmoact2]"),
}


# --------------------------------------------------------------------------- #
# Policy runtime requirements
# --------------------------------------------------------------------------- #
# Sibling of POLICY_EXTRAS above, and the same shape of question: not "which
# package does this policy need installed" but "which flags does this lerobot
# pin REQUIRE to run this policy at all". Kept here rather than in rollout.py
# because three front-ends will want the same answer — the rollout/eval/coaching
# subprocess builders today, a MolmoAct2 training config later, and the remote
# inference policy server after that.
#
# Everything below is a pure function of the checkpoint's own saved config.json
# (the dict lerobot writes next to the weights), so it is trivially testable and
# has no import-time cost.

MOLMOACT2 = "molmoact2"

# `--policy.*` overrides are applied by lerobot ON TOP of the checkpoint's saved
# config (RolloutConfig.__post_init__ → PreTrainedConfig.from_pretrained(
# cli_overrides=...)), which is why filling a gap here is enough and why
# overriding a value the checkpoint deliberately saved would be wrong.


def molmoact2_inference_action_mode(policy_config: Mapping[str, Any]) -> str:
    """The action mode a MolmoAct2 rollout will actually run in.

    `MolmoAct2Config.inference_action_mode` defaults to None and
    `_resolve_inference_action_mode` raises on None — "MolmoAct2 inference
    requires `inference_action_mode` to be set explicitly" — so a checkpoint
    that never saved one cannot be rolled out at all without an override.

    The released `lerobot/MolmoAct2-SO100_101-LeRobot` config DOES save
    ``"inference_action_mode": "continuous"``, so this only fills a gap: a
    locally fine-tuned checkpoint, or a raw `allenai/MolmoAct2` repo. The
    default mirrors the checkpoint's TRAINING mode, because forcing continuous
    onto an `action_mode="discrete"` checkpoint raises in `__post_init__` —
    which would turn "no mode saved" into a different, equally fatal error.
    """
    saved = policy_config.get("inference_action_mode")
    if saved in ("continuous", "discrete"):
        return str(saved)
    return "discrete" if policy_config.get("action_mode") == "discrete" else "continuous"


def policy_inference_args(policy_config: Mapping[str, Any]) -> list[str]:
    """Extra ``--policy.*`` rollout flags this pin requires for a checkpoint.

    Keyed on the checkpoint's own ``type``; empty for every policy that needs
    nothing (act, smolvla, pi0, …), which is all of them but MolmoAct2 today.
    Deliberately emits NOTHING when the checkpoint already saved an explicit
    choice — a rollout must run the policy the way its config says.
    """
    if policy_config.get("type") != MOLMOACT2:
        return []
    if policy_config.get("inference_action_mode") in ("continuous", "discrete"):
        return []
    return [f"--policy.inference_action_mode={molmoact2_inference_action_mode(policy_config)}"]


def molmoact2_rtc_conflict(policy_config: Mapping[str, Any]) -> str | None:
    """Why RTC can't drive this MolmoAct2 checkpoint, or None when it can.

    `MolmoAct2Policy.supports_rtc()` is literally
    ``inference_action_mode == "continuous"``, and `build_rollout_context`
    turns a False into a ValueError — but only AFTER loading a multi-GB VLM
    onto the accelerator. Answering from the saved config costs nothing and
    fails in the launch panel instead. Only meaningful for MolmoAct2: every
    other policy's RTC support is decided elsewhere (upstream's
    `supports_rtc_inference`), and this returns None for them rather than
    pretending to know.
    """
    if policy_config.get("type") != MOLMOACT2:
        return None
    if molmoact2_inference_action_mode(policy_config) == "continuous":
        return None
    return (
        "Real-Time Chunking needs continuous actions, and this MolmoAct2 checkpoint runs "
        "discrete ones. Use the standard (sync) inference engine for it."
    )


def molmoact2_device_warning(policy_config: Mapping[str, Any], device: str) -> str | None:
    """Why a MolmoAct2 rollout on `device` is likely to disappoint, or None.

    A WARNING and deliberately not a refusal. Nothing in this lerobot pin
    requires CUDA: the only CUDA-specific code is the action-flow graph cache,
    and `ActionCudaGraphManager.can_use_action_flow` returns False off-CUDA and
    falls back to the eager loop. So refusing would be MakerMods Lab inventing
    a hardware requirement lerobot does not state — the honest thing is to say
    what the operator is in for and let them decide.

    What they are in for is the model's size, not a missing kernel: MolmoAct2
    is a ~7B vision-language model queried inside a 30 Hz control loop.

    `device` is injected (rollout passes `_detect_device()`) so this stays a
    pure function and the tests never touch a real accelerator.
    """
    if policy_config.get("type") != MOLMOACT2:
        return None
    if device == "cuda":
        return None
    return (
        "MolmoAct2 is a ~7B vision-language model and this machine has no CUDA GPU "
        f"(running on {device}). Expect very slow inference, or an out-of-memory failure. "
        "Run it on a machine with an NVIDIA GPU for a usable control rate."
    )


# Policy types whose forward pass is conditioned on a natural-language task
# string. THE single source of truth: jobs.py's `requires_task` (what the launch
# UI gates its task field on) and the two DRTC policy servers' startup refusal
# both read it, so the Lab and the GPU can never disagree about whether a run
# needs a task.
#
# Why a hard requirement and not a nudge: with no task the policies here do not
# fail, they DEGRADE SILENTLY. MolmoAct2 is the worst of them — its processor
# renders `None` as the empty string into a fixed template, so the VLM is
# prompted with the literal "The task is to ." and produces confidently wrong
# actions with nothing in any log to say why. Refusing at startup costs an
# operator one flag; not refusing costs them a session and a diagnosis.
#
# Mirrored by hand from the pinned fork's modeling_<policy>.py, alongside
# POLICY_EXTRAS above — update both on a pin bump.
LANGUAGE_CONDITIONED_POLICY_TYPES = frozenset({"smolvla", "pi0", "pi0_fast", "pi05", MOLMOACT2})


def policy_requires_task(policy_type: object) -> bool:
    """Whether a checkpoint of this architecture needs a ``--task`` string.

    Takes ``object`` rather than ``str`` because both callers hand it a value
    read straight out of a checkpoint's config.json, where the key can be
    missing (None) or, in a corrupt file, some other type entirely. An
    unreadable type is NOT language-conditioned: a False here only means the
    task field stays optional, and a policy that really wanted one still says
    so from inside the subprocess.
    """
    return isinstance(policy_type, str) and policy_type in LANGUAGE_CONDITIONED_POLICY_TYPES


# --------------------------------------------------------------------------- #
# Flow-matching / denoising steps per chunk, by policy family (S3.8f)
# --------------------------------------------------------------------------- #
# WHICH config field decides how many integration steps the action sampler takes
# for ONE chunk. It is the cheapest latency lever a VLA has — the action expert
# is re-run once per step, so halving the count roughly halves the GPU term of
# the round trip — and every family spells it differently.
#
# Verified against the pinned fork's own source, not from memory:
#
#   * smolvla  → ``num_steps`` (configuration_smolvla.py, default 10).
#   * pi0      → ``num_inference_steps`` (configuration_pi0.py, default 10).
#   * pi05     → ``num_inference_steps`` (configuration_pi05.py, default 10).
#   * molmoact2→ ``num_inference_steps`` (configuration_molmoact2.py, default
#     **None**), and NOT ``num_flow_timesteps``. This one is worth spelling out
#     because the names invite the opposite guess: `num_flow_timesteps` (8) is a
#     TRAINING knob — how many flow timesteps are sampled per example to build
#     the loss (`_prepare_flow_matching_tensors`, and the joint-flow loss) — and
#     is never read on an inference path. `predict_action_chunk` reads
#     ``kwargs.get("num_steps", config.num_inference_steps)`` and passes it down
#     to `generate_actions_from_inputs`, which resolves ``num_steps or
#     self.config.flow_matching_num_steps`` against the BACKBONE's HF config
#     (default 10). So a MolmoAct2 checkpoint that saved nothing is really
#     taking 10 steps, that 10 is not readable from the checkpoint's own
#     config.json, and setting `num_inference_steps` is the only override that
#     reaches the sampler. The RTC path resolves it identically.
#
# `pi0_fast` is deliberately absent: it decodes action TOKENS autoregressively
# and has no denoising loop, so there is no field to point at and the knob is
# correctly reported as inapplicable.
#
# Mirrored by hand from the pin, alongside POLICY_EXTRAS and
# LANGUAGE_CONDITIONED_POLICY_TYPES — update all three on a pin bump.
POLICY_FLOW_STEPS_FIELDS: dict[str, str] = {
    "smolvla": "num_steps",
    "pi0": "num_inference_steps",
    "pi05": "num_inference_steps",
    MOLMOACT2: "num_inference_steps",
}

# What MolmoAct2 actually samples with when its own config says nothing, which
# is what the PUBLISHED checkpoint says (`num_inference_steps: null`). The
# number is NOT in the checkpoint: `modeling_molmoact2.py` resolves
# `steps = int(num_steps or self.config.flow_matching_num_steps)` against the
# BACKBONE's HF config, whose `flow_matching_num_steps` defaults to 10
# (`molmoact2_hf_model/configuration_molmoact2.py`), and nothing on the rollout
# path passes `num_steps`. Stated here rather than left as "unknown" because
# "unknown" is what made the panel offer 8 — `num_flow_timesteps` (default 8) is
# a TRAINING knob (how many flow timesteps each example is sampled at to build
# the loss) and is never read at inference. Re-verify on a lerobot pin bump: it
# is the pin's default, not ours.
MOLMOACT2_FLOW_STEPS_DEFAULT = 10


def policy_flow_steps_field(policy_type: object) -> str | None:
    """The config field a flow-steps override has to write for this policy.

    ``object`` rather than ``str`` for the same reason `policy_requires_task`
    takes one: every caller reads the type straight out of a checkpoint's
    config.json, where it can be missing or, in a corrupt file, anything at
    all. None means "this architecture has no such knob" — which is the answer
    that makes the caller DROP the override rather than send a flag that dies
    in the container after a paid cold start.
    """
    if not isinstance(policy_type, str):
        return None
    return POLICY_FLOW_STEPS_FIELDS.get(policy_type)


def policy_supports_model_dtype(policy_config: Mapping[str, Any]) -> bool:
    """Whether ``--model-dtype`` has anything to write on this checkpoint.

    Answered from the SAVED config rather than from a table of policy types,
    because a saved config.json is a dataclass dump: every field the config
    class declares is a key in it, so key presence IS "the class has this
    field" — and it stays right through a pin bump that gives another family
    the knob. In this pin MolmoAct2 is the only one (`model_dtype: "bfloat16"`).

    One function so the two places that ask cannot drift: the policy-config
    route (which disables the panel's select) and `modal_launcher` (which drops
    the flag before spending a cold start on a `SystemExit`).
    """
    return "model_dtype" in policy_config


def policy_flow_steps_default(policy_config: Mapping[str, Any]) -> int | None:
    """The step count this checkpoint would run with, or None when unknown.

    Read off the checkpoint's own saved config, so it is what the panel shows
    beside "Checkpoint default".

    ONE exception, and it is the family the knob is for: a MolmoAct2 whose
    ``num_inference_steps`` is absent or null answers
    ``MOLMOACT2_FLOW_STEPS_DEFAULT`` (10) rather than None, because the number
    that applies is a documented constant in the pin's own backbone config, not
    an unknown — see that constant for the citation. A saved value still wins;
    this is the fallback the container itself takes.

    Non-positive and non-integral values answer None (`bool` explicitly, since
    it is an `int` subclass): every policy config validates this itself at
    construction, so anything else means a hand-edited config, and the honest
    answer downstream is "unknown" rather than a number somebody sets a latency
    budget from. That includes a hand-edited MolmoAct2 — only null (or absent)
    takes the documented fallback.
    """
    policy_type = policy_config.get("type")
    field = policy_flow_steps_field(policy_type)
    if field is None:
        return None
    value = policy_config.get(field)
    if value is None and policy_type == MOLMOACT2:
        return MOLMOACT2_FLOW_STEPS_DEFAULT
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


# Policy families whose IMAGE VIEW COUNT is a property of the checkpoint's
# wrapper rather than of the architecture — the only ones an extra camera role
# may be added to (S3.8g).
#
# MolmoAct2 is the case that motivated it, and the reasoning is specific rather
# than a general "VLAs take any number of pictures". The allenai model takes a
# LIST of images and was fine-tuned across community datasets carrying one to
# three cameras; lerobot's wrapper
# (`lerobot/MolmoAct2-SO100_101-LeRobot`) simply FIXED that list at two by
# declaring `observation.images.cam0` / `cam1` in `input_features`. Nothing
# downstream of that declaration counts: verified against the pin's own
# `policies/molmoact2/processor_molmoact2.py`, the pack step builds its image
# list by iterating whatever image keys it resolves (`_extract_images`), and the
# prompt and the sequence budget are both computed from `len(images)`
# (`_build_prompt`'s `Image {i}<|image|>` join, `image_tokens = num_images *
# 196`). So a third declared feature is a third picture, not a crash.
#
# CLOSED and small on purpose. Adding a view to a policy whose vision tower
# really is fixed at N is not a degraded run, it is a shape error inside the
# container after a paid cold start — and worse, some architectures would accept
# the feature and silently ignore it. A family goes in here only after someone
# has read ITS processor and confirmed the same three things.
#
# Mirrored by the container (`drtc/policy.py`, `drtc/policy_rtc.py`), the
# launcher (which drops the knob for a checkpoint that is not in it) and
# `remote_inference` (which refuses an unknown camera role for one), so all
# three answer identically. Update on a lerobot pin bump.
VARIABLE_VIEW_POLICY_TYPES: frozenset[str] = frozenset({MOLMOACT2})

# How many extra roles a single launch may add. Not a model limit — it is a
# LATENCY one, and it is deliberately conservative: each view is another 196
# image tokens through the prefill of a ~7B VLM, on a round trip an rtc run has
# 867 ms to spend. Two is enough for "the two the checkpoint declares plus one
# or two more"; a bench that wants five should measure first and raise it here.
MAX_EXTRA_IMAGE_ROLES = 2

# A camera role, as `observation.images.<role>` will spell it. Deliberately
# narrower than "any string": the role becomes a policy feature key, a Portal
# VIDEO TRACK NAME, a `--robot.cameras` dict key inside a draccus-parsed argv,
# and a `--extra-image-roles` element in a COMMA-separated flag. A comma, a
# space, a brace or a dot in it breaks one of those four in a place that
# presents as "the session receives nothing" rather than as an error.
_IMAGE_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def is_valid_image_role(name: object) -> bool:
    """Whether ``name`` is a camera role this stack will carry end to end.

    ``object`` rather than ``str`` for the same reason `policy_flow_steps_field`
    takes one: the callers read these off a request body and a comma-split CLI
    string, either of which can hand over something that is not a string at all.
    """
    return isinstance(name, str) and bool(_IMAGE_ROLE_RE.match(name))


def policy_supports_extra_image_roles(policy_type: object) -> bool:
    """Whether extra image roles may be added to this policy type (S3.8g).

    The single reading of :data:`VARIABLE_VIEW_POLICY_TYPES`, so the container,
    the launcher and the robot side cannot disagree about which checkpoints take
    a third camera. False for a type this pin has never heard of, which is the
    safe direction: an unknown architecture gets the checkpoint's own views, and
    the operator gets a refusal instead of a shape error ninety seconds into a
    cold start.
    """
    return isinstance(policy_type, str) and policy_type in VARIABLE_VIEW_POLICY_TYPES


# One install manager per install target (lerobot[smolvla] / lerobot[pi] / …),
# created lazily so pi0 and pi0_fast share the lerobot[pi] install.
_policy_install_managers: dict[str, InstallManager] = {}


def _policy_install_manager(policy_type: str) -> InstallManager | None:
    spec = POLICY_EXTRAS.get(policy_type)
    if spec is None:
        return None
    target = spec[1]
    mgr = _policy_install_managers.get(target)
    if mgr is None:
        mgr = InstallManager(target)
        _policy_install_managers[target] = mgr
    return mgr


def handle_get_policy_extra(policy_type: str) -> dict[str, Any]:
    """Whether the optional extra a policy needs is importable right now.

    Probed live (not cached at import) so a restart after installing is picked
    up. Policies that need nothing report ``available`` so the UI never blocks
    them.
    """
    spec = POLICY_EXTRAS.get(policy_type)
    if spec is None:
        return {
            "policy_type": policy_type,
            "needs_extra": False,
            "available": True,
            "package": "",
            "install_target": "",
            "install_hint": "",
        }
    probe, target = spec
    try:
        available = importlib.util.find_spec(probe) is not None
    except (ImportError, ValueError):
        available = False
    return {
        "policy_type": policy_type,
        "needs_extra": True,
        "available": available,
        "package": probe,
        "install_target": target,
        "install_hint": f"pip install '{target}'",
    }


def handle_install_policy_extra(policy_type: str) -> dict[str, Any]:
    mgr = _policy_install_manager(policy_type)
    if mgr is None:
        return {"started": False, "message": f"'{policy_type}' needs no extra package."}
    return mgr.start()


def handle_install_policy_extra_status(policy_type: str) -> dict[str, Any]:
    mgr = _policy_install_manager(policy_type)
    if mgr is None:
        return {"state": "done", "error": None, "logs": []}
    return mgr.get_status()


# --------------------------------------------------------------------------- #
# Self-restart
# --------------------------------------------------------------------------- #
# POST /api/v1/system/restart re-execs this process in place so a remote
# operator (the node proxies) can bounce a headless station without a shell on
# it. os.execv keeps PID, argv, env, cwd and the std FDs (a nohup log redirect
# survives), while every other FD — the uvicorn listen socket included — closes
# on exec (PEP 446), so the relaunched process binds the port cleanly.

RESTART_DELAY_S = 1.0
# The launcher entry points (pyproject [project.scripts]) — the only argv[0]s
# we KNOW re-run the launcher when re-executed.
_RESTART_ENTRY_POINTS = ("makermodslab", "makermodslab-station")


def install_in_progress() -> str | None:
    """The package a live InstallManager is installing right now, or None.

    A restart guard: re-exec would orphan the pip subprocess mid-write and
    leave a half-installed site-packages, so the restart route refuses while
    any install (training/wandb/policy extras) is running.
    """
    managers = [training_install_manager, wandb_install_manager, *_policy_install_managers.values()]
    for mgr in managers:
        if mgr.state == "installing":
            return mgr.package
    return None


def restart_supported() -> tuple[bool, str]:
    """Whether this process can safely re-exec itself; (False, why) if not.

    Only a POSIX process whose argv[0] is one of our launcher entry points
    qualifies: there, ``execv(sys.executable, [sys.executable, *sys.argv])``
    re-runs the launcher with identical arguments. A dev-mode reload worker
    (uvicorn --reload spawns it via multiprocessing; argv is not the
    launcher) must never execv — uvicorn's reloader already restarts it on
    any code change. Windows is excluded: its execv spawns a NEW process and
    returns in the caller, which would leave two servers fighting over the
    port.
    """
    if os.name != "posix":
        return False, "restart-in-place is not supported on this platform"
    argv0 = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if argv0 not in _RESTART_ENTRY_POINTS:
        return False, (
            f"this process was not started by a MakerMods Lab entry point (argv[0]={argv0!r}) — "
            "restart it the way it was started (dev mode auto-reloads on code changes)"
        )
    return True, ""


def schedule_restart(delay_s: float = RESTART_DELAY_S, execv=os.execv) -> threading.Thread:
    """Re-exec this process after ``delay_s`` (from a daemon thread).

    The delay lets the HTTP response that announced the restart actually
    reach the client before the connection dies with the process. ``execv``
    is injectable for tests — the real one never returns. Returns the thread
    so a test can join an injected no-op execv.
    """

    def _restart() -> None:
        time.sleep(delay_s)
        logger.info("🔄 Restarting: re-exec %s %s", sys.executable, " ".join(sys.argv))
        execv(sys.executable, [sys.executable, *sys.argv])

    thread = threading.Thread(target=_restart, name="self-restart", daemon=True)
    thread.start()
    return thread


class RestartResponse(BaseModel):
    restarting: bool
    message: str


# Detect the common Windows/MakerMods Lab mismatch where an NVIDIA GPU is visible to the
# OS, but the active PyTorch build cannot use CUDA. Do not auto-install torch.

CUDA_TORCH_DOCS_URL = "https://pytorch.org/get-started/locally/"
CUDA_TORCH_INSTALL_HINT = (
    "To use the GPU, install a CUDA build of PyTorch. Pick your CUDA version at "
    f"{CUDA_TORCH_DOCS_URL} "
    "(for example: pip install torch --index-url https://download.pytorch.org/whl/cu124), "
    "then restart MakerMods Lab."
)


def _nvidia_gpu_present() -> bool:
    """True if an NVIDIA GPU is visible to the OS (``nvidia-smi -L`` lists one).

    Dependency-free and cheap: requires nvidia-smi on PATH, then confirms it
    actually reports a GPU. Any failure (no driver, no GPU, timeout) → False.
    """
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().startswith("GPU")


def _torch_cuda() -> tuple[bool, str | None]:
    """Return (cuda_available, torch_version). Missing/broken torch → (False, None)."""
    try:
        import torch
    except Exception:  # torch absent or import error — treat as no CUDA
        logger.debug("torch import failed during CUDA check", exc_info=True)
        return False, None
    try:
        return bool(torch.cuda.is_available()), torch.__version__
    except Exception:
        logger.debug("torch.cuda.is_available() raised", exc_info=True)
        return False, getattr(torch, "__version__", None)


def detect_cuda_status() -> dict[str, Any]:
    """Detect the 'NVIDIA GPU present but PyTorch is CPU-only' mismatch (issue #30)."""
    gpu_present = _nvidia_gpu_present()
    cuda_available, torch_version = _torch_cuda()
    return {
        "gpu_present": gpu_present,
        "cuda_available": cuda_available,
        "mismatch": gpu_present and not cuda_available,
        "torch_version": torch_version,
        "install_hint": CUDA_TORCH_INSTALL_HINT,
        "docs_url": CUDA_TORCH_DOCS_URL,
    }


def warn_if_cuda_mismatch() -> None:
    """Log a prominent warning when a GPU is present but torch is CPU-only.

    Called at server startup so the user sees actionable guidance in the same
    terminal where LeRobot's easily-missed 'Switching to cpu' line appears.
    """
    status = detect_cuda_status()
    if not status["mismatch"]:
        return
    logger.warning(
        "⚠️  NVIDIA GPU detected but PyTorch can't use CUDA (torch=%s). "
        "Training and inference will run on CPU and may be much slower. %s",
        status["torch_version"],
        status["install_hint"],
    )


# --- GPU capability probe -----------------------------------------------------

# Cached for the process lifetime: hardware doesn't hotplug under a running
# server, and /health is polled by every peer's registry. _GPU_UNPROBED is the
# not-yet-probed sentinel (None is a real "no accelerator" answer).
_GPU_UNPROBED = object()
_gpu_cache: object = _GPU_UNPROBED


def _probe_gpu_uncached() -> dict[str, str] | None:
    """The accelerator torch sees, display-ready, or None.

    torch is already resident in the server process (the lerobot import chain
    pulls it in), so this costs microseconds — but stays defensive anyway: any
    probe failure means "no accelerator", never an error, because /health is
    the node-registry handshake and must not break over a driver hiccup.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "vram": f"{props.total_memory / 2**30:.0f}GB",
            }
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Apple unified memory: no discrete VRAM figure to report.
            return {"name": "Apple Silicon (MPS)", "vram": ""}
    except Exception:  # noqa: BLE001 — health must answer regardless
        logger.debug("GPU probe failed", exc_info=True)
    return None


def probe_gpu() -> dict[str, str] | None:
    """Cached _probe_gpu_uncached; the capabilities.gpu value for /health."""
    global _gpu_cache
    if _gpu_cache is _GPU_UNPROBED:
        _gpu_cache = _probe_gpu_uncached()
    return _gpu_cache  # type: ignore[return-value]


# --- torchcodec loadability probe ---------------------------------------------

# Cached like the GPU probe. Subprocess-isolated on purpose: probing means
# dlopening torchcodec's FFmpeg-linked dylibs, and a broken dylib can do worse
# than raise — it can hard-crash the process. The trainer is a subprocess, so
# the probe mirrors exactly what the trainer will experience.
_TORCHCODEC_UNPROBED = object()
_torchcodec_cache: object = _TORCHCODEC_UNPROBED


def _probe_torchcodec_uncached() -> bool:
    """True when torchcodec's native libraries actually LOAD on this host.

    `import torchcodec` succeeding is NOT enough — lerobot picks torchcodec as
    the video backend whenever the package imports, but the FFmpeg shared
    libraries it dlopens (libavutil & co) load lazily at first decode, and a
    host without FFmpeg installed dies on the first training batch
    (field-debugged on a Mac with no Homebrew ffmpeg). torchcodec._core.ops
    performs that dlopen at import, so importing it in a subprocess is the
    honest preflight.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torchcodec._core.ops"],
            capture_output=True,
            timeout=30,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 — a probe failure means "can't use it"
        logger.debug("torchcodec probe failed to run", exc_info=True)
        return False


def torchcodec_loads() -> bool:
    """Cached _probe_torchcodec_uncached."""
    global _torchcodec_cache
    if _torchcodec_cache is _TORCHCODEC_UNPROBED:
        _torchcodec_cache = _probe_torchcodec_uncached()
        if not _torchcodec_cache:
            logger.warning(
                "torchcodec's native libraries don't load on this host (FFmpeg shared "
                "libraries missing?) — local training will decode video with pyav instead. "
                "Install ffmpeg (brew install ffmpeg / apt install ffmpeg) to use torchcodec."
            )
    return _torchcodec_cache  # type: ignore[return-value]
