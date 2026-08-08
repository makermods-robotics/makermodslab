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

"""HF Jobs runner — runs a training as an HF Jobs job on HuggingFace's GPUs.

Uses huggingface/lerobot-gpu:latest as the runtime image; the in-container
wrapper replaces its bundled lerobot with MakerMods Lab's exact pyproject pin before
launching the trainer (the image's "latest" drifts from the CLI surface our
argv builder targets). Tails logs via HfApi.fetch_job_logs and reuses the
existing parse_metrics_into parser since stdout format is identical to a
local lerobot run.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import netrc
import os
import re
import shlex
import threading
import time
import tomllib
from importlib.metadata import requires
from pathlib import Path
from queue import Empty, Queue

from huggingface_hub import get_token
from huggingface_hub.errors import RepositoryNotFoundError
from packaging.requirements import Requirement

from ..jobs import LogLine, TrainingMetrics, extract_wandb_run_url, parse_metrics_into
from ..train import TrainingRequest, build_training_command, parse_hf_duration
from ..utils.config import with_makermodslab_tag
from ..utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)

LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"

# The :latest image ships whatever lerobot was current when it was built —
# which drifts from the pin in our pyproject.toml that build_training_command's
# argv is shaped for (a real job died on `--eval_freq`, renamed upstream).
# The wrapper therefore pip-installs the exact pin (below) before launching
# the trainer, so container and host agree on the CLI surface.

# Extras from the pyproject pin that only matter on the host machine (serial
# motor buses). Dropped from the container install.
_HOST_ONLY_EXTRAS = frozenset({"feetech"})

# policy_type -> lerobot extra that carries the policy's model dependencies
# at the pinned ref (e.g. transformers for smolvla). Policies without an
# entry (act, tdmpc, vqbet, gaussian_actor) need nothing beyond the core install.
_POLICY_CLOUD_EXTRAS = {
    "smolvla": "smolvla",
    "diffusion": "diffusion",
    "pi0": "pi",
    "pi0_fast": "pi",
    "pi05": "pi",
}

# "git+https://github.com/<org>/<repo>(.git)@<ref>" — the shape of our pin.
_GIT_PIN_RE = re.compile(r"^git\+(?P<repo>https://github\.com/[^@#]+?)(?:\.git)?@(?P<ref>[^#]+)$")


def _pinned_lerobot_requirement() -> Requirement:
    """The exact lerobot requirement MakerMods Lab was installed with (the pyproject pin).

    Primary source is the installed distribution's metadata — generated from
    pyproject.toml at install time, so there is no second hardcoded copy of the
    sha and, crucially, it matches the lerobot actually importable on this host
    (the one build_training_command's argv is shaped for). Falls back to parsing
    pyproject.toml directly when running from a source tree without installed
    metadata.
    """
    candidates: list[str] = []
    with contextlib.suppress(Exception):
        candidates = requires("makermodslab") or []
    if not any("lerobot" in c for c in candidates):
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        candidates = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    for line in candidates:
        with contextlib.suppress(Exception):
            parsed = Requirement(line)
            if parsed.name.lower() == "lerobot":
                return parsed
    raise RuntimeError("Could not resolve the lerobot pin from MakerMods Lab metadata or pyproject.toml")


def cloud_lerobot_spec(policy_type: str) -> str:
    """Pip requirement the cloud container must install so its lerobot matches
    the pin that build_training_command targets.

    Derived from the pyproject pin at submission time (never hardcoded), with
    two container-side adjustments to the extras: host-only extras are dropped,
    and the trained policy's model-deps extra is added. A GitHub `git+…@ref`
    pin is rewritten to the equivalent source archive tarball
    (github.com/<org>/<repo>/archive/<ref>.tar.gz — same tree, but pip can
    install it without git in the image; lerobot's version is static, no scm).
    """
    req = _pinned_lerobot_requirement()
    extras = {e for e in req.extras if e not in _HOST_ONLY_EXTRAS}
    policy_extra = _POLICY_CLOUD_EXTRAS.get(policy_type)
    if policy_extra:
        extras.add(policy_extra)
    name = f"lerobot[{','.join(sorted(extras))}]" if extras else "lerobot"
    if req.url:
        m = _GIT_PIN_RE.match(req.url)
        url = f"{m.group('repo')}/archive/{m.group('ref')}.tar.gz" if m else req.url
        return f"{name} @ {url}"
    # Future-proofing: a PyPI-version pin flows through as a plain specifier.
    return f"{name}{req.specifier}"


def _install_plan(spec, python, uv_path, has_pip, has_ensurepip):
    """Pick how to install `spec` into `python`'s environment.

    Returns (label, commands): an installer name for logging plus the argv
    lists to run in order, or (None, []) when the environment has none.

    uv first: the lerobot-gpu image's venv is created with `uv venv`, which
    ships NO pip module — a real job died on `python -m pip` with "No module
    named pip". `--python` pins the install into this interpreter's env,
    mirroring _build_install_cmd in makermodslab/utils/system.py. pip stays as the
    fallback for future image changes; ensurepip is the last resort.

    Pure stdlib and self-contained by design: its source is inlined verbatim
    into WRAPPER_SOURCE (via inspect.getsource) so the in-container wrapper
    and the unit tests exercise the same implementation.
    """
    if uv_path:
        return "uv", [[uv_path, "pip", "install", "--python", python, "--no-cache", spec]]
    if has_pip:
        return "pip", [[python, "-m", "pip", "install", "--no-cache-dir", spec]]
    if has_ensurepip:
        return "ensurepip+pip", [
            [python, "-m", "ensurepip", "--upgrade"],
            [python, "-m", "pip", "install", "--no-cache-dir", spec],
        ]
    return None, []


def _checkpoint_step_ready(step_dir):
    """Whether `step_dir` (a lerobot checkpoints/<step>/ directory) has been
    fully written and is safe to upload.

    Reads lerobot's own completion signal instead of reconstructing its
    internal write order. `checkpoints/last` is a relative symlink that
    lerobot_train.py points at a step directory via update_last_checkpoint()
    (lerobot/common/train_utils.py) strictly *after* save_checkpoint() has
    returned for that step — so the link having advanced to (or past) this
    step number is proof every file under step_dir is already on disk. The
    `<=` (not `==`) covers two checkpoints completing inside one poll window.
    See test_lerobot_last_checkpoint_symlink_matches_our_readiness_check for
    the upstream contract this assumes.

    Pure and self-contained (stdlib only) so its source can be inlined
    verbatim into WRAPPER_SOURCE the same way _install_plan is, keeping the
    in-container check and the unit-tested implementation identical.
    """
    import os

    try:
        target = os.readlink(step_dir.parent / "last")
    except OSError:
        return False
    target = os.path.basename(target.rstrip("/"))
    return target.isdigit() and step_dir.name.isdigit() and int(step_dir.name) <= int(target)


def _cloud_device(flavor: str) -> str:
    """HF Jobs flavors are NVIDIA GPU boxes except the cpu-* tiers."""
    return "cpu" if flavor.startswith("cpu") else "cuda"


def localize_config_for_cloud(config: TrainingRequest, flavor: str) -> None:
    """Strip host-machine specifics from the request at the cloud-submission
    boundary, before build_training_command runs. Mutates in place — the
    mutated config is what gets persisted on the JobRecord, so the historical
    record reflects what actually ran. Local runs are untouched.

    Raises ValueError (→ HTTP 400) for host-path inputs that cannot work in
    the container, so the user gets a clear message instead of a remote crash.
    """
    # A host-local config_path (the local-resume signal) can't exist in the
    # container. Cloud resume uses resume_from_hub_repo instead — the wrapper
    # downloads the checkpoint from the Hub and reconstructs the layout there,
    # so that path is allowed and config_path is set to the container path later
    # (in HfCloudJobRunner.start), never here.
    if config.config_path and not config.resume_from_hub_repo:
        raise ValueError(
            "Resuming on a cloud job from a local checkpoint isn't supported: the "
            "source checkpoint's train_config.json lives on this machine, not in "
            "the container. Resume a cloud run from its Hub output instead."
        )
    # MT42, stated as an invariant rather than a hope: a cloud run that CALLS
    # itself a resume must name the Hub checkpoint it continues from. Without
    # one, build_training_command falls through to its fresh-run branch and the
    # pod starts a brand-new run at step 0 while every UI surface reports a
    # continuation. Refuse instead — the registry only reaches this point after
    # the checkpoint is confirmed on the Hub (its own resume resolvers), so this
    # can only fire for a request that bypassed them.
    if config.resume and not config.resume_from_hub_repo:
        raise ValueError(
            "This cloud run is marked as a continuation but names no Hub checkpoint "
            "to continue from, so it would silently start over at step 0. Resume "
            "from a run's checkpoint via Continue rather than setting resume by hand."
        )
    # A fine-tune base is allowed to be a HUB ref — either a bare repo id (whose
    # root lerobot resolves itself) or the step-suffixed 'repo@checkpoints/<step>'
    # form, which the wrapper materializes pod-side before launching the trainer
    # (the twin of the resume download above). What cannot work is a host path:
    # the container has no view of this machine's disk.
    #
    # Belt-and-braces since F7's fine-tune quadrant landed: a local base picked
    # in the UI never arrives here as a path any more — JobRegistry.start stages
    # its weights to a private Hub repo (with the user's consent) and rewrites
    # the request to the resulting ref before the runner is reached. Only a
    # request that bypasses the registry can still trip this, so the message
    # names both ways out rather than claiming the case is unsupported.
    if config.policy_pretrained_path and Path(config.policy_pretrained_path).is_absolute():
        raise ValueError(
            "A cloud job can't fine-tune from a checkpoint on this machine — the "
            "container has no view of this disk. Launch it from the training form, "
            "which offers to upload the base checkpoint to a private Hub repo first, "
            "or push the source model to the Hub and fine-tune from the Hub copy."
        )
    # The container resolves the dataset from the Hub by repo_id; a host-local
    # dataset root doesn't exist there.
    config.dataset_root = None
    # The host's auto-detected device (mps on a Mac) is meaningless on the
    # remote pod; pin the flavor's real backend instead.
    config.policy_device = _cloud_device(flavor)


# Where the trainer writes checkpoints inside the HF Jobs container. The host
# path the registry hands us (under ~/.cache/...) doesn't exist on the remote
# pod, so we ignore it and pin a writable container-local path instead. The
# wrapper reads --output_dir from the trainer argv and uploads checkpoints from
# here to the Hub, so the MakerMods Lab UI never reads this path directly.
_CONTAINER_OUTPUT_DIR = "/tmp/makermodslab/train"  # nosec B108 — fixed path inside the remote HF Jobs container, not host-local

# lerobot's per-checkpoint layout under <output_dir>/checkpoints/<step_dir>/.
# Cloud resume reconstructs exactly this so the trainer's own resume path (which
# reads config_path.parent.parent as the checkpoint dir) finds pretrained_model/
# and training_state/ where it expects them.
_CONTAINER_TRAIN_CONFIG_NAME = "train_config.json"

# Inlined sidecar uploader for HF Jobs. Spawns the lerobot trainer as a
# subprocess and concurrently uploads new <output_dir>/checkpoints/<step>/
# directories to the Hub model repo, so the MakerMods Lab UI can list them while
# training is in progress.
#
# Sent verbatim as the value of `python -c '...'`. Wrapper-side arguments
# (the pinned lerobot spec) come before `--`; anything after `--` is
# forwarded to the trainer. The __INSTALL_PLAN_SOURCE__ and
# __CHECKPOINT_READY_SOURCE__ placeholders are replaced with _install_plan's
# and _checkpoint_step_ready's own source below, so the wrapper's installer
# choice and checkpoint-completeness check are the exact functions the unit
# tests exercise.
_WRAPPER_TEMPLATE = r'''
import importlib.util
import os, re, shlex, shutil, sys, threading, subprocess
from pathlib import Path
from huggingface_hub import HfApi

__INSTALL_PLAN_SOURCE__

__CHECKPOINT_READY_SOURCE__

argv = sys.argv[1:]
if "--" not in argv:
    print("[wrapper] missing -- separator", flush=True)
    sys.exit(2)
sep = argv.index("--")
wrapper_args = argv[:sep]
trainer_argv = argv[sep + 1:]

# Wrapper-side args: the pinned lerobot spec (first non---option token) plus
# optional directives. --resume-from=<repo>@checkpoints/<step_dir> tells us to
# download that checkpoint tree and reconstruct lerobot's output-dir layout so
# the trainer's own resume path finds it (config_path.parent.parent).
lerobot_spec = next((a for a in wrapper_args if not a.startswith("--")), None)
resume_from = None
for a in wrapper_args:
    if a.startswith("--resume-from="):
        resume_from = a.split("=", 1)[1]


def _arg(name):
    """Return the value of --name=foo or --name foo from trainer_argv."""
    for i, tok in enumerate(trainer_argv):
        if tok == name and i + 1 < len(trainer_argv):
            return trainer_argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def _set_arg(name, value):
    """Rewrite --name=foo / --name foo in trainer_argv in place. False if absent.

    Mutates the list the trainer is launched with (Popen receives it directly),
    which is how a Hub ref that only this pod can resolve becomes a real path.
    Both spellings are handled because the value is written back in whichever
    form the argv builder used."""
    for i, tok in enumerate(trainer_argv):
        if tok == name and i + 1 < len(trainer_argv):
            trainer_argv[i + 1] = value
            return True
        if tok.startswith(name + "="):
            trainer_argv[i] = name + "=" + value
            return True
    return False


output_dir = _arg("--output_dir")
repo_id = _arg("--policy.repo_id")
if not output_dir or not repo_id:
    print(f"[wrapper] need --output_dir and --policy.repo_id; got {output_dir} / {repo_id}", flush=True)
    sys.exit(2)

# The image ships whatever lerobot was latest when it was built; the trainer
# argv is shaped for MakerMods Lab's pinned lerobot. Install the exact pin (passed as a
# wrapper arg) before launching, or the argument surfaces drift apart (a real
# run died on argparse rc=2 over --eval_freq).
if lerobot_spec:
    install_label, install_cmds = _install_plan(
        lerobot_spec,
        sys.executable,
        shutil.which("uv"),
        importlib.util.find_spec("pip") is not None,
        importlib.util.find_spec("ensurepip") is not None,
    )
    if install_label is None:
        print("[wrapper] cannot install pinned lerobot: no uv, pip, or ensurepip in image", flush=True)
        sys.exit(1)
    print(f"[wrapper] installing pinned lerobot via {install_label}: {lerobot_spec}", flush=True)
    for install_cmd in install_cmds:
        install_rc = subprocess.run(install_cmd).returncode
        if install_rc != 0:
            print(f"[wrapper] pinned lerobot install failed rc={install_rc}: {shlex.join(install_cmd)}", flush=True)
            sys.exit(install_rc)

api = HfApi()
# lerobot only calls push_to_hub at the end of training, so the repo doesn't
# exist when our checkpoint watcher fires. Create it up front (idempotent).
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"[wrapper] repo ready: {repo_id}", flush=True)
except Exception as exc:
    print(f"[wrapper] create_repo failed: {exc}", flush=True)

seen = set()
# Per-step count of consecutive not-ready polls, so a stalled gate (or a
# genuinely slow write) surfaces in the log instead of looking identical to
# "no checkpoint yet" for the whole run.
waits = {}

# Resume: download the parent checkpoint tree (pretrained_model/ +
# training_state/) into <output_dir>/checkpoints/<step_dir>/ so lerobot's own
# resume path (config_path.parent.parent) finds the optimizer + step state. The
# step dir is pre-seeded into `seen` so the watcher never re-uploads the
# checkpoint we just pulled down.
if resume_from:
    m = re.match(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$", resume_from)
    if not m:
        print(f"[wrapper] bad --resume-from ref: {resume_from}", flush=True)
        sys.exit(2)
    src_repo, step_dir = m.group("repo"), m.group("step_dir")
    dest = Path(output_dir) / "checkpoints" / step_dir
    print(f"[wrapper] resuming: downloading {src_repo}@checkpoints/{step_dir}", flush=True)
    try:
        from huggingface_hub import snapshot_download

        local_root = snapshot_download(
            repo_id=src_repo,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step_dir}/*"],
        )
        src = Path(local_root) / "checkpoints" / step_dir
        dest.parent.mkdir(parents=True, exist_ok=True)
        # copytree from the snapshot cache (symlinked files) into a real tree the
        # trainer can read/rewrite; resolve symlinks so lerobot sees plain files.
        shutil.copytree(src, dest, symlinks=False)
        # A bare training_state/ is_dir() check would pass a checkpoint that
        # was itself partially uploaded before this fix existed (the exact
        # bug this watcher fixes). There is no checkpoints/last symlink to
        # check here — it lives beside the run's local output dir and is
        # never pushed to the Hub — so check the files a resume actually
        # needs directly instead.
        has_weights = any((dest / "pretrained_model").glob("*.safetensors"))
        has_training_state = (
            (dest / "training_state" / "training_step.json").is_file()
            and (dest / "training_state" / "rng_state.safetensors").is_file()
        )
        if not (has_weights and has_training_state):
            print(
                f"[wrapper] resume checkpoint at {dest} is incomplete "
                f"(weights: {has_weights}, training_state: {has_training_state}); cannot resume",
                flush=True,
            )
            sys.exit(1)
        seen.add(step_dir)
        print(f"[wrapper] resume checkpoint ready at {dest}", flush=True)
    except Exception as exc:
        print(f"[wrapper] resume download failed: {exc}", flush=True)
        sys.exit(1)

# Fine-tune from a specific Hub step: --policy.pretrained_path may carry the
# ref 'repo@checkpoints/<step_dir>' instead of a path, because lerobot's
# pretrained_path addresses a local dir or a repo ROOT and cannot express a
# sub-path. Materialize it here — the pod-side twin of the host-side
# jobs.localize_pretrained_path — and rewrite the argv to the real directory.
# A bare repo id is left ALONE: lerobot resolves a repo root itself, and that
# flow already works.
#
# Downloaded into the snapshot cache and used from there, NOT copied under
# --output_dir like the resume tree: fine-tuning only READS these weights, and
# anything under <output_dir>/checkpoints/ would be picked up by the uploader
# below and republished as if this run had produced it.
pretrained_ref = _arg("--policy.pretrained_path")
if pretrained_ref:
    m = re.match(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$", pretrained_ref)
    if m:
        src_repo, step_dir = m.group("repo"), m.group("step_dir")
        print(f"[wrapper] fine-tuning: downloading {src_repo}@checkpoints/{step_dir}", flush=True)
        try:
            from huggingface_hub import snapshot_download

            local_root = snapshot_download(
                repo_id=src_repo,
                repo_type="model",
                allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"],
            )
            base_dir = Path(local_root) / "checkpoints" / step_dir / "pretrained_model"
            if not (base_dir / "config.json").is_file():
                print(f"[wrapper] fine-tune base has no config.json at {base_dir}", flush=True)
                sys.exit(1)
            if not _set_arg("--policy.pretrained_path", str(base_dir)):
                print("[wrapper] could not rewrite --policy.pretrained_path", flush=True)
                sys.exit(1)
            print(f"[wrapper] fine-tune base ready at {base_dir}", flush=True)
        except Exception as exc:
            print(f"[wrapper] fine-tune base download failed: {exc}", flush=True)
            sys.exit(1)

stop_event = threading.Event()


def _scan_and_upload():
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return
    # Snapshot before iterating so deletions during the walk do not raise.
    entries = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    for entry in entries:
        if not re.fullmatch(r"\d+", entry.name):
            continue
        if entry.name in seen:
            continue
        # config.json can exist while the rest of the step is still being
        # written. See _checkpoint_step_ready.
        if not _checkpoint_step_ready(entry):
            waits[entry.name] = waits.get(entry.name, 0) + 1
            if waits[entry.name] in (1, 20, 80):  # ~15s, ~5min, ~20min at the 15s poll interval
                try:
                    last_target = os.readlink(root / "last")
                except OSError:
                    last_target = None
                print(
                    f"[wrapper] checkpoint {entry.name} not complete yet "
                    f"(poll {waits[entry.name]}); checkpoints/last -> {last_target}",
                    flush=True,
                )
            continue
        try:
            api.upload_folder(
                folder_path=str(entry),
                repo_id=repo_id,
                path_in_repo=f"checkpoints/{entry.name}",
                commit_message=f"checkpoint {entry.name}",
            )
            seen.add(entry.name)
            waits.pop(entry.name, None)
            print(f"[wrapper] uploaded checkpoint {entry.name}", flush=True)
        except Exception as exc:
            print(f"[wrapper] upload failed for {entry.name}: {exc}", flush=True)


def _watch():
    while not stop_event.is_set():
        try:
            _scan_and_upload()
        except Exception as exc:
            print(f"[wrapper] scan error: {exc}", flush=True)
        stop_event.wait(15)


watch_thread = threading.Thread(target=_watch, name="ckpt-watcher", daemon=True)
watch_thread.start()

# Run the trainer on this same interpreter so it sees the just-installed pin.
if trainer_argv and trainer_argv[0] == "python":
    trainer_argv[0] = sys.executable

# trainer_argv is passed to Popen as a LIST (never joined and re-split), so
# values with spaces stay one argument; shlex.join is only for a faithful log.
print(f"[wrapper] launching trainer: {shlex.join(trainer_argv)}", flush=True)
proc = subprocess.Popen(list(trainer_argv), env=os.environ.copy())
try:
    rc = proc.wait()
finally:
    stop_event.set()
    # One final pass picks up any checkpoint saved in the last 15s window.
    try:
        _scan_and_upload()
    except Exception as exc:
        print(f"[wrapper] final scan error: {exc}", flush=True)

print(f"[wrapper] trainer exited with rc={rc}", flush=True)
sys.exit(rc)
'''

WRAPPER_SOURCE = _WRAPPER_TEMPLATE.replace(
    "__INSTALL_PLAN_SOURCE__", inspect.getsource(_install_plan)
).replace("__CHECKPOINT_READY_SOURCE__", inspect.getsource(_checkpoint_step_ready))

# HF Jobs' platform default timeout has killed legitimate runs that pushed
# the model successfully but were still uploading auxiliary files. 2h covers
# our typical ACT/SmolVLA runs on t4-small with comfortable headroom. This is
# the FALLBACK: used only when the request carries no explicit hf_job_timeout.
HF_JOB_TIMEOUT = "2h"


def resolve_job_timeout(config: TrainingRequest) -> int | str:
    """The value to hand HfApi.run_job's `timeout` for this job.

    Precedence: an explicit, already-validated request value
    (config.hf_job_timeout) wins and is normalised to an int of SECONDS —
    run_job's own string parser only understands a single unit suffix
    (float(timeout[:-1]) * factor[timeout[-1]]), so compound forms like
    "3h30m" must be pre-resolved here rather than passed through as a string.
    When the request leaves the field unset we fall back to the HF_JOB_TIMEOUT
    constant (a plain single-unit string run_job parses natively), preserving
    the platform-default-killed-legit-runs safety net.
    """
    if config.hf_job_timeout:
        return parse_hf_duration(config.hf_job_timeout)
    return HF_JOB_TIMEOUT


# Cadence at which the status poller hits inspect_job. inspect_job is the
# authoritative source for job liveness; the log stream is best-effort and
# may drop during long runs (NAT eviction, laptop sleep, proxy idle timeout)
# without the job actually ending.
_STATUS_POLL_INTERVAL_S = 5.0

# Stages we treat as terminal. Allowlist (not "anything except RUNNING") so
# freshly-submitted jobs in transient stages like QUEUED / BUILDING / STARTING
# aren't mistaken for failures before they get a chance to run.
_TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})

# How long _tail_loop waits before reconnecting after a clean stream end
# (gives the status poller a chance to confirm the job is actually terminal,
# so we don't reconnect and re-replay the entire buffered log).
_TAIL_CLEAN_END_WAIT_S = 15.0

# How long _tail_loop waits before reconnecting after an SSE exception
# (transient network blip during a long training).
_TAIL_RECONNECT_BACKOFF_S = 5.0


def resolve_wandb_api_key() -> str | None:
    """Look up the host's wandb API key for forwarding to a cloud job.

    Checks WANDB_API_KEY first, then falls back to ~/.netrc (where
    `wandb login` writes the key under machine api.wandb.ai). Returns None
    if neither source has it; the caller decides how to surface that.
    """
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        rc = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = rc.authenticators("api.wandb.ai")
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


class HfCloudJobRunner:
    """Run a training as an HF Jobs job. Single-shot — instantiate per job."""

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        flavor: str,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._flavor = flavor
        # Shared HfApi: its in-process whoami cache covers run_job's
        # internal self.whoami(token=...) call too (see utils/hf_auth.py),
        # so submitting many jobs doesn't hammer /whoami-v2.
        self._api = shared_hf_api()
        self._hf_job_id: str | None = None
        self._hf_job_url: str | None = None
        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        # _status_thread polls inspect_job and is the sole writer of
        # _terminal_status (except for stop(), which pre-sets CANCELED).
        # Decoupling liveness from the log stream means a flaky SSE
        # connection no longer makes us declare a running job as failed.
        self._status_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_file = None  # type: ignore[assignment]
        # Cached terminal status once the job ends; None while live.
        self._terminal_status: str | None = None
        # Status.message at the terminal tick (e.g. "Job timeout"), so the
        # registry can surface it to the UI instead of a synthetic exit code.
        self._terminal_message: str | None = None
        self._wandb_run_url: str | None = None
        # Count of log lines processed across (possibly multiple) SSE
        # connections, so reconnects skip the replayed prefix.
        self._lines_processed: int = 0

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # output_dir is the host-local path the registry pins for local jobs;
        # it doesn't exist on the remote pod, so cloud jobs write to a
        # container-local path instead (checkpoints reach the UI via the Hub).
        del output_dir
        if self._hf_job_id is not None:
            raise RuntimeError("HfCloudJobRunner already started")

        token = get_token()
        if not token:
            raise RuntimeError("HF token not found. Run 'hf auth login' before launching cloud jobs.")

        whoami = cached_whoami()
        username = whoami.get("name") if whoami else None
        if not username:
            raise RuntimeError("Could not resolve HF username from whoami()")

        # Strip host-machine specifics (auto-detected device, local dataset
        # root, host checkpoint paths) BEFORE the potentially-slow dataset
        # upload, so invalid requests fail fast with a 400.
        localize_config_for_cloud(config, self._flavor)

        # Open the log file early so dataset-upload progress is recorded
        # before the cloud job is submitted.
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

        # Cloud pods can't see the host's LeRobot cache. If the dataset
        # only exists locally, push it to the Hub before submitting.
        self._ensure_dataset_on_hub(config.dataset_repo_id)

        # Mutate the config so build_training_command emits the right flags.
        # The mutated config is what gets persisted in JobRecord.config, so
        # the historical record reflects what actually ran.
        config.policy_push_to_hub = True
        # A fresh run gets its own repo named after its unique job id slug (e.g.
        # "act_dataset_2026-05-04_10-22-03"); a continuation picks between that
        # and the parent's repo — see the branch below.
        resume_directive: str | None = None
        if config.resume and config.resume_from_hub_repo:
            # Where this run PUBLISHES is not always where it resumes FROM. A
            # cloud→cloud continuation keeps the parent's output repo so the
            # lineage lives in one place. A local→cloud one (F7) resumes from a
            # private staging repo that exists only to carry the local parent's
            # uploaded checkpoint, so it publishes to its own repo instead —
            # otherwise the staging repo would accumulate the child's
            # checkpoints too and parent and child would again be
            # indistinguishable inside one tree.
            config.policy_repo_id = (
                f"{username}/{job_id}"
                if config.resume_from_uploaded_checkpoint
                else config.resume_from_hub_repo
            )
            step_dir = config.resume_from_hub_step or "last"
            # The wrapper downloads checkpoints/<step_dir>/ into this exact path;
            # lerobot's resume reads config_path.parent.parent as the checkpoint
            # dir, so both pretrained_model/ and training_state/ must live here.
            config.config_path = (
                f"{_CONTAINER_OUTPUT_DIR}/checkpoints/{step_dir}/pretrained_model/"
                f"{_CONTAINER_TRAIN_CONFIG_NAME}"
            )
            resume_directive = f"--resume-from={config.resume_from_hub_repo}@checkpoints/{step_dir}"
        else:
            config.policy_repo_id = f"{username}/{job_id}"

        trainer_argv = build_training_command(config, _CONTAINER_OUTPUT_DIR)
        # The wrapper expects `python -c WRAPPER_SOURCE <spec> [directives] -- <trainer argv>`.
        # `python -c` consumes the first non-option argument as the script,
        # so we prepend a "--" sentinel of our own; the pinned-lerobot spec and
        # any wrapper directives (e.g. --resume-from) ride before it as
        # wrapper-side arguments.
        wrapper_side_args = [cloud_lerobot_spec(config.policy_type)]
        if resume_directive is not None:
            wrapper_side_args.append(resume_directive)
        wrapped_command = [
            "python",
            "-c",
            WRAPPER_SOURCE,
            *wrapper_side_args,
            "--",
            *trainer_argv,
        ]
        logger.info(
            "Submitting HF Cloud job %s on %s (wrapped trainer): %s",
            job_id,
            self._flavor,
            shlex.join(trainer_argv),
        )

        # HF_TOKEN goes via `secrets` (not `env`) so it doesn't show up in
        # the job's environment variable inspection / logs.
        secrets = {"HF_TOKEN": token}
        if config.wandb_enable:
            wandb_key = resolve_wandb_api_key()
            if not wandb_key:
                # ValueError so main.py maps it to a 400 + detail the UI shows.
                raise ValueError(
                    "WANDB_API_KEY not found on this machine. "
                    "Run `wandb login` or export WANDB_API_KEY before launching "
                    "cloud jobs with W&B enabled."
                )
            secrets["WANDB_API_KEY"] = wandb_key

        job = self._api.run_job(
            image=LEROBOT_IMAGE,
            command=wrapped_command,
            flavor=self._flavor,
            secrets=secrets,
            timeout=resolve_job_timeout(config),
        )
        self._hf_job_id = job.id
        self._hf_job_url = getattr(job, "url", None)

        self._start_worker_threads(job_id)

    def reattach(self, hf_job_id: str) -> None:
        """Take over an existing HF job after a process restart.

        Skips submission; just opens the log file in append mode and starts
        the log-tailing + status-polling threads.
        """
        if self._hf_job_id is not None:
            raise RuntimeError("HfCloudJobRunner already started")
        self._hf_job_id = hf_job_id
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)
        self._start_worker_threads(f"{hf_job_id}-reattach")

    def _start_worker_threads(self, label: str) -> None:
        """Start the log tail and status poll threads. Both run for the
        life of the runner; the status poller is what marks the job terminal."""
        self._tail_thread = threading.Thread(target=self._tail_loop, name=f"hf-job-{label}-logs", daemon=True)
        self._tail_thread.start()
        self._status_thread = threading.Thread(
            target=self._status_poll_loop, name=f"hf-job-{label}-status", daemon=True
        )
        self._status_thread.start()

    def _set_terminal(self, status: str, message: str | None = None) -> None:
        """Record the job's terminal stage. Idempotent. Wakes the tail loop."""
        if self._terminal_status is not None:
            return
        self._terminal_status = status
        if message:
            self._terminal_message = message
        self._stop_event.set()

    def _log_line(self, message: str) -> None:
        """Append a wrapper-style line to the job's log file."""
        if self._log_file is None:
            return
        line = LogLine(timestamp=time.time(), message=message)
        try:
            self._log_file.write(line.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("Could not write upload log line: %s", exc)

    def _ensure_dataset_on_hub(self, repo_id: str) -> None:
        """If the dataset is local-only, push it to the Hub.

        The cloud pod resolves the dataset by repo_id; it can't see the
        host's `~/.cache/huggingface/lerobot`. We push synchronously and
        let any failure bubble up — JobRegistry.start marks the record
        as failed with the exception message.
        """
        try:
            self._api.dataset_info(repo_id)
            return
        except RepositoryNotFoundError:
            pass

        cache_root = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
        if not (cache_root / repo_id / "meta" / "info.json").is_file():
            # Neither local nor on Hub. Let the trainer surface the error
            # — same behaviour as before.
            return

        self._log_line(f"[upload] dataset {repo_id} not on Hub; pushing local copy (public)...")
        from lerobot.datasets import LeRobotDataset

        try:
            # Public by default: MakerMods Lab's global policy is that datasets it pushes
            # to the Hub are public and carry the required org/product tags (see
            # with_makermodslab_tag / REQUIRED_HUB_TAGS). This implicit cloud-run upload
            # follows that same default so all MakerMods Lab-produced datasets are
            # discoverable. (This intentionally reverses the earlier private
            # default — an implicit upload of a local-only dataset is now public.)
            LeRobotDataset(repo_id).push_to_hub(tags=with_makermodslab_tag(None), private=False)
        except Exception as exc:
            msg = f"Failed to upload local dataset {repo_id} to Hub: {exc}"
            self._log_line(f"[upload] {msg}")
            raise RuntimeError(msg) from exc
        self._log_line(f"[upload] dataset {repo_id} uploaded.")

    def _tail_loop(self) -> None:
        """Stream HfApi.fetch_job_logs, teeing each line to disk and the
        in-memory queue. Reconnects on stream end or transient error while
        the status poller still says the job is alive — SSE death is no
        longer fatal. Exits when _stop_event is set (status poller saw a
        terminal stage, or stop() was called).
        """
        assert self._hf_job_id is not None
        try:
            while not self._stop_event.is_set():
                clean_end = False
                try:
                    seen = 0
                    for raw in self._api.fetch_job_logs(job_id=self._hf_job_id, follow=True):
                        if self._stop_event.is_set():
                            return
                        seen += 1
                        # Skip the replayed prefix from a reconnect.
                        if seen <= self._lines_processed:
                            continue
                        self._lines_processed = seen
                        stripped = raw.rstrip()
                        if not stripped:
                            continue
                        parse_metrics_into(stripped, self._metrics)
                        if self._wandb_run_url is None:
                            url = extract_wandb_run_url(stripped)
                            if url is not None:
                                self._wandb_run_url = url
                        log_line = LogLine(timestamp=time.time(), message=stripped)
                        if self._log_file is not None:
                            try:
                                self._log_file.write(log_line.model_dump_json() + "\n")
                            except Exception as exc:  # pragma: no cover
                                logger.exception("Error writing HF log: %s", exc)
                        if self._log_queue.qsize() >= 1000:
                            with contextlib.suppress(Empty):
                                self._log_queue.get_nowait()
                        self._log_queue.put(log_line)
                    clean_end = True
                except Exception as exc:
                    logger.info("HF log tail disconnected, will reconnect: %s", exc)

                wait_s = _TAIL_CLEAN_END_WAIT_S if clean_end else _TAIL_RECONNECT_BACKOFF_S
                if self._stop_event.wait(wait_s):
                    return
        finally:
            if self._log_file is not None:
                with contextlib.suppress(Exception):
                    self._log_file.close()
                self._log_file = None

    def _status_poll_loop(self) -> None:
        """Poll inspect_job until the job reaches a terminal stage.

        Sole writer of _terminal_status under normal operation. Decoupled
        from the log stream: a dropped SSE connection during a long run
        (NAT eviction, sleep, proxy idle timeout) no longer causes MakerMods Lab
        to declare a still-running job as failed.
        """
        assert self._hf_job_id is not None
        while not self._stop_event.is_set():
            try:
                info = self._api.inspect_job(job_id=self._hf_job_id)
                status_obj = getattr(info, "status", None)
                stage = getattr(status_obj, "stage", None) if status_obj is not None else None
                if stage is not None:
                    stage_str = str(stage).upper()
                    if stage_str in _TERMINAL_STAGES:
                        msg = getattr(status_obj, "message", None)
                        self._set_terminal(stage_str, str(msg) if msg else None)
                        return
            except Exception as exc:
                logger.warning("inspect_job poll failed for %s: %s", self._hf_job_id, exc)
            if self._stop_event.wait(_STATUS_POLL_INTERVAL_S):
                return

    def stop(self) -> None:
        if self._hf_job_id is None:
            return
        # Pre-set CANCELED so the watchdog finalises as canceled regardless
        # of whether the status poller observed a terminal stage first.
        self._set_terminal("CANCELED")
        try:
            self._api.cancel_job(job_id=self._hf_job_id)
        except Exception as exc:
            # Already-completed jobs may 404; that's fine.
            logger.info("cancel_job(%s) ignored: %s", self._hf_job_id, exc)

    def is_running(self) -> bool:
        # Liveness is driven by _status_poll_loop's inspect_job calls.
        if self._hf_job_id is None:
            return False
        return self._terminal_status is None

    def returncode(self) -> int | None:
        if self._terminal_status is None:
            return None
        return 0 if self._terminal_status == "COMPLETED" else 1

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def hf_job_id(self) -> str | None:
        return self._hf_job_id

    def hf_job_url(self) -> str | None:
        return self._hf_job_url

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    def terminal_message(self) -> str | None:
        """Status.message captured when the job reached a terminal stage.

        Set by _status_poll_loop when it observes a terminal stage. Used by
        the registry watchdog to surface platform reasons like 'Job timeout'
        rather than a synthetic 'exit code 1'.
        """
        return self._terminal_message
