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

"""Job lifecycle and registry for trainings (and, in future, other long-running
work). One JobRunner instance owns one subprocess; the JobRegistry owns the
overall state, including history persisted to disk under outputs/train/."""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal, Protocol, runtime_checkable

from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm.auto import tqdm as _base_tqdm

from .datasets import CAMERA_FEATURE_PREFIX, read_dataset_features
from .train import TrainingRequest
from .utils.config import is_valid_robot_name
from .utils.hf_auth import LOGIN_COMMAND, cached_whoami, hf_hub_offline, shared_hf_api
from .utils.naming import (
    dedupe_display_names,
    derive_imported_title,
    imported_name_suffixes,
)

logger = logging.getLogger(__name__)


JobState = Literal["running", "done", "failed", "interrupted"]


class JobTarget(BaseModel):
    """Where a job should run. `local` ⇒ LocalJobRunner. `hf_cloud` requires
    a non-empty `flavor` from HfApi.list_jobs_hardware()."""

    runner: Literal["local", "hf_cloud"] = "local"
    flavor: str | None = None


class TrainingMetrics(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    current_loss: float | None = None
    current_lr: float | None = None
    grad_norm: float | None = None
    eta_seconds: float | None = None


class LogLine(BaseModel):
    timestamp: float
    message: str


class JobRecord(BaseModel):
    id: str
    name: str
    # User-editable display alias set via JobRegistry.rename. Display-only:
    # the immutable identity (id / output_dir / hf_repo_id) never changes on
    # rename, so resume lineage, imported-model dedup, and remote HF/W&B
    # names stay intact. None ⇒ the UI falls back to `name`.
    display_name: str | None = None
    state: JobState
    config: TrainingRequest
    output_dir: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    metrics: TrainingMetrics = TrainingMetrics()
    runner: Literal["local", "hf_cloud", "imported"] = "local"
    # PID of the detached subprocess (local runner only); survives uvicorn
    # --reload so a fresh registry can re-attach by tailing the log file.
    process_pid: int | None = None
    # HF Jobs identifiers (hf_cloud runner only)
    hf_job_id: str | None = None
    hf_flavor: str | None = None
    hf_repo_id: str | None = None
    hf_job_url: str | None = None
    # Number of checkpoints currently visible (local: filesystem; cloud:
    # Hub repo). Filled in by JobRegistry.list/get; persisted as zero.
    checkpoint_count: int = 0
    # Where THIS run's on-disk checkpoints were uploaded so a cloud
    # continuation could read them (local runner only; F7's local→cloud
    # direction). Distinct from `hf_repo_id`, which is a cloud run's own output
    # repo: this one is a staging repo MakerMods Lab creates, private, holding
    # only the steps listed below. Recorded on the PARENT record so a second
    # cross-runner resume of the same step re-uses the upload instead of
    # pushing the same GBs again.
    checkpoints_hub_repo_id: str | None = None
    # Step dirs known to be present under checkpoints/ in that repo.
    checkpoints_hub_steps: list[str] = []


class JobCheckpoint(BaseModel):
    """One checkpoint produced by a training job.

    `ref` is opaque to the frontend; the inference handler resolves it back
    to a usable `--policy.path` value (a local dir for both sources, after
    snapshot_download for hub refs)."""

    step: int
    source: Literal["local", "hub"]
    ref: str


class MetricsHistoryPoint(BaseModel):
    """One (step, metrics) sample reconstructed from a job's log.jsonl.

    Used by GET /jobs/{id}/metrics-history to seed the monitoring charts.
    A point is emitted for each log line that carried a `step: ... loss: ...`
    payload (the log-freq lines from lerobot). Tqdm progress lines are
    skipped — they carry step + ETA but no loss/lr/grdn."""

    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID exists. Cheap; uses signal 0."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@runtime_checkable
class JobRunner(Protocol):
    """Backend interface for running one job. LocalJobRunner is the only impl
    today; remote runners (SSH, Slurm) drop in here later. @runtime_checkable
    lets `isinstance(r, JobRunner)` work in tests / sanity checks.

    Three OPTIONAL hooks are read defensively by the registry watchdog via
    `_runner_hook` rather than declared here, so a runner that can't answer
    them still satisfies this Protocol (HfCloudJobRunner has no
    `stop_signalled`; the local runners have no `terminal_stage`):

      * `stop_signalled() -> bool` — did this runner actually deliver the
        stop signal to a live process? False means the process was already
        gone, so a nonzero exit code belongs to the process, not to us.
      * `terminal_stage() -> str | None` — a platform-reported terminal stage
        (HF Jobs' COMPLETED / CANCELED / ERROR / DELETED), which is richer
        than an exit code and is preferred over it when present.
      * `terminal_message() -> str | None` — the platform's own reason string
        (e.g. HF Jobs' "Job timeout"), surfaced instead of a synthetic
        "exited with code N".
    """

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def returncode(self) -> int | None: ...
    def stream_log_lines(self) -> list[LogLine]: ...


# Written to `error_message` when a run ends because we asked it to. The field
# is named for its usual content but is rendered as neutral subtext, and
# leaving it None would show the user nothing at all where a misleading
# "Subprocess exited with code 1" used to be. The real code stays in
# `exit_code` for anyone debugging.
STOPPED_BY_REQUEST_MESSAGE = "Stopped at your request, not by a training error."


def classify_terminal_state(
    *,
    returncode: int | None,
    stop_requested: bool,
    terminal_stage: str | None = None,
) -> JobState:
    """Decide the final state of a run that has just stopped running.

    Exists because an exit code alone cannot tell a deliberate stop from a
    crash: SIGTERM'ing the trainer yields a nonzero code, so before this every
    press of the Stop button landed in history as `failed` with a synthetic
    "Subprocess exited with code N" — indistinguishable from a real failure,
    and read by at least one user as "the model is broken" when nothing was.

    `stop_requested` is the registry's recorded intent, ANDed with the runner's
    own account of whether it actually signalled anything (see
    `JobRunner`'s optional `stop_signalled` hook).

    Precedence, and the reasoning for the ambiguous cases:

    1. `terminal_stage` wins when the platform reports one (hf_cloud). It is
       set once and never overwritten, so a stage observed BEFORE our cancel
       landed is the truth about a run that beat us to the finish:
         * COMPLETED → `done`, even with a stop pending. A run that finished
           on its own is never relabelled.
         * ERROR     → `failed`, even with a stop pending. The poller can only
           have seen ERROR before our cancel took effect, so this is a genuine
           crash that merely coincided with the stop; laundering it into
           `interrupted` would hide a real failure.
         * CANCELED  → `interrupted`, but only if we asked. An unsolicited
           CANCELED (cancelled in the HF web UI, or however the platform
           chooses to report an enforced timeout) is left as `failed` rather
           than guessed at — see the module note in `JobRegistry.stop`.
         * DELETED   → `failed`, unchanged from before.
    2. `returncode == 0` → `done`, whatever else is true. A clean exit means
       the trainer ran its own shutdown path to completion, which SIGTERM does
       not produce; intent never overrides it.
    3. A nonzero code with a stop we actually signalled → `interrupted`.
       Deliberately not narrowed to signal-shaped codes (-15/-9): a trainer
       that installs its own SIGTERM handler and exits 1 is still a run we
       ended, and requiring -15 would leave the bug unfixed for exactly that
       case. The cost is that a crash landing inside the microseconds between
       the runner's "is it still alive?" check and its `terminate()` call is
       misread as a stop. That window is unobservably narrow and carries no
       evidence that could separate the two; every wider window (process
       already dead before the signal, platform stage already terminal) is
       caught above.
    4. Anything else → `failed`, including a missing code.
    """
    if terminal_stage is not None:
        stage = terminal_stage.upper()
        if stage == "COMPLETED":
            return "done"
        if stage == "CANCELED" and stop_requested:
            return "interrupted"
        return "failed"
    if returncode == 0:
        return "done"
    if stop_requested and returncode is not None:
        return "interrupted"
    return "failed"


def _runner_hook(runner: object, name: str):
    """Call an optional zero-arg runner hook, or return None.

    None means "this runner can't answer", never a substantive answer — a
    runner that doesn't implement the hook and one whose hook raised are
    deliberately indistinguishable, because neither is evidence."""
    fn = getattr(runner, name, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # pragma: no cover — a hook must never break finalisation
        return None


# tqdm progress: "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
_TQDM_RE = re.compile(r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+)")

def _parse_duration(s: str) -> float | None:
    """Parse tqdm's HH:MM:SS or MM:SS into seconds. Returns None on '?'."""
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def parse_metrics_into(line: str, metrics: TrainingMetrics, resume_total: int | None = None) -> None:
    """Update `metrics` in-place from one stdout line.

    Two complementary sources:
      * tqdm progress for current_step + total_steps + ETA (~1s cadence).
      * 'INFO ... step:N smpl:... loss:X grdn:Y lr:Z ...' for loss/lr/grdn
        (only at log_freq cadence, default every 250 steps).

    One `line` can carry MANY tqdm frames. tqdm separates its redraws with \\r,
    and a log transport that doesn't split on \\r hands us the whole burst as one
    line with the trailing 'INFO ... step:N ...' appended (HF Jobs' SSE log
    stream batches ~log_freq frames per message this way; a local subprocess
    read with universal_newlines gets one frame per line). The LAST frame is the
    one the appended INFO line belongs to, so we must take the last match, not
    the first — taking the first understated every step by ~log_freq−1, which
    the abbreviated 'step:4K' token below cannot correct.

    `resume_total` is the run's full step target for a *resumed* run (None for a
    fresh run). On resume lerobot's tqdm bar counts only the remaining window
    (0 → steps−checkpoint), so the raw bar understates the true global step; we
    rebase it to `checkpoint + bar = resume_total − remaining_total + bar` so
    the UI shows e.g. 150/200 instead of 50/100. The `step:N` log line already
    carries the true global step, so it needs no rebasing.
    """
    try:
        tqdm_frames = _TQDM_RE.findall(line)
        if tqdm_frames:
            try:
                raw_step, raw_total, raw_eta = tqdm_frames[-1]
                tqdm_step = int(raw_step)
                total = int(raw_total)
                if resume_total is not None and total > 0:
                    metrics.current_step = resume_total - total + tqdm_step
                    metrics.total_steps = resume_total
                else:
                    metrics.current_step = tqdm_step
                    if total > 0:
                        metrics.total_steps = total
                eta = _parse_duration(raw_eta)
                if eta is not None:
                    metrics.eta_seconds = eta
            except (ValueError, IndexError):
                pass

        if "step:" in line and "loss:" in line:
            # Only useful below 1000 steps: lerobot renders this through
            # format_big_number, so the token becomes "4K" and int() raises —
            # suppressed, leaving the (now correct) tqdm step in place. Don't
            # try to expand the K suffix; it's rounded, hence lossy.
            with contextlib.suppress(ValueError):
                metrics.current_step = int(line.split("step:")[1].split()[0].replace(",", ""))
            with contextlib.suppress(ValueError):
                metrics.current_loss = float(line.split("loss:")[1].split()[0])
            if "lr:" in line:
                with contextlib.suppress(ValueError):
                    metrics.current_lr = float(line.split("lr:")[1].split()[0])
            if "grdn:" in line:
                with contextlib.suppress(ValueError):
                    metrics.grad_norm = float(line.split("grdn:")[1].split()[0])

    except Exception as exc:
        logger.debug("Error parsing log line %r: %s", line, exc)


def _resume_total_steps(config: TrainingRequest) -> int | None:
    """The full step target to rebase a resumed run's tqdm bar against (see
    parse_metrics_into). None for a fresh run — its bar is already global."""
    return config.steps if config.resume else None


def _resume_start_step(config: TrainingRequest) -> int | None:
    """The GLOBAL step a resumed run starts at — the step of the checkpoint it
    continues from. None for a fresh run (which starts at 0 by definition), and
    None when nothing on the request names the checkpoint.

    Complements _resume_total_steps rather than replacing it. Once lerobot's
    tqdm bar exists, parse_metrics_into rebases off the bar's own total
    (`resume_total − remaining_total + bar`), which is the better source
    because it reflects the step lerobot ACTUALLY restored rather than what the
    request asked for. This function covers the window BEFORE the first bar
    frame — dataset scan, video-backend init, checkpoint load — which is many
    seconds even on a small local dataset and minutes on a real one. See
    _initial_metrics for why that window mattered.

    `resume_from_step` is the request's own answer and is preferred. It is
    None when the user resumed "the latest checkpoint": the resolvers
    (_resolve_resume_config_path / _resolve_cloud_resume) pick the checkpoint
    and write their choice back onto the config, so the fallbacks read the step
    out of that — a zero-padded checkpoint dir name either way. A resume driven
    by a hand-supplied `config_path` need not have that layout, hence the digit
    check and the None it can still return.
    """
    if not config.resume:
        return None
    if config.resume_from_step is not None:
        return config.resume_from_step
    if config.resume_from_hub_step and config.resume_from_hub_step.isdigit():
        return int(config.resume_from_hub_step)
    if config.config_path:
        # <output_dir>/checkpoints/<step_dir>/pretrained_model/train_config.json
        step_dir = Path(config.config_path).parent.parent.name
        if step_dir.isdigit():
            return int(step_dir)
    return None


def _initial_metrics(config: TrainingRequest) -> TrainingMetrics:
    """The metrics a job record starts life with.

    A FRESH run starts at 0/0 — genuinely unknown until tqdm speaks, and the UI
    reads total_steps == 0 as "Training starting…".

    A RESUMED run does NOT start at zero, and saying so was the bug: for the
    ~12s (small local dataset) to several minutes (real one) between launching
    and lerobot's first tqdm frame, every progress reading in the app showed
    the run at step 0 — a confident "0 / 60 · 0.0%" where the truth was
    "10 / 60 · 16.7%". Worse, the monitoring chart treats a step going
    BACKWARDS as a new run and clears its history, so the seeded parent-run
    loss curve was wiped on mount by that same 0 before the first frame
    restored it.

    Seeding the known floor fixes every consumer at once, because they all read
    these two fields. It is a floor, not a claim about progress: the parser
    still owns the value from the first tqdm frame onwards.
    """
    start = _resume_start_step(config)
    if start is None or not config.steps:
        return TrainingMetrics()
    return TrainingMetrics(current_step=start, total_steps=config.steps)


def _read_log_metrics(path: Path, resume_total: int | None) -> builtins.list[MetricsHistoryPoint]:
    """Parse one job's log.jsonl into (step, loss, lr, grad_norm) points.

    Feed every line through ONE accumulator rather than a fresh one per line.
    lerobot formats the log-line step with format_big_number, so at >=1000 steps
    its token becomes "1K"/"2K" and int() can't parse it; a fresh-per-line parse
    would leave current_step at 0 and silently drop every point past step 1000.
    Carrying state keeps the exact integer step from the interleaved tqdm lines
    for the loss lines that follow.
    """
    if not path.exists():
        return []
    points: list[MetricsHistoryPoint] = []
    acc = TrainingMetrics()
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                log_line = LogLine.model_validate_json(raw)
            except Exception:
                continue  # skip malformed line, same as read_persisted_logs
            msg = log_line.message
            parse_metrics_into(msg, acc, resume_total)
            # Only the log-freq lines carry loss/lr; tqdm lines just advance the
            # step. Emit a point only when a loss value is present so we don't
            # add a flat point per tqdm tick.
            if "loss:" not in msg or acc.current_step <= 0 or acc.current_loss is None:
                continue
            point = MetricsHistoryPoint(
                step=acc.current_step,
                loss=acc.current_loss,
                lr=acc.current_lr,
                grad_norm=acc.grad_norm,
            )
            # Dedupe by step: overwrite on consecutive same-step lines.
            if points and points[-1].step == point.step:
                points[-1] = point
            else:
                points.append(point)
    return points


class LocalJobRunner:
    """Run a training as a local subprocess.

    The runner is single-shot: instantiate a fresh one per job. Lifetime of
    the underlying subprocess is bounded by this object's existence in memory.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path | None = None,
    ) -> None:
        self._metrics = metrics
        self._process: subprocess.Popen | None = None
        self._log_queue: Queue[LogLine] = Queue()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_file_path = log_file_path
        self._log_file = None  # type: ignore[assignment]
        self._resume_total: int | None = None
        # True only once we have actually signalled a LIVE process. Lets the
        # registry tell "we killed this" from "it had already died", which the
        # exit code alone cannot express.
        self._stop_signalled = False

    def start(
        self,
        job_id: str,
        config: TrainingRequest,
        output_dir: str,
    ) -> None:
        if self._process is not None:
            raise RuntimeError("LocalJobRunner already started")

        self._resume_total = _resume_total_steps(config)

        # Build the command via the helper that lives in train.py.
        from .train import build_training_command  # avoid import cycle at module load

        cmd = build_training_command(config, output_dir, sys.executable)
        logger.info("Starting job %s: %s", job_id, " ".join(cmd))

        # Open the persistent log sink (one JSON line per stdout line). Held
        # open for the subprocess's lifetime so we don't reopen per write.
        if self._log_file_path is not None:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_file_path.open("a", buffering=1)

        # PYTHONUNBUFFERED makes the child's stdout flush per line. Without it
        # block-buffering hides log lines from our parser for many seconds.
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"

        # start_new_session=True puts the child in its own session/process
        # group. Without it, signals sent to the uvicorn worker (e.g. when
        # --reload restarts it on a .py file change) cascade to the child
        # and kill the training. With it, the child survives reloads; the
        # next worker re-attaches via TailingJobRunner using job.json's pid.
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=child_env,
            start_new_session=True,
        )

        self._monitor_thread = threading.Thread(
            target=self._pump_stdout, name=f"job-{job_id}-stdout", daemon=True
        )
        self._monitor_thread.start()

    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def stop(self) -> None:
        # Early return on an already-exited process is what makes
        # stop_signalled() meaningful: the exit code we are about to hand the
        # watchdog is then the process's own, and must stay a failure.
        if self._process is None or self._process.poll() is not None:
            return
        self._stop_event.set()
        self._stop_signalled = True
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess did not terminate in 10s, killing")
                self._process.kill()
                self._process.wait()
        except Exception as exc:
            logger.exception("Error stopping subprocess: %s", exc)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop_signalled(self) -> bool:
        """Whether stop() actually delivered a signal to a live process."""
        return self._stop_signalled

    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def stream_log_lines(self) -> list[LogLine]:
        """Drain whatever has accumulated since the last call."""
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    # -- internals --

    def _pump_stdout(self) -> None:
        assert self._process is not None
        try:
            for line in iter(self._process.stdout.readline, ""):
                if self._stop_event.is_set():
                    break
                stripped = line.rstrip()
                if not stripped:
                    continue
                parse_metrics_into(stripped, self._metrics, self._resume_total)
                log_line = LogLine(timestamp=time.time(), message=stripped)
                if self._log_file is not None:
                    try:
                        self._log_file.write(log_line.model_dump_json() + "\n")
                    except Exception as exc:  # pragma: no cover — best-effort persist
                        logger.exception("Error writing to log file: %s", exc)
                # Cap queue so a chatty subprocess can't grow memory unbounded.
                if self._log_queue.qsize() >= 1000:
                    with contextlib.suppress(Empty):
                        self._log_queue.get_nowait()
                self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Error reading subprocess stdout: %s", exc)
        finally:
            if self._log_file is not None:
                with contextlib.suppress(Exception):
                    self._log_file.close()
                self._log_file = None


class TailingJobRunner:
    """Re-attaches to a detached subprocess after a uvicorn reload.

    We can't recover the original Popen object across processes, so we don't
    own stdout. Instead we tail the persisted log file and watch the pid.
    Implements the JobRunner Protocol so JobRegistry can use it interchangeably
    with LocalJobRunner.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        pid: int,
        resume_total: int | None = None,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._pid = pid
        self._resume_total = resume_total
        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Replay everything that's already on disk so the parser catches up
        # on metrics, then tail from the current EOF.
        self._tail_offset = 0
        # See stop() / returncode(): we have no Popen to reap, so this flag is
        # the only record that the pid's disappearance was our doing.
        self._stop_signalled = False

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # Required by JobRunner Protocol but irrelevant here; the subprocess
        # we're tailing was started by a previous uvicorn worker.
        raise RuntimeError("TailingJobRunner reattaches to an existing pid; use start_tailing() instead")

    def start_tailing(self) -> None:
        if self._tail_thread is not None:
            return
        self._tail_thread = threading.Thread(
            target=self._tail_loop, name=f"job-tail-{self._pid}", daemon=True
        )
        self._tail_thread.start()

    def stop(self) -> None:
        try:
            os.kill(self._pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already gone — we signalled nothing, so its disappearance is not
            # ours to claim and returncode() keeps its optimistic default.
            pass
        else:
            self._stop_signalled = True
        self._stop_event.set()

    def is_running(self) -> bool:
        return _pid_alive(self._pid)

    def stop_signalled(self) -> bool:
        """Whether stop() actually delivered SIGTERM to a live pid."""
        return self._stop_signalled

    def returncode(self) -> int | None:
        # We can't reap a process from another session, so we never learn the
        # real exit code and have to synthesise one.
        #
        # Default is 0 once the pid is gone — the watchdog finalises as "done"
        # rather than "failed", the better default for a detached training that
        # completed normally.
        #
        # After a stop we actually delivered, report SIGTERM instead. A plain 0
        # there would classify a run the user deliberately ended as "done",
        # which is a worse lie than naming the signal we know we sent to a pid
        # that was alive at the time and is now gone.
        if _pid_alive(self._pid):
            return None
        return -signal.SIGTERM if self._stop_signalled else 0

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def pid(self) -> int | None:
        return self._pid

    # -- internals --

    def _tail_loop(self) -> None:
        """Read lines as they arrive in log_file_path. Exits when pid dies
        AND there are no more new lines to read."""
        try:
            while not self._stop_event.is_set():
                if not self._log_file_path.exists():
                    if not _pid_alive(self._pid):
                        return
                    self._stop_event.wait(0.5)
                    continue
                with self._log_file_path.open() as f:
                    f.seek(self._tail_offset)
                    while not self._stop_event.is_set():
                        raw = f.readline()
                        if not raw:
                            self._tail_offset = f.tell()
                            if not _pid_alive(self._pid):
                                return
                            self._stop_event.wait(0.5)
                            continue
                        try:
                            log_line = LogLine.model_validate_json(raw.strip())
                        except Exception:
                            continue
                        parse_metrics_into(
                            log_line.message, self._metrics, self._resume_total
                        )
                        if self._log_queue.qsize() >= 1000:
                            with contextlib.suppress(Empty):
                                self._log_queue.get_nowait()
                        self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Tailing loop error: %s", exc)


class PreparingJobRunner:
    """Stand-in runner for a local job whose base checkpoint is still downloading.

    A local fine-tune from a Hub checkpoint has to materialize multi-GB weights
    before the trainer can start. That used to happen inside POST /jobs/training,
    which blocked the request for minutes with nothing on screen. The record is
    now created first (state "running") and the download runs in a background
    thread; this object stands in for the real LocalJobRunner meanwhile, so the
    registry's existing seams keep working with no special cases:

      * `stream_log_lines` feeds the monitor's 1Hz /jobs/{id}/logs poll — that
        is how download progress reaches the screen. `emit` is the writing end:
        it appends to the SAME log.jsonl the trainer will append to next, so
        the download is part of the run's log rather than a separate channel.
      * `stop` records the user's cancel, which the materialize thread reads
        between the download and the spawn (see
        JobRegistry._materialize_then_start).
      * `is_running` is deliberately always True. The registry ends this
        runner's role by REPLACING it in `_runners` (handoff) or removing it
        (finalisation), both under the registry lock. Answering False would let
        a watchdog tick that raced the handoff finalise the job as failed while
        the trainer it had just spawned kept running.

    Not a real process: `returncode` is always None, and `start` is never called
    (the registry constructs this runner directly).
    """

    def __init__(self, log_file_path: Path | None = None) -> None:
        self._log_file_path = log_file_path
        self._log_queue: Queue[LogLine] = Queue()
        self._cancelled = threading.Event()

    def emit(self, message: str) -> None:
        """Append one line to the job's log, for the file and the live poll.

        Opens and closes the file per line on purpose: this runs at most every
        few seconds, and holding no handle keeps the file free for
        LocalJobRunner to open in append mode at handoff."""
        line = LogLine(timestamp=time.time(), message=message)
        if self._log_file_path is not None:
            try:
                self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_file_path.open("a") as f:
                    f.write(line.model_dump_json() + "\n")
            except Exception as exc:  # pragma: no cover — best-effort persist
                logger.exception("Error writing to log file: %s", exc)
        # Same cap as LocalJobRunner: never grow unbounded if nobody polls.
        if self._log_queue.qsize() >= 1000:
            with contextlib.suppress(Empty):
                self._log_queue.get_nowait()
        self._log_queue.put(line)

    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    # -- JobRunner protocol --

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        raise RuntimeError("PreparingJobRunner stands in for a runner; it starts nothing")

    def stop(self) -> None:
        # A huggingface_hub download can't be interrupted mid-flight, so this
        # only records the intent; the materialize thread acts on it when the
        # download returns.
        self._cancelled.set()

    def is_running(self) -> bool:
        return True

    def returncode(self) -> int | None:
        return None

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out


_PERSIST_THROTTLE_SECONDS = 1.0


def _list_local_checkpoints(output_dir: str) -> list[JobCheckpoint]:
    """Scan an output dir for valid checkpoint subdirectories.

    A directory under <output_dir>/checkpoints/ is a valid checkpoint iff
    its name parses to an int and it contains pretrained_model/config.json.
    """
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return []
    out: list[JobCheckpoint] = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            step = int(entry.name)
        except ValueError:
            continue
        config_json = entry / "pretrained_model" / "config.json"
        if not config_json.is_file():
            continue
        out.append(
            JobCheckpoint(
                step=step,
                source="local",
                ref=str((entry / "pretrained_model").resolve()),
            )
        )
    out.sort(key=lambda c: c.step)
    return out


# lerobot writes this per-checkpoint config inside pretrained_model/; resuming
# needs it as --config_path so lerobot can reconstruct the run.
_TRAIN_CONFIG_NAME = "train_config.json"


# What a COMPLETE lerobot checkpoint looks like, and how to test one.
#
# save_checkpoint writes a checkpoint over several seconds in a fixed order:
# pretrained_model/config.json, then the weights, then train_config.json, then
# training_state/ (training_step.json, rng_state, and the large
# optimizer_state.safetensors last). A directory holding only the leading files
# is a snapshot of a save IN PROGRESS, not a resumable checkpoint — the cloud
# uploader used to publish exactly that and seal it forever, leaving Hub
# checkpoints that pass a naive guard and then die inside the trainer on
# `training_state/optimizer_state.safetensors`.
#
# Both helpers below must stay SELF-CONTAINED (stdlib only, no module-level
# names, no annotations): runners/hf_cloud.py inlines their source verbatim
# into the in-container HF Jobs wrapper the same way it inlines _install_plan,
# so the uploader's readiness rule is exactly the one these tests exercise.


def scan_checkpoint_dir(checkpoint_dir):
    """Inspect one checkpoints/<step>/ directory on disk.

    Returns (names, fingerprint): `names` is every file below it as a relative
    posix path ("training_state/training_step.json"); `fingerprint` is the
    sorted (path, size, mtime_ns) tuple of the same files. Comparing the
    fingerprint across two polls is how the uploader tells "finished" from
    "still being written" without having to know lerobot's exact file list.

    Unreadable entries are skipped rather than raised on — the walk races an
    in-flight save, and safetensors' atomic-write temp files vanish mid-scan.
    """
    names = set()
    fingerprint = []
    for path in sorted(checkpoint_dir.rglob("*")):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(checkpoint_dir).as_posix()
        names.add(rel)
        fingerprint.append((rel, stat.st_size, stat.st_mtime_ns))
    return names, tuple(fingerprint)


def missing_checkpoint_files(names):
    """Which required artifacts are absent from `names` (relative posix paths
    inside one checkpoints/<step>/). An empty list means complete and resumable.

    scheduler_state.json is deliberately NOT required: save_training_state
    writes it only `if scheduler is not None`, so its presence is a property of
    the policy preset rather than of completeness. The weights are matched by
    suffix because the filename varies (model.safetensors, or a PEFT adapter),
    and optimizer_state.safetensors is matched at ANY depth under
    training_state/ because a MultiAdam policy nests one directory per
    optimizer there.
    """
    missing = []
    if "pretrained_model/config.json" not in names:
        missing.append("pretrained_model/config.json")
    if not any(n.startswith("pretrained_model/") and n.endswith(".safetensors") for n in names):
        missing.append("pretrained_model/*.safetensors")
    if "pretrained_model/train_config.json" not in names:
        missing.append("pretrained_model/train_config.json")
    if "training_state/training_step.json" not in names:
        missing.append("training_state/training_step.json")
    if not any(
        n.startswith("training_state/") and n.rsplit("/", 1)[-1] == "optimizer_state.safetensors"
        for n in names
    ):
        missing.append("training_state/optimizer_state.safetensors")
    return missing


# Shared tail for both resume refusals: name the remedy, since "incomplete
# checkpoint" is otherwise a dead end for the user.
_INCOMPLETE_REMEDY = "Resume an earlier checkpoint, or fine-tune from its weights instead."

# Plain-language names for the runner ids, so a user-facing refusal reads like
# the Compute control the user actually clicked rather than an internal literal.
_RUNNER_LABELS = {"local": "Local — your machine", "hf_cloud": "Hugging Face Cloud"}


# Plain-language names for the runner ids, so a user-facing refusal reads like
# the Compute control the user actually clicked rather than an internal literal.
_RUNNER_LABELS = {"local": "Local — your machine", "hf_cloud": "Hugging Face Cloud"}


def hub_checkpoint_has_training_state(api, repo_id: str, step_dir: str) -> bool:
    """Is checkpoints/<step_dir>/ on `repo_id` resumable — i.e. did its
    training_state/ (optimizer + step counter) reach the Hub?

    Reads the repo's file listing (no bytes), so an unresumable checkpoint can
    be refused before anything is downloaded or a GPU is rented.
    `_HUB_TRAINING_STATE_FILE` is the cheapest existence probe for that subtree,
    the same one the cloud runner's uploader uses.

    Shared by every consumer of a Hub checkpoint's completeness so they agree:
    the cloud→cloud resolver, the cloud→local resolver, and the re-use check
    that decides whether a local checkpoint still sits on the Hub from an
    earlier local→cloud resume.

    Raises ValueError when the listing itself can't be read — an unverifiable
    checkpoint is refused rather than assumed good (that assumption is MT4's
    failure mode: the run dies inside the trainer instead of at the form).
    """
    try:
        files = set(api.list_repo_files(repo_id, repo_type="model"))
    except Exception as exc:
        raise ValueError(
            f"Could not read {repo_id!r} to verify its checkpoint at step {step_dir}: {exc}"
        ) from exc
    return f"checkpoints/{step_dir}/{_HUB_TRAINING_STATE_FILE}" in files


def _resolve_cloud_resume(source: JobRecord, step: int | None) -> tuple[str, str]:
    """Return (repo_id, step_dir) identifying the Hub checkpoint a run continuing
    a CLOUD parent resumes from (`step` = None ⇒ the latest available on the Hub).

    Whoever runs the trainer downloads checkpoints/<step_dir>/ (both
    pretrained_model/ and training_state/) from `repo_id` and hands lerobot the
    reconstructed output-dir layout, so resume restores the optimizer and step
    counter — true resume, not a weights-only re-init. That download happens
    pod-side for a cloud→cloud continuation (the HF Jobs wrapper) and host-side
    for a cloud→local one (download_hub_resume_checkpoint); this resolver only
    NAMES the checkpoint, identically for both, which is why the cross-runner
    direction needs no second resolver.

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    can't be resumed from the Hub: not a cloud run, no output repo, no
    checkpoints at all (the run died before its first save), an unknown step, or
    a checkpoint that is only partly on the Hub (missing weights or any of the
    training_state/ files — see missing_checkpoint_files).
    """
    if source.runner != "hf_cloud":
        raise ValueError(
            "This resume path is for cloud runs; local runs resume from their on-disk checkpoint instead."
        )
    if not source.hf_repo_id:
        raise ValueError(f"Cloud run {source.id!r} has no output repo on the Hub to resume from.")
    api = shared_hf_api()
    # Keep only the checkpoints/<step>/ entries. A repo with no checkpoint tree
    # but a policy at its root lists a single '@root' entry (_list_hub_checkpoints'
    # fallback) — deployable and fine-tunable, but root weights carry no
    # training_state/, so there is nothing to resume from. Filtering here keeps
    # the refusal a plain-language one instead of the ref-shape error below.
    listed = _list_hub_checkpoints(api, source.hf_repo_id)
    checkpoints = [c for c in listed if _HUB_CKPT_REF_RE.match(c.ref)]
    if not checkpoints:
        if listed:
            raise ValueError(
                f"Cloud run {source.id!r} published a final policy but saved no "
                "checkpoints, so it has no optimizer state to resume from. "
                "Fine-tune from its weights instead."
            )
        raise ValueError(
            f"Cloud run {source.id!r} left no checkpoints on the Hub — nothing to "
            "resume from (the run died before its first save)."
        )
    if step is None:
        chosen = checkpoints[-1]  # step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Cloud run {source.id!r} has no checkpoint at step {step}.")
    # chosen.ref is 'repo@checkpoints/<step_dir>'; recover the zero-padded dir.
    m = _HUB_CKPT_REF_RE.match(chosen.ref)
    if not m:
        raise ValueError(f"Unexpected checkpoint ref for cloud run {source.id!r}: {chosen.ref!r}")
    step_dir = m.group("step_dir")
    try:
        files = set(api.list_repo_files(source.hf_repo_id, repo_type="model"))
    except Exception as exc:
        raise ValueError(
            f"Could not read cloud run {source.id!r}'s repo to verify the "
            f"checkpoint at step {chosen.step}: {exc}"
        ) from exc
    prefix = f"checkpoints/{step_dir}/"
    missing = missing_checkpoint_files({f[len(prefix) :] for f in files if f.startswith(prefix)})
    if missing:
        raise ValueError(
            f"Checkpoint at step {chosen.step} is incomplete on the Hub (a known "
            f"uploader race) — missing {', '.join(missing)}. {_INCOMPLETE_REMEDY}"
        )
    return source.hf_repo_id, step_dir


def _resolve_resume_config_path(source: JobRecord, step: int | None) -> str:
    """Return the train_config.json path lerobot needs to resume a LOCAL
    `source` from `step` (or its latest checkpoint if step is None).

    The path names a real directory on this machine, so it is what a local
    continuation passes as --config_path. A CLOUD continuation of the same
    local parent uses it differently: `<path>.parent.parent` is the
    checkpoints/<step>/ directory whose bytes get uploaded to the Hub (see
    JobRegistry._resolve_upload_resume), because the pod cannot see this disk.
    Either way the validation below is the gate, which is why both directions
    come through here.

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    can't be resumed: not a local run, no checkpoints, unknown step, a
    weights-only checkpoint missing the training_state/ (optimizer + step)
    needed to continue, or a checkpoint left partly written by an interrupted
    save.
    """
    if source.runner != "local":
        # Not a claim about lerobot — the pin resumes from a Hub repo id just
        # fine (TrainPipelineConfig._resolve_resume_checkpoint downloads the
        # LATEST checkpoint of `--config_path=<repo id>`). It's a claim about
        # THIS function: a cloud parent's checkpoints are on the Hub, so it is
        # _resolve_cloud_resume that names them, and the chosen step is
        # materialized explicitly rather than left to lerobot's latest-only
        # rule (which would silently ignore the step the user picked).
        raise ValueError(
            "This resume path is for local runs; a cloud run's checkpoints live "
            "on the Hub and are resolved from there instead."
        )
    checkpoints = _list_local_checkpoints(source.output_dir)
    if not checkpoints:
        raise ValueError(f"Run {source.id!r} has no saved checkpoints to resume from.")
    if step is None:
        chosen = checkpoints[-1]  # list is step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Run {source.id!r} has no checkpoint at step {step}.")
    # chosen.ref is <output_dir>/checkpoints/<step>/pretrained_model
    pretrained_dir = Path(chosen.ref)
    checkpoint_dir = pretrained_dir.parent
    train_config = pretrained_dir / _TRAIN_CONFIG_NAME
    training_state = checkpoint_dir / "training_state"
    if not train_config.is_file():
        raise ValueError(
            f"Checkpoint at step {chosen.step} is missing {_TRAIN_CONFIG_NAME}, so it can't be resumed."
        )
    # No training_state/ at all is the weights-only shape (an imported model),
    # which deserves its own wording; a training_state/ that exists but is
    # short of files is an interrupted save and gets the incomplete message.
    if not training_state.is_dir():
        raise ValueError(
            f"Checkpoint at step {chosen.step} has no optimizer/step state "
            "(training_state/), so it can't be resumed. Weights-only models "
            "(e.g. imported) can only start a fresh run."
        )
    names, _fingerprint = scan_checkpoint_dir(checkpoint_dir)
    missing = missing_checkpoint_files(names)
    if missing:
        raise ValueError(
            f"Checkpoint at step {chosen.step} is incomplete — missing "
            f"{', '.join(missing)}. {_INCOMPLETE_REMEDY}"
        )
    return str(train_config.resolve())


def _resolve_finetune_pretrained_path(source: JobRecord, step: int | None) -> str:
    """Return a `--policy.pretrained_path` value that initializes a FRESH run's
    weights from `source`'s checkpoint at `step` (or its latest if step is None).

    Unlike resume, this does NOT require training_state/ — weights-only is the
    whole point of fine-tuning. lerobot's PreTrainedConfig.pretrained_path loads
    the policy weights (and processors) from a local pretrained_model dir or a
    Hub repo on a non-resume run.

    Handles every source shape via its own checkpoint listing:
      * imported local / normal local run → a `local` ref that is the absolute
        pretrained_model dir; returned directly (e.g. a flat imported dir
        becomes a step-0 checkpoint whose ref is the dir itself).
      * imported hub / cloud run at a specific step → the step-suffixed hub ref
        'repo@checkpoints/<step_dir>', VERBATIM. lerobot can't load a hub
        sub-path directly, so the ref is materialized into a real directory
        before the trainer starts — host-side by localize_pretrained_path for a
        local run, pod-side by the HF Jobs wrapper for a cloud one. This
        function does not download: it names the checkpoint, and the caller
        materializes it wherever the trainer will run (MT2).
      * imported hub / cloud run whose checkpoint is the whole repo ('repo@root')
        → the plain repo id. The root IS the pretrained_model, which lerobot
        resolves by itself, so there is nothing to materialize.

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    has no usable checkpoint.
    """
    if source.runner == "imported":
        if source.hf_repo_id:
            checkpoints = _list_imported_hub(shared_hf_api(), source.hf_repo_id)
        else:
            checkpoints = _list_imported_local(source.output_dir)
    elif source.runner == "local":
        checkpoints = _list_local_checkpoints(source.output_dir)
    else:  # hf_cloud
        checkpoints = _list_hub_checkpoints(shared_hf_api(), source.hf_repo_id)

    if not checkpoints:
        raise ValueError(f"Source {source.id!r} has no usable checkpoint to fine-tune from.")
    if step is None:
        chosen = checkpoints[-1]  # step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Source {source.id!r} has no checkpoint at step {step}.")

    if chosen.source == "local":
        # chosen.ref is the absolute pretrained_model dir lerobot loads directly.
        return chosen.ref
    if _HUB_ROOT_REF_RE.match(chosen.ref):
        # The repo root IS the model; lerobot loads a repo id as-is, so hand back
        # the repo portion and let it fetch. Nothing to materialize.
        return chosen.ref.split("@", 1)[0]
    # A specific step: keep the whole ref. Truncating it here is what MT2 was —
    # the run silently trained from repo-ROOT weights while the UI reported the
    # step the user picked. The ref is turned into a directory downstream, on
    # whichever machine will run the trainer.
    return chosen.ref


# register_imported's fallback when a checkpoint's config.json can't be read.
# It's a display label, not an architecture, so it can never be compared
# against a requested policy type.
_UNKNOWN_POLICY_TYPE = "model"


def _check_finetune_policy_type(source: JobRecord, requested: str) -> None:
    """Reject a fine-tune whose requested policy type contradicts its source.

    A fine-tune is launched as ``--policy.type <requested>`` plus
    ``--policy.pretrained_path <source checkpoint>`` (see
    train.build_training_command). lerobot builds the policy class from
    ``--policy.type`` and then loads the checkpoint's safetensors NON-strictly
    (``PreTrainedPolicy.from_pretrained`` defaults ``strict=False``, and
    ``make_policy`` doesn't override it), so a mismatched pair does not fail
    loudly — it trains an essentially randomly-initialized `requested` policy
    while the run claims to be a fine-tune of `source`. Fail up front instead.

    Only compared when the source record carries a real architecture: imported
    models fall back to the `_UNKNOWN_POLICY_TYPE` placeholder when their
    config.json couldn't be read, and that says nothing about the weights.

    Raises ValueError (→ HTTP 400) naming both types.
    """
    source_type = (source.config.policy_type or "").strip()
    requested_type = (requested or "").strip()
    if not source_type or source_type == _UNKNOWN_POLICY_TYPE:
        return
    if not requested_type or requested_type == source_type:
        return
    raise ValueError(
        f"This run is set to train {requested_type!r}, but the fine-tune source "
        f"{source.id!r} is a {source_type!r} checkpoint. Fine-tuning loads the "
        f"source's weights into the policy you pick, so the two must match — "
        f"set the policy to {source_type!r}, or pick a {requested_type!r} base "
        "model."
    )


def read_pretrained_policy_type(pretrained_path: str) -> str | None:
    """Read the architecture of the checkpoint at ``--policy.pretrained_path``.

    `pretrained_path` is whatever the run will be started from: an absolute
    local ``pretrained_model`` directory, a Hub repo id whose ROOT holds the
    model, or a step-suffixed hub ref ('repo@checkpoints/<step_dir>') that has
    not been materialized yet — the check runs BEFORE the download, so that a
    contradicting pair is refused without first pulling gigabytes.

    Only the checkpoint's ``config.json`` is fetched in every case (a few KB).

    Returns the ``type`` field of the checkpoint's own ``config.json``, or None
    when it can't be read — missing file, malformed JSON, private/absent repo,
    or no network. None means "not established", never "fine": callers must
    treat it as a reason to stay silent rather than a clean bill of health.
    """
    path = Path(pretrained_path)
    if path.is_dir():
        with contextlib.suppress(Exception), open(path / "config.json") as f:
            return _clean_policy_type(json.load(f).get("type"))
        return None

    # A step-suffixed ref addresses one checkpoint inside the repo; anything
    # else is a repo id whose root holds the config.
    m = _HUB_CKPT_REF_RE.match(pretrained_path)
    if m:
        repo_id = m.group("repo")
        filename = f"checkpoints/{m.group('step_dir')}/{_HUB_CKPT_SUBDIR}/config.json"
    else:
        repo_id, filename = pretrained_path, "config.json"

    with contextlib.suppress(Exception):
        local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        with open(local) as f:
            return _clean_policy_type(json.load(f).get("type"))
    return None


def _clean_policy_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _check_pretrained_policy_type(pretrained_path: str, requested: str) -> None:
    """Reject a run whose ``--policy.type`` contradicts the checkpoint it loads.

    The last line of defence before the trainer is spawned, and the only one
    that consults the WEIGHTS' OWN config rather than MakerMods Lab's bookkeeping
    about them. That matters twice over:

    * ``_check_finetune_policy_type`` compares the source JobRecord's recorded
      ``policy_type``, which is absent for a hand-built request and is the
      `_UNKNOWN_POLICY_TYPE` placeholder whenever an import couldn't read its
      config — precisely the cases it therefore skips.
    * ``policy_pretrained_path`` is a plain field of the public
      ``TrainingRequest``, so a caller can POST it directly with no
      ``finetune_from_job_id`` at all and bypass that guard entirely.

    MakerMods Lab cannot make the load itself strict — it shells out to
    ``lerobot_train``, and ``make_policy`` hardcodes the non-strict default with
    no CLI surface to override — so validating the pair up front is the only
    available equivalent. See makermodslab/finetune_audit.py for why a mismatch is
    unrecoverable after the fact: the architectures share no parameter names, so
    nothing is loaded and the written checkpoint is indistinguishable from a
    from-scratch run.

    Silent when the checkpoint's architecture can't be read: an unverifiable
    source must not block a launch. Raises ValueError (→ HTTP 400) only on a
    contradiction we can actually prove.
    """
    requested_type = _clean_policy_type(requested)
    if not requested_type:
        return
    actual = read_pretrained_policy_type(pretrained_path)
    if actual is None or actual == requested_type:
        return
    raise ValueError(
        f"This run is set to train {requested_type!r}, but the checkpoint it "
        f"starts from ({pretrained_path}) is a {actual!r} model. lerobot loads "
        f"those weights into a {requested_type!r} policy without checking, and "
        f"the two share no parameters — the run would silently train a brand-new "
        f"{requested_type!r} while reporting itself as a fine-tune. Set the "
        f"policy to {actual!r}, or pick a {requested_type!r} base model."
    )


_CLOUD_CKPT_TTL_SECONDS = 30.0
_CKPT_PATH_RE = re.compile(r"^checkpoints/(\d+)/pretrained_model/config\.json$")


def _hub_checkpoints_from_files(files, repo_id: str) -> list[JobCheckpoint]:
    """Parse a repo file listing into checkpoints. The ref preserves the
    original zero-padded directory name (e.g. 000050); JobCheckpoint.step is
    the int form for sorting and UI display."""
    seen: dict[int, JobCheckpoint] = {}
    for path in files:
        m = _CKPT_PATH_RE.match(path)
        if not m:
            continue
        step_dir = m.group(1)
        step = int(step_dir)
        seen[step] = JobCheckpoint(
            step=step,
            source="hub",
            ref=f"{repo_id}@checkpoints/{step_dir}",
        )
    out = list(seen.values())
    out.sort(key=lambda c: c.step)
    return out


def _list_imported_local(path: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported local directory.

    A checkpoints/<step>/pretrained_model tree → reuse _list_local_checkpoints.
    Otherwise, if the dir itself is a pretrained_model (config.json present) →
    a single step-0 checkpoint. Neither → empty (source moved/unusable)."""
    tree = _list_local_checkpoints(path)
    if tree:
        return tree
    if (Path(path) / "config.json").is_file():
        return [JobCheckpoint(step=0, source="local", ref=str(Path(path).resolve()))]
    return []


def _list_imported_hub(api, repo_id: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported Hub model repo.

    A checkpoints/<step>/pretrained_model tree → the tree parse. Otherwise, a
    root config.json → a single step-0 checkpoint with a 'repo@root' ref (the
    whole repo is the pretrained_model).

    That is now exactly _list_hub_checkpoints' rule — a trained run's repo and
    an imported one are the same two layouts — so this delegates rather than
    keeping a second copy that can drift. The name survives because call sites
    (register_imported, _checkpoints_for) pass it explicitly to say WHICH kind
    of record they are listing for."""
    return _list_hub_checkpoints(api, repo_id)


def _list_hub_checkpoints(api, repo_id: str) -> list[JobCheckpoint]:
    """List checkpoints by introspecting the model repo file tree.

    Falls back to the repo ROOT when there is no checkpoints/ tree but the root
    holds a policy (config.json) — the same fallback _list_imported_hub applies,
    for the same reason: a run trained with checkpoint saving off still pushes
    its final policy to the root at the end of training, and without this the
    job card reports zero checkpoints while a loadable model sits in the repo.
    The '@root' ref is what the inference handler already resolves for imported
    flat repos (rollout._resolve_policy_path), so the entry is runnable, not
    merely listed. Resume is the one thing it can't do (root weights carry no
    training_state/) — see _resolve_cloud_resume, which says so explicitly."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        # Repo may not exist yet (training just started, sidecar hasn't
        # uploaded anything). Treat as no checkpoints.
        return []
    tree = _hub_checkpoints_from_files(files, repo_id)
    if tree:
        return tree
    if "config.json" in files:
        return [JobCheckpoint(step=0, source="hub", ref=f"{repo_id}@root")]
    return []


_LANGUAGE_CONDITIONED_POLICY_TYPES = {"smolvla", "pi0", "pi0_fast", "pi05"}


_HUB_CKPT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")

# What a hub ref names inside the snapshot, once downloaded.
_HUB_CKPT_SUBDIR = "pretrained_model"


def make_snapshot_progress_tqdm(
    report: Callable[[int, int | None], None],
) -> type[_base_tqdm]:
    """A ``tqdm_class`` for ``snapshot_download`` that reports byte progress.

    Lives here, next to ``download_hub_checkpoint_ref``, because both consumers
    of a Hub download need it: the inference page's progress bar (rollout.py,
    which imports it from this module) and the local fine-tune's
    base-checkpoint download (JobRegistry._materialize_then_start). Verified
    against the pinned huggingface_hub 1.21.0 contract:
    ``snapshot_download(tqdm_class=cls)`` instantiates ``cls`` twice — a
    file-count bar and ONE shared bytes bar (``unit="B"``). Both the plain-HTTP
    and xet download paths funnel their chunk updates into that shared bar: as
    each file's metadata arrives its size is added by mutating ``bar.total`` in
    place followed by ``bar.refresh()``, and downloaded chunks arrive as
    ``bar.update(n)``. So the recorder keys off ``unit == "B"``, hooks
    ``update`` for bytes done, and hooks ``refresh`` as the signal that the
    (growing) total changed. The total keeps growing while file metadata is
    discovered, so percent can legitimately drop — honest, since the real total
    isn't known upfront.

    `report` is called from huggingface_hub's download worker threads, so an
    implementation that touches shared state must do its own locking.

    Subclasses the vanilla tqdm on purpose: huggingface_hub hands non-hf
    subclasses full responsibility (no ``disable``/``name`` injection, no
    HF_HUB_DISABLE_PROGRESS_BARS gating), so reporting can't be silently turned
    off by env/log-level. The bar itself is force-disabled — nothing is drawn to
    the server's stderr — which also means tqdm's own ``n`` never advances; bytes
    are accumulated in ``_bytes_done`` instead. ``total`` IS still set and mutable
    on a disabled tqdm, which is all ``refresh`` needs to read."""

    class _ProgressTqdm(_base_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._is_bytes_bar = kwargs.get("unit") == "B"
            self._bytes_done = int(kwargs.get("initial") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def _report(self) -> None:
            total = getattr(self, "total", None)
            report(self._bytes_done, int(total) if total else None)

        def update(self, n: float | None = 1) -> bool | None:
            if self._is_bytes_bar:
                if n:
                    self._bytes_done += int(n)
                self._report()
            return super().update(n)

        def refresh(self, *args: Any, **kwargs: Any) -> bool | None:
            if self._is_bytes_bar:
                self._report()
            return super().refresh(*args, **kwargs)

    return _ProgressTqdm


def _format_bytes(n: int) -> str:
    """Human byte size for a log line: 540 MB, 1.2 GB."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


# Written to `error_message` when a fine-tune is stopped during its
# base-checkpoint download — before any trainer existed, so there is no exit
# code and no crash to report, and leaving the field None would show the user
# nothing at all where the card says why the run ended.
_PREPARE_STOPPED_MESSAGE = "Stopped at your request, before training started."


# How far apart two download-progress log lines have to be to be worth writing.
# Either condition is enough. snapshot_download calls back per CHUNK (thousands
# of times for a multi-GB checkpoint) and every line is a JSON write plus a row
# in the monitor's log panel, so an unthrottled hook would drown the log the
# progress is meant to make readable.
_DOWNLOAD_LOG_PERCENT_STEP = 5.0
_DOWNLOAD_LOG_INTERVAL_SECONDS = 5.0


class _DownloadProgressLogger:
    """Turn snapshot_download's byte callbacks into occasional job-log lines.

    Shaped for `make_snapshot_progress_tqdm`'s `report` contract, which fires
    from several download worker threads at once — hence the lock, which also
    keeps the emitted lines in ascending order.

    A total of None (huggingface_hub is still discovering file sizes) is
    reported as bytes-so-far rather than a fake percentage; the total also
    grows as metadata arrives, so the percentage can legitimately go down."""

    def __init__(self, emit: Callable[[str], None], label: str) -> None:
        self._emit = emit
        self._label = label
        self._lock = threading.Lock()
        self._last_at = 0.0
        self._last_percent: float | None = None
        self._last_bytes = 0

    def __call__(self, bytes_done: int, bytes_total: int | None) -> None:
        with self._lock:
            if bytes_done <= self._last_bytes:
                # A refresh with no new bytes (the total changed, or a bar was
                # rebuilt). Nothing to say.
                return
            now = time.monotonic()
            percent = (bytes_done / bytes_total * 100) if bytes_total else None
            due = (now - self._last_at) >= _DOWNLOAD_LOG_INTERVAL_SECONDS
            if percent is not None:
                due = (
                    due
                    or self._last_percent is None
                    or (percent - self._last_percent) >= _DOWNLOAD_LOG_PERCENT_STEP
                )
            if not due:
                return
            self._last_at = now
            self._last_percent = percent
            self._last_bytes = bytes_done
            if percent is None:
                message = f"Downloading base checkpoint {self._label} — {_format_bytes(bytes_done)} so far"
            else:
                message = (
                    f"Downloading base checkpoint {self._label} — {percent:.0f}% "
                    f"({_format_bytes(bytes_done)} / {_format_bytes(bytes_total or 0)})"
                )
            self._emit(message)


def download_hub_checkpoint_ref(ref: str, *, tqdm_class=None, with_training_state: bool = False) -> str:
    """Download the model a hub checkpoint ref names; return its local dir.

    THE resolution rule for a hub ref, shared by every consumer so a ref means
    one thing everywhere:

      * 'repo@checkpoints/<step_dir>' → only that step's ``pretrained_model/``
        is pulled, and the returned path is that directory. lerobot's
        ``pretrained_path`` addresses a local dir or a repo ROOT and has no
        subfolder argument (``PreTrainedConfig.from_pretrained`` hands the id
        straight to ``hf_hub_download``), so materializing the sub-path here is
        the only way a specific step can be loaded at all.
      * 'repo@root' → the whole repo IS the pretrained_model; ``checkpoints/**``
        and ``training_state/**`` are excluded because neither is needed to load
        the policy and both can be multi-GB.

    ``with_training_state`` widens the step-ref case to the WHOLE
    checkpoints/<step_dir>/ tree — the optimizer/rng/step state a RESUME needs
    and a deploy or fine-tune never reads (hundreds of MB per step, so it stays
    opt-in). The return value is unchanged (still the ``pretrained_model/``
    dir); its parent is then the reconstructed checkpoint dir. See
    download_hub_resume_checkpoint, the only caller that wants it.

    ``tqdm_class`` is forwarded to snapshot_download for byte-progress reporting
    (the inference page's download bar). This function itself has no session or
    UI state — callers own their own progress/phase reporting — so it is safe to
    call from a training request without touching a live rollout.

    Raises ValueError for anything that isn't a hub ref. Downloads can take
    minutes and pull GBs: never call it while holding a lock others need.
    """
    # Imported at CALL time, not module scope, so that patching
    # `huggingface_hub.snapshot_download` intercepts it — the seam the inference
    # download tests already use, and the same reason _read_checkpoint_config
    # re-imports hf_hub_download.
    from huggingface_hub import snapshot_download

    dl_kwargs: dict[str, Any] = {}
    if tqdm_class is not None:
        dl_kwargs["tqdm_class"] = tqdm_class
    m = _HUB_CKPT_REF_RE.match(ref)
    if m:
        repo_id, step_dir = m.group("repo"), m.group("step_dir")
        subtree = "" if with_training_state else f"/{_HUB_CKPT_SUBDIR}"
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step_dir}{subtree}/*"],
            **dl_kwargs,
        )
        return str(Path(local_root) / "checkpoints" / step_dir / _HUB_CKPT_SUBDIR)
    m = _HUB_ROOT_REF_RE.match(ref)
    if m:
        return snapshot_download(
            repo_id=m.group("repo"),
            repo_type="model",
            ignore_patterns=["checkpoints/**", "training_state/**"],
            **dl_kwargs,
        )
    raise ValueError(f"Unrecognised policy ref: {ref!r}")


def download_hub_resume_checkpoint(ref: str, *, tqdm_class=None) -> str:
    """Materialize a Hub checkpoint HERE so a local trainer can resume from it,
    and return the --config_path lerobot wants (its train_config.json).

    The host-side port of the HF Jobs wrapper's resume block (F7, cloud→local):
    pull checkpoints/<step_dir>/ whole — pretrained_model/ AND training_state/ —
    so lerobot finds the optimizer, scheduler, rng and step counter where its own
    resume path looks for them. lerobot reads `config_path.parent.parent` as the
    checkpoint dir, which the snapshot's own layout already satisfies, so unlike
    the wrapper this does NOT copy the tree into the run's output dir:

      * the trainer only READS the resumed checkpoint (lerobot_train's
        load_training_state; update_last_checkpoint runs on the checkpoints it
        WRITES, under --output_dir), so a shared-cache path is enough;
      * the weights half then stays in the shared HF cache where a later deploy
        of the same checkpoint reuses it instead of pulling it again;
      * and the child's own checkpoints/ tree stays free of a step it did not
        produce.

    Refuses (ValueError → the job's error_message) a checkpoint that arrives
    incomplete, rather than letting the trainer die on a missing training_state/
    halfway through startup — MT4's failure mode. The Hub listing is checked
    before this runs (hub_checkpoint_has_training_state); this second check is on
    the bytes that actually landed.

    Downloads GBs: never call it while holding a lock others need.
    """
    pretrained_dir = Path(download_hub_checkpoint_ref(ref, tqdm_class=tqdm_class, with_training_state=True))
    checkpoint_dir = pretrained_dir.parent
    train_config = pretrained_dir / _TRAIN_CONFIG_NAME
    missing = [
        name
        for name, path in (
            (_TRAIN_CONFIG_NAME, train_config),
            (_HUB_TRAINING_STATE_FILE, checkpoint_dir / _HUB_TRAINING_STATE_FILE),
        )
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            f"The downloaded checkpoint {hub_ref_step_label(ref)} from "
            f"{hub_ref_repo_id(ref)} is incomplete — missing {', '.join(missing)}. "
            "Resume an earlier checkpoint, or fine-tune from its weights instead."
        )
    return str(train_config)


def upload_local_checkpoint(checkpoint_dir: Path, repo_id: str, step_dir: str, *, api=None) -> None:
    """Push one local checkpoints/<step_dir>/ tree to `repo_id` on the Hub.

    F7's local→cloud direction: a pod cannot see this machine's disk, so the
    parent's checkpoint has to exist on the Hub before the continuation is
    submitted. The whole tree goes up (pretrained_model/ AND training_state/) —
    a resume without the optimizer state is a fine-tune wearing a resume label.

    PRIVATE at creation, deliberately and unlike the dataset uploader: this is a
    by-product of clicking Continue, not a model the user chose to publish, and
    `exist_ok` never downgrades an existing repo's visibility. The caller is
    responsible for having asked first (see JobRegistry's consent gate) — this
    function only moves the bytes.

    Laid out to match what _list_hub_checkpoints/_resolve_cloud_resume expect to
    find, so the uploaded step reads back as an ordinary Hub checkpoint.
    """
    api = api or shared_hf_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo=f"checkpoints/{step_dir}",
        commit_message=f"checkpoint {step_dir} (uploaded for a cloud continuation)",
    )


def checkpoints_staging_repo_id(username: str, job_id: str) -> str:
    """The Hub repo a LOCAL run's checkpoints are staged in for a cloud resume.

    Follows hf_cloud's own minting convention for a run's repo
    (`f"{username}/{job_id}"`) plus a `_checkpoints` suffix, which keeps two
    things true that matter:

      * it can never collide with the output repo of a cloud run of the same
        name, so a staging upload can't land in a repo full of someone else's
        lineage; and
      * the continuation it feeds pushes to its OWN output repo rather than back
        into this one, so parent and child checkpoints stay distinguishable.
    """
    return f"{username}/{job_id}_checkpoints"


def needs_local_materialization(pretrained_path: str) -> bool:
    """Would `localize_pretrained_path` have to DOWNLOAD this value?

    The cheap predicate twin of that function's early return, so the start path
    can decide — without touching the network — whether a job needs the
    background materialization step (see JobRegistry.start). Keep the two in
    step: anything this answers False for must pass straight through there."""
    return bool(_HUB_CKPT_REF_RE.match(pretrained_path))


def hub_ref_step_label(ref: str) -> str:
    """The step directory a hub ref names ('012000'), for log lines. Falls back
    to the whole ref when it isn't step-suffixed."""
    m = _HUB_CKPT_REF_RE.match(ref)
    return m.group("step_dir") if m else ref


def hub_ref_repo_id(ref: str) -> str:
    """The repo half of a hub ref, for log lines. The whole ref if unparsable."""
    m = _HUB_CKPT_REF_RE.match(ref) or _HUB_ROOT_REF_RE.match(ref)
    return m.group("repo") if m else ref


def localize_pretrained_path(pretrained_path: str, *, tqdm_class=None) -> str:
    """Make a ``--policy.pretrained_path`` value loadable by a LOCAL trainer.

    A step-suffixed hub ref is materialized here, on this machine, into the real
    directory lerobot will load (see download_hub_checkpoint_ref). Everything
    else passes through untouched: an absolute local dir is already what lerobot
    wants, and a bare repo id is a repo ROOT, which lerobot resolves itself —
    downloading it here would just duplicate the trainer's own fetch.

    The cloud twin of this call happens IN the container (the HF Jobs wrapper
    materializes the same ref pod-side and rewrites the argv), because a host
    path means nothing on the pod. One rule, two execution sites.

    A failed download becomes a ValueError naming the checkpoint, rather than a
    raw Hub exception with nothing actionable in it. On the local start path the
    download no longer runs inside the request (see JobRegistry.start), so that
    message is now written onto the job record's `error_message` instead of
    becoming an HTTP 400 — same words, later delivery.

    `tqdm_class` is forwarded to the download for byte-progress reporting."""
    if not needs_local_materialization(pretrained_path):
        return pretrained_path
    try:
        return download_hub_checkpoint_ref(pretrained_path, tqdm_class=tqdm_class)
    except Exception as exc:
        raise ValueError(
            f"Could not download the base checkpoint {pretrained_path!r} to fine-tune from: {exc}"
        ) from exc


def _read_checkpoint_config(ckpt: JobCheckpoint) -> dict[str, object]:
    """Load the pretrained_model/config.json for one checkpoint.

    Keyed on the checkpoint's own source/ref shape so it works for training
    jobs and imports alike:
      * local  → ckpt.ref is the absolute pretrained_model dir.
      * hub    → 'repo@checkpoints/<step_dir>' (a tree) or 'repo@root' (a flat
                 model repo); both resolve via hf_hub_download.
    """
    if ckpt.source == "local":
        with open(Path(ckpt.ref) / "config.json") as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    m = _HUB_CKPT_REF_RE.match(ckpt.ref)
    if m:
        repo_id = m.group("repo")
        filename = f"checkpoints/{m.group('step_dir')}/pretrained_model/config.json"
    else:
        m = _HUB_ROOT_REF_RE.match(ckpt.ref)
        if not m:
            raise ValueError(f"Bad hub ref: {ckpt.ref!r}")
        repo_id = m.group("repo")
        filename = "config.json"
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    with open(local_path) as f:
        return json.load(f)


def _flat_feature_dim(feat: object) -> int | None:
    """Flat width of a policy feature (e.g. observation.state, action).

    Checkpoint config features carry a `shape` list; for the proprioceptive
    state and action these are 1-D — `[6]` for a single SO-101 arm, `[12]` for
    a bimanual (two-arm) checkpoint. Returns the single dim, or None when the
    feature is absent or not 1-D (nothing downstream should guess in that
    case)."""
    if not isinstance(feat, dict):
        return None
    shape = feat.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def read_pretrained_config(pretrained_path: str) -> dict[str, Any] | None:
    """Read the whole ``config.json`` of the checkpoint at
    ``--policy.pretrained_path``.

    `pretrained_path` is whatever the run will be started from: an absolute
    local ``pretrained_model`` directory, a Hub repo id whose ROOT holds the
    model, or a step-suffixed hub ref ('repo@checkpoints/<step_dir>') that has
    not been materialized yet — the checks that use this run BEFORE the
    download, so that a contradicting pair is refused without first pulling
    gigabytes.

    Only the checkpoint's ``config.json`` is fetched in every case (a few KB).

    Returns the parsed object, or None when it can't be read — missing file,
    malformed JSON, private/absent repo, or no network. None means "not
    established", never "fine": callers must treat it as a reason to stay
    silent rather than a clean bill of health.
    """
    path = Path(pretrained_path)
    if path.is_dir():
        with contextlib.suppress(Exception), open(path / "config.json") as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else None
        return None

    # A step-suffixed ref addresses one checkpoint inside the repo; anything
    # else is a repo id whose root holds the config.
    m = _HUB_CKPT_REF_RE.match(pretrained_path)
    if m:
        repo_id = m.group("repo")
        filename = f"checkpoints/{m.group('step_dir')}/{_HUB_CKPT_SUBDIR}/config.json"
    else:
        repo_id, filename = pretrained_path, "config.json"

    with contextlib.suppress(Exception):
        local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        with open(local) as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else None
    return None


def read_pretrained_feature_space(
    pretrained_path: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The checkpoint's ``(input_features, output_features)`` maps.

    Read out of the same single read_pretrained_config fetch any other
    checkpoint-side guard uses. These are the feature specs lerobot WOULD have
    built the policy from had the run used ``--policy.path``; on MakerMods Lab's
    launch path (``--policy.type`` + ``--policy.pretrained_path``) they are
    never consulted, which is exactly why the preflight has to read them itself
    (MT44).

    Each map is ``{feature key: {"type": "STATE"|"VISUAL"|"ACTION", "shape":
    [...]}}``; image shapes are CHW. None when the config can't be read or
    carries neither map — "not established", not "fine". A config with only
    one of the two yields the other as ``{}``, so a state-dim check can still
    run when output_features is missing.
    """
    cfg = read_pretrained_config(pretrained_path)
    if cfg is None:
        return None
    inputs = cfg.get("input_features")
    outputs = cfg.get("output_features")
    if not isinstance(inputs, dict) and not isinstance(outputs, dict):
        return None
    return (
        inputs if isinstance(inputs, dict) else {},
        outputs if isinstance(outputs, dict) else {},
    )


def _camera_features(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The camera entries of a feature map, keyed by BARE camera name.

    Both sides of the preflight name cameras the same way — the dataset's
    ``meta/info.json`` and the checkpoint's ``config.json`` both use
    ``observation.images.<name>`` — so stripping the prefix gives directly
    comparable key sets. Every ``observation.images.*`` key counts here, video
    and raw-image alike (unlike datasets._video_camera_names, which filters to
    what the episode viewer can play): the trainer consumes both as camera
    inputs, so both belong in the comparison.
    """
    return {
        key[len(CAMERA_FEATURE_PREFIX) :]: spec
        for key, spec in features.items()
        if key.startswith(CAMERA_FEATURE_PREFIX) and isinstance(spec, dict)
    }


# A camera key a pretrained BASE ships as a placeholder rather than as the name
# of a real mount: lerobot/smolvla_base's camera1/camera2/camera3. Matched
# against the BARE tail (the observation.images. prefix already stripped).
_PLACEHOLDER_CAMERA_RE = re.compile(r"^camera\d+$")


def _is_placeholder_camera_set(names: set[str]) -> bool:
    """True when EVERY camera key is a generic placeholder — the signature of a
    pretrained base that was never tied to one rig. All-or-nothing on purpose: a
    checkpoint mixing placeholders with real mounts (camera1 + wrist) is not a
    generic base, so it does not qualify."""
    return bool(names) and all(_PLACEHOLDER_CAMERA_RE.match(name) for name in names)


def _image_height_width(spec: dict[str, Any]) -> tuple[int, int] | None:
    """(height, width) of a camera feature, normalising the two shape
    conventions this file has to compare.

    A dataset's ``meta/info.json`` stores image shapes HWC — ``[480, 640, 3]``
    with ``names: ["height", "width", "channels"]`` — while a policy
    checkpoint's ``config.json`` stores them CHW — ``[3, 480, 640]``, no names
    at all. Locate the channel axis by name when the spec carries names, else
    by the only axis narrow enough to be channels (1 or 3), and return the
    remaining two dims in order.

    None when the shape isn't a 3-axis image or the channel axis is
    ambiguous — the caller then simply doesn't compare resolutions, which in
    phase 1 only costs a log line.
    """
    shape = spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return None

    axis: int | None = None
    names = spec.get("names")
    if isinstance(names, (list, tuple)) and len(names) == 3:
        for i, name in enumerate(names):
            if isinstance(name, str) and name.strip().lower().rstrip("s") == "channel":
                axis = i
                break
    if axis is None:
        narrow = [i for i, dim in enumerate(shape) if dim in (1, 3)]
        if len(narrow) != 1:
            return None
        axis = narrow[0]

    try:
        height, width = (int(dim) for i, dim in enumerate(shape) if i != axis)
    except (TypeError, ValueError):
        return None
    return height, width


def _describe_cameras(names: set[str]) -> str:
    return ", ".join(sorted(names)) if names else "none"


def _check_pretrained_feature_space(pretrained_path: str, dataset_repo_id: str) -> None:
    """Reject a fine-tune whose checkpoint and dataset describe different robots.

    Necessary because on MakerMods Lab's launch path (``--policy.type`` +
    ``--policy.pretrained_path``) lerobot builds the policy FROM THE DATASET's
    features and then loads the checkpoint's weights with ``strict=False``, so
    it never compares the two. lerobot's own
    ``validate_visual_features_consistency`` is a tautology here — it compares
    the dataset against itself. Nothing downstream will notice the mismatch
    (MT44).

    What that costs, by class:

    * STATE/ACTION WIDTH — ACT fails loudly but late, with a raw size-mismatch
      traceback after the dataset download. SmolVLA/pi0/pi05 pad the
      proprioceptive dims to 32, so a 6-dof checkpoint loads CLEANLY into a
      12-dof (bimanual) run and trains garbage that is recorded as a
      fine-tune. Refused.
    * RENAMED CAMERAS — the same number of cameras under different keys (the
      bimanual ``left_``-prefix case). Refused as a SELECTION mistake, not an
      architectural one: ACT drives every camera through one shared backbone
      (``ACT.backbone`` + a spatial-only position embedding, no per-camera
      parameters) and SmolVLA through one shared vision tower, so the weights
      would in fact carry over fine. What the rename actually signals is that
      the dataset and the base model came from DIFFERENT RIGS — which in this
      UI is never deliberate. Catching the bimanual ``left_``-prefix case is
      the whole point of the rule.

      The same rationale extends to camera sets that are FULLY DISJOINT at ANY
      count (a 1-camera ``left`` dataset against a ``wrist``/``front``
      checkpoint). Unequal counts would otherwise route that to the benign
      count-change warning below, but zero overlap is not a sensor-suite
      change: none of the checkpoint's visual inputs survive, its cameras are
      all dropped and the dataset's all start from scratch. Same selection
      mistake, so the same refusal.

      ONE EXEMPTION — a GENERIC BASE. When every one of the CHECKPOINT's camera
      keys is a placeholder (``camera1``/``camera2``/… — see
      _is_placeholder_camera_set), the checkpoint was never tied to a rig at
      all: ``lerobot/smolvla_base`` ships exactly that, and adapting it to a
      named 3-camera rig is the CANONICAL SmolVLA fine-tune, not a mixup.
      That case demotes to the warn path. The gate is checkpoint-side only —
      a user's dataset always carries real mount names, so the dataset side
      tells us nothing — and it is all-or-nothing, so a checkpoint mixing a
      placeholder with a real mount stays refused. Phase 2 may replace this
      exemption with an explicit confirm once there is a UI to confirm in.
    * MISSING / EXTRA CAMERA, DIFFERENT RESOLUTION — legitimate choices
      (ACT's shared backbone handles a changed sensor count; resolution only
      moves the token count and VRAM). Phase 1 logs a warning; the
      warn-and-confirm UI is phase 2. This path applies only when the two
      sides SHARE at least one camera — with no overlap it is a different rig,
      not a changed sensor suite, and the disjoint rule above refuses it.

    Silent when either side can't be read, matching the neighbouring guards:
    an unverifiable pair must not block a launch. Raises ValueError (→ HTTP
    400) only on a contradiction we can actually prove.
    """
    # Checkpoint first: it is the side most often unreadable (a hub ref with no
    # network, a hand-built path), and bailing here spares the dataset read.
    feature_space = read_pretrained_feature_space(pretrained_path)
    if feature_space is None:
        return
    ckpt_inputs, ckpt_outputs = feature_space

    dataset_features = read_dataset_features(dataset_repo_id)
    if not dataset_features:
        return

    # -- state / action width ------------------------------------------------
    # The dataset's own dims are what lerobot will build the policy from, so a
    # difference here is exactly the width the loaded weights won't fit.
    for label, ckpt_feature, dataset_key in (
        ("robot state", ckpt_inputs.get("observation.state"), "observation.state"),
        ("action", ckpt_outputs.get("action"), "action"),
    ):
        ckpt_dim = _flat_feature_dim(ckpt_feature)
        dataset_dim = _flat_feature_dim(dataset_features.get(dataset_key))
        if ckpt_dim is None or dataset_dim is None or ckpt_dim == dataset_dim:
            continue
        raise ValueError(
            f"The checkpoint this run starts from ({pretrained_path}) was trained "
            f"with {ckpt_dim}-dim {label}, but the dataset {dataset_repo_id!r} "
            f"records {dataset_dim}-dim {label} — a different robot (an SO-101 arm "
            f"is 6 dims, a bimanual pair 12). Fine-tuning builds the policy from "
            f"the DATASET and loads the checkpoint's weights into it without "
            f"checking the widths, so the run would train from largely random "
            f"weights while reporting itself as a fine-tune. Pick a dataset "
            f"recorded on the same robot as this checkpoint, pick a base model "
            f"trained on this robot, or train from scratch."
        )

    # -- cameras -------------------------------------------------------------
    ckpt_cameras = _camera_features(ckpt_inputs)
    dataset_cameras = _camera_features(dataset_features)
    ckpt_names, dataset_names = set(ckpt_cameras), set(dataset_cameras)

    renamed = ckpt_names != dataset_names and len(ckpt_names) == len(dataset_names)
    # Zero overlap is the same selection mistake as a rename, at any count: none
    # of the checkpoint's visual inputs survive. Equal-count disjoint sets fall
    # to `renamed` first (its wording is accurate there), so this branch is in
    # practice the unequal-count case the rename rule alone would let through.
    disjoint = bool(ckpt_names) and bool(dataset_names) and ckpt_names.isdisjoint(dataset_names)
    if (renamed or disjoint) and _is_placeholder_camera_set(ckpt_names):
        # A generic base being bound to a named rig for the first time — the
        # canonical smolvla_base fine-tune. Recorded, not refused.
        logger.warning(
            "Fine-tune feature space: checkpoint %s carries placeholder camera "
            "names (%s); binding them to dataset %s's cameras (%s).",
            pretrained_path,
            _describe_cameras(ckpt_names),
            dataset_repo_id,
            _describe_cameras(dataset_names),
        )
    elif renamed:
        raise ValueError(
            f"The checkpoint this run starts from ({pretrained_path}) expects "
            f"cameras {_describe_cameras(ckpt_names)}, but the dataset "
            f"{dataset_repo_id!r} provides {_describe_cameras(dataset_names)} — "
            f"the same number of cameras under different names. Cameras are "
            f"matched by name, so the two were almost certainly recorded on "
            f"different robots, and the fine-tune would be building on a base "
            f"that never saw this rig. Pick a base model whose camera names "
            f"match this dataset, or train from scratch."
        )
    elif disjoint:
        raise ValueError(
            f"The checkpoint this run starts from ({pretrained_path}) expects "
            f"cameras {_describe_cameras(ckpt_names)}, but the dataset "
            f"{dataset_repo_id!r} provides {_describe_cameras(dataset_names)} — "
            f"no camera in common. Cameras are matched by name, so none of the "
            f"checkpoint's visual inputs would survive: its cameras are dropped "
            f"and the dataset's are trained from scratch, on a base that never "
            f"saw this rig. Pick a base model whose camera names match this "
            f"dataset, or train from scratch."
        )
    elif ckpt_names != dataset_names:
        # Unequal counts: a real sensor-suite change, but a legitimate one.
        # Phase 1 only records it (phase 2 asks the user to confirm).
        missing = ckpt_names - dataset_names
        extra = dataset_names - ckpt_names
        if missing:
            logger.warning(
                "Fine-tune feature space: checkpoint %s expects camera(s) %s that "
                "dataset %s does not provide; those inputs will be dropped.",
                pretrained_path,
                _describe_cameras(missing),
                dataset_repo_id,
            )
        if extra:
            logger.warning(
                "Fine-tune feature space: dataset %s adds camera(s) %s the "
                "checkpoint %s was not trained on; those inputs start from scratch.",
                dataset_repo_id,
                _describe_cameras(extra),
                pretrained_path,
            )

    # Resolution differences on the cameras both sides DO share. Allowed —
    # lerobot resizes/re-tokenizes — but it moves ACT's token count and VRAM,
    # so it should not pass unrecorded.
    resized = []
    for name in sorted(ckpt_names & dataset_names):
        ckpt_hw = _image_height_width(ckpt_cameras[name])
        dataset_hw = _image_height_width(dataset_cameras[name])
        if ckpt_hw is None or dataset_hw is None or ckpt_hw == dataset_hw:
            continue
        resized.append(f"{name} {ckpt_hw[0]}x{ckpt_hw[1]} -> {dataset_hw[0]}x{dataset_hw[1]}")
    if resized:
        logger.warning(
            "Fine-tune feature space: checkpoint %s and dataset %s disagree on "
            "camera resolution (%s); training proceeds at the dataset's size.",
            pretrained_path,
            dataset_repo_id,
            "; ".join(resized),
        )


def _generate_job_id(policy_type: str, dataset_repo_id: str) -> str:
    """Build a sortable, collision-free job id from policy type and dataset slug."""
    from .train import _SLUG_RE

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_slug = _SLUG_RE.sub("_", dataset_repo_id).strip("_") or "dataset"
    return f"{policy_type}_{dataset_slug}_{timestamp}"


# Accepted in place of a bare repo id when importing from the Hub — users
# paste the model page URL as often as the id.
_HUB_URL_PREFIXES = (
    "https://huggingface.co/",
    "http://huggingface.co/",
    "https://hf.co/",
    "http://hf.co/",
    "huggingface.co/",
    "hf.co/",
)


def _normalize_import_source(source: str) -> str:
    """Boundary normalization for import sources, applied before both storing
    and comparing: trim whitespace, strip a pasted Hub URL prefix down to the
    bare repo id, and drop trailing slashes. Local absolute paths start with
    '/' so the URL prefixes never match them."""
    src = source.strip()
    lowered = src.lower()
    for prefix in _HUB_URL_PREFIXES:
        if lowered.startswith(prefix):
            src = src[len(prefix) :]
            break
    return src.rstrip("/")


# What register_imported used to auto-name a card before titles were derived
# ("Imported · makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30"). The
# prefix duplicated the card's own provenance chip and pushed the one useful
# segment — the task — past the truncation. Recognized so existing records
# re-derive on load instead of keeping a name no new import would produce.
_LEGACY_IMPORT_NAME_PREFIX = "Imported · "


def _auto_imported_name(record: JobRecord) -> str | None:
    """The title an imported record's source derives to, or None to leave it be.

    Returns None for a record whose `name` the CALLER chose (POST /jobs/import
    takes an optional name): an explicit name is the user's, and re-deriving
    would silently throw it away. Everything the auto-namer has ever produced is
    recognizable — the legacy prefixed form, the bare derived title, and a
    derived title carrying a collision suffix — so the test is precise rather
    than a guess about which names look generated.
    """
    source = record.hf_repo_id or record.output_dir
    if record.runner != "imported" or not source:
        return None
    derived = derive_imported_title(source)
    if record.name.startswith(_LEGACY_IMPORT_NAME_PREFIX):
        return derived
    if record.name == derived or record.name.startswith(f"{derived} ("):
        return derived
    return None


def _paths_are_same_dir(a: str, b: str) -> bool:
    """True when two path strings refer to the same directory on disk.

    os.path.samefile compares device+inode, so it survives spellings that a
    string compare misses — most importantly case variants on the (default)
    case-insensitive macOS filesystem: a real duplicate pair was registered as
    '/Users/mokuroh54/…' and '/Users/Mokuroh54/…' because Path.resolve()
    preserves the typed case. Falls back to exact string equality when either
    path can't be stat'ed (e.g. the recorded source has since moved)."""
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _job_dir(output_root: Path, job_id: str) -> Path:
    return output_root / job_id


def _job_log_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "log.jsonl"


def _job_meta_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "job.json"


class JobAlreadyRunningError(Exception):
    """Raised when start() is called while another local job is running."""


class JobNotFoundError(Exception):
    """Raised when a lookup hits an unknown id."""


class JobNotRunningError(Exception):
    """Raised when stop() is called on a non-running job."""


class DatasetNotOnHubError(Exception):
    """Raised by JobRegistry.start when a cloud (hf_cloud) run is requested on a
    dataset that isn't on the Hub. HF Jobs pods resolve the dataset by repo_id
    from the Hub — they can't see this machine's local cache — so a local-only
    dataset would make the remote job fail. The UI's upload-then-train flow
    makes this unreachable from the browser; this guard exists for non-UI
    callers (and as belt-and-braces) so they get a clear 409 instead of a
    remote crash. `repo_id` is the offending dataset."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        super().__init__(
            f"Dataset '{repo_id}' is not on the Hugging Face Hub. Cloud training "
            "runs from the Hub, so upload the dataset first (or record/select one "
            "that's already on the Hub)."
        )


class JobRegistry:
    """Owns the registry of training jobs and their persistence.

    On instantiation, scans outputs/train/ for existing job.json files. For
    each record marked 'running': local jobs reattach if the pid is alive
    (else 'interrupted'); hf_cloud jobs always reattach and let the tail loop
    drive finalisation.
    """

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._runners: dict[str, JobRunner] = {}
        self._last_persist_at: dict[str, float] = {}
        # Ids we have asked to stop, recorded BEFORE the signal goes out so the
        # watchdog can tell a deliberate stop from a crash (see
        # classify_terminal_state). Guarded by _lock; entries are dropped when
        # the record is finalised or deleted. Deliberately in-memory only: a
        # stop cannot outlive the process that issued it, and a record found
        # 'running' after a restart is reconciled by _load_from_disk instead.
        self._stop_requested: set[str] = set()

        # job_id -> the thread materializing that job's base checkpoint before
        # its trainer can start (see _materialize_then_start). Entries are left
        # behind once the thread finishes — a finished Thread object is inert
        # and joinable, and dropping it would race a caller that wants to join.
        self._prepare_threads: dict[str, threading.Thread] = {}

        self._stop_watchdog = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

        # repo_id -> (expires_at_epoch, checkpoint list)
        self._cloud_ckpt_cache: dict[str, tuple[float, list[JobCheckpoint]]] = {}

        # Fired (best-effort) on every state change: new job, stop initiated,
        # watchdog finalisation, delete. Server wires this to a WebSocket
        # broadcast so the frontend can refetch on-event instead of polling.
        self._on_change: Callable[[], None] | None = None

        # Fired from the watchdog at ~1Hz with a compact snapshot of every
        # running job (id, state, metrics, checkpoint count) so
        # the dashboard keeps the progress bar live without refetching /jobs.
        self._on_progress: Callable[[builtins.list[dict]], None] | None = None

        self._migrate_legacy_cwd_jobs()
        self._load_from_disk()
        self._dedupe_imported_records()
        # After the dedupe, so a collapsed duplicate can't be counted as a
        # colliding title and drag a suffix onto the card that survived it.
        self._resolve_imported_names()
        self._start_watchdog()

    def _migrate_legacy_cwd_jobs(self) -> None:
        """One-shot migration from cwd-relative `outputs/train/` to the new
        absolute root.

        Older MakerMods Lab versions wrote job dirs to `<cwd>/outputs/train/`, which
        meant history disappeared when you launched from a different cwd. We
        now anchor to ~/.cache/.../outputs/train. On first boot under the new
        layout, move any pre-existing cwd-relative job dirs over and rewrite
        each job.json's `output_dir` field to the new absolute path.

        Idempotent: skipped if (a) the new root is the legacy one itself
        (MAKERMODSLAB_OUTPUT_ROOT=outputs/train still wins for tests), or (b) the
        legacy dir is absent / already empty.
        """
        legacy_root = (Path.cwd() / "outputs" / "train").resolve()
        if legacy_root == self._output_root or not legacy_root.is_dir():
            return

        legacy_dirs = [p for p in legacy_root.iterdir() if p.is_dir()]
        if not legacy_dirs:
            return

        logger.info(
            "Migrating %d legacy job dirs from %s to %s",
            len(legacy_dirs),
            legacy_root,
            self._output_root,
        )
        for src in legacy_dirs:
            dst = self._output_root / src.name
            if dst.exists():
                logger.warning("Migration: %s already exists at destination; skipping", src.name)
                continue
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:
                logger.warning("Migration: failed to move %s: %s", src.name, exc)
                continue
            self._rewrite_output_dir_in_meta(dst)

        # If the legacy dir is now empty, remove it so subsequent boots skip
        # the scan. A leftover non-dir file keeps it around — that's fine.
        with contextlib.suppress(OSError):
            legacy_root.rmdir()

    def _rewrite_output_dir_in_meta(self, job_dir: Path) -> None:
        """Repoint `output_dir` in a migrated job.json to its new absolute
        path. Pre-migration records stored `outputs/train/<id>/run` which
        no longer resolves once cwd has moved."""
        meta = job_dir / "job.json"
        if not meta.is_file():
            return
        try:
            data = json.loads(meta.read_text())
        except Exception as exc:
            logger.warning("Migration: could not parse %s: %s", meta, exc)
            return
        data["output_dir"] = str(job_dir / "run")
        tmp = meta.with_suffix(meta.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, meta)

    def set_on_change(self, callback: Callable[[], None] | None) -> None:
        """Register a single observer fired when registry state changes."""
        self._on_change = callback

    def set_on_progress(self, callback: Callable[[builtins.list[dict]], None] | None) -> None:
        """Register an observer fired each watchdog tick with one dict per
        running job. Quiet when no job runs: a tick with no running jobs
        produces no callback."""
        self._on_progress = callback

    def _notify_change(self) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:
            logger.exception("JobRegistry on_change callback failed: %s", exc)

    def _notify_progress(self, snapshots: builtins.list[dict]) -> None:
        cb = self._on_progress
        if cb is None or not snapshots:
            return
        try:
            cb(snapshots)
        except Exception as exc:
            logger.exception("JobRegistry on_progress callback failed: %s", exc)

    # -- public API --

    def _assert_no_running_local(self, target: JobTarget) -> None:
        """Raise JobAlreadyRunningError if a local training is already running.

        Local trainings are bounded by this machine's GPU/USB resources, so at
        most one runs at a time. Cloud trainings each get their own remote
        container, so any number can be in flight in parallel.

        Takes NO lock, so it is callable both inside and outside the registry's
        critical section (self._lock is a plain Lock — re-entering it would
        deadlock). Iterates a snapshot, so a concurrent mutation can't break the
        walk. `start` calls it twice on purpose: once early to fail fast, and
        once under the lock that inserts the record — the latter is the one that
        actually enforces the invariant."""
        if target.runner != "local":
            return
        for r in list(self._records.values()):
            if r.state == "running" and r.runner == "local":
                raise JobAlreadyRunningError(r.id)

    def list(self, limit: int = 10) -> builtins.list[JobRecord]:
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda r: r.started_at, reverse=True)
        records = records[:limit]
        for r in records:
            r.checkpoint_count = self._count_checkpoints(r)
        return records

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        record.checkpoint_count = self._count_checkpoints(record)
        return record

    def start(self, config: TrainingRequest, target: JobTarget | None = None) -> JobRecord:
        from .runners.hf_cloud import HfCloudJobRunner  # lazy import to avoid circular import

        target = target or JobTarget()
        if target.runner == "hf_cloud" and not target.flavor:
            raise ValueError("flavor is required when runner is hf_cloud")

        # Cloud preflight (belt-and-braces): the HF Jobs pod resolves the
        # dataset by repo_id from the Hub and can't see this machine's local
        # cache, so a local-only dataset would fail the remote job. Reject up
        # front with a clear error instead of submitting a doomed job. Only a
        # definitive "local_only" blocks; "unknown" (offline / transient
        # transport error) is left to the existing _ensure_dataset_on_hub
        # fallback so a network blip doesn't wrongly refuse a Hub dataset. The
        # browser flow uploads-then-trains before ever reaching here, so this
        # path is primarily for non-UI callers.
        if target.runner == "hf_cloud":
            from .datasets import get_hub_status

            if get_hub_status(config.dataset_repo_id).get("status") == "local_only":
                raise DatasetNotOnHubError(config.dataset_repo_id)

        # Resume and fine-tune are distinct and mutually exclusive: resume
        # continues optimizer+step from a checkpoint (needs training_state);
        # fine-tune starts a FRESH run whose weights are init'd from a
        # checkpoint (weights-only is fine). Reject the nonsensical combo up
        # front rather than letting one silently win.
        if config.resume and config.finetune_from_job_id:
            raise ValueError(
                "A run can't both resume and fine-tune. Resume continues an "
                "existing run's optimizer/step; fine-tune starts a fresh run "
                "from a checkpoint's weights."
            )

        # Fail fast on the local-run mutex before any of the (possibly slow)
        # pretrained-path work below. Re-checked under the lock at record
        # creation — THAT is the authoritative check; this one only spares a
        # doomed request a needless download.
        self._assert_no_running_local(target)

        # ------------------------------------------------------------------
        # Pretrained-path resolution runs OUTSIDE the registry lock.
        #
        # It reads the Hub (the source's checkpoint listing) — seconds at worst.
        # self._lock is taken by list / get / stop / delete, so holding it
        # across that would freeze the whole job interface (the MT23 coupling).
        # Nothing is registered yet, so a bad selection still fails with no
        # orphaned record; the registry lookup it needs takes the lock briefly
        # on its own.
        #
        # What does NOT happen here any more is the multi-GB DOWNLOAD a local
        # fine-tune needs. Every cheap refusal above and below stays synchronous
        # (a bad request must still fail fast with a clear 400), but the
        # materialization itself is deferred to a background thread that starts
        # after the record exists — see the `deferred_hub_ref` branch below.
        # ------------------------------------------------------------------
        if config.finetune_from_job_id:
            # Fine-tune: turn the selected source run + step into the
            # pretrained_path lerobot loads weights from. A fresh run (resume
            # stays false); no training_state required.
            with self._lock:
                source = self._records.get(config.finetune_from_job_id)
            if source is None:
                raise ValueError(f"Fine-tune source {config.finetune_from_job_id!r} not found.")
            # The requested policy type must be the source checkpoint's own
            # architecture — lerobot loads the weights non-strictly, so a
            # mismatch trains a fresh `policy_type` policy that only *looks*
            # like a fine-tune. Checked before the pretrained path resolves
            # (cheap, no Hub call) so a contradicting request never starts.
            _check_finetune_policy_type(source, config.policy_type)
            config.policy_pretrained_path = _resolve_finetune_pretrained_path(
                source, config.finetune_from_step
            )

        # The three things that can still have to MOVE before a trainer starts,
        # each resolved (and refused) synchronously below but carried out in the
        # preparing thread because each is potentially GBs over the network:
        #   * a local fine-tune's base checkpoint, downloaded here;
        #   * a cloud parent's checkpoint, downloaded here for a local resume;
        #   * a local parent's checkpoint, uploaded to the Hub for a cloud one.
        # At most one is ever set: fine-tune and resume are mutually exclusive
        # (refused above), and a resume moves bytes in one direction only.
        deferred_hub_ref: str | None = None
        deferred_resume_ref: str | None = None
        deferred_resume_upload: tuple[Path, str, str] | None = None
        if config.policy_pretrained_path and not config.resume:
            # Deliberately BEFORE the materialization: this reads only the
            # checkpoint's config.json (a few KB), so a contradicting pair is
            # refused without first downloading the weights it names — and
            # refused SYNCHRONOUSLY, as a 400, with no job record left behind.
            # lerobot sizes the policy from the DATASET on this launch path and
            # loads the weights non-strictly, so a checkpoint whose robot or
            # camera set contradicts the selected dataset would otherwise train
            # silently wrong (MT44).
            _check_pretrained_feature_space(config.policy_pretrained_path, config.dataset_repo_id)
            if target.runner == "local" and needs_local_materialization(config.policy_pretrained_path):
                # A step-suffixed hub ref becomes the real directory the local
                # trainer loads — but off-request, in _materialize_then_start,
                # which rewrites policy_pretrained_path once the bytes are here.
                # A cloud run keeps the ref: its container materializes the same
                # ref pod-side (see the HF Jobs wrapper), because a host path is
                # meaningless there.
                deferred_hub_ref = config.policy_pretrained_path

        with self._lock:
            # Authoritative local-run mutex: re-checked here, under the lock
            # that also inserts the record, so two concurrent starts can't both
            # pass. (The pre-flight copy above is only a fail-fast.)
            self._assert_no_running_local(target)

        # Whatever put a pretrained_path on this request — the fine-tune
        # resolution above, or a caller setting the public field directly —
        # its architecture must match --policy.type before we spend a GPU on
        # it. Checked against the CHECKPOINT'S OWN config.json, so it holds
        # even when the registry's record of that checkpoint is missing or
        # carries the "model" placeholder. Skipped on resume: that path
        # passes --config_path instead and never emits pretrained_path (see
        # train.build_training_command), so there is no pair to contradict.
        if config.policy_pretrained_path and not config.resume:
            # Deliberately BEFORE the materialization below: this reads only the
            # checkpoint's config.json, so a contradicting pair is refused
            # without first downloading the weights it names.
            _check_pretrained_policy_type(config.policy_pretrained_path, config.policy_type)
            if target.runner == "local":
                # A step-suffixed hub ref becomes the real directory the local
                # trainer loads. A cloud run keeps the ref: its container
                # materializes the same ref pod-side (see the HF Jobs wrapper),
                # because a host path is meaningless there.
                config.policy_pretrained_path = localize_pretrained_path(config.policy_pretrained_path)

        with self._lock:
            # Authoritative local-run mutex: re-checked here, under the lock
            # that also inserts the record, so two concurrent starts can't both
            # pass. (The pre-flight copy above is only a fail-fast.)
            self._assert_no_running_local(target)

            # Resume: turn the selected source run + step into the config_path
            # lerobot needs. Do this under the lock (source lookup) and before
            # creating the record so a bad selection fails cleanly with no
            # orphaned job.
            if config.resume:
                if config.resume_from_job_id:
                    source = self._records.get(config.resume_from_job_id)
                    if source is None:
                        raise ValueError(f"Resume source {config.resume_from_job_id!r} not found.")
                    # A completed run is not resumable — on ANY runner, and
                    # regardless of how the request got here (the UI hides the
                    # button; this catches a direct API call).
                    #
                    # Resume restores the optimizer AND the LR schedule's
                    # position, and a run that reached its target has spent its
                    # schedule: SmolVLA's preset cosine-decays to a 2.5e-6 floor
                    # over a fixed 30k-step horizon, so continuing past the
                    # target trains at floor LR. The loss chart flattens and
                    # reads as convergence while the run is barely learning —
                    # the failure is silent, which is why this refuses rather
                    # than warns. Fine-tune starts a fresh schedule from the
                    # same weights, which is the intended way to build on a
                    # completed run.
                    if source.state == "done":
                        raise ValueError(
                            f"Run {source.id!r} already reached its step target, so there is "
                            "nothing to resume — its learning-rate schedule is finished and a "
                            "continuation would train at the schedule's floor. Fine-tune from "
                            "its final checkpoint instead."
                        )
                    # A resume may continue on EITHER runner (F7). What changes
                    # across the four combinations is only where the parent's
                    # checkpoint has to end up before the trainer can read it —
                    # the pod cannot see this disk, and this disk does not have
                    # the pod's. Each branch below either points the trainer at
                    # bytes that already exist where it runs, or records the one
                    # move that has to happen first. What NEVER happens is a
                    # launch that couldn't find the checkpoint: every branch
                    # refuses (or fails the job) instead, because a resume that
                    # quietly starts at step 0 while the record calls itself a
                    # continuation is MT42, the worst outcome available here.
                    #
                    # Only local/hf_cloud parents reach here: an imported record
                    # is created with state="done", so the refusal above already
                    # took it (imported models are fine-tune sources, never
                    # resume sources).
                    if source.runner == "hf_cloud":
                        # The parent's checkpoints are on the Hub. Naming the
                        # chosen one is the same job for both targets, and
                        # _resolve_cloud_resume refuses an incomplete or absent
                        # one here — synchronously, as a 400, before any record
                        # or GPU exists.
                        repo_id, step_dir = _resolve_cloud_resume(source, config.resume_from_step)
                        if target.runner == "hf_cloud":
                            # An HF Job is immutable once ended: resuming a cloud
                            # run launches a NEW cloud job that continues from the
                            # parent's Hub checkpoint. The HfCloudJobRunner turns
                            # these two into an in-container download +
                            # reconstruct + --config_path. The dataset-on-Hub
                            # guard (target.runner == hf_cloud above) still
                            # applies, so a run whose dataset vanished fails the
                            # same way a fresh cloud run would.
                            config.resume_from_hub_repo = repo_id
                            config.resume_from_hub_step = step_dir
                        else:
                            # cloud → Local: the same download the wrapper does
                            # pod-side, done here instead — but off-request (GBs),
                            # so config_path is filled in by the preparing thread.
                            # `resume_from_step` is pinned to the resolved step so
                            # the record's progress is rebased from the inherited
                            # step immediately, rather than reading 0 until the
                            # download finishes and the first tqdm frame lands
                            # (see _initial_metrics / _resume_start_step).
                            deferred_resume_ref = f"{repo_id}@checkpoints/{step_dir}"
                            config.resume_from_step = int(step_dir)
                    elif target.runner == "local":
                        config.config_path = _resolve_resume_config_path(source, config.resume_from_step)
                    else:
                        # local → Cloud: the parent's checkpoint exists only on
                        # this machine, so it must reach the Hub before the pod
                        # starts. Resolve + validate it here (the same gate a
                        # local→local resume passes), then either reuse an
                        # upload a previous continuation already made or defer a
                        # fresh, consented one.
                        (
                            deferred_resume_upload,
                            hub_repo,
                            hub_step,
                        ) = self._resolve_upload_resume(source, config)
                        config.resume_from_hub_repo = hub_repo
                        config.resume_from_hub_step = hub_step
                        config.resume_from_uploaded_checkpoint = True
                        config.resume_from_step = int(hub_step)
                elif not config.config_path:
                    raise ValueError(
                        "Resume is on but no source checkpoint was selected. Use "
                        '"Continue" on a local run that stopped short of its step '
                        "target rather than toggling resume manually."
                    )

            job_id = self._unique_job_id(config.policy_type, config.dataset_repo_id)
            job_dir = _job_dir(self._output_root, job_id)
            lerobot_output_dir = str(job_dir / "run")
            name = (
                config.job_name.strip()
                if (config.job_name and config.job_name.strip())
                else f"{config.policy_type.upper()} · {config.dataset_repo_id}"
            )
            record = JobRecord(
                id=job_id,
                name=name,
                state="running",
                config=config,
                output_dir=lerobot_output_dir,
                started_at=time.time(),
                runner=target.runner,
                hf_flavor=target.flavor,
                # Built AFTER the resume block above, which is what resolves a
                # "latest checkpoint" request into a concrete step — see
                # _initial_metrics / _resume_start_step.
                metrics=_initial_metrics(config),
            )

            job_dir.mkdir(parents=True, exist_ok=True)
            self._records[job_id] = record
            self._persist(record, force=True)

            log_path = _job_log_path(self._output_root, job_id)

            deferred = (
                deferred_hub_ref is not None
                or deferred_resume_ref is not None
                or deferred_resume_upload is not None
            )
            if deferred:
                # Something still has to move (GBs, minutes). Hand the caller its
                # job id NOW — the record exists, is "running", and has a log
                # file — and do the transfer in a thread, which then starts the
                # real trainer. The monitor opens immediately and tails the
                # transfer instead of the request hanging with nothing on screen.
                #
                # No new JobState for this window: the job is "running" from the
                # user's point of view (they asked for a run and one is being
                # got ready), and a fourth state would ripple through the
                # watchdog's finalisation, the library chips and isJobActive.
                # What stands in for the missing process is PreparingJobRunner —
                # registered here, under the same lock as the record, so /logs
                # and Stop find a runner from the first request onwards.
                prep = PreparingJobRunner(log_file_path=log_path)
                self._runners[job_id] = prep
                if deferred_hub_ref is not None:
                    prep.emit(
                        f"Preparing fine-tune: downloading base checkpoint "
                        f"{hub_ref_step_label(deferred_hub_ref)} from "
                        f"{hub_ref_repo_id(deferred_hub_ref)}…"
                    )
                    thread_target: Callable[..., None] = self._materialize_then_start
                    thread_args: tuple = (job_id, deferred_hub_ref, lerobot_output_dir, prep)
                elif deferred_resume_ref is not None:
                    prep.emit(
                        f"Preparing continuation: downloading checkpoint "
                        f"{hub_ref_step_label(deferred_resume_ref)} from "
                        f"{hub_ref_repo_id(deferred_resume_ref)} to this machine…"
                    )
                    thread_target = self._download_resume_then_start
                    thread_args = (job_id, deferred_resume_ref, lerobot_output_dir, prep)
                else:
                    ckpt_dir, upload_repo, upload_step = deferred_resume_upload
                    prep.emit(
                        f"Preparing continuation: uploading checkpoint {upload_step} "
                        f"to the private repo {upload_repo} so the cloud job can read it…"
                    )
                    thread_target = self._upload_resume_then_start
                    thread_args = (job_id, ckpt_dir, upload_repo, upload_step, target, prep)
                thread = threading.Thread(
                    target=thread_target,
                    args=thread_args,
                    name=f"job-{job_id}-prepare",
                    daemon=True,
                )
                # Kept (not popped on completion) so a caller — today only the
                # tests — can join the thread instead of polling for its effect.
                self._prepare_threads[job_id] = thread
                thread.start()
            else:
                if target.runner == "local":
                    runner = LocalJobRunner(record.metrics, log_file_path=log_path)
                else:
                    runner = HfCloudJobRunner(
                        record.metrics,
                        log_path,
                        target.flavor,
                        _resume_total_steps(config),
                    )

                try:
                    runner.start(job_id, config, lerobot_output_dir)
                except Exception as exc:
                    logger.exception("Failed to start runner for job %s", job_id)
                    record.state = "failed"
                    record.ended_at = time.time()
                    record.error_message = f"Failed to start runner: {exc}"
                    self._persist(record, force=True)
                    raise

                # Capture runner-specific identifiers.
                if target.runner == "local":
                    record.process_pid = runner.pid()
                else:
                    record.hf_job_id = runner.hf_job_id()
                    record.hf_job_url = runner.hf_job_url()
                    # config was mutated by HfCloudJobRunner.start to set
                    # policy_repo_id; mirror it onto the record for the UI.
                    record.hf_repo_id = config.policy_repo_id

                self._persist(record, force=True)
                self._runners[job_id] = runner
        self._notify_change()
        return record

    def _materialize_then_start(
        self,
        job_id: str,
        ref: str,
        output_dir: str,
        prep: PreparingJobRunner,
    ) -> None:
        """Download a local fine-tune's base checkpoint, then start its trainer.

        Runs in its own thread so POST /jobs/training can return the job id
        immediately (see JobRegistry.start). Everything that can refuse the
        request cheaply already ran, synchronously, before the record existed;
        what is left here can only fail for reasons that belong ON the record:

          * download failed  → `failed`, carrying localize_pretrained_path's own
            "Could not download the base checkpoint …" wording, which used to
            reach the user as an HTTP 400.
          * user pressed Stop → `interrupted`, with a message saying so, exactly
            like stopping a live trainer would.
          * trainer failed to spawn → `failed`, same wording as the synchronous
            path's "Failed to start runner: …".

        No synthetic exit code is invented for any of them: there was no
        process, so `exit_code` stays None.

        On the cancel check: a huggingface_hub download cannot be interrupted
        mid-flight, so a Stop pressed while bytes are moving takes effect HERE —
        after the download returns and before the trainer is spawned. The bytes
        are already on disk (and cached for the next attempt); what the user
        gets is a run that never starts training, which is what they asked for.
        The spawn + runner handoff happen inside the registry lock, so a stop can
        neither be missed (spawning a trainer nobody will signal) nor land on a
        runner that has already been replaced.
        """
        reporter = _DownloadProgressLogger(prep.emit, hub_ref_step_label(ref))
        try:
            local_path = localize_pretrained_path(ref, tqdm_class=make_snapshot_progress_tqdm(reporter))
        except Exception as exc:
            logger.exception("Base-checkpoint download failed for job %s", job_id)
            self._fail_prepare(job_id, prep, str(exc))
            return

        self._start_after_prepare(
            job_id,
            output_dir,
            prep,
            JobTarget(runner="local"),
            lambda config: setattr(config, "policy_pretrained_path", local_path),
            "Base checkpoint ready — starting the trainer.",
        )

    def _download_resume_then_start(
        self,
        job_id: str,
        ref: str,
        output_dir: str,
        prep: PreparingJobRunner,
    ) -> None:
        """Bring a CLOUD parent's checkpoint to this machine, then resume locally.

        F7's cloud→local direction, and the twin of _materialize_then_start: same
        thread, same failure vocabulary, different cargo — the whole
        checkpoints/<step>/ tree rather than weights alone, because a resume
        needs the optimizer state (see download_hub_resume_checkpoint).

        The step was already chosen and verified against the Hub's file listing
        before the record existed, so an unresumable selection is still a
        synchronous 400. What is left here can only fail on the transfer itself,
        which belongs on the record: a failed or incomplete download finalises
        the job `failed` with the message naming the checkpoint, and NO trainer
        is spawned. That is the MT4 lesson — the old behaviour handed lerobot a
        --config_path that never existed and let it die at startup.
        """
        reporter = _DownloadProgressLogger(prep.emit, hub_ref_step_label(ref))
        try:
            config_path = download_hub_resume_checkpoint(
                ref, tqdm_class=make_snapshot_progress_tqdm(reporter)
            )
        except Exception as exc:
            logger.exception("Resume-checkpoint download failed for job %s", job_id)
            self._fail_prepare(
                job_id,
                prep,
                f"Could not download the checkpoint {hub_ref_step_label(ref)} from "
                f"{hub_ref_repo_id(ref)} to continue from: {exc}",
            )
            return

        self._start_after_prepare(
            job_id,
            output_dir,
            prep,
            JobTarget(runner="local"),
            lambda config: setattr(config, "config_path", config_path),
            "Checkpoint ready — starting the trainer.",
        )

    def _upload_resume_then_start(
        self,
        job_id: str,
        checkpoint_dir: Path,
        repo_id: str,
        step_dir: str,
        target: JobTarget,
        prep: PreparingJobRunner,
    ) -> None:
        """Push a LOCAL parent's checkpoint to the Hub, then resume on the cloud.

        F7's local→cloud direction. The pod downloads its resume checkpoint from
        the Hub, so these bytes have to be there before the job is submitted —
        and the request already carries the user's explicit consent for this
        upload (see _resolve_upload_resume).

        The upload is re-verified from the Hub's own file listing before anything
        is submitted, and the job is finalised `failed` if it can't be confirmed.
        That is the MT42 guarantee stated as code: a run recorded as a resume
        either continues from the checkpoint it names or does not start at all —
        it never becomes a fresh run that quietly begins at step 0 on rented
        hardware.

        On success the upload is recorded on the PARENT's record, so continuing
        the same step again reuses it instead of pushing the same GBs twice.
        """
        try:
            prep.emit(f"Uploading {step_dir} from {checkpoint_dir}…")
            upload_local_checkpoint(checkpoint_dir, repo_id, step_dir)
            if not hub_checkpoint_has_training_state(shared_hf_api(), repo_id, step_dir):
                raise ValueError(
                    f"the upload finished but {repo_id} is still missing {_HUB_TRAINING_STATE_FILE}"
                )
        except Exception as exc:
            logger.exception("Resume-checkpoint upload failed for job %s", job_id)
            self._fail_prepare(
                job_id,
                prep,
                f"Could not upload the checkpoint at step {step_dir} to {repo_id}, "
                f"so this continuation cannot run on the cloud: {exc}",
            )
            return

        self._remember_uploaded_checkpoint(job_id, repo_id, step_dir)
        self._start_after_prepare(
            job_id,
            # A cloud runner ignores the host output dir (its pod writes to a
            # container path); pass it anyway so the callers stay identical.
            str(_job_dir(self._output_root, job_id) / "run"),
            prep,
            target,
            None,
            f"Checkpoint {step_dir} is on the Hub — submitting the cloud job.",
        )

    def _remember_uploaded_checkpoint(self, job_id: str, repo_id: str, step_dir: str) -> None:
        """Record on the PARENT run where its checkpoint was uploaded.

        Keyed off the child's `resume_from_job_id` rather than passed in, so the
        note always lands on the record whose bytes were actually pushed. A
        missing parent (deleted mid-upload) is not an error: the upload still
        happened and the child still runs; only the re-use shortcut is lost."""
        with self._lock:
            child = self._records.get(job_id)
            parent_id = child.config.resume_from_job_id if child else None
            parent = self._records.get(parent_id) if parent_id else None
            if parent is None:
                return
            parent.checkpoints_hub_repo_id = repo_id
            if step_dir not in parent.checkpoints_hub_steps:
                parent.checkpoints_hub_steps = [*parent.checkpoints_hub_steps, step_dir]
            self._persist(parent, force=True)

    def _fail_prepare(self, job_id: str, prep: PreparingJobRunner, message: str) -> None:
        """Finalise a preparing job whose transfer failed, with the message on
        both the log and the record — the wording that used to reach the user as
        an HTTP 400 back when the transfer happened inside the request."""
        prep.emit(message)
        self._finalize_prepare(job_id, "failed", message)

    def _start_after_prepare(
        self,
        job_id: str,
        output_dir: str,
        prep: PreparingJobRunner,
        target: JobTarget,
        apply_to_config: Callable[[TrainingRequest], None] | None,
        ready_message: str,
    ) -> None:
        """The locked handoff every deferred preparation ends with: apply what
        the transfer produced to the job's config, spawn the real runner, and
        replace the PreparingJobRunner with it.

        Shared by all three preparations so the stop/delete/spawn-failure
        semantics can't drift between them. `apply_to_config` is whatever the
        slow step learned (a materialized path, a config_path) and is applied
        under the lock, immediately before the runner reads the config; None when
        the transfer taught the config nothing new (the upload path resolved its
        repo + step up front, so the record described the run correctly from the
        moment it was created).

        A Stop pressed while bytes were moving takes effect HERE — neither a
        huggingface_hub download nor an upload can be interrupted mid-flight, so
        the cancel is read after the transfer returns and before anything is
        spawned. The bytes are already moved (and cached for the next attempt);
        what the user gets is a run that never starts training, which is what
        they asked for. Reading `_stop_requested` under the same lock
        JobRegistry.stop records it in — with the spawn and runner handoff in the
        same critical section — is what keeps a stop from being missed (spawning
        a trainer nobody will signal) or landing on an already-replaced runner.
        """
        from .runners.hf_cloud import HfCloudJobRunner  # lazy import to avoid circular import

        notify = False
        try:
            with self._lock:
                if prep.cancelled():
                    prep.emit("Stopped before the trainer started.")
                    self._finalize_prepare_locked(job_id, "interrupted", _PREPARE_STOPPED_MESSAGE)
                    notify = True
                    return
                record = self._records.get(job_id)
                if record is None or record.state != "running":
                    # Deleted, or finalised by someone else, while the bytes
                    # moved. Nothing to start, nothing to report.
                    self._runners.pop(job_id, None)
                    return
                prep.emit(ready_message)
                if apply_to_config is not None:
                    apply_to_config(record.config)
                log_path = _job_log_path(self._output_root, job_id)
                if target.runner == "local":
                    runner = LocalJobRunner(record.metrics, log_file_path=log_path)
                else:
                    runner = HfCloudJobRunner(
                        record.metrics,
                        log_path,
                        target.flavor,
                        _resume_total_steps(record.config),
                    )
                try:
                    runner.start(job_id, record.config, output_dir)
                except Exception as exc:
                    logger.exception("Failed to start runner for job %s", job_id)
                    self._finalize_prepare_locked(job_id, "failed", f"Failed to start runner: {exc}")
                    notify = True
                    return
                if target.runner == "local":
                    record.process_pid = runner.pid()
                else:
                    record.hf_job_id = runner.hf_job_id()
                    record.hf_job_url = runner.hf_job_url()
                    # config was mutated by HfCloudJobRunner.start to set
                    # policy_repo_id; mirror it onto the record for the UI.
                    record.hf_repo_id = record.config.policy_repo_id
                self._runners[job_id] = runner
                self._persist(record, force=True)
                notify = True
        finally:
            if notify:
                self._notify_change()

    def _resolve_upload_resume(
        self, source: JobRecord, config: TrainingRequest
    ) -> tuple[tuple[Path, str, str] | None, str, str]:
        """Plan the Hub side of a LOCAL parent → CLOUD continuation.

        Returns (pending_upload, repo_id, step_dir): `pending_upload` is
        (checkpoint dir, repo id, step dir) when bytes still have to be pushed,
        or None when a previous continuation already pushed this exact step and
        the Hub still has it — re-resuming a step must not re-upload GBs.

        Called with the registry lock HELD (from the resume block in `start`),
        like the cloud→cloud resolver beside it — both read the Hub, and both do
        so before any record exists so a bad selection leaves nothing behind.

        Every refusal below is a ValueError (→ HTTP 400) raised before a record
        exists, because each one describes something the user has to change:
          * the checkpoint isn't resumable at all (delegated wholesale to
            _resolve_resume_config_path, so local→local and local→cloud can't
            disagree about what "resumable" means);
          * the upload wasn't consented to — an upload is a disclosure, so it is
            never a silent side effect of clicking Continue;
          * there is no Hub identity to upload as, or the server is offline.
        """
        # The same validation a local→local resume passes: complete checkpoint,
        # known step, real training_state/. Its train_config.json is
        # <dir>/checkpoints/<step>/pretrained_model/train_config.json, so the
        # checkpoint dir to upload — and the step dir naming it — come straight
        # back out of the path.
        train_config = Path(_resolve_resume_config_path(source, config.resume_from_step))
        checkpoint_dir = train_config.parent.parent
        step_dir = checkpoint_dir.name

        # Already on the Hub from an earlier continuation? Trust the record only
        # as far as the Hub confirms it: a deleted repo (or a half-finished push)
        # must produce a fresh upload, not a job that dies looking for bytes.
        if source.checkpoints_hub_repo_id and step_dir in source.checkpoints_hub_steps:
            with contextlib.suppress(Exception):
                if hub_checkpoint_has_training_state(
                    shared_hf_api(), source.checkpoints_hub_repo_id, step_dir
                ):
                    return None, source.checkpoints_hub_repo_id, step_dir

        if not config.upload_resume_checkpoint:
            raise ValueError(
                f"Continuing this run on {_RUNNER_LABELS['hf_cloud']} needs its "
                f"checkpoint at step {int(step_dir)} on the Hub, and it is only on "
                "this machine. Confirm the upload in the training form (it goes to "
                "a private repo), or continue the run locally instead."
            )
        if hf_hub_offline():
            raise ValueError(
                "Offline mode is on, so this run's checkpoint can't be uploaded to "
                "the Hub — continue it locally, or switch offline mode off."
            )
        whoami = cached_whoami()
        username = (whoami or {}).get("name")
        if not username:
            raise ValueError(
                "Continuing a local run on the cloud uploads its checkpoint to your "
                f"Hugging Face account, so you have to be signed in. Run '{LOGIN_COMMAND}' "
                "or paste a token in the app, then try again."
            )
        repo_id = checkpoints_staging_repo_id(username, source.id)
        return (checkpoint_dir, repo_id, step_dir), repo_id, step_dir

    def _finalize_prepare(self, job_id: str, state: JobState, error_message: str) -> None:
        """Lock-taking wrapper around _finalize_prepare_locked."""
        with self._lock:
            self._finalize_prepare_locked(job_id, state, error_message)
        self._notify_change()

    def _finalize_prepare_locked(self, job_id: str, state: JobState, error_message: str) -> None:
        """Finalise a job that never reached its trainer. Caller holds _lock.

        The watchdog's finalisation minus the exit code: it only ever sees jobs
        with a runner that HAD a process, and this one never did. Removing the
        PreparingJobRunner from `_runners` here is what ends its role — until
        then it answers is_running() True precisely so the watchdog leaves this
        window alone."""
        record = self._records.get(job_id)
        if record is None or record.state != "running":
            return
        record.state = state
        record.ended_at = time.time()
        record.error_message = error_message
        self._runners.pop(job_id, None)
        self._persist(record, force=True)

    def _unique_job_id(self, policy_type: str, dataset_repo_id: str) -> str:
        """_generate_job_id with a collision guard. The generated id embeds a
        second-granularity timestamp, so two jobs created within the same
        second would otherwise share an id and silently overwrite each other
        in the registry (and on disk). Suffix -2, -3, … until unused."""
        base = _generate_job_id(policy_type, dataset_repo_id)
        job_id = base
        n = 2
        while job_id in self._records or _job_dir(self._output_root, job_id).exists():
            job_id = f"{base}-{n}"
            n += 1
        return job_id

    def find_imported(self, source: str) -> JobRecord | None:
        """Return the already-registered imported record for `source`, if any.

        `source` is normalized first (whitespace, pasted Hub URLs, trailing
        slashes — see _normalize_import_source). Identity per import kind:
          * local dir → filesystem identity of the resolved path vs the stored
            output_dir (_paths_are_same_dir: samefile, so case variants on a
            case-insensitive filesystem and moved-cwd spellings still match —
            a plain string compare demonstrably missed a real duplicate pair);
          * hub repo → hf_repo_id compared CASE-INSENSITIVELY. Reversal of the
            earlier exact-match choice: HF repo ids are practically unique
            case-insensitively (the Hub redirects across casings), and the
            failure mode of exact matching is silent duplicates.
        """
        src = _normalize_import_source(source)
        if not src:
            return None
        local_path = Path(src).expanduser()
        local_key = str(local_path.resolve()) if local_path.is_dir() else None
        with self._lock:
            for r in self._records.values():
                if r.runner != "imported":
                    continue
                if local_key is not None:
                    if not r.hf_repo_id and r.output_dir and _paths_are_same_dir(r.output_dir, local_key):
                        return r
                elif (r.hf_repo_id or "").lower() == src.lower():
                    return r
        return None

    def register_imported(self, source: str, name: str | None = None) -> JobRecord:
        """Register an externally-trained model as a pointer-only pseudo-job.

        `source` is either an existing local directory (its path is stored in
        output_dir) or, failing that, a Hugging Face repo id (stored in
        hf_repo_id). The source must expose at least one checkpoint under the
        auto-detect rules, else ValueError. Nothing is copied; delete only
        removes the pointer.

        Idempotent per source: importing an already-registered path/repo
        returns the EXISTING record (its id and display alias untouched)
        instead of creating a second entry — see find_imported for the
        identity keys. The source is normalized first (whitespace, pasted Hub
        URLs, trailing slashes), and the normalized form is what gets stored."""
        src = _normalize_import_source(source)
        if not src:
            raise ValueError("source is required")

        existing = self.find_imported(src)
        if existing is not None:
            return existing

        local_path = Path(src).expanduser()
        if local_path.is_dir():
            resolved = str(local_path.resolve())
            ckpts = _list_imported_local(resolved)
            output_dir, hf_repo_id = resolved, None
        else:
            ckpts = _list_imported_hub(shared_hf_api(), src)
            output_dir, hf_repo_id = "", src

        if not ckpts:
            raise ValueError(
                f"No usable model at {src!r}. For a local path, expected a "
                "pretrained_model (config.json) or a checkpoints/<step>/"
                "pretrained_model tree. For a Hugging Face repo, the repo may "
                "not exist, be private without auth, or lack a model config."
            )

        # Best-effort policy type for the display name; inference reads the
        # real config from the checkpoint, so a wrong guess here is harmless.
        policy_type = "model"
        with contextlib.suppress(Exception):
            policy_type = str(_read_checkpoint_config(ckpts[-1]).get("type") or "model")

        with self._lock:
            job_id = self._unique_job_id(policy_type, "imported")
            record = JobRecord(
                id=job_id,
                # Identity only — no "Imported ·" prefix (the card's own
                # provenance chip says that) and no namespace or policy token
                # (the policy row says that). See derive_imported_title.
                name=name or derive_imported_title(hf_repo_id or output_dir),
                state="done",
                config=TrainingRequest(dataset_repo_id="(imported)", policy_type=policy_type),
                output_dir=output_dir,
                started_at=time.time(),
                ended_at=time.time(),
                runner="imported",
                hf_repo_id=hf_repo_id,
            )
            self._records[job_id] = record
            self._persist(record, force=True)
        # Two imports of one task derive the same title; re-run over the whole
        # set (not just the newcomer) so BOTH cards get a disambiguator and
        # neither is left as the bare, ambiguous-looking one.
        self._resolve_imported_names()
        self._notify_change()
        return record

    def rename(self, job_id: str, new_name: str) -> JobRecord:
        """Set a job's display alias. Metadata-only by design: the immutable
        identity (run id, output_dir, hub repo id) is never touched, so resume
        lineage (charts stitch across runs by id), live training/inference
        reads, imported-model hub identity (dedup on re-import), and remote
        HF Jobs / W&B names all keep working. The UI shows the alias and falls
        back to `name` when unset.

        Aliases are display-only, so uniqueness is NOT enforced (unlike
        calibration/robot renames, where the name is a file key). The same
        is_valid-style guard is applied for consistency (rejects path-ish
        characters); trimmed; empty ⇒ ValueError (→ HTTP 400)."""
        name = new_name.strip()
        if not name:
            raise ValueError("Display name cannot be empty.")
        if not is_valid_robot_name(name):
            raise ValueError("Invalid display name.")
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            record.display_name = name
            self._persist(record, force=True)
        self._notify_change()
        return record

    def stop(self, job_id: str) -> JobRecord:
        """Ask a running job to stop, and record that we asked.

        The intent is registered under the lock BEFORE any signal or cancel
        leaves this process, so the watchdog can never finalise a stop it
        doesn't know about — the reason every Stop press used to land in
        history as `failed`.

        Intent alone does not decide the outcome: the runner still gets to
        report that it finished on its own, or that it was already dead when we
        went to signal it. See classify_terminal_state for the full precedence.

        Works during a local fine-tune's base-checkpoint download too: that
        window has a PreparingJobRunner registered in place of the trainer, so
        the intent is recorded here as usual and the materialize thread
        finalises the run as `interrupted` when the download returns (it can't
        be aborted mid-flight — see _materialize_then_start). The 2s wait below
        will usually time out on that path, so the caller sees `running` and
        learns the outcome from the next poll.

        NOT covered: a cloud job cancelled outside MakerMods Lab (the HF web UI, or
        a platform-side kill that HF reports as CANCELED rather than ERROR).
        There is no intent recorded here for those, and HF's stage does not say
        who asked, so they stay `failed` rather than being guessed into
        `interrupted`.
        """
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
            # Raised under the lock (it used to be checked outside it) so the
            # intent below can't be recorded for a job that just finalised.
            if record.state != "running" or runner is None:
                raise JobNotRunningError(job_id)
            self._stop_requested.add(job_id)
        runner.stop()
        # The watchdog will finalise the record (state, ended_at, exit_code).
        # Wait briefly so the caller sees the new state in the response.
        for _ in range(20):
            time.sleep(0.1)
            with self._lock:
                if record.state != "running":
                    return record
        return record

    def drain_logs(self, job_id: str) -> builtins.list[LogLine]:
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
        if runner is None:
            return []
        return runner.stream_log_lines()

    def read_persisted_logs(self, job_id: str) -> builtins.list[LogLine]:
        """Read all log lines that have been written to disk for this job.

        Used by the frontend on Monitoring-page mount to seed the log panel
        with history (e.g. after navigating away and back, or after a MakerMods Lab
        restart marked the job 'interrupted').
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
        path = _job_log_path(self._output_root, job_id)
        if not path.exists():
            return []
        out: list[LogLine] = []
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(LogLine.model_validate_json(raw))
                except Exception:
                    continue  # skip a malformed line rather than 500ing
        return out

    def read_metrics_history(self, job_id: str) -> builtins.list[MetricsHistoryPoint]:
        """Reconstruct the per-step loss/lr/grad-norm series from log.jsonl.

        Walks the resume lineage (job -> resume source -> …, oldest first) and
        concatenates each run's points, so a resumed run's curve is continuous
        across the whole training rather than starting at the resume step. Stops
        at a missing ancestor (a deleted source) — the curve just starts later.

        Used by the frontend on Monitoring-page mount to seed the curves so they
        survive page reloads, navigation, and MakerMods Lab restarts. Re-parses on every
        call; cache later if a slow file ever shows up.
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
            chain: list[JobRecord] = []
            seen: set[str] = set()
            cur: JobRecord | None = self._records[job_id]
            while cur is not None and cur.id not in seen:
                chain.append(cur)
                seen.add(cur.id)
                parent_id = cur.config.resume_from_job_id
                cur = self._records.get(parent_id) if parent_id else None
        chain.reverse()  # oldest (root) first so steps ascend across the chain

        # Concatenate each run's points; dedupe by step (later run wins) in case
        # a resume boundary overlaps, then sort for a clean ascending curve.
        by_step: dict[int, MetricsHistoryPoint] = {}
        for record in chain:
            log_path = _job_log_path(self._output_root, record.id)
            for point in _read_log_metrics(log_path, _resume_total_steps(record.config)):
                by_step[point.step] = point
        return sorted(by_step.values(), key=lambda p: p.step)

    def _checkpoints_for(self, record: JobRecord) -> builtins.list[JobCheckpoint]:
        if record.runner == "imported":
            if record.hf_repo_id:
                return self._list_cloud_cached(record.hf_repo_id, _list_imported_hub)
            return _list_imported_local(record.output_dir)
        if record.runner == "local":
            return _list_local_checkpoints(record.output_dir)
        return self._list_cloud_cached(record.hf_repo_id)

    def list_checkpoints(self, job_id: str) -> builtins.list[JobCheckpoint]:
        """Return checkpoints saved for this job, ascending by step.

        Local jobs scan <output_dir>/checkpoints/. Cloud jobs introspect the
        Hub repo (30s TTL cache). Imported jobs auto-detect single-model vs
        checkpoints-tree from their local path or Hub repo id."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return self._checkpoints_for(record)

    def _list_cloud_cached(
        self, repo_id: str | None, fetch=_list_hub_checkpoints
    ) -> builtins.list[JobCheckpoint]:
        """30s-TTL cache over a hub checkpoint listing. `fetch(api, repo_id)`
        defaults to the training-job tree scan; imported hub models pass
        `_list_imported_hub` so they share the same cache + rate-limit budget."""
        if not repo_id:
            return []
        now = time.time()
        cached = self._cloud_ckpt_cache.get(repo_id)
        if cached is not None and cached[0] > now:
            return cached[1]
        result = fetch(shared_hf_api(), repo_id)
        self._cloud_ckpt_cache[repo_id] = (now + _CLOUD_CKPT_TTL_SECONDS, result)
        return result

    def _count_checkpoints(self, record: JobRecord) -> int:
        return len(self._checkpoints_for(record))

    def get_policy_config_summary(self, job_id: str, step: int) -> dict[str, object]:
        """Read the checkpoint's pretrained_model/config.json and return only
        the UX-relevant slice: policy type, expected camera names + their
        height/width, and whether the policy needs a --task string."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        ckpts = self.list_checkpoints(job_id)
        match = next((c for c in ckpts if c.step == step), None)
        if match is None:
            raise FileNotFoundError(f"No checkpoint at step {step} for job {record.id}")
        cfg = _read_checkpoint_config(match)
        policy_type = cfg.get("type")
        input_features = cfg.get("input_features") or {}
        image_features: dict[str, dict[str, int]] = {}
        for full_name, feat in input_features.items():
            if feat.get("type") != "VISUAL":
                continue
            shape = feat.get("shape") or []
            if len(shape) != 3:
                continue
            _channels, height, width = shape
            # The policy keys are 'observation.images.<name>'; the rollout CLI
            # takes just the suffix.
            name = full_name.split(".")[-1]
            image_features[name] = {"height": int(height), "width": int(width)}
        return {
            "policy_type": policy_type,
            "image_features": image_features,
            "requires_task": policy_type in _LANGUAGE_CONDITIONED_POLICY_TYPES,
            # Flat proprioceptive state / action widths. For an SO-101 arm this
            # is 6 (one per joint); a bimanual-trained checkpoint carries 12
            # (two arms). The inference modal compares this against the selected
            # robot's arm count to explain a single-arm/bimanual mismatch before
            # the user hits Start. None when the checkpoint omits the feature.
            "state_dim": _flat_feature_dim(input_features.get("observation.state")),
            "action_dim": _flat_feature_dim((cfg.get("output_features") or {}).get("action")),
        }

    def delete(self, job_id: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.state == "running":
                raise JobNotRunningError(job_id)
            self._records.pop(job_id, None)
            self._runners.pop(job_id, None)
            self._last_persist_at.pop(job_id, None)
            self._stop_requested.discard(job_id)
            self._prepare_threads.pop(job_id, None)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(_job_dir(self._output_root, job_id))
        self._notify_change()

    def shutdown(self) -> None:
        """For tests / orderly process exit. Not wired to FastAPI lifespan today."""
        self._stop_watchdog.set()

    # -- internals --

    def _load_from_disk(self) -> None:
        for job_dir in self._output_root.glob("*/"):
            meta = job_dir / "job.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text())
                record = JobRecord.model_validate(data)
            except Exception as exc:
                logger.warning("Skipping malformed job.json at %s: %s", meta, exc)
                continue
            if record.state == "running":
                if record.runner == "local":
                    pid = record.process_pid
                    if pid is not None and _pid_alive(pid):
                        logger.info(
                            "Re-attaching to detached local job %s (pid %d)",
                            record.id,
                            pid,
                        )
                        runner = TailingJobRunner(
                            record.metrics,
                            _job_log_path(self._output_root, record.id),
                            pid,
                            _resume_total_steps(record.config),
                        )
                        runner.start_tailing()
                        self._runners[record.id] = runner
                    else:
                        record.state = "interrupted"
                        if record.ended_at is None:
                            record.ended_at = time.time()
                        self._write_meta(record)
                elif record.runner == "hf_cloud" and record.hf_job_id and record.hf_flavor:
                    # Always reattach; the status poller is the source of truth
                    # for terminal state. If the HF job already finished, the
                    # next inspect_job call resolves the final stage and the
                    # watchdog finalises the record. A transient HF API hiccup
                    # at startup no longer strands the record as "interrupted".
                    logger.info(
                        "Re-attaching to HF Cloud job %s (hf_job_id=%s)",
                        record.id,
                        record.hf_job_id,
                    )
                    from .runners.hf_cloud import HfCloudJobRunner

                    runner = HfCloudJobRunner(
                        record.metrics,
                        _job_log_path(self._output_root, record.id),
                        record.hf_flavor,
                        _resume_total_steps(record.config),
                    )
                    runner.reattach(record.hf_job_id)
                    self._runners[record.id] = runner
                else:
                    # Malformed running record — mark interrupted.
                    record.state = "interrupted"
                    if record.ended_at is None:
                        record.ended_at = time.time()
                    self._write_meta(record)
            self._records[record.id] = record

    def _dedupe_imported_records(self) -> None:
        """One-time collapse of duplicate imported pointers left behind before
        dedup-at-registration existed (same local path or hub repo id
        registered more than once).

        Runs at boot, after _load_from_disk and before the watchdog starts
        (single-threaded, so no lock needed). Per identity group: keep the
        OLDEST record; if the keeper has no alias, migrate the newest
        duplicate's display_name onto it. Duplicates are dropped from the
        in-memory map; their job dir is deleted ONLY when it contains nothing
        but job.json — anything else (weights, logs, leftovers) means the
        files stay put and we just log. Conservative by design: malformed
        pointers (no identity) are left alone entirely."""
        groups: dict[tuple[str, str], list[JobRecord]] = {}
        for r in self._records.values():
            if r.runner != "imported":
                continue
            if r.hf_repo_id:
                # Case-insensitive: HF repo ids are practically unique
                # case-insensitively (same reversal as find_imported).
                key = ("hub", r.hf_repo_id.lower())
            elif r.output_dir:
                # Filesystem identity (device:inode) so spellings that differ
                # only by case on a case-insensitive filesystem — the real
                # duplicate pair — group together. Unstat-able paths (source
                # moved/deleted) fall back to the raw string: conservative,
                # they only group with byte-identical spellings.
                try:
                    st = os.stat(r.output_dir)
                    key = ("local", f"{st.st_dev}:{st.st_ino}")
                except OSError:
                    key = ("local", r.output_dir)
            else:
                continue  # no identity — when in doubt, keep it
            groups.setdefault(key, []).append(r)

        for (kind, _ident), records in groups.items():
            if len(records) < 2:
                continue
            records.sort(key=lambda r: r.started_at)
            keeper, dupes = records[0], records[1:]
            if keeper.display_name is None:
                # Newest aliased duplicate wins — it's the user's latest word.
                for dup in reversed(dupes):
                    if dup.display_name:
                        keeper.display_name = dup.display_name
                        self._write_meta(keeper)
                        break
            for dup in dupes:
                self._records.pop(dup.id, None)
                dup_dir = _job_dir(self._output_root, dup.id)
                removed = False
                if dup_dir.is_dir():
                    try:
                        only_meta = [p.name for p in dup_dir.iterdir()] == ["job.json"]
                    except OSError:
                        only_meta = False
                    if only_meta:
                        shutil.rmtree(dup_dir, ignore_errors=True)
                        removed = True
                    else:
                        logger.info(
                            "Duplicate imported model %s: leaving %s in place (contains more than job.json).",
                            dup.id,
                            dup_dir,
                        )
                logger.info(
                    "Collapsed duplicate imported model %s into %s (same %s %r)%s",
                    dup.id,
                    keeper.id,
                    kind,
                    keeper.hf_repo_id or keeper.output_dir,
                    "" if removed else " — pointer dropped from the registry only",
                )

    def _resolve_imported_names(self) -> None:
        """Re-derive every auto-named imported record's title, collisions and all.

        Idempotent and whole-set by design, which is what lets it run both at
        boot and after each import: the title is recomputed from the source each
        time, so a record named before titles were derived ("Imported · …") is
        upgraded, and a suffix added to break a collision is recomputed rather
        than accumulated. Records the user named — explicitly at import, or via
        `rename`, which writes `display_name` and always wins on the card — are
        never touched (see _auto_imported_name).

        Collisions are counted per (title, policy type), the pair a card
        actually renders: an ACT and a SmolVLA of one task read as two rows
        already, because the Policy row below the title says so. Grouping on the
        stored `config.policy_type` rather than a re-derivation keeps the rule
        honest — when the import could not read a policy type, the two cards
        really are indistinguishable and really do need the suffix.

        Oldest-first so the disambiguated labels are stable across restarts and
        don't depend on which record happened to be read from disk first.
        """
        with self._lock:
            records = sorted(
                (r for r in self._records.values() if r.runner == "imported"),
                key=lambda r: (r.started_at, r.id),
            )
        targets: builtins.list[JobRecord] = []
        entries: builtins.list[tuple[str, builtins.list[str]]] = []
        for record in records:
            derived = _auto_imported_name(record)
            if derived is None:
                continue
            targets.append(record)
            entries.append((derived, imported_name_suffixes(record.hf_repo_id or record.output_dir)))

        resolved_names = dedupe_display_names(entries, group_keys=[r.config.policy_type for r in targets])
        for record, resolved in zip(targets, resolved_names, strict=True):
            if record.name == resolved:
                continue
            logger.info("Renamed imported model %s: %r -> %r", record.id, record.name, resolved)
            record.name = resolved
            self._write_meta(record)

    def _start_watchdog(self) -> None:
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="job-registry-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Watchdog tick failed: %s", exc)
            self._stop_watchdog.wait(1.0)

    def _tick(self) -> None:
        with self._lock:
            running_ids = [jid for jid, r in self._records.items() if r.state == "running"]

        progress_snapshots: builtins.list[dict] = []

        for jid in running_ids:
            with self._lock:
                runner = self._runners.get(jid)
                record = self._records.get(jid)
            if runner is None or record is None:
                continue
            if runner.is_running():
                # Persist metric snapshot at most once per second.
                self._persist(record, force=False)
                progress_snapshots.append(
                    {
                        "id": record.id,
                        "state": record.state,
                        "metrics": record.metrics.model_dump(),
                        "checkpoint_count": self._count_checkpoints(record),
                    }
                )
                continue

            # Subprocess exited since the last tick. Finalise.
            rc = runner.returncode()
            # A stop counts only if we asked for it AND the runner didn't tell
            # us it never got to signal anything (already-dead process). A
            # runner that can't answer abstains rather than vetoing.
            with self._lock:
                asked_to_stop = jid in self._stop_requested
            signalled = _runner_hook(runner, "stop_signalled")
            stop_requested = asked_to_stop and signalled is not False
            state = classify_terminal_state(
                returncode=rc,
                stop_requested=stop_requested,
                terminal_stage=_runner_hook(runner, "terminal_stage"),
            )
            with self._lock:
                record.state = state
                record.ended_at = time.time()
                record.exit_code = rc
                if record.error_message is None:
                    if state == "interrupted":
                        # Never the synthetic exit-code text here: that message
                        # on a run the user stopped themselves is what made a
                        # deliberate pause look like a broken model.
                        record.error_message = STOPPED_BY_REQUEST_MESSAGE
                    elif state == "failed":
                        # Prefer a runner-supplied reason (e.g. HF Jobs'
                        # 'Job timeout') over the synthetic exit-code message.
                        reason = _runner_hook(runner, "terminal_message")
                        record.error_message = reason or f"Subprocess exited with code {rc}"
                self._runners.pop(jid, None)
                self._stop_requested.discard(jid)
            self._persist(record, force=True)
            self._notify_change()

        self._notify_progress(progress_snapshots)

    def _persist(self, record: JobRecord, force: bool) -> None:
        now = time.time()
        last = self._last_persist_at.get(record.id, 0.0)
        if not force and (now - last) < _PERSIST_THROTTLE_SECONDS:
            return
        self._last_persist_at[record.id] = now
        self._write_meta(record)

    def _write_meta(self, record: JobRecord) -> None:
        path = _job_meta_path(self._output_root, record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write never strands a half-written file
        # that would skip the job on next _load_from_disk.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)


# Module-level singleton. Anchored to ~/.cache so history survives launches
# from different cwds. JobRegistry.__init__ migrates legacy `<cwd>/outputs/train/`
# job dirs into this root on first boot. MAKERMODSLAB_OUTPUT_ROOT overrides for tests.
_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("MAKERMODSLAB_OUTPUT_ROOT")
    or (Path.home() / ".cache" / "huggingface" / "lerobot" / "outputs" / "train")
).expanduser()
job_registry = JobRegistry(_DEFAULT_OUTPUT_ROOT)

__all__ = [
    "JobState",
    "JobTarget",
    "TrainingMetrics",
    "LogLine",
    "JobRecord",
    "JobCheckpoint",
    "MetricsHistoryPoint",
    "JobRunner",
    "LocalJobRunner",
    "PreparingJobRunner",
    "JobRegistry",
    "make_snapshot_progress_tqdm",
    "JobAlreadyRunningError",
    "JobNotFoundError",
    "JobNotRunningError",
    "job_registry",
    "parse_metrics_into",
    "classify_terminal_state",
    "STOPPED_BY_REQUEST_MESSAGE",
]
