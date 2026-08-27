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

"""LAN-node runner — runs a training job on another MakerMods Lab node.

The peer's own typed v1 job API is the whole transport: submit via
POST /api/v1/jobs/training (as a LOCAL run on the peer), poll the remote
JobRecord for liveness/metrics/terminal state, drain GET /jobs/{id}/logs for
log lines, and POST /jobs/{id}/stop to stop. The peer is addressed by
instance_id through the node registry (nodes.py), never by a raw URL, so a
machine that changed address since it was registered still resolves.

Design decisions, stated once:

* **Datasets travel via the Hub** (deliberate for this phase). A LAN peer can
  no more see this machine's LeRobot cache than an HF pod can, so start()
  resolves the dataset's Hub id and pushes a local-only dataset first — the
  same shared rule hf_cloud uses (runners/_dataset.py).
* **The run's outputs live on the peer.** The job runs as a plain local run
  there, so its checkpoints sit on the peer's disk; this side records the
  outcome, metrics and log. Browsing/fetching remote checkpoints comes later
  with the SDK.
* **No timer threads.** The registry watchdog already ticks every runner at
  1Hz; is_running() turns that into at most one remote record probe per
  REMOTE_POLL_INTERVAL_S, and stream_log_lines() (driven by the monitor's
  /logs poll) is one drain per call. hf_cloud needs its own threads because
  SSE streams must be consumed continuously; plain polling doesn't.
* **Blip tolerance.** A failed probe never fails the job by itself: the job
  stays `running` while probes keep failing for up to PEER_LOST_GRACE_S
  (measured from the first failure of an unbroken streak; any success resets
  it). Only a peer that stays gone past the grace window becomes a terminal
  failure — stage "UNREACHABLE", with a message naming the node — because at
  that point the record would otherwise say "running" forever with nothing
  behind it. The run may well still be training on the peer; the message says
  so instead of blaming the model.
* **Terminal mapping relays the peer's own verdict.** The remote registry
  already classified its run with evidence local to it (its stop intent, its
  exit status), so its terminal state maps 1:1 onto a stage for
  classify_terminal_state: done → COMPLETED, failed → ERROR, interrupted →
  INTERRUPTED (adopted as `interrupted` regardless of local stop intent — a
  stop pressed on the peer must not read as a failure here). The remote
  error_message rides along as terminal_message.
* **Logs reuse the peer's incremental mechanism.** GET /jobs/{id}/logs drains
  the peer runner's live queue server-side, so each call returns only what is
  new — nothing to invent. (That drain is shared with anyone watching the
  same job on the peer itself; a line taken by one watcher won't reach the
  other's live tail.) Drained lines are teed into this job's local log.jsonl,
  and when the run ends the peer's full persisted log (GET /jobs/{id}/log-file)
  replaces the local file, so history is complete here even if nobody watched.

Single-shot, like every runner: instantiate per job. reattach() takes over a
persisted lan_node record after a restart using its stored url + remote id.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..datasets import resolve_hub_repo_id
from ..jobs import LogLine, TrainingMetrics
from ..train import TrainingRequest
from ._dataset import ensure_dataset_on_hub

logger = logging.getLogger(__name__)

# Floor between two remote-record probes. The watchdog ticks at 1Hz; this
# keeps that from becoming 1Hz of HTTP against the peer. Matches the cadence
# hf_cloud polls inspect_job at.
REMOTE_POLL_INTERVAL_S = 5.0

# How long an unbroken streak of failed probes may last before the job is
# declared lost (terminal failure). 120s rides out a peer reboot, a Wi-Fi
# handoff or a laptop lid-close on either end, while still bounding how long
# a record can claim "running" with nothing observable behind it.
PEER_LOST_GRACE_S = 120.0

# Per-request budgets. Submission spawns a subprocess on the peer (its
# registry start is synchronous), so it gets more room than a poll.
_SUBMIT_TIMEOUT_S = 30.0
_POLL_TIMEOUT_S = 5.0


def localize_config_for_lan_node(config: TrainingRequest) -> None:
    """Strip host-machine specifics at the node-submission boundary, before
    the config goes over the wire. Mutates in place — the mutated config is
    what gets persisted on the JobRecord, so the record reflects what the
    peer actually ran. The lan twin of localize_config_for_cloud.

    Raises ValueError (→ HTTP 400) for host-path inputs that cannot work on
    another machine, so the user gets a clear message instead of a remote
    crash.
    """
    # A host-local config_path (the local-resume signal) can't exist on the
    # peer, and unlike the cloud there is no Hub-resume channel wired for a
    # peer's local run yet — the registry refuses resume on lan_node up front,
    # so this is belt-and-braces for requests that bypass it.
    if config.resume or config.config_path:
        raise ValueError(
            "Resuming a run on another node isn't supported yet: the source "
            "checkpoint lives on this machine, not on the peer. Continue the "
            "run locally or on Hugging Face Cloud instead."
        )
    # A fine-tune base may be a Hub ref (the peer resolves it there like any
    # local run would); a host path cannot work — the peer has no view of this
    # machine's disk.
    if config.policy_pretrained_path and Path(config.policy_pretrained_path).is_absolute():
        raise ValueError(
            "A job on another node can't fine-tune from a checkpoint on this "
            "machine — the peer has no view of this disk. Push the source model "
            "to the Hub and fine-tune from the Hub copy instead."
        )
    # The peer resolves the dataset from the Hub by repo_id; a host-local
    # dataset root doesn't exist there.
    config.dataset_root = None
    # The host's auto-detected device (mps on a Mac) is meaningless on the
    # peer, whose hardware we don't know — reset to "auto" so the peer's own
    # trainer detects it. An explicit "cpu" is an instruction, not a
    # detection, so it travels.
    if config.policy_device != "cpu":
        config.policy_device = "auto"


class LanNodeJobRunner:
    """Run a training on another MakerMods Lab node. Single-shot — one per job."""

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        node_instance_id: str,
        *,
        registry=None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._node_instance_id = node_instance_id
        # None ⇒ the module singleton, read at call time so tests (and a
        # future dependency injection) can swap it.
        self._registry = registry
        self._transport = transport
        self._clock = clock
        self._node_url: str | None = None
        self._remote_job_id: str | None = None
        self._log_file = None  # type: ignore[assignment]
        # Poll pacing + blip tracking (see the module docstring).
        self._last_poll_at: float | None = None
        self._probe_failing_since: float | None = None
        # Terminal snapshot, set once (idempotent) — the mapped stage, the
        # peer's message, and the remote exit code.
        self._terminal_stage: str | None = None
        self._terminal_message: str | None = None
        self._remote_exit_code: int | None = None
        self._wandb_run_url: str | None = None
        # Tri-state: None until stop() learns anything, True when the peer
        # accepted the stop for a live run, False when it refused (the run
        # had already ended). See stop_signalled().
        self._stop_accepted: bool | None = None

    # -- plumbing --

    def _node_registry(self):
        if self._registry is not None:
            return self._registry
        from .. import nodes  # module attribute, so tests can swap the singleton

        return nodes.node_registry

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(transport=self._transport, timeout=timeout)

    def _log_line(self, message: str) -> None:
        """Append a wrapper-style line to the job's local log file."""
        if self._log_file is None:
            return
        line = LogLine(timestamp=time.time(), message=message)
        try:
            self._log_file.write(line.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("Could not write LAN-job log line: %s", exc)

    def _ensure_dataset_on_hub(self, local_repo_id: str, hub_repo_id: str) -> None:
        """Shared remote-runner rule — see runners/_dataset.ensure_dataset_on_hub."""
        ensure_dataset_on_hub(local_repo_id, hub_repo_id, self._log_line)

    # -- lifecycle --

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # output_dir is the host-local path the registry pins; the peer's own
        # registry pins its own for the remote record.
        del output_dir
        if self._remote_job_id is not None:
            raise RuntimeError("LanNodeJobRunner already started")

        from ..nodes import NodeUnreachableError  # local import to avoid a cycle at module load

        peer = self._node_registry().resolve(self._node_instance_id)
        self._node_url = peer.url

        localize_config_for_lan_node(config)

        # Open the log file early so dataset-upload progress is recorded
        # before the job is submitted.
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

        # Same reasoning as hf_cloud: the peer resolves datasets from the Hub
        # by NAMESPACED id, so resolve once, pin it into the config (that is
        # what the record persists as "what actually ran"), and push a
        # local-only dataset first.
        local_dataset_repo_id = config.dataset_repo_id
        config.dataset_repo_id = resolve_hub_repo_id(local_dataset_repo_id)
        self._ensure_dataset_on_hub(local_dataset_repo_id, config.dataset_repo_id)

        body = {"config": json.loads(config.model_dump_json()), "target": {"runner": "local"}}
        logger.info("Submitting job %s to node %s (%s)", job_id, self._node_instance_id, self._node_url)
        try:
            with self._client(_SUBMIT_TIMEOUT_S) as client:
                response = client.post(f"{self._node_url}/api/v1/jobs/training", json=body)
        except httpx.HTTPError as exc:
            raise NodeUnreachableError(
                f"could not submit the job to the node at {self._node_url}: {exc}"
            ) from exc
        if response.status_code != 201:
            detail = None
            with contextlib.suppress(Exception):
                detail = response.json().get("detail")
            raise RuntimeError(
                f"The node at {self._node_url} refused the job "
                f"(HTTP {response.status_code}): {detail or response.text}"
            )
        remote = response.json()
        self._remote_job_id = remote["id"]
        self._log_line(f"[node] job submitted to {self._node_url} as {self._remote_job_id}")
        # The submission itself just proved the peer live; start the poll
        # clock here so the first watchdog tick doesn't immediately re-ask.
        self._last_poll_at = self._clock()

    def reattach(self, remote_job_id: str, node_url: str) -> None:
        """Take over a persisted lan_node record after a process restart.

        Uses the record's stored URL directly (no registry probe at boot — a
        moved peer surfaces through the normal grace window instead) and lets
        the watchdog's polling drive finalisation, exactly like hf_cloud's
        reattach lets its status poller.
        """
        if self._remote_job_id is not None:
            raise RuntimeError("LanNodeJobRunner already started")
        self._remote_job_id = remote_job_id
        self._node_url = node_url
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

    # -- terminal bookkeeping --

    def _set_terminal(self, stage: str, message: str | None, exit_code: int | None) -> None:
        """Record the run's terminal snapshot. Idempotent, like hf_cloud's."""
        if self._terminal_stage is not None:
            return
        self._terminal_stage = stage
        self._terminal_message = message
        self._remote_exit_code = exit_code
        self._adopt_peer_log_file()
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None

    def _adopt_peer_log_file(self) -> None:
        """Replace the local log.jsonl with the peer's full persisted log.

        The live /logs drain only captured what somebody here was watching;
        the peer's log-file endpoint has the authoritative whole. Best-effort:
        an unreachable peer (the UNREACHABLE terminal) just keeps whatever was
        already tee'd locally.
        """
        if self._remote_job_id is None or self._node_url is None:
            return
        try:
            with self._client(_POLL_TIMEOUT_S) as client:
                response = client.get(f"{self._node_url}/api/v1/jobs/{self._remote_job_id}/log-file")
            response.raise_for_status()
            lines = [LogLine.model_validate(item) for item in response.json()["logs"]]
        except Exception as exc:
            logger.info("Could not adopt the peer's log file: %s", exc)
            return
        if not lines:
            return
        try:
            tmp = self._log_file_path.with_suffix(self._log_file_path.suffix + ".tmp")
            tmp.write_text("".join(line.model_dump_json() + "\n" for line in lines))
            os.replace(tmp, self._log_file_path)
        except Exception as exc:
            logger.warning("Could not write the adopted peer log: %s", exc)

    def _adopt_remote_record(self, doc: dict) -> None:
        """Mirror one polled remote JobRecord onto this runner's state."""
        try:
            for field, value in TrainingMetrics.model_validate(doc.get("metrics") or {}):
                setattr(self._metrics, field, value)
        except Exception as exc:
            logger.debug("Could not mirror remote metrics: %s", exc)
        if self._wandb_run_url is None and doc.get("wandb_run_url"):
            self._wandb_run_url = doc["wandb_run_url"]
        state = doc.get("state")
        if state in ("running", "queued"):
            # Both are in-flight. "queued" appears when the PEER's local slot
            # (or robot) is busy and its registry parked our submission in its
            # own training queue (PR #83) — the run has not started, but it is
            # coming. Filing it as terminal here classified a healthy queued
            # offload as ERROR on the very first poll.
            return
        # The peer's registry already classified the outcome with the evidence
        # local to the run; relay its verdict as a stage (see module docstring).
        stage = {"done": "COMPLETED", "interrupted": "INTERRUPTED"}.get(state, "ERROR")
        self._set_terminal(stage, doc.get("error_message"), doc.get("exit_code"))

    def _poll_remote_record(self) -> None:
        """One bounded probe of the remote JobRecord, with blip tracking."""
        now = self._clock()
        if self._last_poll_at is not None and now - self._last_poll_at < REMOTE_POLL_INTERVAL_S:
            return
        self._last_poll_at = now
        try:
            with self._client(_POLL_TIMEOUT_S) as client:
                response = client.get(f"{self._node_url}/api/v1/jobs/{self._remote_job_id}")
        except httpx.HTTPError as exc:
            self._note_probe_failure(now, exc)
            return
        if response.status_code == 404:
            # Definitive, not a blip: the peer answered and doesn't know the
            # job (deleted there). Nothing will ever finish.
            self._set_terminal(
                "DELETED",
                f"The node at {self._node_url} no longer has this job — it was deleted on the node.",
                None,
            )
            return
        if response.status_code >= 400:
            self._note_probe_failure(now, RuntimeError(f"HTTP {response.status_code}"))
            return
        self._probe_failing_since = None
        try:
            self._adopt_remote_record(response.json())
        except Exception as exc:
            logger.warning("Malformed job record from %s: %s", self._node_url, exc)

    def _note_probe_failure(self, now: float, exc: Exception) -> None:
        if self._probe_failing_since is None:
            self._probe_failing_since = now
            logger.info("Probe of node %s failed (tolerating): %s", self._node_url, exc)
            return
        if now - self._probe_failing_since >= PEER_LOST_GRACE_S:
            self._set_terminal(
                "UNREACHABLE",
                f"Lost contact with the node at {self._node_url} for over "
                f"{int(PEER_LOST_GRACE_S)}s. The run may still be training on that "
                "machine — check it there; this record can no longer follow it.",
                None,
            )

    # -- JobRunner protocol --

    def stop(self) -> None:
        if self._remote_job_id is None:
            return
        try:
            with self._client(_SUBMIT_TIMEOUT_S) as client:
                response = client.post(f"{self._node_url}/api/v1/jobs/{self._remote_job_id}/stop")
        except httpx.HTTPError as exc:
            # Delivery unproven either way; stop_signalled stays None (abstain).
            logger.warning("Could not deliver stop to node %s: %s", self._node_url, exc)
            return
        if response.status_code < 400:
            self._stop_accepted = True
            with contextlib.suppress(Exception):
                doc = response.json()
                self._adopt_remote_record(doc)
                # A stop accepted for a QUEUED run is the peer's CANCEL: its
                # registry removed the record outright (nothing ever started,
                # so there is no history to keep), and the body we adopted is
                # that removed record — still saying "queued", which the
                # adoption above rightly reads as in-flight. Without settling
                # it here, the next poll 404s and DELETED classifies a
                # deliberate cancel as `failed`. INTERRUPTED is the relayed
                # verdict a stop of a live run would have produced.
                if doc.get("state") == "queued":
                    self._set_terminal(
                        "INTERRUPTED",
                        "Cancelled while it was still queued on the node — the run never started.",
                        None,
                    )
        else:
            # 409 (not running) / 404: the run had already ended on its own —
            # the next poll adopts its real outcome, and we must not claim it.
            self._stop_accepted = False

    def stop_signalled(self) -> bool | None:
        """True once the peer verifiably accepted the stop for a LIVE run;
        False when it refused because the run had already ended; None when
        nothing is known (never asked, or the request never arrived)."""
        return self._stop_accepted

    def is_running(self) -> bool:
        if self._remote_job_id is None:
            return False
        if self._terminal_stage is None:
            self._poll_remote_record()
        return self._terminal_stage is None

    def returncode(self) -> int | None:
        # The peer's exit code, mirrored for the record; classification runs
        # on terminal_stage(), which is present whenever this is.
        if self._terminal_stage is None:
            return None
        return self._remote_exit_code

    def stream_log_lines(self) -> list[LogLine]:
        """Drain the peer's incremental /logs feed; tolerant of blips ([])."""
        if self._remote_job_id is None:
            return []
        try:
            with self._client(_POLL_TIMEOUT_S) as client:
                response = client.get(f"{self._node_url}/api/v1/jobs/{self._remote_job_id}/logs")
            response.raise_for_status()
            lines = [LogLine.model_validate(item) for item in response.json()["logs"]]
        except Exception as exc:
            logger.debug("Log drain from %s failed: %s", self._node_url, exc)
            return []
        for line in lines:
            if self._log_file is not None:
                try:
                    self._log_file.write(line.model_dump_json() + "\n")
                except Exception as exc:
                    logger.warning("Could not persist LAN log line: %s", exc)
        return lines

    # -- optional hooks / accessors --

    def terminal_stage(self) -> str | None:
        """The relayed terminal stage: COMPLETED / INTERRUPTED / ERROR from
        the remote record's state, DELETED for a job the peer dropped, or
        UNREACHABLE for a peer that stayed gone past the grace window."""
        return self._terminal_stage

    def terminal_message(self) -> str | None:
        """The remote record's error_message (which carries the peer's own
        stopped-at-request / failure wording), or this side's lost-contact
        text for UNREACHABLE."""
        return self._terminal_message

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    def node_url(self) -> str | None:
        return self._node_url

    def remote_job_id(self) -> str | None:
        return self._remote_job_id
