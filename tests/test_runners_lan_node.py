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
"""Tests for makermodslab.runners.lan_node — running a training job on another
MakerMods Lab node over its own typed job API.

The peer is an httpx.MockTransport standing in for a real node (the same
no-sockets pattern as test_nodes.py), programmed with authentic wire shapes:
its responses are built from the real JobRecord/LogLine models, so a drift in
the job API's shape fails here rather than only against a live peer. The
runner's clock is injected, so the poll pacing and the lost-peer grace window
are driven without sleeps.

Follows test_runners_hf_cloud.py's split: config localization and terminal
classification as pure-function tests, the runner exercised over mocked
transport only — no subprocess, no thread happy paths (repo policy; the local
trainer on the peer's side is out of scope here).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

LOCAL_ID = "aa" * 16
PEER_ID = "bb" * 16
PEER_URL = "http://peer-a:8000"


# ---------------------------------------------------------------------------
# Fakes: clock, peer node.
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakePeer:
    """A programmable MakerMods Lab node behind an httpx.MockTransport.

    Serves the identity handshake (/api/v1/health) and the slice of the job
    API the runner drives: submit, poll, incremental logs, full log file,
    stop. Wire shapes come from the real Pydantic models so the fake cannot
    quietly diverge from the app's own API.
    """

    def __init__(self, instance_id: str = PEER_ID, url: str = PEER_URL) -> None:
        self.instance_id = instance_id
        self.url = url
        self.down = False
        self.next_remote_id = "remote-job-1"
        self.records: dict[str, dict] = {}  # remote_job_id -> JobRecord dict
        self.pending_logs: dict[str, list[dict]] = {}  # drained by GET /logs
        self.log_history: dict[str, list[dict]] = {}  # full log, GET /log-file
        self.submissions: list[dict] = []
        self.refuse_submission: tuple[int, str] | None = None  # (status, detail)
        self.record_probes = 0  # GET /jobs/{id} count

    # -- programming the peer --

    def finish(self, remote_id: str, *, state: str, exit_code=None, error_message=None) -> None:
        rec = self.records[remote_id]
        rec["state"] = state
        rec["exit_code"] = exit_code
        rec["error_message"] = error_message
        rec["ended_at"] = time.time()

    def set_metrics(self, remote_id: str, **fields) -> None:
        self.records[remote_id]["metrics"].update(fields)

    def add_log(self, remote_id: str, message: str) -> None:
        from makermodslab.jobs import LogLine

        line = json.loads(LogLine(timestamp=time.time(), message=message).model_dump_json())
        self.pending_logs.setdefault(remote_id, []).append(line)
        self.log_history.setdefault(remote_id, []).append(line)

    def _health_doc(self) -> dict:
        return {
            "status": "ok",
            "message": "FastAPI server is running",
            "version": "1.2.3",
            "instance_id": self.instance_id,
            "capabilities": {"serves_ui": True, "accepts_jobs": True},
        }

    def _record_for(self, submitted_config: dict) -> dict:
        from makermodslab.jobs import JobRecord
        from makermodslab.train import TrainingRequest

        record = JobRecord(
            id=self.next_remote_id,
            job_number=7,
            name="remote run",
            state="running",
            config=TrainingRequest.model_validate(submitted_config),
            output_dir="/peer/outputs/train/remote-job-1/run",
            started_at=time.time(),
            runner="local",
            process_pid=4242,
        )
        return json.loads(record.model_dump_json())

    # -- the wire --

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.down:
                raise httpx.ConnectError("connection refused", request=request)
            path = request.url.path
            if path == "/api/v1/health":
                return httpx.Response(200, json=self._health_doc())
            if path == "/api/v1/jobs/training" and request.method == "POST":
                if self.refuse_submission is not None:
                    status, detail = self.refuse_submission
                    return httpx.Response(status, json={"detail": detail})
                body = json.loads(request.content)
                self.submissions.append(body)
                remote_id = self.next_remote_id
                record = self._record_for(body["config"])
                self.records[remote_id] = record
                return httpx.Response(201, json=record)
            if path.startswith("/api/v1/jobs/"):
                parts = path.removeprefix("/api/v1/jobs/").split("/")
                remote_id = parts[0]
                record = self.records.get(remote_id)
                if record is None:
                    return httpx.Response(404, json={"detail": f"Job {remote_id!r} not found"})
                if len(parts) == 1 and request.method == "GET":
                    self.record_probes += 1
                    return httpx.Response(200, json=record)
                if parts[1:] == ["logs"]:
                    drained = self.pending_logs.pop(remote_id, [])
                    return httpx.Response(200, json={"logs": drained})
                if parts[1:] == ["log-file"]:
                    return httpx.Response(200, json={"logs": self.log_history.get(remote_id, [])})
                if parts[1:] == ["stop"] and request.method == "POST":
                    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE

                    if record["state"] == "queued":
                        # Mirror JobRegistry.stop's cancel path: a queued run
                        # is removed outright, and the response body is the
                        # removed record — still saying "queued".
                        del self.records[remote_id]
                        return httpx.Response(200, json=record)
                    if record["state"] != "running":
                        return httpx.Response(409, json={"detail": "not running"})
                    self.finish(
                        remote_id,
                        state="interrupted",
                        exit_code=-15,
                        error_message=STOPPED_BY_REQUEST_MESSAGE,
                    )
                    return httpx.Response(200, json=record)
            raise AssertionError(f"unexpected request to fake peer: {request.method} {path}")

        return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Fixtures (the same isolation test_nodes.py uses).
# ---------------------------------------------------------------------------


@pytest.fixture
def local_identity(monkeypatch: pytest.MonkeyPatch) -> str:
    from makermodslab.utils import config as cfg

    monkeypatch.setattr(cfg, "_instance_id_cache", LOCAL_ID)
    return LOCAL_ID


@pytest.fixture
def nodes_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from makermodslab.utils import config as cfg

    path = tmp_path / "nodes.json"
    monkeypatch.setattr(cfg, "NODES_FILE", str(path))
    return path


@pytest.fixture
def peer() -> FakePeer:
    return FakePeer()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _node_registry(peer: FakePeer, clock: FakeClock):
    from makermodslab.nodes import NodeRegistry

    registry = NodeRegistry(clock=clock, transport=peer.transport())
    registry.add(peer.url)
    return registry


def _request(**overrides):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def _runner(peer: FakePeer, clock: FakeClock, tmp_path: Path, registry=None):
    from makermodslab.jobs import TrainingMetrics
    from makermodslab.runners.lan_node import LanNodeJobRunner

    return LanNodeJobRunner(
        TrainingMetrics(),
        tmp_path / "log.jsonl",
        peer.instance_id,
        registry=registry if registry is not None else _node_registry(peer, clock),
        transport=peer.transport(),
        clock=clock,
    )


@pytest.fixture
def no_dataset_push(monkeypatch: pytest.MonkeyPatch):
    """Neutralize the Hub round-trips start() makes for the dataset: id
    resolution and the push-if-absent are covered by their own tests."""
    from makermodslab.runners import lan_node

    monkeypatch.setattr(lan_node, "resolve_hub_repo_id", lambda repo_id: repo_id)
    monkeypatch.setattr(lan_node.LanNodeJobRunner, "_ensure_dataset_on_hub", lambda self, a, b: None)


def _started_runner(peer, clock, tmp_path, registry=None):
    runner = _runner(peer, clock, tmp_path, registry=registry)
    runner.start("local-job-1", _request(), "/host/out")
    return runner


# ---------------------------------------------------------------------------
# Config localization at the node-submission boundary.
# ---------------------------------------------------------------------------


def test_localize_clears_dataset_root_and_resets_auto_device() -> None:
    """The peer resolves the dataset from the Hub and detects its own
    hardware; a host-local dataset root or the host's auto-detected device
    (mps on a Mac) must never reach it."""
    from makermodslab.runners.lan_node import localize_config_for_lan_node

    for host_device in ("auto", "mps", "cuda", None):
        config = _request(dataset_root="/Users/me/.cache/huggingface/lerobot/user/ds")
        config.policy_device = host_device
        localize_config_for_lan_node(config)
        assert config.dataset_root is None
        assert config.policy_device == "auto"


def test_localize_keeps_an_explicit_cpu_choice() -> None:
    """Force-CPU is the one device choice that is an instruction rather than a
    hardware detection, so it travels."""
    from makermodslab.runners.lan_node import localize_config_for_lan_node

    config = _request(policy_device="cpu")
    localize_config_for_lan_node(config)
    assert config.policy_device == "cpu"


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from makermodslab.runners.lan_node import localize_config_for_lan_node

    config = _request(
        resume=True, config_path="/host/run/checkpoints/5000/pretrained_model/train_config.json"
    )
    with pytest.raises(ValueError, match="[Rr]esum"):
        localize_config_for_lan_node(config)


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from makermodslab.runners.lan_node import localize_config_for_lan_node

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_lan_node(local)

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_lan_node(hub)  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


# ---------------------------------------------------------------------------
# start(): resolve, submit, remember.
# ---------------------------------------------------------------------------


def test_start_resolves_the_peer_and_submits_a_localized_config(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _runner(peer, clock, tmp_path)
    config = _request(dataset_root="/Users/me/.cache/huggingface/lerobot/user/ds")

    runner.start("local-job-1", config, "/host/out")

    assert runner.remote_job_id() == "remote-job-1"
    assert runner.node_url() == PEER_URL
    assert runner.is_running() is True
    [submission] = peer.submissions
    assert submission["target"] == {"runner": "local"}
    assert submission["config"]["dataset_repo_id"] == "user/ds"
    assert submission["config"]["dataset_root"] is None  # localized before the wire


def test_start_raises_node_not_found_for_an_unknown_peer(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    from makermodslab.nodes import NodeNotFoundError, NodeRegistry

    empty = NodeRegistry(clock=clock, transport=peer.transport())
    runner = _runner(peer, clock, tmp_path, registry=empty)
    with pytest.raises(NodeNotFoundError):
        runner.start("local-job-1", _request(), "/host/out")


def test_start_fails_cleanly_when_the_peer_is_down(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """A dead peer at submission is a clean, coded startup failure — the shape
    JobRegistry.start turns into a failed record — not a half-submitted job."""
    from makermodslab.nodes import NodeUnreachableError

    runner = _runner(peer, clock, tmp_path)
    peer.down = True
    with pytest.raises(NodeUnreachableError):
        runner.start("local-job-1", _request(), "/host/out")
    assert runner.remote_job_id() is None


def test_start_surfaces_the_peers_refusal_detail(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """The peer's own refusal (e.g. its local-run mutex) must reach the user's
    record verbatim, not as a bare status code."""
    peer.refuse_submission = (409, "Job already running: remote-job-0")
    runner = _runner(peer, clock, tmp_path)
    with pytest.raises(RuntimeError, match="Job already running: remote-job-0"):
        runner.start("local-job-1", _request(), "/host/out")


# ---------------------------------------------------------------------------
# Liveness polling: bounded frequency, blip tolerance, lost-peer failure.
# ---------------------------------------------------------------------------


def test_record_polls_are_bounded_by_the_poll_interval(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """The watchdog ticks at 1Hz; the runner must not turn that into 1Hz of
    HTTP against the peer."""
    from makermodslab.runners.lan_node import REMOTE_POLL_INTERVAL_S

    runner = _started_runner(peer, clock, tmp_path)
    clock.advance(REMOTE_POLL_INTERVAL_S + 0.1)
    for _ in range(5):
        assert runner.is_running() is True
    assert peer.record_probes == 1
    clock.advance(REMOTE_POLL_INTERVAL_S + 0.1)
    runner.is_running()
    assert peer.record_probes == 2


def test_a_transient_blip_does_not_fail_the_job(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """Network blips mid-run must NOT immediately fail the job: failed probes
    inside the grace window leave the job running, and a recovered peer resets
    the window."""
    from makermodslab.runners.lan_node import PEER_LOST_GRACE_S, REMOTE_POLL_INTERVAL_S

    runner = _started_runner(peer, clock, tmp_path)
    peer.down = True
    clock.advance(REMOTE_POLL_INTERVAL_S + 1)
    assert runner.is_running() is True  # first failure starts the window
    clock.advance(PEER_LOST_GRACE_S / 2)
    assert runner.is_running() is True  # still inside the window
    peer.down = False
    clock.advance(REMOTE_POLL_INTERVAL_S + 1)
    assert runner.is_running() is True  # recovered; window reset
    assert runner.terminal_stage() is None
    peer.down = True
    clock.advance(PEER_LOST_GRACE_S - 1)
    assert runner.is_running() is True  # a fresh window, not the old one
    assert runner.terminal_stage() is None


def test_a_peer_that_stays_gone_becomes_a_terminal_failure(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    from makermodslab.jobs import classify_terminal_state
    from makermodslab.runners.lan_node import PEER_LOST_GRACE_S, REMOTE_POLL_INTERVAL_S

    runner = _started_runner(peer, clock, tmp_path)
    peer.down = True
    clock.advance(REMOTE_POLL_INTERVAL_S + 1)
    assert runner.is_running() is True
    clock.advance(PEER_LOST_GRACE_S + 1)
    assert runner.is_running() is False
    assert (
        classify_terminal_state(
            returncode=runner.returncode(),
            stop_requested=False,
            terminal_stage=runner.terminal_stage(),
        )
        == "failed"
    )
    message = runner.terminal_message()
    assert message is not None and PEER_URL in message


def test_a_job_the_peer_no_longer_knows_is_a_terminal_failure(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """A 404 from the peer is a definitive answer, not a blip: the remote
    record was deleted, so nothing will ever finish."""
    runner = _started_runner(peer, clock, tmp_path)
    del peer.records["remote-job-1"]
    clock.advance(6)
    assert runner.is_running() is False
    assert runner.terminal_stage() == "DELETED"


# ---------------------------------------------------------------------------
# Terminal mapping: the peer's record is the platform stage.
# ---------------------------------------------------------------------------


def test_clean_finish_maps_to_completed(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    from makermodslab.jobs import classify_terminal_state

    runner = _started_runner(peer, clock, tmp_path)
    peer.finish("remote-job-1", state="done", exit_code=0)
    clock.advance(6)
    assert runner.is_running() is False
    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0
    assert classify_terminal_state(returncode=0, stop_requested=False, terminal_stage="COMPLETED") == "done"


def test_remote_failure_maps_to_error_with_the_peers_message(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _started_runner(peer, clock, tmp_path)
    peer.finish("remote-job-1", state="failed", exit_code=1, error_message="Subprocess exited with code 1")
    clock.advance(6)
    assert runner.is_running() is False
    assert runner.terminal_stage() == "ERROR"
    assert runner.returncode() == 1
    assert runner.terminal_message() == "Subprocess exited with code 1"


def test_a_peer_side_queued_run_is_in_flight_not_terminal(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """A busy peer parks the offloaded submit in ITS local training queue
    (PR #83) and its record says "queued". That is a run on its way, not an
    outcome: filing it as terminal classified a healthy queued offload as
    ERROR on the first poll. The runner keeps following it through the
    promotion to the real finish."""
    runner = _started_runner(peer, clock, tmp_path)
    peer.records["remote-job-1"]["state"] = "queued"
    clock.advance(6)
    assert runner.is_running() is True
    assert runner.terminal_stage() is None

    # The peer's watchdog promotes it, it trains, it finishes.
    peer.records["remote-job-1"]["state"] = "running"
    clock.advance(6)
    assert runner.is_running() is True
    peer.finish("remote-job-1", state="done", exit_code=0)
    clock.advance(6)
    assert runner.is_running() is False
    assert runner.terminal_stage() == "COMPLETED"


def test_stopping_a_peer_side_queued_run_relays_the_cancel_as_interrupted(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """Stop of a run the peer still holds queued is the peer's CANCEL: its
    registry removes the record outright and answers with the removed record,
    still saying "queued". The runner must settle INTERRUPTED right there —
    otherwise the next poll 404s and DELETED files a deliberate cancel as
    `failed`."""
    from makermodslab.jobs import classify_terminal_state

    runner = _started_runner(peer, clock, tmp_path)
    peer.records["remote-job-1"]["state"] = "queued"

    runner.stop()

    assert runner.stop_signalled() is True
    assert runner.is_running() is False
    assert runner.terminal_stage() == "INTERRUPTED"
    message = runner.terminal_message()
    assert message is not None and "never started" in message
    assert (
        classify_terminal_state(
            returncode=runner.returncode(),
            stop_requested=True,
            terminal_stage=runner.terminal_stage(),
        )
        == "interrupted"
    )


def test_remote_metrics_are_mirrored_onto_the_local_record(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _started_runner(peer, clock, tmp_path)
    peer.set_metrics("remote-job-1", current_step=1200, total_steps=10000, current_loss=0.42)
    clock.advance(6)
    assert runner.is_running() is True
    assert runner._metrics.current_step == 1200
    assert runner._metrics.total_steps == 10000
    assert runner._metrics.current_loss == 0.42


# ---------------------------------------------------------------------------
# stop(): honest stop_signalled, stopped-by-request classification.
# ---------------------------------------------------------------------------


def test_stop_posts_to_the_peer_and_classifies_as_interrupted(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, classify_terminal_state

    runner = _started_runner(peer, clock, tmp_path)
    runner.stop()

    assert peer.records["remote-job-1"]["state"] == "interrupted"
    assert runner.stop_signalled() is True
    assert runner.terminal_stage() == "INTERRUPTED"
    assert runner.terminal_message() == STOPPED_BY_REQUEST_MESSAGE
    assert runner.is_running() is False
    assert (
        classify_terminal_state(
            returncode=runner.returncode(), stop_requested=True, terminal_stage=runner.terminal_stage()
        )
        == "interrupted"
    )


def test_stop_does_not_claim_a_run_that_had_already_finished(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """The peer refusing the stop (409, not running) means the run ended on
    its own first — stop_signalled must say so, and the next poll adopts the
    peer's real outcome."""
    runner = _started_runner(peer, clock, tmp_path)
    peer.finish("remote-job-1", state="done", exit_code=0)
    runner.stop()

    assert runner.stop_signalled() is False
    clock.advance(6)
    assert runner.is_running() is False
    assert runner.terminal_stage() == "COMPLETED"


def test_stop_before_submission_is_a_noop(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _runner(peer, clock, tmp_path)
    runner.stop()
    assert runner.terminal_stage() is None
    assert runner.stop_signalled() is None


# ---------------------------------------------------------------------------
# Logs: the peer's /logs drain IS the incremental mechanism; the full log
# file is adopted at the end.
# ---------------------------------------------------------------------------


def test_logs_stream_incrementally_and_persist_locally(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _started_runner(peer, clock, tmp_path)
    peer.add_log("remote-job-1", "step:100 loss:0.5")
    peer.add_log("remote-job-1", "step:200 loss:0.4")

    first = runner.stream_log_lines()
    assert [line.message for line in first] == ["step:100 loss:0.5", "step:200 loss:0.4"]
    assert runner.stream_log_lines() == []  # drained: nothing new

    peer.add_log("remote-job-1", "step:300 loss:0.3")
    second = runner.stream_log_lines()
    assert [line.message for line in second] == ["step:300 loss:0.3"]

    on_disk = [json.loads(raw)["message"] for raw in (tmp_path / "log.jsonl").read_text().splitlines()]
    # Every drained line reached the local log file (after the runner's own
    # submission note).
    assert on_disk[-3:] == ["step:100 loss:0.5", "step:200 loss:0.4", "step:300 loss:0.3"]


def test_terminal_detection_adopts_the_peers_full_log_file(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    """Nobody polled /logs while the run trained (monitor closed), so the local
    file is empty — the peer's persisted log is adopted at the end so history
    survives on this side too."""
    runner = _started_runner(peer, clock, tmp_path)
    peer.add_log("remote-job-1", "line one")
    peer.add_log("remote-job-1", "line two")
    peer.finish("remote-job-1", state="done", exit_code=0)
    clock.advance(6)
    assert runner.is_running() is False

    on_disk = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert [json.loads(raw)["message"] for raw in on_disk] == ["line one", "line two"]


def test_stream_log_lines_returns_empty_on_a_blip(
    peer, clock, tmp_path, local_identity, nodes_file, no_dataset_push
) -> None:
    runner = _started_runner(peer, clock, tmp_path)
    peer.down = True
    assert runner.stream_log_lines() == []


# ---------------------------------------------------------------------------
# classify_terminal_state and the watchdog message path.
# ---------------------------------------------------------------------------


def test_classify_adopts_a_relayed_interrupted_stage_without_local_intent() -> None:
    """INTERRUPTED relays another registry's own verdict (which already told a
    deliberate stop from a crash, next to the run); it must not be re-derived
    here into `failed` just because OUR registry recorded no stop intent —
    the remote stop-by-request case."""
    from makermodslab.jobs import classify_terminal_state

    assert classify_terminal_state(returncode=-15, stop_requested=False, terminal_stage="INTERRUPTED") == (
        "interrupted"
    )
    assert classify_terminal_state(returncode=None, stop_requested=True, terminal_stage="INTERRUPTED") == (
        "interrupted"
    )


class _FakeRelayRunner:
    """Runner stand-in with the remote-verdict hooks the watchdog reads."""

    def __init__(self, stage: str, message: str | None, rc: int | None = None) -> None:
        self._stage = stage
        self._message = message
        self._rc = rc

    def is_running(self) -> bool:
        return False

    def returncode(self):
        return self._rc

    def terminal_stage(self):
        return self._stage

    def terminal_message(self):
        return self._message

    def wandb_run_url(self):
        return None


def test_tick_uses_the_runners_message_for_a_relayed_interruption(tmp_path) -> None:
    """A stop pressed on the PEER (never requested here) must finalize as
    `interrupted` with the peer's own wording — not as a failure, and not with
    the local restart/unconfirmed text."""
    from makermodslab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    from .test_jobs import _inject_running_job

    reg = JobRegistry(tmp_path / "root")
    record = _inject_running_job(
        reg, tmp_path, rc=None, runner=_FakeRelayRunner("INTERRUPTED", STOPPED_BY_REQUEST_MESSAGE, rc=-15)
    )

    reg._tick()

    finalized = reg._records[record.id]
    assert finalized.state == "interrupted"
    assert finalized.error_message == STOPPED_BY_REQUEST_MESSAGE


# ---------------------------------------------------------------------------
# Registry / record integration.
# ---------------------------------------------------------------------------


def _swap_node_registry(monkeypatch, registry) -> None:
    from makermodslab import nodes

    monkeypatch.setattr(nodes, "node_registry", registry)


def _stub_hub_status(monkeypatch) -> None:
    """Keep the registry's remote-dataset preflight off the real Hub — BOTH
    probes: the status lookup and the direct emptiness check the preflight
    makes for an on_hub answer (without the second stub, these tests issue a
    real get_paths_info request to huggingface.co)."""
    from makermodslab import datasets

    monkeypatch.setattr(datasets, "get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr(datasets, "hub_copy_has_data", lambda repo_id, **kwargs: True)


def _quiesced_registry(root):
    """A JobRegistry whose watchdog thread is verifiably stopped, so manual
    _tick() calls (and job.json writes) can't race a background tick — the
    same discipline as test_jobs._inject_running_job."""
    from makermodslab.jobs import JobRegistry

    reg = JobRegistry(root)
    reg.shutdown()
    if reg._watchdog_thread is not None:
        reg._watchdog_thread.join(timeout=2)
    return reg


def _lan_runner_factory(monkeypatch, peer, clock) -> None:
    """Make JobRegistry.start's LanNodeJobRunner construction network-free by
    injecting the fake transport + clock into the class's __init__ defaults."""
    from makermodslab.runners import lan_node

    real = lan_node.LanNodeJobRunner

    def factory(metrics, log_file_path, node_instance_id):
        return real(
            metrics,
            log_file_path,
            node_instance_id,
            registry=_node_registry(peer, clock),
            transport=peer.transport(),
            clock=clock,
        )

    monkeypatch.setattr(lan_node, "LanNodeJobRunner", factory)


def test_job_target_requires_a_node_for_lan_runner(tmp_path) -> None:
    from makermodslab.jobs import JobTarget

    reg = _quiesced_registry(tmp_path / "root")
    with pytest.raises(ValueError, match="node_instance_id"):
        reg.start(_request(), JobTarget(runner="lan_node"))


def test_registry_start_refuses_an_unknown_node(
    tmp_path, monkeypatch, peer, clock, local_identity, nodes_file
) -> None:
    from makermodslab.jobs import JobTarget
    from makermodslab.nodes import NodeNotFoundError, NodeRegistry

    _swap_node_registry(monkeypatch, NodeRegistry(clock=clock, transport=peer.transport()))
    reg = _quiesced_registry(tmp_path / "root")
    with pytest.raises(NodeNotFoundError):
        reg.start(_request(), JobTarget(runner="lan_node", node_instance_id=PEER_ID))
    assert reg.list(limit=100) == []  # refused before any record existed


def test_registry_refuses_resume_and_finetune_on_a_lan_node(tmp_path) -> None:
    from makermodslab.jobs import JobTarget

    reg = _quiesced_registry(tmp_path / "root")
    target = JobTarget(runner="lan_node", node_instance_id=PEER_ID)
    with pytest.raises(ValueError, match="[Cc]ontinu"):
        reg.start(_request(resume=True, resume_from_job_id="parent"), target)
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        reg.start(_request(finetune_from_job_id="parent"), target)


def test_registry_start_records_the_target_node_and_remote_id(
    tmp_path, monkeypatch, peer, clock, local_identity, nodes_file, no_dataset_push
) -> None:
    from makermodslab.jobs import JobTarget
    from makermodslab.nodes import NodeRegistry

    registry = NodeRegistry(clock=clock, transport=peer.transport())
    registry.add(peer.url)
    _swap_node_registry(monkeypatch, registry)
    _stub_hub_status(monkeypatch)
    _lan_runner_factory(monkeypatch, peer, clock)

    reg = _quiesced_registry(tmp_path / "root")
    record = reg.start(_request(), JobTarget(runner="lan_node", node_instance_id=PEER_ID))

    assert record.runner == "lan_node"
    assert record.node_instance_id == PEER_ID
    assert record.node_url == PEER_URL
    assert record.remote_job_id == "remote-job-1"
    assert record.state == "running"

    persisted = json.loads((tmp_path / "root" / record.id / "job.json").read_text())
    assert persisted["node_instance_id"] == PEER_ID
    assert persisted["node_url"] == PEER_URL
    assert persisted["remote_job_id"] == "remote-job-1"

    # Checkpoints live on the peer's disk in this phase — never a crash, just none.
    assert reg.list_checkpoints(record.id) == []

    # Finalisation via the watchdog's normal path.
    peer.finish("remote-job-1", state="done", exit_code=0)
    clock.advance(6)
    reg._tick()
    assert reg.get(record.id).state == "done"


def test_an_offloaded_submit_never_enters_the_local_queue(
    tmp_path, monkeypatch, peer, clock, local_identity, nodes_file, no_dataset_push
) -> None:
    """The local queue exists because this machine has one training slot; a
    lan_node run spends the PEER's slot, so a busy local slot — and even a
    non-empty local queue — must not park it. It starts on the peer
    immediately, and the waiting local runs keep their places."""
    from makermodslab.jobs import JobRecord, JobTarget
    from makermodslab.nodes import NodeRegistry

    registry = NodeRegistry(clock=clock, transport=peer.transport())
    registry.add(peer.url)
    _swap_node_registry(monkeypatch, registry)
    _stub_hub_status(monkeypatch)
    _lan_runner_factory(monkeypatch, peer, clock)

    reg = _quiesced_registry(tmp_path / "root")
    with reg._lock:
        reg._records["busy"] = JobRecord(
            id="busy",
            name="busy",
            state="running",
            config=_request(),
            output_dir=str(tmp_path / "root" / "busy" / "run"),
            started_at=0.0,
            runner="local",
            process_pid=4242,
        )
        reg._records["waiting"] = JobRecord(
            id="waiting",
            name="waiting",
            state="queued",
            config=_request(),
            output_dir=str(tmp_path / "root" / "waiting" / "run"),
            started_at=0.0,
            runner="local",
            queue_seq=1,
        )
        reg._next_queue_seq = 2

    record = reg.start(_request(), JobTarget(runner="lan_node", node_instance_id=PEER_ID))

    assert record.state == "running"  # on the peer now, not in our line
    assert record.queue_position == 0
    assert [r.id for r in reg.list_queue()] == ["waiting"]
    assert peer.submissions  # it really was handed to the peer


def test_old_job_json_without_node_fields_still_loads(tmp_path) -> None:
    """Records persisted before the lan_node fields existed must load with the
    new fields defaulting to null — no migration, no skipped record."""
    job_dir = tmp_path / "root" / "old-job"
    job_dir.mkdir(parents=True)
    old = {
        "id": "old-job",
        "name": "OLD · user/ds",
        "state": "done",
        "config": {"dataset_repo_id": "user/ds"},
        "output_dir": str(job_dir / "run"),
        "started_at": 100.0,
        "ended_at": 200.0,
        "exit_code": 0,
        "runner": "local",
    }
    (job_dir / "job.json").write_text(json.dumps(old))

    reg = _quiesced_registry(tmp_path / "root")
    record = reg.get("old-job")
    assert record.state == "done"
    assert record.node_instance_id is None
    assert record.node_url is None
    assert record.remote_job_id is None


def test_a_running_lan_node_record_reattaches_at_boot(
    tmp_path, monkeypatch, peer, clock, local_identity, nodes_file, no_dataset_push
) -> None:
    """Like a running hf_cloud record, a running lan_node record reattaches and
    lets the poll drive finalisation — the peer kept training while we were
    down, and a restart must not orphan the record."""
    from makermodslab.jobs import JobTarget
    from makermodslab.nodes import NodeRegistry

    registry = NodeRegistry(clock=clock, transport=peer.transport())
    registry.add(peer.url)
    _swap_node_registry(monkeypatch, registry)
    _stub_hub_status(monkeypatch)
    _lan_runner_factory(monkeypatch, peer, clock)

    first = _quiesced_registry(tmp_path / "root")
    record = first.start(_request(), JobTarget(runner="lan_node", node_instance_id=PEER_ID))

    second = _quiesced_registry(tmp_path / "root")
    assert second.get(record.id).state == "running"
    assert record.id in second._runners

    peer.finish("remote-job-1", state="done", exit_code=0)
    clock.advance(6)
    second._tick()
    assert second.get(record.id).state == "done"


def test_runner_labels_cover_lan_node() -> None:
    from makermodslab.jobs import _RUNNER_LABELS

    assert "lan_node" in _RUNNER_LABELS
    assert _RUNNER_LABELS["lan_node"]  # non-empty, human-facing


# ---------------------------------------------------------------------------
# Submission path over HTTP: proper codes for a missing/unknown node.
# ---------------------------------------------------------------------------


def test_post_training_with_lan_runner_but_no_node_is_422_with_code(client) -> None:
    response = client.post(
        "/api/v1/jobs/training",
        json={"config": {"dataset_repo_id": "user/ds"}, "target": {"runner": "lan_node"}},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "request.validation"
    assert "node_instance_id" in body["detail"]


def test_post_training_naming_an_unknown_node_is_400_with_code(
    client, monkeypatch, peer, clock, local_identity, nodes_file
) -> None:
    from makermodslab.nodes import NodeRegistry

    _swap_node_registry(monkeypatch, NodeRegistry(clock=clock, transport=peer.transport()))
    response = client.post(
        "/api/v1/jobs/training",
        json={
            "config": {"dataset_repo_id": "user/ds"},
            "target": {"runner": "lan_node", "node_instance_id": "ff" * 16},
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "node.not_found"
