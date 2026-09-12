"""The ``jobs`` namespace: training-job lifecycle, logs/metrics/checkpoints,
Hub jobs and models, runner hardware.

Response models mirror makermodslab/schemas/jobs.py (whose registry shapes are
the wire models in makermodslab/jobs.py). Enum-like server fields (``state``,
``runner``, checkpoint ``source``) are typed ``str`` on purpose: an older SDK
must keep decoding values a newer server adds.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import pydantic
from pydantic import ConfigDict

from makermodslab_sdk._operations import operation
from makermodslab_sdk.errors import InvalidRequestError, MakerModsError
from makermodslab_sdk.resources._base import Resource, SdkModel

# The server's job lifecycle (makermodslab/jobs.py):
#   JobState = Literal["queued", "running", "done", "failed", "interrupted"]
# "queued" (waiting for the machine's one local-training slot) and "running"
# are the live states; wait() polls through both to the terminal three.
TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "interrupted"})


class JobWaitTimeout(MakerModsError, TimeoutError):  # noqa: N818 — "Timeout" IS the suffix, matching builtins.TimeoutError
    """wait() gave up before the job reached a terminal state.

    Not an error from the server — the job is still running; the message says
    how to keep waiting. Catchable as either ``MakerModsError`` or the
    built-in ``TimeoutError``.
    """

    def __init__(self, message: str, *, job_id: str, waited: float, last_state: str) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.waited = waited
        self.last_state = last_state


class TrainingMetrics(SdkModel):
    """Live progress numbers on a job record (zeros/None until training logs)."""

    current_step: int = 0
    total_steps: int = 0
    current_loss: float | None = None
    current_lr: float | None = None
    grad_norm: float | None = None
    eta_seconds: float | None = None


class TrainingConfig(SdkModel):
    """The job's full TrainingRequest as persisted. Only the core fields are
    declared; every other server-side knob rides along via ``extra="allow"``."""

    dataset_repo_id: str
    policy_type: str = "act"
    steps: int = 10000
    batch_size: int = 8


class TrainingOptions(SdkModel):
    """EVERY user-settable training knob the server accepts — the SDK exposes
    the backend's full power, deliberately wider than the web UI's form.

    All fields default to unset (None) and only set fields are sent, so the
    server's own defaults always rule. ``create_training`` takes these as
    keyword arguments and validates them CLIENT-SIDE (``extra="forbid"``): a
    typo'd knob fails immediately with the close matches named, before any
    request is sent. tests/test_jobs.py parity-asserts this field set against
    the server's TrainingRequest, so a new server knob breaks the build here
    until it is typed (or excluded with a reason) — the surface cannot
    silently fall behind the backend.

    Field semantics are the server's (makermodslab/train.py TrainingRequest);
    the groups below mirror its layout. Registry-managed internals
    (``resume_from_hub_repo``, ``policy_repo_id``, …) are deliberately not
    here — the server sets those itself.
    """

    model_config = ConfigDict(extra="forbid")

    # Dataset
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None
    # Policy
    policy_type: str | None = None
    # Core
    steps: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    num_workers: int | None = None
    # Logging / checkpointing
    log_freq: int | None = None
    save_freq: int | None = None
    env_eval_freq: int | None = None
    save_checkpoint: bool | None = None
    # Output / naming
    output_dir: str | None = None
    job_name: str | None = None
    # Resume ("Continue training") — the API is deliberately wider than the
    # UI here: an arbitrary chain rewind (resume_from_checkpoint_job_id) is
    # reachable only through this surface; the server validates lineage.
    resume: bool | None = None
    resume_from_job_id: str | None = None
    resume_from_step: int | None = None
    resume_from_checkpoint_job_id: str | None = None
    upload_resume_checkpoint: bool | None = None
    # Fine-tune (fresh run, weights initialized from a checkpoint)
    finetune_from_job_id: str | None = None
    finetune_from_step: int | None = None
    policy_pretrained_path: str | None = None
    upload_finetune_checkpoint: bool | None = None
    # Weights & Biases
    wandb_enable: bool | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = None
    wandb_disable_artifact: bool | None = None
    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int | None = None
    eval_batch_size: int | None = None
    eval_use_async_envs: bool | None = None
    # Policy runtime
    policy_device: str | None = None
    policy_use_amp: bool | None = None
    # Optimizer
    optimizer_type: str | None = None
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None
    # Advanced
    use_policy_training_preset: bool | None = None
    config_path: str | None = None
    # Cloud (hf_cloud runner) only — HF-Jobs duration string ("2h", "3h30m")
    hf_job_timeout: str | None = None


def _validate_training_knobs(knobs: dict[str, Any]) -> dict[str, Any]:
    """Client-side knob validation: unknown or mistyped knobs fail HERE, with
    the fix named, instead of as a server 422 (or worse, a silently ignored
    key on an older server). Returns only the set fields."""
    try:
        options = TrainingOptions.model_validate(knobs)
    except pydantic.ValidationError as exc:
        problems = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            if error["type"] == "extra_forbidden":
                close = difflib.get_close_matches(field, TrainingOptions.model_fields, n=2)
                hint = f" — did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
                problems.append(f"unknown knob {field!r}{hint}")
            else:
                problems.append(f"{field}: {error['msg']}")
        raise InvalidRequestError(
            "create_training rejected client-side (no request was sent): "
            + "; ".join(problems)
            + "\nNext step: help(makermodslab_sdk.TrainingOptions) lists every knob; for a field "
            "newer than this SDK, pass it via config={...} which skips this validation.",
            status=0,
            detail="; ".join(problems),
            suggestion="help(makermodslab_sdk.TrainingOptions) lists every valid training knob.",
        ) from None
    return options.model_dump(exclude_none=True)


class Job(SdkModel):
    """One training-run record (server JobRecord — GET /api/v1/jobs/{job_id}).

    ONE shape for all runners ("local" / "hf_cloud" / "imported" / "lan_node"):
    uniform-with-nulls, so runner-specific fields (``process_pid`` for local,
    ``hf_*`` for hf_cloud, ``node_*`` for lan_node) are simply None outside
    their runner. ``state`` is "running" until the run ends as one of
    "done" / "failed" / "interrupted". ``job_number`` is the short human-facing
    run number ("#46"); 0 means unassigned — don't print "#0".
    """

    id: str
    job_number: int = 0
    name: str
    display_name: str | None = None
    state: str
    config: TrainingConfig
    output_dir: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    metrics: TrainingMetrics = TrainingMetrics()
    runner: str = "local"
    process_pid: int | None = None
    node_instance_id: str | None = None
    node_url: str | None = None
    remote_job_id: str | None = None
    hf_job_id: str | None = None
    hf_flavor: str | None = None
    hf_repo_id: str | None = None
    hf_job_url: str | None = None
    checkpoint_count: int = 0
    checkpoints_hub_repo_id: str | None = None
    wandb_run_url: str | None = None
    checkpoints_hub_steps: list[str] = []
    child_ids: list[str] = []
    ancestor_ids: list[str] = []
    # Local training queue (state == "queued"): promotion order and FIFO seq;
    # both 0 outside the queue.
    queue_position: int = 0
    queue_seq: int = 0
    queued_hub_ref: str | None = None
    queued_resume_ref: str | None = None


class JobList(SdkModel):
    """GET /api/v1/jobs — records newest first."""

    jobs: list[Job]


class LogLine(SdkModel):
    """One captured line of training output."""

    timestamp: float
    message: str


class JobLogs(SdkModel):
    """The shared body of /logs (live tail) and /log-file (whole history)."""

    logs: list[LogLine]


class MetricsPoint(SdkModel):
    """One (step, loss/lr/grad_norm) sample from the job's persisted log."""

    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None


class MetricsHistory(SdkModel):
    """GET /api/v1/jobs/{job_id}/metrics-history — chart-seeding series."""

    points: list[MetricsPoint]


class Checkpoint(SdkModel):
    """One saved checkpoint. ``ref`` is opaque — hand it back to the server
    (inference start), don't parse it. ``source`` is "local" or "hub"."""

    step: int
    source: str
    ref: str


class JobCheckpoints(SdkModel):
    """GET /api/v1/jobs/{job_id}/checkpoints — ascending by step."""

    checkpoints: list[Checkpoint]


class CheckpointImageFeature(SdkModel):
    """One camera's expected input size in a checkpoint's policy config."""

    height: int
    width: int


class CheckpointPolicyConfig(SdkModel):
    """The UX-relevant slice of a checkpoint's pretrained config —
    state_dim/action_dim of 6 means single-arm, 12 bimanual."""

    policy_type: str | None
    image_features: dict[str, CheckpointImageFeature]
    requires_task: bool
    state_dim: int | None
    action_dim: int | None


class HubJobStatus(SdkModel):
    """The {stage, message} pair of one Hub job (stage e.g. "RUNNING")."""

    stage: str
    message: str | None = None


class HubJob(SdkModel):
    """One HF Jobs run visible to the server's Hub account."""

    id: str
    name: str | None = None
    created_at: str | None = None
    docker_image: str | None = None
    space_id: str | None = None
    flavor: str | None = None
    status: HubJobStatus | None = None
    owner: str | None = None
    url: str


class HubModel(SdkModel):
    """One model repo of the server's Hub account shown beside its jobs."""

    repo_id: str
    last_modified: str | None = None
    private: bool


class HubJobs(SdkModel):
    """GET /api/v1/jobs/hub. Heterogeneous by branch: when the server is not
    authenticated the body has no ``jobs_permission`` key at all (reads None
    here); when authenticated it is a real True/False."""

    authenticated: bool
    jobs_permission: bool | None = None
    jobs: list[HubJob]
    models: list[HubModel]


class HubJobDismissed(SdkModel):
    """POST /api/v1/jobs/hub/jobs/{job_id}/dismiss — job_id is the stripped id
    that was persisted, not necessarily the raw input."""

    status: str
    job_id: str


class HubModelDeleted(SdkModel):
    """DELETE /api/v1/jobs/hub/models/{repo_id} — idempotent success."""

    status: str
    repo_id: str


class RunnerFlavor(SdkModel):
    """One HF Jobs hardware flavor. ``accelerator`` ("Nvidia A10G") and
    ``vram`` ("24 GB") are None on cpu-* flavors."""

    name: str
    pretty_name: str
    cpu: str
    ram: str
    accelerator: str | None = None
    vram: str | None = None
    unit_cost_usd: float
    unit_label: str


class RunnersHardware(SdkModel):
    """GET /api/v1/jobs/runners/hardware — flavor catalog + Hub auth state."""

    authenticated: bool
    username: str | None
    flavors: list[RunnerFlavor]
    offline: bool


def _job_path(job_id: str, suffix: str = "") -> str:
    return f"/api/v1/jobs/{quote(job_id, safe='')}{suffix}"


class JobsResource(Resource):
    """``client.jobs`` — training jobs: create, watch, and manage runs.

    Example:
        >>> job = client.jobs.create_training("user/so101-pick", steps=20000)
        >>> done = client.jobs.wait(job.id, timeout=4 * 3600)
        >>> done.state, client.jobs.checkpoints(job.id).checkpoints[-1].step
        ('done', 20000)
    """

    @operation("list_jobs")
    def list(self, limit: int = 10) -> JobList:
        """The run history, newest first.

        Records carry live ``metrics`` while running and the resume lineage
        (``child_ids`` / ``ancestor_ids``) computed at read time.

        Example:
            >>> for job in client.jobs.list(limit=5).jobs:
            ...     print(f"#{job.job_number}", job.display_name or job.name, job.state)
        """
        return JobList.model_validate(
            self._transport.request("GET", "/api/v1/jobs", params={"limit": limit}, action="List jobs")
        )

    @operation("list_job_queue")
    def queue(self) -> JobList:
        """The local training queue, in promotion order (a submit made while
        the machine's one training slot is busy QUEUES instead of refusing —
        these are those runs, state "queued").

        Example:
            >>> [(j.queue_position, j.name) for j in client.jobs.queue().jobs]
            [(1, 'act_run_a'), (2, 'act_run_b')]
        """
        return JobList.model_validate(
            self._transport.request("GET", "/api/v1/jobs/queue", action="List job queue")
        )

    @operation("reorder_job_queue")
    def reorder_queue(self, job_ids: Sequence[str]) -> JobList:
        """Reorder the queue to exactly ``job_ids`` (every queued id, in the
        new order). A 409 job.queue_stale means the queue changed under you —
        refetch ``queue()`` and retry with the current ids.

        Example:
            >>> ids = [j.id for j in client.jobs.queue().jobs]
            >>> client.jobs.reorder_queue(ids[::-1])  # reverse the order
        """
        return JobList.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/jobs/queue/reorder",
                json={"job_ids": list(job_ids)},
                action="Reorder job queue",
            )
        )

    @operation("get_job")
    def get(self, job_id: str) -> Job:
        """One job by id (the long id, not the "#46" run number).

        Example:
            >>> job = client.jobs.get(job.id)
            >>> job.state, job.metrics.current_step, job.metrics.total_steps
            ('running', 1200, 20000)
        """
        return Job.model_validate(
            self._transport.request("GET", _job_path(job_id), action=f"Get job {job_id!r}")
        )

    @operation("stop_job")
    def stop(self, job_id: str) -> Job:
        """Stop a running job; returns the final record.

        A deliberate stop is filed as state "interrupted", never "failed" —
        checkpoints already saved stay usable. 409 when the job is not running.

        Example:
            >>> client.jobs.stop(job.id).state
            'interrupted'
        """
        return Job.model_validate(
            self._transport.request("POST", _job_path(job_id, "/stop"), action=f"Stop job {job_id!r}")
        )

    @operation("delete_job")
    def delete(self, job_id: str) -> None:
        """Delete a finished job's record and outputs (checkpoints included).

        Refuses (409) while the job is running — stop it first — and for a run
        that a later run resumed from (delete the continuation first).

        Example:
            >>> client.jobs.delete(job.id)
        """
        self._transport.request("DELETE", _job_path(job_id), action=f"Delete job {job_id!r}")

    @operation("rename_job")
    def rename(self, job_id: str, new_name: str) -> Job:
        """Set a job's display alias (shown in place of the generated name).

        Metadata-only: ids, output dirs, and Hub repo names never change.

        Example:
            >>> client.jobs.rename(job.id, "pick v2").display_name
            'pick v2'
        """
        return Job.model_validate(
            self._transport.request(
                "POST",
                _job_path(job_id, "/rename"),
                json={"new_name": new_name},
                action=f"Rename job {job_id!r}",
            )
        )

    @operation("get_job_logs")
    def logs(self, job_id: str) -> JobLogs:
        """Drain the job's live log tail — lines that arrived since the last
        call. For everything from the start, use ``log_file``.

        Example:
            >>> for line in client.jobs.logs(job.id).logs:
            ...     print(line.message)
        """
        return JobLogs.model_validate(
            self._transport.request("GET", _job_path(job_id, "/logs"), action=f"Get logs of job {job_id!r}")
        )

    @operation("get_job_log_file")
    def log_file(self, job_id: str) -> JobLogs:
        """The job's whole persisted log, from the first line (JSON, not a
        download). Also drains the live tail so a following ``logs`` call
        returns only newer lines.

        Example:
            >>> lines = client.jobs.log_file(job.id).logs
            >>> lines[0].message
            'Starting training...'
        """
        return JobLogs.model_validate(
            self._transport.request(
                "GET", _job_path(job_id, "/log-file"), action=f"Get log file of job {job_id!r}"
            )
        )

    @operation("get_job_metrics_history")
    def metrics_history(self, job_id: str) -> MetricsHistory:
        """The per-step loss/lr/grad-norm series reconstructed from the job's
        log — resolution is the run's ``log_freq``. Survives restarts and
        spans the resume chain.

        Example:
            >>> points = client.jobs.metrics_history(job.id).points
            >>> points[-1].step, points[-1].loss
            (20000, 0.041)
        """
        return MetricsHistory.model_validate(
            self._transport.request(
                "GET", _job_path(job_id, "/metrics-history"), action=f"Get metrics history of job {job_id!r}"
            )
        )

    @operation("get_job_checkpoints")
    def checkpoints(self, job_id: str) -> JobCheckpoints:
        """The job's saved checkpoints, ascending by step (local runs list
        disk, cloud runs list the Hub repo).

        Example:
            >>> latest = client.jobs.checkpoints(job.id).checkpoints[-1]
            >>> latest.step, latest.source
            (20000, 'local')
        """
        return JobCheckpoints.model_validate(
            self._transport.request(
                "GET", _job_path(job_id, "/checkpoints"), action=f"List checkpoints of job {job_id!r}"
            )
        )

    @operation("get_checkpoint_policy_config")
    def checkpoint_policy_config(self, job_id: str, step: int) -> CheckpointPolicyConfig:
        """What one checkpoint's policy expects as input: per-camera image
        sizes, whether it needs a task string, and state/action dims (6 =
        single arm, 12 = bimanual) — check these before starting inference.

        Example:
            >>> cfg = client.jobs.checkpoint_policy_config(job.id, 20000)
            >>> cfg.policy_type, cfg.action_dim, list(cfg.image_features)
            ('act', 6, ['observation.images.front'])
        """
        return CheckpointPolicyConfig.model_validate(
            self._transport.request(
                "GET",
                _job_path(job_id, f"/checkpoints/{step}/policy-config"),
                action=f"Get policy config of job {job_id!r} checkpoint {step}",
            )
        )

    @operation("create_training_job")
    def create_training(
        self,
        dataset_repo_id: str,
        *,
        runner: str = "local",
        flavor: str | None = None,
        node_instance_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        **knobs: Any,
    ) -> Job:
        """Start a training run; returns its record immediately (state
        "running") — follow with ``wait``, ``logs`` or ``metrics_history``.

        ``**knobs`` accepts EVERY training knob the server has —
        :class:`TrainingOptions` is the full typed catalog (steps,
        batch_size, seed, log/save freq, wandb_*, optimizer_*, resume and
        fine-tune lineage, eval, device/AMP, hf_job_timeout, …). Knobs are
        validated client-side: a typo fails instantly with the close matches
        named, before any request is sent. Unset knobs are not sent, so the
        server's defaults rule.

        ``runner`` picks where it runs: "local" (this machine), "hf_cloud"
        (HF Jobs GPU — requires ``flavor``, see ``runners_hardware``), or
        "lan_node" (a registered peer — requires ``node_instance_id``, see
        ``client.nodes``). ``config`` passes raw fields WITHOUT client-side
        validation (for a server newer than this SDK) and overrides knobs on
        key collisions.

        Example:
            >>> job = client.jobs.create_training(
            ...     "user/so101-pick",
            ...     steps=20000,
            ...     save_freq=5000,
            ...     optimizer_lr=1e-5,
            ...     wandb_enable=True,
            ... )
            >>> client.jobs.wait(job.id, timeout=4 * 3600).state
            'done'
        """
        cfg: dict[str, Any] = _validate_training_knobs(knobs)
        cfg["dataset_repo_id"] = dataset_repo_id
        if config:
            cfg.update(config)
        target: dict[str, Any] = {"runner": runner}
        if flavor is not None:
            target["flavor"] = flavor
        if node_instance_id is not None:
            target["node_instance_id"] = node_instance_id
        return Job.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/jobs/training",
                json={"config": cfg, "target": target},
                action="Create training job",
            )
        )

    @operation("import_model")
    def import_model(self, source: str, *, name: str | None = None) -> Job:
        """Register an external model (local checkpoint dir or Hub repo id) as
        a pseudo-job so it can be fine-tuned or run for inference.

        Idempotent: importing an already-registered source returns the
        existing record, marked with an extra ``already_imported: True`` key
        (readable via ``getattr(job, "already_imported", False)``).

        Example:
            >>> job = client.jobs.import_model("lerobot/act_so101", name="base act")
            >>> job.runner
            'imported'
        """
        body: dict[str, Any] = {"source": source, "name": name}
        return Job.model_validate(
            self._transport.request(
                "POST", "/api/v1/jobs/import", json=body, action=f"Import model {source!r}"
            )
        )

    @operation("list_hub_jobs")
    def list_hub(self) -> HubJobs:
        """The server's HF Jobs runs and MakerMods-created model repos as the
        Hub sees them — including runs this install no longer tracks locally.

        ``authenticated`` False means no Hub token; ``jobs_permission`` says
        whether the token may use HF Jobs (None when unauthenticated).

        Example:
            >>> hub = client.jobs.list_hub()
            >>> [(j.name, j.status.stage if j.status else None) for j in hub.jobs]
            [('act_user_so101-pick_2026-08-27_10-00-00', 'RUNNING')]
        """
        return HubJobs.model_validate(
            self._transport.request("GET", "/api/v1/jobs/hub", action="List Hub jobs")
        )

    @operation("dismiss_hub_job")
    def dismiss_hub(self, job_id: str) -> HubJobDismissed:
        """Hide a finished Hub job from ``list_hub`` (a local, persisted hide —
        the HF Jobs API has no delete). A job still RUNNING/QUEUED/SCHEDULING
        keeps showing until it ends.

        Example:
            >>> client.jobs.dismiss_hub("64f1c9a2").status
            'success'
        """
        return HubJobDismissed.model_validate(
            self._transport.request(
                "POST",
                f"/api/v1/jobs/hub/jobs/{quote(job_id, safe='')}/dismiss",
                action=f"Dismiss Hub job {job_id!r}",
            )
        )

    @operation("delete_hub_model")
    def delete_hub_model(self, repo_id: str) -> HubModelDeleted:
        """Permanently delete a model repo from the Hugging Face Hub — this
        destroys weights on the Hub, not a local record. Only repos under the
        server's own Hub username are allowed; already-gone repos succeed
        (idempotent).

        Example:
            >>> client.jobs.delete_hub_model("user/act_user_so101-pick_2026-08-27_10-00-00")
        """
        return HubModelDeleted.model_validate(
            self._transport.request(
                "DELETE",
                f"/api/v1/jobs/hub/models/{quote(repo_id, safe='/')}",
                action=f"Delete Hub model {repo_id!r}",
            )
        )

    @operation("get_runners_hardware")
    def runners_hardware(self) -> RunnersHardware:
        """The HF Jobs hardware catalog + Hub auth state — pick a
        ``flavor.name`` here for ``create_training(runner="hf_cloud")``.

        Example:
            >>> hw = client.jobs.runners_hardware()
            >>> [(f.name, f.accelerator, f.unit_cost_usd) for f in hw.flavors[:2]]
            [('cpu-basic', None, 0.0), ('a10g-small', 'Nvidia A10G', 1.05)]
        """
        return RunnersHardware.model_validate(
            self._transport.request("GET", "/api/v1/jobs/runners/hardware", action="Get runner hardware")
        )

    # Deliberately NOT @operation-tagged: a convenience over get_job, not an
    # API surface of its own. Polling-only for now — a realtime hint-driven
    # variant arrives with the integration stage.
    def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Job:
        """Block until the job ends; returns the final record.

        Terminal means ``state`` in "done" / "failed" / "interrupted" (the
        server's JobState minus "running" — makermodslab/jobs.py). This
        returns on ANY terminal state; check ``.state`` (and
        ``.error_message``) yourself — a failed run is a normal return, not an
        exception. Raises :class:`JobWaitTimeout` when ``timeout`` seconds of
        polling pass first (None = wait forever), and an unknown ``job_id``
        raises the same error ``get`` would. ``sleep_fn`` is injectable so
        tests can wait without sleeping.

        Example:
            >>> job = client.jobs.wait(job.id, timeout=3600)
            >>> job.state, job.error_message
            ('done', None)
        """
        waited = 0.0
        while True:
            job = self.get(job_id)  # a missing job surfaces get's error as-is
            if job.state in TERMINAL_STATES:
                return job
            if timeout is not None and waited + poll_interval > timeout:
                raise JobWaitTimeout(
                    f"Job {job_id!r} is still {job.state!r} after {waited:.0f}s "
                    f"(timeout={timeout}). The run may simply need longer — call "
                    f"client.jobs.wait({job_id!r}, timeout=<more seconds>) to keep waiting, "
                    f"check progress with client.jobs.get({job_id!r}).metrics, "
                    f"or end it with client.jobs.stop({job_id!r}).",
                    job_id=job_id,
                    waited=waited,
                    last_state=job.state,
                )
            sleep_fn(poll_interval)
            waited += poll_interval
