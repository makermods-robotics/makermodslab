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

"""Response models for the "jobs" route group (training-job lifecycle, Hub
jobs/models listing, checkpoints). See the package docstring for the fidelity
rules; the shape authority is always the handler, named next to each model.

The registry shapes are re-exported straight from makermodslab/jobs.py: the
handlers there already build responses FROM these very Pydantic models
(JobRecord, LogLine, MetricsHistoryPoint, JobCheckpoint), so schema and wire
format cannot drift. JobRecord itself is deliberately ONE model for all three
runners (local / hf_cloud / imported): the record is uniform-with-nulls, not a
union — every key is persisted and serialized on every record, and the
runner-specific fields (process_pid for local, hf_* for hf_cloud) are simply
null outside their runner. Only /jobs/hub is heterogeneous BY BRANCH: the
unauthenticated body carries no jobs_permission key at all (never null), while
sibling keys in the authenticated rows (name, created_at, status, owner,
last_modified) are legitimately null — so that one route serializes with
``response_model_exclude_unset=True`` and every other route takes a plain
model with its real nulls declared.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# The registry's own wire models — the handlers return these instances (or
# dicts wrapping them), so re-exporting is what keeps schema == wire.
from makermodslab.jobs import JobCheckpoint, JobRecord, LogLine, MetricsHistoryPoint

__all__ = [
    "CheckpointImageFeature",
    "CheckpointPolicyConfigResponse",
    "HubJobDismissResponse",
    "HubJobItem",
    "HubJobStatus",
    "HubJobsResponse",
    "HubModelDeleteResponse",
    "HubModelItem",
    "JobCheckpoint",
    "JobCheckpointsResponse",
    "JobListResponse",
    "JobLogsResponse",
    "JobQueueResponse",
    "JobMetricsHistoryResponse",
    "JobRecord",
    "LogLine",
    "MetricsHistoryPoint",
    "RunnerFlavor",
    "RunnersHardwareResponse",
]


class JobListResponse(BaseModel):
    """server.py list_jobs — JobRegistry.list() records (checkpoint_count and
    the resume lineage annotated at read time), newest first."""

    jobs: list[JobRecord]


class JobQueueResponse(BaseModel):
    """server.py list_job_queue / reorder_job_queue — the WHOLE local training
    queue (JobRegistry.list_queue / reorder_queue), in the order it will run,
    each record annotated with its 1-based queue_position. Same JobRecord model
    as the history list — a queued record is uniform-with-defaults, not a
    different shape — but the ordering contract differs (run order, uncapped),
    which is why this is not JobListResponse."""

    jobs: list[JobRecord]


class JobLogsResponse(BaseModel):
    """server.py get_job_logs / get_job_log_file — both return {"logs": [...]}
    of the same LogLine model: the live drained tail for /logs, the whole
    persisted log.jsonl for /log-file (JSON, not a file download)."""

    logs: list[LogLine]


class JobMetricsHistoryResponse(BaseModel):
    """server.py get_job_metrics_history — the per-step series JobRegistry.
    read_metrics_history reconstructs from log.jsonl across the resume chain."""

    points: list[MetricsHistoryPoint]


class JobCheckpointsResponse(BaseModel):
    """server.py get_job_checkpoints — JobRegistry.list_checkpoints, ascending
    by step.

    ``?lineage=true`` serves the same shape from list_chain_checkpoints (the
    whole resume chain); those rows are the ones that carry JobCheckpoint's
    owner_* stamps, which a single-run listing leaves null."""

    checkpoints: list[JobCheckpoint]


class CheckpointImageFeature(BaseModel):
    """One camera's expected input size in a checkpoint's policy config
    (jobs.py JobRegistry.get_policy_config_summary)."""

    height: int
    width: int


class CheckpointPolicyConfigResponse(BaseModel):
    """jobs.py JobRegistry.get_policy_config_summary — the UX-relevant slice
    of a checkpoint's pretrained_model/config.json. policy_type passes through
    from the file's "type" key (null when absent); state_dim/action_dim are
    null when the checkpoint omits the feature."""

    policy_type: str | None
    image_features: dict[str, CheckpointImageFeature]
    requires_task: bool
    state_dim: int | None
    action_dim: int | None


class HubJobStatus(BaseModel):
    """The {stage, message} pair of one Hub job (server.py list_hub_jobs, from
    huggingface_hub's JobStatus)."""

    stage: str
    message: str | None


class HubJobItem(BaseModel):
    """One row of GET /jobs/hub `jobs` (server.py list_hub_jobs). Every key is
    always present; the nullables mirror huggingface_hub's JobInfo (docker_image
    and space_id are mutually exclusive on the Hub side, status/owner can be
    absent objects → null, name is _hub_job_run_name's best effort)."""

    id: str
    name: str | None
    created_at: str | None
    docker_image: str | None
    space_id: str | None
    flavor: str | None
    status: HubJobStatus | None
    owner: str | None
    url: str


class HubModelItem(BaseModel):
    """One row of GET /jobs/hub `models` (server.py list_hub_jobs `_add`)."""

    repo_id: str
    last_modified: str | None
    private: bool


class HubJobsResponse(BaseModel):
    """server.py list_hub_jobs. Heterogeneous by branch: the unauthenticated
    body is exactly {authenticated, jobs, models} — jobs_permission is absent
    there (never null), and present (true/false) when authenticated. The route
    serializes with exclude_unset so each branch keeps its exact keys while the
    rows' legitimate nulls (name, created_at, status, …) still go out."""

    authenticated: bool
    jobs_permission: bool | None = None
    jobs: list[HubJobItem]
    models: list[HubModelItem]


class HubModelDeleteResponse(BaseModel):
    """server.py delete_hub_model (success path only; refusals raise) —
    idempotent, so an already-gone repo reports success too."""

    status: Literal["success"]
    repo_id: str


class HubJobDismissResponse(BaseModel):
    """server.py dismiss_hub_job — job_id is the stripped id that was persisted
    to the dismissal file, not necessarily the caller's raw input."""

    status: Literal["success"]
    job_id: str


class RunnerFlavor(BaseModel):
    """One HF Jobs hardware flavor (server.py get_runners_hardware, flattened
    from huggingface_hub's JobHardwareInfo). accelerator is the label
    _format_accelerator renders ("2× Nvidia A100"), null on cpu-* flavors, as
    is vram (the Hub words it as a string, "16 GB"). unit_cost_usd is
    `int | float` so it passes through exactly as the Hub sent it."""

    name: str
    pretty_name: str
    cpu: str
    ram: str
    accelerator: str | None
    vram: str | None
    unit_cost_usd: int | float
    unit_label: str


class RunnersHardwareResponse(BaseModel):
    """server.py get_runners_hardware — every branch (unauthenticated, flavor
    fetch failed, cached catalog) carries all four keys; username is null (not
    absent) when unauthenticated."""

    authenticated: bool
    username: str | None
    flavors: list[RunnerFlavor]
    offline: bool
