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

"""Machine-readable error codes for the API.

Grammar: ``<domain>.<condition>[.<detail>]`` — dots separate hierarchy levels,
underscores separate words within a level, so ``code.split(".")`` always
recovers the structure. Level 1 comes from a closed domain set; both rules are
contract-tested in tests/test_api_errors.py, which is where extending the
namespace starts.

Codes are ADDITIVE metadata: they ride beside the human-readable message
(``code`` next to ``detail`` in HTTP error bodies, a ``code`` key in the
``{"success": False, ...}`` refusal dicts), and clients that predate them keep
working unchanged. The message wording stays free to improve; the code is the
stable contract an SDK dispatches on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    # request.* — the request itself is malformed or violates naming rules.
    REQUEST_VALIDATION = "request.validation"
    REQUEST_INVALID_NAME = "request.invalid_name"

    # robot.* — the persisted robot record, and the busy family: the
    # mutual-exclusion refusals, one discriminant per feature so a client can
    # tell WHAT holds the robot. `releasing` is the post-session grace window
    # (arm still energized, driving back to rest) — retry shortly.
    ROBOT_NOT_FOUND = "robot.not_found"
    ROBOT_NOT_READY = "robot.not_ready"
    ROBOT_BUSY_RECORDING = "robot.busy.recording"
    ROBOT_BUSY_TELEOPERATION = "robot.busy.teleoperation"
    ROBOT_BUSY_INFERENCE = "robot.busy.inference"
    ROBOT_BUSY_REPLAY = "robot.busy.replay"
    ROBOT_BUSY_CALIBRATION = "robot.busy.calibration"
    ROBOT_BUSY_AUTO_CALIBRATION = "robot.busy.auto_calibration"
    ROBOT_BUSY_WIGGLE = "robot.busy.wiggle"
    # Remote inference (makermodslab/remote_inference.py): a policy on a remote
    # GPU driving this machine's follower over LiveKit. Its own discriminant
    # rather than `inference` because the two are different sessions with
    # different stop machinery — a client refused by one and pointed at the
    # other's Stop button would get an endpoint that reports idle.
    ROBOT_BUSY_REMOTE_INFERENCE = "robot.busy.remote_inference"
    ROBOT_BUSY_RELEASING = "robot.busy.releasing"
    # A live LOCAL training run holds the machine (GPU + the arms' USB bus).
    # The reverse direction never refuses: a submit made while a feature runs
    # QUEUES instead (jobs.JobRegistry._robot_busy).
    ROBOT_BUSY_TRAINING = "robot.busy.training"

    # hardware.* — the physical layer (serial bus, servos), distinct from the
    # robot record.
    HARDWARE_PORT_UNAVAILABLE = "hardware.port_unavailable"
    HARDWARE_CONNECT_FAILED = "hardware.connect_failed"
    HARDWARE_IDENTITY_MISMATCH = "hardware.identity_mismatch"
    HARDWARE_TORQUE_RELEASE_FAILED = "hardware.torque_release_failed"

    # hub.* — the Hugging Face Hub as an external service.
    HUB_UNAUTHENTICATED = "hub.unauthenticated"
    HUB_OFFLINE = "hub.offline"
    HUB_REPO_NOT_FOUND = "hub.repo_not_found"
    HUB_UPLOAD_FAILED = "hub.upload_failed"

    # job.* — training-run lifecycle (maps 1:1 from jobs.py's exception types).
    JOB_NOT_FOUND = "job.not_found"
    JOB_ALREADY_RUNNING = "job.already_running"
    JOB_ALREADY_CONTINUED = "job.already_continued"
    JOB_HAS_CHILDREN = "job.has_children"
    JOB_NOT_RUNNING = "job.not_running"
    JOB_DATASET_NOT_ON_HUB = "job.dataset_not_on_hub"
    # The local training queue (PR #83). `queue_stale`: a reorder named a set
    # of runs that is no longer the queue — refetch and retry, the one 409 in
    # this family a retry can clear. `state_changed`: a stop/cancel whose
    # expect_state precondition no longer holds. `has_queued_dependents`: a
    # delete/cancel would take the checkpoint a QUEUED run will train from —
    # the fine-tune edge build_child_index deliberately does not model.
    # `removal_failed`: the record's job.json could not be unlinked; nothing
    # was removed and the request is safe to retry.
    JOB_QUEUE_STALE = "job.queue_stale"
    JOB_STATE_CHANGED = "job.state_changed"
    JOB_HAS_QUEUED_DEPENDENTS = "job.has_queued_dependents"
    JOB_REMOVAL_FAILED = "job.removal_failed"
    # `not_terminal`: a surface that deletes a run's ARTIFACTS (POST
    # /models/delete) was aimed at a run that hasn't finished — running or
    # queued. Distinct from `not_running` (a stop aimed at nothing stoppable):
    # here the run is very much alive or about to be, and the remedy is to
    # stop/cancel it where it lives (the jobs surface), not to retry the
    # delete.
    JOB_NOT_TERMINAL = "job.not_terminal"

    # Library resources.
    DATASET_NOT_FOUND = "dataset.not_found"
    MODEL_NOT_FOUND = "model.not_found"
    CHECKPOINT_NOT_FOUND = "checkpoint.not_found"
    CHECKPOINT_INCOMPLETE = "checkpoint.incomplete"

    # node.* — the peer-node registry (other MakerMods Lab servers on the
    # LAN/tailnet). `unreachable` covers both a dead host and one that answers
    # /api/v1/health with something that isn't a node identity document.
    NODE_NOT_FOUND = "node.not_found"
    NODE_UNREACHABLE = "node.unreachable"
    NODE_DUPLICATE = "node.duplicate"
    NODE_SELF = "node.self"

    # session.* — the /api/v1/sessions surface (sessions.py). `held`: another
    # session holds the hardware (details name the holder). `not_found`: a
    # stop or heartbeat aimed at anything but the current session's id.
    # `not_owner`: a heartbeat whose owner doesn't match the lease — never
    # stop, which is deliberately owner-ungated (safety over ownership).
    # `lease_expired`: doubles as last_ended's safety-stop reason, and as an
    # HTTP 409 only for a heartbeat in the window between the expiry
    # watchdog's stop dispatch and the session's release.
    SESSION_HELD = "session.held"
    SESSION_NOT_OWNER = "session.not_owner"
    SESSION_LEASE_EXPIRED = "session.lease_expired"
    SESSION_NOT_FOUND = "session.not_found"

    # transport.* — the LiveKit path remote inference runs over (the SFU and
    # the room), an external service this node depends on. Its own domain for
    # the same reason `hub` has one: folding it into `hardware.connect_failed`
    # would lie (that is the serial bus) and so would `system.*` (it is not
    # this process). `no_policy` is the empty-room case — the room answers but
    # no GPU-side operator is in it, caught BEFORE the arm is energized.
    TRANSPORT_EXTRA_MISSING = "transport.extra_missing"
    TRANSPORT_NOT_CONFIGURED = "transport.not_configured"
    TRANSPORT_UNREACHABLE = "transport.unreachable"
    TRANSPORT_UNAUTHORIZED = "transport.unauthorized"
    TRANSPORT_NO_POLICY = "transport.no_policy"

    # system.* — the server process itself. `restart_unsupported`: this
    # process cannot safely re-exec (a dev reload worker, or a launch whose
    # argv isn't one of our entry points) — the remedy is restarting it the
    # way it was started, not retrying. `install_in_progress`: a restart
    # would orphan a live pip subprocess mid-write — retry once it finishes.
    SYSTEM_RESTART_UNSUPPORTED = "system.restart_unsupported"
    SYSTEM_INSTALL_IN_PROGRESS = "system.install_in_progress"

    # gpu.* — the remote GPU that runs the policy for a remote-inference run
    # (modal_launcher.py), reached through the `modal` CLI. Its own level-1
    # domain by the same argument `transport` earned one: a second external
    # service this node depends on, with a different remedy set. `transport.*`
    # would blunt four rungs that are carefully distinguished (the GPU is
    # neither the SFU nor the room), and `system.*` would lie — `unauthenticated`
    # and `launch_failed` are facts about Modal, not about this process.
    # `cli_missing`: the binary isn't on PATH (remedy: `uv tool install modal`).
    # `unauthenticated`: Modal rejected this machine (remedy: `modal token new`;
    # the Lab never touches ~/.modal.toml). `already_running`/`not_running`: a
    # start against a live launcher, a stop against a dead one — the GPU is a
    # Lab-level resource, so these are its own, not `robot.busy.*`.
    GPU_CLI_MISSING = "gpu.cli_missing"
    GPU_UNAUTHENTICATED = "gpu.unauthenticated"
    GPU_ALREADY_RUNNING = "gpu.already_running"
    GPU_NOT_RUNNING = "gpu.not_running"
    GPU_LAUNCH_FAILED = "gpu.launch_failed"

    # sfu.* — the bundled LiveKit server (sfu.py). `disabled`: this process
    # was started without --sfu (or an external SFU configured), so there is
    # no secret to sign room tokens with — the remedy is restarting the
    # launcher with --sfu, not retrying.
    SFU_DISABLED = "sfu.disabled"

    # The residual 500.
    INTERNAL_UNEXPECTED = "internal.unexpected"


class ApiError(HTTPException):
    """HTTPException that carries an ErrorCode into the response body.

    The registered handler (install_error_handlers) serializes it as
    ``{"detail": <message>, "code": <code>}`` — the legacy string ``detail``
    unchanged, ``code`` an additive sibling. ``details`` (optional) adds a
    machine-readable sibling of the same name for structured context a client
    dispatches on beyond the code — e.g. session.held names its holder there.
    """

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        code: ErrorCode | str | None = None,
        headers: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code
        self.details = details


def install_error_handlers(app: FastAPI) -> None:
    """Register the ApiError → JSON body handler on `app`."""

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        body: dict[str, Any] = {"detail": exc.detail}
        if exc.code is not None:
            body["code"] = str(exc.code)
        if exc.details is not None:
            body["details"] = exc.details
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's 422, with `code` as an additive sibling.

        The body keeps FastAPI's exact `detail` shape (the pydantic error
        list), so clients that parse it keep working; `request.validation`
        rides beside it the way every ApiError's code does — a schema-level
        refusal (an unparsable body, an out-of-range field, a reorder list
        past its 512-id bound) is the request domain's own condition, and
        an SDK should not have to sniff the 422 status to know that.
        """
        body: dict[str, Any] = {
            "detail": jsonable_encoder(exc.errors()),
            "code": str(ErrorCode.REQUEST_VALIDATION),
        }
        return JSONResponse(status_code=422, content=body)
