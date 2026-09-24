# Copyright 2026 MakerMods. All rights reserved.
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
#
"""Deterministic training-to-Hub composition, separate from wire operations.

``train_and_publish`` composes ``jobs.create_training``, ``jobs.wait``,
``models.checkpoints``, ``models.publish`` and ``models.publish_status``.
Use those primitives directly when a run needs a different publish policy.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from makermodslab_sdk.errors import CompatibilityWarning, MakerModsError
from makermodslab_sdk.resources.jobs import Job
from makermodslab_sdk.resources.models import PublishStatus

if TYPE_CHECKING:
    from makermodslab_sdk.client import Client


class TrainingFlowError(MakerModsError):
    """A run or publish could not complete this flow; ``job_id`` is resumable."""

    def __init__(self, message: str, *, job_id: str) -> None:
        super().__init__(message)
        self.job_id = job_id


class PublishWaitTimeout(TrainingFlowError, TimeoutError):  # noqa: N818
    """Publishing is still running; ``publish_id`` identifies its slot attempt
    (``None`` when the server predates attempt ids)."""

    def __init__(self, *, job_id: str, publish_id: str | None, waited: float, last_state: str) -> None:
        if publish_id is not None:
            confirm = f"confirm .publish_id == {publish_id!r}"
        else:
            # An older server names no attempt: the job is the best identity,
            # and it cannot tell this attempt from a later publish of the job.
            confirm = (
                f"confirm .model_id == {job_id!r} (this server reports no publish attempt id, "
                "so a later publish of the same job looks identical)"
            )
        super().__init__(
            f"Publish for job {job_id!r} is still {last_state!r} after {waited:g}s. "
            "The publish remains active; call client.models.publish_status() to keep checking it "
            f"and {confirm}. Next step: client.models.publish_status().",
            job_id=job_id,
        )
        self.publish_id = publish_id
        self.waited = waited
        self.last_state = last_state


@dataclass(frozen=True)
class TrainAndPublishResult:
    """The terminal local job and its successful Hub publish status."""

    job: Job
    publish: PublishStatus


class TrainingFlows:
    """High-level local-training sequences bound to a ``Client``."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def train_and_publish(
        self,
        dataset_repo_id: str,
        *,
        repo_id: str | None = None,
        publish_steps: Sequence[int] | None = None,
        steps: int | None = None,
        train_timeout: float,
        publish_timeout: float,
        poll_interval: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        **training_knobs: Any,
    ) -> TrainAndPublishResult:
        """Train locally, wait for success, then publish its checkpoints.

        Composes ``jobs.create_training`` → ``jobs.wait`` →
        ``models.checkpoints`` → ``models.publish`` →
        ``models.publish_status``. Both timeout budgets start at their phase;
        neither timeout cancels work. A training timeout is
        ``JobWaitTimeout``, whose
        ``job_id`` can be passed to ``client.jobs.wait`` again. A publish
        timeout is ``PublishWaitTimeout``; use
        ``client.models.publish_status()`` to resume observing it.

        Only local training is supported. ``training_knobs`` are the same
        validated options as ``jobs.create_training``. ``steps`` is the
        training length; ``publish_steps`` selects checkpoints to publish.

        Example:
            >>> result = client.flows.train_and_publish(
            ...     "me/pick", repo_id="me/act-pick", steps=20000, train_timeout=14400, publish_timeout=3600
            ... )
            >>> result.publish.repo_id
            'me/act-pick'
        """
        if train_timeout < 0 or publish_timeout < 0:
            raise ValueError("train_timeout and publish_timeout must be nonnegative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        if steps is not None:
            training_knobs["steps"] = steps
        job = self._client.jobs.create_training(dataset_repo_id, runner="local", **training_knobs)
        job_id = job.id
        job = self._client.jobs.wait(
            job_id, timeout=train_timeout, poll_interval=poll_interval, sleep_fn=sleep_fn, clock=clock
        )
        if job.state != "done":
            raise TrainingFlowError(
                f"Job {job_id!r} ended {job.state!r}: {job.error_message or 'no error message'}. "
                f"Next step: inspect client.jobs.get({job_id!r}) and client.jobs.logs({job_id!r}); "
                "no publish was started.",
                job_id=job_id,
            )
        if job.runner != "local":
            raise TrainingFlowError(
                f"Job {job_id!r} finished with runner {job.runner!r}, not 'local'; "
                f"no publish was started. Next step: client.jobs.get({job_id!r}).",
                job_id=job_id,
            )

        checkpoints = self._client.models.checkpoints(job_id)
        if checkpoints.id != job_id:
            raise TrainingFlowError(
                f"Checkpoint listing for job {job_id!r} belongs to {checkpoints.id!r}; no publish was started. "
                f"Next step: retry client.models.checkpoints({job_id!r}).",
                job_id=job_id,
            )
        if not checkpoints.checkpoints:
            raise TrainingFlowError(
                f"Job {job_id!r} finished without publishable checkpoints; no publish was started. "
                f"Next step: inspect client.models.checkpoints({job_id!r}) "
                f"and client.jobs.logs({job_id!r}).",
                job_id=job_id,
            )
        selected_steps = list(publish_steps) if publish_steps is not None else None
        if selected_steps is not None:
            available = {item.step for item in checkpoints.checkpoints}
            missing = sorted(set(selected_steps) - available)
            if not selected_steps or missing:
                raise TrainingFlowError(
                    f"Job {job_id!r} has no checkpoint for requested steps {missing or selected_steps!r}; "
                    f"Next step: choose steps from client.models.checkpoints({job_id!r}).checkpoints. "
                    "No publish was started.",
                    job_id=job_id,
                )

        expected_repo_id = repo_id or checkpoints.default_repo_id
        started = self._client.models.publish(job_id, repo_id=repo_id, steps=selected_steps)
        if not started.started or started.model_id != job_id:
            raise TrainingFlowError(
                f"Publish for job {job_id!r} was not accepted (slot model_id={started.model_id!r}): "
                f"{started.message}. Next step: check client.models.publish_status() before retrying.",
                job_id=job_id,
            )

        if started.publish_id is None:
            # Server skew: a build before publish attempt ids. Comparing None
            # with None would pass vacuously, so say plainly what is not checked.
            warnings.warn(
                f"the server at {self._client.base_url} returned no publish_id, so this flow cannot "
                f"verify the publish attempt it started; it checks only that the slot belongs to job "
                f"{job_id!r} and repository {expected_repo_id!r}. Update the server for attempt identity.",
                CompatibilityWarning,
                stacklevel=2,
            )

        started_at = clock()
        while True:
            status = self._client.models.publish_status()
            if started.publish_id is not None and status.publish_id != started.publish_id:
                raise TrainingFlowError(
                    f"The publish slot now belongs to attempt {status.publish_id!r}, not "
                    f"{started.publish_id!r} for job {job_id!r}. Next step: inspect "
                    f"client.models.publish_status() and client.models.checkpoints({job_id!r}); "
                    "this flow cannot confirm the result.",
                    job_id=job_id,
                )
            if status.model_id != job_id:
                raise TrainingFlowError(
                    f"The publish slot now belongs to {status.model_id!r}, not job {job_id!r}. "
                    f"Next step: inspect client.models.publish_status() and client.models.checkpoints({job_id!r}); "
                    "this flow cannot confirm the result.",
                    job_id=job_id,
                )
            if status.repo_id is not None and status.repo_id != expected_repo_id:
                raise TrainingFlowError(
                    f"The publish slot now targets {status.repo_id!r}, not {expected_repo_id!r}. "
                    "Next step: inspect client.models.publish_status(); this flow cannot confirm the result.",
                    job_id=job_id,
                )
            if status.state == "done":
                if status.repo_id != expected_repo_id:
                    raise TrainingFlowError(
                        f"Publish for job {job_id!r} finished without the expected repository "
                        f"{expected_repo_id!r} (slot repo_id={status.repo_id!r}). "
                        "Next step: inspect client.models.publish_status(); this flow cannot confirm the result.",
                        job_id=job_id,
                    )
                return TrainAndPublishResult(job=job, publish=status)
            if status.state == "error":
                raise TrainingFlowError(
                    f"Publish for job {job_id!r} failed: {status.error or status.message or 'unknown error'}. "
                    f"Next step: inspect client.models.publish_status() and client.models.checkpoints({job_id!r}); "
                    "already published checkpoints may remain on the Hub.",
                    job_id=job_id,
                )
            if status.state != "running":
                raise TrainingFlowError(
                    f"Publish for job {job_id!r} has unexpected state {status.state!r}. "
                    "Next step: inspect client.models.publish_status().",
                    job_id=job_id,
                )
            waited = max(0.0, clock() - started_at)
            if waited >= publish_timeout:
                raise PublishWaitTimeout(
                    job_id=job_id,
                    publish_id=started.publish_id,
                    waited=waited,
                    last_state=status.state,
                )
            sleep_fn(min(poll_interval, publish_timeout - waited))
