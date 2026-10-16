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
"""Deterministic recording composition over sessions and recording primitives."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.resources.recording import RecordingStatus
from makermodslab_sdk.resources.sessions import EndedSessionInfo, SessionWaitTimeout

if TYPE_CHECKING:
    from makermodslab_sdk.client import Client


class RecordingFlowError(MakerModsError):
    """The recording run failed or a task prompt was refused."""

    def __init__(self, message: str, *, session_id: str, dataset_repo_id: str, saved_episodes: int) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.dataset_repo_id = dataset_repo_id
        self.saved_episodes = saved_episodes


class RecordingFlowTimeout(RecordingFlowError, TimeoutError):  # noqa: N818 - matches builtins.TimeoutError
    """The recording timed out; the owned session was stopped on exit."""


@dataclass(frozen=True)
class RecordingFlowResult:
    """The final recording summary and its matching session end."""

    session: EndedSessionInfo
    dataset_repo_id: str
    saved_episodes: int
    outcome: str | None
    discarded_empty: bool | None
    status: RecordingStatus
    start_warnings: tuple[str, ...]


class RecordingFlows:
    """Recording flow implementation; ``client.flows`` is the public entry."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def record_episodes(
        self,
        robot: str,
        dataset_repo_id: str,
        *,
        task: str | Sequence[str],
        episodes: int | None = None,
        timeout: float = 3600.0,
        poll_interval: float = 1.0,
        on_progress: Callable[[RecordingStatus], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        **record_options: Any,
    ) -> RecordingFlowResult:
        """Record a fixed task or one prompt per episode, then await completion.

        Composes ``sessions.record``, ``sessions.recording_status``,
        ``sessions.recording_episode_task`` (for a sequence of tasks), and
        ``ActiveSession.wait``. A string task requires ``episodes``; a task
        sequence determines the count and must match ``episodes`` if supplied.
        ``record_options`` passes remaining ``sessions.record`` options such
        as ``fps`` and ``push_to_hub``. The server owns return-to-rest on stop.

        A timeout stops this flow's session and raises RecordingFlowTimeout;
        the error carries the session id, dataset id and last saved count.
        ``on_progress`` receives each polled status. The result keeps the
        terminal status (including warning, error and hint) and any start
        warnings. Inject ``sleep_fn`` and ``clock`` in tests so no test sleeps
        or touches hardware.

        Example:
            >>> result = client.flows.record_episodes("bench", "me/pick", task="pick the cube", episodes=5)
            >>> result.saved_episodes, result.dataset_repo_id
            (5, 'me/pick_20260922_120000')
            >>> result.status.warning  # inspect even when outcome is ran_with_warning
        """
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("timeout must be nonnegative and poll_interval positive")
        if isinstance(task, str):
            if episodes is None or episodes < 1 or not task.strip():
                raise ValueError("a nonempty string task requires episodes >= 1")
            tasks: list[str] | None = None
            single_task, count = task, episodes
        else:
            tasks = list(task)
            if not tasks or any(not isinstance(item, str) or not item.strip() for item in tasks):
                raise ValueError("task sequence must contain nonempty strings")
            if episodes is not None and episodes != len(tasks):
                raise ValueError("episodes must equal the task sequence length")
            single_task, count = tasks[0], len(tasks)
        reserved = {
            "dataset_repo_id",
            "single_task",
            "num_episodes",
            "per_episode_task",
        } & record_options.keys()
        if reserved:
            raise ValueError(f"record_options must not override flow fields: {sorted(reserved)}")

        saved = 0
        started_at = clock()
        actual_dataset_id: str | None = None
        prompted_episode: int | None = None
        with self._client.sessions.record(
            robot,
            dataset_repo_id=dataset_repo_id,
            single_task=single_task,
            num_episodes=count,
            per_episode_task=tasks is not None,
            **record_options,
        ) as session:
            while True:
                status = self._client.sessions.recording_status(session.id)
                saved = status.saved_episodes
                if status.dataset_repo_id is None:
                    raise RecordingFlowError(
                        f"Recording {session.id} status omitted its actual dataset id. "
                        "Next step: client.recording.status() should report dataset_repo_id.",
                        session_id=session.id,
                        dataset_repo_id=actual_dataset_id or dataset_repo_id,
                        saved_episodes=saved,
                    )
                if actual_dataset_id is None:
                    actual_dataset_id = status.dataset_repo_id
                elif status.dataset_repo_id != actual_dataset_id:
                    raise RecordingFlowError(
                        f"Recording status switched from dataset {actual_dataset_id!r} "
                        f"to {status.dataset_repo_id!r}. Next step: client.sessions.current() "
                        "identifies the live session; do not submit another prompt.",
                        session_id=session.id,
                        dataset_repo_id=actual_dataset_id,
                        saved_episodes=saved,
                    )
                if on_progress is not None:
                    on_progress(status)
                if status.session_ended:
                    elapsed = max(0.0, clock() - started_at)
                    try:
                        ended = session.wait(
                            timeout=max(0.0, timeout - elapsed),
                            poll_interval=poll_interval,
                            sleep_fn=sleep_fn,
                            clock=clock,
                        )
                    except SessionWaitTimeout as exc:
                        raise RecordingFlowTimeout(
                            f"Recording {session.id} reached a terminal status but its session has not released. "
                            "Next step: client.sessions.current() and client.recording.status() show its state.",
                            session_id=session.id,
                            dataset_repo_id=actual_dataset_id,
                            saved_episodes=saved,
                        ) from exc
                    if ended.phase == "error" or status.outcome == "failed":
                        raise RecordingFlowError(
                            f"Recording {session.id} failed: {status.error or status.hint or 'no detail'}. "
                            "Next step: client.recording.status() has the terminal outcome.",
                            session_id=session.id,
                            dataset_repo_id=actual_dataset_id,
                            saved_episodes=saved,
                        )
                    if saved != count:
                        raise RecordingFlowError(
                            f"Recording {session.id} ended after saving {saved} of {count} requested episodes. "
                            "Next step: client.recording.status() shows the final outcome; "
                            "use client.datasets.list() to inspect the partial dataset before resuming.",
                            session_id=session.id,
                            dataset_repo_id=actual_dataset_id,
                            saved_episodes=saved,
                        )
                    return RecordingFlowResult(
                        ended,
                        actual_dataset_id,
                        saved,
                        status.outcome,
                        status.discarded_empty,
                        status,
                        tuple(session.warnings),
                    )
                elapsed = max(0.0, clock() - started_at)
                if elapsed >= timeout:
                    raise RecordingFlowTimeout(
                        f"Recording {session.id} is still {status.current_phase!r} after {elapsed:g}s "
                        f"(timeout={timeout:g}); its session is being stopped. "
                        "Next step: client.recording.status() shows how many episodes were saved.",
                        session_id=session.id,
                        dataset_repo_id=actual_dataset_id,
                        saved_episodes=saved,
                    )
                if tasks is not None and status.current_phase == "naming":
                    current = self._client.sessions.current().session
                    if current is None or current.id != session.id:
                        raise RecordingFlowError(
                            f"Recording {session.id} no longer owns the live session. "
                            "Next step: client.sessions.current() identifies the new owner; "
                            "no prompt was submitted.",
                            session_id=session.id,
                            dataset_repo_id=actual_dataset_id,
                            saved_episodes=saved,
                        )
                    episode = status.current_episode
                    if episode is None or not 1 <= episode <= len(tasks):
                        raise RecordingFlowError(
                            f"Recording {session.id} requested a prompt for episode {episode!r}, "
                            f"outside the {len(tasks)} supplied tasks. "
                            "Next step: client.recording.status() shows the current episode.",
                            session_id=session.id,
                            dataset_repo_id=actual_dataset_id,
                            saved_episodes=saved,
                        )
                    if episode != prompted_episode:
                        answer = self._client.sessions.recording_episode_task(session.id, tasks[episode - 1])
                        if not answer.success:
                            raise RecordingFlowError(
                                f"Recording {session.id} refused episode {episode}'s task: {answer.message}. "
                                "Next step: client.recording.status() shows which phase accepts a task.",
                                session_id=session.id,
                                dataset_repo_id=actual_dataset_id,
                                saved_episodes=saved,
                            )
                        prompted_episode = episode
                else:
                    prompted_episode = None
                elapsed = max(0.0, clock() - started_at)
                if elapsed >= timeout:
                    raise RecordingFlowTimeout(
                        f"Recording {session.id} is still {status.current_phase!r} after {elapsed:g}s "
                        f"(timeout={timeout:g}); its session is being stopped. "
                        "Next step: client.recording.status() shows how many episodes were saved.",
                        session_id=session.id,
                        dataset_repo_id=actual_dataset_id,
                        saved_episodes=saved,
                    )
                sleep_fn(min(poll_interval, timeout - elapsed))
