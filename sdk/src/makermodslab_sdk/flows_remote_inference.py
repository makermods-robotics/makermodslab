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
"""Managed remote-inference composition, separate from wire operations.

``remote_inference`` composes ``sessions.remote_inference_transport``,
``sessions.gpu_start``, ``sessions.gpu_status``, ``sessions.remote_infer``
and the matching session/GPU stops. Use those primitives directly when the
GPU is launched elsewhere or its lifetime should outlive the robot session.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.resources.sessions import (
    ActiveSession,
    GpuLaunch,
    GpuStatus,
    RemoteInferenceTransport,
    StoppedSession,
)

if TYPE_CHECKING:
    from types import TracebackType

    from makermodslab_sdk.client import Client


class RemoteInferenceFlowError(MakerModsError):
    """A managed remote run could not start or clean up; IDs make recovery scoped."""

    def __init__(
        self,
        message: str,
        *,
        launch_id: str | None,
        session_id: str | None,
        last_state: str,
        last_phase: str | None = None,
        recovery_call: str,
    ) -> None:
        super().__init__(message)
        self.launch_id = launch_id
        self.session_id = session_id
        self.last_state = last_state
        self.last_phase = last_phase
        self.recovery_call = recovery_call
        #: Cleanup failure retained when another exception has priority.
        self.cleanup_error: BaseException | None = None


class RemoteInferenceStartupTimeout(RemoteInferenceFlowError, TimeoutError):  # noqa: N818
    """The owned GPU launch did not become ready inside the startup budget."""

    def __init__(
        self,
        *,
        launch_id: str,
        waited: float,
        last_state: str,
        last_phase: str | None,
    ) -> None:
        recovery = "client.sessions.gpu_status()"
        super().__init__(
            f"GPU launch {launch_id!r} is still {last_state!r} "
            f"(phase={last_phase!r}) after {waited:g}s and was asked to stop. "
            f"Next step: inspect {recovery}; if it is still stopping, wait before retrying.",
            launch_id=launch_id,
            session_id=None,
            last_state=last_state,
            last_phase=last_phase,
            recovery_call=recovery,
        )
        self.waited = waited


class RemoteInferenceRun:
    """The live robot session and the exact GPU launch this context owns."""

    def __init__(
        self,
        client: Client,
        *,
        robot: str,
        policy_ref: str,
        policy_hub_id: str,
        gpu: str | None,
        profile: str | None,
        environment: str | None,
        region: str | None,
        task: str,
        horizon: int,
        fps: int,
        video_codec: Literal["H264", "MJPEG"],
        engine: Literal["sync", "rtc"],
        s_min: int,
        duration_s: int | None,
        owner: str | None,
        lease_timeout_s: float | None,
        startup_timeout: float,
        poll_interval: float,
        sleep_fn: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        self._client = client
        self._robot = robot
        self._policy_ref = policy_ref
        self._policy_hub_id = policy_hub_id
        self._gpu = gpu
        self._profile = profile
        self._environment = environment
        self._region = region
        self._task = task
        self._horizon = horizon
        self._fps = fps
        self._video_codec = video_codec
        self._engine = engine
        self._s_min = s_min
        self._duration_s = duration_s
        self._owner = owner
        self._lease_timeout_s = lease_timeout_s
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._sleep_fn = sleep_fn
        self._clock = clock

        self._transport: RemoteInferenceTransport | None = None
        self._launch: GpuLaunch | None = None
        self._ready: GpuStatus | None = None
        self._session: ActiveSession | None = None
        self.session_stop: StoppedSession | None = None
        self.gpu_stop: GpuStatus | None = None
        self.cleanup_error: BaseException | None = None
        self._entered = False
        self._closed = False

    @property
    def launch_id(self) -> str | None:
        """The owned transient GPU launch ID, once startup has accepted it."""
        return self._launch.launch_id if self._launch is not None else None

    @property
    def session_id(self) -> str | None:
        """The robot session ID, once the room gate has accepted the start."""
        return self._session.id if self._session is not None else None

    @property
    def transport(self) -> RemoteInferenceTransport:
        """The successful transport preflight captured on context entry."""
        if self._transport is None:
            raise RuntimeError("Enter the remote-inference flow before reading its transport")
        return self._transport

    @property
    def launch(self) -> GpuLaunch:
        """The accepted GPU start response captured on context entry."""
        if self._launch is None:
            raise RuntimeError("Enter the remote-inference flow before reading its launch")
        return self._launch

    @property
    def ready(self) -> GpuStatus:
        """The exact-launch GPU status that triggered the robot start attempt."""
        if self._ready is None:
            raise RuntimeError("The owned GPU launch has not become ready")
        return self._ready

    @property
    def session(self) -> ActiveSession:
        """The managed robot session, available inside the entered context."""
        if self._session is None:
            raise RuntimeError("The remote-inference robot session has not started")
        return self._session

    @property
    def start_warnings(self) -> tuple[str, ...]:
        """Warn-but-allow findings returned by the robot session start."""
        return tuple(self._session.warnings) if self._session is not None else ()

    def _flow_error(
        self,
        message: str,
        *,
        last_state: str,
        last_phase: str | None = None,
        recovery_call: str,
    ) -> RemoteInferenceFlowError:
        return RemoteInferenceFlowError(
            f"{message} Next step: {recovery_call}.",
            launch_id=self.launch_id,
            session_id=self.session_id,
            last_state=last_state,
            last_phase=last_phase,
            recovery_call=recovery_call,
        )

    def _check_transport(self) -> RemoteInferenceTransport:
        status = self._client.sessions.remote_inference_transport()
        self._transport = status
        # An absent policy peer is the expected pre-launch observation. Every
        # other coded failure means the GPU would be launched into a transport
        # that the robot cannot use.
        unusable_code = status.error_code not in (None, "transport.no_policy")
        if (
            not status.extra_installed
            or not status.configured
            or status.endpoint_reachable is False
            or unusable_code
        ):
            state = status.error_code or "transport.unavailable"
            detail = status.message or "the remote-inference transport is unavailable"
            raise self._flow_error(
                f"Remote-inference preflight failed ({state}): {detail}.",
                last_state=state,
                recovery_call="client.sessions.remote_inference_transport()",
            )
        return status

    def _start_gpu(self) -> GpuLaunch:
        fields: dict[str, object] = {
            "policy_hub_id": self._policy_hub_id,
            "engine": self._engine,
            "task": self._task,
            "horizon": self._horizon,
            "fps": self._fps,
            "video_codec": self._video_codec,
            "s_min": self._s_min,
        }
        for name, value in (
            ("gpu", self._gpu),
            ("profile", self._profile),
            ("environment", self._environment),
            ("region", self._region),
        ):
            if value is not None:
                fields[name] = value
        launch = self._client.sessions.gpu_start(**fields)
        self._launch = launch
        if not launch.started or launch.gpu.launch_id != launch.launch_id:
            raise self._flow_error(
                f"GPU launch was not accepted with one stable identity "
                f"(start={launch.launch_id!r}, status={launch.gpu.launch_id!r}): {launch.message}",
                last_state=launch.gpu.state,
                last_phase=launch.gpu.phase,
                recovery_call="client.sessions.gpu_status()",
            )
        return launch

    def _wait_for_gpu(self) -> GpuStatus:
        assert self._launch is not None
        started_at = self._clock()
        while True:
            status = self._client.sessions.gpu_status()
            if status.launch_id != self._launch.launch_id:
                raise self._flow_error(
                    f"GPU slot now belongs to launch {status.launch_id!r}, not this flow's "
                    f"{self._launch.launch_id!r}; no robot session was started.",
                    last_state=status.state,
                    last_phase=status.phase,
                    recovery_call="client.sessions.gpu_status()",
                )
            if status.state == "ready":
                self._ready = status
                return status
            if status.state == "failed":
                diagnosis = status.hint or status.message or status.code or "no server diagnosis"
                raise self._flow_error(
                    f"GPU launch {self._launch.launch_id!r} failed in phase {status.phase!r}: {diagnosis}.",
                    last_state=status.state,
                    last_phase=status.phase,
                    recovery_call="client.sessions.gpu_status()",
                )
            if status.state != "starting":
                raise self._flow_error(
                    f"GPU launch {self._launch.launch_id!r} entered unexpected state {status.state!r}; "
                    "no robot session was started.",
                    last_state=status.state,
                    last_phase=status.phase,
                    recovery_call="client.sessions.gpu_status()",
                )
            waited = max(0.0, self._clock() - started_at)
            if waited >= self._startup_timeout:
                raise RemoteInferenceStartupTimeout(
                    launch_id=self._launch.launch_id,
                    waited=waited,
                    last_state=status.state,
                    last_phase=status.phase,
                )
            self._sleep_fn(min(self._poll_interval, self._startup_timeout - waited))

    def _start_session(self) -> ActiveSession:
        # Matching wire knobs are deliberately sourced from the same fields as
        # _start_gpu. The caller gets no pair of dictionaries that can drift.
        session = self._client.sessions.remote_infer(
            self._robot,
            policy_ref=self._policy_ref,
            policy_hub_id=self._policy_hub_id,
            task=self._task,
            duration_s=self._duration_s,
            horizon=self._horizon,
            fps=self._fps,
            video_codec=self._video_codec,
            engine=self._engine,
            s_min=self._s_min,
            owner=self._owner,
            lease_timeout_s=self._lease_timeout_s,
        )
        self._session = session
        return session

    def _stop_gpu_after_failed_start(self, primary: BaseException) -> None:
        if self.launch_id is None:
            return
        try:
            self.gpu_stop = self._client.sessions.gpu_stop(launch_id=self.launch_id)
        except MakerModsError as cleanup_error:
            self.cleanup_error = cleanup_error
            if isinstance(primary, RemoteInferenceFlowError):
                primary.cleanup_error = cleanup_error

    def __enter__(self) -> RemoteInferenceRun:
        if self._entered:
            raise RuntimeError("A remote-inference flow context cannot be entered twice")
        self._entered = True
        self._check_transport()
        try:
            self._start_gpu()
            self._wait_for_gpu()
            try:
                self._start_session()
            except MakerModsError as exc:
                ready = self._ready
                raise self._flow_error(
                    f"GPU launch {self.launch_id!r} became ready, but the robot session start failed: {exc}",
                    last_state=ready.state if ready else "ready",
                    last_phase=ready.phase if ready else None,
                    recovery_call="client.sessions.current()",
                ) from exc
        except BaseException as exc:
            self._stop_gpu_after_failed_start(exc)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        session_error: BaseException | None = None
        gpu_error: BaseException | None = None
        try:
            if self._session is not None:
                try:
                    # ActiveSession's exit preserves lease-loss classification
                    # while stopping the robot session.
                    self._session.__exit__(exc_type, exc, tb)
                    self.session_stop = self._session._stop_result
                except BaseException as stop_error:
                    session_error = stop_error
        finally:
            if self.launch_id is not None:
                try:
                    self.gpu_stop = self._client.sessions.gpu_stop(launch_id=self.launch_id)
                except BaseException as stop_error:
                    gpu_error = stop_error
                    self.cleanup_error = stop_error

        # Cleanup must never replace an exception from the context body.
        if exc_type is not None:
            return
        if session_error is not None:
            raise session_error
        if gpu_error is not None:
            ready = self._ready
            error = self._flow_error(
                f"Robot session {self.session_id!r} stopped, but conditional cleanup of GPU launch "
                f"{self.launch_id!r} failed: {gpu_error}",
                last_state=ready.state if ready else "unknown",
                last_phase=ready.phase if ready else None,
                recovery_call="client.sessions.gpu_status()",
            )
            error.cleanup_error = gpu_error
            raise error from gpu_error


class RemoteInferenceFlows:
    """High-level remote-inference sequences bound to a ``Client``."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def remote_inference(
        self,
        robot: str,
        *,
        policy_ref: str,
        policy_hub_id: str | None = None,
        gpu: str | None = None,
        profile: str | None = None,
        environment: str | None = None,
        region: str | None = None,
        task: str = "",
        horizon: int = 16,
        fps: int = 30,
        video_codec: Literal["H264", "MJPEG"] = "H264",
        engine: Literal["sync", "rtc"] = "sync",
        s_min: int = 4,
        duration_s: int | None = None,
        startup_timeout: float = 180.0,
        poll_interval: float = 2.0,
        owner: str | None = None,
        lease_timeout_s: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> RemoteInferenceRun:
        """Launch a matching GPU, start remote inference, and own both lifetimes.

        Composes ``sessions.remote_inference_transport`` → ``gpu_start`` →
        exact-``launch_id`` ``gpu_status`` reads → ``remote_infer``. The call
        itself starts nothing; entering the context performs startup. Exit
        stops the robot session first, then conditionally calls
        ``gpu_stop(launch_id=...)``, so it cannot stop a replacement launch.
        The server's room probe remains the authoritative gate before the arm
        energizes; GPU ``ready`` is only the hint that it is time to try it.

        ``policy_hub_id`` defaults to ``policy_ref``. ``engine``, ``task``,
        ``horizon``, ``fps``, ``video_codec`` and ``s_min`` are accepted once
        and sent identically to both halves, preventing a silent wire mismatch.
        A startup timeout stops only this flow's launch and raises
        ``RemoteInferenceStartupTimeout`` with its ID and last state.

        Example:
            >>> with client.flows.remote_inference("bench", policy_ref="me/act-pick", gpu="A10G") as run:
            ...     print(run.session_id, run.launch_id, run.start_warnings)
            ...     run.session.wait(timeout=300)
        """
        if startup_timeout < 0:
            raise ValueError("startup_timeout must be nonnegative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not policy_ref.strip():
            raise ValueError("policy_ref must not be empty")
        hub_id = policy_hub_id if policy_hub_id is not None else policy_ref
        if not hub_id.strip():
            raise ValueError("policy_hub_id must not be empty")
        return RemoteInferenceRun(
            self._client,
            robot=robot,
            policy_ref=policy_ref,
            policy_hub_id=hub_id,
            gpu=gpu,
            profile=profile,
            environment=environment,
            region=region,
            task=task,
            horizon=horizon,
            fps=fps,
            video_codec=video_codec,
            engine=engine,
            s_min=s_min,
            duration_s=duration_s,
            owner=owner,
            lease_timeout_s=lease_timeout_s,
            startup_timeout=startup_timeout,
            poll_interval=poll_interval,
            sleep_fn=sleep_fn,
            clock=clock,
        )
