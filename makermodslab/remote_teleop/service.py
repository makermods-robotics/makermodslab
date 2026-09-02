"""Simulation-only lifecycle used by the v1 API and dashboard status panel."""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Mapping
from typing import Any

from .contracts import SessionSpec
from .executor import JointLimit, RemoteExecutor
from .simulation import InMemoryRecorder, SimulatedFollower


class RemoteSimulationService:
    """Runs only deterministic doubles; there is no hardware adapter here."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor: RemoteExecutor | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._recorder: InMemoryRecorder | None = None

    def start(
        self,
        spec: SessionSpec,
        limits: Mapping[str, JointLimit],
        *,
        tick_hz: int = 50,
        watchdog_ms: int = 200,
    ) -> dict[str, Any]:
        with self._lock:
            if self._executor is not None and self._executor.authority.grant is not None:
                raise RuntimeError("a remote teleoperation simulation is already active")
            follower = SimulatedFollower(spec.joint_names)
            recorder = InMemoryRecorder()
            executor = RemoteExecutor(
                follower,
                limits,
                tick_hz=tick_hz,
                watchdog_ns=watchdog_ms * 1_000_000,
                first_action_deadline_ns=watchdog_ms * 1_000_000,
                mode="simulation",
                recorder=recorder,
            )
            grant = executor.open_session(spec)
            # A prior watchdog-stopped simulation thread retains its own stop
            # event and executor. It must never begin ticking this new session.
            self._stop.set()
            stop_event = threading.Event()
            self._executor = executor
            self._recorder = recorder
            self._stop = stop_event
            self._thread = threading.Thread(
                target=self._run,
                args=(executor, stop_event),
                name="remote-teleop-simulation",
                daemon=True,
            )
            self._thread.start()
            return {
                "simulation_only": True,
                "session": grant.public(),
                "credentials": {
                    "key_id": grant.key_id,
                    "action_key_base64": base64.b64encode(grant.action_key).decode("ascii"),
                },
                "status": executor.status(),
            }

    def _run(self, executor: RemoteExecutor, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if executor.authority.grant is None:
                return
            started = time.monotonic()
            try:
                executor.tick()
            except Exception:
                return
            period = 1.0 / executor.tick_hz
            stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    def submit(self, session_id: str, datagram_base64: str) -> dict[str, Any]:
        try:
            raw = base64.b64decode(datagram_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("datagram_base64 is invalid") from exc
        with self._lock:
            executor = self._require(session_id)
            executor.submit_datagram(raw)
            return executor.status()

    def stop(self, session_id: str, reason: str = "api_stop") -> dict[str, Any]:
        with self._lock:
            executor = self._require(session_id)
            transition = executor.stop(reason)
            self._stop.set()
            return {"transition": transition, "status": executor.status()}

    def _require(self, session_id: str) -> RemoteExecutor:
        executor = self._executor
        grant = executor.authority.grant if executor is not None else None
        if grant is None or grant.session_id != session_id:
            raise KeyError("remote teleoperation simulation session not found")
        return executor

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._executor is None:
                return {
                    "simulation_only": True,
                    "live_hardware_enabled": False,
                    "state": "idle",
                    "status": None,
                    "recorded_events": 0,
                }
            status = self._executor.status()
            return {
                "simulation_only": True,
                "live_hardware_enabled": False,
                "state": status["authority"]["state"],
                "status": status,
                "recorded_events": len(self._recorder.events) if self._recorder else 0,
            }

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            if executor is not None and executor.authority.grant is not None:
                executor.stop("server_shutdown")
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


remote_simulation_service = RemoteSimulationService()
