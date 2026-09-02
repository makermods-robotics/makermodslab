"""Robot-local liveness watchdogs independent of browsers and remote hosts."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class WatchdogDeadlines:
    action_ns: int = 200_000_000
    first_action_ns: int = 1_000_000_000
    control_ns: int = 1_000_000_000
    browser_ns: int = 2_000_000_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 20_000_000 <= value <= 10_000_000_000:
                raise ValueError(f"{name} must be in [20ms,10s]")


class RobotLivenessWatchdog:
    """Choose one local stop reason when any required liveness proof expires."""

    def __init__(
        self,
        stop: Callable[[str], object],
        *,
        deadlines: WatchdogDeadlines = WatchdogDeadlines(),
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.stop = stop
        self.deadlines = deadlines
        self.clock_ns = clock_ns
        self._lock = threading.RLock()
        self._session_started_ns: int | None = None
        self._last_action_ns: int | None = None
        self._last_control_ns: int | None = None
        self._last_browser_ns: int | None = None
        self._tripped_reason: str | None = None

    def arm(self) -> None:
        now = self.clock_ns()
        with self._lock:
            self._session_started_ns = now
            self._last_action_ns = None
            self._last_control_ns = now
            self._last_browser_ns = now
            self._tripped_reason = None

    def disarm(self) -> None:
        with self._lock:
            self._session_started_ns = None
            self._last_action_ns = None
            self._last_control_ns = None
            self._last_browser_ns = None

    def mark_action(self, *, now_ns: int | None = None) -> None:
        with self._lock:
            if self._session_started_ns is not None:
                self._last_action_ns = self.clock_ns() if now_ns is None else now_ns

    def mark_control(
        self,
        *,
        operator_process_live: bool = True,
        browser_live: bool = True,
        now_ns: int | None = None,
    ) -> str | None:
        now = self.clock_ns() if now_ns is None else now_ns
        with self._lock:
            if self._session_started_ns is None:
                return None
            self._last_control_ns = now
            if browser_live:
                self._last_browser_ns = now
        if not operator_process_live:
            return self.trip("operator_process_lost")
        if not browser_live:
            return self.trip("operator_browser_lost")
        return None

    def poll(self, *, now_ns: int | None = None) -> str | None:
        now = self.clock_ns() if now_ns is None else now_ns
        with self._lock:
            started = self._session_started_ns
            if started is None or self._tripped_reason is not None:
                return self._tripped_reason
            if self._last_control_ns is None or now - self._last_control_ns >= self.deadlines.control_ns:
                reason = "control_heartbeat_timeout"
            elif self._last_browser_ns is None or now - self._last_browser_ns >= self.deadlines.browser_ns:
                reason = "operator_browser_timeout"
            elif self._last_action_ns is None:
                reason = "first_action_timeout" if now - started >= self.deadlines.first_action_ns else None
            else:
                reason = (
                    "action_watchdog_timeout"
                    if now - self._last_action_ns >= self.deadlines.action_ns
                    else None
                )
        return self.trip(reason) if reason is not None else None

    def trip(self, reason: str) -> str:
        with self._lock:
            if self._tripped_reason is not None:
                return self._tripped_reason
            self._tripped_reason = reason
        # Hardware teardown is intentionally outside the watchdog lock.
        self.stop(reason)
        return reason

    def status(self, *, now_ns: int | None = None) -> dict[str, object]:
        now = self.clock_ns() if now_ns is None else now_ns
        with self._lock:
            started = self._session_started_ns
            if started is None:
                return {"armed": False, "tripped_reason": self._tripped_reason}

            def remaining(last: int | None, budget: int) -> float | None:
                if last is None:
                    return None
                return max(0.0, (budget - (now - last)) / 1_000_000)

            first_reference = self._last_action_ns if self._last_action_ns is not None else started
            first_budget = (
                self.deadlines.action_ns
                if self._last_action_ns is not None
                else self.deadlines.first_action_ns
            )
            return {
                "armed": True,
                "tripped_reason": self._tripped_reason,
                "action_remaining_ms": remaining(first_reference, first_budget),
                "control_remaining_ms": remaining(self._last_control_ns, self.deadlines.control_ns),
                "browser_remaining_ms": remaining(self._last_browser_ns, self.deadlines.browser_ns),
            }


class WatchdogRunner:
    """Explicitly started polling thread; construction never starts background work."""

    def __init__(self, watchdog: RobotLivenessWatchdog, *, poll_interval_s: float = 0.02) -> None:
        if not 0.005 <= poll_interval_s <= 0.25:
            raise ValueError("watchdog poll interval must be in [5ms,250ms]")
        self.watchdog = watchdog
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("watchdog runner is already active")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="remote-teleop-watchdog", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            self.watchdog.poll()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
