"""Policy execution duration, independent of connection and first-action setup."""

from __future__ import annotations

import time
from collections.abc import Callable


class RunTimer:
    def __init__(self, duration_s: float, *, clock: Callable[[], float] = time.monotonic):
        self.duration_s = duration_s
        self._clock = clock
        self._started_at: float | None = None

    def start(self) -> bool:
        """Start once, after easing; return whether this starts a new run."""
        if self._started_at is not None:
            return False
        self._started_at = self._clock()
        return True

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._started_at is None else max(0.0, self._clock() - self._started_at)

    @property
    def expired(self) -> bool:
        return self.duration_s > 0 and self.elapsed_s >= self.duration_s
