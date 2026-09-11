"""Limit paired camera/state sends without changing the action clock."""

import math


class ObservationRateLimiter:
    def __init__(self, hz: float = 0):
        if not math.isfinite(hz) or hz < 0:
            raise ValueError("Observation rate must be finite and nonnegative")
        self.interval = 1 / hz if hz else 0
        self.last_sent: float | None = None

    def ready(self, now: float) -> bool:
        return self.last_sent is None or now - self.last_sent >= self.interval - 1e-9

    def sent(self, now: float) -> None:
        # Anchor to the actual send, so a stall never creates a catch-up burst.
        self.last_sent = now
