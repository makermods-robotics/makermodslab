"""Jacobson-Karels round-trip latency estimation — the one canonical copy.

Both chunk-execution regimes need the same latency estimate, so it lives here
rather than being duplicated in each:

  * ``_rtc.py`` / ``robot_rtc.py``     — full DRTC with RTC in-painting, where
    ``estimate_steps`` is the in-painting prefix length ``d``.
  * ``_sync_player.py`` / ``robot_sync.py`` — adaptive-sync, where
    ``estimate_steps`` is the latency-driven component of the prefetch lead.

This module is deliberately PURE STANDARD LIBRARY (``math`` only). Keep it that
way: it is earmarked to move into another project, and any numpy / lerobot /
livekit import would break that portability.
"""

from __future__ import annotations

import math


class JKLatencyEstimator:
    """Jacobson–Karels round-trip estimator (the same one TCP uses).

    Exponentially smooths measured policy round-trip latency and adds a scaled
    mean-deviation margin, so a single spike decays out instead of pinning the
    estimate high for a fixed window — i.e. it recovers fast after a spike
    rather than staying pessimistic for N rounds. Mirrors ``JKLatencyEstimator``
    in the upstream DRTC reference implementation
    (``async_inference/utils/latency_estimation.py`` there — another codebase,
    not a file in this repo): same update eqns + ``estimate_steps`` clamp.
    """

    def __init__(
        self,
        fps: int,
        action_chunk_size: int,
        s_min: int = 4,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.5,
    ) -> None:
        self.fps = fps
        self.action_chunk_size = action_chunk_size
        self.s_min = s_min
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.smoothed_rtt = 0.0
        self.rtt_deviation = 0.0
        self._initialized = False

    def update(self, measured_latency_s: float) -> None:
        if not self._initialized:
            self.smoothed_rtt = measured_latency_s
            self.rtt_deviation = 0.0
            self._initialized = True
            return
        error = measured_latency_s - self.smoothed_rtt
        self.smoothed_rtt = (1 - self.alpha) * self.smoothed_rtt + self.alpha * measured_latency_s
        self.rtt_deviation = (1 - self.beta) * self.rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        if not self._initialized:
            return self.s_min / self.fps
        return self.smoothed_rtt + self.k * self.rtt_deviation

    @property
    def estimate_steps(self) -> int:
        """Latency in whole action steps, clamped to ``[1, H // 2]``.

        The ``H // 2`` ceiling originates as the RTC constraint that the
        frozen/overlap prefix can never exceed half the chunk — beyond that a
        chunk is too stale to be worth stitching (``d <= H - s`` with ``s = d``
        gives ``d <= H/2``).

        Both regimes inherit that ceiling, for related but distinct reasons:

          * DRTC (``_rtc.py``): ``estimate_steps`` IS the in-painting prefix
            length ``d``, so the clamp is the constraint itself.
          * Adaptive-sync (``_sync_player.py``): the same ceiling caps how early
            we prefetch off the estimate alone; past it the ``margin``
            (``base_lead``) and the degrade clause take over, so a latency
            estimate beyond half a chunk cannot on its own drive the lead
            further out.
        """
        raw = max(1, math.ceil(self.estimate_seconds * self.fps))
        d_max = max(1, self.action_chunk_size // 2)
        return min(raw, d_max)
