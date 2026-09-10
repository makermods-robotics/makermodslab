"""Adaptive-sync block player for remote robot inference over Portal.

This module implements a NEW chunk-execution regime — *adaptive-sync* — and a
headless simulation that quantitatively tests whether it helps a
**non-flow-matching / non-inpainting** policy (ACT, diffusion, or any policy you
do not want to guide with dynamically-consistent denoising) survive network
latency.

Where it sits between the two existing regimes
----------------------------------------------
(Both regimes below live in EARLIER PROTOTYPE PROJECTS — ``lerobot_inference``
and ``lerobot_drtc`` — neither of which is a directory in this repo.)

* ``lerobot_inference``'s ``robot.py`` — ``BlockChunkPlayer``. Plays each chunk to
  completion (ONE seam per chunk boundary), staging the next block and requesting
  a fresh one when the runway drops to a *fixed* ``prefetch_lead``. Smooth for
  non-inpainting policies, but the fixed lead is either too small (starves at
  high latency) or too big (needless prefetch/overlap at low latency).

* ``lerobot_drtc``'s ``_scheduler.py`` — ``ActionSchedule``. Re-plans as fast as
  latency allows and merges overlapping chunks last-write-wins on an absolute
  step axis. That is correct *only* if the policy in-paints the seams (the
  flow/diffusion guided-denoising half); for a policy WITHOUT inpainting every
  re-plan is a hard switch, so faster re-planning multiplies hard seams.

Adaptive-sync keeps ``BlockChunkPlayer``'s play-to-completion property — it does
NOT re-plan mid-chunk and does NOT overwrite the executing chunk, so there is
still exactly ONE seam per chunk boundary — but it makes the prefetch lead
ADAPTIVE to measured round-trip latency via a Jacobson–Karels estimator:

    lead = estimator.estimate_steps + margin            # clamped to [1, H-1]

  * LOW latency  -> ``estimate_steps`` small -> lead small -> request near the
    END of the chunk -> minimal prefetch, near-zero overlap, smoothest.
  * HIGH latency -> ``estimate_steps`` grows -> lead grows -> request EARLIER so
    the next chunk lands before the current one drains -> avoids starvation.
  * If latency exceeds what one chunk can cover (``lead >= H - s_min``) the
    player degrades gracefully to strict play-to-drain: it requests at the
    boundary and, if the next chunk has not arrived, HOLDS the last action
    (freeze) rather than jumping to a stale plan.

The slogan: adaptive-sync changes *WHEN* the robot asks, never *HOW OFTEN* it
overwrites the plan. It moves the single seam earlier or later; it never adds a
second seam. That is what makes it safe for policies that were never trained to
stitch overlapping plans.

The class is transport-agnostic and unit-testable (``push`` / ``step`` /
``should_request`` / ``remaining`` / ``record_roundtrip``); ``robot_sync.py``
drives it over Portal, and the ``__main__`` block below drives it against a
synthetic network + ground-truth trajectory to measure the payoff.
"""

from __future__ import annotations

import time

from ._latency import JKLatencyEstimator


class AdaptiveBlockPlayer:
    """Open-loop block execution with an ADAPTIVE, latency-driven prefetch lead.

    Same play-to-completion contract as ``BlockChunkPlayer``:

      * ``step`` plays the current chunk one action per control tick, in order,
        to the end — never a per-tick cursor reset, never a mid-chunk re-plan;
      * an incoming chunk is *staged* (``push``) and swapped in only when the
        current block drains, so an early (prefetched) arrival can never
        truncate the block being played — the property that keeps a
        non-inpainting policy in-distribution;
      * ``should_request`` fires at most once per block.

    The one difference: the runway threshold that triggers ``should_request`` is
    ``lead = estimator.estimate_steps + margin`` (clamped to ``[1, H-1]``)
    instead of a constant. See the module docstring for the regime rationale.

    Robustness to drops: ``should_request`` sets ``_pending`` and will not
    re-ask until the chunk lands (``push`` clears it) OR the request has been
    outstanding longer than a latency-scaled deadline — otherwise a single
    silently-dropped chunk would deadlock the loop into permanent starvation.
    """

    def __init__(
        self,
        horizon: int,
        fps: int = 30,
        *,
        adaptive: bool = True,
        margin: int = 2,
        s_min: int = 4,
        align: bool = True,
        action_delay: int = 1,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.5,
        estimator: JKLatencyEstimator | None = None,
    ) -> None:
        self.horizon = horizon
        self.fps = fps
        self.adaptive = adaptive
        self.margin = max(0, margin)
        self.s_min = s_min
        self.align = align
        self.action_delay = action_delay
        # Absolute control tick, written by the caller each loop; the alignment
        # on swap reads it to know how much of a fresh chunk is already in the past.
        self.current_step = 0
        self.estimator = estimator or JKLatencyEstimator(
            fps, horizon, s_min=s_min, alpha=alpha, beta=beta, k=k
        )

        self._chunk = None  # block currently playing
        self._next = None  # prefetched block, staged
        self._chunk_src = None  # src control tick of the playing block
        self._next_src = None  # src control tick of the staged block
        self._cursor = 0
        self._pending = False  # an obs is in flight, awaiting its chunk
        self._pending_age = 0  # ticks since the outstanding request went out
        self._received_at: float | None = None

        # Observability (read by the sim harness / robot logging).
        self.boundary_swap = False  # True on the tick a real chunk->chunk swap happened
        self.holds = 0  # cumulative frozen (starvation) ticks

    # -- lead policy -------------------------------------------------------
    def current_lead(self) -> int:
        """Runway (in actions) at which we request the next block.

        Adaptive: ``estimate_steps + margin``. Fixed (``--adaptive false``): the
        constant ``margin`` (i.e. the baseline ``prefetch_lead``). Always clamped
        to ``[1, H-1]`` — ``H-1`` is the degrade limit "request one tick after the
        boundary", ``1`` never lets it collapse to zero prefetch.
        """
        raw = (self.estimator.estimate_steps + self.margin) if self.adaptive else self.margin
        return max(1, min(int(raw), self.horizon - 1))

    @property
    def in_degrade(self) -> bool:
        """True when the estimated round-trip no longer fits inside one chunk's
        executable budget (``lead >= H - s_min``): prefetch alone cannot hide the
        latency, so we are effectively play-to-drain-and-hold."""
        raw = (self.estimator.estimate_steps + self.margin) if self.adaptive else self.margin
        return raw >= (self.horizon - self.s_min)

    # -- core player -------------------------------------------------------
    def push(self, chunk, src_step: int | None = None) -> None:
        # Stage it; step() swaps it in when the current block drains. Staging
        # (rather than adopting immediately) is what keeps an early arrival from
        # cutting the current block short. ``src_step`` is the control tick of the
        # observation this chunk answered — used to align the swap to NOW.
        self._next = chunk
        self._next_src = src_step
        self._pending = False
        self._pending_age = 0
        self._received_at = time.monotonic()

    def record_roundtrip(self, latency_s: float) -> None:
        """Feed one measured obs->chunk round-trip (seconds) to the estimator."""
        self.estimator.update(latency_s)

    def step(self) -> dict[str, float] | None:
        self.boundary_swap = False
        if self._pending:
            self._pending_age += 1
        chunk_h = 0 if self._chunk is None else self._chunk.horizon
        if self._chunk is None or self._cursor >= chunk_h:
            if self._next is None:
                self.holds += 1
                return None  # dry schedule -> caller holds the last action (freeze)
            # Swap the staged block in. Only a swap *from* a real, drained block
            # is a chunk-boundary seam; the very first fill (from None) is not.
            self.boundary_swap = self._chunk is not None
            self._chunk = self._next
            self._chunk_src = self._next_src
            self._next = None
            self._next_src = None
            # Alignment: resume the fresh chunk at the index that lines up with
            # NOW, dropping the prefix already in the past by the time the old
            # block drained. Without it the swap starts at index 0 — an action
            # planned lead+latency ticks ago — so the arm snaps BACKWARD by that
            # many steps at every boundary (the "twitch back" you saw). This is
            # DRTC's absolute-step alignment applied at block granularity: it does
            # NOT merge/re-plan mid-block, so there is still exactly ONE seam per
            # boundary — but it is now forward-consistent, not a rewind.
            if self.align and self._chunk_src is not None:
                aligned = self.current_step - (self._chunk_src + self.action_delay)
                self._cursor = min(max(aligned, 0), self._chunk.horizon - 1)
            else:
                self._cursor = 0
        action = {name: float(column[self._cursor]) for name, column in self._chunk.data.items()}
        self._cursor += 1
        return action

    def remaining(self) -> int:
        cur = 0 if self._chunk is None else max(0, self._chunk.horizon - self._cursor)
        nxt = self._next.horizon if self._next is not None else 0
        return cur + nxt

    def should_request(self) -> bool:
        lead = self.current_lead()
        if self._pending:
            # Give up on a request that has been outstanding longer than the
            # runway it was meant to cover plus a small slack (its chunk was
            # almost certainly dropped), so a drop can't wedge us into permanent
            # starvation. Tied to ``lead`` (which already tracks latency) so the
            # retry is prompt at low RTT and patient at high RTT; a late-but-not-
            # dropped chunk simply re-stages harmlessly.
            deadline = lead + self.s_min
            if self._pending_age > deadline:
                self._pending = False
            else:
                return False
        if self._next is not None:
            return False
        if self.remaining() <= lead:
            self._pending = True
            self._pending_age = 0
            return True
        return False

    # -- observability -----------------------------------------------------
    @property
    def current_chunk(self):
        return self._chunk

    def age_ms(self) -> float | None:
        if self._received_at is None:
            return None
        return (time.monotonic() - self._received_at) * 1000.0


# ===========================================================================
# Headless simulation:  `python -m makermodslab.drtc._sync_player`  (numpy only;
# no torch, no hardware, no network).  Deterministic (seeded) so results are
# reproducible.
# ===========================================================================
if __name__ == "__main__":
    import numpy as np

    FPS = 30
    H = 16  # action-chunk horizon
    A = 6  # action dimension (6-DoF arm)
    N_TICKS = 1500  # ~50 s of control at 30 fps
    BASE_LEAD = 2  # margin (adaptive) == fixed prefetch_lead (baseline)
    S_MIN = 4

    # --- synthetic GROUND-TRUTH continuous trajectory ---------------------
    # Smooth, slow per-joint sinusoids. Deterministic constants (NOT drawn per
    # run) so the adaptive and fixed players are graded on the identical target.
    _FREQ = np.array([0.11, 0.17, 0.23, 0.13, 0.19, 0.29])  # Hz, all sub-second
    _AMP = np.array([1.0, 0.8, 1.2, 0.6, 0.9, 0.5])
    _PHASE = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])

    def gt(t: float) -> np.ndarray:
        """Ground-truth action vector the policy 'wants' at absolute step t."""
        return _AMP * np.sin(2.0 * np.pi * _FREQ * (t / FPS) + _PHASE)

    class SimChunk:
        """A chunk as the (perfect) policy would return it: ground truth sampled
        for the H steps starting at the observation tick it answered."""

        __slots__ = ("horizon", "data", "src_step", "latency_s")

        def __init__(self, src_step: int, horizon: int):
            self.horizon = horizon
            self.src_step = src_step
            self.data = {
                f"a{i}": np.array([gt(src_step + j)[i] for j in range(horizon)], dtype=float)
                for i in range(A)
            }
            self.latency_s = 0.0

    def _p95(x):
        return float(np.percentile(x, 95)) if len(x) else 0.0

    def simulate(mean_rtt_ms, jitter_ms, drop_prob, adaptive, seed):
        rng = np.random.default_rng(seed)
        player = AdaptiveBlockPlayer(H, FPS, adaptive=adaptive, margin=BASE_LEAD, s_min=S_MIN)
        inflight = []  # list of (deliver_tick, SimChunk)

        starvation = 0
        requests = 0
        seams = []  # L2 discontinuity at each chunk-boundary swap
        staleness = []  # executed-action age (ticks) vs its obs-time
        last_real = None  # last executed non-frozen action vector

        for w in range(N_TICKS):
            # 1. Deliver any chunks whose network delay elapsed this tick.
            due = [c for (d, c) in inflight if d == w]
            inflight = [(d, c) for (d, c) in inflight if d != w]
            for chunk in due:
                player.record_roundtrip(chunk.latency_s)
                player.push(chunk)

            # 2. Execute one action (or freeze on a dry schedule).
            action = player.step()
            if action is None:
                starvation += 1
                # the arm holds its last action (freeze): no fresh staleness or
                # seam sample is recorded for this tick.
            else:
                vec = np.array([action[f"a{i}"] for i in range(A)])
                staleness.append(w - player.current_chunk.src_step)
                if player.boundary_swap and last_real is not None:
                    seams.append(float(np.linalg.norm(vec - last_real)))
                last_real = vec

            # 3. Request the next block only when the (adaptive) runway threshold
            #    is crossed — one inference request per block.
            if player.should_request():
                requests += 1
                chunk = SimChunk(src_step=w, horizon=H)
                lat_ms = max(1.0, mean_rtt_ms + rng.normal(0.0, jitter_ms))
                lat_ticks = max(1, int(round(lat_ms / 1000.0 * FPS)))
                if rng.random() >= drop_prob:  # otherwise silently dropped
                    chunk.latency_s = lat_ticks / FPS
                    inflight.append((w + lat_ticks, chunk))

        return {
            "starv": starvation,
            "seams": len(seams),
            "seam_mean": float(np.mean(seams)) if seams else 0.0,
            "seam_max": float(np.max(seams)) if seams else 0.0,
            "stale_mean": float(np.mean(staleness)) if staleness else 0.0,
            "stale_p95": _p95(staleness),
            "reqs": requests,
        }

    # --- latency sweep ----------------------------------------------------
    RTTS = [30, 60, 120, 250, 400]  # mean round-trip, ms
    DROPS = [0.0, 0.05]
    np.random.seed(0)  # global determinism for good measure

    print("=" * 96)
    print(
        "adaptive-sync vs fixed-lead block execution  |  fps=30 H=16 A=6  "
        f"ticks={N_TICKS} base_lead={BASE_LEAD}"
    )
    print(
        "ground truth: smooth per-joint sinusoids; policy returns gt sampled from the obs tick; block player"
    )
    print("plays each chunk to completion (one seam / boundary) and freezes on a dry schedule.")
    print("=" * 96)
    header = (
        f"{'RTT':>5} {'drop':>5} {'player':>8} | {'starv':>6} {'seams':>5} "
        f"{'seamL2 mean':>11} {'seamL2 max':>10} {'stale mean':>10} "
        f"{'stale p95':>9} {'reqs':>5}"
    )
    print(header)
    print("-" * len(header))

    seed = 1234
    for rtt in RTTS:
        jitter = max(5.0, 0.25 * rtt)  # jitter scales with RTT (min 5 ms)
        for drop in DROPS:
            rows = {}
            for adaptive in (True, False):
                rows[adaptive] = simulate(rtt, jitter, drop, adaptive, seed)
                seed += 1
            for adaptive in (True, False):
                r = rows[adaptive]
                tag = "adaptive" if adaptive else "fixed"
                print(
                    f"{rtt:>5} {drop * 100:>4.0f}% {tag:>8} | {r['starv']:>6} "
                    f"{r['seams']:>5} {r['seam_mean']:>11.4f} {r['seam_max']:>10.4f} "
                    f"{r['stale_mean']:>10.2f} {r['stale_p95']:>9.1f} {r['reqs']:>5}"
                )
            # per-setting delta line (starvation reduction is the headline)
            a, f = rows[True], rows[False]
            starv_cut = (1 - a["starv"] / f["starv"]) * 100 if f["starv"] else 0.0
            print(
                f"{'':>5} {'':>5} {'Δ':>8} | starvation {a['starv']} vs {f['starv']} "
                f"({starv_cut:+.0f}%),  max-seam {a['seam_max']:.3f} vs {f['seam_max']:.3f},  "
                f"stale-p95 {a['stale_p95']:.0f} vs {f['stale_p95']:.0f} ticks"
            )
            print("-" * len(header))

    print("\nReading: 'starv' = frozen ticks (schedule dry, arm held). 'seams' = chunk-boundary swaps;")
    print("seamL2 = L2 jump between the last action of one chunk and the first of the next. 'stale' =")
    print("executed-action age (ticks) vs the obs it came from. Lower is better on every column except")
    print("that some staleness is the unavoidable cost of prefetching earlier under latency.")
