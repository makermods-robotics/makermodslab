"""Robot-side RTC scheduling primitives for the FULL DRTC port.

This is the offline-unit-testable core of ``robot_rtc.py``: the LWW
``ActionSchedule``, the Jacobson-Karels latency estimator, and single-source
prefix/span extraction. It has NO Portal / hardware / GPU dependency, so it is
exercised by ordinary unit tests: ``tests/test_drtc_schedule.py`` (which is
where this module's ``__main__`` self-test block went when it was ported into
the Lab).

It extends the half-DRTC scheduler from the earlier ``lerobot_drtc`` prototype
(``_scheduler.py`` there — a separate project, NOT a file in this repo) in
exactly one way:
the schedule now records enough provenance to reconstruct the RTC in-painting
*prefix descriptor* the policy server needs. Concretely, each scheduled action
remembers:

  * ``src_ts``      — the source observation's timestamp in µs
                      (``ActionChunk.in_reply_to_ts_us``). This is BOTH the LWW
                      freshness key (larger µs == fresher, since obs timestamps
                      are monotone) AND the key into the server's raw-chunk cache
                      (the upstream DRTC reference's
                      ``policy_server_drtc.ActionChunkCache`` keys by source
                      step; here we key by source obs timestamp).
  * ``chunk_index`` — the action's index ``j`` within its source chunk. The
                      server cached the RAW (pre-postprocess) chunk under
                      ``src_ts`` as a length-``H`` tensor; index ``j`` there is
                      the same ``j`` the robot received. So the prefix
                      ``raw_cache[src_ts][start : start+len]`` is recovered from
                      ``(src_ts, chunk_index_of_first, run_length)``.

The absolute-step axis is unchanged from the half-DRTC scheduler: a chunk that
answered the observation captured at control tick ``t_src`` describes actions for
steps ``[t_src + action_delay, t_src + action_delay + H)``; steps already in the
past (``< current_step``) are dropped on merge, which yields the chunk-alignment
offset for free. See the ``lerobot_drtc`` prototype's ``_scheduler.py`` for the
full rationale.

References mirrored (all of these live in OTHER codebases, not here):
  * upstream DRTC reference implementation,
    ``async_inference/robot_client_drtc.py`` (ActionSchedule.merge LWW,
    get_masking_chunk_spans — we take its *single* freshest span)
  * upstream DRTC reference implementation,
    ``async_inference/utils/latency_estimation.py`` (JK update eqns)
  * the installed lerobot package, ``lerobot/rollout/inference/rtc.py`` (how d
    and the prefix drive ``predict_action_chunk`` on the server; the client only
    computes them)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Scheduled action + merge stats
# ---------------------------------------------------------------------------
@dataclass
class ScheduledAction:
    """One action pinned to an absolute step, tagged with RTC provenance.

    ``src_ts`` (source obs timestamp, µs) is the LWW freshness key: a later
    observation (larger µs) wins on overlap. It is also the key into the policy
    server's raw-chunk cache. ``chunk_index`` is the action's index within that
    source chunk, so a contiguous run recovers a ``[start:start+len]`` slice.
    """

    action: dict[str, float]
    src_ts: int  # source obs timestamp (µs); freshness + cache key
    chunk_index: int  # index j of this action within its source raw chunk
    chunk_start_step: int  # absolute step where the source chunk starts (debug)


@dataclass
class MergeStats:
    """Discontinuity summary over steps where the incoming chunk overlapped
    actions from an *older* chunk. ``mean_l2`` near zero means the two chunks
    agreed; large values flag a seam the guided in-painting had to smooth
    (or, on a starved/hard switch, could not)."""

    inserted: int = 0
    overwritten: int = 0
    stale_dropped: int = 0
    overlap_count: int = 0
    mean_l2: float = 0.0
    max_l2: float = 0.0


@dataclass
class PrefixDescriptor:
    """Single-source in-painting prefix: slice ``[start : start+length]`` of the
    raw chunk the server cached under ``src_ts``. ``length == 0`` means no
    prefix (first inference / starved schedule)."""

    src_ts: int
    start: int
    length: int

    @property
    def is_empty(self) -> bool:
        return self.length <= 0


def _action_dict(chunk, j: int) -> dict[str, float]:
    """Extract chunk action at index ``j`` as ``{field: value}``. Works with a
    Portal ``ActionChunk`` (``.data`` maps field -> numpy array) or the fake
    chunk used in the self-test."""
    return {name: float(column[j]) for name, column in chunk.data.items()}


def _l2(a: dict[str, float], b: dict[str, float]) -> float:
    keys = a.keys() | b.keys()
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


# ---------------------------------------------------------------------------
# ActionSchedule: absolute step -> freshest action, with RTC provenance.
# ---------------------------------------------------------------------------
class ActionSchedule:
    """The action timeline ``ψ``: absolute step -> the freshest action for it.

    ``current_step`` is the robot's control tick, set by the caller each loop.
    Steps strictly before it are frozen (already executed) and can never be
    modified — the monotone property that makes out-of-order/duplicate chunks
    safe. Same contract as the ``lerobot_drtc`` prototype's ``_scheduler.py``
    (a separate project, not in this repo); the only additions
    are ``src_ts``/``chunk_index`` provenance (for prefix extraction) and
    :meth:`prefix_span`.
    """

    def __init__(self) -> None:
        self._sched: dict[int, ScheduledAction] = {}
        self.current_step: int = 0

    def pop_current(self) -> dict[str, float] | None:
        """Return (and remove) the action for ``current_step``, pruning anything
        older. Returns ``None`` on starvation; the caller should hold rather than
        advance ``current_step``."""
        stale = [s for s in self._sched if s < self.current_step]
        for s in stale:
            del self._sched[s]
        entry = self._sched.pop(self.current_step, None)
        return entry.action if entry is not None else None

    def remaining(self) -> int:
        """How many actions are queued at or beyond ``current_step`` (the runway
        the pacing logic watches)."""
        return sum(1 for s in self._sched if s >= self.current_step)

    def merge(self, chunk, src_ts: int, chunk_start_step: int) -> MergeStats:
        """Fold an incoming chunk into the schedule on the absolute-step axis.

        - step ``= chunk_start_step + j`` for chunk index ``j``.
        - steps ``< current_step`` are frozen -> dropped (alignment offset for
          free).
        - empty step -> insert.
        - occupied step -> overwrite only if this chunk's obs is strictly fresher
          (``src_ts`` larger). This is the LWW / CRDT join that absorbs
          duplicates and ignores reordered stale chunks.

        ``src_ts`` is the source observation timestamp (µs) — freshness key AND
        server raw-cache key. ``chunk_index`` is stored as ``j`` so a later
        prefix extraction can slice the cached raw chunk.
        """
        stats = MergeStats()
        l2_sum = 0.0
        for j in range(chunk.horizon):
            step = chunk_start_step + j
            if step < self.current_step:
                stats.stale_dropped += 1
                continue
            incoming = _action_dict(chunk, j)
            existing = self._sched.get(step)
            if existing is None:
                self._sched[step] = ScheduledAction(incoming, src_ts, j, chunk_start_step)
                stats.inserted += 1
                continue
            # Overlap: measure discontinuity for observability, then LWW.
            d = _l2(incoming, existing.action)
            stats.overlap_count += 1
            l2_sum += d
            stats.max_l2 = max(stats.max_l2, d)
            if src_ts > existing.src_ts:
                self._sched[step] = ScheduledAction(incoming, src_ts, j, chunk_start_step)
                stats.overwritten += 1
        if stats.overlap_count:
            stats.mean_l2 = l2_sum / stats.overlap_count
        return stats

    def prefix_span(self, current_step: int, max_len: int) -> PrefixDescriptor:
        """Extract the SINGLE-SOURCE in-painting prefix at/after ``current_step``.

        Walks the schedule forward from ``current_step`` and takes the maximal
        contiguous run that (a) is step-consecutive and (b) shares one
        ``src_ts`` with consecutive ``chunk_index`` — i.e. one unbroken slice of
        one source chunk. Length is capped at ``max_len`` (the caller passes
        ``overlap_end``; beyond it the RTC prefix weights are zero anyway, so
        there's nothing to guide toward).

        This is the single-source simplification of the multi-span
        ``get_masking_chunk_spans`` in the upstream DRTC reference
        implementation (``robot_client_drtc.py`` there): we keep only the
        freshest near-future span. Returns an empty descriptor
        (``length == 0``) when the schedule is dry or ``max_len <= 0``.
        """
        if max_len <= 0:
            return PrefixDescriptor(0, 0, 0)
        steps = sorted(s for s in self._sched if s >= current_step)
        if not steps:
            return PrefixDescriptor(0, 0, 0)

        first = self._sched[steps[0]]
        src_ts = first.src_ts
        start = first.chunk_index
        length = 0
        prev_step: int | None = None
        prev_idx: int | None = None
        for s in steps:
            e = self._sched[s]
            if e.src_ts != src_ts:
                break  # different source chunk -> single-source span ends
            if prev_step is not None and (s != prev_step + 1 or e.chunk_index != prev_idx + 1):
                break  # a gap in steps or indices -> span ends
            prev_step, prev_idx = s, e.chunk_index
            length += 1
            if length >= max_len:
                break
        return PrefixDescriptor(src_ts, start, length)
