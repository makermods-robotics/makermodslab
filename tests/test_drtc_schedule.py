"""Pure-helper tests for the DRTC action schedule and latency estimator.

Ported from the `livekit-drtc` repo, where these ran as a `__main__` self-test
block inside `_rtc.py` (`python3 _rtc.py` -> "20/20 pass"). The assertions and
the reasoning comments are carried over unchanged; only the harness differs —
a vendored module should not ship its own test runner when the repo has pytest.

These cover the offline core of `robot_rtc.py`: absolute-step alignment,
last-write-wins merging keyed by source-observation timestamp, single-source
prefix extraction (what the RTC in-painting server needs), and the
Jacobson-Karels latency estimator's convergence and clamping. No Portal, no
hardware, no GPU — which is exactly why they are worth having.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from makermodslab.drtc._latency import JKLatencyEstimator
from makermodslab.drtc._rtc import ActionSchedule

H = 16


@dataclass
class _FakeChunk:
    """Stand-in for a Portal ActionChunk: `.horizon` + `.data` columns."""

    horizon: int
    data: dict[str, list[float]]


def make_chunk(horizon: int, base: float) -> _FakeChunk:
    # a0 ramps from `base`; distinct per source so overlaps have nonzero L2.
    return _FakeChunk(horizon=horizon, data={"a0": [float(i) + base for i in range(horizon)]})


# --- (a) alignment ---------------------------------------------------------
def test_merge_drops_the_stale_prefix_of_a_late_chunk():
    """A chunk answering an old observation resumes at the CURRENT step."""
    sched = ActionSchedule()
    sched.current_step = 5  # 5 ticks elapsed since the obs at src control step 0
    stats = sched.merge(make_chunk(H, 0.0), src_ts=1000, chunk_start_step=1)

    # steps 1..4 (< 5) dropped; steps 5..16 kept.
    assert stats.stale_dropped == 4
    assert sched.remaining() == H - 4

    first = sched.pop_current()
    assert first is not None
    assert first["a0"] == 4.0  # resumed at chunk index 4 -> step 5


# --- (b) last-write-wins, keyed by source obs timestamp --------------------
def test_a_staler_chunk_never_overwrites_a_fresher_one():
    sched = ActionSchedule()
    sched.current_step = 0
    sched.merge(make_chunk(H, 100.0), src_ts=2000, chunk_start_step=0)  # fresh

    stats = sched.merge(make_chunk(H, 0.0), src_ts=1000, chunk_start_step=0)  # staler ts

    assert stats.overwritten == 0
    assert stats.overlap_count == H
    assert sched._sched[0].action["a0"] == 100.0


def test_a_fresher_chunk_overwrites_the_whole_overlap():
    sched = ActionSchedule()
    sched.current_step = 0
    sched.merge(make_chunk(H, 100.0), src_ts=2000, chunk_start_step=0)

    stats = sched.merge(make_chunk(H, 200.0), src_ts=3000, chunk_start_step=0)  # fresher ts

    assert stats.overwritten == H
    assert sched._sched[0].action["a0"] == 200.0


# --- (c) prefix span: what the in-painting server is handed ----------------
def test_prefix_span_reports_source_start_index_and_length():
    sched = ActionSchedule()
    sched.current_step = 3
    # obs at control tick t_src=2, action_delay=1 -> chunk_start_step=3, so
    # step s maps to chunk index j = s - 3.
    sched.merge(make_chunk(H, 0.0), src_ts=5000, chunk_start_step=3)

    pd = sched.prefix_span(current_step=3, max_len=6)

    assert pd.src_ts == 5000
    assert pd.start == 0  # step 3 -> chunk index 0
    assert pd.length == 6  # capped at max_len


def test_prefix_span_starts_mid_chunk_once_the_run_has_advanced():
    sched = ActionSchedule()
    sched.current_step = 3
    sched.merge(make_chunk(H, 0.0), src_ts=5000, chunk_start_step=3)

    pd = sched.prefix_span(current_step=7, max_len=100)

    # step 7 -> chunk index 4; the horizon caps the run at chunk index 15.
    assert pd.start == 4
    assert pd.length == H - 4


def test_prefix_span_stops_at_a_source_boundary():
    """The single-source guarantee: a span never straddles two chunks."""
    sched = ActionSchedule()
    sched.current_step = 0
    sched.merge(make_chunk(H, 0.0), src_ts=1000, chunk_start_step=0)  # old source
    # Fresher but SHORTER overlay: only steps 0..3 get overwritten.
    short = _FakeChunk(horizon=4, data={"a0": [900.0 + i for i in range(4)]})
    sched.merge(short, src_ts=2000, chunk_start_step=0)  # fresh source

    pd = sched.prefix_span(current_step=0, max_len=100)

    # steps 0..3 belong to src_ts=2000; step 4 belongs to src_ts=1000, so the
    # span stops at the provenance change rather than reporting a mixed prefix.
    assert pd.src_ts == 2000
    assert pd.length == 4


@pytest.mark.parametrize(
    ("max_len", "primed"),
    [(0, True), (8, False)],
    ids=["max_len_zero", "dry_schedule"],
)
def test_prefix_span_is_empty_when_there_is_nothing_to_report(max_len, primed):
    sched = ActionSchedule()
    if primed:
        sched.current_step = 0
        sched.merge(make_chunk(H, 0.0), src_ts=1000, chunk_start_step=0)

    assert sched.prefix_span(current_step=0, max_len=max_len).is_empty


# --- (d) Jacobson-Karels latency estimator ---------------------------------
def test_estimator_falls_back_to_s_min_before_its_first_sample():
    est = JKLatencyEstimator(fps=30, action_chunk_size=H, s_min=4)

    assert est.estimate_seconds == 4 / 30


def test_estimator_converges_on_a_steady_round_trip():
    est = JKLatencyEstimator(fps=30, action_chunk_size=H, s_min=4)
    for _ in range(50):
        est.update(0.100)  # 100 ms round trip

    assert abs(est.estimate_seconds - 0.100) < 0.01
    assert est.estimate_steps == 3  # 100 ms @ 30 fps


def test_estimator_clamps_a_hopeless_round_trip_to_half_the_horizon():
    """5 s would be 150 steps; the clamp is what keeps the schedule sane."""
    est = JKLatencyEstimator(fps=30, action_chunk_size=H, s_min=4)
    for _ in range(50):
        est.update(5.0)

    assert est.estimate_steps == H // 2
