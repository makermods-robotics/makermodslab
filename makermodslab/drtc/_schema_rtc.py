"""The EXTENDED wire contract shared by ``robot_rtc.py`` and ``policy_rtc.py``.

This is the full-DRTC (RTC in-painting) sibling of ``_schema.py``. It reuses
``_schema.py``'s canonical, index-based state/action layout and *appends* five
scalar RTC-control fields to the state schema so the robot can carry the
in-painting request metadata to the policy server over Portal's typed-state
channel.

Why piggy-back on the state channel?
------------------------------------
Portal's robot->operator path only carries typed STATE scalars + video. The
policy ``Operator`` receives synced ``Observation``s (state + frames) and replies
with ``act`` chunks. There is no side channel for request metadata, so the RTC
request rides as EXTRA STATE FIELDS appended after the base ``s0..s{D-1}``:

    rtc_d            F64  inference delay in *steps* (client JK estimator, d)
    rtc_overlap_end  F64  execution-horizon boundary = H - max(s_min, d)
    rtc_src_ts       F64  obs timestamp (µs) that the still-to-execute prefix
                          came from -> keys the server's raw-chunk cache.
                          0 = no prefix / first inference. µs epoch ~1.75e15
                          fits exactly in f64's 53-bit mantissa (< 2^53).
    rtc_prefix_start F64  start index within that cached raw chunk
    rtc_prefix_len   F64  number of prefix steps; 0 = no prefix

Portal fingerprints the state schema and *silently drops* packets whose schema
disagrees on both peers (same field names, order, dtype). So BOTH sides MUST
build the state schema through :func:`rtc_state_fields` below — never hand-roll
one side. The ``act`` action-chunk schema is unchanged from ``_schema.py``
(post-processed executable actions, horizon, ``a0..a{A-1}``): the raw in-painting
prefix never crosses the wire as actions — it is referenced by
``(rtc_src_ts, rtc_prefix_start, rtc_prefix_len)`` into the server's own raw
cache, exactly like the reference DRTC server keys its cache by source step
(``policy_server_drtc.ActionChunkCache``, in the upstream DRTC reference
implementation — not a module in this repo).

SIMPLIFICATION (documented, deliberate): single-source prefix.
------------------------------------------------------------
The reference DRTC client sends ``action_schedule_spans`` — a *list* of
``(src_step, start, end)`` slices, because after LWW merging the near-future
schedule can be stitched from several source chunks
(``robot_client_drtc.ActionSchedule.get_masking_chunk_spans``, likewise
upstream). Here we send a
SINGLE span: the maximal contiguous still-to-execute run drawn from ONE source
chunk (the newest one dominating the near future). This is the common case, it
keeps the wire compact (5 scalars vs. a variable-length list), and it is a
faithful first port. Multi-span is a future extension — it would need either a
repeated-field encoding or a small fixed cap of ``(src_ts, start, len)`` triples
appended to the state schema. See ``robot_rtc.py`` for where the single span is
extracted (``_rtc.ActionSchedule.prefix_span``).
"""

from __future__ import annotations

from livekit.portal import DType

# Re-export the base contract so callers can import everything RTC-related from
# one module and never accidentally mix a base schema on one side with an
# extended schema on the other.
from ._schema import (  # noqa: F401
    CHUNK_NAME,
    IMAGE_PREFIX,
    WireSchema,
    policy_wire_schema,
    robot_wire_schema,
)

# The five RTC control fields, in declaration order. Order is load-bearing:
# Portal's schema fingerprint is order-sensitive, so this list is the single
# source of truth for both peers.
RTC_STATE_FIELDS: list[tuple[str, DType]] = [
    ("rtc_d", DType.F64),  # inference delay d (steps)
    ("rtc_overlap_end", DType.F64),  # H - max(s_min, d): where the fresh region starts
    ("rtc_src_ts", DType.F64),  # source obs timestamp (µs) of the prefix; 0 = none
    ("rtc_prefix_start", DType.F64),  # start index within the cached raw chunk
    ("rtc_prefix_len", DType.F64),  # prefix length in steps; 0 = none
]

# Field-name constants so the two sides never disagree on a string literal.
RTC_D = "rtc_d"
RTC_OVERLAP_END = "rtc_overlap_end"
RTC_SRC_TS = "rtc_src_ts"
RTC_PREFIX_START = "rtc_prefix_start"
RTC_PREFIX_LEN = "rtc_prefix_len"


def rtc_state_fields(schema: WireSchema) -> list[tuple[str, DType]]:
    """The full state schema for the RTC example: ``s0..s{D-1}`` then the RTC
    control fields. Both ``robot_rtc.py`` and ``policy_rtc.py`` MUST declare
    state through this function so their fingerprints match.
    """
    return schema.state_fields() + RTC_STATE_FIELDS


def empty_rtc_meta() -> dict[str, float]:
    """The neutral RTC metadata for a first inference / no-prefix request.

    Sent verbatim on the very first observation (nothing to in-paint yet) and as
    a safe default. ``rtc_prefix_len == 0`` tells the server to run a plain
    ``predict_action_chunk`` with no guidance.
    """
    return {
        RTC_D: 0.0,
        RTC_OVERLAP_END: 0.0,
        RTC_SRC_TS: 0.0,
        RTC_PREFIX_START: 0.0,
        RTC_PREFIX_LEN: 0.0,
    }
