"""The line protocol `makermodslab.drtc_protocol` — both directions.

Pure-helper test: no LiveKit, no hardware, no subprocess. `drtc_protocol` is
deliberately free of heavy imports (it is the module the FastAPI-side parent
and the `livekit.portal`-importing child BOTH depend on), so this runs in
ordinary CI with the `drtc` extra absent.

What it pins, and why each matters:

- `format_event`/`parse_event` round-trip, INCLUDING the "match the prefix
  anywhere in the line" rule — a log record flushed without its newline would
  otherwise swallow the event behind it on the merged pipe.
- Compact JSON survives `format_event`'s whitespace collapse. That collapse is
  the reason STATS must be serialized with `separators=(",", ":")`; a
  pretty-printed payload would still parse but would no longer round-trip, and
  the failure would only show up under a payload that happened to be long.
- Every STATS key is always present. The parent's response model materializes
  declared-but-absent optionals as `null`, so an exact model is only honest if
  the wire really always carries the whole set.
- The two-STOP rule, which is the difference between a graceful return and a
  cut-short one on an energized arm.
"""

from __future__ import annotations

import json
import threading

import pytest

from makermodslab import drtc_protocol as proto

# --- format_event / parse_event ---------------------------------------------


@pytest.mark.parametrize(
    ("event", "payload"),
    [
        (proto.EVENT_READY, "url=wss://x.livekit.cloud room=portal-lerobot-inference"),
        (proto.EVENT_EASING, ""),
        (proto.EVENT_CONNECTED, ""),
        (proto.EVENT_ACTIVE, "operator=policy"),
        (proto.EVENT_RETURNING, ""),
        (proto.EVENT_BYE, ""),
    ],
)
def test_format_and_parse_round_trip(event, payload):
    assert proto.parse_event(proto.format_event(event, payload)) == (event, payload)


def test_the_prefix_is_matched_anywhere_in_the_line():
    """A log record flushed without its newline must not swallow the event."""
    line = "INFO 2026-09-02 rollout: connected" + proto.format_event(proto.EVENT_CONNECTED)

    assert proto.parse_event(line) == (proto.EVENT_CONNECTED, "")


def test_a_multiline_payload_is_collapsed_onto_one_line():
    message = "Traceback (most recent call last):\n  File 'x.py'\nRuntimeError: boom"

    line = proto.format_event(proto.EVENT_ERROR, message)

    assert "\n" not in line
    event, payload = proto.parse_event(line)
    assert event == proto.EVENT_ERROR
    assert payload == "Traceback (most recent call last): File 'x.py' RuntimeError: boom"


@pytest.mark.parametrize(
    "line",
    [
        "",
        "INFO nothing to see here",
        "MAKERMODSLAB-EVAL READY",  # the OTHER protocol; must not cross-parse
        proto.EVENT_PREFIX,  # prefix with no event word
        f"{proto.EVENT_PREFIX}   ",
    ],
)
def test_non_protocol_lines_parse_as_none(line):
    assert proto.parse_event(line) is None


# --- READY / ACTIVE key=value payloads --------------------------------------


def test_ready_reports_the_effective_transport():
    line = proto.format_event(proto.EVENT_READY, proto.format_ready("ws://127.0.0.1:7880", "lab"))

    _event, payload = proto.parse_event(line)
    assert proto.parse_kv(payload) == {"url": "ws://127.0.0.1:7880", "room": "lab"}


def test_parse_kv_skips_tokens_without_a_value_rather_than_guessing():
    assert proto.parse_kv("url=wss://x room") == {"url": "wss://x"}


def test_parse_kv_tolerates_an_empty_value():
    """An unset room must read as empty, not as absent — the parent errors on it."""
    assert proto.parse_kv("url=wss://x room=") == {"url": "wss://x", "room": ""}


# --- STATS ------------------------------------------------------------------


def _sample() -> dict[str, object]:
    return {
        "t": 1,
        "chunks": 3,
        "reqs": 4,
        "sched": 6,
        "lead": 10,
        "s_min": 4,
        "horizon": 16,
        "lat_steps": 8,
        "lat_ms": 264.0,
        "holds": 41,
        "degrade": False,
        "chunk_age_ms": 320.0,
        "active": "policy",
        "e2e_p50_us": 221000,
        "e2e_p95_us": 256000,
        "rtt_us": 74000,
        "uncorr": 0,
    }


def test_format_stats_emits_every_key_in_order():
    payload = proto.format_stats(_sample())

    assert payload.startswith('{"t":1,')
    decoded = proto.parse_stats(payload)
    assert list(decoded) == list(proto.STATS_KEYS)
    assert decoded == _sample()


def test_format_stats_is_compact_so_the_whitespace_collapse_is_lossless():
    """The load-bearing one: format_event collapses runs of whitespace."""
    payload = proto.format_stats(_sample())
    line = proto.format_event(proto.EVENT_STATS, payload)

    event, round_tripped = proto.parse_event(line)
    assert event == proto.EVENT_STATS
    assert round_tripped == payload  # byte-for-byte, not merely "still parses"
    assert proto.parse_stats(round_tripped) == _sample()


def test_a_pretty_printed_payload_would_not_survive_the_collapse():
    """Documents WHY separators=(",", ":") is required, not a style choice."""
    pretty = json.dumps(_sample(), indent=2)

    _event, round_tripped = proto.parse_event(proto.format_event(proto.EVENT_STATS, pretty))

    assert round_tripped != pretty


def test_unknown_values_are_emitted_as_null_not_dropped():
    partial = dict(_sample())
    for key in ("chunk_age_ms", "active", "e2e_p50_us", "e2e_p95_us", "rtt_us"):
        del partial[key]

    decoded = proto.parse_stats(proto.format_stats(partial))

    assert decoded is not None
    assert list(decoded) == list(proto.STATS_KEYS)
    assert decoded["chunk_age_ms"] is None
    assert decoded["active"] is None
    assert decoded["rtt_us"] is None


def test_format_stats_rejects_an_unknown_key():
    """The key set is a contract S3.2/S3.3 model exactly; a typo must not ship."""
    with pytest.raises(ValueError, match="lattency_ms"):
        proto.format_stats({**_sample(), "lattency_ms": 1})


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{",  # truncated: a partially-flushed line
        '{"t":1,',  # truncated mid-document
        "not json at all",
        "[1,2,3]",  # valid JSON, wrong shape
        '"a string"',
        "null",
    ],
)
def test_malformed_stats_degrade_to_no_sample(payload):
    assert proto.parse_stats(payload) is None


def test_stats_interleaved_with_lerobot_chatter_still_parses():
    noisy = "INFO robot: sending action" + proto.format_event(
        proto.EVENT_STATS, proto.format_stats(_sample())
    )

    event, payload = proto.parse_event(noisy)

    assert event == proto.EVENT_STATS
    assert proto.parse_stats(payload) == _sample()


# --- commands ---------------------------------------------------------------


@pytest.fixture
def events():
    return threading.Event(), threading.Event(), threading.Event()


def test_the_first_stop_asks_for_a_graceful_return(events):
    stop, abort, quit_ = events

    assert proto.apply_command("STOP\n", stop, abort, quit_) == proto.CMD_STOP

    assert stop.is_set()
    assert not abort.is_set()
    assert not quit_.is_set()


def test_a_second_stop_cuts_the_return_short(events):
    stop, abort, quit_ = events
    proto.apply_command("STOP", stop, abort, quit_)

    assert proto.apply_command("STOP", stop, abort, quit_) == proto.CMD_STOP

    assert stop.is_set()
    assert abort.is_set()
    assert not quit_.is_set()


def test_quit_sets_everything_so_no_return_is_attempted(events):
    stop, abort, quit_ = events

    assert proto.apply_command("quit", stop, abort, quit_) == proto.CMD_QUIT

    assert quit_.is_set()
    assert stop.is_set()
    assert abort.is_set()


@pytest.mark.parametrize("line", ["", "   ", "\n", "PAUSE", "stop now", "42"])
def test_blank_and_unknown_lines_are_ignored(line, events):
    stop, abort, quit_ = events

    assert proto.apply_command(line, stop, abort, quit_) is None

    assert not stop.is_set()
    assert not abort.is_set()
    assert not quit_.is_set()


def test_pump_drives_a_whole_stream_to_the_cut_short_state(events):
    stop, abort, quit_ = events

    proto.pump_commands(["\n", "STOP\n", "STOP\n"], stop, abort, quit_)

    assert stop.is_set()
    assert abort.is_set()
    assert not quit_.is_set()


def test_pump_stops_reading_after_quit(events):
    stop, abort, quit_ = events

    def stream():
        yield "QUIT\n"
        raise AssertionError("the pump must not read past QUIT")

    proto.pump_commands(stream(), stop, abort, quit_)

    assert quit_.is_set()


def test_pump_returns_on_eof_without_stopping_the_run(events):
    """EOF is the parent's pipe closing, not a stop.

    A child that killed itself on a closed pipe would drop an energized arm the
    moment a log pump hiccuped; an abandoned session is the server-side lease
    watchdog's job."""
    stop, abort, quit_ = events

    proto.pump_commands([], stop, abort, quit_)

    assert not stop.is_set()
    assert not abort.is_set()
    assert not quit_.is_set()
