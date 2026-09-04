# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for makermodslab.dagger_protocol — the coaching line protocol.

Pure string handling with no imports beyond the module itself, which is the
whole point of the module existing: the orchestrator and the runner sit on
opposite sides of a torch-shaped dependency wall and can only agree on these
markers if they both import them from here.

Worth covering properly despite being small. A parser that silently fails to
recognise an event doesn't crash — it leaves the session UI frozen on a stale
phase while the arm keeps moving, which is the exact failure mode the events
exist to prevent."""

from __future__ import annotations

import pytest

from makermodslab.dagger_protocol import (
    CMD_CANCEL,
    CMD_HANDBACK,
    CMD_HOLD,
    CMD_QUIT,
    CMD_RESUME,
    CMD_TAKEOVER,
    COMMANDS,
    EVENT_ALIGN_REQUIRED,
    EVENT_BYE,
    EVENT_CORRECTION_CANCELLED,
    EVENT_CORRECTION_SAVED,
    EVENT_DATASET,
    EVENT_ERROR,
    EVENT_PHASE,
    EVENT_PREFIX,
    EVENT_READY,
    PHASE_AUTONOMOUS,
    PHASE_CORRECTING,
    PHASE_PAUSED,
    PHASES,
    format_event,
    parse_event,
    parse_fields,
)

ALL_EVENTS = [
    EVENT_READY,
    EVENT_DATASET,
    EVENT_PHASE,
    EVENT_CORRECTION_SAVED,
    EVENT_CORRECTION_CANCELLED,
    EVENT_ALIGN_REQUIRED,
    EVENT_ERROR,
    EVENT_BYE,
]


# --- Command vocabulary ------------------------------------------------------


def test_commands_set_contains_every_verb() -> None:
    """The orchestrator validates against COMMANDS before writing to the pipe,
    so a verb missing from it is a 400 on a control that should work."""
    from makermodslab.dagger_protocol import (
        CMD_DROP_LAST,
        CMD_RECOVERED,
        CMD_RESET,
    )

    assert {
        CMD_TAKEOVER,
        CMD_HANDBACK,
        CMD_CANCEL,
        CMD_HOLD,
        CMD_RESUME,
        CMD_RESET,
        CMD_RECOVERED,
        # Un-records the correction that is still held in memory. In COMMANDS
        # like every other verb, so a browser that sends it reaches the runner
        # rather than a 400 from the orchestrator's validation.
        CMD_DROP_LAST,
        CMD_QUIT,
    } == COMMANDS


def test_commands_are_distinct_bare_words() -> None:
    """The runner reads one bare word per line and upper-cases it, so a command
    carrying a space or lower case would never match."""
    for command in COMMANDS:
        assert command == command.strip().upper()
        assert " " not in command


def test_phases_match_lerobot_dagger_phase_values() -> None:
    """These are passed through to the wire unrenamed so a status payload and a
    lerobot log line agree. Pinned as a literal set rather than imported from
    lerobot: this module must stay import-free, and a pin that has to be updated
    by hand on a lerobot bump is exactly the prompt we want.

    Four phases are deliberately NOT in this set — they are ours, not lerobot's,
    and this assertion keeps the two vocabularies distinguishable if someone
    later adds a phase without deciding which side of the line it belongs on.
    Three describe windows where the runner is BLOCKED and no lerobot phase is
    true: driving an arm into position, writing an episode to disk, and easing
    the arms home. `poised` is different again — the loop is running and
    lerobot's own phase is PAUSED, but PAUSED cannot distinguish "the policy is
    frozen" from "both arms are held on the follower's pose waiting for you", and
    those ask the operator for opposite things."""
    from makermodslab.dagger_protocol import (
        PHASE_HANDING_OVER,
        PHASE_POISED,
        PHASE_RESETTING,
        PHASE_SAVING,
    )

    ours = {PHASE_HANDING_OVER, PHASE_SAVING, PHASE_RESETTING, PHASE_POISED}
    assert {"autonomous", "paused", "correcting"} == PHASES - ours
    assert (PHASE_AUTONOMOUS, PHASE_PAUSED, PHASE_CORRECTING) == (
        "autonomous",
        "paused",
        "correcting",
    )


# --- format_event / parse_event round-trip -----------------------------------


@pytest.mark.parametrize("event", ALL_EVENTS)
def test_every_event_round_trips_without_a_payload(event: str) -> None:
    assert parse_event(format_event(event)) == (event, "")


@pytest.mark.parametrize("event", ALL_EVENTS)
def test_every_event_round_trips_with_a_payload(event: str) -> None:
    assert parse_event(format_event(event, "n=3 frames=90")) == (event, "n=3 frames=90")


def test_format_event_collapses_a_multiline_payload_to_one_line() -> None:
    """A traceback in an ERROR payload must not split one event across several
    lines — the reader is line-oriented and would read the continuation lines as
    unknown events (or, worse, as nothing at all)."""
    line = format_event(EVENT_ERROR, "RuntimeError: bus went away\n  File x.py\n    raise")
    assert "\n" not in line
    event, payload = parse_event(line)
    assert event == EVENT_ERROR
    assert payload == "RuntimeError: bus went away File x.py raise"


def test_format_event_has_no_trailing_space_without_a_payload() -> None:
    assert format_event(EVENT_BYE) == f"{EVENT_PREFIX} {EVENT_BYE}"


def test_parse_event_returns_none_for_an_ordinary_log_line() -> None:
    assert parse_event("INFO 2026-08-18 lerobot.rollout: Connecting robot ...") is None


def test_parse_event_returns_none_for_an_empty_line() -> None:
    assert parse_event("") is None
    assert parse_event("\n") is None


def test_parse_event_returns_none_for_a_bare_prefix() -> None:
    """A prefix with no event name carries nothing to dispatch on."""
    assert parse_event(f"{EVENT_PREFIX}   \n") is None


def test_parse_event_finds_an_event_appended_to_an_unterminated_log_record() -> None:
    """The runner's logging handler shares the pipe. A record flushed without its
    trailing newline would otherwise swallow the event that follows it — which is
    why the prefix is matched anywhere in the line, not just at the start."""
    line = f"INFO some half-flushed record {EVENT_PREFIX} {EVENT_PHASE} phase=correcting"
    assert parse_event(line) == (EVENT_PHASE, "phase=correcting")


def test_parse_event_is_distinct_from_the_eval_protocol() -> None:
    """Both runners tee into the same log directory. A coaching parser that
    accepted an eval line (or vice versa) would make the two indistinguishable."""
    from makermodslab.eval_protocol import EVENT_PREFIX as EVAL_PREFIX, format_event as format_eval_event

    assert EVENT_PREFIX != EVAL_PREFIX
    assert parse_event(format_eval_event("EPISODE_STARTED")) is None


# --- parse_fields ------------------------------------------------------------


def test_parse_fields_reads_key_value_pairs() -> None:
    assert parse_fields("n=3 frames=90 seconds=4.2") == {
        "n": "3",
        "frames": "90",
        "seconds": "4.2",
    }


def test_parse_fields_returns_empty_for_an_empty_payload() -> None:
    assert parse_fields("") == {}


def test_parse_fields_drops_tokens_without_a_separator() -> None:
    """Guessed-at tokens are worse than absent ones: the caller falls back to a
    sane default for a missing key, but a wrong value silently mis-tallies."""
    assert parse_fields("n=3 garbage frames=90") == {"n": "3", "frames": "90"}


def test_parse_fields_keeps_a_value_containing_a_path() -> None:
    """DATASET carries a filesystem root; slashes and dots must survive."""
    fields = parse_fields("repo_id=rollout_fixes_20260818_120000 root=/tmp/lerobot/rollout_fixes")
    assert fields["repo_id"] == "rollout_fixes_20260818_120000"
    assert fields["root"] == "/tmp/lerobot/rollout_fixes"


def test_parse_fields_keeps_a_comma_joined_joint_list() -> None:
    """ALIGN_REQUIRED names every offending joint in one space-free token."""
    fields = parse_fields("max_delta=42 joints=shoulder_pan:42,elbow_flex:19")
    assert fields["joints"] == "shoulder_pan:42,elbow_flex:19"


def test_parse_fields_takes_the_last_value_for_a_repeated_key() -> None:
    """Not a case the runner produces, but a defined one beats an arbitrary one."""
    assert parse_fields("n=1 n=2") == {"n": "2"}


def test_correction_saved_payload_round_trips_through_both_helpers() -> None:
    """The path an actual event takes end to end: formatted by the runner,
    parsed by the orchestrator's pump, then read field by field."""
    line = format_event(EVENT_CORRECTION_SAVED, "n=4 frames=120 seconds=4.0")
    event, payload = parse_event(line)
    assert event == EVENT_CORRECTION_SAVED
    assert parse_fields(payload) == {"n": "4", "frames": "120", "seconds": "4.0"}


def test_handing_over_is_a_phase_but_not_one_of_lerobots() -> None:
    """It has to travel on the same PHASE event as the real phases, but it is
    OURS — lerobot has no state for "the arm is currently moving into position",
    which is exactly the gap it fills."""
    from makermodslab.dagger_protocol import PHASE_HANDING_OVER

    assert PHASE_HANDING_OVER in PHASES
    assert PHASE_HANDING_OVER not in {"autonomous", "paused", "correcting"}
    assert parse_event(format_event(EVENT_PHASE, f"phase={PHASE_HANDING_OVER}")) == (
        EVENT_PHASE,
        f"phase={PHASE_HANDING_OVER}",
    )


def test_recover_is_distinct_from_recovered() -> None:
    """RECOVER and RECOVERED differ by two letters and mean opposite things:
    one throws the correction away and resets, the other marks a boundary
    inside a correction it intends to keep. The runner matches on the whole
    bare word, so this pins that they can never collide."""
    from makermodslab.dagger_protocol import CMD_CANCEL, CMD_RECOVERED

    assert CMD_CANCEL != CMD_RECOVERED
    assert CMD_RECOVERED in COMMANDS
