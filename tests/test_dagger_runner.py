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
"""Tests for makermodslab.dagger_runner — command translation and the cancel flag.

The control loop itself needs a policy, a robot and a leader arm, so it stays
untested here in line with the tests/ policy. What IS covered is the pure state
logic that decides which of lerobot's phase transitions each operator command
becomes — the one-key takeover is built entirely out of that mapping, and a
mistranslation would put the arm under human control when the operator asked
only to freeze it, or vice versa."""

from __future__ import annotations

import pytest

from lerobot.rollout.configs import DAggerStrategyConfig
from lerobot.rollout.strategies.dagger import DAggerPhase
from makermodslab.dagger_protocol import (
    CMD_CANCEL,
    CMD_HANDBACK,
    CMD_HOLD,
    CMD_RESUME,
    CMD_TAKEOVER,
)
from makermodslab.dagger_runner import (
    _EV_CORRECTION,
    _EV_PAUSE_RESUME,
    WebDAggerStrategy,
)

AUTONOMOUS = DAggerPhase.AUTONOMOUS
PAUSED = DAggerPhase.PAUSED
CORRECTING = DAggerPhase.CORRECTING


@pytest.fixture
def strategy() -> WebDAggerStrategy:
    """A strategy with no hardware behind it.

    `__init__` only builds queues and counters — the engine, dataset and robot
    are all attached later by `setup`/`run`, neither of which is called here."""
    return WebDAggerStrategy(DAggerStrategyConfig(num_episodes=5))


# --- Composite commands ------------------------------------------------------


def test_takeover_from_autonomous_expands_to_pause_then_correct(strategy) -> None:
    """The one-key takeover. Upstream lerobot needs two keys and four presses per
    correction cycle; the composition lives here so the operator's job is one
    button. The intermediate PAUSED step is not cosmetic — it is where lerobot
    drives the leader arm to the follower's pose."""
    assert strategy._translate(CMD_TAKEOVER, AUTONOMOUS) == [
        _EV_PAUSE_RESUME,
        _EV_CORRECTION,
    ]


def test_takeover_from_paused_only_needs_the_correction_step(strategy) -> None:
    """Already frozen (the operator held first, or a previous takeover was
    refused): the pause half is done, so re-issuing it would resume the policy."""
    assert strategy._translate(CMD_TAKEOVER, PAUSED) == [_EV_CORRECTION]


def test_handback_saves_and_returns_control_to_the_policy(strategy) -> None:
    assert strategy._translate(CMD_HANDBACK, CORRECTING) == [
        _EV_CORRECTION,
        _EV_PAUSE_RESUME,
    ]
    assert strategy._cancel_correction is False


def test_cancel_stops_at_held_rather_than_resuming(strategy) -> None:
    """A discarded correction usually means something went wrong. Dropping the
    operator straight back into an autonomous policy is the wrong default when
    they have just said the last few seconds were a mess."""
    assert strategy._translate(CMD_CANCEL, CORRECTING) == [_EV_CORRECTION]
    assert strategy._cancel_correction is True


def test_hold_and_resume_are_the_bare_pause_toggle(strategy) -> None:
    assert strategy._translate(CMD_HOLD, AUTONOMOUS) == [_EV_PAUSE_RESUME]
    assert strategy._translate(CMD_RESUME, PAUSED) == [_EV_PAUSE_RESUME]


# --- Commands that don't apply ----------------------------------------------


@pytest.mark.parametrize(
    ("command", "phase"),
    [
        (CMD_TAKEOVER, CORRECTING),  # already driving
        (CMD_HANDBACK, AUTONOMOUS),  # nothing to hand back
        (CMD_HANDBACK, PAUSED),
        (CMD_CANCEL, AUTONOMOUS),  # nothing to discard
        (CMD_CANCEL, PAUSED),
        (CMD_HOLD, PAUSED),  # already held
        (CMD_HOLD, CORRECTING),
        (CMD_RESUME, AUTONOMOUS),  # already running
        (CMD_RESUME, CORRECTING),
    ],
)
def test_a_command_that_makes_no_sense_expands_to_nothing(strategy, command, phase) -> None:
    """The browser's phase is always one poll stale, so a click can race a
    transition. Doing nothing is the right answer — far better than forcing a
    transition the state machine considers invalid."""
    assert strategy._translate(command, phase) == []


def test_an_unknown_command_expands_to_nothing(strategy) -> None:
    assert strategy._translate("LAUNCH_MISSILES", AUTONOMOUS) == []


def test_cancel_is_not_armed_by_a_command_that_does_not_apply(strategy) -> None:
    """The cancel flag is consumed at the CORRECTING→PAUSED edge. Arming it from
    a phase that can't reach that edge would leave it set, and the NEXT
    correction — a real one — would be silently discarded."""
    strategy._translate(CMD_CANCEL, AUTONOMOUS)
    assert strategy._cancel_correction is False


def test_handback_after_an_armed_cancel_clears_the_flag(strategy) -> None:
    """Guards the same latch from the other direction: a cancel that never
    reached the save edge must not turn a later hand-back into a discard."""
    strategy._translate(CMD_CANCEL, CORRECTING)
    assert strategy._cancel_correction is True
    strategy._translate(CMD_HANDBACK, CORRECTING)
    assert strategy._cancel_correction is False


# --- Command draining --------------------------------------------------------


class _FakeEvents:
    """Stands in for lerobot's `DAggerEvents`, recording what was requested."""

    def __init__(self, phase: DAggerPhase) -> None:
        self.phase = phase
        self.requested: list[str] = []

    def request_transition(self, event: str) -> None:
        self.requested.append(event)


def test_drain_applies_one_transition_per_tick(strategy) -> None:
    """Each transition's hardware side-effects — the ~2s leader glide, the torque
    flip — must complete before the next is requested, so a composite command is
    spread across control ticks rather than fired at once."""
    events = _FakeEvents(AUTONOMOUS)
    strategy.submit(CMD_TAKEOVER)

    strategy._drain_commands(events)
    assert events.requested == [_EV_PAUSE_RESUME]

    # Second tick: the phase has moved on, and the queued half is applied.
    events.phase = PAUSED
    strategy._drain_commands(events)
    assert events.requested == [_EV_PAUSE_RESUME, _EV_CORRECTION]

    strategy._drain_commands(events)
    assert events.requested == [_EV_PAUSE_RESUME, _EV_CORRECTION]


def test_drain_is_a_noop_when_no_command_is_queued(strategy) -> None:
    events = _FakeEvents(AUTONOMOUS)
    strategy._drain_commands(events)
    assert events.requested == []


def test_drain_does_not_start_a_new_command_mid_composite(strategy) -> None:
    """A second command arriving while a composite is half-applied waits its
    turn. Interleaving the two would produce a transition sequence neither
    command asked for."""
    events = _FakeEvents(AUTONOMOUS)
    strategy.submit(CMD_TAKEOVER)
    strategy.submit(CMD_HOLD)

    strategy._drain_commands(events)  # pause_resume, from TAKEOVER
    events.phase = PAUSED
    strategy._drain_commands(events)  # correction, still from TAKEOVER
    assert events.requested == [_EV_PAUSE_RESUME, _EV_CORRECTION]

    # Only now is HOLD looked at — and from CORRECTING it means nothing.
    events.phase = CORRECTING
    strategy._drain_commands(events)
    assert events.requested == [_EV_PAUSE_RESUME, _EV_CORRECTION]


# --- Continuous mode is refused ---------------------------------------------


def test_run_refuses_continuous_recording() -> None:
    """merge.py's "drop the intervention column" shortcut is only lossless
    because every recorded frame in this mode is a human correction. The runner
    refuses the other mode outright rather than letting that reasoning quietly
    stop being true."""
    strategy = WebDAggerStrategy(DAggerStrategyConfig(num_episodes=5, record_autonomous=True))
    with pytest.raises(ValueError, match="record_autonomous"):
        strategy.run(object())


def test_a_fresh_strategy_starts_with_nothing_armed(strategy) -> None:
    assert strategy._corrections_saved == 0
    assert strategy._cancel_correction is False
    assert list(strategy._pending) == []


# --- The handover announcement ----------------------------------------------


class _FakeTeleop:
    """Actuated or not, which is what picks the handover path upstream."""

    def __init__(self, actuated: bool) -> None:
        self.feedback_features = {"x": float} if actuated else {}

    def enable_torque(self) -> None: ...
    def disable_torque(self) -> None: ...


class _FakeCtx:
    def __init__(self, actuated: bool) -> None:
        self.hardware = type("HW", (), {"teleop": _FakeTeleop(actuated)})()


def test_actuated_teleop_announces_travel_on_the_pause_edge(strategy) -> None:
    """Single SO-101 leader: AUTONOMOUS->PAUSED drives the LEADER to the
    follower's pose for ~2s. Announcing it is what stops the banner insisting
    the policy is still driving while the operator's arm moves under them."""
    ctx = _FakeCtx(actuated=True)
    assert strategy._transition_moves_the_arm(AUTONOMOUS, PAUSED, ctx) is True
    assert strategy._transition_moves_the_arm(PAUSED, CORRECTING, ctx) is False


def test_non_actuated_teleop_announces_travel_on_the_correction_edge(strategy) -> None:
    """BiSOLeader on this pin: PAUSED->CORRECTING slides BOTH FOLLOWERS across
    the workspace to meet the leaders. That edge used to render as "HELD — the
    arm is frozen", which is the worst thing this UI could have said."""
    ctx = _FakeCtx(actuated=False)
    assert strategy._transition_moves_the_arm(PAUSED, CORRECTING, ctx) is True
    assert strategy._transition_moves_the_arm(AUTONOMOUS, PAUSED, ctx) is False


def test_edges_with_no_motion_are_not_announced(strategy) -> None:
    """Resuming the policy and ending a correction are a torque flip and an
    engine reset. Claiming travel there would be its own small lie."""
    for actuated in (True, False):
        ctx = _FakeCtx(actuated=actuated)
        assert strategy._transition_moves_the_arm(PAUSED, AUTONOMOUS, ctx) is False
        assert strategy._transition_moves_the_arm(CORRECTING, PAUSED, ctx) is False
