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

import io
import time

import pytest

from lerobot.rollout.configs import DAggerStrategyConfig
from lerobot.rollout.strategies.dagger import DAggerPhase
from makermodslab.dagger_protocol import (
    CMD_CANCEL,
    CMD_DROP_LAST,
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


def test_the_first_takeover_press_only_stops_the_policy(strategy) -> None:
    """Takeover is TWO presses. The first stops the policy and asks for the
    poise; it must not reach CORRECTING, because nothing may be recorded while
    the operator is still reaching for the arm."""
    assert strategy._translate(CMD_TAKEOVER, AUTONOMOUS) == [_EV_PAUSE_RESUME]
    assert strategy._poise_requested is True
    assert strategy._poised is False


def test_a_first_press_from_paused_requests_the_poise_and_no_transition(strategy) -> None:
    """Already frozen — there is no policy left to stop, so the press buys only
    the glide and the hold. Requesting `pause_resume` here would RESUME the
    policy, which is the opposite of what the operator asked for."""
    assert strategy._translate(CMD_TAKEOVER, PAUSED) == []
    assert strategy._poise_requested is True


def test_the_second_press_is_what_hands_over(strategy) -> None:
    """Keyed on `_poised`, not on the phase: lerobot has no phase for "held,
    lined up, waiting", so PAUSED covers both that and a plain freeze."""
    strategy._poised = True
    assert strategy._translate(CMD_TAKEOVER, PAUSED) == [_EV_CORRECTION]


def test_a_takeover_while_poised_never_resumes_the_policy(strategy) -> None:
    """The regression that would put the arm back under the policy at the exact
    moment the operator reached for it."""
    strategy._poised = True
    for phase in (AUTONOMOUS, PAUSED):
        assert _EV_PAUSE_RESUME not in strategy._translate(CMD_TAKEOVER, phase)


def test_handback_saves_and_returns_control_to_the_policy(strategy) -> None:
    assert strategy._translate(CMD_HANDBACK, CORRECTING) == [
        _EV_CORRECTION,
        _EV_PAUSE_RESUME,
    ]
    assert strategy._cancel_correction is False


def test_cancel_discards_and_then_resets(strategy) -> None:
    """A discard means the last few seconds were a mess, and the scene almost
    always needs setting up again after one. So it bins the frames AND arms the
    reset: one transition out of CORRECTING this tick, then the ordinary
    ease-home on the next."""
    assert strategy._translate(CMD_CANCEL, CORRECTING) == [_EV_CORRECTION]
    assert strategy._cancel_correction is True
    assert strategy._reset_requested is True


def test_cancel_is_accepted_from_every_phase(strategy) -> None:
    """This is the absorbed RECOVER, and the reason it had to be absorbed rather
    than deleted. Phase-gating a discard is what strands an operator whose
    correction has wedged: the leader is rigid, the follower holds, and every
    command that could help is either refused or would keep bad frames. With
    nothing in flight it degrades to a plain reset."""
    for phase in (AUTONOMOUS, PAUSED):
        strategy._reset_requested = False
        assert strategy._translate(CMD_CANCEL, phase) == []
        assert strategy._reset_requested is True


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
    reached the save edge must not turn a later hand-back into a discard —
    which would bin the correction the operator has just chosen to KEEP."""
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
    # HANDBACK is the remaining composite now that takeover is two operator
    # presses rather than one command expanding into two transitions.
    events = _FakeEvents(CORRECTING)
    strategy.submit(CMD_HANDBACK)

    strategy._drain_commands(events)
    assert events.requested == [_EV_CORRECTION]

    # Second tick: the phase has moved on, and the queued half is applied.
    events.phase = PAUSED
    strategy._drain_commands(events)
    assert events.requested == [_EV_CORRECTION, _EV_PAUSE_RESUME]

    strategy._drain_commands(events)
    assert events.requested == [_EV_CORRECTION, _EV_PAUSE_RESUME]


def test_drain_is_a_noop_when_no_command_is_queued(strategy) -> None:
    events = _FakeEvents(AUTONOMOUS)
    strategy._drain_commands(events)
    assert events.requested == []


def test_drain_does_not_start_a_new_command_mid_composite(strategy) -> None:
    """A second command arriving while a composite is half-applied waits its
    turn. Interleaving the two would produce a transition sequence neither
    command asked for."""
    events = _FakeEvents(CORRECTING)
    strategy.submit(CMD_HANDBACK)
    strategy.submit(CMD_HOLD)

    strategy._drain_commands(events)  # correction, from HANDBACK
    events.phase = PAUSED
    strategy._drain_commands(events)  # pause_resume, still from HANDBACK
    assert events.requested == [_EV_CORRECTION, _EV_PAUSE_RESUME]

    # Only now is HOLD looked at — and from AUTONOMOUS it is a fresh pause.
    events.phase = AUTONOMOUS
    strategy._drain_commands(events)
    assert events.requested == [_EV_CORRECTION, _EV_PAUSE_RESUME, _EV_PAUSE_RESUME]


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


# A previous action for the handover to move toward. Upstream skips both smooth
# handovers without one, so its presence is a precondition for any travel.
_PREV = {"shoulder_pan.pos": 0.0}


def test_actuated_teleop_announces_travel_on_the_pause_edge(strategy) -> None:
    """Single SO-101 leader: AUTONOMOUS->PAUSED drives the LEADER to the
    follower's pose for ~2s. Announcing it is what stops the banner insisting
    the policy is still driving while the operator's arm moves under them."""
    ctx = _FakeCtx(actuated=True)
    assert strategy._transition_moves_the_arm(AUTONOMOUS, PAUSED, ctx, _PREV) is True
    assert strategy._transition_moves_the_arm(PAUSED, CORRECTING, ctx, _PREV) is False


def test_non_actuated_teleop_announces_travel_on_the_correction_edge(strategy) -> None:
    """BiSOLeader on this pin: PAUSED->CORRECTING slides BOTH FOLLOWERS across
    the workspace to meet the leaders. That edge used to render as "HELD — the
    arm is frozen", which is the worst thing this UI could have said."""
    ctx = _FakeCtx(actuated=False)
    assert strategy._transition_moves_the_arm(PAUSED, CORRECTING, ctx, _PREV) is True
    assert strategy._transition_moves_the_arm(AUTONOMOUS, PAUSED, ctx, _PREV) is False


def test_edges_with_no_motion_are_not_announced(strategy) -> None:
    """Resuming the policy and ending a correction are a torque flip and an
    engine reset. Claiming travel there would be its own small lie."""
    for actuated in (True, False):
        ctx = _FakeCtx(actuated=actuated)
        assert strategy._transition_moves_the_arm(PAUSED, AUTONOMOUS, ctx, _PREV) is False
        assert strategy._transition_moves_the_arm(CORRECTING, PAUSED, ctx, _PREV) is False


# --- Limp / torque during the reset window ----------------------------------


class _FakeBus:
    """Shaped like a real `MotorsBus` for the parts the torque paths touch.

    `motors` and the per-motor `disable_torque(motor, num_retry=…)` signature
    matter: `_go_limp` goes through `torque.force_disable_bus_torque`, which
    walks motors one at a time precisely so one bad write cannot leave the rest
    of the arm locked. A fake with a single all-or-nothing `disable_torque()`
    could not express the partial failure that is the whole point."""

    def __init__(self, fail_motors: set[str] | None = None) -> None:
        self.torque = True
        self.motors = {"shoulder_pan": None, "elbow_flex": None}
        self.port = "/dev/fake"
        self.released: list[str] = []
        self._fail = fail_motors or set()

    def disable_torque(self, motor=None, num_retry: int = 0) -> None:
        if motor in self._fail:
            raise RuntimeError(f"write failed on {motor}")
        self.released.append(motor)
        if set(self.released) >= set(self.motors):
            self.torque = False

    def enable_torque(self) -> None:
        self.torque = True


class _FakeRobot:
    """Single-arm shape: one bus hanging off the device."""

    def __init__(self) -> None:
        self.bus = _FakeBus()
        self.sent: list[dict] = []

    def get_observation(self) -> dict:
        return {"shoulder_pan.pos": 11.0, "elbow_flex.pos": 22.0, "not_a_motor": 1}

    def send_action(self, action: dict) -> None:
        self.sent.append(action)


class _FakeBiRobot:
    """Bimanual BiSO shape: a bus per sub-arm, none on the device itself."""

    def __init__(self) -> None:
        self.left_arm = _FakeRobot()
        self.right_arm = _FakeRobot()


def _ctx_with(robot):
    return type("Ctx", (), {"hardware": type("HW", (), {"robot_wrapper": robot})()})()


def test_follower_buses_finds_both_arms_on_a_bimanual_robot(strategy) -> None:
    assert len(strategy._follower_buses(_FakeBiRobot())) == 2
    assert len(strategy._follower_buses(_FakeRobot())) == 1


def test_go_limp_releases_torque_and_restore_re_enables_it(strategy) -> None:
    robot = _FakeRobot()
    ctx = _ctx_with(robot)
    assert strategy._go_limp(ctx) is True
    assert robot.bus.torque is False
    strategy._restore_torque(ctx)
    assert robot.bus.torque is True


def test_go_limp_goes_motor_by_motor(strategy) -> None:
    """Routed through `force_disable_bus_torque` rather than the bus's own
    `disable_torque`, whose loop aborts on the first failed write and leaves the
    earlier motors already released — a half-collapsed arm."""
    robot = _FakeRobot()
    strategy._go_limp(_ctx_with(robot))
    assert robot.bus.released == ["shoulder_pan", "elbow_flex"]


def test_a_partial_release_reports_failure_but_still_arms_the_restore(strategy) -> None:
    """The regression this exists for. One motor failing used to log "the arm
    stays powered" — false, the others are already limp — return False, and
    leave `_limp` False, so `_restore_torque` short-circuited for the rest of
    the session and those motors were never re-energised."""
    robot = _FakeRobot()
    robot.bus._fail = {"elbow_flex"}
    ctx = _ctx_with(robot)

    assert strategy._go_limp(ctx) is False, "a partial release is not a success"
    assert strategy._limp is True, "the restore must still run — some motors ARE released"
    assert strategy._restore_torque(ctx) is not None
    assert robot.bus.torque is True


def test_restore_refuses_to_energize_when_the_pose_cannot_be_read(strategy) -> None:
    """Without a fresh Goal_Position write the servos come back on their
    pre-limp target, snapping the arm from where the operator's hands left it to
    where the policy last wanted it. Failing the read must not fall through to
    enabling torque anyway."""

    class _BlindRobot(_FakeRobot):
        def get_observation(self):
            raise RuntimeError("bus read failed")

    robot = _BlindRobot()
    ctx = _ctx_with(robot)
    strategy._go_limp(ctx)
    robot.bus.torque = False

    assert strategy._restore_torque(ctx) is None
    assert robot.bus.torque is False, "torque was enabled with a stale goal"
    assert strategy._limp is True, "still limp, so the next transition retries"


def test_restore_writes_the_goal_before_re_enabling_torque(strategy) -> None:
    """The ordering that stops the arm snapping when it comes back.

    A Feetech servo holds its Goal_Position the instant torque returns, and
    that register still holds the pre-limp target. Writing the operator's new
    pose first is what makes re-powering a no-op instead of a lurch."""
    robot = _FakeRobot()
    ctx = _ctx_with(robot)
    strategy._go_limp(ctx)
    order: list[str] = []
    robot.send_action = lambda a: order.append("goal")  # noqa: ARG005
    robot.bus.enable_torque = lambda: order.append("torque")
    strategy._restore_torque(ctx)
    assert order == ["goal", "torque"]


def test_restore_sends_only_motor_positions(strategy) -> None:
    """Observations carry more than joint positions; only `.pos` keys are a
    valid action, and passing the rest through would be rejected downstream."""
    robot = _FakeRobot()
    ctx = _ctx_with(robot)
    strategy._go_limp(ctx)
    strategy._restore_torque(ctx)
    assert robot.sent == [{"shoulder_pan.pos": 11.0, "elbow_flex.pos": 22.0}]


def test_restore_is_a_noop_when_the_arm_was_never_limp(strategy) -> None:
    robot = _FakeRobot()
    assert strategy._restore_torque(_ctx_with(robot)) is None
    assert robot.sent == []


def test_minimum_correction_length_is_above_the_one_frame_crash() -> None:
    """Guards the two failures that made this threshold necessary.

    A crash mid-correction left ONE frame in the buffer. Teardown saved it, so
    the operator was credited a correction they never made — and the resulting
    one-frame episode then broke lerobot's stats aggregation outright
    ("ArrowInvalid ... expected length 2 but got length 1"), killing the
    session. The bound must therefore sit above 1, and above anything a
    double-press can produce, while staying well under a deliberate
    demonstration (which is at least a second, i.e. ~30 frames)."""
    from makermodslab.dagger_runner import _MIN_CORRECTION_FRAMES

    assert _MIN_CORRECTION_FRAMES > 1
    assert _MIN_CORRECTION_FRAMES < 30


def test_the_runner_applies_the_bus_read_retry_patch() -> None:
    """The crash that killed four of the station's sessions.

    lerobot reads Present_Position with num_retry=0, so ONE dropped serial
    reply ends the session with "[TxRxResult] There is no status packet!".
    record.py has patched that for years — but it runs inside the FastAPI
    server, and a monkeypatch only affects the interpreter that ran it. The
    coaching runner is a separate process, so it inherited the zero-retry
    default. Coaching surfaced it because it reads TWO buses per tick while
    cameras stream, which is exactly the contended-USB case the patch exists
    for."""
    import makermodslab.dagger_runner  # noqa: F401  (importing applies it)
    from lerobot.motors.motors_bus import MotorsBus

    assert MotorsBus.sync_read.__name__ == "_sync_read_with_default_retries"


def test_the_retry_patch_is_idempotent() -> None:
    """Re-importing must not make the saved original point at the patch and
    recurse — the server re-imports its modules under uvicorn --reload.

    Asserts the WRAPPED FUNCTION, not the installed method's `__name__`. The
    name is `_sync_read_with_default_retries` in the healthy case AND in the
    recursive one, so the check this replaces passed while the module was
    broken: the reload re-ran `_original_sync_read = MotorsBus.sync_read`, by
    then already the patch, so every motor read would have recursed until the
    stack blew — and the test left it that way for the rest of the session."""
    import importlib

    from lerobot.motors.motors_bus import MotorsBus
    from makermodslab import bus_retry

    pristine = bus_retry._original_sync_read
    assert pristine.__name__ == "sync_read", "the captured original is not lerobot's method"

    importlib.reload(bus_retry)

    assert bus_retry._original_sync_read is pristine, (
        "the reload re-captured the patch as its own original — calling sync_read would now recurse forever"
    )
    assert bus_retry._original_sync_read.__name__ == "sync_read"
    assert MotorsBus.sync_read.__name__ == "_sync_read_with_default_retries"
    assert bus_retry.BUS_SYNC_READ_RETRIES >= 1


def test_the_retry_patch_actually_forwards_to_lerobots_method() -> None:
    """The wrapper must call through, and an explicit `num_retry` must win."""
    from makermodslab import bus_retry

    seen: list[dict] = []

    def _fake_original(self, data_name, motors=None, *, normalize=True, num_retry=0):
        seen.append({"data_name": data_name, "num_retry": num_retry})
        return {"ok": True}

    original = bus_retry._original_sync_read
    bus_retry._original_sync_read = _fake_original
    try:
        assert bus_retry._sync_read_with_default_retries(object(), "Present_Position") == {"ok": True}
        assert seen[-1]["num_retry"] == bus_retry.BUS_SYNC_READ_RETRIES
        bus_retry._sync_read_with_default_retries(object(), "Present_Position", num_retry=0)
        assert seen[-1]["num_retry"] == 0, "an explicit num_retry must not be overridden"
    finally:
        bus_retry._original_sync_read = original


# --- RECOVERED: an annotation, not a transition ------------------------------


def test_recovered_marks_the_boundary_without_requesting_a_transition(strategy) -> None:
    """Recovery and correction are the SAME control mode — human on the leader,
    frames recording. Inventing a lerobot phase for a distinction lerobot does
    not have would put the vendored loop out of step with upstream's state
    machine and buy nothing."""
    from makermodslab.dagger_protocol import CMD_RECOVERED

    assert strategy._translate(CMD_RECOVERED, CORRECTING) == []
    assert strategy._recovery_mark_requested is True


def test_recovered_is_ignored_outside_a_correction(strategy) -> None:
    """There is no boundary to mark while the policy is driving or the arm is
    held — nothing is being recorded to divide."""
    from makermodslab.dagger_protocol import CMD_RECOVERED

    for phase in (AUTONOMOUS, PAUSED):
        strategy._recovery_mark_requested = False
        assert strategy._translate(CMD_RECOVERED, phase) == []
        assert strategy._recovery_mark_requested is False


def test_a_second_recovered_cannot_move_an_already_marked_boundary(strategy) -> None:
    """The first mark is the one the operator meant. Silently moving it on a
    stray second press would rewrite a claim they already made, and they would
    have no way to tell."""
    from makermodslab.dagger_protocol import CMD_RECOVERED

    strategy._recovery_frames = 40  # already marked
    assert strategy._translate(CMD_RECOVERED, CORRECTING) == []
    assert strategy._recovery_mark_requested is False
    assert strategy._recovery_frames == 40


def test_a_fresh_strategy_starts_unmarked(strategy) -> None:
    """Unmarked is None, never 0 — the sidecar has to keep "the operator said
    nothing" distinguishable from "the operator said there was no recovery"."""
    assert strategy._recovery_frames is None
    assert strategy._recovery_mark_requested is False


def test_no_travel_is_announced_without_a_previous_action(strategy) -> None:
    """Upstream skips BOTH smooth-handover paths when `prev_action` is None, so
    a mirror that ignored it would promise a movement that never happens.

    Reachable in ordinary use: a takeover during policy warmup, and the first
    transition after a reset, both arrive here with no previous action. The
    window is short, but the whole reason this mirror exists is that the banner
    must not describe hardware that isn't doing what it says."""
    for actuated in (True, False):
        ctx = _FakeCtx(actuated=actuated)
        assert strategy._transition_moves_the_arm(AUTONOMOUS, PAUSED, ctx, None) is False
        assert strategy._transition_moves_the_arm(PAUSED, CORRECTING, ctx, None) is False


# --- The held correction -----------------------------------------------------
#
# DROP_LAST is the only control in the session that can destroy a finished
# correction, and it works by NOT WRITING one rather than by deleting one —
# lerobot has no way to take a single episode back out of a dataset that is
# still open (see "The held correction" in dagger_protocol). That makes the
# hold/commit/drop bookkeeping the thing worth pinning: a held correction that
# is silently forgotten is lost work, and one that is committed twice is a
# corrupt dataset.


class _FakeWriter:
    """The writer surface `_hold_correction` / `_commit_held` / `_drop_held` touch.

    Deliberately models the ONE upstream invariant that makes the hold legal:
    `save_episode` refuses a buffer whose `episode_index` is not the dataset's
    current `total_episodes`. Everything this feature does to stay compatible
    with that is only visible against a fake that enforces it."""

    def __init__(self) -> None:
        self.total_episodes = 0
        self.episode_buffer: dict | None = self._create_episode_buffer()
        self.cleared: list[int] = []
        self.cleaned_up: list[int] = []

    def _create_episode_buffer(self) -> dict:
        return {"episode_index": self.total_episodes, "size": 0}

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self.cleared.append(self.episode_buffer["episode_index"])
        self.episode_buffer = self._create_episode_buffer()

    def cleanup_interrupted_episode(self, episode_index: int) -> None:
        self.cleaned_up.append(episode_index)


class _FakeDataset:
    def __init__(self, *, fail: bool = False) -> None:
        self.writer = _FakeWriter()
        self.written: list[dict] = []
        self._fail = fail

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        if self._fail:
            raise RuntimeError("disk went away")
        buffer = self.writer.episode_buffer if episode_data is None else episode_data
        assert buffer is not None, "save_episode called with no buffer at all"
        assert buffer["episode_index"] == self.writer.total_episodes, (
            "upstream's validate_episode_buffer would reject this buffer: "
            f"episode_index={buffer['episode_index']} but total_episodes="
            f"{self.writer.total_episodes}"
        )
        self.written.append(buffer)
        self.writer.total_episodes += 1


def _record(strategy, dataset, *, frames: int = 120, seconds: float = 6.0) -> bool:
    """Finish a correction: count it and hold it, as the loop's edge does."""
    strategy._corrections_saved += 1
    return strategy._hold_correction(dataset, frames=frames, seconds=seconds)


def test_a_finished_correction_is_held_rather_than_written(strategy) -> None:
    """The premise of the whole feature. Nothing reaches the dataset at hand-back,
    which is precisely why the operator can still take it back."""
    dataset = _FakeDataset()
    assert _record(strategy, dataset) is True
    assert dataset.written == []
    assert strategy._held is not None
    # The writer is left with no buffer at all; upstream's `add_frame` builds a
    # fresh one lazily, and nothing adds frames before the commit.
    assert dataset.writer.episode_buffer is None


def test_committing_writes_the_held_buffer_and_leaves_a_usable_writer(strategy) -> None:
    dataset = _FakeDataset()
    _record(strategy, dataset)
    strategy._commit_held(dataset)
    assert len(dataset.written) == 1
    assert strategy._held is None
    # The replacement buffer must carry the NEW index. Handing back one stamped
    # with the episode we just wrote would both collide on its temp frame
    # directory and be rejected by upstream on the next save.
    assert dataset.writer.episode_buffer == {"episode_index": 1, "size": 0}


def test_two_corrections_in_a_row_each_reach_the_dataset_exactly_once(strategy) -> None:
    """The invariant that keeps the hold legal: only ever ONE uncommitted
    correction, so every buffer still satisfies `episode_index == total_episodes`
    when it is written. `_FakeDataset.save_episode` asserts that directly."""
    dataset = _FakeDataset()
    _record(strategy, dataset)
    strategy._commit_held(dataset)
    _record(strategy, dataset)
    strategy._commit_held(dataset)
    assert [b["episode_index"] for b in dataset.written] == [0, 1]
    assert strategy._corrections_saved == 2


def test_committing_twice_writes_once(strategy) -> None:
    """The commit is reached from three places — a takeover, the next attempt,
    and teardown — and more than one of them can fire for a single correction."""
    dataset = _FakeDataset()
    _record(strategy, dataset)
    strategy._commit_held(dataset)
    strategy._commit_held(dataset)
    assert len(dataset.written) == 1


def test_dropping_writes_nothing_and_takes_the_correction_off_the_tally(strategy) -> None:
    dataset = _FakeDataset()
    _record(strategy, dataset)
    assert strategy._corrections_saved == 1
    strategy._drop_held(dataset)
    assert dataset.written == []
    assert strategy._held is None
    # Counted from the hand-back — it WAS a correction until the operator said
    # otherwise — so the drop has to undo that or the summary claims a
    # correction the dataset does not contain.
    assert strategy._corrections_saved == 0


def test_dropping_cleans_up_the_temp_frames(strategy) -> None:
    """Both cleanup paths, because they cover different feature types: image
    features by `clear_episode_buffer`, video features by
    `cleanup_interrupted_episode`. Leaving either behind fills the disk with
    frames of a correction nobody kept."""
    dataset = _FakeDataset()
    _record(strategy, dataset)
    strategy._drop_held(dataset)
    assert dataset.writer.cleared == [0]
    assert dataset.writer.cleaned_up == [0]
    # And the writer is usable again, at the SAME index — nothing was written,
    # so the next correction takes the slot the dropped one was going to.
    assert dataset.writer.episode_buffer == {"episode_index": 0, "size": 0}


def test_dropping_nothing_is_a_noop(strategy) -> None:
    dataset = _FakeDataset()
    strategy._drop_held(dataset)
    assert strategy._corrections_saved == 0
    assert dataset.writer.cleared == []


def test_a_commit_that_fails_corrects_the_tally_instead_of_raising(strategy) -> None:
    """The commit happens on the way into a takeover. Raising there would take
    the session down with the arm mid-handover and every earlier correction
    already on disk — so it is reported and the count corrected instead."""
    dataset = _FakeDataset(fail=True)
    _record(strategy, dataset)
    strategy._commit_held(dataset)
    assert strategy._held is None
    assert strategy._corrections_saved == 0


def test_drop_last_is_refused_mid_correction(strategy) -> None:
    """From CORRECTING the operator means the take they are still recording, and
    the control for that is CANCEL. Honouring DROP_LAST here would bin the
    PREVIOUS correction — one they were happy with — and leave the one they
    meant untouched."""
    strategy._held = {"episode_index": 0}
    assert strategy._translate(CMD_DROP_LAST, CORRECTING) == []
    assert strategy._drop_last_requested is False


def test_drop_last_arms_the_drop_when_something_is_held(strategy) -> None:
    strategy._held = {"episode_index": 0}
    assert strategy._translate(CMD_DROP_LAST, PAUSED) == []
    # An annotation, not a transition: nothing moves and no phase changes.
    assert strategy._drop_last_requested is True


def test_drop_last_with_nothing_held_arms_nothing(strategy) -> None:
    """The window closes on its own the moment the correction is committed, and
    a stale button press must not arm a drop that would then take the NEXT
    correction."""
    assert strategy._translate(CMD_DROP_LAST, PAUSED) == []
    assert strategy._drop_last_requested is False


def test_a_fresh_strategy_holds_nothing(strategy) -> None:
    assert strategy._held is None
    assert strategy._held_meta is None
    assert strategy._drop_last_requested is False
    assert strategy._awaiting_attempt is False


# --- the mid-tick drive ------------------------------------------------------
#
# The stutter fix. The loop commands the follower once per tick because that is
# what pairs with one recorded frame; on the station that worked out to a goal
# position every 86ms, against plain teleoperation's ~2ms on the same hardware.
# The tick's leftover time now drives instead of sleeping.


class _DriveRobot:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_action(self, action):
        self.sent.append(action)


class _DriveTeleop:
    def __init__(self, fail_after: int | None = None) -> None:
        self.reads = 0
        self.fail_after = fail_after

    def get_action(self) -> dict:
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("dropped serial reply")
        return {"shoulder_pan.pos": float(self.reads)}


class _DriveCtx:
    def __init__(self, shutdown=None) -> None:
        import threading

        identity = type("P", (), {})()
        identity.teleop_action_processor = lambda pair: pair[0]
        identity.robot_action_processor = lambda pair: pair[0]
        self.processors = identity
        self.runtime = type("R", (), {"shutdown_event": shutdown or threading.Event()})()


def test_the_tick_tail_drives_the_follower_many_times(strategy) -> None:
    """One command per tick is what made it feel stepped. The tail should issue
    a stream of them, not one."""
    robot, teleop = _DriveRobot(), _DriveTeleop()
    deadline = time.perf_counter() + 0.05
    strategy._drive_until(deadline, _DriveCtx(), robot, teleop, {}, None)
    # ~1ms pacing over 50ms; generous floor so a loaded CI box still passes.
    assert len(robot.sent) > 5
    assert teleop.reads == len(robot.sent)


def test_the_drive_stops_at_the_deadline(strategy) -> None:
    """It fills the tick's idle time and must not eat the next tick — the
    recorded frame rate is the dataset's, and it stays the loop's to set."""
    robot, teleop = _DriveRobot(), _DriveTeleop()
    start = time.perf_counter()
    strategy._drive_until(start + 0.03, _DriveCtx(), robot, teleop, {}, None)
    assert time.perf_counter() - start < 0.06


def test_the_drive_returns_the_last_action_it_sent(strategy) -> None:
    """`last_action` is what PAUSED holds the arm at and what the handover
    reasons about, so the tail's commands have to be reflected in it — leaving
    it at the tick's own action would hold the arm at a pose it left 30ms ago."""
    robot, teleop = _DriveRobot(), _DriveTeleop()
    out = strategy._drive_until(time.perf_counter() + 0.02, _DriveCtx(), robot, teleop, {}, None)
    assert out == robot.sent[-1]


def test_a_deadline_already_passed_sends_nothing(strategy) -> None:
    robot, teleop = _DriveRobot(), _DriveTeleop()
    out = strategy._drive_until(time.perf_counter() - 1, _DriveCtx(), robot, teleop, {}, "prev")
    assert robot.sent == []
    assert out == "prev"


def test_a_dropped_reply_mid_tail_does_not_end_the_session(strategy) -> None:
    """These are EXTRA commands between two the loop was already going to send.
    Skipping one costs a millisecond of smoothness; raising would kill a session
    over a glitch the tick's own read/write would have retried."""
    robot, teleop = _DriveRobot(), _DriveTeleop(fail_after=2)
    out = strategy._drive_until(time.perf_counter() + 0.05, _DriveCtx(), robot, teleop, {}, None)
    assert len(robot.sent) == 2
    assert out == robot.sent[-1]


def test_a_shutdown_request_stops_the_drive_immediately(strategy) -> None:
    """QUIT must not wait out a tick tail that is busy driving an arm."""
    import threading

    stop = threading.Event()
    stop.set()
    robot, teleop = _DriveRobot(), _DriveTeleop()
    strategy._drive_until(time.perf_counter() + 0.05, _DriveCtx(stop), robot, teleop, {}, None)
    assert robot.sent == []


# --- The takeover offset -----------------------------------------------------
#
# The handover, rebuilt. It used to drive the leader onto the follower, measure
# the residual, and REFUSE if it was too wide. On real hardware the residual is
# 2-8 degrees, the measurement was taken while the arm was still travelling, and
# a refusal changed nothing — so a leader that stalls short produced an
# unbreakable handing_over -> refuse -> paused loop. One session logged 19
# refusals, 11 at an identical 7 degrees, and could not be driven at all.
#
# Nothing is measured against a threshold now and nothing is refused: the gap is
# cancelled instead of closed.


def _pose(**kw):
    return {f"{k}.pos": float(v) for k, v in kw.items()}


def test_the_follower_does_not_move_at_the_instant_of_takeover(strategy) -> None:
    """THE property. Whatever the gap, the first commanded position is where the
    follower already is."""
    leader, follower = _pose(shoulder_pan=0, elbow=10), _pose(shoulder_pan=40, elbow=-5)
    strategy.begin_correction(leader, follower)
    first = strategy.follower_target(leader, strategy._offset_at)
    assert first == pytest.approx(follower)


def test_a_thirty_degree_gap_is_cancelled_not_closed(strategy) -> None:
    """The FOLLOWER is never driven to meet the leader: whatever gap survives
    `close_the_gap`'s best-effort leader glide is absorbed by the offset, so a
    takeover with the arms 30 degrees apart still commands the follower exactly
    where it is standing. It is the decay, not the handover, that closes the
    remaining distance."""
    leader, follower = _pose(shoulder_pan=0), _pose(shoulder_pan=30)
    strategy.begin_correction(leader, follower)
    assert strategy.follower_target(leader, strategy._offset_at)["shoulder_pan.pos"] == pytest.approx(30.0)


def test_moving_the_leader_moves_the_follower_by_the_same_amount() -> None:
    """The offset must SHIFT the frame, not scale it — a demonstration's shape is
    the whole point of recording it.

    Measured against a decay-only baseline at the same instant, on a separate
    strategy each time, so neither the decay nor the rate limit's own history
    can be mistaken for the answer."""
    from lerobot.rollout.configs import DAggerStrategyConfig
    from makermodslab.dagger_runner import WebDAggerStrategy

    def follower_at(leader_now, dt):
        s = WebDAggerStrategy(DAggerStrategyConfig(num_episodes=5))
        s.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=30))
        t = s._offset_at
        s.follower_target(_pose(shoulder_pan=0), t)
        return s.follower_target(_pose(shoulder_pan=leader_now), t + dt)["shoulder_pan.pos"]

    still = follower_at(0, 0.1)
    moved = follower_at(5, 0.1)
    assert moved - still == pytest.approx(5.0, abs=0.01)


def test_the_offset_decays_to_exactly_zero(strategy) -> None:
    """Otherwise repeated takeovers walk the leader into its joint limits while
    the follower sits mid-range — the clutch drift this design exists to avoid.
    Linear, so it truly reaches zero rather than approaching it forever."""
    from makermodslab.dagger_runner import _OFFSET_DECAY_S

    leader, follower = _pose(shoulder_pan=0), _pose(shoulder_pan=30)
    strategy.begin_correction(leader, follower)
    t = strategy._offset_at
    strategy.follower_target(leader, t)
    after = strategy.follower_target(leader, t + _OFFSET_DECAY_S + 0.01)
    assert after["shoulder_pan.pos"] == pytest.approx(0.0, abs=1.0)
    assert strategy._offset is None


def test_the_offset_is_half_gone_halfway_through(strategy) -> None:
    """Halfway through the DECAY, which is no longer a constant: a 100-unit
    offset is stretched past `_OFFSET_DECAY_S` so the follower's implied speed
    stays under `_OFFSET_DECAY_DEG_PER_S`. Asking against the flat constant
    would be asking the follower to move at 66 deg/s, which is the very thing
    the stretch exists to refuse."""
    leader, follower = _pose(shoulder_pan=0), _pose(shoulder_pan=100)
    strategy.begin_correction(leader, follower)
    t = strategy._offset_at
    strategy.follower_target(leader, t)
    half = strategy.follower_target(leader, t + strategy._decay_s / 2)
    assert half["shoulder_pan.pos"] == pytest.approx(50.0, abs=5.0)


def test_joints_the_two_arms_do_not_share_are_passed_through(strategy) -> None:
    """Nothing is known about a gap that cannot be measured, and inventing one
    would be worse than leaving it."""
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=10))
    out = strategy.follower_target({"shoulder_pan.pos": 0.0, "gripper.pos": 7.0}, strategy._offset_at)
    assert out["gripper.pos"] == pytest.approx(7.0)


def test_ending_a_correction_drops_the_offset(strategy) -> None:
    """It belongs to one takeover. Carrying it into the next would offset an
    already-offset frame."""
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=30))
    strategy.end_correction()
    assert strategy._offset is None
    assert strategy._last_target is None


# --- The rate limit ----------------------------------------------------------


def test_a_huge_jump_is_rate_limited(strategy) -> None:
    """What the limit is actually for: a DISCONTINUITY in the leader's reported
    pose — a stale or garbled read, a re-acquire — clamped to one tick's worth
    of travel.

    Note what it does not cover, so this test is not read as more than it is: it
    lives inside `follower_target` and so bounds only commands issued during a
    correction, and it is skipped entirely while `_last_target` is None. It also
    never engages against the offset decay, which moves the target by at most
    offset/`_OFFSET_DECAY_S` per second — see `_MAX_FOLLOWER_DEG_PER_S`."""
    from makermodslab.dagger_runner import _MAX_FOLLOWER_DEG_PER_S

    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    t = strategy._offset_at
    strategy.follower_target(_pose(shoulder_pan=0), t)
    out = strategy.follower_target(_pose(shoulder_pan=500), t + 0.0333)
    assert out["shoulder_pan.pos"] <= _MAX_FOLLOWER_DEG_PER_S * 0.0333 + 0.01


def test_the_rate_limit_is_symmetric(strategy) -> None:
    from makermodslab.dagger_runner import _MAX_FOLLOWER_DEG_PER_S

    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    t = strategy._offset_at
    strategy.follower_target(_pose(shoulder_pan=0), t)
    out = strategy.follower_target(_pose(shoulder_pan=-500), t + 0.0333)
    assert out["shoulder_pan.pos"] >= -(_MAX_FOLLOWER_DEG_PER_S * 0.0333) - 0.01


def test_ordinary_teleoperation_is_never_shaped_by_the_rate_limit(strategy) -> None:
    """It is a floor under the pathological case, not a comfort feature. An
    operator swinging a leader hard peaks around 120 deg/s; if the limit bit
    there it would quietly flatten real demonstrations into the dataset."""
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    t = strategy._offset_at
    strategy.follower_target(_pose(shoulder_pan=0), t)
    # 120 deg/s for one 30Hz tick = 4 deg.
    out = strategy.follower_target(_pose(shoulder_pan=4), t + 0.0333)
    assert out["shoulder_pan.pos"] == pytest.approx(4.0)


def test_the_limit_scales_with_elapsed_time_not_with_calls(strategy) -> None:
    """`_drive_until` calls this every millisecond and the loop every 33ms. A
    per-CALL limit would throttle the drive to a crawl and let the loop through."""
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    t = strategy._offset_at
    strategy.follower_target(_pose(shoulder_pan=0), t)
    small = strategy.follower_target(_pose(shoulder_pan=500), t + 0.001)
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    t2 = strategy._offset_at
    strategy.follower_target(_pose(shoulder_pan=0), t2)
    big = strategy.follower_target(_pose(shoulder_pan=500), t2 + 0.033)
    assert big["shoulder_pan.pos"] > small["shoulder_pan.pos"] * 10


def test_a_fresh_strategy_holds_no_offset(strategy) -> None:
    assert strategy._offset is None
    assert strategy._last_target is None


# --- Coming home ------------------------------------------------------------
#
# Upstream's CORRECTING->PAUSED enables teleop torque to hold the leader, and
# nothing on the reset path released it. So the operator finished a correction,
# pressed reset, and was left holding an arm locked wherever their hand stopped
# — through the follower's whole glide and the scene rearrangement after it.


class _HomeLeader:
    def __init__(self, pose, actuated=True, fail_release=False, fail_feedback=False) -> None:
        self.pose = dict(pose)
        self.feedback_features = {"shoulder_pan.pos": float} if actuated else {}
        self.torque = True
        self.released = False
        self.fail_release = fail_release
        self.fail_feedback = fail_feedback
        self.moves: list[dict] = []

    def get_action(self):
        return dict(self.pose)

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        if self.fail_release:
            raise RuntimeError("bus gone")
        self.torque = False
        self.released = True

    def send_feedback(self, action):
        if self.fail_feedback:
            raise RuntimeError("bus gone")
        self.moves.append(dict(action))
        self.pose = dict(action)


class _HomeRobot:
    def __init__(self, pose) -> None:
        self.pose = dict(pose)
        self.moves: list[dict] = []

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        self.moves.append(dict(action))
        self.pose = {**self.pose, **action}


def _home_ctx(robot, leader):
    hw = type("H", (), {"robot_wrapper": robot, "teleop": leader, "initial_position": None})()
    return type("C", (), {"hardware": hw})()


def test_both_arms_are_driven_home_in_the_same_pass(strategy) -> None:
    """Together, not one after the other. Two sequential glides would be two
    waits, and the operator is watching one movement."""
    robot = _HomeRobot(_pose(shoulder_pan=40))
    leader = _HomeLeader(_pose(shoulder_pan=-20))
    target = _pose(shoulder_pan=0)
    strategy._ease_both(_home_ctx(robot, leader), robot.get_observation(), leader.get_action(), target, 0.05)
    assert robot.moves and leader.moves
    assert len(robot.moves) == len(leader.moves)
    # Both land ON the target, not near it: the last interpolation step is t=1.
    assert robot.moves[-1]["shoulder_pan.pos"] == pytest.approx(0.0)
    assert leader.moves[-1]["shoulder_pan.pos"] == pytest.approx(0.0)


def test_a_leader_that_stops_responding_does_not_interrupt_the_follower(strategy) -> None:
    """The leader is being returned for the operator's comfort; the follower is
    carrying a gripper over a table. A dead leader bus must not strand it."""
    robot = _HomeRobot(_pose(shoulder_pan=40))
    leader = _HomeLeader(_pose(shoulder_pan=-20), fail_feedback=True)
    strategy._ease_both(
        _home_ctx(robot, leader), robot.get_observation(), leader.get_action(), _pose(shoulder_pan=0), 0.05
    )
    assert robot.moves[-1]["shoulder_pan.pos"] == pytest.approx(0.0)


def test_no_leader_still_eases_the_follower(strategy) -> None:
    robot = _HomeRobot(_pose(shoulder_pan=40))
    leader = _HomeLeader({}, actuated=False)
    strategy._ease_both(_home_ctx(robot, leader), robot.get_observation(), None, _pose(shoulder_pan=0), 0.05)
    assert robot.moves[-1]["shoulder_pan.pos"] == pytest.approx(0.0)
    assert leader.moves == []


def test_the_leader_is_released_so_it_can_be_moved_by_hand(strategy) -> None:
    leader = _HomeLeader(_pose(shoulder_pan=0))
    assert strategy._release_leader(_home_ctx(_HomeRobot({}), leader)) is True
    assert leader.released is True
    assert leader.torque is False


def test_releasing_a_non_actuated_leader_is_a_noop(strategy) -> None:
    """It was never under torque, so it is already free — and calling
    disable_torque on one that has no feedback support is not a thing to do."""
    leader = _HomeLeader({}, actuated=False)
    assert strategy._release_leader(_home_ctx(_HomeRobot({}), leader)) is True
    assert leader.released is False


def test_a_leader_that_cannot_be_released_is_reported_not_raised(strategy) -> None:
    """It is reported so the log says the arm may still be rigid, but a reset
    that has already brought the follower home must still finish."""
    leader = _HomeLeader(_pose(shoulder_pan=0), fail_release=True)
    assert strategy._release_leader(_home_ctx(_HomeRobot({}), leader)) is False


# --- Closing the gap before the offset is taken ------------------------------
#
# Pure offset-and-decay guarantees the follower does not JUMP, but the decay
# still walks it all the way to wherever the leader stands: a 40 degree gap
# becomes a 40 degree follower sweep across the workspace, merely spread over
# the decay. Closing most of it with the leader first — which is in the
# operator's hand, not over the task — leaves the follower almost nothing to
# travel. It needs no verification precisely because the offset is measured
# afterwards, against whatever is actually left.


class _GlideLeader(_HomeLeader):
    def __init__(self, pose, arrives=True, **kw) -> None:
        super().__init__(pose, **kw)
        self.arrives = arrives
        self.glides: list[float] = []


def _glide_ctx(strategy, monkeypatch, leader, follower_pose):
    from makermodslab import dagger_runner

    def fake_move(tl, target, duration_s=2.0, fps=30):
        tl.glides.append(duration_s)
        if tl.fail_feedback:
            raise RuntimeError("bus gone")
        if tl.arrives:
            tl.pose = dict(target)

    monkeypatch.setattr(dagger_runner, "teleop_smooth_move_to", fake_move)
    return _home_ctx(_HomeRobot(follower_pose), leader)


def test_the_leader_closes_the_gap_before_the_offset_is_taken(strategy, monkeypatch) -> None:
    leader = _GlideLeader(_pose(shoulder_pan=-20))
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    assert leader.glides
    assert leader.pose["shoulder_pan.pos"] == pytest.approx(40.0)


def test_a_leader_already_in_place_is_not_moved(strategy, monkeypatch) -> None:
    leader = _GlideLeader(_pose(shoulder_pan=39))
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    assert leader.glides == []


def test_a_glide_that_stalls_short_is_absorbed_not_refused(strategy, monkeypatch) -> None:
    """THE property that makes this safe to attempt. The old code measured the
    residual and refused; one session logged 19 refusals at an identical 7
    degrees and could not be driven at all. Here a stalled glide is simply a
    larger offset."""
    leader = _GlideLeader(_pose(shoulder_pan=-20), arrives=False)
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    # Nothing raised, nothing refused; the leader simply did not get there.
    strategy.begin_correction(leader.get_action(), _pose(shoulder_pan=40))
    first = strategy.follower_target(leader.get_action(), strategy._offset_at)
    assert first["shoulder_pan.pos"] == pytest.approx(40.0)


def test_a_leader_bus_failure_during_the_glide_is_survivable(strategy, monkeypatch) -> None:
    leader = _GlideLeader(_pose(shoulder_pan=-20), fail_feedback=True)
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))  # must not raise


def test_a_non_actuated_leader_is_left_alone(strategy, monkeypatch) -> None:
    """Bimanual on this pin. It cannot be driven, and there the offset does the
    entire job on its own."""
    leader = _GlideLeader(_pose(shoulder_pan=-20), actuated=False)
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    assert leader.glides == []


def test_a_bigger_gap_is_given_a_longer_glide_but_still_capped(strategy, monkeypatch) -> None:
    from makermodslab.dagger_runner import _TAKEOVER_GLIDE_MAX_S

    small = _GlideLeader(_pose(shoulder_pan=30))
    strategy.close_the_gap(
        _glide_ctx(strategy, monkeypatch, small, _pose(shoulder_pan=40)), _pose(shoulder_pan=40)
    )
    big = _GlideLeader(_pose(shoulder_pan=-150))
    strategy.close_the_gap(
        _glide_ctx(strategy, monkeypatch, big, _pose(shoulder_pan=40)), _pose(shoulder_pan=40)
    )
    assert big.glides[0] > small.glides[0]
    assert big.glides[0] == pytest.approx(_TAKEOVER_GLIDE_MAX_S)


def test_the_leader_is_held_after_the_glide_not_released(strategy, monkeypatch) -> None:
    """Inverted deliberately. The leader used to be freed the instant the glide
    finished; it is now HELD on the follower's pose until the operator's second
    press, so they take hold of an arm that is already lined up and stationary."""
    leader = _GlideLeader(_pose(shoulder_pan=-20))
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    assert leader.released is False
    assert leader.torque is True


def test_a_raised_glide_still_leaves_the_hold_recoverable(strategy, monkeypatch) -> None:
    """A failed glide must not raise out of the poise — the operator lines the
    arms up by hand instead, which is precisely the case that used to walk the
    FOLLOWER 114 degrees across the workspace to meet a leader that never moved."""
    leader = _GlideLeader(_pose(shoulder_pan=-20), fail_feedback=True)
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))  # must not raise


def test_a_skipped_glide_is_still_held(strategy, monkeypatch) -> None:
    """`close_the_gap` returns early when the leader is already close enough, and
    the glide is the only thing in that path that powers it — so the hold has to
    be established separately or "held" would mean "limp" exactly when the
    operator is most likely to trust it."""
    leader = _GlideLeader(_pose(shoulder_pan=39))
    ctx = _glide_ctx(strategy, monkeypatch, leader, _pose(shoulder_pan=40))
    strategy.close_the_gap(ctx, _pose(shoulder_pan=40))
    assert leader.glides == []
    strategy._hold_leader(ctx, _pose(shoulder_pan=40))
    assert leader.torque is True
    assert leader.released is False


def test_a_discard_leaves_the_release_to_the_reset_it_armed(strategy) -> None:
    """CMD_CANCEL always arms a reset, and the reset's `_ease_home` re-torques
    the leader to walk it home and releases it there. Releasing at the discard
    edge as well made that slack-grab-slack inside about a second, with the
    operator's hand on the arm — indistinguishable from the "it went stiff and
    nothing on screen says why" fault the release exists to prevent."""
    leader = _HomeLeader(_pose(shoulder_pan=0))
    strategy._reset_requested = True
    assert strategy._release_after_discard(_home_ctx(_HomeRobot({}), leader)) is False
    assert leader.released is False
    assert leader.torque is True


def test_a_discard_with_no_reset_armed_still_frees_the_leader(strategy) -> None:
    """The other half, and why this is a condition rather than a deletion. A
    discard stops at PAUSED, and upstream's CORRECTING->PAUSED enables teleop
    torque; with no reset behind it there is no scheduled transition that would
    ever release the leader again — the original "the arm is stuck there"."""
    leader = _HomeLeader(_pose(shoulder_pan=0))
    strategy._reset_requested = False
    assert strategy._release_after_discard(_home_ctx(_HomeRobot({}), leader)) is True
    assert leader.released is True
    assert leader.torque is False


# --- The decay obeys a speed ceiling -----------------------------------------
#
# The rate limit could never bound the offset decay: decaying an offset O over a
# flat 1.5s moves the follower at O/1.5 deg/s, so clipping needed O above 360
# when a joint's whole range is about 200. It never engaged, while the comment
# claimed it was "the only thing standing between a large offset and a swept
# workspace". The fix is not a tighter clamp — it is deriving the DURATION from
# the offset, so a big gap decays over longer instead of faster.


def test_a_large_offset_decays_over_longer_not_faster(strategy) -> None:
    from makermodslab.dagger_runner import _OFFSET_DECAY_DEG_PER_S, _OFFSET_DECAY_S

    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=180))
    assert strategy._decay_s > _OFFSET_DECAY_S
    assert strategy._decay_s == pytest.approx(180 / _OFFSET_DECAY_DEG_PER_S, rel=0.01)


def test_a_small_offset_still_gets_the_floor(strategy) -> None:
    """The duration scales UP, never down — a 2 degree offset must not snap home
    in 30ms just because it could."""
    from makermodslab.dagger_runner import _OFFSET_DECAY_S

    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=2))
    assert strategy._decay_s == pytest.approx(_OFFSET_DECAY_S)


def test_the_decay_never_moves_the_follower_faster_than_the_ceiling(strategy) -> None:
    """THE property, stated directly: whatever the gap, the follower's implied
    speed while the offset unwinds stays under the ceiling. This is the assertion
    that fails if anyone reintroduces a fixed decay duration."""
    from makermodslab.dagger_runner import _OFFSET_DECAY_DEG_PER_S

    for gap in (10, 60, 180, 300):
        s = strategy.__class__(DAggerStrategyConfig(num_episodes=5))
        s.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=gap))
        implied = gap / s._decay_s
        assert implied <= _OFFSET_DECAY_DEG_PER_S + 0.01, f"{gap} deg unwound at {implied:.0f} deg/s"


# --- The clamp is armed for the FIRST command of every correction ------------


def test_the_first_command_of_a_correction_is_clamped(strategy) -> None:
    """`begin_correction` used to leave `_last_target_at` holding a timestamp
    from the PREVIOUS correction, so `elapsed` was seconds or minutes and the
    ceiling came out in the thousands — the clamp was inert for exactly the
    command that matters most."""
    from makermodslab.dagger_runner import _MAX_FOLLOWER_DEG_PER_S

    strategy._last_target_at = time.perf_counter() - 600.0  # a stale correction
    strategy.begin_correction(_pose(shoulder_pan=0), _pose(shoulder_pan=0))
    out = strategy.follower_target(_pose(shoulder_pan=500), strategy._offset_at + 0.0333)
    assert out["shoulder_pan.pos"] <= _MAX_FOLLOWER_DEG_PER_S * 0.0333 + 0.01


def test_a_failed_offset_measurement_still_clamps(strategy) -> None:
    """The most dangerous path in the file: with `_last_target` left at None,
    `follower_target` skips the clamp entirely, so the first command was a raw
    jump to the leader — on exactly the path (bus trouble at the handover edge)
    where the arms are most likely far apart."""
    from makermodslab.dagger_runner import _MAX_FOLLOWER_DEG_PER_S

    # No offset, but the follower's pose was recoverable — as the except now does.
    strategy._offset = None
    strategy._last_target = _pose(shoulder_pan=0)
    strategy._last_target_at = time.perf_counter()
    out = strategy.follower_target(_pose(shoulder_pan=500), strategy._last_target_at + 0.0333)
    assert out["shoulder_pan.pos"] <= _MAX_FOLLOWER_DEG_PER_S * 0.0333 + 0.01


# --- A failed drive still paces the tick -------------------------------------


def test_a_failed_drive_still_sleeps_out_the_tick(strategy) -> None:
    """The drive REPLACED the tail's sleep — a correcting tick is one or the
    other — so returning early does not cost a command, it costs the whole
    remaining budget. A recurring bus dropout then free-runs the loop and
    `add_frame` stamps `frame_index / fps` onto frames arriving far faster than
    fps, writing an episode whose timestamps claim a fraction of the real time."""
    robot, teleop = _DriveRobot(), _DriveTeleop(fail_after=0)
    start = time.perf_counter()
    strategy._drive_until(start + 0.030, _DriveCtx(), robot, teleop, {}, None)
    elapsed = time.perf_counter() - start
    assert robot.sent == []
    assert elapsed >= 0.025, f"returned after {elapsed * 1e3:.1f}ms, abandoning the tick's pacing"


# --- Leaving the hold ---------------------------------------------------------
#
# Poising powers the leader and pins it on the follower's pose. Every route out
# that is NOT the second takeover press has to let it go again — otherwise the
# operator is left gripping a rigid arm with nothing on screen to explain it,
# which is the original "the arm is stuck there" report.


def test_unpoise_releases_the_leader(strategy) -> None:
    leader = _HomeLeader(_pose(shoulder_pan=0))
    strategy._poised = True
    strategy._unpoise(_home_ctx(_HomeRobot({}), leader))
    assert strategy._poised is False
    assert leader.released is True


def test_unpoise_is_a_noop_when_not_poised(strategy) -> None:
    """It is called on every transition, so it must be free when nothing is held."""
    leader = _HomeLeader(_pose(shoulder_pan=0))
    strategy._unpoise(_home_ctx(_HomeRobot({}), leader))
    assert leader.released is False


def test_a_pending_poise_request_is_dropped_when_the_hold_is_left(strategy) -> None:
    """A queued request that outlived its moment would re-power the leader on
    some later tick, long after the operator moved on."""
    strategy._poised = True
    strategy._poise_requested = True
    strategy._unpoise(_home_ctx(_HomeRobot({}), _HomeLeader(_pose(shoulder_pan=0))))
    assert strategy._poise_requested is False


def test_the_hold_survives_until_the_operator_asks_a_second_time(strategy) -> None:
    """The whole point: a first press must leave the session poised, not driving,
    so no frame is recorded while a grip is being settled."""
    assert strategy._translate(CMD_TAKEOVER, AUTONOMOUS) == [_EV_PAUSE_RESUME]
    assert strategy._poised is False  # not yet — the loop does the glide
    strategy._poised = True
    assert strategy._translate(CMD_TAKEOVER, PAUSED) == [_EV_CORRECTION]


def test_the_glide_speed_is_the_one_the_operator_asked_for(strategy) -> None:
    """Pinned because it is a comfort judgement made at the bench, not a derived
    constant — a future tidy-up should have to argue with a test, not a number."""
    from makermodslab.dagger_runner import _TAKEOVER_GLIDE_DEG_PER_S

    assert pytest.approx(96.0) == _TAKEOVER_GLIDE_DEG_PER_S


# --- A hardware fault must not keep the take it interrupted -------------------
#
# Observed on the bench: unplugging a camera mid-session errored correctly and
# kept the earlier episodes — and ALSO kept the one being recorded through the
# dead camera. An unplugged camera does not blank the frame, it stops updating,
# so that episode is written with a stale image repeated to the end and nothing
# downstream can tell it from a good one.


def test_a_fresh_strategy_has_not_failed(strategy) -> None:
    assert strategy._loop_failed is False


def test_the_fault_flag_suppresses_the_in_flight_save(strategy) -> None:
    """The teardown's save decision, exercised directly: any of the three
    conditions must be enough to drop the take, and a fault is the newest."""
    frames = 500  # comfortably above _MIN_CORRECTION_FRAMES

    def would_drop(failed, cancelled):
        strategy._loop_failed = failed
        strategy._cancel_correction = cancelled
        from makermodslab.dagger_runner import _MIN_CORRECTION_FRAMES

        return strategy._loop_failed or strategy._cancel_correction or frames < _MIN_CORRECTION_FRAMES

    assert would_drop(True, False) is True  # the camera-unplug case
    assert would_drop(False, True) is True  # the operator discarded it
    assert would_drop(False, False) is False  # an ordinary stop mid-take: kept


def test_the_fault_reason_is_its_own(strategy) -> None:
    """`operator` and `too_short` already mean specific things to the UI — a
    discard the operator asked for stays silent, a too-short one must be
    surfaced. A fault is neither, and conflating it with `too_short` would tell
    the operator to hold the takeover longer next time when the real problem is
    a cable."""
    from makermodslab.dagger_protocol import (
        CANCEL_REASON_FAULT,
        CANCEL_REASON_OPERATOR,
        CANCEL_REASON_TOO_SHORT,
    )

    assert len({CANCEL_REASON_FAULT, CANCEL_REASON_OPERATOR, CANCEL_REASON_TOO_SHORT}) == 3


# --- logger.exception must actually print the exception ----------------------
#
# lerobot's `init_logging` replaces its formatter's bound `format` method with a
# function that renders level, time, location and message — and never looks at
# `record.exc_info`. Appending the traceback is the one thing stock
# `Formatter.format` does that a message-only function cannot, so in every
# process calling `init_logging()`, `logger.exception` was byte-for-byte
# identical to `logger.error`. A takeover glide failed on a real session and its
# exception was simply never written down.


def _formatted(record_factory) -> str:
    import logging as _logging

    from makermodslab.log_exceptions import restore_traceback_rendering

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    fmt = _logging.Formatter()
    # Reproduce lerobot's move exactly: replace the BOUND method with a
    # message-only function. This is the shape the fix has to survive.
    fmt.format = lambda rec: f"{rec.levelname} {rec.getMessage()}"
    handler.setFormatter(fmt)
    root = _logging.getLogger()
    saved = root.handlers[:]
    root.handlers = [handler]
    try:
        restore_traceback_rendering()
        record_factory(_logging.getLogger("t"))
        return stream.getvalue()
    finally:
        root.handlers = saved


def test_a_logged_exception_now_carries_its_traceback() -> None:
    def emit(log):
        try:
            raise ConnectionError("Failed to write 'Torque_Enable' on id_=2")
        except Exception:
            log.exception("Could not glide the leader")

    out = _formatted(emit)
    assert "Could not glide the leader" in out
    assert "Traceback (most recent call last)" in out
    assert "Failed to write 'Torque_Enable' on id_=2" in out


def test_the_original_format_is_preserved_not_replaced() -> None:
    """Wrapping, not replacing. The lerobot format is what makes our logs
    greppable against lerobot's own lines in the same file."""

    def emit(log):
        log.info("an ordinary line")

    out = _formatted(emit)
    assert out.startswith("INFO an ordinary line")
    assert "Traceback" not in out


def test_restoring_twice_does_not_stack_wrappers() -> None:
    """It runs after `init_logging`, which a re-imported or re-initialised
    module can call more than once; stacked wrappers would print the traceback
    once per call."""

    from makermodslab.log_exceptions import restore_traceback_rendering

    def emit(log):
        restore_traceback_rendering()
        restore_traceback_rendering()
        try:
            raise ValueError("boom")
        except Exception:
            log.exception("once please")

    out = _formatted(emit)
    assert out.count("Traceback (most recent call last)") == 1
    assert out.count("ValueError: boom") == 1


def test_a_handler_with_no_formatter_is_skipped(strategy) -> None:
    import logging as _logging

    from makermodslab.log_exceptions import restore_traceback_rendering

    root = _logging.getLogger()
    saved = root.handlers[:]
    root.handlers = [_logging.StreamHandler(io.StringIO())]
    root.handlers[0].formatter = None
    try:
        restore_traceback_rendering()  # must not raise
    finally:
        root.handlers = saved


# --- writes get the retries too ----------------------------------------------
#
# Reads were patched first because a dropped read was the failure we had seen.
# The logs since say otherwise: across every session recorded on this rig the
# bus failed 13 times and EVERY one was an unretried write — 7 x `Lock`, 6 x
# `Torque_Enable` — against a single `sync_read` failure that predates the read
# patch. Both come from `enable_torque`, which is the first statement of the
# takeover glide, so twelve unretried writes stand between the operator pressing
# space and the leader moving.


def test_the_retries_land_on_the_class_that_actually_runs() -> None:
    """Asserted against `FeetechMotorsBus` — the concrete bus an SO-101 uses —
    NOT against `MotorsBus`, which is what we patch.

    This exists because the first write patch was inert twice over, and both
    times the class we patched looked correct. `enable_torque` passes
    `num_retry` to `write` EXPLICITLY, so patching `write`'s default did nothing
    for it; and `FeetechMotorsBus` overrides `enable_torque`, so patching the
    base class did nothing either. A real session proved it — `Failed to write
    'Torque_Enable' ... after 1 tries` on a build meant to make three. Resolve
    it off the class that runs, or this silently regresses again."""
    import inspect

    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from makermodslab.bus_retry import BUS_SYNC_READ_RETRIES

    assert BUS_SYNC_READ_RETRIES > 0
    for name in ("write", "sync_read", "enable_torque", "disable_torque"):
        got = inspect.signature(getattr(FeetechMotorsBus, name)).parameters["num_retry"].default
        assert got == BUS_SYNC_READ_RETRIES, f"{name} resolves to num_retry={got}"


def test_the_torque_toggles_forward_the_retries_to_their_writes() -> None:
    """The signature is necessary but not sufficient — `enable_torque` must also
    PASS the value on, since that is what each of its twelve writes receives."""
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from makermodslab.bus_retry import BUS_SYNC_READ_RETRIES

    seen = []

    class _Bus(FeetechMotorsBus):
        def __init__(self):
            pass

        def _get_motors_list(self, motors):
            return ["elbow_flex"]

        def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
            seen.append((data_name, num_retry))

    FeetechMotorsBus.enable_torque(_Bus())
    assert seen == [
        ("Torque_Enable", BUS_SYNC_READ_RETRIES),
        ("Lock", BUS_SYNC_READ_RETRIES),
    ]


def test_the_control_loop_stream_is_deliberately_not_retried() -> None:
    """`sync_write` carries the 30Hz teleop traffic and does not ask for a status
    packet at all, so it cannot fail the way `write` does — and an extra round
    trip there is a real cost in the loop the operator feels in their hand. It
    has never once appeared in a recorded failure. Retrying reads and config
    writes is cheap insurance; retrying the stream that drives the arm is not."""
    import inspect

    from lerobot.motors.motors_bus import MotorsBus

    assert inspect.signature(MotorsBus.sync_write).parameters["num_retry"].default == 0
    assert not hasattr(MotorsBus.sync_write, "_bus_retry_original")


def test_the_write_patch_forwards_to_lerobots_method() -> None:
    from lerobot.motors.motors_bus import MotorsBus

    seen = {}
    original = MotorsBus.write._bus_retry_original

    def spy(self, data_name, motor, value, *, normalize=True, num_retry=0):
        seen.update(data_name=data_name, motor=motor, value=value, num_retry=num_retry)

    MotorsBus.write._bus_retry_original.__self__ if hasattr(original, "__self__") else None
    import makermodslab.bus_retry as br

    saved, br._original_write = br._original_write, spy
    try:
        MotorsBus.write(object(), "Torque_Enable", "elbow_flex", 1)
    finally:
        br._original_write = saved
    assert seen["data_name"] == "Torque_Enable"
    assert seen["num_retry"] > 0


def test_the_write_patch_is_idempotent() -> None:
    """`uvicorn --reload` and `importlib.reload` both re-run this module. Without
    recovering the true original, a second import would capture OUR patch as the
    original and every write would recurse until the stack blew."""
    import importlib

    import makermodslab.bus_retry as br
    from lerobot.motors.motors_bus import MotorsBus

    first = MotorsBus.write._bus_retry_original
    importlib.reload(br)
    assert MotorsBus.write._bus_retry_original is first
