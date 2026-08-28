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

"""Browser-driven DAgger runner for MakerMods Lab's coaching sessions.

    python -m makermodslab.dagger_runner <exactly the lerobot-rollout argv>

What a coaching session is
--------------------------
The policy drives the follower. The operator watches, and the moment it is about
to fail, takes over with the leader arm, teleoperates the recovery and the
correction, then hands control back. Only the human-driven windows are recorded,
each as one episode. Fine-tuning on that dataset targets the failure modes of
the exact checkpoint that produced it — the states a blind demonstration never
visits. (HG-DAgger, Kelly et al. 2019; the recovery/correction split is RaC,
Hu et al. 2025.)

Why this module exists
----------------------
lerobot v0.6.0 already implements all of that in `DAggerStrategy`. What it does
NOT have is a way to drive it from anything but a keyboard or a USB foot pedal
plugged into the machine running the arm. `DAggerStrategy.setup` installs a
`create_key_listener`, which on a headless host falls back to reading the
terminal's stdin — the very pipe this runner needs as its command channel. So
the listener is not merely unhelpful here, it is actively in the way.

This runner therefore subclasses the upstream strategy, drops the listener, and
feeds the same thread-safe `DAggerEvents` inbox from stdin instead. Everything
that touches hardware — the phase transitions, the smooth leader handover, the
dataset writes — is upstream's code, reached through inheritance, so an upstream
fix to any of it lands here too.

What is vendored, and why
-------------------------
One method is copied rather than inherited: `_run_corrections_only`, the control
loop. Three behaviours we need are not expressible from outside it:

  1. **Cancel.** Upstream saves EVERY correction, unconditionally, at the
     CORRECTING→PAUSED edge. A fumbled takeover — grabbed the leader badly, the
     gripper caught, corrected the wrong thing — is poison training data that
     the operator has no way to reject. The vendored loop calls
     `clear_episode_buffer()` instead of `save_episode()` when cancel is armed.
  2. **The alignment gate** (see `_alignment_error`).
  3. **Protocol events** at the phase edges, so the browser can show what the
     arm is actually doing rather than poll for it.

Vendoring one loop is this repo's established answer to exactly this problem:
`record.py`'s `record_with_web_events` is lerobot's record loop, copied, with a
web-events dict where the keystrokes were. It beats monkey-patching the dataset
object or the transition table, because the seam is visible in one place instead
of three. The cost is real and worth naming: **when the lerobot pin moves, diff
`DAggerStrategy._run_corrections_only` against `_run_corrections` below.**

Not implemented here, deliberately: continuous recording
(`--strategy.record_autonomous=true`). The orchestrator always passes false —
corrections-only is the mode whose dataset is entirely human-driven, and the
merge shortcut in `merge.py` depends on that being true.
"""

# NO `from __future__ import annotations` here, deliberately: lerobot's
# parser.wrap() reads run()'s RAW annotation (getfullargspec, not
# get_type_hints) to find the draccus config class. PEP 563 would turn it
# into the string 'RolloutConfig' and draccus dies with "must be called
# with a dataclass type or instance". Same reason no lerobot script uses it
# (and the same reason eval_runner.py doesn't).

import contextlib
import faulthandler
import logging
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers --robot.cameras type)
from lerobot.common.control_utils import teleop_smooth_move_to, teleop_supports_feedback
from lerobot.configs import parser
from lerobot.datasets import VideoEncodingManager
from lerobot.robots import (  # noqa: F401  (registers --robot.type=so101_follower / bi_so_follower)
    bi_so_follower,
    so_follower,
)
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.configs import DAggerStrategyConfig
from lerobot.rollout.strategies.core import send_next_action
from lerobot.rollout.strategies.dagger import DAggerPhase, DAggerStrategy
from lerobot.teleoperators import (  # noqa: F401  (registers --teleop.type=so101_leader / bi_so_leader)
    bi_so_leader,
    so_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

# Imported for its SIDE EFFECT: a non-zero sync_read retry default for this
# process. Without it a single dropped serial reply — routine when arm
# adapters share a USB bus with streaming cameras — kills the session with
# "[TxRxResult] There is no status packet!". See makermodslab/bus_retry.py.
from . import bus_retry  # noqa: F401
from .dagger_protocol import (
    CMD_CANCEL,
    CMD_HANDBACK,
    CMD_HOLD,
    CMD_QUIT,
    CMD_RESET,
    CMD_RESUME,
    CMD_TAKEOVER,
    EVENT_ALIGN_REQUIRED,
    EVENT_ATTEMPT_RESET,
    EVENT_BYE,
    EVENT_CORRECTION_CANCELLED,
    EVENT_CORRECTION_SAVED,
    EVENT_DATASET,
    EVENT_ERROR,
    EVENT_PHASE,
    EVENT_READY,
    PHASE_HANDING_OVER,
    PHASE_RESETTING,
    PHASE_SAVING,
    format_event,
)

logger = logging.getLogger(__name__)

# Upstream's own transition-request names, as used by `DAggerEvents`. Ours are
# the operator-facing composites; these are the primitives they expand into.
_EV_PAUSE_RESUME = "pause_resume"
_EV_CORRECTION = "correction"

# How far (degrees, worst joint) the leader may sit from the follower before a
# takeover is refused on a NON-ACTUATED teleop. See `_alignment_error` for why
# only that case is gated, and why the number is what it is.
_ALIGN_TOLERANCE_DEG = 15.0

# Shortest correction worth keeping, in recorded frames (~0.33s at 30fps).
#
# Below this a "correction" is never a deliberate demonstration — it is a
# double-press, or the buffer that happened to exist when the session died. Two
# observed failures make discarding them mandatory rather than tidy:
#
#   * a crash mid-correction left 1 frame, which the teardown saved and counted,
#     so the operator was credited a correction they never made;
#   * a ONE-FRAME episode then breaks lerobot's stats aggregation outright —
#     "ArrowInvalid: Column 95 named stats/observation.images.front/min expected
#     length 2 but got length 1" — which killed the session and left the dataset
#     with a malformed episode.
#
# Anything a human actually meant to demonstrate is at least a second long.
_MIN_CORRECTION_FRAMES = 10

# Logged (not emitted) once setup is behind us, purely so the familiar lerobot
# landmark still appears in the inference log a human — or a future grep — goes
# looking for. Mirrors eval_runner's `_SETUP_COMPLETE_LOG` and the orchestrator's
# `_ROLLOUT_START_MARKER`; the orchestrator does NOT key any phase off it here,
# because PHASE events say what the session is doing far more precisely.
_SETUP_COMPLETE_LOG = "Rollout setup complete"


# How long `save_episode()` may run before we dump every thread's stack.
#
# Not a timeout — nothing is cancelled — purely a diagnostic. A correction of
# 132 frames x 2 cameras measures 0.4-2.3s on the station, so 30s means
# something is genuinely stuck rather than slow. A hang here was observed once
# (0% CPU, PNGs undrained, no video written) and could not be explained from
# the log afterwards, because a blocked thread writes nothing. The dump turns a
# silent wedge into a stack trace naming the exact call.
_SAVE_WATCHDOG_S = 30.0


@contextlib.contextmanager
def _watchdog(seconds: float, label: str):
    """Dump all thread stacks to the log if the block takes longer than `seconds`.

    `faulthandler.dump_traceback_later` fires from a C-level timer thread, so it
    still works when the GIL is held by a thread stuck in a native call — which
    is exactly the case a pure-Python watchdog would miss."""
    logger.info("Starting %s (stacks dump at %.0fs if it hangs)", label, seconds)
    faulthandler.dump_traceback_later(seconds, exit=False)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _emit(event: str, payload: str = "") -> None:
    """Write one protocol event to stdout, flushed.

    stdout is a pipe here, so it is block-buffered by default — without the
    flush the orchestrator would not see a phase change until 4-8 KB of
    unrelated log had accumulated behind it. For eval that meant a late episode
    boundary; here it would mean the browser showing "watching" while the
    operator is already driving the arm, which is a safety-relevant lie."""
    print(format_event(event, payload), flush=True)


def _read_commands(commands: queue.Queue, shutdown_event: threading.Event) -> None:
    """Read the command protocol off stdin until EOF (runs on its own thread).

    QUIT is acted on HERE as well as queued, because its whole point is to
    interrupt a session whose main thread is parked inside lerobot's control
    loop. Setting the shutdown event breaks that loop on its next tick; the
    queued copy is what stops the command drain from serving anything after it.

    Started only AFTER the robot is connected: `SOFollower.calibrate()` prompts
    with `input()` during `connect()`, reading this very stdin, and the
    orchestrator pre-seeds a newline per ARM (leaders included — a coaching
    session connects both sides, unlike every other inference run) to answer it.
    Racing that prompt with this reader would let a real command be eaten by the
    prompt, or a seed newline be read as a command. Any seed newline the prompt
    didn't consume arrives here as a blank line and is ignored."""
    try:
        for raw in sys.stdin:
            cmd = raw.strip().upper()
            if not cmd:
                continue
            if cmd == CMD_QUIT:
                logger.info("QUIT received — ending the coaching session")
                shutdown_event.set()
            commands.put(cmd)
    except Exception:
        logger.exception("DAgger runner stdin reader failed")
    finally:
        # EOF (the orchestrator died or closed the pipe) must not leave a
        # session driving the arm with nobody watching.
        logger.info("Command stdin closed — shutting the session down")
        shutdown_event.set()
        commands.put(None)


class WebDAggerStrategy(DAggerStrategy):
    """`DAggerStrategy` driven by a line protocol instead of a keyboard.

    Subclassed rather than reimplemented: `setup`'s engine init, `teardown`'s
    dataset finalize + hardware release, `_apply_transition`'s physical
    handovers and `_background_push` are all inherited untouched.
    """

    def __init__(self, config: DAggerStrategyConfig) -> None:
        super().__init__(config)
        self._commands: queue.Queue = queue.Queue()
        # Upstream transition names still owed to the command we are executing.
        # A composite (TAKEOVER / HANDBACK) puts two here; the loop applies ONE
        # per control tick so each transition's hardware side-effects — the ~2s
        # leader glide, the torque flip — complete before the next is requested.
        self._pending: deque = deque()
        # Armed by CANCEL, consumed at the CORRECTING→PAUSED edge.
        self._cancel_correction = False
        self._corrections_saved = 0
        # Armed by RESET, consumed once the loop reaches PAUSED. The ease-home
        # can only run with the engine stopped, and getting to PAUSED is itself
        # a transition, so the request has to outlive the tick that made it.
        self._reset_requested = False
        self._attempts = 0
        # True while the follower is deliberately unpowered so the operator can
        # reposition it by hand during a reset. Everything that could move the
        # arm has to restore torque first — see `_restore_torque`.
        self._limp = False
        # Set by a reset: the leader was never driven to the follower (a reset
        # is not a handover), so the next takeover must align it first or the
        # follower would snap to wherever the leader happens to be.
        self._needs_leader_align = False

    # -- setup: everything upstream does, minus the input listener -----------

    def setup(self, ctx) -> None:
        """Arm the engine and the push executor WITHOUT an input listener.

        Deliberately does not call `super().setup()`. That method's last act is
        to install a keyboard or pedal listener, and `create_key_listener`'s
        headless backend reads the process's stdin — the same pipe this runner
        serves commands on. There is no upstream flag to suppress it (the
        `input_device` field accepts only "keyboard" or "pedal"), so the four
        lines that precede it are reproduced here instead.

        Kept deliberately parallel to upstream's body so the diff is obvious on
        a pin bump; see the module docstring.
        """
        from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
        from lerobot.rollout.strategies.core import estimate_max_episode_seconds

        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        # Unused in corrections-only mode (episodes end at HANDBACK, not on a
        # size-based rotation), but the attribute is part of the base class's
        # contract and teardown paths read it.
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )
        logger.info(
            "Coaching session ready (browser-driven, corrections only, target=%s)",
            self.config.num_episodes,
        )

    # -- command translation -------------------------------------------------

    def submit(self, command: str) -> None:
        """Hand one protocol command to the control loop (called off-thread)."""
        self._commands.put(command)

    def _translate(self, command: str, phase: DAggerPhase) -> list:
        """Expand one operator command into upstream transition requests.

        The expansion depends on the phase at the moment of translation, which
        is why it happens on the control-loop thread rather than in the stdin
        reader: a command written while the arm was still gliding into position
        would otherwise be expanded against a phase that had already moved on.

        A command that makes no sense in the current phase expands to nothing —
        the browser and the arm can disagree for a tick or two (a click races a
        transition), and the correct answer to that is to do nothing, not to
        force a transition the state machine considers invalid.
        """
        if command == CMD_TAKEOVER:
            if phase == DAggerPhase.AUTONOMOUS:
                return [_EV_PAUSE_RESUME, _EV_CORRECTION]
            if phase == DAggerPhase.PAUSED:
                return [_EV_CORRECTION]
        elif command in (CMD_HANDBACK, CMD_CANCEL):
            if phase == DAggerPhase.CORRECTING:
                # CANCEL stops at PAUSED: a discarded correction usually means
                # something went wrong, and dropping the operator straight back
                # into an autonomous policy is the wrong default when they have
                # just told us the last few seconds were a mess.
                self._cancel_correction = command == CMD_CANCEL
                return [_EV_CORRECTION] if command == CMD_CANCEL else [_EV_CORRECTION, _EV_PAUSE_RESUME]
        elif command == CMD_HOLD:
            if phase == DAggerPhase.AUTONOMOUS:
                return [_EV_PAUSE_RESUME]
        elif command == CMD_RESUME:
            if phase == DAggerPhase.PAUSED:
                return [_EV_PAUSE_RESUME]
        elif command == CMD_RESET:
            # Deliberately ignored mid-correction: the operator is holding the
            # leader and has frames in the buffer, and silently discarding or
            # saving those on their behalf is not a call this should make. Hand
            # back or discard first, then reset.
            if phase == DAggerPhase.CORRECTING:
                logger.info("Ignoring RESET during a correction — hand back or discard first")
                return []
            self._reset_requested = True
            # Deliberately requests NO transition. Routing a reset through
            # `pause_resume` made `_apply_transition` treat it as "the operator
            # is about to grab the leader" and drive the LEADER up to the
            # follower's pose under torque — then, when the policy resumed, that
            # same function released the leader and it fell from mid-air.
            # A reset is not a handover. The loop pauses the engine itself.
            return []
        else:
            logger.warning("Ignoring unrecognised coaching command %r", command)
            return []
        logger.info("Ignoring %s — not valid from phase %s", command, phase.value)
        return []

    def _drain_commands(self, events) -> None:
        """Feed at most one transition request into `events` this tick."""
        if not self._pending:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command is None:
                return
            self._pending.extend(self._translate(command, events.phase))
        if self._pending:
            events.request_transition(self._pending.popleft())

    @staticmethod
    def _transition_moves_the_arm(old_phase, new_phase, ctx) -> bool:
        """True when `_apply_transition` is about to drive hardware for ~2s.

        Mirrors upstream's own branching in `_apply_transition`, which picks its
        handover by whether the teleop is actuated:

          * actuated (single SO-101 leader): AUTONOMOUS→PAUSED glides the LEADER
            to the follower's pose;
          * non-actuated (BiSOLeader on this pin): PAUSED→CORRECTING slides BOTH
            FOLLOWERS to meet the leaders.

        Every other edge is a torque flip or an engine reset — fast enough that
        announcing travel would be its own small lie.
        """
        actuated = teleop_supports_feedback(ctx.hardware.teleop)
        if actuated:
            return old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED
        return old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING

    # -- limp / torque for the reset window ----------------------------------

    @staticmethod
    def _follower_buses(robot_wrapper):
        """Every serial bus behind the follower — two for a bimanual BiSO.

        Mirrors `arm_identity._device_arms`: a BiSO robot exposes `left_arm` /
        `right_arm` sub-devices that each own a bus, a single-arm robot owns one
        directly. Reaches through `ThreadSafeRobot.inner` because torque lives
        on the raw device, not the wrapper."""
        robot = getattr(robot_wrapper, "inner", robot_wrapper)
        arms = [
            a for a in (getattr(robot, "left_arm", None), getattr(robot, "right_arm", None)) if a is not None
        ]
        return [bus for d in (arms or [robot]) if (bus := getattr(d, "bus", None)) is not None]

    def _go_limp(self, ctx) -> bool:
        """Unpower the follower so it can be repositioned by hand. True if it went.

        Callers MUST have driven the arm somewhere safe first. An SO-101 that
        loses torque mid-reach falls under its own weight, so this is only ever
        called after a successful ease-home — the same order `rest_pose` states
        as its contract ("torque must still be enabled; call BEFORE
        force_disable_torque")."""
        buses = self._follower_buses(ctx.hardware.robot_wrapper)
        if not buses:
            logger.warning("No follower bus found; leaving the arm powered")
            return False
        for bus in buses:
            try:
                bus.disable_torque()
            except Exception:
                logger.exception("Could not release torque; the arm stays powered")
                return False
        self._limp = True
        logger.info("Follower is limp — reposition by hand")
        return True

    def _restore_torque(self, ctx):
        """Re-power the follower where the operator left it. Returns that pose.

        The goal position is written FIRST, while the motors are still limp.
        A Feetech servo holds its Goal_Position register the instant torque
        comes back, and that register still contains the pre-limp target — so
        enabling torque without this would snap the arm from wherever the
        operator moved it to wherever the policy last wanted it."""
        if not self._limp:
            return None
        robot = ctx.hardware.robot_wrapper
        hold = None
        try:
            obs = robot.get_observation()
            hold = {k: v for k, v in obs.items() if k.endswith(".pos")}
            if hold:
                robot.send_action(hold)
        except Exception:
            logger.exception("Could not read the arm's pose before re-powering")
        for bus in self._follower_buses(robot):
            with contextlib.suppress(Exception):
                bus.enable_torque()
        self._limp = False
        logger.info("Follower re-powered at its current pose")
        return hold

    # -- the alignment gate --------------------------------------------------

    def _alignment_error(self, ctx):
        """`(max_delta_deg, ["joint:delta", …])` when a takeover is unsafe, else None.

        Only NON-ACTUATED teleops are gated, and on this lerobot pin that means
        exactly one configuration: bimanual. `SO101Leader.feedback_features`
        returns its action features, so a single-arm session takes upstream's
        actuated path — pausing drives the LEADER to the follower's pose and the
        operator picks up an arm already sitting where the robot is.
        `BiSOLeader.feedback_features` returns `{}` (bi_so_leader.py:69), so
        bimanual takes the opposite path: the FOLLOWERS glide to wherever the
        leaders happen to be. That glide is smooth but it is not short — leaders
        parked on the bench mean both arms sweep across the workspace, through
        whatever the task involves, the instant the operator takes over.
        Upstream PR #4028 gives BiSOLeader feedback support, but it landed after
        v0.6.0 and this repo pins the release tag.

        So: measure first, refuse if the trip is long, and tell the operator
        which joints to move. The tolerance is a workspace judgement, not a
        derived constant — 15° is comfortably more than the slop of parking the
        leaders by hand and comfortably less than a reach across the table.

        Reads the bus from the control-loop thread (the caller is the loop), the
        same thread and the same two calls upstream makes one line later in
        `_apply_transition`; it must never be called from the stdin reader.
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper
        if teleop_supports_feedback(teleop):
            return None
        obs = robot.get_observation()
        teleop_action = teleop.get_action()
        processed = ctx.processors.teleop_action_processor((teleop_action, obs))
        target = ctx.processors.robot_action_processor((processed, obs))

        offenders = []
        worst = 0.0
        for key, want in target.items():
            have = obs.get(key)
            if have is None or not isinstance(want, (int, float)) or not isinstance(have, (int, float)):
                continue
            delta = abs(float(want) - float(have))
            worst = max(worst, delta)
            if delta > _ALIGN_TOLERANCE_DEG:
                # Strip the ".pos" suffix the action keys carry so the browser
                # can print "shoulder_pan" rather than "shoulder_pan.pos".
                offenders.append(f"{key.removesuffix('.pos')}:{delta:.0f}")
        if not offenders:
            return None
        return worst, offenders

    # -- run -----------------------------------------------------------------

    def run(self, ctx) -> None:
        """Serve a corrections-only session. Continuous mode is not supported."""
        if self.config.record_autonomous:
            raise ValueError(
                "record_autonomous=true is not supported by the coaching runner; "
                "the orchestrator always passes false (see the module docstring)"
            )
        self._run_corrections(ctx)

    def _run_corrections(self, ctx) -> None:
        """Vendored `DAggerStrategy._run_corrections_only`, browser-driven.

        Differences from upstream, all of them called out at their site below:
        commands replace keystrokes, the CORRECTING→PAUSED edge can discard
        instead of save, PAUSED→CORRECTING is gated on leader alignment, and
        every phase edge emits a protocol event. The autonomous and correcting
        control paths themselves are upstream's, line for line — that is the
        part that drives the arm, and it should stay diffable.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action = None
        record_tick = 0
        correction_frames = 0
        correction_started_at = 0.0
        target = self.config.num_episodes
        _emit(EVENT_PHASE, f"phase={DAggerPhase.AUTONOMOUS.value}")
        logger.info("Coaching session started (target: %s corrections)", target)

        with VideoEncodingManager(dataset):
            try:
                while (
                    self._corrections_saved < target
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    loop_start = time.perf_counter()

                    # MakerMods Lab: browser commands where upstream reads a key
                    # listener. One transition per tick — see `_pending`.
                    self._drain_commands(events)

                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition

                        # MakerMods Lab: refuse a takeover that would drag the
                        # followers across the workspace. Reverting the phase is
                        # safe here and only here — `_apply_transition` has not
                        # run, so no torque flip or motion has been committed.
                        if old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
                            misalignment = self._alignment_error(ctx)
                            if misalignment is not None:
                                worst, offenders = misalignment
                                events.phase = DAggerPhase.PAUSED
                                self._pending.clear()
                                self._cancel_correction = False
                                logger.warning(
                                    "Takeover refused — leader is %.0f deg from the follower (%s)",
                                    worst,
                                    ", ".join(offenders),
                                )
                                _emit(
                                    EVENT_ALIGN_REQUIRED,
                                    f"max_delta={worst:.0f} joints={','.join(offenders)}",
                                )
                                transition = None

                    if transition is not None:
                        # Anything past here can move the arm, so it must be
                        # powered again first — including the handover glide,
                        # which would otherwise drive a limp follower.
                        restored = self._restore_torque(ctx)
                        if restored:
                            last_action = restored
                        old_phase, new_phase = transition
                        # Announce the travel BEFORE the blocking call. Both
                        # smooth-handover paths inside `_apply_transition` drive
                        # a physical arm for ~2s, and until this existed the
                        # banner spent that entire window asserting the opposite
                        # of what the hardware was doing (see PHASE_HANDING_OVER).
                        # stdout is flushed per event, so the orchestrator reads
                        # this while the control-loop thread is still blocked.
                        # After a reset the leader was never brought to the
                        # follower, so do it now — otherwise the first
                        # CORRECTING tick sends the follower straight to
                        # wherever the operator left the leader.
                        aligning = (
                            self._needs_leader_align
                            and new_phase == DAggerPhase.CORRECTING
                            and teleop_supports_feedback(ctx.hardware.teleop)
                        )
                        if self._transition_moves_the_arm(old_phase, new_phase, ctx) or aligning:
                            _emit(EVENT_PHASE, f"phase={PHASE_HANDING_OVER}")
                        if aligning:
                            self._needs_leader_align = False
                            obs = ctx.hardware.robot_wrapper.get_observation()
                            here = {k: v for k, v in obs.items() if k.endswith(".pos")}
                            logger.info("Aligning the leader to the follower after a reset")
                            with contextlib.suppress(Exception):
                                teleop_smooth_move_to(ctx.hardware.teleop, here)
                        self._apply_transition(old_phase, new_phase, engine, interpolator, ctx, last_action)
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                            # Back under policy control: the leader is released
                            # and free, so there is no alignment debt to carry.
                            self._needs_leader_align = False

                        if old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
                            correction_frames = 0
                            correction_started_at = time.perf_counter()

                        # Correction ended. Upstream saves unconditionally; we
                        # honour a CANCEL armed by the operator instead.
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            # Announce the write BEFORE it starts. It is
                            # synchronous and can be slow (video encoding), and
                            # the operator has already handed back — leaving the
                            # banner on "you're driving" for the duration is the
                            # same class of lie as the handover window.
                            if not self._cancel_correction:
                                _emit(EVENT_PHASE, f"phase={PHASE_SAVING}")
                            seconds = time.perf_counter() - correction_started_at
                            if self._cancel_correction:
                                self._cancel_correction = False
                                with self._episode_lock:
                                    dataset.clear_episode_buffer()
                                logger.info(
                                    "Correction discarded (%.1fs, %d frames)", seconds, correction_frames
                                )
                                _emit(EVENT_CORRECTION_CANCELLED)
                            elif correction_frames < _MIN_CORRECTION_FRAMES:
                                # Too short to be a demonstration — a
                                # double-press, or a hand-back inside a tick or
                                # two. Discarding is not just tidiness: a
                                # one-frame episode breaks lerobot's stats
                                # aggregation and takes the session with it
                                # (see _MIN_CORRECTION_FRAMES).
                                with self._episode_lock:
                                    dataset.clear_episode_buffer()
                                logger.info(
                                    "Discarded a %d-frame correction (too short to be deliberate)",
                                    correction_frames,
                                )
                                _emit(EVENT_CORRECTION_CANCELLED)
                            else:
                                with self._episode_lock, _watchdog(_SAVE_WATCHDOG_S, "save_episode"):
                                    dataset.save_episode()
                                self._corrections_saved += 1
                                self._needs_push.set()
                                logger.info(
                                    "Correction %d/%s saved (%.1fs, %d frames)",
                                    self._corrections_saved,
                                    target,
                                    seconds,
                                    correction_frames,
                                )
                                _emit(
                                    EVENT_CORRECTION_SAVED,
                                    f"n={self._corrections_saved} frames={correction_frames} "
                                    f"seconds={seconds:.1f}",
                                )

                        _emit(EVENT_PHASE, f"phase={new_phase.value}")

                    # MakerMods Lab: an attempt at the TASK is over. Corrections-only
                    # DAgger has no task-episode of its own, so without this the
                    # policy just kept driving at a finished scene. Runs only
                    # from PAUSED — the engine must be stopped before the arm
                    # moves — and blocks the loop for the duration of the glide,
                    # which is why it announces itself first.
                    if self._reset_requested and events.phase != DAggerPhase.CORRECTING:
                        self._reset_requested = False
                        self._attempts += 1
                        # Stop the policy WITHOUT going through
                        # `_apply_transition` — see the note in `_translate`.
                        # Setting the phase directly is safe here precisely
                        # because none of the transition side-effects apply: no
                        # handover, no torque flip on the leader.
                        engine.pause()
                        events.phase = DAggerPhase.PAUSED
                        _emit(EVENT_PHASE, f"phase={PHASE_RESETTING}")
                        logger.info("Attempt %d ended — easing the follower home", self._attempts)
                        homed = True
                        try:
                            self._return_to_initial_position(ctx.hardware)
                        except Exception:
                            # A failed ease-home must not end the session: the
                            # arm simply stays where it is and the operator can
                            # reposition the scene around it.
                            homed = False
                            logger.exception("Return-to-initial-position failed; leaving the arm in place")
                        # Limp ONLY from a pose we know is safe. If the ease-home
                        # did not arrive, the arm is somewhere arbitrary — quite
                        # possibly extended — and cutting torque there drops it.
                        if homed:
                            self._go_limp(ctx)
                        else:
                            logger.warning("Skipping limp: the arm is not at its home pose")
                        last_action = None
                        self._needs_leader_align = True
                        _emit(EVENT_ATTEMPT_RESET, f"n={self._attempts}")
                        _emit(EVENT_PHASE, f"phase={DAggerPhase.PAUSED.value}")

                    phase = events.phase
                    obs = robot.get_observation()

                    # --- CORRECTING: human teleop control + recording ---
                    if phase == DAggerPhase.CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        if record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            dataset.add_frame(
                                {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([True], dtype=bool),
                                }
                            )
                            correction_frames += 1
                        record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            robot.send_action(last_action)

                    # --- AUTONOMOUS: policy control (no recording) ---
                    else:
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Coaching loop is running slower ({1 / dt:.1f} Hz) than the target FPS "
                            f"({cfg.fps} Hz). Dataset frames might be dropped and robot control might be "
                            f"unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy "
                            f"inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("Coaching control loop ended — pausing engine")
                engine.pause()
                # Teardown eases the arm home and that needs torque; a session
                # stopped during a reset would otherwise disconnect limp,
                # leaving the arm wherever the operator's hand left it.
                with contextlib.suppress(Exception):
                    self._restore_torque(ctx)

                # A session stopped mid-correction still has frames in the
                # buffer, and those are real work: the operator was
                # demonstrating right up to the stop, so the correction is
                # saved and counted.
                #
                # UNLESS it is too short to be one. A session that DIES
                # mid-take — a serial dropout, say — leaves a frame or two
                # behind, and saving that both credited a correction nobody
                # made and wrote the one-frame episode that broke lerobot's
                # stats aggregation, taking the session down with it.
                #
                # Deliberately no `return` in either branch: returning from a
                # `finally` swallows the exception that brought us here, which
                # is exactly the crash the operator needs reported.
                if correction_frames < _MIN_CORRECTION_FRAMES:
                    if correction_frames:
                        logger.info(
                            "Dropping a %d-frame in-flight correction on shutdown (too short)",
                            correction_frames,
                        )
                    with contextlib.suppress(Exception):
                        dataset.clear_episode_buffer()
                else:
                    saved = False
                    try:
                        with self._episode_lock:
                            dataset.save_episode()
                        self._corrections_saved += 1
                        self._needs_push.set()
                        saved = True
                        logger.info("Final in-progress correction saved")
                    except Exception:
                        logger.exception("Could not save the in-flight correction")
                    # Emitted only on a save that actually happened. The
                    # orchestrator's tally comes solely from these events, so a
                    # save that occurs here and nowhere else would otherwise
                    # leave the summary one short — the ordinary outcome of
                    # pressing Stop while driving.
                    if saved:
                        _emit(
                            EVENT_CORRECTION_SAVED,
                            f"n={self._corrections_saved} frames={correction_frames} "
                            f"seconds={time.perf_counter() - correction_started_at:.1f}",
                        )


@parser.wrap()
def run(cfg: RolloutConfig) -> None:
    """Load once, connect once, then serve coaching commands off stdin.

    Takes the identical argv `lerobot-rollout` takes (the orchestrator builds
    one arg list and points it at either entry point), so `--strategy.*`,
    `--policy.*`, `--robot.*`, `--teleop.*` and `--dataset.*` all mean exactly
    what they mean there.
    """
    init_logging()

    if not isinstance(cfg.strategy, DAggerStrategyConfig):
        raise ValueError(f"The coaching runner requires --strategy.type=dagger, got {cfg.strategy.type!r}")

    # Same handler lerobot's own CLI installs: a SIGTERM (the orchestrator's
    # fallback when a QUIT goes unanswered) sets an event rather than killing us
    # mid-motion, so the arm still eases home.
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building rollout context (policy, robot and leader load ONCE for the session)...")
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = WebDAggerStrategy(cfg.strategy)

    # AFTER build_rollout_context, never before: lerobot stamps a timestamp onto
    # the repo_id in there, so this is the first moment the name on disk exists.
    # The orchestrator has no other way to learn it (issue #3722 closed without
    # an opt-out) and must not reconstruct it by guessing the wall clock.
    if cfg.dataset is not None:
        _emit(EVENT_DATASET, f"repo_id={cfg.dataset.repo_id} root={cfg.dataset.root}")

    logger.info(
        "Robot: %s | Teleop: %s | FPS: %.0f | Target corrections: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.teleop.type if cfg.teleop else "?",
        cfg.fps,
        cfg.strategy.num_episodes,
    )

    failure = None
    try:
        strategy.setup(ctx)
        logger.info("%s — coaching runner waiting for commands", _SETUP_COMPLETE_LOG)

        # Only now is it safe to read stdin: the calibration prompt inside
        # `robot.connect()` / `teleop.connect()` has had its turn.
        commands: queue.Queue = queue.Queue()
        threading.Thread(
            target=_read_commands,
            args=(commands, shutdown_event),
            name="dagger-runner-stdin",
            daemon=True,
        ).start()

        # Bridge the reader thread to the control loop. The loop drains this
        # queue itself (`_drain_commands`), so all this thread does is forward —
        # it must never touch the bus, which is why translation lives in the loop.
        def _forward():
            while True:
                command = commands.get()
                if command is None:
                    return
                strategy.submit(command)

        threading.Thread(target=_forward, name="dagger-runner-commands", daemon=True).start()

        _emit(EVENT_READY)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except BaseException as exc:  # noqa: BLE001 — reported, then re-raised as a non-zero exit
        failure = exc
        logger.exception("Coaching runner failed")
        # Give the orchestrator the real exception text on the wire. It also
        # mines the log tail, but that is a heuristic over a traceback; this is
        # the message itself, delivered before the process is gone.
        _emit(EVENT_ERROR, f"{type(exc).__name__}: {exc}")
    finally:
        # Full lerobot teardown: finalize the dataset (encode any pending video),
        # ease home, disconnect the bus and every camera. Guarded because the
        # failure that brought us here is often exactly what teardown will trip
        # over again, and masking it with a second traceback helps nobody.
        try:
            strategy.teardown(ctx)
        except Exception:
            logger.exception("Coaching runner teardown failed")
        _emit(EVENT_BYE)

    if failure is not None:
        sys.exit(1)


def main() -> None:
    """Entry point for `python -m makermodslab.dagger_runner`."""
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
