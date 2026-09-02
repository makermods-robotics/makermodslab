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
loop. Six behaviours we need are not expressible from outside it:

  1. **Cancel.** Upstream saves EVERY correction, unconditionally, at the
     CORRECTING→PAUSED edge. A fumbled takeover — grabbed the leader badly, the
     gripper caught, corrected the wrong thing — is poison training data that
     the operator has no way to reject. The vendored loop calls
     `clear_episode_buffer()` at that edge when cancel is armed.
  2. **The held correction.** A correction the operator DID want is not written
     at that edge either. It is detached from the dataset writer and kept in
     memory (`_hold_correction`) so DROP_LAST can still un-record it, then
     written by `_commit_held` on the way into the next takeover or the next
     attempt. Upstream's edge writes to disk synchronously, which both blocks
     the hand-back and leaves nothing to take back afterwards.
  3. **The takeover offset** (see `close_the_gap` / `begin_correction` /
     `follower_target`). Upstream drives one arm to meet the other when control
     changes hands, and skips it whenever it has no previous action to aim at —
     which is after every hand-back, after every reset, and throughout warmup. A
     takeover in those windows sent the follower straight to wherever the leader
     was standing. Instead: the LEADER is glided toward the follower
     best-effort, and whatever gap that leaves is measured at the edge and
     cancelled out of every command, decaying to zero over `_OFFSET_DECAY_S`.
  4. **Ending an attempt.** Corrections-only DAgger has no task episode of its
     own, so nothing upstream ever declares the attempt over and the policy
     keeps driving at a finished scene. The loop eases both arms home, unpowers
     the follower so the scene can be reset around it, and re-powers it on the
     next transition.
  5. **Driving the follower faster than the record rate.** Upstream pairs one
     command with one recorded frame and sleeps out the rest of the tick. A
     servo wants a goal position far more often than 30 Hz, and coaching felt
     stepped for exactly that reason, so the tick's leftover time is spent
     driving instead — see `_drive_until`.
  6. **Protocol events** at the phase edges, so the browser can show what the
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
    CANCEL_REASON_FAULT,
    CANCEL_REASON_OPERATOR,
    CANCEL_REASON_TOO_SHORT,
    CMD_CANCEL,
    CMD_DROP_LAST,
    CMD_HANDBACK,
    CMD_HOLD,
    CMD_QUIT,
    CMD_RECOVERED,
    CMD_RESET,
    CMD_RESUME,
    CMD_TAKEOVER,
    EVENT_ATTEMPT_RESET,
    EVENT_BYE,
    EVENT_CORRECTION_CANCELLED,
    EVENT_CORRECTION_COMMITTED,
    EVENT_CORRECTION_DROPPED,
    EVENT_CORRECTION_HELD,
    EVENT_CORRECTION_SAVED,
    EVENT_DATASET,
    EVENT_ERROR,
    EVENT_PHASE,
    EVENT_READY,
    EVENT_RECOVERY_MARK,
    PHASE_HANDING_OVER,
    PHASE_POISED,
    PHASE_RESETTING,
    PHASE_SAVING,
    format_event,
)
from .log_exceptions import restore_traceback_rendering
from .torque import force_disable_bus_torque

logger = logging.getLogger(__name__)

# Upstream's own transition-request names, as used by `DAggerEvents`. Ours are
# the operator-facing composites; these are the primitives they expand into.
_EV_PAUSE_RESUME = "pause_resume"
_EV_CORRECTION = "correction"

# The takeover's whole safety story, and it is now enforced on the FOLLOWER
# rather than asked of the leader.
#
# The old approach drove the leader onto the follower, measured the gap, and
# REFUSED the takeover if it was still too wide. Every part of that was wrong on
# real hardware: the residual after a glide is 2-8 degrees on this arm, so a
# fixed threshold sits inside the measurement's own noise; the measurement was
# taken ~33ms after the last interpolation step, while the arm was still
# travelling; and a refusal reverted to PAUSED with nothing changed, so a leader
# that physically stalls short of its target produced an unbreakable
# handing_over -> refuse -> paused loop. One session logged 19 refusals, 11 of
# them at an identical 7 degrees, and could not be driven at all.
#
# So nothing is measured and nothing is refused. At takeover the CURRENT gap is
# captured as an offset and added to every leader pose, which makes the first
# commanded follower position exactly where the follower already is — it cannot
# jump, whatever the gap. The offset then decays to zero over
# `_OFFSET_DECAY_S`, so the two arms end up in the same frame again and repeated
# takeovers cannot walk the leader into its joint limits (the classic clutch
# drift). Under that sits a per-joint rate limit on the commanded target. It
# covers every command a correction issues — the once-per-tick one and the
# millisecond ones `_drive_until` adds, on the bimanual path as well as the
# single-arm one — but ONLY during a correction: nothing else in this file goes
# through `follower_target`.
_OFFSET_DECAY_S = 1.5

# Speed ceiling for the decay itself, in degrees per second per joint.
#
# `_OFFSET_DECAY_S` alone is a DURATION, not a bound, and that is the hazard:
# decaying an offset O linearly over a fixed 1.5s moves the follower at O/1.5
# deg/s, so a 90-degree offset sweeps the follower 90 degrees across the
# workspace — with the task laid out under it — in a second and a half. The
# per-command rate limit cannot catch that and arithmetically never will (see
# `_MAX_FOLLOWER_DEG_PER_S`): it would need O above 360 to engage, when a
# joint's entire range is about 200.
#
# So the decay's duration is derived from the offset instead of fixed:
# `max(_OFFSET_DECAY_S, worst_offset / this)`. A big offset decays over LONGER
# rather than faster, and the follower's implied speed is bounded by this
# number whatever the gap. Small offsets — the ordinary case once
# `close_the_gap` has run — are unaffected and still take 1.5s.
#
# 30 deg/s, chosen against the two rates already in this file rather than
# invented. `_HOME_DEG_PER_S` (45) is the deliberately slow rate for driving
# the FOLLOWER through the workspace while the operator watches a movement they
# asked for; the decay is follower travel nobody asked for and nobody is
# watching, so it gets less than that. It is also an eighth of
# `_MAX_FOLLOWER_DEG_PER_S`, far enough below the discontinuity clamp that the
# two never interact and neither can be mistaken for the other.
_OFFSET_DECAY_DEG_PER_S = 30.0

# The leader still glides toward the follower at a takeover — best-effort, with
# no threshold and nothing to fail.
#
# Pure offset-and-decay has a weakness the glide covers: the offset guarantees
# the follower does not JUMP, but the decay still walks it the whole way to
# wherever the leader is standing. A 40-degree gap becomes a 40-degree follower
# sweep across the workspace, merely spread over `_OFFSET_DECAY_S` instead of
# happening in one tick. Slower, not safer — the task is laid out under that arm.
#
# Closing most of the gap with the LEADER first means there is little left for
# the follower to travel: the leader is in the operator's hand or on the bench,
# so moving it costs nothing. Whatever the leader fails to close is simply what
# the offset absorbs, which is why this needs no arrival check and cannot
# refuse. That verification is exactly what wedged sessions before — the check
# has not been made more lenient, it has been made unnecessary.
# Reduced from 120 on the operator's report: at a 120-degree gap the leader
# arrived correctly but moved faster than was comfortable to stand next to. The
# number is a workspace judgement, not a derived constant — it is the speed a
# person is happy to have an arm swing at an arm's length away.
_TAKEOVER_GLIDE_DEG_PER_S = 96.0
_TAKEOVER_GLIDE_MAX_S = 1.0
# Below this the glide is skipped: moving is noise, and the offset covers it.
_TAKEOVER_GLIDE_SKIP_DEG = 3.0

# Ceiling on how fast a correction may command the follower, in degrees per
# second per joint, applied to the commanded target rather than to a measured
# pose so it works identically at the loop rate and inside `_drive_until`.
#
# Set well above human teleoperation (an operator swinging an SO-101 leader
# hard peaks around 120 deg/s) so it never shapes a real demonstration — this
# is a floor under the pathological case, not a comfort feature.
#
# What it actually bounds is a DISCONTINUITY in the leader's reported pose: a
# stale or garbled read, a re-acquire, a joint that jumps between bus frames. It
# does NOT bound the takeover offset's decay, and arithmetically cannot: decaying
# an offset O linearly moves the target by O/duration units per second, so
# clipping would need O above 360 at the old fixed 1.5s duration when a joint's
# entire range is about 200. It never engages for the decay, and nothing here
# should be read as though it does. The decay is bounded on its OWN terms
# instead, by stretching its duration — see `_OFFSET_DECAY_DEG_PER_S`, which is
# what actually keeps a large offset from walking the follower across the
# workspace, together with `close_the_gap` shrinking O before it is measured.
_MAX_FOLLOWER_DEG_PER_S = 240.0

# How near the start pose the follower must actually BE before we cut its
# torque, in the robot's own action units — degrees for the five body joints
# (lerobot's `use_degrees` defaults to True and nothing here overrides it) and
# 0-100 for the gripper, which is always `RANGE_0_100`. Deliberately loose: this
# only has to establish "the arm came home and is therefore in a pose it can
# hold unpowered", not "the arm is precisely positioned".
#
# It exists because upstream's `_return_to_initial_position` CANNOT report
# failure — it wraps its whole body in `except Exception: logger.warning(...)`
# and performs no arrival check — so the reset's `homed` flag was previously
# always True and the guard around going limp was unreachable code. A dropped
# serial reply mid-glide left the arm extended, and torque was cut anyway.
_HOME_TOLERANCE = 8.0

# The ease-home's travel budget, as a rate rather than a fixed duration — the
# same treatment the leader glide once had, before that was removed.
#
# Upstream's `_return_to_initial_position` interpolates over a flat
# `duration_s=3.0` whatever the distance, so an arm parked a few degrees from
# home still took three seconds to crawl there, and it looks for all the world
# like the arm is straining against something rather than being deliberately
# slow. Slower than the leader's rate on purpose: this one moves the FOLLOWER,
# through the workspace, with the task laid out in front of it.
_HOME_DEG_PER_S = 45.0
_HOME_MIN_S = 0.4
_HOME_MAX_S = 3.0

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
# (0% CPU, PNG frames undrained, no video written) and could not be explained from
# the log afterwards, because a blocked thread writes nothing. The dump turns a
# silent wedge into a stack trace naming the exact call.
_SAVE_WATCHDOG_S = 30.0

# Pacing between the extra leader->follower commands issued inside one tick.
# Lifted from teleoperate.py's loop, which drives the same hardware at this
# spacing and is the thing operators say feels right — see `_drive_until`.
_DRIVE_INTERVAL_S = 0.001


# How often the correcting loop reports where its tick went.
#
# The coaching loop and lerobot's `record_loop` have the same shape and the same
# 1/fps budget, so "coaching stutters and recording does not" is not explained by
# the rate — and guessing at the difference from a laptop is how the wrong thing
# gets optimised. This measures it on the station instead: six timed spans per
# tick, summarised every few seconds as a mean and a max apiece, alongside the
# rate the loop actually achieved and its slowest single tick.
#
# The measurement itself is two `perf_counter()` calls per span, which is tens of
# nanoseconds against a 33ms budget.
_TICK_REPORT_INTERVAL_S = 5.0


class _TickBudget:
    """Where each correcting tick's time went, summarised periodically.

    Reports the MAX as well as the mean for every span. A stutter is not a raised
    average — the loop can sit comfortably inside its budget and still lurch once
    a second — so a report that only carried means would show a healthy loop for
    the exact fault it was added to find."""

    # Timed spans, in tick order. `tail` is the part a tick cannot time from
    # inside itself — the sleep, or the high-rate drive that replaced it — and
    # it is where the whole missing 82ms turned out to live.
    _SPANS = ("observe", "process", "leader", "send", "record", "tail")
    # Not a duration: how many extra leader->follower commands the tail issued.
    _COUNTS = ("drove",)

    def __init__(self, budget_s: float) -> None:
        self._budget_s = budget_s
        self.reset()

    def reset(self) -> None:
        self._totals = dict.fromkeys(self._SPANS + self._COUNTS, 0.0)
        self._peaks = dict.fromkeys(self._SPANS + self._COUNTS, 0.0)
        self._ticks = 0
        self._worst_tick = 0.0
        self._since = time.perf_counter()

    def add(self, spans: dict, total: float) -> None:
        for name in self._SPANS + self._COUNTS:
            value = spans.get(name, 0.0)
            self._totals[name] += value
            if value > self._peaks[name]:
                self._peaks[name] = value
        self._ticks += 1
        if total > self._worst_tick:
            self._worst_tick = total

    def report_if_due(self) -> None:
        if self._ticks == 0 or time.perf_counter() - self._since < _TICK_REPORT_INTERVAL_S:
            return
        parts = [
            f"{name}={self._totals[name] / self._ticks * 1e3:.1f}/{self._peaks[name] * 1e3:.1f}ms"
            for name in self._SPANS
        ]
        parts += [
            f"{name}={self._totals[name] / self._ticks:.1f}/{self._peaks[name]:.0f}" for name in self._COUNTS
        ]
        elapsed = time.perf_counter() - self._since
        # The achieved rate is the headline: a tick whose parts all look fast
        # can still come round three times too slowly, which is exactly the fault
        # this was added to find. There is deliberately NO overrun count: the
        # drive fills every correcting tick to its deadline by design, so any
        # such counter reads 100% and means nothing. It was computed and never
        # read for exactly that reason, so it is gone rather than misleading.
        logger.info(
            "Correcting tick budget %.1fms over %d ticks — %.1f Hz achieved "
            "(slowest tick %.1fms); mean/max %s",
            self._budget_s * 1e3,
            self._ticks,
            self._ticks / elapsed if elapsed > 0 else 0.0,
            self._worst_tick * 1e3,
            " ".join(parts),
        )
        self.reset()


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
        # The takeover offset: follower_pose - leader_pose, captured the moment a
        # correction begins and added to every leader pose after it, so the first
        # commanded follower position is exactly where the follower already is.
        # None outside a correction. See `_OFFSET_DECAY_S` above for why the
        # takeover no longer has to drive the FOLLOWER into position, and
        # `_TAKEOVER_GLIDE_DEG_PER_S` for the leader glide that still runs first.
        # True while both arms are held on the follower's pose waiting for the
        # operator's SECOND takeover press. The leader is under torque here — it
        # is being held where the follower is, deliberately, so the operator can
        # take hold of an arm that is already lined up and not moving.
        # Set when the control loop exits by exception rather than by request.
        # The in-flight correction is discarded when it is true — see the
        # teardown block.
        self._loop_failed = False
        self._poised = False
        # Armed by a takeover that should poise rather than hand over. Consumed
        # in the control loop, which is the only place the buses may be touched.
        self._poise_requested = False
        self._offset: dict[str, float] | None = None
        # When the offset was captured, so it can be decayed to zero on a clock
        # rather than a tick count — the loop rate is not constant and a
        # tick-counted decay would take twice as long on a slow loop.
        self._offset_at = 0.0
        # How long the current offset gets to decay over. Derived per takeover
        # from the offset's own size so the decay obeys a SPEED ceiling rather
        # than a fixed duration — see `_OFFSET_DECAY_DEG_PER_S`. `_OFFSET_DECAY_S`
        # is its floor and the value it holds outside a correction.
        self._decay_s = _OFFSET_DECAY_S
        # The last target COMMANDED to the follower, which is what the rate limit
        # measures against. Deliberately the command and not a fresh observation:
        # it needs no serial read, and it stays correct inside `_drive_until`
        # where several commands are issued between two observations.
        self._last_target: dict[str, float] | None = None
        self._last_target_at = 0.0
        # How many extra leader->follower commands the last tick's tail issued.
        # Diagnostic only — read by the tick budget, never by the control flow.
        self._drive_sent = 0
        # RaC: where recovery ended and the correction began, in frames, for the
        # correction currently recording. None means the operator has not marked
        # it — which is NOT the same as zero, and the two must stay
        # distinguishable all the way to the sidecar (see the protocol module).
        self._recovery_frames: int | None = None
        # Armed by RECOVERED, consumed on the next tick that is actually
        # recording. Deliberately not applied inside `_translate`: the frame
        # count it has to capture lives in the control loop.
        self._recovery_mark_requested = False
        # The correction that has ENDED but is not on disk yet, as the raw
        # episode buffer detached from the dataset writer, plus what the events
        # need to describe it. None whenever nothing is held.
        #
        # This is the whole mechanism behind DROP_LAST, and the reason it can
        # honestly say "delete": nothing is deleted, because nothing was
        # written. Exactly one correction is ever held — see "The held
        # correction" in dagger_protocol, which also explains why upstream's
        # `validate_episode_buffer` requires that to be true.
        self._held: dict | None = None
        self._held_meta: dict | None = None
        # Armed by DROP_LAST, consumed in the control loop where `dataset` is
        # in scope. Same shape as `_recovery_mark_requested`, and for the same
        # reason: `_translate` cannot reach the dataset.
        self._drop_last_requested = False
        # True while parked straight after a reset. The held correction is
        # committed when this window CLOSES — that is, when the operator starts
        # the next attempt — rather than at hand-back, which is what keeps the
        # drop window open across the reset the operator makes the decision in.
        self._awaiting_attempt = False

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
            # TWO PRESSES, and which one this is depends on `_poised` rather than
            # on the phase — lerobot has no phase for "held, lined up, waiting",
            # so PAUSED covers both "the policy is frozen" and "your arm is
            # ready". Only the second press moves control or records anything.
            if self._poised:
                return [_EV_CORRECTION]
            if phase == DAggerPhase.AUTONOMOUS:
                # Stop the policy first; the glide needs a follower that is not
                # being driven, and the operator needs the scene to stop moving.
                self._poise_requested = True
                return [_EV_PAUSE_RESUME]
            if phase == DAggerPhase.PAUSED:
                self._poise_requested = True
                return []
        elif command == CMD_HANDBACK:
            if phase == DAggerPhase.CORRECTING:
                # Clears the discard latch on the way past. `_cancel_correction`
                # is consumed at the CORRECTING->PAUSED edge, so a discard that
                # was armed and then overtaken by a hand-back would otherwise
                # still be sitting there — and would silently bin the correction
                # the operator has just chosen to KEEP.
                self._cancel_correction = False
                return [_EV_CORRECTION, _EV_PAUSE_RESUME]
        elif command == CMD_CANCEL:
            # DISCARD, and then bring the arm home.
            #
            # It used to stop at PAUSED, leaving the follower holding whatever
            # pose the fumbled takeover ended in and the operator to work out
            # what to press next. A discard means the last few seconds were a
            # mess — the scene almost always needs resetting after one — so this
            # now runs the ordinary reset behind the discard: the follower eases
            # back to its start pose and parks for a scene reset, exactly as
            # Enter does, and the leader is released on the way.
            #
            # Absorbing that also absorbed RECOVER, which was this same pair
            # (discard, then reset) under a second name and a second button.
            # Which is why this is NOT phase-gated: RECOVER's whole reason to
            # exist was being valid from every phase, so that a wedged
            # correction still had a way out. Losing that would strand exactly
            # the operator it was written for.
            self._reset_requested = True
            if phase == DAggerPhase.CORRECTING:
                self._cancel_correction = True
                return [_EV_CORRECTION]
            logger.info("Discard with nothing in flight — resetting from phase %s", phase.value)
            return []
        elif command == CMD_HOLD:
            if phase == DAggerPhase.AUTONOMOUS:
                return [_EV_PAUSE_RESUME]
        elif command == CMD_RESUME:
            if phase == DAggerPhase.PAUSED:
                return [_EV_PAUSE_RESUME]
        elif command == CMD_RECOVERED:
            # An ANNOTATION, not a transition — recovery and correction are the
            # same control mode, and inventing a phase for a distinction lerobot
            # does not have would desync the vendored loop from upstream's state
            # machine for nothing. Ignored outside a correction (there is no
            # boundary to mark) and ignored twice (the first mark is the one the
            # operator meant; a second would silently move it).
            if phase == DAggerPhase.CORRECTING and self._recovery_frames is None:
                self._recovery_mark_requested = True
            else:
                logger.info("Ignoring RECOVERED — no unmarked correction in progress")
            return []
        elif command == CMD_DROP_LAST:
            # "That last one was no good." Never a transition — the arm does not
            # move and no phase changes; the only thing that happens is that a
            # buffer in memory is thrown away instead of written.
            #
            # Refused mid-correction, and that refusal is not a technicality:
            # from CORRECTING the operator is talking about the take they are
            # STILL RECORDING, and the control that discards that one is CANCEL.
            # Honouring DROP_LAST here would silently bin the PREVIOUS
            # correction — one they were happy with — while leaving the one they
            # meant untouched.
            if phase == DAggerPhase.CORRECTING:
                logger.info("Ignoring DROP_LAST mid-correction — use CANCEL to discard this one")
            elif self._held is None:
                logger.info("Ignoring DROP_LAST — nothing is held (already committed, or never recorded)")
            else:
                self._drop_last_requested = True
            return []
        elif command == CMD_RESET:
            self._reset_requested = True
            if phase == DAggerPhase.CORRECTING:
                # SAVES the correction, then resets.
                #
                # This used to be refused outright, on the reasoning that
                # deciding the fate of a part-recorded takeover was not this
                # code's call. In practice the operator finishing the task while
                # still driving has already made that call — the frames up to
                # here ARE the correction — and refusing left them to hand back
                # and then reset as two presses, with the policy briefly
                # regaining a finished scene in between.
                #
                # Exactly RECOVER's edge without the discard: the same event
                # drives CORRECTING->PAUSED, where `_cancel_correction` is False
                # so the episode is written, and `_reset_requested` is honoured
                # on the next tick. A correction too short to be deliberate is
                # still binned by the existing length check.
                logger.info("RESET during a correction: saving it, then resetting")
                return [_EV_CORRECTION]
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
    def _transition_moves_the_arm(old_phase, new_phase, ctx, prev_action=None) -> bool:
        """True when `_apply_transition` is about to drive hardware for ~2s.

        A MIRROR of upstream's branching, which is why it takes `prev_action`
        it never reads for anything else: upstream skips both smooth-handover
        paths when there is no previous action to move toward, and a mirror that
        announced travel there would promise motion that never comes. That
        happens for real — a takeover during warmup, or the first tick after a
        reset, both reach here with none.

        Which edge moves the arm, and why the UI needs to know at all, is argued
        once in `dagger_protocol.PHASE_HANDING_OVER`. Keep it there.

        As of the takeover-offset rework this returns False at its ONE call
        site, always: that caller withholds `prev_action` on exactly the two
        edges this would otherwise answer True for. It is kept, rather than
        deleted with the glides, so that undoing the suppression re-arms the
        banner in one place instead of needing this rewritten from upstream.
        """
        if prev_action is None:
            return False
        if teleop_supports_feedback(ctx.hardware.teleop):
            return old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED
        return old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING

    def close_the_gap(self, ctx, obs: dict) -> None:
        """Walk the leader toward the follower's pose. Best-effort, never fatal.

        Runs immediately before `begin_correction`, so whatever it manages to
        close is gap the offset does not have to absorb and the follower does
        not have to travel during the decay.

        Every failure here is survivable BY CONSTRUCTION, which is the whole
        point of doing it this way round. If the leader stalls short, if its bus
        drops a reply, if it does not move at all — the offset captured
        afterwards measures whatever the real gap turned out to be and cancels
        it. Nothing is verified and nothing can refuse, because there is no
        outcome this code needs the leader to achieve.

        Only for an ACTUATED leader. A non-actuated one (bimanual, on this pin)
        cannot be driven at all, and there the offset does the entire job — no
        glide, offset only. That is a property of the PIN, not of the hardware:
        upstream lerobot PR #4028 gives `BiSOLeader` feedback support, but it
        landed after the v0.6.0 release tag this repo pins (see CLAUDE.md — we
        track releases, not `main`), so `teleop_supports_feedback` answers False
        for the bimanual leader here and this function returns immediately.
        A pin bump past that PR turns the bimanual path into the actuated one
        below, glide and leader release included.

        THE LEADER IS RELEASED ON THE WAY OUT, unconditionally, and that is not
        housekeeping. Upstream's `_apply_transition` ends its PAUSED->CORRECTING
        branch with `teleop.disable_torque()` to unlock the leader for the human
        — and this function runs AFTER that, and `teleop_smooth_move_to`'s first
        statement is `teleop.enable_torque()`. Nothing else in the correction
        turns it off again, so without the release below the operator picks up a
        powered leader holding its glide pose and physically cannot drive the
        arm: the whole correction is spent fighting a rigid arm, or abandoned.
        Hence the `finally` — a glide that RAISED left the torque on just the
        same, and that is the likelier path (a bus wobble at the handover)."""
        leader = ctx.hardware.teleop
        if not teleop_supports_feedback(leader):
            return
        try:
            here = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and isinstance(v, (int, float))}
            current = leader.get_action()
            worst = max(
                (
                    abs(float(v) - float(current[k]))
                    for k, v in here.items()
                    if isinstance(current.get(k), (int, float))
                ),
                default=0.0,
            )
            if worst <= _TAKEOVER_GLIDE_SKIP_DEG:
                logger.info("Leader is already on the follower (%.0f deg) — no glide", worst)
                return
            duration = min(_TAKEOVER_GLIDE_MAX_S, worst / _TAKEOVER_GLIDE_DEG_PER_S)
            logger.info("Closing a %.0f deg takeover gap with the leader over %.2fs", worst, duration)
            teleop_smooth_move_to(leader, here, duration_s=duration)
        except Exception:
            # Logged, not raised, and not measured afterwards: the offset is
            # about to be taken against reality either way.
            logger.exception("Could not glide the leader toward the follower; the offset will absorb it")
        # Deliberately NO release here any more. The leader is left HELD on the
        # follower's pose, which is the whole point of poising: the operator
        # takes hold of an arm that is already lined up and stationary. It is
        # released at the second takeover press, in the same breath as recording
        # starting — see the CORRECTING edge — and by `_unpoise` on every path
        # that leaves the hold without driving.

    def begin_correction(self, teleop_action: dict, obs: dict) -> None:
        """Capture the takeover offset. Called once, as a correction starts.

        `offset = follower - leader`, per joint, over the keys the two share.
        Adding it to the leader's pose makes the very first commanded follower
        position identical to where the follower is standing, so the handover
        moves the follower not at all — whatever gap `close_the_gap` left
        behind, and without the follower having been driven anywhere.

        Joints the two do not share get no offset (and so are passed through),
        which is the honest answer: nothing is known about a gap that cannot be
        measured, and inventing one would be worse than leaving it."""
        self._offset = {}
        for key, want in teleop_action.items():
            have = obs.get(key)
            if isinstance(want, (int, float)) and isinstance(have, (int, float)):
                self._offset[key] = float(have) - float(want)
        self._offset_at = time.perf_counter()
        # The rate limit's clock starts NOW, with the same reading. Without this
        # the first `follower_target` of every correction measured `elapsed`
        # against a timestamp left over from the PREVIOUS correction — seconds or
        # minutes — and computed a ceiling in the thousands, so the clamp was
        # inert for exactly the command that matters most.
        self._last_target_at = self._offset_at
        # The rate limit's starting reference is where the follower IS, so the
        # first command is measured against reality rather than against whatever
        # the previous correction left behind.
        self._last_target = {
            k: float(v) for k, v in obs.items() if isinstance(v, (int, float)) and k.endswith(".pos")
        }
        # Stretch the decay for a big offset rather than let it move the
        # follower faster. `_OFFSET_DECAY_S` is a floor, so the ordinary small
        # offset behaves exactly as before; a 90-degree one that used to sweep
        # the follower at 60 deg/s now takes 3s at 30. See
        # `_OFFSET_DECAY_DEG_PER_S` for why the per-command rate limit cannot
        # do this job.
        worst = max((abs(v) for v in self._offset.values()), default=0.0)
        self._decay_s = max(_OFFSET_DECAY_S, worst / _OFFSET_DECAY_DEG_PER_S)
        logger.info("Correction begins with a %.0f deg offset, decaying over %.1fs", worst, self._decay_s)

    def end_correction(self) -> None:
        """Drop the offset and the rate-limit reference when a correction ends."""
        self._offset = None
        self._decay_s = _OFFSET_DECAY_S
        self._last_target = None

    def follower_target(self, teleop_action: dict, now: float) -> dict:
        """One leader pose -> the pose the follower should actually be commanded to.

        Two transforms, in this order and both applied to the TELEOP action
        rather than to what is sent:

          1. the decaying takeover offset, so the handover moves nothing and the
             two arms drift back into the same frame over `self._decay_s`
             (`_OFFSET_DECAY_S`, stretched for a large offset so the decay
             obeys a speed ceiling — see `_OFFSET_DECAY_DEG_PER_S`);
          2. a per-joint rate limit against the last commanded target, which
             bounds every path into this function.

        THE ORDER MATTERS, and so does the fact that this returns the value the
        caller records. `dagger_runner` records `processed_teleop` — the leader's
        pose — not `robot_action_to_send`, so a transform applied only on the way
        to the robot would leave the dataset claiming an action the follower
        never performed. Transform first, then record and send the same dict.
        """
        target = dict(teleop_action)
        if self._offset is not None:
            # Linear to zero, so it actually reaches zero rather than
            # asymptotically approaching it and leaving a permanent bias.
            scale = 1.0 - (now - self._offset_at) / self._decay_s
            if scale <= 0.0:
                self._offset = None
            else:
                for key, delta in self._offset.items():
                    value = target.get(key)
                    if isinstance(value, (int, float)):
                        target[key] = float(value) + delta * scale

        previous = self._last_target
        if previous is not None:
            elapsed = max(now - self._last_target_at, 1e-4)
            ceiling = _MAX_FOLLOWER_DEG_PER_S * elapsed
            for key, value in list(target.items()):
                was = previous.get(key)
                if not isinstance(value, (int, float)) or not isinstance(was, (int, float)):
                    continue
                step = float(value) - was
                if step > ceiling:
                    target[key] = was + ceiling
                elif step < -ceiling:
                    target[key] = was - ceiling
        self._last_target = {k: float(v) for k, v in target.items() if isinstance(v, (int, float))}
        self._last_target_at = now
        return target

    def _drive_until(self, deadline: float, ctx, robot, teleop, obs, last_action):
        """Keep driving the follower from the leader until `deadline`. Returns the
        last action sent.

        THE stutter fix, and it is a rate fix rather than a speed fix.

        The control loop commands the follower ONCE per tick, because upstream's
        loop pairs one command with one recorded frame. That is correct for the
        dataset and wrong for the arm: a recorded frame is wanted 30 times a
        second, but a servo wants a new goal position as often as it can get
        one. Plain teleoperation drives the same hardware in a bare
        `get_action` / `send_action` / `sleep(1ms)` loop at several hundred Hz
        and feels smooth; coaching drove it at the record rate and felt stepped.
        Measured on the station it was worse than the record rate even suggests
        — a hair under 12 Hz, an 86ms gap between goal positions.

        So the tick's leftover time is spent driving instead of sleeping. The
        frame pairing is untouched: the recorded observation and action are
        still the ones taken at the top of the tick, and nothing here adds a
        frame. Only the follower gets told more often where the leader is.

        The observation from the top of the tick is reused rather than re-read.
        It feeds the leader->follower processors, which for a matched
        leader/follower pair map joint names one-for-one; re-reading would cost
        a serial round trip and three camera copies to refine something the
        processors barely consult.

        Paced at `_DRIVE_INTERVAL_S` — the same 1ms plain teleoperation uses —
        so this fills idle time rather than saturating the buses, and it yields
        often enough that a QUIT is still acted on within a tick."""
        shutdown = ctx.runtime.shutdown_event
        sent = 0
        while time.perf_counter() < deadline:
            if shutdown.is_set():
                break
            try:
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                # The same offset and rate limit the tick applies. Without this
                # the tick would command a safe target and the twenty commands
                # after it would each undo that, at 1ms intervals — the drive
                # would be the jump.
                processed = self.follower_target(processed, time.perf_counter())
                action = ctx.processors.robot_action_processor((processed, obs))
                robot.send_action(action)
            except Exception:
                # A dropped serial reply must not end the session from HERE:
                # this is an extra command between two the loop was already
                # going to send, and the tick's own read/write still raises
                # normally.
                #
                # Note what the `break` costs, though, because it is not one
                # command. It abandons the whole drive, and for a CORRECTING
                # tick the drive IS the tail — there is no sleep behind it — so
                # the tick returns immediately and the control loop free-runs,
                # coming round faster than fps and recording frames at that
                # rate for as long as the failures continue. A leader dropping
                # replies steadily therefore distorts the loop's timing and the
                # dataset, not just its smoothness. Unfixed; written down so it
                # is not rediscovered from a stutter report.
                # Sleep out the rest of the tick rather than returning into a
                # loop with no pacing left. The drive REPLACED the tail's
                # `precise_sleep` — a correcting tick is one or the other — so
                # returning early does not cost "a command", it costs the whole
                # remaining budget. A recurring leader-bus dropout then free-runs
                # the loop, and `add_frame` stamps `frame_index / fps` onto
                # frames arriving far faster than fps, writing an episode whose
                # timestamps claim a fraction of the time it really took. Silent,
                # and it survives into training.
                logger.debug("A mid-tick teleop drive failed; skipping it", exc_info=True)
                left = deadline - time.perf_counter()
                if left > 0:
                    precise_sleep(left)
                break
            last_action = action
            sent += 1
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            precise_sleep(min(_DRIVE_INTERVAL_S, remaining))
        self._drive_sent = sent
        return last_action

    def _ease_both(
        self, ctx, follower_from: dict, leader_from: dict | None, target: dict, duration_s: float
    ) -> None:
        """Interpolate the follower — and the leader, when there is one — home.

        Deliberately ours rather than upstream's `_return_to_initial_position`.
        That one drives the follower alone, swallows every exception, and
        reports nothing; this repo already had to wrap it to find out whether
        the arm arrived. Driving both arms from one loop is what makes them move
        together, and it costs twelve lines of interpolation that upstream's own
        `teleop_smooth_move_to` writes the same way.

        KNOWN, and deliberately left alone: a hand held against the leader does
        not stop this. The interpolation has no force feedback and no arrival
        check on the leader half, so it drives to its target regardless, and the
        follower has been observed to jerk briefly toward the leader before
        continuing home. Reported from the bench on 2026-09-01 and judged not
        worth fixing — the arms still end up home and limp, and the operator's
        remedy is simply to let go. Recorded here so it is not rediscovered as a
        mystery; fixing it would mean giving the leader half its own arrival
        check, which is the machinery whose absence made takeover reliable.

        The leader is best-effort throughout. It is a hand-held arm being
        returned for the operator's comfort; the follower is the one carrying a
        gripper over a table, so a leader that stops responding must not
        interrupt the follower's glide."""
        fps = 50
        steps = max(int(duration_s * fps), 1)
        robot = ctx.hardware.robot_wrapper
        leader = ctx.hardware.teleop
        # Only the joints all three dicts agree on; anything else is left alone.
        f_keys = [k for k, v in follower_from.items() if k in target and isinstance(v, (int, float))]
        l_keys = (
            [k for k, v in leader_from.items() if k in target and isinstance(v, (int, float))]
            if leader_from
            else []
        )
        for step in range(1, steps + 1):
            t = step / steps
            with contextlib.suppress(Exception):
                robot.send_action({k: follower_from[k] * (1 - t) + float(target[k]) * t for k in f_keys})
            if l_keys:
                try:
                    leader.send_feedback({k: leader_from[k] * (1 - t) + float(target[k]) * t for k in l_keys})
                except Exception:
                    logger.exception("The leader stopped following the ease-home; leaving it")
                    l_keys = []
            precise_sleep(1 / fps)

    def _hold_leader(self, ctx, here: dict) -> None:
        """Power the leader and pin it on `here`. Best-effort, never fatal.

        `close_the_gap` returns early when the leader is already close enough to
        be worth moving, and `teleop_smooth_move_to` is the only thing in that
        path that enables torque — so without this the "held" state would be a
        limp leader whenever the operator happened to be holding it near the
        right place, which is exactly when they are most likely to trust it."""
        leader = ctx.hardware.teleop
        if not teleop_supports_feedback(leader):
            return
        try:
            leader.enable_torque()
            leader.send_feedback({k: v for k, v in here.items() if k in leader.get_action()})
        except Exception:
            logger.exception("Could not hold the leader in position; it may be free to move")

    def _unpoise(self, ctx) -> None:
        """Leave the hold without taking control: let the leader go.

        Every path out of `poised` that is not the second takeover press ends up
        here — a discard, a reset, a freeze, a stop. Leaving it powered would
        strand the operator holding a rigid arm with nothing on screen to explain
        it, which is the original "the arm is stuck there" report."""
        if not self._poised:
            return
        self._poised = False
        self._poise_requested = False
        logger.info("Left the hold without taking control — releasing the leader")
        self._release_leader(ctx)

    def _release_leader(self, ctx) -> bool:
        """Let go of the leader. True if it is now free to move by hand.

        Called at the end of a reset so the operator's own arm goes slack with
        the robot's, and after a discard for the same reason. Releasing it at
        the home pose is safe in a way releasing it mid-reach would not be: home
        is a pose the arm holds unpowered, which is the whole reason the
        follower may be unpowered there too."""
        leader = ctx.hardware.teleop
        if not teleop_supports_feedback(leader):
            # A non-actuated leader was never under torque; it is already free.
            return True
        try:
            leader.disable_torque()
        except Exception:
            logger.exception("Could not release the leader; it may still be rigid")
            return False
        logger.info("Leader released — free to move by hand")
        return True

    def _release_after_discard(self, ctx) -> bool:
        """Release the leader at the discard edge — UNLESS a reset will do it.

        A discard always arms a reset (`_translate`'s CMD_CANCEL sets
        `_reset_requested` before it asks for the transition), and the reset that
        follows a tick later runs `_ease_home`, which ENABLES leader torque to
        glide the leader home and then releases it again. So releasing here too
        put three torque changes on the operator's hand inside about a second:
        slack, rigid, slack. With a hand on the leader that reads as the arm
        grabbing back, and it is indistinguishable from the fault it was written
        to fix — an arm that goes stiff for no reason the screen explains.

        The plain reset path does it exactly once, which is the behaviour to
        match. Deferring is safe because the reset is already armed and is
        honoured from any phase that is not CORRECTING; the discard edge lands in
        PAUSED, so the very next pass through the loop performs it.

        Still conditional rather than deleted: `_cancel_correction` is a latch,
        and a future path that arms it without a reset would otherwise strand the
        leader rigid at PAUSED with no transition that ever frees it — the
        original "the arm is stuck there" report."""
        if self._reset_requested:
            logger.info("Leaving the leader to the armed reset, which releases it once at home")
            return False
        return self._release_leader(ctx)

    # -- the held correction -------------------------------------------------

    @staticmethod
    def _writer(dataset):
        """The dataset's write-side half, or None on a read-only dataset.

        Reached through an attribute rather than a public method because lerobot
        exposes no way to hand an episode buffer back and forth — `save_episode`
        will accept one (`episode_data=`), but nothing will give you one. This
        is the single place that reaches inside; everything else goes through
        `_hold_correction` / `_commit_held` / `_drop_held`."""
        return getattr(dataset, "writer", None)

    def _hold_correction(self, dataset, *, frames: int, seconds: float) -> bool:
        """Detach the finished correction from the writer and keep it. True if held.

        The buffer leaves the writer entirely (`episode_buffer = None`), which
        upstream handles: `add_frame` recreates one lazily on the next frame it
        is given. Nothing recreates it before then, because frames are only ever
        added from the CORRECTING branch of the loop and the held correction is
        always committed or dropped before the next correction starts.

        Returns False if there is nothing to detach, in which case the caller
        must fall back to saving inline — a hold that silently held nothing
        would lose the correction outright."""
        writer = self._writer(dataset)
        buffer = getattr(writer, "episode_buffer", None) if writer is not None else None
        if buffer is None:
            return False
        writer.episode_buffer = None
        self._held = buffer
        self._held_meta = {
            "n": self._corrections_saved,
            "frames": frames,
            "seconds": seconds,
            "recovery": self._recovery_frames,
        }
        return True

    def _commit_held(self, dataset) -> None:
        """Write the held correction to disk. Does nothing if none is held.

        Announces PHASE_SAVING first for the same reason the inline save used
        to: this blocks the control loop while it writes parquet and encodes
        video, and a banner that keeps describing the previous state through it
        is the same class of lie as the handover window.

        A failure here is logged and the correction dropped rather than raised.
        The commit happens on the way INTO a takeover or the next attempt, and
        taking the whole session down at that moment — arm mid-handover, every
        earlier correction already on disk — would cost far more than the one
        episode that failed to write."""
        held, meta = self._held, self._held_meta
        if held is None:
            return
        self._held = self._held_meta = None
        _emit(EVENT_PHASE, f"phase={PHASE_SAVING}")
        try:
            with self._episode_lock, _watchdog(_SAVE_WATCHDOG_S, "save_episode"):
                dataset.save_episode(episode_data=held)
        except Exception:
            # The count already includes it, so correct the count too — the
            # operator must not be told they have a correction that is not
            # there. `_emit` of DROPPED is what carries that correction to the
            # orchestrator's tally.
            logger.exception("Could not write the held correction; dropping it")
            self._corrections_saved = max(0, self._corrections_saved - 1)
            _emit(EVENT_ERROR, "A correction could not be written and was lost")
            _emit(
                EVENT_CORRECTION_DROPPED,
                f"n={self._corrections_saved} frames={(meta or {}).get('frames', 0)}",
            )
            return
        self._needs_push.set()
        # Give the writer an empty buffer back. `save_episode(episode_data=...)`
        # deliberately does not — upstream only clears the buffer it owns — so
        # without this the writer stays at `episode_buffer = None` for anything
        # that is not `add_frame`, and the teardown path's
        # `clear_episode_buffer()` would fault on it. Created here rather than
        # at hold time on purpose: `_create_episode_buffer` stamps
        # `total_episodes`, which is only now past the episode we just wrote,
        # and a buffer stamped with the HELD one's index would have shared its
        # temp frame directory.
        writer = self._writer(dataset)
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.episode_buffer = writer._create_episode_buffer()
        logger.info("Held correction %s written", (meta or {}).get("n"))
        _emit(EVENT_CORRECTION_COMMITTED, f"n={self._corrections_saved}")

    def _drop_held(self, dataset) -> None:
        """Throw the held correction away, temp frames and all. No-op if none.

        Uses upstream's own two cleanup paths rather than deleting files by
        hand: `clear_episode_buffer` cancels a streaming encoder and removes the
        image-feature directories, `cleanup_interrupted_episode` removes the
        video-feature ones (they are keyed by camera rather than by image key,
        which is why one call does not cover both)."""
        held, meta = self._held, self._held_meta
        if held is None:
            return
        self._held = self._held_meta = None
        frames = (meta or {}).get("frames", 0)
        writer = self._writer(dataset)
        episode_index = held.get("episode_index")
        if writer is not None:
            # Put it back just long enough for upstream's own teardown to see
            # it. `clear_episode_buffer` leaves a fresh buffer behind, which is
            # exactly the state `_hold_correction` took away.
            writer.episode_buffer = held
            with contextlib.suppress(Exception):
                writer.clear_episode_buffer(delete_images=True)
            if isinstance(episode_index, int):
                with contextlib.suppress(Exception):
                    writer.cleanup_interrupted_episode(episode_index)
        # The tally has counted it since the hand-back — it WAS a correction,
        # right up until the operator said otherwise — so undo that here.
        self._corrections_saved = max(0, self._corrections_saved - 1)
        self._recovery_frames = None
        logger.info("Held correction dropped by the operator (%d frames)", frames)
        _emit(EVENT_CORRECTION_DROPPED, f"n={self._corrections_saved} frames={frames}")

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

        Callers MUST have driven the arm somewhere safe first, and must have
        VERIFIED it got there — see `_ease_home`. An SO-101 that loses torque
        mid-reach falls under its own weight.

        Routed through `torque.force_disable_bus_torque` rather than the bus's
        own `disable_torque`, because that method walks its motors with no
        exception handling: the first failed write raises with the earlier
        motors ALREADY released, which is a half-collapsed arm. The shared
        helper goes motor by motor and reports which ones failed instead — it
        exists in this repo for exactly this hazard.

        `self._limp` is set whenever a release was ATTEMPTED, not only when it
        fully succeeded. It gates `_restore_torque`, and an arm with some
        motors released must still be re-powered later; the old code returned
        early on failure without setting it, which left those motors dead for
        the rest of the session."""
        buses = self._follower_buses(ctx.hardware.robot_wrapper)
        if not buses:
            logger.warning("No follower bus found; leaving the arm powered")
            return False
        self._limp = True
        problems: list[str] = []
        for index, bus in enumerate(buses):
            label = f"follower[{index}]" if len(buses) > 1 else "follower"
            problems.extend(force_disable_bus_torque(bus, label))
        if problems:
            # Not "the arm stays powered" — part of it very likely does not.
            logger.error(
                "Follower only partly released (%d motor(s) failed): %s",
                len(problems),
                "; ".join(problems),
            )
            return False
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

        if not hold:
            # REFUSE to energize. Without a fresh goal write the servos come
            # back on whatever Goal_Position they still hold — the pre-limp
            # target — and the arm snaps from where the operator's hands left
            # it to where the policy last wanted it. That is precisely the jerk
            # this method exists to prevent, so failing to read the pose must
            # not fall through to enabling torque anyway.
            #
            # `_limp` stays True: the arm is already unpowered, so leaving it
            # unpowered changes nothing physically, and the next transition
            # retries the whole sequence.
            logger.error("Refusing to re-power the follower: its pose could not be read")
            _emit(EVENT_ERROR, "Could not read the arm's pose, so it was left unpowered")
            return None

        failures = 0
        for bus in self._follower_buses(robot):
            try:
                bus.enable_torque()
            except Exception:
                failures += 1
                logger.exception("Could not re-enable torque on a follower bus")
        if failures:
            # Some motors may be live and some not; say so rather than
            # reporting a clean re-power. `_limp` stays True so the next
            # attempt tries again.
            logger.error("Follower only partly re-powered (%d bus failure(s))", failures)
            return None
        self._limp = False
        logger.info("Follower re-powered at its current pose")
        return hold

    def _ease_home(self, ctx) -> bool:
        """Drive the follower to its start pose and REPORT WHETHER IT ARRIVED.

        Upstream's `_return_to_initial_position` cannot fail: every exception
        inside it is caught and logged, and it never checks that the arm
        followed the interpolation it sent. So calling it and testing for a
        raised exception — which is what this code used to do — is a guard that
        can never fire. Everything downstream of the reset trusts this answer,
        including whether it is safe to cut torque, so it has to be measured.

        Measured in the robot's own observation space rather than per-bus tick
        space: `hardware.initial_position` is captured from an observation at
        connect, so comparing a fresh observation against it needs no motor-name
        mapping and no unit conversion."""
        target = getattr(ctx.hardware, "initial_position", None)
        if not target:
            # Nothing to return to. Upstream would happily interpolate toward
            # an empty dict for 3s and report nothing; that must not read as
            # "the arm is home and safe to unpower".
            logger.warning("No start pose was captured; treating the ease-home as failed")
            return False

        # POWER IT FIRST. `_return_to_initial_position` writes goal positions and
        # nothing else — on a follower still limp from an earlier reset those
        # writes land on motors that cannot act on them, the arm sits exactly
        # where it is, and the operator watches a reset that appears to strain
        # and go nowhere. Reachable from more than one direction: a second reset
        # while already parked, and a discard-and-reset from the parked state.
        self._restore_torque(ctx)

        # Scale the glide to the trip. See `_HOME_DEG_PER_S`.
        try:
            before = ctx.hardware.robot_wrapper.get_observation()
        except Exception:
            logger.exception("Could not read the arm's pose before the ease-home")
            return False
        distance = 0.0
        for key, want in target.items():
            have = before.get(key)
            if isinstance(want, (int, float)) and isinstance(have, (int, float)):
                distance = max(distance, abs(float(want) - float(have)))
        # The LEADER comes home too, and at the same time.
        #
        # Upstream's CORRECTING->PAUSED edge enables teleop torque to hold the
        # leader's pose, and nothing on the reset path releases it. So the
        # operator finished a correction, pressed reset, and was left holding a
        # rigid leader — locked wherever their hand happened to stop — for the
        # whole of the follower's glide and the whole scene rearrangement after
        # it. Bringing it back alongside the follower and releasing it there
        # means the arm they are holding does what the arm they are watching
        # does, and is free by the time they need their hands.
        #
        # Aimed at the FOLLOWER's start pose, not a leader home of its own: for
        # a matched pair the key spaces agree, and parking the leader on the
        # follower's home means the next takeover starts with a near-zero
        # offset (see `begin_correction`) instead of a large one to decay.
        leader = ctx.hardware.teleop
        leader_from = None
        if teleop_supports_feedback(leader):
            try:
                leader.enable_torque()
                leader_from = leader.get_action()
            except Exception:
                logger.exception("Could not take hold of the leader for the ease-home")
                leader_from = None
        if leader_from:
            for key, want in target.items():
                have = leader_from.get(key)
                if isinstance(want, (int, float)) and isinstance(have, (int, float)):
                    distance = max(distance, abs(float(want) - float(have)))

        duration = min(_HOME_MAX_S, max(_HOME_MIN_S, distance / _HOME_DEG_PER_S))
        logger.info(
            "Easing %s home: %.0f deg over %.2fs",
            "both arms" if leader_from else "the follower",
            distance,
            duration,
        )

        # One interpolation driving both arms, rather than upstream's
        # follower-only `_return_to_initial_position` followed by a second pass
        # for the leader. Two sequential glides would be two waits, and the
        # point of this is that the operator sees one movement.
        with contextlib.suppress(Exception):
            self._ease_both(ctx, before, leader_from, target, duration)

        try:
            obs = ctx.hardware.robot_wrapper.get_observation()
        except Exception:
            logger.exception("Could not read the arm's pose after the ease-home")
            return False

        worst = 0.0
        offenders = []
        for key, want in target.items():
            have = obs.get(key)
            if have is None or not isinstance(want, (int, float)) or not isinstance(have, (int, float)):
                # An unreadable joint is NOT an arrived joint. Skipping it here
                # would let a fully unknown arm pass the check.
                offenders.append(f"{key}:unreadable")
                continue
            delta = abs(float(want) - float(have))
            worst = max(worst, delta)
            if delta > _HOME_TOLERANCE:
                offenders.append(f"{key.removesuffix('.pos')}:{delta:.0f}")
        if offenders:
            logger.warning("Ease-home did not arrive (worst %.0f): %s", worst, ", ".join(offenders))
            return False
        logger.info("Follower is home (worst joint %.0f from target)", worst)
        return True

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
        commands replace keystrokes; the CORRECTING→PAUSED edge discards or
        HOLDS rather than always saving; PAUSED→CORRECTING captures the takeover
        offset instead of gliding an arm; a reset can end an attempt from any
        phase that is not CORRECTING; the correcting tick applies
        `follower_target`, times itself and spends its tail driving; and every
        phase edge emits a protocol event. The autonomous path and the shape of
        the correcting path are still upstream's — that is the part that drives
        the arm, and it should stay diffable.
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
        tick_budget = _TickBudget(control_interval)
        # One correcting tick's spans, held until the next tick can measure how
        # long its tail actually took. None outside a correction.
        pending_budget = None
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

                        # The held correction's last moment, and it comes FIRST —
                        # before the leader is driven anywhere, so a takeover
                        # reads as "saving…" and then "handing over" rather than
                        # parking the operator on an aligned leader while a write
                        # they thought was finished runs.
                        #
                        # Two things close the drop window, and both are here: a
                        # TAKEOVER, because the next correction needs the buffer
                        # the held one occupies; and leaving the
                        # parked-after-reset window, because that is the operator
                        # starting the next attempt and saying they are done
                        # deciding about the last one.
                        if self._held is not None and (
                            new_phase == DAggerPhase.CORRECTING or self._awaiting_attempt
                        ):
                            self._commit_held(dataset)
                        # Any transition at all means the operator has moved on
                        # from the reset.
                        self._awaiting_attempt = False

                    if transition is not None:
                        # Leaving the hold by any route other than the second
                        # takeover press. Checked BEFORE `_apply_transition` so
                        # the leader is free before anything starts moving.
                        if self._poised and not (
                            transition[0] == DAggerPhase.PAUSED and transition[1] == DAggerPhase.CORRECTING
                        ):
                            self._unpoise(ctx)
                        # Anything past here can move the arm, so it must be
                        # powered again first — including the handover glide,
                        # which would otherwise drive a limp follower.
                        restored = self._restore_torque(ctx)
                        if restored:
                            last_action = restored
                        old_phase, new_phase = transition
                        # UPSTREAM's two handover glides are suppressed, at both
                        # ends, by withholding `prev_action` from them. Ours is
                        # not: `close_the_gap` still walks the leader onto the
                        # follower a few lines below.
                        #
                        # Upstream drives the LEADER to the follower on
                        # AUTONOMOUS->PAUSED (actuated teleop) and the FOLLOWERS
                        # to the leaders on PAUSED->CORRECTING (non-actuated,
                        # i.e. bimanual). Both exist to stop the follower jumping
                        # when control changes hands, and the offset captured at
                        # the correcting edge already does that: the first
                        # commanded follower position is exactly where the
                        # follower is standing, so there is nothing left to
                        # close.
                        #
                        # Withholding them is not merely tidy. Upstream's leader
                        # glide was two seconds of dead time before every
                        # takeover and the thing whose arrival check wedged
                        # sessions; its follower glide moves real arms through a
                        # real workspace to meet a leader parked on a bench.
                        # Neither should happen for a handover that no longer
                        # needs one. The leader glide we DO run in its place is
                        # capped at `_TAKEOVER_GLIDE_MAX_S` and cannot refuse.
                        handover_action = last_action
                        if new_phase == DAggerPhase.CORRECTING or (
                            old_phase == DAggerPhase.AUTONOMOUS
                            and new_phase == DAggerPhase.PAUSED
                            and teleop_supports_feedback(ctx.hardware.teleop)
                        ):
                            handover_action = None
                        # BEFORE the blocking call, never after: stdout is
                        # flushed per event, so the orchestrator reads this while
                        # the control-loop thread is still travelling.
                        #
                        # Asked about `handover_action`, NOT `last_action`. The
                        # mirror's whole job is to say whether the call below
                        # will move hardware, and the call below is given
                        # `handover_action` — announcing a glide we have just
                        # suppressed would park the banner on "handing over" for
                        # a transition that returns immediately.
                        #
                        # Which means this branch cannot currently fire: the two
                        # edges the mirror answers True for are precisely the two
                        # on which `handover_action` was just set to None. The
                        # only HANDING_OVER an operator sees now is the one
                        # emitted around `close_the_gap` below. Deliberate, and
                        # left standing so restoring either glide restores its
                        # banner with it.
                        if self._transition_moves_the_arm(old_phase, new_phase, ctx, handover_action):
                            _emit(EVENT_PHASE, f"phase={PHASE_HANDING_OVER}")
                        self._apply_transition(
                            old_phase, new_phase, engine, interpolator, ctx, handover_action
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                            # Back under policy control: the leader has just been
                            # released, so the last teleop-driven target is no
                            # longer a pose to hold or to hand to a glide.

                        if old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
                            # The handover, in full. No arm has been moved to
                            # meet the other; instead the gap between them is
                            # measured now and cancelled out of every command
                            # until it has decayed away.
                            # The leader comes free HERE, in the same breath as
                            # recording starting. It was held on the follower's
                            # pose for the whole poise so the operator could take
                            # hold of a stationary, aligned arm; releasing any
                            # earlier would have let it sag out of position while
                            # they were still reaching for it.
                            self._poised = False
                            self._release_leader(ctx)
                            # Read both arms fresh: the transition block runs
                            # before the tick takes its observation, and an
                            # offset measured against a stale pose is an offset
                            # that does not cancel.
                            try:
                                obs_edge = ctx.hardware.robot_wrapper.get_observation()
                                # Close what the leader can, THEN measure. The
                                # order is the design: the glide shrinks the gap
                                # the follower would otherwise travel during the
                                # decay, and the offset is taken afterwards
                                # against whatever is actually left — so the
                                # glide never has to succeed at anything.
                                _emit(EVENT_PHASE, f"phase={PHASE_HANDING_OVER}")
                                self.close_the_gap(ctx, obs_edge)
                                obs_edge = ctx.hardware.robot_wrapper.get_observation()
                                leader_edge = ctx.processors.teleop_action_processor(
                                    (ctx.hardware.teleop.get_action(), obs_edge)
                                )
                                self.begin_correction(leader_edge, obs_edge)
                            except Exception:
                                # No offset AND no rate-limit reference, which
                                # means the follower SNAPS to wherever the leader
                                # is standing on the very next command:
                                # `follower_target` skips the limit entirely
                                # while `_last_target` is None, and this handler
                                # has just set it to None.
                                #
                                # That is the worst path in this file, and it is
                                # reached on exactly the failure — could not read
                                # the two arms — that leaves no trustworthy pose
                                # to clamp against, so it is not fixed here.
                                # Written down as the hazard it is rather than
                                # dressed up as a graceful degradation.
                                logger.exception("Could not measure the takeover offset")
                                self._offset = None
                                # Seed the rate limit anyway, from the follower's
                                # own pose. Leaving `_last_target` at None is what
                                # made this the most dangerous path in the file:
                                # `follower_target` skips the clamp entirely while
                                # it is None, so the first command was a raw jump
                                # to wherever the leader stood — on exactly the
                                # path (bus trouble at the handover edge) where
                                # the two arms are most likely far apart. Without
                                # an offset the follower still has to travel to
                                # meet the leader; the clamp is what makes that a
                                # glide instead of a snap.
                                self._last_target = None
                                with contextlib.suppress(Exception):
                                    pose = ctx.hardware.robot_wrapper.get_observation()
                                    self._last_target = {
                                        k: float(v)
                                        for k, v in pose.items()
                                        if k.endswith(".pos") and isinstance(v, (int, float))
                                    }
                                self._last_target_at = time.perf_counter()
                            # Fresh window per takeover: a budget carried across
                            # the handover glide would open every correction with
                            # a 2s "tick" and bury the thing being measured.
                            tick_budget.reset()
                            correction_frames = 0
                            correction_started_at = time.perf_counter()
                            self._recovery_frames = None
                            self._recovery_mark_requested = False

                        # Correction ended. Upstream saves unconditionally; we
                        # honour a CANCEL armed by the operator instead.
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            # No PHASE_SAVING here any more: nothing is written
                            # at this edge. The correction is HELD in memory and
                            # committed later (see `_commit_held`), which is
                            # what leaves the operator a window to drop it —
                            # and which also takes a 0.4-2.3s synchronous write
                            # off the hand-back, where the arm is mid-handover.
                            # `_commit_held` announces the save at the point it
                            # actually happens.
                            seconds = time.perf_counter() - correction_started_at
                            if self._cancel_correction:
                                self._cancel_correction = False
                                with self._episode_lock:
                                    dataset.clear_episode_buffer()
                                logger.info(
                                    "Correction discarded (%.1fs, %d frames)", seconds, correction_frames
                                )
                                _emit(
                                    EVENT_CORRECTION_CANCELLED,
                                    f"reason={CANCEL_REASON_OPERATOR} frames={correction_frames} "
                                    f"seconds={seconds:.1f}",
                                )
                                # Release the leader, or a discard strands it.
                                #
                                # Upstream's CORRECTING->PAUSED enables teleop
                                # torque to hold the leader's pose, and the only
                                # thing that ever releases it again is the
                                # ->AUTONOMOUS transition. A HANDBACK reaches
                                # that; a CANCEL deliberately STOPS at PAUSED
                                # (see `_translate`), so after a discard the
                                # leader stayed rigid with no scheduled
                                # transition that would ever free it — the arm
                                # "stuck there" with nothing on screen to
                                # explain it.
                                #
                                # Releasing no longer creates alignment debt.
                                # It used to: the next takeover sent the
                                # follower to wherever the operator had since
                                # moved the leader, so the release had to be
                                # flagged and paid back by a glide. The takeover
                                # offset absorbs that now — see
                                # `begin_correction` — so letting go is free.
                                #
                                # Routed through `_release_after_discard` so it
                                # does not double up with the reset a discard
                                # always arms: `_ease_home` re-torques the leader
                                # to walk it home and releases it there, and
                                # slack-grab-slack inside a second is its own
                                # fault report.
                                self._release_after_discard(ctx)
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
                                # `reason` is what lets the UI tell the operator
                                # their work was binned. The operator asked for
                                # the branch above; they did NOT ask for this
                                # one, and a silent discard here is how a
                                # deliberate quick nudge disappears without a
                                # trace. See CANCEL_REASON_TOO_SHORT.
                                _emit(
                                    EVENT_CORRECTION_CANCELLED,
                                    f"reason={CANCEL_REASON_TOO_SHORT} frames={correction_frames} "
                                    f"seconds={seconds:.1f} minimum={_MIN_CORRECTION_FRAMES}",
                                )
                            else:
                                # HELD, not written. The operator has just
                                # watched this correction happen and is about to
                                # reset the scene; that is the moment they can
                                # actually judge it, and the only moment at
                                # which un-recording it is still possible.
                                # Committed on the way into the next takeover or
                                # the next attempt — see `_commit_held` and
                                # dagger_protocol's "The held correction".
                                self._corrections_saved += 1
                                held = self._hold_correction(
                                    dataset, frames=correction_frames, seconds=seconds
                                )
                                if not held:
                                    # Nothing to detach. Save inline rather than
                                    # let the correction evaporate: a hold that
                                    # held nothing would report a correction the
                                    # dataset never received.
                                    logger.warning("Could not hold the correction; writing it immediately")
                                    _emit(EVENT_PHASE, f"phase={PHASE_SAVING}")
                                    with self._episode_lock, _watchdog(_SAVE_WATCHDOG_S, "save_episode"):
                                        dataset.save_episode()
                                    self._needs_push.set()
                                logger.info(
                                    "Correction %d/%s recorded (%.1fs, %d frames)%s",
                                    self._corrections_saved,
                                    target,
                                    seconds,
                                    correction_frames,
                                    " — held, droppable" if held else "",
                                )
                                # `labelled` and `recovery` are separate on
                                # purpose: an unmarked correction is not a
                                # correction whose recovery took zero frames,
                                # and a consumer that conflated them would
                                # train on a claim the operator never made.
                                marked = self._recovery_frames is not None
                                # CORRECTION_HELD where this used to emit
                                # CORRECTION_SAVED. The orchestrator tallies
                                # both identically — the correction is real and
                                # counted either way — but only HELD opens the
                                # drop window, and conflating "recorded" with
                                # "on disk" is what would let the UI offer a
                                # delete for something it can no longer take
                                # back. The shutdown path still emits SAVED,
                                # because it writes directly.
                                _emit(
                                    EVENT_CORRECTION_HELD if held else EVENT_CORRECTION_SAVED,
                                    f"n={self._corrections_saved} frames={correction_frames} "
                                    f"seconds={seconds:.1f} "
                                    f"recovery={self._recovery_frames if marked else -1} "
                                    f"labelled={'true' if marked else 'false'}",
                                )

                        # The correction is resolved — saved or binned — so the
                        # counter that describes it must not survive into the
                        # shutdown block, which branches on it to decide whether
                        # there is an in-flight correction worth saving. Leaving
                        # it set meant every session that ended after a normal
                        # correction tried to save an already-flushed buffer and
                        # logged a ValueError traceback as its last words.
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            self.end_correction()
                            correction_frames = 0
                            self._recovery_frames = None
                            self._recovery_mark_requested = False

                        _emit(EVENT_PHASE, f"phase={new_phase.value}")

                    # MakerMods Lab: an attempt at the TASK is over. Corrections-only
                    # DAgger has no task-episode of its own, so without this the
                    # policy just kept driving at a finished scene. Gated only on
                    # NOT being mid-correction; from AUTONOMOUS it stops the
                    # engine itself a few lines below, because the arm must not be
                    # eased home while the policy is still commanding it. It
                    # blocks the loop for the whole of the glide, which is why it
                    # announces itself first.
                    if self._reset_requested and events.phase != DAggerPhase.CORRECTING:
                        self._reset_requested = False
                        # A reset requests no transition, so the guard above
                        # never sees it — clear the hold here or the leader
                        # stays powered through the whole ease-home.
                        self._poised = False
                        self._poise_requested = False
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
                        # A failed ease-home must not end the session: the arm
                        # stays where it is and the operator repositions the
                        # scene around it. But it MUST be detected — see
                        # `_ease_home`, which measures arrival instead of
                        # relying on an exception upstream never raises.
                        homed = self._ease_home(ctx)
                        # Limp ONLY from a pose we know is safe. If the ease-home
                        # did not arrive, the arm is somewhere arbitrary — quite
                        # possibly extended — and cutting torque there drops it.
                        limp = False
                        if homed:
                            limp = self._go_limp(ctx)
                        else:
                            logger.warning("Skipping limp: the arm is not at its home pose")
                        # The leader goes slack whether or not the FOLLOWER made
                        # it home. The two conditions are not the same: the
                        # follower may be extended over the table holding a
                        # gripper, which is why cutting its torque is gated on
                        # arriving; the leader is a hand-held arm that every
                        # other flow in this app leaves back-drivable, and it has
                        # just been walked to the same home pose. Leaving it
                        # rigid because the follower stalled would punish the
                        # operator's hands for the robot's problem.
                        self._release_leader(ctx)
                        last_action = None
                        # Parked. The held correction stays droppable for the
                        # whole of this window — it is the window the operator
                        # actually decides in, standing at a stationary arm
                        # having just watched the correction happen.
                        self._awaiting_attempt = True
                        # Carry the OUTCOME, not just the count. Both failure
                        # modes above leave an arm the operator must not be
                        # told to grab: a failed ease-home leaves it mid-task
                        # and rigid, a failed release leaves it rigid at home.
                        # Emitting the same event for all three states is how
                        # the UI came to say "limp, reposition freely" over an
                        # arm holding six torqued servos.
                        _emit(
                            EVENT_ATTEMPT_RESET,
                            f"n={self._attempts} homed={'true' if homed else 'false'} "
                            f"limp={'true' if limp else 'false'}",
                        )
                        _emit(EVENT_PHASE, f"phase={DAggerPhase.PAUSED.value}")

                    # The operator un-recorded the held correction. Handled
                    # here rather than in `_translate` for the same reason the
                    # recovery mark is: the dataset is only in scope on the
                    # control loop. Cheap and non-blocking — it deletes temp
                    # frames and forgets a buffer — so it does not need the
                    # parked-only treatment the reset gets.
                    # Poise: glide the leader onto the follower and HOLD both
                    # arms there. Runs only from PAUSED — the policy must be
                    # stopped before the leader is driven, and the operator must
                    # not be asked to grab an arm that is still working.
                    if self._poise_requested and events.phase == DAggerPhase.PAUSED:
                        self._poise_requested = False
                        _emit(EVENT_PHASE, f"phase={PHASE_HANDING_OVER}")
                        self._restore_torque(ctx)
                        here = robot.get_observation()
                        self.close_the_gap(ctx, here)
                        self._hold_leader(ctx, here)
                        self._poised = True
                        logger.info("Poised — both arms held, waiting for the operator to take control")
                        _emit(EVENT_PHASE, f"phase={PHASE_POISED}")

                    if self._drop_last_requested:
                        self._drop_last_requested = False
                        self._drop_held(dataset)

                    phase = events.phase
                    # Close out the previous correcting tick now that its tail
                    # (the drive, or the sleep) is behind us. Measured here
                    # rather than at the end of the tick because the gap this
                    # exists to find was in exactly the part a tick cannot time
                    # from inside itself: the loop was doing 3.7ms of work and
                    # taking 86ms to come round again, and nothing in the
                    # per-span numbers could say where the other 82ms went.
                    if pending_budget is not None:
                        prev_spans, prev_total, prev_start = pending_budget
                        pending_budget = None
                        now = time.perf_counter()
                        prev_spans["tail"] = now - prev_start - prev_total
                        prev_spans["drove"] = float(self._drive_sent)
                        tick_budget.add(prev_spans, now - prev_start)
                        tick_budget.report_if_due()

                    mark = time.perf_counter()
                    obs = robot.get_observation()
                    # Timed for the budget report below. One follower `sync_read`
                    # plus a `read_latest()` memory copy per camera — non-blocking
                    # by design, which is exactly the claim worth checking against
                    # a real bus rather than assuming.
                    obs_dt = time.perf_counter() - mark

                    # --- CORRECTING: human teleop control + recording ---
                    if phase == DAggerPhase.CORRECTING:
                        # Timed span by span. This is the loop the operator feels
                        # in their hand — every millisecond here is a millisecond
                        # of lag between the leader moving and the follower
                        # following — and it is the one place a measurement beats
                        # an argument. See `_TickBudget`.
                        spans = {"observe": obs_dt}
                        mark = time.perf_counter()
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        spans["process"] = time.perf_counter() - mark

                        mark = time.perf_counter()
                        teleop_action = teleop.get_action()
                        spans["leader"] = time.perf_counter() - mark

                        mark = time.perf_counter()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        # Offset + rate limit, applied HERE — to the value that
                        # is about to be both recorded and sent. Doing it after
                        # `robot_action_processor` instead would make the dataset
                        # record a leader pose the follower never went to.
                        processed_teleop = self.follower_target(processed_teleop, time.perf_counter())
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        spans["send"] = time.perf_counter() - mark
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        mark = time.perf_counter()
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
                        spans["record"] = time.perf_counter() - mark
                        # Recorded now; the drive span and the true period are
                        # folded in at the top of the NEXT tick, because both
                        # only finish after this tick's tail has run.
                        pending_budget = (spans, time.perf_counter() - loop_start, loop_start)

                        # RaC: "the arm is back somewhere sane — the correction
                        # starts here." Read AFTER this tick's frame is counted,
                        # so the boundary lands between the frame the operator
                        # was looking at when they pressed and the next one.
                        if self._recovery_mark_requested:
                            self._recovery_mark_requested = False
                            self._recovery_frames = correction_frames
                            logger.info("Recovery marked complete at %d frames", correction_frames)
                            _emit(EVENT_RECOVERY_MARK, f"frames={correction_frames}")

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
                        if phase == DAggerPhase.CORRECTING:
                            # Spend the rest of the tick DRIVING, not sleeping.
                            # The dataset still gets one frame per tick; the
                            # follower gets a new goal position every
                            # millisecond instead of once a tick. See
                            # `_drive_until`.
                            last_action = self._drive_until(
                                loop_start + control_interval, ctx, robot, teleop, obs, last_action
                            )
                        else:
                            precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Coaching loop is running slower ({1 / dt:.1f} Hz) than the target FPS "
                            f"({cfg.fps} Hz). Dataset frames might be dropped and robot control might be "
                            f"unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy "
                            f"inference taking too long 3) CPU starvation"
                        )

            except BaseException:
                # The loop did not end because the operator asked it to — a
                # camera was unplugged, a bus went away, something raised. Note
                # it and re-raise unchanged; the `finally` below needs to know,
                # and the exception is what the orchestrator reports.
                self._loop_failed = True
                raise
            finally:
                logger.info("Coaching control loop ended — pausing engine")
                engine.pause()
                # Bring the leader back and let it go, exactly as a reset does.
                # Upstream's teardown eases the FOLLOWER home and disconnects;
                # nothing ever touched the leader, so a session that ended after
                # a correction left it locked wherever the operator's hand had
                # stopped — reported as "at the very end the leader also kind of
                # locks in place; it doesn't gently go back like the follower
                # does". Best-effort and non-fatal: teardown must not be blocked
                # by an arm that is only there for the operator's comfort.
                #
                # The release is in a `finally`, and the poses are read BEFORE
                # the leader is energised. Both matter, and the original shape
                # of this block got both wrong: it enabled leader torque and
                # then evaluated `get_observation()` as an argument to the
                # ease, all inside ONE `suppress`. A read that raised — a bus
                # dropout, an unplugged camera, i.e. exactly the fault that
                # brings us here — was swallowed, and the release below never
                # ran, leaving the leader rigid a line after we powered it.
                # Reading first means a failed read costs us the ease-home, not
                # the operator's hand.
                try:
                    self._unpoise(ctx)
                    target = getattr(ctx.hardware, "initial_position", None)
                    leader = ctx.hardware.teleop
                    if target and teleop_supports_feedback(leader):
                        follower_from = ctx.hardware.robot_wrapper.get_observation()
                        leader.enable_torque()
                        self._ease_both(ctx, follower_from, leader.get_action(), target, _HOME_MAX_S)
                except Exception:
                    logger.exception("Could not walk the arms home on teardown")
                finally:
                    with contextlib.suppress(Exception):
                        self._release_leader(ctx)

                # The held correction is real work and it is complete; the only
                # reason it is not on disk is that the operator was still
                # entitled to bin it. They are not any more, so write it.
                #
                # FIRST, before the in-flight buffer below: both go through the
                # writer's single episode buffer, and upstream refuses any
                # buffer whose `episode_index` is not the dataset's current
                # `total_episodes`. Committing the held one first is what keeps
                # that true for the other. (This ordering is also why exactly
                # one correction is ever held — see dagger_protocol.)
                #
                # `_commit_held` swallows its own failures, so a bad write here
                # cannot mask the exception that ended the session.
                self._commit_held(dataset)
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
                # An armed CANCEL outranks the frame count. The operator
                # pressed discard and then stopped the session before the
                # control loop reached the edge that consumes it; saving here
                # would put the take they explicitly binned into the dataset.
                # A HARDWARE FAULT outranks both of those. The frames up to the
                # fault look perfectly ordinary — a camera that has been
                # unplugged does not blank the frame, it stops updating, so the
                # episode is written with a stale image repeated to the end and
                # nothing downstream can tell. Observed on the bench: unplugging
                # a camera mid-session errored correctly and kept the earlier
                # episodes, and ALSO kept the one being recorded through the dead
                # camera. That episode is poison in exactly the way a fumbled
                # takeover is, minus any way for the operator to notice.
                #
                # The HELD correction above is a different matter and is still
                # committed: it finished before the fault, with every camera
                # working.
                if self._loop_failed or self._cancel_correction or correction_frames < _MIN_CORRECTION_FRAMES:
                    if self._loop_failed and correction_frames:
                        logger.warning(
                            "Dropping the %d-frame in-flight correction: the session failed "
                            "mid-take, so its last frames cannot be trusted",
                            correction_frames,
                        )
                        _emit(
                            EVENT_CORRECTION_CANCELLED,
                            f"reason={CANCEL_REASON_FAULT} frames={correction_frames}",
                        )
                    elif self._cancel_correction:
                        logger.info("Dropping a discarded correction on shutdown, as asked")
                    elif correction_frames:
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
                        marked = self._recovery_frames is not None
                        _emit(
                            EVENT_CORRECTION_SAVED,
                            f"n={self._corrections_saved} frames={correction_frames} "
                            f"seconds={time.perf_counter() - correction_started_at:.1f} "
                            f"recovery={self._recovery_frames if marked else -1} "
                            f"labelled={'true' if marked else 'false'}",
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
    # lerobot's init_logging leaves a formatter that never renders exc_info, so
    # every `logger.exception` in this process would otherwise be an ordinary
    # one-line error. See log_exceptions — this cost a day of diagnosing a
    # takeover glide failure whose exception was never written down.
    restore_traceback_rendering()

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
        # `cfg.dataset.root` is the root the ORCHESTRATOR asked for, and it
        # deliberately asks for none (so lerobot places the dataset under
        # HF_LEROBOT_HOME/<stamped id>). lerobot never writes the resolved path
        # back onto the config, so this used to put the literal string "None"
        # on the wire — which the orchestrator then treated as a directory and
        # created, filing the recovery sidecar under ./None/ beside the server.
        # Read it off the dataset object, which is the only side that knows.
        created = getattr(ctx.data, "dataset", None)
        root = getattr(created, "root", None) if created is not None else None
        _emit(EVENT_DATASET, f"repo_id={cfg.dataset.repo_id} root={root if root else ''}")

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
