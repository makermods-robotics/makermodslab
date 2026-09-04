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

"""Line protocol spoken between the coaching orchestrator and the DAgger runner.

Its own module for the same reason `makermodslab.eval_protocol` is: BOTH ends need
the strings and they live on opposite sides of a dependency wall — the
orchestrator (`makermodslab.rollout`) is imported by the FastAPI server at boot and
must NOT drag in `lerobot.rollout` (torch, datasets, …), while the runner
(`makermodslab.dagger_runner`) is exactly the process that does. Restating the
markers in both modules would be a silent-drift bug the first time one side
renamed one: the orchestrator would simply stop recognising phase changes and
the session UI would freeze on a stale phase while the arm kept moving.

Deliberately kept separate from `eval_protocol` rather than merged into one
"runner protocol" module. The two sessions have disjoint vocabularies (episodes
vs. corrections) and disjoint failure modes, and a shared event prefix would
make a log line from one indistinguishable from the other.

Commands — orchestrator → runner stdin, one bare word per line:
    TAKEOVER  ONE STEP OF TWO. From `autonomous` it stops the policy, glides the
              leader onto the follower and HOLDS both arms there (`poised`),
              recording nothing. Sent again from `poised` it releases the leader
              and starts the correction. Two presses, so the operator settles
              their grip on a stationary arm before any frame is kept.
    HANDBACK  end the correction, SAVE it, and return control to the policy
    CANCEL    discard the correction and reset: bin the frames, ease the follower
              home, park for a scene reset. Valid from EVERY phase — with
              nothing in flight it is a plain reset — which is what makes it the
              way out of a wedged takeover as well as the fumbled-takeover
              escape.
    HOLD      freeze the policy without taking over
    RESUME    unfreeze, returning control to the policy
    RECOVERED mid-correction: "the arm is back somewhere sane, the correction
              starts here" — records a boundary, changes no phase (see below)
    RESET     end this attempt: ease the follower home and park for a scene reset.
              Mid-correction it SAVES the correction in flight first — the
              operator finishing the task while still driving has already
              decided those frames are the correction.
    DROP_LAST un-record the correction that is still HELD (see "The held
              correction" below). A no-op if nothing is held.
    QUIT      finalize the dataset, ease the arm home, disconnect, exit

TAKEOVER and HANDBACK are COMPOSITE: each drives two of lerobot's phase
transitions back to back (AUTONOMOUS→PAUSED→CORRECTING and the reverse). That
composition lives here, on the runner side, rather than in the UI, because each
transition carries hardware side effects — the leader's torque is flipped on at
PAUSED and off again on the way back to AUTONOMOUS — and the runner applies one
per control tick so each completes before the next is asked for. A UI that sent
two separate commands could interleave something between them; a UI that sent
them "atomically" would still be guessing when the first had finished. One
command, one hardware sequence.

Upstream ALSO glides an arm at those edges (leader-to-follower over ~2s on an
actuated teleop, followers-to-leaders on a bimanual one). The runner suppresses
both and does its own bounded leader glide at the PAUSED→CORRECTING edge
instead — see `close_the_gap` and `_OFFSET_DECAY_S` in `dagger_runner`.

Events — runner → stdout, one line each, carrying a grep-able prefix so the
orchestrator's log pump can pick them out of lerobot's own INFO chatter (the
runner's stderr is merged into the same pipe):
    MAKERMODSLAB-DAGGER READY
    MAKERMODSLAB-DAGGER DATASET repo_id=<final id> root=<path>
    MAKERMODSLAB-DAGGER PHASE phase=autonomous|paused|correcting
    MAKERMODSLAB-DAGGER CORRECTION_SAVED n=<count> frames=<n> seconds=<s>
                                        recovery=<n|-1> labelled=<true|false>
    MAKERMODSLAB-DAGGER CORRECTION_CANCELLED reason=<operator|too_short> …
    MAKERMODSLAB-DAGGER CORRECTION_HELD n=<count> frames=<n> seconds=<s>
                                        recovery=<n|-1> labelled=<true|false>
    MAKERMODSLAB-DAGGER CORRECTION_COMMITTED n=<count>
    MAKERMODSLAB-DAGGER CORRECTION_DROPPED n=<count after the drop> frames=<n>
    MAKERMODSLAB-DAGGER RECOVERY_MARK frames=<n>
    MAKERMODSLAB-DAGGER ATTEMPT_RESET n=<count> homed=<true|false>
                                      limp=<true|false>
    MAKERMODSLAB-DAGGER ERROR <message, whitespace collapsed to one line>
    MAKERMODSLAB-DAGGER BYE

ALIGN_REQUIRED is NOT in that list any more. Nothing emits it; the alignment
gate it belonged to was deleted along with the takeover refusal. The constant
and the orchestrator's handler for it still exist — see `EVENT_ALIGN_REQUIRED`
below.

The held correction, and why DROP_LAST is not a delete
------------------------------------------------------

An operator watching their own correction back in their head immediately after
an attempt knows, most of the time, whether it was any good — and until now the
only thing they could do about a bad one was throw away the WHOLE session's
dataset from the summary screen afterwards.

There is no lerobot API for removing one episode from a dataset that is still
being written. `save_episode` interleaves the episode's frames into a shared
per-chunk parquet file and appends its video into a shared per-chunk video file
(see `DatasetWriter._save_episode_data` / `_save_episode_video`), then folds its
stats into the running aggregate. `dataset_tools.delete_episodes` exists, but it
rebuilds a *finalized* dataset into a new directory by copying and re-encoding
every episode that survives — minutes of work, and invalid against an open
writer. Doing that surgery by hand mid-session is how a coaching dataset ends up
subtly corrupt in a way nobody notices until training.

So the correction is not deleted. It is NOT YET WRITTEN.

When a correction ends, the runner detaches the writer's episode buffer and
HOLDS it instead of saving it, emitting CORRECTION_HELD. The frames are complete
and in memory; nothing has touched the dataset. The hold ends, one way or the
other, at the next moment the buffer is needed:

  * a new takeover begins        -> commit it (CORRECTION_COMMITTED), then
                                    record the new correction over a fresh
                                    buffer;
  * the next attempt begins      -> commit it. Leaving the parked-after-reset
                                    window IS the operator saying they are done
                                    deciding about the last one;
  * the session ends             -> commit it in the teardown path;
  * DROP_LAST arrives first      -> discard it (CORRECTION_DROPPED). The temp
                                    frame images go with it and the dataset
                                    never learns it existed.

Exactly ONE correction is ever held, which is what keeps this compatible with
upstream's `validate_episode_buffer`: it refuses any buffer whose
`episode_index` is not the dataset's current `total_episodes`, and a held buffer
keeps that property precisely because nothing else can be written while it is
held.

DELIBERATELY ONE LEVEL, FOR NOW
-------------------------------

Only the most recent correction can be taken back. A second DROP_LAST has
nothing to act on, and the UI says so rather than failing quietly.

That is a staging limit, not a decision that one level is the right number. The
owner wants multi-level undo and asked for it to be deferred so the single-level
feature could be isolated and shipped first. What stands in the way is concrete:
`add_frame` writes each episode's temporary frames into a directory keyed by
`episode_index`, and an uncommitted buffer takes its index from
`total_episodes` — which does not advance until something is written. Two held
corrections therefore carry the SAME index and would overwrite each other's
frames. Supporting more than one means staging each held episode's frames under
an index of its own and rewriting the buffer's stored paths at commit time,
which is surgery on the one part of the data path that has been verified end to
end (episode counts, frame counts, timestamps and the RaC sidecar all agree
after a drop).

It may also turn out to be the wrong shape entirely. A design that deferred
every commit to the end of the session, or that rebuilt the dataset at finalize
from a keep-list, would make this whole mechanism unnecessary rather than
extend it. Worth scoping before building.

The window is therefore real but narrow — from hand-back until the operator
either takes over again or starts the next attempt — and it is deliberately the
window the operator is already standing in: the arm is parked, the scene is
being reset, and they have just watched the thing they are deciding about. The
orchestrator reports the window as `droppable_correction`; the browser must not
infer it from the phase.

The cost is honest and worth naming: a session that dies between the hand-back
and the commit loses that one correction. That is the same exposure the
in-flight correction has always had, moved back by a few seconds, and it buys
the only version of "delete that one" that cannot corrupt the dataset.

RECOVERED and the recovery boundary
------------------------------------

An intervention is two different things wearing one name. lerobot's own HIL
guide says so: the DAgger strategy follows RaC (Hu et al., 2025,
arXiv:2509.07953), which "explicitly decompos[es] interventions into recovery
(teleoperating back to a good state) and correction (demonstrating the right
behavior from there)". RaC reports 10x less collection time than baselines from
that decomposition alone, and that policy performance scales linearly in the
number of recovery maneuvers the trained policy exhibits.

The strategy records neither half separately — every frame between takeover and
hand-back gets one undifferentiated `intervention=True`. We cannot fix that
where it belongs: the dataset's feature dict is assembled inside
`build_rollout_context` with `intervention` hardcoded (lerobot rollout/context.py)
and no hook to add a column, so a per-frame label would mean monkeypatching or
vendoring dataset creation — the coupling this branch already has too much of.

So the boundary is recorded OUT OF BAND. RECOVERED marks the frame at which the
operator judges the arm to be back in a sane state; the runner reports it on
CORRECTION_HELD (or CORRECTION_SAVED, on the shutdown path that writes
directly), and the orchestrator writes per-episode counts to a sidecar
next to the dataset. Nothing consumes it at training time yet. It is written
anyway because it is unrecoverable after the fact — nobody can look at a
finished episode later and say where recovery ended — and because the gesture
itself is most of the value: RaC's mechanism is the OPERATOR working in
recovery-then-correction terms, which they will not do while the UI calls the
whole thing "you're driving".

RECOVERED deliberately requests NO phase transition. Recovery and correction are
the same control mode — human on the leader, frames recording — and inventing a
lerobot phase for a distinction lerobot does not have would put the vendored
loop and the upstream state machine out of step for no gain. It is an
annotation, not a state change.

`labelled=false` means the operator never pressed it, which is NOT the same as
"recovery took zero frames": one says unannotated, the other says the operator
went straight to correcting. A consumer that conflates them would train on a lie.

DATASET is not a nicety. lerobot stamps a timestamp onto the rollout dataset's
repo_id inside `build_rollout_context` (`DatasetRecordConfig.stamp_repo_id`,
called unconditionally — see lerobot issue #3722, closed without an opt-out), so
the name the user typed is NOT the name on disk. The orchestrator must be told
the resolved id rather than reconstruct it, because reconstructing it means
guessing the second at which the subprocess reached that line.
"""

from __future__ import annotations

# --- Commands (stdin) -------------------------------------------------------
CMD_TAKEOVER = "TAKEOVER"
CMD_HANDBACK = "HANDBACK"
# Discard AND reset, in one verb and from any phase.
#
# It used to stop at PAUSED and a second verb, RECOVER, existed to do the
# discard-then-reset pair — same two effects, a second name, a second button and
# a second endpoint. The operator has to make the same decision either way
# ("this take is rubbish and the scene needs setting up again"), so it is one
# control now. The from-any-phase behaviour is RECOVER's, kept deliberately:
# that is what gives a wedged correction a way out.
CMD_CANCEL = "CANCEL"
CMD_HOLD = "HOLD"
CMD_RESUME = "RESUME"
CMD_RESET = "RESET"
# "I have the arm back somewhere sane; the correction starts here." Valid only
# mid-correction, records a boundary, changes no phase. See the module docstring.
CMD_RECOVERED = "RECOVERED"
# "That last one was no good." Un-records the HELD correction — the one that
# ended at the last hand-back and has not been written yet. A no-op once it has
# been committed, which is the point: see "The held correction" above.
CMD_DROP_LAST = "DROP_LAST"
CMD_QUIT = "QUIT"

# Every command the runner accepts. The orchestrator validates against this
# before writing to the pipe so a typo'd endpoint fails at the HTTP layer with a
# 400 rather than silently reaching a runner that logs "unrecognised" and
# carries on driving the arm.
COMMANDS = frozenset(
    {
        CMD_TAKEOVER,
        CMD_HANDBACK,
        CMD_CANCEL,
        CMD_HOLD,
        CMD_RESUME,
        CMD_RESET,
        CMD_RECOVERED,
        CMD_DROP_LAST,
        CMD_QUIT,
    }
)

# --- Events (stdout) --------------------------------------------------------
# Distinct from eval_protocol's prefix on purpose: both runners tee into
# `inference_logs/`, and a shared prefix would make a coaching line and an eval
# line indistinguishable in a log a human is reading after the fact.
EVENT_PREFIX = "MAKERMODSLAB-DAGGER"

EVENT_READY = "READY"
EVENT_DATASET = "DATASET"
EVENT_PHASE = "PHASE"
EVENT_CORRECTION_SAVED = "CORRECTION_SAVED"
EVENT_CORRECTION_CANCELLED = "CORRECTION_CANCELLED"

# DEAD ON THE WIRE. Nothing emits this any more: the takeover no longer measures
# a leader gap and no longer refuses, so there is no alignment to require (see
# `_OFFSET_DECAY_S` in `dagger_runner` for what replaced it, and why the refusal
# had to go — it wedged sessions). The constant and `rollout`'s handler for it
# survive so an OLD runner still speaking it is understood rather than logged as
# an unknown event; delete both together, or not at all.
EVENT_ALIGN_REQUIRED = "ALIGN_REQUIRED"

# The three events of the held correction's life. Separate from
# CORRECTION_SAVED rather than a flag on it, because the orchestrator has to
# tell "this exists and can still be un-recorded" from "this is on disk", and a
# consumer that misses one event must not silently read the other as the whole
# story. See "The held correction" in the module docstring.
#
# CORRECTION_HELD is emitted where CORRECTION_SAVED used to be — the correction
# is complete and counted, it simply is not written yet.
EVENT_CORRECTION_HELD = "CORRECTION_HELD"
# The held correction reached disk. Carries the count so the orchestrator's
# tally and the runner's cannot drift.
EVENT_CORRECTION_COMMITTED = "CORRECTION_COMMITTED"
# The operator un-recorded it. `n` is the count AFTER the drop, so the UI can
# take the number verbatim rather than decrementing its own and hoping.
EVENT_CORRECTION_DROPPED = "CORRECTION_DROPPED"

# `CORRECTION_CANCELLED reason=` values. A discard has two very different
# causes and the operator needs to be able to tell them apart:
#
#   * `operator` — they pressed discard. They already know; the count not
#     moving is feedback enough, and saying more would be nagging.
#   * `too_short` — the runner binned it under `_MIN_CORRECTION_FRAMES`. The
#     operator did NOT ask for this. They took over, did something they meant,
#     handed back, and their work went in the bin with nothing on screen to say
#     so. That is the one case that MUST be surfaced.
#
# The distinction is not cosmetic. The correction literature (CR-DAgger,
# arXiv:2506.16685) finds that the frames right after an intervention starts
# are the *most* valuable in the episode — "sampling denser right after
# intervention starts leads to more reactive and accurate corrections" — so a
# quick, deliberate nudge is exactly the data we most want and exactly what the
# frame floor throws away. Discarding is still correct (a one-frame episode
# breaks lerobot's stats aggregation and takes the session down with it), but
# it has to be loud.
CANCEL_REASON_OPERATOR = "operator"
CANCEL_REASON_TOO_SHORT = "too_short"
# The session died mid-take — a camera unplugged, a bus went away. The frames up
# to the fault look ordinary (an unplugged camera stops updating rather than
# blanking, so the episode ends with a stale image repeated), which is precisely
# why this correction must be thrown away rather than trusted: nothing
# downstream could tell it apart from a good one.
CANCEL_REASON_FAULT = "fault"
# One attempt at the task ended. Carries the running attempt count so the UI can
# show "attempt 4" without keeping its own tally that a dropped event would
# desynchronise, AND the outcome of the two things that can fail on the way:
# `homed` (did the follower actually reach its start pose) and `limp` (did its
# torque actually come off). Both can be false with the attempt still over, and
# the UI must not say "reposition it freely" over an arm holding six torqued
# servos — which is exactly what a bare count once let it do.
EVENT_ATTEMPT_RESET = "ATTEMPT_RESET"
# The operator marked the end of recovery inside the correction in progress.
# Carries the frame count at the boundary so the UI can show it live; the
# authoritative per-episode record arrives on CORRECTION_HELD (or, on the
# shutdown path, CORRECTION_SAVED).
EVENT_RECOVERY_MARK = "RECOVERY_MARK"

# Filename of the sidecar the orchestrator writes into the dataset directory,
# holding the recovery/correction split per episode. Named for this app rather
# than dropped into `meta/` so it can never be mistaken for something lerobot
# wrote or expects, and so a `LeRobotDataset` load ignores it entirely.
RAC_SIDECAR_NAME = "makermodslab_rac.json"
EVENT_ERROR = "ERROR"
EVENT_BYE = "BYE"

# `PHASE phase=` values. These are lerobot's own `DAggerPhase` values verbatim
# (`lerobot.rollout.strategies.dagger.DAggerPhase`), passed through rather than
# renamed, so a log line here and a log line from lerobot agree. The operator-
# facing names ("Watching" / "Holding" / "You're driving") are a FRONTEND
# concern — see InferenceSessionDialog. Don't translate them on the wire: the
# wire should stay greppable against upstream's own logging.
PHASE_AUTONOMOUS = "autonomous"
PHASE_PAUSED = "paused"
PHASE_CORRECTING = "correcting"

# NOT one of lerobot's phases — ours. It covers a window in which an arm is
# PHYSICALLY TRAVELLING and no lerobot phase describes the truth.
#
# It was written for upstream's handovers: single-arm, the leader is driven
# under torque to the follower's pose (`teleop_smooth_move_to`, ~2s) while
# lerobot still reports AUTONOMOUS; bimanual, BOTH followers slide to meet the
# leaders while lerobot already reports PAUSED. That second case is why it
# exists — PAUSED renders as "the arm is frozen", and showing that sentence to
# an operator while two arms sweep the workspace is the worst thing this UI
# could say.
#
# The runner now suppresses both of those glides, so in practice this covers
# `close_the_gap`: the runner's own leader glide at the PAUSED→CORRECTING edge,
# capped at `_TAKEOVER_GLIDE_MAX_S`. Emitted before the blocking call, with the
# real phase after it.
PHASE_HANDING_OVER = "handing_over"

# Also ours. `save_episode()` runs synchronously on the control loop — it writes
# the parquet and encodes the episode's video — so the arm sits frozen for its
# duration, 0.4-2.3s on the station. Without a state of its own that window
# inherited the phase either side of it, which meant the banner kept insisting
# the operator was still driving and recording long after they had let go.
#
# No longer emitted at the hand-back edge: nothing is written there any more.
# `_commit_held` emits it at the moment the write actually happens — on the way
# into the next takeover or the next attempt, and in the teardown path.
PHASE_SAVING = "saving"

# Also ours. The operator has declared the current ATTEMPT at the task over
# (the cube is in the tray, or it went unrecoverably wrong) and the follower is
# easing back to the pose captured at connect, so the next attempt starts from
# the same place the first one did.
#
# Corrections-only DAgger has no task-episode concept of its own — an "episode"
# there is one takeover — so without this the policy simply kept driving at a
# finished scene, and the only way to reset was to freeze the arm mid-pose and
# work around it.
PHASE_RESETTING = "resetting"

# Also ours, and the takeover's new middle step. The leader has been driven onto
# the follower's pose and is HELD there under torque; both arms are stationary
# and nothing is being recorded. The session waits here until the operator asks
# for control a second time.
#
# It exists because the one-press takeover put the operator's hand on a moving
# arm. Worse, when the leader glide failed — which happens, and did on the very
# first takeover of a real session — the takeover proceeded anyway, and the
# offset that was supposed to absorb the gap instead walked the FOLLOWER 114
# degrees across the workspace to meet a leader that had never moved. Nothing
# was wrong from the software's point of view; the arm simply went somewhere
# nobody asked it to.
#
# Poising makes that failure visible instead of automatic: the operator sees
# two arms that do not line up, moves the leader by hand, and only then presses
# space. The second press is also what starts recording, so no frames are
# captured while a grip is being settled.
PHASE_POISED = "poised"

PHASES = frozenset(
    {
        PHASE_AUTONOMOUS,
        PHASE_PAUSED,
        PHASE_CORRECTING,
        PHASE_HANDING_OVER,
        PHASE_SAVING,
        PHASE_RESETTING,
        PHASE_POISED,
    }
)


def format_event(event: str, payload: str = "") -> str:
    """Render one protocol line.

    The payload's whitespace is collapsed so a multi-line exception message can
    never split one event across several lines — the reader is line-oriented and
    a wrapped traceback would otherwise be read as several unknown events."""
    body = " ".join(str(payload).split())
    return f"{EVENT_PREFIX} {event} {body}".rstrip()


def parse_event(line: str) -> tuple[str, str] | None:
    """`(event, payload)` for a protocol line, or None when the line isn't one.

    Matches the prefix anywhere in the line rather than only at the start: the
    runner's logging handler writes to the same merged pipe, and a log record
    flushed without its trailing newline would otherwise swallow the event that
    follows it on the wire. A line that merely mentions the prefix can't be
    produced by the runner (it only ever writes it via `format_event`)."""
    idx = line.find(EVENT_PREFIX)
    if idx < 0:
        return None
    rest = line[idx + len(EVENT_PREFIX) :].strip()
    if not rest:
        return None
    event, _, payload = rest.partition(" ")
    return event, payload.strip()


def parse_fields(payload: str) -> dict[str, str]:
    """Split a `key=value key=value` payload into a dict.

    Tokens without an `=` are dropped rather than guessed at. Values cannot
    contain spaces — every field this protocol carries (a repo id, a path, an
    integer, a `true`/`false`) is space-free by construction, and
    `format_event` collapses whitespace anyway, so a value that did contain one
    would already have been mangled upstream of here."""
    fields: dict[str, str] = {}
    for token in payload.split():
        key, sep, value = token.partition("=")
        if sep and key:
            fields[key] = value
    return fields
