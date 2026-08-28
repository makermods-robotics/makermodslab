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
    TAKEOVER  hand control to the human and start recording a correction
    HANDBACK  end the correction, SAVE it, and return control to the policy
    CANCEL    end the correction and DISCARD it (the fumbled-takeover escape)
    HOLD      freeze the policy without taking over
    RESUME    unfreeze, returning control to the policy
    RECOVERED mid-correction: "the arm is back somewhere sane, the correction
              starts here" — records a boundary, changes no phase (see below)
    RESET     end this attempt: ease the follower home and park for a scene reset
    QUIT      finalize the dataset, ease the arm home, disconnect, exit

TAKEOVER and HANDBACK are COMPOSITE: each drives two of lerobot's phase
transitions back to back (AUTONOMOUS→PAUSED→CORRECTING and the reverse). That
composition lives here, on the runner side, rather than in the UI, because the
intermediate PAUSED step is where lerobot performs the physical handover — it
drives an actuated leader arm to the follower's pose over ~2s, synchronously, on
the control-loop thread. A UI that sent two separate commands could interleave
something between them; a UI that sent them "atomically" would still be
guessing when the first had finished. One command, one hardware sequence.

Events — runner → stdout, one line each, carrying a grep-able prefix so the
orchestrator's log pump can pick them out of lerobot's own INFO chatter (the
runner's stderr is merged into the same pipe):
    MAKERMODSLAB-DAGGER READY
    MAKERMODSLAB-DAGGER DATASET repo_id=<final id> root=<path>
    MAKERMODSLAB-DAGGER PHASE phase=autonomous|paused|correcting
    MAKERMODSLAB-DAGGER CORRECTION_SAVED n=<count> frames=<n> seconds=<s>
                                        recovery=<n|-1> labelled=<true|false>
    MAKERMODSLAB-DAGGER CORRECTION_CANCELLED reason=<operator|too_short> …
    MAKERMODSLAB-DAGGER RECOVERY_MARK frames=<n>
    MAKERMODSLAB-DAGGER ALIGN_REQUIRED max_delta=<deg> joints=<name:delta,…>
    MAKERMODSLAB-DAGGER ERROR <message, whitespace collapsed to one line>
    MAKERMODSLAB-DAGGER BYE

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
CORRECTION_SAVED, and the orchestrator writes per-episode counts to a sidecar
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

`labelled=false` on CORRECTION_SAVED means the operator never pressed it, which
is NOT the same as "recovery took zero frames": one says unannotated, the other
says the operator went straight to correcting. A consumer that conflates them
would train on a lie.

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
CMD_CANCEL = "CANCEL"
CMD_HOLD = "HOLD"
CMD_RESUME = "RESUME"
CMD_RESET = "RESET"
# "I have the arm back somewhere sane; the correction starts here." Valid only
# mid-correction, records a boundary, changes no phase. See the module docstring.
CMD_RECOVERED = "RECOVERED"
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
EVENT_ALIGN_REQUIRED = "ALIGN_REQUIRED"

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
# One attempt at the task ended and the arm is back at its start pose. Carries
# the running attempt count so the UI can show "attempt 4" without keeping its
# own tally that a dropped event would desynchronise.
EVENT_ATTEMPT_RESET = "ATTEMPT_RESET"
# The operator marked the end of recovery inside the correction in progress.
# Carries the frame count at the boundary so the UI can show it live; the
# authoritative per-episode record arrives on CORRECTION_SAVED.
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

# NOT one of lerobot's phases — ours, and the only one that is. It covers the
# window inside `_apply_transition` where the arm is PHYSICALLY TRAVELLING and
# no lerobot phase describes the truth:
#
#   * single-arm: the leader is driven under torque to the follower's pose
#     (`teleop_smooth_move_to`, ~2s) while lerobot still reports AUTONOMOUS;
#   * bimanual: BOTH followers slide to meet the leaders
#     (`follower_smooth_move_to`, ~2s) while lerobot already reports PAUSED.
#
# The second case is why this exists. PAUSED renders as "the arm is frozen",
# and showing that sentence to an operator while two arms sweep across the
# workspace is the worst thing this UI could say. The runner emits this before
# the blocking call and the real phase after it.
PHASE_HANDING_OVER = "handing_over"

# Also ours. `save_episode()` runs synchronously on the control loop at the
# hand-back edge — it writes the parquet and encodes the episode's video — so
# the arm sits frozen for its duration and the policy has not resumed yet.
# Without a state of its own that window inherited the phase either side of it,
# which meant the banner kept insisting the operator was still driving and
# recording long after they had let go.
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

PHASES = frozenset(
    {
        PHASE_AUTONOMOUS,
        PHASE_PAUSED,
        PHASE_CORRECTING,
        PHASE_HANDING_OVER,
        PHASE_SAVING,
        PHASE_RESETTING,
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
    integer, a comma-joined joint list) is space-free by construction, and
    `format_event` collapses whitespace anyway, so a value that did contain one
    would already have been mangled upstream of here."""
    fields: dict[str, str] = {}
    for token in payload.split():
        key, sep, value = token.partition("=")
        if sep and key:
            fields[key] = value
    return fields
