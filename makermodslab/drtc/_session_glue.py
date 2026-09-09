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

"""Supervised-session glue shared by BOTH robot entrypoints.

`robot_sync.py` grew all of this in S3.1 (the stdin STOP protocol, the
first-action ease-in, the start-pose capture and return, the interrupt-shielded
teardown, the transport-pinning flags). S3.5 brings `robot_rtc.py` under the
same session, and the only honest way to do that is to lift the pieces here
rather than to copy them: two divergent copies of a teardown whose entire job
is to make an energized arm safe is exactly the bug this module prevents.

Importable WITHOUT the `[remote]` extra, deliberately — same rule as `._pose`,
and for the same reason. Nothing here imports `livekit.portal` (the FFI dylib),
`livekit.api` or `python-dotenv`; the two `portal`-typed helpers take the portal
object as an argument and only call methods on it. So the pure parts stay
unit-testable in ordinary CI, and only the two entrypoints' own module tops
carry the extra.

What deliberately did NOT move here
-----------------------------------
* **The teardown's CALL SEQUENCE.** `shielded` and `_shielded_disconnect` are
  here; the `finally:` that composes them stays written out in each entrypoint.
  `tests/test_drtc_robot_sync.py::test_every_teardown_step_goes_through_the_shield`
  reads that block off the SOURCE and asserts every step is routed through the
  shield — the one guarantee no runtime test can reach, since the block only
  runs with a real arm attached. Hiding the sequence behind one helper call
  would silently retire that guard. `tests/test_drtc_robot_rtc.py` now applies
  the identical assertion to the other entrypoint, so the visible-call-sites
  rule is pinned on both sides rather than traded away.
* **`reset_torque_limit(robot, FOLLOWER)`.** Same reason: pinned by a source
  assertion in each entrypoint's test (and by `tests/test_motor_power_call_sites`'s
  rule that call sites name their side rather than pass a bare string). Four
  lines, stated where they happen.
* **Credential resolution** (`load_env` / `mint_token`). Those live in
  `._common`, which imports `livekit.api` — importing it here would drag the
  extra into this module and cost the CI-importability above. The flags
  themselves DO live here (see the field factories), so the two entrypoints
  cannot drift on defaults or help text; only the three-line resolution is
  restated.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass, field

from ..drtc_protocol import (
    EVENT_ACTIVE,
    EVENT_EASING,
    EVENT_STATS,
    format_event,
    format_stats,
    pump_commands,
)
from ._pose import capture_start_poses, ease_to_action, ensure_uncapped, return_to_start_poses

# --- the stdout half of the protocol ----------------------------------------


def emit(event: str, payload: str = "") -> None:
    """Write one protocol event to stdout, flushed.

    stdout is a pipe under a supervising parent, so it is block-buffered by
    default — without the flush the parent would not see CONNECTED (or a
    per-second STATS) until 4-8 KB of unrelated lerobot log had accumulated
    behind it, which is most of a short run.

    Deliberately RAISES on a dead pipe, and that is load-bearing: a parent that
    died without saying STOP (SIGKILL, an OOM, a crash) leaves this child
    holding an energized arm with no other signal — stdin EOF is ignored by
    design (see `drtc_protocol.pump_commands`). The read end closing makes the
    next write raise BrokenPipeError, which unwinds the control loop into the
    teardown within the second it takes the next STATS to be emitted, and the
    teardown returns the arm and releases torque. Swallowing it here would
    retire the only detector the child has. The teardown's OWN narration goes
    through :func:`say` instead, so the same broken pipe cannot then unwind the
    teardown itself."""
    print(format_event(event, payload), flush=True)


def say(message: str) -> None:
    """`print` for the TEARDOWN, where a dead stdout must not cost the release.

    Same broken pipe as above, one block later: a bare `print` in the `finally:`
    raises BrokenPipeError, propagates out of the block, and skips
    `robot.disconnect()` — the call that RELEASES TORQUE. The arm is then left
    energized holding the policy's last command, on the one path whose entire
    job is to make it safe, and precisely in the case (the parent is gone) where
    nobody is left to notice. So the teardown narrates through here, and
    :func:`shielded`'s own diagnostics with it.

    `ValueError` as well as `OSError`: a closed (rather than broken) stream
    raises "I/O operation on closed file"."""
    with contextlib.suppress(OSError, ValueError):
        print(message, flush=True)


def emit_stats(values: dict[str, object]) -> None:
    """Emit one STATS line from a full :data:`drtc_protocol.STATS_KEYS` dict.

    `format_stats` fills a missing key with null and RAISES on an unknown one,
    so the always-present contract the parent's response model depends on is
    enforced here rather than trusted. The key set is FROZEN: an engine that
    has no analogue for a key reports null (or its own equivalent, documented
    at the call site) and puts its engine-specific numbers on the human
    `[robot]` line instead."""
    emit(EVENT_STATS, format_stats(values))


def note_first_operator(portal, seen: bool) -> bool:
    """Emit ACTIVE the first time an operator joins; return the new `seen`.

    Polled every tick until it fires, not once a second: this is the transition
    a supervising parent's "did the GPU ever join?" watchdog waits on, so a
    1 Hz sample would cost it a whole second of its budget."""
    if seen:
        return True
    operator = portal.active_operator()
    if operator is None:
        return False
    emit(EVENT_ACTIVE, f"operator={operator}")
    return True


def or_none(value):
    """Portal's metrics report 0 for "no sample yet"; STATS reports that as null.

    A genuine 0 µs e2e or RTT is not physically reachable, so collapsing the
    two is safe and keeps the "null means unknown" contract honest for the
    parent's typed status model."""
    return value or None


# --- the stdin half: STOP / STOP / QUIT --------------------------------------


@dataclass
class LoopControl:
    """The three events a supervising parent drives the child with.

    One object rather than three loose locals so the entrypoints cannot get the
    two-STOP semantics subtly different: the FIRST STOP asks for a graceful
    return, a LATER one sets `abort_event` so the in-flight return is cut short,
    and QUIT sets all three. That rule lives in
    `drtc_protocol.apply_command`; this is only the carrier."""

    stop_event: threading.Event = field(default_factory=threading.Event)
    abort_event: threading.Event = field(default_factory=threading.Event)
    quit_event: threading.Event = field(default_factory=threading.Event)

    def start_command_pump(self, stream=None) -> threading.Thread:
        """Read commands off `stream` (default stdin) on a daemon thread.

        Call this only AFTER `robot.connect()`. `SOFollower.calibrate()` prompts
        with `input()` during connect on an uncalibrated arm, reading the very
        same stdin, and racing that prompt with this reader would let a real
        command be eaten by the prompt. Daemon so an EOF-blocked read never
        holds up exit."""
        thread = threading.Thread(
            target=pump_commands,
            args=(
                sys.stdin if stream is None else stream,
                self.stop_event,
                self.abort_event,
                self.quit_event,
            ),
            name="drtc-robot-stdin",
            daemon=True,
        )
        thread.start()
        return thread


# --- start pose, ease-in, return ---------------------------------------------


def capture_start_poses_or_warn(robot, enabled: bool) -> list:
    """The pose to return to, captured after connect and before anything moves.

    Empty for a non-Feetech (Koch/OMX) arm, which makes the return a logged
    no-op rather than a wrong-units move; see `_pose.py`. The gripper is
    excluded — the policy may have left it holding something at stop time."""
    if not enabled:
        return []
    poses = capture_start_poses(robot)
    if not poses:
        print(
            "[robot] WARNING: no Feetech bus to capture a start pose from; torque will be released in place"
        )
    return poses


def ease_into_first_action(robot, action: dict[str, float], control: LoopControl) -> tuple[float, bool]:
    """Ramp into the policy's FIRST commanded pose. Returns `(seconds, stopped)`.

    Until this runs nothing has been sent, so the arm is still exactly where the
    operator left it and this is the one send that can be an arbitrarily large
    jump. `_pose.ease_to_action` ramps into it at the gentle profile speed
    instead.

    The caller's loop is deliberately BLOCKED for the duration. `seconds` is
    returned so the caller can re-base its pacing clock by that much rather than
    letting the loop burst to catch up; `stopped` is True when a STOP landed
    during the ease (the ramp came back `cut-short`) and the caller must skip
    execution entirely.

    `ensure_uncapped` runs UNCONDITIONALLY, not just on arrival: the ease stamps
    a gentle `RETURN_POS_SPEED` profile cap into RAM `Goal_Velocity` on every
    exit path, and a survivor would throttle the whole run. An ease that settled
    short still proceeds — the arm is closer to the plan than it was, and
    refusing to run would leave it energized mid-air for no gain."""
    started = time.monotonic()
    emit(EVENT_EASING)
    _arrived, reason = ease_to_action(robot, action, abort_event=control.stop_event)
    print(f"[robot] first-action ease-in: {reason}")
    ensure_uncapped(robot)
    if control.stop_event.is_set():
        print("[robot] stop requested during the ease-in")
    return time.monotonic() - started, control.stop_event.is_set()


def return_step(start_poses: list, abort_event: threading.Event):
    """The teardown's return-to-rest step, as a callable for :func:`shielded`.

    A closure rather than a direct call so a Ctrl-C during the RETURN is the
    SECOND stop press: it sets `abort_event` and re-raises, `shielded` retries
    once, and the retry sees the flag and unwinds at once — torque then releases
    where the arm is, nearer rest than it started."""

    def _return_or_cut_short() -> None:
        if abort_event.is_set():
            return
        try:
            return_to_start_poses(start_poses, abort_event=abort_event)
        except KeyboardInterrupt:
            abort_event.set()
            raise

    return _return_or_cut_short


# --- the interrupt shield -----------------------------------------------------


def shielded(what: str, fn, *args, attempts: int = 2, reraise: bool = False, **kwargs):
    """Run one TEARDOWN step so that no interrupt can skip the steps after it.

    `contextlib.suppress(Exception)` does NOT cover `KeyboardInterrupt` — it
    derives from BaseException — so a Ctrl-C landing inside the return-to-rest,
    or inside the transport teardown, propagates straight out of the `finally`
    block and skips `robot.disconnect()`: the call that RELEASES TORQUE. The
    arm is then left energized, holding the policy's last command, with no BYE
    for a supervising parent to read. That is the worst outcome available on
    the one path whose entire job is to make the arm safe.

    So every teardown step is individually shielded, and one that was
    interrupted is retried once — by then the abort event is set (see the call
    sites), so the retry unwinds immediately instead of resuming a long move.
    Returns the step's value, or None when it failed or was given up on.

    `reraise=True` shields the step from INTERRUPTS only and lets a genuine
    exception propagate, which is what the torque release itself wants: an arm
    that could not be disconnected is a failed run and the process should exit
    saying so, exactly as it did before this shield existed.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            remaining = attempts - attempt
            say(f"[robot] Ctrl-C during {what}; {'retrying' if remaining else 'giving up on it'}")
        except Exception as exc:
            say(f"[robot] {what} failed: {exc}")
            if reraise:
                raise
            return None
    return None


async def _shielded_disconnect(portal) -> None:
    """`await portal.disconnect()`, swallowing errors AND interrupts.

    The async twin of :func:`shielded` — one step, not retried: a transport
    that would not close is not going to close on a second ask, and the arm's
    torque release is waiting behind it."""
    try:
        await portal.disconnect()
    except KeyboardInterrupt:
        say("[robot] Ctrl-C during the transport disconnect; continuing to the torque release")
    except Exception as exc:
        say(f"[robot] transport disconnect failed: {exc}")


# --- the shared draccus flags -------------------------------------------------
#
# Field FACTORIES rather than a shared base dataclass: both configs open with
# `robot: RobotConfig` (no default), and inheriting defaulted fields from a base
# would put them BEFORE it and make the dataclass unconstructible. Factories
# keep the defaults and the help text identical across the two entrypoints,
# which is what the parent's arg builder and the docs both assume.
#
# draccus has NO `--no-<flag>` form (it is not argparse's BooleanOptionalAction):
# turn one of these off with `--<flag> false` / `--<flag>=False`.


def livekit_url_field():
    return field(
        default="",
        metadata={
            "help": "SFU URL to dial. Unset falls back to LIVEKIT_URL from the "
            "credential files (see _env.load_env's precedence). Pin it when "
            "a parent has already verified the transport, so the whole "
            "'parent probed room X, the child's .env.local said room Y' "
            "class of failure cannot happen; the effective value is echoed "
            "back in the READY event."
        },
    )


def livekit_room_field():
    return field(
        default="",
        metadata={
            "help": "Room to join. Unset falls back to LIVEKIT_ROOM. Same "
            "rationale as --livekit_url; note the GPU side's room comes ONLY "
            "from its Modal secret unless it is launched with --livekit-room."
        },
    )


def livekit_token_field():
    return field(
        default="",
        metadata={
            "help": "Pre-minted room token to join with. The Lab always passes "
            "one (it signs it from its own SFU's 0600 key file, and the child "
            "— which needs no credential beyond a token scoped to one room "
            "and one identity — is never given the secret). Unset mints one "
            "from LIVEKIT_API_KEY/SECRET in the environment: the hand-run "
            "bench fallback."
        },
    )


def return_to_rest_field():
    return field(
        default=True,
        metadata={
            "help": "Capture the pose the arm starts in and drive it back there "
            "before releasing torque, on every exit path (STOP, Ctrl-C, "
            "duration elapsed, or a crash). ON by default: the safe "
            "behaviour is the right default everywhere, and cutting torque "
            "wherever the policy left the arm drops it. `--return_to_rest false` "
            "is a bench A/B flag. The GRIPPER is excluded from the captured "
            "pose (the policy may have left it holding something), matching "
            "teleoperation and recording rather than replay."
        },
    )


def ease_in_field():
    return field(
        default=True,
        metadata={
            "help": "Ramp the arm from wherever it is to the first chunk's step-0 "
            "pose before executing, instead of stepping straight to it at "
            "full speed. ON by default for the same reason as "
            "--return_to_rest; `--ease_in false` is the A/B baseline that "
            "reproduces the pre-2026-09-02 snap."
        },
    )
