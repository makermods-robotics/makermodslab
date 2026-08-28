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

"""Persistent multi-episode rollout runner for MakerLab's evaluation mode.

    python -m makermodslab.eval_runner <exactly the lerobot-rollout argv>

Why it exists
-------------
Evaluation runs N episodes of the SAME policy on the SAME robot. Spawning
`lerobot-rollout` once per episode re-pays the two expensive setup steps every
single time — loading the policy onto the accelerator (15-40 s for SmolVLA on
MPS) and connecting the serial bus + every camera (several more seconds, plus a
camera-handover window where a preview can steal a device). Over a 20-episode
eval that is 5-10 minutes of dead time and 20 chances for a camera to come back
on a different index.

This runner pays both costs ONCE. It builds lerobot's rollout context, then
parks on stdin serving a tiny line protocol (see `makermodslab.eval_protocol`):
`EPISODE` runs one rollout, `STOP` ends the running one early, `QUIT` tears
down. Everything that actually drives the arm is lerobot's own code —
`build_rollout_context` for the policy/robot/processor wiring, `BaseStrategy`
for the control loop, and `RolloutStrategy._return_to_initial_position` for the
ease-home — so an upstream change to how a rollout behaves lands here too.

Deliberate behaviour differences from N separate `lerobot-rollout` processes
-----------------------------------------------------------------------------
1. **The follower stays energized between episodes.** A per-episode subprocess
   disconnects at exit, which drops torque and leaves the arm limp during the
   reset window. Here the robot is only disconnected at QUIT, so between
   episodes the arm HOLDS the initial pose it was just eased back to. Nothing
   moves at the transition (it is already at that pose), and the arm can't flop
   while the user rearranges the scene — but it also can't be repositioned by
   hand without stopping the session.
2. **`initial_position` is captured once**, at the session's connect, and every
   episode ends by easing back to it. Per-episode subprocesses each captured
   wherever the arm happened to be sitting, so "home" could drift across a long
   eval. Here every episode starts from the same pose.

Neither the policy nor the control loop is reimplemented; the episode loop only
decides WHEN lerobot's loop runs and when the arm goes home.
"""

# NO `from __future__ import annotations` here, deliberately: lerobot's
# parser.wrap() reads run()'s RAW annotation (getfullargspec, not
# get_type_hints) to find the draccus config class. PEP 563 would turn it
# into the string 'RolloutConfig' and draccus dies with "must be called
# with a dataclass type or instance". Same reason no lerobot script uses it.
import logging
import queue
import sys
import threading
from collections.abc import Callable

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers --robot.cameras type)
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401  (registers --robot.type=so101_follower / bi_so_follower)
    bi_so_follower,
    so_follower,
)
from lerobot.rollout import (
    RolloutConfig,
    RolloutContext,
    RolloutStrategy,
    build_rollout_context,
    create_strategy,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

# Imported for its SIDE EFFECT: a non-zero sync_read retry default for this
# process. Without it a single dropped serial reply — routine when arm
# adapters share a USB bus with streaming cameras — kills the session with
# "[TxRxResult] There is no status packet!". See makermodslab/bus_retry.py.
from . import bus_retry  # noqa: F401
from .eval_protocol import (
    CMD_EPISODE,
    CMD_QUIT,
    CMD_STOP,
    EVENT_BYE,
    EVENT_EPISODE_ENDED,
    EVENT_EPISODE_STARTED,
    EVENT_ERROR,
    EVENT_READY,
    REASON_DURATION,
    REASON_STOPPED,
    format_event,
)

logger = logging.getLogger(__name__)

# Logged (not emitted as an event) once the one-time setup is behind us, purely
# so the familiar lerobot landmark still appears in the inference log a human —
# or a future grep — goes looking for. The orchestrator does NOT key its
# `running` phase off this line in runner mode: setup happens once per SESSION
# but the phase has to flip once per EPISODE, which is what EPISODE_STARTED is
# for. See `_ROLLOUT_START_MARKER` in makermodslab/rollout.py.
_SETUP_COMPLETE_LOG = "Rollout setup complete"


def _emit(event: str, payload: str = "") -> None:
    """Write one protocol event to stdout, flushed.

    stdout is a pipe here, so it is block-buffered by default — without the
    flush the orchestrator would not see an episode boundary until 4-8 KB of
    unrelated log had accumulated behind it, which is most of an episode."""
    print(format_event(event, payload), flush=True)


def _read_commands(
    commands: queue.Queue,
    stop_event: threading.Event,
    quit_event: threading.Event,
) -> None:
    """Read the command protocol off stdin until EOF (runs on its own thread).

    STOP and QUIT are acted on HERE rather than being queued, because their
    whole point is to interrupt an episode that is currently running — the main
    thread is inside lerobot's control loop and won't drain the queue until that
    loop returns. Both set `stop_event`, which is the loop's shutdown event, so
    the episode breaks out on its next tick.

    Started only AFTER the robot is connected: `SOFollower.calibrate()` prompts
    with `input()` during `connect()`, reading the very same stdin, and the
    orchestrator pre-seeds a newline per arm to answer it. Racing that prompt
    with this reader would let a real command be eaten by the prompt (or the
    seed newline be read as a command). Any seed newline the prompt didn't
    consume arrives here as a blank line and is ignored."""
    try:
        for raw in sys.stdin:
            cmd = raw.strip().upper()
            if not cmd:
                continue
            if cmd == CMD_STOP:
                logger.info("STOP received — ending the current episode early")
                stop_event.set()
                continue
            if cmd == CMD_QUIT:
                logger.info("QUIT received — tearing down")
                quit_event.set()
                # Break out of a running episode first; the main loop then sees
                # quit_event and skips reporting the (unscored) partial episode.
                stop_event.set()
            commands.put(cmd)
    except Exception:
        logger.exception("Eval runner stdin reader failed")
    finally:
        # EOF (the orchestrator died or closed the pipe) must not leave the main
        # thread blocked on an empty queue forever.
        commands.put(None)


def _run_episode(
    strategy: RolloutStrategy,
    ctx: RolloutContext,
    stop_event: threading.Event,
) -> str:
    """Run ONE episode and ease the arm home. Returns the end reason.

    `strategy.setup` is lerobot's own per-run arming step (fresh action
    interpolator, `engine.reset()` to clear the policy's hidden state and
    processor queues, `engine.start()`, cleared observation cache) — calling it
    per episode is what makes episode k+1 indistinguishable from a fresh
    process, minus the load and the connect. Its counterpart `engine.stop()` is
    called at the end so an async backend's worker thread is not leaked per
    episode; both are no-op logs for the default sync engine.

    The ease-home runs in a `finally` so a crashed episode still puts the arm
    back before the failure is reported."""
    stop_event.clear()
    strategy.setup(ctx)
    _emit(EVENT_EPISODE_STARTED)
    try:
        strategy.run(ctx)
    finally:
        _return_to_initial(ctx)
        ctx.policy.inference.stop()
    return REASON_STOPPED if stop_event.is_set() else REASON_DURATION


def _return_to_initial(ctx: RolloutContext) -> None:
    """Ease the follower back to the pose captured at connect, between episodes.

    Delegates to lerobot's own helper — the exact routine
    `--return_to_initial_position=true` runs at process teardown — rather than
    re-deriving the interpolation, so the promise the stop dialog makes ("eases
    the follower back to its start pose") stays literally the same code. It is
    a `staticmethod` on `RolloutStrategy` with a leading underscore, i.e. not a
    stability contract: if an upstream bump moves it, this call site is the one
    to fix (the arm would stop easing home between episodes, loudly).

    Unlike teardown this does NOT disconnect, so the arm holds the pose under
    torque through the reset window — see the module docstring."""
    if not ctx.runtime.cfg.return_to_initial_position:
        logger.info("Skipping return-to-initial-position between episodes (disabled by config)")
        return
    if not ctx.hardware.initial_position:
        logger.warning("No initial position was captured; leaving the arm where the episode ended")
        return
    logger.info("Returning robot to initial position before the reset...")
    RolloutStrategy._return_to_initial_position(ctx.hardware)


# How long the idle loop blocks on the command queue before re-checking whether
# it should quit. Bounded rather than infinite because a signal that lands while
# we are parked between episodes only flips a flag — the blocking `get()` itself
# resumes waiting once the handler returns, so without a timeout a SIGTERM
# arriving during a reset would go unanswered until the orchestrator's SIGKILL.
_COMMAND_POLL_S = 0.5


def _serve(
    strategy: RolloutStrategy,
    ctx: RolloutContext,
    stop_event: threading.Event,
    should_quit: Callable[[], bool],
    commands: queue.Queue,
) -> None:
    """Main command loop: one episode per EPISODE, until QUIT, a signal, or EOF."""
    while not should_quit():
        try:
            cmd = commands.get(timeout=_COMMAND_POLL_S)
        except queue.Empty:
            continue
        if cmd is None or cmd == CMD_QUIT:
            return
        if cmd != CMD_EPISODE:
            logger.warning("Ignoring unrecognised eval-runner command %r", cmd)
            continue
        reason = _run_episode(strategy, ctx, stop_event)
        if should_quit():
            # Aborted mid-episode. The orchestrator does not score the in-flight
            # episode of an aborted session, so reporting an end here would race
            # a verdict in that the user never asked for.
            logger.info("Episode cut short by QUIT — not reporting an episode end")
            return
        logger.info("Episode ended (reason=%s)", reason)
        _emit(EVENT_EPISODE_ENDED, f"reason={reason}")


@parser.wrap()
def run(cfg: RolloutConfig) -> None:
    """Load once, connect once, then serve episodes off stdin.

    Takes the identical argv `lerobot-rollout` takes (the orchestrator builds
    one arg list and points it at either entry point), so `--strategy.type`,
    `--policy.*`, `--robot.*`, `--task` and `--duration` all mean exactly what
    they mean there. `--duration` is the per-EPISODE limit, not a session
    budget: the runner has no idea how many episodes it will be asked for."""
    init_logging()

    # Same handler lerobot's own CLI installs: a SIGTERM (the orchestrator's
    # fallback when a QUIT goes unanswered) sets an event rather than killing us
    # mid-motion, so the arm still eases home. Its `counter` is what lets us
    # treat a signal as QUIT rather than as a per-episode STOP — otherwise the
    # signalled episode would end and the runner would park for the next one.
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    stop_event = signal_handler.shutdown_event
    quit_event = threading.Event()

    def _should_quit() -> bool:
        return quit_event.is_set() or signal_handler.counter > 0

    logger.info("Building rollout context (policy + robot are loaded ONCE for the whole session)...")
    # Hands `stop_event` to lerobot as the control loop's shutdown event; we
    # clear it at the start of every episode, which is what turns a one-shot
    # shutdown flag into a per-episode "end this one now".
    ctx = build_rollout_context(cfg, stop_event)
    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Episode duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )
    logger.info("%s — eval runner waiting for episode commands", _SETUP_COMPLETE_LOG)

    # Only now is it safe to read stdin: the calibration prompt inside
    # `robot.connect()` has had its turn (see `_read_commands`).
    commands: queue.Queue = queue.Queue()
    threading.Thread(
        target=_read_commands,
        args=(commands, stop_event, quit_event),
        name="eval-runner-stdin",
        daemon=True,
    ).start()

    failure: BaseException | None = None
    try:
        _emit(EVENT_READY)
        _serve(strategy, ctx, stop_event, _should_quit, commands)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except BaseException as exc:  # noqa: BLE001 — reported, then re-raised as a non-zero exit
        failure = exc
        logger.exception("Eval runner failed")
        # Give the orchestrator the real exception text on the wire. It also
        # mines the log tail, but that is a heuristic over a traceback; this is
        # the message itself, delivered before the process is gone.
        _emit(EVENT_ERROR, f"{type(exc).__name__}: {exc}")
    finally:
        # Full lerobot teardown: ease home (already there, so a no-op move),
        # disconnect the bus and every camera. Guarded because the failure that
        # brought us here is often exactly what teardown will trip over again,
        # and masking it with a second traceback helps nobody.
        try:
            strategy.teardown(ctx)
        except Exception:
            logger.exception("Eval runner teardown failed")
        _emit(EVENT_BYE)

    if failure is not None:
        # Exit non-zero so the orchestrator classifies the in-flight episode an
        # `error` (excluded from the accuracy denominator) rather than a verdict.
        sys.exit(1)


def main() -> None:
    """Entry point for `python -m makermodslab.eval_runner`."""
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
