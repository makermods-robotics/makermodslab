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
"""`robot_sync`'s two safety fixtures: the interrupt-shielded teardown and the
torque-cap reset at connect.

The shield is a pure function, but it lives in a module that imports
`livekit.portal` (an FFI dylib behind the optional `drtc` extra) at top level —
so those tests importorskip the extra, exactly as `test_drtc_env.py` does for
python-dotenv. The call-site test below reads the SOURCE instead and therefore
runs everywhere, which matters: it is the half that cannot be reached by any
ordinary test (the call sits inside an async control loop that needs a real
arm), and it is the half a refactor is most likely to drop.
"""

from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import pytest

ROBOT_SYNC = Path(__file__).resolve().parents[1] / "makermodslab" / "drtc" / "robot_sync.py"


@pytest.fixture
def shielded():
    pytest.importorskip("livekit.portal")
    from makermodslab.drtc.robot_sync import shielded as fn

    return fn


# ---------------------------------------------------------------------------
# The interrupt shield
# ---------------------------------------------------------------------------


def test_a_step_that_succeeds_returns_its_value(shielded) -> None:
    assert shielded("a step", lambda: "done") == "done"


def test_an_interrupted_step_is_retried_once(shielded) -> None:
    """The retry is what makes a Ctrl-C during the return mean "cut it short"
    rather than "skip the torque release": by the time it runs, the call site
    has set the abort event, so the return unwinds at once."""
    attempts: list[int] = []

    def _interrupt_once() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt
        return "second"

    assert shielded("a step", _interrupt_once) == "second"
    assert len(attempts) == 2


def test_a_step_interrupted_every_time_is_given_up_on_not_propagated(shielded) -> None:
    """The whole point. `contextlib.suppress(Exception)` does not cover
    KeyboardInterrupt, so a THIRD Ctrl-C used to escape the teardown and skip
    everything behind it — including `robot.disconnect()`, the call that
    releases torque. The arm was then left energized, holding the policy's last
    command, with no BYE for a supervising parent to read."""

    def _always_interrupted() -> None:
        raise KeyboardInterrupt

    assert shielded("a step", _always_interrupted) is None


def test_a_failing_step_is_swallowed_by_default(shielded) -> None:
    def _boom() -> None:
        raise RuntimeError("transport is already closed")

    assert shielded("a step", _boom) is None


def test_a_failing_step_can_still_propagate(shielded) -> None:
    """`reraise=True` is what the torque release uses: an arm that could not be
    disconnected is a failed run and the process must exit saying so, exactly
    as it did before the shield existed. Only the interrupt is swallowed."""

    def _boom() -> None:
        raise RuntimeError("the bus is gone")

    with pytest.raises(RuntimeError):
        shielded("the torque release", _boom, reraise=True)


def test_reraise_still_shields_an_interrupt(shielded) -> None:
    attempts: list[int] = []

    def _interrupt_once() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt
        return "released"

    assert shielded("the torque release", _interrupt_once, reraise=True) == "released"


# ---------------------------------------------------------------------------
# A dead stdout (the parent died)
# ---------------------------------------------------------------------------
#
# `_session_glue` is importable WITHOUT the `[drtc]` extra (see its module
# docstring), so these run everywhere rather than importorskip-ing.


class _DeadStdout:
    """stdout after the supervising parent's death closed the read end."""

    def write(self, data: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


def test_emit_still_raises_on_a_dead_pipe() -> None:
    """The child's ONLY parent-death detector, and therefore not swallowed.

    stdin EOF is ignored by design (`drtc_protocol.pump_commands`), and the
    child is spawned into its own session so no signal reaches it either. The
    once-a-second STATS write raising is what unwinds the control loop into the
    teardown — where the arm is returned and torque released."""
    from makermodslab.drtc._session_glue import emit

    with contextlib.redirect_stdout(_DeadStdout()), pytest.raises(BrokenPipeError):
        emit("STATS", "{}")


def test_say_survives_a_dead_pipe_and_a_closed_stream() -> None:
    """The teardown's narration must never be what skips the torque release."""
    from makermodslab.drtc._session_glue import say

    with contextlib.redirect_stdout(_DeadStdout()):
        say("[robot] returning to the start pose ...")  # must not raise

    closed = io.StringIO()
    closed.close()
    with contextlib.redirect_stdout(closed):
        say("[robot] disconnecting...")  # must not raise


def test_the_shield_itself_survives_a_dead_pipe() -> None:
    """`shielded` reports a failed step by printing — on a dead pipe that print
    would raise INSIDE the handler and unwind the teardown anyway, which is why
    its diagnostics go through `say` too. The shielded RETURNING event is the
    step this actually protects."""
    from makermodslab.drtc._session_glue import emit, shielded

    with contextlib.redirect_stdout(_DeadStdout()):
        assert shielded("the RETURNING event", emit, "RETURNING", attempts=1) is None


# ---------------------------------------------------------------------------
# The teardown's shape, read off the source
# ---------------------------------------------------------------------------


def _finally_body() -> ast.Try:
    """The `try/finally` in `run()` that owns the teardown."""
    tree = ast.parse(ROBOT_SYNC.read_text(), filename=str(ROBOT_SYNC))
    run = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    tries = [node for node in ast.walk(run) if isinstance(node, ast.Try) and node.finalbody]
    assert tries, "run() no longer has a try/finally — the teardown must still be one"
    return max(tries, key=lambda node: len(node.finalbody))


def _direct_calls(nodes: list[ast.stmt]) -> list[ast.Call]:
    """Every call made BY these statements, not descending into nested defs.

    The distinction is the whole point: `return_to_start_poses(...)` inside a
    helper the teardown hands to `shielded` is protected; the same call written
    inline is not."""
    found: list[ast.Call] = []
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Call):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _callee(node: ast.Call) -> str | None:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def test_every_teardown_step_goes_through_the_shield() -> None:
    """A bare call in that `finally` is a call an interrupt can unwind the
    block from, skipping the torque release behind it. The shield is only worth
    anything if EVERY step is inside it."""
    names = {_callee(node) for node in _direct_calls(_finally_body().finalbody)}
    # The steps that touch the arm or the transport are reached only through
    # `shielded` / `_shielded_disconnect`, never called directly here.
    assert "shielded" in names
    assert "_shielded_disconnect" in names
    for direct in ("disconnect", "close", "return_to_start_poses"):
        assert direct not in names, (
            f"{direct}() is called directly in run()'s finally — an interrupt there would "
            "unwind the teardown and skip the torque release. Route it through `shielded`."
        )


def test_the_teardown_never_writes_to_stdout_unprotected() -> None:
    """S3.8d: the teardown's own NARRATION can unwind it too.

    When the parent is what died, this stdout pipe's read end is closed, so a
    bare `print` (or a bare `emit`) in that `finally:` raises BrokenPipeError,
    propagates out of the block, and skips `robot.disconnect()` — the torque
    release — leaving the arm energized with nobody left to reach it. `say`
    swallows a dead stream; `emit` deliberately does not (it is how the child
    NOTICES the parent is gone), so in the teardown it goes through `shielded`
    like every other step."""
    names = [_callee(node) for node in _direct_calls(_finally_body().finalbody)]
    assert "say" in names
    for direct in ("print", "emit"):
        assert direct not in names, (
            f"{direct}() is called directly in run()'s finally — a dead stdout (the parent "
            "died) would unwind the teardown and skip the torque release. Use `say`, or "
            "route it through `shielded`."
        )


def test_the_torque_cap_is_reset_right_after_connect() -> None:
    """`Torque_Limit` is a RAM register that SURVIVES between sessions on one
    power-up, so a cap an earlier auto-calibration left behind would silently
    throttle the whole run — the arm tracks sluggishly, everything looks
    healthy, and nothing says why. Every other Lab session resets it at start;
    this entrypoint was the one that did not.

    Read off the source because the call sits inside an async control loop that
    needs a real arm. `tests/test_motor_power_call_sites.py` separately pins
    that it names its side (FOLLOWER) rather than a bare string."""
    source = ROBOT_SYNC.read_text()
    assert "reset_torque_limit(robot, FOLLOWER)" in source
    assert source.index("robot.connect()") < source.index("reset_torque_limit(robot, FOLLOWER)"), (
        "the torque cap must be cleared AFTER connect — lerobot's configure() stamps the "
        "gripper's Max_Torque_Limit first, and that is the value the reset re-seeds from"
    )
