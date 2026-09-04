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
"""`robot_rtc`'s half of the S3.5 session glue, read off the SOURCE.

The sibling of `tests/test_drtc_robot_sync.py`'s source-level half, and it
exists for the same reason: the teardown and the torque-cap reset sit inside an
async control loop that needs a real arm, so no ordinary test can reach them —
and they are exactly the code a refactor drops.

It is also why the teardown's CALL SEQUENCE deliberately did NOT move into
`_session_glue` when S3.5 lifted the rest of the glue there. `shielded` and
`_shielded_disconnect` are shared; the `finally:` that composes them stays
written out in each entrypoint, so this assertion has something to read. Hiding
the sequence behind one helper call would have retired the only guard the
torque-release path has, on BOTH engines at once.

Nothing here imports the module (that would need the `drtc` extra and its FFI
dylib): every assertion is over the parsed source, so it runs in ordinary CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

DRTC = Path(__file__).resolve().parents[1] / "makermodslab" / "drtc"
ROBOT_RTC = DRTC / "robot_rtc.py"
ROBOT_SYNC = DRTC / "robot_sync.py"


def _finally_body(path: Path) -> ast.Try:
    """The `try/finally` in `run()` that owns the teardown."""
    tree = ast.parse(path.read_text(), filename=str(path))
    run = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    tries = [node for node in ast.walk(run) if isinstance(node, ast.Try) and node.finalbody]
    assert tries, "run() no longer has a try/finally — the teardown must still be one"
    return max(tries, key=lambda node: len(node.finalbody))


def _direct_calls(nodes: list[ast.stmt]) -> list[ast.Call]:
    """Every call made BY these statements, not descending into nested defs.

    The distinction is the whole point: `return_to_start_poses(...)` inside the
    closure `return_step` hands to `shielded` is protected; the same call
    written inline is not."""
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


# ---------------------------------------------------------------------------
# The teardown's shape
# ---------------------------------------------------------------------------


def test_every_teardown_step_goes_through_the_shield() -> None:
    """A bare call in that `finally` is a call an interrupt can unwind the block
    from, skipping the torque release behind it.

    Before S3.5 this entrypoint's teardown was four bare calls — `await
    portal.disconnect()`, `portal.close()`, `robot.disconnect()` — with no
    return-to-rest at all, so a Ctrl-C dropped the arm wherever the policy had
    left it."""
    names = {_callee(node) for node in _direct_calls(_finally_body(ROBOT_RTC).finalbody)}
    assert "shielded" in names
    assert "_shielded_disconnect" in names
    for direct in ("disconnect", "close", "return_to_start_poses"):
        assert direct not in names, (
            f"{direct}() is called directly in run()'s finally — an interrupt there would "
            "unwind the teardown and skip the torque release. Route it through `shielded`."
        )


def test_the_teardown_never_writes_to_stdout_unprotected() -> None:
    """The dead-stdout twin of the assertion above, on this entrypoint.

    Pinned on BOTH children for the same reason the shield itself is: the
    parent's death breaks this pipe, and a bare `print`/`emit` in the `finally:`
    would unwind the teardown and skip the torque release. See
    `tests/test_drtc_robot_sync.py` for the full rationale."""
    names = [_callee(node) for node in _direct_calls(_finally_body(ROBOT_RTC).finalbody)]
    assert "say" in names
    for direct in ("print", "emit"):
        assert direct not in names, (
            f"{direct}() is called directly in run()'s finally — a dead stdout (the parent "
            "died) would unwind the teardown and skip the torque release. Use `say`, or "
            "route it through `shielded`."
        )


def test_both_engines_tear_down_the_same_way() -> None:
    """The two entrypoints are one session's two children; a teardown that
    diverged would mean an arm that is safe on one engine and not the other.

    Compared as the SEQUENCE of `shielded`-step labels, which is what actually
    orders the return, the transport close and the torque release."""

    def labels(path: Path) -> list[str]:
        # Sorted by line: `_direct_calls` walks with a LIFO stack (it is written
        # for membership, not order), and the ORDER is the thing under test here.
        steps = [
            node
            for node in _direct_calls(_finally_body(path).finalbody)
            if _callee(node) == "shielded" and node.args and isinstance(node.args[0], ast.Constant)
        ]
        return [node.args[0].value for node in sorted(steps, key=lambda n: n.lineno)]

    assert labels(ROBOT_RTC) == labels(ROBOT_SYNC)
    assert labels(ROBOT_RTC) == [
        "the RETURNING event",
        "the return to the start pose",
        "closing the transport",
        "the torque release (robot.disconnect)",
        "the BYE event",
    ]


# ---------------------------------------------------------------------------
# The torque cap
# ---------------------------------------------------------------------------


def test_the_torque_cap_is_reset_right_after_connect() -> None:
    """`Torque_Limit` is a RAM register that SURVIVES between sessions on one
    power-up, so a cap an earlier auto-calibration left behind would silently
    throttle the whole run — the arm tracks sluggishly, everything looks
    healthy, and nothing says why.

    `tests/test_motor_power_call_sites.py` separately pins that a call site
    names its side (FOLLOWER) rather than passing a bare string."""
    source = ROBOT_RTC.read_text()
    assert "reset_torque_limit(robot, FOLLOWER)" in source
    assert source.index("robot.connect()") < source.index("reset_torque_limit(robot, FOLLOWER)"), (
        "the torque cap must be cleared AFTER connect — lerobot's configure() stamps the "
        "gripper's Max_Torque_Limit first, and that is the value the reset re-seeds from"
    )


# ---------------------------------------------------------------------------
# The glue is shared, not copied
# ---------------------------------------------------------------------------


def test_neither_entrypoint_redefines_the_shared_glue() -> None:
    """S3.5's whole point. Two divergent copies of the code that makes an
    energized arm safe is the bug this module exists to prevent, and the way it
    comes back is somebody re-adding a local `shielded` or `_emit` to one file
    "just for now"."""
    shared = {
        "shielded",
        "_shielded_disconnect",
        "emit",
        "emit_stats",
        "or_none",
        "note_first_operator",
        "ease_into_first_action",
        "return_step",
        "capture_start_poses_or_warn",
    }
    for path in (ROBOT_RTC, ROBOT_SYNC):
        tree = ast.parse(path.read_text(), filename=str(path))
        defined = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert not (defined & shared), (
            f"{path.name} redefines {sorted(defined & shared)} instead of importing it from "
            "_session_glue — the two engines would then drift on the arm's safety behaviour."
        )
