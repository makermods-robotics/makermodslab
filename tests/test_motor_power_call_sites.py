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
"""Every motor-register call site must SAY which arm it is holding.

A tripwire, not a behaviour test — the same genre as
`tests/test_dagger_upstream_pin.py`, and here for the same reason: the thing it
protects cannot be reached by an ordinary test.

`makermodslab.motor_power` writes RAM registers on a live servo bus. CLAUDE.md
deliberately excludes hardware paths from unit testing, and `tests/test_rollout.py`
monkeypatches the preflights away entirely, so the guard inside
`clear_goal_velocity` has NO runtime coverage in this repo and would rot unnoticed.
This walks the source instead: pure AST over files on disk, no bus, no subprocess.

WHAT THIS CAUGHT, AND WHY IT EXISTS
-----------------------------------
`_preflight_leader_registers` was written by copying its follower twin. The copy
kept `+ clear_goal_velocity(...)`, whose docstring says NEVER to call it on a
leader, and passed it the string "leader arm". Nothing objected: `label` was
free-form prose used only for log interpolation, it was optional, and it
defaulted to the safe kind — so the wrong caller read exactly like a right one.
Types could not object either (mypy is `ignore_errors = true` for
`makermodslab.*`), and ruff has no rule that could express it.

WHAT TO DO WHEN THIS FAILS
--------------------------
1. Find the flagged call and decide which arm it really holds.
2. If it is a follower, pass `FOLLOWER`. If a leader, pass `LEADER` — and note
   that `clear_goal_velocity` will refuse it at runtime, on purpose.
3. Do NOT pass `FOLLOWER` to silence this on a device that is a leader. Read
   `clear_goal_velocity`'s docstring first: the register bounds the stale-goal
   snap when torque is re-enabled mid-handover, with the arm in someone's hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "makermodslab"

# Helpers that take a required `side`, and which sides each one accepts.
#
# `reset_torque_limit` serves BOTH arms: coaching drives the leader under its
# own torque, so a leader left capped by an earlier auto-calibration cannot
# carry itself. `clear_goal_velocity` serves the follower only.
_GUARDED = {
    "reset_torque_limit": {"LEADER", "FOLLOWER"},
    "clear_goal_velocity": {"FOLLOWER"},
}

# Why each restriction exists, printed on failure so the next person does not
# have to reconstruct the argument from scratch.
_WHY = {
    "clear_goal_velocity": (
        "Goal_Velocity is the only bound on the leader's stale-goal snap: "
        "teleop_smooth_move_to calls enable_torque() BEFORE writing any goal, and "
        "Feetech's enable_torque does not seed Goal_Position, so the servo drives "
        "at the previous handover pose until the first waypoint lands — while the "
        "operator is holding the arm. Clearing it gains nothing (the 2s/30fps "
        "waypoint schedule is already the rate limiter)."
    ),
    "reset_torque_limit": (
        "Valid on both arms, but the side must still be stated: it is the "
        "declaration clear_goal_velocity refuses on, and an unstated side is how "
        "a follower-only register reached a leader in the first place."
    ),
}


def _call_sites() -> list[tuple[str, int, str, ast.Call]]:
    """(file, line, callee, node) for every guarded call in the package."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in path.parts:
            continue  # vendored upstream is not ours to re-shape
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _GUARDED:
                found.append((path.name, node.lineno, name, node))
    return found


def test_the_sweep_finds_the_call_sites_at_all() -> None:
    """Guards the guard. An AST walk that silently matches nothing passes every
    assertion below it — a green tripwire that protects nothing is worse than no
    tripwire, because it is believed."""
    sites = _call_sites()
    assert len(sites) >= 10, f"expected the known motor-register call sites, found {len(sites)}"
    assert {name for _, _, name, _ in sites} == set(_GUARDED)


@pytest.mark.parametrize("callee", sorted(_GUARDED))
def test_every_call_states_its_side_as_a_named_constant(callee: str) -> None:
    """The side must be `LEADER`/`FOLLOWER`, not a bare string.

    A literal like "follower" would satisfy the runtime check just as well, but
    a NAME is what makes the constant's definition — and the argument written
    above it — one hop from the call site. The original defect was a hand-typed
    string that nobody had reason to look up."""
    offenders = []
    for file, line, name, node in _call_sites():
        if name != callee:
            continue
        side = node.args[1] if len(node.args) >= 2 else None
        if side is None:
            kw = {k.arg: k.value for k in node.keywords}
            side = kw.get("side")
        if not isinstance(side, ast.Name) or side.id not in _GUARDED[callee]:
            shown = ast.dump(side) if side is not None else "<no side argument>"
            offenders.append(f"{file}:{line} -> {shown}")
    assert not offenders, (
        f"{callee} must be called with a side in {sorted(_GUARDED[callee])}, "
        f"as a named constant imported from makermodslab.motor_power.\n"
        f"Offending call sites:\n  " + "\n  ".join(offenders) + "\n\n"
        f"WHY: {_WHY[callee]}\n\n"
        "Do not silence this by passing FOLLOWER for a leader device — read "
        "clear_goal_velocity's docstring first."
    )
