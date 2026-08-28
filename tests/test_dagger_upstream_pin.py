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
"""Pins the lerobot code the coaching runner vendors, mirrors or reaches into.

`WebDAggerStrategy` does three things a subclass normally must not do, each for
a reason argued at its own call site:

  * it COPIES `DAggerStrategy._run_corrections_only` (cancel, the alignment gate
    and the phase events are not reachable from outside that loop);
  * it MIRRORS `_apply_transition`'s branching in `_transition_moves_the_arm`,
    to announce the ~2s handover glide BEFORE the blocking call;
  * it CALLS the private `RolloutStrategy._return_to_initial_position` and
    WRITES `DAggerEvents.phase` directly during a reset.

All four are silent couplings: if upstream changes any of them, nothing in this
repo raises. The vendored loop keeps running yesterday's logic, and the mirror
keeps announcing an arm movement that no longer happens (or, worse, stays quiet
through one that does).

So this test fails on a lerobot pin bump, on purpose, and is the only test here
whose correct response is "read the diff, then update the constant". It is not
asserting that upstream is right — only that a human has looked.

WHAT TO DO WHEN THIS FAILS

  1. `git diff` the named object between the old and new lerobot tags, or read
     it under `.venv/lib/python3.12/site-packages/lerobot/`.
  2. Port anything material into `dagger_runner._run_corrections` /
     `_transition_moves_the_arm`, keeping the MakerMods Lab deltas marked in
     that file's comments.
  3. Re-run with `--update-upstream-pins` printed below to get the new digests.

Digests are truncated sha256 of `inspect.getsource`, so they move on comment and
whitespace changes too. That is deliberate: upstream states load-bearing
reasoning in comments (the `prev_action` key-space caveat in `_apply_transition`
is a comment, not code), and a review that skipped those would miss it.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

from lerobot.rollout.strategies.core import RolloutStrategy
from lerobot.rollout.strategies.dagger import DAggerEvents, DAggerPhase, DAggerStrategy
from makermodslab.dagger_runner import WebDAggerStrategy

# Truncated sha256 of each object's source, as of the pinned lerobot (v0.6.0).
# Update ONLY after re-reading the upstream diff — see the module docstring.
_PINNED = {
    "DAggerStrategy._run_corrections_only": "036ca02102fc3277",
    "DAggerStrategy._apply_transition": "7102f74d661b5dd4",
    "RolloutStrategy._return_to_initial_position": "90abcaf4804bbf3b",
    "DAggerEvents": "0a13fbe73fb79909",
}

_OBJECTS = {
    "DAggerStrategy._run_corrections_only": DAggerStrategy._run_corrections_only,
    "DAggerStrategy._apply_transition": DAggerStrategy._apply_transition,
    "RolloutStrategy._return_to_initial_position": RolloutStrategy._return_to_initial_position,
    "DAggerEvents": DAggerEvents,
}

_WHY = {
    "DAggerStrategy._run_corrections_only": (
        "dagger_runner._run_corrections is a hand-maintained copy of this loop. "
        "Diff the two and port anything material."
    ),
    "DAggerStrategy._apply_transition": (
        "dagger_runner._transition_moves_the_arm mirrors this method's branching to "
        "decide when to announce the handover glide. If the conditions moved, the "
        "mirror now lies to the operator about whether the arm is travelling."
    ),
    "RolloutStrategy._return_to_initial_position": (
        "The coaching reset calls this private method to ease the follower home "
        "before cutting torque. A changed signature or a changed safety contract "
        "means the arm goes limp somewhere it should not."
    ),
    "DAggerEvents": (
        "The reset writes `events.phase` directly, bypassing consume_transition and "
        "the transition table. That is only safe while phase is the class's whole "
        "state. If it grew an invariant, the reset now corrupts it."
    ),
}


def _digest(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(_PINNED))
def test_vendored_upstream_source_is_unchanged(name: str) -> None:
    """Fails loudly on a lerobot bump so the hand-maintained copies get re-diffed."""
    actual = _digest(_OBJECTS[name])
    assert actual == _PINNED[name], (
        f"\n\nlerobot's {name} has changed.\n\n"
        f"{_WHY[name]}\n\n"
        f"Read the upstream diff, update makermodslab/dagger_runner.py to match, "
        f'then set _PINNED["{name}"] = "{actual}" in this file.\n'
    )


def test_return_to_initial_position_is_still_callable_the_way_the_reset_calls_it() -> None:
    """The reset passes `ctx.hardware` positionally and nothing else.

    A digest change tells you the body moved; this tells you the CALL broke,
    which is the failure the operator would actually meet (a reset that raises
    mid-session, with the arm parked wherever the policy left it)."""
    sig = inspect.signature(RolloutStrategy._return_to_initial_position)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    required = [p for p in params if p.default is inspect.Parameter.empty]
    assert len(required) == 1, (
        f"_return_to_initial_position now takes {len(required)} required arguments; "
        "dagger_runner's reset passes exactly one (ctx.hardware)."
    )


def test_the_phases_the_runner_switches_on_all_still_exist() -> None:
    """`_run_corrections` branches on these three by name; a rename is a silent
    fall-through to the autonomous branch, which drives the policy while the
    operator believes they are holding the arm."""
    assert {p.name for p in DAggerPhase} >= {"AUTONOMOUS", "PAUSED", "CORRECTING"}


def test_the_runner_still_inherits_what_it_claims_to_inherit() -> None:
    """Everything NOT vendored is inherited untouched. If upstream deletes one of
    these, the runner loses engine setup, the dataset finalize or the handovers
    themselves — and does so at teardown on the station, not here."""
    for name in ("_init_engine", "_apply_transition", "_background_push", "teardown", "_handle_warmup"):
        assert hasattr(WebDAggerStrategy, name), f"WebDAggerStrategy lost {name} on this lerobot pin"
