# Copyright 2026 MakerMods. All rights reserved.
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
"""TB6a — the pre-rollout follower preflight is a family method, always called.

rollout used to branch on ``uses_feetech_bus`` around two module-level
SO-101 helpers; a family the core did not ship could never preflight. Now
every arg builder calls ``family.preflight_ports([...], skip_identity=...)``
once with one ``FollowerPreflight`` per follower (left, right), the base
answers "nothing to check", and the SO-101 override runs the identity check
then the register priming per follower, in order. The helpers moved into
arms/so101.py and are stubbed THERE. No bus is opened anywhere here.
"""

from __future__ import annotations

import logging

import pytest

from makermodslab.arm_identity import ArmIdentityError
from makermodslab.arms import ArmFamily, registry
from tests.mocks import make_arm_family, scratch_registry

# ---------------------------------------------------------------------------
# The contract object and the base default
# ---------------------------------------------------------------------------


def test_follower_preflight_is_a_frozen_value_with_an_optional_config_name() -> None:
    from makermodslab.arms.base import FollowerPreflight

    entry = FollowerPreflight("/dev/f", "cal")
    assert (entry.port, entry.calibration_id, entry.config_name) == ("/dev/f", "cal", None)
    assert FollowerPreflight("/dev/f", "dual_left", config_name="FC").config_name == "FC"
    with pytest.raises(AttributeError):
        entry.port = "/dev/other"  # type: ignore[misc]
    assert FollowerPreflight("/dev/f", "cal") == entry


def test_the_base_preflight_has_nothing_to_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A family whose motors keep their zero and drive gains internally has
    no register to prime and no fingerprint to compare — the base answers an
    empty warning list and never raises, whatever the entries."""
    from makermodslab.arms.base import FollowerPreflight

    family = make_arm_family("nine")
    entries = [FollowerPreflight("/dev/a", "A"), FollowerPreflight("/dev/b", "B", config_name="Bstem")]
    assert ArmFamily.preflight_ports(family, entries) == []
    assert ArmFamily.preflight_ports(family, entries, skip_identity=True) == []
    assert ArmFamily.preflight_ports(family, []) == []
    # A family that does not override it inherits that answer.
    plain = make_arm_family("plain")
    assert plain.preflight_ports(entries) == []


@pytest.mark.parametrize("family_id", ["maker", "metal"])
def test_the_can_families_keep_the_base_answer(family_id: str) -> None:
    from makermodslab.arms.base import FollowerPreflight

    family = registry.get(family_id)
    assert type(family).preflight_ports is ArmFamily.preflight_ports
    assert family.preflight_ports([FollowerPreflight("/dev/can0", "cal")]) == []


# ---------------------------------------------------------------------------
# The SO-101 override: identity then registers, per follower, in order
# ---------------------------------------------------------------------------


@pytest.fixture
def so101_helpers(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Stub the two moved helpers in arms/so101.py and record their calls."""
    from makermodslab.arms import so101

    calls: list[tuple] = []

    def identity(port, follower_id, config_name=None):
        calls.append(("identity", port, follower_id, config_name))
        return [f"id-warn:{port}"]

    def registers(port, follower_id):
        calls.append(("registers", port, follower_id))
        return [f"reg-warn:{port}"]

    monkeypatch.setattr(so101, "_preflight_arm_identity", identity)
    monkeypatch.setattr(so101, "_preflight_motor_registers", registers)
    return calls


def test_so101_runs_identity_then_registers_per_follower_in_order(so101_helpers: list[tuple]) -> None:
    """Each follower is opened and released in turn — never two at once —
    and its identity is verified BEFORE its registers are written, so a
    swapped arm is refused before anything is primed on it. The warnings
    come back in the same order."""
    from makermodslab.arms.base import FollowerPreflight

    so = registry.get("so101")
    warnings = so.preflight_ports(
        [
            FollowerPreflight("/dev/a", "dual_left", config_name="FC"),
            FollowerPreflight("/dev/b", "dual_right"),
        ]
    )
    assert so101_helpers == [
        ("identity", "/dev/a", "dual_left", "FC"),
        ("registers", "/dev/a", "dual_left"),
        ("identity", "/dev/b", "dual_right", None),
        ("registers", "/dev/b", "dual_right"),
    ]
    assert warnings == ["id-warn:/dev/a", "reg-warn:/dev/a", "id-warn:/dev/b", "reg-warn:/dev/b"]


def test_so101_skip_identity_still_primes_the_registers_and_warns(
    so101_helpers: list[tuple], caplog: pytest.LogCaptureFixture
) -> None:
    """``skip_identity`` is the operator's explicit "I know this arm" — it
    skips the fingerprint only. The torque-limit reset is never optional (a
    previous auto-calibration's cap would otherwise throttle the whole
    rollout), and the skip is logged as the warning it always was."""
    from makermodslab.arms.base import FollowerPreflight

    so = registry.get("so101")
    with caplog.at_level(logging.WARNING):
        warnings = so.preflight_ports([FollowerPreflight("/dev/a", "A")], skip_identity=True)
    assert so101_helpers == [("registers", "/dev/a", "A")]
    assert warnings == ["reg-warn:/dev/a"]
    assert any("SKIPPED" in record.getMessage() for record in caplog.records)


def test_so101_identity_error_propagates_before_the_registers_are_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.arms import so101
    from makermodslab.arms.base import FollowerPreflight

    calls: list[str] = []

    def refuse(port, follower_id, config_name=None):
        raise ArmIdentityError(f"The follower arm on {port} carries another calibration — SWAPPED")

    monkeypatch.setattr(so101, "_preflight_arm_identity", refuse)
    monkeypatch.setattr(so101, "_preflight_motor_registers", lambda port, fid: calls.append(port) or [])

    with pytest.raises(ArmIdentityError, match="SWAPPED"):
        registry.get("so101").preflight_ports([FollowerPreflight("/dev/a", "A")])
    assert calls == []


def test_the_helpers_left_rollout_for_the_so101_family() -> None:
    """The move is complete: rollout keeps only the coaching LEADER preflights
    (SO-101-specific by construction, gated by supports_dagger) and no
    follower helper or feetech branch of its own."""
    from makermodslab import rollout
    from makermodslab.arms import so101

    for name in ("_open_follower", "_preflight_arm_identity", "_preflight_motor_registers"):
        assert callable(getattr(so101, name)), f"arms/so101.py must define {name}"
        assert not hasattr(rollout, name), f"rollout.py must no longer define {name}"
    for name in ("_open_leader", "_preflight_leader_identity", "_preflight_leader_registers"):
        assert callable(getattr(rollout, name)), f"the coaching leader preflight {name} stays in rollout"
    assert not hasattr(rollout, "uses_feetech_bus"), "no feetech branch is left around the preflight"


def test_the_moved_helpers_keep_their_docstrings() -> None:
    from makermodslab.arms import so101

    assert "read-only" in so101._open_follower.__doc__.lower()
    assert "subprocess" in so101._preflight_arm_identity.__doc__
    assert "Torque_Limit" in so101._preflight_motor_registers.__doc__


# ---------------------------------------------------------------------------
# rollout's arg builders call it exactly once, with the right entries
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list, bool]] = []

    def __call__(self, followers, *, skip_identity=False):
        self.calls.append((list(followers), skip_identity))
        return ["family-warning"]


@pytest.fixture
def nine(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """A registered fake family that records what rollout asked it to
    preflight; every staging helper is stubbed so no library file is read."""
    from makermodslab import rollout

    recorder = _Recorder()
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine", preflight_ports=recorder))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name, arm_type="so101": name)
    monkeypatch.setattr(
        rollout, "setup_calibration_files", lambda leader, follower, *a, **k: (leader, follower)
    )
    monkeypatch.setattr(rollout, "bimanual_base_id", lambda name: "dual")
    monkeypatch.setattr(
        rollout, "stage_bimanual_follower_calibrations", lambda *a, **k: ("/staging/follower", "dual")
    )
    monkeypatch.setattr(
        rollout,
        "stage_bimanual_calibrations",
        lambda *a, **k: ("/staging/leader", "/staging/follower", "dual"),
    )
    monkeypatch.setattr(rollout, "_session_cameras", lambda request: {})
    # The coaching LEADER preflights stay in rollout and are not under test here.
    monkeypatch.setattr(rollout, "_preflight_leader_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_leader_registers", lambda *a, **k: [])
    return recorder


def _request(**overrides):
    from makermodslab.rollout import InferenceRequest

    fields = {"follower_port": "/dev/f", "follower_config": "FC", "policy_ref": "ref", "arm_type": "nine"}
    fields.update(overrides)
    return InferenceRequest(**fields)


def test_prepare_robot_single_preflights_the_one_follower(nine: _Recorder) -> None:
    from makermodslab.arms.base import FollowerPreflight
    from makermodslab.rollout import _prepare_robot

    args, warnings = _prepare_robot(_request(skip_identity_check=True))
    assert nine.calls == [([FollowerPreflight("/dev/f", "FC")], True)]
    assert warnings == ["family-warning"]
    assert "--robot.type=nine_follower" in args


def test_prepare_robot_bimanual_preflights_left_then_right_with_the_library_stems(nine: _Recorder) -> None:
    """The sub-arm ids are BiSO's staging aliases; the identity guard compares
    against the LIBRARY stems, so each entry carries its real config name."""
    from makermodslab.arms.base import FollowerPreflight
    from makermodslab.rollout import _prepare_robot

    args, warnings = _prepare_robot(
        _request(mode="bimanual", right_follower_port="/dev/rf", right_follower_config="RFC")
    )
    assert nine.calls == [
        (
            [
                FollowerPreflight("/dev/f", "dual_left", config_name="FC"),
                FollowerPreflight("/dev/rf", "dual_right", config_name="RFC"),
            ],
            False,
        )
    ]
    assert warnings == ["family-warning"]
    assert "--robot.type=bi_nine_follower" in args


def test_prepare_coaching_robot_single_preflights_the_follower_through_the_family(nine: _Recorder) -> None:
    from makermodslab.arms.base import FollowerPreflight
    from makermodslab.rollout import _prepare_coaching_robot

    _args, warnings = _prepare_coaching_robot(
        _request(coaching=True, leader_port="/dev/l", leader_config="LC", skip_identity_check=True)
    )
    assert nine.calls == [([FollowerPreflight("/dev/f", "FC")], True)]
    assert "family-warning" in warnings


def test_prepare_coaching_robot_bimanual_preflights_both_followers_once(nine: _Recorder) -> None:
    from makermodslab.arms.base import FollowerPreflight
    from makermodslab.rollout import _prepare_coaching_robot

    _args, warnings = _prepare_coaching_robot(
        _request(
            coaching=True,
            mode="bimanual",
            leader_port="/dev/l",
            leader_config="LC",
            right_leader_port="/dev/rl",
            right_leader_config="RLC",
            right_follower_port="/dev/rf",
            right_follower_config="RFC",
        )
    )
    assert nine.calls == [
        (
            [
                FollowerPreflight("/dev/f", "dual_left", config_name="FC"),
                FollowerPreflight("/dev/rf", "dual_right", config_name="RFC"),
            ],
            False,
        )
    ]
    assert "family-warning" in warnings


def test_an_identity_error_from_the_family_propagates_out_of_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup worker's existing hard-mismatch refusal
    (tests/test_arm_identity.py::test_start_inference_refuses_on_identity_error)
    catches ArmIdentityError out of _prepare_robot; the family's raise must
    reach it unwrapped."""
    from makermodslab import rollout

    def refuse(followers, *, skip_identity=False):
        raise ArmIdentityError("The follower arm on /dev/f carries calibration 'leader_a' — SWAPPED")

    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine", preflight_ports=refuse))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name, arm_type="so101": name)

    with pytest.raises(ArmIdentityError, match="SWAPPED"):
        rollout._prepare_robot(_request())


def test_the_so101_builders_reach_the_moved_helpers(so101_helpers: list[tuple], monkeypatch) -> None:
    """End to end for the built-in: rollout → SO101.preflight_ports → the
    helpers in arms/so101.py, with the request's skip flag honoured."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name, arm_type="so101": name)
    monkeypatch.setattr(rollout, "_session_cameras", lambda request: {})

    _args, warnings = rollout._prepare_robot(_request(arm_type="so101"))
    assert so101_helpers == [("identity", "/dev/f", "FC", None), ("registers", "/dev/f", "FC")]
    assert warnings == ["id-warn:/dev/f", "reg-warn:/dev/f"]

    so101_helpers.clear()
    rollout._prepare_robot(_request(arm_type="so101", skip_identity_check=True))
    assert so101_helpers == [("registers", "/dev/f", "FC")]
