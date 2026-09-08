"""Arm-type capability predicates.

Pure functions with no I/O — the seam every hardware path branches on, so the
point of pinning them is that a future refactor cannot quietly flip one and
silently re-enable a guard (or a feature) on hardware that cannot support it.
"""

import pytest

from makermodslab import arm_capabilities
from makermodslab.arm_capabilities import (
    arm_type_from_robot_type,
    arm_type_of_robot_config,
    joints_per_arm,
    supports_auto_calibration,
    supports_dagger,
    uses_feetech_bus,
)


def test_joints_per_arm_differs_between_arm_types() -> None:
    """The Maker arm is 7-DOF (6 joints + a permanent gripper), the SO-101 6.

    This is the number the inference arm-count guard measures a checkpoint's
    observation.state against, so getting it wrong disables that guard rather
    than tripping it.
    """
    assert joints_per_arm("so101") == 6
    assert joints_per_arm("maker") == 7


def test_feetech_only_helpers_are_off_for_the_maker_arm() -> None:
    """The fingerprint/torque-cap/rest-pose helpers are Feetech-register-based."""
    assert uses_feetech_bus("so101") is True
    assert uses_feetech_bus("maker") is False


def test_auto_calibration_is_so101_only_and_the_can_pair_are_step_wizards() -> None:
    """Each arm type has exactly one calibration procedure, named by its kind.

    The SO-101 sweeps each joint's range under torque (``range_sweep``); the
    CAN arms' limits are fixed constants, so all they need is a zero pose,
    which the family runs as a step wizard (``steps``). The boolean
    ``uses_zero_calibration`` is gone: an extension's kind may be ``panel``,
    which no boolean could name.
    """
    from makermodslab.arm_capabilities import calibration_kind

    assert supports_auto_calibration("so101") is True
    assert supports_auto_calibration("maker") is False
    assert calibration_kind("so101") == "range_sweep"
    assert calibration_kind("maker") == "steps"
    assert calibration_kind("metal") == "steps"
    assert not hasattr(arm_capabilities, "uses_zero_calibration")


def test_calibration_kind_reads_the_registry_live(monkeypatch) -> None:
    from makermodslab.arm_capabilities import calibration_kind
    from makermodslab.arms import registry
    from tests.mocks import make_arm_family, scratch_registry

    scratch_registry(monkeypatch)
    registry.register(
        make_arm_family("paneled", calibration_kind="panel", calibration_panel_url="/api/v1/ext/p/static/cal")
    )
    assert calibration_kind("paneled") == "panel"


def test_dagger_is_refused_on_the_maker_and_metal_arms() -> None:
    """A hardware limit, not a policy choice.

    The Star Arm 102 leader that drives a Maker or Metal follower has encoders
    and no motors in its joints, so there is nothing to back-drive it with
    during a policy-to-human handover. Coaching (DAgger) consults this in
    rollout.handle_start_inference — see
    test_rollout.test_coaching_is_refused_on_a_can_arm for the enforcement.
    """
    assert supports_dagger("maker") is False
    assert supports_dagger("metal") is False
    assert supports_dagger("so101") is True


@pytest.mark.parametrize("value", [None, "", 7, object()])
def test_missing_or_non_string_arm_types_fall_back_to_so101(value: object) -> None:
    """A record written before the Maker arm existed carries no arm_type and
    IS an SO-101; a non-string is a corrupted field, not a family. Both read
    as the default rather than making the robot unopenable."""
    assert uses_feetech_bus(value) is True
    assert joints_per_arm(value) == 6


@pytest.mark.parametrize("value", ["SO101", "star", "nope"])
def test_unknown_arm_type_strings_raise_instead_of_masquerading_as_so101(value: str) -> None:
    """TB5's locked decision: an unknown STRING is a family this install does
    not have, and answering "SO-101" for it would send a Feetech serial path
    at whatever the hardware really is. The predicates raise the registry's
    UnknownArmType (a KeyError); the refusal gates upstream make the raise
    unreachable from a request."""
    from makermodslab.arms.registry import UnknownArmType

    with pytest.raises(UnknownArmType):
        uses_feetech_bus(value)
    with pytest.raises(UnknownArmType) as excinfo:
        joints_per_arm(value)
    assert isinstance(excinfo.value, KeyError)
    assert value in str(excinfo.value)


def test_arm_type_read_back_off_a_built_robot_config() -> None:
    """Recording is handed a RecordConfig, not the original request, so it
    reads the arm type back off the assembled config instead of taking a
    parallel parameter that could drift out of agreement with it."""
    from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig
    from lerobot.robots.bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase
    from lerobot.robots.so_follower import SO101FollowerConfig

    assert arm_type_of_robot_config(MakerFollowerConfig(port="/dev/can")) == "maker"
    assert arm_type_of_robot_config(SO101FollowerConfig(port="/dev/tty")) == "so101"
    assert (
        arm_type_of_robot_config(
            BiMakerFollowerConfig(
                left_arm_config=MakerFollowerConfigBase(port="/dev/a"),
                right_arm_config=MakerFollowerConfigBase(port="/dev/b"),
            )
        )
        == "maker"
    )
    assert (
        arm_type_of_robot_config(
            BiSOFollowerConfig(
                left_arm_config=SO101FollowerConfig(port="/dev/a"),
                right_arm_config=SO101FollowerConfig(port="/dev/b"),
            )
        )
        == "so101"
    )


def test_a_config_without_a_type_reads_as_so101() -> None:
    """Defensive: the getattr fallback must not raise on an unexpected object."""
    assert arm_type_of_robot_config(object()) == "so101"
    assert arm_type_of_robot_config(None) == "so101"


@pytest.mark.parametrize(
    ("robot_type", "expected"),
    [
        # What this app writes into a recorded dataset's meta/info.json
        # (lerobot stores the robot object's .name there).
        ("so101_follower", "so101"),
        ("bi_so_follower", "so101"),
        ("maker_follower", "maker"),
        ("bi_maker_follower", "maker"),
        ("metal_follower", "metal"),
        ("bi_metal_follower", "metal"),
        # Legacy / community datasets recorded elsewhere.
        ("so100_follower", "so101"),
        ("so-101", "so101"),
        ("SO101", "so101"),
        ("  Maker_Follower  ", "maker"),
    ],
)
def test_arm_type_from_robot_type_maps_known_strings(robot_type: str, expected: str) -> None:
    """The dataset-side counterpart to arm_type_of_robot_config: it takes the
    free-form robot_type STRING from meta/info.json rather than a built config."""
    assert arm_type_from_robot_type(robot_type) == expected


@pytest.mark.parametrize("value", [None, "", "   ", "aloha", "widowx", "unknown", 7, object()])
def test_arm_type_from_robot_type_returns_none_when_unrecognized(value: object) -> None:
    """Unlike arm_type_of_robot_config, this must NOT default to so101: the
    compatibility warnings that call it have to stay silent on "don't know"
    rather than cry wolf about a dataset whose arm can't be established."""
    assert arm_type_from_robot_type(value) is None


def test_arm_type_from_robot_type_marker_match_is_deliberately_greedy() -> None:
    """A family marker anywhere in the string wins. That's the point — a
    dataset whose robot_type merely *contains* "metal"/"maker" is almost
    certainly that arm — but pin it so the behaviour is a choice, not an
    accident."""
    assert arm_type_from_robot_type("experimental_metal_rig") == "metal"
    assert arm_type_from_robot_type("bi_so100_follower") == "so101"
    assert arm_type_from_robot_type("so_leader") == "so101"
