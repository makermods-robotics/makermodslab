"""Arm-type capability predicates.

Pure functions with no I/O — the seam every hardware path branches on, so the
point of pinning them is that a future refactor cannot quietly flip one and
silently re-enable a guard (or a feature) on hardware that cannot support it.
"""

import pytest

from makermodslab.arm_capabilities import (
    arm_type_from_robot_type,
    arm_type_of_robot_config,
    joints_per_arm,
    supports_auto_calibration,
    supports_dagger,
    uses_feetech_bus,
    uses_zero_calibration,
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


def test_auto_calibration_is_so101_only_and_zero_calibration_is_maker_only() -> None:
    """Each arm type has exactly one calibration procedure, and they differ.

    The SO-101 sweeps each joint's range under torque; the Maker arm's limits
    are fixed constants, so all it needs is a zero pose.
    """
    assert supports_auto_calibration("so101") is True
    assert supports_auto_calibration("maker") is False
    assert uses_zero_calibration("maker") is True
    assert uses_zero_calibration("so101") is False


def test_dagger_is_refused_on_the_maker_arm() -> None:
    """A hardware limit, not a policy choice.

    The Star Arm 102 leader that drives a Maker follower has encoders and no
    motors in its joints, so there is nothing to back-drive it with during a
    policy-to-human handover.
    """
    assert supports_dagger("maker") is False
    assert supports_dagger("so101") is True


def test_makermodslab_never_requests_a_dagger_rollout() -> None:
    """The other half of the DAgger guarantee: nothing here can ask for one.

    `supports_dagger` is a value nothing reads yet, so on its own it would not
    stop a DAgger run. What actually stops one is that rollout.py hardcodes
    `--strategy.type=base` and exposes no strategy option at all. Pin that, so
    adding a strategy picker has to come past this test.
    """
    from pathlib import Path

    rollout_src = Path("makermodslab/rollout.py").read_text()
    assert "--strategy.type=base" in rollout_src
    assert "dagger" not in rollout_src.lower()


@pytest.mark.parametrize("value", [None, "", "SO101", "star", 7, object()])
def test_unknown_arm_types_fall_back_to_so101(value: object) -> None:
    """A corrupted or future-dated record must never make a robot unopenable.

    so101 is the safe default: it is what every record written before the
    Maker arm existed implicitly is.
    """
    assert uses_feetech_bus(value) is True
    assert joints_per_arm(value) == 6


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
