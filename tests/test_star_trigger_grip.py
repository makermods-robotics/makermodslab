"""Measured Maker trigger-grip mapping through real LeRobot drivers, without hardware."""

from types import SimpleNamespace

import pytest

from lerobot.teleoperators.utils import make_teleoperator_from_config
from makermodslab.arms import MAKER
from makermodslab.arms.base import leader_kwargs
from makermodslab.star_gripper import (
    TRIGGER_FAR_STOP_DEG,
    TRIGGER_USABLE_TRAVEL_DEG,
    measured_gripper_scale,
)

# 120 deg of jaw over the first 93.4 deg of pull, positive because the trigger
# turns the servo the opposite way to the lever.
TRIGGER_SCALE = -120 / -93.4


def test_trigger_profile_changes_only_gripper_and_keeps_regular_star():
    stock = MAKER.single_leader_config("fake", "test")
    trigger = MAKER.single_leader_config("fake", "test", "star_trigger")
    assert trigger.joint_ranges == stock.joint_ranges
    assert trigger.joint_ids == stock.joint_ids
    assert trigger.joint_directions["gripper"] == pytest.approx(TRIGGER_SCALE)
    assert stock.joint_directions["gripper"] == pytest.approx(-1.983613)
    for joint in stock.joint_directions:
        if joint != "gripper":
            assert trigger.joint_directions[joint] == stock.joint_directions[joint]
    # The lever pulls the servo one way, the trigger the other.
    assert trigger.joint_directions["gripper"] > 0 > stock.joint_directions["gripper"]


def test_the_maker_jaw_opens_towards_its_negative_limit():
    """The Maker jaw's open target is the low end of its range, so the scale is
    derived from a negative output. The helper must accept that, not just Metal's +115."""
    assert measured_gripper_scale(0.0, -93.4, -120) == pytest.approx(TRIGGER_SCALE)
    with pytest.raises(ValueError):
        measured_gripper_scale(0.0, -93.4, 0)


@pytest.mark.parametrize(
    "angle,target",
    [
        (0, -2),  # closed stop: the jaw's zero-pose end of the range
        (-46.7, -60),  # quarter pull: half open
        (-93.4, -120),  # half pull: fully open
        (-140, -120),  # past half: clamped open
        (TRIGGER_FAR_STOP_DEG, -120),  # far hard stop: still clamped open, no branch flip
        (5, -2),  # a hair past the closed stop: clamped closed
        (-93.4 - 360, -120),  # an extra full turn on the counter unwraps
    ],
)
def test_real_driver_maps_zeroed_encoder_travel_and_clips(angle, target, tmp_lerobot_home):
    config = MAKER.single_leader_config("fake", "trigger-test", "star_trigger")
    leader = make_teleoperator_from_config(config)
    leader.bus = SimpleNamespace(close=lambda: None)
    leader._read_raw_positions = lambda: {**dict.fromkeys(config.joint_ids, 0.0), "gripper": angle}
    try:
        assert leader.get_action()["gripper.pos"] == pytest.approx(target, abs=0.5)
    finally:
        leader.disconnect()


def test_the_far_stop_is_well_inside_the_unwrap_window():
    """The lever preset snapped the jaw at raw -150 because the far stop crossed the
    edge of the driver's 360 deg unwrap window. With the trigger scale the window is
    centred on the mapped pull, and the far stop must sit inside it with margin."""
    config = MAKER.single_leader_config("fake", "test", "star_trigger")
    lo, hi = config.joint_ranges["gripper"]
    direction = config.joint_directions["gripper"]
    center = (lo / direction + hi / direction) / 2
    assert center - 180 + 30 < TRIGGER_FAR_STOP_DEG < TRIGGER_USABLE_TRAVEL_DEG < 0 < center + 180 - 30


def test_bimanual_driver_preserves_the_trigger_mapping_for_both_leaders(tmp_path):
    request = SimpleNamespace(
        leader_kind="star_trigger",
        leader_port="left",
        right_leader_port="right",
        follower_port="f-left",
        right_follower_port="f-right",
    )
    _, config = MAKER.build_bimanual_configs(request, None, "trigger", str(tmp_path), str(tmp_path))
    assert config.type == "bi_rebot_102_leader_maker"
    leader = make_teleoperator_from_config(config)
    for arm in (leader.left_arm, leader.right_arm):
        assert arm.config.joint_directions["gripper"] == pytest.approx(TRIGGER_SCALE)
        assert arm.config.joint_ranges["gripper"] == [-120, -2]


def test_trigger_choice_is_available_and_persists(client, tmp_lerobot_home):
    arms = {arm["id"]: arm for arm in client.get("/api/v1/arms").json()["arms"]}
    assert [o["id"] for o in arms["maker"]["leader_options"]] == ["star", "star_trigger"]
    option = arms["maker"]["leader_options"][1]
    assert option["label"] == "Star arm trigger grip"
    assert option["available"] is True
    assert option["energized"] is False
    # A multi-leader family always hears the kind (its default when unset).
    assert leader_kwargs(MAKER, None) == {"leader_kind": "star"}
    response = client.post(
        "/api/v1/robots/trigger?create=true", json={"arm_type": "maker", "leader_kind": "star_trigger"}
    )
    assert response.status_code == 200, response.text
    assert client.get("/api/v1/robots/trigger").json()["robot"]["leader_kind"] == "star_trigger"
    assert MAKER.leader_calibration_dir("star_trigger") != MAKER.leader_calibration_dir("star")
    assert MAKER.leader_calibration_dir("star_trigger").endswith("rebot_102_leader_trigger")
    config = MAKER.single_leader_config("fake", "trigger", "star_trigger")
    assert str(config.calibration_dir) == MAKER.leader_calibration_dir("star_trigger")
    # The lever kind is what it always was: lerobot's own directory, no explicit override.
    assert MAKER.single_leader_config("fake", "lever").calibration_dir is None
