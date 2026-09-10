"""Measured vertical grip mapping through real LeRobot drivers, without hardware."""

from types import SimpleNamespace

import pytest

from lerobot.teleoperators.utils import make_teleoperator_from_config
from makermodslab.arms import METAL
from makermodslab.star_gripper import measured_gripper_scale


def test_vertical_profile_changes_only_gripper_and_keeps_regular_star():
    stock = METAL.single_leader_config("fake", "test")
    vertical = METAL.single_leader_config("fake", "test", "star_vertical")
    assert vertical.joint_ranges == stock.joint_ranges
    assert vertical.joint_ids == stock.joint_ids
    assert vertical.joint_directions["gripper"] == pytest.approx(115 / 41.2)
    assert stock.joint_directions["gripper"] == 1.895
    for joint in stock.joint_directions:
        if joint != "gripper":
            assert vertical.joint_directions[joint] == stock.joint_directions[joint]


@pytest.mark.parametrize(
    "angle,target", [(0, 0), (20.6, 57.5), (41.2, 115), (-5, 0), (46, 115), (401.2, 115)]
)
def test_real_driver_maps_zeroed_encoder_travel_and_clips(angle, target):
    config = METAL.single_leader_config("fake", "vertical-test", "star_vertical")
    leader = make_teleoperator_from_config(config)
    leader.bus = SimpleNamespace(close=lambda: None)
    leader._read_raw_positions = lambda: {**dict.fromkeys(config.joint_ids, 0.0), "gripper": angle}
    try:
        assert leader.get_action()["gripper.pos"] == pytest.approx(target)
    finally:
        leader.disconnect()


def test_bimanual_driver_preserves_the_vertical_mapping_for_both_leaders(tmp_path):
    request = SimpleNamespace(
        leader_kind="star_vertical",
        leader_port="left",
        right_leader_port="right",
        follower_port="f-left",
        right_follower_port="f-right",
    )
    _, config = METAL.build_bimanual_configs(request, None, "vertical", str(tmp_path), str(tmp_path))
    leader = make_teleoperator_from_config(config)
    for arm in (leader.left_arm, leader.right_arm):
        assert arm.config.joint_directions["gripper"] == pytest.approx(115 / 41.2)
        assert arm.config.joint_ranges["gripper"] == [0, 115]


def test_vertical_choice_is_available_and_persists(client, tmp_lerobot_home):
    arms = {arm["id"]: arm for arm in client.get("/api/v1/arms").json()["arms"]}
    option = next(o for o in arms["metal"]["leader_options"] if o["id"] == "star_vertical")
    assert option["label"] == "Star arm vertical grip"
    assert option["available"] is True
    assert option["energized"] is False
    response = client.post(
        "/robots/vertical?create=true", json={"arm_type": "metal", "leader_kind": "star_vertical"}
    )
    assert response.status_code == 200
    assert client.get("/robots/vertical").json()["robot"]["leader_kind"] == "star_vertical"
    assert METAL.leader_calibration_dir("star_vertical") != METAL.leader_calibration_dir("star")
    config = METAL.single_leader_config("fake", "vertical", "star_vertical")
    assert str(config.calibration_dir) == METAL.leader_calibration_dir("star_vertical")


@pytest.mark.parametrize("closed,opened", [(0, 0), (0, 0.5), (0, 360), (float("nan"), 10), (0, float("inf"))])
def test_invalid_endpoint_measurements_are_rejected(closed, opened):
    with pytest.raises(ValueError):
        measured_gripper_scale(closed, opened)
