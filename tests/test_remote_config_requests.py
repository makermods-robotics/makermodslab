"""Build remote device configs from the actual one-sided request models."""

from unittest.mock import Mock

import pytest

from makermodslab.remote_host import HostingRequest
from makermodslab.remote_teleoperate import RemoteTeleoperateRequest
from makermodslab.utils import robot_factory


@pytest.mark.parametrize(
    "arm_type,leader_kind", [("so101", ""), ("maker", ""), ("metal", "star"), ("metal", "metal")]
)
@pytest.mark.parametrize("mode", ["single", "bimanual"])
@pytest.mark.parametrize("side", ["follower", "leader"])
def test_remote_request_builds_only_its_selected_side(
    monkeypatch, tmp_path, arm_type, leader_kind, mode, side
):
    staging = tmp_path / side
    stages = {
        "setup_follower_calibration_file": Mock(return_value="selected-follower"),
        "setup_leader_calibration_file": Mock(return_value="selected-leader"),
        "stage_bimanual_follower_calibrations": Mock(return_value=(staging, None)),
        "stage_bimanual_leader_calibrations": Mock(return_value=(staging, None)),
    }
    for name, mock in stages.items():
        monkeypatch.setattr(robot_factory, name, mock)

    fields = {
        "arm_type": arm_type,
        "mode": mode,
        "robot_name": "remote-test",
        f"{side}_port": "/not-opened-left",
        f"right_{side}_port": "/not-opened-right",
        f"{side}_config": "selected-left",
        f"right_{side}_config": "selected-right",
    }
    if side == "follower":
        request = HostingRequest(**fields)
        build = robot_factory.build_follower_config
    else:
        request = RemoteTeleoperateRequest(**fields, leader_kind=leader_kind, station="test-station")
        build = robot_factory.build_leader_config
    before = request.model_dump()
    before_fields_set = set(request.model_fields_set)

    # Config construction only: no device, serial port, Portal or worker is opened.
    config = build(request)

    assert request.model_dump() == before
    assert request.model_fields_set == before_fields_set
    unused = "leader" if side == "follower" else "follower"
    assert not hasattr(request, f"{unused}_port")
    assert not hasattr(request, f"right_{unused}_port")
    stages[f"setup_{unused}_calibration_file"].assert_not_called()
    stages[f"stage_bimanual_{unused}_calibrations"].assert_not_called()

    if mode == "single":
        assert config.id == f"selected-{side}"
        assert config.port == "/not-opened-left"
        stages[f"setup_{side}_calibration_file"].assert_called_once()
        stages[f"stage_bimanual_{side}_calibrations"].assert_not_called()
    else:
        assert config.calibration_dir == staging
        assert config.left_arm_config.port == "/not-opened-left"
        assert config.right_arm_config.port == "/not-opened-right"
        stages[f"stage_bimanual_{side}_calibrations"].assert_called_once()
        stages[f"setup_{side}_calibration_file"].assert_not_called()

    if side == "leader" and arm_type == "metal":
        expected = "metal_leader" if leader_kind == "metal" else "rebot_102_leader_metal"
        if mode == "bimanual":
            expected = "bi_metal_leader" if leader_kind == "metal" else "bi_rebot_102_leader"
        assert config.type == expected
