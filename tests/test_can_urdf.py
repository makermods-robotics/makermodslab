"""CAN viewer mappings and asset contracts, without connecting to hardware."""

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock

import pytest

from makermodslab.arms.base import ArmFamily
from makermodslab.arms.maker import MAKER
from makermodslab.arms.metal import METAL
from makermodslab.teleoperate import get_can_joint_data


@pytest.mark.parametrize("family,joint", [(MAKER, "link_002_joint"), (METAL, "joint1")])
@pytest.mark.parametrize("bimanual", [False, True])
def test_one_observation_drives_each_arm_and_raw_readout(family, joint, bimanual):
    observation = (
        {"left_shoulder_pan.pos": 30.0, "right_shoulder_pan.pos": -60.0}
        if bimanual
        else {"shoulder_pan.pos": 30.0}
    )
    robot = Mock()
    robot.get_observation.return_value = observation
    data = get_can_joint_data(robot, family, bimanual, 12.0)
    robot.get_observation.assert_called_once_with()
    assert data["type"] == "joint_update"
    assert data["timestamp"] == 12.0
    assert data["joints"] == {joint: pytest.approx(math.pi / 6)}
    assert data["joints_deg"] == {"shoulder_pan": 30.0}
    if bimanual:
        assert data["joints_right"] == {joint: pytest.approx(-math.pi / 3)}
        assert data["joints_deg_right"] == {"shoulder_pan": -60.0}
    else:
        assert "joints_right" not in data


@pytest.mark.parametrize("family", [MAKER, METAL])
def test_bad_readings_hold_pose_instead_of_resetting_or_poisoning_the_viewer(family):
    robot = Mock()
    robot.get_observation.return_value = {
        "shoulder_pan.pos": float("nan"),
        "wrist_yaw.pos": float("inf"),
        "gripper.pos": "bad",
        "camera": object(),
        "elbow_flex.pos": 90.0,
    }
    data = get_can_joint_data(robot, family, False, 0)
    assert data["joints_deg"] == {"elbow_flex": 90.0}
    assert len(data["joints"]) == 1
    robot.get_observation.side_effect = RuntimeError("offline")
    data = get_can_joint_data(robot, family, True, 0)
    assert data["joints"] == data["joints_right"] == {}
    assert data["joints_deg"] == data["joints_deg_right"] == {}


def test_family_without_model_keeps_numeric_readout():
    family = Mock()
    family.urdf_joint_positions = lambda degrees: ArmFamily.urdf_joint_positions(family, degrees)
    robot = Mock()
    robot.get_observation.return_value = {"custom.pos": 23.0}
    data = get_can_joint_data(robot, family, False, 0)
    assert data["joints"] == {}
    assert data["joints_deg"] == {"custom": 23.0}


def test_metal_all_six_joints_preserve_vendor_direction_and_units():
    values = dict(
        zip(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"),
            (30, -90, 90, -45, 45, 60),
            strict=True,
        )
    )
    assert METAL.urdf_joint_positions(values) == {
        f"joint{i}": pytest.approx(math.radians(deg)) for i, deg in enumerate(values.values(), 1)
    }


@pytest.mark.parametrize(
    "angle,travel",
    [
        (-0.1, 0.0),
        (0.002, 0.0),
        (0.295, -0.005),
        (1.10383, -0.025),
        ((1.10383 + 1.11897) / 2, -0.02525),
        (1.97527, -0.05),
        (2.4, -0.05),
    ],
)
def test_metal_gripper_uses_vendor_nonlinear_curve_and_clamps(angle, travel):
    assert METAL.urdf_joint_positions({"gripper": math.degrees(angle)}) == {
        "joint7": pytest.approx(travel),
    }


@pytest.mark.parametrize(
    "family,folder,filename,mimic",
    [
        (MAKER, "maker-urdf", "robot.urdf", "gripper_right_joint"),
        (METAL, "metal-urdf", "metal_with_gripper.urdf", "joint8"),
    ],
)
def test_shipped_geometry_resolves_and_telemetry_targets_real_joints(family, folder, filename, mimic):
    root = Path(__file__).parents[1] / "frontend/public" / folder
    urdf = ET.parse(root / filename)
    joints = {j.attrib["name"]: j for j in urdf.findall("joint")}
    motors = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    )
    data = family.urdf_joint_positions(dict.fromkeys(motors, 0.0))
    assert len(data) == 7
    assert all(name in joints for name in data)
    assert joints[mimic].find("mimic").attrib["joint"] in data
    for mesh in urdf.findall(".//mesh"):
        relative = "meshes/" + mesh.attrib["filename"].split("/meshes/")[-1].removeprefix("meshes/")
        path = root / relative
        # Exact case matters on Linux; macOS exists() alone cannot verify it.
        assert path.name in {f.name for f in path.parent.iterdir()}
        content = path.read_bytes()
        assert not content.startswith(b"version https://git-lfs")
        triangles = struct.unpack_from("<I", content, 80)[0]
        assert len(content) == 84 + 50 * triangles
