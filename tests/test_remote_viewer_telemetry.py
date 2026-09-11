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
"""Remote CAN telemetry uses the same family mapping as local viewers."""

import pytest

from makermodslab import remote_host, teleoperate
from makermodslab.arms import registry


@pytest.mark.parametrize("arm_type", ["maker", "metal"])
@pytest.mark.parametrize("bimanual", [False, True])
def test_remote_viewer_matches_local_family_mapping(arm_type, bimanual):
    family = registry.get(arm_type)
    sides = [("left_", 30), ("right_", -45)] if bimanual else [("", 30)]
    observation = {
        f"{prefix}{motor}.pos": angle
        for prefix, angle in sides
        for motor in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        )
    }
    actual = remote_host.host_joint_data(observation, {}, arm_type, bimanual)
    local = teleoperate.get_can_joint_data(None, family, bimanual, 123, observation=observation)
    del local["type"]
    del local["timestamp"]
    assert actual == local
    assert actual["joints"]
    assert actual["joints_deg"]["shoulder_pan"] == 30
    if bimanual:
        assert actual["joints_right"] != actual["joints"]
        assert actual["joints_deg_right"]["shoulder_pan"] == -45
