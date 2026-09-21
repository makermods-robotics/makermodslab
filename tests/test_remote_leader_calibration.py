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
"""Remote leader builders must stage from the selected leader's real library."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from makermodslab.utils import config as cfg
from makermodslab.utils.robot_factory import build_leader_config


def _request(mode, leader_kind):
    return SimpleNamespace(
        arm_type="metal",
        mode=mode,
        leader_kind=leader_kind,
        robot_name="remote",
        leader_config="left",
        right_leader_config="right",
        leader_port="/dev/leader",
        right_leader_port="/dev/right-leader",
        follower_port="",
        right_follower_port="",
    )


def _libraries(leader_kind):
    star = Path(cfg.MAKER_LEADER_CONFIG_PATH)
    metal = Path(cfg.METAL_LEADER_CONFIG_PATH)
    return (metal, star) if leader_kind == "metal" else (star, metal)


@pytest.mark.parametrize("mode", ["single", "bimanual"])
@pytest.mark.parametrize("leader_kind", [None, "star", "metal"])
def test_remote_leader_build_uses_selected_library(tmp_lerobot_home, mode, leader_kind):
    selected, other = _libraries(leader_kind)
    # The same names in both libraries must not make a wrong-library read pass.
    for side in ("left", "right"):
        (selected / f"{side}.json").write_text(f"selected-{side}")
        (other / f"{side}.json").write_text(f"wrong-{side}")

    leader = build_leader_config(_request(mode, leader_kind))
    is_metal = leader_kind == "metal"
    if mode == "single":
        assert leader.type == ("metal_leader" if is_metal else "rebot_102_leader_metal")
        assert (leader.id, leader.port) == ("left", "/dev/leader")
    else:
        assert leader.type == ("bi_metal_leader" if is_metal else "bi_rebot_102_leader")
        assert (leader.left_arm_config.port, leader.right_arm_config.port) == (
            "/dev/leader",
            "/dev/right-leader",
        )
        for side in ("left", "right"):
            staged = leader.calibration_dir / f"{leader.id}_{side}.json"
            assert staged.read_text() == f"selected-{side}"


@pytest.mark.parametrize("mode", ["single", "bimanual"])
@pytest.mark.parametrize("leader_kind", [None, "star", "metal"])
def test_remote_leader_build_rejects_calibration_only_in_other_library(tmp_lerobot_home, mode, leader_kind):
    selected, other = _libraries(leader_kind)
    for side in ("left", "right"):
        (other / f"{side}.json").write_text("wrong-library")

    with pytest.raises(FileNotFoundError) as error:
        build_leader_config(_request(mode, leader_kind))
    assert str(selected) in str(error.value)
    assert "left.json" in str(error.value)
