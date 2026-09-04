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
"""The SO-101 family: Feetech STS3215 smart servos on a USB serial bus.

Registers are readable and writable (EEPROM + RAM), which is what the
fingerprint, torque-cap and rest-pose machinery all depend on — hence
uses_feetech_bus. Calibration is a range sweep, manual or automatic
(the vendored Feetech autocal). 6 joints per arm. The leader has motors in
its joints, so it can be back-driven for a DAgger handover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ArmFamily


class SO101Family(ArmFamily):
    id = "so101"
    label = "SO-101"
    short_label = "SO-101"
    indefinite_label = "an SO-101 arm"

    joints_per_arm = 6
    supports_bimanual = True

    uses_feetech_bus = True
    supports_auto_calibration = True
    uses_zero_calibration = False
    supports_dagger = True

    single_robot_type = "so101_follower"
    bimanual_robot_type = "bi_so_follower"

    # The SO family is every string carrying an so100/so101 marker or the bare
    # so_follower/so_leader device names lerobot writes for a bimanual SO rig
    # (bi_so_follower). Loose on purpose; scanned last for that reason.
    robot_type_markers = ("so100", "so101", "so-100", "so-101", "so_follower", "so_leader")

    leader_library_attr = "LEADER_CONFIG_PATH"
    follower_library_attr = "FOLLOWER_CONFIG_PATH"

    # Leader and follower are the same Feetech bus: nothing to tell them apart
    # by protocol, so the SO-101 identifies by the hand-swing gesture only.
    follower_probe_protocol = None
    motion_identify_energizes_follower = False

    def default_calibration_name(self, record_name: str) -> str:
        """Historical default: the bare record name, no family suffix."""
        return record_name

    def single_follower_config(self, port: str, config_id: str):
        from lerobot.robots.so_follower import SO101FollowerConfig

        return SO101FollowerConfig(port=port, id=config_id)

    def single_leader_config(self, port: str, config_id: str):
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        return SO101LeaderConfig(port=port, id=config_id)

    async def identify_by_motion(self, device_type: str, ports: list[str] | None = None) -> dict:
        # Both halves speak Feetech serial on motor id 1, so the side asked
        # about changes nothing — identify.py watches the same register either way.
        from .. import identify

        return await identify.identify_arm_by_motion(ports)

    def build_single_configs(self, request: Any, cameras: dict | None, leader_id: str, follower_id: str):
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        if cameras is None:
            robot_config = SO101FollowerConfig(
                port=request.follower_port,
                id=follower_id,
            )
        else:
            robot_config = SO101FollowerConfig(
                port=request.follower_port,
                id=follower_id,
                cameras=cameras,
            )

        teleop_config = SO101LeaderConfig(
            port=request.leader_port,
            id=leader_id,
        )

        return robot_config, teleop_config

    def build_bimanual_configs(
        self,
        request: Any,
        cameras: dict | None,
        base: str,
        leader_staging: str,
        follower_staging: str,
    ):
        from lerobot.robots.bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        if cameras is None:
            left_follower = SO101FollowerConfig(port=request.follower_port)
        else:
            left_follower = SO101FollowerConfig(port=request.follower_port, cameras=cameras)

        robot_config = BiSOFollowerConfig(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=left_follower,
            right_arm_config=SO101FollowerConfig(port=request.right_follower_port),
        )
        teleop_config = BiSOLeaderConfig(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=SO101LeaderConfig(port=request.leader_port),
            right_arm_config=SO101LeaderConfig(port=request.right_leader_port),
        )

        return robot_config, teleop_config


SO101 = SO101Family()
