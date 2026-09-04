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
"""The Maker Arm v1 family: a 7-DOF RobStride follower on classic CAN.

See can_common for everything shared with the Metal arm. Maker-specific:
the RobStride motor stores its zero internally and exposes no register
equivalent to Feetech's Homing_Offset (nothing to fingerprint), and the
fork registers a dedicated bimanual leader type (bi_rebot_102_leader_maker).
"""

from __future__ import annotations

from .can_common import CanArmFamily, CanDeviceClasses


class MakerFamily(CanArmFamily):
    id = "maker"
    label = "Maker Arm v1"
    short_label = "Maker"
    indefinite_label = "a Maker arm"

    follower_zero_pose = (
        "Move the arm by hand to its ZERO POSE — folded against the base, gripper fully open — then confirm."
    )

    single_robot_type = "maker_follower"
    bimanual_robot_type = "bi_maker_follower"
    robot_type_markers = ("maker",)

    follower_library_attr = "MAKER_FOLLOWER_CONFIG_PATH"

    def _device_classes(self) -> CanDeviceClasses:
        from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig
        from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase
        from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderMakerConfig
        from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMakerConfig
        from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (
            RebotArm102LeaderMakerTeleopConfig,
        )

        return CanDeviceClasses(
            follower=MakerFollowerConfig,
            follower_base=MakerFollowerConfigBase,
            bi_follower=BiMakerFollowerConfig,
            teleop=RebotArm102LeaderMakerTeleopConfig,
            leader_sub=RebotArm102LeaderMakerConfig,
            bi_teleop=BiRebot102LeaderMakerConfig,
        )


MAKER = MakerFamily()
