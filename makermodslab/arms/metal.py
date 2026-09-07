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
"""The Metal arm family: a 7-DOF Damiao follower on classic CAN.

See can_common for everything shared with the Maker arm. The Metal
difference that matters everywhere a bus is touched casually: the Damiao
HANDSHAKE is the per-motor enable command, so even a "read-only" ping
energizes the arm — the flows that connect one disable torque first thing.
Bimanual uses the GENERIC BiRebot102LeaderConfig carrying Metal-preset
sub-configs; the fork registers no bi_rebot_102_leader_metal type, and
the Metal mapping travels entirely in the sub-configs, which is the shape the
fork's own Metal docs prescribe.
"""

from __future__ import annotations

from .can_common import CanArmFamily, CanDeviceClasses


class MetalFamily(CanArmFamily):
    id = "metal"
    label = "Metal Arm"
    short_label = "Metal"
    indefinite_label = "a Metal arm"

    follower_zero_pose = (
        "Move the arm by hand to its ZERO POSE — standing upright, all "
        "joints at 0 degrees, gripper closed — then confirm."
    )

    single_robot_type = "metal_follower"
    bimanual_robot_type = "bi_metal_follower"
    robot_type_markers = ("metal",)

    follower_library_attr = "METAL_FOLLOWER_CONFIG_PATH"

    # Damiao frames — and the handshake that opens the bus IS the enable
    # command, so watching this follower's joints would energize it
    # mid-gesture. Identify a bimanual Metal rig by its leaders instead.
    follower_probe_protocol = "damiao"
    motion_identify_energizes_follower = True

    def _device_classes(self) -> CanDeviceClasses:
        from lerobot.robots.bi_metal_follower import BiMetalFollowerConfig
        from lerobot.robots.metal_follower import MetalFollowerConfig, MetalFollowerConfigBase
        from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderConfig
        from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMetalConfig
        from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_metal import (
            RebotArm102LeaderMetalTeleopConfig,
        )

        return CanDeviceClasses(
            follower=MetalFollowerConfig,
            follower_base=MetalFollowerConfigBase,
            bi_follower=BiMetalFollowerConfig,
            teleop=RebotArm102LeaderMetalTeleopConfig,
            leader_sub=RebotArm102LeaderMetalConfig,
            bi_teleop=BiRebot102LeaderConfig,
        )


METAL = MetalFamily()
