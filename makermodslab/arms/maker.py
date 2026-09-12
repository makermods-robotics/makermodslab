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
"""The Maker Arm v1 family: a 7-DOF RobStride follower on classic CAN.

See can_common for everything shared with the Metal arm. Maker-specific:
the RobStride motor stores its zero internally and exposes no register
equivalent to Feetech's Homing_Offset (nothing to fingerprint), and the
fork registers a dedicated bimanual leader type (bi_rebot_102_leader_maker).

Two leader kinds, one physical leader. Kind "star" (the default) is the stock
Star Arm 102 with its gripper lever. Kind "star_trigger" is the same arm fitted
with MakerMods' trigger grip: the preset is the stock Maker one with the gripper
scale replaced by the measured trigger travel, built in star_gripper.py the way
the Metal arm's vertical grip is, so the mechanism never moves the lerobot pin.
Neither kind is energized, so every leader-side hook treats them alike; the kind
picks the preset and the calibration library.
"""

from __future__ import annotations

from .base import LeaderOption
from .can_common import STAR_LEADER_KIND, CanArmFamily, CanDeviceClasses
from .urdf import maker_joint_positions


class MakerFamily(CanArmFamily):
    recording_realign_speed_deg_s = 60.0
    # Match SO-101's 400 steps/s nominal angular rate (4096 steps/revolution).
    # https://www.waveshare.com/wiki/ST3215_Servo
    recording_home_speed_deg_s = 400.0 * 360.0 / 4096.0
    recording_home_is_rest_pose = True
    id = "maker"
    label = "Maker Arm v1"
    short_label = "Maker"
    indefinite_label = "a Maker arm"

    supports_gripper_wiggle = True

    follower_zero_pose = "Move the arm by hand to its ZERO POSE — folded against the base, gripper fully closed — then confirm."

    single_robot_type = "maker_follower"
    bimanual_robot_type = "bi_maker_follower"
    robot_type_markers = ("maker",)

    follower_library_attr = "MAKER_FOLLOWER_CONFIG_PATH"

    # Bundled model plus the family's motor-degrees to URDF mapping.
    telemetry_kind = "urdf"

    def urdf_joint_positions(self, degrees: dict[str, float]) -> dict[str, float]:
        return maker_joint_positions(degrees)

    # RobStride frames; the probe is strictly read-only, so the gesture that
    # tells a bimanual rig's two followers apart is safe to watch.
    follower_probe_protocol = "robstride"
    motion_identify_energizes_follower = False

    def leader_options(self) -> tuple[LeaderOption, ...]:
        from ..star_gripper import STAR_TRIGGER_LEADER_KIND

        return (
            LeaderOption(id=STAR_LEADER_KIND, label="Star Arm 102 leader"),
            LeaderOption(id=STAR_TRIGGER_LEADER_KIND, label="Star arm trigger grip"),
        )

    def leader_calibration_dir(self, leader_kind: str | None = None) -> str:
        from ..star_gripper import STAR_TRIGGER_LEADER_KIND, trigger_calibration_dir

        if leader_kind == STAR_TRIGGER_LEADER_KIND:
            return trigger_calibration_dir()
        return super().leader_calibration_dir()

    def gripper_bus(self, port: str):
        from lerobot.motors.robstride import RobstrideMotorsBus
        from lerobot.robots.maker_follower.maker_follower import MOTOR_MODELS

        return self._build_gripper_bus(port, RobstrideMotorsBus, MOTOR_MODELS)

    def _device_classes(self, leader_kind: str | None = None) -> CanDeviceClasses:
        from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig
        from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase
        from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderMakerConfig
        from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMakerConfig
        from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (
            RebotArm102LeaderMakerTeleopConfig,
        )

        from ..star_gripper import STAR_TRIGGER_LEADER_KIND, trigger_sub_config, trigger_teleop_config

        if leader_kind == STAR_TRIGGER_LEADER_KIND:
            return CanDeviceClasses(
                follower=MakerFollowerConfig,
                follower_base=MakerFollowerConfigBase,
                bi_follower=BiMakerFollowerConfig,
                teleop=trigger_teleop_config,
                leader_sub=trigger_sub_config,
                bi_teleop=BiRebot102LeaderMakerConfig,
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
