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

Two leader kinds, ONE physical leader. Kind "star" (the default) is the stock
Star Arm 102, whose gripper servo is worked by a left-right lever through
about 60 deg. Kind "star_trigger" is the same arm fitted with MakerMods'
trigger gripper, which turns that servo about 187 deg the OTHER way between
two hard stops. Same bus, same servo ids, same zero pose, same joints; the
two differ in one number, the gripper's scale factor, which lives in
lerobot's rebot_102_leader_maker_trigger preset. Run with the lever preset a
trigger leader barely moves the jaw (its travel lies outside the band the
lever mapped, so the jaw snaps closed-to-open only at the edge of the
multi-turn unwrap window). Neither kind is energized, so every leader-side
hook treats them alike; what the kind changes is the preset the session and
calibration configs are built from and the calibration library they read.
"""

from __future__ import annotations

from .base import LeaderOption
from .can_common import STAR_LEADER_KIND, CanArmFamily, CanDeviceClasses
from .urdf import maker_joint_positions

# The leader kind a robot record stores for a Star Arm 102 with the trigger gripper.
STAR_TRIGGER_LEADER_KIND = "star_trigger"

_TRIGGER_LEADER_ZERO_POSE = (
    "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, "
    "gripper trigger at its CLOSED stop — then confirm."
)


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
        return (
            LeaderOption(id=STAR_LEADER_KIND, label="Star Arm 102 leader"),
            LeaderOption(id=STAR_TRIGGER_LEADER_KIND, label="Star Arm 102 leader (trigger gripper)"),
        )

    def _is_trigger(self, leader_kind: str | None) -> bool:
        return self.normalize_leader_kind(leader_kind) == STAR_TRIGGER_LEADER_KIND

    def leader_calibration_dir(self, leader_kind: str | None = None) -> str:
        # One lerobot class for both kinds, so lerobot would derive ONE
        # directory; the registry refuses two kinds of a family sharing a
        # library, and the Lab names the trigger's itself (the single-arm
        # configs carry it as an explicit calibration_dir, see can_common).
        if self._is_trigger(leader_kind):
            from ..utils import config

            return config.MAKER_TRIGGER_LEADER_CONFIG_PATH
        return super().leader_calibration_dir()

    def zero_pose_instructions(
        self, device_type: object | None = None, leader_kind: str | None = None
    ) -> str:
        # The trigger's zero is the stop that closes the jaw; the rest of the
        # pose is the shared Star-leader one.
        if device_type == "teleop" and self._is_trigger(leader_kind):
            return _TRIGGER_LEADER_ZERO_POSE
        return super().zero_pose_instructions(device_type, leader_kind)

    def gripper_bus(self, port: str):
        from lerobot.motors.robstride import RobstrideMotorsBus
        from lerobot.robots.maker_follower.maker_follower import MOTOR_MODELS

        return self._build_gripper_bus(port, RobstrideMotorsBus, MOTOR_MODELS)

    def _device_classes(self, leader_kind: str | None = None) -> CanDeviceClasses:
        from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig
        from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase

        if self._is_trigger(leader_kind):
            from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderMakerTriggerConfig
            from lerobot.teleoperators.rebot_102_leader import (
                RebotArm102LeaderMakerTriggerConfig,
                RebotArm102LeaderMakerTriggerTeleopConfig,
            )

            return CanDeviceClasses(
                follower=MakerFollowerConfig,
                follower_base=MakerFollowerConfigBase,
                bi_follower=BiMakerFollowerConfig,
                teleop=RebotArm102LeaderMakerTriggerTeleopConfig,
                leader_sub=RebotArm102LeaderMakerTriggerConfig,
                bi_teleop=BiRebot102LeaderMakerTriggerConfig,
            )

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
