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
"""The Metal arm family: a 7-DOF Damiao follower on classic CAN.

See can_common for everything shared with the Maker arm. The Metal
difference that matters everywhere a bus is touched casually: the Damiao
HANDSHAKE is the per-motor enable command, so even a "read-only" ping
energizes the arm — the flows that connect one disable torque first thing.
Bimanual uses the GENERIC BiRebot102LeaderConfig carrying Metal-preset
sub-configs; the fork registers no bi_rebot_102_leader_metal type, and
the Metal mapping travels entirely in the sub-configs, which is the shape the
fork's own Metal docs prescribe.

Two leaders. The Star Arm 102 (kind "star", the default) is the encoder-only
leader every CAN family shares. Kind "metal" is the fork's gravity-compensated
METAL LEADER (lerobot's metal_leader / bi_metal_leader): a second Metal arm,
powered while the human moves it — a background thread streams Pinocchio
gravity-compensation torque with kp=0 so the operator feels only the arm's
inertia. Metal-to-Metal teleop is therefore the one flow where BOTH halves
are Damiao CAN devices, and everything the follower needs because the Damiao
handshake energizes, the leader needs too (see can_common's leader-kinds
note). Its availability is a dependency question: Pinocchio (``pin``) ships no
Windows wheels and is NOT in the default install — it comes with the opt-in
``makermodslab[metal-leader]`` extra (pyproject.toml), and until it is
installed the option lists as unavailable and every session that opens the
leader refuses with robot.leader_kind.unavailable. The first connect also
downloads the arm's URDF (the fork caches it under HF_LEROBOT_HOME/metal).
"""

from __future__ import annotations

import importlib.util

from .base import LeaderOption
from .can_common import STAR_LEADER_KIND, CanArmFamily, CanDeviceClasses

# The leader kind a robot record stores to be driven by a second Metal arm.
METAL_LEADER_KIND = "metal"

# The opt-in extra that installs Pinocchio for the gravity-compensated leader.
METAL_LEADER_EXTRA = "metal-leader"

METAL_LEADER_UNAVAILABLE = (
    "The gravity-compensated Metal leader needs Pinocchio, which is not installed. "
    f"Install it with `pip install 'makermodslab[{METAL_LEADER_EXTRA}]'` "
    "(or `uv pip install -e '.[metal-leader]'` from a checkout) and restart the server."
)


def _metal_leader_available() -> bool:
    """True when the gravity-compensated leader's one heavy dependency is importable.

    The fork's metal_leader package imports cleanly without Pinocchio (it is
    required inside MetalGravityModel, at connect), so an import probe of the
    device class would answer "available" on every install; the honest test
    is whether ``pinocchio`` itself resolves. Answered live, never cached: an
    install made while the server runs is picked up by the next manifest read.
    """
    return importlib.util.find_spec("pinocchio") is not None


class MetalFamily(CanArmFamily):
    id = "metal"
    label = "Metal Arm"
    short_label = "Metal"
    indefinite_label = "a Metal arm"

    follower_zero_pose = (
        "Move the arm by hand to its ZERO POSE — standing upright, all "
        "joints at 0 degrees, gripper fully closed — then confirm."
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

    # Two Damiao arms that answer the same protocol cannot be told apart by
    # probe or gesture; a gripper wiggle on one port can (can_wiggle.py).
    supports_gripper_wiggle = True

    def leader_options(self) -> tuple[LeaderOption, ...]:
        available = _metal_leader_available()
        return (
            LeaderOption(id=STAR_LEADER_KIND, label="Star Arm 102 leader"),
            LeaderOption(
                id=METAL_LEADER_KIND,
                label="Metal arm leader (gravity-compensated)",
                available=available,
                unavailable_reason=None if available else METAL_LEADER_UNAVAILABLE,
                energized=True,
            ),
        )

    def leader_calibration_dir(self, leader_kind: str | None = None) -> str:
        # The Metal leader is its own lerobot class (metal_leader), so lerobot
        # derives a directory of its own for it; the Star leader keeps the
        # library shared with the Maker arm.
        if self.leader_holds_torque(leader_kind):
            from ..utils import config

            return config.METAL_LEADER_CONFIG_PATH
        return super().leader_calibration_dir()

    def gripper_bus(self, port: str):
        from lerobot.motors.damiao import DamiaoMotorsBus
        from lerobot.robots.metal_follower.metal_follower import MOTOR_MODELS

        return self._build_gripper_bus(port, DamiaoMotorsBus, MOTOR_MODELS)

    def _device_classes(self, leader_kind: str | None = None) -> CanDeviceClasses:
        from lerobot.robots.bi_metal_follower import BiMetalFollowerConfig
        from lerobot.robots.metal_follower import MetalFollowerConfig, MetalFollowerConfigBase

        if self.leader_holds_torque(leader_kind):
            return CanDeviceClasses(
                follower=MetalFollowerConfig,
                follower_base=MetalFollowerConfigBase,
                bi_follower=BiMetalFollowerConfig,
                teleop=self._metal_leader_config_class(),
                leader_sub=self._metal_leader_sub_config_class(),
                bi_teleop=self._bi_metal_leader_config_class(),
            )

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

    # --- the Metal leader's config classes ---------------------------------------
    # Imported lazily, and ONLY when the metal leader kind is selected: the
    # fork's package imports without Pinocchio, but nothing here should touch
    # it for a Star-leader session. Each class is wrapped so the ONE knob the
    # stop path depends on is pinned: hold_kp_on_disconnect=0. The fork's
    # default (50) makes disconnect() freeze the arm in place with torque ON —
    # its answer to "the gravity thread stopped, do not let the arm fall" —
    # which is exactly the state every MakerMods Lab stop path exists to rule
    # out. Ours returns the arm to its start pose and releases it BEFORE
    # disconnect (capture_leader_rest_poses → return_to_rest → release_torque),
    # so the hold must not re-energize it on the way out.

    @staticmethod
    def _metal_leader_config_class():
        from lerobot.teleoperators.metal_leader import MetalLeaderConfig

        def build(port: str, id: str | None = None, **kwargs):  # noqa: A002 — lerobot's own field name
            kwargs.setdefault("hold_kp_on_disconnect", 0.0)
            return MetalLeaderConfig(port=port, id=id, **kwargs)

        return build

    @staticmethod
    def _metal_leader_sub_config_class():
        from lerobot.teleoperators.metal_leader import MetalLeaderConfigBase

        def build(port: str, **kwargs):
            kwargs.setdefault("hold_kp_on_disconnect", 0.0)
            return MetalLeaderConfigBase(port=port, **kwargs)

        return build

    @staticmethod
    def _bi_metal_leader_config_class():
        from lerobot.teleoperators.bi_metal_leader import BiMetalLeaderConfig

        return BiMetalLeaderConfig


METAL = MetalFamily()
