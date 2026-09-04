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
"""What the two CAN families (Maker, Metal) share.

Both are 7-DOF followers on classic CAN (via an slcan adapter) driven by a
Star Arm 102 (reBot 102) leader on FashionStar UART servos — the follower's
motor protocol (RobStride vs Damiao) is the only hardware difference, and it
lives entirely inside lerobot's device classes. Neither bus speaks the
Feetech register protocol, so every register-by-name helper is off; their
joint limits are measured constants, so calibration is a zero pose set by
hand with torque off; and the Star leader's joints hold encoders and no
motors, so there is nothing to back-drive for a DAgger handover.

The leader preset MUST match the follower family — never the bare
rebot_102_leader: each preset carries the joint directions and ranges of
ITS follower, and a mismatched one runs joints the wrong way or saturates
them against the follower's soft limits while teleop keeps reporting a
healthy loop. Each family names its own preset classes in _device_classes
and the builders here never choose one.

Both leaders share ONE calibration library (rebot_102_leader): the
presets are config-only variants of one lerobot class and the class name
picks the directory. That sharing is why default_calibration_name mints
the family id into the slot name (inherited from the base contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ArmFamily


@dataclass(frozen=True)
class CanDeviceClasses:
    """The six lerobot config classes a CAN family builds sessions from.

    The bimanual configs embed the UNREGISTERED base dataclasses
    (follower_base / leader_sub) rather than the registered ones,
    which is what keeps draccus's choice-registry tree from becoming
    self-referential. Passing a registered config there would still typecheck
    but re-enters that tree.
    """

    follower: type  # registered single follower config (port, id, cameras)
    follower_base: type  # unregistered follower base, for the bimanual arms
    bi_follower: type  # registered bimanual follower config
    teleop: type  # registered single-leader preset (port, id)
    leader_sub: type  # unregistered leader preset, for the bimanual arms
    bi_teleop: type  # bimanual leader config carrying two leader_sub configs


# Both CAN families use the same Star Arm 102 leader, and it has ONE zero
# pose: folded against the base, gripper closed. The follower poses are
# family-specific (follower_zero_pose) and opposite on the gripper.
_LEADER_ZERO_POSE = "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, gripper closed — then confirm."


class CanArmFamily(ArmFamily):
    """Shared shape of the CAN families; subclasses name their classes and
    their follower's zero pose."""

    joints_per_arm = 7
    supports_bimanual = True

    # The follower's zero-pose text; set by each family (the two are
    # opposites on the gripper — Maker folded/open, Metal upright/closed).
    follower_zero_pose: str

    uses_feetech_bus = False
    supports_auto_calibration = False
    uses_zero_calibration = True
    supports_dagger = False

    leader_library_attr = "MAKER_LEADER_CONFIG_PATH"

    def _device_classes(self) -> CanDeviceClasses:  # pragma: no cover - abstract by convention
        """Import (lazily — python-can / motorbridge) and return this family's classes."""
        raise NotImplementedError

    def zero_pose_instructions(self, device_type: object | None = None) -> str:
        if device_type == "teleop":
            return _LEADER_ZERO_POSE
        return self.follower_zero_pose

    def single_follower_config(self, port: str, config_id: str):
        return self._device_classes().follower(port=port, id=config_id)

    def single_leader_config(self, port: str, config_id: str):
        return self._device_classes().teleop(port=port, id=config_id)

    def build_single_configs(self, request: Any, cameras: dict | None, leader_id: str, follower_id: str):
        # The follower config's defaults carry the CAN wiring (slcan @ 1 Mbps,
        # the per-joint ids, soft limits and MIT gains); only the adapter port
        # and the calibration id vary per session.
        cls = self._device_classes()
        if cameras is None:
            robot_config = cls.follower(
                port=request.follower_port,
                id=follower_id,
            )
        else:
            robot_config = cls.follower(
                port=request.follower_port,
                id=follower_id,
                cameras=cameras,
            )

        teleop_config = cls.teleop(
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
        cls = self._device_classes()
        if cameras is None:
            left_follower = cls.follower_base(port=request.follower_port)
        else:
            left_follower = cls.follower_base(port=request.follower_port, cameras=cameras)

        robot_config = cls.bi_follower(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=left_follower,
            right_arm_config=cls.follower_base(port=request.right_follower_port),
        )
        teleop_config = cls.bi_teleop(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=cls.leader_sub(port=request.leader_port),
            right_arm_config=cls.leader_sub(port=request.right_leader_port),
        )

        return robot_config, teleop_config
