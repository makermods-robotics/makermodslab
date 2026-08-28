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
"""Shared leader/follower config-object assembly, per arm type.

Teleoperation and recording both build the same pair of lerobot config
objects — a follower ``robot_config`` and a leader ``teleop_config`` — from
a request that carries the four ports and calibration names (plus the four
bimanual variants). This module owns that assembly so the two call sites
stay byte-for-byte identical; each caller then instantiates the concrete
robot/teleop devices from the returned configs.

Two arm types are supported, chosen by the request's ``arm_type``:

* ``so101`` — SO-101 leader/follower, Feetech STS3215 over USB serial.
  ``SO101FollowerConfig`` / ``SO101LeaderConfig``, ``BiSO*`` when bimanual.
* ``maker`` — Maker Arm v1: a 7-DOF RobStride CAN follower driven by a Star
  Arm 102 (reBot 102) leader on FashionStar UART servos. ``MakerFollowerConfig``
  / ``RebotArm102LeaderMakerTeleopConfig``, ``BiMakerFollowerConfig`` /
  ``BiRebot102LeaderMakerConfig`` when bimanual. The leader MUST be the
  ``_maker`` preset, not the bare ``rebot_102_leader``: the base config
  describes the reBot B601 follower, and pointed at a Maker arm unchanged most
  joints run the wrong way or saturate against the follower's soft limits while
  teleop keeps reporting a healthy loop.

The camera wiring is the only difference between the two callers: recording
puts the session's cameras on the (left) follower arm; teleoperation passes
no cameras at all. Callers that want cameras pass ``cameras=<dict>``; callers
that don't (teleop) leave it as ``None`` and the follower config is built
without a ``cameras`` kwarg — preserving the exact object each site built
before this module existed.

Bimanual cameras go on the LEFT ARM config, not the bimanual config's
top-level ``cameras``, for both arm types. Both bimanual followers prefix
per-arm camera keys with ``left_``/``right_`` and leave top-level keys
unprefixed, so this choice is what keeps a bimanual dataset's camera feature
keys identical across arm types (and identical to what BiSO recordings have
always produced).

Inference (rollout.py) drives followers only and assembles its robot config
as subprocess CLI args, not as config objects, so it does not use this
module — the follower-only asymmetry lives there, not here.
"""

from pathlib import Path

from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderMakerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMakerConfig
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (
    RebotArm102LeaderMakerTeleopConfig,
)
from lerobot.teleoperators.so_leader import SO101LeaderConfig

from .config import (
    bimanual_base_id,
    normalize_arm_type,
    setup_calibration_files,
    stage_bimanual_calibrations,
)


def request_arm_type(request) -> str:
    """The arm type a start request targets, normalized.

    Requests that predate the Maker arm carry no ``arm_type`` at all, so a
    missing attribute reads as the SO-101 default rather than raising.
    """
    return normalize_arm_type(getattr(request, "arm_type", None))


def build_single_configs(request, cameras=None):
    """Build (robot_config, teleop_config) for a single leader/follower pair.

    Stages the selected library calibrations into lerobot's expected
    locations (via ``setup_calibration_files``) and returns a follower/leader
    config pair of the request's arm type. When ``cameras`` is provided it is
    wired onto the follower; when ``None`` the follower config is built without
    a ``cameras`` kwarg (teleoperation).
    """
    arm_type = request_arm_type(request)
    leader_config_name, follower_config_name = setup_calibration_files(
        request.leader_config, request.follower_config, arm_type
    )

    if arm_type == "maker":
        # MakerFollowerConfig defaults carry the CAN wiring (slcan @ 1 Mbps, the
        # per-joint ids, soft limits and MIT gains); only the adapter port and
        # the calibration id vary per session.
        if cameras is None:
            robot_config = MakerFollowerConfig(
                port=request.follower_port,
                id=follower_config_name,
            )
        else:
            robot_config = MakerFollowerConfig(
                port=request.follower_port,
                id=follower_config_name,
                cameras=cameras,
            )

        teleop_config = RebotArm102LeaderMakerTeleopConfig(
            port=request.leader_port,
            id=leader_config_name,
        )

        return robot_config, teleop_config

    if cameras is None:
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
        )
    else:
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
            cameras=cameras,
        )

    teleop_config = SO101LeaderConfig(
        port=request.leader_port,
        id=leader_config_name,
    )

    return robot_config, teleop_config


def build_bimanual_configs(request, cameras=None):
    """Build (robot_config, teleop_config) for a bimanual pair.

    Stages the four arbitrarily-named library calibrations into the bimanual
    "<base>_left/right.json" convention (via ``stage_bimanual_calibrations``)
    and returns a follower/leader config pair of the request's arm type,
    pointed at the per-device staging dirs. When ``cameras`` is provided it is
    wired onto the left follower arm; when ``None`` the left follower arm is
    built without a ``cameras`` kwarg (teleoperation).
    """
    arm_type = request_arm_type(request)
    base = bimanual_base_id(request.robot_name)
    leader_staging, follower_staging, _ = stage_bimanual_calibrations(
        base,
        request.leader_config,
        request.right_leader_config,
        request.follower_config,
        request.right_follower_config,
        arm_type,
    )

    if arm_type == "maker":
        # The bimanual configs embed the UNREGISTERED base dataclasses
        # (MakerFollowerConfigBase / RebotArm102LeaderMakerConfig) rather than
        # the registered ones, which is what keeps draccus's choice-registry
        # tree from becoming self-referential. Passing a registered config here
        # would still typecheck but re-enters that tree.
        if cameras is None:
            left_follower = MakerFollowerConfigBase(port=request.follower_port)
        else:
            left_follower = MakerFollowerConfigBase(port=request.follower_port, cameras=cameras)

        robot_config = BiMakerFollowerConfig(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=left_follower,
            right_arm_config=MakerFollowerConfigBase(port=request.right_follower_port),
        )
        teleop_config = BiRebot102LeaderMakerConfig(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=RebotArm102LeaderMakerConfig(port=request.leader_port),
            right_arm_config=RebotArm102LeaderMakerConfig(port=request.right_leader_port),
        )

        return robot_config, teleop_config

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


# --- single-device configs, for the calibration flows -------------------------
# Calibration connects ONE arm at a time, so it needs a config for that arm
# alone rather than the leader/follower pair the session builders return. These
# stay here so every Maker config object in the app is still built in one file.


def maker_follower_config(port: str, config_id: str) -> MakerFollowerConfig:
    """A single Maker follower config for zero-pose calibration.

    No cameras: calibration never opens one, and opening a camera here would
    hold it for the duration of a flow that is otherwise pure motor work.
    """
    return MakerFollowerConfig(port=port, id=config_id)


def maker_leader_config(port: str, config_id: str) -> RebotArm102LeaderMakerTeleopConfig:
    """A single Star Arm 102 leader config for zero-pose calibration.

    The ``_maker`` preset rather than the bare ``rebot_102_leader``, so the
    calibration file this run writes carries the Maker joint ranges the teleop
    session will later expect to find in it.
    """
    return RebotArm102LeaderMakerTeleopConfig(port=port, id=config_id)
