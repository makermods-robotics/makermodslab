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
"""Shared leader/follower config-object assembly, per arm type.

Teleoperation and recording both build the same pair of lerobot config
objects — a follower ``robot_config`` and a leader ``teleop_config`` — from
a request that carries the four ports and calibration names (plus the four
bimanual variants). This module owns that assembly so the two call sites
stay byte-for-byte identical; each caller then instantiates the concrete
robot/teleop devices from the returned configs.

The per-family half — WHICH config classes, and how they nest for bimanual —
lives on the arm family (``makermodslab/arms``), chosen by the request's
``arm_type`` through the registry. What stays here is the part every family
shares: staging the selected library calibrations into lerobot's expected
locations first, then handing the staged ids to the family's builder. The
staging helpers are looked up on THIS module at call time (tests patch
``robot_factory.setup_calibration_files`` and friends).

The leader preset MUST match the follower family (a mismatched Star Arm 102
preset runs joints the wrong way or saturates them against the follower's
soft limits while teleop keeps reporting a healthy loop); that pairing is the
family's own, decided from ``arm_type`` alone — no caller ever picks a teleop
type string directly.

The camera wiring is the only difference between the two callers: recording
puts the session's cameras on the (left) follower arm; teleoperation passes
no cameras at all. Callers that want cameras pass ``cameras=<dict>``; callers
that don't (teleop) leave it as ``None`` and the follower config is built
without a ``cameras`` kwarg — preserving the exact object each site built
before this module existed.

Bimanual cameras go on the LEFT ARM config, not the bimanual config's
top-level ``cameras``, for every family. Both bimanual followers prefix
per-arm camera keys with ``left_``/``right_`` and leave top-level keys
unprefixed, so this choice is what keeps a bimanual dataset's camera feature
keys identical across arm types (and identical to what BiSO recordings have
always produced).

Inference (rollout.py) drives followers only and assembles its robot config
as subprocess CLI args, not as config objects, so it does not use this
module — the follower-only asymmetry lives there, not here.
"""

# Eager, deliberately: every device config class the three families build is
# imported HERE at module import (server.py imports this module through
# teleoperate/record), so a missing or broken CAN extra — `maker`, `damiao`,
# `rebot` — refuses to start the server, as it always has, instead of
# surfacing as the first Maker/Metal session's start failure. The families
# themselves import lazily inside their builders so that importing
# `makermodslab.arms` (which utils.config does) stays cheap; this block is
# the one place the whole stack is pulled in up front. Unused names on
# purpose.
from lerobot.robots.bi_maker_follower import BiMakerFollowerConfig  # noqa: F401
from lerobot.robots.bi_metal_follower import BiMetalFollowerConfig  # noqa: F401
from lerobot.robots.bi_so_follower import BiSOFollowerConfig  # noqa: F401
from lerobot.robots.maker_follower import MakerFollowerConfig  # noqa: F401
from lerobot.robots.metal_follower import MetalFollowerConfig  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102LeaderConfig  # noqa: F401
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig  # noqa: F401
from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMakerConfig  # noqa: F401
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (  # noqa: F401
    RebotArm102LeaderMakerTeleopConfig,
)
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_metal import (  # noqa: F401
    RebotArm102LeaderMetalTeleopConfig,
)
from lerobot.teleoperators.so_leader import SO101LeaderConfig  # noqa: F401

from ..arms import MAKER, METAL, registry as arm_registry
from .config import (
    bimanual_base_id,
    normalize_arm_type,
    setup_calibration_files,
    setup_follower_calibration_file,
    setup_leader_calibration_file,
    stage_bimanual_calibrations,
    stage_bimanual_follower_calibrations,
    stage_bimanual_leader_calibrations,
)


def request_arm_type(request) -> str:
    """The arm type a start request targets, normalized.

    Requests that predate the Maker arm carry no ``arm_type`` at all, so a
    missing attribute reads as the SO-101 default rather than raising.
    """
    return normalize_arm_type(getattr(request, "arm_type", None))


def request_leader_kind(request) -> str | None:
    """The leader kind a start request names, or None (the family's default)
    on a request that predates leader kinds or leaves it blank."""
    return getattr(request, "leader_kind", None) or None


def _leader_staging_kwargs(request, arm_type: str) -> dict[str, str]:
    """``leader_kind=`` for the staging helpers, ONLY for a multi-leader family
    (arms.base.leader_kwargs): a single-leader family's staging is unchanged,
    keyword and all, so nothing that patches those helpers sees a new argument."""
    from ..arms.base import leader_kwargs

    return leader_kwargs(arm_registry.get(arm_type), request_leader_kind(request))


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
        request.leader_config, request.follower_config, arm_type, **_leader_staging_kwargs(request, arm_type)
    )
    family = arm_registry.get(arm_type)
    return family.build_single_configs(request, cameras, leader_config_name, follower_config_name)


def build_follower_config(request, cameras=None):
    """Build only the follower side for a hosted robot station.

    The family builders still return a pair, but the unused leader config is
    never instantiated. This keeps construction and leader-kind rules in the
    arm registry while staging only the follower calibration files.
    """
    arm_type = request_arm_type(request)
    family = arm_registry.get(arm_type)
    if getattr(request, "mode", "single") == "bimanual":
        base = bimanual_base_id(request.robot_name)
        follower_staging, _ = stage_bimanual_follower_calibrations(
            base,
            request.follower_config,
            request.right_follower_config,
            arm_type,
        )
        robot_config, _ = family.build_bimanual_configs(
            request,
            cameras,
            base,
            follower_staging,
            follower_staging,
        )
        return robot_config

    follower_id = setup_follower_calibration_file(request.follower_config, arm_type)
    robot_config, _ = family.build_single_configs(request, cameras, "", follower_id)
    return robot_config


def build_leader_config(request):
    """Build only the leader side for a remote controller."""
    arm_type = request_arm_type(request)
    family = arm_registry.get(arm_type)
    if getattr(request, "mode", "single") == "bimanual":
        base = bimanual_base_id(request.robot_name)
        leader_staging, _ = stage_bimanual_leader_calibrations(
            base,
            request.leader_config,
            request.right_leader_config,
            arm_type,
            **_leader_staging_kwargs(request, arm_type),
        )
        _, teleop_config = family.build_bimanual_configs(
            request,
            None,
            base,
            leader_staging,
            leader_staging,
        )
        return teleop_config

    leader_id = setup_leader_calibration_file(
        request.leader_config,
        arm_type,
        **_leader_staging_kwargs(request, arm_type),
    )
    _, teleop_config = family.build_single_configs(request, None, leader_id, "")
    return teleop_config


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
        **_leader_staging_kwargs(request, arm_type),
    )
    family = arm_registry.get(arm_type)
    return family.build_bimanual_configs(request, cameras, base, leader_staging, follower_staging)


# --- single-device configs, for the calibration flows -------------------------
# Calibration connects ONE arm at a time, so it needs a config for that arm
# alone rather than the leader/follower pair the session builders return.
# These are the CAN families' single_follower_config / single_leader_config,
# kept under their historical names for the callers and tests that import
# them; new code asks the family directly.


def maker_follower_config(port: str, config_id: str):
    """A single Maker follower config for zero-pose calibration (no cameras)."""
    return MAKER.single_follower_config(port, config_id)


def maker_leader_config(port: str, config_id: str):
    """A single Star Arm 102 leader config with the Maker preset."""
    return MAKER.single_leader_config(port, config_id)


def metal_follower_config(port: str, config_id: str):
    """A single Metal follower config for zero-pose calibration (no cameras).

    NOTE for callers that connect it: the Damiao bus HANDSHAKE is the motor
    enable command, so the first thing to do after ``bus.connect()`` is
    ``bus.disable_torque()``.
    """
    return METAL.single_follower_config(port, config_id)


def metal_leader_config(port: str, config_id: str):
    """A single Star Arm 102 leader config with the Metal preset."""
    return METAL.single_leader_config(port, config_id)
