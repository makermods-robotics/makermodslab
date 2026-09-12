"""Star vertical gripper preset, maintained in MakerMods Lab instead of LeRobot."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

STAR_VERTICAL_LEADER_KIND = "star_vertical"
# Encoder readings captured on 2026-09-10, ten identical samples per endpoint.
VERTICAL_CLOSED_DEG = -17.2
VERTICAL_OPEN_DEG = 24.0


def measured_gripper_scale(closed_deg: float, open_deg: float, output_open_deg: float = 115.0) -> float:
    """Scale travel relative to the fully closed encoder origin.

    Calibration sets the Star's hardware origin with its gripper fully closed.
    These two readings measure travel only; the old origin is not a preset offset.
    Preserve the signed sweep so an oppositely rotating grip opens correctly.
    """
    if not all(math.isfinite(v) for v in (closed_deg, open_deg, output_open_deg)):
        raise ValueError("Gripper endpoints must be finite")
    travel = open_deg - closed_deg
    if not 1 <= abs(travel) < 360:
        raise ValueError("Gripper travel must be at least 1 and less than 360 degrees")
    if output_open_deg <= 0:
        raise ValueError("Follower opening must be positive")
    return output_open_deg / travel


def vertical_gripper_config(
    config: Any, *, closed_deg: float = VERTICAL_CLOSED_DEG, open_deg: float = VERTICAL_OPEN_DEG
) -> Any:
    """Copy the stock Metal mapping, replacing only the gripper scale."""
    return replace(
        config,
        joint_directions={
            **config.joint_directions,
            "gripper": measured_gripper_scale(closed_deg, open_deg, config.joint_ranges["gripper"][1]),
        },
    )


def vertical_calibration_dir() -> str:
    from .utils import config

    return str(Path(config.MAKER_LEADER_CONFIG_PATH).with_name("rebot_102_leader_vertical"))


def vertical_teleop_config(**kwargs):
    from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_metal import (
        RebotArm102LeaderMetalTeleopConfig,
    )

    kwargs.setdefault("calibration_dir", Path(vertical_calibration_dir()))
    return vertical_gripper_config(RebotArm102LeaderMetalTeleopConfig(**kwargs))


def vertical_sub_config(**kwargs):
    from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMetalConfig

    return vertical_gripper_config(RebotArm102LeaderMetalConfig(**kwargs))
