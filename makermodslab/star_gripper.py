"""Star gripper-mechanism presets, maintained in MakerMods Lab instead of LeRobot.

The Star Arm 102 leader ships with a left-right gripper lever; MakerMods fits other
mechanisms to the same servo. Each one is a Star leader kind that differs from the
stock preset in ONE number, the gripper's scale factor, measured on the hardware
and kept here so a gripper mechanism never moves the lerobot pin:

- ``star_vertical`` (Metal arm): the vertical grip, 41.2 deg of travel.
- ``star_trigger`` (Maker arm): the trigger, ~187 deg of travel the other way,
  of which the first half opens the jaw fully (see the trigger constants).
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

STAR_VERTICAL_LEADER_KIND = "star_vertical"
# Encoder readings captured on 2026-09-10, ten identical samples per endpoint.
VERTICAL_CLOSED_DEG = -17.2
VERTICAL_OPEN_DEG = 24.0

# The Maker arm's trigger grip. Measured 2026-09-10 on the trigger unit (servo id 6,
# multi-turn counter reset, read_raw_angle, two runs, 0.1 deg spread): a hard stop at
# both ends, 0.0 deg at the closed stop (the calibration origin) and -186.8 deg at the
# far stop. Only the FIRST HALF of the pull is mapped onto the jaw: fully open at
# -93.4 deg of trigger, and the rest of the pull clamps there, because pulling the
# trigger to its far stop is physically awkward. Change TRIGGER_USABLE_TRAVEL_DEG to
# use more or less of the pull; TRIGGER_FAR_STOP_DEG is the measurement, kept so the
# tests can prove the far stop still unwraps on the right 360 deg branch.
STAR_TRIGGER_LEADER_KIND = "star_trigger"
TRIGGER_CLOSED_DEG = 0.0
TRIGGER_FAR_STOP_DEG = -186.8
TRIGGER_USABLE_TRAVEL_DEG = TRIGGER_FAR_STOP_DEG / 2  # -93.4


def measured_gripper_scale(closed_deg: float, open_deg: float, output_open_deg: float = 115.0) -> float:
    """Scale travel relative to the fully closed encoder origin.

    Calibration sets the Star's hardware origin with its gripper fully closed.
    These two readings measure travel only; the old origin is not a preset offset.
    Preserve the signed sweep so an oppositely rotating grip opens correctly.
    ``output_open_deg`` is the follower's open target, signed: the Metal jaw opens
    towards +115, the Maker jaw towards -120.
    """
    if not all(math.isfinite(v) for v in (closed_deg, open_deg, output_open_deg)):
        raise ValueError("Gripper endpoints must be finite")
    travel = open_deg - closed_deg
    if not 1 <= abs(travel) < 360:
        raise ValueError("Gripper travel must be at least 1 and less than 360 degrees")
    if output_open_deg == 0:
        raise ValueError("Follower opening must be non-zero")
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


# --- the Maker arm's trigger grip -----------------------------------------------


def trigger_gripper_config(
    config: Any, *, closed_deg: float = TRIGGER_CLOSED_DEG, pulled_deg: float = TRIGGER_USABLE_TRAVEL_DEG
) -> Any:
    """Copy the stock Maker mapping, replacing only the gripper scale.

    The Maker jaw's open target is the LOW end of its range (-120; raw 0 lands on the
    -2 end), so that end is the output the pull maps onto.
    """
    open_target, _closed_target = config.joint_ranges["gripper"]
    return replace(
        config,
        joint_directions={
            **config.joint_directions,
            "gripper": measured_gripper_scale(closed_deg, pulled_deg, open_target),
        },
    )


def trigger_calibration_dir() -> str:
    # The trigger shares lerobot's rebot_102_leader class with the lever, so lerobot
    # would derive the lever's directory; the registry refuses two kinds of one
    # family in one library, and the configs below name this one outright.
    from .utils import config

    return str(Path(config.MAKER_LEADER_CONFIG_PATH).with_name("rebot_102_leader_trigger"))


def trigger_teleop_config(**kwargs):
    from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (
        RebotArm102LeaderMakerTeleopConfig,
    )

    kwargs.setdefault("calibration_dir", Path(trigger_calibration_dir()))
    return trigger_gripper_config(RebotArm102LeaderMakerTeleopConfig(**kwargs))


def trigger_sub_config(**kwargs):
    from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMakerConfig

    return trigger_gripper_config(RebotArm102LeaderMakerConfig(**kwargs))
