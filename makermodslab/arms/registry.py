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
"""The arm-family registry: the one place the core resolves an arm type.

Families register once, at import of makermodslab.arms (the built-ins) —
and, once the extension system lands, from an extension's activation. Lookups
are by the arm_type string; callers normalize first
(utils.config.normalize_arm_type) so an unknown value reads as the
default family rather than raising here.

Registration order is meaningful in exactly one place: the dataset
robot_type marker scan checks the default family last. Everything else
treats the registry as a set.
"""

from __future__ import annotations

from .base import REQUIRED_ATTRIBUTES, ArmFamily

# The family every record written before arm types existed implicitly is, and
# what an unrecognized value falls back to (see utils.config.normalize_arm_type).
DEFAULT_ID = "so101"

_FAMILIES: dict[str, ArmFamily] = {}


def register(family: ArmFamily) -> None:
    """Add a family. Refuses a duplicate id and an incomplete family, loudly.

    Completeness is checked here rather than left to the ABC because most of
    a family is data, and a missing attribute would otherwise surface as an
    AttributeError deep in whichever flow first read it.
    """
    if not isinstance(family, ArmFamily):
        raise TypeError(f"arm family must be an ArmFamily instance, got {type(family).__name__}")
    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(family, name)]
    if missing:
        raise TypeError(f"arm family {type(family).__name__} is missing required attributes: {missing}")
    if family.id in _FAMILIES:
        raise ValueError(f"arm family {family.id!r} is already registered")
    _FAMILIES[family.id] = family


def get(arm_type: str) -> ArmFamily:
    """The family registered under arm_type. KeyError when unknown — normalize first."""
    return _FAMILIES[arm_type]


def default() -> ArmFamily:
    """The default family (DEFAULT_ID)."""
    return _FAMILIES[DEFAULT_ID]


def ids() -> tuple[str, ...]:
    """Registered arm types, in registration order."""
    return tuple(_FAMILIES)


def families() -> tuple[ArmFamily, ...]:
    """Registered families, in registration order."""
    return tuple(_FAMILIES.values())


def family_for_robot_config_type(robot_type: object) -> ArmFamily:
    """The family whose followers register under this lerobot RobotConfig type string.

    Falls back to the default family for an unknown (or missing) type, which is
    the contract arm_capabilities.arm_type_of_robot_config has always had.
    """
    for family in _FAMILIES.values():
        if robot_type in family.robot_config_types():
            return family
    return default()
