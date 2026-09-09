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
"""The arm-family registry: the one place the core resolves an arm type.

Families register once, at import of makermodslab.arms (the built-ins) —
and, once the extension system lands, from an extension's activation, which
records its name as the family's provenance. Lookups are by the arm_type
string and an unknown id RAISES (UnknownArmType, a KeyError): there is no
fallback family, because answering "SO-101" for hardware nobody registered
would open a Feetech serial path at whatever it really is. The API layer
refuses such an id before any lookup (arm_capabilities.require_known_arm_type
→ 400 robot.arm_type.unavailable); utils.config.normalize_arm_type only
defaults a MISSING value (a record written before arm types existed), never
an unknown string.

Registration order is meaningful in exactly one place: the dataset
robot_type marker scan checks the default family last. Everything else
treats the registry as a set.
"""

from __future__ import annotations

from .base import CALIBRATION_KINDS, REQUIRED_ATTRIBUTES, ArmFamily

# The family every record written before arm types existed implicitly is, and
# what a MISSING arm_type reads as (see utils.config.normalize_arm_type). An
# unknown string never falls back to it.
DEFAULT_ID = "so101"

# Provenance of the built-ins; an extension's registration passes its own name.
BUILTIN_PROVIDER = "builtin"

_FAMILIES: dict[str, ArmFamily] = {}
# arm_type -> who registered it (the manifest reports it). A module-level
# dict beside _FAMILIES on purpose: tests scratch every dict in this module
# together so a family registered inside a test takes its provenance with it.
_PROVIDERS: dict[str, str] = {}


class UnknownArmType(KeyError):  # noqa: N818 — the spec and the tests name it; it is the registry's KeyError
    """Lookup of an arm_type nothing has registered.

    A KeyError so existing ``except KeyError`` sites keep working; the message
    names what was asked for and what exists, which is what a log line needs
    to show when a refusal gate upstream was bypassed.
    """

    def __init__(self, arm_type: object) -> None:
        self.arm_type = arm_type
        super().__init__(f"unknown arm type {arm_type!r}; registered: {', '.join(_FAMILIES) or '(none)'}")

    def __str__(self) -> str:
        # KeyError.__str__ repr()s its single argument, which would wrap the
        # message in quotes; show it plainly.
        return self.args[0]


def register(family: ArmFamily, *, provided_by: str = BUILTIN_PROVIDER) -> None:
    """Add a family. Refuses a duplicate id and an incomplete family, loudly.

    Completeness is checked here rather than left to the ABC because most of
    a family is data, and a missing attribute would otherwise surface as an
    AttributeError deep in whichever flow first read it. ``provided_by`` is
    the provenance the manifest reports: the built-ins take the default, an
    extension passes its own name.
    """
    if not isinstance(family, ArmFamily):
        raise TypeError(f"arm family must be an ArmFamily instance, got {type(family).__name__}")
    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(family, name)]
    if missing:
        raise TypeError(f"arm family {type(family).__name__} is missing required attributes: {missing}")
    if family.id in _FAMILIES:
        raise ValueError(f"arm family {family.id!r} is already registered")
    _check_calibration_dirs(family)
    _check_calibration_kind(family)
    _FAMILIES[family.id] = family
    _PROVIDERS[family.id] = provided_by


def _calibration_dir(family: ArmFamily, method: str) -> str:
    """Call one of the family's dir methods, refusing a family that cannot answer.

    The library dirs are read by every readiness check and every staging
    copy; a family that raises (the base's NotImplementedError when neither
    the method nor the library attr is provided) or answers something other
    than a non-empty path would fail deep in the first flow that asks.
    """
    try:
        value = getattr(family, method)()
    except Exception as exc:
        raise TypeError(f"arm family {family.id!r} cannot answer {method}(): {exc}") from exc
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"arm family {family.id!r} must answer {method}() with a non-empty path, got {value!r}"
        )
    return value


def _check_calibration_dirs(family: ArmFamily) -> None:
    """Both dir methods answer; the follower dir is the family's alone; a
    shared leader dir is allowed only behind a calibration_name_suffix."""
    leader_dir = _calibration_dir(family, "leader_calibration_dir")
    follower_dir = _calibration_dir(family, "follower_calibration_dir")
    for other in _FAMILIES.values():
        if other.follower_calibration_dir() == follower_dir:
            raise ValueError(
                f"arm family {family.id!r} names follower calibration dir {follower_dir!r}, which "
                f"{other.id!r} already owns: two families writing one follower library would load "
                "each other's files by name"
            )
        if other.leader_calibration_dir() == leader_dir and not family.calibration_name_suffix:
            raise ValueError(
                f"arm family {family.id!r} shares leader calibration dir {leader_dir!r} with "
                f"{other.id!r} but has an empty calibration_name_suffix: sharing a leader library "
                "(one lerobot leader class, several presets — the Maker/Metal Star-leader case) is "
                "legitimate only when the suffix mints each family's id into its default calibration "
                "names, so the two never reuse a calibration written for the other"
            )


def _check_calibration_kind(family: ArmFamily) -> None:
    """The kind is one the core can run, and the family carries its prerequisites."""
    kind = family.calibration_kind
    if kind not in CALIBRATION_KINDS:
        raise ValueError(
            f"arm family {family.id!r} has calibration_kind {kind!r}; expected one of {CALIBRATION_KINDS}"
        )
    if kind == "range_sweep" and not family.uses_feetech_bus:
        raise ValueError(
            f"arm family {family.id!r} declares calibration_kind 'range_sweep' with uses_feetech_bus=False: "
            "the sweep managers read and write Feetech servo registers by name"
        )
    if kind == "steps":
        not_overridden = [
            name
            for name in ("calibrate", "open_for_calibration")
            if getattr(type(family), name) is getattr(ArmFamily, name)
        ]
        if not_overridden:
            raise TypeError(
                f"arm family {family.id!r} declares calibration_kind 'steps' but does not override "
                f"{' and '.join(not_overridden)}(): the step manager runs calibrate() on the device "
                "open_for_calibration() returns, and the base versions raise NotImplementedError"
            )
    if kind == "panel":
        url = family.calibration_panel_url
        if not isinstance(url, str) or not url:
            raise ValueError(
                f"arm family {family.id!r} declares calibration_kind 'panel' but calibration_panel_url "
                f"is {url!r}: the config dialog needs the served page to mount"
            )


def get(arm_type: str) -> ArmFamily:
    """The family registered under arm_type.

    Raises UnknownArmType (a KeyError) for an id nothing registered — never
    a fallback. Callers that take an arm type from a request or a record
    refuse it first with arm_capabilities.require_known_arm_type.
    """
    try:
        return _FAMILIES[arm_type]
    except KeyError:
        raise UnknownArmType(arm_type) from None


def provided_by(arm_type: str) -> str:
    """Who registered this family: BUILTIN_PROVIDER, or an extension's name."""
    get(arm_type)  # the same UnknownArmType for an unknown id
    return _PROVIDERS[arm_type]


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
