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
"""What each arm type can and cannot do.

Every feature module branches on arm type somewhere: which preflight guards
apply, which calibration flow to run, how many joints a checkpoint should
have. Those predicates live here, once, rather than as `arm_type == "maker"`
literals scattered across teleoperate/record/rollout/replay — a scattered
check is a check somebody forgets to add to the next flow.

The set of arm types is OPEN: the registry (``makermodslab/arms``) holds
the three built-ins below plus whatever an extension registers, and every
predicate here asks it live. A MISSING arm type (None — a record written
before arm types existed) reads as the default SO-101; an unknown STRING
raises the registry's ``UnknownArmType`` rather than masquerading as an
SO-101, and ``require_known_arm_type`` is the one refusal every request
gate uses so that raise is never reached from a request.

The three built-in arm types:

* ``so101`` — SO-101 leader/follower. Feetech STS3215 smart servos on a USB
  serial bus. Registers are readable and writable (EEPROM + RAM), which is
  what the fingerprint, torque-cap and rest-pose machinery all depend on.
* ``maker`` — Maker Arm v1. A 7-DOF RobStride follower on classic CAN (via an
  slcan adapter) driven by a Star Arm 102 (reBot 102) leader on FashionStar
  UART servos. Neither bus speaks the Feetech register protocol.
* ``metal`` — Metal arm. A 7-DOF Damiao follower on classic CAN, driven by
  the same Star Arm 102 leader with a Metal joint-mapping preset. Shares the
  Maker arm's integration seams (zero-pose calibration, MIT-setpoint rest
  return, no Feetech registers) with one hardware difference that matters
  everywhere a bus is touched casually: the Damiao HANDSHAKE is the motor
  enable command, so even a "read-only" ping energizes the arm.

Import this instead of writing the comparison inline. The answers come from
the arm-family registry (``makermodslab/arms``); each predicate here is a
named lookup that keeps the hardware reasoning next to the flag it gates.
"""

from collections.abc import Iterator, Mapping

from .api_errors import ApiError, ErrorCode
from .arms import registry as _registry
from .utils.config import DEFAULT_ARM_TYPE, is_known_arm_type, normalize_arm_type


def _family(arm_type: object):
    """The family for an arm type: None/non-string → the default; an unknown
    string → UnknownArmType (KeyError). See utils.config.normalize_arm_type."""
    return _registry.get(normalize_arm_type(arm_type))


def require_known_arm_type(arm_type: object) -> None:
    """Refuse (400 robot.arm_type.unavailable) an arm type nothing registered.

    THE gate every request path calls before an arm type reaches the
    registry or a device builder: the sessions front door, the legacy start
    handlers, the robot-record upsert, the calibration-library routes and
    the CAN-only routes. One helper so every refusal carries the same
    status, code and remedy.

    Reads its input the way normalize_arm_type does: None, "" and a
    non-string mean "unspecified" and pass as the default family (an absent
    ``arm_type`` in a request body or an empty ``?arm_type=`` query is an
    SO-101, exactly as a pre-Maker record on disk is); only a STRING nothing
    registered is refused. A known id returns None.
    """
    resolved = normalize_arm_type(arm_type)
    if not is_known_arm_type(resolved):
        raise ApiError(
            status_code=400,
            detail=(
                f"Arm type {resolved!r} is not installed. Install the extension that "
                "provides it, or delete this robot and create it again with an installed arm type."
            ),
            code=ErrorCode.ROBOT_ARM_TYPE_UNAVAILABLE,
        )


def joints_per_arm(arm_type: object) -> int:
    """Joint count of a single follower arm of this type.

    Flat proprioceptive width of ONE follower arm — one dim per joint. The
    SO-101 has 6; the CAN arms have 7 (6 joints plus a permanent gripper).
    This is the number a bimanual robot doubles, and the number a trained
    checkpoint's observation.state must match.
    """
    return _family(arm_type).joints_per_arm


def uses_feetech_bus(arm_type: object) -> bool:
    """True when this arm type's servos speak the Feetech register protocol.

    Gates every helper that reads or writes a servo register by name:

    * ``arm_identity`` — fingerprints an arm by reading Homing_Offset out of
      EEPROM. A RobStride motor stores its zero internally and exposes no
      equivalent register, and the Maker calibration writes homing_offset=0 for
      every joint, so there is literally nothing to fingerprint.
    * ``motor_power`` — caps Torque_Limit and clears Goal_Velocity. Both are
      Feetech RAM registers; the Maker follower's drive effort is set by its
      MIT position-control gains at connect() instead.
    * ``rest_pose`` — eases an arm home by writing Goal_Position in raw ticks.
    * ``identify`` / ``wiggle`` — port detection by watching (or driving)
      Present_Position on Feetech motor id 1.

    A Maker or Metal session skips all of them; ``maker_ports`` provides the
    CAN/UART port detection that replaces identify/wiggle.
    """
    return _family(arm_type).uses_feetech_bus


def supports_auto_calibration(arm_type: object) -> bool:
    """True when this arm type has an automatic (driven) calibration.

    Only the SO-101 does. Auto-calibration drives the arm under torque against
    its own stops to record each joint's range, and writes the result to servo
    EEPROM — a Feetech-specific procedure end to end (see the vendored script
    in ``vendor/feetech_autocal``).

    The CAN arms need no range sweep at all: their joint limits are fixed
    constants (``MakerFollowerConfig.joint_limits`` /
    ``MetalFollowerConfig.joint_limits``), measured once against the arms'
    mechanical stops. All their calibration has to establish is where zero
    is, which their families do as a step wizard (``step_calibrate``) — with
    torque OFF, by hand.
    """
    return _family(arm_type).supports_auto_calibration


def calibration_kind(arm_type: object) -> str:
    """How this arm type is calibrated: one of ``arms.base.CALIBRATION_KINDS``.

    ``range_sweep`` — the SO-101's Feetech sweep (``calibrate.py``, manual,
    and ``auto_calibrate.py``, driven); ``steps`` — the family's own
    procedure run by the generic step wizard (``step_calibrate.py``: the CAN
    arms' zero pose); ``panel`` — an extension's own page, mounted by the
    config dialog. Not a boolean on purpose: it picks the manager in
    sessions.py, and a third kind fits a name where a flag could not.
    """
    return _family(arm_type).calibration_kind


def supports_dagger(arm_type: object) -> bool:
    """True when this arm type can run a DAgger / smooth-handover rollout.

    Always False for the CAN arms, and it is a HARDWARE limit, not a policy
    choice. DAgger hands control back and forth between the policy and a human
    on the leader, which requires the leader to be back-driven to the
    follower's pose when the policy has control. The Star Arm 102 leader that
    drives a Maker or Metal arm has no motors in its joints at all — they are
    encoders only. There is no actuator to drive it with, so a handover would
    silently read a stale human pose. (The gravity-compensated metal_leader
    COULD back-drive, but MakerMods Lab does not integrate it yet — this
    becomes a per-leader question, not a per-arm-type one, if it ever does.)

    MakerMods Lab does not expose any rollout strategy other than ``base``
    (see rollout.py's ``--strategy.type=base``), so nothing consults this
    today. It exists so that if a strategy picker is ever added, the constraint
    is a value to read rather than a fact somebody has to rediscover from the
    hardware; ``tests/test_arm_capabilities.py`` pins both halves.
    """
    return _family(arm_type).supports_dagger


def arm_type_of_robot_config(robot_config: object) -> str:
    """The arm type a built lerobot robot config describes.

    For flows that are handed an assembled ``RobotConfig`` rather than the
    original request (recording's ``record_with_web_events`` takes a
    ``RecordConfig``), this reads the arm type back off the config instead of
    threading a parallel parameter that could drift out of agreement with it.
    Matched on the config's REGISTERED type string rather than by isinstance
    so this module never imports the device classes (which would drag the
    python-can / motorbridge stack into every import of it).
    """
    return _registry.family_for_robot_config_type(getattr(robot_config, "type", None)).id


class _ArmTypeLabels(Mapping[str, str]):
    """Human-readable name per arm type, for prose a user reads (merge /
    fine-tune / replay compatibility warnings). Not localized — the backend
    never is (see frontend/docs/localization.md). A live view of the registry
    rather than a dict captured at import, so a family registered later (an
    extension's) has a label the moment ``arm_type_from_robot_type`` can
    return its id."""

    def __getitem__(self, arm_type: str) -> str:
        return _registry.get(arm_type).indefinite_label

    def __iter__(self) -> Iterator[str]:
        return iter(_registry.ids())

    def __len__(self) -> int:
        return len(_registry.ids())


ARM_TYPE_LABEL: Mapping[str, str] = _ArmTypeLabels()


def _marker_scan_order():
    """Families in registry order with the default family LAST: its markers
    are the loosest (``so_follower``, ``so_leader``), so a string naming a
    specific family must get that family."""
    families = _registry.families()
    return [f for f in families if f.id != DEFAULT_ARM_TYPE] + [
        f for f in families if f.id == DEFAULT_ARM_TYPE
    ]


def arm_type_from_robot_type(robot_type: object) -> str | None:
    """Best-effort arm type for a dataset's ``meta/info.json`` ``robot_type``.

    lerobot writes the recording robot's ``.name`` there — ``so101_follower``,
    ``bi_maker_follower``, ``metal_follower`` — but a dataset recorded outside
    this app (or imported from the Hub) can carry anything: ``so100``,
    ``so-101``, ``aloha``, a custom string, or nothing at all. Each family
    declares the substrings that identify it (``robot_type_markers``); a
    marker anywhere in the string wins, deliberately greedily.

    Returns ``None`` — NOT the ``so101`` default ``arm_type_of_robot_config``
    falls back to — when the string is missing, non-string or unrecognized.
    The callers here are the merge / fine-tune compatibility warnings, which
    must stay silent when an arm can't be established rather than raise a false
    alarm about a community dataset tagged ``so100`` or an untagged one.
    """
    if not isinstance(robot_type, str):
        return None
    text = robot_type.strip().lower()
    if not text:
        return None
    for family in _marker_scan_order():
        for marker in family.robot_type_markers:
            if marker in text:
                return family.id
    return None
