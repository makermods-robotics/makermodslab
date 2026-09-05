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
"""What each arm type can and cannot do.

Every feature module branches on arm type somewhere: which preflight guards
apply, which calibration flow to run, how many joints a checkpoint should
have. Those predicates live here, once, rather than as `arm_type == "maker"`
literals scattered across teleoperate/record/rollout/replay — a scattered
check is a check somebody forgets to add to the next flow.

The three arm types:

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

Import this instead of writing the comparison inline.
"""

from .utils.config import normalize_arm_type

# Flat proprioceptive width of ONE follower arm — one dim per joint. The SO-101
# has 6; the CAN arms have 7 (6 joints plus a permanent gripper). This is the
# number a bimanual robot doubles, and the number a trained checkpoint's
# observation.state must match.
_JOINTS_PER_ARM = {"so101": 6, "maker": 7, "metal": 7}


def joints_per_arm(arm_type: object) -> int:
    """Joint count of a single follower arm of this type."""
    return _JOINTS_PER_ARM[normalize_arm_type(arm_type)]


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
    return normalize_arm_type(arm_type) == "so101"


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
    is, which is what ``zero_calibrate`` does — with torque OFF, by hand.
    """
    return normalize_arm_type(arm_type) == "so101"


def uses_zero_calibration(arm_type: object) -> bool:
    """True when calibrating this arm type means setting a zero pose.

    The exact complement of ``supports_auto_calibration`` today, but they are
    not the same question and need not stay complementary as arm types
    arrive — keep them separate.
    """
    return normalize_arm_type(arm_type) in ("maker", "metal")


def ships_urdf(arm_type: object) -> bool:
    """True when a URDF for this arm type ships in ``frontend/public/``.

    Two things read it, and they must agree:

    * ``teleoperate.py`` — an arm type that ships a URDF broadcasts live joint
      angles in radians under ``joints`` (keyed by URDF joint name) for the 3D
      viewer to drive. One that does not broadcasts ``joints_deg`` (raw degrees
      by motor name) for the numeric readout instead.
    * the frontend — ``ships_urdf`` decides the teleop panel shows ``UrdfViewer``
      rather than ``JointAngleReadout`` (mirrored in ``lib/armTypes.ts``).

    The SO-101 (``frontend/public/so-101-urdf``) and the Maker arm
    (``frontend/public/maker-urdf``) each ship one. The Metal arm does not yet —
    same 7-DOF class as the Maker follower but a different geometry, so the
    Maker model cannot stand in for it — so a Metal session stays on the
    readout.
    """
    return normalize_arm_type(arm_type) in ("so101", "maker")


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
    return normalize_arm_type(arm_type) == "so101"


# lerobot `RobotConfig` choice-registry keys, mapped to the arm type they
# describe. Kept as REGISTERED type strings rather than an isinstance check
# so this module never has to import the device classes (which would drag the
# python-can / motorbridge stack into every import of it).
_ROBOT_TYPE_TO_ARM_TYPE = {
    "maker_follower": "maker",
    "bi_maker_follower": "maker",
    "metal_follower": "metal",
    "bi_metal_follower": "metal",
}


def arm_type_of_robot_config(robot_config: object) -> str:
    """The arm type a built lerobot robot config describes.

    For flows that are handed an assembled ``RobotConfig`` rather than the
    original request (recording's ``record_with_web_events`` takes a
    ``RecordConfig``), this reads the arm type back off the config instead of
    threading a parallel parameter that could drift out of agreement with it.
    """
    return _ROBOT_TYPE_TO_ARM_TYPE.get(getattr(robot_config, "type", None), "so101")


# Human-readable name per arm type, for prose a user reads (merge/fine-tune
# compatibility warnings). Not localized — the backend never is (see
# frontend/docs/localization.md).
ARM_TYPE_LABEL = {"so101": "an SO-101 arm", "maker": "a Maker arm", "metal": "a Metal arm"}

# Substrings that identify an arm family inside a dataset's free-form
# ``robot_type`` string. "maker"/"metal" are unambiguous; the SO family is
# every string carrying an ``so100``/``so101`` marker or the bare
# ``so_follower``/``so_leader`` device names lerobot writes for a bimanual SO
# rig (``bi_so_follower``).
_ROBOT_TYPE_STRING_MARKERS = (
    ("maker", "maker"),
    ("metal", "metal"),
    ("so100", "so101"),
    ("so101", "so101"),
    ("so-100", "so101"),
    ("so-101", "so101"),
    ("so_follower", "so101"),
    ("so_leader", "so101"),
)


def arm_type_from_robot_type(robot_type: object) -> str | None:
    """Best-effort arm type for a dataset's ``meta/info.json`` ``robot_type``.

    lerobot writes the recording robot's ``.name`` there — ``so101_follower``,
    ``bi_maker_follower``, ``metal_follower`` — but a dataset recorded outside
    this app (or imported from the Hub) can carry anything: ``so100``,
    ``so-101``, ``aloha``, a custom string, or nothing at all.

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
    for marker, arm_type in _ROBOT_TYPE_STRING_MARKERS:
        if marker in text:
            return arm_type
    return None
