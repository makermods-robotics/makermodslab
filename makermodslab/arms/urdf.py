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
"""Pure CAN-arm telemetry conversions; values are radians or metres for the viewer."""

import math
from bisect import bisect_left

# Maker arm → its shipped URDF (`frontend/public/maker-urdf/`, vendored from
# the makermods-robotics/maker-arm-sdk release). Unlike the SO-101 path, this
# is a DIRECT degrees→radians map, not an affine range→range remap: a Maker
# follower's observation is already the true joint angle in degrees about the
# folded/gripper-open calibration zero (the SO-101 needs the remap only because
# a Feetech-normalized value is not a physical angle without calibration).
#
# The URDF joints are `link_002_joint`..`link_007_joint` (base→wrist), one per
# Maker motor. `sign` is whether motor-increasing maps to URDF-increasing;
# `offset` (radians) covers a URDF zero pose that differs from the arm's
# calibration zero. Both can only be pinned against real hardware — neutral
# here. The viewer ignores the URDF's own joint limits for this model
# (`urdfConfigs.ts` `ignoreLimits`), so a raw angle never freezes a joint
# short of the real one while sign/offset are still being validated.
_MAKER_URDF_JOINTS: dict[str, tuple[str, int, float]] = {
    # motor_name: (urdf_joint, sign, offset_rad)   # HARDWARE: confirm sign/offset
    "shoulder_pan": ("link_002_joint", +1, 0.0),
    "shoulder_lift": ("link_003_joint", +1, 0.0),
    "elbow_flex": ("link_004_joint", +1, 0.0),
    "wrist_flex": ("link_005_joint", +1, 0.0),
    "wrist_yaw": ("link_006_joint", +1, 0.0),
    "wrist_roll": ("link_007_joint", +1, 0.0),
}

# The URDF's gripper is a symmetric sliding jaw: `gripper_left_joint`
# (prismatic, metres) drives it and `gripper_right_joint` mimics it. The Maker
# `gripper` motor reports an angle (degrees, like the other joints); map it
# onto the jaw's 0..TRAVEL metres from the two endpoints the SDK's
# `revision_report.json` records (`motor_calibration`: closed ≈ +0.0067 rad,
# commanded-open ≈ −2.079 rad). The SDK calls this a visual-preview
# interpolation, not a calibrated transmission.
_MAKER_URDF_GRIPPER_JOINT = "gripper_left_joint"
_MAKER_GRIPPER_CLOSED_RAD = 0.0067132066834521
_MAKER_GRIPPER_OPEN_RAD = -2.078984206912338
_MAKER_GRIPPER_JAW_TRAVEL_M = 0.0524125  # HARDWARE: confirm the motor→gap curve


def _maker_gripper_joint_metres(motor_deg: float) -> float:
    """The Maker gripper motor angle (degrees) as `gripper_left_joint` travel
    in metres, clamped to the jaw's real 0..TRAVEL range."""
    motor_rad = math.radians(motor_deg)
    span = _MAKER_GRIPPER_CLOSED_RAD - _MAKER_GRIPPER_OPEN_RAD
    frac = (_MAKER_GRIPPER_CLOSED_RAD - motor_rad) / span if span else 0.0
    frac = min(1.0, max(0.0, frac))
    return frac * _MAKER_GRIPPER_JAW_TRAVEL_M


def maker_joint_positions(degrees: dict[str, float]) -> dict[str, float]:
    joints = {
        joint: math.radians(degrees[motor]) * sign + offset
        for motor, (joint, sign, offset) in _MAKER_URDF_JOINTS.items()
        if motor in degrees
    }
    if "gripper" in degrees:
        joints[_MAKER_URDF_GRIPPER_JOINT] = _maker_gripper_joint_metres(degrees["gripper"])
    return joints


# metal-python-ros at ef4181f: native/can_manager.h's measured motor-angle
# table for total gripper opening (mm). MIT, Copyright (c) 2025 MakerMods;
# the complete notice is in frontend/public/metal-urdf/LICENSE.
# Interpolate between measured samples for a smooth display; clamp to the
# URDF's 100 mm opening (the vendor table extends to 102.5 mm).
_METAL_GRIPPER_ANGLES_RAD = (
    0.002,
    0.01407,
    0.0368934,
    0.08634,
    0.115854,
    0.161084,
    0.176609,
    0.194816,
    0.22893,
    0.262277,
    0.295,
    0.314215,
    0.344305,
    0.362704,
    0.393177,
    0.411575,
    0.441473,
    0.46773,
    0.482871,
    0.508743,
    0.527718,
    0.550333,
    0.576397,
    0.60227,
    0.621053,
    0.639452,
    0.662833,
    0.683915,
    0.707105,
    0.722054,
    0.748693,
    0.763451,
    0.782233,
    0.805039,
    0.815964,
    0.841837,
    0.856786,
    0.879593,
    0.894158,
    0.91639,
    0.92789,
    0.94648,
    0.969287,
    0.987494,
    0.998418,
    1.02084,
    1.03617,
    1.05112,
    1.0701,
    1.08505,
    1.10383,
    1.11897,
    1.13411,
    1.15557,
    1.16382,
    1.18221,
    1.20117,
    1.21594,
    1.23779,
    1.24986,
    1.26424,
    1.28264,
    1.29452,
    1.31273,
    1.3317,
    1.34359,
    1.36505,
    1.38019,
    1.39169,
    1.40683,
    1.42542,
    1.44401,
    1.45609,
    1.47468,
    1.48943,
    1.50419,
    1.52259,
    1.53735,
    1.55575,
    1.57453,
    1.582,
    1.60462,
    1.62302,
    1.64218,
    1.65694,
    1.66825,
    1.68722,
    1.70984,
    1.729,
    1.73973,
    1.76216,
    1.78477,
    1.79991,
    1.82214,
    1.84399,
    1.86297,
    1.88577,
    1.9036,
    1.92717,
    1.94844,
    1.97527,
    2.03143,
)
_METAL_GRIPPER_STROKES_MM = (*range(101), 102.5)
_METAL_URDF_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)


def _metal_gripper_joint_metres(motor_deg: float) -> float:
    angle = math.radians(motor_deg)
    i = bisect_left(_METAL_GRIPPER_ANGLES_RAD, angle)
    if i == 0:
        stroke = 0.0
    elif i == len(_METAL_GRIPPER_ANGLES_RAD):
        stroke = _METAL_GRIPPER_STROKES_MM[-1]
    else:
        lo, hi = _METAL_GRIPPER_ANGLES_RAD[i - 1 : i + 1]
        start, end = _METAL_GRIPPER_STROKES_MM[i - 1 : i + 1]
        stroke = start + (end - start) * (angle - lo) / (hi - lo)
    # ROS controller: joint7 = -total opening mm / 2000; joint8 mimics -joint7.
    return -min(100.0, max(0.0, stroke)) / 2000.0


def metal_joint_positions(degrees: dict[str, float]) -> dict[str, float]:
    # The vendor ROS controller passes joint1..joint6 directly to the motors
    # in radians, with no sign or zero offset.
    joints = {
        f"joint{i}": math.radians(degrees[motor])
        for i, motor in enumerate(_METAL_URDF_MOTORS, start=1)
        if motor in degrees
    }
    if "gripper" in degrees:
        joints["joint7"] = _metal_gripper_joint_metres(degrees["gripper"])
    return joints
