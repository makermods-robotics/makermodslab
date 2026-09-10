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

"""CAN follower transitions for remote hosting (Maker and Metal).

MIT setpoints are commands, not Feetech registers. Clear effort before
disabling so a later enable cannot resurrect an old position or velocity.
All operations run on the hosting worker, never on Portal's callback thread.
"""

from __future__ import annotations

import math

from .maker_rest_pose import MAKER_RETURN_SETTLE_DEG, maker_follower_arms, return_maker_to_pose
from .torque import de_energize_can_device

CAN_TRACKING_SPEED_DEG_S = 30.0
# The pinned RobStride update_motor_state allows only 3 ms for a reply.
# USB serial delivery and camera/Portal activity can exceed that deadline.
# This is a maximum wait, not a sleep: healthy replies return immediately.
CAN_FEEDBACK_TIMEOUT_S = 0.05


def checked_pose(values: dict, keys) -> dict[str, float]:
    """Require every commanded joint, including grippers, to be readable."""
    pose = {}
    for key in keys:
        value = values.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or non-finite joint position: {key}")
        pose[key] = float(value)
    if not pose:
        raise ValueError("The follower has no joint positions")
    return pose


def rest_pose(pose: dict[str, float]) -> dict[str, float]:
    """Bare keys for the shared return helper, preserving bimanual prefixes."""
    return {
        key.removesuffix(".pos"): value
        for key, value in pose.items()
        if key.removesuffix(".pos") not in ("gripper", "left_gripper", "right_gripper")
    }


def limit_action(previous: dict[str, float], action: dict, elapsed_s: float) -> dict[str, float]:
    """Limit target travel in degrees per second, including after packet gaps.

    A late packet gets at most 100 ms of travel; a network stall must not
    authorize a large jump. Device soft limits still apply in send_action.
    """
    target = checked_pose(action, previous)
    step = CAN_TRACKING_SPEED_DEG_S * min(0.1, max(0.0, elapsed_s))
    return {key: value + max(-step, min(step, target[key] - value)) for key, value in previous.items()}


def prepare(robot) -> None:
    # Network actions arrive irregularly. Metal's local velocity estimator
    # would turn that jitter (or a Home/Engage transition) into feedforward.
    for arm, _label in maker_follower_arms(robot):
        if hasattr(arm.config, "velocity_feedforward"):
            arm.config.velocity_feedforward = False


def get_observation(robot) -> dict:
    """Require fresh replies; the CAN drivers' batch reads reuse stale state."""
    for arm, _label in maker_follower_arms(robot):
        for motor in arm.bus.motors:
            query = getattr(arm.bus, "_query_status_via_clear_fault", None)
            if query is not None:
                # Same request/decoder as the pinned RobStride driver's
                # update_motor_state, with a USB-appropriate reply deadline.
                # Never substitute its cached pose for a missing response.
                fault, message = query(motor, timeout=CAN_FEEDBACK_TIMEOUT_S)
                if message is None:
                    raise ConnectionError(f"No fresh feedback from motor '{motor}' within 50 ms")
                if fault:
                    raise RuntimeError(f"Motor '{motor}' reported a fault during state update")
                arm.bus._decode_motor_state(message.data)
            else:  # Damiao: individual reads fail on missing replies; batch reads do not
                arm.bus.read("Present_Position", motor)
    # Use the device's observation for Maker's full-turn zero correction.
    observation = robot.get_observation()
    checked_pose(observation, robot.action_features)
    return observation


def read_pose(robot) -> dict[str, float]:
    """Accept the supported rest pose, but reject a materially wrong zero.

    Maker's limits deliberately back off the mechanical stops: folded zero
    is about 3 degrees outside shoulder_lift/elbow/gripper's command limits.
    Use the shared rest-arrival tolerance, then ease inward on engagement.
    """
    pose = checked_pose(get_observation(robot), robot.action_features)
    for key, (low, high) in joint_limits(robot).items():
        if key in pose and not low - MAKER_RETURN_SETTLE_DEG <= pose[key] <= high + MAKER_RETURN_SETTLE_DEG:
            raise ValueError(
                f"{key} at {pose[key]:.1f} degrees is outside its soft limits "
                f"({low:.1f} to {high:.1f}); check the arm's zero calibration"
            )
    return pose


def joint_limits(robot) -> dict[str, tuple[float, float]]:
    limits = {}
    bimanual = getattr(robot, "left_arm", None) is not None
    for index, (arm, _label) in enumerate(maker_follower_arms(robot)):
        prefix = ("left_" if index == 0 else "right_") if bimanual else ""
        limits.update({f"{prefix}{motor}.pos": bounds for motor, bounds in arm.config.joint_limits.items()})
    return limits


def clamp_action(robot, action: dict) -> dict[str, float]:
    """Clamp the destination BEFORE interpolation from an outside rest pose."""
    pose = checked_pose(action, robot.action_features)
    for key, (low, high) in joint_limits(robot).items():
        if key in pose:
            pose[key] = min(high, max(low, pose[key]))
    return pose


def send_action(robot, action: dict[str, float], previous: dict[str, float]):
    """Let an inward ramp start at the actual resting position.

    The native send_action normally clips even the FIRST setpoint to the
    soft limit, skipping our rate limit if the arm starts just outside it.
    For this call only, include the previous ramp point in the permitted
    interval. Destinations remain clamped to the original limits, so this
    cannot authorize travel farther outward. Restore even on a write error.
    """
    saved = []
    bimanual = getattr(robot, "left_arm", None) is not None
    try:
        for index, (arm, _label) in enumerate(maker_follower_arms(robot)):
            prefix = ("left_" if index == 0 else "right_") if bimanual else ""
            limits = arm.config.joint_limits
            expanded = dict(limits)
            for motor, (low, high) in limits.items():
                position = previous[f"{prefix}{motor}.pos"]
                if not low - MAKER_RETURN_SETTLE_DEG <= position <= high + MAKER_RETURN_SETTLE_DEG:
                    raise ValueError(f"Cannot ease {prefix}{motor} from outside the resting tolerance")
                expanded[motor] = (min(low, position), max(high, position))
            saved.append((arm.config, limits))
            arm.config.joint_limits = expanded
        return robot.send_action(action)
    finally:
        for config, limits in saved:
            config.joint_limits = limits


def neutralize(robot) -> None:
    """Send zero-effort MIT frames, retaining configured gains for next engage.

    Do this BEFORE torque-off as a disabled motor may ignore new setpoints.
    Restoring Kp/Kd only changes the bus's local cache; it sends no MIT frame.
    """
    for arm, _label in maker_follower_arms(robot):
        bus = arm.bus
        gains = arm._resolved_gains
        zeros = dict.fromkeys(bus.motors, 0.0)
        try:
            bus.sync_write("Kp", zeros)
            bus.sync_write("Kd", zeros)
            bus.sync_write("Goal_Position", zeros)
        finally:
            bus.sync_write("Kp", {motor: kp for motor, (kp, _kd) in gains.items()})
            bus.sync_write("Kd", {motor: kd for motor, (_kp, kd) in gains.items()})


def engage(robot) -> dict[str, float]:
    # Validate first, while limp. Park has already replaced the retained MIT
    # command with zero effort. Enable, refresh again, then hold THIS pose.
    read_pose(robot)
    neutralize(robot)
    for arm, _label in maker_follower_arms(robot):
        arm.bus.enable_torque()
    pose = read_pose(robot)
    send_action(robot, pose, pose)
    return pose


class _ReturningFollower:
    """Keep the shared rest-return helper on fresh CAN feedback, too."""

    def __init__(self, robot):
        self.robot = robot

    def get_observation(self):
        return get_observation(self.robot)

    def send_action(self, action):
        return self.robot.send_action(action)


def return_to_rest(robot, pose, abort_event):
    return return_maker_to_pose(_ReturningFollower(robot), pose, abort_event=abort_event)


def release(robot) -> str | None:
    """Release even a failed Damiao handshake, then close each opened camera."""
    problems = []
    try:
        if all(arm.bus.is_connected for arm, _label in maker_follower_arms(robot)):
            neutralize(robot)
    except Exception as exc:
        problems.append(f"Could not clear the follower's retained motor command: {exc}")
    problems += de_energize_can_device(robot, "hosted follower arm")
    for name, camera in (getattr(robot, "cameras", None) or {}).items():
        if not camera.is_connected:
            continue
        try:
            camera.disconnect()
        except Exception as exc:
            problems.append(f"Could not release follower camera {name}: {exc}")
    # Bimanual followers own an I/O pool whose guarded disconnect cannot run
    # once the individual buses have been closed.
    pool = getattr(robot, "_io_pool", None)
    if pool is not None:
        try:
            pool.shutdown(wait=True)
        except Exception as exc:
            problems.append(f"Could not close the follower's I/O pool: {exc}")
    return " ".join(problems) or None
