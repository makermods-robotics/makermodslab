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
"""Metal's optional closing-position-error limit for local teleop/record.

This bounds the commanded spring displacement, NOT motor current, torque or
jaw force. Existing damping, inertia and external loads remain. The pinned
Damiao adapter has no guaranteed sample-age API: we explicitly refresh and
require a recent host-receipt timestamp, rather than accepting its stale cache.
One extra gripper round trip is required per action; no hardware timing claim.
"""

import math
import time

from .utils.config import get_robot_record, validate_gripper_closing_error


class _ActionBus:
    """Per-follower adapter; cleanup and other direct bus users pass through."""

    def __init__(self, bus, closing_error, limits):
        self._bus = bus
        self._closing_error = closing_error
        self._limits = limits
        self.active = False
        self.sent_gripper = None

    def __getattr__(self, name):
        return getattr(self._bus, name)

    def _target(self, requested):
        if not math.isfinite(requested):
            raise ValueError("Nonfinite gripper target")
        # read()/sync_read() can reuse stale cache, including decode failures.
        # Decode the actual response and propagate errors instead. These private
        # calls intentionally match the SHA-pinned Damiao dependency.
        msg = self._bus._refresh_motor("gripper")
        if msg is None:
            raise ConnectionError("Gripper soft limit: no feedback response")
        timestamp = getattr(msg, "timestamp", None)
        now = time.time()
        if timestamp is None or not math.isfinite(timestamp) or not -0.01 <= now - timestamp <= 0.1:
            raise ConnectionError("Gripper soft limit: missing or stale feedback timestamp")
        present, *_ = self._bus._decode_motor_state(msg.data, self._bus._motor_types["gripper"])
        if not math.isfinite(present):
            raise ConnectionError("Gripper soft limit: nonfinite feedback")
        # Metal preset: zero is shut; increasing degrees opens. Recompute from
        # measured jaws every action, never from a previous unreachable target.
        target = max(requested, present - self._closing_error)
        if not self._limits[0] <= target <= self._limits[1]:
            raise ValueError("Gripper soft limit conflicts with joint range; check measured position")
        self.sent_gripper = target
        return target

    def sync_write(self, data_name, values, *args, **kwargs):
        if self.active and data_name == "Goal_Position" and "gripper" in values:
            values = dict(values)
            values["gripper"] = self._target(values["gripper"])
        return self._bus.sync_write(data_name, values, *args, **kwargs)

    def sync_write_metal(self, commands, *args, **kwargs):
        if self.active and "gripper" in commands:
            commands = dict(commands)
            kp, kd, position, _velocity, _torque = commands["gripper"]
            # Gripper feedforward could add closing effort independent of the
            # position error. Keep damping, but use zero velocity/torque target.
            commands["gripper"] = (kp, kd, self._target(position), 0.0, 0.0)
        return self._bus.sync_write_metal(commands, *args, **kwargs)


def install_gripper_soft_limit(robot, family, closing_error):
    """Install before connect; stock single/bimanual send_action stays in charge.

    Every local action, including record reset, runs through this final bus
    boundary AFTER stock startup/range handling. On invalid feedback the entire
    affected arm batch raises before writing; the session's normal error cleanup
    releases hardware. Bimanual buses remain independently scheduled.
    """
    validate_gripper_closing_error(closing_error)
    if closing_error is None:
        return
    if not family.supports_gripper_soft_limit:
        raise ValueError("Gripper soft limit is only supported for Metal followers")
    arms = [robot.left_arm, robot.right_arm] if hasattr(robot, "left_arm") else [robot]
    for arm in arms:
        if isinstance(arm.bus, _ActionBus):
            raise RuntimeError("Gripper soft limit already installed")
        bus = _ActionBus(arm.bus, closing_error, arm.config.joint_limits["gripper"])
        original_send = arm.send_action

        def send_action(action, _bus=bus, _send=original_send):
            if "gripper.pos" in action and not math.isfinite(action["gripper.pos"]):
                raise ValueError("Nonfinite gripper target")
            _bus.active = True
            _bus.sent_gripper = None
            try:
                sent = _send(action)
                if _bus.sent_gripper is not None:
                    sent = {**sent, "gripper.pos": _bus.sent_gripper}
                return sent
            finally:
                _bus.active = False

        arm.bus = bus
        arm.send_action = send_action


def resolve_request_gripper_soft_limit(request):
    """Legacy named starts inherit the saved setting when the field is omitted."""
    from .api_errors import ApiError, ErrorCode
    from .arms import registry

    value = request.gripper_closing_error_deg
    if "gripper_closing_error_deg" not in request.model_fields_set and request.robot_name:
        record = get_robot_record(request.robot_name)
        if record is not None:
            value = record.get("gripper_closing_error_deg")
    try:
        value = validate_gripper_closing_error(value)
        if value is not None and not registry.get(request.arm_type).supports_gripper_soft_limit:
            raise ValueError("Gripper soft limit is only supported for Metal followers")
    except ValueError as exc:
        raise ApiError(status_code=400, detail=str(exc), code=ErrorCode.REQUEST_VALIDATION) from exc
    request.gripper_closing_error_deg = value


def gripper_action_columns(robot, features):
    """Validate named dataset columns before creating recording artifacts."""
    arms = (
        {"left_gripper.pos": robot.left_arm, "right_gripper.pos": robot.right_arm}
        if hasattr(robot, "left_arm")
        else {"gripper.pos": robot}
    )
    names = features["action"]["names"]
    return [(names.index(name), arm) for name, arm in arms.items()]


def record_limited_gripper_actions(dataset, robot):
    """The pinned record loop otherwise records the pre-send leader target.

    Replace only limited gripper columns at the dataset boundary; keep the
    existing action convention for all arm joints and never mutate the caller's
    frame. This is local to this dataset instance, including reset/episode reuse.
    """
    columns = gripper_action_columns(robot, dataset.features)
    original_add = dataset.add_frame

    def add_frame(frame, *args, **kwargs):
        action = frame["action"].copy()
        for index, arm in columns:
            sent = arm.bus.sent_gripper
            if sent is None:
                raise RuntimeError("No successfully sent gripper action for recording frame")
            action[index] = sent
        return original_add({**frame, "action": action}, *args, **kwargs)

    dataset.add_frame = add_frame
