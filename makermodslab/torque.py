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
"""Shared motor-torque release helper.

An SO-101 arm whose servos keep torque enabled stays rigid (and warm) until
its power is pulled, so every feature that drives an arm needs a
belt-and-braces "disable torque, motor by motor, and be loud on failure"
cleanup step. Auto-calibration uses this as a fallback after its subprocess
dies. (Deliberately mirrors the per-bus loop of
``teleoperate.force_disable_torque`` rather than refactoring it — same
pattern as ``motor_power._device_buses``.)
"""

import logging

logger = logging.getLogger(__name__)


def force_disable_bus_torque(bus, label: str = "device") -> list[str]:
    """Explicitly disable torque on every motor of one bus, motor by motor.

    lerobot's ``disconnect()`` does disable torque itself, but any exception on
    the way there leaves the arm energized: one motor's failed write aborts the
    disable for all remaining motors (and skips closing the port), and the
    error is easy to swallow on a cleanup path. Going motor by motor means one
    bad motor can't leave the other joints locked.

    Returns a list of problem descriptions — empty when torque was disabled on
    every motor. Each problem is also logged at ERROR level naming the port.
    """
    failed: list[str] = []
    for motor in getattr(bus, "motors", None) or {}:
        try:
            bus.disable_torque(motor, num_retry=5)
        except Exception as e:
            failed.append(f"{motor}: {e}")
    if not failed:
        return []
    port = getattr(bus, "port", None) or "unknown port"
    message = (
        f"TORQUE MAY STILL BE ENABLED on {port} ({label}; failed motors — {'; '.join(failed)}). "
        "The arm can stay rigid; unplug its power to release it."
    )
    logger.error(message)
    return [message]


def release_maker_torque(device, label: str = "device") -> list[str]:
    """Disable torque on a Maker device before disconnect, loudly on failure.

    The Feetech path above cannot be reused: it walks each bus motor-by-motor
    through Feetech register writes and a Dynamixel-style ``port_handler``,
    neither of which a RobStride CAN bus has. RobStride exposes one
    whole-bus ``disable_torque()`` instead, so that is what this calls.

    The Star Arm 102 leader is skipped, and that is not an oversight: its
    joints contain encoders and no motors, so it has no torque to disable and
    its FashionStar bus handle has no ``disable_torque`` to call. That same
    fact is why a Maker arm can never run a DAgger handover — see
    ``arm_capabilities.supports_dagger``.

    Returns a list of problem descriptions — empty when every bus released.
    """
    problems: list[str] = []
    for bus in _maker_device_buses(device):
        disable = getattr(bus, "disable_torque", None)
        if disable is None:
            continue  # the leader's FashionStar bus — nothing to release
        try:
            disable()
        except Exception as e:
            port = getattr(bus, "port", None) or "unknown port"
            message = (
                f"Failed to disable torque on the {label} ({port}): {e}. "
                "TORQUE MAY STILL BE ENABLED — the arm can stay rigid; unplug its power to release it."
            )
            logger.error(message)
            problems.append(message)
    return problems


def _maker_device_buses(device) -> list:
    """The motor bus(es) of a device: ``.bus``, or both sub-arms' when bimanual.

    A local copy of ``teleoperate._device_buses`` for the same reason the loop
    above is a local copy of ``force_disable_torque`` — this module is imported
    by teleoperate, so reaching back into it would close an import cycle.
    """
    if device is None:
        return []
    arms = [
        arm
        for arm in (getattr(device, "left_arm", None), getattr(device, "right_arm", None))
        if arm is not None
    ]
    targets = arms if arms else [device]
    return [target.bus for target in targets if getattr(target, "bus", None) is not None]
