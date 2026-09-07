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
"""Shared motor-torque release helper.

An SO-101 arm whose servos keep torque enabled stays rigid (and warm) until
its power is pulled, so every feature that drives an arm needs a
belt-and-braces "disable torque, motor by motor, and be loud on failure"
cleanup step. Auto-calibration uses this as a fallback after its subprocess
dies. (Deliberately mirrors the per-bus loop of
``teleoperate.force_disable_torque`` rather than refactoring it — same
pattern as ``motor_power._device_buses``.)
"""

import contextlib
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


def device_buses(device) -> list:
    """The motor bus(es) of a robot/teleop device — the one shared copy.

    A single-arm device exposes ``.bus``; a bimanual BiSO device exposes
    ``left_arm``/``right_arm`` sub-arms which each carry their own bus.
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


# Historical name; the CAN helpers above were written when this module could
# not import teleoperate's copy without a cycle. One copy now, here.
_maker_device_buses = device_buses


def bus_has_a_responding_motor(bus) -> bool:
    """True when at least one motor on ``bus`` answers a ping.

    Used only to choose the *wording* of the alarm after the torque-disable
    pass below has already failed on every motor on this bus — it never gates
    whether that pass runs. A degraded-but-recoverable bus (a brownout that
    recovers, EMI, a long cable, the moments right after the control loop died
    on a comms error) can fail every zero-retry ping while a retried write
    would have landed, so the probe must not be allowed to veto the write.

    ``is_connected`` can't stand in for this: it's literally
    ``port_handler.is_open`` (lerobot ``motors_bus.py``), and
    ``MotorsBus._connect`` calls ``openPort()`` *before* ``_handshake()`` and
    does not close the port when the handshake fails — so an unpowered,
    browned-out, or wrong-baud arm leaves ``is_connected`` True on a bus that
    no motor is listening on.

    ``ping`` is the right probe: it is a read (so it works whatever the torque
    state is) and it returns ``None`` rather than raising on comm failure.
    Short-circuits on the first motor that answers.

    Defaults to True (assume the bus is alive, keep the loud alarm) for
    anything that can't be probed: a bus with no ``motors``, a bus without a
    ``ping`` method, or a test double that models neither.
    """
    motors = getattr(bus, "motors", None) or {}
    ping = getattr(bus, "ping", None)
    if not motors or not callable(ping):
        return True
    for motor in motors:
        try:
            if ping(motor) is not None:
                return True
        except Exception:
            continue
    return False


def force_disable_torque(device, label: str = "device") -> list[str]:
    """Explicitly disable torque on every motor of a device, motor by motor.

    Belt-and-braces step to run *before* ``device.disconnect()``. lerobot's
    disconnect does disable torque itself, but any exception on the way there
    leaves the arm energized: one motor's failed write aborts the disable for
    all remaining motors (and skips closing the port), and the error is easy
    to swallow on a cleanup path. Going motor by motor means one bad motor
    can't leave the other joints locked.

    Returns a list of problem descriptions — empty when torque was disabled on
    every motor. Each problem is also logged at ERROR level naming the port.
    """
    problems: list[str] = []
    for bus in device_buses(device):
        # A bus whose port never opened at all (the device node was missing or
        # busy, so openPort() itself raised) has nothing to disable — writing
        # to a closed port just produces a false "TORQUE MAY STILL BE ENABLED"
        # alarm. Note this covers *only* that case: is_connected is
        # port_handler.is_open, so it stays True when the port opened and the
        # handshake then failed. The unpowered/wrong-baud arm is caught by the
        # liveness probe below, not here. Test doubles that don't model
        # connection state default to True so torque-only tests are unaffected.
        if not getattr(bus, "is_connected", True):
            continue
        # The Dynamixel SDK port handler can be left flagged "in use" after a
        # failed read/write in the control loop. LeRobot's normal
        # bus.disconnect() clears this before disabling torque; mirror that here
        # because this belt-and-braces path deliberately bypasses disconnect()
        # to disable motors one by one.
        port_handler = getattr(bus, "port_handler", None)
        if port_handler is not None:
            with contextlib.suppress(Exception):
                port_handler.clearPort()
            with contextlib.suppress(Exception):
                port_handler.is_using = False
        port = getattr(bus, "port", None) or "unknown port"
        # Always attempt the disable, whatever the liveness probe below would
        # say: a degraded-but-recoverable bus can fail every zero-retry ping
        # while the retried write here still lands, and skipping the write on
        # a failed probe would leave a genuinely energized arm rigid. The
        # probe is only consulted after a failure, to pick the wording.
        failed: list[str] = []
        for motor in getattr(bus, "motors", None) or {}:
            try:
                bus.disable_torque(motor, num_retry=5)
            except Exception as e:
                failed.append(f"{motor}: {e}")
        if failed:
            if bus_has_a_responding_motor(bus):
                # Something is listening, so the failed writes are a real
                # "this joint may stay rigid" condition.
                message = (
                    f"TORQUE MAY STILL BE ENABLED on {port} ({label}; failed motors — {'; '.join(failed)}). "
                    "The arm can stay rigid; unplug its power to release it."
                )
                logger.error(message)
            else:
                # Nothing on this bus is listening: the port opened but no
                # motor answers (arm unpowered, browned-out servos, wrong
                # baud, or a valid-but-wrong serial device). Reporting that as
                # "TORQUE MAY STILL BE ENABLED — unplug its power" is actively
                # misleading on an arm that has no power. Say what was
                # actually observed, and keep the rigid-arm advice as a
                # conditional rather than an assertion.
                message = (
                    f"No motor answered on {port} ({label}) — the torque disable failed on every motor. "
                    "Check the arm's power and USB cable. If the arm is rigid, unplug its power to release it."
                )
                logger.warning(message)
            problems.append(message)
    return problems


def de_energize_can_bus(bus, label: str = "device") -> list[str]:
    """Free a CAN arm that may be energized behind a dead-looking bus object.

    Two situations land here that ``release_maker_torque`` cannot reach,
    both specific to how CAN buses fail:

    * a **partial Damiao handshake** — the handshake frame IS the motor
      enable command, sent motor by motor, so a handshake that raises on the
      Nth motor has already energized the first N-1 while ``is_connected``
      still reads False (nothing sets it on the failure path);
    * a **previous process killed hard** (SIGKILL, power loss) — the Damiao
      motors hold their last MIT command indefinitely, and the new process's
      bus object naturally starts disconnected.

    The recovery is to (re)open WITHOUT the handshake — reopening with it
    would re-energize the very motors being freed — then broadcast the
    disable, then close with ``disable_torque=False`` (the disable already
    ran; re-running it inside disconnect would just re-raise on a bad motor
    after the port is half-closed).

    Never raises; returns problem descriptions, empty on success. Loud on a
    failed disable — same alarm contract as force_disable_bus_torque.
    """
    problems: list[str] = []
    port = getattr(bus, "port", None) or "unknown port"
    if not getattr(bus, "is_connected", False):
        try:
            bus.connect(handshake=False)
        except Exception as e:
            message = (
                f"Could not reopen the CAN bus on {port} ({label}) to release it: {e}. "
                "If the arm is rigid, unplug its power to release it."
            )
            logger.warning(message)
            return [message]
    try:
        bus.disable_torque()
    except Exception as e:
        message = (
            f"Failed to disable torque on {port} ({label}): {e}. "
            "TORQUE MAY STILL BE ENABLED — the arm can stay rigid; unplug its power to release it."
        )
        logger.error(message)
        problems.append(message)
    try:
        bus.disconnect(disable_torque=False)
    except Exception as e:
        message = f"Could not close the CAN bus on {port} ({label}): {e}"
        logger.warning(message)
        problems.append(message)
    return problems


def de_energize_can_device(device, label: str = "device") -> list[str]:
    """``de_energize_can_bus`` over every CAN bus of a device.

    Walks single and bimanual devices alike (same shape as
    ``_maker_device_buses``) and skips any bus without ``disable_torque`` —
    the Star leader's FashionStar handle, whose joints hold encoders and no
    motors.
    """
    problems: list[str] = []
    for bus in _maker_device_buses(device):
        if getattr(bus, "disable_torque", None) is None:
            continue
        problems += de_energize_can_bus(bus, label)
    return problems
