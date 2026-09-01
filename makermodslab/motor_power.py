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
"""Follower motor-register hygiene at session start, plus torque helpers.

The per-robot "motor power" percentage (10-100) is the AUTO-CALIBRATION drive
torque: it is threaded into the vendored autocal subprocess as its
``--torque-limit`` (percent × 10; see makermodslab/auto_calibrate.py). Regular
sessions — teleoperation, recording, policy runs — deliberately run at stock
LeRobot torque instead: lerobot never writes ``Torque_Limit``, so "stock"
means the register's power-on value.

Register facts (pinned lerobot, lerobot/motors/feetech/tables.py):

- ``Torque_Limit`` (address 48, 2 bytes) sits in the SRAM section of
  ``STS_SMS_SERIES_CONTROL_TABLE``. It scales output torque 0-1000
  (0.1% units, so percent × 10) and, being RAM, re-seeds from its EEPROM
  twin on every power cycle — but SURVIVES between sessions on the same
  power-up, so an auto-calibration's working torque would silently throttle
  the next session unless reset (that's ``reset_torque_limit`` below).
- ``Max_Torque_Limit`` (address 16, 2 bytes) is the persistent EEPROM twin
  and the power-on source of ``Torque_Limit``; lerobot's
  ``SOFollower.configure()`` writes it to 500 for the gripper only. We NEVER
  write it here — this project has been burned by persistent EEPROM state
  (see makermodslab/wiggle.py).

Only ever touch FOLLOWER arms. The leader is human-held with torque
disabled; limiting it does nothing useful and risks confusing state.
"""

import asyncio
import logging

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .utils.config import clamp_motor_power

logger = logging.getLogger(__name__)

# RAM register scaled 0-1000 = 0-100% of max torque (see module docstring).
_TORQUE_LIMIT_REGISTER = "Torque_Limit"
_TORQUE_LIMIT_PER_PERCENT = 10

# "Goal_Velocity" (address 46, 2 bytes, RAM section of the STS3215 control
# table) is the profile-speed CAP in position mode: 0 means "uncapped, run at
# the servo default", any positive value throttles every Goal_Position move to
# that speed. Like Torque_Limit it lives in RAM, so it RESETS TO 0 on power
# cycle — but survives across sessions on the same power-up.
#
# This is a leftover-state hazard. Any feature that drives the follower with a
# capped speed stamps a nonzero value that the NEXT session inherits:
#   - auto-calibration's fold/unfold moves run at DEFAULT_POS_SPEED=1000, and
#     its graceful-stop freeze reuses that value;
#   - makermodslab/rest_pose.py's return-to-rest writes a gentle 400.
# Neither lerobot's configure() nor any MakerMods Lab start path resets Goal_Velocity,
# so the last arm-driving feature's cap silently throttles the next
# teleop/record/inference session (bench-confirmed: all six follower motors
# read Goal_Velocity=1000 after an auto-cal day; teleop tracked sluggishly
# until it was cleared to 0). We clear it to 0 at every session start, right
# where reset_torque_limit runs, so a stale cap can't outlive the power-up.
_GOAL_VELOCITY_REGISTER = "Goal_Velocity"

# "Present_Voltage" (address 62, 1 byte, read-only in the STS3215 table) is the
# measured servo-bus supply voltage in 0.1 V units. It is a REAL reading shown
# alongside the power slider — it is NOT what the Torque_Limit percentage
# controls (that's a torque fraction), so the two are labelled separately.
_PRESENT_VOLTAGE_REGISTER = "Present_Voltage"
_PRESENT_VOLTAGE_SCALE = 0.1
_VOLTAGE_TIMEOUT_S = 10.0


def torque_limit_from_percent(percent: object) -> int:
    """Register value for a motor-power percentage (clamped to 10-100)."""
    return clamp_motor_power(percent) * _TORQUE_LIMIT_PER_PERCENT


def _device_buses(device) -> list:
    """The motor bus(es) of a robot device.

    A single-arm device exposes ``.bus``; a bimanual BiSO device exposes
    ``left_arm``/``right_arm`` sub-arms which each carry their own bus.
    (Deliberately mirrors ``teleoperate._device_buses`` rather than importing
    it — teleoperate imports this module, so importing back would cycle.)
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


def _for_each_motor(device, action, on_fail_message, on_success_message=None) -> list[str]:
    """Apply ``action`` to every motor of every bus on ``device``, tolerating
    per-motor failures.

    For each bus: run ``action(bus, motor)`` on each motor, collecting
    "<motor>: <exc>" for the ones that raise. When any failed, build a warning
    via ``on_fail_message(port, failed)``, log it at WARNING, and add it to the
    returned list; otherwise (and only when supplied) log
    ``on_success_message(port)`` at INFO. ``port`` is the bus's ``.port`` or
    "unknown port". Never raises — this is the shared, failure-tolerant
    scaffold behind reset_torque_limit / clear_goal_velocity. Returns the
    accumulated warning messages (empty when every motor succeeded).
    """
    warnings: list[str] = []
    for bus in _device_buses(device):
        failed: list[str] = []
        for motor in getattr(bus, "motors", None) or {}:
            try:
                action(bus, motor)
            except Exception as e:
                failed.append(f"{motor}: {e}")
        port = getattr(bus, "port", None) or "unknown port"
        if failed:
            message = on_fail_message(port, failed)
            logger.warning(message)
            warnings.append(message)
        elif on_success_message is not None:
            logger.info(on_success_message(port))
    return warnings


def reset_torque_limit(device, label: str = "follower arm") -> list[str]:
    """Restore stock torque on every motor of a FOLLOWER device.

    Sessions (teleop, recording, policy runs) run at LeRobot-default torque —
    the robot's torque slider only sets AUTO-CALIBRATION's drive torque. But
    ``Torque_Limit`` is RAM: a lower value written by a previous
    auto-calibration survives until a power cycle, so simply not writing
    would silently inherit it. Re-seed the RAM register from each motor's
    persistent ``Max_Torque_Limit`` — exactly the value the servo boots
    with — so the session behaves as if freshly power-cycled (stock lerobot).

    Call after the device is connected and configured: lerobot's configure()
    stamps the gripper's Max_Torque_Limit (500) first, so the value read here
    matches what the next power-up would load.

    Never raises: a failed read/write is logged as a warning and the motor is
    left at whatever limit it had — a degraded but safe outcome that must not
    abort the session start. Returns the warning messages so callers can
    surface them to the user.
    """

    def _reset(bus, motor):
        stock = bus.read("Max_Torque_Limit", motor, normalize=False)
        bus.write(_TORQUE_LIMIT_REGISTER, motor, stock, normalize=False, num_retry=2)

    def _fail(port, failed):
        # Wording note: never say "calibration" here — tests (and log triage)
        # tell benign motor-register warnings apart from arm-identity/
        # calibration-mismatch warnings by that word.
        return (
            f"Could not restore stock torque (Torque_Limit) on {port} "
            f"({label}; failed motors — {'; '.join(failed)}). "
            "Those motors keep whatever torque cap was previously set, for this session."
        )

    return _for_each_motor(
        device,
        _reset,
        _fail,
        lambda port: f"Stock torque restored (Torque_Limit = Max_Torque_Limit) on {port} ({label})",
    )


def clear_goal_velocity(device, label: str = "follower arm") -> list[str]:
    """Reset the RAM speed cap (Goal_Velocity=0) on every motor of a FOLLOWER device.

    Call at session start, alongside reset_torque_limit (same post-configure
    point, same buses). A previous arm-driving feature — auto-calibration's
    fold/unfold at 1000, the rest-pose return at 400 — leaves a nonzero
    Goal_Velocity stamped in RAM that this session would otherwise inherit,
    throttling every follower move (see module-level _GOAL_VELOCITY_REGISTER
    note). Clearing to 0 restores the servo's uncapped default speed.

    NEVER call this on the leader: in teleop the leader is human-held with
    torque disabled, so its motion registers are read-only and irrelevant.

    Never raises: mirrors reset_torque_limit's failure tolerance — a failed
    write is logged as a warning and the motor keeps whatever cap it had (a
    degraded but safe outcome that must not abort the session start). Returns
    the warning messages so callers can surface them.
    """

    def _fail(port, failed):
        return (
            f"Could not clear the speed cap (Goal_Velocity) on {port} "
            f"({label}; failed motors — {'; '.join(failed)}). "
            "Those motors keep any leftover speed cap from a previous session for this run."
        )

    return _for_each_motor(
        device,
        lambda bus, motor: bus.write(_GOAL_VELOCITY_REGISTER, motor, 0, normalize=False, num_retry=2),
        _fail,
        lambda port: f"Speed cap cleared (Goal_Velocity=0) on {port} ({label})",
    )


def voltage_from_raw(raw: object) -> float:
    """Convert a raw Present_Voltage register value (0.1 V units) to volts."""
    return round(float(raw) * _PRESENT_VOLTAGE_SCALE, 1)


def _read_voltage_sync(port: str) -> float:
    """Connect to the arm on `port`, read the supply voltage, and release the port.

    Reads Present_Voltage from the gripper (motor id 6 — present on every SO-101
    arm; the supply rail is shared by all motors on the bus). Read-only: torque
    is never enabled, so disconnect skips the torque-disable write and just
    closes the port. Blocking; run in a worker thread. Mirrors the one-shot
    connect/act/disconnect pattern of makermodslab/wiggle.py so the serial port stays
    free for calibration/teleoperation between reads.
    """
    bus = FeetechMotorsBus(
        port=port,
        motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    bus.connect()
    try:
        raw = bus.read(_PRESENT_VOLTAGE_REGISTER, "gripper", normalize=False)
        return voltage_from_raw(raw)
    finally:
        bus.disconnect(disable_torque=False)


async def read_supply_voltage(port: str) -> dict:
    """One-shot supply-voltage read with a timeout. Returns a result dict
    ({"success": bool, "voltage": float} or {"success": False, "message": str});
    logical failures (port busy, arm off) are reported, not raised, so the
    endpoint stays HTTP 200 like the rest of the feature handlers."""
    if not port or not port.strip():
        return {"success": False, "message": "No port provided."}
    try:
        voltage = await asyncio.wait_for(
            asyncio.to_thread(_read_voltage_sync, port.strip()),
            timeout=_VOLTAGE_TIMEOUT_S,
        )
        return {"success": True, "voltage": voltage}
    except TimeoutError:
        return {
            "success": False,
            "message": "Voltage read timed out — is the arm powered on and the port correct?",
        }
    except Exception as e:
        logger.exception("Voltage read failed")
        return {"success": False, "message": f"Failed to read voltage: {e}"}
