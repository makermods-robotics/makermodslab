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
"""Experimental DM4310 V7018 force-position commands, local followers only.

Protocol: dmBots/motor-sdk Python example u2can/DM_CAN.py, control_pos_force,
__read_RID_param, __write_motor_param and __control_cmd; manufacturer V1.4
protocol feedback status and firmware revision notes. Independently implemented
against python-can; no dependency patch and no flash-save command.

Each active mode-4 command requests normalized current saturation (I/Imax).
This is not a jaw-force guarantee. Firmware behavior between enable and the
first command is undocumented: startup current transients are NOT bounded by
this implementation. Never preload a command while disabled and assume latching.
"""

import math
import struct
import time
from contextlib import contextmanager

from .utils.config import validate_gripper_current_limit


class MetalGripperBus:
    """Own mode transitions and every gripper write, including direct cleanup.

    Other motors retain the pinned bus's MIT implementations. An ordinary
    Metal session checks the gripper is in MIT before the energizing handshake,
    so a process killed in mode 4 cannot silently start in an incompatible mode.
    """

    def __init__(self, bus, ratio=None, velocity=None, limits=(0.0, 137.5)):
        self._bus = bus
        self.ratio, self.velocity = validate_gripper_current_limit(ratio, velocity)
        self.limits = limits
        self.sent_gripper = None
        self._original_mode = None
        self._mode_changed = False
        self._ready = False
        self._enabled = False

    def __getattr__(self, name):
        return getattr(self._bus, name)

    def _send(self, can_id, data):
        import can

        if self._bus.canbus is None:
            raise ConnectionError("Gripper CAN bus is not open")
        self._bus.canbus.send(
            can.Message(arbitration_id=can_id, data=data, is_extended_id=False, is_fd=False)
        )

    def _drain(self):
        # Bound even a noisy bus; never loop indefinitely before a control action.
        for _ in range(128):
            if self._bus.canbus.recv(timeout=0) is None:
                return
        raise ConnectionError("Gripper CAN receive queue did not drain")

    def _exchange(self, can_id, data, matches):
        self._drain()
        started = time.time()
        self._send(can_id, data)
        deadline = time.monotonic() + 0.1
        for _ in range(256):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            msg = self._bus.canbus.recv(timeout=min(remaining, 0.01))
            if msg is None:
                continue
            stamp = getattr(msg, "timestamp", None)
            if stamp is None or not math.isfinite(stamp) or not started - 0.01 <= stamp <= time.time() + 0.01:
                continue
            if (
                msg.arbitration_id == self._bus._get_motor_recv_id("gripper")
                and len(msg.data) == 8
                and matches(msg.data)
            ):
                return msg
        raise ConnectionError("Gripper current-limit transaction timed out or returned invalid feedback")

    def _parameter(self, rid, value=None):
        motor_id = self._bus._get_motor_id("gripper")
        op = 0x33 if value is None else 0x55
        prefix = struct.pack("<HBB", motor_id, op, rid)
        msg = self._exchange(
            0x7FF,
            prefix + (bytes(4) if value is None else struct.pack("<I", value)),
            lambda data: bytes(data[:4]) == prefix,
        )
        return bytes(msg.data[4:])

    def _mode(self):
        return struct.unpack("<I", self._parameter(10))[0]

    def _set_mode(self, mode):
        self._parameter(10, mode)
        if self._mode() != mode:
            raise ConnectionError("Gripper control mode readback mismatch")

    def _state_matches(self, data):
        motor_id = self._bus._get_motor_id("gripper")
        # Parameter echoes use the same feedback CAN ID and begin with slave
        # ID low/high. They must NEVER count as disabled state feedback.
        if bytes(data[:2]) == struct.pack("<H", motor_id) and data[2] in (0x33, 0x55):
            return False
        return data[0] & 0x0F == motor_id & 0x0F

    def _state(self, msg, expected):
        status = msg.data[0] >> 4
        if status != expected:
            raise ConnectionError(f"Gripper status {status} does not confirm expected state {expected}")
        position, *_ = self._bus._decode_motor_state(msg.data, self._bus._motor_types["gripper"])
        if not math.isfinite(position):
            raise ConnectionError("Gripper returned nonfinite position")
        self._bus._process_response("gripper", msg)
        return position

    def _command(self, command, expected):
        msg = self._exchange(
            self._bus._get_motor_id("gripper"), bytes([0xFF] * 7 + [command]), self._state_matches
        )
        position = self._state(msg, expected)
        self._enabled = expected == 1
        return position

    def _verify_profile(self):
        if self._bus._motor_types["gripper"].name != "METAL_JHI":
            raise ValueError("Experimental current limit supports only the Metal DM4310 gripper")
        if self._bus._get_motor_id("gripper") != 7 or self._bus._get_motor_recv_id("gripper") != 0x17:
            raise ValueError("Experimental gripper current limit requires Metal J7 IDs 7/0x17")
        if self._parameter(14) != b"7018":
            raise ValueError("Experimental gripper current limit requires verified V7018 firmware")
        # These are serialization scales, NOT a current cap. Read, never write.
        for rid, expected in ((21, 6.28), (22, 30.0), (23, 20.0)):
            actual = struct.unpack("<f", self._parameter(rid))[0]
            if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=0.001):
                raise ValueError("Gripper feedback scales differ from the supported Metal DM4310 profile")

    def connect(self, handshake=True):
        self._ready = False
        self._enabled = False
        self._bus.connect(handshake=False)
        if not handshake:
            return  # Recovery open must never enable or change mode.
        if self.ratio is None:
            if self._mode() != 1:
                self._command(0xFD, 0)
                raise ValueError(
                    "Metal gripper is not in MIT mode; release torque and restore its mode before starting"
                )
            self._bus._handshake()
            return
        try:
            self._command(0xFD, 0)
            self._verify_profile()
            self._original_mode = self._mode()
            if self._original_mode not in (1, 4):
                raise ValueError("Unsupported original gripper control mode")
            # Mark before write: a failed echo may still mean the mode changed.
            self._mode_changed = self._original_mode != 4
            self._set_mode(4)
            self._ready = True
            self.enable_torque()
        except Exception:
            self._ready = False
            # Leave the open bus available to the caller's normal CAN recovery.
            # No arm enable has occurred before the gripper preflight succeeds.
            raise

    def _hold_and_enable(self):
        # Read fresh jaws while disabled. Manufacturer requires enable BEFORE
        # control: the short enable-to-first-frame interval remains unverified.
        position = self._command(0xFD, 0)
        if not self.limits[0] <= position <= self.limits[1]:
            raise ValueError("Gripper initial position is outside configured joint limits")
        self._drain()
        self._send(self._bus._get_motor_id("gripper"), bytes([0xFF] * 7 + [0xFC]))
        self._enabled = True
        # Do not wait for an enable round trip before sending the hold. The
        # force-position reply below must confirm enabled status.
        self._force_position(position)

    def enable_torque(self, motors=None, num_retry=0):
        names = self._bus._get_motors_list(motors)
        if self.ratio is None:
            return self._bus.enable_torque(motors, num_retry=num_retry)
        if "gripper" in names:
            if not self._ready or self._mode() != 4:
                raise ConnectionError("Gripper current-limit mode is not armed")
            if not self._enabled:
                self._hold_and_enable()
        others = [name for name in names if name != "gripper"]
        for motor in others:
            # The vendor enable_torque silently tolerates dropped replies;
            # connect historically used a handshake which rejects missing motors.
            self._drain()
            self._send(self._bus._get_motor_id(motor), bytes([0xFF] * 7 + [0xFC]))
            reply = self._bus._recv_motor_response(
                expected_recv_id=self._bus._get_motor_recv_id(motor), timeout=0.1
            )
            if reply is None or len(reply.data) != 8 or reply.data[0] >> 4 != 1:
                raise ConnectionError(f"Metal follower motor {motor} did not confirm enable")
            self._bus._process_response(motor, reply)

    def disable_torque(self, motors=None, num_retry=0):
        names = self._bus._get_motors_list(motors)
        failures = []
        if "gripper" in names:
            try:
                self._command(0xFD, 0)
            except Exception as exc:
                failures.append(exc)
            self._enabled = False
        others = [name for name in names if name != "gripper"]
        if others:
            try:
                self._bus.disable_torque(others, num_retry=num_retry)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ConnectionError(f"Could not confirm motor disable: {failures}")

    def disconnect(self, disable_torque=True):
        error = None
        try:
            # Even vendor disconnect(False) cannot leave this experimental
            # gripper enabled while restoring an unrestricted original mode.
            if self.ratio is not None or self._mode_changed:
                self.disable_torque()
                if self._mode_changed:
                    self._set_mode(self._original_mode)
                    self._mode_changed = False
            elif disable_torque:
                self.disable_torque()
        except Exception as exc:
            error = exc
        finally:
            self._ready = False
            self._enabled = False
            self._bus.disconnect(disable_torque=False)
        if error is not None:
            raise error

    def _force_position(self, position):
        if not self._ready or not self._enabled:
            raise ConnectionError("Gripper force-position command refused before mode setup/enable")
        if not math.isfinite(position) or not self.limits[0] <= position <= self.limits[1]:
            raise ValueError("Invalid gripper position")
        # Round DOWN, so quantization cannot increase the requested ceiling.
        velocity_units = int(math.radians(self.velocity) * 100)
        current_units = int(self.ratio * 10000)
        if not 1 <= velocity_units <= 10000 or not 1 <= current_units <= 10000:
            raise ValueError("Gripper velocity/current is not representable")
        payload = struct.pack("<fHH", math.radians(position), velocity_units, current_units)
        try:
            msg = self._exchange(0x300 + self._bus._get_motor_id("gripper"), payload, self._state_matches)
            self._state(msg, 1)
        except Exception:
            self._ready = False
            raise
        self.sent_gripper = position

    def sync_write(self, data_name, values, *args, **kwargs):
        values = {self._bus._get_motor_name(k): v for k, v in values.items()}
        if self.ratio is not None and data_name == "Goal_Position" and "gripper" in values:
            self._force_position(values["gripper"])
            values = {k: v for k, v in values.items() if k != "gripper"}
        if values:
            return self._bus.sync_write(data_name, values, *args, **kwargs)

    def sync_write_metal(self, commands, *args, **kwargs):
        commands = {self._bus._get_motor_name(k): v for k, v in commands.items()}
        if self.ratio is not None and "gripper" in commands:
            self._force_position(commands["gripper"][2])
            commands = {k: v for k, v in commands.items() if k != "gripper"}
        if commands:
            return self._bus.sync_write_metal(commands, *args, **kwargs)

    def write(self, data_name, motor, value):
        name = self._bus._get_motor_name(motor)
        if self.ratio is not None and name == "gripper" and data_name == "Goal_Position":
            return self._force_position(value)
        return self._bus.write(data_name, motor, value)

    def _mit_control(self, motor, kp, kd, position_degrees, velocity_deg_per_sec, torque):
        if self.ratio is not None and self._bus._get_motor_name(motor) == "gripper":
            return self._force_position(position_degrees)
        return self._bus._mit_control(motor, kp, kd, position_degrees, velocity_deg_per_sec, torque)

    def _mit_control_batch(self, commands):
        return self.sync_write_metal(commands)

    def configure_motors(self):
        return self.enable_torque()

    @contextmanager
    def torque_disabled(self, motors=None):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def _send_simple_command(self, motor, command):
        if self.ratio is not None and self._bus._get_motor_name(motor) == "gripper":
            if command == 0xFC:
                return self.enable_torque("gripper")
            if command == 0xFD:
                return self.disable_torque("gripper")
            raise ValueError(
                "Gripper zero/calibration commands are not supported during current-limited sessions"
            )
        return self._bus._send_simple_command(motor, command)

    def set_zero_position(self, motors=None):
        if self.ratio is not None and "gripper" in self._bus._get_motors_list(motors):
            raise ValueError("Cannot zero the gripper during a current-limited session")
        return self._bus.set_zero_position(motors)
