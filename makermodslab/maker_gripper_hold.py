"""RobStride 00 holding for local Maker followers using standard-frame MIT.

Requires RS00 firmware with MIT parameter access (0.0.3.27+) and packed
operating-status feedback. Protocol: RS00 manual, sections 6.1 and 6.17-18.
No fault clears, zero changes, protocol switches, or flash writes.
"""

import math
import struct
import threading
import time

import can

from .gripper_settings import GRIPPER_RESTART_C
from .metal_gripper import GripperSafetyError, _active, _registry_lock
from .metal_gripper_hold import HoldingController, MetalHoldingBus

QUERY_TIMEOUT_S = 0.12
CAN_TIMEOUT_INDEX = 0x7028
WATCHDOG_TICKS = 10_000  # RS00: 20,000 ticks/s, 500 ms.


class MakerHoldingBus(MetalHoldingBus):
    """Share regulation/serialization, replacing every Damiao wire operation."""

    def __init__(self, arm, robot_name, hold_torque_nm):
        from lerobot.motors.robstride.tables import MOTOR_LIMIT_PARAMS, MotorType

        super().__init__(
            arm.bus,
            robot_name,
            hold_torque_nm,
            arm.config.joint_limits["gripper"],
            gains=arm.config.gains["gripper"],
        )
        if self._base._motor_types["gripper"] != MotorType.O0:
            raise GripperSafetyError("Maker holding requires a RobStride 00 gripper")
        if MOTOR_LIMIT_PARAMS[MotorType.O0] != (12.57, 33, 14):
            raise GripperSafetyError("Unexpected RobStride 00 MIT scaling")
        self._arm = arm
        self._reply_id = 0xFD  # Standard MIT host ID; payload byte 0 identifies the motor.
        self._feedback_motor_id = self._base._get_motor_recv_id("gripper")
        self.controller = HoldingController(
            hold_torque_nm,
            tuple(map(math.radians, self.joint_limits)),
            closing_direction=1,
        )

    def _query(self, arbitration_id, data, matches):
        channel = self._base.canbus
        if channel is None:
            raise ConnectionError("Gripper CAN bus is closed")
        # Drain BEFORE sending: queued state/parameter replies cannot acknowledge
        # a new command. Bound the drain so a noisy bus fails closed.
        for _ in range(256):
            if channel.recv(timeout=0) is None:
                break
        else:
            raise GripperSafetyError("Gripper CAN receive queue did not drain")
        reply_id = arbitration_id if arbitration_id & 0x700 in (0x300, 0x400) else self._reply_id
        channel.send(can.Message(arbitration_id=arbitration_id, is_extended_id=False, data=data))
        deadline = time.monotonic() + QUERY_TIMEOUT_S
        while time.monotonic() < deadline:
            msg = channel.recv(timeout=0.005)
            if (
                msg is not None
                and not msg.is_extended_id
                and not msg.is_error_frame
                and not msg.is_remote_frame
                and msg.arbitration_id == reply_id
                and len(msg.data) == 8
                and matches(bytes(msg.data))
            ):
                return msg
        raise ConnectionError("No fresh RobStride gripper response; grip stopped")

    def _param(self, index, value=None, fmt="I"):
        header = struct.pack("<H", index) + bytes(2)
        payload = bytes(4) if value is None else struct.pack("<" + fmt, value)
        reply = self._query(
            self._motor_id | (0x300 if value is None else 0x400),
            header + payload,
            lambda d: d[:4] == header,
        )
        result = struct.unpack("<" + fmt, reply.data[4:])[0]
        if value is not None and (result != value or self._param(index, fmt=fmt) != value):
            raise GripperSafetyError(f"RobStride parameter 0x{index:X} write did not verify")
        return result

    def _is_state(self, data):
        return data[0] == self._feedback_motor_id

    def _state(self, msg):
        data = bytearray(msg.data)
        # Fault-query replies have bytes 5..7 zero, unlike a valid stationary
        # MIT state (whose encoded velocity is around 2047).
        if data[5:] == bytes(3):
            raise GripperSafetyError("Unexpected RobStride fault-status frame")
        mode = data[6] >> 6
        if data[6] & 0x30:
            raise GripperSafetyError("RobStride gripper reports a motor fault or warning")
        self._status = {0: 0, 2: 1}.get(mode, 8)
        # Firmware packs mode/fault flags above the 12-bit temperature. The
        # pinned driver predates those flags, so strip them before decoding.
        data[6] &= 0x0F
        self._base._decode_motor_state(data)
        self.temperature_c = float(self._base._last_known_states["gripper"]["temp_mos"])
        self._last_feedback = time.monotonic()

    def _simple(self, command):
        self._state(self._query(self._motor_id, bytes([255] * 7 + [command]), self._is_state))

    def _refresh(self):
        # Repeat the last bounded target to get state without the upstream
        # driver's CLEAR_FAULT poll. Never replay the leader's full closure.
        target = self.controller.last_command
        goal = self._goal if target is None else math.degrees(target)
        self._send_mit(self.kp if self._enabled else 0.0, self.kd if self._enabled else 0.0, goal, 0.0, 0.0)

    def connect(self, handshake=True):
        with self._lock:
            self._base.connect(handshake=False)
            if not handshake:
                return  # Failure cleanup opens only to disable.
            try:
                self._simple(0xFD)
                if self._status != 0:
                    raise GripperSafetyError("RobStride gripper stop was not confirmed")
                self._check()
                if self.temperature_c >= GRIPPER_RESTART_C:
                    raise GripperSafetyError("Let the gripper cool below 45 C before starting")
                # These are RobStride parameter indices, never Damiao RIDs.
                if self._param(0x7005) != 0:
                    raise GripperSafetyError(
                        "Set the RobStride gripper to MIT operation mode before starting"
                    )
                self._original_timeout = self._param(CAN_TIMEOUT_INDEX)
                self._goal = float(self._base._last_known_states["gripper"]["position"])
                self._prepared = True
            except Exception:
                self.disconnect()
                raise
            with _registry_lock:
                _active.setdefault(self.robot_name, []).append(self)

    def enable_torque(self, motors=None, num_retry=0):
        with self._lock:
            if "gripper" in self._base._get_motors_list(motors) and not self._enabled:
                self._check()
                if not self._prepared:
                    raise GripperSafetyError("Gripper setup did not complete")
                if "gripper" in self._arm._stale_zero:
                    raise GripperSafetyError("Recalibrate the Maker gripper before holding")
                # Maker detects whole-turn offsets after bus.connect(), and
                # sends raw motor goals. Shift bounds into that same frame.
                offset = self._arm._turn_offset["gripper"]
                self.joint_limits = tuple(x - offset for x in self._arm.config.joint_limits["gripper"])
                self.controller.limits = tuple(map(math.radians, self.joint_limits))
                try:
                    self._param(CAN_TIMEOUT_INDEX, WATCHDOG_TICKS)
                    self._refresh()
                    self._goal = float(self._base._last_known_states["gripper"]["position"])
                    self.controller = HoldingController(
                        self.controller.torque_nm,
                        self.controller.limits,
                        closing_direction=1,
                    )
                except Exception as exc:
                    self._trip(exc)
                    raise
            return super().enable_torque(motors, num_retry=num_retry)

    def read(self, data_name, motor):
        with self._lock:
            self._check()
            if motor != "gripper":
                return self._base.read(data_name, motor)
            try:
                self._refresh()
                return self._base._get_cached_value(motor, data_name)
            except Exception as exc:
                self._trip(exc)
                raise

    def sync_read(self, data_name, motors=None):
        with self._lock:
            return {m: self.read(data_name, m) for m in self._base._get_motors_list(motors)}

    def sync_read_all_states(self, motors=None, *, num_retry=0):
        with self._lock:
            names = self._base._get_motors_list(motors)
            for motor in names:
                self.read("Present_Position", motor)
            return {m: self._base._last_known_states[m].copy() for m in names}

    def sync_write(self, data_name, values):
        with self._lock:
            if "gripper" in values and data_name != "Goal_Position":
                expected = {"Kp": self.kp, "Kd": self.kd}
                if data_name not in expected or values["gripper"] != expected[data_name]:
                    exc = GripperSafetyError("Stop the session before changing gripper gains or control mode")
                    self._trip(exc)
                    raise exc
            return super().sync_write(data_name, values)

    def write(self, data_name, motor, value):
        return self.sync_write(data_name, {motor: value})

    def sync_write_metal(self, commands):
        # Preserve the shared MIT command API, with RobStride batching for the
        # other joints (the Maker follower normally uses individual writes).
        with self._lock:
            super().sync_write_metal({k: v for k, v in commands.items() if k == "gripper"})
            other = {k: v for k, v in commands.items() if k != "gripper"}
            if other:
                self._base._mit_control_batch(other)

    def disconnect(self, disable_torque=True):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._lock:
            errors = []
            try:
                if self._base.canbus is not None:
                    if disable_torque:
                        for motor in self._base.motors:
                            if motor != "gripper":
                                try:
                                    self._base.disable_torque(motor)
                                except Exception as exc:
                                    errors.append(str(exc))
                    try:
                        self.disable_torque("gripper")
                        if self._status != 0:
                            raise GripperSafetyError("RobStride gripper stop was not confirmed")
                        # Never disarm the watchdog unless release is confirmed.
                        if self._original_timeout is not None:
                            self._param(CAN_TIMEOUT_INDEX, self._original_timeout)
                    except Exception as exc:
                        errors.append(str(exc))
            finally:
                if self._base.is_connected:
                    self._base.disconnect(disable_torque=False)
                self._prepared = False
                with _registry_lock:
                    group = _active.get(self.robot_name, [])
                    if self in group:
                        group.remove(self)
                    if not group:
                        _active.pop(self.robot_name, None)
            if errors:
                raise GripperSafetyError("Gripper cleanup could not be verified: " + "; ".join(errors))
