"""Gripper-only Damiao force-position control for local teleop and recording.

Wrap one follower bus BEFORE connect. The remaining joints use the original
MIT driver. All bus transactions, including the independent thermal/holding
worker, share one lock. No flash writes. Failure is latched, never automatic
re-enabling. The motor watchdog covers a stopped process; the worker covers
recording pauses when the normal action loop is idle.
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

import can

from .gripper_settings import GRIPPER_RESTART_C, GRIPPER_STOP_C, validate_gripper_current

logger = logging.getLogger(__name__)
_registry_lock = threading.RLock()
_active: dict[str, list[MetalGripperBus]] = {}
WATCHDOG_TICKS = 10_000  # Damiao 50 us units: 500 ms.
HOLD_INTERVAL_S = 0.05
QUERY_TIMEOUT_S = 0.12


class GripperSafetyError(RuntimeError):
    """Latched gripper fault: the session must stop and be restarted deliberately."""


def encode_force_position(position_deg: float, speed_rad_s: float, current_a: float, imax_a: float) -> bytes:
    if not all(math.isfinite(v) for v in (position_deg, speed_rad_s, current_a, imax_a)):
        raise ValueError("Non-finite gripper command")
    if not 0 < imax_a <= 100 or not 0 <= current_a <= imax_a or not 0 <= speed_rad_s <= 100:
        raise ValueError("Invalid gripper current/speed scale")
    # Floor, never round upward past the requested ceiling.
    return struct.pack(
        "<fHH", math.radians(position_deg), int(speed_rad_s * 100), int(current_a / imax_a * 10_000)
    )


class MetalGripperBus:
    """A transaction-serialized adapter around the pinned LeRobot Damiao bus."""

    @property
    def control_enabled(self) -> bool:
        return self.limit_a is not None

    def _preload(self) -> None:
        self._force(self._goal, current_a=0.0)

    def __init__(
        self,
        bus: Any,
        robot_name: str,
        limit_a: float | None,
        joint_limits: tuple[float, float],
        *,
        speed_limit_deg_s: float,
    ):
        self._base = bus
        self.robot_name = robot_name
        self.limit_a = validate_gripper_current(limit_a)
        self.joint_limits = joint_limits
        # Mode 4 has a separate velocity ceiling. Use the follower's velocity
        # setting, not the deliberately slow 0.35 rad/s unloaded-test cap.
        self.speed_limit_rad_s = math.radians(speed_limit_deg_s)
        if not math.isfinite(self.speed_limit_rad_s) or not 0 < self.speed_limit_rad_s <= 100:
            raise ValueError("Invalid gripper speed ceiling")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._prepared = False
        self._enabled = False
        self._mode = 1
        self._original_mode: int | None = None
        self._original_timeout: int | None = None
        self.imax_a = 0.0
        self.temperature_c = 0.0
        self.effective_current_a = 0.0
        self._goal = 0.0
        self._status = 0
        self._last_feedback = 0.0
        motor = bus.motors["gripper"]
        self._motor_id = motor.id
        self._reply_id = motor.recv_id

    def __getattr__(self, name: str):
        target = getattr(self._base, name)
        if not callable(target):
            return target

        def locked(*args, **kwargs):
            with self._lock:
                if self._prepared:
                    self._check()
                return target(*args, **kwargs)

        return locked

    def _query(self, arbitration_id: int, data: bytes, matches: Callable[[bytes], bool]) -> can.Message:
        if self._base.canbus is None:
            raise ConnectionError("Gripper CAN bus is closed")
        for _ in range(2):
            self._base.canbus.send(
                can.Message(arbitration_id=arbitration_id, is_extended_id=False, data=data)
            )
            deadline = time.monotonic() + QUERY_TIMEOUT_S
            while time.monotonic() < deadline:
                msg = self._base.canbus.recv(timeout=0.01)
                if (
                    msg is not None
                    and not msg.is_extended_id
                    and msg.arbitration_id == self._reply_id
                    and len(msg.data) == 8
                    and matches(bytes(msg.data))
                ):
                    return msg
        raise ConnectionError("No fresh gripper response; grip stopped")

    def _param(self, rid: int, value: int | None = None, fmt: str = "I") -> float | int:
        op = 0x33 if value is None else 0x55
        header = struct.pack("<HBB", self._motor_id, op, rid)
        payload = bytes(4) if value is None else struct.pack("<" + fmt, value)
        reply = self._query(0x7FF, header + payload, lambda d: d[:4] == header)
        result = struct.unpack("<" + fmt, reply.data[4:])[0]
        if value is not None and (result != value or self._param(rid, fmt=fmt) != value):
            raise GripperSafetyError(f"Gripper register {rid} did not retain requested value {value}")
        return result

    def _state(self, msg: can.Message) -> None:
        self._base._process_response("gripper", msg)
        self._status = msg.data[0] >> 4
        self.temperature_c = float(max(msg.data[6], msg.data[7]))
        self._last_feedback = time.monotonic()

    def _is_state(self, data: bytes) -> bool:
        # Parameter acknowledgments share the feedback CAN ID. A delayed
        # parameter reply must never be decoded as fresh thermal feedback.
        return (
            data[0] & 15 == self._motor_id
            and data[0] >> 4 in (0, 1, 8, 9, 10, 11, 12, 13, 14)
            and not (data[1] == 0 and data[2] in (0x33, 0x55, 0xAA))
        )

    def _refresh(self) -> None:
        header = struct.pack("<HB", self._motor_id, 0xCC)
        msg = self._query(0x7FF, header + bytes(5), self._is_state)
        self._state(msg)

    def _simple(self, command: int) -> None:
        offset = {1: 0, 2: 0x100, 3: 0x200, 4: 0x300}[self._mode]
        msg = self._query(self._motor_id + offset, bytes([255] * 7 + [command]), self._is_state)
        self._state(msg)

    def _check(self) -> None:
        if self._error:
            raise GripperSafetyError(self._error)
        if self.temperature_c >= GRIPPER_STOP_C:
            raise GripperSafetyError(
                f"Gripper reached {self.temperature_c:.0f} C; grip released. Let it cool before restarting."
            )
        if self._status not in (0, 1):
            raise GripperSafetyError(f"Gripper motor fault 0x{self._status:X}; session must stop")
        if self._enabled and (self._status != 1 or time.monotonic() - self._last_feedback > 0.3):
            raise GripperSafetyError("Gripper disabled or feedback stale; session must stop")

    def _trip(self, exc: Exception) -> None:
        self._error = str(exc)
        self._enabled = False
        self.effective_current_a = 0.0
        self._stop.set()
        try:
            self._simple(0xFD)
        except Exception:
            logger.exception("Gripper disable could not be confirmed; motor watchdog remains armed")
        logger.error("Metal gripper stopped: %s", exc)

    def connect(self, handshake: bool = True) -> None:
        with self._lock:
            if not self.control_enabled:
                self._base.connect(handshake=handshake)
            else:
                self._base.connect(handshake=False)  # normal handshake sends enable!
                if not handshake:
                    # The shared CAN failure cleanup reopens buses with this
                    # flag solely to disable them. Do not configure a session.
                    self._mode = int(self._param(10))
                    return
                try:
                    self._mode = int(self._param(10))
                    self._original_mode = self._mode
                    if self._mode not in (1, 4):
                        raise GripperSafetyError("Gripper must start in MIT or force-position mode")
                    self._refresh()
                    if self._status != 0:
                        raise GripperSafetyError(
                            "Release the gripper before starting current-limited control"
                        )
                    if self.temperature_c >= GRIPPER_RESTART_C:
                        raise GripperSafetyError(
                            "Gripper is warm; let both sensors cool below 45 C before starting"
                        )
                    self.imax_a = float(self._param(59, fmt="f"))
                    if not math.isfinite(self.imax_a) or not self.limit_a <= self.imax_a <= 100:
                        raise GripperSafetyError("Invalid gripper maximum-current scale")
                    self._goal = float(self._base._last_known_states["gripper"]["position"])
                    self._original_timeout = int(self._param(9))
                    # Set before selecting/energizing mode 4. Never persist to flash.
                    self._param(9, WATCHDOG_TICKS)
                    self._param(10, 4)
                    self._mode = 4
                    self._prepared = True
                    self._preload()
                    # Read all other joints without enabling any of them.
                    if handshake:
                        for motor in self._base.motors:
                            if motor != "gripper":
                                self._base.read("Present_Position", motor)
                except Exception:
                    self.disconnect()
                    raise
            with _registry_lock:
                _active.setdefault(self.robot_name, []).append(self)

    def _force(self, goal: float, current_a: float | None = None) -> None:
        self._check()
        if not math.isfinite(goal):
            raise GripperSafetyError("Invalid gripper target")
        self._goal = min(self.joint_limits[1], max(self.joint_limits[0], goal))
        # Fixed ceiling at every operating temperature. Heat trips a latched
        # safety stop; it never silently changes the operator's grip strength.
        self.effective_current_a = self.limit_a if current_a is None else current_a
        data = encode_force_position(
            self._goal, self.speed_limit_rad_s, self.effective_current_a, self.imax_a
        )
        self._state(self._query(self._motor_id + 0x300, data, self._is_state))
        self._check()

    def _hold(self) -> None:
        while not self._stop.wait(HOLD_INTERVAL_S):
            with self._lock:
                if not self._enabled:
                    return
                try:
                    self._refresh()
                    self._force(self._goal)
                except Exception as exc:
                    self._trip(exc)
                    return

    def enable_torque(self, motors=None, num_retry: int = 0) -> None:
        with self._lock:
            if not self.control_enabled:
                return self._base.enable_torque(motors, num_retry=num_retry)
            self._check()
            for motor in self._base._get_motors_list(motors):
                if motor != "gripper":
                    self._base.enable_torque(motor, num_retry=num_retry)
                    continue
                if not self._prepared:
                    raise GripperSafetyError("Gripper setup did not complete")
                if self._enabled:
                    continue
                try:
                    self._preload()
                    self._simple(0xFC)
                    if self._status != 1:
                        raise GripperSafetyError("Gripper did not enable in the configured mode")
                    self._enabled = True
                    self._stop.clear()
                    self._thread = threading.Thread(target=self._hold, name="metal-gripper-hold", daemon=True)
                    self._thread.start()
                except Exception as exc:
                    self._trip(exc)
                    raise

    def configure_motors(self) -> None:
        self.enable_torque()

    def disable_torque(self, motors=None, num_retry: int = 0) -> None:
        with self._lock:
            if not self.control_enabled:
                return self._base.disable_torque(motors, num_retry=num_retry)
            errors = []
            for motor in self._base._get_motors_list(motors):
                try:
                    if motor == "gripper":
                        self._enabled = False
                        self._stop.set()
                        self._simple(0xFD)
                        if self._status == 1:
                            raise GripperSafetyError("Gripper disable was not confirmed")
                    else:
                        self._base.disable_torque(motor, num_retry=num_retry)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise GripperSafetyError("; ".join(errors))

    def disconnect(self, disable_torque: bool = True) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._lock:
            errors = []
            try:
                if self.control_enabled and self._base.canbus is not None:
                    # Attempt other-joint shutdown even if gripper parameter
                    # reads fail. Each device gets its own best-effort stop.
                    if disable_torque:
                        for motor in self._base.motors:
                            if motor != "gripper":
                                try:
                                    self._base.disable_torque(motor)
                                except Exception as exc:
                                    errors.append(str(exc))
                    # A mode write may have landed even if its reply was lost.
                    self._mode = int(self._param(10))
                    # Always release the gripper, even when the upstream config
                    # requests leaving arm-joint torque on at disconnect.
                    self.disable_torque("gripper")
                    if self._original_mode is not None:
                        self._param(10, self._original_mode)
                        self._mode = self._original_mode
                    if self._original_timeout is not None:
                        self._param(9, self._original_timeout)
                elif disable_torque and self._base.canbus is not None:
                    self._base.disable_torque()
            except Exception as exc:
                errors.append(str(exc))
            finally:
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

    def sync_write_metal(self, commands: dict) -> None:
        with self._lock:
            if not self.control_enabled:
                return self._base.sync_write_metal(commands)
            try:
                self._check()
                other = {k: v for k, v in commands.items() if k != "gripper"}
                if "gripper" in commands:
                    self._force(float(commands["gripper"][2]))
                if other:
                    self._base.sync_write_metal(other)
            except Exception as exc:
                self._trip(exc)
                raise

    def sync_write(self, data_name: str, values: dict) -> None:
        with self._lock:
            if not self.control_enabled or data_name != "Goal_Position":
                return self._base.sync_write(data_name, values)
            try:
                self._check()
                if "gripper" in values:
                    self._force(float(values["gripper"]))
                other = {k: v for k, v in values.items() if k != "gripper"}
                if other:
                    self._base.sync_write(data_name, other)
            except Exception as exc:
                self._trip(exc)
                raise

    def write(self, data_name: str, motor: str, value: Any) -> None:
        with self._lock:
            if self.control_enabled and motor == "gripper" and data_name == "Goal_Position":
                return self.sync_write(data_name, {motor: value})
            return self._base.write(data_name, motor, value)

    def apply_current(self, limit_a: float) -> None:
        with self._lock:
            self._check()
            if not self._prepared or self.limit_a is None:
                raise GripperSafetyError(
                    "Stop the session before enabling or disabling gripper current limiting"
                )
            old = self.limit_a
            self.limit_a = validate_gripper_current(limit_a)
            try:
                self._refresh()
                self._force(self._goal, current_a=None if self._enabled else 0.0)
            except Exception as exc:
                self.limit_a = old
                self._trip(exc)
                raise


def install_metal_gripper(robot: Any, robot_name: str) -> None:
    """Attach to Metal FOLLOWERS only; invoked before connect by both sessions."""
    from .utils.config import get_robot_record

    record = get_robot_record(robot_name) if robot_name else None
    limit = validate_gripper_current((record or {}).get("gripper_current_limit_a"))
    from .gripper_settings import DEFAULT_GRIPPER_HOLD_TORQUE_NM, validate_gripper_hold_torque
    from .metal_gripper_hold import MetalHoldingBus

    # Unnamed legacy teleop/record requests also use the Metal default. An
    # existing record's explicit null remains an opt-out.
    holding = validate_gripper_hold_torque(
        record.get("gripper_hold_torque_nm") if record is not None else DEFAULT_GRIPPER_HOLD_TORQUE_NM
    )
    if holding is not None and limit is not None:
        raise GripperSafetyError("Choose holding torque or current limiting, not both")
    arms = [getattr(robot, "left_arm", None), getattr(robot, "right_arm", None)]
    for arm in [a for a in arms if a is not None] or [robot]:
        if getattr(arm.config, "type", "") != "metal_follower":
            continue
        if isinstance(arm.bus, MetalGripperBus):
            raise RuntimeError("Metal gripper adapter is already installed")
        if holding is not None:
            arm.bus = MetalHoldingBus(
                arm.bus,
                robot_name,
                holding,
                arm.config.joint_limits["gripper"],
                gains=arm.config.gains["gripper"],
            )
        else:
            arm.bus = MetalGripperBus(
                arm.bus,
                robot_name,
                limit,
                arm.config.joint_limits["gripper"],
                speed_limit_deg_s=arm.config.velocity_ff_max_deg_s,
            )


def apply_live_hold_torque(robot_name: str, value: float | None) -> str:
    from .gripper_settings import validate_gripper_hold_torque
    from .metal_gripper_hold import MetalHoldingBus

    value = validate_gripper_hold_torque(value)
    with _registry_lock:
        buses = list(_active.get(robot_name, []))
    if not buses:
        from .sessions import tracker

        session = tracker.current()
        if session and session.get("robot") == robot_name:
            raise GripperSafetyError("Stop the current session before changing gripper settings")
        return "next_session"
    if value is None or any(not isinstance(b, MetalHoldingBus) for b in buses):
        raise GripperSafetyError("Stop the session before enabling, disabling or switching gripper control")
    try:
        for bus in buses:
            bus.apply_hold_torque(value)
    except Exception as exc:
        for bus in buses:
            with bus._lock:
                bus._trip(exc)
        raise
    return "live"


def apply_live_current(robot_name: str, limit_a: float | None) -> str:
    """Apply on the existing session bus; never open hardware from the API."""
    limit_a = validate_gripper_current(limit_a)
    with _registry_lock:
        buses = list(_active.get(robot_name, []))
    if not buses:
        from .sessions import tracker

        session = tracker.current()
        if session and session.get("robot") == robot_name:
            raise GripperSafetyError("Stop the current robot session before changing gripper settings")
        return "next_session"
    if any(b.limit_a is None for b in buses) or limit_a is None:
        raise GripperSafetyError(
            "Stop teleoperation/recording before enabling or disabling gripper current limiting"
        )
    try:
        for bus in buses:
            bus.apply_current(limit_a)
    except Exception as exc:
        # Bimanual partial apply must not leave an unreported mixture running.
        for bus in buses:
            with bus._lock:
                bus._trip(exc)
        raise
    return "live"


def stop_live_grippers(robot_name: str, reason: str) -> None:
    """Latch a stop if Apply succeeded but its persisted configuration failed."""
    with _registry_lock:
        buses = list(_active.get(robot_name, []))
    for bus in buses:
        with bus._lock:
            bus._trip(GripperSafetyError(reason))


def gripper_status(robot_name: str) -> list[dict]:
    with _registry_lock:
        buses = list(_active.get(robot_name, []))
    return [
        {
            "port": b.port,
            "enabled": b._enabled,
            "limit_a": b.limit_a,
            "effective_current_a": b.effective_current_a,
            "temperature_c": b.temperature_c,
            "fault": b._error,
            "hold_torque_nm": b.controller.torque_nm if hasattr(b, "controller") else None,
            "holding": b._enabled and b.controller.holding if hasattr(b, "controller") else False,
            "measured_torque_nm": b._base._last_known_states.get("gripper", {}).get("torque"),
        }
        for b in buses
    ]
