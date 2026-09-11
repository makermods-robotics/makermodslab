"""MIT gripper holding-effort regulation, inspired by YAM's target correction.

This is not a hard instantaneous torque/current limit. Only closing contact
changes the position command; motor Kp/Kd remain fixed. No thermal derating.
"""

import math
import time
from collections import deque

from .gripper_settings import GRIPPER_RESTART_C, validate_gripper_hold_torque
from .metal_gripper import WATCHDOG_TICKS, GripperSafetyError, MetalGripperBus


class HoldingController:
    """All positions/velocities here are radians; positive position opens Metal."""

    def __init__(self, torque_nm: float, limits_rad: tuple[float, float]):
        self.torque_nm = validate_gripper_hold_torque(torque_nm)
        if self.torque_nm is None:
            raise ValueError("Holding controller requires a torque")
        self.limits = limits_rad
        self.holding = False
        self.last_command = None
        self.filtered = None
        self.history = deque()
        self.last_time = None
        self.contact_started = None
        self.low_effort_since = None

    def update(self, goal: float, position: float, velocity: float, effort: float, kp: float, now: float):
        if not all(math.isfinite(x) for x in (goal, position, velocity, effort, kp, now)) or kp <= 0:
            raise GripperSafetyError("Invalid gripper holding feedback/gain")
        goal = min(self.limits[1], max(self.limits[0], goal))
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        self.history.append((now, effort))
        while self.history and self.history[0][0] < now - 0.1:
            self.history.popleft()
        # Signed closing effort avoids treating opening acceleration as a grasp.
        closing_effort = -sum(e for _, e in self.history) / len(self.history)
        opening = goal > position + math.radians(0.25)
        if self.holding:
            # Target relaxation initially reduces effort below its final value.
            # Require sustained loss of contact so it cannot oscillate between
            # full-close tracking and holding during this normal transient.
            if closing_effort < min(0.2, self.torque_nm * 0.4):
                if self.low_effort_since is None:
                    self.low_effort_since = now
            else:
                self.low_effort_since = None
            lost_contact = self.low_effort_since is not None and now - self.low_effort_since >= 0.3
            if opening or lost_contact:
                self.holding = False
                self.contact_started = None
                self.low_effort_since = None
        elif goal < position and closing_effort > 0.5 and abs(velocity) < 0.3:
            if self.contact_started is None:
                self.contact_started = now
            # Require 100 ms of contact evidence, not one acceleration sample.
            if now - self.contact_started >= 0.1:
                self.holding = True
        else:
            self.contact_started = None

        if self.holding:
            # q_raw = q_last + s * (target_effort - measured_effort) / Kp;
            # Metal closes toward smaller angles, so s=-1.
            raw = self.last_command - (self.torque_nm - abs(effort)) / kp
            # Time-based smoothing avoids dependence on recording/teleop rate:
            # alpha=0.1 at a 20 ms update. Normal movement has no such filter.
            alpha = 1.0 - math.exp(-min(dt, 0.05) / 0.19)
            self.filtered += alpha * (raw - self.filtered)
            # Never close beyond the leader request; prevent integrator windup
            # at joint bounds by storing the actual bounded command.
            command = min(self.limits[1], max(goal, self.limits[0], self.filtered))
            self.filtered = command
        else:
            command = goal
            self.filtered = position
        self.last_command = command
        return command


class MetalHoldingBus(MetalGripperBus):
    """Reuse serialized CAN lifecycle/stops, but route gripper through MIT."""

    def __init__(self, bus, robot_name, hold_torque_nm, joint_limits, *, gains):
        super().__init__(bus, robot_name, None, joint_limits, speed_limit_deg_s=120)
        self.controller = HoldingController(hold_torque_nm, tuple(map(math.radians, joint_limits)))
        self.kp, self.kd = gains
        if not all(math.isfinite(x) for x in gains) or not 0 < self.kp <= 500 or not 0 < self.kd <= 5:
            raise GripperSafetyError("Holding control requires valid nonzero Kp/Kd")
        self._velocity = 0.0
        self._feedforward = 0.0
        self._last_action = 0.0

    @property
    def control_enabled(self):
        return True

    def connect(self, handshake=True):
        from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS

        from .metal_gripper import _active, _registry_lock

        with self._lock:
            self._base.connect(handshake=False)
            try:
                self._mode = int(self._param(10))
                if not handshake:
                    return  # Shared failure cleanup opens only to disable.
                self._original_mode = self._mode
                if self._mode not in (1, 4):
                    raise GripperSafetyError("Gripper must start in MIT or force-position mode")
                self._refresh()
                if self._status != 0:
                    raise GripperSafetyError("Release the gripper before starting holding control")
                if self.temperature_c >= GRIPPER_RESTART_C:
                    raise GripperSafetyError("Let both gripper sensors cool below 45 C before starting")
                expected = MOTOR_LIMIT_PARAMS[self._base._motor_types["gripper"]]
                for rid, scale in zip((21, 22, 23), expected, strict=True):
                    if not math.isclose(float(self._param(rid, fmt="f")), scale, rel_tol=1e-5):
                        raise GripperSafetyError(
                            "Gripper MIT scaling differs from the configured Metal driver"
                        )
                self._goal = float(self._base._last_known_states["gripper"]["position"])
                self._original_timeout = int(self._param(9))
                self._param(9, WATCHDOG_TICKS)
                if self._mode != 1:
                    self._param(10, 1)
                self._mode = 1
                self._prepared = True
                self._preload()
                for motor in self._base.motors:
                    if motor != "gripper":
                        self._base.read("Present_Position", motor)
            except Exception:
                self.disconnect()
                raise
            with _registry_lock:
                _active.setdefault(self.robot_name, []).append(self)

    def _send_mit(self, kp, kd, goal, velocity, ff):
        packet = bytes(
            self._base._encode_mit_packet(
                self._base._motor_types["gripper"],
                kp,
                kd,
                goal,
                velocity,
                ff,
            )
        )
        self._state(self._query(self._motor_id, packet, self._is_state))
        self._check()

    def _preload(self):
        self._send_mit(0.0, 0.0, self._goal, 0.0, 0.0)

    def _force(self, goal, current_a=None):
        self._check()
        if not math.isfinite(goal):
            raise GripperSafetyError("Invalid gripper target")
        self._goal = min(self.joint_limits[1], max(self.joint_limits[0], goal))
        # Always refresh here: the underlying multi-motor read also consumes
        # feedback, and the holding equation must use a fresh matching sample.
        self._refresh()
        self._check()
        state = self._base._last_known_states["gripper"]
        now = time.monotonic()
        target = self.controller.update(
            math.radians(self._goal),
            math.radians(state["position"]),
            math.radians(state["velocity"]),
            state["torque"],
            self.kp,
            now,
        )
        paused = now - self._last_action > 0.1
        velocity = 0.0 if self.controller.holding or paused else self._velocity
        ff = 0.0 if self.controller.holding or paused else self._feedforward
        self._send_mit(self.kp, self.kd, math.degrees(target), velocity, ff)

    def _hold(self):
        # Runs even during recording pauses; normal action writes use this same
        # lock. Independent of the recording frame rate.
        while not self._stop.wait(0.02):
            with self._lock:
                if not self._enabled:
                    return
                try:
                    self._force(self._goal)
                except Exception as exc:
                    self._trip(exc)
                    return

    def sync_write_metal(self, commands):
        with self._lock:
            try:
                self._check()
                if "gripper" in commands:
                    kp, kd, goal, velocity, ff = commands["gripper"]
                    if not all(math.isfinite(x) for x in commands["gripper"]):
                        raise GripperSafetyError("Invalid MIT gripper command")
                    # Follow the configured gains; do not silently retune the
                    # force equation when an upstream command changes gains.
                    if (kp, kd) != (self.kp, self.kd):
                        raise GripperSafetyError("Stop the session before changing gripper gains")
                    self._velocity, self._feedforward = velocity, ff
                    self._last_action = time.monotonic()
                    self._force(goal)
                other = {k: v for k, v in commands.items() if k != "gripper"}
                if other:
                    self._base.sync_write_metal(other)
            except Exception as exc:
                self._trip(exc)
                raise

    def sync_write(self, data_name, values):
        with self._lock:
            if data_name == "Goal_Position" and "gripper" in values:
                self._velocity, self._feedforward = 0.0, 0.0
                self._last_action = time.monotonic()
            return super().sync_write(data_name, values)

    def apply_hold_torque(self, value):
        with self._lock:
            value = validate_gripper_hold_torque(value)
            if value is None:
                raise GripperSafetyError("Stop the session before disabling holding control")
            self._check()
            old = self.controller.torque_nm
            self.controller.torque_nm = value
            try:
                if self._enabled:
                    self._force(self._goal)
                else:
                    self._refresh()
                    self._check()
            except Exception as exc:
                self.controller.torque_nm = old
                self._trip(exc)
                raise
