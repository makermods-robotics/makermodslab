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
"""Gravity feed-forward on the Maker follower's MIT frames.

The pinned driver's ``write("Goal_Position")`` sends zero torque, so a raised
arm sags by holding torque / Kp. ``maker_can.install`` attaches this to the bus
of every ``MakerFollower`` built, in every process, so each frame of
``send_action`` carries the torque whatever flow called it.

Per joint: its axis dotted with the moment of the weights of every link it
carries, in motor degrees and motor sign, from the URDF's masses and origins
with the jaws lumped into link_007 closed.
"""

import contextlib
import logging
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
_LINKS = ("link_002", "link_003", "link_004", "link_005", "link_006", "link_007")
_JAWS = ("gripper_left", "gripper_right")
_ROOT_RPY = (math.pi / 2, 0.0, math.pi)
# Motor-positive against the URDF's own axis, per joint.
_MOTOR_SIGN = (1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
_GRAVITY_M_S2 = 9.81

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
# A checkout's source copy first; a wheel ships only dist.
URDF_PATHS = (
    _FRONTEND / "public" / "maker-urdf" / "robot.urdf",
    _FRONTEND / "dist" / "maker-urdf" / "robot.urdf",
)

MAX_SCALE = 1.2
RAMP_S = 0.5
RAMP_STEP_S = 0.05
FRESH_S = 0.15
# About 1.5x the model's largest torque inside the soft limits, under each motor's
# limit: a guard against a bad pose, not reached in normal use.
LIMIT_NM = np.array([0.5, 15.0, 7.5, 2.0, 0.5, 0.5])


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return _rot_z(yaw) @ rot_y @ rot_x


def _floats(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.get(attribute, "0 0 0").split())


class GravityModel:
    """The chain and link weights of one URDF."""

    def __init__(self, urdf_path: Path) -> None:
        root = ET.parse(urdf_path).getroot()
        joints = {joint.find("child").get("link"): joint for joint in root.findall("joint")}

        def inertial(name: str) -> tuple[float, np.ndarray]:
            element = root.find(f"link[@name='{name}']/inertial")
            origin = element.find("origin")
            if any(_floats(origin, "rpy")):
                raise ValueError(f"{name}: inertial rpy must be 0")
            return float(element.find("mass").get("value")), np.array(_floats(origin, "xyz"))

        self._fixed = []
        for link, sign in zip(_LINKS, _MOTOR_SIGN, strict=True):
            joint = joints[link]
            origin = joint.find("origin")
            axis = _floats(joint.find("axis"), "xyz")
            if axis[:2] != (0.0, 0.0) or abs(axis[2]) != 1.0:
                raise ValueError(f"{link}: joint axis must be +-z, got {axis}")
            self._fixed.append(
                (np.array(_floats(origin, "xyz")), _rpy(*_floats(origin, "rpy")), sign * axis[2])
            )

        links = [inertial(name) for name in _LINKS]
        last_mass, last_com = links[-1]
        for jaw in _JAWS:
            origin = joints[jaw].find("origin")
            if any(_floats(origin, "xyz")) or any(_floats(origin, "rpy")):
                raise ValueError(f"{jaw}: expected its joint at link_007's origin")
            mass, com = inertial(jaw)
            last_com = (last_mass * last_com + mass * com) / (last_mass + mass)
            last_mass += mass
        links[-1] = (last_mass, last_com)

        self._coms = [com for _mass, com in links]
        masses = np.array([mass for mass, _com in links])
        self._masses = masses[:, None]
        self._carried = np.cumsum(masses[::-1])[::-1]
        self._root = _rpy(*_ROOT_RPY)

    def holding_nm(self, q_deg) -> np.ndarray:
        """Per joint (motor sign): the torque that holds the arm at ``q_deg`` against gravity."""
        rot, origin = self._root, np.zeros(3)
        axes, pivots, coms = np.empty((6, 3)), np.empty((6, 3)), np.empty((6, 3))
        for i, ((xyz, fixed, axis_z), com) in enumerate(zip(self._fixed, self._coms, strict=True)):
            origin = origin + rot @ xyz
            rot = rot @ fixed
            axes[i], pivots[i] = rot[:, 2] * axis_z, origin
            rot = rot @ _rot_z(math.radians(q_deg[i]) * axis_z)
            coms[i] = origin + rot @ com
        # Weights are vertical: axis . ((c - p) x (0, 0, m g)) = m g (a_x (c_y - p_y) - a_y (c_x - p_x)).
        moment = np.cumsum((self._masses * coms)[::-1], axis=0)[::-1]
        return _GRAVITY_M_S2 * (
            axes[:, 0] * (moment[:, 1] - self._carried * pivots[:, 1])
            - axes[:, 1] * (moment[:, 0] - self._carried * pivots[:, 0])
        )


@lru_cache(maxsize=1)
def shipped_model() -> GravityModel:
    path = next((path for path in URDF_PATHS if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no Maker URDF at {[str(p) for p in URDF_PATHS]}")
    return GravityModel(path)


class GravityFeedforward:
    """Adds the model's torque to the arm joints' Goal_Position frames on one bus.

    Eased in from zero after torque-on, at the measured pose when every joint
    replied within ``FRESH_S`` and otherwise at each joint's latest goal. A joint
    commanded again starts the next frame. Gripper frames carry no torque.
    """

    def __init__(
        self, bus, model: GravityModel, scale: float, clock: Callable[[], float] = time.time
    ) -> None:
        if not 0.0 <= scale <= MAX_SCALE:
            raise ValueError(f"Gravity feed-forward scale must be 0..{MAX_SCALE}, got {scale}")
        self.bus = bus
        self.model = model
        self.scale = scale
        self._clock = clock
        self._levels: dict[str, float] = {}  # the energized arm joints' ease-in, from 0 at their torque-on
        self._goals: dict[str, float] = {}
        self._written: set[str] = set()
        self._retained: dict[str, float] = {}
        self._frame_t: float | None = None
        self._frame_nm: np.ndarray | None = None
        self._bus_write = bus.write
        self._bus_sync_write = bus.sync_write
        self._bus_enable_torque = bus.enable_torque
        self._bus_disable_torque = bus.disable_torque
        bus.write = self._write
        bus.sync_write = self._sync_write
        bus.enable_torque = self._enable_torque
        bus.disable_torque = self._disable_torque

    def _start_frame(self) -> None:
        now = self._clock()
        if self._frame_t is not None:
            step = min(RAMP_STEP_S, max(0.0, now - self._frame_t))
            for motor, level in self._levels.items():
                self._levels[motor] = min(1.0, level + step / RAMP_S)
        self._frame_t = now
        self._written.clear()
        self._frame_nm = None

    def _measured_pose(self) -> list[float] | None:
        now = self._clock()
        pose = []
        for motor in ARM_JOINTS:
            stamp = self.bus.last_feedback_time.get(motor)
            if stamp is None or not 0.0 <= now - stamp <= FRESH_S:
                return None
            position = float(self.bus._last_known_states[motor]["position"])
            if not math.isfinite(position):
                return None
            pose.append(position)
        return pose

    def _goal_pose(self) -> list[float]:
        pose = []
        for motor in ARM_JOINTS:
            if motor in self._goals:
                pose.append(self._goals[motor])
            elif self.bus.last_feedback_time.get(motor) is not None:
                pose.append(float(self.bus._last_known_states[motor]["position"]))
            else:
                pose.append(math.nan)
        return pose

    def _torques(self, pose: list[float]) -> np.ndarray:
        if not all(math.isfinite(q) for q in pose):
            return np.zeros(6)
        levels = np.array([self._levels.get(motor, 0.0) for motor in ARM_JOINTS])
        nm = self.model.holding_nm(pose) * self.scale * levels
        return np.clip(np.nan_to_num(nm, nan=0.0, posinf=0.0, neginf=0.0), -LIMIT_NM, LIMIT_NM)

    def torque(self, motor: str, goal: float) -> float:
        """The torque for this joint's frame, advancing the frame and the ease-in."""
        self._goals[motor] = float(goal)
        if motor not in self._levels:
            return 0.0
        if self._frame_t is None or motor in self._written:
            self._start_frame()
        self._written.add(motor)
        if self._levels[motor] == 0.0:
            return 0.0
        joint = ARM_JOINTS.index(motor)
        if self._frame_nm is not None:
            return float(self._frame_nm[joint])
        measured = self._measured_pose()
        if measured is None:
            return float(self._torques(self._goal_pose())[joint])
        self._frame_nm = self._torques(measured)
        return float(self._frame_nm[joint])

    def _write(self, data_name: str, motor: str, value) -> None:
        if data_name != "Goal_Position" or motor not in ARM_JOINTS:
            return self._bus_write(data_name, motor, value)
        tau = self.torque(motor, value)
        if tau == 0.0:
            result = self._bus_write(data_name, motor, value)
            self._retained.pop(motor, None)
            return result
        gains = self.bus._gains[motor]
        self._retained[motor] = value
        return self.bus._mit_control(motor, gains["kp"], gains["kd"], value, 0, tau)

    def _sync_write(self, data_name: str, values: dict) -> None:
        result = self._bus_sync_write(data_name, values)
        if data_name == "Goal_Position":
            for motor in values:
                self._retained.pop(motor, None)
        return result

    def _forget_frame(self, motors) -> None:
        self._frame_nm = None
        self._written.difference_update(motors)
        if not self._levels:
            self._frame_t = None

    def _enable_torque(self, motors=None, num_retry: int = 0) -> None:
        result = self._bus_enable_torque(motors, num_retry=num_retry)
        joints = [motor for motor in self.bus._get_motors_list(motors) if motor in ARM_JOINTS]
        if joints:
            for motor in joints:
                self._levels[motor] = 0.0
                self._goals.pop(motor, None)
            self._forget_frame(joints)
        return result

    def _disable_torque(self, motors=None, num_retry: int = 0) -> None:
        # A motor keeps its last MIT frame across a disable, so a later enable
        # would apply this torque at once.
        targets = self.bus._get_motors_list(motors)
        for motor in targets:
            goal = self._retained.pop(motor, None)
            if goal is not None:
                gains = self.bus._gains[motor]
                with contextlib.suppress(Exception):
                    self.bus._mit_control(motor, gains["kp"], gains["kd"], goal, 0, 0)
        joints = [motor for motor in targets if motor in ARM_JOINTS]
        if joints:
            for motor in joints:
                self._levels.pop(motor, None)
            self._forget_frame(joints)
        return self._bus_disable_torque(motors, num_retry=num_retry)


def attach_to_follower(follower) -> GravityFeedforward | None:
    """Fit a follower's bus with the family's feed-forward, if it asks for one.

    A model that cannot load leaves the follower as the driver built it; a scale
    out of range refuses the follower before anything is energized.
    """
    from .arms.maker import MAKER

    scale = MAKER.gravity_feedforward_scale
    if scale == 0.0:
        return None
    try:
        model = shipped_model()
    except (OSError, ET.ParseError, ValueError, KeyError, AttributeError, TypeError) as exc:
        logger.error("Maker gravity feed-forward is OFF: could not load the URDF model: %s", exc)
        return None
    follower.gravity_feedforward = GravityFeedforward(follower.bus, model, scale)
    return follower.gravity_feedforward
