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
"""What the two CAN families (Maker, Metal) share.

Both are 7-DOF followers on classic CAN (via an slcan adapter) driven by a
Star Arm 102 (reBot 102) leader on FashionStar UART servos — the follower's
motor protocol (RobStride vs Damiao) is the only hardware difference, and it
lives entirely inside lerobot's device classes. Neither bus speaks the
Feetech register protocol, so every register-by-name helper is off; their
joint limits are measured constants, so calibration is a zero pose set by
hand with torque off; and the Star leader's joints hold encoders and no
motors, so there is nothing to back-drive for a DAgger handover.

The leader preset MUST match the follower family — never the bare
rebot_102_leader: each preset carries the joint directions and ranges of
ITS follower, and a mismatched one runs joints the wrong way or saturates
them against the follower's soft limits while teleop keeps reporting a
healthy loop. Each family names its own preset classes in _device_classes
and the builders here never choose one.

Both leaders share ONE calibration library (rebot_102_leader): the
presets are config-only variants of one lerobot class and the class name
picks the directory. That sharing is why default_calibration_name mints
the family id into the slot name (inherited from the base contract).
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ArmFamily, CalibrationUI

logger = logging.getLogger(__name__)

# Settle time after unlocking a FashionStar servo before writing its origin
# point. Mirrors lerobot's own `_SETTLE_SEC` in rebot_102_leader.py — the servo
# needs a moment between the unlock and the write or the origin lands on a
# stale reading.
_SETTLE_SEC = 0.01


@dataclass(frozen=True)
class CanDeviceClasses:
    """The six lerobot config classes a CAN family builds sessions from.

    The bimanual configs embed the UNREGISTERED base dataclasses
    (follower_base / leader_sub) rather than the registered ones,
    which is what keeps draccus's choice-registry tree from becoming
    self-referential. Passing a registered config there would still typecheck
    but re-enters that tree.
    """

    follower: type  # registered single follower config (port, id, cameras)
    follower_base: type  # unregistered follower base, for the bimanual arms
    bi_follower: type  # registered bimanual follower config
    teleop: type  # registered single-leader preset (port, id)
    leader_sub: type  # unregistered leader preset, for the bimanual arms
    bi_teleop: type  # bimanual leader config carrying two leader_sub configs


# Both CAN families use the same Star Arm 102 leader, and it has ONE zero
# pose: folded against the base, gripper closed. The follower poses are
# family-specific (follower_zero_pose) and opposite on the gripper.
_LEADER_ZERO_POSE = "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, gripper closed — then confirm."


class CanArmFamily(ArmFamily):
    """Shared shape of the CAN families; subclasses name their classes and
    their follower's zero pose."""

    joints_per_arm = 7
    supports_bimanual = True

    # The follower's zero-pose text; set by each family (the two are
    # opposites on the gripper — Maker folded/open, Metal upright/closed).
    follower_zero_pose: str

    uses_feetech_bus = False
    supports_auto_calibration = False
    # The zero-pose procedure below, run as the generic step wizard.
    calibration_kind = "steps"
    supports_dagger = False

    leader_library_attr = "MAKER_LEADER_CONFIG_PATH"

    # No Maker or Metal URDF ships, so the viewer's slot shows the numeric
    # readout fed by `joints_deg`.
    telemetry_kind = "degrees"

    def _device_classes(self) -> CanDeviceClasses:  # pragma: no cover - abstract by convention
        """Import (lazily — python-can / motorbridge) and return this family's classes."""
        raise NotImplementedError

    # --- the zero-pose calibration, as a step wizard ----------------------------
    # The CAN arms need no range sweep: their joint limits are fixed constants
    # measured once against the arms' mechanical stops (the follower configs'
    # joint_limits, and the Star 102 leader presets' joint_ranges copied from
    # them). The only thing calibration has to establish is WHERE ZERO IS: torque
    # off, the user poses the arm by hand, we tell the motors "this is zero",
    # and the calibration file's ranges come from the config. One step, no
    # driving. A web reimplementation of lerobot's MakerFollower.calibrate() /
    # RebotArm102Leader.calibrate(), which block on input().

    def zero_pose_instructions(self, device_type: object | None = None) -> str:
        """The physical pose to ask for: the shared Star-leader pose for
        "teleop", this family's follower pose otherwise (opposites on the
        gripper between Maker and Metal). The families' own helper, not part
        of the base contract — calibration_summary and calibrate serve it."""
        if device_type == "teleop":
            return _LEADER_ZERO_POSE
        return self.follower_zero_pose

    def calibration_summary(self, device_type: object | None = None) -> dict | None:
        # No served image: the frontend keeps its bundled photos for the built-ins.
        return {"text": self.zero_pose_instructions(device_type), "image_url": None}

    def open_for_calibration(self, device_type: str, port: str, config_id: str) -> Any:
        """Open ONE device's bus with torque OFF, ready for the user to pose the arm."""
        if device_type == "robot":
            from lerobot.robots import make_robot_from_config

            device = make_robot_from_config(self.single_follower_config(port, config_id))
            # NOT device.connect(): both followers' connect() finishes by
            # calling enable_torque(), which would lock the arm rigid exactly
            # when the user needs to move it by hand. Open the bus directly
            # and disable torque, which is what lerobot's own calibrate() does
            # internally. The disable is not optional for Metal even here:
            # the Damiao HANDSHAKE inside bus.connect() is itself the enable
            # command, so the arm comes up energized and this write is what
            # frees it for the user's hands.
            try:
                device.bus.connect()
                device.bus.disable_torque()
            except Exception:
                # Either the handshake raised partway (a Damiao handshake is
                # the per-motor enable, so the motors that answered are
                # energized while is_connected reads False) or it completed
                # and the write that frees the arm failed. Both: de-energize
                # the bus the recovery way (reopen without the handshake if
                # needed, broadcast the disable), close the port, then let the
                # failure propagate. Same recovery the teleop/replay connect
                # paths run; without it the arm holds torque with no device
                # object left to release it through.
                from .. import torque

                torque.de_energize_can_device(device, f"{self.short_label} follower arm")
                # Best effort: the original error is the one to raise.
                with contextlib.suppress(Exception):
                    device.bus.disconnect(False)
                raise
            return device

        from lerobot.teleoperators import make_teleoperator_from_config

        device = make_teleoperator_from_config(self.single_leader_config(port, config_id))
        # The leader's bus is constructed inside connect(), so there is no
        # bus-only path — but there is nothing to disable either: its joints
        # hold encoders and no motors. connect(calibrate=False) leaves it
        # unlocked and back-drivable, which is the state we want.
        device.connect(calibrate=False)
        return device

    def calibrate(self, device: Any, device_type: str, ui: CalibrationUI) -> dict[str, Any]:
        """One live-positions step at the zero pose, then set zero and build the file.

        The two device SIDES differ in exactly one place, the zero write: the
        followers speak CAN (RobStride and Damiao alike), and one whole-bus
        ``set_zero_position()`` zeroes every motor at once; the Star 102 leader
        speaks FashionStar UART, and each servo has to be unlocked and given
        ``set_origin_point`` individually.
        """
        is_follower = device_type == "robot"
        # Torque is off and stays off for this entire wait: the user is
        # physically moving the arm.
        ui.step(self.zero_pose_instructions(device_type), image_url=None, live_positions=True)
        ui.message("Setting zero…")

        pre_zero: dict[str, float] = {}
        try:
            pre_zero = self.read_positions(device)
        except Exception as e:
            logger.warning(f"Could not read pre-zero positions: {e}")
        for motor, value in pre_zero.items():
            # Logged so an offset against the PREVIOUS zero stays recoverable
            # from the logs if this one turns out to have been taken in the
            # wrong pose — same reason lerobot's own calibrate() logs them.
            logger.info(f"Pre-zero position of {motor}: {value:.2f} deg")

        self._set_zero(device, is_follower)
        logger.info("Arm zero position set.")
        return self._build_calibration(device, is_follower)

    @staticmethod
    def _set_zero(device: Any, is_follower: bool) -> None:
        """Tell the motors that where they are now is zero."""
        bus = device.bus
        if is_follower:
            # CAN (RobStride and Damiao alike): one broadcast zeroes every
            # motor on the bus.
            bus.set_zero_position()
            # Mirror what MakerFollower.calibrate() resets alongside the zero,
            # so the freshly zeroed arm is not still carrying the previous
            # zero's multi-turn bookkeeping (which would make send_action
            # refuse with a stale-zero error).
            for attr, value in (
                ("_turn_offset", dict.fromkeys(getattr(device, "_joint_motor_names", []), 0.0)),
                ("_stale_zero", {}),
                ("_last_positions", {}),
            ):
                if hasattr(device, attr):
                    setattr(device, attr, value)
            return

        # FashionStar UART: no broadcast — unlock and origin each servo in turn.
        for motor_id in dict(device.config.joint_ids).values():
            bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
            bus.set_origin_point(motor_id)

    @staticmethod
    def _build_calibration(device: Any, is_follower: bool) -> dict[str, Any]:
        """The calibration file's contents: fixed ranges from the config.

        ``homing_offset`` is 0 for every joint and that is correct, not a
        placeholder — the zero now lives INSIDE the motor (RobStride's zero
        position / FashionStar's origin point), so there is no software offset
        left to apply on top. It is also why the Feetech EEPROM fingerprint in
        ``arm_identity.py`` can say nothing about a Maker arm: every Maker
        calibration file has the same all-zero offsets.
        """
        from lerobot.motors import MotorCalibration

        config = device.config
        if is_follower:
            ids = config.motor_can_ids
            ranges = config.joint_limits
        else:
            ids = config.joint_ids
            ranges = {m: tuple(v) for m, v in config.joint_ranges.items()}
        default = (-360.0, 360.0)

        calibration: dict[str, MotorCalibration] = {}
        for motor_name, motor_id in ids.items():
            # The two CAN followers disagree about the id field's shape:
            # Maker motor_can_ids are plain ints, Metal's are (send, recv)
            # tuples. MotorCalibration.id is an int, and lerobot's own
            # MetalFollower.calibrate() stores the SEND id.
            if isinstance(motor_id, tuple):
                motor_id = motor_id[0]
            range_min, range_max = ranges.get(motor_name, default)
            calibration[motor_name] = MotorCalibration(
                id=motor_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )
        return calibration

    def single_follower_config(self, port: str, config_id: str):
        return self._device_classes().follower(port=port, id=config_id)

    def single_leader_config(self, port: str, config_id: str):
        return self._device_classes().teleop(port=port, id=config_id)

    def capture_rest_poses(self, robot: Any, *, include_gripper: bool = False) -> list[tuple[Any, dict]]:
        # Degrees by bare motor name, one entry per drivable sub-arm (the
        # bimanual wrapper's sub-arms drive directly, on their own buses).
        from .. import maker_rest_pose

        return [
            (arm, maker_rest_pose.capture_maker_pose(arm, include_gripper=include_gripper))
            for arm, _label in maker_rest_pose.maker_follower_arms(robot)
        ]

    def return_to_rest(self, rest_poses: list[tuple[Any, dict]], abort_event: Any = None) -> None:
        # The MIT setpoint interpolated at a bounded rate, arrival judged by
        # CONVERGENCE (a loaded joint holds a standing error), all arms at once.
        from .. import maker_rest_pose

        maker_rest_pose.return_maker_arms_to_rest(rest_poses, abort_event)

    def release_torque(self, device: Any, label: str = "device") -> list[str]:
        # One whole-bus disable per bus; the Star leader has no motors and is skipped.
        from .. import torque

        return torque.release_maker_torque(device, label)

    async def probe_ports(self, ports: list[str] | None = None) -> dict:
        from .. import maker_ports

        return await maker_ports.probe_maker_ports(ports, self.id)

    async def identify_by_motion(self, device_type: str, ports: list[str] | None = None) -> dict:
        from .. import maker_ports

        return await maker_ports.identify_maker_arm_by_motion(device_type, ports, self.id)

    def build_single_configs(self, request: Any, cameras: dict | None, leader_id: str, follower_id: str):
        # The follower config's defaults carry the CAN wiring (slcan @ 1 Mbps,
        # the per-joint ids, soft limits and MIT gains); only the adapter port
        # and the calibration id vary per session.
        cls = self._device_classes()
        if cameras is None:
            robot_config = cls.follower(
                port=request.follower_port,
                id=follower_id,
            )
        else:
            robot_config = cls.follower(
                port=request.follower_port,
                id=follower_id,
                cameras=cameras,
            )

        teleop_config = cls.teleop(
            port=request.leader_port,
            id=leader_id,
        )

        return robot_config, teleop_config

    def build_bimanual_configs(
        self,
        request: Any,
        cameras: dict | None,
        base: str,
        leader_staging: str,
        follower_staging: str,
    ):
        cls = self._device_classes()
        if cameras is None:
            left_follower = cls.follower_base(port=request.follower_port)
        else:
            left_follower = cls.follower_base(port=request.follower_port, cameras=cameras)

        robot_config = cls.bi_follower(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=left_follower,
            right_arm_config=cls.follower_base(port=request.right_follower_port),
        )
        teleop_config = cls.bi_teleop(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=cls.leader_sub(port=request.leader_port),
            right_arm_config=cls.leader_sub(port=request.right_leader_port),
        )

        return robot_config, teleop_config
