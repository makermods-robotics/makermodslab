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
"""The SO-101 family: Feetech STS3215 smart servos on a USB serial bus.

Registers are readable and writable (EEPROM + RAM), which is what the
fingerprint, torque-cap and rest-pose machinery all depend on — hence
uses_feetech_bus. Calibration is a range sweep, manual or automatic
(the vendored Feetech autocal). 6 joints per arm. The leader has motors in
its joints, so it can be back-driven for a DAgger handover.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .base import ArmFamily, FollowerPreflight

logger = logging.getLogger(__name__)


# --- the port-based follower preflights rollout runs before its subprocess -----
# Module-level helpers (the family's preflight_ports override calls them) so a
# test can stub them here. Everything they need is imported INSIDE them: this
# package is imported by utils.config, i.e. by everything, and arm_identity /
# motor_power import utils.config back.


@contextmanager
def _open_follower(port: str, follower_id: str):
    """Open a bare follower bus on `port`, yield the connected robot, and
    release the port read-only on exit.

    Both rollout preflights connect one follower, do read-only work, then must
    free the port for the subprocess to reopen. Torque is never enabled here,
    so the release skips the torque-disable write (``disconnect(
    disable_torque=False)``) — a plain port close. The disconnect runs on any
    exit path (success or exception)."""
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(SO101.single_follower_config(port, follower_id))
    robot.bus.connect()
    try:
        yield robot
    finally:
        robot.bus.disconnect(disable_torque=False)


def _preflight_arm_identity(port: str, follower_id: str, config_name: str | None = None) -> list[str]:
    """Read-only identity check of ONE follower arm before the rollout
    subprocess starts.

    The subprocess itself can't be guarded (its stdin is pre-seeded with a
    newline, which auto-confirms lerobot's "use the calibration file" prompt
    and stamps the file into EEPROM on mismatch), so the check happens here:
    connect the bare bus, verify, and release the port for the subprocess to
    reopen. Raises ArmIdentityError on a hard mismatch; returns the
    warn-but-allow messages otherwise.

    `follower_id` names the calibration the arm loads and is what identifies the
    slot by default. For a bimanual staging alias id ("<base>_left"), pass the
    real library stem as `config_name` so the guard compares against the library
    entry rather than the alias (mirrors verify_devices' config_names in
    record/teleop). Bimanual runs each follower bus through this separately —
    each opens and releases its own port — so the two are never open at once."""
    from ..arm_identity import verify_devices

    # The counterpart lookup stays in rollout: it is that flow's knowledge
    # (inference has no leader in the session, so the slot comes from the
    # robot records), not the family's. Lazy — rollout imports this package.
    from ..rollout import _counterpart_leader_slots

    with _open_follower(port, follower_id) as robot:
        return verify_devices(
            ((robot, "follower"),),
            extra_slots=_counterpart_leader_slots(config_name or follower_id),
            config_names=[config_name] if config_name is not None else None,
        )


def _preflight_motor_registers(port: str, follower_id: str) -> list[str]:
    """Prime the follower's RAM motor registers before the rollout subprocess
    starts.

    The subprocess itself can't be instrumented, but Torque_Limit and
    Goal_Velocity are both RAM registers: they survive closing the serial port
    (only a power cycle resets them), and the subprocess's connect()/configure()
    never writes them — so setting them here and releasing the port is enough
    for the whole rollout. Two priming steps:
      - reset_torque_limit: restore stock torque (a previous auto-calibration's
        working torque would otherwise cap the whole rollout).
      - clear_goal_velocity: reset any leftover speed cap a previous
        arm-driving feature stamped (auto-cal fold/unfold=1000, rest-pose
        return=400), which would otherwise throttle the whole rollout.
    Never raises: a failure degrades to the previous register value (logged)
    and returns warning messages instead of aborting the start."""
    from ..motor_power import FOLLOWER, clear_goal_velocity, reset_torque_limit

    try:
        with _open_follower(port, follower_id) as robot:
            return reset_torque_limit(robot, FOLLOWER) + clear_goal_velocity(robot, FOLLOWER)
    except Exception as exc:
        message = (
            f"Could not reset the motor registers on {port}: {exc}. "
            "The arm runs at its previous torque/speed limits for this rollout."
        )
        logger.warning(message)
        return [message]


class SO101Family(ArmFamily):
    id = "so101"
    label = "SO-101"
    short_label = "SO-101"
    indefinite_label = "an SO-101 arm"

    joints_per_arm = 6
    supports_bimanual = True

    uses_feetech_bus = True
    supports_auto_calibration = True
    # The Feetech sweep managers (calibrate.py / auto_calibrate.py).
    calibration_kind = "range_sweep"
    supports_dagger = True

    single_robot_type = "so101_follower"
    bimanual_robot_type = "bi_so_follower"

    # The SO family is every string carrying an so100/so101 marker or the bare
    # so_follower/so_leader device names lerobot writes for a bimanual SO rig
    # (bi_so_follower). Loose on purpose; scanned last for that reason.
    robot_type_markers = ("so100", "so101", "so-100", "so-101", "so_follower", "so_leader")

    leader_library_attr = "LEADER_CONFIG_PATH"
    follower_library_attr = "FOLLOWER_CONFIG_PATH"

    # Leader and follower are the same Feetech bus: nothing to tell them apart
    # by protocol, so the SO-101 identifies by the hand-swing gesture only.
    follower_probe_protocol = None
    motion_identify_energizes_follower = False

    telemetry_kind = "urdf"

    @property
    def calibration_name_suffix(self) -> str:
        """Historical default: the bare record name, no family suffix."""
        return ""

    def single_follower_config(self, port: str, config_id: str):
        from lerobot.robots.so_follower import SO101FollowerConfig

        return SO101FollowerConfig(port=port, id=config_id)

    def single_leader_config(self, port: str, config_id: str):
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        return SO101LeaderConfig(port=port, id=config_id)

    def verify_identity(self, pairs: Any, *, skip: bool = False, config_names: Any = None) -> list[str]:
        from .. import arm_identity

        return arm_identity.verify_devices(pairs, skip=skip, config_names=config_names)

    def prepare_follower_registers(self, robot: Any, label: str | None = None) -> list[str]:
        # Both are Feetech RAM registers on the FOLLOWER (see motor_power.py
        # for why the leader must never get the speed-cap clear).
        from ..motor_power import FOLLOWER, clear_goal_velocity, reset_torque_limit

        warnings = reset_torque_limit(robot, FOLLOWER, label)
        warnings += clear_goal_velocity(robot, FOLLOWER, label)
        return warnings

    def preflight_ports(
        self, followers: list[FollowerPreflight], *, skip_identity: bool = False
    ) -> list[str]:
        """Identity check THEN register priming, PER FOLLOWER, in order.

        For two followers that is ``identity(a), registers(a), identity(b),
        registers(b)`` — not identity on both and then registers on both, as
        rollout once did: each follower's fingerprint is verified before
        anything is primed on it, so a swapped arm is refused with nothing
        written to it, and the ports are still opened one at a time. The
        register reset is never optional (a previous auto-calibration's cap
        would otherwise throttle the whole rollout); ``skip_identity`` skips
        the fingerprint only, and is logged as the warning it always was.
        """
        if skip_identity:
            logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
        warnings: list[str] = []
        for follower in followers:
            if not skip_identity:
                warnings += _preflight_arm_identity(
                    follower.port, follower.calibration_id, config_name=follower.config_name
                )
            warnings += _preflight_motor_registers(follower.port, follower.calibration_id)
        return warnings

    def capture_rest_poses(self, robot: Any, *, include_gripper: bool = False) -> list[tuple[Any, dict]]:
        # One bus per follower arm (a bimanual BiSO robot exposes two), raw
        # ticks — directly replayable as Goal_Position later.
        from .. import rest_pose, torque

        poses = []
        for bus in torque.device_buses(robot):
            pose = rest_pose.capture_rest_pose(bus)
            if not include_gripper:
                pose = {m: v for m, v in pose.items() if m != "gripper"}
            poses.append((bus, pose))
        return poses

    def return_to_rest(self, rest_poses: list[tuple[Any, dict]], abort_event: Any = None) -> None:
        # A Feetech profile-velocity move per bus, all buses at once.
        from .. import rest_pose

        rest_pose.return_buses_to_rest(rest_poses, abort_event)

    def release_torque(self, device: Any, label: str = "device") -> list[str]:
        # Motor by motor, so one bad motor cannot leave the other joints locked.
        from .. import torque

        return torque.force_disable_torque(device, label)

    async def identify_by_motion(self, device_type: str, ports: list[str] | None = None) -> dict:
        # Both halves speak Feetech serial on motor id 1, so the side asked
        # about changes nothing — identify.py watches the same register either way.
        from .. import identify

        return await identify.identify_arm_by_motion(ports)

    def build_single_configs(self, request: Any, cameras: dict | None, leader_id: str, follower_id: str):
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        if cameras is None:
            robot_config = SO101FollowerConfig(
                port=request.follower_port,
                id=follower_id,
            )
        else:
            robot_config = SO101FollowerConfig(
                port=request.follower_port,
                id=follower_id,
                cameras=cameras,
            )

        teleop_config = SO101LeaderConfig(
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
        from lerobot.robots.bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        if cameras is None:
            left_follower = SO101FollowerConfig(port=request.follower_port)
        else:
            left_follower = SO101FollowerConfig(port=request.follower_port, cameras=cameras)

        robot_config = BiSOFollowerConfig(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=left_follower,
            right_arm_config=SO101FollowerConfig(port=request.right_follower_port),
        )
        teleop_config = BiSOLeaderConfig(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=SO101LeaderConfig(port=request.leader_port),
            right_arm_config=SO101LeaderConfig(port=request.right_leader_port),
        )

        return robot_config, teleop_config


SO101 = SO101Family()
