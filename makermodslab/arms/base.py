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
"""The arm-family contract: what the core asks an arm type for, in one place.

Every robot flow branches on arm_type somewhere — which lerobot configs to
build, how many joints a checkpoint must have, which calibration library to
read, whether a servo register can be touched by name. Before this package
those branches were if arm_type == "maker" literals spread over the
capability predicates, the robot factory, the config paths and the rollout
argument builder; each new family meant finding every one of them again (the
Metal arm was the second family through those seams and is why they are
centralized now). An ArmFamily answers all of those questions for ONE
family, and the registry (registry.py) is the only place the core looks
them up. Nothing outside this package compares an arm type to a literal.

What the contract covers TODAY (refactor step "4a" of docs/extensions/plan.md):

* identity — id (the arm_type string on disk and on the wire),
  label (a display name), indefinite_label (for prose: "a Metal arm");
* shape — joints_per_arm, supports_bimanual;
* capability flags — uses_feetech_bus, supports_auto_calibration,
  uses_zero_calibration, supports_dagger (the predicates in
  makermodslab/arm_capabilities.py read these; their docstrings carry the
  hardware reasoning and stay the reference for what each flag gates);
* lerobot registry keys — single_robot_type / bimanual_robot_type,
  the RobotConfig choice-registry names this family's followers register
  under, used both to name --robot.type for a subprocess and to read the
  family back off a built config;
* dataset provenance — robot_type_markers, substrings that identify the
  family inside a dataset's free-form meta/info.json robot_type;
* calibration libraries — leader_library_attr / follower_library_attr
  name the makermodslab.utils.config path constants for this family's
  libraries (resolved at CALL time so a relocated or monkeypatched path is
  honoured), and default_calibration_name is the naming rule for a robot
  record's empty calibration slot;
* device construction — build_single_configs / build_bimanual_configs
  assemble the lerobot follower/leader config pair for a session. Device
  config classes are imported INSIDE these methods, never at module import:
  the CAN families' configs drag in python-can / motorbridge, and this package
  is imported by utils.config, i.e. by everything.

* calibration procedure — zero_pose_instructions (the pose text the
  zero-calibration flow shows; only the CAN families have one) and the
  single-device configs single_follower_config / single_leader_config that
  calibration and crash recovery connect ONE arm with (no leader/follower
  pair, no cameras).

What step "4b" still adds (deliberately NOT declared yet): port probing and
motion identify, the pre-torque preflight (identity fingerprint, motor-power
cap), the stop-path pair return_to_rest / release_torque, and the telemetry
kind the loops broadcast (URDF joints vs degrees by motor name). Until then
those flows keep their own per-family modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Attributes a family MUST set. Checked by registry.register (an ABC can
# only enforce methods, and most of a family is data), so an incomplete family
# is refused at registration with the missing names spelled out rather than
# failing later in whichever flow first reads the gap.
REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "id",
    "label",
    "short_label",
    "indefinite_label",
    "joints_per_arm",
    "supports_bimanual",
    "uses_feetech_bus",
    "supports_auto_calibration",
    "uses_zero_calibration",
    "supports_dagger",
    "single_robot_type",
    "bimanual_robot_type",
    "robot_type_markers",
    "leader_library_attr",
    "follower_library_attr",
)


class ArmFamily(ABC):
    """One arm family. Subclass, set every attribute in REQUIRED_ATTRIBUTES,
    implement the two builders, and register ONE instance.

    Instances are stateless singletons: the registry hands the same object to
    every caller, so nothing here may hold per-session state.
    """

    # --- identity ---------------------------------------------------------
    id: str
    label: str
    # The one-word family name status messages and logs splice into prose
    # ("Connecting to the Metal follower arm..."). Not localized — the backend
    # never is (see frontend/docs/localization.md).
    short_label: str
    # For prose a user reads ("recorded on a Metal arm").
    indefinite_label: str

    # --- shape ------------------------------------------------------------
    # Flat proprioceptive width of ONE follower arm — one dim per joint. The
    # number a bimanual robot doubles, and the number a trained checkpoint's
    # observation.state must match.
    joints_per_arm: int
    supports_bimanual: bool

    # --- capability flags -------------------------------------------------
    uses_feetech_bus: bool
    supports_auto_calibration: bool
    uses_zero_calibration: bool
    supports_dagger: bool

    # --- lerobot RobotConfig choice-registry keys ---------------------------
    # Registered type STRINGS rather than classes so the family can be read
    # back off a config (or written into a CLI) without importing the device
    # stack.
    single_robot_type: str
    bimanual_robot_type: str

    # --- dataset provenance -------------------------------------------------
    # Lower-case substrings that identify this family in a dataset's
    # robot_type string. Matched greedily and in registry order with the
    # default family LAST, because its markers are the loosest (so_leader).
    robot_type_markers: tuple[str, ...]

    # --- calibration libraries -----------------------------------------------
    # Names of makermodslab.utils.config module constants, not paths:
    # lerobot derives a device's calibration directory from the device CLASS's
    # name, the constants pin those names, and the test fixtures redirect
    # them by monkeypatching the constant — a path captured at import would
    # silently ignore the patch.
    leader_library_attr: str
    follower_library_attr: str

    def robot_config_types(self) -> frozenset[str]:
        """Every lerobot RobotConfig type string a follower of this family registers under."""
        return frozenset({self.single_robot_type, self.bimanual_robot_type})

    def robot_cli_type(self, bimanual: bool) -> str:
        """The --robot.type= value for a subprocess driving this family."""
        return self.bimanual_robot_type if bimanual else self.single_robot_type

    def leader_calibration_dir(self) -> str:
        """The calibration library dir holding this family's LEADER configs (resolved now)."""
        from ..utils import config

        return getattr(config, self.leader_library_attr)

    def follower_calibration_dir(self) -> str:
        """The calibration library dir holding this family's FOLLOWER configs (resolved now)."""
        from ..utils import config

        return getattr(config, self.follower_library_attr)

    def default_calibration_name(self, record_name: str) -> str:
        """The default calibration id for a robot record's empty slot (single mode).

        Mints the family id into the name: the CAN families' Star-leader
        calibrations live in ONE shared library while the presets' zero poses
        differ, so an unsuffixed default would let a Maker robot and a Metal
        robot silently share a zero that is wrong for one of them. The SO-101
        overrides this to keep its historical bare name.
        """
        return f"{record_name}_{self.id}"

    # --- calibration procedure -----------------------------------------------

    def zero_pose_instructions(self, device_type: object | None = None) -> str:
        """The physical pose to ask the user for during a zero-pose calibration.

        Only meaningful when uses_zero_calibration is True; a family calibrated
        by a range sweep has no zero pose and answers "". ``device_type`` is
        "teleop" (the leader) or "robot" (the follower): the two poses differ,
        and on the CAN families they are OPPOSITES on the gripper.
        """
        return ""

    @abstractmethod
    def single_follower_config(self, port: str, config_id: str):
        """A config for ONE follower arm, alone — no leader, no cameras.

        What the zero-pose calibration and the crash-recovery torque release
        connect with: calibration never opens a camera (holding one for a flow
        that is pure motor work would only block it for everyone else), and
        recovery reaches the bus through a throwaway id.
        """

    @abstractmethod
    def single_leader_config(self, port: str, config_id: str):
        """A config for ONE leader arm, alone — the family's own preset, so the
        calibration file this run writes carries THIS follower's joint ranges."""

    # --- device construction -------------------------------------------------

    @abstractmethod
    def build_single_configs(self, request: Any, cameras: dict | None, leader_id: str, follower_id: str):
        """Return (robot_config, teleop_config) for one leader/follower pair.

        leader_id / follower_id are the calibration ids already staged
        into lerobot's expected locations by the caller. When cameras is
        None the follower config must be built WITHOUT a cameras kwarg
        (teleoperation); otherwise the dict goes on the follower.
        """

    @abstractmethod
    def build_bimanual_configs(
        self,
        request: Any,
        cameras: dict | None,
        base: str,
        leader_staging: str,
        follower_staging: str,
    ):
        """Return (robot_config, teleop_config) for a bimanual pair.

        base is the staged "<base>_left/right.json" id and the two staging
        dirs are the per-device calibration_dir roots. Cameras go on the
        LEFT follower arm config, never the bimanual config's top level — that
        is what keeps a bimanual dataset's camera feature keys identical across
        families. Same cameras is None rule as the single builder.
        """
