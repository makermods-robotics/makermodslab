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

* port detection — probe_ports (which ports answer which protocol, no user
  gesture; only the CAN families have one, because their two halves speak
  different protocols) and identify_by_motion (the hand-swing gesture that
  tells one arm from its twin), with the two facts the CAN probes branch on:
  follower_probe_protocol and motion_identify_energizes_follower.

* preflight — verify_identity (the read-only EEPROM fingerprint that
  catches a swapped or mis-assigned arm BEFORE a calibration is written into
  it) and prepare_follower_registers (re-seed the session torque limit and
  clear a leftover speed cap). Both are Feetech register work; the CAN
  families answer with nothing to do, because a RobStride/Damiao motor
  keeps its zero internally and takes its drive effort from the MIT gains
  connect() writes.

* stop path — capture_rest_poses at session start, return_to_rest before
  torque is released, release_torque last. The ORDER is the core's and every
  flow that energizes an arm keeps it (teleoperation, recording, replay;
  inference gets it from lerobot's --return_to_initial_position); the
  MECHANISM is the family's: a Feetech profile-velocity move for the SO-101,
  an interpolated MIT setpoint judged by convergence for the CAN arms, which
  have no brakes and drop under gravity if released anywhere but near rest.

* telemetry — telemetry_kind, what the live loops broadcast for the
  viewer: "urdf" (normalized joint fractions that drive the 3D model) or
  "degrees" (angles by motor name for the numeric readout, because no URDF
  ships for that family yet).

Adding a family means adding a module here and registering it; nothing
outside this package compares an arm type to a literal
(tests/test_arm_registry.py sweeps every module for one).
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
    "follower_probe_protocol",
    "motion_identify_energizes_follower",
    "telemetry_kind",
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

    # --- port detection ---------------------------------------------------------
    # The protocol the follower probe speaks ("robstride", "damiao"), or None
    # for a family whose two halves share one protocol and so cannot be told
    # apart by asking — the SO-101's leader and follower are both Feetech
    # serial, which is why it identifies by gesture instead.
    follower_probe_protocol: str | None
    # True when merely OPENING the follower's bus to watch its joints would
    # energize the motors (the Damiao handshake is the enable command). The
    # motion-identify gesture is refused for such a follower rather than run
    # behind the user's back; the leader side still works.
    motion_identify_energizes_follower: bool

    # --- telemetry --------------------------------------------------------------
    # What the teleop loop broadcasts each tick: "urdf" — normalized joint
    # fractions under `joints`, driving the 3D viewer — or "degrees" — angles
    # by motor name under `joints_deg`, for the numeric readout a family
    # without a URDF gets (see teleoperate.get_maker_joint_degrees).
    telemetry_kind: str

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

    @property
    def calibration_name_suffix(self) -> str:
        """What default_calibration_name appends to a robot record's name.

        Mints the family id into the name: the CAN families' Star-leader
        calibrations live in ONE shared library while the presets' zero poses
        differ, so an unsuffixed default would let a Maker robot and a Metal
        robot silently share a zero that is wrong for one of them. The SO-101
        overrides this to "" to keep its historical bare name. Published in
        the arms manifest so the UI predicts the same id the server mints —
        one property serves both, so the two cannot drift.
        """
        return f"_{self.id}"

    def default_calibration_name(self, record_name: str) -> str:
        """The default calibration id for a robot record's empty slot (single mode)."""
        return f"{record_name}{self.calibration_name_suffix}"

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

    # --- port detection ---------------------------------------------------------

    async def probe_ports(self, ports: list[str] | None = None) -> dict:
        """Classify ports by which protocol answers on them, with no user gesture.

        Same response shape as maker_ports.probe_maker_ports. A family without
        a protocol probe answers a plain refusal naming the alternative.
        """
        return {
            "success": False,
            "follower_ports": [],
            "leader_ports": [],
            "unknown_ports": list(ports or []),
            "message": (
                f"Protocol probing is not available for the {self.short_label}: its leader and follower "
                "speak the same protocol. Identify the arm by motion instead."
            ),
        }

    @abstractmethod
    async def identify_by_motion(self, device_type: str, ports: list[str] | None = None) -> dict:
        """Report which port saw a hand gesture. Read-only on every family.

        ``device_type`` is "robot" (the follower) or "teleop" (the leader);
        a family whose two halves share one bus driver may ignore it.
        """

    # --- preflight, before any torque --------------------------------------------

    def verify_identity(self, pairs: Any, *, skip: bool = False, config_names: Any = None) -> list[str]:
        """Check each connected (device, side) pair against its assigned calibration.

        Runs after the buses connect and strictly before anything writes a
        calibration or enables torque; raises on a hard mismatch, returns
        warn-but-allow messages otherwise. The default is "nothing to
        compare": only a family whose servos hold a fingerprint in EEPROM
        (the SO-101) can answer, and it overrides this.
        """
        return []

    def prepare_follower_registers(self, robot: Any, label: str | None = None) -> list[str]:
        """Put a connected FOLLOWER's drive registers into session state.

        Re-seed the RAM torque limit from EEPROM (clearing any cap a previous
        auto-calibration left) and clear a leftover Goal_Velocity speed cap.
        Followers only, never the human-held leader. Returns warnings; a
        failed write degrades rather than aborts. The default is a no-op: a
        CAN follower's drive effort is its MIT gains, set at connect().
        """
        return []

    # --- stop path ----------------------------------------------------------------
    # The one sequence every energizing flow follows: capture where the arm
    # started, drive it back there before torque goes, then release. The core
    # calls these in that order on every stop path — including error paths
    # and the expiry watchdog — and never skips a step; the family decides
    # how each is done for its hardware.

    @abstractmethod
    def capture_rest_poses(self, robot: Any, *, include_gripper: bool = False) -> list[tuple[Any, dict]]:
        """Where each follower arm is right now, as (handle, pose) pairs.

        Called once at session start, after connect and before anything
        moves. FOLLOWERS only, never the human-held leader. One pair per
        drivable arm (a bimanual robot yields two). The gripper is excluded by
        default — at stop time it may be holding something, and returning it
        to its (likely open) starting width would drop that object
        mid-return; replay passes include_gripper=True because the dataset
        drives the gripper and its start width is part of the pose restored.
        Never raises: a session must not fail to start over this.
        """

    @abstractmethod
    def return_to_rest(self, rest_poses: list[tuple[Any, dict]], abort_event: Any = None) -> None:
        """Drive every captured arm back to its pose, concurrently, then return.

        Runs on a NORMAL stop, immediately before release_torque, and returns
        only when every arm has finished (arrived, settled, stalled, hit the
        ceiling, or been cut short). ``abort_event`` (a second stop press)
        cuts every arm's return short promptly, leaving it nearer rest than it
        started. Best-effort, never raises: every outcome falls through to
        the unconditional release.
        """

    @abstractmethod
    def release_torque(self, device: Any, label: str = "device") -> list[str]:
        """Disable torque on every motor of ``device``, loudly on failure.

        The last step before disconnect, and belt-and-braces on purpose:
        lerobot's disconnect disables torque too, but any exception on the way
        leaves the arm energized (rigid). Returns problem descriptions, empty
        when every motor released; each is also logged at ERROR level naming
        the port. Never raises.
        """

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
