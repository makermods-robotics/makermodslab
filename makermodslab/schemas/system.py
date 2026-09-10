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

"""Response models for the "system" route group (health, auth, optional
extras, updates, port/camera discovery). See the package docstring for the
fidelity rules; the shape authority is always the handler, named next to each
model. Update-check/update and the extra-install shapes are re-exported from
the modules whose handlers already build their responses from these models,
so the schema cannot drift from the wire format.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Handlers in makermodslab/update.py return these models' dumps directly.
from makermodslab.update import UpdateResult, UpdateStatus

# Handlers in makermodslab/utils/system.py return dicts with exactly these
# fields (InstallManager.start / .get_status and handle_get_*_extra).
from makermodslab.utils.system import (
    ExtraStatus,
    InstallStartResponse,
    InstallStatusResponse,
    RestartResponse,
)

__all__ = [
    "ArmCalibrationInfo",
    "ArmCalibrationSide",
    "ArmCalibrationSummary",
    "ArmCapabilities",
    "ArmFamiliesResponse",
    "ArmFamilyInfo",
    "AvailableCamerasResponse",
    "AvailablePortsResponse",
    "CameraInfo",
    "ExtraStatus",
    "HealthCapabilities",
    "HealthResponse",
    "HfAuthStatusResponse",
    "HfLoginResponse",
    "InstallStartResponse",
    "InstallStatusResponse",
    "MakerIdentifyArmResponse",
    "MakerProbePortsResponse",
    "ReleaseCanTorqueResponse",
    "PolicyExtraStatus",
    "PolicyOptimizerDefaultsResponse",
    "PolicyOptimizerPreset",
    "RestartResponse",
    "RobotPortResponse",
    "SupplyVoltageResponse",
    "UpdateResult",
    "UpdateStatus",
]


class HealthCapabilities(BaseModel):
    """Capabilities block of /health (server.py health_check).

    The health doc grows additively as the node registry needs more
    (gpu, hardware inventory, …) — extra="allow" keeps keys the handler adds
    before this model learns about them, instead of silently filtering them
    out of the handshake. Absent-or-present keys (`gpu`, and `sfu` — the
    bundled LiveKit server's signalling URL, `{"url": "ws://host:7880"}`,
    only when started with --sfu) are deliberately NOT declared here: a
    declared optional would materialize as null on nodes without one, and
    the contract is "absent means none/unknown".
    """

    model_config = ConfigDict(extra="allow")

    serves_ui: bool
    accepts_jobs: bool


class HealthResponse(BaseModel):
    """Node identity + capability document (server.py health_check)."""

    status: str
    message: str
    version: str
    instance_id: str
    capabilities: HealthCapabilities


class HfAuthStatusResponse(BaseModel):
    """utils/hf_auth.py handle_hf_auth_status — username is null (not absent)
    when unauthenticated."""

    authenticated: bool
    username: str | None
    orgs: list[str]
    writable_namespaces: list[str]
    login_command: str


class HfLoginResponse(BaseModel):
    """utils/hf_auth.py handle_hf_login (success path only; failures raise)."""

    authenticated: bool
    username: str
    orgs: list[str]
    login_command: str


class PolicyExtraStatus(BaseModel):
    """utils/system.py handle_get_policy_extra — every key is always set
    (core policies report needs_extra=False with empty-string fields)."""

    policy_type: str
    needs_extra: bool
    available: bool
    package: str
    install_target: str
    install_hint: str


class AvailablePortsResponse(BaseModel):
    """server.py get_available_ports. Success carries ports, failure carries
    message — never both, never null; the route excludes None to keep each
    branch's exact keys."""

    status: str
    ports: list[str] | None = None
    message: str | None = None


class CameraInfo(BaseModel):
    """One camera from the platform enumerators in server.py — unique_id is
    the AVFoundation uniqueID, present on macOS only (absent elsewhere, never
    null; the route excludes None)."""

    index: int
    name: str
    available: bool
    unique_id: str | None = None


class AvailableCamerasResponse(BaseModel):
    """server.py get_available_cameras. cameras is always present (empty list
    on failure); message only on the error branch — the route excludes None."""

    status: str
    cameras: list[CameraInfo]
    message: str | None = None


class SupplyVoltageResponse(BaseModel):
    """motor_power.py read_supply_voltage. Success carries voltage, failure
    carries message — never both, never null; the route excludes None."""

    success: bool
    voltage: float | None = None
    message: str | None = None


class RobotPortResponse(BaseModel):
    """server.py get_robot_port — saved_port is null (not absent) when no
    port file exists, so None must NOT be excluded on this route."""

    status: str
    saved_port: str | None
    default_port: str


class PolicyOptimizerPreset(BaseModel):
    """One entry of /policy-optimizer-defaults `defaults` (server.py
    get_policy_optimizer_defaults); lerobot OptimizerConfig presets type all
    three numbers as float."""

    optimizer: str
    lr: float
    weight_decay: float
    grad_clip_norm: float


class PolicyOptimizerDefaultsResponse(BaseModel):
    """server.py get_policy_optimizer_defaults — `defaults` values are null
    (legitimately) for unavailable policies or unreadable presets, so None
    must NOT be excluded on this route."""

    defaults: dict[str, PolicyOptimizerPreset | None]
    available: dict[str, bool]


class MakerProbePortsResponse(BaseModel):
    """maker_ports.probe_maker_ports — which ports answered which protocol.

    Every list is always present (empty rather than absent) so a client can
    read them unconditionally; `message` is always a human-readable summary,
    including on the nothing-found path.
    """

    success: bool
    follower_ports: list[str]
    leader_ports: list[str]
    unknown_ports: list[str]
    message: str


class ReleaseCanTorqueResponse(BaseModel):
    """can_recovery.handle_release_can_torque — the crash-recovery release.

    `problems` is always present (empty on success) so a client can render
    the loud per-bus alarms unconditionally.
    """

    success: bool
    message: str
    problems: list[str]


class MakerIdentifyArmResponse(BaseModel):
    """maker_ports.identify_maker_arm_by_motion — which port saw the gesture.

    `port` is absent on failure rather than null, so the route excludes None.
    `skipped` lists ports that could not be opened (usually the other half of
    the rig, which speaks a different protocol).
    """

    success: bool
    message: str
    port: str | None = None
    skipped: list[str] = []
    # The identification of last resort the client should offer next —
    # "wiggle" (can_wiggle.py) when the gesture was refused or found nothing
    # on a family with a gripper wiggle; absent otherwise (never null).
    fallback: Literal["wiggle"] | None = None


class CanGripperWiggleResponse(BaseModel):
    """can_wiggle.wiggle_can_gripper — one port's gripper was jogged (or not).

    `code` is present only on a busy refusal (robot.busy.*), like every
    other hardware handler's refusal dict; the route excludes None.
    """

    success: bool
    message: str
    code: str | None = None


class ArmCalibrationSide(BaseModel):
    """What the config dialog shows BEFORE Start for one device side (the
    family's own calibration_summary, verbatim — the backend is never
    localized): the text and an optional served image."""

    text: str
    image_url: str | None


class ArmCalibrationSummary(BaseModel):
    """The pre-start summary per side; a side is null when the family has
    nothing to show for it."""

    leader: ArmCalibrationSide | None
    follower: ArmCalibrationSide | None


class ArmCalibrationInfo(BaseModel):
    """How a family is calibrated: a range sweep (the SO-101's manual or
    driven flows), a step wizard the family drives (the CAN arms' zero pose),
    or an extension's own panel at `panel_url`. `summary` and `panel_url` are
    null — not absent — when they do not apply, so the route must NOT
    exclude None."""

    kind: Literal["range_sweep", "steps", "panel"]
    summary: ArmCalibrationSummary | None
    panel_url: str | None


class ArmCapabilities(BaseModel):
    """The family's capability flags (arms/base.py), plus two derived from
    its port-detection facts: `supports_port_probe` (a protocol probe exists,
    so no gesture is needed) and `motion_identify_energizes_follower` (the
    follower side of the gesture is refused because opening its bus would
    energize it)."""

    uses_feetech_bus: bool
    supports_auto_calibration: bool
    supports_dagger: bool
    supports_remote_inference: bool
    supports_port_probe: bool
    motion_identify_energizes_follower: bool
    # The family can jog ONE port's gripper so the user sees which arm it is
    # (POST /api/v1/maker/wiggle-gripper) — the identification of last resort
    # when neither the probe nor the gesture can tell two arms apart.
    supports_gripper_soft_limit: bool
    supports_gripper_wiggle: bool


class LeaderOptionInfo(BaseModel):
    """One leader arm a family can be driven by (arms/base.py LeaderOption).

    `id` is what a robot record stores as `leader_kind`; `available` is
    false when this install cannot drive it, with `unavailable_reason`
    naming what to install (null when available); `energized` marks a leader
    that holds torque while the human moves it (the Metal arm's
    gravity-compensated leader): it answers the follower's protocol, refuses
    the gesture, and is returned and released on a stop like a follower.
    `calibration_summary` is the pre-start summary for THIS leader's side
    (null for a family with nothing to summarize).
    """

    id: str
    label: str
    available: bool
    unavailable_reason: str | None
    energized: bool
    calibration_summary: ArmCalibrationSide | None


class ArmFamilyInfo(BaseModel):
    """One entry of the arms manifest (arms/manifest.py describe_family).

    `robot_types` are the lerobot RobotConfig type strings the family's
    followers register under (single, then bimanual); `robot_type_markers`
    the substrings that identify it in a dataset's free-form robot_type;
    `calibration_name_suffix` what the server appends to a robot record's
    name when it mints a default calibration id ("" for the SO-101);
    `image_url` a served image for the create dialog (null for the built-ins,
    whose photos the frontend bundles); `leader_options` the leader arms the
    family can be driven by, default first (`default_leader_kind` names it —
    what a record with no `leader_kind` reads as).
    """

    id: str
    label: str
    short_label: str
    provided_by: str
    joints_per_arm: int
    supports_bimanual: bool
    image_url: str | None
    calibration: ArmCalibrationInfo
    telemetry_kind: Literal["urdf", "degrees"]
    capabilities: ArmCapabilities
    robot_types: list[str]
    robot_type_markers: list[str]
    calibration_name_suffix: str
    default_leader_kind: str
    leader_options: list[LeaderOptionInfo]


class ArmFamiliesResponse(BaseModel):
    """GET /api/v1/arms (server.py list_arm_families) — registry order,
    default family first."""

    arms: list[ArmFamilyInfo]
