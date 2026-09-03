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

"""Response models for the "system" route group (health, auth, optional
extras, updates, port/camera discovery). See the package docstring for the
fidelity rules; the shape authority is always the handler, named next to each
model. Update-check/update and the extra-install shapes are re-exported from
the modules whose handlers already build their responses from these models,
so the schema cannot drift from the wire format.
"""

from __future__ import annotations

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
