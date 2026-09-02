"""Typed v1 contracts for remote teleoperation and owner-fed servo health."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RemoteRobotConfigBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    robot_name: str = Field(min_length=1, max_length=256)
    bind_address: str = Field(min_length=1, max_length=64)
    control_port: int = Field(ge=1, le=65535)
    udp_port: int = Field(ge=1, le=65535)
    tls_certificate_path: str = Field(min_length=1, max_length=4096)
    tls_private_key_path: str = Field(min_length=1, max_length=4096)
    leader_calibration_id: str = Field(min_length=1, max_length=128)
    leader_calibration_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_rate_hz: int = Field(default=50, ge=10, le=100)
    action_watchdog_ms: int = Field(default=200, ge=20, le=2000)
    first_action_deadline_ms: int = Field(default=1000, ge=20, le=5000)
    control_deadline_ms: int = Field(default=1000, ge=100, le=5000)
    browser_deadline_ms: int = Field(default=2000, ge=100, le=10000)
    max_velocity_per_s: float = Field(default=60.0, gt=0)
    max_acceleration_per_s2: float = Field(default=300.0, gt=0)
    recording_enabled: bool = True


class RemoteOperatorConfigBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    robot_id: str = Field(min_length=1, max_length=128)
    leader_robot_name: str = Field(min_length=1, max_length=256)
    control_uri: str = Field(min_length=1, max_length=512)
    certificate_fingerprint: str = Field(min_length=1, max_length=128)
    action_rate_hz: int = Field(default=50, ge=10, le=100)


class RemoteConfigurationBody(BaseModel):
    """Exactly one role configuration; saving it never enables that role."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["robot", "operator"]
    robot: RemoteRobotConfigBody | None = None
    operator: RemoteOperatorConfigBody | None = None

    @model_validator(mode="after")
    def selected_role_only(self) -> RemoteConfigurationBody:
        if self.role == "robot" and (self.robot is None or self.operator is not None):
            raise ValueError("robot role requires only the robot configuration")
        if self.role == "operator" and (self.operator is None or self.robot is not None):
            raise ValueError("operator role requires only the operator configuration")
        return self


class RemotePairBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairing_token: str = Field(min_length=20, max_length=256)
    operator_label: str = Field(min_length=1, max_length=128)


class RemoteCommissionBody(BaseModel):
    """Physical safeguards must be true before the follower is opened."""

    model_config = ConfigDict(extra="forbid")

    arm_secured: Literal[True]
    workspace_clear: Literal[True]
    physical_power_cutoff_reachable: Literal[True]
    acknowledge_live_torque_enable_risk: Literal[True]


class RemoteStopBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="api_stop", min_length=1, max_length=128)


class RemoteRuntimeStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str
    configured: bool
    role: Literal["robot", "operator"] | None
    config: dict[str, Any] | None
    runtime_enabled: bool
    live_hardware_enabled: bool
    state: str
    runtime: dict[str, Any] | None
    simulation: dict[str, Any]
    commissioning: dict[str, Any]
    durable_fault: dict[str, Any]
    hardware_registry: dict[str, Any]


class RemoteJointLimitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: float
    maximum: float
    max_velocity_per_s: float = Field(gt=0)
    max_acceleration_per_s2: float = Field(gt=0)


class RemoteSimulationStartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    rig_id: str
    rig_digest: str
    leader_calibration_id: str
    leader_calibration_digest: str
    follower_calibration_id: str
    follower_calibration_digest: str
    joint_names: list[str] = Field(min_length=1, max_length=32)
    units: list[str] = Field(min_length=1, max_length=32)
    limits: dict[str, RemoteJointLimitBody]
    tick_hz: int = Field(default=50, ge=10, le=200)
    watchdog_ms: int = Field(default=200, ge=20, le=2000)


class RemoteSimulationCredentials(BaseModel):
    key_id: str
    action_key_base64: str


class RemoteSimulationStartResponse(BaseModel):
    simulation_only: bool
    session: dict[str, Any]
    credentials: RemoteSimulationCredentials
    status: dict[str, Any]


class RemoteSimulationDatagramBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datagram_base64: str


class RemoteSimulationStopBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="api_stop", min_length=1, max_length=128)


class RemoteSimulationStatusResponse(BaseModel):
    simulation_only: bool
    live_hardware_enabled: bool
    state: str
    status: dict[str, Any] | None
    recorded_events: int


class ServoHealthResponse(BaseModel):
    """Snapshot shape is additive; individual register values may be null."""

    model_config = ConfigDict(extra="allow")

    protocol_version: str
    source_revision: str
    read_only: bool
    available: bool
    complete: bool
    owner: str | None
    arms: list[dict[str, Any]]
    last_error: str | None
    maintenance: dict[str, Any]
