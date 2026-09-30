"""Live gripper state for the robot configuration API."""

from pydantic import BaseModel


class GripperStatus(BaseModel):
    port: str
    enabled: bool
    limit_a: float | None
    effective_current_a: float
    temperature_c: float
    fault: str | None
    fault_word: int | None = None
    legacy_firmware: bool | None = None
    hold_torque_nm: float | None
    holding: bool
    measured_torque_nm: float | None
    goal_position_deg: float | None = None
    measured_position_deg: float | None = None
    command_position_deg: float | None = None
    command_kp: float | None = None
    command_kd: float | None = None
    leader_target_deg: float | None = None


class GripperStatusResponse(BaseModel):
    grippers: list[GripperStatus]
