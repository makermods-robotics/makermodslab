"""Live gripper state for the robot configuration API."""

from pydantic import BaseModel


class GripperStatus(BaseModel):
    port: str
    enabled: bool
    limit_a: float | None
    effective_current_a: float
    temperature_c: float
    fault: str | None
    hold_torque_nm: float | None
    holding: bool
    measured_torque_nm: float | None


class GripperStatusResponse(BaseModel):
    grippers: list[GripperStatus]
