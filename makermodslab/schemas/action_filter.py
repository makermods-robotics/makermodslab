"""Validated, per-run action filtering for the native RTC worker."""

from pydantic import Field, model_validator

from .remote_network import RobotNetworkOptions


class ActionFilterOptions(RobotNetworkOptions):
    lpf_hz: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    lpf_order: int = Field(default=2, ge=1, le=4, strict=True)

    @model_validator(mode="after")
    def validate_action_filter(self):
        if self.lpf_hz > 0:
            if getattr(self, "engine", None) != "rtc":
                raise ValueError("Action filtering requires the rtc engine")
            if self.lpf_hz >= getattr(self, "fps", 0) / 2:
                raise ValueError("Action filter cutoff must be below half the control FPS")
        return self
