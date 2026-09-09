"""Validated network settings shared by API requests and launchers."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RegionChoice = Literal[
    "auto", "us-west", "us-east", "us-central", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx"
]


class GpuNetworkOptions(BaseModel):
    region: RegionChoice = "us-west"
    tolerance: float = Field(default=1.5, ge=0.1, le=5, allow_inf_nan=False)


class RobotNetworkOptions(BaseModel):
    video_quality: int = Field(default=90, ge=1, le=100, strict=True)
    video_bitrate_kbps: int = Field(default=4000, ge=256, le=20000, strict=True)
    camera_send_hz: float = Field(default=0, ge=0, le=120, allow_inf_nan=False)
    latency_k: float = Field(default=1.5, ge=0, le=5, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_camera_send_rate(self):
        if self.camera_send_hz > 0:
            if getattr(self, "engine", None) != "rtc":
                raise ValueError("Camera send rate requires the rtc engine")
            if self.camera_send_hz > getattr(self, "fps", 0):
                raise ValueError("Camera send rate cannot exceed the control rate")
        return self
