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

"""Response models for the "remote" route group: the station's hosting
descriptor (remote_host.py) and the operator's remote-teleoperation status
(remote_teleoperate.py). Shape authority: the two modules' status handlers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = [
    "HostingCamera",
    "HostingDescriptor",
    "HostingStatusResponse",
    "RemoteStation",
    "RemoteTeleoperationMetrics",
    "RemoteTeleoperationStatusResponse",
]


class HostingCamera(BaseModel):
    name: str
    width: int
    height: int


class HostingDescriptor(BaseModel):
    """What an operator needs to join this station's room and agree on the
    wire contract (remote_host.build_descriptor). `motors` are the bare
    motor names (bimanual: `left_`/`right_` prefixed) — Portal's state and
    action schema; both peers must declare the same set in the same order.
    `joint_ranges_deg` is the SO-101 follower's calibrated full travel per
    body joint, so the OPERATOR can render the remote arm on the URDF viewer
    without holding the station's calibration file; absent for the CAN
    arms, whose angles are already degrees."""

    robot: str
    arm_type: str
    mode: str
    room: str
    url: str
    fps: int
    video_codec: Literal["H264", "MJPEG", "PNG", "RAW"]
    motors: list[str]
    cameras: list[HostingCamera]
    joint_ranges_deg: dict[str, float]
    active_operator: str | None


class HostingStatusResponse(BaseModel):
    """remote_host.handle_hosting_status — the descriptor rides only while a
    hosting session is live (null otherwise); the rest mirrors the
    teleoperation status payload's post-session fields."""

    hosting_active: bool
    hosting: HostingDescriptor | None
    releasing: bool
    last_cleanup_error: str | None
    outcome: str | None
    error: str | None
    hint: str | None
    message: str


class RemoteStation(BaseModel):
    instance_id: str
    name: str | None
    url: str


class RemoteTeleoperationMetrics(BaseModel):
    """Portal transport metrics, in ms (None until the first sample)."""

    rtt_ms_last: float | None
    rtt_ms_mean: float | None
    rtt_ms_p95: float | None
    observations: int
    states_dropped: int


class RemoteTeleoperationStatusResponse(BaseModel):
    """remote_teleoperate.handle_remote_teleoperation_status."""

    remote_teleoperation_active: bool
    station: RemoteStation | None
    room: str | None
    cameras: list[str]
    metrics: RemoteTeleoperationMetrics | None
    last_cleanup_error: str | None
    outcome: str | None
    error: str | None
    hint: str | None
    message: str
