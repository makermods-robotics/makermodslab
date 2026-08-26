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

"""Schemas for the /api/v1/sessions surface (makermodslab/sessions.py).

Unlike its siblings this module also carries the REQUEST side: the start body
and the per-kind option models. The options are the kind-specific fields a
client must supply because they cannot come from the robot record — everything
hardware-shaped (ports, configs, mode, right-arm fields, cameras) is resolved
server-side from the record, which is the point of the surface. Each options
model is ``extra="forbid"`` so a field sent under the wrong kind (or a typo'd
one) is a loud 422, never a silently ignored knob.

The response models mirror the dicts SessionTracker builds; the shape
authority is the tracker (see the package docstring's fidelity rules).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# The frame-size model the inference launch flow already speaks — reused, not
# duplicated, so the sessions surface can never drift from InferenceRequest.
from makermodslab.rollout import PolicyCameraDims

__all__ = [
    "CurrentSessionResponse",
    "EndedSessionInfo",
    "InferenceOptions",
    "PolicyCameraDims",
    "RecordingOptions",
    "ReplayOptions",
    "SessionInfo",
    "SessionStartBody",
    "SessionStartResponse",
    "SessionStopResponse",
    "TeleoperationOptions",
]


# --- requests ---------------------------------------------------------------


class TeleoperationOptions(BaseModel):
    """Teleoperation needs nothing beyond the robot record."""

    model_config = ConfigDict(extra="forbid")

    skip_identity_check: bool = False


class RecordingOptions(BaseModel):
    """Dataset-shaped fields of record.py's RecordingRequest. Cameras are NOT
    here: they resolve server-side from the robot record, as they already do
    on the legacy endpoint."""

    model_config = ConfigDict(extra="forbid")

    dataset_repo_id: str
    single_task: str
    num_episodes: int = 5
    episode_time_s: int = 30
    reset_time_s: int = 10
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    tags: list[str] = []
    private: bool = False
    resume: bool = False
    streaming_encoding: bool = True
    skip_identity_check: bool = False


class InferenceOptions(BaseModel):
    """Policy-shaped fields of rollout.py's InferenceRequest. `camera_bindings`
    maps policy-expected camera names to robot-record camera names — the
    devices themselves still come from the record, server-side."""

    model_config = ConfigDict(extra="forbid")

    policy_ref: str
    task: str = ""
    camera_bindings: dict[str, str] = {}
    camera_dims: dict[str, PolicyCameraDims] = {}
    duration_s: int = 60
    checkpoint_state_dim: int | None = None
    eval_episodes: int = 1
    skip_identity_check: bool = False


class ReplayOptions(BaseModel):
    """Episode selection for replay.py's ReplayRequest."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    episode_index: int
    skip_identity_check: bool = False


class SessionStartBody(BaseModel):
    """POST /api/v1/sessions. `options` is validated against the kind's model
    above in the handler (422 request.validation on mismatch) — a plain dict
    here keeps the error a single coded shape instead of a four-armed union
    blob."""

    kind: Literal["teleoperation", "recording", "inference", "replay"]
    robot: str
    owner: str | None = None
    options: dict[str, Any] = {}


# --- responses (shape authority: sessions.SessionTracker) -------------------


class SessionInfo(BaseModel):
    """Identity of the current session (SessionTracker._current). `robot` and
    `owner` are known only for sessions started through /api/v1/sessions —
    legacy-started sessions carry null (the tracker never guesses)."""

    id: str
    kind: str
    robot: str | None
    owner: str | None
    started_at: float
    revision: int
    phase: str | None


class EndedSessionInfo(BaseModel):
    """The last_ended summary (SessionTracker._last_ended). `phase` is the
    phase carried by the release event — the session's final phase."""

    id: str
    kind: str
    ended_at: float
    phase: str | None


class SessionStartResponse(BaseModel):
    session: SessionInfo


class CurrentSessionResponse(BaseModel):
    session: SessionInfo | None
    last_ended: EndedSessionInfo | None


class SessionStopResponse(BaseModel):
    """`result` is the kind's existing stop handler's response, verbatim —
    rich per-kind status stays on the feature endpoints this phase."""

    session: SessionInfo
    result: dict[str, Any]
