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

from pydantic import BaseModel, ConfigDict, Field

# The frame-size model the inference launch flow already speaks — reused, not
# duplicated, so the sessions surface can never drift from InferenceRequest.
from makermodslab.rollout import PolicyCameraDims

__all__ = [
    "LEASE_TIMEOUT_AUTO_CALIBRATION_S",
    "LEASE_TIMEOUT_DEFAULT_S",
    "LEASE_TIMEOUT_MAX_S",
    "LEASE_TIMEOUT_MIN_S",
    "OWNER_MAX_LENGTH",
    "AutoCalibrationArmOption",
    "AutoCalibrationOptions",
    "CalibrationOptions",
    "CoachingCommandResponse",
    "CurrentSessionResponse",
    "EndedSessionInfo",
    "InferenceOptions",
    "PolicyCameraDims",
    "RecordingOptions",
    "ReplayOptions",
    "SessionCoachingBody",
    "SessionCoachingResponse",
    "SessionHeartbeatBody",
    "SessionHeartbeatResponse",
    "SessionInfo",
    "SessionLeaseInfo",
    "SessionStartBody",
    "SessionStartResponse",
    "SessionStopResponse",
    "TeleoperationOptions",
]

# Lease bounds (inclusive), enforced in sessions.handle_start_session so the
# refusal carries the coded 422 shape (`request.validation`) like the per-kind
# options do — pydantic Field constraints would produce an uncoded 422.
LEASE_TIMEOUT_DEFAULT_S = 60.0
# auto_calibration runs ~60s+ on real hardware (measured), so the generic
# default leaves no closed-tab margin for the reopen-and-recover flow; give it
# headroom. Applies only when the client sends no explicit lease_timeout_s.
LEASE_TIMEOUT_AUTO_CALIBRATION_S = 90.0
LEASE_TIMEOUT_MIN_S = 10.0
LEASE_TIMEOUT_MAX_S = 600.0
OWNER_MAX_LENGTH = 128


# --- requests ---------------------------------------------------------------


class TeleoperationOptions(BaseModel):
    """Teleoperation needs nothing beyond the robot record."""

    model_config = ConfigDict(extra="forbid")

    skip_identity_check: bool = False


class HostingOptions(BaseModel):
    """Station side of remote teleoperation (remote_host.py): publish this
    robot's follower + cameras into the node's LiveKit room and execute the
    active operator's actions. `fps` paces the whole loop (frames, state and
    action application — one knob); `video_codec` is the wire transport for
    frames and MUST match what the operator will request, so it is published
    in the hosting descriptor and the operator copies it from there."""

    model_config = ConfigDict(extra="forbid")

    fps: int = Field(30, ge=5, le=60)
    video_codec: Literal["H264", "MJPEG", "PNG", "RAW"] = "H264"
    skip_identity_check: bool = False


class RemoteTeleoperationOptions(BaseModel):
    """Operator side (remote_teleoperate.py): drive a STATION's follower with
    this node's leader. `station` is the peer's instance id from the node
    registry; everything else (room, codec, fps, motors, cameras) comes from
    the station's live hosting descriptor."""

    model_config = ConfigDict(extra="forbid")

    station: str = Field(min_length=1, max_length=64)
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
    # The two remaining policy-shaped knobs of InferenceRequest — without them
    # a UI-selected RTC engine / ACT temporal ensembling would be silently
    # dropped by this surface (extra="forbid" refuses them; omitting them runs
    # the arm under a different engine than the user chose).
    inference_engine: Literal["sync", "rtc"] = "sync"
    temporal_ensemble_coeff: float | None = None
    # Coaching (DAgger) — the third inference session shape. `coaching=True`
    # records each leader-arm takeover as one episode of a corrections dataset.
    #
    # Policy-shaped like the rest of this model: the LEADER arms a coaching
    # session drives are hardware, so they resolve from the robot record
    # server-side — see `_build_inference_request` in sessions.py. That also
    # means a coaching session is NOT follower-only, unlike a plain rollout or
    # an eval: it opens the leader bus, so the start-time readiness gate has to
    # cover the leader arm too (see `handle_start_session`).
    #
    # `target_corrections` and `coaching_dataset_name` (given without the
    # mandatory `rollout_` prefix, which is applied server-side) are clamped and
    # validated in rollout, not here.
    coaching: bool = False
    target_corrections: int = 10
    coaching_dataset_name: str = ""


class ReplayOptions(BaseModel):
    """Episode selection for replay.py's ReplayRequest."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    episode_index: int
    skip_identity_check: bool = False


class CalibrationOptions(BaseModel):
    """Manual step-by-step calibration (calibrate.py's CalibrationRequest).

    Calibration is the flow that CREATES the record's hardware picture, so —
    unlike every other kind — `port` and `config_file` may ride in the options:
    a fresh robot has no saved port yet (the UI's port pick is a draft until
    Save, and the backend writes port+config back into the record only on a
    successful calibration). Omitted, both resolve from the record: the slot's
    saved port, and the slot's assigned config name (else the robot's default
    name for that slot — "<robot>_<arm>" bimanual, "<robot>" single)."""

    model_config = ConfigDict(extra="forbid")

    # Which physical arm slot to calibrate. "robot" = follower, "teleop" =
    # leader; "left" is also the single-arm pair.
    device_type: Literal["robot", "teleop"]
    arm: Literal["left", "right"] = "left"
    port: str | None = None
    config_file: str | None = None
    # Must be explicitly true to replace an existing config file of the name.
    overwrite: bool = False


class AutoCalibrationArmOption(BaseModel):
    """One arm slot in an auto-calibration run — the caller-chosen half of
    auto_calibrate.py's AutoCalibrationBatchArm; `port`/`config_file` resolve
    from the record like CalibrationOptions' (same setup-flow reasoning)."""

    model_config = ConfigDict(extra="forbid")

    device_type: Literal["robot", "teleop"]
    arm: Literal["left", "right"] = "left"
    port: str | None = None
    config_file: str | None = None


class AutoCalibrationOptions(BaseModel):
    """Auto-calibration (auto_calibrate.py). One options shape covers the
    single-arm AND concurrent multi-arm flows: `arms` is a list of per-arm
    slots, and the start wrapper always builds AutoCalibrationBatchRequest —
    the batch of one is exactly how the UI already runs a single arm, and the
    aggregate `auto_calibration` session-event kind makes a batch ONE session
    (arms finishing are phase transitions, not releases), so the batch fits
    the one-session model cleanly."""

    model_config = ConfigDict(extra="forbid")

    arms: list[AutoCalibrationArmOption] = Field(min_length=1)
    # Calibration drive torque (percent, clamped 10-100 server-side), applied
    # to every arm. None resolves to the record's persisted motor_power — the
    # UI passes its slider draft so what you see is what it drives at.
    motor_power: int | None = None
    overwrite: bool = False


class SessionStartBody(BaseModel):
    """POST /api/v1/sessions. `options` is validated against the kind's model
    above in the handler (422 request.validation on mismatch) — a plain dict
    here keeps the error a single coded shape instead of a four-armed union
    blob.

    `owner` (non-empty, ≤ OWNER_MAX_LENGTH chars) attaches a lease to the
    session: it must then be renewed via the heartbeat endpoint or the expiry
    watchdog safety-stops it after `lease_timeout_s`. Without an owner there
    is no lease and no timeout-stop. Both fields' shape checks live in the
    handler (see the constants above)."""

    kind: Literal[
        "teleoperation",
        "recording",
        "inference",
        "replay",
        "calibration",
        "auto_calibration",
        "hosting",
        "remote_teleoperation",
    ]
    robot: str
    owner: str | None = None
    # None → the per-kind default (LEASE_TIMEOUT_AUTO_CALIBRATION_S for
    # auto_calibration, LEASE_TIMEOUT_DEFAULT_S otherwise), resolved in the
    # handler so an explicit client value always wins.
    lease_timeout_s: float | None = None
    options: dict[str, Any] = {}


class SessionHeartbeatBody(BaseModel):
    """POST /api/v1/sessions/{session_id}/heartbeat. `owner` must match the
    lease's owner (shape-checked in the handler like SessionStartBody's)."""

    owner: str


# --- responses (shape authority: sessions.SessionTracker) -------------------


class SessionLeaseInfo(BaseModel):
    """The public face of a session's lease. `expires_in_s` is computed at
    read time from the internal monotonic deadline (never exposed) and is
    never negative; only a heartbeat pushes it back up — reads don't renew."""

    owner: str
    timeout_s: float
    expires_in_s: float


class SessionInfo(BaseModel):
    """Identity of the current session (SessionTracker._current). `robot` and
    `owner` are known only for sessions started through /api/v1/sessions —
    legacy-started sessions carry null (the tracker never guesses). `lease`
    is null for owner-less and legacy-started sessions — those are never
    timeout-stopped."""

    id: str
    kind: str
    robot: str | None
    owner: str | None
    started_at: float
    revision: int
    phase: str | None
    lease: SessionLeaseInfo | None


class EndedSessionInfo(BaseModel):
    """The last_ended summary (SessionTracker._last_ended). `phase` is the
    phase carried by the release event — the session's final phase. `reason`
    is "session.lease_expired" when the expiry watchdog safety-stopped the
    session, null for every normal ending."""

    id: str
    kind: str
    ended_at: float
    phase: str | None
    reason: str | None


class SessionHeartbeatResponse(BaseModel):
    """The renewed identity (`lease.expires_in_s` back at `timeout_s`); for a
    session with no lease the heartbeat is a documented no-op and `lease`
    stays null."""

    session: SessionInfo


class SessionStartResponse(BaseModel):
    """`warnings` relays the feature start handler's warn-but-allow findings
    (teleoperation/replay arm-identity checks: the session RUNS, but e.g. the
    servos' EEPROM disagrees with the saved calibration file). Backend prose,
    rendered verbatim; null when the start raised none — kinds whose warnings
    surface via status polling (recording, inference) keep them there."""

    session: SessionInfo
    warnings: list[str] | None = None


class CurrentSessionResponse(BaseModel):
    session: SessionInfo | None
    last_ended: EndedSessionInfo | None


class CoachingCommandResponse(BaseModel):
    """The 200 body of the coaching control verbs (rollout.handle_coaching_command).

    Only the ACCEPTED shape is modelled: a refusal carries its own status code
    and is raised as an HTTPException by the route, so it never reaches here.
    """

    success: bool
    message: str


class SessionStopResponse(BaseModel):
    """`result` is the kind's existing stop handler's response, verbatim —
    rich per-kind status stays on the feature endpoints this phase."""

    session: SessionInfo
    result: dict[str, Any]


class SessionCoachingBody(BaseModel):
    """POST /api/v1/sessions/{session_id}/coaching. One operator command for a
    coaching (DAgger) inference session. The runner interprets the verb against
    the live phase — this layer forwards it and never pre-checks (a stale phase
    copy would sometimes reject a command the arm was ready for)."""

    model_config = ConfigDict(extra="forbid")

    # `recover` is deliberately NOT here any more: it did discard-then-reset,
    # which is what `cancel` now does on its own, from every phase. `drop_last`
    # is the new one — it un-records the correction the runner is still holding.
    command: Literal[
        "takeover",
        "handback",
        "cancel",
        "hold",
        "resume",
        "reset",
        "recovered",
        "drop_last",
    ]


class SessionCoachingResponse(BaseModel):
    """The runner's command result, verbatim (`success` plus a status message).
    Rich coaching state stays on the /inference-status poll + the pushed
    `coaching_state` websocket event."""

    result: dict[str, Any]
