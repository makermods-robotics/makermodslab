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
    "GpuLaunchResponse",
    "GpuStatusResponse",
    "InferenceOptions",
    "PolicyCameraDims",
    "RecordingOptions",
    "RemoteInferenceOptions",
    "RemoteInferenceStats",
    "RemoteInferenceStatusResponse",
    "RemoteInferenceTransport",
    "RemoteInferenceTransportStatusResponse",
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


class RemoteInferenceOptions(BaseModel):
    """Policy + transport fields of remote_inference.py's RemoteInferenceRequest.

    Everything hardware-shaped resolves server-side from the robot record, as
    for every other kind. What is here that inference does NOT have is the
    transport triple — horizon / fps / video_codec — because Portal
    fingerprints the wire schema and SILENTLY DROPS packets whose fingerprint
    differs: a disagreement with the GPU side presents as a healthy session
    with zero chunks, never as an error. They are options, not constants,
    precisely so the panel can generate the matching `modal run` line from the
    same object.

    Deliberately NOT exposed (constants in remote_inference's arg builder, each
    with a `# why` there): adaptive, base_lead, align, action_delay, the
    latency coefficients, video_quality/bitrate, reliable_state, and — on the
    rtc engine — slack / tolerance / max_guidance_weight / rtc_schedule. Their
    wrong values present as "the arm freezes" or "the arm snaps at every
    boundary" rather than as an error.
    """

    model_config = ConfigDict(extra="forbid")

    # Two vocabularies, deliberately not collapsed: `policy_ref` is the opaque
    # Lab ref /jobs/{id}/checkpoints yields (and what checkpoint_state_dim and
    # camera_dims come from); `policy_hub_id` is the "<owner>/<repo>" the GPU
    # container resolves with from_pretrained. Merging them would either make
    # the Lab download a checkpoint it never runs, or leave the arm-count guard
    # with nothing to check. `policy_hub_id` is advisory in this slice — the
    # backend never reads it — and is kept because it is what lets the panel
    # generate the other terminal's `modal run` line from this same object.
    policy_ref: str
    policy_hub_id: str = ""
    task: str = ""
    camera_bindings: dict[str, str] = {}
    camera_dims: dict[str, PolicyCameraDims] = {}
    checkpoint_state_dim: int | None = None
    duration_s: int = 60  # 0 = unbounded
    horizon: int = 16  # MUST match the GPU side
    fps: int = 30  # MUST match the GPU side
    video_codec: Literal["H264", "MJPEG"] = "H264"  # MUST match the GPU side
    # WHICH chunk player runs on the arm, and therefore which GPU server the
    # other terminal must be running:
    #   sync -> makermodslab.drtc.robot_sync  <-> modal_policy.py
    #   rtc  -> makermodslab.drtc.robot_rtc   <-> modal_policy_rtc.py
    # The two are not interchangeable: `rtc` ships the still-to-execute prefix
    # so the server can GUIDE denoising, which only a flow/diffusion policy
    # (smolvla, pi0, pi05, diffusion) can act on — an ACT checkpoint serves
    # plain chunks and the extra state fields are ignored, so `rtc` buys it
    # nothing and costs it the play-to-completion contract it was evaluated in.
    #
    # The BACKEND DOES NOT VERIFY THIS. It never reads the checkpoint (the GPU
    # container is what loads it), so it cannot tell a flow policy from an ACT
    # one; the UI is the gate, and `deployGuards` refuses `rtc` for a
    # non-flow policy_type before the launch. A caller driving this API
    # directly owns that choice.
    engine: Literal["sync", "rtc"] = "sync"
    # Minimum execution budget, in action steps. MUST match the GPU side's
    # `--s-min` on the rtc engine: the robot computes `overlap_end =
    # H - max(s_min, d)` per request and the server trusts it, falling back to
    # its OWN `H - s_min` when the field is absent — so two different values
    # give two different fresh-region boundaries and the in-painting guidance
    # is computed against the wrong mask. Only sent for `engine="rtc"`; the
    # sync engine's own s_min stays at its (identical) child default.
    s_min: int = 4
    skip_identity_check: bool = False


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
        "remote_inference",
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


class RemoteInferenceStats(BaseModel):
    """One 1 Hz STATS sample from the child, decoded by drtc_protocol.parse_stats.

    Field-for-field :data:`makermodslab.drtc_protocol.STATS_KEYS`, in its order,
    with that module's own nullability. Every key is ALWAYS present —
    `format_stats` fills missing ones with null and raises on an unknown one,
    and `parse_stats` refills the full set on the way back — which is what
    makes this model exact instead of `exclude_none`. Adding a field here
    without adding it to STATS_KEYS is a lie; change STATS_KEYS and nowhere
    else.
    """

    t: int
    chunks: int
    reqs: int
    sched: int
    lead: int
    s_min: int
    horizon: int
    lat_steps: int
    lat_ms: float
    holds: int
    degrade: bool
    # Null until the first chunk / the first operator / the first correlated
    # round trip — LEGITIMATE nulls, which is why no exclusion mode may be
    # applied to the status route (see RemoteInferenceStatusResponse).
    chunk_age_ms: float | None
    active: str | None
    e2e_p50_us: int | None
    e2e_p95_us: int | None
    rtt_us: int | None
    uncorr: int


class RemoteInferenceTransport(BaseModel):
    """The transport the session actually resolved (the READY echo, not what
    the parent believed it passed — see drtc_protocol.format_ready).

    `source` is `remote_inference._transport_source`'s range, the SAME set the
    transport ROUTE reports: `sfu` (this process runs the Lab's own LiveKit
    server, so the url, room and the child's token were minted in-process),
    `process_env`, `cloud` (livekit.env) or `none`. It used to be narrower than
    the route's; one walker now serves both, and the two values it lost —
    `local_override` and `cwd` — retired with the shell SFU scripts in S3.6."""

    url: str
    room: str
    source: Literal["sfu", "cloud", "process_env", "none"]
    operator_present: bool


class RemoteInferenceStatusResponse(BaseModel):
    """GET /api/v1/remote-inference-status — shape authority:
    remote_inference.handle_remote_inference_status (pinned by that module's
    tests as an equality-asserted key set).

    Modelled EXACTLY: the handler funnels every branch (live,
    idle-with-terminal-result, plain idle) through one payload builder
    (`_payload_locked`) so the key set never varies, which is what lets this
    route carry no exclusion mode at all. Contrast /inference-status, whose
    live and terminal branches carry different keys — the reason it is still in
    UNTYPED_V1_ROUTES.

    `response_model_exclude_none` would be actively WRONG here: pydantic's
    exclude_none recurses, so it would strip `chunk_age_ms` / `active` /
    `e2e_*` / `rtt_us` out of `stats` exactly while the run is warming up —
    the moment the operator most needs "no sample yet" rendered as an explicit
    null rather than as a missing key.
    """

    remote_inference_active: bool
    phase: str | None
    policy_ref: str | None
    # Which chunk player the run was started with ("sync" / "rtc"), so a panel
    # that did not start it can still say which regime is driving the arm — and
    # which of the two `modal run` lines the other terminal has to be running.
    # Null only when no run has been started since boot; a live or terminal
    # payload always carries it.
    engine: str | None
    started_at: float | None
    # Seconds from `started_at`. FROZEN at the exit for a terminal payload
    # (built once, from the globals, before they are cleared) rather than reset
    # to 0 — a finished run that reports "0s" reads as a run that never
    # happened, which is precisely the opposite of what a failed one needs to
    # say.
    elapsed_s: float
    duration_s: int | None
    log_path: str | None
    # Terminal-run fields, reusing rollout's contracts verbatim
    # (_classify_outcome / friendly_hint) so terminal handling matches the
    # local sibling.
    exited: bool
    exit_code: int | None
    outcome: str | None
    error: str | None
    hint: str | None
    # Warn-but-allow arm-identity finding, surfaced once the run is up.
    warning: str | None
    # True inside the `stopping` phase while the child is easing the arm back
    # to its captured start pose. Exposed as a flag rather than as a phase name
    # BECAUSE sessions._WINDING_DOWN_PHASES must keep matching "stopping" — a
    # `returning` phase would let an expiry tick dispatch a second stop into an
    # in-flight return.
    returning_to_rest: bool
    shutting_down: bool
    stats: RemoteInferenceStats | None
    transport: RemoteInferenceTransport | None


class RemoteInferenceTransportStatusResponse(BaseModel):
    """GET /api/v1/remote-inference/transport — shape authority:
    remote_inference.handle_remote_inference_transport.

    Every key always present, so no exclusion mode: the four probe-shaped
    fields are null when the probe DID NOT RUN (no extra, or not configured),
    which is a third state distinct from false."""

    extra_installed: bool
    # Under the Lab's own SFU this is True by construction (the credentials are
    # minted in-process); on the Cloud path it means all four LIVEKIT_* vars
    # resolved.
    configured: bool
    missing_vars: list[str]  # [] when configured
    url: str  # "" when unresolved — never null
    room: str
    source: Literal["sfu", "cloud", "process_env", "none"]
    # --- the Lab-owned SFU (makermodslab --sfu; see makermodslab/sfu.py) ---
    # Everything below is null/false when this process does not run one, so a
    # panel can render the whole block from one flag.
    sfu_enabled: bool
    sfu_url: str | None  # the loopback signalling URL a local child dials
    # The URL a MODAL container should dial: ws://<tailnet ipv4>:7880. Null
    # when tailscale is absent or not logged in — a container has no route to
    # loopback or to a LAN address, so there is nothing honest to offer then.
    sfu_modal_url: str | None
    # Whether the SFU advertises its STUN-discovered public IP (the launcher's
    # --sfu-external-ip). Without it a Modal container can reach signalling
    # over the tailnet and still never punch a media path.
    sfu_external_ip: bool
    # The key's NAME — the `--livekit-api-key` half of the Modal line. It
    # identifies rather than authorizes. THE SECRET IS NEVER RETURNED; the
    # panel names the file below and a human reads it.
    sfu_key_id: str | None
    sfu_key_file: str | None
    # The per-OS install line, present ONLY when `livekit-server` is missing
    # from PATH — otherwise null, which is how the panel knows `--sfu` is a
    # flag the user can actually pass.
    sfu_install_hint: str | None
    # Null (not false) when the probe did not run.
    endpoint_reachable: bool | None
    operator_present: bool | None
    # The probe's coded failure ("transport.unreachable" / ".unauthorized" /
    # ".no_policy"), or "transport.extra_missing" / ".not_configured" when the
    # probe never ran. Null on success.
    error_code: str | None
    message: str | None


class GpuStatusResponse(BaseModel):
    """GET /api/v1/remote-inference/gpu — shape authority:
    modal_launcher.status().

    Modelled EXACTLY, with no exclusion mode: the launcher funnels every branch
    through one builder (`_status_locked`) that emits every key, null where
    unknown, which is what lets a `response_model` sit on it without
    materializing absences as lies.

    The GPU is a LAB-LEVEL RESOURCE, not a session — it holds no hardware, has
    no lease, and stopping it is not a safety action. That is why it has its
    own three routes rather than a field on `RemoteInferenceOptions`: a 1-3
    minute cold start inside `handle_start_remote_inference` would hold the
    `robot.busy.remote_inference` discriminant while the arm sat completely
    free.
    """

    # idle | starting | ready | failed | stopping.
    state: str
    # The container's own progress, parsed from its stdout: tailscale_up |
    # loading | warmup | connecting | connected | claimed. Null before the
    # first recognizable line. `connected` is what flips `state` to ready —
    # `claimed` is a display refinement, because policy.py claims control in a
    # background task and a healthy run may never print it.
    phase: str | None
    # Which wrapper is running ("sync" / "rtc"), i.e. which robot-side chunk
    # player it pairs with. Null while idle.
    engine: str | None
    policy_hub_id: str | None
    # The room the launcher pinned with --livekit-room. The one value whose
    # mismatch is invisible by construction, so it is reported.
    room: str | None
    # WHICH WORKSPACE IS PAYING, as launched. The Modal profile went to the
    # child as MODAL_PROFILE and the environment as `modal run --env`; both are
    # null when the operator left the choice to the CLI's own resolution, which
    # is a different fact from an empty selection. Data — profile, workspace
    # and environment names are never translated.
    profile: str | None
    environment: str | None
    # The Modal app this run created ("ap-…"), parsed from the client's own
    # "View run at …" url. Null while idle and until that line arrives. It is
    # the argument `modal app stop` takes — which is how the Lab stops an app
    # whose client it had to kill, and how an operator stops one by hand.
    app_id: str | None
    # The transport tuple AS LAUNCHED, null while idle. Half of it is the
    # Portal wire schema (horizon/fps/video_codec/s_min — a disagreement drops
    # every packet in silence rather than erroring) and `task` steers the
    # policy itself, so the panel warns when the form drifts away from these.
    # Echoing them SERVER-SIDE is what makes that warning survive a page
    # reload and cover a GPU another tab started; before it, the comparison
    # could only be made against the launching tab's own memory.
    task: str | None
    horizon: int | None
    fps: int | None
    video_codec: str | None
    # As launched, for both engines. It only reaches the wire for `rtc`
    # (`modal_launcher.build_argv` omits the flag otherwise), which is why the
    # panel compares it only when the engine is rtc.
    s_min: int | None
    # WHAT IT RUNS AS and WHAT IT RUNS ON, as launched; null only while idle
    # (S3.8e). Both are DATA — `bfloat16` is a torch dtype name and `A100-80GB`
    # a Modal GPU spec, matched verbatim on both sides and never translated.
    #
    # The empty string is a REAL value for both, unlike `profile` above: it
    # means the checkpoint's own saved dtype (no `--model-dtype` flag was
    # passed) and the wrapper's own pinned GPU (no `DRTC_GPU` was exported).
    # That is why they are echoed as sent rather than folded into null — a
    # client comparing its form against this record is comparing against what
    # the launch actually said.
    model_dtype: str | None
    # Named for Modal's own `@app.function(gpu=…)` kwarg, which is what it
    # becomes. It cannot ride the command line at all: the wrapper's decorator
    # is evaluated at import, before `modal run` parses a flag, so the launcher
    # exports it as DRTC_GPU in the child's environment.
    gpu: str | None
    # Whether each per-checkpoint knob actually reached the wire (S3.8f).
    #
    # A knob is DROPPED when the target checkpoint's own config has no field
    # for it — a precision remembered from a MolmoAct2 run, still selected when
    # the operator switches to SmolVLA, whose config has no `model_dtype` at
    # all. Before this that was a paid cold start ending in the container's own
    # refusal; now the launcher drops it and says so here.
    #
    # The value above is the ASK, so a client comparing its form against this
    # record still matches after a drop (a dropped knob must not read as
    # drift). These two say whether it went. False beside an EMPTY value is
    # simply "nobody chose one".
    model_dtype_applied: bool
    # How many flow-matching / denoising steps the sampler takes per chunk, as
    # ASKED. Null while idle AND when nothing was asked — an int has no empty
    # string the way `model_dtype` has one, so those two share a value and
    # `flow_steps_applied` is what separates a drop from a non-choice.
    #
    # Which config field it writes is per family and is resolved inside the
    # container (`utils.system.policy_flow_steps_field`): `num_steps` for
    # smolvla, `num_inference_steps` for pi0/pi05/MolmoAct2.
    flow_steps: int | None
    flow_steps_applied: bool
    # The EXTRA camera views this launch asked to declare on the checkpoint
    # (S3.8g) — role names, e.g. ["cam2"]. Null while idle; [] is a real answer
    # ("the checkpoint's own views") and is what an unasked launch echoes.
    #
    # The only one of these knobs that changes the WIRE: an added role is one
    # more `observation.images.<role>` input feature, so the policy expects one
    # more video track and the robot side publishes it. Dropped, on the same
    # rule as the two above, for a checkpoint whose view count is fixed by its
    # architecture rather than by its wrapper — `extra_image_roles_applied`
    # separates that drop from "nobody asked". Role names are DATA and are
    # rendered verbatim in every language.
    extra_image_roles: list[str] | None
    extra_image_roles_applied: bool
    # WHAT THE CONTAINER SAID IT IS RUNNING ON — e.g. "NVIDIA A100-SXM4-40GB
    # (39.6 GiB)", parsed out of the policy server's own `[policy] device:`
    # line. Null while idle and until that line arrives.
    #
    # The only EVIDENCE in this payload about hardware: `gpu` above is what the
    # launch asked Modal for, which is a different fact and was the one being
    # read as this. DATA — a vendor device string, never translated.
    device_name: str | None
    # Survives an idle transition on purpose: after a failure the log is the
    # most useful thing left. Null before the first launch since boot.
    log_path: str | None
    started_at: float | None  # unix seconds
    elapsed_s: float  # 0.0 while idle
    # Terminal reason while failed/stopping, null otherwise. Backend prose.
    message: str | None
    hint: str | None
    # The `gpu.*` code behind a FAILED state ("gpu.unauthenticated",
    # "gpu.launch_failed"), null in every other state. The stable contract an
    # SDK dispatches on — the prose above it is free to improve, and Modal's
    # own auth failure and an expired tailnet auth key have entirely different
    # remedies. A string rather than the enum because a newer server may name a
    # condition this client has never heard of; match on it, do not exhaust it.
    code: str | None
    # The most recent non-empty output line, for the "what is it doing right
    # now" strip while starting. Data — never translated.
    last_line: str | None
    # Seconds until the idle auto-stop fires. Null unless `ready` AND no remote
    # session is running: a busy GPU is not idle, and a countdown that keeps
    # ticking through a live run would be a lie.
    idle_stop_in_s: float | None


class ModalProfileEntry(BaseModel):
    """One row of `modal profile list --json`.

    All three fields are DATA: a profile name and a workspace name are
    identifiers the CLI matches on, never prose to translate."""

    name: str
    # The Modal workspace the profile bills to — the thing an operator actually
    # recognizes, and why the picker shows it beside the name.
    workspace: str
    # Whether this is the machine-wide active profile. The picker's DEFAULT,
    # never a constraint: the Lab picks per launch and never activates one.
    active: bool


class ModalEnvironmentEntry(BaseModel):
    """One row of `modal environment list --json`, for one profile's workspace.

    The CLI emits `active` there as the STRING "True" and carries a key with a
    space in it; both are normalized in `modal_launcher.parse_environments`, so
    this model sees a plain boolean."""

    name: str
    active: bool


class GpuTargetsError(BaseModel):
    """Why the listing could not be made — in the BODY, never as a 500.

    A machine with no `modal`, an expired token or a slow API must still be
    able to launch: with no selection the CLI resolves the profile and
    environment itself. So a listing failure costs the two pickers and nothing
    else, and the panel shows this message verbatim beside a Start button that
    still works."""

    # `gpu.cli_missing` | `gpu.unauthenticated` | `gpu.targets_unavailable`.
    # A string rather than the enum: a newer server may name a condition this
    # client has never heard of — match on it, do not exhaust it.
    code: str
    # Backend prose. Rendered as sent; the Python backend is never localized.
    message: str


class GpuTargetsResponse(BaseModel):
    """GET /api/v1/remote-inference/gpu/targets — shape authority:
    modal_launcher.list_targets().

    What THIS MACHINE can bill, so a launch can be pointed at one workspace
    without `modal profile activate` rewriting the ~/.modal.toml every other
    terminal on this machine shares. Read-only: two `modal … list --json`
    subprocesses, bounded, and no mutating subcommand anywhere near this path.
    `~/.modal.toml` itself — which holds token_id AND token_secret per profile
    — is never opened, so nothing here can carry a credential.
    """

    profiles: list[ModalProfileEntry]
    # The environments of `profile` below — NOT of every profile. Empty
    # whenever `error` is set.
    environments: list[ModalEnvironmentEntry]
    # Which profile the environments were listed for: the `profile` query when
    # one was given, otherwise the active one. Null when they could not be
    # listed at all, which is what stops a stale list reading as current.
    profile: str | None
    error: GpuTargetsError | None


class GpuLaunchResponse(BaseModel):
    """POST /api/v1/remote-inference/gpu/start — shape authority:
    modal_launcher.start().

    `started` is always true here (every refusal is a coded ApiError raised
    before the spawn), and it is kept so the shape matches the install-manager
    surface this one is modelled on. `gpu` is the status as of the spawn."""

    started: bool
    message: str
    gpu: GpuStatusResponse


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
