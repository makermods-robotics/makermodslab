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
"""Tests for makermodslab.remote_inference — schema, arg builder, the mutex,
the preflight ladder, the STATS pump, the two watchdogs and the stop machine.

Per the repo's testing policy the subprocess happy path (spawn → connect →
chunks) is deliberately NOT tested: it needs LiveKit, a GPU and an arm. What is
tested is everything that decides whether that subprocess is allowed to exist,
and everything that decides how it dies.

No sleeps anywhere: the watchdogs read an injected `_clock`
(tests/test_session_lease.py's FakeClock pattern) and the stop machine drives a
fake Popen. The room probe is monkeypatched at its one seam — livekit-api is
aiohttp-based, so httpx.MockTransport does not apply and no new test dependency
is warranted for a function this thin.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from makermodslab import remote_inference as ri
from makermodslab.drtc_protocol import STATS_KEYS, format_event, format_stats


class FakeClock:
    """A controllable stand-in for time.monotonic (mirrors test_session_lease)."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeProc:
    """The minimum `subprocess.Popen` surface the stop machine touches."""

    def __init__(self, *, alive: bool = True, accepts_stdin: bool = True) -> None:
        self.returncode: int | None = None if alive else 0
        self.stdin = _FakeStdin(accepts=accepts_stdin) if accepts_stdin else None
        self.waits: list[float | None] = []
        self.pid = 4242

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        self.returncode = 0
        return 0


class _FakeStdin:
    def __init__(self, accepts: bool = True) -> None:
        self.written: list[bytes] = []
        self.accepts = accepts

    def write(self, data: bytes) -> None:
        if not self.accepts:
            raise BrokenPipeError("the child is gone")
        self.written.append(data)

    def flush(self) -> None:
        if not self.accepts:
            raise BrokenPipeError("the child is gone")


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset the module's session state around each test, so a leaked
    `remote_inference_active=True` cannot poison the next one."""
    monkeypatch.setattr(ri, "remote_inference_active", False)
    monkeypatch.setattr(ri, "_remote_proc", None)
    monkeypatch.setattr(ri, "_remote_started_at", None)
    monkeypatch.setattr(ri, "_remote_running_started_at", None)
    monkeypatch.setattr(ri, "_remote_meta", {})
    monkeypatch.setattr(ri, "_last_result", None)
    monkeypatch.setattr(ri, "_remote_cancel", None)
    monkeypatch.setattr(ri, "_startup_thread", None)
    monkeypatch.setattr(ri, "_transport", None)
    monkeypatch.setattr(ri, "_stats", None)
    monkeypatch.setattr(ri, "_connected_at", None)
    monkeypatch.setattr(ri, "_active_at", None)
    monkeypatch.setattr(ri, "_chunks", 0)
    monkeypatch.setattr(ri, "_returning_to_rest", False)


def _request(**overrides):
    fields = {"follower_port": "/dev/ttyUSB0", "follower_config": "robot_a"}
    fields.update(overrides)
    return ri.RemoteInferenceRequest(**fields)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


def test_request_has_expected_defaults() -> None:
    """The wire-contract defaults are the ones the GPU side also defaults to —
    a run started with no options must not need a matching flag on the other
    terminal to work."""
    req = _request()
    assert req.horizon == 16
    assert req.fps == 30
    assert req.video_codec == "H264"
    assert req.duration_s == 60
    assert req.mode == "single"
    assert req.arm_type == "so101"
    assert req.camera_bindings == {}
    assert req.checkpoint_state_dim is None
    assert req.skip_identity_check is False
    # The default engine is the one that works for ANY policy. `rtc` guides
    # denoising, which only a flow/diffusion checkpoint can act on, so defaulting
    # to it would break every ACT run started without options.
    assert req.engine == "sync"
    # Same value modal_policy_rtc.py's `--s-min` defaults to. The two MUST agree
    # (the robot computes overlap_end from it and the server trusts the field),
    # so a run started with neither side's flag set still agrees.
    assert req.s_min == 4


def test_request_rejects_an_unknown_engine() -> None:
    """An engine the map does not know would spawn the sync child while the
    operator's other terminal runs the rtc server — a schema-fingerprint
    mismatch, which Portal reports by silently dropping every packet."""
    with pytest.raises(ValidationError):
        _request(engine="inpaint")


def test_request_rejects_unknown_fields() -> None:
    """`extra="forbid"`: every field here is a hardware address or half of a
    wire contract, so a typo silently ignored is a run that connects, looks
    healthy and never receives a chunk."""
    with pytest.raises(ValidationError):
        _request(horizen=8)


def test_request_rejects_an_unknown_codec() -> None:
    """A codec name the child cannot resolve would raise AFTER the bus is open
    and the arm energized (`getattr(VideoCodec, name)`)."""
    with pytest.raises(ValidationError):
        _request(video_codec="AV1")


# ---------------------------------------------------------------------------
# The argument builder
# ---------------------------------------------------------------------------


def _robot_record_with_cam(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A one-camera robot record on redirected disk (mirrors test_rollout's)."""
    from makermodslab.utils import config as cfg

    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    cfg.save_robot_record(
        name,
        {
            "cameras": [
                {
                    "id": "camera_1",
                    "name": "wrist",
                    "type": "opencv",
                    "camera_index": 0,
                    "device_id": "browser-device-id",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                }
            ]
        },
        allow_create=True,
    )


def test_robot_sync_args_carry_the_wire_settings_and_the_pinned_transport() -> None:
    req = _request(horizon=32, fps=15, duration_s=120, video_codec="MJPEG")
    args = ri._robot_sync_args(req, ["--robot.type=so101_follower"], url="ws://sfu:7880", room="rm")

    assert args[0] == "--robot.type=so101_follower"
    assert "--horizon=32" in args
    assert "--fps=15" in args
    assert "--duration_s=120" in args
    assert "--video_codec=MJPEG" in args
    assert "--livekit_url=ws://sfu:7880" in args
    assert "--livekit_room=rm" in args


def test_robot_sync_args_ask_for_the_safe_stop_and_the_ease_in() -> None:
    """Both default true in the child; a supervised run states them anyway so
    the safety behaviour is in the argv a log records."""
    args = ri._robot_sync_args(_request(), [], url="u", room="r")
    assert "--return_to_rest=true" in args
    assert "--ease_in=true" in args


def test_the_engine_picks_the_child_module() -> None:
    """The two entrypoints are not interchangeable: `robot_rtc` publishes five
    extra RTC state fields, and Portal fingerprints the whole state schema — so
    a child that disagrees with the GPU server it is paired with connects, looks
    healthy, and receives nothing."""
    assert ri._child_module("sync") == "makermodslab.drtc.robot_sync"
    assert ri._child_module("rtc") == "makermodslab.drtc.robot_rtc"


def test_an_unknown_engine_falls_back_to_the_sync_child() -> None:
    """Unreachable through the API (the options model's Literal refuses it), and
    raising HERE would be a poor trade: `_child_module` runs on the startup
    worker, after the arm has been claimed and preflighted."""
    assert ri._child_module("nonsense") == "makermodslab.drtc.robot_sync"


def test_only_the_rtc_engine_is_sent_s_min() -> None:
    """`--s_min` is half a contract on the rtc engine — the robot computes
    `overlap_end = H - max(s_min, d)` and `policy_rtc` trusts that field, so the
    two sides must agree. On the sync engine it only tunes when the player calls
    itself degraded, and it stays at the child's (identical) default with the
    rest of the scheduler knobs."""
    rtc = ri._robot_sync_args(_request(engine="rtc", s_min=6), [], url="u", room="r")
    assert "--s_min=6" in rtc

    sync = ri._robot_sync_args(_request(engine="sync", s_min=6), [], url="u", room="r")
    assert not [a for a in sync if a.startswith("--s_min")]


def test_both_engines_get_the_same_wire_settings_and_the_same_safe_stop() -> None:
    """The engines share their whole session surface (`_session_glue`), and the
    arg builder is the place that could quietly stop being true — an rtc run
    without `--return_to_rest` would drop the arm at every stop."""
    common = ["--fps=30", "--horizon=16", "--duration_s=60", "--video_codec=H264"]
    safety = ["--livekit_url=u", "--livekit_room=r", "--return_to_rest=true", "--ease_in=true"]
    for engine in ("sync", "rtc"):
        args = ri._robot_sync_args(_request(engine=engine), [], url="u", room="r")
        for flag in common + safety:
            assert flag in args, f"{engine} is missing {flag}"


def test_robot_sync_args_never_emit_a_no_flag() -> None:
    """draccus has NO `--no-<flag>` form — it is not argparse's
    BooleanOptionalAction. A `--no-return_to_rest` would be parsed as an
    unknown argument, not as "off"."""
    args = ri._robot_sync_args(_request(), [], url="u", room="r")
    assert not [a for a in args if a.startswith("--no-")]


def test_robot_sync_args_take_the_cameras_from_the_record_keyed_by_policy_name(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding names a RECORD camera ("wrist"); the child's flag is keyed by
    the POLICY-expected name ("front"). That keying is the whole mitigation for
    Portal's silent fingerprint mismatch — Portal derives the robot's track
    names from these keys and the policy's from the checkpoint's
    observation.images.*, so they must agree exactly."""
    from makermodslab.rollout import _single_robot_args

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "solo")
    req = _request(robot_name="solo", camera_bindings={"front": "wrist"})
    robot_args = _single_robot_args(ri._robot_request(req), "robot_a")

    args = ri._robot_sync_args(req, robot_args, url="u", room="r")
    cam_arg = next(a for a in args if a.startswith("--robot.cameras="))
    assert "front:" in cam_arg
    assert "wrist" not in cam_arg
    assert "index_or_path: 0" in cam_arg
    assert "width: 640" in cam_arg


def test_robot_sync_args_capture_at_the_checkpoints_resolution(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing resizes frames to the policy's input shape, so the checkpoint's
    dims must win over the record's configured size."""
    from makermodslab.rollout import _single_robot_args

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "solo")
    req = _request(
        robot_name="solo",
        camera_bindings={"front": "wrist"},
        camera_dims={"front": {"width": 320, "height": 240}},
    )
    robot_args = _single_robot_args(ri._robot_request(req), "robot_a")
    cam_arg = next(a for a in robot_args if a.startswith("--robot.cameras="))

    assert "width: 320" in cam_arg
    assert "height: 240" in cam_arg
    assert "width: 640" not in cam_arg


def test_the_robot_request_adapter_carries_every_field_the_preflight_reads() -> None:
    """`_prepare_robot`/`_session_cameras` are imported from rollout, so they
    take an InferenceRequest. Building one explicitly (rather than relying on
    the two models happening to share attribute names) is what keeps that reuse
    from breaking silently the day either model gains a field."""
    req = _request(
        robot_name="solo",
        arm_type="so101",
        camera_bindings={"front": "wrist"},
        checkpoint_state_dim=6,
        skip_identity_check=True,
    )
    adapted = ri._robot_request(req)
    assert adapted.follower_port == req.follower_port
    assert adapted.follower_config == req.follower_config
    assert adapted.robot_name == "solo"
    assert adapted.camera_bindings == {"front": "wrist"}
    assert adapted.checkpoint_state_dim == 6
    assert adapted.skip_identity_check is True
    # A remote run is neither a coaching session nor an eval.
    assert adapted.coaching is False
    assert adapted.eval_episodes == 1


# ---------------------------------------------------------------------------
# The mutex — all eight refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "attr", "busy_value", "expected_code"),
    [
        ("teleoperate", "teleoperation_active", True, "robot.busy.teleoperation"),
        ("record", "recording_active", True, "robot.busy.recording"),
        ("rollout", "inference_active", True, "robot.busy.inference"),
        ("replay", "replay_active", True, "robot.busy.replay"),
        ("calibrate", "calibration_is_active", lambda: True, "robot.busy.calibration"),
        ("auto_calibrate", "auto_calibration_is_active", lambda: True, "robot.busy.auto_calibration"),
        ("wiggle", "wiggle_active", True, "robot.busy.wiggle"),
        ("jobs", "training_is_active", lambda: "ACT · user/ds", "robot.busy.training"),
    ],
)
def test_start_is_refused_by_every_peer(
    monkeypatch: pytest.MonkeyPatch, module_name, attr, busy_value, expected_code
) -> None:
    """All eight legs of the mutex, one case each.

    A new robot-driving feature must refuse against every existing one, and
    `tests/test_api_errors.py::test_busy_discriminants_cover_mutex_matrix` is
    an equality assertion — so half a matrix is not an option. These are the
    checks that keep two processes from opening the same serial port."""
    import importlib

    module = importlib.import_module(f"makermodslab.{module_name}")
    monkeypatch.setattr(module, attr, busy_value, raising=False)

    result = ri.handle_start_remote_inference(_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert result["code"] == expected_code
    # Refused BEFORE the claim: the flag must be untouched for the peer that
    # actually holds the arm.
    assert ri.remote_inference_active is False


def test_start_refuses_a_second_remote_session() -> None:
    ri.remote_inference_active = True
    try:
        result = ri.handle_start_remote_inference(_request())
    finally:
        ri.remote_inference_active = False
    assert result["code"] == "robot.busy.remote_inference"


def test_start_refuses_while_a_previous_startup_worker_is_still_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag is already False but the worker is still inside `_prepare_robot`
    with the serial port open. Starting now would race it for that port."""

    class _AliveThread:
        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(ri, "_startup_thread", _AliveThread())
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "robot.busy.releasing"


# ---------------------------------------------------------------------------
# The preflight ladder
# ---------------------------------------------------------------------------


_GOOD_ENV = {
    "LIVEKIT_URL": "wss://x.livekit.cloud",
    "LIVEKIT_ROOM": "portal-lerobot-inference",
    "LIVEKIT_API_KEY": "key",
    "LIVEKIT_API_SECRET": "secret",
}


@pytest.fixture
def preflight(monkeypatch: pytest.MonkeyPatch):
    """Everything up to (and excluding) the rung under test, made to pass.

    The extra is reported present, the credentials resolve, the room answers
    with a policy in it, and the startup worker is replaced by a recorder — so
    a test can knock out exactly one rung and see its refusal."""
    started: list[tuple] = []

    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", lambda: dict(_GOOD_ENV))
    monkeypatch.setattr(ri, "_transport_source", lambda url: "cloud")
    # The LiveKit Cloud branch. The Lab-owned SFU is the OTHER branch and gets
    # its own fixture below; nothing in this file ever runs `livekit-server`.
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )
    monkeypatch.setattr(ri.camera_preview_manager, "stop_all", lambda: None)

    class _Recorder:
        def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None) -> None:
            self.args = args

        def start(self) -> None:
            started.append(self.args)

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(ri.threading, "Thread", _Recorder)
    return started


def test_preflight_passes_and_hands_off_to_the_worker(preflight) -> None:
    result = ri.handle_start_remote_inference(_request())
    assert result["success"] is True
    assert ri.remote_inference_active is True
    assert len(preflight) == 1
    # The verified transport — not whatever the child would resolve on its own
    # — is what gets pinned onto the child's argv.
    _req, _robot_req, url, room, token, _cancel = preflight[0]
    assert (url, room) == (_GOOD_ENV["LIVEKIT_URL"], _GOOD_ENV["LIVEKIT_ROOM"])
    # No token on the Cloud path: the child mints its own from the credentials
    # it resolves, and there is no secret here for the parent to sign with.
    assert token == ""
    assert ri._transport == {
        "url": _GOOD_ENV["LIVEKIT_URL"],
        "room": _GOOD_ENV["LIVEKIT_ROOM"],
        "source": "cloud",
        "operator_present": True,
    }
    ri._go_idle_locked()


def test_preflight_refuses_when_the_extra_is_missing(monkeypatch, preflight) -> None:
    monkeypatch.setattr(ri, "_extra_missing", lambda: True)
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.extra_missing"
    assert result["status_code"] == 400
    # Never "run this here": an editable install re-points the shared venv at
    # whatever directory it runs from, which silently breaks every other
    # session's makermodslab.
    assert "PRIMARY checkout" in result["message"]
    assert ri.remote_inference_active is False


def test_preflight_refuses_incomplete_credentials(monkeypatch, preflight) -> None:
    monkeypatch.setattr(ri, "_read_env", lambda: {"LIVEKIT_URL": "wss://x"})
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.not_configured"
    assert "LIVEKIT_ROOM" in result["message"]
    assert ri.remote_inference_active is False


def test_preflight_refuses_an_unreachable_sfu(monkeypatch, preflight) -> None:
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(False, False, False, False, error="refused")
    )
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.unreachable"
    assert ri.remote_inference_active is False


def test_the_unreachable_message_names_the_sfu_when_that_is_the_transport(monkeypatch, preflight) -> None:
    """The Lab's own SFU and LiveKit Cloud have disjoint remedies — a process
    on this machine that stops with the launcher, versus a file of credentials.
    An "unreachable" that does not say which sends the operator off to check
    their internet connection when the answer was a flag they did not pass."""
    monkeypatch.setattr(ri, "_transport_source", lambda url: "sfu")
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: True)
    monkeypatch.setattr(
        ri, "_sfu_transport", lambda: ri.SfuTransport("ws://127.0.0.1:7880", "r", "k", "s", "j")
    )
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(False, False, False, False, error="refused")
    )
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.unreachable"
    assert "--sfu" in result["message"]


def test_preflight_refuses_bad_credentials(monkeypatch, preflight) -> None:
    monkeypatch.setattr(ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, False, False, False))
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.unauthorized"
    assert ri.remote_inference_active is False


def test_preflight_refuses_an_empty_room_and_names_the_three_causes(monkeypatch, preflight) -> None:
    """The empty room, caught BEFORE torque. The Lab cannot detect a room
    mismatch (the GPU's room comes only from its Modal secret), so the message
    has to name all three ways this happens."""
    monkeypatch.setattr(ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, False))
    result = ri.handle_start_remote_inference(_request())
    assert result["code"] == "transport.no_policy"
    assert "modal run" in result["message"]
    assert "LIVEKIT_ROOM" in result["message"]
    assert "TS_AUTHKEY" in result["message"]
    assert ri.remote_inference_active is False


@pytest.mark.parametrize(
    ("arm_type", "mode"),
    [("maker", "single"), ("metal", "single"), ("so101", "bimanual")],
)
def test_preflight_refuses_can_arms_and_bimanual_and_releases_the_slot(preflight, arm_type, mode) -> None:
    """Both are WIRING limits, and both must be refused SYNCHRONOUSLY and
    pre-spawn: a CAN arm fails at draccus CLI-parse time INSIDE the child, by
    which point the session has claimed and preflighted the arm; a bimanual one
    would run, but its first move would be a full-speed snap (no ease-in).

    The slot has already been claimed by the time this rung runs, so the
    refusal must release it — otherwise one bad launch wedges the arm until a
    restart."""
    result = ri.handle_start_remote_inference(_request(arm_type=arm_type, mode=mode))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert ri.remote_inference_active is False
    assert ri._remote_meta == {}


def test_preflight_refuses_an_arm_count_mismatch_and_releases_the_slot(preflight) -> None:
    result = ri.handle_start_remote_inference(_request(checkpoint_state_dim=12))
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "bimanual" in result["message"]
    assert ri.remote_inference_active is False


def test_preflight_refuses_an_unresolvable_camera_binding_and_releases_the_slot(
    preflight, monkeypatch
) -> None:
    from makermodslab.utils.config import CameraResolutionError

    def _boom(_request):
        raise CameraResolutionError("robot 'solo' has no camera named 'wrist'")

    monkeypatch.setattr(ri, "_session_cameras", _boom)
    result = ri.handle_start_remote_inference(_request(robot_name="solo", camera_bindings={"f": "wrist"}))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert ri.remote_inference_active is False


# ---------------------------------------------------------------------------
# The event pump
# ---------------------------------------------------------------------------


def _live_session(phase: str = ri.PHASE_STARTING, **meta) -> None:
    """Put the module into the shape a running session leaves it in."""
    ri.remote_inference_active = True
    ri._remote_started_at = 1000.0
    ri._remote_meta = {"phase": phase, "policy_ref": "u/r@root", "duration_s": 60, **meta}
    ri._transport = {
        "url": "wss://x",
        "room": "rm",
        "source": "cloud",
        "operator_present": True,
    }


def test_the_pump_walks_the_phase_vocabulary() -> None:
    _live_session()
    ri._handle_line(format_event("READY", "url=wss://x room=rm") + "\n")
    assert ri._remote_meta["phase"] == ri.PHASE_CONNECTING

    ri._handle_line(format_event("CONNECTED") + "\n")
    assert ri._remote_meta["phase"] == ri.PHASE_WARMING_UP

    ri._handle_line(format_event("EASING") + "\n")
    assert ri._remote_meta["phase"] == ri.PHASE_EASING

    ri._handle_line(format_event("STATS", format_stats({"chunks": 3})) + "\n")
    assert ri._remote_meta["phase"] == ri.PHASE_RUNNING
    ri._go_idle_locked()


def test_ready_with_a_different_transport_fails_the_run_before_the_room_is_joined() -> None:
    """READY echoes the EFFECTIVE url/room the child resolved. A mismatch means
    the ground moved between the probe and the spawn (the local-SFU script
    restarted, a credential file rewritten) — a failure to report now, not a
    puzzle to debug later with an arm energized."""
    _live_session()
    ri._remote_proc = FakeProc()
    ri._handle_line(format_event("READY", "url=ws://127.0.0.1:7880 room=other") + "\n")

    assert ri._remote_meta["phase"] == ri.PHASE_STOPPING
    assert "transport changed" in ri._remote_meta["error"]
    assert ri._remote_proc.stdin.written == [b"STOP\n"]
    ri._go_idle_locked()


def test_stats_are_recorded_with_every_key() -> None:
    """Every STATS_KEYS key is always present (null where unknown) — that is
    what lets S3.3 put an EXACT response model on the status dict rather than
    an `exclude_none` one."""
    _live_session()
    ri._handle_line(format_event("STATS", format_stats({"t": 1, "chunks": 3, "holds": 41})) + "\n")

    assert set(ri._stats) == set(STATS_KEYS)
    assert ri._stats["chunks"] == 3
    assert ri._stats["holds"] == 41
    assert ri._stats["rtt_us"] is None
    ri._go_idle_locked()


@pytest.mark.parametrize(
    "payload",
    [
        "{not json at all}",
        '{"t":1,"chunks":3',  # truncated: the pipe cut mid-line
        "[1,2,3]",  # valid JSON, wrong shape
        "",
    ],
)
def test_a_malformed_stats_line_is_dropped_not_half_applied(payload) -> None:
    """ "No sample this second" is the only honest degradation — a
    half-populated status is one the UI would render as real."""
    _live_session()
    ri._stats = dict.fromkeys(STATS_KEYS)
    ri._stats["chunks"] = 7

    ri._handle_line(format_event("STATS", payload) + "\n")
    assert ri._stats["chunks"] == 7
    ri._go_idle_locked()


def test_a_protocol_event_is_recognised_inside_lerobot_chatter() -> None:
    """The child's logging handler shares the pipe, and a log record flushed
    without its trailing newline would otherwise swallow the event behind it —
    which is why parse_event matches the prefix ANYWHERE in the line."""
    _live_session()
    line = "INFO 2026-09-03 lerobot.robots: connected" + format_event("CONNECTED") + "\n"
    ri._handle_line(line)
    assert ri._remote_meta["phase"] == ri.PHASE_WARMING_UP
    ri._go_idle_locked()


def test_a_non_protocol_line_changes_nothing() -> None:
    _live_session()
    ri._handle_line("INFO lerobot: Connecting robot ...\n")
    assert ri._remote_meta["phase"] == ri.PHASE_STARTING
    ri._go_idle_locked()


def test_returning_stays_in_the_stopping_phase() -> None:
    """A phase name of its own would fall outside
    `sessions._WINDING_DOWN_PHASES`, and an expiry tick landing during the
    return would then dispatch a SECOND stop into a live return."""
    _live_session(phase=ri.PHASE_STOPPING)
    ri._handle_line(format_event("RETURNING") + "\n")

    assert ri._remote_meta["phase"] == ri.PHASE_STOPPING
    assert ri._returning_to_rest is True
    assert ri.handle_remote_inference_status()["returning_to_rest"] is True
    ri._go_idle_locked()


def test_a_late_line_cannot_resurrect_a_finished_session() -> None:
    """The pump outlives the session by a beat; a phase stamped onto an empty
    meta would broadcast a phantom transition."""
    ri._handle_line(format_event("CONNECTED") + "\n")
    assert ri._remote_meta == {}


# ---------------------------------------------------------------------------
# The watchdogs
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(ri, "_clock", fake)
    return fake


def test_no_operator_within_the_timeout_fails_the_run(clock) -> None:
    """Layer (b) of the empty-room defence: we are in the room and nobody ever
    joined. Converts "silent forever with an energized arm" into "stopped in
    fifteen seconds"."""
    _live_session(phase=ri.PHASE_WARMING_UP)
    ri._remote_proc = FakeProc()
    ri._connected_at = clock()

    clock.advance(ri._ACTIVE_TIMEOUT_S - 0.1)
    assert ri._check_watchdogs() is None

    clock.advance(0.2)
    failure = ri._check_watchdogs()
    assert failure is not None
    assert "No policy joined room 'rm'" in failure
    assert ri._remote_meta["phase"] == ri.PHASE_STOPPING
    # Asked for, never forced: the child returns the arm to rest and exits, and
    # the pump's EOF path finalises. Calling the stop handler from here would
    # block the very pump that has to drain stdout for that return to finish.
    assert ri._remote_proc.stdin.written == [b"STOP\n"]
    ri._go_idle_locked()


def test_an_operator_that_sends_no_chunks_fails_the_run(clock) -> None:
    """Layer (c), the one that earns its keep: it catches the schema-fingerprint
    mismatch, where the room matches, the operator joins, and Portal silently
    drops every packet. Invisible by construction — a healthy-looking session
    with 0 chunks."""
    _live_session(phase=ri.PHASE_WARMING_UP, fingerprint="horizon=16, fps=30")
    ri._remote_proc = FakeProc()
    ri._connected_at = clock()
    ri._active_at = clock()

    clock.advance(ri._CHUNK_TIMEOUT_S - 0.1)
    assert ri._check_watchdogs() is None

    clock.advance(0.2)
    failure = ri._check_watchdogs()
    assert failure is not None
    assert "no action chunks" in failure
    assert "horizon=16, fps=30" in failure
    ri._go_idle_locked()


def test_the_watchdogs_stand_down_once_chunks_are_flowing(clock) -> None:
    _live_session(phase=ri.PHASE_WARMING_UP)
    ri._connected_at = clock()
    ri._active_at = clock()
    ri._chunks = 1
    clock.advance(600.0)
    assert ri._check_watchdogs() is None
    ri._go_idle_locked()


def test_the_watchdogs_are_disarmed_outside_the_warm_up_window(clock) -> None:
    """`running` has nothing to wait for and `stopping` already has a stop in
    flight — re-triggering there would write a SECOND STOP, which the child
    reads as "cut the return short"."""
    _live_session(phase=ri.PHASE_STOPPING)
    ri._connected_at = clock()
    clock.advance(600.0)
    assert ri._check_watchdogs() is None
    ri._go_idle_locked()


# ---------------------------------------------------------------------------
# The stop state machine
# ---------------------------------------------------------------------------


def test_stop_when_idle_is_a_409() -> None:
    result = ri.handle_stop_remote_inference()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_writes_stop_and_waits_for_the_return() -> None:
    """Never a signal: the child owns the bus, and a SIGTERM would run its
    `finally:` from wherever the policy left the arm. STOP makes it drive back
    to the pose it captured at connect FIRST."""
    _live_session(phase=ri.PHASE_RUNNING)
    proc = FakeProc()
    ri._remote_proc = proc

    result = ri.handle_stop_remote_inference()
    assert result["success"] is True
    assert proc.stdin.written == [b"STOP\n"]
    # Bounded by the return's own ceiling plus teardown, not unbounded.
    assert proc.waits == [ri._STOP_WAIT_S]
    assert ri.remote_inference_active is False
    assert ri._last_result["exited"] is True
    assert ri._last_result["phase"] == ri.PHASE_STOPPED


def test_a_second_stop_while_returning_writes_the_abort_and_returns_at_once() -> None:
    """The second press is the abort: the child cuts the return short and
    releases torque where the arm is — nearer rest than it started. It must not
    queue a second bounded wait behind the first caller's."""
    _live_session(phase=ri.PHASE_STOPPING)
    proc = FakeProc()
    ri._remote_proc = proc
    ri._returning_to_rest = True

    result = ri.handle_stop_remote_inference()
    assert result["success"] is True
    assert proc.stdin.written == [b"STOP\n"]
    assert proc.waits == []
    # The first caller is still inside its wait; this one must not have torn
    # the session down underneath it.
    assert ri.remote_inference_active is True
    ri._go_idle_locked()


def test_a_child_that_cannot_be_talked_to_is_terminated(monkeypatch) -> None:
    """A child that will not take STOP cannot return the arm either — waiting
    out the ceiling for nothing just leaves it energized for longer."""
    terminated: list = []
    monkeypatch.setattr(ri, "_terminate_tree", lambda p, **kw: terminated.append(p))

    _live_session(phase=ri.PHASE_RUNNING)
    proc = FakeProc(accepts_stdin=False)
    ri._remote_proc = proc

    result = ri.handle_stop_remote_inference()
    assert result["success"] is True
    assert terminated == [proc]
    assert ri.remote_inference_active is False


def test_a_child_that_ignores_stop_is_terminated_after_the_ceiling(monkeypatch) -> None:
    terminated: list = []
    monkeypatch.setattr(ri, "_terminate_tree", lambda p, **kw: terminated.append(p))

    class _Wedged(FakeProc):
        def wait(self, timeout=None):
            import subprocess

            raise subprocess.TimeoutExpired(cmd="robot_sync", timeout=timeout)

    _live_session(phase=ri.PHASE_RUNNING)
    proc = _Wedged()
    ri._remote_proc = proc

    ri.handle_stop_remote_inference()
    assert terminated == [proc]
    ri._go_idle_locked()


def test_stop_before_the_child_spawned_abandons_via_the_cancel_event() -> None:
    """The pre-spawn window (the arm preflight) has no process to terminate;
    the cancel event is the only way to abandon it, and the worker bails at its
    next check."""
    import threading

    _live_session(phase=ri.PHASE_PREFLIGHT)
    cancel = threading.Event()
    ri._remote_cancel = cancel

    result = ri.handle_stop_remote_inference()
    assert result["success"] is True
    assert cancel.is_set()
    assert ri.remote_inference_active is False


def test_stop_pressed_again_while_a_worker_is_still_shutting_down(monkeypatch) -> None:
    """`_prepare_robot` cannot be interrupted mid-call, so the honest answer is
    a bounded wait and a report — not a blanket "nothing to stop" that hides a
    worker still holding the serial bus."""

    class _StuckThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            pass

    monkeypatch.setattr(ri, "_startup_thread", _StuckThread())
    result = ri.handle_stop_remote_inference()
    assert result["success"] is True
    assert result["shutting_down"] is True


# ---------------------------------------------------------------------------
# The shutdown stop (S3.8d)
# ---------------------------------------------------------------------------
#
# This is the one robot-driving flow whose arm is held by a child process that
# OUTLIVES the worker: `_spawn` gives it `start_new_session=True`, so the
# SIGTERM/SIGINT that ends a `--reload` save, a Ctrl-C or a `makermodslab
# --stop` never reaches it, and stdin EOF is ignored by design. `STOP` on that
# stdin is the only thing that returns the arm before torque is released.


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Every `session_changed` emission this module makes, in order."""
    events: list[tuple] = []

    def _record(kind, active, **kwargs):
        events.append((kind, active, kwargs.get("phase")))

    monkeypatch.setattr(ri, "notify_session_changed", _record)
    return events


def test_shutdown_stop_is_a_no_op_when_nothing_is_running(released) -> None:
    """Idle is the normal case on the way out, not an exceptional one — and it
    must cost nothing and say so, exactly like `modal_launcher`'s twin."""
    assert ri.stop_for_shutdown() is False
    assert released == []


def test_shutdown_stop_drives_the_same_stop_the_stop_control_does(released) -> None:
    """STOP on stdin, then BLOCK until the child has actually returned the arm
    and exited — a stop that outlives the process it runs in is not a stop."""
    _live_session(phase=ri.PHASE_RUNNING)
    proc = FakeProc()
    ri._remote_proc = proc

    assert ri.stop_for_shutdown() is True
    assert proc.stdin.written == [b"STOP\n"]
    # Bounded by the return's own ceiling plus teardown, never unbounded.
    assert proc.waits == [ri._STOP_WAIT_S]
    assert ri.remote_inference_active is False
    assert ri._last_result["exited"] is True
    assert ri._last_result["phase"] == ri.PHASE_STOPPED
    # The SessionTracker and the lease must see a clean end, exactly as they do
    # for a Stop press: the stopping hint, then the release.
    assert released[0] == (ri.KIND, True, ri.PHASE_STOPPING)
    assert released[-1] == (ri.KIND, False, ri.PHASE_STOPPED)


def test_shutdown_stop_escalates_when_the_child_ignores_stop(monkeypatch, released) -> None:
    """Bounded, and the bound has teeth: past the ceiling the whole process
    group goes, because a child that will not answer is a child that is still
    holding the arm."""
    terminated: list = []
    monkeypatch.setattr(ri, "_terminate_tree", lambda p, **kw: terminated.append(p))

    class _Wedged(FakeProc):
        def wait(self, timeout=None):
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired(cmd="robot_sync", timeout=timeout)

    _live_session(phase=ri.PHASE_RUNNING)
    proc = _Wedged()
    ri._remote_proc = proc

    assert ri.stop_for_shutdown() is True
    assert proc.waits == [ri._STOP_WAIT_S]
    assert terminated == [proc]
    ri._go_idle_locked()


def test_shutdown_stop_waits_out_a_stop_already_in_flight_rather_than_aborting_it(
    released,
) -> None:
    """A second STOP is the ABORT gesture — it cuts the return short. On the way
    out we can still afford to wait, so a stop already in flight (a Stop press,
    or a watchdog) is waited out instead of pressed again."""
    _live_session(phase=ri.PHASE_STOPPING)
    proc = FakeProc()
    ri._remote_proc = proc
    ri._returning_to_rest = True

    assert ri.stop_for_shutdown() is True
    assert proc.stdin.written == [], "a second STOP would have cut the return to rest short"
    assert proc.waits == [ri._STOP_WAIT_S]
    assert ri.remote_inference_active is False


def test_shutdown_stop_joins_a_startup_worker_still_holding_the_bus(monkeypatch, released) -> None:
    """The pre-spawn preflight has the follower's serial port open and cannot be
    interrupted mid-call. Leaving while it does is how the NEXT boot finds the
    bus busy — so it is joined, bounded, like the "press Stop again" gesture."""
    joins: list[float | None] = []

    class _Worker:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            joins.append(timeout)

    monkeypatch.setattr(ri, "_startup_thread", _Worker())
    assert ri.stop_for_shutdown() is True
    assert joins == [ri._STARTUP_STOP_JOIN_TIMEOUT_S]


def test_shutdown_stop_never_raises_into_the_shutdown_handler(monkeypatch, released) -> None:
    """It is called from `shutdown_event`, where an exception would skip every
    cleanup queued behind it."""

    def _boom() -> dict:
        raise RuntimeError("the stop machine fell over")

    monkeypatch.setattr(ri, "handle_stop_remote_inference", _boom)
    _live_session(phase=ri.PHASE_RUNNING)
    ri._remote_proc = FakeProc()

    assert ri.stop_for_shutdown() is True
    ri._go_idle_locked()


# ---------------------------------------------------------------------------
# Status payload
# ---------------------------------------------------------------------------

# The exact key set S3.3's response_model must describe. Equality-asserted: a
# model that materializes absent optionals as null must be describing a payload
# that really always carries them, and `response_model` silently FILTERS
# undeclared fields — so a key added here without the model is a key the API
# drops on the floor.
STATUS_KEYS = {
    "remote_inference_active",
    "exited",
    "exit_code",
    "outcome",
    "error",
    "hint",
    "warning",
    "phase",
    "policy_ref",
    "engine",
    "started_at",
    "elapsed_s",
    "duration_s",
    "log_path",
    "returning_to_rest",
    "shutting_down",
    "transport",
    "stats",
}


def test_the_idle_status_carries_every_key() -> None:
    status = ri.handle_remote_inference_status()
    assert set(status) == STATUS_KEYS
    assert status["remote_inference_active"] is False
    assert status["phase"] is None
    assert status["transport"] is None
    assert status["stats"] is None


def test_the_live_status_carries_every_key() -> None:
    _live_session(phase=ri.PHASE_RUNNING, log_path="/tmp/x.log", warning="arm warning")
    ri._stats = dict.fromkeys(STATS_KEYS)
    status = ri.handle_remote_inference_status()

    assert set(status) == STATUS_KEYS
    assert status["phase"] == ri.PHASE_RUNNING
    assert status["warning"] == "arm warning"
    assert status["transport"]["room"] == "rm"
    assert set(status["stats"]) == set(STATS_KEYS)
    ri._go_idle_locked()


def test_the_terminal_status_carries_every_key_and_is_idempotent() -> None:
    """Several surfaces poll this concurrently; a consume-once payload lets one
    poller swallow the error the user needed to see."""
    _live_session(phase=ri.PHASE_RUNNING)
    proc = FakeProc()
    proc.returncode = 0
    ri._remote_proc = proc

    first = ri.handle_remote_inference_status()
    second = ri.handle_remote_inference_status()
    assert set(first) == STATUS_KEYS
    assert first["exited"] is True
    assert first["outcome"] == "ok"
    assert second == first


def test_the_terminal_status_freezes_the_elapsed_time_instead_of_zeroing_it() -> None:
    """S3.4 overrode `elapsed_s` with 0.0 on the terminal payload, so a run that
    had just failed 40 seconds in reported "0s / 60" — reading as a run that
    never started, and losing the one number that says whether it died at once
    or ran most of its course first.

    The freeze needs no new clock: `_terminal_payload_locked` runs ONCE, at the
    exit, and builds on `_payload_locked`, whose `time.time() - started_at` is
    therefore measured to that moment and then stored verbatim in
    `_last_result`."""
    _live_session(phase=ri.PHASE_RUNNING)
    ri._remote_started_at = time.time() - 40.0

    payload = ri._terminal_payload_locked(
        exit_code=1, outcome="failed", error="the room went empty", phase=ri.PHASE_ERROR
    )
    assert 39.0 < payload["elapsed_s"] < 45.0
    ri._go_idle_locked()


def test_the_frozen_elapsed_time_does_not_keep_growing_after_the_exit() -> None:
    """The whole point of freezing it: once stored, the payload is a RECORD, and
    a later poll must report the run's length rather than "time since it
    started" ticking on forever."""
    _live_session(phase=ri.PHASE_RUNNING)
    ri._remote_started_at = time.time() - 12.0
    proc = FakeProc()
    proc.returncode = 0
    ri._remote_proc = proc

    first = ri.handle_remote_inference_status()
    ri._remote_started_at = time.time() - 9000.0  # a live run would now say 9000
    second = ri.handle_remote_inference_status()

    assert 11.0 < first["elapsed_s"] < 17.0
    assert second["elapsed_s"] == first["elapsed_s"]


def test_the_status_always_names_the_engine() -> None:
    """A panel that did not start the run still has to say which regime is
    driving the arm — and therefore which of the two `modal run` lines the other
    terminal must be running."""
    idle = ri.handle_remote_inference_status()
    assert idle["engine"] is None

    _live_session(phase=ri.PHASE_RUNNING, engine="rtc")
    assert ri.handle_remote_inference_status()["engine"] == "rtc"

    proc = FakeProc()
    proc.returncode = 0
    ri._remote_proc = proc
    assert ri.handle_remote_inference_status()["engine"] == "rtc"


def test_the_fingerprint_leads_with_the_engine() -> None:
    """The engine decides WHICH `modal run` line the operator's other terminal
    has to be running, and comparing that line against this phrase is the whole
    remedy the no-chunks watchdog offers."""
    text = ri._fingerprint(_request(engine="rtc", horizon=50, camera_bindings={"front": "wrist"}))
    assert text.startswith("engine=rtc, ")
    assert "horizon=50" in text
    assert "cameras=front" in text


def test_a_recorded_diagnosis_survives_a_clean_child_exit() -> None:
    """A watchdog asks the child to stop, and the child obliges cleanly (rc 0).
    The STOP was the remedy, not the verdict — reporting "ok" here would tell
    the operator their zero-chunk run finished fine."""
    _live_session(phase=ri.PHASE_STOPPING, error="no chunks in 10s")
    proc = FakeProc()
    proc.returncode = 0
    ri._remote_proc = proc

    status = ri.handle_remote_inference_status()
    assert status["outcome"] == "failed"
    assert status["error"] == "no chunks in 10s"
    assert status["phase"] == ri.PHASE_ERROR


def test_a_startup_failure_is_reported_the_way_local_inference_reports_one() -> None:
    _live_session(phase=ri.PHASE_PREFLIGHT)
    ri._fail_startup("The arm doesn't match its calibration.")

    status = ri.handle_remote_inference_status()
    assert status["exited"] is True
    assert status["outcome"] == "failed"
    assert status["phase"] == ri.PHASE_ERROR
    assert status["error"] == "The arm doesn't match its calibration."


# ---------------------------------------------------------------------------
# Transport provenance
# ---------------------------------------------------------------------------


def test_the_sfu_is_the_source_whenever_this_process_runs_one(monkeypatch) -> None:
    """Not conditioned on the url matching: when the SFU is up the session uses
    it unconditionally, so the url IS the SFU's by construction."""
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: True)

    assert ri._transport_source("ws://127.0.0.1:7880") == "sfu"


def test_the_saved_file_is_recognised_as_the_source(tmp_path, monkeypatch) -> None:
    """Which layer supplied the effective url decides what the "unreachable"
    message says, so it is read from the environment and the file rather than
    assumed."""
    pytest.importorskip("dotenv")
    saved = tmp_path / "livekit.env"
    saved.write_text("LIVEKIT_URL=wss://x.livekit.cloud\n")
    monkeypatch.setattr(ri, "DRTC_ENV_PATH", str(saved))
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)

    assert ri._transport_source("wss://x.livekit.cloud") == "cloud"


def test_the_process_environment_is_named_when_it_is_what_won(tmp_path, monkeypatch) -> None:
    """A remedy of its own: telling an operator to edit `livekit.env` while
    their shell is overriding it is the worst of the three answers."""
    pytest.importorskip("dotenv")
    saved = tmp_path / "livekit.env"
    saved.write_text("LIVEKIT_URL=wss://from-file\n")
    monkeypatch.setattr(ri, "DRTC_ENV_PATH", str(saved))
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")

    assert ri._transport_source("wss://from-shell") == "process_env"


def test_nothing_anywhere_is_none_rather_than_cloud(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ri, "DRTC_ENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)

    assert ri._transport_source("") == "none"
    assert ri._transport_source("wss://x.livekit.cloud") == "none"


# ---------------------------------------------------------------------------
# The Lab-owned SFU (makermodslab --sfu)
# ---------------------------------------------------------------------------


@pytest.fixture
def sfu_preflight(monkeypatch, preflight):
    """The `preflight` ladder, with the SFU branch taken instead of Cloud.

    `_read_env` is made to explode: the point of the branch is that no
    credential file is consulted at all when the Lab hosts the SFU."""

    def _boom():
        raise AssertionError("livekit.env was read while the Lab's own SFU is running")

    monkeypatch.setattr(ri, "_read_env", _boom)
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ri.sfu, "api_keys", lambda *a, **k: ("APIkey123", "s3cret"))
    monkeypatch.setattr(ri.sfu, "local_url", lambda *a, **k: "ws://127.0.0.1:7880")
    monkeypatch.setattr(ri.sfu, "default_room", lambda instance_id: "mml-abcdef012345")
    monkeypatch.setattr(ri.sfu, "mint_token", lambda **kw: (f"jwt.{kw['identity']}.{kw['room']}", 0))
    return preflight


def test_the_sfu_supplies_the_url_room_and_the_child_token(sfu_preflight) -> None:
    """The whole S3.6 contract in one assertion: the transport is minted here,
    the child is handed a token instead of a secret, and the identity is the
    exact string the room probe looks for on the other side."""
    result = ri.handle_start_remote_inference(_request())
    assert result["success"] is True
    _req, _robot_req, url, room, token, _cancel = sfu_preflight[0]
    assert url == "ws://127.0.0.1:7880"
    assert room == "mml-abcdef012345"
    assert token == "jwt.robot.mml-abcdef012345"
    assert ri._transport == {
        "url": "ws://127.0.0.1:7880",
        "room": "mml-abcdef012345",
        "source": "sfu",
        "operator_present": True,
    }
    ri._go_idle_locked()


def test_the_child_token_rides_the_argv_and_the_secret_never_does(sfu_preflight) -> None:
    """A token is short-lived and scoped to one room and one identity, which is
    why it may be passed on a command line at all; the API SECRET signs every
    token for the life of the install and must stay in its 0600 file."""
    args = ri._robot_sync_args(
        _request(), ["--robot.type=so101_follower"], url="ws://127.0.0.1:7880", room="r", token="jwt.x"
    )
    assert "--livekit_token=jwt.x" in args
    assert not any("s3cret" in a for a in args)


def test_no_token_flag_is_emitted_on_the_cloud_path() -> None:
    """The child mints its own there, and an empty `--livekit_token=` would
    read to draccus as a value rather than as an absence."""
    args = ri._robot_sync_args(_request(), [], url="wss://x", room="r", token="")
    assert not any(a.startswith("--livekit_token") for a in args)


def test_an_unreadable_key_file_refuses_before_the_arm_is_claimed(monkeypatch, sfu_preflight) -> None:
    """The launcher wrote that file before the app started, so this is a broken
    install rather than something the operator can fix in the panel — but it
    still has to be a refusal, not a session that energizes and then dies."""
    monkeypatch.setattr(ri.sfu, "api_keys", _raise_oserror)

    result = ri.handle_start_remote_inference(_request())
    assert result["success"] is False
    assert result["code"] == "transport.not_configured"
    assert ri.remote_inference_active is False
    assert sfu_preflight == []


def _raise_oserror(*args, **kwargs):
    raise OSError("permission denied")
