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
"""Tests for makermodslab.rollout — request schema, pure helpers, and the
non-subprocess branches of the start/stop/status handlers.

handle_start_inference's happy path spawns a real subprocess and a stdout-
pumping thread; covering it would require mocking subprocess.Popen, threading,
and setup_follower_calibration_file. We test only the early-return mutex
branches here — the parts that matter for safety."""

from __future__ import annotations

import io
import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_rollout_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset rollout's module-level state around each test so a leaking
    `inference_active=True` from one case can't poison the next."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_started_at", None)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_inference_cancel", None)
    monkeypatch.setattr(rollout, "_last_result", None)
    monkeypatch.setattr(rollout, "_inference_startup_thread", None)


class _SyncThread:
    """A ``threading.Thread`` stand-in whose ``.start()`` runs the target inline.

    The start handler now hands the heavy work (download → preflight → spawn) to
    a background ``threading.Thread``; patching it with this lets a test drive
    that worker — and the stdout-pump thread it in turn spawns — deterministically
    in the calling thread, no real threads or sleeps. Only the keyword call shape
    the code uses (``Thread(target=..., args=..., name=..., daemon=...)``) is
    supported."""

    def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class _EmptyStdout:
    """A subprocess ``stdout`` that is immediately at EOF, so the stdout pump's
    ``iter(proc.stdout.readline, b"")`` loop exits at once when a test runs it
    synchronously."""

    def readline(self) -> bytes:
        return b""


def test_inference_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from makermodslab.rollout import InferenceRequest

    with pytest.raises(ValidationError):
        InferenceRequest()


def test_inference_request_has_expected_defaults() -> None:
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.task == ""
    assert req.cameras == {}
    assert req.duration_s == 60


def test_inference_request_bimanual_fields_default_to_single() -> None:
    """A request that omits the bimanual block is single-arm — the right-arm
    fields are inert and `mode` defaults to 'single'."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.mode == "single"
    assert req.right_follower_port == ""
    assert req.right_follower_config == ""
    assert req.robot_name == ""
    assert req.checkpoint_state_dim is None


def test_inference_request_accepts_bimanual_block() -> None:
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        robot_name="dual_arm",
        checkpoint_state_dim=12,
    )
    assert req.mode == "bimanual"
    assert req.right_follower_port == "/dev/right"
    assert req.right_follower_config == "right_cal"
    assert req.robot_name == "dual_arm"
    assert req.checkpoint_state_dim == 12


# ---------------------------------------------------------------------------
# _arm_count_mismatch — the pre-spawn checkpoint/robot arm-count guard
# ---------------------------------------------------------------------------


def test_arm_count_mismatch_none_when_state_dim_unknown() -> None:
    """A checkpoint with no observation.state (state_dim None) can't be judged
    cheaply — defer to the subprocess's own shape check."""
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", None) is None
    assert _arm_count_mismatch("bimanual", None) is None


def test_arm_count_mismatch_none_when_single_matches_single() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 6) is None


def test_arm_count_mismatch_none_when_bimanual_matches_bimanual() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("bimanual", 12) is None


def test_arm_count_mismatch_flags_bimanual_checkpoint_on_single_robot() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("single", 12)
    assert msg is not None
    assert "bimanual" in msg
    assert "single-arm" in msg


def test_arm_count_mismatch_flags_single_checkpoint_on_bimanual_robot() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("bimanual", 6)
    assert msg is not None
    assert "single-arm" in msg
    assert "bimanual" in msg


def test_arm_count_mismatch_none_for_unrecognised_width() -> None:
    """A width that's neither a single arm nor a clean multiple is left to the
    subprocess rather than guessed at (e.g. 7 = 6 + an extra sensor dim)."""
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 7) is None
    assert _arm_count_mismatch("bimanual", 7) is None


def test_detect_device_returns_cpu_when_neither_cuda_nor_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _detect_device() == "cpu"


def test_detect_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "cuda"


def test_detect_device_falls_back_to_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "mps"


def test_detect_device_returns_cpu_when_torch_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function wraps both probes in a broad try/except — if torch is
    broken at runtime we still need a sensible fallback."""
    import torch

    from makermodslab.rollout import _detect_device

    def _boom() -> bool:
        raise RuntimeError("simulated torch.cuda failure")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    assert _detect_device() == "cpu"


def test_resolve_policy_path_returns_local_dir_unchanged(tmp_path) -> None:
    from makermodslab.rollout import _resolve_policy_path

    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()
    assert _resolve_policy_path(str(pretrained)) == str(pretrained)


def test_resolve_policy_path_raises_on_unparsable_ref() -> None:
    from makermodslab.rollout import _resolve_policy_path

    with pytest.raises(ValueError, match="Unrecognised policy ref"):
        _resolve_policy_path("not-a-real-ref-no-at-sign")


def test_resolve_policy_path_resolves_hub_ref(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Hub refs ('user/repo@checkpoints/000050') must be passed through
    snapshot_download and joined to the standard checkpoints/<step>/pretrained_model
    layout."""
    from makermodslab.rollout import _resolve_policy_path

    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen_kwargs: dict = {}

    def fake_snapshot_download(**kwargs):
        seen_kwargs.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = _resolve_policy_path("user/my-repo@checkpoints/000050")

    assert seen_kwargs["repo_id"] == "user/my-repo"
    assert seen_kwargs["repo_type"] == "model"
    assert seen_kwargs["allow_patterns"] == ["checkpoints/000050/pretrained_model/*"]
    assert result == str(fake_root / "checkpoints" / "000050" / "pretrained_model")


def test_resolve_policy_path_resolves_hub_root_ref(monkeypatch, tmp_path) -> None:
    """A flat-model ref ('user/repo@root') downloads the repo root and returns
    it — but excludes the checkpoints/ and training_state/ sub-trees (neither is
    needed to run inference, both can be multi-GB) so only the root pretrained
    files are pulled."""
    from makermodslab.rollout import _resolve_policy_path

    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = _resolve_policy_path("user/repo@root")
    assert seen["repo_id"] == "user/repo"
    # Byte-scoping: no allow_patterns (the whole root IS the model), but the
    # heavy sibling sub-trees are ignored.
    assert "allow_patterns" not in seen
    assert seen["ignore_patterns"] == ["checkpoints/**", "training_state/**"]
    assert result == str(fake_root)


def test_format_cameras_arg_empty_yields_empty_braces() -> None:
    from makermodslab.rollout import _format_cameras_arg

    assert _format_cameras_arg({}) == "{}"


def test_format_cameras_arg_renames_camera_index_to_index_or_path() -> None:
    """lerobot's CLI expects `index_or_path`, but the frontend posts
    `camera_index`. The rename is the whole point of this helper."""
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    )
    assert "index_or_path: 0" in result
    assert "camera_index" not in result
    assert result.startswith("{front: {")
    assert result.endswith("}}")


def test_format_cameras_arg_omits_none_values() -> None:
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg({"front": {"camera_index": 0, "fps": None}})
    assert "fps" not in result
    assert "index_or_path: 0" in result


def test_format_cameras_arg_handles_multiple_cameras() -> None:
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {
            "front": {"camera_index": 0, "fps": 30},
            "wrist": {"camera_index": 1, "fps": 30},
        }
    )
    assert "front: {" in result
    assert "wrist: {" in result


def test_handle_stop_inference_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_stop_inference

    result = handle_stop_inference()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_handle_inference_status_when_idle_returns_dict_with_expected_keys() -> None:
    from makermodslab.rollout import handle_inference_status

    result = handle_inference_status()
    assert isinstance(result, dict)
    assert result["inference_active"] is False
    assert result["phase"] is None
    for key in ("started_at", "rollout_started_at", "elapsed_s", "rollout_elapsed_s"):
        assert key in result


def _stub_request():
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )


def test_handle_start_inference_blocked_when_teleoperation_active(monkeypatch) -> None:
    """If teleop owns the bus, inference must refuse rather than race for
    the serial port."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_handle_start_inference_blocked_when_recording_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Recording" in result["message"]


def test_handle_start_inference_blocked_when_already_active(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"]


def test_handle_start_inference_blocked_when_calibration_active(monkeypatch) -> None:
    """Inference must refuse to start while manual calibration owns the same
    serial bus, rather than opening a second connection on a live port."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Calibration" in result["message"]


def test_handle_start_inference_blocked_when_auto_calibration_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Auto-calibration" in result["message"]


def test_handle_start_inference_blocked_when_wiggle_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "wiggle" in result["message"].lower()


def test_handle_start_inference_blocked_when_replay_active(monkeypatch) -> None:
    """Replay drives the same follower bus open-loop — inference must refuse
    to start while it's active, or both threads race to write goal positions
    to the same servos."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.replay.replay_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "replay" in result["message"].lower()


def test_handle_start_inference_pins_return_to_initial_position(monkeypatch, tmp_path) -> None:
    """The stop dialog promises the follower eases back to its start pose on
    teardown. That behaviour is lerobot's `return_to_initial_position`, which
    defaults to True today — but we pin it explicitly so an upstream default
    flip can't silently break the promise. Capture the rollout command and
    assert the flag is present.

    This is the one command-construction test: it stubs out the subprocess and
    every hardware-touching preflight so nothing real is started, runs the
    background startup worker synchronously (via the _SyncThread stub), and
    redirects HOME so the worker's log file lands in tmp rather than the real
    cache — we only inspect the argv handed to Popen. The resolve stub takes the
    `report` kwarg the worker now passes for download progress."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg: cfg)
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(
        rollout, "_resolve_policy_path", lambda ref, report=None: str(tmp_path / "pretrained_model")
    )
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    captured: dict = {}

    class _FakeProc:
        pid = 4321

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            # A stdin for the newline-seeding block, a stdout the pump can drain.
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    # Run the startup worker (and its stdout pump) inline.
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--return_to_initial_position=true" in cmd
    # Sanity: the core rollout invocation is intact around our pinned flag.
    assert "lerobot.scripts.lerobot_rollout" in cmd
    assert "--strategy.type=base" in cmd


# ---------------------------------------------------------------------------
# --robot.* arg construction — single vs bimanual (pure, no I/O)
# ---------------------------------------------------------------------------


def _bimanual_request():
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        robot_name="dual_arm",
    )


def test_single_robot_args_uses_so101_follower_type() -> None:
    from makermodslab.rollout import _single_robot_args

    args = _single_robot_args(_stub_request(), "robot_a")
    assert "--robot.type=so101_follower" in args
    assert "--robot.port=/dev/ttyUSB0" in args
    assert "--robot.id=robot_a" in args
    # No cameras on the stub request → no --robot.cameras arg.
    assert not any(a.startswith("--robot.cameras=") for a in args)


def test_single_robot_args_appends_cameras_when_present() -> None:
    from makermodslab.rollout import InferenceRequest, _single_robot_args

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        cameras={"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480}},
    )
    args = _single_robot_args(req, "robot_a")
    cam_arg = next(a for a in args if a.startswith("--robot.cameras="))
    assert "front:" in cam_arg
    assert "index_or_path: 0" in cam_arg


def test_bimanual_robot_args_uses_bi_so_follower_with_both_ports() -> None:
    from makermodslab.rollout import _bimanual_robot_args

    args = _bimanual_robot_args(_bimanual_request(), "dual_arm", "/staging/follower")
    assert "--robot.type=bi_so_follower" in args
    assert "--robot.id=dual_arm" in args
    assert "--robot.calibration_dir=/staging/follower" in args
    assert "--robot.left_arm_config.port=/dev/left" in args
    assert "--robot.right_arm_config.port=/dev/right" in args


def test_bimanual_robot_args_puts_cameras_on_left_arm_only() -> None:
    from makermodslab.rollout import InferenceRequest, _bimanual_robot_args

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        cameras={"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480}},
    )
    args = _bimanual_robot_args(req, "dual_arm", "/staging/follower")
    assert any(a.startswith("--robot.left_arm_config.cameras=") for a in args)
    assert not any(a.startswith("--robot.right_arm_config.cameras=") for a in args)


def test_build_rollout_cmd_wraps_robot_args_with_shared_flags() -> None:
    from makermodslab.rollout import _build_rollout_cmd

    robot_args = ["--robot.type=so101_follower", "--robot.port=/dev/ttyUSB0"]
    cmd = _build_rollout_cmd(_stub_request(), "/local/pretrained_model", robot_args)
    assert "lerobot.scripts.lerobot_rollout" in cmd
    assert "--strategy.type=base" in cmd
    assert "--policy.path=/local/pretrained_model" in cmd
    assert "--robot.type=so101_follower" in cmd
    assert "--return_to_initial_position=true" in cmd
    assert "--duration=60" in cmd


# ---------------------------------------------------------------------------
# handle_start_inference — the arm-count 409 guard (fires before any port opens)
# ---------------------------------------------------------------------------


def test_handle_start_inference_rejects_bimanual_checkpoint_on_single_robot() -> None:
    """A bimanual checkpoint on a single-arm robot returns 409 without opening
    any port or spawning a subprocess."""
    from makermodslab.rollout import InferenceRequest, handle_start_inference

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        mode="single",
        checkpoint_state_dim=12,
    )
    result = handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "bimanual" in result["message"]


def test_handle_start_inference_rejects_single_checkpoint_on_bimanual_robot() -> None:
    from makermodslab.rollout import InferenceRequest, handle_start_inference

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        checkpoint_state_dim=6,
    )
    result = handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "single-arm" in result["message"]


def test_handle_start_inference_arm_count_guard_releases_slot() -> None:
    """A rejected start must leave inference_active False so the next request
    isn't wedged behind a phantom session."""
    from makermodslab import rollout

    req = rollout.InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        mode="single",
        checkpoint_state_dim=12,
    )
    rollout.handle_start_inference(req)
    assert rollout.inference_active is False


def test_handle_start_inference_bimanual_builds_bi_so_follower_command(monkeypatch, tmp_path) -> None:
    """End-to-end (no hardware): a bimanual request stages the two follower
    calibrations and hands Popen a `bi_so_follower` argv with both ports and
    two stdin newlines (one prompt per sub-arm's connect()).

    Mirrors the pin-test's stub pattern: subprocess, the two preflights, and the
    staging helper are all replaced so nothing real runs; the startup worker (and
    its stdout pump) run inline via _SyncThread and HOME is redirected so the log
    file lands in tmp."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "bimanual_base_id", lambda name: "dual_arm")
    monkeypatch.setattr(
        rollout,
        "stage_bimanual_follower_calibrations",
        lambda *a, **k: ("/staging/follower", "dual_arm"),
    )
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(
        rollout, "_resolve_policy_path", lambda ref, report=None: str(tmp_path / "pretrained_model")
    )
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeProc:
        pid = 9999

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdin = _FakeStdin()
            self.stdout = _EmptyStdout()
            captured["stdin"] = self.stdin

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    result = rollout.handle_start_inference(_bimanual_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--robot.type=bi_so_follower" in cmd
    assert "--robot.left_arm_config.port=/dev/left" in cmd
    assert "--robot.right_arm_config.port=/dev/right" in cmd
    assert "--robot.calibration_dir=/staging/follower" in cmd
    # Two sub-arms → two seeded newlines (single-arm seeds only one).
    assert captured["stdin"].written == b"\n\n"


# ---------------------------------------------------------------------------
# Startup phase model — the "which substep am I in" status (download / subprocess
# fully MOCKED; no real inference, no hardware, no port opened).
# ---------------------------------------------------------------------------


def test_resolve_policy_path_sets_downloading_model_phase(monkeypatch, tmp_path) -> None:
    """During the Hub snapshot_download, an active session's phase must read
    `downloading_model` so the UI can name that (multi-second) wait."""
    from makermodslab import rollout

    # Seed a live meta the way handle_start_inference does before the download.
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})

    seen_phase: dict = {}

    def fake_snapshot_download(**kwargs):
        # Capture the phase *at the moment of download*, not after.
        seen_phase["phase"] = rollout._inference_meta.get("phase")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@root")

    assert seen_phase["phase"] == rollout.PHASE_DOWNLOADING_MODEL


def test_resolve_policy_path_local_dir_leaves_phase_untouched(monkeypatch, tmp_path) -> None:
    """A local checkpoint dir needs no download, so it must NOT flip the phase
    to downloading_model."""
    from makermodslab import rollout

    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})

    rollout._resolve_policy_path(str(pretrained))

    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_set_phase_noops_without_active_session(monkeypatch) -> None:
    """A late stdout line arriving after teardown (empty meta) can't resurrect
    a phase on an empty dict."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {})
    rollout._set_phase(rollout.PHASE_CONNECTING)
    assert rollout._inference_meta == {}


class _LineFeeder:
    def __init__(self, lines: list[bytes]) -> None:
        self._it = iter(lines + [b""])

    def readline(self) -> bytes:
        return next(self._it)


class _NullLog:
    def write(self, *a) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_pump_stdout_advances_phases_through_setup(monkeypatch) -> None:
    """The stdout pump walks loading_policy → connecting → running off the
    stable lerobot setup lines, then pins running at the rollout marker."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    phases_seen: list[str] = []

    real_set_phase = rollout._set_phase

    def recording_set_phase(phase: str) -> None:
        real_set_phase(phase)
        phases_seen.append(phase)

    monkeypatch.setattr(rollout, "_set_phase", recording_set_phase)

    class _Proc:
        stdout = _LineFeeder(
            [
                b"INFO Loading policy from 'user/repo'...\n",
                b"INFO Policy loaded: type=act, device=cpu\n",
                b"INFO Connecting robot (so101_follower)...\n",
                b"INFO Robot connected: so101_follower\n",
                b"INFO Rollout setup complete, starting rollout...\n",
                b"INFO step 0\n",
            ]
        )

    rollout._pump_stdout(_Proc(), _NullLog())

    assert phases_seen == [
        rollout.PHASE_LOADING_POLICY,
        rollout.PHASE_CONNECTING,
        rollout.PHASE_RUNNING,
    ]
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING
    # The marker also stamped the rollout-start time.
    assert rollout._inference_rollout_started_at is not None


def test_pump_stdout_does_not_regress_phase_after_marker(monkeypatch) -> None:
    """A setup-looking line AFTER the rollout marker must not drag a running
    session back to `connecting`."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    class _Proc:
        stdout = _LineFeeder(
            [
                b"INFO Rollout setup complete, starting rollout...\n",
                b"INFO Connecting robot (stray later mention)...\n",
            ]
        )

    rollout._pump_stdout(_Proc(), _NullLog())
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING


def test_start_inference_seeds_starting_phase(monkeypatch) -> None:
    """The start handler seeds a `starting` phase synchronously before handing
    off to the background worker, so the very first status poll can already name
    the wait. Here the worker Thread is a no-op — modelling the instant after the
    POST returns, before the worker has run — so the phase stays `starting`."""
    from makermodslab import rollout

    # A no-op Thread: the background startup worker is never actually run, so the
    # meta shows the state the POST left behind.
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True, result
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_STARTING


def test_stop_inference_sets_stopping_phase(monkeypatch) -> None:
    """A stop request stamps `stopping` on the meta before terminate/wait, so a
    racing status poll doesn't report a stale `running`."""
    from makermodslab import rollout

    phase_at_terminate: dict = {}

    class _FakeProc:
        def terminate(self):
            phase_at_terminate["phase"] = rollout._inference_meta.get("phase")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _FakeProc())
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert phase_at_terminate["phase"] == rollout.PHASE_STOPPING


def test_status_finalisation_reports_stopped_on_clean_exit(monkeypatch) -> None:
    """A subprocess that exited rc=0 finalises to the terminal `stopped` phase."""
    from makermodslab import rollout

    class _ExitedProc:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _ExitedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    result = rollout.handle_inference_status()
    assert result["exited"] is True
    assert result["phase"] == rollout.PHASE_STOPPED


def test_status_finalisation_reports_error_on_nonzero_exit(monkeypatch) -> None:
    """A non-zero exit code finalises to the terminal `error` phase."""
    from makermodslab import rollout

    class _CrashedProc:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _CrashedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_CONNECTING})

    result = rollout.handle_inference_status()
    assert result["exited"] is True
    assert result["phase"] == rollout.PHASE_ERROR


def test_terminal_status_is_idempotent_across_polls(monkeypatch) -> None:
    """The terminal payload must survive repeated polls, not report-once.

    Several surfaces poll /inference-status concurrently (session dialog +
    Deploy panel); with a consume-once payload, whichever poll lands first
    after the subprocess dies swallows the outcome/error/hint and the dialog
    misreports a crash as a clean finish. A new start clears the stored
    result so the next session's first poll reflects THAT session."""
    from makermodslab import rollout

    class _CrashedProc:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _CrashedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    first = rollout.handle_inference_status()  # finalises the exit
    second = rollout.handle_inference_status()  # a second poller must see the same
    assert first["exited"] is True and first["outcome"] == "failed"
    assert second == first

    # A new start supersedes the stored result.
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )
    assert rollout.handle_start_inference(_stub_request())["success"] is True
    status = rollout.handle_inference_status()
    assert status["inference_active"] is True
    assert "outcome" not in status


# ---------------------------------------------------------------------------
# I1: a stopped startup worker has no tracked thread handle, so it keeps
# touching hardware after stop() and a new session can't tell it's still
# running (unlike teleoperate.py's `teleoperation_thread`, which the start
# guard checks with `.is_alive()` before claiming the slot).
# ---------------------------------------------------------------------------


def test_stopped_startup_worker_blocks_a_new_session_from_starting(monkeypatch) -> None:
    """`_inference_cancel` only aborts the worker at coarse boundaries (before
    entering `_prepare_robot`, after it returns) — nothing checks the cancel
    flag WHILE `_prepare_robot` itself runs, and nothing tracks a handle to the
    worker thread. So a stop pressed mid-preflight lets the orphaned worker
    keep opening the bus / writing motor registers, and — because there is no
    handle to ask "is that worker still alive" — a brand-new start goes ahead
    and races it for the same serial port.

    Real thread, real Event: the worker blocks inside a stubbed
    `_prepare_robot` (standing in for the hardware-touching preflight) until
    released, letting the test control the exact interleaving stop() must
    protect against."""
    from makermodslab import rollout

    entered_preflight = threading.Event()
    release_preflight = threading.Event()

    def _blocking_prepare_robot(request):
        entered_preflight.set()
        assert release_preflight.wait(timeout=5), "test setup: preflight release never signalled"
        return [], []

    monkeypatch.setattr(rollout, "_prepare_robot", _blocking_prepare_robot)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: ref)

    created_threads: list[threading.Thread] = []
    real_thread = threading.Thread

    def _tracking_thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        created_threads.append(t)
        return t

    monkeypatch.setattr(rollout.threading, "Thread", _tracking_thread)

    try:
        first = rollout.handle_start_inference(_stub_request())
        assert first["success"] is True
        assert entered_preflight.wait(timeout=5), "worker never reached _prepare_robot"

        # Stop while the worker is INSIDE _prepare_robot — hardware is already
        # being touched, and the worker has no way to be interrupted mid-call.
        stopped = rollout.handle_stop_inference()
        assert stopped["success"] is True
        assert rollout.inference_active is False

        # The orphaned worker from the stopped session is still alive (stuck in
        # _prepare_robot) and still driving hardware. A new session must be
        # refused rather than being allowed to open the same serial port out
        # from under it.
        second = rollout.handle_start_inference(_stub_request())
        assert second["success"] is False, (
            "a new session was allowed to start while a stopped session's "
            "startup worker was still alive and touching hardware"
        )
        assert second["status_code"] == 409
    finally:
        release_preflight.set()
        for t in created_threads:
            t.join(timeout=5)


# ---------------------------------------------------------------------------
# I6: I1 added the is_alive() guard that refuses a NEW session while a
# stopped session's startup worker is still alive, but gave the operator no
# way to see that from /inference-status (idle looks identical either way)
# and no way to force/confirm it from a second stop-inference call (unlike
# teleoperate.py's second-stop, which joins the worker with a timeout and
# reports honestly). These tests cover that gap.
# ---------------------------------------------------------------------------


class _FakeStartupWorker:
    """Thread double for the orphaned inference-startup worker.

    ``dies_on_join`` controls whether ``join()`` simulates the worker actually
    finishing (mirrors teleoperate.py's test double) or simulates a worker
    still stuck inside the unjoinable ``_prepare_robot`` call (stays alive no
    matter how long the caller waits)."""

    def __init__(self, dies_on_join: bool) -> None:
        self._alive = True
        self._dies_on_join = dies_on_join
        self.joined_with_timeout: float | None = None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined_with_timeout = timeout
        if self._dies_on_join:
            self._alive = False


def test_inference_status_reports_shutting_down_when_startup_worker_orphaned(monkeypatch) -> None:
    """A stopped session whose startup worker hasn't exited yet must be
    visible on /inference-status — not indistinguishable from true idle."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_startup_thread", _FakeStartupWorker(dies_on_join=False))

    status = rollout.handle_inference_status()

    assert status["shutting_down"] is True


def test_inference_status_not_shutting_down_when_truly_idle() -> None:
    from makermodslab.rollout import handle_inference_status

    status = handle_inference_status()

    assert status["shutting_down"] is False


def test_second_stop_while_startup_worker_alive_joins_with_timeout_and_reports(monkeypatch) -> None:
    """Pressing Stop again while the orphaned startup worker is still alive
    must actually wait (bounded) for it, not just repeat a blanket 409 —
    mirrors teleoperate.py's second-stop-during-grace behavior, adapted for a
    worker with no cooperative cancellation checkpoint to force through."""
    from makermodslab import rollout

    worker = _FakeStartupWorker(dies_on_join=False)
    monkeypatch.setattr(rollout, "_inference_startup_thread", worker)

    result = rollout.handle_stop_inference()

    assert result["success"] is True
    assert worker.joined_with_timeout is not None, "second stop must join() the orphaned worker"
    assert result["shutting_down"] is True
    assert "shutting down" in result["message"].lower()


def test_second_stop_while_startup_worker_exits_during_join_reports_finished(monkeypatch) -> None:
    from makermodslab import rollout

    worker = _FakeStartupWorker(dies_on_join=True)
    monkeypatch.setattr(rollout, "_inference_startup_thread", worker)

    result = rollout.handle_stop_inference()

    assert result["success"] is True
    assert result.get("shutting_down") is not True
    assert "finished" in result["message"].lower()


def test_fail_startup_result_is_idempotent_across_polls(monkeypatch) -> None:
    """A pre-subprocess failure (download/preflight) persists the same way."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"policy_ref": "user/repo@root"})
    rollout._fail_startup("Failed to download the model: boom")

    first = rollout.handle_inference_status()
    second = rollout.handle_inference_status()
    assert first["exited"] is True and first["outcome"] == "failed"
    assert first["error"] == "Failed to download the model: boom"
    assert first["policy_ref"] == "user/repo@root"
    assert second == first


def test_classify_outcome_ok_warns_and_fails() -> None:
    from makermodslab.rollout import _classify_outcome

    # rc 0/None => the run was fine.
    assert _classify_outcome(0, True, "overload") == "ok"
    assert _classify_outcome(None, True, None) == "ok"
    # Non-zero AFTER the rollout started, with a torque-disable/overload on
    # shutdown => the skill ran; only cleanup tripped.
    assert _classify_outcome(1, True, "Motor 6 overload, torque_enable failed") == "ran_with_warning"
    # Never started, or an unrelated error => a real failure.
    assert _classify_outcome(1, False, "overload") == "failed"
    assert _classify_outcome(1, True, "could not connect to the arm") == "failed"
    # A connection lost mid-run (cable bumped while the policy is driving) is a
    # real failure, not a shutdown/cleanup warning — connection-loss markers are
    # deliberately excluded from the cleanup set.
    assert _classify_outcome(1, True, "DeviceNotConnectedError: follower is not connected") == "failed"


def test_friendly_hint_maps_common_failures() -> None:
    from makermodslab.utils.errors import friendly_hint

    assert "gripper" in (friendly_hint("Motor overload detected") or "").lower()
    assert "connect" in (friendly_hint("Failed to connect to the follower") or "").lower()
    assert friendly_hint("some unrecognised traceback") is None
    assert friendly_hint(None) is None


def test_extract_error_from_log_pulls_exception_tail(tmp_path) -> None:
    from makermodslab.rollout import _extract_error_from_log

    log = tmp_path / "rollout.log"
    log.write_text(
        "INFO starting rollout\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "RuntimeError: gripper overload during shutdown\n",
        encoding="utf-8",
    )
    out = _extract_error_from_log(str(log))
    assert out is not None and "RuntimeError: gripper overload during shutdown" in out
    assert _extract_error_from_log(None) is None
    assert _extract_error_from_log(str(tmp_path / "missing.log")) is None


def test_inference_in_use_path_none_when_idle() -> None:
    """No active inference -> no in-use path (delete guards stay open)."""
    from makermodslab import rollout

    assert rollout.inference_in_use_path() is None


def test_inference_in_use_path_returns_resolved_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """While a session is active, the accessor exposes the RESOLVED local
    checkpoint dir captured at start (not the possibly-hub policy_ref)."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"policy_ref": "user/repo@root", "policy_path": "/tmp/ckpt/pretrained_model"},
    )
    assert rollout.inference_in_use_path() == "/tmp/ckpt/pretrained_model"


# ---------------------------------------------------------------------------
# Navigate-first startup: the POST returns immediately and the heavy work
# (download → preflight → spawn) runs in the background worker. All of these
# fully MOCK snapshot_download / the subprocess — no network, no hardware.
# ---------------------------------------------------------------------------


def test_start_inference_returns_immediately_without_downloading(monkeypatch) -> None:
    """The whole point of the rework: the POST must not block on the Hub
    download. With the worker Thread stubbed to a no-op, the handler still
    returns success and claims the session — and snapshot_download is never
    touched on the request thread (it would raise here if it were)."""
    from makermodslab import rollout

    def _boom(**kwargs):
        raise AssertionError("snapshot_download must not run on the request thread")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True
    assert rollout.inference_active is True
    # Visible from the very first status poll, before the worker has run.
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_download_progress_reported_into_status(monkeypatch) -> None:
    """While a Hub checkpoint downloads, snapshot_download's byte updates flow
    through the progress tqdm into the meta, and /inference-status exposes them
    as download_bytes_done / _total / _percent. The total can arrive after some
    bytes (metadata discovery), which is exactly the refresh()-then-update()
    order huggingface_hub uses on the shared bytes bar."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def fake_snapshot_download(**kwargs):
        # huggingface_hub instantiates the shared bytes bar (unit="B"); a file's
        # size becoming known grows total via refresh(), chunks arrive via
        # update(n).
        cls = kwargs["tqdm_class"]
        bar = cls(total=None, unit="B")
        bar.total = 1000
        bar.refresh()
        bar.update(250)
        return "/tmp/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@checkpoints/000050", report=rollout._report_download_progress)

    assert rollout._inference_meta["phase"] == rollout.PHASE_DOWNLOADING_MODEL
    status = rollout.handle_inference_status()
    assert status["download_bytes_done"] == 250
    assert status["download_bytes_total"] == 1000
    assert status["download_percent"] == 25.0


def test_download_percent_is_none_until_total_known(monkeypatch) -> None:
    """Before any file size is known the total is None, so download_percent is
    None too → the UI shows an indeterminate bar rather than a bogus 0/0%."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@root"}
    )

    def fake_snapshot_download(**kwargs):
        cls = kwargs["tqdm_class"]
        bar = cls(total=None, unit="B")
        bar.update(128)  # bytes trickling in before any total is known
        return "/tmp/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@root", report=rollout._report_download_progress)

    status = rollout.handle_inference_status()
    assert status["download_bytes_done"] == 128
    assert status["download_bytes_total"] is None
    assert status["download_percent"] is None


def test_startup_download_failure_reports_failed_and_hint_without_spawn(monkeypatch) -> None:
    """A Hub download that raises (offline / 404 / disk full) is finalised as a
    `failed` outcome carrying the error text + a friendly hint — and no arm
    preflight runs and no subprocess spawns."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def _raise(ref, report=None):
        raise RuntimeError("Repository Not Found for url: https://huggingface.co/api/models/x")

    monkeypatch.setattr(rollout, "_resolve_policy_path", _raise)

    def _no_prepare(*a, **k):
        raise AssertionError("preflight must not run after a download failure")

    def _no_popen(*a, **k):
        raise AssertionError("no subprocess may spawn after a download failure")

    monkeypatch.setattr(rollout, "_prepare_robot", _no_prepare)
    monkeypatch.setattr(rollout.subprocess, "Popen", _no_popen)

    rollout._run_inference_startup(_stub_request(), threading.Event())

    assert rollout.inference_active is False
    status = rollout.handle_inference_status()
    assert status["exited"] is True
    assert status["outcome"] == "failed"
    assert status["phase"] == rollout.PHASE_ERROR
    assert "download" in (status["error"] or "").lower()
    # friendly_hint recognises the Hub-not-found token and adds a hint.
    assert status["hint"] is not None and "Hub" in status["hint"]


def test_stop_during_download_leaves_clean_idle_without_spawn(monkeypatch) -> None:
    """Pressing Stop while the model is still downloading tears the session down
    to a clean idle: the worker abandons after the download returns, never
    opening the bus (_prepare_robot) or spawning a subprocess. Models the real
    ordering — stop() with no subprocess yet flips the session idle and sets the
    cancel event; the in-flight download still finishes into the cache."""
    from makermodslab import rollout

    cancel = threading.Event()
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_cancel", cancel)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_DOWNLOADING_MODEL, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def _resolve_then_stop(ref, report=None):
        rollout.handle_stop_inference()
        return "/tmp/snap/pretrained_model"

    def _no_prepare(*a, **k):
        raise AssertionError("no bus may be opened after a stop during download")

    def _no_popen(*a, **k):
        raise AssertionError("no subprocess may spawn after a stop during download")

    monkeypatch.setattr(rollout, "_resolve_policy_path", _resolve_then_stop)
    monkeypatch.setattr(rollout, "_prepare_robot", _no_prepare)
    monkeypatch.setattr(rollout.subprocess, "Popen", _no_popen)

    rollout._run_inference_startup(_stub_request(), cancel)

    assert rollout.inference_active is False
    assert rollout._inference_proc is None
    assert rollout._inference_meta == {}
    assert rollout.handle_inference_status()["inference_active"] is False


def test_run_inference_startup_local_ref_skips_download_phase(monkeypatch, tmp_path) -> None:
    """A local checkpoint dir needs no download: the worker resolves it instantly,
    never enters the downloading_model phase, and proceeds straight to preflight
    + spawn."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING, "policy_ref": str(pretrained)}
    )

    phases: list[str] = []
    real_set_phase = rollout._set_phase

    def _rec(phase: str) -> None:
        phases.append(phase)
        real_set_phase(phase)

    monkeypatch.setattr(rollout, "_set_phase", _rec)
    monkeypatch.setattr(rollout, "_prepare_robot", lambda req: (["--robot.type=so101_follower"], []))
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    class _FakeProc:
        pid = 1

        def __init__(self, cmd, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    req = rollout.InferenceRequest(follower_port="/dev/x", follower_config="c", policy_ref=str(pretrained))
    rollout._run_inference_startup(req, threading.Event())

    assert rollout.PHASE_DOWNLOADING_MODEL not in phases
    assert rollout._inference_proc is not None
