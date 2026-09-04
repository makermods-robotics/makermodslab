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

import contextlib
import io
import json
import threading
import types
from pathlib import Path

import pytest

from makermodslab.eval_protocol import (
    CMD_EPISODE,
    CMD_QUIT,
    CMD_STOP,
    REASON_DURATION,
    REASON_STOPPED,
)


@pytest.fixture(autouse=True)
def _reset_rollout_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset rollout's module-level state around each test so a leaking
    `inference_active=True` from one case can't poison the next."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_started_at", None)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)
    monkeypatch.setattr(rollout, "_runner_ready", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_inference_cancel", None)
    monkeypatch.setattr(rollout, "_last_result", None)
    monkeypatch.setattr(rollout, "_inference_startup_thread", None)
    monkeypatch.setattr(rollout, "_eval_session", None)
    monkeypatch.setattr(rollout, "_coach_session", None)


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
    assert req.camera_bindings == {}
    assert req.camera_dims == {}
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


def test_inference_request_defaults_to_sync_engine() -> None:
    """Absent an explicit choice the request pins lerobot's own default, so
    adding the A/B knob can't change what an existing caller gets."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.inference_engine == "sync"


def test_inference_request_rejects_unknown_engine() -> None:
    """The field is a Literal, so a typo is a 422 at the API edge rather than a
    draccus parse crash inside the rollout subprocess."""
    from pydantic import ValidationError

    from makermodslab.rollout import InferenceRequest

    with pytest.raises(ValidationError):
        InferenceRequest(
            follower_port="/dev/ttyUSB0",
            follower_config="robot_a",
            policy_ref="user/repo@checkpoints/000050",
            inference_engine="async",
        )


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


# ---------------------------------------------------------------------------
# Hub ref → MakerMods Lab's own models store (design-debt F6).
#
# The Models page downloads with `local_dir=<makermodslab_models>/<repo_id>`, which
# huggingface_hub keeps entirely outside the shared hub cache — so a model the
# user already pulled there used to be downloaded a SECOND time on first
# inference. _resolve_policy_path now checks the local store first, but ONLY
# when the hub cache holds nothing for the repo (with a cache entry,
# snapshot_download is the revision-aware, self-deduping path and must win).
# No network anywhere: snapshot_download is monkeypatched to explode.
# ---------------------------------------------------------------------------

_FAKE_POLICY_CONFIG = '{"type": "act"}'


def _seed_models_store(
    monkeypatch,
    tmp_path,
    repo_id: str,
    *,
    step: str | None = None,
    flat: bool = False,
    weights: bool = True,
):
    """Point models._local_models_root at a tmp dir and populate one repo in it.

    `step` writes a `checkpoints/<step>/pretrained_model` tree (what a training
    repo download looks like); `flat` writes a root config (what a `@root` repo
    looks like). Both may be set, to build the ambiguous tree the `@root` branch
    deliberately refuses. Returns the repo dir, RESOLVED — _downloaded_model_dir
    resolves, and tmp_path is behind a symlink on macOS.

    `weights=False` reproduces an INTERRUPTED local_dir download: config plus the
    pre/post-processor safetensors, but no `model.safetensors`. That is the exact
    shape that bit the user — note it does contain `.safetensors` files, so any
    check looking merely for "some safetensors" would still be fooled.
    """
    import makermodslab.models as m

    store = tmp_path / "makermodslab_models"
    monkeypatch.setattr(m, "_local_models_root", lambda: store)
    repo_dir = store / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)

    def _populate(d):
        (d / "config.json").write_text(_FAKE_POLICY_CONFIG)
        (d / "train_config.json").write_text(_FAKE_POLICY_CONFIG)
        # Processor weights land early in a download and are NOT policy weights.
        (d / "preprocessor.safetensors").write_text("processor")
        (d / "postprocessor.safetensors").write_text("processor")
        if weights:
            (d / "model.safetensors").write_text("weights")

    if step is not None:
        pretrained = repo_dir / "checkpoints" / step / "pretrained_model"
        pretrained.mkdir(parents=True)
        _populate(pretrained)
    if flat:
        _populate(repo_dir)
    return repo_dir.resolve()


def _seed_hub_cache(monkeypatch, tmp_path, *, cached_repo: str | None = None, with_snapshot: bool = True):
    """Redirect HF_HUB_CACHE at a tmp dir, optionally holding one model repo."""
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.file_download import repo_folder_name

    cache = tmp_path / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    if cached_repo is not None:
        snapshots = cache / repo_folder_name(repo_id=cached_repo, repo_type="model") / "snapshots"
        snapshots.mkdir(parents=True)
        if with_snapshot:
            (snapshots / "deadbeef").mkdir()
    return cache


def _explode_snapshot_download(monkeypatch) -> None:
    """Any snapshot_download call in these tests is the bug under test."""

    def _boom(**kwargs):
        raise AssertionError(f"snapshot_download must not be called: {kwargs}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)


def test_resolve_policy_path_uses_local_store_for_checkpoint_ref(monkeypatch, tmp_path) -> None:
    """A `@checkpoints/<step>` ref whose repo is already in the local models
    store (and nowhere in the hub cache) resolves with zero network."""
    from makermodslab import rollout

    repo_dir = _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)
    _explode_snapshot_download(monkeypatch)

    result = rollout._resolve_policy_path("user/repo@checkpoints/000050")

    assert result == str(repo_dir / "checkpoints" / "000050" / "pretrained_model")


def test_resolve_policy_path_uses_local_store_for_root_ref(monkeypatch, tmp_path) -> None:
    """Same for a flat `@root` repo — the store's repo dir IS the pretrained
    dir, so it is returned verbatim."""
    from makermodslab import rollout

    repo_dir = _seed_models_store(monkeypatch, tmp_path, "user/flat", flat=True)
    _seed_hub_cache(monkeypatch, tmp_path)
    _explode_snapshot_download(monkeypatch)

    assert rollout._resolve_policy_path("user/flat@root") == str(repo_dir)


def test_resolve_policy_path_local_store_hit_leaves_phase_untouched(monkeypatch, tmp_path) -> None:
    """Nothing is fetched, so the UI must not be told it is downloading a model
    — the same contract the plain local-dir branch has."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)
    _explode_snapshot_download(monkeypatch)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})

    rollout._resolve_policy_path("user/repo@checkpoints/000050")

    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_resolve_policy_path_prefers_hub_when_repo_is_cached(monkeypatch, tmp_path) -> None:
    """With the repo in the hub cache, snapshot_download is the right call even
    though a local-store copy exists: it is revision-aware and only pulls what
    changed, so the user stays on `main` instead of being pinned to a possibly
    stale local copy."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/repo")
    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen: dict = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = rollout._resolve_policy_path("user/repo@checkpoints/000050")

    assert seen["repo_id"] == "user/repo"
    assert result == str(fake_root / "checkpoints" / "000050" / "pretrained_model")


def test_resolve_policy_path_downloads_when_requested_step_is_missing(monkeypatch, tmp_path) -> None:
    """The local store has the repo but not the step the user picked — that is a
    miss, not a substitution: fall through to the Hub."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)
    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: str(fake_root))

    result = rollout._resolve_policy_path("user/repo@checkpoints/000100")

    assert result == str(fake_root / "checkpoints" / "000100" / "pretrained_model")


def test_local_store_root_ref_refuses_a_checkpoints_tree(monkeypatch, tmp_path) -> None:
    """`@root` means "the repo root IS the pretrained_model". A local copy that
    resolves to a checkpoints sub-tree is a DIFFERENT tree than the Hub path
    would return, so the shortcut declines rather than substituting it."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)

    assert rollout._local_store_policy_path("user/repo", None) is None


def test_local_store_declines_unknown_repo(monkeypatch, tmp_path) -> None:
    """Nothing in the store for this repo → no shortcut."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)

    assert rollout._local_store_policy_path("someone/else", "000050") is None


def test_local_store_refuses_traversal_repo_id(monkeypatch, tmp_path) -> None:
    """`policy_ref` is user input and the hub-ref regex accepts any `[^@]+`, so
    a repo id that escapes the models root must never resolve (the guard comes
    from models._downloaded_model_dir)."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)

    assert rollout._local_store_policy_path("../../etc", None) is None


def test_hub_cache_has_repo_reads_the_on_disk_layout(monkeypatch, tmp_path) -> None:
    """A repo dir whose snapshots/ is empty (an interrupted or wiped entry) has
    nothing to dedupe against, so it counts as absent."""
    from makermodslab import rollout

    _seed_hub_cache(monkeypatch, tmp_path, cached_repo="user/repo")
    assert rollout._hub_cache_has_repo("user/repo") is True
    assert rollout._hub_cache_has_repo("user/other") is False

    _seed_hub_cache(monkeypatch, tmp_path / "b", cached_repo="user/repo", with_snapshot=False)
    assert rollout._hub_cache_has_repo("user/repo") is False


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


def test_handle_start_inference_blocked_when_a_training_run_is_active(monkeypatch) -> None:
    """The worst pairing of the six: both want several GB of VRAM.

    The training queue already waits for inference. Until it was mutual, this
    direction walked straight over a run the watchdog had just promoted — and
    whichever process lost the OOM lost hours, reported only as "Subprocess
    exited with code 1" with nothing tying it to this click."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.jobs.training_is_active", lambda: "ACT · user/ds")
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "ACT · user/ds" in result["message"]


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
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg, arm_type="so101": cfg)
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


def test_rollout_cli_args_emits_sync_engine_by_default() -> None:
    """`inference` is a draccus ChoiceRegistry field defaulting to sync
    upstream; we name it explicitly so an upstream flip can't silently change
    which engine drives the arm."""
    from makermodslab.rollout import _rollout_cli_args

    args = _rollout_cli_args(_stub_request(), "/tmp/pretrained_model", [])
    assert "--inference.type=sync" in args


def test_rollout_cli_args_forwards_the_engine_to_both_front_ends() -> None:
    """_rollout_cli_args is shared by the single-episode command and the eval
    runner — guard against the eval path drifting away from the rollout path."""
    from makermodslab.rollout import (
        InferenceRequest,
        _build_eval_runner_cmd,
        _build_rollout_cmd,
    )

    request = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        inference_engine="rtc",
        eval_episodes=5,
    )
    single = _build_rollout_cmd(request, "/tmp/pretrained_model", [])
    evalrun = _build_eval_runner_cmd(request, "/tmp/pretrained_model", [])
    assert "--inference.type=rtc" in single
    assert "--inference.type=rtc" in evalrun
    assert "--inference.type=sync" not in single
    assert "--inference.type=sync" not in evalrun


# ---------------------------------------------------------------------------
# --robot.* arg construction — single vs bimanual
#
# Cameras are no longer carried on the request: it names a robot record and
# binds each policy-expected camera name to one of that record's cameras, so
# the camera-bearing cases need a record on disk (via `_robot_record_with_cam`).
# ---------------------------------------------------------------------------


def _robot_record_with_cam(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Write a one-camera robot record into a redirected ROBOTS_PATH.

    ROBOTS_PATH is a module-level constant not covered by `tmp_lerobot_home`
    (same pattern as tests/test_utils_config.py's autouse fixture)."""
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


def test_single_robot_args_appends_bound_record_cameras(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding names a RECORD camera ("wrist"); the CLI arg is keyed by the
    POLICY-expected name ("front") and carries the record's own settings."""
    from makermodslab.rollout import InferenceRequest, _single_robot_args

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "solo")
    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        robot_name="solo",
        camera_bindings={"front": "wrist"},
    )
    args = _single_robot_args(req, "robot_a")
    cam_arg = next(a for a in args if a.startswith("--robot.cameras="))
    assert "front:" in cam_arg
    assert "wrist" not in cam_arg
    assert "index_or_path: 0" in cam_arg
    assert "width: 640" in cam_arg
    # Record-keeping keys never reach lerobot's config parser, and neither does
    # the record's own device identity (lerobot has no `unique_id` field).
    assert "device_id" not in cam_arg
    assert "id:" not in cam_arg
    assert "unique_id" not in cam_arg


def test_single_robot_args_captures_at_the_checkpoints_resolution(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollout pipeline doesn't resize frames to the policy's input shape,
    so `camera_dims` (from the checkpoint) must win over the record's own
    configured size — while identity still comes from the record."""
    from makermodslab.rollout import InferenceRequest, _single_robot_args

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "solo")
    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        robot_name="solo",
        camera_bindings={"front": "wrist"},
        camera_dims={"front": {"width": 320, "height": 240}},
    )
    cam_arg = next(a for a in _single_robot_args(req, "robot_a") if a.startswith("--robot.cameras="))

    assert "width: 320" in cam_arg
    assert "height: 240" in cam_arg
    assert "width: 640" not in cam_arg
    assert "index_or_path: 0" in cam_arg


def test_bimanual_robot_args_uses_bi_so_follower_with_both_ports() -> None:
    from makermodslab.rollout import _bimanual_robot_args

    args = _bimanual_robot_args(_bimanual_request(), "dual_arm", "/staging/follower")
    assert "--robot.type=bi_so_follower" in args
    assert "--robot.id=dual_arm" in args
    assert "--robot.calibration_dir=/staging/follower" in args
    assert "--robot.left_arm_config.port=/dev/left" in args
    assert "--robot.right_arm_config.port=/dev/right" in args


def test_bimanual_robot_args_puts_cameras_on_left_arm_only(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab.rollout import InferenceRequest, _bimanual_robot_args

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "dual_arm")
    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        robot_name="dual_arm",
        camera_bindings={"front": "wrist"},
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


def test_build_rollout_cmd_omits_temporal_ensemble_when_unset() -> None:
    """Default (None) must leave the checkpoint's own config untouched — no
    --policy.temporal_ensemble_coeff, and crucially no n_action_steps override
    that would silently re-tune a policy the user didn't ask to change."""
    from makermodslab.rollout import _build_rollout_cmd

    cmd = _build_rollout_cmd(_stub_request(), "/local/pretrained_model", [])
    assert not any(a.startswith("--policy.temporal_ensemble_coeff=") for a in cmd)
    assert not any(a.startswith("--policy.n_action_steps=") for a in cmd)


def test_build_rollout_cmd_pins_n_action_steps_with_temporal_ensemble() -> None:
    """ACT raises NotImplementedError when temporal_ensemble_coeff is set with
    n_action_steps > 1 (checkpoints ship 100), so the two flags must always
    travel together."""
    from makermodslab.rollout import InferenceRequest, _build_rollout_cmd

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        temporal_ensemble_coeff=0.01,
    )
    cmd = _build_rollout_cmd(req, "/local/pretrained_model", [])
    assert "--policy.temporal_ensemble_coeff=0.01" in cmd
    assert "--policy.n_action_steps=1" in cmd


def test_eval_runner_cmd_carries_temporal_ensemble_flags() -> None:
    """The flags live in the shared _rollout_cli_args, so evaluation mode's
    separate entry point gets them too — a run scored over N episodes must use
    the same action selection the single-rollout path would."""
    from makermodslab.rollout import InferenceRequest, _build_eval_runner_cmd

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        eval_episodes=5,
        temporal_ensemble_coeff=0.01,
    )
    cmd = _build_eval_runner_cmd(req, "/local/pretrained_model", [])
    assert "makermodslab.eval_runner" in cmd
    assert "--policy.temporal_ensemble_coeff=0.01" in cmd
    assert "--policy.n_action_steps=1" in cmd


def test_handle_start_inference_rejects_non_positive_ensemble_coeff() -> None:
    """Weights are exp(-coeff * i): 0 weights the whole chunk equally and a
    negative coefficient makes the STALEST prediction dominate. Reject before
    the arm moves under it — and release the session slot on the way out."""
    from makermodslab import rollout
    from makermodslab.rollout import InferenceRequest, handle_start_inference

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        temporal_ensemble_coeff=0,
    )
    result = handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "temporal_ensemble_coeff" in result["message"]
    assert rollout.inference_active is False


# ---------------------------------------------------------------------------
# Real-Time Chunking capability — the table, and the pre-flight guard that
# refuses an rtc launch on an architecture that cannot run it. Both matter
# because the fork only discovers this AFTER the policy is loaded and the arm
# is claimed (lerobot/rollout/context.py, supports_rtc_inference).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy_type", "expected"),
    [
        # Declare supports_rtc() -> True in the pinned fork.
        ("smolvla", True),
        ("pi0", True),
        ("pi05", True),
        ("evo1", True),
        ("groot", True),
        # Per-CONFIG in the fork (inference_action_mode == "continuous"), so the
        # architecture answer is True and the subprocess enforces the rest.
        ("molmoact2", True),
        # Inherit PreTrainedPolicy.supports_rtc -> False.
        ("act", False),
        ("diffusion", False),
        ("tdmpc", False),
        ("vqbet", False),
        # The sibling that does NOT support it, unlike pi0/pi05.
        ("pi0_fast", False),
        # Not registered in the pinned fork: "not established", never "no".
        ("some_future_policy", None),
        ("", None),
    ],
)
def test_policy_type_supports_rtc_decisions(policy_type: str, expected: bool | None) -> None:
    from makermodslab.rollout import policy_type_supports_rtc

    assert policy_type_supports_rtc(policy_type) is expected


def test_rtc_table_matches_the_pinned_fork() -> None:
    """Drift guard: the table is a hand-mirrored read of lerobot's policy
    classes, so re-derive it from the fork's SOURCES on every run.

    Reads the files rather than importing them — importing
    lerobot.policies.factory costs ~2 s and pulls transformers, which is
    exactly the cost policy_type_supports_rtc exists to avoid.
    """
    import re as _re

    import lerobot.policies as _policies_pkg
    from makermodslab.jobs import _KNOWN_POLICY_TYPES, _RTC_CAPABLE_POLICY_TYPES

    root = Path(_policies_pkg.__file__).parent
    registered: set[str] = set()
    declares_rtc: set[str] = set()
    for cfg in root.rglob("configuration_*.py"):
        for name in _re.findall(r'@PreTrainedConfig\.register_subclass\("([^"]+)"\)', cfg.read_text()):
            registered.add(name)
            modeling = cfg.with_name(cfg.name.replace("configuration_", "modeling_", 1))
            if modeling.is_file() and "def supports_rtc(" in modeling.read_text():
                declares_rtc.add(name)

    assert registered, f"no registered policy types found under {root}"
    assert registered == set(_KNOWN_POLICY_TYPES)
    assert declares_rtc == set(_RTC_CAPABLE_POLICY_TYPES)


def _rtc_request(policy_ref: str = "user/repo@checkpoints/000050", **kwargs):
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref=policy_ref,
        inference_engine="rtc",
        **kwargs,
    )


def test_handle_start_inference_refuses_rtc_on_an_act_checkpoint(monkeypatch) -> None:
    """400 in the launch panel, before the model download and before any port
    is opened — and the session slot is handed back."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "read_pretrained_policy_type", lambda ref: "act")
    monkeypatch.setattr(
        rollout.camera_preview_manager,
        "stop_all",
        lambda: pytest.fail("refused start must not disturb the camera previews"),
    )
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: pytest.fail("no startup worker may spawn")
    )

    result = rollout.handle_start_inference(_rtc_request())

    assert result["success"] is False
    assert result["status_code"] == 400
    assert result["message"] == (
        "Real-Time Chunking isn't available for act checkpoints; use the standard (sync) engine."
    )
    assert rollout.inference_active is False


def test_handle_start_inference_allows_rtc_on_a_supporting_checkpoint(monkeypatch) -> None:
    """A smolvla checkpoint passes the RTC gate — proven by the request landing
    on the NEXT guard (camera bindings with no robot record) instead."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "read_pretrained_policy_type", lambda ref: "smolvla")

    result = rollout.handle_start_inference(_rtc_request(camera_bindings={"front": "wrist"}))

    assert result["success"] is False
    assert "Real-Time Chunking" not in result["message"]
    assert "No robot selected" in result["message"]
    assert rollout.inference_active is False


def test_handle_start_inference_allows_rtc_on_an_unknown_policy_type(monkeypatch) -> None:
    """A type newer than our table, and an unreadable config, both mean "not
    established" — neither may refuse a run the subprocess would accept."""
    from makermodslab import rollout

    for policy_type in ("some_future_policy", None):
        monkeypatch.setattr(rollout, "read_pretrained_policy_type", lambda ref, t=policy_type: t)
        result = rollout.handle_start_inference(_rtc_request(camera_bindings={"front": "wrist"}))
        assert "Real-Time Chunking" not in result["message"]
        assert rollout.inference_active is False


def test_handle_start_inference_reads_no_config_for_the_sync_engine(monkeypatch) -> None:
    """sync runs on every architecture, so the gate must not cost a config read
    (which is an hf_hub_download for a Hub ref) on the ordinary path."""
    from makermodslab import rollout

    monkeypatch.setattr(
        rollout,
        "read_pretrained_policy_type",
        lambda ref: pytest.fail("sync must not read the checkpoint config"),
    )
    req = _rtc_request(camera_bindings={"front": "wrist"})
    req.inference_engine = "sync"
    result = rollout.handle_start_inference(req)
    # Stopped by the NEXT guard (bindings with no robot record), not this one.
    assert result["success"] is False
    assert "No robot selected" in result["message"]


def test_rtc_gate_asks_about_a_root_ref_by_its_bare_repo_id(monkeypatch) -> None:
    """'user/repo@root' means the repo root IS the pretrained dir, which is
    where read_pretrained_policy_type looks for a plain repo id. Passing the
    suffixed ref through would fail the lookup and silently skip the guard."""
    from makermodslab import rollout

    seen: list[str] = []

    def _record(ref: str) -> str:
        seen.append(ref)
        return "act"

    monkeypatch.setattr(rollout, "read_pretrained_policy_type", _record)
    result = rollout.handle_start_inference(_rtc_request(policy_ref="user/repo@root"))

    assert seen == ["user/repo"]
    assert result["status_code"] == 400
    assert "Real-Time Chunking" in result["message"]


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


# ---------------------------------------------------------------------------
# handle_start_inference — camera bindings resolve against the ROBOT RECORD
# (cheap: one JSON read, no hardware, no subprocess). The pure resolution
# helpers are covered in tests/test_utils_config.py.
# ---------------------------------------------------------------------------


def test_handle_start_inference_rejects_a_binding_with_no_robot() -> None:
    """Bindings name a record camera, so they're meaningless without a record."""
    from makermodslab import rollout

    req = rollout.InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        camera_bindings={"front": "wrist"},
    )
    result = rollout.handle_start_inference(req)

    assert result["success"] is False
    assert result["status_code"] == 400
    assert "No robot selected" in result["message"]
    # A rejected start must not wedge the slot.
    assert rollout.inference_active is False


def test_handle_start_inference_rejects_a_binding_to_an_unknown_camera(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 400 names what the robot actually has, so the panel can say which
    camera to pick — this is the whole point of resolving server-side."""
    from makermodslab import rollout

    _robot_record_with_cam(tmp_lerobot_home, monkeypatch, "solo")
    req = rollout.InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        robot_name="solo",
        camera_bindings={"front": "gone"},
    )
    result = rollout.handle_start_inference(req)

    assert result["success"] is False
    assert result["status_code"] == 400
    assert "'gone'" in result["message"]
    assert "wrist" in result["message"]
    assert rollout.inference_active is False


def test_handle_start_inference_ignores_a_stale_cameras_payload() -> None:
    """An older frontend still posts full `cameras` configs. The field is gone
    from the model, so pydantic drops them rather than letting a request-side
    camera set drive the run."""
    from makermodslab import rollout

    req = rollout.InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        cameras={"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480}},
    )

    assert not hasattr(req, "cameras")
    assert req.camera_bindings == {}


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
    # shutdown => the policy ran; only cleanup tripped.
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


def test_is_out_of_memory_matches_every_allocator_backend() -> None:
    """The wording differs per backend, so each one is keyed separately."""
    from makermodslab.utils.errors import is_out_of_memory

    for text in (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: HIP out of memory. Tried to allocate 512.00 MiB",
        "RuntimeError: MPS backend out of memory (MPS allocated: 9.06 GB)",
        "RuntimeError: CUDA error: out of memory",
        "RuntimeError: [enforce fail] DefaultCPUAllocator: can't allocate memory: you tried to allocate",
    ):
        assert is_out_of_memory(text), text
    assert not is_out_of_memory("RuntimeError: shape mismatch")
    assert not is_out_of_memory("")
    assert not is_out_of_memory(None)


def test_is_out_of_memory_ignores_a_bare_killed() -> None:
    """ "Killed" alone is not evidence: the host OOM killer prints nothing, and
    plenty of benign lines contain the word. That case is recognised from the
    exit code instead (see jobs._oom_failure_reason)."""
    from makermodslab.utils.errors import is_out_of_memory

    assert not is_out_of_memory("Killed")
    assert not is_out_of_memory("stop requested: killed the training subprocess")


def test_friendly_hint_names_the_three_oom_remedies() -> None:
    from makermodslab.utils.errors import friendly_hint

    hint = (friendly_hint("torch.OutOfMemoryError: CUDA out of memory.") or "").lower()
    assert "mixed precision" in hint
    assert "batch size" in hint
    assert "larger gpu" in hint


def test_friendly_hint_servo_bus_error_is_not_a_download_failure() -> None:
    """A servo that stops answering must read as an ARM problem.

    lerobot's motors bus raises every serial failure as `ConnectionError`, and
    the type name itself contains "connect" — keying the Hub-download hint on
    that token labelled arm-side startup crashes "couldn't download the model"
    (observed 2026-08-03: a `Failed to write 'Lock' ... [TxRxResult] Incorrect
    status packet!` at robot.connect() reported as a failed model download)."""
    from makermodslab.utils.errors import friendly_hint

    for text in (
        "ConnectionError: Failed to write 'Lock' on id_=3 with '1' after 1 tries. "
        "[TxRxResult] Incorrect status packet!",
        "Failed to start inference: ConnectionError: Failed to sync read 'Present_Position' "
        "on ids=[1, 2, 3] after 1 tries. [TxRxResult] There is no status packet!",
    ):
        hint = friendly_hint(text) or ""
        assert "motor" in hint.lower()
        assert "download" not in hint.lower()


def test_friendly_hint_still_names_real_download_failures() -> None:
    """The other side of the tightening: a genuine fetch failure keeps its Hub
    hint. Download-step failures reach here with rollout's own
    "Failed to download the model: …" prefix (see _inference_startup_thread)."""
    from makermodslab.utils.errors import friendly_hint

    network = friendly_hint(
        "Failed to download the model: (MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', "
        'port=443): Max retries exceeded with url: /api/models/user/repo"))'
    )
    assert network is not None and "download the model" in network.lower()
    # No hub host in the text at all — rollout's prefix is what identifies it.
    offline = friendly_hint(
        "Failed to download the model: An error happened while trying to locate the file on the Hub "
        "and we cannot find the requested files in the local cache. Please check your connection."
    )
    assert offline is not None and "download the model" in offline.lower()
    missing = friendly_hint(
        "Failed to download the model: RepositoryNotFoundError: 404 Client Error. "
        "Repository Not Found for url: https://huggingface.co/api/models/user/repo"
    )
    assert missing is not None and "hub" in missing.lower()
    full = friendly_hint("Failed to download the model: OSError: [Errno 28] No space left on device")
    assert full is not None and "disk space" in full.lower()


def test_friendly_hint_for_capture_size_mismatch_also_suggests_restart() -> None:
    """Hardware-confirmed 2026-07-31: a camera reporting the wrong
    width/height on connect is not always a genuine capability mismatch —
    twice on the bench, the SAME camera kept failing with this identical
    error for 15+ seconds and across a retry with no unplugging involved,
    and only restarting the makermodslab process (not reconfiguring the camera)
    cleared it. Sending the user to "click Auto" alone, with no escalation
    path, is a dead end when the session is actually wedged rather than
    misconfigured — the hint must mention restarting as the fallback."""
    from makermodslab.utils.errors import friendly_hint

    hint = (
        friendly_hint(
            "OpenCVCamera(0) failed to set capture_width=640 (actual_width=1920, width_success=True)."
        )
        or ""
    )
    assert "restart" in hint.lower()

    height_hint = (
        friendly_hint(
            "OpenCVCamera(0) failed to set capture_height=480 (actual_height=1080, height_success=True)."
        )
        or ""
    )
    assert "restart" in height_hint.lower()


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


# ---------------------------------------------------------------------------
# Multi-episode EVALUATION mode
#
# Pure helpers (clamping, accuracy math, verdict classification, protocol
# parsing), the request schema, the status-payload shape, the orchestrator state
# machine driven over a fake runner pipe, and the idle/mutex branches of the two
# new endpoints. Nothing here spawns a process or touches hardware: the eval
# runner is never executed, only stood in for. Per CLAUDE.md the subprocess
# happy path stays untested — what IS tested is the bookkeeping either side of
# it, which is where the verdicts and the crash containment live.
# ---------------------------------------------------------------------------


class _ExitedProc:
    """A `subprocess.Popen` stand-in that has already exited with `rc`."""

    def __init__(self, rc: int = 0) -> None:
        self.returncode = rc
        self.pid = 4242
        # A dead process has no usable command pipe — writing to it is exactly
        # how the orchestrator discovers the runner is gone.
        self.stdin = None

    def poll(self) -> int:
        return self.returncode


class _CommandPipe:
    """A subprocess `stdin` that records the command lines written to it."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, data: bytes) -> None:
        self._sink.append(data.decode().strip())

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeRunner:
    """A live `makermodslab.eval_runner` subprocess stand-in.

    Unlike `_ExitedProc` it is ALIVE (`poll()` → None) and stays alive across
    episodes, which is the property the redesign turns on: a live process no
    longer implies a live episode, so the tests drive episode boundaries through
    the protocol handlers rather than by pretending a process exited. Every
    command the orchestrator sends lands in `.commands`."""

    def __init__(self, rc: int | None = None) -> None:
        self.pid = 4242
        self.returncode = rc
        self.commands: list[str] = []
        self.stdin = _CommandPipe(self.commands)

    def poll(self) -> int | None:
        return self.returncode


def _eval_request(episodes: int = 3):
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        eval_episodes=episodes,
    )


def _arm_eval_session(
    monkeypatch,
    rollout,
    episodes: int = 3,
    *,
    running: bool = True,
    proc=None,
):
    """Put the module into a mid-eval state with a live runner.

    `running` is whether an EPISODE is in flight, which is now independent of
    whether the runner process is up — pass `proc=None` for the (recoverable)
    state left behind by a runner that died."""
    session = rollout._EvalSession(request=_eval_request(episodes), episodes_total=episodes)
    session.policy_path = "/tmp/policy"
    session.robot_args = ["--robot.type=so101_follower"]
    session.episode_running = running
    monkeypatch.setattr(rollout, "_eval_session", session)
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_started_at", 1000.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", 1005.0 if running else None)
    # An episode in flight means the runner reported READY long ago; a session
    # parked in a reset with a dead runner has not (its respawn earns a fresh
    # one).
    monkeypatch.setattr(rollout, "_runner_ready", running)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {
            "phase": rollout.PHASE_RUNNING if running else rollout.PHASE_RESETTING,
            "policy_ref": "user/repo@checkpoints/000050",
            "duration_s": 60,
        },
    )
    monkeypatch.setattr(rollout, "_inference_proc", _FakeRunner() if (proc is None and running) else proc)
    return session


def test_eval_episodes_defaults_to_one() -> None:
    """The historical single-rollout request is unchanged: no eval fields set."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.eval_episodes == 1


def test_inference_request_accepts_eval_episodes() -> None:
    assert _eval_request(10).eval_episodes == 10


def test_clamp_eval_episodes_bounds_and_fallbacks() -> None:
    from makermodslab.rollout import MAX_EVAL_EPISODES, clamp_eval_episodes

    assert clamp_eval_episodes(1) == 1
    assert clamp_eval_episodes(10) == 10
    assert clamp_eval_episodes(0) == 1
    assert clamp_eval_episodes(-5) == 1
    assert clamp_eval_episodes(MAX_EVAL_EPISODES) == MAX_EVAL_EPISODES
    assert clamp_eval_episodes(10_000) == MAX_EVAL_EPISODES
    # Junk degrades to a single episode rather than raising out of the POST.
    assert clamp_eval_episodes(None) == 1
    assert clamp_eval_episodes("nope") == 1


def test_eval_accuracy_scores_successes_over_scored_episodes() -> None:
    from makermodslab.rollout import eval_accuracy

    assert eval_accuracy(["success", "success", "failure", "failure"]) == 0.5
    assert eval_accuracy(["success"]) == 1.0
    assert eval_accuracy(["failure", "failure"]) == 0.0


def test_eval_accuracy_excludes_errored_episodes_from_the_denominator() -> None:
    """A serial glitch must not poison the number: an errored episode counts
    neither for nor against the policy."""
    from makermodslab.rollout import eval_accuracy

    # 1 success, 1 failure, 2 crashes -> 1/2, not 1/4.
    assert eval_accuracy(["success", "failure", "error", "error"]) == 0.5


def test_eval_accuracy_is_none_when_nothing_scoreable() -> None:
    from makermodslab.rollout import eval_accuracy

    assert eval_accuracy([]) is None
    assert eval_accuracy(["error", "error"]) is None


def test_classify_episode_early_stop_is_a_success_whatever_the_exit_code() -> None:
    """We terminated the subprocess ourselves, so its exit code is our own
    SIGTERM and says nothing about the run."""
    from makermodslab.rollout import EPISODE_SUCCESS, classify_episode

    assert classify_episode(-15, True, True, None) == EPISODE_SUCCESS
    assert classify_episode(0, True, True, None) == EPISODE_SUCCESS
    assert classify_episode(1, True, True, "RuntimeError: boom") == EPISODE_SUCCESS


def test_classify_episode_clean_timeout_is_a_failure() -> None:
    from makermodslab.rollout import EPISODE_FAILURE, classify_episode

    assert classify_episode(0, False, True, None) == EPISODE_FAILURE


def test_classify_episode_cleanup_warning_is_a_failure_not_an_error() -> None:
    """The rollout ran its full duration and only teardown was noisy — the
    episode legitimately timed out, so it's a failure, not a crash."""
    from makermodslab.rollout import EPISODE_FAILURE, classify_episode

    # "overload" is one of errors.CLEANUP_MARKERS — a gripper still holding an
    # object when torque is released at teardown.
    verdict = classify_episode(1, False, True, "RuntimeError: motor 6 overload during shutdown")
    assert verdict == EPISODE_FAILURE


def test_classify_episode_crash_is_an_error() -> None:
    from makermodslab.rollout import EPISODE_ERROR, classify_episode

    assert classify_episode(1, False, False, "DeviceNotConnectedError: bus is gone") == EPISODE_ERROR


def test_eval_fields_are_null_shaped_for_a_single_episode_run() -> None:
    """The status shape stays stable: a plain run reports eval_mode False with
    null companions rather than omitting the keys."""
    from makermodslab.rollout import _eval_fields

    fields = _eval_fields(None)
    assert fields["eval_mode"] is False
    assert fields["episode_index"] is None
    assert fields["episodes_total"] is None
    assert fields["episode_results"] is None
    assert fields["accuracy"] is None


def test_idle_status_reports_single_episode_shape() -> None:
    from makermodslab.rollout import handle_inference_status

    result = handle_inference_status()
    assert result["eval_mode"] is False
    assert result["episodes_total"] is None
    assert result["accuracy"] is None


def test_start_inference_seeds_the_eval_session(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    result = rollout.handle_start_inference(_eval_request(5))
    assert result["success"] is True
    assert rollout._eval_session is not None
    assert rollout._eval_session.episodes_total == 5
    assert rollout._eval_session.episode_index == 1
    status = rollout.handle_inference_status()
    assert status["eval_mode"] is True
    assert status["episodes_total"] == 5
    assert status["episode_index"] == 1
    assert status["episode_results"] == []
    assert status["accuracy"] is None


def test_start_inference_with_one_episode_leaves_eval_session_none(monkeypatch) -> None:
    """eval_episodes=1 must be bit-for-bit the historical flow."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    rollout.handle_start_inference(_eval_request(1))
    assert rollout._eval_session is None
    assert rollout.handle_inference_status()["eval_mode"] is False


def test_start_inference_clamps_the_requested_episode_count(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    rollout.handle_start_inference(_eval_request(10_000))
    assert rollout._eval_session.episodes_total == rollout.MAX_EVAL_EPISODES


def test_start_inference_guard_failure_clears_the_eval_session(monkeypatch) -> None:
    """A rejected start must not leave eval bookkeeping behind for the next run."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: False)
    result = rollout.handle_start_inference(_eval_request(4))
    assert result["success"] is False
    assert rollout._eval_session is None
    assert rollout.inference_active is False


def test_episode_timeout_parks_the_session_in_the_reset_phase(monkeypatch) -> None:
    """An episode that runs out its duration scores a FAILURE and keeps the
    session — and its hold on the inference slot, the cameras AND the loaded
    policy — alive for the reset."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert status["episode_results"] == ["failure"]
    assert status["episode_index"] == 2
    assert status["accuracy"] is None
    # The slot stays claimed through the reset — recording/teleop stay blocked.
    assert rollout.inference_active is True
    # And so does the runner: the whole point is that the next episode does not
    # re-pay the policy load. It is only told to QUIT once the session is over.
    assert rollout._inference_proc is runner
    assert runner.commands == []


def test_early_stop_scores_the_episode_a_success(monkeypatch) -> None:
    """The success button asks the runner to end the episode — no signal, no
    kill — and the verdict lands when the runner reports the end."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc

    assert rollout.handle_stop_episode()["success"] is True
    assert runner.commands == [CMD_STOP]
    assert session.stop_requested is True
    # Still running until the runner says otherwise — nothing was scored yet.
    assert session.results == []

    rollout._on_episode_ended(REASON_STOPPED)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success"]
    assert status["phase"] == rollout.PHASE_RESETTING
    # The flag is one-shot: the next episode starts unstopped.
    assert session.stop_requested is False


def test_episode_end_reason_stopped_scores_a_success_without_the_flag(monkeypatch) -> None:
    """The reason IS the STOP we sent, so it stands on its own — a lost flag
    can't turn the user's success into a timeout."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    rollout._on_episode_ended(REASON_STOPPED)
    assert rollout.handle_inference_status()["episode_results"] == ["success"]


def test_episode_end_with_no_episode_in_flight_is_ignored(monkeypatch) -> None:
    """A duplicate/late end line must not append a phantom verdict."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    rollout._on_episode_ended(REASON_DURATION)
    assert session.results == []


def test_crashed_runner_parks_the_episode_with_the_error_visible(monkeypatch) -> None:
    """A crash is neither success nor failure: the in-flight episode is scored
    `error` and the session parks in the reset phase with the error on show, so
    the user can continue (paying one reload) or abort."""
    from makermodslab import rollout

    monkeypatch.setattr(
        rollout,
        "_extract_error_from_log",
        lambda p: "DeviceNotConnectedError: could not connect to the follower bus",
    )
    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    rollout._handle_runner_exit(runner, 1)

    status = rollout.handle_inference_status()
    assert status["episode_results"] == ["error"]
    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert "DeviceNotConnectedError" in status["error"]
    # The existing error taxonomy still applies to an episode-level crash.
    assert status["hint"]
    # The dead runner is dropped, which is what makes the continue respawn.
    assert rollout._inference_proc is None


def test_crashed_runner_prefers_its_own_error_line(monkeypatch) -> None:
    """The runner's ERROR event is the exception itself; log mining is a
    heuristic over a traceback, so the event wins when both exist."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "mined from the log tail")
    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc

    rollout._on_runner_error("RuntimeError: the gripper stalled")
    rollout._handle_runner_exit(runner, 1)

    assert rollout.handle_inference_status()["error"] == "RuntimeError: the gripper stalled"


def test_runner_death_during_a_reset_keeps_the_session_recoverable(monkeypatch) -> None:
    """Nothing to score — but the user has to learn that continuing now costs a
    reload, and the tally must survive."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "OSError: the port vanished")
    session = _arm_eval_session(monkeypatch, rollout, episodes=4, running=False, proc=_FakeRunner())
    session.results.extend(["success", "failure"])
    runner = rollout._inference_proc

    rollout._handle_runner_exit(runner, 1)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success", "failure"]  # no phantom verdict
    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert "OSError" in status["error"]
    assert rollout._inference_proc is None


def test_expected_runner_exit_after_quit_is_not_scored(monkeypatch) -> None:
    """An abort asks the runner to quit; that exit must not be read as a crash
    and score the episode the abort deliberately left unscored."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    session.quitting = True

    rollout._handle_runner_exit(runner, 0)
    assert session.results == []


def test_last_episode_finishes_the_session_with_accuracy(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    session.results.extend(["success", "success"])
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["phase"] == rollout.PHASE_FINISHED
    assert status["inference_active"] is False
    assert status["exited"] is True
    assert status["episode_results"] == ["success", "success", "failure"]
    assert status["episodes_total"] == 3
    assert status["episode_index"] == 3
    assert status["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
    # The slot is released for the next session.
    assert rollout.inference_active is False
    assert rollout._eval_session is None
    # And the runner is sent home rather than left holding the bus and cameras.
    assert runner.commands == [CMD_QUIT]


def test_finished_eval_payload_is_idempotent_across_polls(monkeypatch) -> None:
    """Several surfaces poll concurrently; the summary must survive every one."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=2)
    session.results.append("success")
    rollout._on_episode_ended(REASON_DURATION)
    first = rollout.handle_inference_status()
    second = rollout.handle_inference_status()
    third = rollout.handle_inference_status()

    assert first["accuracy"] == second["accuracy"] == third["accuracy"] == 0.5
    assert second["episode_results"] == third["episode_results"] == ["success", "failure"]
    assert third["phase"] == rollout.PHASE_FINISHED


def test_accuracy_excludes_errors_in_the_finished_payload(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    session.results.extend(["success", "error"])
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success", "error", "failure"]
    # 1 success out of 2 scored episodes, not out of 3.
    assert status["accuracy"] == 0.5


def test_stop_episode_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_stop_episode

    result = handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_refuses_outside_eval_mode(monkeypatch) -> None:
    """A single-episode run has no tally to record a success into."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _ExitedProc(0))
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_refuses_while_parked_in_a_reset(monkeypatch) -> None:
    """The runner is still ALIVE between episodes, so a live process is no
    longer proof an episode is running — `episode_running` is."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_reports_a_dead_runner_instead_of_claiming_success(monkeypatch) -> None:
    """A success can't be recorded against a runner that isn't there to end the
    episode — the crash path will score it an error instead."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3, proc=_ExitedProc(1))
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_next_episode

    result = handle_next_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_refuses_while_an_episode_is_still_running(monkeypatch) -> None:
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    # Phase is `running`, not `resetting` — there is nothing to continue from.
    result = rollout.handle_next_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_on_a_live_runner_costs_one_command(monkeypatch) -> None:
    """The headline of the redesign: continuing does NOT spawn anything. The
    policy is still resident and the bus and cameras are still open, so the
    whole cost is one line on the runner's stdin."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.results.append("failure")
    session.error = "old crash"
    session.hint = "old hint"

    def _explode(*a, **k):
        raise AssertionError("continuing must not spawn a process")

    monkeypatch.setattr(rollout, "_launch_eval_runner", _explode)
    monkeypatch.setattr(rollout, "_launch_rollout_subprocess", _explode)

    result = rollout.handle_next_episode()
    assert result["success"] is True
    assert runner.commands == [CMD_EPISODE]
    assert rollout._inference_proc is runner
    # The previous episode's crash banner is cleared on continue.
    assert session.error is None
    assert session.hint is None
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING
    # Both timers restart so the dialog clocks the EPISODE, not the session.
    assert rollout._inference_rollout_started_at is None


def test_next_episode_respawns_a_dead_runner_and_carries_the_tally(monkeypatch) -> None:
    """Crash containment: a runner that died costs ONE reload, not the session.
    The resolved policy path and preflighted `--robot.*` args are reused
    verbatim — no re-download, no second arm-identity pass — and the tally so
    far is carried forward."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=None)
    session.results.append("error")
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    launched = {}

    def _fake_launch(request, policy_path, robot_args):
        launched["policy_path"] = policy_path
        launched["robot_args"] = robot_args
        proc = _FakeRunner()
        proc.stdout = _EmptyStdout()
        return proc, io.StringIO(), __import__("pathlib").Path("/tmp/ep2.log")

    monkeypatch.setattr(rollout, "_launch_eval_runner", _fake_launch)
    # The pump would otherwise reap the fake proc and fire crash containment.
    monkeypatch.setattr(rollout, "_handle_runner_exit", lambda proc, rc: None)

    result = rollout.handle_next_episode()
    assert result["success"] is True
    assert launched["policy_path"] == "/tmp/policy"
    assert launched["robot_args"] == ["--robot.type=so101_follower"]
    assert session.results == ["error"]
    # The episode is PENDING, not started: the respawned runner has to finish
    # loading first, and its READY is what issues the command.
    assert session.episode_pending is True
    assert rollout._inference_meta["log_path"] == "/tmp/ep2.log"


def test_runner_ready_issues_the_pending_episode(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.episode_pending = True

    rollout._on_runner_ready()
    assert runner.commands == [CMD_EPISODE]
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_runner_ready_stays_idle_when_no_episode_is_pending(monkeypatch) -> None:
    """A READY from a respawn the user hasn't continued from — or one that lands
    after an abort — must not put the arm in motion."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.episode_pending = False
    rollout._on_runner_ready()
    assert runner.commands == []

    session.episode_pending = True
    session.quitting = True
    rollout._on_runner_ready()
    assert runner.commands == []


def test_episode_started_flips_the_phase_to_running(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    session.episode_pending = True

    rollout._on_episode_started()
    assert session.episode_running is True
    assert session.episode_pending is False
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING
    assert rollout._inference_rollout_started_at is not None


def test_stop_inference_quits_the_runner_instead_of_signalling_it(monkeypatch) -> None:
    """Abort mid-episode: the runner is asked to wind down so the follower still
    eases home, and the in-flight episode stays deliberately unscored."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=5)
    session.results.extend(["success", "failure"])
    runner = rollout._inference_proc
    quit_calls = []
    monkeypatch.setattr(rollout, "_quit_runner", lambda proc, **kw: quit_calls.append((proc, kw)))

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert [c[0] for c in quit_calls] == [runner]
    assert session.quitting is True

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ABORTED
    assert status["episode_results"] == ["success", "failure"]  # the cut episode isn't scored
    assert status["accuracy"] is None


def test_stop_inference_aborts_an_eval_parked_in_a_reset(monkeypatch) -> None:
    """Abort reports the partial tally and deliberately claims NO accuracy."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=5, running=False, proc=None)
    session.results.extend(["success", "failure"])

    result = rollout.handle_stop_inference()
    assert result["success"] is True

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ABORTED
    assert status["inference_active"] is False
    assert status["episode_results"] == ["success", "failure"]
    assert status["episodes_total"] == 5
    assert status["accuracy"] is None
    assert rollout._eval_session is None


def test_stop_inference_still_ends_a_single_run_the_old_way(monkeypatch) -> None:
    """No eval session -> the historical idle-with-no-payload teardown."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_DOWNLOADING_MODEL})

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert rollout.inference_active is False
    assert rollout._last_result is None


# ---------------------------------------------------------------------------
# Eval-runner line protocol + argv (makermodslab/eval_protocol.py)
#
# Pure string handling, no process anywhere near it. The runner itself is NEVER
# executed by the suite — it drives real servos.
# ---------------------------------------------------------------------------


def test_protocol_round_trips_an_event_and_its_payload() -> None:
    from makermodslab.eval_protocol import format_event, parse_event

    line = format_event("EPISODE_ENDED", "reason=duration")
    assert parse_event(line) == ("EPISODE_ENDED", "reason=duration")
    assert parse_event(format_event("READY")) == ("READY", "")


def test_protocol_collapses_a_multiline_payload_onto_one_line() -> None:
    """A traceback in an ERROR payload must not split one event across several
    lines — the reader is line-oriented and would read the tail as junk."""
    from makermodslab.eval_protocol import format_event, parse_event

    line = format_event("ERROR", "RuntimeError: boom\n  File 'x.py', line 3\n")
    assert "\n" not in line
    event, payload = parse_event(line)
    assert event == "ERROR"
    assert payload.startswith("RuntimeError: boom")


def test_protocol_ignores_ordinary_log_lines() -> None:
    from makermodslab.eval_protocol import parse_event

    assert parse_event("INFO 2026-07-31 Connecting robot (so101_follower)...\n") is None
    assert parse_event("") is None


def test_protocol_finds_an_event_appended_to_a_log_line() -> None:
    """The runner's logging handler shares the pipe; a record flushed without
    its newline must not swallow the event behind it."""
    from makermodslab.eval_protocol import parse_event

    assert parse_event("INFO some log MAKERMODSLAB-EVAL EPISODE_STARTED") == ("EPISODE_STARTED", "")


def test_protocol_reason_parsing_is_conservative() -> None:
    """An unrecognised (or renamed) reason yields "" rather than a guess: the
    orchestrator must not score an episode off a reason it doesn't understand."""
    from makermodslab.eval_protocol import parse_episode_end_reason

    assert parse_episode_end_reason("reason=stopped") == "stopped"
    assert parse_episode_end_reason("elapsed=30 reason=duration") == "duration"
    assert parse_episode_end_reason("") == ""
    assert parse_episode_end_reason("something-else") == ""


def test_unknown_episode_end_reason_scores_a_failure_not_a_success(monkeypatch) -> None:
    """Degrade the way the pre-runner code did — an episode that ended without
    the user calling it a success is a failure — rather than inventing a verdict."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    rollout._on_episode_ended("")
    assert rollout.handle_inference_status()["episode_results"] == ["failure"]


def test_eval_runner_and_rollout_argv_share_every_flag() -> None:
    """One flag list, two entry points. A flag added for the single-episode path
    must never be missing from the eval runner's."""
    from makermodslab.rollout import _build_eval_runner_cmd, _build_rollout_cmd

    request = _eval_request(4)
    args = ("/local/pretrained_model", ["--robot.type=so101_follower", "--robot.port=/dev/ttyUSB0"])
    rollout_cmd = _build_rollout_cmd(request, *args)
    runner_cmd = _build_eval_runner_cmd(request, *args)

    assert rollout_cmd[1:3] == ["-m", "lerobot.scripts.lerobot_rollout"]
    assert runner_cmd[1:3] == ["-m", "makermodslab.eval_runner"]
    assert rollout_cmd[3:] == runner_cmd[3:]
    assert "--return_to_initial_position=true" in runner_cmd
    assert "--strategy.type=base" in runner_cmd


def test_eval_start_spawns_the_runner_with_stdin_left_open(monkeypatch, tmp_path) -> None:
    """Eval mode gets ONE long-lived runner whose stdin is the command channel;
    the single-episode path still gets `lerobot-rollout` with stdin closed."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name, arm_type="so101": name)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: "/local/model")
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_pump_runner_stdout", lambda proc, log: None)

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

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

    assert rollout.handle_start_inference(_eval_request(6))["success"] is True
    assert captured["cmd"][1:3] == ["-m", "makermodslab.eval_runner"]
    # The command channel has to stay open — closing it is what a one-shot
    # rollout does, and it would make every later EPISODE unsendable.
    assert captured["stdin"].closed is False
    assert captured["stdin"].written == b"\n"
    # Episode 1 is pending: it is issued when the runner reports READY, after
    # the one-time policy load.
    assert rollout._eval_session.episode_pending is True
    assert rollout._eval_session.episode_running is False


def test_single_episode_start_still_spawns_lerobot_rollout(monkeypatch, tmp_path) -> None:
    """`eval_episodes == 1` is untouched by the redesign: same module, and stdin
    closed straight after the calibration seed."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name, arm_type="so101": name)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: "/local/model")
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

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

    assert rollout.handle_start_inference(_eval_request(1))["success"] is True
    assert captured["cmd"][1:3] == ["-m", "lerobot.scripts.lerobot_rollout"]
    assert captured["stdin"].closed is True
    assert rollout._eval_session is None


def test_runner_death_before_the_first_episode_fails_the_session(monkeypatch) -> None:
    """A bad policy path / missing camera / busy bus kills the runner before any
    episode starts. That is a startup failure, not an evaluation with one bad
    episode — parking in a reset would offer a continue that can only fail the
    same way."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "FileNotFoundError: no such checkpoint")
    session = _arm_eval_session(monkeypatch, rollout, episodes=4, running=False, proc=_FakeRunner())
    session.episode_pending = True
    runner = rollout._inference_proc

    rollout._handle_runner_exit(runner, 1)

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ERROR
    assert status["outcome"] == "failed"
    assert status["inference_active"] is False
    assert "FileNotFoundError" in status["error"]
    assert rollout._eval_session is None
    # Idempotent, like every other terminal payload.
    assert rollout.handle_inference_status()["phase"] == rollout.PHASE_ERROR


# ---------------------------------------------------------------------------
# /inference-log identity: the endpoint may only ever serve a log this PROCESS
# opened, and must say whose it is.
#
# The old resolver fell back to "newest *.log in inference_logs" whenever the
# active meta had no path — true during a new session's pre-spawn phases, and
# after a run that failed before spawning. Both windows served an earlier run's
# log unlabelled. Live incident: a run that failed in _prepare_robot on a
# calibration error produced no log at all, and the user was shown a three-day-old
# RTC run's output, concluding their sync run was executing RTC code.
# ---------------------------------------------------------------------------


def _seed_log_dir(monkeypatch, tmp_path, *, stale_text: str = "STALE RTC RUN") -> Path:
    """An inference_logs dir holding an old log from some previous run."""
    log_dir = tmp_path / "inference_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stale = log_dir / "1000.log"
    stale.write_text(stale_text + "\n")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return stale


def test_inference_log_ignores_stale_files_during_startup(monkeypatch, tmp_path) -> None:
    """THE INCIDENT: a session in its pre-spawn phases has no log of its own.

    `log_path` is only committed once the subprocess is launched, so during the
    download/preflight window the endpoint must report nothing rather than the
    newest file on disk.
    """
    from makermodslab import rollout

    stale = _seed_log_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_DOWNLOADING_MODEL})
    monkeypatch.setattr(rollout, "_last_log_path", None)

    result = rollout.handle_inference_log()

    assert result["belongs_to"] is None
    assert result["logs"] == "" and result["log_path"] is None
    assert stale.is_file(), "the stale file is still there — it is simply not ours to show"


def test_inference_log_after_a_startup_failure_is_empty(monkeypatch, tmp_path) -> None:
    """A run that failed BEFORE spawning never opened a log.

    `_fail_startup` wipes the meta, so nothing points anywhere — and the previous
    run's file must not fill the gap. This is the shape of the live incident: the
    user's calibration error produced no log, and they were shown someone else's.
    """
    from makermodslab import rollout

    _seed_log_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_last_log_path", None)

    result = rollout.handle_inference_log()

    assert result == {"logs": "", "log_path": None, "belongs_to": None}


def test_inference_log_serves_the_active_sessions_own_log(monkeypatch, tmp_path) -> None:
    from makermodslab import rollout

    _seed_log_dir(monkeypatch, tmp_path)
    mine = tmp_path / "inference_logs" / "2000.log"
    mine.write_text("MY RUN\n")
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"log_path": str(mine)})
    monkeypatch.setattr(rollout, "_last_log_path", str(mine))

    result = rollout.handle_inference_log()

    assert result["belongs_to"] == "active"
    assert "MY RUN" in result["logs"]


def test_inference_log_serves_a_finished_run_as_last_run(monkeypatch, tmp_path) -> None:
    """The legitimate case the glob fallback existed for, now explicit.

    A finished run's log stays readable — `_go_idle_locked` clears the meta but
    not `_last_log_path` — and is labelled so the UI can say it is not live.
    """
    from makermodslab import rollout

    _seed_log_dir(monkeypatch, tmp_path)
    mine = tmp_path / "inference_logs" / "2000.log"
    mine.write_text("MY FINISHED RUN\n")
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_last_log_path", str(mine))

    result = rollout.handle_inference_log()

    assert result["belongs_to"] == "last_run"
    assert "MY FINISHED RUN" in result["logs"]
    assert result["log_path"] == str(mine)


def test_inference_log_never_globs_the_directory(monkeypatch, tmp_path) -> None:
    """The fallback is gone, not merely deprioritised.

    Pinned directly: with a populated log dir and no session state at all, the
    answer is None. If someone reintroduces a "be helpful, show the newest file"
    shortcut, this fails.
    """
    from makermodslab import rollout

    log_dir = tmp_path / "inference_logs"
    log_dir.mkdir(parents=True)
    for name in ("1000.log", "2000.log", "3000.log"):
        (log_dir / name).write_text(f"run {name}\n")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_last_log_path", None)

    assert rollout._resolve_inference_log_path() == (None, None)


def test_inference_log_does_not_serve_a_previous_runs_log_to_a_new_session(monkeypatch, tmp_path) -> None:
    """A new claim clears the pointer, so run N+1 cannot inherit run N's log."""
    from makermodslab import rollout

    previous = tmp_path / "inference_logs" / "2000.log"
    previous.parent.mkdir(parents=True)
    previous.write_text("PREVIOUS RUN\n")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(rollout, "_last_log_path", str(previous))

    # What handle_start_inference does when it claims the slot.
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    monkeypatch.setattr(rollout, "_last_log_path", None)

    result = rollout.handle_inference_log()
    assert result["belongs_to"] is None and result["logs"] == ""


def test_start_inference_clears_the_previous_runs_log_pointer(monkeypatch, tmp_path) -> None:
    """The clear happens in the real claim path, not just in test setup.

    Driven through `handle_start_inference` so the lifecycle wiring itself is
    covered; the start is failed immediately after the claim (arm-count mismatch)
    so nothing spawns.
    """
    from makermodslab import rollout
    from makermodslab.rollout import InferenceRequest

    monkeypatch.setattr(rollout, "_last_log_path", "/tmp/some-previous-run.log")
    monkeypatch.setattr(rollout, "_arm_count_mismatch", lambda mode, dim, arm_type="so101": "nope")

    # Built inline rather than via a shared helper so this case stays
    # self-contained on this branch.
    request = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        checkpoint_state_dim=12,
    )
    result = rollout.handle_start_inference(request)

    assert result["success"] is False
    assert rollout._last_log_path is None, "a new claim must not inherit the previous run's log"


# ---------------------------------------------------------------------------
# F6 store entries must hold LOADABLE WEIGHTS, not just a config.
#
# Live incident: the user's sock ACT model had a 68 KB store entry — an
# interrupted local_dir download with config.json, train_config.json and both
# processor safetensors, but no model.safetensors. F6's short-circuit served it,
# and lerobot died with FileNotFoundError on the weights. Pre-F6 the hub path
# would have downloaded and worked, so the short-circuit converted a silent
# partial into a hard failure.
# ---------------------------------------------------------------------------


def test_local_store_refuses_a_weightless_checkpoint_ref(monkeypatch, tmp_path) -> None:
    """The incident, at the `@checkpoints/<step>` branch that hit it."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050", weights=False)
    _seed_hub_cache(monkeypatch, tmp_path)

    assert rollout._local_store_policy_path("user/repo", "000050") is None, (
        "a store entry with a config but no model.safetensors must fall through to the "
        "download instead of being served as a usable checkpoint"
    )


def test_local_store_refuses_a_weightless_root_ref(monkeypatch, tmp_path) -> None:
    """Same for the flat `@root` shape."""
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/flat", flat=True, weights=False)
    _seed_hub_cache(monkeypatch, tmp_path)

    assert rollout._local_store_policy_path("user/flat", None) is None


def test_weightless_store_entry_falls_through_to_the_hub(monkeypatch, tmp_path) -> None:
    """End-to-end: the partial entry must not break inference, just be skipped.

    This is the behaviour the incident should have had — resolve via the Hub, as
    it did before F6 existed.
    """
    from makermodslab import rollout

    _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050", weights=False)
    _seed_hub_cache(monkeypatch, tmp_path)
    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: str(fake_root))

    result = rollout._resolve_policy_path("user/repo@checkpoints/000050")

    assert result == str(fake_root / "checkpoints" / "000050" / "pretrained_model")


def test_processor_safetensors_alone_do_not_count_as_weights(monkeypatch, tmp_path) -> None:
    """The trap, pinned explicitly.

    The broken tree DOES contain .safetensors files (pre/post-processor), so a
    check for "any safetensors present" would have served it just the same. The
    requirement is the POLICY weights specifically.
    """
    from makermodslab import models

    _seed_models_store(monkeypatch, tmp_path, "user/repo", flat=True, weights=False)
    repo_dir = tmp_path / "makermodslab_models" / "user" / "repo"

    assert list(repo_dir.glob("*.safetensors")), "fixture precondition: safetensors ARE present"
    assert models._has_loadable_weights(repo_dir) is False
    assert models._resolve_pretrained_dir(repo_dir) is None


def test_complete_store_entry_is_still_served(monkeypatch, tmp_path) -> None:
    """Control: the fix must not stop F6 serving a COMPLETE local copy."""
    from makermodslab import rollout

    repo_dir = _seed_models_store(monkeypatch, tmp_path, "user/repo", step="000050")
    _seed_hub_cache(monkeypatch, tmp_path)
    _explode_snapshot_download(monkeypatch)

    assert rollout._local_store_policy_path("user/repo", "000050") == str(
        repo_dir / "checkpoints" / "000050" / "pretrained_model"
    )


def test_adapter_shaped_checkpoint_counts_as_loadable(tmp_path) -> None:
    """A PEFT/LoRA adapter dir has no model.safetensors and is still loadable.

    `make_policy`'s use_peft branch reads the adapter config and calls
    `PeftModel.from_pretrained(policy, dir)`, pulling the BASE weights from the
    adapter config's `base_model_name_or_path` — so requiring model.safetensors
    unconditionally would make the user's pi05 LoRA repos vanish.
    """
    from makermodslab import models

    d = tmp_path / "adapter"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "adapter_config.json").write_text('{"base_model_name_or_path": "lerobot/pi05"}')
    (d / "adapter_model.safetensors").write_text("lora")

    assert models._has_loadable_weights(d) is True
    assert models._resolve_pretrained_dir(d) == d


def test_partial_adapter_dir_is_not_loadable(tmp_path) -> None:
    """Half an adapter is not an adapter: PeftConfig.from_pretrained needs the
    config, and there is nothing to load without the adapter weights."""
    from makermodslab import models

    d = tmp_path / "half_adapter"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_text("lora")  # no adapter_config.json

    assert models._has_loadable_weights(d) is False


def test_sharded_weights_are_not_accepted(tmp_path) -> None:
    """The pinned lerobot cannot load sharded weights from a local dir.

    `from_pretrained` joins SAFETENSORS_SINGLE_FILE and opens it — there is no
    index-file branch — and the save side pins max_shard_size above the total so
    output stays one file. Accepting shards would recreate this very bug: a tree
    we call usable that lerobot then fails to open.
    """
    from makermodslab import models

    d = tmp_path / "sharded"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_text("shard")
    (d / "model-00002-of-00002.safetensors").write_text("shard")
    (d / "model.safetensors.index.json").write_text("{}")

    assert models._has_loadable_weights(d) is False


# ---------------------------------------------------------------------------
# Coaching (DAgger) mode
#
# The runner's control loop is a subprocess and stays untested per the module
# docstring above. What is covered here is everything that decides WHETHER and
# HOW that subprocess is launched: the request schema, the refusals, and the
# argv — the places a mistake reaches the arm.
# ---------------------------------------------------------------------------


def _coaching_request(**overrides):
    """A minimally valid coaching request, single-arm."""
    from makermodslab.rollout import InferenceRequest

    fields = {
        "follower_port": "/dev/ttyUSB0",
        "follower_config": "robot_a",
        "policy_ref": "user/repo@checkpoints/000050",
        "task": "fold the shirt",
        "coaching": True,
        "leader_port": "/dev/ttyUSB1",
        "leader_config": "teleop_a",
        "coaching_dataset_name": "shirt_fixes",
        "target_corrections": 5,
    }
    fields.update(overrides)
    return InferenceRequest(**fields)


def test_inference_request_defaults_to_no_coaching() -> None:
    """Coaching must be opt-in: every existing caller omits the field and has to
    keep getting a plain rollout."""
    request = _stub_request()
    assert request.coaching is False
    assert request.leader_port == ""
    assert request.leader_config == ""
    assert request.right_leader_port == ""
    assert request.right_leader_config == ""
    assert request.coaching_dataset_name == ""
    assert request.target_corrections == 10


def test_clamp_coaching_corrections_bounds_and_defaults() -> None:
    from makermodslab.rollout import MAX_COACHING_CORRECTIONS, clamp_coaching_corrections

    assert clamp_coaching_corrections(5) == 5
    assert clamp_coaching_corrections(0) == 1
    assert clamp_coaching_corrections(-3) == 1
    assert clamp_coaching_corrections(10_000) == MAX_COACHING_CORRECTIONS
    assert clamp_coaching_corrections(None) == 10
    assert clamp_coaching_corrections("nonsense") == 10


# --- Dataset naming ---------------------------------------------------------


def test_coaching_dataset_repo_id_applies_the_rollout_prefix() -> None:
    """lerobot REFUSES a rollout dataset whose name lacks this prefix, so it is
    applied for the operator rather than demanded of them."""
    from makermodslab.rollout import _coaching_dataset_repo_id

    assert _coaching_dataset_repo_id(_coaching_request()) == "rollout_shirt_fixes"


def test_coaching_dataset_repo_id_does_not_double_the_prefix() -> None:
    from makermodslab.rollout import _coaching_dataset_repo_id

    request = _coaching_request(coaching_dataset_name="rollout_shirt_fixes")
    assert _coaching_dataset_repo_id(request) == "rollout_shirt_fixes"


def test_coaching_dataset_repo_id_falls_back_for_a_blank_name() -> None:
    from makermodslab.rollout import _coaching_dataset_repo_id

    assert _coaching_dataset_repo_id(_coaching_request(coaching_dataset_name="")) == ("rollout_corrections")


def test_coaching_dataset_repo_id_is_not_pre_stamped() -> None:
    """lerobot stamps its own timestamp inside the subprocess. Stamping here too
    would produce `rollout_x_20260818_120000_20260818_120001`, and the app would
    then be looking for a directory that doesn't exist."""
    import re

    from makermodslab.rollout import _coaching_dataset_repo_id

    assert not re.search(r"\d{8}_\d{6}", _coaching_dataset_repo_id(_coaching_request()))


# --- CLI arguments ----------------------------------------------------------


def _coaching_args(request=None) -> list[str]:
    from makermodslab.rollout import _rollout_cli_args

    return _rollout_cli_args(request or _coaching_request(), "/tmp/policy", ["--robot.type=x"])


def test_coaching_args_select_the_dagger_strategy() -> None:
    args = _coaching_args()
    assert "--strategy.type=dagger" in args
    assert "--strategy.type=base" not in args


def test_coaching_args_pin_corrections_only_recording() -> None:
    """merge.py's "drop the intervention column" shortcut is only lossless
    because every recorded frame in this mode is a human correction. If this
    flag ever flips, that reasoning has to be revisited first."""
    assert "--strategy.record_autonomous=false" in _coaching_args()


def test_coaching_args_carry_the_clamped_correction_target() -> None:
    from makermodslab.rollout import MAX_COACHING_CORRECTIONS

    assert "--strategy.num_episodes=5" in _coaching_args()
    args = _coaching_args(_coaching_request(target_corrections=10_000))
    assert f"--strategy.num_episodes={MAX_COACHING_CORRECTIONS}" in args


def test_coaching_args_force_the_sync_engine_even_when_rtc_is_requested() -> None:
    """Defence in depth. The request layer already refuses rtc + coaching; this
    makes the argv incapable of expressing the unsafe combination even if that
    guard is ever bypassed. On the pinned lerobot, RTC resumes from the
    PRE-correction observation and snaps the arm back toward it (issue #3747)."""
    args = _coaching_args(_coaching_request(inference_engine="rtc"))
    assert "--inference.type=sync" in args
    assert "--inference.type=rtc" not in args


def test_coaching_args_run_without_a_duration() -> None:
    """A coaching session ends on its correction target or the Stop button, never
    on a clock — a timeout could fire mid-takeover, with a hand on the leader."""
    args = _coaching_args(_coaching_request(duration_s=60))
    assert "--duration=0" in args


def test_coaching_args_omit_the_dataset_root() -> None:
    """The one place this diverges from record.py, which pins its root. lerobot
    stamps the repo_id inside the subprocess and derives the root from the
    STAMPED name; a root computed out here would name a directory the dataset
    no longer lives in, and the library would never find it."""
    assert not any(a.startswith("--dataset.root") for a in _coaching_args())


def test_coaching_args_carry_the_dataset_and_task() -> None:
    args = _coaching_args()
    assert "--dataset.repo_id=rollout_shirt_fixes" in args
    assert "--dataset.single_task=fold the shirt" in args
    assert "--dataset.push_to_hub=false" in args


def test_coaching_args_keep_control_and_dataset_fps_in_step() -> None:
    """The dataset's timestamps are derived from the control loop's tick rate;
    the two disagreeing produces a dataset whose playback speed is wrong."""
    from makermodslab.rollout import _COACHING_FPS

    args = _coaching_args()
    assert f"--fps={_COACHING_FPS}" in args
    assert f"--dataset.fps={_COACHING_FPS}" in args


def test_coaching_args_still_pin_return_to_initial_position() -> None:
    assert "--return_to_initial_position=true" in _coaching_args()


def test_non_coaching_args_are_unchanged_by_the_coaching_branch() -> None:
    """The regression that matters most: every existing run shape must produce
    exactly the argv it did before coaching existed."""
    from makermodslab.rollout import _rollout_cli_args

    args = _rollout_cli_args(_stub_request(), "/tmp/policy", ["--robot.type=x"])
    assert "--strategy.type=base" in args
    assert "--duration=60" in args
    assert not any(a == "--strategy.type=dagger" for a in args)
    assert not any(a.startswith("--dataset.") for a in args)
    assert not any(a.startswith("--teleop.") for a in args)
    assert not any(a.startswith("--fps=") for a in args)


def test_dagger_runner_cmd_targets_the_coaching_entry_point() -> None:
    from makermodslab.rollout import _build_dagger_runner_cmd

    cmd = _build_dagger_runner_cmd(_coaching_request(), "/tmp/policy", [])
    assert cmd[1:3] == ["-m", "makermodslab.dagger_runner"]
    assert "--strategy.type=dagger" in cmd


# --- Teleop argv ------------------------------------------------------------


def test_teleop_args_single_arm_names_the_leader_port_and_calibration() -> None:
    from makermodslab.rollout import _teleop_args

    args = _teleop_args(_coaching_request(), "teleop_a", None)
    assert args == [
        "--teleop.type=so101_leader",
        "--teleop.port=/dev/ttyUSB1",
        "--teleop.id=teleop_a",
    ]


def test_teleop_args_bimanual_uses_the_biso_leader_and_staging_dir() -> None:
    from makermodslab.rollout import _teleop_args

    request = _coaching_request(
        mode="bimanual",
        right_leader_port="/dev/ttyUSB3",
        right_leader_config="teleop_b",
    )
    args = _teleop_args(request, "base_id", "/staging/leader")
    assert "--teleop.type=bi_so_leader" in args
    assert "--teleop.id=base_id" in args
    assert "--teleop.calibration_dir=/staging/leader" in args
    assert "--teleop.left_arm_config.port=/dev/ttyUSB1" in args
    assert "--teleop.right_arm_config.port=/dev/ttyUSB3" in args


# --- stdin seeding ----------------------------------------------------------


def test_stdin_seed_covers_both_sides_for_coaching() -> None:
    """One newline per ARM to pre-answer lerobot's calibration prompt. Coaching
    connects leaders as well as followers, so it needs twice as many as any
    other inference run — a short seed leaves connect() blocked on input()."""
    from makermodslab.rollout import _stdin_seed

    assert _stdin_seed(_stub_request()) == b"\n"
    assert _stdin_seed(_coaching_request()) == b"\n\n"
    assert _stdin_seed(_coaching_request(mode="bimanual")) == b"\n\n\n\n"


# --- Start-request refusals -------------------------------------------------


def test_coaching_start_refuses_rtc(monkeypatch) -> None:
    from makermodslab import rollout

    result = rollout.handle_start_inference(_coaching_request(inference_engine="rtc"))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "Real-Time Chunking" in result["message"]
    # The slot must be released, or the next launch 409s on a session that never
    # started.
    assert rollout.inference_active is False
    assert rollout._coach_session is None


def test_coaching_is_refused_on_a_can_arm(monkeypatch) -> None:
    """A Maker or Metal robot has an unmotorised Star Arm 102 leader — nothing to
    drive to the follower's pose between takeovers — so coaching is refused in
    the launch panel rather than failing once the arms are connected."""
    from makermodslab import rollout

    for arm_type in ("maker", "metal"):
        result = rollout.handle_start_inference(_coaching_request(arm_type=arm_type))
        assert result["success"] is False, arm_type
        assert result["status_code"] == 400
        assert "leader" in result["message"].lower()
        assert rollout.inference_active is False
        assert rollout._coach_session is None


def test_coaching_start_refuses_a_simultaneous_evaluation(monkeypatch) -> None:
    from makermodslab import rollout

    result = rollout.handle_start_inference(_coaching_request(eval_episodes=20))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "evaluation" in result["message"].lower()
    assert rollout.inference_active is False


def test_coaching_start_refuses_a_missing_leader(monkeypatch) -> None:
    """Without a leader there is nothing to take over WITH — the session would
    load a policy, connect the arm, and then fail deep inside lerobot."""
    from makermodslab import rollout

    result = rollout.handle_start_inference(_coaching_request(leader_port=""))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "leader" in result["message"].lower()
    assert rollout.inference_active is False


def test_bimanual_coaching_is_refused_before_the_per_arm_checks(monkeypatch) -> None:
    """This used to assert the missing-right-leader message. It cannot any more,
    and the reason is the point: bimanual coaching is refused outright, ahead of
    every per-arm check, so an operator is told the real answer instead of being
    sent to find a leader port that would not have helped.

    Restore the per-arm assertion when bimanual coaching is supported again —
    see `_bimanual_robot_args`' docstring for what that needs."""
    from makermodslab import rollout

    request = _coaching_request(mode="bimanual", right_follower_port="/dev/ttyUSB2")
    result = rollout.handle_start_inference(request)
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "bimanual" in result["message"].lower()
    assert "right leader" not in result["message"].lower()


def test_coaching_start_refuses_a_blank_task(monkeypatch) -> None:
    """The task is written into every recorded frame and is what a
    language-conditioned policy is fine-tuned against; a blank one silently
    produces a dataset that can't be used with SmolVLA or pi0."""
    from makermodslab import rollout

    result = rollout.handle_start_inference(_coaching_request(task="   "))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert rollout.inference_active is False


def test_coaching_start_refuses_an_invalid_dataset_name(monkeypatch) -> None:
    from makermodslab import rollout

    result = rollout.handle_start_inference(_coaching_request(coaching_dataset_name="a/b/c"))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert rollout.inference_active is False


def test_coaching_start_is_blocked_while_recording(monkeypatch) -> None:
    """Coaching drives the same bus as everything else, so it inherits the whole
    mutual-exclusion table — it holds `inference_active`, no new global."""
    from makermodslab import rollout

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = rollout.handle_start_inference(_coaching_request())
    assert result["success"] is False
    assert result["status_code"] == 409


# --- Coaching commands ------------------------------------------------------


def test_coaching_command_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_coaching_command

    result = handle_coaching_command("TAKEOVER")
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "No coaching session" in result["message"]


def test_coaching_command_rejects_an_unknown_verb() -> None:
    from makermodslab.rollout import handle_coaching_command

    result = handle_coaching_command("LAUNCH_MISSILES")
    assert result["success"] is False
    assert result["status_code"] == 400


def test_coaching_command_rejects_quit() -> None:
    """Ending the session is /stop-inference, which also releases the slot and
    writes the terminal payload. A QUIT sent here would leave the orchestrator
    believing a dead session is still live."""
    from makermodslab.rollout import handle_coaching_command

    result = handle_coaching_command("QUIT")
    assert result["success"] is False
    assert result["status_code"] == 400


def test_coaching_command_refuses_before_the_subprocess_exists(monkeypatch) -> None:
    """During the model download / arm preflight there is nothing to command.
    Saying so beats a silent no-op the operator reads as a dead button."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_coach_session",
        rollout._CoachSession(request=_coaching_request(), corrections_target=5),
    )
    result = rollout.handle_coaching_command("TAKEOVER")
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "starting up" in result["message"]


def test_coaching_command_is_written_to_the_runner(monkeypatch) -> None:
    from makermodslab import rollout

    written: list[str] = []
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_coach_session",
        rollout._CoachSession(request=_coaching_request(), corrections_target=5),
    )
    monkeypatch.setattr(rollout, "_inference_proc", object())
    monkeypatch.setattr(rollout, "_send_runner_command", lambda proc, cmd: (written.append(cmd), True)[1])

    assert rollout.handle_coaching_command("takeover")["success"] is True
    assert written == ["TAKEOVER"]


# --- Status payload ---------------------------------------------------------


def test_coach_fields_are_present_and_null_for_a_non_coaching_run() -> None:
    """The payload shape is stable so the frontend branches on `coaching` alone,
    exactly as it does on `eval_mode`."""
    from makermodslab.rollout import handle_inference_status

    result = handle_inference_status()
    assert result["coaching"] is False
    for key in (
        "coaching_phase",
        "corrections_saved",
        "corrections_target",
        "correction_seconds",
        "coaching_dataset",
        "align_error",
    ):
        assert result[key] is None


def test_coach_session_starts_with_no_phase_at_all() -> None:
    """Before the first PHASE event lands, the session must make NO claim about
    who holds the arm.

    It used to default to `paused`, on the reasoning that claiming the policy
    was driving would be worse. But `paused` renders as "the arm is frozen",
    which is equally false and shown at exactly the moment the policy starts
    driving. None renders as "Starting…" — the only honest answer in a window
    where neither state is known to be true."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    assert session.phase is None
    assert session.corrections_saved == 0


def test_handing_over_maps_to_its_own_app_phase(monkeypatch) -> None:
    """The runner announces travel before the ~2s blocking handover. It must
    reach the UI as its own phase and not collapse into watching/holding —
    collapsing is what made the banner claim the arm was frozen while both
    followers swept across the workspace."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_WATCHING})
    rollout._on_dagger_phase("handing_over")
    assert session.phase == "handing_over"
    assert rollout._inference_meta["phase"] == rollout.PHASE_HANDING_OVER


def test_on_dagger_phase_ignores_an_unrecognised_value(monkeypatch) -> None:
    """The phase drives a banner telling the operator who holds the arm. A stale
    value they can still act on beats an unknown one they can't."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.phase = "autonomous"
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_dagger_phase("banana")
    assert session.phase == "autonomous"


def test_on_dagger_phase_clears_a_stale_alignment_refusal(monkeypatch) -> None:
    """The refusal describes the LAST takeover attempt; once the phase actually
    moves it is history and must stop being shown."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.align_error = "too far"
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_dagger_phase("correcting")
    assert session.phase == "correcting"
    assert session.align_error is None


def test_on_correction_saved_trusts_the_runner_count(monkeypatch) -> None:
    """The runner is the side that knows whether an episode was written; a
    dropped event would otherwise leave the two counts permanently out of step."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "3", "frames": "90", "seconds": "4.5"})
    assert session.corrections_saved == 3
    assert session.correction_seconds == pytest.approx(4.5)


def test_on_dagger_dataset_records_the_stamped_name(monkeypatch, tmp_path) -> None:
    """The only place the app learns the real dataset name — it cannot be
    derived from what the operator typed."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_dagger_dataset({"repo_id": "rollout_shirt_fixes_20260818_120000", "root": str(tmp_path)})
    assert session.dataset_repo_id == "rollout_shirt_fixes_20260818_120000"
    assert session.dataset_root == str(tmp_path)


@pytest.mark.parametrize("raw", ["None", "none", "null", "", "/definitely/not/here"])
def test_an_unusable_dataset_root_is_dropped_rather_than_believed(monkeypatch, raw) -> None:
    """THE bug this guards. Coaching passes no `--dataset.root`, and lerobot
    never writes the resolved path back to the config — so the runner used to
    put the literal string "None" on the wire. It is non-empty and therefore
    truthy, so it was stored, and `_atomic_write_text` CREATED a directory
    called "None" beside the server rather than failing. Every recovery
    boundary the operator marked went there instead of to the dataset."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_dagger_dataset({"repo_id": "rollout_x_20260819_120000", "root": raw})
    assert session.dataset_root is None


def test_coaching_terminal_payload_keeps_the_dataset_and_tally(monkeypatch) -> None:
    """The coaching block must OUTLIVE the session: the follow-up actions (merge,
    fine-tune) need the dataset name, and it is gone from the live state the
    moment the session ends."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.corrections_saved = 4
    session.dataset_repo_id = "rollout_shirt_fixes_20260818_120000"
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {"policy_ref": "user/repo@root"})

    with rollout._state_lock:
        rollout._finalise_coaching_locked(0, session)

    assert rollout.inference_active is False
    assert rollout._coach_session is None
    result = rollout._last_result
    assert result["phase"] == "finished"
    assert result["coaching"] is True
    assert result["corrections_saved"] == 4
    assert result["coaching_dataset"] == "rollout_shirt_fixes_20260818_120000"


def test_coaching_finalise_invalidates_the_dataset_listing(monkeypatch) -> None:
    """A coaching session leaves a corrections dataset on disk (or removes an
    empty one). Either way the /datasets listing changed, so its cache must be
    dropped — nothing on the coaching teardown path did this, so the corrections
    dataset stayed invisible until the 45s TTL lapsed."""
    import time

    from makermodslab import datasets, rollout

    for aborted in (False, True):
        session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
        session.corrections_saved = 3
        session.dataset_repo_id = "rollout_shirt_fixes_20260818_120000"
        monkeypatch.setattr(rollout, "inference_active", True)
        monkeypatch.setattr(rollout, "_coach_session", session)
        monkeypatch.setattr(rollout, "_inference_meta", {})
        with datasets._listing_cache_lock:
            datasets._listing_cache = {"at": time.monotonic(), "value": [{"repo_id": "old/ds"}]}

        with rollout._state_lock:
            rollout._finalise_coaching_locked(0 if not aborted else None, session, aborted=aborted)

        with datasets._listing_cache_lock:
            assert datasets._listing_cache is None, f"aborted={aborted}"


def test_coaching_finalise_leaves_the_listing_cache_alone_when_no_dataset_was_created(
    monkeypatch,
) -> None:
    """A coaching session killed before lerobot wrote meta/info.json has no
    dataset_repo_id — nothing landed on disk, so the listing cache must be left
    intact rather than forcing a needless Hub re-fan-out."""
    import time

    from makermodslab import datasets, rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    # dataset_repo_id stays None — the runner never reported one.
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    sentinel = {"at": time.monotonic(), "value": []}
    with datasets._listing_cache_lock:
        datasets._listing_cache = sentinel

    with rollout._state_lock:
        rollout._finalise_coaching_locked(None, session, aborted=True)

    with datasets._listing_cache_lock:
        assert datasets._listing_cache is sentinel


def test_coaching_stop_reports_aborted_but_keeps_the_partial_tally(monkeypatch) -> None:
    """Unlike an aborted EVAL — which must not claim an accuracy it never
    measured — a stopped coaching session loses nothing by reporting its count.
    Every correction it saved is on disk and just as useful."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.corrections_saved = 2
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    with rollout._state_lock:
        rollout._finalise_coaching_locked(None, session, aborted=True)

    assert rollout._last_result["phase"] == "aborted"
    assert rollout._last_result["outcome"] == "ok"
    assert rollout._last_result["corrections_saved"] == 2


def test_coaching_runner_exit_after_a_stop_reports_aborted(monkeypatch) -> None:
    """A stopped session exits cleanly (rc 0) because we asked it to via QUIT.
    Without the `quitting` check the summary would read "Coaching complete" and
    congratulate the operator on finishing a run they cut short."""
    from makermodslab import rollout

    proc = object()
    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.corrections_saved = 2
    session.quitting = True
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_proc", proc)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    rollout._handle_dagger_exit(proc, 0)

    assert rollout._last_result["phase"] == "aborted"
    assert rollout._last_result["corrections_saved"] == 2


def test_coaching_runner_exit_at_target_reports_finished(monkeypatch) -> None:
    """The control: a session the runner ended on its own must NOT read as an
    abort just because the abort path exists."""
    from makermodslab import rollout

    proc = object()
    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.corrections_saved = 5
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_proc", proc)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    rollout._handle_dagger_exit(proc, 0)

    assert rollout._last_result["phase"] == "finished"


def test_coaching_runner_crash_reports_error_even_when_quitting(monkeypatch) -> None:
    """A non-zero exit is a failure whether or not a stop was in flight — the
    operator needs to know the dataset may be incomplete."""
    from makermodslab import rollout

    proc = object()
    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.quitting = True
    session.runner_error = "RuntimeError: bus went away"
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_proc", proc)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    rollout._handle_dagger_exit(proc, 1)

    assert rollout._last_result["phase"] == "error"
    assert rollout._last_result["error"] == "RuntimeError: bus went away"


def test_coaching_args_pin_streaming_encoding_but_not_a_hardware_codec() -> None:
    """Measured on the station, 132 frames x 2 cameras at 480x640:
    `save_episode` takes 2.32s at lerobot's default and 0.44s with streaming
    encoding on, for identical output. It runs synchronously on the control
    loop at the hand-back edge, so that gap is time the operator waits with the
    arm frozen.

    `rgb_encoder.vcodec=auto` is deliberately NOT set, even though record.py
    sets it. On this hardware "auto" resolves to h264_nvenc, which lerobot's
    `detect_available_encoders` claims is available but PyAV cannot open
    (`avcodec_open2(h264_nvenc)`, Errno 22) — pinning it breaks encoding
    outright. The software default encodes the same episode in 0.8s, which is
    not a bottleneck worth risking that on. This assertion is the guard against
    someone "optimising" it back."""
    args = _coaching_args()
    assert "--dataset.streaming_encoding=true" in args
    assert not any("vcodec" in a for a in args)


def test_saving_maps_to_its_own_app_phase(monkeypatch) -> None:
    """Without a phase of its own, the write window inherited `correcting` —
    so the banner told the operator they were still driving and recording for
    the whole (potentially minute-long) save."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_CORRECTING})
    rollout._on_dagger_phase("saving")
    assert session.phase == "saving"
    assert rollout._inference_meta["phase"] == rollout.PHASE_SAVING


def test_signal_group_refuses_to_signal_our_own_process_group() -> None:
    """The guard that stops a teardown killing the FastAPI server.

    `start_new_session=True` puts the runner in its own group, so this should
    never fire — but if it ever did, `killpg` would take the whole app down
    with the runner. A wedged camera is a far better outcome than a dead
    server, so the group route bails and the caller signals the process alone."""
    import os

    from makermodslab import rollout

    class _OurselvesProc:
        pid = os.getpid()

    assert rollout._signal_group(_OurselvesProc(), 15) is False


def test_signal_group_declines_a_process_without_a_pid() -> None:
    """Degrades instead of raising, so `_terminate_tree` still falls back to
    terminate()/kill() on anything that isn't a real Popen."""
    from makermodslab import rollout

    assert rollout._signal_group(object(), 15) is False


# --- The happy path, against real processes ----------------------------------
#
# Everything above pins a REFUSAL. The behaviour those refusals exist to protect
# — one killpg reaping the runner AND the children it forked — was only ever
# exercised on the station, and only in coaching mode, even though all three
# session shapes (single run, eval, coaching) now route their force-kills
# through `_terminate_tree`. The bug it fixes is not subtle but it is invisible
# from the parent: `Popen.kill()` returns cleanly while `LeRobotDataset`'s image
# writers keep running and keep /dev/video* open, so the NEXT session is the one
# that fails. That is worth a real fork to test.
#
# Deliberately real subprocesses rather than fakes: a fake cannot show that the
# GRANDCHILD died, which is the entire point.

_FORK_AND_WAIT = (
    # Spawn a grandchild that would outlive us, announce its pid, then idle.
    # `sys.stdout.flush` matters — the parent reads the pid before signalling.
    "import subprocess, sys, time; "
    "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']); "
    "print(kid.pid, flush=True); "
    "time.sleep(300)"
)


def _alive(pid: int) -> bool:
    """True while the pid exists and has not been reaped."""
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(not hasattr(__import__("os"), "getpgid"), reason="posix process groups only")
def test_terminate_tree_reaps_the_grandchild_the_runner_forked() -> None:
    """The orphaned-image-writer bug, reproduced and fixed in one test.

    Spawns a process that forks a child of its own, exactly as the runner does,
    then force-kills it the way a teardown does. Both must go."""
    import os
    import subprocess
    import sys
    import time

    from makermodslab import rollout

    proc = subprocess.Popen(
        [sys.executable, "-c", _FORK_AND_WAIT],
        stdout=subprocess.PIPE,
        text=True,
        # The same flag `_spawn_rollout_process` passes. Without it the two
        # processes share OUR group and `_signal_group` correctly refuses.
        start_new_session=True,
    )
    try:
        grandchild = int(proc.stdout.readline().strip())
        assert _alive(grandchild)
        # Precondition for the whole mechanism: the runner leads its own group.
        assert os.getpgid(proc.pid) != os.getpgid(0)

        rollout._terminate_tree(proc, timeout=5.0)

        assert proc.poll() is not None, "the runner itself survived _terminate_tree"
        # SIGKILL delivery is asynchronous; give the grandchild a moment to go.
        deadline = time.monotonic() + 5.0
        while _alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(grandchild), (
            "the forked grandchild survived _terminate_tree — this is the orphaned "
            "image-writer bug that wedged the cameras for the next session"
        )
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.mark.skipif(not hasattr(__import__("os"), "getpgid"), reason="posix process groups only")
def test_terminate_tree_returns_promptly_for_an_already_dead_process() -> None:
    """Teardown runs on paths where the runner exited on its own — the common
    case, in fact, since QUIT is tried before the force-kill. It must not spend
    the escalation timeout discovering that."""
    import subprocess
    import sys
    import time

    from makermodslab import rollout

    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait(timeout=10)

    started = time.monotonic()
    rollout._terminate_tree(proc, timeout=5.0)
    assert time.monotonic() - started < 2.0, "_terminate_tree waited on a corpse"


@pytest.mark.skipif(not hasattr(__import__("os"), "getpgid"), reason="posix process groups only")
def test_terminate_tree_escalates_to_sigkill_when_sigterm_is_ignored() -> None:
    """A runner wedged in a blocking serial read is the reason the SIGKILL leg
    exists — the >60s hand-back that started the whole encoding investigation
    left a process at 0% CPU that SIGTERM did not shift."""
    import signal
    import subprocess
    import sys

    from makermodslab import rollout

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(300)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        rollout._terminate_tree(proc, timeout=1.0)
        assert proc.poll() is not None, "a SIGTERM-ignoring runner survived _terminate_tree"
        assert proc.returncode == -signal.SIGKILL
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


def test_reset_saves_the_correction_in_flight_then_resets() -> None:
    """Finishing the task while still driving means those frames ARE the
    correction.

    RESET used to be refused mid-correction so it could not decide the fate of a
    part-recorded takeover. In practice the operator had already decided, and
    the refusal cost them two presses — hand back, then reset — with the policy
    briefly regaining a finished scene in between.

    It now takes the same edge as RECOVER, minus the discard: one correction
    event (which drives CORRECTING->PAUSED, where the episode is written because
    no cancel is armed) with the reset armed for the following tick."""
    from lerobot.rollout.configs import DAggerStrategyConfig
    from lerobot.rollout.strategies.dagger import DAggerPhase
    from makermodslab.dagger_protocol import CMD_RESET
    from makermodslab.dagger_runner import _EV_CORRECTION, WebDAggerStrategy

    s = WebDAggerStrategy(DAggerStrategyConfig(num_episodes=5))
    assert s._translate(CMD_RESET, DAggerPhase.CORRECTING) == [_EV_CORRECTION]
    assert s._reset_requested is True
    # The distinction from RECOVER: no cancel is armed, so the correction is
    # SAVED rather than binned.
    assert s._cancel_correction is False


def test_reset_requests_no_transition_from_the_non_correcting_phases() -> None:
    """From every phase that is not mid-correction it arms the flag and requests
    NO transition. Routing it through `pause_resume` made `_apply_transition`
    treat the reset as a handover: it drove the LEADER up to the follower's pose
    under torque, and released it again when the policy resumed, so the leader
    fell out of the air. A reset is not a handover — the loop pauses the engine
    itself, with none of the transition's side effects."""
    from lerobot.rollout.configs import DAggerStrategyConfig
    from lerobot.rollout.strategies.dagger import DAggerPhase
    from makermodslab.dagger_protocol import CMD_RESET
    from makermodslab.dagger_runner import WebDAggerStrategy

    s = WebDAggerStrategy(DAggerStrategyConfig(num_episodes=5))
    for phase in (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED):
        s._reset_requested = False
        assert s._translate(CMD_RESET, phase) == []
        assert s._reset_requested is True


def test_attempt_reset_event_updates_the_tally(monkeypatch) -> None:
    """The count comes from the runner, not a local increment, so a dropped
    event cannot leave the two permanently out of step."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_attempt_reset({"n": "3"})
    assert session.attempts == 3


def test_coaching_start_refuses_a_leader_calibration_that_no_longer_exists(monkeypatch, tmp_path) -> None:
    """A name being non-empty is not the same as the calibration existing.

    A record can point at a calibration that was deleted or renamed since, and
    a stem like "None" is perfectly legal so it cannot be treated as unset.
    Without this the failure surfaced deep inside the runner, after the model
    download and the arm preflight."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "LEADER_CONFIG_PATH", str(tmp_path))
    result = rollout.handle_start_inference(_coaching_request(leader_config="ghost"))
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "ghost" in result["message"]
    assert rollout.inference_active is False


def test_coaching_start_accepts_a_leader_calibration_that_does_exist(monkeypatch, tmp_path) -> None:
    """Control: an odd-but-real stem (the station's is literally "None") must
    pass. Treating that string as "unset" would break a working rig."""
    from makermodslab import rollout

    (tmp_path / "None.json").write_text("{}")
    monkeypatch.setattr(rollout, "LEADER_CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: False)
    result = rollout.handle_start_inference(_coaching_request(leader_config="None"))
    # Gets PAST the calibration gate and fails on the next check instead.
    assert "calibration" not in result["message"]


# --- Discards: who did it decides whether the operator hears about it --------


def test_operator_pressed_discard_stays_silent(monkeypatch) -> None:
    """They asked for it. The count not moving is the whole feedback, and a
    banner here would be nagging the person about their own button press."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_cancelled({"reason": "operator", "frames": "120", "seconds": "4.0"})
    assert session.align_error is None
    assert session.corrections_saved == 0


def test_too_short_discard_tells_the_operator_their_work_was_binned(monkeypatch) -> None:
    """The regression this exists for: a deliberate quick nudge used to vanish
    with nothing on screen. The operator did not press anything — the frame
    floor decided — so they can only discover it by counting episodes later."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_cancelled(
        {"reason": "too_short", "frames": "7", "seconds": "0.2", "minimum": "10"}
    )
    assert session.discard_notice is not None
    # Names the numbers, not just "discarded" — the operator has to be able to
    # work out how much longer to hold it.
    assert "7 frames" in session.discard_notice
    assert "10-frame" in session.discard_notice
    assert "longer" in session.discard_notice
    # And it must NOT live in align_error, which the very next phase event
    # clears — see test_the_discard_notice_survives_the_runners_own_event_order.
    assert session.align_error is None


def test_a_cancel_without_a_reason_is_treated_as_the_operator(monkeypatch) -> None:
    """Forwards-compatibility with a runner older than this field: silence is
    the safe default, since the noisy branch accuses the system of eating work
    the operator may well have discarded deliberately."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_cancelled({})
    assert session.align_error is None


def test_the_too_short_notice_clears_only_when_the_next_takeover_begins(monkeypatch) -> None:
    """It reads as "your last takeover", not as session history — but it must
    outlive the `paused` event the runner emits immediately after the discard,
    which is why it does not share `align_error`'s slot."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_cancelled({"reason": "too_short", "frames": "3", "minimum": "10"})
    assert session.discard_notice is not None
    rollout._on_dagger_phase("paused")
    assert session.discard_notice is not None, "wiped by the event that always follows it"
    rollout._on_dagger_phase("correcting")
    assert session.discard_notice is None


# --- The push channel --------------------------------------------------------
#
# The banner naming who holds the arm used to reach the operator on a 1 Hz poll,
# which meant up to half of the ~2s handover window could pass before the words
# "don't fight it" could possibly have been read. These pin that every state
# change pushes, and that the push can never be the thing that breaks a session.


def _capture_pushes(monkeypatch) -> list[dict]:
    from makermodslab import rollout

    pushes: list[dict] = []
    monkeypatch.setattr(rollout, "_on_coaching_state", pushes.append)
    return pushes


def test_a_phase_change_pushes_immediately(monkeypatch) -> None:
    """The whole point. A poll would deliver this up to a second later."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    pushes = _capture_pushes(monkeypatch)
    rollout._on_dagger_phase("handing_over")
    assert len(pushes) == 1
    assert pushes[0]["coaching_phase"] == "handing_over"


def test_every_coaching_handler_pushes(monkeypatch) -> None:
    """Not just the phase: a saved correction, a refused takeover and an attempt
    reset all change what the operator is being told, and all of them used to
    wait for the poll."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    pushes = _capture_pushes(monkeypatch)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "3.0"})
    rollout._on_correction_cancelled({"reason": "too_short", "frames": "4", "minimum": "10"})
    rollout._on_align_required({"max_delta": "40", "joints": "shoulder_pan:40"})
    rollout._on_attempt_reset({"n": "2"})
    assert len(pushes) == 4
    assert pushes[0]["corrections_saved"] == 1
    assert pushes[1]["discard_notice"] is not None
    assert pushes[3]["attempts"] == 2


def test_the_push_carries_the_same_shape_the_poll_does(monkeypatch) -> None:
    """One shape for the frontend to understand. If these diverged, the browser
    would have to merge two different objects and decide which wins."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    pushes = _capture_pushes(monkeypatch)
    rollout._on_dagger_phase("correcting")
    assert pushes[0].keys() == rollout._coach_fields(session).keys()


def test_a_failing_push_never_takes_the_session_down(monkeypatch) -> None:
    """It runs on the runner's stdout pump. A websocket that has gone away must
    not stop the pump reading the runner, or the session goes blind entirely —
    strictly worse than the lag this feature exists to remove."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)

    def _explode(_fields):
        raise RuntimeError("websocket is gone")

    monkeypatch.setattr(rollout, "_on_coaching_state", _explode)
    rollout._on_dagger_phase("correcting")  # must not raise
    assert session.phase == "correcting"


def test_no_push_wired_is_fine(monkeypatch) -> None:
    """The default. Tests and any embedding without a websocket still work; the
    poll is the reconciler, not an optimisation."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_on_coaching_state", None)
    rollout._on_dagger_phase("correcting")
    assert session.phase == "correcting"


# --- RaC: the recovery/correction boundary -----------------------------------
#
# An intervention is two things wearing one name — rewind to a state the policy
# has seen, then demonstrate what should follow. lerobot's own HIL guide names
# RaC (arXiv:2509.07953) as the protocol its DAgger strategy follows, and RaC's
# data-efficiency claim rests entirely on that decomposition; the strategy then
# records both halves as one undifferentiated `intervention=True`. We record the
# boundary out of band because lerobot's dataset feature dict has no hook.
#
# The distinction these pin over and over: UNMARKED is not ZERO. One says the
# operator never annotated it, the other says they went straight to correcting.


def test_a_marked_correction_records_both_halves(monkeypatch) -> None:
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved(
        {"n": "1", "frames": "120", "seconds": "4.0", "recovery": "40", "labelled": "true"}
    )
    assert session.rac_episodes[0] == {
        "recovery_frames": 40,
        "correction_frames": 80,
        "labelled": True,
        # First correction of the first scene.
        "attempt_index": 0,
        "index_in_attempt": 0,
    }


def test_corrections_are_grouped_by_scene(monkeypatch) -> None:
    """THE grouping. A scene routinely takes several corrections — take over,
    hand back, the policy fails at the same place, take over again with more
    help — and training is IID over shuffled frames, so nothing downstream can
    reconstruct which correction belonged to which attempt unless it is written
    down live. Without it, "keep only the correction that ended each scene" is
    not a filter anyone can express."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=9)
    monkeypatch.setattr(rollout, "_coach_session", session)

    # Scene 0 took three goes.
    for n in (1, 2, 3):
        rollout._on_correction_saved({"n": str(n), "frames": "90", "seconds": "3.0"})
    rollout._on_attempt_reset({"n": "1"})
    # Scene 1 took two.
    for n in (4, 5):
        rollout._on_correction_saved({"n": str(n), "frames": "90", "seconds": "3.0"})

    grouped = [(e["attempt_index"], e["index_in_attempt"]) for _, e in sorted(session.rac_episodes.items())]
    assert grouped == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    # The point of the pair: the last correction of each scene is selectable.
    last_per_scene = {e["attempt_index"]: i for i, e in sorted(session.rac_episodes.items())}
    assert last_per_scene == {0: 2, 1: 4}


def test_a_scene_reset_restarts_the_within_scene_count_even_if_its_number_is_junk(
    monkeypatch,
) -> None:
    """The attempt NUMBER is parsed defensively; the scene boundary is not
    conditional on it. A reset whose `n` fails to parse still ended the scene,
    and carrying the old within-scene position into the next one would mislabel
    every correction that follows it."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)

    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "3.0"})
    rollout._on_attempt_reset({"n": "not-a-number"})
    rollout._on_correction_saved({"n": "2", "frames": "90", "seconds": "3.0"})

    assert session.rac_episodes[1]["index_in_attempt"] == 0


def test_an_unmarked_correction_is_recorded_as_unlabelled_not_as_zero_recovery(monkeypatch) -> None:
    """THE distinction. A consumer that read `recovery_frames: 0` here would
    believe the operator asserted there was no recovery phase, when in fact they
    asserted nothing at all."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved(
        {"n": "1", "frames": "120", "seconds": "4.0", "recovery": "-1", "labelled": "false"}
    )
    assert session.rac_episodes[0]["labelled"] is False
    assert session.rac_episodes[0]["recovery_frames"] is None
    assert session.rac_episodes[0]["correction_frames"] == 120


def test_a_recovery_of_zero_frames_is_kept_as_a_real_claim(monkeypatch) -> None:
    """The operator pressed the key immediately: they DID assert there was no
    recovery to do. That is information, and distinct from the case above."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved(
        {"n": "1", "frames": "90", "seconds": "3.0", "recovery": "0", "labelled": "true"}
    )
    assert session.rac_episodes[0]["labelled"] is True
    assert session.rac_episodes[0]["recovery_frames"] == 0


def test_a_nonsensical_boundary_is_demoted_to_unlabelled(monkeypatch) -> None:
    """A recovery longer than the episode cannot be true. Recording it would put
    a negative correction length in the sidecar; dropping to unlabelled loses
    only an annotation nothing reads yet."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved(
        {"n": "1", "frames": "50", "seconds": "2.0", "recovery": "999", "labelled": "true"}
    )
    assert session.rac_episodes[0]["labelled"] is False
    assert session.rac_episodes[0]["correction_frames"] == 50


def test_episodes_are_keyed_by_dataset_episode_index(monkeypatch) -> None:
    """The sidecar is useless if its keys don't line up with the episodes on
    disk. A coaching dataset is created fresh per session, so index = n - 1."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    for n in (1, 2, 3):
        rollout._on_correction_saved(
            {"n": str(n), "frames": "60", "seconds": "2.0", "recovery": "10", "labelled": "true"}
        )
    assert sorted(session.rac_episodes) == [0, 1, 2]


def test_the_live_recovery_marker_is_exposed_and_cleared_per_takeover(monkeypatch) -> None:
    """Shown while the correction is still recording, then cleared when the next
    takeover begins — it describes the correction in progress, not a history."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_recovery_mark({"frames": "35"})
    assert session.recovery_marked_at == 35
    assert rollout._coach_fields(session)["recovery_marked_at"] == 35
    rollout._on_dagger_phase("correcting")  # a fresh takeover
    assert session.recovery_marked_at is None


def test_the_sidecar_is_written_next_to_the_dataset(monkeypatch, tmp_path) -> None:
    """The boundary is unrecoverable after the fact — nobody can look at a saved
    episode later and say where recovery ended — so it has to reach disk."""
    import json

    from makermodslab import rollout
    from makermodslab.dagger_protocol import RAC_SIDECAR_NAME

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.dataset_root = str(tmp_path)
    session.dataset_repo_id = "user/rollout_fixes_20260819_120000"
    session.rac_episodes = {
        0: {"recovery_frames": 40, "correction_frames": 80, "labelled": True},
        1: {"recovery_frames": None, "correction_frames": 95, "labelled": False},
    }
    rollout._write_rac_sidecar(session)

    written = json.loads((tmp_path / RAC_SIDECAR_NAME).read_text())
    assert written["version"] == 2
    assert written["dataset_repo_id"] == "user/rollout_fixes_20260819_120000"
    # JSON has no integer keys; the reader has to know they are indices.
    assert written["episodes"]["0"]["recovery_frames"] == 40
    assert written["episodes"]["1"]["labelled"] is False


def test_no_sidecar_is_written_when_nothing_was_recorded(monkeypatch, tmp_path) -> None:
    """An empty annotation file beside a dataset invites the reader to conclude
    the operator marked nothing, when the session may simply have saved nothing."""
    from makermodslab import rollout
    from makermodslab.dagger_protocol import RAC_SIDECAR_NAME

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.dataset_root = str(tmp_path)
    rollout._write_rac_sidecar(session)
    assert not (tmp_path / RAC_SIDECAR_NAME).exists()


def test_an_unwritable_sidecar_never_fails_the_session(monkeypatch) -> None:
    """The corrections are the deliverable; this is a note about them. A session
    whose episodes are safely on disk must not be reported as failed because an
    annotation could not be written."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    session.dataset_root = "/definitely/not/a/directory/anywhere"
    session.rac_episodes = {0: {"recovery_frames": 1, "correction_frames": 2, "labelled": True}}
    rollout._write_rac_sidecar(session)  # must not raise


# --- Real event ORDER, not handlers in isolation -----------------------------
#
# THE gap that let two features ship broken. Every other coaching test calls an
# `_on_*` handler directly, which cannot see what the runner emits NEXT — and
# what it emits next is a PHASE event, on the very same tick, after every
# transition. A notice parked in a field that any phase clears is therefore
# destroyed about a millisecond after it is written, with every unit test green.
#
# These drive `_handle_dagger_line` with the exact lines `dagger_runner` writes,
# in the order it writes them. Pure string -> state, so squarely within the
# tests/ policy.


def _drive(lines: list[str]) -> None:
    """Feed real protocol lines through the real dispatcher."""
    from makermodslab import rollout
    from makermodslab.dagger_protocol import EVENT_PREFIX

    for payload in lines:
        rollout._handle_dagger_line(f"{EVENT_PREFIX} {payload}")


def test_the_discard_notice_survives_the_runners_own_event_order(monkeypatch) -> None:
    """A too-short discard is followed immediately by `PHASE phase=paused`.

    Before the notice had a field of its own, that one line wiped it and the
    operator saw nothing at all — the exact bug the reason code was added to
    fix, still unfixed, now with an invisible message."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_CORRECTING})

    _drive(
        [
            "CORRECTION_CANCELLED reason=too_short frames=7 seconds=0.2 minimum=10",
            "PHASE phase=paused",
        ]
    )
    assert session.discard_notice is not None, "destroyed by the phase event that always follows"
    assert "7 frames" in session.discard_notice


def test_an_alignment_refusal_still_survives_its_own_sequence(monkeypatch) -> None:
    """The refusal keeps `align_error` because that path sets `transition =
    None` and emits no phase — the property the discard notice does NOT have,
    which is why copying the pattern across was wrong."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    _drive(["ALIGN_REQUIRED max_delta=40 joints=shoulder_pan:40"])
    assert session.align_error is not None


def test_a_full_correction_cycle_leaves_a_consistent_tally(monkeypatch) -> None:
    """Takeover, mark recovery, hand back, save — as the runner emits it."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    _drive(
        [
            "PHASE phase=correcting",
            "RECOVERY_MARK frames=40",
            "PHASE phase=saving",
            "CORRECTION_SAVED n=1 frames=120 seconds=4.0 recovery=40 labelled=true",
            "PHASE phase=paused",
        ]
    )
    assert session.corrections_saved == 1
    assert session.rac_episodes[0] == {
        "recovery_frames": 40,
        "correction_frames": 80,
        "labelled": True,
        # First correction of the first scene.
        "attempt_index": 0,
        "index_in_attempt": 0,
    }
    # The live marker describes the correction in progress; a new takeover
    # clears it, and there isn't one yet.
    assert session.recovery_marked_at == 40


def test_the_reset_outcome_reaches_the_session(monkeypatch) -> None:
    """A failed ease-home and a failed release must be distinguishable from a
    good reset — the UI tells the operator to grab the arm based on this."""
    from makermodslab import rollout

    session = rollout._CoachSession(request=_coaching_request(), corrections_target=5)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    _drive(["ATTEMPT_RESET n=1 homed=false limp=false", "PHASE phase=paused"])
    assert session.attempts == 1
    assert session.reset_homed is False
    assert session.reset_limp is False

    _drive(["ATTEMPT_RESET n=2 homed=true limp=true", "PHASE phase=paused"])
    assert session.reset_homed is True
    assert session.reset_limp is True


# --- The held correction, orchestrator side ----------------------------------


def _coach(**overrides):
    from makermodslab import rollout

    return rollout._CoachSession(request=_coaching_request(), corrections_target=5, **overrides)


def test_a_held_correction_counts_and_opens_the_drop_window(monkeypatch) -> None:
    """CORRECTION_HELD tallies exactly like CORRECTION_SAVED — the correction is
    real and recorded — and additionally says the operator can still take it
    back."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "4.5", "labelled": "false"}, held=True)
    assert session.corrections_saved == 1
    assert session.droppable_correction == {"n": 1, "frames": 90, "seconds": 4.5}


def test_a_plain_saved_correction_opens_no_drop_window(monkeypatch) -> None:
    """CORRECTION_SAVED means it is already on disk, and there is no supported
    way to take one episode back out of an open lerobot dataset. Offering the
    button anyway would be a delete that cannot happen."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "4.5"})
    assert session.corrections_saved == 1
    assert session.droppable_correction is None


def test_committing_closes_the_window_without_counting_again(monkeypatch) -> None:
    """THE double-count trap. The correction was tallied when it was recorded;
    a commit handler that tallied it again would report two corrections for
    every one the operator gave."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "4.5"}, held=True)
    rollout._on_correction_committed()
    assert session.corrections_saved == 1
    assert session.droppable_correction is None


def test_dropping_takes_the_correction_back_off_the_session(monkeypatch) -> None:
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "4.5"}, held=True)
    rollout._on_correction_dropped({"n": "0", "frames": "90"})
    assert session.corrections_saved == 0
    assert session.droppable_correction is None
    # The seconds were added when it was recorded, and it is not part of the
    # session any more.
    assert session.correction_seconds == pytest.approx(0.0)


def test_dropping_removes_the_rac_entry_for_that_episode(monkeypatch) -> None:
    """The sidecar is keyed by episode index. Leaving the dropped episode's
    entry behind would describe an episode the dataset does not contain, and
    every later entry would then be attributed to the wrong one."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved(
        {"n": "1", "frames": "90", "seconds": "4.5", "recovery": "30", "labelled": "true"},
        held=True,
    )
    assert 0 in session.rac_episodes
    rollout._on_correction_dropped({"n": "0", "frames": "90"})
    assert session.rac_episodes == {}


def test_a_dropped_correction_leaves_the_next_one_at_the_right_index(monkeypatch) -> None:
    """Two corrections, the second dropped, then a third. The third has to land
    on episode 1 — the slot the dropped one vacated — or the sidecar describes
    the wrong episodes for the rest of the session."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "60", "seconds": "3.0"}, held=True)
    rollout._on_correction_committed()
    rollout._on_correction_saved({"n": "2", "frames": "90", "seconds": "4.5"}, held=True)
    rollout._on_correction_dropped({"n": "1", "frames": "90"})
    rollout._on_correction_saved({"n": "2", "frames": "70", "seconds": "3.5"}, held=True)
    assert session.corrections_saved == 2
    assert sorted(session.rac_episodes) == [0, 1]


def test_the_drop_window_is_reported_to_the_browser(monkeypatch) -> None:
    """The browser must not infer the window from the phase — the runner owns
    when it opens and closes, and it closes on a commit the UI never sees."""
    from makermodslab import rollout

    session = _coach()
    monkeypatch.setattr(rollout, "_coach_session", session)
    rollout._on_correction_saved({"n": "1", "frames": "90", "seconds": "4.5"}, held=True)
    assert rollout._coach_fields(session)["droppable_correction"] == {
        "n": 1,
        "frames": 90,
        "seconds": 4.5,
    }
    assert rollout._coach_fields(None)["droppable_correction"] is None


# --- Stopping before the runner can hear us ----------------------------------
#
# Both runners start their stdin reader only AFTER the robot is connected,
# because `SOFollower.calibrate()` prompts with `input()` on that same stdin
# during `connect()`. So a Stop pressed while the policy is loading or the arms
# are connecting writes QUIT into a buffer nobody is reading — and the stop path
# then waited the full 45s for an answer that could not come, while the runner
# carried on loading the policy, opening the cameras and connecting both arms.
# Watched from the browser that is a Stop that does nothing for most of a minute
# and then aborts after everything has connected.


def test_a_stop_before_ready_terminates_instead_of_waiting_for_quit(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=5)
    monkeypatch.setattr(rollout, "_runner_ready", False)
    seen = []
    monkeypatch.setattr(rollout, "_quit_runner", lambda proc, **kw: seen.append(kw))

    rollout.handle_stop_inference()
    assert seen == [{"listening": False}]
    assert session.quitting is True


def test_a_stop_after_ready_still_asks_the_runner_to_wind_down(monkeypatch) -> None:
    """The escalation is ONLY for the window where nothing is reading. Once the
    runner is live, QUIT is what finalises the dataset and eases the arm home —
    signalling there would risk the corrections the operator just collected."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=5)
    monkeypatch.setattr(rollout, "_runner_ready", True)
    seen = []
    monkeypatch.setattr(rollout, "_quit_runner", lambda proc, **kw: seen.append(kw))

    rollout.handle_stop_inference()
    assert seen == [{"listening": True}]


def test_quit_runner_skips_the_wait_entirely_when_nothing_is_listening(monkeypatch) -> None:
    """The point is the ABSENCE of the 45s wait, so assert the wait never
    happens rather than that a terminate eventually does."""
    from makermodslab import rollout

    waited, terminated = [], []
    proc = types.SimpleNamespace(wait=lambda timeout=None: waited.append(timeout), stdin=None)
    monkeypatch.setattr(rollout, "_terminate_tree", lambda p, **kw: terminated.append(p))
    monkeypatch.setattr(rollout, "_send_runner_command", lambda p, c: waited.append("sent"))

    rollout._quit_runner(proc, listening=False)
    assert waited == []
    assert terminated == [proc]


# --- A stop is an abort whenever it lands ------------------------------------
#
# Stopping a coaching session while the policy was still loading ended with
# `phase: "error", outcome: "failed"`, and the `error` string was whatever the
# runner had last written to stderr — before READY that is only macOS's benign
# "objc[…]: Class AVFFrameReceiver is implemented in both …/cv2/… and …/av/…"
# warning. The frontend renders `outcome === "failed"` as a red destructive
# toast showing that last line, so cancelling a startup looked exactly like the
# app crashing on a dylib fault. Confirmed on hardware three times.


def _quitting_coach(monkeypatch, rollout, proc):
    """A coaching session whose operator has pressed Stop, runner still up."""
    session = _coach()
    session.quitting = True
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_proc", proc)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    return session


def test_a_stop_before_ready_is_an_abort_not_a_crash(monkeypatch) -> None:
    """The pre-READY stop path SIGNALS the runner, so it exits non-zero. That
    exit code used to drive the classification, turning an operator-requested
    stop into `failed` — the bug the operator saw as a crash toast."""
    from makermodslab import rollout

    proc = object()
    session = _quitting_coach(monkeypatch, rollout, proc)
    session.corrections_saved = 0
    monkeypatch.setattr(
        rollout,
        "_extract_error_from_log",
        lambda path: (
            "objc[41521]: Class AVFFrameReceiver is implemented in both "
            ".../cv2/cv2.abi3.so and .../av/_core.cpython-311-darwin.so. "
            "This may cause spurious casting failures and mysterious crashes."
        ),
    )

    rollout._handle_dagger_exit(proc, -15)

    assert rollout._last_result["phase"] == "aborted"
    assert rollout._last_result["outcome"] == "ok"
    # The benign dylib warning must never reach the operator as an error.
    assert rollout._last_result["error"] is None


def test_a_stop_after_ready_is_still_an_abort(monkeypatch) -> None:
    """The other window: a live session stopped with QUIT already classified
    correctly, and must keep doing so."""
    from makermodslab import rollout

    proc = object()
    session = _quitting_coach(monkeypatch, rollout, proc)
    session.corrections_saved = 3
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", 1005.0)

    rollout._handle_dagger_exit(proc, 0)

    assert rollout._last_result["phase"] == "aborted"
    assert rollout._last_result["outcome"] == "ok"
    assert rollout._last_result["corrections_saved"] == 3


def test_a_runner_that_reported_its_own_error_is_still_a_failure(monkeypatch) -> None:
    """The one exit a stop does NOT excuse: the runner said what broke. Hiding
    that behind "stopped" would hide a half-written dataset."""
    from makermodslab import rollout

    proc = object()
    session = _quitting_coach(monkeypatch, rollout, proc)
    session.runner_error = "RuntimeError: bus went away"

    rollout._handle_dagger_exit(proc, 1)

    assert rollout._last_result["phase"] == "error"
    assert rollout._last_result["outcome"] == "failed"
    assert rollout._last_result["error"] == "RuntimeError: bus went away"


def test_a_status_poll_that_wins_the_race_reports_the_abort_too(monkeypatch) -> None:
    """`handle_inference_status` is the backstop for the pump, and whichever
    path arrives first writes the terminal payload — so it has to reach the
    same verdict or the operator's toast depends on a thread race."""
    from makermodslab import rollout

    proc = types.SimpleNamespace(poll=lambda: -15, returncode=-15)
    _quitting_coach(monkeypatch, rollout, proc)
    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda path: "objc[41521]: Class …")

    payload = rollout.handle_inference_status()

    assert payload["phase"] == "aborted"
    assert payload["outcome"] == "ok"
    assert payload["error"] is None


# --- drop_last must not claim it dropped something ---------------------------
#
# POST /sessions/{id}/coaching {"command": "drop_last"} answered
# 200 {"success": true, "message": "Drop_last sent"} even with nothing held —
# seen mid-correction and again straight after a previous drop. The runner logs
# its refusal and does nothing, so the caller was told a correction had been
# un-recorded when none had been.


def _armed_coaching_command(monkeypatch, rollout, written):
    """A live coaching session whose commands land in `written`."""
    session = _coach()
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_proc", object())
    monkeypatch.setattr(rollout, "_send_runner_command", lambda proc, cmd: (written.append(cmd), True)[1])
    return session


def test_drop_last_is_refused_when_no_correction_is_held(monkeypatch) -> None:
    from makermodslab import rollout

    written: list[str] = []
    _armed_coaching_command(monkeypatch, rollout, written)

    result = rollout.handle_coaching_command("drop_last")

    assert result["success"] is False
    assert result["status_code"] == 409
    assert result["code"] == "coaching.nothing_to_drop"
    # And nothing was written: a refusal the runner would only have logged.
    assert written == []


def test_drop_last_is_forwarded_while_the_window_is_open(monkeypatch) -> None:
    """The refusal reads `droppable_correction`, so an open window still sends."""
    from makermodslab import rollout

    written: list[str] = []
    session = _armed_coaching_command(monkeypatch, rollout, written)
    session.droppable_correction = {"n": 1, "frames": 90, "seconds": 4.5}

    result = rollout.handle_coaching_command("drop_last")

    assert result["success"] is True
    assert written == ["DROP_LAST"]


def test_every_other_verb_is_still_forwarded_without_a_state_check(monkeypatch) -> None:
    """The narrowness is the point: a server-side phase copy is one event stale,
    so pre-checking the rest would reject commands the arm was ready for."""
    from makermodslab import rollout

    written: list[str] = []
    _armed_coaching_command(monkeypatch, rollout, written)

    for verb in ("takeover", "handback", "cancel", "hold", "resume", "reset", "recovered"):
        assert rollout.handle_coaching_command(verb)["success"] is True
    assert written == [
        "TAKEOVER",
        "HANDBACK",
        "CANCEL",
        "HOLD",
        "RESUME",
        "RESET",
        "RECOVERED",
    ]


def test_the_command_message_is_not_a_protocol_token(monkeypatch) -> None:
    """ "Drop_last sent" is the wire verb leaking into a sentence a person reads."""
    from makermodslab import rollout

    written: list[str] = []
    session = _armed_coaching_command(monkeypatch, rollout, written)
    session.droppable_correction = {"n": 1, "frames": 90, "seconds": 4.5}

    assert rollout.handle_coaching_command("drop_last")["message"] == "Drop last sent"
    assert rollout.handle_coaching_command("takeover")["message"] == "Takeover sent"


# --- The pyav decoder fallback, at every dataset call site --------------------
#
# torchcodec is lerobot's default video decoder and its native libraries do not
# load on a host without FFmpeg (`dlopen … libavutil.56.dylib` fails — this
# machine). The training path already probes for that (jobs.py, via
# utils.system.torchcodec_loads) and asks for pyav, which bundles its own
# FFmpeg. Two dataset call sites did not, and would raise on first frame
# access: merge's `_strip_features` and datasets' `push_dataset_to_hub`.


class _RecordingDataset:
    """A LeRobotDataset stand-in that records the kwargs it was built with."""

    def __init__(self, *args, **kwargs) -> None:
        _RecordingDataset.seen = kwargs
        self.meta = types.SimpleNamespace(features={})
        self.repo_id = args[0] if args else None
        self.num_episodes = 0

    def push_to_hub(self, **kwargs) -> None:
        pass


@pytest.mark.parametrize(
    ("loads", "expected"),
    [(False, "pyav"), (True, None)],
)
def test_merge_strip_features_falls_back_to_pyav(monkeypatch, tmp_path, loads, expected) -> None:
    """`None` is not "no opinion by accident": it is what leaves lerobot's own
    default in place on a host where torchcodec works."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    from makermodslab import merge

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", _RecordingDataset)
    monkeypatch.setattr(merge, "torchcodec_loads", lambda: loads)

    # No feature to drop, so it returns before touching dataset_tools — the
    # construction is the whole point of the test.
    assert merge._strip_features("user/set", tmp_path, ["intervention"], tmp_path, 0) is None
    assert _RecordingDataset.seen["video_backend"] == expected


@pytest.mark.parametrize(
    ("loads", "expected"),
    [(False, "pyav"), (True, None)],
)
def test_push_dataset_to_hub_falls_back_to_pyav(monkeypatch, loads, expected) -> None:
    import lerobot.datasets as lerobot_datasets
    from makermodslab import datasets

    monkeypatch.setattr(lerobot_datasets, "LeRobotDataset", _RecordingDataset)
    monkeypatch.setattr(datasets, "torchcodec_loads", lambda: loads)
    monkeypatch.setattr(datasets, "resolve_hub_repo_id", lambda repo_id: f"someone/{repo_id}")
    monkeypatch.setattr(datasets, "invalidate_hub_status", lambda repo_id: None)
    monkeypatch.setattr(datasets, "invalidate_hub_dataset_info", lambda repo_id: None)

    datasets.push_dataset_to_hub("my_set", tags=None, private=False)
    assert _RecordingDataset.seen["video_backend"] == expected


# --- The recovery mark must not outlive its correction ------------------------
#
# `recovery_marked_at` only ever cleared on the NEXT takeover, so it still read
# 96 after that correction was cancelled — through the whole parked window —
# and still read 66 after a drop_last. Both describe an episode that no longer
# exists, and a UI reading the field shows it as this session's live state.


def test_a_cancelled_correction_clears_the_recovery_mark(monkeypatch) -> None:
    from makermodslab import rollout

    session = _coach()
    session.recovery_marked_at = 96
    monkeypatch.setattr(rollout, "_coach_session", session)

    rollout._on_correction_cancelled({"reason": "operator", "frames": "140"})

    assert session.recovery_marked_at is None
    assert rollout._coach_fields(session)["recovery_marked_at"] is None


def test_a_too_short_discard_clears_the_recovery_mark_too(monkeypatch) -> None:
    """The other cancel reason takes an early return of its own — the mark has
    to be gone before either path leaves."""
    from makermodslab import rollout

    session = _coach()
    session.recovery_marked_at = 12
    monkeypatch.setattr(rollout, "_coach_session", session)

    rollout._on_correction_cancelled({"reason": "too_short", "frames": "4", "seconds": "0.2"})

    assert session.recovery_marked_at is None
    # …and the notice the operator does need still arrives.
    assert session.discard_notice is not None


def test_a_dropped_correction_clears_the_recovery_mark(monkeypatch) -> None:
    from makermodslab import rollout

    session = _coach()
    session.corrections_saved = 1
    session.droppable_correction = {"n": 1, "frames": 120, "seconds": 6.0}
    session.recovery_marked_at = 66
    monkeypatch.setattr(rollout, "_coach_session", session)

    rollout._on_correction_dropped({"n": "0"})

    assert session.recovery_marked_at is None
    assert rollout._coach_fields(session)["recovery_marked_at"] is None


# --- "Listening" has to mean READY for BOTH runner kinds ----------------------
#
# `_quit_runner(listening=...)` used to be derived from
# `_inference_rollout_started_at`. Coaching sets that from `_on_dagger_ready`,
# so it really was READY there; EVAL sets it from `_on_episode_started`, one
# command later. A stop landing after the eval runner started reading its pipe
# but before episode 1 began therefore signalled a runner that could have been
# asked to quit. Short window and lerobot's signal handler kept it graceful, so
# this fixes a false premise rather than a lost dataset — but the premise is the
# thing the escalation is built on.


def test_an_eval_stop_between_ready_and_the_first_episode_asks_nicely(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    session.episode_pending = True
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    seen = []
    monkeypatch.setattr(rollout, "_quit_runner", lambda proc, **kw: seen.append(kw))

    # READY arrives; the first episode has been issued but has not started, so
    # `_inference_rollout_started_at` is still None.
    rollout._on_runner_ready()
    assert rollout._inference_rollout_started_at is None
    rollout.handle_stop_inference()

    assert seen == [{"listening": True}]


def test_ready_with_no_episode_pending_still_counts_as_listening(monkeypatch) -> None:
    """READY says the runner is reading its pipe. Whether we had an episode to
    give it is a different question, and the early return must not swallow it."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    session.episode_pending = False

    rollout._on_runner_ready()

    assert rollout._runner_ready is True


def test_a_coaching_ready_still_marks_the_runner_listening(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", _coach())
    monkeypatch.setattr(rollout, "_inference_started_at", 1000.0)

    rollout._on_dagger_ready()

    assert rollout._runner_ready is True


def test_a_dead_eval_runner_is_no_longer_listening(monkeypatch) -> None:
    """Its respawn earns a fresh READY — a stop before that must not write QUIT
    into a pipe belonging to a process that has gone."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    monkeypatch.setattr(rollout, "_runner_ready", True)

    with rollout._state_lock:
        rollout._finalise_runner_exit_locked(1, session)

    assert rollout._runner_ready is False


def test_a_finished_episode_leaves_a_live_runner_listening(monkeypatch) -> None:
    """The control: between episodes the runner never stopped reading its pipe,
    so the next stop must still be a QUIT."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    monkeypatch.setattr(rollout, "_runner_ready", True)

    with rollout._state_lock:
        rollout._finalise_eval_episode_locked(0, session, keep_runner=True)

    assert rollout._runner_ready is True


def test_going_idle_clears_the_listening_flag(monkeypatch) -> None:
    """It describes ONE process. Leaking it into the next session would have a
    stop QUIT a runner that is still loading its policy."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    monkeypatch.setattr(rollout, "_runner_ready", True)

    with rollout._state_lock:
        rollout._go_idle_locked()

    assert rollout._runner_ready is False


# --- Aborted startups must not leave empty datasets behind --------------------
#
# A coaching session killed during startup left a dataset directory whose
# meta/info.json reported `total_episodes: 0, total_frames: 0`. GET
# /api/v1/datasets filters those out, so they were invisible — and accumulated,
# one per cancelled start.


def _coaching_dataset_dir(tmp_path, *, episodes: int, frames: int) -> Path:
    root = tmp_path / "rollout_shirt_fixes_20260901_120000"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes, "total_frames": frames}))
    return root


def _abort_coaching(monkeypatch, rollout, session) -> None:
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    with rollout._state_lock:
        rollout._finalise_coaching_locked(None, session, aborted=True)


def test_an_abort_with_no_episodes_removes_the_dataset_directory(monkeypatch, tmp_path) -> None:
    from makermodslab import rollout

    root = _coaching_dataset_dir(tmp_path, episodes=0, frames=0)
    session = _coach()
    session.dataset_root = str(root)
    session.dataset_repo_id = root.name

    _abort_coaching(monkeypatch, rollout, session)

    assert not root.exists()
    # And the abort itself is still reported exactly as before.
    assert rollout._last_result["phase"] == "aborted"


def test_an_abort_after_one_correction_keeps_everything(monkeypatch, tmp_path) -> None:
    """The whole point of the guards: corrections on disk are the deliverable,
    and a stop is the normal way a session ends."""
    from makermodslab import rollout

    root = _coaching_dataset_dir(tmp_path, episodes=1, frames=120)
    session = _coach()
    session.dataset_root = str(root)
    session.dataset_repo_id = root.name
    session.corrections_saved = 1

    _abort_coaching(monkeypatch, rollout, session)

    assert (root / "meta" / "info.json").exists()


def test_a_zero_tally_disagreeing_with_the_dataset_is_left_alone(monkeypatch, tmp_path) -> None:
    """Two independent sources say whether anything was recorded, and it takes
    BOTH to remove a directory — a dropped CORRECTION_SAVED event must not cost
    the operator the episodes lerobot did write."""
    from makermodslab import rollout

    root = _coaching_dataset_dir(tmp_path, episodes=2, frames=240)
    session = _coach()
    session.dataset_root = str(root)

    _abort_coaching(monkeypatch, rollout, session)

    assert root.exists()


def test_an_unreadable_info_json_is_never_removed(monkeypatch, tmp_path) -> None:
    """A directory we cannot prove is empty is one we keep."""
    from makermodslab import rollout

    root = tmp_path / "rollout_shirt_fixes_20260901_120000"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{not json")
    session = _coach()
    session.dataset_root = str(root)

    _abort_coaching(monkeypatch, rollout, session)

    assert root.exists()


def test_a_session_that_ran_to_target_is_never_cleaned_up(monkeypatch, tmp_path) -> None:
    """Cleanup is the ABORT path only. A finished (or crashed) session's
    directory is somebody's dataset, however it reads."""
    from makermodslab import rollout

    root = _coaching_dataset_dir(tmp_path, episodes=0, frames=0)
    session = _coach()
    session.dataset_root = str(root)
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_coach_session", session)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    with rollout._state_lock:
        rollout._finalise_coaching_locked(0, session)

    assert root.exists()


# --- A stop before READY should not wait out a grace nobody can use ----------
#
# `dagger_runner` installs lerobot's signal handler before `build_rollout_context`,
# and that handler only sets an event which `build_rollout_context` never reads.
# So pre-READY the child provably cannot act on SIGTERM: the policy load and the
# camera opens run to completion regardless. Measured on the station, the full
# 5s grace elapsed every time and cameras kept connecting after the stop.


def test_a_stop_before_ready_uses_the_short_grace(monkeypatch) -> None:
    from makermodslab import rollout

    seen = []
    monkeypatch.setattr(rollout, "_terminate_tree", lambda p, **kw: seen.append(kw))
    monkeypatch.setattr(rollout, "_send_runner_command", lambda p, c: pytest.fail("must not send QUIT"))

    rollout._quit_runner(object(), listening=False)
    assert seen == [{"timeout": rollout._PRE_READY_TERMINATE_TIMEOUT_S}]
    assert pytest.approx(1.0) == rollout._PRE_READY_TERMINATE_TIMEOUT_S


def test_the_post_ready_fallback_keeps_the_patient_grace(monkeypatch) -> None:
    """THE one that must not regress. After READY the runner may be mid
    `save_episode()` encoding video; SIGKILLing it there leaves the parquet
    footers unwritten while the summary offers to fine-tune on the result. That
    is why the short grace is passed at the pre-READY sites rather than made the
    default — a forgotten keyword must fail towards patience."""
    import subprocess

    from makermodslab import rollout

    seen = []
    monkeypatch.setattr(rollout, "_terminate_tree", lambda p, **kw: seen.append(kw))
    monkeypatch.setattr(rollout, "_send_runner_command", lambda p, c: True)

    class _Wedged:
        stdin = None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="runner", timeout=timeout)

    rollout._quit_runner(_Wedged(), listening=True)
    # No timeout override: _terminate_tree's patient default stands.
    assert seen == [{}]


def test_the_short_grace_is_never_the_default(monkeypatch) -> None:
    """If someone flips the default, every post-READY caller silently becomes
    aggressive and the failure is a corrupted dataset, not a test error."""
    import inspect

    from makermodslab import rollout

    default = inspect.signature(rollout._terminate_tree).parameters["timeout"].default
    assert default > rollout._PRE_READY_TERMINATE_TIMEOUT_S


def test_bimanual_coaching_is_refused_before_any_hardware_is_touched(monkeypatch) -> None:
    """It cannot work on this pin — the left_* camera prefix defeats every policy
    trained on unprefixed names — and without this guard the operator waits ~30s
    while the policy downloads and four arms connect, only to be told about a
    CLI flag they cannot reach. Refuse in the launch panel instead."""
    from makermodslab import rollout

    req = _coaching_request(mode="bimanual")
    result = rollout.handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "bimanual" in result["message"].lower()
    # And nothing was claimed: the slot must be free for the next attempt.
    assert rollout.inference_active is False


def test_single_arm_coaching_is_not_caught_by_that_guard(monkeypatch) -> None:
    """The guard keys on mode alone, so it is exactly one `==` away from
    disabling the feature entirely."""

    req = _coaching_request()
    assert req.mode != "bimanual"


# --- MolmoAct2: the flags the lerobot pin requires --------------------------
#
# `_rollout_cli_args` reads the checkpoint's own config.json to decide these,
# so the tests below write a real (tiny) one into tmp_path. Every other test in
# this file passes a path that does not exist, which is the "{} → add nothing"
# fallback and is why none of them changed.


def _checkpoint_dir(tmp_path, config: dict) -> str:
    """A pretrained_model dir holding just the config.json the builder reads."""
    d = tmp_path / "pretrained_model"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    return str(d)


def test_rollout_cli_args_add_nothing_for_act_and_smolvla(tmp_path) -> None:
    """The required-flag machinery must be invisible to every policy that
    doesn't need it — including on the eval and coaching front-ends."""
    from makermodslab.rollout import _rollout_cli_args

    for policy_type in ("act", "smolvla"):
        path = _checkpoint_dir(tmp_path / policy_type, {"type": policy_type})
        args = _rollout_cli_args(_stub_request(), path, [])
        assert not any(a.startswith("--policy.inference_action_mode") for a in args)


def test_rollout_cli_args_set_molmoact2_inference_action_mode(tmp_path) -> None:
    """MolmoAct2Config.inference_action_mode has no usable default — the policy
    raises "requires `inference_action_mode` to be set explicitly" on None — so
    a checkpoint that saved none is unrunnable without this override."""
    from makermodslab.rollout import _rollout_cli_args

    path = _checkpoint_dir(tmp_path, {"type": "molmoact2", "action_mode": "both"})
    args = _rollout_cli_args(_stub_request(), path, [])
    assert "--policy.inference_action_mode=continuous" in args


def test_rollout_cli_args_respect_a_checkpoints_own_action_mode(tmp_path) -> None:
    """The released lerobot/MolmoAct2-*-LeRobot config already saves
    inference_action_mode=continuous. Overriding a saved choice would be how a
    discrete checkpoint silently gets run through the wrong head."""
    from makermodslab.rollout import _rollout_cli_args

    path = _checkpoint_dir(
        tmp_path,
        {"type": "molmoact2", "action_mode": "continuous", "inference_action_mode": "continuous"},
    )
    args = _rollout_cli_args(_stub_request(), path, [])
    assert not any(a.startswith("--policy.inference_action_mode") for a in args)


def test_eval_runner_cmd_carries_the_molmoact2_flag(tmp_path) -> None:
    """Same reason the temporal-ensemble flags are tested on both front-ends:
    an eval must drive the policy exactly the way a single rollout would."""
    from makermodslab.rollout import InferenceRequest, _build_eval_runner_cmd

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        eval_episodes=5,
    )
    path = _checkpoint_dir(tmp_path, {"type": "molmoact2", "action_mode": "both"})
    cmd = _build_eval_runner_cmd(req, path, [])
    assert "makermodslab.eval_runner" in cmd
    assert "--policy.inference_action_mode=continuous" in cmd


def test_rtc_is_refused_for_a_discrete_molmoact2_checkpoint(tmp_path) -> None:
    """MolmoAct2Policy.supports_rtc() is `inference_action_mode ==
    "continuous"`, and build_rollout_context raises on a False — but only after
    loading a multi-GB VLM onto the accelerator and with a message naming
    `--inference.type`, a flag no UI user can reach."""
    from makermodslab.rollout import InferenceRequest, _rollout_cli_args

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        inference_engine="rtc",
    )
    path = _checkpoint_dir(tmp_path, {"type": "molmoact2", "inference_action_mode": "discrete"})
    with pytest.raises(ValueError, match="Real-Time Chunking"):
        _rollout_cli_args(req, path, [])


def test_rtc_is_allowed_for_a_continuous_molmoact2_checkpoint(tmp_path) -> None:
    from makermodslab.rollout import InferenceRequest, _rollout_cli_args

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        inference_engine="rtc",
    )
    path = _checkpoint_dir(tmp_path, {"type": "molmoact2", "inference_action_mode": "continuous"})
    args = _rollout_cli_args(req, path, [])
    assert "--inference.type=rtc" in args


def test_molmoact2_hints_name_the_extra_and_the_action_head() -> None:
    """Both failures are otherwise opaque: lerobot's require_package message
    names a pip extra, and the action-mode refusal names a config field."""
    from makermodslab.utils.errors import friendly_hint

    extra = friendly_hint(
        "ImportError: 'scipy' is required but not installed. "
        "Install it with: pip install 'lerobot[molmoact2]'"
    )
    assert extra is not None and "lerobot[molmoact2]" in extra

    head = friendly_hint(
        "ValueError: MolmoAct2 checkpoint was trained with action_mode='discrete' "
        "and cannot run continuous inference."
    )
    assert head is not None and "action_mode" in head

    # The bare field name is NOT a trigger: lerobot logs the effective policy
    # config, so it appears in the log tail of unrelated failures too.
    assert friendly_hint("inference_action_mode: continuous") is None


def test_molmoact2_on_a_non_cuda_host_warns_but_still_starts(monkeypatch, tmp_path) -> None:
    """Warn-but-allow, on the same `meta["warning"]` channel the arm-identity
    findings use. Nothing in this lerobot pin REQUIRES CUDA — the action-flow
    CUDA graph falls back off-CUDA — so a refusal would be MakerMods Lab
    inventing a hardware requirement lerobot does not state. The device is
    injected; nothing here touches a real accelerator.

    Same harness as the return-to-initial-position test: every hardware-touching
    preflight and the subprocess itself are stubbed, the startup worker runs
    inline, and HOME is redirected so the log lands in tmp."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg, arm_type="so101": cfg)
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_detect_device", lambda: "mps")
    # Restored by monkeypatch at teardown, so this test can't leak a claimed
    # session into the next one.
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    checkpoint = _checkpoint_dir(tmp_path, {"type": "molmoact2", "inference_action_mode": "continuous"})
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: checkpoint)

    class _FakeProc:
        pid = 4321

        def __init__(self, cmd, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    result = rollout.handle_start_inference(_stub_request())
    # Started, not refused — that is the whole point of a warning.
    assert result["success"] is True, result
    warning = rollout._inference_meta.get("warning")
    assert warning is not None
    assert "mps" in warning
    assert "CUDA" in warning


def test_an_act_checkpoint_on_the_same_host_gets_no_device_warning(monkeypatch, tmp_path) -> None:
    """The guard keys on the checkpoint's policy type, so it is one `!=` away
    from warning about every ACT run on every Mac in the building."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg, arm_type="so101": cfg)
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_detect_device", lambda: "mps")
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_meta", {})

    checkpoint = _checkpoint_dir(tmp_path, {"type": "act"})
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: checkpoint)

    class _FakeProc:
        pid = 4322

        def __init__(self, cmd, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    assert rollout.handle_start_inference(_stub_request())["success"] is True
    assert rollout._inference_meta.get("warning") is None
