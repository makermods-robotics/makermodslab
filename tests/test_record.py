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
"""Tests for makermodslab.record — request schemas and handler entry points."""

from __future__ import annotations

import threading
import time

import pytest


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from makermodslab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_recording_status_handler_exposes_state_fields() -> None:
    from makermodslab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result


def test_recording_status_surfaces_preparing_substeps(monkeypatch) -> None:
    """record_with_web_events refines the coarse "preparing" window into named
    substeps ("connecting_robot", "connecting_teleop") by writing current_phase.
    The status handler must pass those through verbatim so the UI can name the
    substep — verified here without touching hardware by driving the module
    global the worker sets."""
    from makermodslab import record

    for substep in ("connecting_robot", "connecting_teleop"):
        monkeypatch.setattr(record, "current_phase", substep)
        # An active session with no config still surfaces current_phase.
        result = record.handle_recording_status()
        assert result["current_phase"] == substep
        # A preparing substep is not a completed/errored session.
        assert result["session_ended"] is False


class _FakeWorker:
    """Thread double: reports alive until joined."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.joined = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
        self._alive = False


def test_stop_recording_during_release_grace_releases_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second Stop while the session-end cleanup is holding torque for the
    release grace must cut the hold short (release immediately).
    """
    import threading

    import makermodslab.record as record

    release_now = threading.Event()
    monkeypatch.setattr(record, "releasing", True)
    monkeypatch.setattr(record, "_release_now", release_now)

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert release_now.is_set()
    assert "releasing" in result["message"].lower()


def test_stop_recording_mentions_rest_pose_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first stop must tell the user the arm returns to its starting
    position, then goes limp — no timed hold anymore (same as teleop)."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", {"stop_recording": False, "exit_early": False})

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert "returns to its starting position" in result["message"]
    assert "holds its pose" not in result["message"]  # the timed hold is gone
    assert "Stop again" in result["message"]


def test_record_finish_pending_release_cuts_grace_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import makermodslab.record as record

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "releasing", True)
    monkeypatch.setattr(record, "_release_now", release_now)

    assert record.finish_pending_release() is True
    assert release_now.is_set()
    assert worker.joined is True


def test_record_finish_pending_release_leaves_live_session_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import makermodslab.record as record

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "_release_now", release_now)

    assert record.finish_pending_release() is False
    assert not release_now.is_set()
    assert worker.joined is False


def test_record_finish_pending_release_noop_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_thread", None)
    assert record.finish_pending_release() is True


def test_recording_status_reports_releasing(monkeypatch: pytest.MonkeyPatch) -> None:
    """During the post-stop return the status must say the arm is still
    energized and going home (releasing) rather than pretending the session
    is fully over."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "releasing", True)
    status = record.handle_recording_status()
    assert status["releasing"] is True
    assert "returning the arm" in status["message"].lower()


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import makermodslab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
    )

    # Cameras come from the robot record, resolved by the caller; passing them
    # in explicitly is the same path handle_start_recording takes.
    config = record.create_record_config(
        request,
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_create_record_config_builds_biso_for_bimanual(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bimanual request stages the four arbitrarily-named library configs and
    builds a BiSO pair pointed at the per-device staging dirs."""
    import makermodslab.record as record
    from lerobot.robots.bi_so_follower import BiSOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig

    staged: dict = {}

    def _fake_stage(base, leader_left, leader_right, follower_left, follower_right):
        staged.update(
            base=base,
            leader=(leader_left, leader_right),
            follower=(follower_left, follower_right),
        )
        return (f"/staging/{base}/leader", f"/staging/{base}/follower", base)

    monkeypatch.setattr("makermodslab.utils.robot_factory.stage_bimanual_calibrations", _fake_stage)

    # Config names are ARBITRARY — no "<base>_left/right" convention required.
    request = record.RecordingRequest(
        leader_port="/dev/ll",
        follower_port="/dev/lf",
        leader_config="alice",
        follower_config="bob",
        mode="bimanual",
        right_leader_port="/dev/rl",
        right_follower_port="/dev/rf",
        right_leader_config="carol",
        right_follower_config="dave",
        robot_name="mybot",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
    )

    # No cameras for this session — passed explicitly so the bimanual assembly
    # is exercised without needing a "mybot" record on disk.
    config = record.create_record_config(request, cameras={})
    assert isinstance(config.robot, BiSOFollowerConfig)
    assert isinstance(config.teleop, BiSOLeaderConfig)
    # BiSO id + calibration_dir come from the staging helper (base = robot name).
    assert config.robot.id == "mybot"
    assert config.teleop.id == "mybot"
    assert str(config.robot.calibration_dir) == "/staging/mybot/follower"
    assert str(config.teleop.calibration_dir) == "/staging/mybot/leader"
    assert config.robot.right_arm_config.port == "/dev/rf"
    # Helper received the four library stems, grouped per device.
    assert staged["base"] == "mybot"
    assert staged["leader"] == ("alice", "carol")
    assert staged["follower"] == ("bob", "dave")


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    # fourcc defaults to MJPG (compressed) to avoid USB isochronous-bandwidth
    # exhaustion on multi-camera Linux rigs; an explicit choice still wins.
    assert configs["cam"].fourcc == "MJPG"
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_defaults_fourcc_to_mjpg() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _DEFAULT_FOURCC, _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert _DEFAULT_FOURCC == "MJPG"
    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_fourcc_overrides_mjpg_default() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "YUYV"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "YUYV"


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lerobot.cameras.configs import Cv2Backends
    from makermodslab.record import _build_camera_configs

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


def test_is_transient_camera_error_markers_pinned_against_lerobot_source() -> None:
    """Pins the marker set against lerobot's *actual* raise sites, not
    hand-typed copies of them — a `str.__contains__` over a literal tuple
    can't fail on its own, so the thing worth testing is whether upstream
    still emits these strings, not whether the tuple matches itself. If
    lerobot rewords one of these RuntimeErrors, this test goes red instead of
    the retry silently stopping firing.

    "failed to set capture_" is deliberately NOT pinned here — it's excluded
    from `_is_transient_camera_error` on purpose (see its docstring)."""
    import inspect

    from lerobot.cameras.opencv import camera_opencv

    source = inspect.getsource(camera_opencv)

    assert "failed to set fps={self.fps} ({actual_fps=})" in source
    assert "do not match configured width={self.capture_width} or height={self.capture_height}" in source
    assert "Timed out waiting for frame from camera {self}" in source


def test_is_transient_camera_error_matches_existing_markers() -> None:
    from makermodslab.record import _is_transient_camera_error

    assert _is_transient_camera_error("OpenCVCamera(0) failed to set fps=30 (actual_fps=5.0).")
    assert _is_transient_camera_error(
        "OpenCVCamera(0) frame width=360 or height=640 do not match configured width=480 or height=640."
    )
    assert _is_transient_camera_error("Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms.")


def test_is_transient_camera_error_false_for_unrelated_errors() -> None:
    from makermodslab.record import _is_transient_camera_error

    assert not _is_transient_camera_error("Could not connect on port '/dev/ttyUSB0'.")
    assert not _is_transient_camera_error("Failed to open OpenCVCamera(97).")


def test_is_transient_camera_error_false_for_capture_size_mismatch() -> None:
    """`friendly_hint()` (utils/errors.py) already classifies "failed to set
    capture_" as a permanent misconfiguration ("camera doesn't support the
    configured resolution — click Auto"). Retrying it wastes ~9s telling the
    operator the same thing three times. Don't add this marker here without
    reconciling the two classifiers first."""
    from makermodslab.record import _is_transient_camera_error

    assert not _is_transient_camera_error(
        "OpenCVCamera(0) failed to set capture_width=640 (actual_width=1920, width_success=True)."
    )


def test_is_transient_camera_error_false_for_fourcc_mismatch() -> None:
    """lerobot logs a fourcc mismatch as a warning (camera_opencv.py's
    `_validate_fourcc`) rather than raising — it never reaches this
    classifier as an exception message. Not retried, but for a different
    reason than the negative cases above: excluded by construction, not by
    policy."""
    from makermodslab.record import _is_transient_camera_error

    assert not _is_transient_camera_error(
        "OpenCVCamera(0) failed to set fourcc=MJPG (actual=YUYV, success=False)."
    )


def _make_dataset_dir(cache, repo_id: str, total_episodes: int):
    """Create a minimal on-disk LeRobot dataset dir (meta/info.json) under the
    tmp cache root, plus a fake video file so 'removed' is observable."""
    import json
    from pathlib import Path

    target = Path(cache) / repo_id
    (target / "meta").mkdir(parents=True, exist_ok=True)
    (target / "meta" / "info.json").write_text(json.dumps({"total_episodes": total_episodes}))
    (target / "videos").mkdir(parents=True, exist_ok=True)
    (target / "videos" / "ep.mp4").write_bytes(b"\x00" * 1024)
    return target


def test_discard_empty_dataset_removes_zero_episode_dir(tmp_lerobot_home) -> None:
    """A non-resume session that saved zero episodes has its directory removed."""
    import makermodslab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/big_20260703_120000", total_episodes=0)
    assert target.exists()

    removed = record._discard_empty_dataset("tester/big_20260703_120000", resume=False)

    assert removed is True
    assert not target.exists()


def test_discard_empty_dataset_keeps_nonempty_dir(tmp_lerobot_home) -> None:
    """A directory that recorded >=1 episode is never removed."""
    import makermodslab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/good_20260703_120000", total_episodes=3)

    removed = record._discard_empty_dataset("tester/good_20260703_120000", resume=False)

    assert removed is False
    assert target.exists()


def test_discard_empty_dataset_never_touches_resume_session(tmp_lerobot_home) -> None:
    """A resume/append session writes into a pre-existing dataset — even at zero
    NEW episodes on disk, the directory must never be removed."""
    import makermodslab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/preexisting", total_episodes=0)

    removed = record._discard_empty_dataset("tester/preexisting", resume=True)

    assert removed is False
    assert target.exists()


def test_discard_empty_dataset_rejects_path_traversal(tmp_lerobot_home) -> None:
    """A repo_id escaping the cache root is refused (no deletion outside cache)."""
    import makermodslab.record as record

    removed = record._discard_empty_dataset("../../etc", resume=False)
    assert removed is False


def test_discard_empty_dataset_invalidates_hub_status(tmp_lerobot_home) -> None:
    """Removing an empty dataset drops any cached Hub-existence probe for it."""
    import makermodslab.datasets as datasets
    import makermodslab.record as record

    _make_dataset_dir(tmp_lerobot_home, "tester/probed_20260703", total_episodes=0)
    # Seed a cached probe answer for the repo id.
    with datasets._HUB_STATUS_LOCK:
        datasets._HUB_STATUS_CACHE["tester/probed_20260703"] = "local_only"

    assert record._discard_empty_dataset("tester/probed_20260703", resume=False) is True

    with datasets._HUB_STATUS_LOCK:
        assert "tester/probed_20260703" not in datasets._HUB_STATUS_CACHE


# ---------------------------------------------------------------------------
# Quit-without-saving (discard) — _discard_session_dataset + the stop handler.
# The resume-protection test comes FIRST and is the load-bearing data-safety
# guarantee: a quit must never delete a pre-existing (resume) dataset.
# ---------------------------------------------------------------------------


def test_discard_session_dataset_never_touches_resume_session(tmp_lerobot_home) -> None:
    """A QUIT on a RESUME session must NEVER delete the pre-existing dataset —
    lerobot already committed its earlier episodes, so they must survive. The
    resume guard is checked first; this is the load-bearing safety property."""
    import makermodslab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/preexisting", total_episodes=5)

    removed = record._discard_session_dataset("tester/preexisting", resume=True)

    assert removed is False
    assert target.exists()  # every already-saved episode is intact


def test_discard_session_dataset_removes_fresh_dir_with_episodes(tmp_lerobot_home) -> None:
    """A QUIT on a FRESH session removes the whole stamped directory even when
    episodes were saved earlier THIS session — quit discards everything the
    session created (unlike _discard_empty_dataset, which keeps a non-empty dir)."""
    import makermodslab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/quit_20260708_120000", total_episodes=3)
    assert target.exists()

    removed = record._discard_session_dataset("tester/quit_20260708_120000", resume=False)

    assert removed is True
    assert not target.exists()


def test_discard_session_dataset_rejects_path_traversal(tmp_lerobot_home) -> None:
    """A repo_id escaping the cache root is refused — no deletion outside cache."""
    import makermodslab.record as record

    removed = record._discard_session_dataset("../../etc", resume=False)
    assert removed is False


def test_discard_session_dataset_invalidates_hub_status(tmp_lerobot_home) -> None:
    """Discarding a quit session drops any cached Hub-existence probe for it."""
    import makermodslab.datasets as datasets
    import makermodslab.record as record

    _make_dataset_dir(tmp_lerobot_home, "tester/quit_probed", total_episodes=2)
    with datasets._HUB_STATUS_LOCK:
        datasets._HUB_STATUS_CACHE["tester/quit_probed"] = "local_only"

    assert record._discard_session_dataset("tester/quit_probed", resume=False) is True

    with datasets._HUB_STATUS_LOCK:
        assert "tester/quit_probed" not in datasets._HUB_STATUS_CACHE


def test_handle_stop_recording_discard_arms_flag_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Quit stop (discard=True) on a live session arms discard_requested, sets
    the same stop events a Done stop does, and echoes discard in the response."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False}
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "discard_requested", False)

    result = record.handle_stop_recording(discard=True)

    assert result["success"] is True
    assert result["discard"] is True
    assert record.discard_requested is True
    assert events["stop_recording"] is True
    assert events["exit_early"] is True
    assert "without saving" in result["message"].lower()


def test_handle_stop_recording_discard_ignored_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A discard stop against no active session is refused and never arms the
    discard flag — an idle/mutex miss can't schedule a dataset deletion."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_events", None)
    monkeypatch.setattr(record, "discard_requested", False)

    result = record.handle_stop_recording(discard=True)

    assert result["success"] is False
    assert record.discard_requested is False


def test_handle_pause_recording_sets_flag_during_reset_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pause only takes effect during the reset phase, and records a wall-clock
    pause-start timestamp for the status handler to use."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(record.time, "time", lambda: 12345.0)

    result = record.handle_pause_recording()

    assert result["success"] is True
    assert events["paused"] is True
    assert record.pause_started_at == 12345.0


def test_handle_pause_recording_ignored_outside_recording_or_resetting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pausing is only meaningful while recording (arms it for the next reset
    gap) or resetting (freezes it live) — any other phase (preparing,
    connecting_*, stopping, completed, error) is refused and never arms the
    paused flag."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "preparing")

    result = record.handle_pause_recording()

    assert result["success"] is False
    assert events["paused"] is False


def test_handle_pause_recording_arms_during_recording_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pausing during the recording phase arms a pending pause for the
    upcoming reset gap: the flag is set, but pause_started_at stays None
    since the countdown it will freeze hasn't started yet — the reset-phase
    preamble seeds it once resetting actually begins."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "pause_started_at", None)

    result = record.handle_pause_recording()

    assert result["success"] is True
    assert events["paused"] is True
    assert record.pause_started_at is None


def test_handle_resume_recording_cancels_pause_armed_during_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming during the recording phase cancels a pending arm before the
    reset gap it targets ever begins. No elapsed time is credited since
    pause_started_at was never set for an armed-but-not-yet-active pause."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": True}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)

    result = record.handle_resume_recording()

    assert result["success"] is True
    assert events["paused"] is False
    assert record.paused_accum_seconds == 0.0


def test_handle_pause_recording_idempotent_when_already_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicate pause request (e.g. a double-click) is a safe no-op, not a
    second timestamp overwrite that would lose accounted pause time."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": True}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "pause_started_at", 999.0)

    result = record.handle_pause_recording()

    assert result["success"] is True
    assert record.pause_started_at == 999.0  # unchanged


def test_handle_resume_recording_credits_paused_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resuming clears the paused flag and folds the just-finished pause
    interval into the running total, so status can freeze-then-continue the
    countdown correctly."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": True}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "pause_started_at", 100.0)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record.time, "time", lambda: 106.0)

    result = record.handle_resume_recording()

    assert result["success"] is True
    assert events["paused"] is False
    assert record.pause_started_at is None
    assert record.paused_accum_seconds == 6.0


def test_handle_resume_recording_ignored_when_not_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resuming when nothing is paused is a safe no-op — it must not credit
    bogus paused time."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 3.0)

    result = record.handle_resume_recording()

    assert result["success"] is True
    assert record.paused_accum_seconds == 3.0  # unchanged


def test_handle_pause_recording_refused_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_events", None)

    result = record.handle_pause_recording()

    assert result["success"] is False


def test_recording_status_freezes_elapsed_time_while_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_elapsed_seconds must stop advancing while paused, accounting for
    the in-progress pause (not just previously-completed ones)."""
    import makermodslab.record as record

    monkeypatch.setattr(record.time, "time", lambda: 1000.0)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "phase_start_time", 990.0)  # phase began 10s ago
    monkeypatch.setattr(record, "recording_start_time", 900.0)
    monkeypatch.setattr(record, "current_episode", 2)
    monkeypatch.setattr(record, "saved_episodes", 1)
    monkeypatch.setattr(
        record,
        "recording_config",
        type("Cfg", (), {"dataset_repo_id": "tester/ds", "num_episodes": 2, "reset_time_s": 20})(),
    )
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", 995.0)  # paused for the last 5s
    monkeypatch.setattr(
        record, "recording_events", {"paused": True, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["paused"] is True
    # 10s since phase start, minus the 5s currently paused = 5s.
    assert status["phase_elapsed_seconds"] == 5
    assert status["available_controls"]["pause_recording"] is False
    assert status["available_controls"]["resume_recording"] is True
    assert status["available_controls"]["exit_early"] is False


def test_recording_status_exposes_pause_recording_when_resetting_and_unpaused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "resetting")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": False, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["paused"] is False
    assert status["available_controls"]["pause_recording"] is True
    assert status["available_controls"]["resume_recording"] is False
    assert status["available_controls"]["exit_early"] is True


def test_recording_status_exposes_pause_recording_during_recording_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause is available during the recording phase too (arms it for the
    upcoming reset gap — the episode itself keeps recording uninterrupted),
    not just during resetting. `paused` itself stays False since nothing is
    actively frozen yet."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": False, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["paused"] is False
    assert status["pause_armed"] is False
    assert status["available_controls"]["pause_recording"] is True
    assert status["available_controls"]["resume_recording"] is False


def test_recording_status_never_exposes_pause_outside_recording_or_resetting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause controls must be structurally impossible outside the recording
    and resetting phases (e.g. while preparing/connecting/stopping)."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "preparing")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": False, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["available_controls"]["pause_recording"] is False
    assert status["available_controls"]["resume_recording"] is False


def test_recording_status_does_not_leak_paused_time_into_recording_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the Critical leak: a leftover paused_accum_seconds
    from a finished reset phase must NOT reduce phase_elapsed_seconds once
    current_phase has moved on to "recording" — this is the test that would
    have caught the bug where the subtraction applied unconditionally to
    every phase instead of being gated on current_phase == "resetting"."""
    import makermodslab.record as record

    monkeypatch.setattr(record.time, "time", lambda: 1000.0)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "phase_start_time", 990.0)  # phase began 10s ago
    monkeypatch.setattr(record, "recording_start_time", 900.0)
    monkeypatch.setattr(record, "current_episode", 2)
    monkeypatch.setattr(record, "saved_episodes", 1)
    monkeypatch.setattr(
        record,
        "recording_config",
        type("Cfg", (), {"dataset_repo_id": "tester/ds", "num_episodes": 2, "episode_time_s": 60})(),
    )
    # Leftover from the reset phase that just ended, not yet cleared —
    # exactly the pre-fix world this test guards against.
    monkeypatch.setattr(record, "paused_accum_seconds", 4.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": False, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    # 10s since phase start; the leftover 4s must NOT be subtracted now that
    # current_phase is "recording" (pre-fix this would have read 6).
    assert status["phase_elapsed_seconds"] == 10


def test_recording_status_armed_pause_during_recording_does_not_block_exit_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """paused=True during the recording phase (armed for the next reset gap,
    or a stale flag a reset loop failed to clear) must read as `paused: False`
    — nothing is actively frozen yet/anymore — and must NOT block ending the
    current episode: only an ACTIVELY frozen reset phase disables exit_early.
    resume_recording IS available, though, so the operator can cancel the arm."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": True, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["paused"] is False
    assert status["pause_armed"] is True
    assert status["available_controls"]["exit_early"] is True
    assert status["available_controls"]["pause_recording"] is False
    assert status["available_controls"]["resume_recording"] is True


def test_recording_status_pause_armed_false_after_session_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pause armed on an episode's recording phase is only cleared by the
    reset-phase cleanup in record_with_web_events — which never runs if that
    episode turns out to be the session's last (no reset gap follows) or the
    session is stopped before reaching one. Once the session has actually
    ended (recording_active False), the stale `paused=True` left in the dead
    events dict must not resurface as pause_armed=True in the completed/
    stopped status — the frontend has nothing to resume and no reset gap is
    coming."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "paused_accum_seconds", 0.0)
    monkeypatch.setattr(record, "pause_started_at", None)
    monkeypatch.setattr(
        record, "recording_events", {"paused": True, "exit_early": False, "stop_recording": False}
    )

    status = record.handle_recording_status()

    assert status["pause_armed"] is False


def test_handle_exit_early_refused_while_paused_in_reset_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip-to-next-episode must be refused server-side while the reset phase
    is paused (design decision: skip is disabled while paused) — this closes
    the race at its root rather than relying only on the advisory frontend
    available_controls gate, which is up to 1s stale via polling."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": True}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")

    result = record.handle_exit_early()

    assert result["success"] is False
    assert events["exit_early"] is False


def test_handle_exit_early_allowed_when_resetting_and_not_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: the new pause guard on handle_exit_early must not block
    the ordinary skip-to-next-episode path when nothing is paused."""
    import makermodslab.record as record

    events = {"stop_recording": False, "exit_early": False, "paused": False}
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "current_phase", "resetting")

    result = record.handle_exit_early()

    assert result["success"] is True
    assert events["exit_early"] is True


class _FakePauseTeleop:
    def __init__(self) -> None:
        self.get_action_calls = 0

    def get_action(self) -> dict:
        self.get_action_calls += 1
        return {"shoulder_pan.pos": 1.0}


class _FakePauseRobot:
    def __init__(self) -> None:
        self.send_action_calls = 0

    def get_observation(self) -> dict:
        return {"shoulder_pan.pos": 0.5}

    def send_action(self, action: dict) -> dict:
        self.send_action_calls += 1
        return action


def _identity(pair):
    return pair[0]


def test_reset_loop_freezes_passthrough_while_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """While paused, the loop must never read the leader or send to the
    follower, and must not exit on its own (freeze = indefinite, no
    auto-timeout) — it only exits here because the fake sleep flips
    exit_early after a few ticks, standing in for a real Stop request."""
    from makermodslab import record

    robot = _FakePauseRobot()
    teleop = _FakePauseTeleop()
    events = {"exit_early": False, "paused": True}
    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            events["exit_early"] = True

    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", _fake_sleep, raising=False)

    record._reset_loop_with_pause(
        robot=robot,
        teleop=teleop,
        events=events,
        fps=30,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        control_time_s=10.0,
    )

    assert teleop.get_action_calls == 0
    assert robot.send_action_calls == 0
    assert len(sleep_calls) == 3


def test_reset_loop_drives_passthrough_when_not_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unpaused, the loop behaves like the original record_loop: reads the
    leader and sends to the follower every tick until control_time_s elapses."""
    from makermodslab import record

    robot = _FakePauseRobot()
    teleop = _FakePauseTeleop()
    events = {"exit_early": False, "paused": False}

    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", lambda seconds: None, raising=False)

    record._reset_loop_with_pause(
        robot=robot,
        teleop=teleop,
        events=events,
        fps=30,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        control_time_s=0.02,
    )

    assert teleop.get_action_calls > 0
    assert robot.send_action_calls == teleop.get_action_calls


def test_reset_loop_exit_early_stops_promptly_even_while_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop (which sets exit_early — see handle_stop_recording) must win
    immediately even if the loop is paused when it's requested; it must not
    be deferred until a resume."""
    from makermodslab import record

    robot = _FakePauseRobot()
    teleop = _FakePauseTeleop()
    events = {"exit_early": True, "paused": True}

    def _fail_sleep(seconds: float) -> None:
        raise AssertionError("must not sleep at all — exit_early is already set")

    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", _fail_sleep, raising=False)

    record._reset_loop_with_pause(
        robot=robot,
        teleop=teleop,
        events=events,
        fps=30,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        control_time_s=10.0,
    )

    assert events["exit_early"] is False  # consumed, matching record_loop's own convention
    assert teleop.get_action_calls == 0


def test_worker_quit_discards_fresh_dataset_with_saved_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """End-to-end through the real worker: a fresh session whose user quit
    (discard_requested set) has its whole stamped directory removed in the
    finally block, even with episodes saved, and reports discarded_empty."""
    import makermodslab.record as record

    def _work_then_quit(cfg, events, **kwargs):
        # Create the dataset dir at the stamped repo id the session recorded into,
        # then simulate a completed loop with saved episodes and a user quit.
        repo_id = record.recording_config.dataset_repo_id
        _make_dataset_dir(tmp_lerobot_home, repo_id, total_episodes=2)
        record.current_phase = "completed"
        record.saved_episodes = 2
        record.discard_requested = True  # handle_stop_recording(discard=True) would set this

    try:
        status = _start_session_with_fake_work(monkeypatch, _work_then_quit)

        assert status["session_ended"] is True
        assert status["discarded_empty"] is True
        assert not (tmp_lerobot_home / status["dataset_repo_id"]).exists()
        # saved_episodes still reports what was recorded before the quit, even
        # though the dataset directory itself was deleted — the two are
        # independent facts in the status payload. This pins backend behavior
        # only: the quit/discard exit path doesn't currently pass this count
        # along to the UI.
        assert status["saved_episodes"] == 2
    finally:
        record.discard_requested = False  # don't leak the armed flag into later tests


def test_recording_status_reports_discarded_empty_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the ended session discarded its empty dataset, the status payload
    tells the frontend honestly that nothing was kept."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "last_session_discarded_empty", True)

    status = record.handle_recording_status()

    assert status["session_ended"] is True
    assert status["discarded_empty"] is True


def test_record_start_clears_stale_release_state_from_previous_double_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-session leak regression (mirrors the teleoperation test): a stale
    _release_now from a previous session's double-stop must be cleared under
    the state lock when a new recording claims the active flag — otherwise
    every later release grace is cut short instantly until a server restart."""
    import threading

    import makermodslab.record as record
    import makermodslab.teleoperate as teleop

    stale = threading.Event()
    stale.set()
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "_release_now", stale)
    monkeypatch.setattr(record, "releasing", True)
    # Teleop side idle so the cross-module pending-release check no-ops.
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)

    # Fail fast AFTER the locked reset, before any hardware is touched.
    def _boom(request, cameras=None):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr(record, "create_record_config", _boom)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/dataset",
            single_task="pick",
        )
    )

    # The start fails, but the per-session reset already ran under the lock.
    assert result["success"] is False
    assert not stale.is_set()
    assert record.releasing is False
    assert record.recording_active is False


# ---------------------------------------------------------------------------
# Rest-pose return on session end (mirrors the teleop stop-path integration).
# record_with_web_events captures each follower's pose at session start and, on
# a NORMAL end, drives it back before releasing torque — same helpers as teleop
# (makermodslab.rest_pose, makermodslab.teleoperate._return_followers_to_rest), so the shared
# return logic itself is covered in tests/test_teleoperate.py. These tests pin
# record's own finally-block wiring: normal end returns then releases, a
# double-stop skips the return, an error skips it, the pose is captured per
# follower, and the gripper is excluded.
# ---------------------------------------------------------------------------


class _RecReturnBus:
    """Follower bus double: serves capture_rest_pose and records nothing else
    (the return itself is spied via _return_followers_to_rest)."""

    _MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, positions: dict[str, int] | None = None, port: str = "COM_FOLLOWER") -> None:
        self.port = port
        self.motors = dict.fromkeys(self._MOTORS)
        self.positions = dict.fromkeys(self._MOTORS, 1000) if positions is None else dict(positions)

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        assert reg == "Present_Position" and normalize is False
        return dict(self.positions)


class _RecRobot:
    """Follower robot double exposing one .bus for _device_buses/capture."""

    def __init__(self, bus: _RecReturnBus) -> None:
        self.bus = bus
        self.disconnected = False

    def disconnect(self) -> None:
        self.disconnected = True


def _run_record_session(
    monkeypatch: pytest.MonkeyPatch,
    robot: _RecRobot,
    *,
    stop_events: dict | None = None,
    raise_in_loop: bool = False,
    preset_release_now: bool = False,
    repo_id: str = "tester/ds",
    num_episodes: int = 1,
    reset_time_s: int = 10,
    record_loop_side_effect=None,
):
    """Drive record_with_web_events with every lerobot dependency mocked so no
    real hardware, dataset, or record_loop runs. Returns the spy call log for
    _return_followers_to_rest (the rest-pose return) and the robot.

    The loop runs a single episode: record_loop sets `stop_recording` (via the
    supplied events) so the session ends normally after one save, unless
    `raise_in_loop` makes record_loop raise (the error path)."""
    import makermodslab.record as record

    return_calls: list[tuple] = []

    def _spy_return(rest_poses, abort_event):
        return_calls.append((list(rest_poses), abort_event))

    monkeypatch.setattr(record, "_return_followers_to_rest", _spy_return)
    monkeypatch.setattr(record, "force_disable_torque", lambda device, label="": [])
    monkeypatch.setattr(record, "reset_torque_limit", lambda *a, **k: [])
    monkeypatch.setattr(record, "clear_goal_velocity", lambda *a, **k: [])
    monkeypatch.setattr(record, "verify_devices", lambda *a, **k: [])

    if preset_release_now:
        record._release_now.set()
    else:
        record._release_now.clear()

    # lerobot symbols resolved at call time inside record_with_web_events.
    monkeypatch.setattr("lerobot.robots.make_robot_from_config", lambda cfg: robot, raising=False)
    monkeypatch.setattr(
        "lerobot.teleoperators.make_teleoperator_from_config", lambda cfg: None, raising=False
    )
    monkeypatch.setattr(
        "lerobot.processor.make_default_processors", lambda: (None, None, None), raising=False
    )
    monkeypatch.setattr(
        "lerobot.utils.feature_utils.hw_to_dataset_features", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr("lerobot.utils.utils.log_say", lambda *a, **k: None, raising=False)

    def _fake_record_loop(*args, **kwargs):
        if raise_in_loop:
            raise RuntimeError("bus died mid-episode")
        events = kwargs.get("events")
        if events is None or kwargs.get("dataset") is None:
            return
        if record_loop_side_effect is not None:
            record_loop_side_effect(events)
        else:
            events.update(stop_events or {"stop_recording": True, "_exit_early_triggered": True})

    monkeypatch.setattr("lerobot.scripts.lerobot_record.record_loop", _fake_record_loop, raising=False)

    dataset_calls: list[str] = []

    class _FakeDataset:
        num_episodes = 1
        num_frames = 1
        fps = 30
        features = {"action": None}
        meta = type("M", (), {"robot_type": "so101"})()

        @staticmethod
        def create(*args, **kwargs):
            return _FakeDataset()

        def save_episode(self) -> None:
            dataset_calls.append("save_episode")

        def clear_episode_buffer(self) -> None:
            dataset_calls.append("clear_episode_buffer")

    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", _FakeDataset, raising=False)

    # robot.connect(calibrate=False) is called on the double.
    robot.connect = lambda **kwargs: None  # type: ignore[attr-defined]
    robot.name = "so101"  # type: ignore[attr-defined]
    robot.cameras = {}  # type: ignore[attr-defined]
    robot.action_features = {}  # type: ignore[attr-defined]
    robot.observation_features = {}  # type: ignore[attr-defined]
    robot.calibration = {}  # type: ignore[attr-defined]

    cfg = record.create_record_config(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id=repo_id,
            single_task="pick",
            num_episodes=num_episodes,
            reset_time_s=reset_time_s,
            video=False,
        )
    )
    # No teleop device: keep the return path follower-only and simple.
    cfg.teleop = None

    web_events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
        "paused": False,
    }
    error: Exception | None = None
    try:
        record.record_with_web_events(cfg, web_events)
    except Exception as e:  # the error-path test expects this
        error = e
    return return_calls, robot, error, dataset_calls


def test_reset_phase_globals_reset_on_both_call_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """paused_accum_seconds/pause_started_at must be freshly reset at the
    START of every reset phase — both the re-record call site and the normal
    inter-episode call site — not just one of them. Verified by spying on
    _reset_loop_with_pause: the spy snapshots the globals on each call (which
    should always read as freshly reset, since record_with_web_events resets
    them immediately before calling it), then pollutes them, so the SECOND
    snapshot only reads clean if the SECOND call site independently reset
    them rather than inheriting the first call's pollution."""
    import makermodslab.record as record

    calls: list[dict] = []

    def _spy_reset_loop(**kwargs):
        calls.append(
            {
                "paused_accum_seconds": record.paused_accum_seconds,
                "pause_started_at": record.pause_started_at,
            }
        )
        if len(calls) == 1:
            record.paused_accum_seconds = 7.5
            record.pause_started_at = 555.0

    monkeypatch.setattr(record, "_reset_loop_with_pause", _spy_reset_loop)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    call_count = [0]

    def _record_loop_side_effect(events: dict) -> None:
        call_count[0] += 1
        if call_count[0] == 1:
            # Episode 1, first take: trigger a re-record (RE-RECORD call site).
            events.update({"rerecord_episode": True, "_exit_early_triggered": True})
        else:
            # Episode 1's re-take, then episode 2: normal skip-to-next
            # (NORMAL call site after episode 1; session completes after 2).
            events.update({"rerecord_episode": False, "_exit_early_triggered": True})

    robot = _RecRobot(_RecReturnBus())
    _run_record_session(
        monkeypatch,
        robot,
        num_episodes=2,
        record_loop_side_effect=_record_loop_side_effect,
    )

    assert len(calls) == 2
    assert calls[0] == {"paused_accum_seconds": 0.0, "pause_started_at": None}
    # The second call site must have reset the polluted values, not inherited them.
    assert calls[1] == {"paused_accum_seconds": 0.0, "pause_started_at": None}


def test_reset_phase_preamble_preserves_pause_armed_during_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """A pause armed during the recording phase (operator clicks Pause
    mid-episode) must survive into the reset phase that follows: `paused`
    stays True and pause_started_at gets seeded the moment resetting
    actually begins, rather than being unconditionally wiped. Verified by
    spying on _reset_loop_with_pause and arming the pause from within the
    fake recording-phase call, mirroring how handle_pause_recording flips
    the flag mid-episode in production."""
    import makermodslab.record as record

    calls: list[dict] = []

    def _spy_reset_loop(**kwargs):
        calls.append(
            {
                "paused": kwargs["events"].get("paused"),
                "pause_started_at": record.pause_started_at,
            }
        )
        kwargs["events"]["exit_early"] = True

    monkeypatch.setattr(record, "_reset_loop_with_pause", _spy_reset_loop)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    monkeypatch.setattr(record.time, "time", lambda: 42.0)

    def _record_loop_side_effect(events: dict) -> None:
        # Simulate the operator clicking Pause mid-episode — arms it exactly
        # like handle_pause_recording does when current_phase == "recording".
        events["paused"] = True
        events.update({"rerecord_episode": False, "_exit_early_triggered": True})

    robot = _RecRobot(_RecReturnBus())
    _run_record_session(
        monkeypatch,
        robot,
        num_episodes=2,
        record_loop_side_effect=_record_loop_side_effect,
    )

    assert len(calls) == 1  # the spy ends the session after the one reset phase it reaches
    assert calls[0]["paused"] is True
    assert calls[0]["pause_started_at"] == 42.0


def test_record_accepts_bare_repo_id(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A bare dataset name (no HF login → no `user/` namespace) records fine —
    lerobot's sanity_check_dataset_name would crash on it, so we don't call it."""

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
    robot = _RecRobot(bus)

    _, robot, error, _ = _run_record_session(monkeypatch, robot, repo_id="bare_local_name")

    assert error is None


def test_record_refuses_eval_prefixed_repo_id(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """eval_ names are reserved for policy-evaluation recordings (rollout flow),
    with or without a namespace."""

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    for repo_id in ("eval_ds", "tester/eval_ds"):
        bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
        robot = _RecRobot(bus)
        _, robot, error, _ = _run_record_session(monkeypatch, robot, repo_id=repo_id)
        assert isinstance(error, ValueError), repo_id
        assert "eval_" in str(error)


def test_record_normal_end_returns_then_releases(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A normal session end drives the follower back to its captured start pose
    (once), then disconnects — same as teleop's stop, no timed hold."""
    import makermodslab.record as record

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
    robot = _RecRobot(bus)

    return_calls, robot, error, _dataset_calls = _run_record_session(monkeypatch, robot)

    assert error is None
    assert len(return_calls) == 1  # the return ran exactly once
    assert robot.disconnected is True
    assert record.releasing is False  # reset in the finally


def test_record_captures_pose_per_follower_excluding_gripper(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """The captured pose is the follower's raw ticks with the gripper removed
    (it may be holding an object at stop time)."""

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    positions = {
        "shoulder_pan": 1111,
        "shoulder_lift": 2222,
        "elbow_flex": 3333,
        "wrist_flex": 4444,
        "wrist_roll": 5555,
        "gripper": 9999,
    }
    robot = _RecRobot(_RecReturnBus(positions=positions))

    return_calls, _robot, error, _dataset_calls = _run_record_session(monkeypatch, robot)

    assert error is None
    (rest_poses, _abort) = return_calls[0]
    assert len(rest_poses) == 1  # one follower bus
    captured_bus, captured_pose = rest_poses[0]
    assert captured_bus is robot.bus
    assert "gripper" not in captured_pose  # excluded — may be holding an object
    assert captured_pose == {k: v for k, v in positions.items() if k != "gripper"}


def test_record_double_stop_skips_the_return(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A second stop (release-now) set before the session-end cleanup runs must
    skip the return and release immediately, mirroring teleop."""

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, _dataset_calls = _run_record_session(
        monkeypatch, robot, preset_release_now=True
    )

    assert error is None
    assert return_calls == []  # release-now skipped the return
    assert robot.disconnected is True


def test_record_error_path_skips_return_and_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """An exception in the loop (dead bus) skips the return entirely — the bus
    may be gone, so release ASAP — but still disconnects."""

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, _dataset_calls = _run_record_session(monkeypatch, robot, raise_in_loop=True)

    assert isinstance(error, RuntimeError)
    assert return_calls == []  # error path never returns to rest
    assert robot.disconnected is True


def test_stop_during_recording_phase_discards_episode_no_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """Stop pressed mid-episode (stop_recording set, but NOT _exit_early_triggered
    — exactly what handle_stop_recording produces) must discard the in-progress
    episode (clear_episode_buffer, never save_episode) and end the session
    immediately, with no reset detour, then return to rest and disconnect once."""
    import makermodslab.record as record

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    robot = _RecRobot(_RecReturnBus())

    # No _exit_early_triggered: the pre-fix classification would have called this
    # a timeout, flipped rerecord on, and run a reset phase before honoring stop.
    return_calls, robot, error, dataset_calls = _run_record_session(
        monkeypatch, robot, stop_events={"stop_recording": True}
    )

    assert error is None
    assert dataset_calls == ["clear_episode_buffer"]  # discarded, never saved
    assert record.current_phase == "completed"  # not "resetting" — no reset detour
    assert len(return_calls) == 1  # rest-pose return ran once
    assert robot.disconnected is True


def test_stop_wins_over_skip_when_both_set_in_same_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """When stop_recording AND _exit_early_triggered land in the same episode,
    stop wins: the short-circuit is checked FIRST, so the episode is discarded,
    not saved. (Stop is a deliberate 'end now, drop this take' action.)"""
    import makermodslab.record as record

    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, dataset_calls = _run_record_session(
        monkeypatch, robot, stop_events={"stop_recording": True, "_exit_early_triggered": True}
    )

    assert error is None
    assert dataset_calls == ["clear_episode_buffer"]  # stop precedence: discard, not save
    assert record.current_phase == "completed"


# ---------------------------------------------------------------------------
# UploadManager — background dataset upload (start → running → done | error).
# The push runs in a worker thread; tests mock LeRobotDataset so no real Hub
# call happens, then join the thread before asserting on the final state.
# ---------------------------------------------------------------------------


def _fake_dataset(num_episodes: int = 3, push=None):
    from unittest.mock import MagicMock

    ds = MagicMock(name="LeRobotDataset")
    ds.num_episodes = num_episodes
    if push is not None:
        ds.push_to_hub = push
    return ds


def _join_upload(mgr, timeout: float = 5.0) -> None:
    thread = mgr._thread
    if thread is not None:
        thread.join(timeout=timeout)


def test_upload_manager_start_runs_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start pushes in a worker thread and lands in state "done" with the
    dataset_url, invalidating the cached hub status."""
    from makermodslab.record import UploadManager, UploadRequest

    ds = _fake_dataset()
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    invalidated: list[str] = []
    monkeypatch.setattr("makermodslab.record.invalidate_hub_status", invalidated.append)

    mgr = UploadManager()
    # Nothing else is writing this dataset — _dataset_in_use must return None.
    monkeypatch.setattr("makermodslab.datasets._dataset_in_use", lambda repo_id: None)

    result = mgr.start(UploadRequest(dataset_repo_id="tester/ds", tags=["x"], private=True))
    assert result == {"started": True, "repo_id": "tester/ds", "message": "Upload started"}

    _join_upload(mgr)
    status = mgr.get_status()
    assert status["state"] == "done"
    assert status["repo_id"] == "tester/ds"
    assert status["dataset_url"] == "https://huggingface.co/datasets/tester/ds"
    assert invalidated == ["tester/ds"]
    # push_to_hub got the MakerModsLab-tagged tags + private flag.
    ds.push_to_hub.assert_called_once()
    kwargs = ds.push_to_hub.call_args.kwargs
    assert kwargs["private"] is True
    assert "x" in kwargs["tags"]


def test_upload_manager_error_maps_auth_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 during push lands in state "error" with the friendly login message
    and the docs_url, not a raw traceback string."""
    from makermodslab.record import UploadManager, UploadRequest

    def _raise_401(**kwargs):
        raise RuntimeError("401 Client Error: you must be authenticated")

    ds = _fake_dataset(push=_raise_401)
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    monkeypatch.setattr("makermodslab.datasets._dataset_in_use", lambda repo_id: None)

    mgr = UploadManager()
    mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    _join_upload(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "hf auth login" in status["message"]
    assert status["docs_url"].startswith("https://huggingface.co/docs")


def test_upload_manager_error_generic_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-auth failure surfaces its message without a docs_url."""
    from makermodslab.record import UploadManager, UploadRequest

    def _boom(**kwargs):
        raise RuntimeError("disk exploded")

    ds = _fake_dataset(push=_boom)
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    monkeypatch.setattr("makermodslab.datasets._dataset_in_use", lambda repo_id: None)

    mgr = UploadManager()
    mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    _join_upload(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "disk exploded" in status["message"]
    assert "docs_url" not in status


def test_upload_manager_rejects_concurrent_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start while one is running is refused (409-mapped by the route),
    naming the repo already uploading; the running upload is untouched."""
    from makermodslab.record import UploadManager, UploadRequest

    mgr = UploadManager()
    monkeypatch.setattr("makermodslab.datasets._dataset_in_use", lambda repo_id: None)
    # Pretend an upload is already running for another repo (don't spawn one).
    mgr.state = "running"
    mgr.repo_id = "tester/first"

    result = mgr.start(UploadRequest(dataset_repo_id="tester/second"))
    assert result["started"] is False
    assert "already running" in result["message"]
    assert "tester/first" in result["message"]
    # State unchanged — the second start didn't clobber the running upload.
    assert mgr.repo_id == "tester/first"


def test_upload_manager_refuses_busy_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start is refused when the dataset is being written by another op —
    _dataset_in_use returns a reason, and no worker thread is spawned."""
    from makermodslab.record import UploadManager, UploadRequest

    monkeypatch.setattr(
        "makermodslab.datasets._dataset_in_use",
        lambda repo_id: "A recording session is writing to this dataset. Stop it before renaming.",
    )
    mgr = UploadManager()
    result = mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    assert result["started"] is False
    assert "recording session" in result["message"]
    assert mgr.state == "idle"
    assert mgr._thread is None


def test_upload_status_idle_shape() -> None:
    from makermodslab.record import UploadManager

    status = UploadManager().get_status()
    assert status["state"] == "idle"
    assert status["repo_id"] is None
    assert status["dataset_url"] is None
    assert "docs_url" not in status


def test_delete_dataset_refused_mid_upload(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset that's being pushed to the Hub is refused, and the
    directory is left on disk."""
    import json

    import makermodslab.record as record
    from makermodslab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/uploading"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 2}))

    monkeypatch.setattr(record.upload_manager, "state", "running")
    monkeypatch.setattr(record.upload_manager, "repo_id", repo_id)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "uploaded" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_dataset_refused_mid_recording(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset an active recording session is writing is refused —
    the delete guard now runs the full _dataset_in_use check, not just the
    upload one."""
    import json
    from unittest.mock import MagicMock

    import makermodslab.record as record
    from makermodslab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/recording_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    cfg = MagicMock()
    cfg.dataset_repo_id = repo_id
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", cfg)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "recording" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_dataset_refused_mid_merge(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting the output dataset of a running merge is refused."""
    import json

    from makermodslab import merge
    from makermodslab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/merging_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    monkeypatch.setattr(merge.merge_manager, "state", "running")
    monkeypatch.setattr(merge.merge_manager, "output_repo_id", repo_id)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "merge" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_dataset_refused_mid_local_training(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset a running local training job reads is refused."""
    import json
    from unittest.mock import MagicMock

    from makermodslab import jobs
    from makermodslab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/training_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    job = MagicMock()
    job.state = "running"
    job.runner = "local"
    job.config.dataset_repo_id = repo_id
    monkeypatch.setattr(jobs.job_registry, "list", lambda limit=200: [job])

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "training" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_refusal_wording_is_action_neutral(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared in-use guard's refusals are action-neutral ("Stop it first."),
    not rename-specific — they now surface from delete too."""
    import json
    from unittest.mock import MagicMock

    import makermodslab.record as record
    from makermodslab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/neutral_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    cfg = MagicMock()
    cfg.dataset_repo_id = repo_id
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", cfg)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "renaming" not in result["message"]
    assert result["message"].endswith("Stop it first.")


def _stub_recording_request(**overrides):
    """Minimal RecordingRequest for exercising handle_start_recording's
    precondition checks — none of these reach real hardware. Pass keyword
    overrides to vary a single field (e.g. an invalid dataset_repo_id)."""
    import makermodslab.record as record

    fields = {
        "leader_port": "/dev/leader",
        "follower_port": "/dev/follower",
        "leader_config": "leader",
        "follower_config": "follower",
        "dataset_repo_id": "tester/ds",
        "single_task": "pick",
    }
    fields.update(overrides)
    return record.RecordingRequest(**fields)


def test_start_recording_blocked_when_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording must refuse to start while manual calibration owns the same
    serial bus, rather than opening a second connection on a live port."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)

    result = record.handle_start_recording(_stub_recording_request())
    assert result == {
        "success": False,
        "status_code": 409,
        "message": "Calibration is currently active. Stop it first.",
    }


def test_start_recording_blocked_when_auto_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)

    result = record.handle_start_recording(_stub_recording_request())
    assert result == {
        "success": False,
        "status_code": 409,
        "message": "Auto-calibration is currently active. Stop it first.",
    }


def test_start_recording_blocked_when_wiggle_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)

    result = record.handle_start_recording(_stub_recording_request())
    assert result == {
        "success": False,
        "status_code": 409,
        "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
    }


def test_start_recording_resume_skips_timestamp_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume must append to the EXISTING directory: the repo_id is used
    verbatim (no '_<timestamp>' suffix), unlike a fresh session which stamps
    one. Regression-guards the `if not request.resume` skip."""
    import re

    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)

    # Fail fast AFTER the stamp point (create_record_config runs right after),
    # before any hardware is touched.
    def _boom(request, cameras=None):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr(record, "create_record_config", _boom)

    def _start(resume: bool):
        req = record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/existing_ds",
            single_task="pick",
            resume=resume,
        )
        result = record.handle_start_recording(req)
        assert result["success"] is False  # the _boom stub stopped the start
        return req.dataset_repo_id

    # Resume: the id is untouched. Fresh: a "_YYYYMMDD_HHMMSS" stamp lands.
    assert _start(resume=True) == "tester/existing_ds"
    monkeypatch.setattr(record, "recording_active", False)  # release the claim
    assert re.fullmatch(r"tester/existing_ds_\d{8}_\d{6}", _start(resume=False))


# ---------------------------------------------------------------------------
# R2 regression: every rejection branch inside handle_start_recording's
# `with _state_lock:` block must carry a "status_code" the route layer can
# turn into a real HTTPException (409 conflict / 400 bad request), instead of
# a plain dict that FastAPI would default to HTTP 200 for. See
# tests/test_server.py for the route-level (actual HTTP status) coverage.
# ---------------------------------------------------------------------------


def test_start_recording_rejects_with_409_when_recording_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    result = record.handle_start_recording(_stub_recording_request())

    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"]


def test_start_recording_rejects_with_409_while_previous_session_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "releasing" variant of the recording_active conflict is still a 409
    (a client should retry shortly), not a 400."""
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "releasing", True)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    result = record.handle_start_recording(_stub_recording_request())

    assert result["success"] is False
    assert result["status_code"] == 409
    assert "still releasing" in result["message"]


def test_start_recording_rejects_with_409_when_teleoperation_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(rollout, "inference_active", False)

    result = record.handle_start_recording(_stub_recording_request())

    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation is currently active" in result["message"]


def test_start_recording_rejects_with_409_when_inference_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", True)

    result = record.handle_start_recording(_stub_recording_request())

    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Inference is currently active" in result["message"]


def test_start_recording_rejects_with_400_for_invalid_dataset_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    result = record.handle_start_recording(_stub_recording_request(dataset_repo_id="too/many/slashes"))

    assert result["success"] is False
    assert result["status_code"] == 400
    assert "'/'" in result["message"]


# ---------------------------------------------------------------------------
# Session cameras come from the ROBOT RECORD named by the request, never from
# the request itself. The resolution helpers are covered in
# tests/test_utils_config.py; these are the start-path rejection branches.
# ---------------------------------------------------------------------------


def _idle_mutexes(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)


def test_start_recording_rejects_with_400_when_the_robot_record_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """A named robot with no record on disk must 400 — recording camera-less
    because the name was wrong throws away a whole session's video silently."""
    import makermodslab.record as record
    from makermodslab.utils import config as cfg

    _idle_mutexes(monkeypatch)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(tmp_lerobot_home / "robots"))

    result = record.handle_start_recording(_stub_recording_request(robot_name="ghost"))

    assert result["success"] is False
    assert result["status_code"] == 400
    assert "ghost" in result["message"]
    # The rejection happens BEFORE the active flag is claimed.
    assert record.recording_active is False


def test_start_recording_rejects_with_400_on_duplicate_camera_names(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """Keying the session cameras by name would drop one of a duplicate pair —
    a camera missing from every episode. Refuse the start instead."""
    import makermodslab.record as record
    from makermodslab.utils import config as cfg

    _idle_mutexes(monkeypatch)
    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    cfg.save_robot_record(
        "twins",
        {
            "cameras": [
                {"name": "wrist", "type": "opencv", "camera_index": 0, "width": 640, "height": 480},
                {"name": "wrist", "type": "opencv", "camera_index": 1, "width": 640, "height": 480},
            ]
        },
        allow_create=True,
    )

    result = record.handle_start_recording(_stub_recording_request(robot_name="twins"))

    assert result["success"] is False
    assert result["status_code"] == 400
    assert "wrist" in result["message"]
    assert record.recording_active is False


def test_start_recording_ignores_a_stale_cameras_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """An older frontend still posts `cameras`. The field is gone from the
    model, so pydantic drops it — the request must not carry it forward."""
    import makermodslab.record as record

    request = record.RecordingRequest(
        leader_port="/dev/leader",
        follower_port="/dev/follower",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="tester/ds",
        single_task="pick",
        cameras={"wrist": {"type": "opencv", "camera_index": 0}},
    )

    assert not hasattr(request, "cameras")


def test_create_record_config_resolves_cameras_from_the_robot_record(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """Called without an explicit set, the config builder reads the record —
    the request has no camera payload to fall back on."""
    import makermodslab.record as record
    from makermodslab.utils import config as cfg

    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    cfg.save_robot_record(
        "lab1",
        {
            "cameras": [
                {
                    "id": "camera_1",
                    "name": "wrist",
                    "type": "opencv",
                    "camera_index": 0,
                    "device_id": "browser-id",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                }
            ]
        },
        allow_create=True,
    )
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    config = record.create_record_config(
        _stub_recording_request(robot_name="lab1", dataset_repo_id="tester/ds")
    )

    assert list(config.robot.cameras) == ["wrist"]
    assert config.robot.cameras["wrist"].width == 640


# ---------------------------------------------------------------------------
# Session error taxonomy — outcome / error / hint (in-process twin of the
# rollout exited payload). The worker's catch site holds the actual exception,
# so the error text is formatted from the object (no log forensics); the
# outcome is classified by catch-site phase: an exception AFTER the recording
# loop finished (phase already "completed" — episodes saved) is only noisy
# teardown, a warning; any earlier phase is a real failure.
# ---------------------------------------------------------------------------


def test_classify_outcome_three_ways() -> None:
    """The pure classifier behind both record and teleop catch sites."""
    from makermodslab.utils.errors import classify_outcome

    # No error: the session was fine, wherever it stood.
    assert classify_outcome(work_completed=True, error_text=None) == "ok"
    assert classify_outcome(work_completed=False, error_text=None) == "ok"
    # The saved-episodes-then-teardown-overload case: the loop finished its
    # real work, then disabling torque on a loaded gripper raised. Data is
    # safe — a warning, NOT a failed session.
    assert (
        classify_outcome(True, "RuntimeError: Overload detected on gripper (torque_enable failed)")
        == "ran_with_warning"
    )
    # The mid-episode-failure case: same-looking error text, but the work was
    # cut short — catch-site phase (not text markers) decides: failed.
    assert classify_outcome(False, "RuntimeError: Overload detected on gripper") == "failed"
    assert classify_outcome(False, "ConnectionError: could not connect to the arm") == "failed"


def test_format_exception_type_message_and_truncation() -> None:
    from makermodslab.utils.errors import format_exception

    assert format_exception(RuntimeError("boom")) == "RuntimeError: boom"
    out = format_exception(RuntimeError("x" * 2000))
    assert out.startswith("RuntimeError: ")
    assert out.endswith("…")
    assert len(out) <= 501  # 500-char cap + ellipsis


def test_recording_status_carries_outcome_error_hint_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session-end status payload exposes the taxonomy fields, with the
    hint derived from the error text via friendly_hint."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "last_session_outcome", "ran_with_warning")
    monkeypatch.setattr(
        record,
        "last_session_error",
        "RuntimeError: Overload detected on gripper (torque_enable failed)",
    )

    status = record.handle_recording_status()

    assert status["session_ended"] is True
    assert status["outcome"] == "ran_with_warning"
    assert "Overload" in status["error"]
    assert "motor overloaded" in status["hint"].lower()


def test_recording_status_omits_outcome_fields_while_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The taxonomy describes an ENDED session only — a live session's status
    carries none of the three fields (mirrors discarded_empty)."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")

    status = record.handle_recording_status()

    assert status["session_ended"] is False
    for key in ("outcome", "error", "hint"):
        assert key not in status


def _start_session_with_fake_work(monkeypatch: pytest.MonkeyPatch, fake_work):
    """Drive handle_start_recording with record_with_web_events replaced by
    `fake_work`, so the REAL worker thread runs the real catch site. Returns
    after joining the worker. All feature mutexes idle; no hardware touched."""
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "current_phase", "preparing")
    monkeypatch.setattr(record, "last_session_outcome", None)
    monkeypatch.setattr(record, "last_session_error", None)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(record, "create_record_config", lambda request, cameras=None: None)
    monkeypatch.setattr(record, "record_with_web_events", fake_work)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/taxonomy_ds",
            single_task="pick",
        )
    )
    assert result["success"] is True
    record.recording_thread.join(timeout=5.0)
    assert not record.recording_thread.is_alive()
    return record.handle_recording_status()


def test_worker_classifies_teardown_failure_as_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """THE headline case: a session whose recording loop finished (episodes
    saved, phase "completed") but whose teardown overloaded the gripper must
    end ran_with_warning with phase "completed" — NOT a failed session."""
    import makermodslab.record as record

    def _work_then_teardown_boom(cfg, events, **kwargs):
        # The loop finished its real work before cleanup raised.
        record.current_phase = "completed"
        record.saved_episodes = 3
        raise RuntimeError("Overload detected on gripper while disabling torque (torque_enable)")

    status = _start_session_with_fake_work(monkeypatch, _work_then_teardown_boom)

    assert status["session_ended"] is True
    assert status["current_phase"] == "completed"  # not "error"
    assert status["outcome"] == "ran_with_warning"
    assert "Overload" in status["error"]
    assert "motor overloaded" in status["hint"].lower()


def test_worker_classifies_midsession_failure_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """An exception mid-episode (loop still in "recording") is a real failure:
    phase "error", outcome "failed", with the camera hint mapped."""

    import makermodslab.record as record

    def _boom_mid_episode(cfg, events, **kwargs):
        record.current_phase = "recording"
        raise RuntimeError("Camera cam0: frame is too old (age 3.2s)")

    status = _start_session_with_fake_work(monkeypatch, _boom_mid_episode)

    assert status["session_ended"] is True
    assert status["current_phase"] == "error"
    assert status["outcome"] == "failed"
    assert "frame is too old" in status["error"]
    assert "camera" in status["hint"].lower()


def test_worker_reports_ok_outcome_on_clean_end(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A session that ends without raising reports outcome "ok" (no error, no
    hint) so the frontend's normal navigate-to-upload path is untouched."""
    import makermodslab.record as record

    class _FakeDataset:
        num_episodes = 2

    def _clean_work(cfg, events, **kwargs):
        record.current_phase = "completed"
        record.saved_episodes = 2
        return _FakeDataset()

    status = _start_session_with_fake_work(monkeypatch, _clean_work)

    assert status["session_ended"] is True
    assert status["outcome"] == "ok"
    assert status["error"] is None
    assert status["hint"] is None
    # Regression (R5): the worker's finally block used to zero saved_episodes
    # the instant it flipped recording_active False, and handle_recording_status
    # only ever echoed saved_episodes while recording_active was True — so the
    # terminal status silently reported nothing saved even though 2 episodes
    # were recorded.
    assert status["saved_episodes"] == 2


# ---------------------------------------------------------------------------
# I8: shutdown_event() has no UI to poll and no "press Stop again" gesture
# available, but handle_stop_recording()'s first call is deliberately
# fire-and-forget -- it only sets the stop_recording/exit_early events and
# returns immediately, relying on the recording worker to notice them, finish
# the session, then return to rest and release. Calling it alone from
# shutdown would let the process exit while the worker is still mid-session,
# with no return-to-rest and no torque release. stop_and_wait() composes the
# stop with a bounded join so a caller with no UI can get the same
# graceful-then-forced guarantee synchronously.
# ---------------------------------------------------------------------------


def test_stop_and_wait_is_a_noop_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)

    record.stop_and_wait(timeout=1.0)  # must return promptly, no exception


def test_stop_and_wait_blocks_until_worker_finishes_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real worker that responds to the stop_recording event (the normal
    graceful path: finish the session, return-to-rest, then release) must be
    allowed to finish on its own within the timeout -- stop_and_wait must not
    return before that, and must not force an early release when it didn't
    need to."""
    import makermodslab.record as record

    released = threading.Event()
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}

    def _worker() -> None:
        while not events["stop_recording"]:
            time.sleep(0.01)
        record.recording_active = False
        released.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "releasing", False)
    record._release_now.clear()
    worker.start()

    record.stop_and_wait(timeout=2.0)

    assert released.is_set(), "stop_and_wait returned before the worker finished releasing"
    assert record.recording_active is False
    assert not record._release_now.is_set(), "a worker that finished gracefully must not be force-released"
    worker.join(timeout=2.0)


def test_stop_and_wait_forces_release_if_worker_does_not_finish_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker stuck past the graceful-release window must be force-released
    via the same "second stop" mechanism the UI's second Stop press uses --
    shutdown has no operator to press it, so stop_and_wait must do it
    automatically once the bound elapses."""
    import makermodslab.record as record

    def _worker() -> None:
        # Ignores the stop_recording event; only responds to a forced
        # release, standing in for a stalled/wedged end-of-session cleanup.
        record._release_now.wait(timeout=5.0)

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "recording_events", {"exit_early": False, "stop_recording": False})
    monkeypatch.setattr(record, "releasing", False)
    record._release_now.clear()
    worker.start()

    record.stop_and_wait(timeout=0.2)

    assert record._release_now.is_set(), "a worker that outlasts the timeout must be force-released"
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_stop_and_wait_lets_an_already_in_progress_release_finish_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recording_active stays True for the worker's entire lifetime -- unlike
    teleoperation_active, which flips False the instant a stop is requested,
    recording_active only goes False after the worker's finally block has
    already released and disconnected. So a session that already received its
    first stop and is mid return-to-rest (releasing=True) still has
    recording_active=True. stop_and_wait must not mistake this for a
    not-yet-stopped session and call handle_stop_recording() again --
    handle_stop_recording treats any call made while releasing is True as a
    *second* stop and force-releases immediately, which would abort a return
    that was about to finish gracefully on its own well within the timeout."""
    import makermodslab.record as record

    forced = threading.Event()
    finished_gracefully = threading.Event()

    def _worker() -> None:
        # Stands in for the tail of a graceful return-to-rest: finishes on its
        # own shortly, but would also honor a forced release if one came in.
        if record._release_now.wait(timeout=0.3):
            forced.set()
        else:
            finished_gracefully.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(
        record, "recording_events", {"exit_early": False, "stop_recording": True, "rerecord_episode": False}
    )
    monkeypatch.setattr(record, "releasing", True)
    record._release_now.clear()
    worker.start()

    record.stop_and_wait(timeout=5.0)

    assert finished_gracefully.is_set(), "an already-in-progress graceful return was aborted early"
    assert not forced.is_set(), "stop_and_wait force-released a session already mid graceful return"
    worker.join(timeout=2.0)


def test_recording_status_reports_saved_episodes_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal status must keep reporting saved_episodes once the session
    has ended — the frontend's exit handoff (RecordingSessionDialog) reads this
    field to tell the user, and the upload flow, how many episodes exist."""
    import makermodslab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "saved_episodes", 5)

    status = record.handle_recording_status()

    assert status["session_ended"] is True
    assert status["saved_episodes"] == 5
