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
"""Tests for makermodslab.camera_preview — shared MJPEG previews of backend cameras.

Everything runs against a fake cv2.VideoCapture: no real camera is ever opened
(on macOS a real open would pop a permission dialog and stall the run).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import makermodslab.camera_preview as camera_preview
import makermodslab.focus_tune as focus_tune
import makermodslab.record as record
import makermodslab.rollout as rollout
import makermodslab.server as server_mod
import makermodslab.teleoperate as teleoperate
from makermodslab.camera_preview import CameraOpenError, CameraPreviewManager


class FakeVideoCapture:
    """cv2.VideoCapture double: serves synthetic frames, records release()."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        self.index = index
        self.backend = backend
        self.opened = True
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 — cv2's camelCase API
        return self.opened

    def read(self):
        if not self.opened:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True
        self.opened = False


class FailingVideoCapture(FakeVideoCapture):
    """A capture whose device can't be opened (unplugged / held elsewhere)."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        super().__init__(index, backend)
        self.opened = False


@pytest.fixture
def fake_captures(monkeypatch: pytest.MonkeyPatch) -> list[FakeVideoCapture]:
    """Patch cv2.VideoCapture (as seen by makermodslab.camera_preview) with a fake
    factory; returns the list of instances it constructed."""
    instances: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FakeVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    return instances


# ---------------------------------------------------------------------------
# CameraPreviewManager — refcounting, stop_all, generator lifecycle
# ---------------------------------------------------------------------------


def test_two_clients_share_one_capture_last_release_frees_it(
    fake_captures: list[FakeVideoCapture],
) -> None:
    manager = CameraPreviewManager()
    gen_a = manager.open_stream(0)
    gen_b = manager.open_stream(0)

    # Both clients stream frames from ONE underlying device.
    assert b"--frame" in next(gen_a)
    assert b"Content-Type: image/jpeg" in next(gen_b)
    assert len(fake_captures) == 1

    # First client detaching must NOT release the shared capture...
    gen_a.close()
    assert not fake_captures[0].released

    # ...but the last one must, and the registry entry goes with it.
    gen_b.close()
    assert fake_captures[0].released
    assert manager._captures == {}


def test_distinct_indices_get_distinct_captures(fake_captures: list[FakeVideoCapture]) -> None:
    manager = CameraPreviewManager()
    gen_a = manager.open_stream(0)
    gen_b = manager.open_stream(1)
    next(gen_a)
    next(gen_b)
    assert [cap.index for cap in fake_captures] == [0, 1]
    gen_a.close()
    gen_b.close()
    assert all(cap.released for cap in fake_captures)


def test_open_failure_raises_and_leaks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FailingVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    with pytest.raises(CameraOpenError):
        manager.open_stream(3)

    # The failed capture was released and no registry entry was left behind.
    assert instances[0].released
    assert manager._captures == {}


def test_generator_exits_when_device_stops_producing(
    fake_captures: list[FakeVideoCapture],
) -> None:
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    fake_captures[0].opened = False  # camera unplugged mid-stream
    with pytest.raises(StopIteration):
        next(gen)
    assert fake_captures[0].released
    assert manager._captures == {}


def test_stop_all_force_releases_a_stalled_client(fake_captures: list[FakeVideoCapture]) -> None:
    """A generator suspended mid-yield (a stalled/dead client) can't detach on
    its own; stop_all must force-release the device after the brief wait, and
    the generator must exit — not re-grab the camera — when it resumes."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)  # suspended at yield, refcount still held

    manager.stop_all(timeout=0.05)

    assert fake_captures[0].released
    assert manager._captures == {}
    # The lagging client's next pull ends the stream (release is a no-op).
    with pytest.raises(StopIteration):
        next(gen)


def test_stop_all_without_streams_is_a_noop(fake_captures: list[FakeVideoCapture]) -> None:
    manager = CameraPreviewManager()
    manager.stop_all(timeout=0.05)
    assert fake_captures == []


def test_stream_after_stop_all_reopens_the_camera(fake_captures: list[FakeVideoCapture]) -> None:
    """stop_all must not poison the index: a later preview gets a fresh capture."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    manager.stop_all(timeout=0.05)

    gen2 = manager.open_stream(0)
    assert b"--frame" in next(gen2)
    assert len(fake_captures) == 2
    gen2.close()
    assert fake_captures[1].released
    gen.close()


# ---------------------------------------------------------------------------
# GET /camera-preview/{index} — status codes and exclusivity
# ---------------------------------------------------------------------------


def test_camera_preview_409_while_recording(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "recording_active", True)
    response = client.get("/camera-preview/0")
    assert response.status_code == 409
    assert "Recording" in response.json()["detail"]


def test_camera_preview_409_while_focus_tuning(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A focus tune opens the same cv2 indices (and drives their UVC controls),
    so a preview must not re-acquire a device mid-sweep."""
    monkeypatch.setattr(focus_tune, "focus_tune_active", True)
    response = client.get("/camera-preview/0")
    assert response.status_code == 409
    assert "Focus tuning" in response.json()["detail"]


def test_camera_preview_allowed_while_teleoperating(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teleop drives the serial bus and opens no cv2 cameras, so a preview during
    teleop does not contend — it must NOT 409. (The manager is patched to a
    finite stream so the TestClient request completes.)"""

    def finite_stream(index: int):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(teleoperate, "teleoperation_active", True)
    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")


def test_camera_preview_503_when_camera_cannot_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", FailingVideoCapture)
    response = client.get("/camera-preview/9")
    assert response.status_code == 503
    assert "could not be opened" in response.json()["detail"]


def test_camera_preview_streams_multipart_mjpeg(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: 200 + the multipart media type, with the manager patched to
    a FINITE stream so the TestClient request completes (the real generator is
    endless by design; its behavior is covered by the manager tests above)."""

    def finite_stream(index: int):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content


# ---------------------------------------------------------------------------
# Exclusivity wiring — recording/teleop start paths stop the previews
# ---------------------------------------------------------------------------

# Verbatim `uvc-util -d` output from the SO-101 rig's two cameras (kept in sync
# with tests/test_focus_tune.py, which parses it in anger).
UVC_DEVICE_TABLE = """\
------------ -------------- ------------ ------------ ------------------------------------------------
Index        Vend:Prod      LocationID   UVC Version  Device name
------------ -------------- ------------ ------------ ------------------------------------------------
0            0x1e45:0x0209  0x01113000   1.00         USB Camera
1            0x1e45:0x0209  0x01114000   1.00         USB Camera
------------ -------------- ------------ ------------ ------------------------------------------------
"""


def test_start_recording_stops_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_start_recording force-releases the previews before any
    robot/camera construction (create_record_config is made to fail right
    after, so no worker or hardware is ever touched)."""
    calls: list[str] = []
    monkeypatch.setattr(record.camera_preview_manager, "stop_all", lambda: calls.append("stop_all"))
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_thread", None)

    def _boom(request):
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

    assert result["success"] is False
    assert calls == ["stop_all"]
    assert record.recording_active is False


def test_start_inference_stops_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_start_inference force-releases the previews before spawning the
    rollout subprocess, which opens the same cv2 indices. The startup thread is
    stubbed so no hardware is touched."""
    calls: list[str] = []
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: calls.append("stop_all"))
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda request, cancel: None)
    # Neutralise the cheap pre-flight guards so this test isolates the preview
    # release; they run BEFORE it and would otherwise early-return.
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)
    monkeypatch.setattr(rollout, "_arm_count_mismatch", lambda mode, dim: None)

    result = rollout.handle_start_inference(
        rollout.InferenceRequest(
            follower_port="COM_FOLLOWER",
            follower_config="follower",
            policy_ref="tester/policy",
            duration_s=10,
        )
    )

    assert result["success"] is True
    assert calls == ["stop_all"]
    rollout.inference_active = False


def test_start_focus_tune_stops_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_start_focus_tune force-releases the previews before the sweep
    thread opens the same cv2 indices. Everything below the release is stubbed:
    no uvc-util is executed and the real worker never runs, so no camera or USB
    control is touched."""
    calls: list[str] = []
    active_when_released: list[bool] = []

    def _stop_all() -> None:
        calls.append("stop_all")
        active_when_released.append(focus_tune.focus_tune_active)

    monkeypatch.setattr(focus_tune.camera_preview_manager, "stop_all", _stop_all)
    monkeypatch.setattr(focus_tune.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(focus_tune, "find_uvc_util", lambda: "/fake/uvc-util")
    monkeypatch.setattr(focus_tune, "focus_tune_active", False)
    # Rebind (not mutate) the module's session state: handle_start_focus_tune
    # writes into `_status` in place, and an in-place mutation is something
    # monkeypatch cannot undo — it would leak a populated, active status into
    # tests/test_focus_tune.py's idle-shape assertions. The module resolves both
    # globals at call time, so a rebind is picked up and then reverted for free.
    monkeypatch.setattr(focus_tune, "_status", {"active": False, "cameras": [], "error": None})
    monkeypatch.setattr(focus_tune, "_tune_thread", None)
    monkeypatch.setattr(
        focus_tune,
        "_uvc_run",
        lambda *a: UVC_DEVICE_TABLE if a[1] == ["-d"] else "auto-focus\nfocus-abs\n",
    )
    worker_args: list[str] = []
    monkeypatch.setattr(focus_tune, "_tune_worker", worker_args.append)

    result = focus_tune.handle_start_focus_tune(
        [{"camera_index": 0, "name": "front"}],
        [{"index": 0, "name": "USB Camera", "unique_id": "0x11130001e450209"}],
    )

    assert result["success"] is True
    assert calls == ["stop_all"]
    # Released only AFTER the active flag is set, so /camera-preview 409s and no
    # client can re-acquire a device in the gap before the sweep's first open.
    assert active_when_released == [True]
    if focus_tune._tune_thread is not None:
        focus_tune._tune_thread.join(timeout=2)
    assert worker_args == ["/fake/uvc-util"]


def test_teleoperation_does_not_touch_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teleoperation opens NO cv2 cameras in this architecture (robot_factory
    passes cameras=None), so it must leave previews alone — killing them would
    blank the user's tiles for no reason. Guards the /camera-preview endpoint's
    "allowed while teleoperating" contract from the other side."""
    assert not hasattr(teleoperate, "camera_preview_manager")
