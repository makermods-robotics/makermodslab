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

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import makermodslab.camera_identity as camera_identity
import makermodslab.camera_preview as camera_preview
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


class BlockingVideoCapture(FakeVideoCapture):
    """A capture whose read() blocks until told otherwise — models the stale
    AVFoundation device after a same-port replug: open succeeds (isOpened()
    True) but the first read never returns."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        super().__init__(index, backend)
        self.unblock = threading.Event()

    def read(self):
        self.unblock.wait()
        return super().read()


class RaisingVideoCapture(FakeVideoCapture):
    """A capture whose read() raises. The opposite of BlockingVideoCapture: the
    read FINISHES (badly), so nothing is in flight and the containment must
    treat it as an ordinary failed read, not as a wedge."""

    def read(self):
        raise RuntimeError("cv2 read blew up")


class OneFrameVideoCapture(FakeVideoCapture):
    """A capture that serves exactly one frame, then reports failure — so the
    endpoint's (endless by design) generator terminates and a TestClient
    request can complete against the REAL manager rather than a stub."""

    def read(self):
        if not self.opened:
            return False, None
        self.opened = False
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def no_host_camera_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let identity resolution touch the host's real AVFoundation list.

    The default ("identity unavailable") is what non-macOS hosts see, and it
    keys previews by index exactly as they were keyed before identity existed,
    so the pre-existing tests below are unaffected. Tests that exercise
    identity patch this with a device list of their own.
    """
    monkeypatch.setattr(camera_identity, "list_cameras_in_process", lambda: None)


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
# Wedged reads — a cap.read() that never returns (stale same-port replug)
# must be contained (leaked, loudly), never allowed to hang stop_all or
# stack later clients. See camera_preview.LOCK_ACQUIRE_TIMEOUT.
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the wedge-containment timeouts so tests don't wait seconds."""
    monkeypatch.setattr(camera_preview, "LOCK_ACQUIRE_TIMEOUT", 0.05)
    monkeypatch.setattr(camera_preview, "FIRST_FRAME_TIMEOUT", 0.05)


def test_stop_all_leaks_instead_of_hanging_on_a_wedged_read(
    fake_captures: list[FakeVideoCapture], fast_timeouts: None
) -> None:
    """stop_all sits on the recording/inference start paths, where a hang
    would freeze Start; a reader wedged inside cap.read() (holding the entry
    lock) must make it leak the capture and move on, not block forever."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    entry = manager._captures[0]
    entry.lock.acquire()  # simulate a reader wedged inside cap.read()

    started = time.monotonic()
    manager.stop_all(timeout=0.05)

    assert time.monotonic() - started < 1.0
    # Leaked, not released: releasing under an in-flight read segfaults cv2.
    assert not fake_captures[0].released
    assert entry.wedged
    # Deregistered anyway, so the index is not poisoned for future previews.
    assert manager._captures == {}
    gen2 = manager.open_stream(0)
    assert b"--frame" in next(gen2)
    assert len(fake_captures) == 2
    gen2.close()
    # Close the wedged generator while the shrunk timeout is still patched, so
    # its _release times out in 0.05s here rather than 2s at GC time.
    gen.close()


def test_new_client_errors_fast_instead_of_stacking_behind_a_wedged_reader(
    fake_captures: list[FakeVideoCapture], fast_timeouts: None
) -> None:
    """Before hardening, every later client for a wedged index blocked forever
    on the entry lock (HTTP 200, zero bytes). Now the first one times out and
    every subsequent one short-circuits on the wedged flag."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    entry = manager._captures[0]
    entry.lock.acquire()  # simulate a reader wedged inside cap.read()

    with pytest.raises(CameraOpenError):
        manager.open_stream(0)
    assert entry.wedged
    with pytest.raises(CameraOpenError):  # fast path, no timeout wait
        manager.open_stream(0)
    # The wedged reader's refcount and capture were left alone.
    assert entry.refcount == 1
    assert not fake_captures[0].released
    gen.close()  # _release times out on the held lock and leaks — no hang
    assert not fake_captures[0].released


def test_first_frame_deadline_ends_the_stream_instead_of_blank_forever(
    monkeypatch: pytest.MonkeyPatch, fast_timeouts: None
) -> None:
    """A stale device opens fine but its first read() never returns. The
    first-frame watchdog must end the generator (a visible stream error for
    the client) and leak the capture rather than release it mid-read."""
    instances: list[BlockingVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> BlockingVideoCapture:
        cap = BlockingVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen = manager.open_stream(0)
    with pytest.raises(StopIteration):  # finite failure, not an eternal blank
        next(gen)

    assert not instances[0].released  # leaked: release mid-read segfaults cv2
    assert manager._captures == {}  # torn down; a retry gets a fresh entry
    instances[0].unblock.set()  # let the leaked watchdog thread finish


def test_first_read_that_raises_is_a_fast_failure_not_a_wedge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first read() that RAISES finishes its watchdog thread, so nothing is
    in flight: the stream must end immediately and the capture must be
    released normally. Leaking is reserved for reads that never return —
    treating an exception as a wedge would strand a capture (and its lock)
    forever on an ordinary, recoverable failure.

    Deliberately runs with the real FIRST_FRAME_TIMEOUT (no fast_timeouts): the
    elapsed-time assertion is what proves the failure is reported at once
    rather than after waiting the full first-frame deadline out.
    """
    instances: list[RaisingVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> RaisingVideoCapture:
        cap = RaisingVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen = manager.open_stream(0)
    entry = manager._captures[0]
    started = time.monotonic()
    with pytest.raises(StopIteration):
        next(gen)

    assert time.monotonic() - started < camera_preview.FIRST_FRAME_TIMEOUT
    assert not entry.wedged  # a finished read is not a wedge
    assert instances[0].released  # released, not leaked: no read in flight
    assert manager._captures == {}


# ---------------------------------------------------------------------------
# Identity keying — the registry is keyed by the camera's uniqueID, not by its
# cv2 index, because the in-process device list is live and indices renumber
# mid-session (camera_identity.pump_avfoundation_runloop).
# ---------------------------------------------------------------------------


def test_same_index_different_identity_get_distinct_captures(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """The bug this keying fixes: camera A is attached, sorts ahead of the
    camera already streaming, and resolves to the index that camera is open on.
    Keyed by that int, the second client was handed the FIRST camera's handle —
    it would then be configured and named while showing the other's picture."""
    manager = CameraPreviewManager()
    gen_b = manager.open_stream(0, "uid-B")  # streaming B, opened at index 0
    gen_a = manager.open_stream(0, "uid-A")  # A now resolves to index 0 too
    next(gen_b)
    next(gen_a)

    assert len(fake_captures) == 2
    assert set(manager._captures) == {"uid-A", "uid-B"}
    assert manager._captures["uid-A"].cap is not manager._captures["uid-B"].cap
    gen_a.close()
    gen_b.close()


def test_same_identity_different_index_shares_one_capture(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """The mirror case, which must NOT regress into a second open: B renumbers
    from 0 to 1 while a client still streams it, and the new client for B
    shares the existing device-bound handle instead of opening the device
    twice (a busy camera would refuse the second open)."""
    manager = CameraPreviewManager()
    gen_first = manager.open_stream(0, "uid-B")
    gen_second = manager.open_stream(1, "uid-B")
    next(gen_first)
    next(gen_second)

    assert len(fake_captures) == 1
    assert fake_captures[0].index == 0  # the index the handle was opened at
    assert list(manager._captures) == ["uid-B"]
    assert manager._captures["uid-B"].refcount == 2

    gen_first.close()
    assert not fake_captures[0].released
    gen_second.close()
    assert fake_captures[0].released
    assert manager._captures == {}


def test_identity_unavailable_falls_back_to_index_keying(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """Where the platform can't establish identity (non-macOS, PyObjC missing),
    the index is the key and sharing behaves exactly as it did before."""
    manager = CameraPreviewManager()
    gen_a = manager.open_stream(0, None)
    gen_b = manager.open_stream(0, None)
    next(gen_a)
    next(gen_b)

    assert len(fake_captures) == 1
    assert list(manager._captures) == [0]
    gen_a.close()
    gen_b.close()
    assert fake_captures[0].released


def test_wedge_does_not_fast_fail_a_different_camera_at_the_same_index(
    fake_captures: list[FakeVideoCapture], fast_timeouts: None
) -> None:
    """A wedge belongs to the device that wedged, not to its slot. Keyed by
    int, a wedged entry fast-failed whatever camera later landed on that index
    while the genuinely dead device answered fresh under its new number."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0, "uid-B")
    next(gen)
    entry = manager._captures["uid-B"]
    entry.lock.acquire()  # simulate a reader wedged inside cap.read()

    with pytest.raises(CameraOpenError):
        manager.open_stream(0, "uid-B")
    assert entry.wedged

    # A different physical camera that now sits at index 0 is unaffected.
    gen_other = manager.open_stream(0, "uid-A")
    assert b"--frame" in next(gen_other)
    assert len(fake_captures) == 2
    gen_other.close()
    gen.close()  # _release times out on the held lock and leaks — no hang


def test_camera_preview_endpoint_keys_by_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the endpoint: a client already streaming camera B on
    index 0, then a request for camera A that resolves to index 0 as well, must
    open a SECOND capture — not inherit B's handle."""
    instances: list[OneFrameVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> OneFrameVideoCapture:
        cap = OneFrameVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    monkeypatch.setattr(server_mod, "camera_preview_manager", manager)

    # Browser 1: B is the only camera, so it is index 0.
    monkeypatch.setattr(
        camera_identity,
        "list_cameras_in_process",
        lambda: [{"index": 0, "name": "cam", "unique_id": "uid-B"}],
    )
    gen_b = manager.open_stream(*camera_identity.identify_cv2_index("uid-B", 0))
    next(gen_b)  # attached and holding the capture

    # A is plugged in and sorts ahead of B, taking index 0.
    monkeypatch.setattr(
        camera_identity,
        "list_cameras_in_process",
        lambda: [
            {"index": 0, "name": "cam", "unique_id": "uid-A"},
            {"index": 1, "name": "cam", "unique_id": "uid-B"},
        ],
    )
    response = client.get("/camera-preview/0", params={"unique_id": "uid-A"})

    assert response.status_code == 200
    assert b"--frame" in response.content
    assert len(instances) == 2  # a fresh, A-bound capture, not B's handle
    assert set(manager._captures) == {"uid-B"}  # A's stream finished and freed
    gen_b.close()


def test_camera_preview_409_while_recording(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "recording_active", True)
    response = client.get("/camera-preview/0")
    assert response.status_code == 409
    assert "Recording" in response.json()["detail"]


def test_camera_preview_allowed_while_teleoperating(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teleop drives the serial bus and opens no cv2 cameras, so a preview during
    teleop does not contend — it must NOT 409. (The manager is patched to a
    finite stream so the TestClient request completes.)"""

    def finite_stream(index: int, key: str | int | None = None):
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

    def finite_stream(index: int, key: str | int | None = None):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content


# ---------------------------------------------------------------------------
# Exclusivity wiring — recording/teleop start paths stop the previews
# ---------------------------------------------------------------------------


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


def test_teleoperation_does_not_touch_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teleoperation opens NO cv2 cameras in this architecture (robot_factory
    passes cameras=None), so it must leave previews alone — killing them would
    blank the user's tiles for no reason. Guards the /camera-preview endpoint's
    "allowed while teleoperating" contract from the other side."""
    assert not hasattr(teleoperate, "camera_preview_manager")
