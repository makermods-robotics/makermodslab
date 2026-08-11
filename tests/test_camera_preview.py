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


class WedgesMidStreamVideoCapture(FakeVideoCapture):
    """A capture that streams normally and then stops returning from read() —
    the device dying AFTER its first frame (a same-port replug while the tile
    is already live), which BlockingVideoCapture cannot model."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        super().__init__(index, backend)
        self.unblock = threading.Event()
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads > 1:
            self.unblock.wait()
        return super().read()


class SlowFirstReadVideoCapture(FakeVideoCapture):
    """A perfectly healthy camera that is merely slow to hand over its first
    frame. Real USB webcams routinely take 1-3s (warm-up, exposure lock), and
    that is exactly the interval a contending client must be willing to wait
    out rather than declare the device wedged."""

    delay = 0.3

    def __init__(self, index: int, backend: int | None = None) -> None:
        super().__init__(index, backend)
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads == 1:
            time.sleep(self.delay)
        return super().read()


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
    """Shrink the wedge-containment timeouts so tests don't wait seconds.

    Preserves the production ordering — LOCK_ACQUIRE_TIMEOUT strictly above
    both read deadlines — because that ordering is load-bearing: a contention
    timeout at or below the read deadline makes a slow-but-healthy read look
    like a wedge, which these tests would then not be able to tell apart.
    """
    monkeypatch.setattr(camera_preview, "FIRST_FRAME_TIMEOUT", 0.05)
    monkeypatch.setattr(camera_preview, "READ_TIMEOUT", 0.05)
    monkeypatch.setattr(camera_preview, "LOCK_ACQUIRE_TIMEOUT", 0.15)


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
    # Kept as a tombstone, NOT deregistered: this camera is dead until a
    # restart, and letting the next client open a fresh capture on it just
    # wedges (and leaks) another thread. Identity keying is what makes this
    # safe — the tombstone names the device, not an index another camera can
    # later occupy, which is checked in the identity tests.
    assert manager._captures[0] is entry
    with pytest.raises(CameraOpenError):
        manager.open_stream(0)
    assert len(fake_captures) == 1  # no second capture was ever opened
    # Close the wedged generator while the shrunk timeout is still patched, so
    # its _release returns at once here rather than waiting at GC time.
    gen.close()


def test_stop_all_skips_tombstones_instead_of_re_waiting_on_them(
    fake_captures: list[FakeVideoCapture], fast_timeouts: None
) -> None:
    """stop_all runs on every recording/inference Start. A camera already known
    wedged has a leaked capture and a lock nobody will ever release, so waiting
    LOCK_ACQUIRE_TIMEOUT on it again would add that much latency to every Start
    for as long as the process lives."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    entry = manager._captures[0]
    entry.lock.acquire()  # simulate a reader wedged inside cap.read()
    manager.stop_all(timeout=0.01)  # first pass: discovers and tombstones it
    assert entry.wedged

    started = time.monotonic()
    manager.stop_all(timeout=0.01)  # second pass: must not re-wait on the lock

    assert time.monotonic() - started < camera_preview.LOCK_ACQUIRE_TIMEOUT
    assert manager._captures[0] is entry  # still tombstoned
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
    assert manager._captures[0].wedged  # tombstoned, so a retry fails fast
    instances[0].unblock.set()  # let the leaked watchdog thread finish


def test_a_wedge_leaks_one_capture_no_matter_how_often_the_tile_retries(
    monkeypatch: pytest.MonkeyPatch, fast_timeouts: None
) -> None:
    """The preview tile retries a failed stream forever (BackendCameraStream
    backs off to a 12s poll and never stops, because most preview failures are
    transient). If a wedge deregistered its entry, every one of those retries
    would open a fresh capture on the same dead device and strand a fresh
    watchdog thread inside cap.read() — an unbounded leak of threads and OS
    handles for as long as the tile stays open. The tombstone must cap it at
    one capture and one thread, total."""
    instances: list[BlockingVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> BlockingVideoCapture:
        cap = BlockingVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen = manager.open_stream(0)
    with pytest.raises(StopIteration):
        next(gen)
    threads_after_wedge = threading.active_count()

    for _ in range(10):  # the retry loop, compressed
        with pytest.raises(CameraOpenError):
            manager.open_stream(0)

    assert len(instances) == 1  # one capture opened, not eleven
    assert threading.active_count() == threads_after_wedge  # no new watchdogs
    instances[0].unblock.set()


def test_a_wedge_after_the_first_frame_also_ends_the_stream(
    monkeypatch: pytest.MonkeyPatch, fast_timeouts: None
) -> None:
    """Only the FIRST read used to run under the watchdog; every later one was
    a raw cap.read(). A device that died after delivering frames therefore left
    the generator blocked inside cv2 forever, holding the entry lock inside a
    live MJPEG response — and uvicorn's graceful shutdown waits on in-flight
    responses, so a --reload restart never completed. Later reads are bounded
    by READ_TIMEOUT now, so the response ends and shutdown can proceed."""
    instances: list[WedgesMidStreamVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> WedgesMidStreamVideoCapture:
        cap = WedgesMidStreamVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)  # streams normally first

    started = time.monotonic()
    with pytest.raises(StopIteration):  # the response ENDS rather than hanging
        next(gen)

    assert time.monotonic() - started < 1.0
    assert manager._captures[0].wedged
    assert not instances[0].released  # leaked: release mid-read segfaults cv2
    instances[0].unblock.set()


def test_contention_timeout_stays_above_every_read_deadline() -> None:
    """The whole containment rests on one inference: if the entry lock is still
    held after LOCK_ACQUIRE_TIMEOUT, the holder is wedged. That is only true
    while every read is bounded BELOW that timeout. Invert the ordering and a
    contender starts declaring healthy cameras dead — permanently, since a
    wedge is now a tombstone that only a restart clears."""
    assert camera_preview.LOCK_ACQUIRE_TIMEOUT > camera_preview.FIRST_FRAME_TIMEOUT
    assert camera_preview.LOCK_ACQUIRE_TIMEOUT > camera_preview.READ_TIMEOUT


def test_a_slow_first_frame_is_not_mistaken_for_a_wedge_by_a_second_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two tiles on one camera. The first spends 0.3s inside a legitimate first
    read; the second must wait that out and stream, not time out on the lock,
    call the camera wedged and tell the user to restart the app moments before
    the first read succeeds. Uses the production ORDERING (contention timeout
    above the read deadlines), scaled down."""
    monkeypatch.setattr(camera_preview, "FIRST_FRAME_TIMEOUT", 1.0)
    monkeypatch.setattr(camera_preview, "READ_TIMEOUT", 1.0)
    monkeypatch.setattr(camera_preview, "LOCK_ACQUIRE_TIMEOUT", 1.5)
    instances: list[SlowFirstReadVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> SlowFirstReadVideoCapture:
        cap = SlowFirstReadVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen_a = manager.open_stream(0)
    frames: list[bytes] = []
    # Drive A's slow first read from a thread so B genuinely contends with it.
    reader = threading.Thread(target=lambda: frames.append(next(gen_a)), daemon=True)
    reader.start()
    time.sleep(0.05)  # let A get inside cap.read()

    gen_b = manager.open_stream(0)  # must not raise: the hold is legitimate
    assert b"--frame" in next(gen_b)

    reader.join(2.0)
    assert frames and b"--frame" in frames[0]
    assert not manager._captures[0].wedged
    assert len(instances) == 1  # both clients shared the one capture
    gen_b.close()
    gen_a.close()


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
