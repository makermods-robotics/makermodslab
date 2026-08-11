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
"""Backend MJPEG camera previews for headless deployments.

The frontend's preview tiles normally use getUserMedia, which only sees the
*viewing* machine's cameras. When MakerMods Lab runs on a headless host (e.g. a Jetson)
with the cameras plugged into the server, no browser deviceId ever matches, so
the tiles would show "No camera selected" forever. This module streams the
backend's cv2 cameras as multipart/x-mixed-replace MJPEG (GET
/camera-preview/{index}) so the tiles can fall back to an ``<img>``.
"""

import logging
import platform
import threading
import time

import cv2

logger = logging.getLogger(__name__)

# A preview is a thumbnail, not a recording: ~15 fps at JPEG quality 70 keeps
# per-client bandwidth modest without visible stutter.
TARGET_FPS = 15.0
JPEG_QUALITY = 70

# A cap.read() against a stale AVFoundation device (camera replugged into the
# SAME port after this process started: the uniqueID survives, so identity
# resolution succeeds and isOpened() returns True, but the device is dead)
# blocks FOREVER while holding the per-index lock. Verified live 2026-08-07:
# an unbounded `with entry.lock` then stacks every later preview client behind
# it, wedges stop_all() (recording/inference start), and hangs uvicorn's
# graceful shutdown so --reload restarts never complete. Every path that
# contends with a reader therefore bounds its lock acquisition and treats a
# timeout as "the reader is wedged": leak the capture, log loudly, move on.
# Leaking beats hanging — and the capture must NEVER be released while a read
# may still be in flight, because cv2 segfaults.
#
# That "lock held too long => the reader is wedged" inference is only sound if
# EVERY read is itself bounded, and bounded BELOW the contention timeout, which
# is why the two deadlines below exist and why LOCK_ACQUIRE_TIMEOUT is derived
# from them rather than picked. A contention timeout under the read deadline
# needs only two clients of one camera to brick it: the second waits less time
# than the first is legitimately allowed to spend inside a slow first read
# (1-3s is ordinary for a real USB webcam warming up), times out, and marks a
# perfectly healthy device wedged.
#
# Deadline on the FIRST cap.read() of a stream, so a wedged device becomes a
# visible stream error for the client instead of an eternally blank tile. Kept
# generous: it covers device warm-up and exposure lock.
FIRST_FRAME_TIMEOUT = 5.0
# Deadline on every LATER read. Once frames are flowing a read takes about
# 1/TARGET_FPS (~66ms), so this is ~30x headroom for a hiccup while still
# catching a device that dies mid-stream — the case where the stream generator,
# not stop_all, is what hangs uvicorn's graceful shutdown (it waits on
# in-flight responses, and an MJPEG response blocked inside cap.read() never
# completes).
READ_TIMEOUT = 2.0
# Must stay strictly above the longest legitimate hold (see above); the margin
# absorbs the watchdog join's own overshoot.
LOCK_ACQUIRE_TIMEOUT = max(FIRST_FRAME_TIMEOUT, READ_TIMEOUT) + 1.0

# Same per-platform backend pin as recording (record._platform_backend) and the
# /available-cameras enumeration: CAP_ANY can pick different backends across
# calls on macOS, silently reordering indices, so a preview could show a
# different physical device than the one the recorder will open.
_CV2_BACKEND = {
    "Darwin": cv2.CAP_AVFOUNDATION,
    "Linux": cv2.CAP_V4L2,
    "Windows": cv2.CAP_DSHOW,
}.get(platform.system(), cv2.CAP_ANY)


class CameraOpenError(RuntimeError):
    """The camera at the requested index could not be opened."""


class _SharedCapture:
    """One refcounted cv2.VideoCapture, shared by every client of a camera."""

    def __init__(self, key: str | int, index: int) -> None:
        # Registry key: the camera's AVFoundation uniqueID when identity is
        # available, else the bare index (see CameraPreviewManager).
        self.key = key
        # The index cv2.VideoCapture() was called with. Kept for the open and
        # for log lines about this handle — it is the honest description of
        # what was opened, even after the device set renumbers underneath it.
        self.index = index
        self.cap: cv2.VideoCapture | None = None
        self.refcount = 0
        # Held around every cap.read() AND every cap.release(): cv2 segfaults
        # if a capture is released while another thread is inside read(), so a
        # release must never happen outside this lock.
        self.lock = threading.Lock()
        # Set by stop_all() so every client generator exits promptly instead of
        # grabbing another frame.
        self.stop = threading.Event()
        # Set (never cleared) once a reader is presumed stuck inside cap.read()
        # — a bounded lock acquisition timed out, or the first-frame deadline
        # fired. New clients fail fast instead of queueing behind the wedge.
        self.wedged = False


class CameraPreviewManager:
    """Refcounted, shared MJPEG streaming of the backend's cv2 cameras.

    One cv2.VideoCapture per camera, shared across all connected preview
    clients; the last client detaching releases the device. Recording and
    teleoperation always win: their start paths call :meth:`stop_all`, which
    tells every client generator to exit and force-releases any capture a
    stalled client would otherwise keep holding.

    The registry is keyed by camera *identity* (AVFoundation uniqueID), not by
    cv2 index, because an index is not a stable name for a device: the
    in-process device list is live (camera_identity.pump_avfoundation_runloop),
    so attaching a camera that sorts ahead of another renumbers it while a
    capture bound to the old device is still open. Keyed by int, a client
    asking for the newly-arrived camera at index 0 would be handed the handle
    bound to the camera that *used* to be index 0 — the user would then
    configure and name one camera while watching another's picture, and that
    name lands in a robot record. Where identity is unavailable (non-macOS,
    PyObjC missing, enumeration failure) the key falls back to the index,
    preserving the original sharing behavior; on those platforms the device
    list is not live-refreshed, so indices do not renumber underneath us.

    A device whose read never returns (see LOCK_ACQUIRE_TIMEOUT above) is
    contained, not cured: its capture and lock are deliberately leaked, the
    entry is marked ``wedged`` and KEPT in the registry as a tombstone so new
    clients get a fast CameraOpenError (see :meth:`_mark_wedged` — dropping it
    instead makes the frontend's retry loop leak a thread and a handle per
    attempt), and everything else keeps working. Only a process restart
    recovers the camera.
    Identity keying scopes that wedge to the device that actually wedged: an
    int-keyed wedge poisoned a *slot*, so after a renumber it fast-failed
    whatever camera had since landed there while the genuinely dead device
    answered fresh under its old number.

    What identity keying does NOT fix: a camera replugged into the same port
    keeps its uniqueID, so a dead handle is still handed back for that device.
    That case is the wedge containment's job, not this key's — the key only
    stops two *different* physical cameras from colliding on one entry.
    """

    def __init__(self) -> None:
        self._captures: dict[str | int, _SharedCapture] = {}
        # Guards the registry dict and the refcounts; never held during device
        # I/O (open/read/release happen under the per-camera lock instead).
        self._registry_lock = threading.Lock()

    def open_stream(self, index: int, key: str | int | None = None):
        """Open (or share) a camera's capture and return a frame generator.

        ``index`` is what cv2 opens; ``key`` is the camera's identity, as
        returned by :func:`camera_identity.identify_cv2_index` — the same
        physical device shares one capture across every index it has been
        known by. ``None`` means identity is unavailable, and the index itself
        becomes the key.

        Raises :class:`CameraOpenError` when the device can't be opened. The
        generator yields ``multipart/x-mixed-replace`` JPEG parts (boundary
        ``frame``) at ~TARGET_FPS and drops its reference on exit — client
        disconnect, :meth:`stop_all`, or the device dying mid-stream.
        """
        entry = self._acquire(index, index if key is None else key)
        return self._frames(entry)

    def stop_all(self, timeout: float = 1.0) -> None:
        """Stop every preview stream and release every capture.

        Sets each capture's stop event so client generators exit on their next
        frame, waits up to ``timeout`` seconds for them to detach, then
        force-releases anything still held — a client stalled mid-yield on a
        dead connection must not keep the device away from recording/teleop.
        The force-release happens under the per-camera lock, so it can never
        pull the capture out from under a thread inside cap.read(); if that
        lock can't be acquired within LOCK_ACQUIRE_TIMEOUT (a reader wedged in
        cap.read() against a stale replugged device), the capture is leaked
        instead — this method must never hang, because it sits on the
        recording/inference start paths, where a hang would freeze Start.
        Uvicorn's graceful shutdown is protected separately: nothing calls
        this at shutdown, but shutdown waits on in-flight responses, and the
        bounded reads in :meth:`_frames` are what guarantee a wedged MJPEG
        stream ends instead of holding that wait hostage.

        Cameras already tombstoned as wedged are skipped outright: their
        capture is long leaked and their lock is held by a thread that will
        never come back, so waiting LOCK_ACQUIRE_TIMEOUT on each of them would
        add seconds to every Start to re-learn what is already known.
        """
        with self._registry_lock:
            entries = [entry for entry in self._captures.values() if not entry.wedged]
        if not entries:
            return
        for entry in entries:
            entry.stop.set()
        deadline = time.monotonic() + timeout
        for entry in entries:
            while time.monotonic() < deadline:
                with self._registry_lock:
                    detached = self._captures.get(entry.key) is not entry
                if detached:
                    break
                time.sleep(0.02)
            if entry.lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT):
                try:
                    if entry.cap is not None:
                        logger.warning(
                            "Force-releasing camera %d preview capture (a client is still attached)",
                            entry.index,
                        )
                        entry.cap.release()
                        entry.cap = None
                finally:
                    entry.lock.release()
            else:
                self._mark_wedged(entry)
                logger.error(
                    "Camera %d preview reader is wedged inside cap.read() (stale device after "
                    "a same-port replug?) — leaking its capture instead of hanging. "
                    "Restart MakerMods Lab to recover this camera.",
                    entry.index,
                )
                continue  # tombstoned by _mark_wedged; leave it registered
            # Deregister so a lagging client's _release becomes a no-op and a
            # future preview starts from a fresh entry (fresh stop event).
            with self._registry_lock:
                if self._captures.get(entry.key) is entry:
                    del self._captures[entry.key]

    def _acquire(self, index: int, key: str | int) -> _SharedCapture:
        with self._registry_lock:
            entry = self._captures.get(key)
            if entry is not None and entry.wedged:
                # Fail fast: a reader is already known to be stuck inside
                # cap.read() on THIS camera (identity, not index — a different
                # device that has since taken this index is unaffected);
                # queueing behind it would just hang this client too (HTTP 200,
                # zero bytes forever).
                raise CameraOpenError(
                    f"Camera {index} is not responding (a previous read never returned — was it "
                    "replugged since MakerMods Lab started?). Restart MakerMods Lab to recover it."
                )
            if entry is None:
                entry = _SharedCapture(key, index)
                self._captures[key] = entry
            entry.refcount += 1
        try:
            if not entry.lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT):
                # Sound because LOCK_ACQUIRE_TIMEOUT sits above every read
                # deadline: no healthy reader can still hold this lock.
                self._mark_wedged(entry)
                logger.error(
                    "Camera %d preview lock held for over %.1fs (reader wedged in cap.read()?) — "
                    "refusing new preview client. Restart MakerMods Lab to recover this camera.",
                    index,
                    LOCK_ACQUIRE_TIMEOUT,
                )
                raise CameraOpenError(
                    f"Camera {index} is not responding (a read is stuck — was it replugged since "
                    "MakerMods Lab started?). Restart MakerMods Lab to recover it."
                )
            try:
                if entry.cap is None:
                    cap = cv2.VideoCapture(index, _CV2_BACKEND)
                    if not cap.isOpened():
                        cap.release()
                        raise CameraOpenError(
                            f"Camera {index} could not be opened — it may be unplugged or in use "
                            "by another application."
                        )
                    entry.cap = cap
                    # Keep entry.index describing the handle that actually
                    # exists: this open may be a re-open of an entry created
                    # when the device answered to a different index.
                    entry.index = index
            finally:
                entry.lock.release()
        except Exception:
            self._release(entry)
            raise
        return entry

    def _mark_wedged(self, entry: _SharedCapture) -> None:
        """Tombstone a camera whose reader is stuck inside cap.read().

        The entry deliberately STAYS in the registry: that is what makes
        :meth:`_acquire`'s fast fail reachable at all. Dropping it instead, so
        a retry "starts fresh", sounds harmless and is not — the next client
        builds a new entry, opens a new capture on the same dead device, and
        wedges a new thread. The preview tile retries forever (it backs off to
        a 12s poll but never gives up, deliberately, because most preview
        failures are transient), so that is one more leaked thread and one more
        leaked capture handle every 12s for as long as the tile stays open.
        The tombstone bounds the damage at exactly one of each per physically
        wedged camera.

        Sound only because the registry is keyed by identity (see the class
        docstring): the tombstone names the device that actually died, not an
        index another camera may later occupy. Recovery is a process restart —
        which is what every message on this path already tells the user.
        Recovering in-process would need a re-enumeration signal to tell "the
        user replugged it and the runloop pump saw it come back" from "still
        dead", and the uniqueID reads identically in both cases.
        """
        entry.wedged = True
        with self._registry_lock:
            current = self._captures.get(entry.key)
            # Never overwrite a live entry that has since taken this key: a
            # wedge can surface long after stop_all deregistered this entry and
            # a later client opened a working capture for the same camera.
            if current is None or current is entry:
                self._captures[entry.key] = entry

    def _release(self, entry: _SharedCapture) -> None:
        with self._registry_lock:
            entry.refcount -= 1
            if entry.refcount > 0:
                return
            if entry.wedged:
                # Tombstone: stays registered so the next client fails fast
                # instead of opening (and wedging) another capture. Its lock is
                # held by a thread still inside cap.read(), so there is nothing
                # here to release and nothing worth waiting on.
                return
            # Last client: deregister before the (slow) device release so a new
            # client never latches onto a capture that is being torn down.
            if self._captures.get(entry.key) is entry:
                del self._captures[entry.key]
        if not entry.lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT):
            # A read is (still) in flight somewhere — releasing the capture
            # under it would segfault cv2. Leak it; only a restart recovers.
            self._mark_wedged(entry)
            logger.error(
                "Camera %d preview capture leaked on release: its lock is still held "
                "(reader wedged in cap.read()?). Restart MakerMods Lab to recover this camera.",
                entry.index,
            )
            return
        try:
            if entry.cap is not None:
                entry.cap.release()
                entry.cap = None
                logger.info("Released camera %d preview capture (last client detached)", entry.index)
        finally:
            entry.lock.release()

    def _read_with_deadline(self, entry: _SharedCapture, timeout: float):
        """Run one cap.read() under a watchdog thread, bounded by ``timeout``.

        Returns the ``(ok, frame)`` tuple; ``(False, None)`` when the read
        finished by raising (an ordinary failed read — nothing is in flight,
        so the caller releases normally); or None when the read did not
        complete within ``timeout`` — meaning the helper thread is still
        blocked inside cap.read() and the caller must treat the device as
        wedged (and must NOT release the capture or the entry lock).
        Caller must hold ``entry.lock``.

        Every read goes through here, not only the first. A device dies
        mid-stream as easily as before its first frame, and an unbounded later
        read is the worse of the two: the capture is already streaming, so the
        blocked thread holds the entry lock inside a live MJPEG response that
        can now never complete — which is exactly what keeps uvicorn's
        graceful shutdown (and so a --reload restart) waiting forever.

        The wedge signal is the watchdog thread still being *alive* after the
        join, not an empty result: a read that raises leaves the thread dead
        and the device idle, and mistaking that for a wedge would leak a
        capture and a lock for what is really a fast, recoverable failure.

        A thread per frame costs ~25us to start and join — about 0.04% of one
        core at TARGET_FPS, orders of magnitude below the JPEG encode already
        in this loop, and the only way to bound an uncancellable blocking call
        in CPython.
        """
        result: list = []

        def _read() -> None:
            try:
                result.append(entry.cap.read())
            except Exception as exc:  # noqa: BLE001 — reported via `result`
                result.append(exc)

        watchdog = threading.Thread(target=_read, daemon=True, name=f"camera-{entry.index}-read")
        watchdog.start()
        watchdog.join(timeout)
        if watchdog.is_alive():
            return None
        if not result or isinstance(result[0], BaseException):
            logger.warning(
                "Camera %d first read failed (%s); ending preview",
                entry.index,
                result[0] if result else "read thread died without a result",
            )
            return False, None
        return result[0]

    def _frames(self, entry: _SharedCapture):
        """Yield multipart JPEG parts from a shared capture until stopped."""
        interval = 1.0 / TARGET_FPS
        first_frame = True
        try:
            while not entry.stop.is_set():
                started = time.monotonic()
                if not entry.lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT):
                    # Another reader is wedged in cap.read(); end this stream
                    # so the client sees an error instead of a frozen tile.
                    entry.wedged = True
                    logger.error(
                        "Camera %d preview lock held for over %.1fs (reader wedged in "
                        "cap.read()?); ending this preview stream.",
                        entry.index,
                        LOCK_ACQUIRE_TIMEOUT,
                    )
                    break
                release_lock = True
                try:
                    if entry.cap is None:  # force-released by stop_all
                        break
                    deadline = FIRST_FRAME_TIMEOUT if first_frame else READ_TIMEOUT
                    read_result = self._read_with_deadline(entry, deadline)
                    if read_result is None:
                        # The watchdog thread is still inside cap.read(). Leak
                        # the lock on purpose: releasing it would let someone
                        # release the capture under the in-flight read, which
                        # segfaults cv2.
                        release_lock = False
                        self._mark_wedged(entry)
                        logger.error(
                            "Camera %d produced no %s within %.1fs (stale device after a "
                            "same-port replug?) — ending the preview and leaking the capture. "
                            "Restart MakerMods Lab to recover this camera.",
                            entry.index,
                            "first frame" if first_frame else "further frame",
                            deadline,
                        )
                        break
                    ok, frame = read_result
                    first_frame = False
                finally:
                    if release_lock:
                        entry.lock.release()
                if not ok:
                    logger.warning("Camera %d stopped producing frames; ending preview", entry.index)
                    break
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue
                data = jpeg.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
                # Pace to ~TARGET_FPS. Waiting on the stop event (not a plain
                # sleep) lets stop_all cut the frame interval short.
                remaining = interval - (time.monotonic() - started)
                if remaining > 0 and entry.stop.wait(remaining):
                    break
        finally:
            self._release(entry)


camera_preview_manager = CameraPreviewManager()
