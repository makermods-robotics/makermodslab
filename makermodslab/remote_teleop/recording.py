"""Bounded, secret-rejecting diagnostics for remote teleoperation sessions."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path

RECORDING_VERSION = "makermodslab.remote-recording.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SECRET_FRAGMENTS = (
    "action_key",
    "credential_secret",
    "pairing_token",
    "private_key",
    "tls_key",
    "browser_data",
    "network_address",
)


class RecordingError(RuntimeError):
    """A recording could not be safely created or finalized."""


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _SECRET_FRAGMENTS):
                return True
            if _contains_secret_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


def _canonical_line(value: Mapping[str, object], max_event_bytes: int) -> bytes:
    if _contains_secret_key(value):
        raise RecordingError("recording event contains a forbidden secret field")
    try:
        line = (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise RecordingError("recording event is not bounded JSON") from exc
    if len(line) > max_event_bytes:
        raise RecordingError("recording event exceeds the size bound")
    return line


class BoundedSessionRecorder:
    """One non-blocking producer and one explicitly started writer thread."""

    def __init__(
        self,
        directory: Path,
        session_label: str,
        *,
        max_queue: int = 512,
        max_event_bytes: int = 32_768,
        max_file_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not _IDENTIFIER.fullmatch(session_label):
            raise ValueError("session_label must be a path-free identifier")
        if not 16 <= max_queue <= 8192:
            raise ValueError("max_queue must be in [16,8192]")
        if not 1024 <= max_event_bytes <= 65_536:
            raise ValueError("max_event_bytes must be in [1KiB,64KiB]")
        if not max_event_bytes <= max_file_bytes <= 1024 * 1024 * 1024:
            raise ValueError("max_file_bytes is outside the supported bound")
        self.directory = directory
        self.session_label = session_label
        self.max_event_bytes = max_event_bytes
        self.max_file_bytes = max_file_bytes
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=max_queue)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._filename: str | None = None
        self._bytes_written = 0
        self._events_written = 0
        self._dropped = 0
        self._writer_fault: str | None = None

    def start(self, header: Mapping[str, object]) -> None:
        with self._lock:
            if self._thread is not None:
                raise RecordingError("recording is already started")
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.directory, 0o700)
            if self.directory.is_symlink():
                raise RecordingError("recording directory must not be a symlink")
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            filename = f"{timestamp}-{self.session_label}.jsonl"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.directory / filename, flags, 0o600)
            try:
                first = _canonical_line(
                    {"record_type": "header", "version": RECORDING_VERSION, **dict(header)},
                    self.max_event_bytes,
                )
                os.write(fd, first)
            except Exception:
                os.close(fd)
                raise
            self._fd = fd
            self._filename = filename
            self._bytes_written = len(first)
            self._thread = threading.Thread(
                target=self._run,
                name="remote-teleop-recording",
                daemon=True,
            )
            self._thread.start()

    def __call__(self, event: Mapping[str, object]) -> bool:
        try:
            line = _canonical_line({"record_type": "event", **dict(event)}, self.max_event_bytes)
        except RecordingError:
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            if self._thread is None or self._writer_fault is not None:
                self._dropped += 1
                return False
        try:
            self._queue.put_nowait(line)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def _run(self) -> None:
        while True:
            line = self._queue.get()
            if line is None:
                return
            with self._lock:
                fd = self._fd
                if fd is None:
                    self._dropped += 1
                    continue
                if self._bytes_written + len(line) > self.max_file_bytes:
                    self._dropped += 1
                    continue
            try:
                os.write(fd, line)
            except OSError as exc:
                with self._lock:
                    self._writer_fault = type(exc).__name__
            else:
                with self._lock:
                    self._bytes_written += len(line)
                    self._events_written += 1

    def close(self, terminal_receipt: Mapping[str, object], *, timeout_s: float = 1.0) -> bool:
        """Queue a terminal receipt and close; timeout/fault returns false."""
        if not 0 <= timeout_s <= 5:
            raise ValueError("timeout_s must be in [0,5]")
        try:
            terminal = _canonical_line(
                {
                    "record_type": "terminal",
                    "dropped_before_terminal": self.status()["dropped"],
                    **dict(terminal_receipt),
                },
                self.max_event_bytes,
            )
        except RecordingError:
            terminal = b""
        if terminal:
            try:
                self._queue.put_nowait(terminal)
            except queue.Full:
                with self._lock:
                    self._dropped += 1
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Never block STOP for diagnostics. The daemon writer may finish
            # queued events, while the caller receives an honest false result.
            return False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        complete = thread is None or not thread.is_alive()
        with self._lock:
            fd = self._fd
            self._fd = None
            if fd is not None and complete:
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self._thread = None if complete else thread
            return complete and self._writer_fault is None

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "active": self._thread is not None and self._thread.is_alive(),
                "filename": self._filename,
                "events_written": self._events_written,
                "bytes_written": self._bytes_written,
                "dropped": self._dropped,
                "writer_fault": self._writer_fault,
            }
