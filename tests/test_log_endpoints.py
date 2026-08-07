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
"""Tests for the read-only log endpoints — inference-log (tail of the rollout's
on-disk log file) and recording-log (tail of the in-memory ring buffer).

Both log SOURCES are mocked: the inference log is a temp file monkeypatched into
the handler's session state, and the recording log is a seeded ring-buffer
handler. No real inference or recording is ever started (that would drive the
arm) — we exercise the pure read paths only."""

from __future__ import annotations

import logging

import pytest

# --- Inference log ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rollout_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear rollout's active-session meta so each case controls the log source."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {})


def test_inference_log_empty_when_no_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No session state → empty text, log_path None, belongs_to None, no raise.

    Log files on disk are deliberately irrelevant here: the handler only serves a
    log this process opened, so a populated inference_logs dir must not change
    the answer (see the removal note below).
    """
    from makermodslab import rollout

    logs_dir = tmp_path / ".cache" / "huggingface" / "lerobot" / "inference_logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "100.log").write_text("some earlier run\n")
    monkeypatch.setattr(rollout.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_last_log_path", None)

    result = rollout.handle_inference_log()
    assert result["logs"] == ""
    assert result["log_path"] is None
    assert result["belongs_to"] is None


def test_inference_log_tails_active_meta_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """With an active meta log_path, the handler returns that file's tail."""
    from makermodslab import rollout

    log_file = tmp_path / "run.log"
    log_file.write_text("line1\nline2\nline3\n")
    # The meta branch serves the log only for a LIVE session — otherwise a
    # lingering meta could get labelled "active" after the run ended, which is the
    # mislabelling the belongs_to contract exists to prevent.
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"log_path": str(log_file)})

    result = rollout.handle_inference_log()
    assert result["log_path"] == str(log_file)
    assert result["logs"] == "line1\nline2\nline3"
    assert result["belongs_to"] == "active"


def test_inference_log_bounded_to_max_lines(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A log longer than max_lines is trimmed to the trailing lines only."""
    from makermodslab import rollout

    log_file = tmp_path / "run.log"
    log_file.write_text("\n".join(f"line{i}" for i in range(1000)) + "\n")
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"log_path": str(log_file)})

    result = rollout.handle_inference_log(max_lines=10)
    lines = result["logs"].splitlines()
    assert len(lines) == 10
    # The LAST ten lines, not the first.
    assert lines[0] == "line990"
    assert lines[-1] == "line999"


# REMOVED: test_inference_log_falls_back_to_newest_file.
#
# It pinned the "no active meta -> tail the newest *.log in inference_logs"
# fallback, which is now the DEFECT rather than the contract. A log file on disk
# carries no evidence of which run wrote it, so during a new session's pre-spawn
# phases — and after a run that failed before spawning — that glob served an
# earlier run's output as though it were the current one. Observed live: a run
# that failed on a calibration error produced no log at all, and the user was
# shown a three-day-old RTC run's output and concluded their sync run was
# executing RTC code.
#
# The replacement guarantees live in tests/test_rollout.py: the endpoint serves
# only logs THIS process opened, labelled `belongs_to` ("active" / "last_run" /
# null), and `test_inference_log_never_globs_the_directory` pins the absence of
# this fallback directly.


# --- Recording log ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_record_handler(monkeypatch: pytest.MonkeyPatch):
    """Ensure no leaked ring-buffer handler from another test, and detach any
    handler this test attaches so loggers aren't left instrumented."""
    from makermodslab import record

    monkeypatch.setattr(record, "_record_log_handler", None)
    yield
    with record._record_log_lock:
        record._detach_record_log_handler_locked()
        record._record_log_handler = None


def test_recording_log_empty_when_no_session() -> None:
    """No handler attached → empty text, no raise."""
    from makermodslab import record

    result = record.handle_recording_log()
    assert result["logs"] == ""


def test_recording_log_captures_logger_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """After attaching the handler, records logged to the record logger appear
    in the ring buffer's snapshot."""
    from makermodslab import record

    record._attach_record_log_handler()
    record.logger.info("recording episode 1")
    record.logger.info("saved episode 1")

    result = record.handle_recording_log()
    assert "recording episode 1" in result["logs"]
    assert "saved episode 1" in result["logs"]


def test_recording_log_ring_buffer_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ring buffer keeps only the last N records — memory cannot grow
    unbounded no matter how many lines are logged."""
    from makermodslab import record

    monkeypatch.setattr(record, "_RECORD_LOG_MAX_LINES", 5)
    record._attach_record_log_handler()
    for i in range(50):
        record.logger.info("line %d", i)

    result = record.handle_recording_log()
    lines = result["logs"].splitlines()
    assert len(lines) == 5
    # Only the trailing 5 survive.
    assert "line 45" in lines[0]
    assert "line 49" in lines[-1]


def test_recording_log_tail_max_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler's own max_lines arg trims the returned tail independently of
    the buffer capacity."""
    from makermodslab import record

    record._attach_record_log_handler()
    for i in range(20):
        record.logger.info("msg %d", i)

    result = record.handle_recording_log(max_lines=3)
    lines = result["logs"].splitlines()
    assert len(lines) == 3
    assert "msg 19" in lines[-1]


def test_attach_replaces_previous_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh attach detaches the previous handler so a new session starts with
    a clean buffer and the record logger isn't left with a stale handler."""
    from makermodslab import record

    record._attach_record_log_handler()
    first = record._record_log_handler
    record.logger.info("from first session")

    record._attach_record_log_handler()
    second = record._record_log_handler
    assert second is not first
    # The old handler is no longer attached to the logger.
    assert first not in record.logger.handlers
    # The new buffer starts empty of the old session's lines.
    record.logger.info("from second session")
    result = record.handle_recording_log()
    assert "from first session" not in result["logs"]
    assert "from second session" in result["logs"]


def test_ring_buffer_handler_capacity_direct() -> None:
    """Unit-test the handler in isolation: it never holds more than `capacity`
    formatted lines."""
    from makermodslab.record import _RingBufferLogHandler

    handler = _RingBufferLogHandler(capacity=3)
    for i in range(10):
        handler.emit(
            logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="entry %d",
                args=(i,),
                exc_info=None,
            )
        )
    snap = handler.snapshot()
    assert len(snap) == 3
    assert "entry 9" in snap[-1]
