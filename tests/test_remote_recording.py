from __future__ import annotations

import json
import stat
from pathlib import Path

from makermodslab.remote_teleop.recording import BoundedSessionRecorder


def test_bounded_recording_is_private_and_rejects_secret_fields(tmp_path: Path) -> None:
    recorder = BoundedSessionRecorder(tmp_path / "records", "session-1", max_queue=16)
    recorder.start({"session_id": "session-1", "clock_uncertainty_ns": 10})
    assert recorder({"event": "action.executed", "sequence": 1}) is True
    assert recorder({"event": "bad", "action_key_base64": "must-not-land"}) is False
    assert recorder.close({"event": "session.stopped", "torque_off_confirmed": True}) is True

    files = list((tmp_path / "records").iterdir())
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["header", "event", "terminal"]
    assert "must-not-land" not in files[0].read_text()
    assert recorder.status()["dropped"] == 1


def test_recording_never_starts_on_construction(tmp_path: Path) -> None:
    recorder = BoundedSessionRecorder(tmp_path / "records", "session-2", max_queue=16)
    assert recorder.status()["active"] is False
    assert not (tmp_path / "records").exists()
