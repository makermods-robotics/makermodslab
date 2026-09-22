# Copyright 2026 MakerMods. All rights reserved.
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

"""Session-scoped recording prompts refuse stale ids before control dispatch."""

from __future__ import annotations

import pytest

from makermodslab import record, sessions
from makermodslab.api_errors import ApiError


def test_stale_prompt_cannot_reach_new_recording(monkeypatch):
    tracker = sessions.SessionTracker()
    tracker._current = {"id": "new", "kind": "recording"}
    monkeypatch.setattr(sessions, "tracker", tracker)
    calls = []
    monkeypatch.setattr(record, "handle_submit_episode_task", lambda task: calls.append(task))

    with pytest.raises(ApiError) as error:
        sessions.handle_recording_episode_task_for_session("old", "pick")
    assert error.value.status_code == 404
    assert calls == []


def test_matching_prompt_forwards_the_soft_result(monkeypatch):
    tracker = sessions.SessionTracker()
    tracker._current = {"id": "current", "kind": "recording"}
    monkeypatch.setattr(sessions, "tracker", tracker)
    monkeypatch.setattr(
        record, "handle_submit_episode_task", lambda task: {"success": False, "message": task}
    )

    assert sessions.handle_recording_episode_task_for_session("current", "pick") == {
        "success": False,
        "message": "pick",
    }


def test_stale_status_cannot_read_replacement_recording(monkeypatch):
    tracker = sessions.SessionTracker()
    tracker._current = {"id": "new", "kind": "recording"}
    tracker._last_ended = {"id": "old", "kind": "recording"}
    monkeypatch.setattr(sessions, "tracker", tracker)
    calls = []
    monkeypatch.setattr(record, "handle_recording_status", lambda: calls.append(True))

    with pytest.raises(ApiError) as error:
        sessions.handle_recording_status_for_session("old")
    assert error.value.status_code == 404
    assert calls == []


def test_matching_terminal_status_remains_readable(monkeypatch):
    tracker = sessions.SessionTracker()
    tracker._last_ended = {"id": "old", "kind": "recording"}
    monkeypatch.setattr(sessions, "tracker", tracker)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "handle_recording_status", lambda: {"session_ended": True})

    assert sessions.handle_recording_status_for_session("old") == {"session_ended": True}


def test_previous_session_cannot_read_new_recording_before_tracker_claim(monkeypatch):
    tracker = sessions.SessionTracker()
    tracker._last_ended = {"id": "old", "kind": "recording"}
    monkeypatch.setattr(sessions, "tracker", tracker)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "preparing")
    calls = []
    monkeypatch.setattr(record, "handle_recording_status", lambda: calls.append(True))

    with pytest.raises(ApiError) as error:
        sessions.handle_recording_status_for_session("old")
    assert error.value.status_code == 404
    assert calls == []
