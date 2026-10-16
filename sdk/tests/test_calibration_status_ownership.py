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
"""Calibration results must stay with their session, even after cached completion."""

from __future__ import annotations

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import CalibrationFlowError
from makermodslab_sdk.resources.sessions import ActiveSession


@pytest.mark.parametrize("kind", ["auto_calibration", "calibration"])
@pytest.mark.parametrize("replacement", ["after_cached_end", "during_status_read"])
def test_replacement_results_are_not_attributed_to_original(monkeypatch, kind, replacement):
    handles = []
    monkeypatch.setattr(ActiveSession, "_start_heartbeat_thread", lambda session: handles.append(session))
    phase = "running"
    reads = []
    original = {
        "id": "original",
        "kind": kind,
        "robot": "bench",
        "owner": "test",
        "started_at": 1.0,
        "revision": 1,
        "phase": "running",
        "lease": {"owner": "test", "timeout_s": 60, "expires_in_s": 60},
    }

    def ended(identifier, outcome):
        return {"id": identifier, "kind": kind, "ended_at": 2.0, "phase": outcome, "reason": None}

    def transport(request):
        nonlocal phase
        path = request.url.path
        if path == "/api/v1/sessions":
            return httpx.Response(201, json={"session": original})
        if path == "/api/v1/robots/bench":
            return httpx.Response(200, json={"robot": {"name": "bench", "arm_type": "maker"}})
        if path.endswith("/heartbeat") or path == "/api/v1/sessions/original/stop":
            return httpx.Response(404, json={"detail": "Ended", "code": "session.not_found"})
        if path == "/api/v1/sessions/current":
            if phase == "running":
                body = {"session": original, "last_ended": None}
            else:
                summary = (
                    ended("original", "error")
                    if phase == "original-ended"
                    else ended("replacement", "completed")
                )
                body = {"session": None, "last_ended": summary}
            return httpx.Response(200, json=body)
        if path in ("/api/v1/auto-calibration-batch-status", "/api/v1/calibration-status"):
            reads.append(path)
            if replacement == "during_status_read":
                phase = "replacement-ended"
            if kind == "auto_calibration":
                body = {
                    "active": phase == "running",
                    "total": 1,
                    "completed": 0 if phase == "running" else 1,
                    "failed": 0,
                    "arms": []
                    if phase == "running"
                    else [{"name": "replacement_follower", "status": "completed"}],
                }
            else:
                body = {
                    "calibration_active": phase == "running",
                    "status": "connecting" if phase == "running" else "completed",
                    "message": "replacement result",
                }
            return httpx.Response(200, json=body)
        raise AssertionError(path)

    def advance(_interval):
        nonlocal phase
        phase = "original-ended"
        assert handles[0]._tick() == "completed"
        phase = "replacement-ended"

    with (
        mock_client(transport) as client,
        pytest.raises(CalibrationFlowError, match="no longer attributable"),
    ):
        if kind == "auto_calibration":
            client.flows.auto_calibrate(
                "bench",
                arms=[{"device_type": "robot"}],
                sleep_fn=advance,
                clock=lambda: 0,
            )
        else:
            client.flows.calibrate_zero(
                "bench",
                device_type="robot",
                confirm=lambda _: True,
                sleep_fn=advance,
                clock=lambda: 0,
            )
    assert len(reads) == 1
    if replacement == "after_cached_end":
        assert handles[0].ended.id == "original"
