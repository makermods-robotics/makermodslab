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


def test_all_failed_launch_attempt_preserves_original_completed_result(monkeypatch):
    from makermodslab import auto_calibrate as ac, sessions

    handles = []
    monkeypatch.setattr(ActiveSession, "_start_heartbeat_thread", lambda session: handles.append(session))
    manager = ac.AutoCalibrationBatchManager()
    monkeypatch.setattr(ac, "auto_calibration_batch_manager", manager)
    tracker = sessions.SessionTracker()
    monkeypatch.setattr(sessions, "tracker", tracker)
    runner = ac._AutoCalArmRunner()
    runner.status = ac.AutoCalibrationStatus(active=True, status="running")
    runner._request = ac.AutoCalibrationRequest(
        device_type="robot", port="/fake/original", config_file="original", robot_name="bench"
    )
    manager._runners = [runner]
    manager._robot_name = "bench"
    tracker.observe({"session": {"kind": "auto_calibration", "active": True, "phase": "running"}})
    original = tracker.current()["id"]

    def transport(request):
        path = request.url.path
        if path == "/api/v1/sessions":
            info = sessions.handle_current_session()["session"]
            info.update(
                robot="bench", owner="test", lease={"owner": "test", "timeout_s": 60, "expires_in_s": 60}
            )
            return httpx.Response(201, json={"session": info})
        if path == "/api/v1/sessions/current":
            return httpx.Response(200, json=sessions.handle_current_session())
        if path.endswith("/heartbeat") or path.endswith("/stop"):
            return httpx.Response(404, json={"detail": "Ended", "code": "session.not_found"})
        if path == "/api/v1/auto-calibration-batch-status":
            return httpx.Response(200, json=manager.get_status())
        raise AssertionError(path)

    def fail_spawn(*args, **kwargs):
        raise OSError("offline simulated spawn failure")

    def advance(_interval):
        runner.status = ac.AutoCalibrationStatus(active=False, status="completed")
        tracker.observe({"session": {"kind": "auto_calibration", "active": False, "phase": "completed"}})
        assert handles[0]._tick() == "completed"
        monkeypatch.setattr(ac.subprocess, "Popen", fail_spawn)
        attempt = manager.start(
            ac.AutoCalibrationBatchRequest(
                arms=[
                    ac.AutoCalibrationBatchArm(
                        device_type="robot", port="/fake/replacement", config_file="replacement"
                    )
                ],
                robot_name="other-bench",
                overwrite=True,
            )
        )
        assert not attempt["success"]
        assert attempt["failed_arms"][0]["name"] == "replacement"

    with mock_client(transport) as client:
        result = client.flows.auto_calibrate(
            "bench",
            arms=[{"device_type": "robot"}],
            sleep_fn=advance,
            clock=lambda: 0,
        )
    assert result.session.id == original
    assert result.completed == 1 and result.failed == 0
    assert result.status.arms[0].name == "original"
    assert manager._robot_name == "bench"
