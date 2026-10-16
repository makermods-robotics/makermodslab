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
"""Real confirmation boundary, mocked session lifecycle; no devices or workers."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import NotFoundError, ServerTooOldError, StepNotConfirmedError
from makermodslab_sdk.resources.sessions import ActiveSession


@pytest.mark.parametrize("replace,confirm", [(False, True), (True, False), (True, True)])
def test_delayed_human_confirmation_cannot_advance_a_replacement(monkeypatch, replace, confirm):
    from makermodslab import server, sessions
    from makermodslab.api_errors import ApiError
    from makermodslab.step_calibrate import StepCalibrationManager, StepCalibrationStatus

    tracker = sessions.SessionTracker()
    monkeypatch.setattr(sessions, "tracker", tracker)
    manager = StepCalibrationManager()
    monkeypatch.setattr(server, "step_calibration_manager", manager)
    monkeypatch.setattr(ActiveSession, "_start_heartbeat_thread", lambda self: None)
    current = "cal-original"
    confirmed_runs = []

    def claim(identifier):
        tracker.observe({"session": {"kind": "calibration", "active": False}})
        tracker.observe({"session": {"kind": "calibration", "active": True}})
        with tracker._lock:
            tracker._current["id"] = identifier
        manager._session_id = identifier
        manager._step_confirmed.clear()
        manager.status = StepCalibrationStatus(
            calibration_active=True,
            status="awaiting_step",
            step=1,
            device_type="robot",
            message=f"Pose {identifier} arm.",
        )

    claim(current)

    def session(identifier):
        return {
            "id": identifier,
            "kind": "calibration",
            "robot": "bench",
            "owner": "test",
            "started_at": 1.0,
            "revision": 1,
            "phase": "awaiting_step",
            "lease": None,
        }

    def transport(request):
        path = request.url.path
        if path == "/api/v1/robots/bench":
            return httpx.Response(200, json={"robot": {"name": "bench", "arm_type": "maker"}})
        if path == "/api/v1/sessions":
            return httpx.Response(201, json={"session": session(current), "warnings": None})
        if path == "/api/v1/calibration-status":
            return httpx.Response(200, json=server.calibration_status(arm_type="maker"))
        if path == "/api/v1/sessions/cal-original/calibration/step":
            body = server.CompleteCalibrationStepRequest(**json.loads(request.read()))
            try:
                answer = server.calibration_step_for_session("cal-original", body)
            except ApiError as error:
                return httpx.Response(error.status_code, json={"detail": error.detail, "code": error.code})
            if manager._step_confirmed.is_set():
                confirmed_runs.append(current)
            manager._update_status(calibration_active=False, status="completed")
            return httpx.Response(200, json=answer)
        if path == "/api/v1/sessions/current":
            return httpx.Response(
                200,
                json={
                    "session": None,
                    "last_ended": {
                        "id": current,
                        "kind": "calibration",
                        "ended_at": 2.0,
                        "phase": "completed",
                        "reason": None,
                    },
                },
            )
        if path == "/api/v1/sessions/cal-original/stop":
            return httpx.Response(404, json={"detail": "Original ended", "code": "session.not_found"})
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    def human_confirmation(status):
        nonlocal current
        assert status.message == "Pose cal-original arm."
        if replace:
            current = "cal-replacement"
            claim(current)
        return confirm

    with mock_client(transport) as client:
        if replace:
            expected = NotFoundError if confirm else StepNotConfirmedError
            with pytest.raises(expected):
                client.flows.calibrate_zero(
                    "bench",
                    device_type="robot",
                    confirm=human_confirmation,
                    sleep_fn=lambda _: None,
                    clock=lambda: 0.0,
                )
        else:
            client.flows.calibrate_zero(
                "bench",
                device_type="robot",
                confirm=human_confirmation,
                sleep_fn=lambda _: None,
                clock=lambda: 0.0,
            )
    assert confirmed_runs == (["cal-original"] if confirm and not replace else [])


def test_confirmation_refuses_new_manager_before_tracker_claim_is_published(monkeypatch):
    from makermodslab import sessions
    from makermodslab.api_errors import ApiError
    from makermodslab.step_calibrate import StepCalibrationManager, StepCalibrationStatus

    tracker = sessions.SessionTracker()
    tracker.observe({"session": {"kind": "calibration", "active": True}})
    old_id = tracker.current()["id"]
    monkeypatch.setattr(sessions, "tracker", tracker)
    manager = StepCalibrationManager()
    # A replacement has claimed the manager but has not published its new id.
    manager._session_id = None
    manager.status = StepCalibrationStatus(calibration_active=True, status="awaiting_step", step=1)
    with pytest.raises(ApiError) as error:
        manager.complete_step_for_session(old_id, 1)
    assert error.value.code == "session.not_found"
    assert not manager._step_confirmed.is_set()


def test_old_server_never_falls_back_to_unscoped_confirmation():
    calls = []

    def transport(request):
        calls.append(request.url.path)
        return httpx.Response(404, json={"detail": "Not Found"})

    with mock_client(transport) as client, pytest.raises(ServerTooOldError):
        client.sessions.calibration_step("old-session", 1)
    assert calls == ["/api/v1/sessions/old-session/calibration/step"]


def test_scoped_confirmation_refuses_a_stale_id_through_the_real_app(sdk_client):
    with pytest.raises(NotFoundError) as error:
        sdk_client.sessions.calibration_step("__sdk_missing_session__", 1)
    assert error.value.code == "session.not_found"
