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
#
"""The two calibration flows. MockTransport only — never real hardware — and
no test sleeps: sleep_fn and clock are injected everywhere."""

from __future__ import annotations

import json

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import (
    CalibrationFlowError,
    CalibrationFlowTimeout,
    StepNotConfirmedError,
)


def session_body(session_id: str = "cal-1", kind: str = "calibration") -> dict:
    return {
        "session": {
            "id": session_id,
            "kind": kind,
            "robot": "bench",
            "owner": "sdk:test:1:ab",
            "started_at": 1.0,
            "revision": 1,
            "phase": None,
            "lease": {"owner": "sdk:test:1:ab", "timeout_s": 60.0, "expires_in_s": 60.0},
        },
        "warnings": None,
    }


def ended_body(session_id: str = "cal-1", kind: str = "calibration", phase: str | None = None) -> dict:
    return {
        "session": None,
        "last_ended": {
            "id": session_id,
            "kind": kind,
            "ended_at": 9.0,
            "phase": phase,
            "reason": None,
        },
    }


def scripted(statuses, *, kind="calibration", status_path, extra=None):
    """A handler serving a session start, a scripted status sequence, the
    session stop, and the end summary. Records what was sent."""
    seen = {"steps": [], "starts": 0, "stops": 0, "status_params": []}
    remaining = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/sessions" and request.method == "POST":
            seen["starts"] += 1
            seen["start_body"] = json.loads(request.read())
            return httpx.Response(201, json=session_body(kind=kind))
        if path.endswith("/stop"):
            seen["stops"] += 1
            return httpx.Response(200, json={"session": session_body(kind=kind)["session"], "result": {}})
        if path == "/api/v1/sessions/current":
            return httpx.Response(200, json=ended_body(kind=kind))
        if path == "/api/v1/complete-calibration-step":
            seen["steps"].append(json.loads(request.read()))
            answer = (extra or {}).get("step_answer", {"success": True, "message": "Step confirmed"})
            return httpx.Response(200, json=answer)
        if path == "/api/v1/robots/bench":
            arm_type = (extra or {}).get("arm_type", "maker")
            return httpx.Response(200, json={"robot": {"name": "bench", "arm_type": arm_type}})
        if path == status_path:
            seen["status_params"].append(dict(request.url.params))
            return httpx.Response(200, json=remaining.pop(0) if remaining else statuses[-1])
        return httpx.Response(500, json={"detail": f"unexpected {path}"})

    return handler, seen


# --- auto-calibration: no human in the loop -----------------------------------


def test_auto_calibrate_polls_to_terminal_and_reports_every_arm():
    running = {"active": True, "arms": [], "total": 2, "completed": 0, "failed": 0, "logs": []}
    done = {
        "active": False,
        "total": 2,
        "completed": 1,
        "failed": 1,
        "logs": [],
        "arms": [
            {
                "status": "completed",
                "name": "bench_follower",
                "port": "/dev/a",
                "device_type": "robot",
                "arm": "left",
            },
            {
                "status": "failed",
                "name": "bench_leader",
                "port": "/dev/b",
                "device_type": "teleop",
                "arm": "left",
                "error": "no reply",
            },
        ],
    }
    handler, seen = scripted([running, running, done], status_path="/api/v1/auto-calibration-batch-status")
    sleeps: list[float] = []

    with mock_client(handler) as client:
        result = client.flows.auto_calibrate(
            "bench",
            arms=[{"device_type": "robot"}, {"device_type": "teleop"}],
            sleep_fn=sleeps.append,
            clock=lambda: 0.0,
        )

    assert seen["start_body"]["kind"] == "auto_calibration"
    assert result.completed == 1 and result.failed == 1
    # a per-arm failure is a REPORTED outcome, not a raise
    assert "no reply" in result.summary()
    assert "bench_leader" in result.summary()
    assert sleeps == [2.0, 2.0]
    assert seen["stops"] == 1


def test_auto_calibrate_times_out_and_stops_its_session():
    running = {"active": True, "arms": [], "total": 1, "completed": 0, "failed": 0, "logs": []}
    handler, seen = scripted([running], status_path="/api/v1/auto-calibration-batch-status")
    ticks = iter([0.0, 0.0, 600.0, 600.0, 600.0])

    with mock_client(handler) as client, pytest.raises(CalibrationFlowTimeout) as excinfo:
        client.flows.auto_calibrate(
            "bench",
            arms=[{"device_type": "robot"}],
            timeout=1.0,
            sleep_fn=lambda _s: None,
            clock=lambda: next(ticks),
        )
    assert excinfo.value.robot == "bench"
    assert "auto_batch_status()" in str(excinfo.value)
    assert seen["stops"] == 1  # the with-block stopped the leased session


def test_auto_calibrate_rejects_an_empty_arm_list():
    handler, _ = scripted([{}], status_path="/api/v1/auto-calibration-batch-status")
    with mock_client(handler) as client, pytest.raises(ValueError):
        client.flows.auto_calibrate("bench", arms=[])


# --- the step wizard: the human is the loop -----------------------------------


AWAITING = {
    "calibration_active": True,
    "status": "awaiting_step",
    "device_type": "robot",
    "message": "Fold the arm and close the gripper.",
    "step": 1,
    "total_steps": 1,
    "image_url": "/media/zero.png",
    "live_positions": True,
}
COMPLETED = {"calibration_active": False, "status": "completed", "device_type": "robot", "step": 1}


def test_calibrate_zero_relays_the_step_then_confirms_it():
    handler, seen = scripted([AWAITING, AWAITING, COMPLETED], status_path="/api/v1/calibration-status")
    shown: list[str] = []

    def confirm(status):
        shown.append(status.message)
        assert status.image_url == "/media/zero.png"  # the flow hands over the whole status
        return True

    with mock_client(handler) as client:
        result = client.flows.calibrate_zero(
            "bench", device_type="robot", confirm=confirm, sleep_fn=lambda _s: None, clock=lambda: 0.0
        )

    assert shown == ["Fold the arm and close the gripper."]  # asked once, not once per poll
    assert seen["steps"] == [{"step": 1}]  # and confirmed the step it was shown
    assert result.steps_confirmed == 1
    assert result.status.status == "completed"
    assert seen["start_body"]["kind"] == "calibration"


def test_declining_writes_nothing_and_stops_the_session():
    """Declining is a VALID answer — the arm was not in the pose, so
    confirming would have recorded a wrong zero."""
    handler, seen = scripted([AWAITING], status_path="/api/v1/calibration-status")

    with mock_client(handler) as client, pytest.raises(StepNotConfirmedError) as excinfo:
        client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=lambda _status: False,
            sleep_fn=lambda _s: None,
            clock=lambda: 0.0,
        )
    assert seen["steps"] == []  # nothing was confirmed
    assert seen["stops"] == 1
    assert "declined" in str(excinfo.value)


def test_a_refused_confirmation_raises_with_the_servers_reason():
    handler, _ = scripted(
        [AWAITING],
        status_path="/api/v1/calibration-status",
        extra={"step_answer": {"success": False, "message": "The session is not waiting for a step"}},
    )
    with mock_client(handler) as client, pytest.raises(CalibrationFlowError) as excinfo:
        client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=lambda _status: True,
            sleep_fn=lambda _s: None,
            clock=lambda: 0.0,
        )
    assert "not waiting for a step" in str(excinfo.value)


def test_a_failed_wizard_raises_with_its_terminal_detail():
    failed = {"calibration_active": False, "status": "error", "error": "no reply from gripper", "step": 1}
    handler, _ = scripted([failed], status_path="/api/v1/calibration-status")
    with mock_client(handler) as client, pytest.raises(CalibrationFlowError) as excinfo:
        client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=lambda _status: True,
            sleep_fn=lambda _s: None,
            clock=lambda: 0.0,
        )
    assert "no reply from gripper" in str(excinfo.value)


def test_confirm_is_required_and_has_no_default():
    """A default would let an agent confirm an unposed arm, writing a wrong
    zero that corrupts every later session silently."""
    import inspect

    from makermodslab_sdk.flows_calibration import CalibrationFlows

    parameter = inspect.signature(CalibrationFlows.calibrate_zero).parameters["confirm"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_multi_step_family_is_asked_once_per_step():
    second = {**AWAITING, "step": 2, "message": "Now straighten the elbow."}
    handler, seen = scripted([AWAITING, second, COMPLETED], status_path="/api/v1/calibration-status")
    shown: list[int] = []

    with mock_client(handler) as client:
        result = client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=lambda status: (shown.append(status.step), True)[1],
            sleep_fn=lambda _s: None,
            clock=lambda: 0.0,
        )
    assert shown == [1, 2]
    assert seen["steps"] == [{"step": 1}, {"step": 2}]
    assert result.steps_confirmed == 2


def test_a_released_wizard_is_read_through_its_arm_type():
    """The server reads a FINISHED step wizard back as idle unless the
    status call names the arm family — the session releases the moment the
    zero is written, so a flow polling without arm_type never sees
    "completed" and spins until its timeout on a calibration that succeeded."""
    idle = {"calibration_active": False, "status": "idle", "step": 0}
    remaining = [AWAITING]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/calibration-status":
            if remaining:
                return httpx.Response(200, json=remaining.pop(0))
            released = COMPLETED if request.url.params.get("arm_type") == "maker" else idle
            return httpx.Response(200, json=released)
        return inner(request)

    inner, seen = scripted([AWAITING], status_path="/api/v1/calibration-status")
    ticks = iter([0.0] * 4 + [600.0] * 10)

    with mock_client(handler) as client:
        result = client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=lambda _status: True,
            timeout=300.0,
            sleep_fn=lambda _s: None,
            clock=lambda: next(ticks),
        )
    assert result.status.status == "completed"
    assert seen["steps"] == [{"step": 1}]


class FakeClock:
    """A clock the test moves by hand: the human's think time inside
    confirm(), and each sleep_fn call, advance it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_a_slow_human_is_not_charged_against_the_timeout():
    """Seen on hardware: the human took ~17 minutes to pose the arm, the
    server wrote the zero the moment the step was confirmed, and the flow
    still raised a timeout for a calibration that had succeeded. Waiting on
    the human is not the server's budget."""
    handler, seen = scripted([AWAITING, COMPLETED], status_path="/api/v1/calibration-status")
    clock = FakeClock()

    def slow_confirm(_status):
        clock.now += 1015.0  # posing the arm by hand
        return True

    with mock_client(handler) as client:
        result = client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=slow_confirm,
            timeout=300.0,
            sleep_fn=clock.sleep,
            clock=clock,
        )
    assert result.status.status == "completed"
    assert seen["steps"] == [{"step": 1}]


def test_a_slow_server_still_times_out_after_the_step_and_stops_the_session():
    """The budget pauses for the human, not for the server: a step accepted
    and then never finished still times out, quoting the FRESH status."""
    saving = {**AWAITING, "status": "saving", "message": "Saving calibration…"}
    handler, seen = scripted([AWAITING, saving], status_path="/api/v1/calibration-status")
    clock = FakeClock()

    def slow_confirm(_status):
        clock.now += 1015.0
        return True

    with mock_client(handler) as client, pytest.raises(CalibrationFlowTimeout) as excinfo:
        client.flows.calibrate_zero(
            "bench",
            device_type="robot",
            confirm=slow_confirm,
            timeout=5.0,
            poll_interval=1.0,
            sleep_fn=clock.sleep,
            clock=clock,
        )
    assert seen["steps"] == [{"step": 1}]
    assert "still 'saving' after 5s" in str(excinfo.value)  # the server's 5s, not the human's 1015
    assert seen["stops"] == 1
