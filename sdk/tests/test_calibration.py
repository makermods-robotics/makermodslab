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
"""client.calibration — the observe-and-advance half of calibration.

MockTransport for everything that would drive an arm; e2e only for the
read-only status routes, which answer idle on a server with no hardware.
"""

from __future__ import annotations

import json

import httpx
from helpers import mock_client

STEP_WAITING = {
    "calibration_active": True,
    "status": "awaiting_step",
    "device_type": "robot",
    "error": None,
    "message": "Fold the arm and close the gripper, then confirm.",
    "step": 1,
    "total_steps": 1,
    "current_positions": {"shoulder_pan": 0.5},
    "recorded_ranges": None,
    "image_url": "/media/calibration/metal-zero.png",
    "live_positions": True,
}

SWEEP_RECORDING = {
    "calibration_active": True,
    "status": "recording",
    "device_type": "robot",
    "message": "Move every joint through its range.",
    "step": 1,
    "total_steps": 1,
    "recorded_ranges": {"shoulder_pan": {"min": 12.0, "max": 170.0, "current": 90.0}},
}


def test_status_reads_the_step_wizard_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=STEP_WAITING)

    with mock_client(handler) as client:
        status = client.calibration.status(arm_type="metal")
    assert seen["path"] == "/api/v1/calibration-status"
    assert seen["params"] == {"arm_type": "metal"}
    assert status.awaiting_step is True
    assert status.finished is False
    assert status.message.startswith("Fold the arm")
    assert status.image_url is not None and status.live_positions is True
    assert status.recorded_ranges is None


def test_status_reads_the_sweep_shape_through_the_same_model():
    """One model reads both flows — the server serves one field-compatible
    shape whichever manager answers."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SWEEP_RECORDING)

    with mock_client(handler) as client:
        status = client.calibration.status()
    assert status.status == "recording"
    assert status.awaiting_step is False
    assert status.recorded_ranges["shoulder_pan"]["max"] == 170.0
    # the wizard's own fields default rather than blowing up
    assert status.image_url is None and status.live_positions is False


def test_status_omits_arm_type_when_not_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"status": "idle"})

    with mock_client(handler) as client:
        client.calibration.status()
    assert seen["params"] == {}


def test_terminal_statuses_read_finished():
    for state in ("completed", "error"):

        def handler(request: httpx.Request, state=state) -> httpx.Response:
            return httpx.Response(200, json={"status": state, "calibration_active": False})

        with mock_client(handler) as client:
            assert client.calibration.status().finished is True


def test_complete_step_sends_the_step_it_confirms():
    """The step number is what stops a late confirmation from being applied to
    whatever step the family published next."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "message": "Step confirmed"})

    with mock_client(handler) as client:
        answer = client.calibration.complete_step(step=1)
    assert seen["path"] == "/api/v1/complete-calibration-step"
    assert seen["body"] == {"step": 1}
    assert answer.success is True


def test_complete_step_without_a_step_sends_an_empty_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "message": "Step confirmed"})

    with mock_client(handler) as client:
        client.calibration.complete_step()
    assert seen["body"] == {}


def test_stale_step_confirmation_is_a_soft_refusal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": False, "message": "Step 1 is not the current step (2); nothing confirmed."},
        )

    with mock_client(handler) as client:
        answer = client.calibration.complete_step(step=1)
    assert answer.success is False
    assert "nothing confirmed" in answer.message


def test_auto_batch_status_counts_and_per_arm_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auto-calibration-batch-status"
        return httpx.Response(
            200,
            json={
                "active": False,
                "arms": [
                    {
                        "active": False,
                        "status": "completed",
                        "message": "done",
                        "error": None,
                        "logs": ["saved"],
                        "name": "bench_follower",
                        "port": "/dev/ttyA",
                        "device_type": "robot",
                        "arm": "left",
                    },
                    {
                        "active": False,
                        "status": "failed",
                        "message": "bus error",
                        "error": "no reply",
                        "logs": [],
                        "name": "bench_leader",
                        "port": "/dev/ttyB",
                        "device_type": "teleop",
                        "arm": "left",
                    },
                ],
                "total": 2,
                "completed": 1,
                "failed": 1,
                "logs": ["[bench_follower] saved"],
            },
        )

    with mock_client(handler) as client:
        batch = client.calibration.auto_batch_status()
    assert batch.finished is True
    assert batch.completed == 1 and batch.failed == 1
    assert batch.arms[1].error == "no reply"
    assert batch.arms[0].name == "bench_follower"


def test_batch_still_running_is_not_finished():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"active": True, "arms": [], "total": 2, "completed": 0, "failed": 0, "logs": []},
        )

    with mock_client(handler) as client:
        assert client.calibration.auto_batch_status().finished is False


def test_configs_sends_arm_type_and_optional_leader_kind():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            json={
                "success": True,
                "device_type": "robot",
                "configs": [
                    {
                        "name": "bench_follower",
                        "filename": "bench_follower.json",
                        "size": 812,
                        "modified": 1.0,
                    }
                ],
            },
        )

    with mock_client(handler) as client:
        listing = client.calibration.configs("robot", arm_type="metal")
        assert [c.name for c in listing.configs] == ["bench_follower"]
        client.calibration.configs("teleop", arm_type="metal", leader_kind="star")
    assert seen[0] == ("/api/v1/calibration-configs/robot", {"arm_type": "metal"})
    assert seen[1] == ("/api/v1/calibration-configs/teleop", {"arm_type": "metal", "leader_kind": "star"})


def test_configs_failure_is_a_soft_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "message": "Invalid device type"})

    with mock_client(handler) as client:
        listing = client.calibration.configs("nope")
    assert listing.success is False and listing.configs == []


# --- end to end: read-only status only ----------------------------------------


def test_status_is_idle_end_to_end(sdk_client):
    status = sdk_client.calibration.status()
    assert status.calibration_active is False
    assert status.status == "idle"


def test_auto_statuses_are_idle_end_to_end(sdk_client):
    assert sdk_client.calibration.auto_status().active is False
    batch = sdk_client.calibration.auto_batch_status()
    assert batch.active is False and batch.total == 0


def test_configs_listing_end_to_end(sdk_client):
    listing = sdk_client.calibration.configs("robot")
    assert listing.success is True
    assert listing.device_type == "robot"


def test_unknown_arm_type_refuses_coded_end_to_end(sdk_client):
    """require_known_arm_type gates the library routes — an unknown id is a
    400, never a silent fallback to the SO-101 library."""
    import pytest
    from makermodslab_sdk import ApiError

    with pytest.raises(ApiError) as excinfo:
        sdk_client.calibration.configs("robot", arm_type="definitely-not-installed")
    assert excinfo.value.status == 400
    assert excinfo.value.code == "robot.arm_type.unavailable"
