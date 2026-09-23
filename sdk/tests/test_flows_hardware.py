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
"""Passive hardware-context flow; MockTransport only, never hardware."""

from __future__ import annotations

import httpx
from helpers import mock_client

ARM_FAMILIES = {
    "arms": [
        {
            "id": "so101",
            "label": "SO-101",
            "joints_per_arm": 6,
            "calibration": {"kind": "range_sweep"},
            "capabilities": {
                "supports_port_probe": False,
                "motion_identify_energizes_follower": False,
                "supports_gripper_wiggle": False,
            },
        },
        {
            "id": "maker",
            "label": "Maker Arm",
            "joints_per_arm": 7,
            "calibration": {"kind": "steps"},
            "capabilities": {
                "supports_port_probe": True,
                "motion_identify_energizes_follower": False,
                "supports_gripper_wiggle": True,
            },
        },
        {
            "id": "metal",
            "label": "Metal Arm",
            "joints_per_arm": 7,
            "calibration": {"kind": "steps"},
            "capabilities": {
                "supports_port_probe": True,
                "motion_identify_energizes_follower": True,
                "supports_gripper_wiggle": True,
            },
        },
    ]
}


def test_inspect_hardware_builds_passive_context_and_literal_next_actions():
    responses = {
        "/api/v1/arms": ARM_FAMILIES,
        "/api/v1/available-ports": {
            "status": "success",
            "ports": ["/dev/shared", "/dev/free", "/dev/leader"],
        },
        "/api/v1/available-cameras": {
            "status": "success",
            "cameras": [{"index": 0, "name": "Wrist", "available": True, "unique_id": "cam-wrist"}],
        },
        "/api/v1/robots": {
            "status": "success",
            "robots": [
                {
                    "name": "bench",
                    "arm_type": "so101",
                    "mode": "single",
                    "leader_port": "/dev/leader",
                    "follower_port": "/dev/missing",
                    "cameras": [{"name": "wrist", "camera_index": 0, "unique_id": "cam-wrist"}],
                },
                {
                    "name": "alias-a",
                    "arm_type": "maker",
                    "mode": "single",
                    "follower_port": "/dev/shared",
                },
                {
                    "name": "alias-b",
                    "arm_type": "maker",
                    "mode": "single",
                    "follower_port": "/dev/shared",
                },
            ],
        },
    }
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert seen == [
        ("GET", "/api/v1/arms"),
        ("GET", "/api/v1/available-ports"),
        ("GET", "/api/v1/available-cameras"),
        ("GET", "/api/v1/robots"),
    ]
    assert context.visible_ports == ("/dev/shared", "/dev/free", "/dev/leader")
    assert [camera.name for camera in context.visible_cameras] == ["Wrist"]
    assert [robot.name for robot in context.robots] == ["bench", "alias-a", "alias-b"]
    bench = context.robots[0]
    assert [(slot.slot, slot.port, slot.present) for slot in bench.ports] == [
        ("leader_port", "/dev/leader", True),
        ("follower_port", "/dev/missing", False),
    ]
    assert bench.cameras[0].name == "wrist" and bench.cameras[0].present is True
    assert [(slot.robot, slot.slot, slot.port) for slot in context.assigned_ports_missing] == [
        ("bench", "follower_port", "/dev/missing")
    ]
    assert context.unassigned_visible_ports == ("/dev/free",)
    shared = [item for item in context.observations if item.kind == "shared_port_reference"]
    assert len(shared) == 1
    assert shared[0].port == "/dev/shared"
    assert shared[0].references == ("alias-a.follower_port", "alias-b.follower_port")
    assert "alias" in shared[0].message.lower()
    assert context.errors == ()

    calls = {action.call: action for action in context.next_actions}
    assert "client.system.identify_maker_arm('robot', arm_type='so101', ports=['/dev/free'])" in calls
    assert calls["client.system.probe_maker_arm_ports(arm_type='maker', ports=['/dev/free'])"].effects == (
        "read_only",
    )
    assert calls["client.system.probe_maker_arm_ports(arm_type='metal', ports=['/dev/free'])"].effects == (
        "may_energize",
    )
    assert calls["client.system.wiggle_can_gripper('metal', 'robot', '/dev/free')"].effects == (
        "may_energize",
        "moves_hardware",
    )
    # The summary names the FIELDS it counts, so a reader's next reach is
    # context.visible_ports rather than a guess like context.ports.
    assert "visible_ports=3" in context.summary()
    assert "visible_cameras=1" in context.summary()
    assert "/dev/missing" in context.summary()


def test_inspect_hardware_keeps_other_sections_when_reads_fail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/arms":
            return httpx.Response(503, json={"detail": "registry unavailable"})
        if request.url.path == "/api/v1/available-ports":
            return httpx.Response(200, json={"status": "error", "message": "port scan failed"})
        if request.url.path == "/api/v1/available-cameras":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "cameras": [{"index": 2, "name": "Scene", "available": True}],
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "error",
                "message": "robot registry exploded",
                "robots": [{"name": "must-not-leak"}],
            },
        )

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert [camera.name for camera in context.visible_cameras] == ["Scene"]
    assert context.visible_ports == ()
    assert context.robots == ()
    assert context.arm_families == ()
    assert [error.section for error in context.errors] == ["arms", "ports", "robots"]
    assert "registry unavailable" in context.errors[0].detail
    assert context.errors[1].detail == "port scan failed"
    assert context.errors[2].detail == "robot registry exploded"
    summary = context.summary()
    assert "Partial errors" in summary
    assert "visible_cameras=1" in summary


def test_inspect_hardware_does_not_suggest_active_fallback_without_free_ports():
    responses = {
        "/api/v1/arms": ARM_FAMILIES,
        "/api/v1/available-ports": {"status": "success", "ports": ["/dev/assigned"]},
        "/api/v1/available-cameras": {"status": "success", "cameras": []},
        "/api/v1/robots": {
            "status": "success",
            "robots": [{"name": "bench", "follower_port": "/dev/assigned"}],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert context.unassigned_visible_ports == ()
    assert context.next_actions == ()


def test_successful_empty_port_scan_marks_saved_assignments_missing():
    responses = {
        "/api/v1/arms": {"arms": []},
        "/api/v1/available-ports": {"status": "success", "ports": []},
        "/api/v1/available-cameras": {"status": "success", "cameras": []},
        "/api/v1/robots": {
            "status": "success",
            "robots": [{"name": "bench", "follower_port": "/dev/unplugged"}],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert [(item.robot, item.port) for item in context.assigned_ports_missing] == [
        ("bench", "/dev/unplugged")
    ]


def test_failed_port_scan_keeps_saved_assignment_presence_unknown():
    responses = {
        "/api/v1/arms": {"arms": []},
        "/api/v1/available-ports": {"status": "error", "message": "scan failed"},
        "/api/v1/available-cameras": {"status": "error", "message": "scan failed", "cameras": []},
        "/api/v1/robots": {
            "status": "success",
            "robots": [
                {
                    "name": "bench",
                    "follower_port": "/dev/maybe",
                    "cameras": [{"name": "wrist", "camera_index": 0}],
                }
            ],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert context.robots[0].ports[0].present is None
    assert context.robots[0].cameras[0].present is None
    assert context.assigned_ports_missing == ()
    assert "/dev/maybe (unknown)" in context.summary()


def test_failed_robot_read_does_not_claim_visible_ports_are_unassigned():
    responses = {
        "/api/v1/arms": ARM_FAMILIES,
        "/api/v1/available-ports": {"status": "success", "ports": ["/dev/maybe-assigned"]},
        "/api/v1/available-cameras": {"status": "success", "cameras": []},
        "/api/v1/robots": {"status": "error", "message": "records unavailable", "robots": []},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert context.visible_ports == ("/dev/maybe-assigned",)
    assert context.unassigned_visible_ports == ()
    assert context.next_actions == ()


def test_summary_caps_detail_while_structured_results_stay_complete():
    ports = [f"/dev/free-{index}" for index in range(12)]
    responses = {
        "/api/v1/arms": ARM_FAMILIES,
        "/api/v1/available-ports": {"status": "success", "ports": ports},
        "/api/v1/available-cameras": {"status": "success", "cameras": []},
        "/api/v1/robots": {"status": "success", "robots": []},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        context = client.flows.inspect_hardware()

    assert context.unassigned_visible_ports == tuple(ports)
    assert len(context.next_actions) > 5
    summary = context.summary()
    assert "4 more ports" in summary
    assert "more next actions" in summary
    assert len(summary) < 2_000


# --- readiness the server already computes ------------------------------------


def _context_with_robots(records: list[dict]):
    """A context whose only interesting input is the saved robot records."""
    responses = {
        "/api/v1/arms": ARM_FAMILIES,
        "/api/v1/available-ports": {"status": "success", "ports": []},
        "/api/v1/available-cameras": {"status": "success", "cameras": []},
        "/api/v1/robots": {"status": "success", "robots": records},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    with mock_client(handler) as client:
        return client.flows.inspect_hardware()


def test_saved_robot_carries_server_readiness_flags():
    """_record_with_clean returns these on every read; the flow must keep them
    — "is this robot usable?" is the whole point of a setup snapshot."""
    context = _context_with_robots(
        [
            {
                "name": "bench",
                "arm_type": "metal",
                "mode": "single",
                "follower_port": "/dev/ttyA",
                "is_clean": False,
                "follower_ready": True,
                "leader_ready": False,
                "arm_available": True,
            }
        ]
    )
    robot = context.robots[0]
    assert robot.is_clean is False
    assert robot.follower_ready is True
    assert robot.leader_ready is False
    assert robot.arm_available is True
    assert "inference/replay" in robot.readiness_summary()
    assert "inference/replay" in context.summary()


def test_readiness_degrades_to_unknown_on_an_older_server():
    context = _context_with_robots([{"name": "bench", "arm_type": "so101", "mode": "single"}])
    robot = context.robots[0]
    assert robot.is_clean is None and robot.arm_available is None
    assert robot.readiness_summary() == "unknown readiness"


def test_unknown_arm_type_reads_as_unavailable_not_merely_unready():
    context = _context_with_robots(
        [
            {
                "name": "bench",
                "arm_type": "someext",
                "mode": "single",
                "is_clean": False,
                "arm_available": False,
            }
        ]
    )
    assert "not installed" in context.robots[0].readiness_summary()


def test_clean_record_reads_ready():
    context = _context_with_robots(
        [{"name": "bench", "arm_type": "so101", "mode": "single", "is_clean": True, "arm_available": True}]
    )
    assert context.robots[0].readiness_summary() == "ready"
