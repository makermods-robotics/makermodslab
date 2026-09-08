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
"""Tests for makermodslab.identify — touch-to-identify port finder (hardware mocked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from makermodslab import identify
from makermodslab.identify import swing_detected

# ---------------------------------------------------------------------------
# swing_detected: the pure both-directions decision
# ---------------------------------------------------------------------------


def test_swing_both_directions_detects() -> None:
    # Swung 130 ticks left and 130 right of a 2000 baseline.
    assert swing_detected(2000, min_seen=1870, max_seen=2130) is True


def test_swing_exactly_at_threshold_detects() -> None:
    assert swing_detected(2000, min_seen=1880, max_seen=2120, threshold=120) is True


def test_one_sided_swing_is_not_detected() -> None:
    # A bump or a single push moves the joint one way only.
    assert swing_detected(2000, min_seen=2000, max_seen=2400) is False
    assert swing_detected(2000, min_seen=1600, max_seen=2000) is False


def test_drift_below_threshold_is_not_detected() -> None:
    # Small wobble on both sides but under the threshold.
    assert swing_detected(2000, min_seen=1901, max_seen=2099) is False


def test_one_side_over_other_side_under_is_not_detected() -> None:
    assert swing_detected(2000, min_seen=1990, max_seen=2500) is False


# ---------------------------------------------------------------------------
# _identify_arm_sync: bus lifecycle with a fake bus (no hardware)
# ---------------------------------------------------------------------------


class FakeBus:
    """Stands in for FeetechMotorsBus: replays a per-port position script and
    records connect/disconnect calls so tests can assert every bus is released."""

    instances: list[FakeBus] = []
    scripts: dict[str, list[int]] = {}
    fail_connect: set[str] = set()

    def __init__(self, port: str, motors: dict) -> None:
        self.port = port
        self.motors = motors
        self.disconnect_calls: list[dict] = []
        FakeBus.instances.append(self)

    def connect(self) -> None:
        if self.port in FakeBus.fail_connect:
            raise ConnectionError(f"could not open {self.port}")

    def sync_read(self, register: str, motor: str, normalize: bool = True):
        assert register == "Present_Position"
        assert normalize is False
        script = FakeBus.scripts[self.port]
        # Hold the last position once the script is exhausted.
        value = script.pop(0) if len(script) > 1 else script[0]
        return {motor: value}

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append({"disable_torque": disable_torque})


@pytest.fixture
def fake_bus(monkeypatch: pytest.MonkeyPatch) -> type[FakeBus]:
    FakeBus.instances = []
    FakeBus.scripts = {}
    FakeBus.fail_connect = set()
    monkeypatch.setattr(identify, "FeetechMotorsBus", FakeBus)
    return FakeBus


def test_sync_detects_the_moving_port_and_releases_all_buses(fake_bus: type[FakeBus]) -> None:
    fake_bus.scripts = {
        "/dev/still": [2000],  # baseline read, then holds forever
        "/dev/moving": [2000, 2150, 1820],  # baseline, right swing, left swing
    }
    result = identify._identify_arm_sync(["/dev/still", "/dev/moving"], timeout_s=5.0)
    assert result["success"] is True
    assert result["port"] == "/dev/moving"
    assert result["skipped"] == []
    # Every opened bus is released, and never with a torque write.
    assert len(fake_bus.instances) == 2
    for bus in fake_bus.instances:
        assert bus.disconnect_calls == [{"disable_torque": False}]


def test_sync_skips_unopenable_ports_but_still_runs(fake_bus: type[FakeBus]) -> None:
    fake_bus.fail_connect = {"/dev/busy"}
    fake_bus.scripts = {"/dev/ok": [1000, 1200, 800]}
    result = identify._identify_arm_sync(["/dev/busy", "/dev/ok"], timeout_s=5.0)
    assert result["success"] is True
    assert result["port"] == "/dev/ok"
    assert result["skipped"] == ["/dev/busy"]


def test_sync_no_motion_times_out_with_message(fake_bus: type[FakeBus]) -> None:
    fake_bus.scripts = {"/dev/still": [2000]}
    result = identify._identify_arm_sync(["/dev/still"], timeout_s=0.2)
    assert result["success"] is False
    assert "swing its base" in result["message"]
    assert fake_bus.instances[0].disconnect_calls == [{"disable_torque": False}]


def test_sync_all_ports_unopenable_reports_failure(fake_bus: type[FakeBus]) -> None:
    fake_bus.fail_connect = {"/dev/a", "/dev/b"}
    fake_bus.scripts = {}
    result = identify._identify_arm_sync(["/dev/a", "/dev/b"], timeout_s=1.0)
    assert result["success"] is False
    assert result["skipped"] == ["/dev/a", "/dev/b"]


# ---------------------------------------------------------------------------
# identify_arm_by_motion + /identify-arm endpoint (sync worker mocked)
# ---------------------------------------------------------------------------


async def test_identify_with_no_ports_anywhere_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identify, "find_available_ports", lambda: [])
    result = await identify.identify_arm_by_motion(None)
    assert result["success"] is False
    assert "No arm ports detected" in result["message"]


async def test_identify_defaults_to_detected_ports_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, list[str]] = {}

    def fake_sync(ports: list[str]) -> dict:
        seen["ports"] = ports
        return {"success": True, "port": ports[0], "message": "ok", "skipped": []}

    monkeypatch.setattr(identify, "find_available_ports", lambda: ["/dev/a", "/dev/b"])
    monkeypatch.setattr(identify, "_identify_arm_sync", fake_sync)

    result = await identify.identify_arm_by_motion(["  ", ""])  # blank → fall back to detected
    assert result["success"] is True
    assert seen["ports"] == ["/dev/a", "/dev/b"]

    result = await identify.identify_arm_by_motion(["/dev/x", "/dev/x", " /dev/y "])
    assert seen["ports"] == ["/dev/x", "/dev/y"]


def test_identify_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "makermodslab.identify._identify_arm_sync",
        lambda ports: {
            "success": True,
            "port": ports[-1],
            "message": f"Detected motion on {ports[-1]}.",
            "skipped": [],
        },
    )
    response = client.post("/identify-arm", json={"ports": ["/dev/fake1", "/dev/fake2"]})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["port"] == "/dev/fake2"
    assert "/dev/fake2" in body["message"]


def test_identify_endpoint_defaults_to_all_detected_ports(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("makermodslab.identify.find_available_ports", lambda: ["/dev/fakeA"])
    monkeypatch.setattr(
        "makermodslab.identify._identify_arm_sync",
        lambda ports: {"success": True, "port": ports[0], "message": "ok", "skipped": []},
    )
    response = client.post("/identify-arm", json={})
    assert response.status_code == 200
    assert response.json()["port"] == "/dev/fakeA"


def test_identify_endpoint_reports_hardware_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ports: list[str]) -> dict:
        raise RuntimeError("serial exploded")

    monkeypatch.setattr("makermodslab.identify._identify_arm_sync", boom)

    response = client.post("/identify-arm", json={"ports": ["/dev/fake"]})
    # Logical failures stay HTTP 200 with success=False (like wiggle).
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "serial exploded" in body["message"]
