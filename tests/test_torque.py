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
"""Tests for makermodslab.torque — the shared per-bus torque release helper."""

from __future__ import annotations


class _FakeBus:
    """Motor bus double (mirrors tests/test_teleoperate.py's _FakeBus)."""

    def __init__(self, port: str = "COM_FAKE", failing: tuple[str, ...] = ()) -> None:
        self.port = port
        self.motors = {"shoulder_pan": 1, "elbow_flex": 3, "gripper": 6}
        self.failing = set(failing)
        self.disabled: list[tuple[str, int]] = []

    def disable_torque(self, motor: str, num_retry: int = 0) -> None:
        if motor in self.failing:
            raise ConnectionError(f"no response from {motor}")
        self.disabled.append((motor, num_retry))


def test_force_disable_bus_torque_disables_every_motor() -> None:
    from makermodslab.torque import force_disable_bus_torque

    bus = _FakeBus()
    problems = force_disable_bus_torque(bus, "auto-calibration arm")

    assert problems == []
    # Every motor is disabled individually, with retries.
    assert [motor for motor, _ in bus.disabled] == list(bus.motors)
    assert all(num_retry == 5 for _, num_retry in bus.disabled)


def test_force_disable_bus_torque_reports_failed_motor_and_port() -> None:
    """One bad motor must not stop the others from being released, and the
    problem message must be unmistakable: it names the port and warns that
    torque may still be enabled (the arm stays rigid until power is pulled).
    """
    from makermodslab.torque import force_disable_bus_torque

    bus = _FakeBus(port="COM_ARM", failing=("elbow_flex",))
    problems = force_disable_bus_torque(bus, "auto-calibration arm")

    assert len(problems) == 1
    assert "TORQUE MAY STILL BE ENABLED" in problems[0]
    assert "COM_ARM" in problems[0]
    assert "elbow_flex" in problems[0]
    # The remaining motors were still disabled despite the failure.
    assert [motor for motor, _ in bus.disabled] == ["shoulder_pan", "gripper"]
