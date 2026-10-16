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
"""The TEMPORARY Damiao frame trace (can_trace.py): off by default, alarms when on."""

import json
import logging

import can
import pytest

from makermodslab import can_trace


def _bus():
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.damiao import DamiaoMotorsBus

    motors = {}
    for index, name in enumerate(("wrist_flex", "gripper"), start=4):
        motor = Motor(index, "dm4310", MotorNormMode.DEGREES)
        motor.recv_id = 0x10 + index
        motor.motor_type_str = "dm4310"
        motors[name] = motor
    return DamiaoMotorsBus(port="gs_usb:T", motors=motors, use_can_fd=False, data_bitrate=None)


class Inner:
    def __init__(self):
        self.replies = []
        self.closed = False

    def send(self, msg, timeout=None):
        return None

    def recv(self, timeout=None):
        return self.replies.pop(0) if self.replies else None

    def shutdown(self):
        self.closed = True


def _state(recv_id, status):
    return can.Message(
        arbitration_id=recv_id,
        is_extended_id=False,
        data=bytes([(status << 4) | (recv_id - 0x10), 0x80, 0x00, 0x80, 0x08, 0x00, 30, 28]),
    )


def test_off_unless_opted_in(monkeypatch):
    monkeypatch.delenv(can_trace.TRACE_ENV, raising=False)
    bus = _bus()
    bus.canbus = inner = Inner()
    can_trace.wrap(bus)
    assert bus.canbus is inner


def test_fault_and_full_silence_raise_alarms_and_frames_reach_the_file(monkeypatch, caplog):
    monkeypatch.setenv(can_trace.TRACE_ENV, "1")
    bus = _bus()
    bus.canbus = inner = Inner()
    can_trace.wrap(bus)
    trace = bus.canbus
    assert isinstance(trace, can_trace.CanTrace)

    caplog.set_level(logging.INFO, logger="makermodslab.can_trace")
    inner.replies = [_state(0x14, 1), _state(0x14, 9)]
    assert trace.recv(0.01).arbitration_id == 0x14
    trace.recv(0.01)
    assert any("wrist_flex FAULT UNDERVOLTAGE" in r.getMessage() for r in caplog.records)

    clock = [1000.0]
    monkeypatch.setattr(can_trace.time, "monotonic", lambda: clock[0])
    command = can.Message(arbitration_id=4, is_extended_id=False, data=bytes(8))
    gripper = can.Message(arbitration_id=5, is_extended_id=False, data=bytes(8))
    for _ in range(3):
        trace.send(command)
        trace.send(gripper)
        clock[0] += 0.1
    assert any("ALL motors silent" in r.getMessage() for r in caplog.records)

    trace.shutdown()
    assert inner.closed
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert any(r.get("status_name") == "UNDERVOLTAGE" for r in rows)
    assert any(r.get("dir") == "tx" and r.get("motor") == "gripper" and r.get("cmd") == "mit" for r in rows)
    assert rows[-1] == {"event": "close"}


@pytest.mark.parametrize("value", ["", "0", "false"])
def test_falsy_values_keep_it_off(monkeypatch, value):
    monkeypatch.setenv(can_trace.TRACE_ENV, value)
    assert not can_trace.enabled()
