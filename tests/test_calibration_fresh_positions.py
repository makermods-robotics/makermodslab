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
"""Zero-pose failures must survive polling and must never save cached positions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from makermodslab.arms import registry
from makermodslab.step_calibrate import StepCalibrationStatus


@pytest.fixture
def manager(monkeypatch):
    manager = SimpleNamespace(family=registry.get("metal"))
    manager.device = SimpleNamespace(
        config=SimpleNamespace(motor_can_ids={"shoulder": 1, "gripper": 2}),
        bus=Mock(),
    )
    manager.device.bus.read.return_value = 0.0
    monkeypatch.setattr(manager.family, "_set_zero", Mock())
    monkeypatch.setattr(manager.family, "_build_calibration", Mock(return_value={"shoulder": "calibration"}))
    return manager


def test_power_loss_before_zero_does_not_set_zero_or_save(manager):
    manager.device.bus.read.side_effect = ConnectionError("No response from motor 'shoulder'")
    manager.family._set_zero = Mock()
    with pytest.raises(ConnectionError, match="No response"):
        manager.family.calibrate(manager.device, "robot", Mock())
    manager.family._set_zero.assert_not_called()
    manager.family._build_calibration.assert_not_called()


def test_power_loss_during_zero_does_not_save(manager):
    manager.device.bus.read.side_effect = [5.0, 10.0, 5.0, 10.0] + [ConnectionError("No response")] * 3
    manager.family._set_zero = Mock()
    with pytest.raises(ConnectionError, match="No response"):
        manager.family.calibrate(manager.device, "robot", Mock())
    manager.family._set_zero.assert_called_once()
    manager.family._build_calibration.assert_not_called()


def test_leader_fresh_near_zero_angle_is_not_a_connection_failure(manager):
    manager.device.config = SimpleNamespace(joint_ids={"shoulder": 1})
    manager.device.bus.read_raw_angle.return_value = 0.1
    manager.device.bus.sync_monitor.return_value = {1: SimpleNamespace(reliable=False, angle_deg=0.1)}
    assert manager.family._read_fresh_positions(manager.device) == {"shoulder": 0.1}
    manager.device.bus.sync_monitor.assert_not_called()


def test_leader_missing_reply_still_fails_with_the_joint_name(manager):
    manager.device.config = SimpleNamespace(joint_ids={"gripper": 6})
    manager.device.bus.read_raw_angle.side_effect = RuntimeError("Timeout")
    with pytest.raises(ConnectionError, match="No response from motor 'gripper'"):
        manager.family._read_fresh_positions(manager.device)
    assert manager.device.bus.read_raw_angle.call_count == 3


def test_transient_follower_read_failure_recovers(manager):
    manager.device.bus.read.side_effect = [ConnectionError("No response"), 0.1, 0.2]
    assert manager.family._read_fresh_positions(manager.device) == {"shoulder": 0.1, "gripper": 0.2}


def test_metal_calibration_connects_without_enabling_handshake(manager):
    manager.family._open_can_bus_torque_off(manager.device, "test")
    manager.device.bus.connect.assert_called_once_with(handshake=False)
    manager.device.bus.disable_torque.assert_called_once()


def test_fresh_responses_build_calibration(manager):
    assert manager.family.calibrate(manager.device, "robot", Mock()) == {"shoulder": "calibration"}
    assert manager.device.bus.read.call_count == 6  # before posing, before zero, after zero


@pytest.mark.parametrize("arm_type", ["maker", "metal"])
@pytest.mark.parametrize("status", ["completed", "error"])
def test_status_keeps_zero_result_after_hardware_is_released(client, monkeypatch, arm_type, status):
    from makermodslab import server

    monkeypatch.setattr(
        server.step_calibration_manager,
        "status",
        StepCalibrationStatus(status=status, error="No response" if status == "error" else None),
    )
    response = client.get(f"/api/v1/calibration-status?arm_type={arm_type}")
    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["calibration_active"] is False
    assert response.json()["error"] == ("No response" if status == "error" else None)
