from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from starlette.requests import Request

from makermodslab.api_errors import ApiError
from makermodslab.remote_teleop.api_service import RemoteTeleoperationRuntimeService
from makermodslab.remote_teleop.config import RemoteRoleConfigStore
from makermodslab.remote_teleop.contracts import ActionSample, encode_action


@pytest.fixture(autouse=True)
def isolated_remote_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never let an API test inspect or mutate the developer's role config."""
    from makermodslab import server

    runtime = RemoteTeleoperationRuntimeService(
        config_store=RemoteRoleConfigStore(tmp_path / "private-remote-runtime")
    )
    monkeypatch.setattr(server, "remote_teleoperation_runtime", runtime)
    yield runtime
    runtime.shutdown()


def start_body():
    return {
        "source_id": "operator-test",
        "rig_id": "rig-test",
        "rig_digest": "a" * 64,
        "leader_calibration_id": "leader-test",
        "leader_calibration_digest": "b" * 64,
        "follower_calibration_id": "follower-test",
        "follower_calibration_digest": "c" * 64,
        "joint_names": ["joint_a", "joint_b"],
        "units": ["rad", "rad"],
        "limits": {
            "joint_a": {
                "minimum": -1,
                "maximum": 1,
                "max_velocity_per_s": 2,
                "max_acceleration_per_s2": 20,
            },
            "joint_b": {
                "minimum": -1,
                "maximum": 1,
                "max_velocity_per_s": 2,
                "max_acceleration_per_s2": 20,
            },
        },
        "watchdog_ms": 2000,
    }


def test_remote_simulation_api_is_authenticated_and_hardware_disabled(client) -> None:
    started = client.post("/api/v1/arms/remote-teleoperation/simulations", json=start_body())
    assert started.status_code == 201, started.text
    payload = started.json()
    assert payload["simulation_only"] is True
    grant = payload["session"]
    credentials = payload["credentials"]

    now = time.monotonic_ns()
    action = ActionSample(
        session_id=grant["session_id"],
        source_id=grant["source_id"],
        executor_generation=grant["executor_generation"],
        rig_id="rig-test",
        rig_digest="a" * 64,
        leader_calibration_id="leader-test",
        leader_calibration_digest="b" * 64,
        follower_calibration_id="follower-test",
        follower_calibration_digest="c" * 64,
        sequence=1,
        source_monotonic_ns=now,
        expires_at_source_monotonic_ns=now + 250_000_000,
        joint_names=("joint_a", "joint_b"),
        units=("rad", "rad"),
        positions=(0.5, -0.5),
    )
    raw = encode_action(
        action,
        key_id=credentials["key_id"],
        key=base64.b64decode(credentials["action_key_base64"]),
    )
    submitted = client.post(
        f"/api/v1/arms/remote-teleoperation/simulations/{grant['session_id']}/actions",
        json={"datagram_base64": base64.b64encode(raw).decode("ascii")},
    )
    assert submitted.status_code == 200, submitted.text
    status = client.get("/api/v1/arms/remote-teleoperation").json()
    assert status["live_hardware_enabled"] is False
    assert status["state"] == "active"

    stopped = client.post(
        f"/api/v1/arms/remote-teleoperation/simulations/{grant['session_id']}/stop",
        json={"reason": "test_complete"},
    )
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"]["authority"]["state"] == "idle"


def test_remote_role_configuration_is_dormant_and_redacted(client, isolated_remote_runtime) -> None:
    before = client.get("/api/v1/arms/remote-teleoperation")
    assert before.status_code == 200
    assert before.json()["state"] == "unconfigured"
    assert isolated_remote_runtime.robot._loop.active is False
    assert isolated_remote_runtime.operator._loop.active is False

    saved = client.put(
        "/api/v1/arms/remote-teleoperation/configuration",
        json={
            "role": "operator",
            "robot": None,
            "operator": {
                "node_id": "operator-test",
                "robot_id": "robot-test",
                "leader_robot_name": "saved-so101-leader",
                "control_uri": "wss://100.64.0.2:7443",
                "certificate_fingerprint": "aa" * 32,
                "action_rate_hz": 50,
            },
        },
    )
    assert saved.status_code == 200, saved.text
    payload = saved.json()
    assert payload["configured"] is True
    assert payload["role"] == "operator"
    assert payload["runtime_enabled"] is False
    assert payload["live_hardware_enabled"] is False
    assert payload["state"] == "idle"
    assert isolated_remote_runtime.robot._loop.active is False
    assert isolated_remote_runtime.operator._loop.active is False

    cleared = client.delete("/api/v1/arms/remote-teleoperation/configuration")
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["configured"] is False
    assert not (isolated_remote_runtime.config_store.root / "role-config.json").exists()


def test_remote_configuration_requires_exactly_one_role(client) -> None:
    refused = client.put(
        "/api/v1/arms/remote-teleoperation/configuration",
        json={"role": "robot", "robot": None, "operator": None},
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "request.validation"


def test_commissioning_and_recovery_require_every_physical_safeguard(client) -> None:
    incomplete = {
        "arm_secured": True,
        "workspace_clear": True,
        "physical_power_cutoff_reachable": True,
        "acknowledge_live_torque_enable_risk": False,
    }
    for suffix in ("commission", "recover-hardware"):
        refused = client.post(
            f"/api/v1/arms/remote-teleoperation/{suffix}",
            json=incomplete,
        )
        assert refused.status_code == 422
        assert refused.json()["code"] == "request.validation"


def test_remote_management_is_local_only_but_stop_route_is_not_guarded() -> None:
    from makermodslab.server import _require_local_remote_request

    local = Request({"type": "http", "client": ("127.0.0.1", 1000), "headers": []})
    _require_local_remote_request(local)

    remote = Request({"type": "http", "client": ("100.64.0.2", 1000), "headers": []})
    with pytest.raises(ApiError) as refused:
        _require_local_remote_request(remote)
    assert refused.value.status_code == 403


def test_servo_health_api_is_read_only_and_unavailable_without_owner(client) -> None:
    response = client.get("/api/v1/arms/servo-health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["read_only"] is True
    assert payload["available"] is False
    assert payload["maintenance"]["state"] == "disabled"
