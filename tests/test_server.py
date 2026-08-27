# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for makermodslab.server — FastAPI app and ConnectionManager."""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import makermodslab.server as server_mod
from makermodslab.utils import config as cfg

# A browser sends an Accept header that prefers HTML on navigations/hard-reloads.
BROWSER_ACCEPT = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

REQUIRED_PATHS = {
    "/health",
    "/skills",
    "/move-arm",
    "/stop-teleoperation",
    "/teleoperation-status",
    "/start-recording",
    "/stop-recording",
    "/recording-status",
    "/start-calibration",
    "/stop-calibration",
    "/calibration-status",
    "/datasets",
    "/jobs",
    "/available-ports",
    "/available-cameras",
    "/hf-auth-status",
    "/ws/joint-data",
}


def test_app_exposes_required_endpoints() -> None:
    from makermodslab.server import app

    paths = {route.path for route in app.routes}
    missing = REQUIRED_PATHS - paths
    assert not missing, f"missing routes: {missing}"


def test_shutdown_stops_active_teleoperation(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI's shutdown handler must wait for an in-flight teleoperation
    session to actually finish releasing the arm(s), not just flip the flag
    and move on.

    teleoperation_thread runs INSIDE this process (not a subprocess, unlike
    inference's rollout or auto-calibration's vendored script) — a plain kill
    or `--reload` restart kills it mid-loop unless something signals it AND
    waits for the result. handle_stop_teleoperation()'s first call is
    fire-and-forget by design (see teleoperate.stop_and_wait), so calling it
    alone from shutdown would let the process exit while the worker is still
    mid-return, with no return-to-rest and no torque release."""
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    released = threading.Event()

    def _worker() -> None:
        while teleop.teleoperation_active:
            time.sleep(0.01)
        released.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)
    teleop._release_now.clear()
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(server_mod, "manager", None)
    worker.start()

    asyncio.run(server_mod.shutdown_event())

    assert released.is_set(), "shutdown returned without waiting for teleoperation to finish releasing"
    assert teleop.teleoperation_active is False
    worker.join(timeout=2.0)


def test_shutdown_stops_active_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server restart mid-replay must wait for the worker to actually
    finish releasing the arm — same class of bug I8 fixed for teleoperation
    (a plain kill or --reload restart otherwise orphans the in-process
    worker thread mid-motion, with no return-to-rest and no torque release)."""
    import makermodslab.record as record
    import makermodslab.replay as replay
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    released = threading.Event()

    def _worker() -> None:
        while replay.replay_active:
            time.sleep(0.01)
        released.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(replay, "replay_active", True)
    monkeypatch.setattr(replay, "replay_thread", worker)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(server_mod, "manager", None)
    worker.start()

    asyncio.run(server_mod.shutdown_event())

    assert released.is_set(), "shutdown returned without waiting for replay to finish releasing"
    assert replay.replay_active is False
    worker.join(timeout=2.0)


def test_shutdown_stops_active_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same requirement as teleoperation, for recording_thread.

    handle_stop_recording() signals via recording_events (stop_recording /
    exit_early) rather than flipping recording_active itself — the real
    recording worker notices those events, winds down, and clears the flag as
    part of its own cleanup. The fake worker here mirrors that division of
    responsibility instead of short-circuiting it, so this test exercises the
    real signal shutdown actually sends."""
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    released = threading.Event()
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}

    def _worker() -> None:
        while not events["stop_recording"]:
            time.sleep(0.01)
        record.recording_active = False
        released.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "releasing", False)
    record._release_now.clear()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(server_mod, "manager", None)
    worker.start()

    asyncio.run(server_mod.shutdown_event())

    assert released.is_set(), "shutdown returned without waiting for recording to finish releasing"
    assert events["stop_recording"] is True
    assert record.recording_active is False
    worker.join(timeout=2.0)


def test_shutdown_stops_active_auto_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI's shutdown handler must terminate an in-flight auto-calibration
    subprocess and release the arm's torque, not just clean up the broadcast
    thread.

    Auto-calibration drives the arm under torque via its own subprocess,
    independent of this server process, and on success writes servo EEPROM.
    Without this, `--reload` or a plain PID kill during a run leaves that
    subprocess orphaned with the arm potentially still energized and nobody
    able to stop it from the API. Uses a real (fake) Popen so the actual
    SIGTERM -> wait -> torque-release sequence runs end to end through
    shutdown_event(), the same way test_shutdown_stops_active_inference
    exercises the inference path."""
    from makermodslab import auto_calibrate as ac

    class _FakeAutocalProc:
        def __init__(self) -> None:
            self._dead = threading.Event()
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self._dead.set()

        def wait(self, timeout: float | None = None) -> int:
            if not self._dead.wait(timeout):
                raise TimeoutError
            return 0

    proc = _FakeAutocalProc()
    released: list[str] = []
    monkeypatch.setattr(ac, "_release_arm_torque", lambda port: (released.append(port), [])[1])
    monkeypatch.setattr(ac, "_STOP_GRACE_S", 0.2)
    monkeypatch.setattr(ac, "_STOP_KILL_WAIT_S", 0.2)
    monkeypatch.setattr(ac, "_READER_JOIN_S", 0.2)

    mgr = ac.auto_calibration_manager
    monkeypatch.setattr(mgr, "status", ac.AutoCalibrationStatus(active=True, status="running"))
    monkeypatch.setattr(mgr, "_proc", proc)
    monkeypatch.setattr(
        mgr,
        "_request",
        ac.AutoCalibrationRequest(device_type="robot", port="/dev/arm", config_file="test_arm"),
    )
    monkeypatch.setattr(mgr, "_thread", None)
    # Broadcast-thread cleanup isn't under test here.
    monkeypatch.setattr(server_mod, "manager", None)

    asyncio.run(server_mod.shutdown_event())

    assert proc.terminated, "shutdown did not terminate the in-flight auto-calibration subprocess"
    assert released == ["/dev/arm"], "shutdown did not release the arm's torque"
    assert mgr.status.active is False


def test_shutdown_stops_active_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI's shutdown handler must terminate an in-flight inference
    subprocess, not just the broadcast thread.

    Without this, `--reload` (uvicorn kills and respawns the worker process
    on a file change) or a plain PID kill leaves the `lerobot-rollout` child
    — which is actively driving the follower under a policy — orphaned and
    running with nobody supervising it, since the parent that would have
    stopped it is already gone.

    Calls shutdown_event() directly (matches the asyncio.run(mgr.connect(...))
    pattern already used in this file) instead of relying on TestClient's
    lifespan + monkeypatch fixture teardown ordering, which isn't guaranteed
    to leave the patched state in place by the time the shutdown fires."""
    from makermodslab import rollout

    terminate_calls: list[bool] = []

    class _FakeProc:
        def terminate(self) -> None:
            terminate_calls.append(True)

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _FakeProc())
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})
    # Broadcast-thread cleanup isn't under test here.
    monkeypatch.setattr(server_mod, "manager", None)

    asyncio.run(server_mod.shutdown_event())

    assert terminate_calls, "shutdown did not terminate the in-flight inference subprocess"
    assert rollout.inference_active is False


def test_health_endpoint_returns_200_with_json_object(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_skills_endpoint_returns_an_envelope_not_a_bare_array(client: TestClient) -> None:
    """/skills answers with {skills, hub}, deliberately unlike /models' array.

    The envelope exists so a caller can tell "the Hub was unreachable" from
    "you own no skills" — two states that used to reach the UI as the same empty
    list, which made an outage read as the user's models having been deleted."""
    from unittest.mock import patch

    with patch(
        "makermodslab.models.list_skills",
        return_value={
            "skills": [],
            "hub": {"ok": False, "authenticated": True, "degraded": True, "stale_rows": True},
        },
    ):
        response = client.get("/skills")

    assert response.status_code == 200
    body = response.json()
    assert body["skills"] == []
    assert body["hub"]["ok"] is False
    assert body["hub"]["degraded"] is True


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/this-does-not-exist")
    assert response.status_code == 404


@pytest.mark.parametrize("unsafe_name", ["evil..name", "..config", "back\\door"])
def test_delete_calibration_config_rejects_unsafe_name(client: TestClient, unsafe_name: str) -> None:
    """A config name with path-traversal characters is rejected before any
    filesystem access — distinct from the "not found" path, so the guard is
    proven to fire. The validator also blocks "/" and "\\"."""
    response = client.delete(f"/calibration-configs/teleop/{unsafe_name}")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "Invalid configuration name" in body["message"]


def test_delete_in_use_calibration_config_unassigns_robots(
    client: TestClient, tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting an in-use config is ALLOWED; the referencing robots are
    unassigned (arm returns to "needs calibration") and reported back."""
    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    # server.py binds LEADER_CONFIG_PATH at import; repoint it at the tmp dir.
    monkeypatch.setattr(server_mod, "LEADER_CONFIG_PATH", cfg.LEADER_CONFIG_PATH)

    config_file = Path(cfg.LEADER_CONFIG_PATH) / "mycal.json"
    config_file.write_text("{}")
    cfg.save_robot_record("armA", {"mode": "single", "leader_config": "mycal"}, allow_create=True)

    resp = client.delete("/calibration-configs/teleop/mycal")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["unassigned"] == [{"robot": "armA", "fields": ["leader_config"]}]
    assert "armA" in body["message"]
    # The file is gone (this dir IS where lerobot loads calibrations from, so
    # no stale copy can keep working) and the record is unassigned + dirty.
    assert not config_file.exists()
    record = cfg.get_robot_record("armA")
    assert record["leader_config"] == ""
    assert cfg.is_robot_record_clean(record) is False


def test_delete_unused_calibration_config_reports_no_unassignments(
    client: TestClient, tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_mod, "LEADER_CONFIG_PATH", cfg.LEADER_CONFIG_PATH)
    config_file = Path(cfg.LEADER_CONFIG_PATH) / "spare.json"
    config_file.write_text("{}")

    resp = client.delete("/calibration-configs/teleop/spare")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["unassigned"] == []
    assert not config_file.exists()


def test_upsert_robot_rejects_same_side_config_conflict(client: TestClient, tmp_lerobot_home) -> None:
    """Assigning one config to both same-side arms of a bimanual robot is a 409."""
    client.post(
        "/robots/bi?create=true",
        json={"mode": "bimanual", "leader_config": "L1", "right_leader_config": "L2"},
    )
    # right leader = left leader -> conflict.
    resp = client.post("/robots/bi", json={"right_leader_config": "L1"})
    assert resp.status_code == 409
    assert "leader" in resp.json()["message"]

    # A non-slot edit (cameras) is never blocked.
    assert client.post("/robots/bi", json={"cameras": []}).status_code == 200


def test_upsert_robot_rejects_shared_port(client: TestClient, tmp_lerobot_home) -> None:
    """Two arms can't share a serial port (each is its own USB device)."""
    client.post("/robots/p?create=true", json={"leader_port": "/dev/a"})
    # follower on the same port as leader -> 409.
    resp = client.post("/robots/p", json={"follower_port": "/dev/a"})
    assert resp.status_code == 409
    assert "/dev/a" in resp.json()["message"]
    # A distinct port is fine.
    assert client.post("/robots/p", json={"follower_port": "/dev/b"}).status_code == 200


def test_upsert_robot_clears_port_with_empty_string(client: TestClient, tmp_lerobot_home) -> None:
    """Posting an empty-string port releases the assignment (disconnect without
    reconnecting), and two cleared ports never count as a shared-port conflict."""
    client.post("/robots/d?create=true", json={"leader_port": "/dev/a", "follower_port": "/dev/b"})

    resp = client.post("/robots/d", json={"leader_port": ""})
    assert resp.status_code == 200
    assert resp.json()["robot"]["leader_port"] == ""

    # Clearing the other arm too must not trip the duplicate-port guard.
    resp = client.post("/robots/d", json={"follower_port": ""})
    assert resp.status_code == 200
    assert resp.json()["robot"]["follower_port"] == ""

    # A cleared port doesn't block re-assigning that port to the other arm.
    assert client.post("/robots/d", json={"leader_port": "/dev/b"}).status_code == 200


@pytest.mark.parametrize("mode", ["single", "bimanual"])
def test_create_robot_accepts_mode(client: TestClient, tmp_lerobot_home, mode: str) -> None:
    """Mode is established at creation for both values."""
    resp = client.post(f"/robots/created_{mode}?create=true", json={"mode": mode})
    assert resp.status_code == 200
    assert resp.json()["robot"]["mode"] == mode


def test_upsert_robot_rejects_mode_change_on_existing_record(client: TestClient, tmp_lerobot_home) -> None:
    """Mode is fixed at creation. A patch that flips the stored mode is a 409;
    creating a new robot is the migration path instead."""
    client.post("/robots/fixed?create=true", json={"mode": "single"})

    resp = client.post("/robots/fixed", json={"mode": "bimanual"})
    assert resp.status_code == 409
    assert "fixed at creation" in resp.json()["message"]
    # The stored mode is untouched by the rejected patch.
    assert client.get("/robots/fixed").json()["robot"]["mode"] == "single"


def test_upsert_robot_allows_same_mode_echo(client: TestClient, tmp_lerobot_home) -> None:
    """Calibration write-backs echo the full record (including its current
    mode); a same-value mode in the body must stay a no-op, not a 409."""
    client.post("/robots/echo?create=true", json={"mode": "bimanual"})

    # Echo the existing mode alongside a real edit — must succeed.
    resp = client.post("/robots/echo", json={"mode": "bimanual", "leader_port": "/dev/a"})
    assert resp.status_code == 200
    robot = resp.json()["robot"]
    assert robot["mode"] == "bimanual"
    assert robot["leader_port"] == "/dev/a"


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Build a LogRecord shaped like uvicorn.access emits:
    args = (client_addr, method, full_path, http_version, status_code)."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", method, path, "1.1", status),
        exc_info=None,
    )


def test_status_poll_access_filter_drops_only_successful_status_gets() -> None:
    """The uvicorn.access filter silences ~2 Hz status polls but keeps errors,
    writes, and every other path."""
    f = server_mod._StatusPollAccessFilter()

    # High-frequency polls with 2xx are dropped (query string ignored).
    assert f.filter(_access_record("GET", "/teleoperation-status", 200)) is False
    assert f.filter(_access_record("GET", "/auto-calibration-status", 200)) is False
    assert f.filter(_access_record("GET", "/jobs?limit=20", 200)) is False

    # Errors on those same paths must still log.
    assert f.filter(_access_record("GET", "/recording-status", 500)) is True
    assert f.filter(_access_record("GET", "/jobs", 404)) is True

    # Writes and non-status paths are untouched.
    assert f.filter(_access_record("POST", "/jobs/training", 201)) is True
    assert f.filter(_access_record("GET", "/health", 200)) is True
    # Subpaths of /jobs (log tails, checkpoints) are NOT silenced.
    assert f.filter(_access_record("GET", "/jobs/abc123/logs", 200)) is True

    # Records that don't look like uvicorn access lines pass through.
    other = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain", None, None)
    assert f.filter(other) is True


def test_policy_optimizer_defaults_reports_availability(client: TestClient) -> None:
    """`available` marks which policy types this lerobot pin can construct.
    act must work everywhere; reward_classifier registers under lerobot's
    rewards registry (not the policy registry) in this pin, so it's out."""
    data = client.get("/policy-optimizer-defaults").json()
    assert set(data["available"]) == set(data["defaults"])
    assert data["available"]["act"] is True
    assert data["defaults"]["act"] is not None
    assert data["available"]["pi0_fast"] is True
    assert data["available"]["pi05"] is True
    assert data["available"]["reward_classifier"] is False
    assert data["defaults"]["reward_classifier"] is None


@pytest.mark.parametrize("unsafe_name", ["evil..name", "..config", "back\\door"])
def test_download_calibration_config_rejects_unsafe_name(client: TestClient, unsafe_name: str) -> None:
    response = client.get(f"/calibration-configs/teleop/{unsafe_name}/download")
    assert response.status_code == 400
    assert "Invalid configuration name" in response.json()["message"]


def test_download_calibration_config_rejects_bad_device_type(client: TestClient) -> None:
    response = client.get("/calibration-configs/bogus/arm/download")
    assert response.status_code == 400


def test_download_calibration_config_returns_file(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leader config downloads byte-for-byte as a raw JSON attachment."""
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    (leader_dir / "armA.json").write_text('{"shoulder_pan": {"id": 1}}')
    # server.py binds its own LEADER_CONFIG_PATH at import — patch that one.
    monkeypatch.setattr("makermodslab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/armA/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="armA.json"'
    assert response.json() == {"shoulder_pan": {"id": 1}}


def test_download_calibration_config_accepts_dot_json_suffix(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Robot records store config names with the .json extension; passing that
    form must resolve to the file, not "<name>.json.json"."""
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    (leader_dir / "so101.json").write_text('{"shoulder_pan": {"id": 1}}')
    monkeypatch.setattr("makermodslab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/so101.json/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="so101.json"'


def test_download_calibration_config_missing_returns_404(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    monkeypatch.setattr("makermodslab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/nope/download")
    assert response.status_code == 404


_GOOD_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1927, "range_min": 741, "range_max": 3472},
}


def test_upload_calibration_config_rejects_bad_device_type(client: TestClient) -> None:
    response = client.post("/calibration-configs/bogus/upload", json={"name": "x", "data": _GOOD_CALIBRATION})
    assert response.status_code == 400


def test_upload_calibration_config_rejects_malformed_data(client: TestClient) -> None:
    response = client.post("/calibration-configs/teleop/upload", json={"name": "x", "data": {"m": {"id": 1}}})
    assert response.status_code == 400
    assert "missing" in response.json()["message"]


def test_upload_calibration_config_writes_then_409_on_collision(client: TestClient, tmp_lerobot_home) -> None:
    """First upload writes; a second under the same name is rejected (no overwrite)."""
    first = client.post(
        "/calibration-configs/teleop/upload", json={"name": "armA", "data": _GOOD_CALIBRATION}
    )
    assert first.status_code == 200
    assert first.json()["name"] == "armA"

    second = client.post(
        "/calibration-configs/teleop/upload", json={"name": "armA", "data": _GOOD_CALIBRATION}
    )
    assert second.status_code == 409


def _spa_mounted(client: TestClient) -> bool:
    return any(getattr(route, "name", None) == "frontend" for route in client.app.routes)


def test_spa_deep_link_serves_index_html(client: TestClient) -> None:
    """A browser hard-reload of a client-side route returns the SPA shell, not a 404."""
    if not _spa_mounted(client):
        pytest.skip("frontend/dist not built; SPA not mounted")
    response = client.get("/recording", headers=BROWSER_ACCEPT)
    assert response.status_code == 200
    assert response.text.lstrip().lower().startswith("<!doctype html")


def test_spa_fallback_does_not_mask_api_404(client: TestClient) -> None:
    """Non-HTML clients (XHR, curl, API typos) still get a real 404, not the SPA shell."""
    response = client.get("/recording", headers={"accept": "application/json"})
    assert response.status_code == 404


def test_spa_fallback_respects_explicit_html_refusal(client: TestClient) -> None:
    """`text/html;q=0` is an explicit refusal — it must not get the SPA shell."""
    response = client.get("/recording", headers={"accept": "application/json,text/html;q=0"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", True),
        ("text/html", True),
        ("text/html;q=0.5", True),
        ("application/json", False),
        ("*/*", False),
        ("", False),
        ("text/html;q=0", False),
        ("application/json,text/html;q=0", False),
        ("text/html;q=bogus", False),
    ],
)
def test_accepts_html(accept: str, expected: bool) -> None:
    from makermodslab.server import _accepts_html

    assert _accepts_html(accept) is expected


def test_connection_manager_tracks_connect_and_disconnect() -> None:
    from makermodslab.server import ConnectionManager

    mgr = ConnectionManager()
    fake_ws = MagicMock()
    fake_ws.accept = AsyncMock()

    asyncio.run(mgr.connect(fake_ws))
    assert fake_ws in mgr.active_connections

    mgr.disconnect(fake_ws)
    assert fake_ws not in mgr.active_connections


def test_connection_manager_broadcast_sync_does_not_block_without_loop() -> None:
    from makermodslab.server import ConnectionManager

    mgr = ConnectionManager()
    # Should enqueue without raising even if there are no consumers.
    mgr.broadcast_joint_data_sync({"shoulder_pan.pos": 1.0})


class _LoopThread:
    """A real asyncio loop on a background thread, standing in for uvicorn's
    event loop in ConnectionManager tests: websockets are accepted on it and
    the broadcast worker must marshal sends back onto it."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=2.0)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)
        self.loop.close()


@pytest.fixture
def ws_loop():
    loop_thread = _LoopThread()
    yield loop_thread
    loop_thread.close()


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _fake_ws(send_json=None) -> MagicMock:
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = send_json if send_json is not None else AsyncMock()
    return ws


def test_broadcast_sends_on_owning_loop_and_survives_dead_connection(ws_loop) -> None:
    """A send failure must drop only that connection — not kill the worker —
    and healthy sends must run on the loop that accepted the websocket
    (regression: 'Task got Future attached to a different loop')."""
    from makermodslab.server import ConnectionManager

    mgr = ConnectionManager()
    seen: dict[str, object] = {}

    async def _record(data):
        seen["loop"] = asyncio.get_running_loop()
        seen["data"] = data

    ws_ok = _fake_ws(send_json=AsyncMock(side_effect=_record))
    ws_dead = _fake_ws(send_json=AsyncMock(side_effect=RuntimeError("client went away")))

    ws_loop.run(mgr.connect(ws_ok))
    ws_loop.run(mgr.connect(ws_dead))
    worker = mgr.broadcast_thread
    try:
        mgr.broadcast_joint_data_sync({"shoulder_pan.pos": 1.0})

        assert _wait_for(lambda: ws_dead not in mgr.active_connections)
        assert ws_ok in mgr.active_connections
        assert seen["loop"] is ws_loop.loop
        assert seen["data"] == {"shoulder_pan.pos": 1.0}
        assert mgr.is_running
        assert worker.is_alive()

        # The surviving connection keeps receiving broadcasts.
        mgr.broadcast_joint_data_sync({"shoulder_pan.pos": 2.0})
        assert _wait_for(lambda: seen.get("data") == {"shoulder_pan.pos": 2.0})
    finally:
        mgr.disconnect(ws_ok)


def test_connection_manager_rapid_reconnect_restarts_worker(ws_loop) -> None:
    """Disconnect-then-reconnect while broadcasts flow (browser reload during
    teleop) must hand off cleanly to a fresh worker with no self-join
    (regression: 'cannot join current thread' killing joint streaming)."""
    from makermodslab.server import ConnectionManager

    mgr = ConnectionManager()
    ws1 = _fake_ws()
    ws2 = _fake_ws()

    ws_loop.run(mgr.connect(ws1))
    first_worker = mgr.broadcast_thread
    mgr.broadcast_joint_data_sync({"n": 1})
    assert _wait_for(lambda: ws1.send_json.call_count >= 1)

    # Last client drops: the worker is signaled to stop but never joined.
    mgr.disconnect(ws1)
    assert not mgr.is_running

    # Immediate reconnect restarts broadcasting on a fresh worker.
    ws_loop.run(mgr.connect(ws2))
    assert mgr.is_running
    second_worker = mgr.broadcast_thread
    assert second_worker is not first_worker

    try:
        mgr.broadcast_joint_data_sync({"n": 2})
        assert _wait_for(lambda: ws2.send_json.call_count >= 1)
        ws2.send_json.assert_called_with({"n": 2})

        # The replaced worker notices it's been superseded and exits on its
        # own even though is_running is True again.
        first_worker.join(timeout=2.0)
        assert not first_worker.is_alive()
    finally:
        mgr.disconnect(ws2)


def _install_fake_pygrabber(monkeypatch: pytest.MonkeyPatch, filter_graph_cls) -> None:
    import sys
    import types

    module = types.ModuleType("pygrabber.dshow_graph")
    module.FilterGraph = filter_graph_cls
    monkeypatch.setitem(sys.modules, "pygrabber", types.ModuleType("pygrabber"))
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", module)


def _install_fake_pygrabber(monkeypatch: pytest.MonkeyPatch, filter_graph_cls) -> None:
    import sys
    import types

    module = types.ModuleType("pygrabber.dshow_graph")
    module.FilterGraph = filter_graph_cls
    monkeypatch.setitem(sys.modules, "pygrabber", types.ModuleType("pygrabber"))
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", module)


def test_windows_cameras_uses_real_directshow_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows path returns pygrabber's real device names in index order so
    the frontend can match each camera to its browser deviceId (issues #12/#16).
    """
    from makermodslab import server

    class _FakeGraph:
        def get_input_devices(self) -> list[str]:
            return ["USB2.0_CAM1", "ASUS FHD webcam"]

    _install_fake_pygrabber(monkeypatch, _FakeGraph)

    assert server._windows_cameras() == [
        {"index": 0, "name": "USB2.0_CAM1", "available": True},
        {"index": 1, "name": "ASUS FHD webcam", "available": True},
    ]


def test_windows_cameras_falls_back_when_pygrabber_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If pygrabber is missing or its COM init fails, enumeration degrades to the
    generic cv2 probe instead of erroring."""
    from makermodslab import server

    class _BoomGraph:
        def __init__(self) -> None:
            raise RuntimeError("DirectShow/COM unavailable")

    _install_fake_pygrabber(monkeypatch, _BoomGraph)
    sentinel = [{"index": 0, "name": "Camera 0", "available": True}]
    monkeypatch.setattr(server, "_generic_cv2_cameras", lambda backend: sentinel)

    assert server._windows_cameras() == sentinel


def test_v4l2_camera_name_reads_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from makermodslab import server

    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO("HD Pro Webcam C920\n"))
    assert server._v4l2_camera_name(0) == "HD Pro Webcam C920"


def test_v4l2_camera_name_returns_none_when_missing() -> None:
    from makermodslab import server

    # No such sysfs node (also the case on non-Linux): graceful None, not error.
    assert server._v4l2_camera_name(999999) is None


def test_import_model_route_returns_record(client, monkeypatch) -> None:
    from makermodslab import server

    fake = {
        "id": "act_imported_x",
        "name": "Imported · model",
        "state": "done",
        "config": {"dataset_repo_id": "(imported)", "policy_type": "act"},
        "output_dir": "/tmp/model",
        "started_at": 1.0,
        "ended_at": 1.0,
        "runner": "imported",
        "hf_repo_id": None,
    }
    from makermodslab.jobs import JobRecord

    # No pre-existing entry for this source → fresh 201 path.
    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: None)
    monkeypatch.setattr(
        server.job_registry,
        "register_imported",
        lambda source, name=None: JobRecord(**fake),
    )
    resp = client.post("/jobs/import", json={"source": "/tmp/model"})
    assert resp.status_code == 201
    assert resp.json()["runner"] == "imported"
    assert "already_imported" not in resp.json()


def test_import_model_route_flags_duplicate_with_200(client, monkeypatch) -> None:
    """Re-importing an already-registered source returns the EXISTING record
    with already_imported=true and a 200 (not 201)."""
    from makermodslab import server
    from makermodslab.jobs import JobRecord

    existing = JobRecord(
        id="act_imported_x",
        name="Imported · model",
        display_name="my alias",
        state="done",
        config={"dataset_repo_id": "(imported)", "policy_type": "act"},
        output_dir="/tmp/model",
        started_at=1.0,
        ended_at=1.0,
        runner="imported",
    )
    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: existing)
    monkeypatch.setattr(
        server.job_registry,
        "register_imported",
        lambda source, name=None: existing,
    )
    resp = client.post("/jobs/import", json={"source": "/tmp/model"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["already_imported"] is True
    assert body["id"] == "act_imported_x"
    assert body["display_name"] == "my alias"  # alias preserved on re-import


def test_import_model_route_maps_value_error_to_400(client, monkeypatch) -> None:
    from makermodslab import server

    def boom(source, name=None):
        raise ValueError("No usable model at '/tmp/x'")

    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: None)
    monkeypatch.setattr(server.job_registry, "register_imported", boom)
    resp = client.post("/jobs/import", json={"source": "/tmp/x"})
    assert resp.status_code == 400
    assert "No usable model" in resp.json()["detail"]


_MINIMAL_RECORDING_REQUEST_BODY = {
    "leader_port": "COM_LEADER",
    "follower_port": "COM_FOLLOWER",
    "leader_config": "leader",
    "follower_config": "follower",
    "dataset_repo_id": "tester/some_dataset",
    "single_task": "pick",
}


def test_start_recording_route_returns_409_when_already_active(client, monkeypatch) -> None:
    """R2 regression: a rejected /start-recording must surface as a real HTTP
    status (so the frontend's `response.ok` check treats it as a failure),
    not the HTTP 200 FastAPI would otherwise default every dict return to."""
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    resp = client.post("/start-recording", json=_MINIMAL_RECORDING_REQUEST_BODY)

    assert resp.status_code == 409
    assert "already active" in resp.json()["detail"]


def test_start_recording_route_returns_409_when_teleoperation_active(client, monkeypatch) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(rollout, "inference_active", False)

    resp = client.post("/start-recording", json=_MINIMAL_RECORDING_REQUEST_BODY)

    assert resp.status_code == 409
    assert "Teleoperation is currently active" in resp.json()["detail"]


def test_start_recording_route_returns_400_for_invalid_dataset_name(client, monkeypatch) -> None:
    import makermodslab.record as record
    import makermodslab.rollout as rollout
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    body = dict(_MINIMAL_RECORDING_REQUEST_BODY, dataset_repo_id="too/many/slashes")
    resp = client.post("/start-recording", json=body)

    assert resp.status_code == 400
    assert "'/'" in resp.json()["detail"]


def test_rename_job_route_returns_updated_record(client, monkeypatch) -> None:
    from makermodslab import server
    from makermodslab.jobs import JobRecord

    fake = {
        "id": "act_ds_x",
        "name": "ACT · user/ds",
        "display_name": "my run",
        "state": "done",
        "config": {"dataset_repo_id": "user/ds", "policy_type": "act"},
        "output_dir": "/tmp/run",
        "started_at": 1.0,
    }
    seen = {}

    def fake_rename(job_id, new_name):
        seen["args"] = (job_id, new_name)
        return JobRecord(**fake)

    monkeypatch.setattr(server.job_registry, "rename", fake_rename)
    resp = client.post("/jobs/act_ds_x/rename", json={"new_name": "my run"})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "my run"
    assert seen["args"] == ("act_ds_x", "my run")


def test_rename_job_route_maps_not_found_to_404(client, monkeypatch) -> None:
    from makermodslab import server
    from makermodslab.jobs import JobNotFoundError

    def boom(job_id, new_name):
        raise JobNotFoundError(job_id)

    monkeypatch.setattr(server.job_registry, "rename", boom)
    resp = client.post("/jobs/nope/rename", json={"new_name": "x"})
    assert resp.status_code == 404


def test_rename_job_route_maps_value_error_to_400(client, monkeypatch) -> None:
    from makermodslab import server

    def boom(job_id, new_name):
        raise ValueError("Display name cannot be empty.")

    monkeypatch.setattr(server.job_registry, "rename", boom)
    resp = client.post("/jobs/act_ds_x/rename", json={"new_name": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


# --- DELETE /jobs/hub/models/{repo_id} -------------------------------------
#
# Deleting an orphaned hub MODEL repo. Uses a mocked shared HfApi so no real
# Hub call is made. The endpoint is scoped to the caller's own namespace and
# treats a missing repo (delete_repo missing_ok=True) as idempotent success.


def _patch_hub_delete(monkeypatch, *, username, api):
    """Point the endpoint at a fake whoami (namespace) and a fake HfApi."""
    monkeypatch.setattr(
        server_mod,
        "cached_whoami",
        lambda: {"name": username} if username else None,
    )
    monkeypatch.setattr(server_mod, "shared_hf_api", lambda: api)


def test_delete_hub_model_success(client: TestClient, monkeypatch) -> None:
    api = MagicMock()
    _patch_hub_delete(monkeypatch, username="makermods", api=api)

    resp = client.delete("/jobs/hub/models/makermods/smolvla_orphan_2026")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["repo_id"] == "makermods/smolvla_orphan_2026"
    api.delete_repo.assert_called_once_with(
        "makermods/smolvla_orphan_2026", repo_type="model", missing_ok=True
    )


def test_delete_hub_model_missing_repo_is_idempotent_success(client: TestClient, monkeypatch) -> None:
    # missing_ok=True means the Hub 404 never surfaces — delete_repo just
    # returns. The endpoint therefore reports success for an already-gone repo.
    api = MagicMock()
    api.delete_repo.return_value = None
    _patch_hub_delete(monkeypatch, username="makermods", api=api)

    resp = client.delete("/jobs/hub/models/makermods/already_gone")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_delete_hub_model_permission_error_is_friendly(client: TestClient, monkeypatch) -> None:
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    response = requests.Response()
    response.status_code = 403
    api = MagicMock()
    api.delete_repo.side_effect = HfHubHTTPError("forbidden", response=response)
    _patch_hub_delete(monkeypatch, username="makermods", api=api)

    resp = client.delete("/jobs/hub/models/makermods/no_write_scope")
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert "write access" in detail


def test_delete_hub_model_refuses_foreign_namespace(client: TestClient, monkeypatch) -> None:
    # The caller is "makermods" but tries to delete a repo under "someoneelse".
    # Refused up front — the Hub is never called.
    api = MagicMock()
    _patch_hub_delete(monkeypatch, username="makermods", api=api)

    resp = client.delete("/jobs/hub/models/someoneelse/their_model")
    assert resp.status_code == 403
    assert "namespace" in resp.json()["detail"]
    api.delete_repo.assert_not_called()


def test_delete_hub_model_unauthenticated_is_401(client: TestClient, monkeypatch) -> None:
    api = MagicMock()
    _patch_hub_delete(monkeypatch, username=None, api=api)

    resp = client.delete("/jobs/hub/models/makermods/whatever")
    assert resp.status_code == 401
    api.delete_repo.assert_not_called()


# --- GET /jobs/hub model listing (tagged + untagged run-repo union) --------
#
# The listing must surface the user's own empty/untagged run repos (orphans a
# crashed cloud run pre-creates) alongside the tagged ones, so the untracked
# cleanup path can reach them. It does this with ONE unfiltered list_models()
# call per author, filtered client-side: a repo qualifies if it carries the
# "lerobot" library tag OR its name matches MakerMods Lab's "_<timestamp>" run-repo
# naming (see _list_author_models). The single unfiltered call replaced an older
# two-pass (filter="lerobot" + unfiltered) approach — half the Hub calls, same
# result set.


class _FakeModel:
    def __init__(self, repo_id, last_modified=None, private=False, tags=None):
        self.id = repo_id
        self.last_modified = last_modified
        self.private = private
        # The `lerobot` library tag is now modeled as a real attribute (the
        # single unfiltered call filters client-side on it), not via a separate
        # filter="lerobot" pass.
        self.tags = list(tags or [])


def _hub_api_with_models(*, all_author):
    """Fake HfApi whose (single, unfiltered) list_models() returns `all_author`
    for any author. list_jobs() returns nothing. The `lerobot` tag lives on each
    model's `.tags`, so the endpoint's client-side filter sees it."""
    api = MagicMock()
    api.list_jobs.return_value = []

    def _list_models(author=None, filter=None, limit=None, expand=None):
        return list(all_author)

    api.list_models.side_effect = _list_models
    return api


def _patch_hub_list(monkeypatch, *, username, api, orgs=None):
    info = {"name": username, "orgs": orgs or []}
    monkeypatch.setattr(server_mod, "cached_whoami", lambda: info)
    monkeypatch.setattr(server_mod, "shared_hf_api", lambda: api)


def test_list_hub_jobs_includes_empty_untagged_run_repos(client: TestClient, monkeypatch) -> None:
    # The motivating case: an empty repo a crashed run pre-created. It has no
    # "lerobot" tag, so it appears ONLY in the unfiltered author listing — and
    # matches the run-repo timestamp suffix, so it must be surfaced.
    empty = _FakeModel(
        "makermods/smolvla_makermods_so101_merged_20260701_2026-07-03_09-15-57",
        last_modified=None,
    )
    api = _hub_api_with_models(all_author=[empty])
    _patch_hub_list(monkeypatch, username="makermods", api=api)

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    repo_ids = [m["repo_id"] for m in resp.json()["models"]]
    assert empty.id in repo_ids


def test_list_hub_jobs_unions_and_dedups_tagged_and_untagged(client: TestClient, monkeypatch) -> None:
    tagged = _FakeModel(
        "makermods/act_makermods_pick_2026-07-03_10-00-00",
        last_modified=_dt.datetime(2026, 7, 3, 10, 0, tzinfo=_dt.UTC),
        tags=["lerobot"],
    )
    empty = _FakeModel(
        "makermods/smolvla_makermods_so101_merged_20260701_2026-07-03_09-15-57",
        last_modified=_dt.datetime(2026, 7, 3, 9, 15, tzinfo=_dt.UTC),
    )
    # The single unfiltered pass returns both. `tagged` qualifies via BOTH its
    # lerobot tag and its run-repo suffix; the client-side filter + _add() dedup
    # must still surface it exactly once, and sort newest-first.
    api = _hub_api_with_models(all_author=[tagged, empty])
    _patch_hub_list(monkeypatch, username="makermods", api=api)

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    repo_ids = [m["repo_id"] for m in resp.json()["models"]]
    assert repo_ids == [tagged.id, empty.id]  # deduped, newest first
    assert repo_ids.count(tagged.id) == 1


def test_list_hub_jobs_excludes_foreign_personal_models(client: TestClient, monkeypatch) -> None:
    # A user's unrelated personal model (no lerobot tag, name doesn't match the
    # run-repo timestamp convention) must NOT be surfaced — it's theirs, not a
    # MakerMods Lab orphan. But a tagged repo is always kept even without the suffix.
    personal = _FakeModel("makermods/my-cool-llm", last_modified=None)
    run_repo = _FakeModel(
        "makermods/smolvla_makermods_so101_merged_20260701_2026-07-03_09-15-57",
        last_modified=None,
    )
    tagged_no_suffix = _FakeModel("makermods/some-tagged-model", last_modified=None, tags=["lerobot"])
    api = _hub_api_with_models(all_author=[personal, run_repo, tagged_no_suffix])
    _patch_hub_list(monkeypatch, username="makermods", api=api)

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    repo_ids = {m["repo_id"] for m in resp.json()["models"]}
    assert run_repo.id in repo_ids  # run-repo naming → surfaced
    assert tagged_no_suffix.id in repo_ids  # tagged → surfaced regardless of name
    assert personal.id not in repo_ids  # foreign personal model → excluded


# --- POST /jobs/hub/jobs/{job_id}/dismiss + listing filter ------------------
#
# The HF Jobs API has no delete — a finished job stays in list_jobs()
# indefinitely — so removing a dead untracked job from the UI is a local,
# persisted dismissal (utils/config.DISMISSED_HUB_JOBS_FILE). The /jobs/hub
# listing drops dismissed ids, but only in a terminal stage: a live run can
# never be dismissed out of sight.


class _FakeHubJob:
    def __init__(self, job_id, stage):
        self.id = job_id
        self.created_at = None
        self.docker_image = "huggingface/lerobot-gpu:latest"
        self.space_id = None
        self.flavor = "a100-large"
        self.status = SimpleNamespace(stage=stage, message=None)
        self.owner = None
        self.url = f"https://huggingface.co/jobs/{job_id}"


def _hub_api_with_jobs(jobs):
    """Fake HfApi whose list_jobs() returns `jobs`. list_models() returns
    nothing (models are irrelevant to the dismissal tests)."""
    api = MagicMock()
    api.list_jobs.return_value = list(jobs)
    api.list_models.return_value = []
    return api


def test_dismiss_hub_job_persists_and_hides_terminal_job(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    dead = _FakeHubJob("job-dead", "ERROR")
    other = _FakeHubJob("job-other", "COMPLETED")
    _patch_hub_list(monkeypatch, username="makermods", api=_hub_api_with_jobs([dead, other]))

    resp = client.post("/jobs/hub/jobs/job-dead/dismiss")
    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "job_id": "job-dead"}
    assert cfg.get_dismissed_hub_jobs() == {"job-dead"}

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    job_ids = [j["id"] for j in resp.json()["jobs"]]
    assert job_ids == ["job-other"]  # dismissed terminal job hidden, rest kept


def test_dismissed_hub_job_in_active_stage_stays_listed(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    # Dismissing an id whose job is still RUNNING must not hide it — the
    # listing keeps it until the job reaches a terminal stage.
    live = _FakeHubJob("job-live", "RUNNING")
    _patch_hub_list(monkeypatch, username="makermods", api=_hub_api_with_jobs([live]))
    cfg.add_dismissed_hub_job("job-live")

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    assert [j["id"] for j in resp.json()["jobs"]] == ["job-live"]


def test_list_hub_jobs_prunes_dismissed_ids_gone_from_listing(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    kept = _FakeHubJob("job-kept", "FAILED")
    _patch_hub_list(monkeypatch, username="makermods", api=_hub_api_with_jobs([kept]))
    cfg.add_dismissed_hub_job("job-kept")
    cfg.add_dismissed_hub_job("job-expired")  # no longer in the Hub listing

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == []
    assert cfg.get_dismissed_hub_jobs() == {"job-kept"}


def test_list_hub_jobs_keeps_dismissals_when_listing_fails(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    # A transient list_jobs() failure returns an empty jobs list; pruning
    # against it would forget every dismissal, so it must be skipped.
    api = _hub_api_with_jobs([])
    api.list_jobs.side_effect = RuntimeError("hub outage")
    _patch_hub_list(monkeypatch, username="makermods", api=api)
    cfg.add_dismissed_hub_job("job-dead")

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == []
    assert cfg.get_dismissed_hub_jobs() == {"job-dead"}


# --- Hub job run names -----------------------------------------------------
#
# Every cloud run launches on the same image, so a job the local registry
# doesn't know about (launched from another machine) would otherwise be titled
# "huggingface/lerobot-gpu:latest" in the library, like every other one.
# _hub_job_run_name recovers a real name from the job itself.


def _job_with(**attrs):
    """A _FakeHubJob carrying extra JobInfo attributes (labels, command, …)."""
    job = _FakeHubJob("job-x", "COMPLETED")
    for k, v in attrs.items():
        setattr(job, k, v)
    return job


def test_hub_job_run_name_prefers_submission_label() -> None:
    job = _job_with(
        labels={"makermodslab.run": "act_cube_2026-08-01_12-00-00"},
        command=["python", "-c", "…", "--", "--policy.repo_id", "makermods/other_name"],
    )
    assert server_mod._hub_job_run_name(job) == "act_cube_2026-08-01_12-00-00"


def test_hub_job_run_name_falls_back_to_repo_id_in_argv() -> None:
    # The backlog: jobs submitted before labelling existed still carry their
    # publish target in argv, and its slug is the run id.
    job = _job_with(
        command=["python", "-c", "…", "--", "--policy.repo_id", "makermods/act_cube_2026-08-01_12-00-00"]
    )
    assert server_mod._hub_job_run_name(job) == "act_cube_2026-08-01_12-00-00"


def test_hub_job_run_name_reads_equals_form_and_arguments_list() -> None:
    job = _job_with(command=None, arguments=["--policy.repo_id=makermods/smolvla_fold_2026-08-02_09-00-00"])
    assert server_mod._hub_job_run_name(job) == "smolvla_fold_2026-08-02_09-00-00"


def test_hub_job_run_name_is_none_when_nothing_identifies_the_run() -> None:
    # A foreign job on the same account: no label, no --policy.repo_id. The
    # card keeps its image-name fallback rather than inventing a name.
    assert server_mod._hub_job_run_name(_job_with(labels={}, command=["python", "train.py"])) is None
    assert server_mod._hub_job_run_name(_FakeHubJob("bare", "COMPLETED")) is None  # no attrs at all


def test_hub_job_run_name_ignores_blank_label_and_trailing_flag() -> None:
    # A blank label must not win over the argv fallback, and a dangling
    # --policy.repo_id with no value must not index past the end.
    job = _job_with(
        labels={"makermodslab.run": "   "},
        command=["--policy.repo_id", "makermods/act_x_2026-08-01_12-00-00"],
    )
    assert server_mod._hub_job_run_name(job) == "act_x_2026-08-01_12-00-00"
    assert server_mod._hub_job_run_name(_job_with(command=["--policy.repo_id"])) is None


# --- _argv_value / _hub_job_provenance ------------------------------------
#
# A cloud card's flavor/created/owner/image rows are near-identical across every
# run on an account. The provenance parser reads what the run started FROM off
# its own argv, which is the only Hub-side record of it: a repo id contains a
# "/", and the Hub's label charset forbids one, so no label could carry it.


def test_argv_value_reads_both_spellings() -> None:
    argv = ["--a", "1", "--b=2"]
    assert server_mod._argv_value(argv, "--a") == "1"
    assert server_mod._argv_value(argv, "--b") == "2"
    assert server_mod._argv_value(argv, "--c") is None


def test_argv_value_treats_blank_and_dangling_as_absent() -> None:
    # A trailing flag must not index past the end, and an empty value carries no
    # more information than no flag at all.
    assert server_mod._argv_value(["--a"], "--a") is None
    assert server_mod._argv_value(["--a", "   "], "--a") is None
    assert server_mod._argv_value(["--a="], "--a") is None


def test_argv_value_does_not_match_a_flag_by_prefix() -> None:
    # "--steps" must not be answered by "--steps_per_epoch".
    assert server_mod._argv_value(["--steps_per_epoch", "7"], "--steps") is None


def test_provenance_finetune_from_a_user_chosen_base() -> None:
    job = _job_with(
        command=["--policy.pretrained_path=makermods/my_base", "--dataset.repo_id", "u/d", "--steps", "9000"]
    )
    prov = server_mod._hub_job_provenance(job)
    assert prov["kind"] == "finetune"
    assert prov["base_repo"] == "makermods/my_base"
    assert prov["dataset_repo_id"] == "u/d"
    assert prov["steps"] == "9000"


def test_the_frontend_foundation_list_cannot_drift_from_the_python_one() -> None:
    # jobsApi.ts carries its own copy of these repo ids (the local JobCard has no
    # backend round-trip to classify against). Nothing links the two, so a fifth
    # policy added in Python would silently mis-chip local VLA runs as
    # "Fine-tune". This test is that link: if it fails, update
    # frontend/src/lib/jobsApi.ts FOUNDATION_BASE_REPO_IDS to match.
    from makermodslab.jobs import _KNOWN_FOUNDATION_BASE_REPO_IDS

    assert (
        frozenset(
            {
                "lerobot/smolvla_base",
                "lerobot/pi0_base",
                "lerobot/pi05_base",
                "lerobot/pi0fast-base",
            }
        )
        == _KNOWN_FOUNDATION_BASE_REPO_IDS
    )


def test_argv_value_rejects_an_option_shaped_value() -> None:
    # A dangling flag followed by another flag must not swallow it as a value:
    # that turned "--policy.pretrained_path --resume true" into a confident
    # fine-tune whose base model was the string "--resume".
    argv = ["--policy.pretrained_path", "--resume", "true"]
    assert server_mod._argv_value(argv, "--policy.pretrained_path") is None
    assert server_mod._hub_job_provenance(_job_with(command=argv))["kind"] == "resume"


def test_argv_value_does_not_close_gaps_left_by_junk_tokens() -> None:
    # Dropping non-string tokens made two tokens adjacent that never were, so a
    # flag read the token AFTER the junk as its value.
    argv = ["--policy.type", None, "act"]
    assert server_mod._argv_value(argv, "--policy.type") is None


def test_provenance_handles_a_non_numeric_checkpoint_ref() -> None:
    # hf_cloud can emit "@checkpoints/last", which the digits-only ref regex
    # will not split — without a guard the whole raw ref reaches the card.
    prov = server_mod._hub_job_provenance(_job_with(command=["--resume-from=u/run@checkpoints/last"]))
    assert prov["base_repo"] == "u/run"
    assert prov["base_step"] == "last"


def test_provenance_calls_a_vla_default_start_foundation_not_finetune() -> None:
    # JobRegistry.start pins policy_pretrained_path to the public foundation
    # checkpoint for ANY smolvla/pi0 run that named no starting point, so
    # `--policy.pretrained_path` is on the argv of every from-scratch VLA run.
    # Reading that as a fine-tune would mislabel most cards on a VLA account.
    for base in ("lerobot/smolvla_base", "lerobot/pi0_base", "lerobot/pi05_base", "lerobot/pi0fast-base"):
        prov = server_mod._hub_job_provenance(_job_with(command=[f"--policy.pretrained_path={base}"]))
        assert prov["kind"] == "foundation", base
        assert prov["base_repo"] == base


def test_provenance_recovers_the_run_id_from_a_staged_checkpoint_base() -> None:
    # A local run's checkpoint uploaded for a cloud fine-tune lives in a
    # "<user>/<job id>_checkpoints" staging repo. The job id is the thing a
    # person recognizes; the raw ref must never reach the card.
    job = _job_with(
        command=[
            "--policy.pretrained_path=makermods/act_cube_2026-08-01_12-00-00_checkpoints@checkpoints/012000"
        ]
    )
    prov = server_mod._hub_job_provenance(job)
    assert prov["kind"] == "finetune"
    assert prov["base_job_id"] == "act_cube_2026-08-01_12-00-00"
    assert prov["base_step"] == "012000"


def test_provenance_reads_a_continuation_from_the_wrapper_directive() -> None:
    # NOT from --config_path: on a cloud continuation that is a container path
    # that names nothing the user could recognize.
    job = _job_with(
        command=[
            "python",
            "-c",
            "…",
            "--resume-from=makermods/act_cube_2026-08-01_12-00-00@checkpoints/020000",
            "--",
            "--config_path=/tmp/makermodslab/train/checkpoints/020000/pretrained_model/train_config.json",
            "--resume",
            "true",
        ]
    )
    prov = server_mod._hub_job_provenance(job)
    assert prov["kind"] == "resume"
    assert prov["base_repo"] == "makermods/act_cube_2026-08-01_12-00-00"
    assert prov["base_step"] == "020000"
    # The container path must not have leaked into anything user-facing.
    assert "/tmp/" not in str(prov["base_repo"])
    assert prov["base_job_id"] is None  # not a staging repo


def test_provenance_knows_an_old_continuation_without_naming_its_source() -> None:
    # Submitted before the wrapper carried --resume-from: we know it continued
    # something, but not what. Honest beats invented.
    prov = server_mod._hub_job_provenance(_job_with(command=["--resume", "true"]))
    assert prov["kind"] == "resume"
    assert prov["base_repo"] is None


def test_provenance_omits_rather_than_guesses_on_a_continuation() -> None:
    # build_training_command's resume branch emits neither --dataset.repo_id nor
    # --policy.type; lerobot rebuilds both from the checkpoint config.
    prov = server_mod._hub_job_provenance(_job_with(command=["--resume-from=u/r@checkpoints/000500"]))
    assert prov["dataset_repo_id"] is None
    assert prov["policy_type"] is None


def test_provenance_of_a_scratch_run_and_a_foreign_job() -> None:
    prov = server_mod._hub_job_provenance(
        _job_with(command=["--dataset.repo_id", "u/d", "--policy.type", "act", "--resume", "false"])
    )
    assert prov["kind"] == "scratch"
    assert prov["base_ref"] is None
    assert prov["policy_type"] == "act"
    # Someone else's job on the same account: no argv we understand at all.
    bare = server_mod._hub_job_provenance(_FakeHubJob("bare", "COMPLETED"))
    assert bare["kind"] == "scratch"
    assert bare["dataset_repo_id"] is None


def test_list_hub_jobs_exposes_the_run_name(client: TestClient, monkeypatch, tmp_lerobot_home: Path) -> None:
    named = _job_with(labels={"makermodslab.run": "act_cube_2026-08-01_12-00-00"})
    named.id = "job-named"
    _patch_hub_list(monkeypatch, username="makermods", api=_hub_api_with_jobs([named]))

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200
    assert resp.json()["jobs"][0]["name"] == "act_cube_2026-08-01_12-00-00"


# --- /jobs/devices --------------------------------------------------------
#
# Cross-device presence is a convenience layered on the Hub. Every failure mode
# below must degrade to "no other devices", never to an error: it sits in the
# same library as the local jobs list, and breaking that would cost the user
# access to their own runs to show them somebody else's.


@pytest.fixture(autouse=True)
def _no_presence_cache(monkeypatch):
    """Drop the /jobs/devices response cache between tests."""
    monkeypatch.setattr(server_mod, "_devices_cache", None)


def test_read_board_is_empty_when_signed_out(monkeypatch, tmp_lerobot_home: Path) -> None:
    # Tests read_board DIRECTLY. Going through the endpoint proved nothing: the
    # autouse fixture already stubs read_board to [], so the handler would have
    # returned an empty list with the signed-out branch deleted.
    called: list[str] = []
    monkeypatch.setattr(server_mod.presence, "cached_whoami", lambda **kw: None)
    monkeypatch.setattr(
        server_mod.presence,
        "shared_hf_api",
        lambda: called.append("api") or MagicMock(),
    )
    assert server_mod.presence.read_board() == []
    assert called == [], "signed out must not reach for a Hub client at all"


def test_list_device_runs_survives_an_unreachable_hub(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    def _boom(**kwargs):
        raise RuntimeError("network is down")

    monkeypatch.setattr(server_mod.presence, "read_board", _boom)
    resp = client.get("/jobs/devices")
    assert resp.status_code == 200
    assert resp.json()["devices"] == []


def test_list_device_runs_reports_a_write_forbidden_token(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    # A read-only token can never publish. Retrying forever while the toggle
    # still reads "on" is the silent lie default-on publishing must not become,
    # so the reason is surfaced instead.
    monkeypatch.setattr(server_mod.presence, "read_board", lambda **kw: [])
    monkeypatch.setattr(
        server_mod.presence_publisher,
        "status",
        lambda: {
            "device_id": "dev-1",
            "disabled_reason": "forbidden",
            "last_error": "403",
            "last_write": 0.0,
        },
    )
    body = client.get("/jobs/devices").json()
    assert body["disabled_reason"] == "forbidden"


def test_presence_settings_round_trip(client: TestClient, monkeypatch, tmp_lerobot_home: Path) -> None:
    assert client.post("/jobs/devices/settings", json={"enabled": False}).json()["enabled"] is False
    assert server_mod.presence.load_settings()["enabled"] is False
    assert client.post("/jobs/devices/settings", json={"label": "desktop"}).json()["label"] == "desktop"
    # A label-only update must not silently flip sharing back on.
    assert server_mod.presence.load_settings()["enabled"] is False


def test_forget_device_refuses_this_device(client: TestClient, monkeypatch, tmp_lerobot_home: Path) -> None:
    # Its file is rewritten on the next publish, so deleting it would be a no-op
    # that looks like it worked.
    mine = server_mod.presence.device_id()
    resp = client.delete(f"/jobs/devices/{mine}")
    assert resp.status_code == 400


def test_dismiss_hub_job_rejects_blank_id(client: TestClient, monkeypatch, tmp_lerobot_home: Path) -> None:
    resp = client.post("/jobs/hub/jobs/%20/dismiss")
    assert resp.status_code == 400
    assert cfg.get_dismissed_hub_jobs() == set()


def test_delete_job_dismisses_its_hub_job_id(client: TestClient, monkeypatch, tmp_lerobot_home: Path) -> None:
    # Deleting a tracked cloud run must also dismiss its hf_job_id, otherwise
    # the Hub job resurfaces as an untracked card on the next /jobs/hub poll.
    record = MagicMock()
    record.hf_job_id = "hub-job-123"
    monkeypatch.setattr(server_mod.job_registry, "get", lambda job_id: record)
    monkeypatch.setattr(server_mod.job_registry, "delete", lambda job_id: None)

    resp = client.delete("/jobs/some-cloud-run")
    assert resp.status_code == 204
    assert cfg.get_dismissed_hub_jobs() == {"hub-job-123"}


def test_delete_local_job_records_no_dismissal(
    client: TestClient, monkeypatch, tmp_lerobot_home: Path
) -> None:
    record = MagicMock()
    record.hf_job_id = None
    monkeypatch.setattr(server_mod.job_registry, "get", lambda job_id: record)
    monkeypatch.setattr(server_mod.job_registry, "delete", lambda job_id: None)

    resp = client.delete("/jobs/some-local-run")
    assert resp.status_code == 204
    assert cfg.get_dismissed_hub_jobs() == set()


def test_recording_pause_route_calls_handler(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.server as server_mod

    called = {}

    def _fake_handle_pause_recording():
        called["hit"] = True
        return {"success": True, "message": "Reset phase paused"}

    monkeypatch.setattr(server_mod, "handle_pause_recording", _fake_handle_pause_recording)

    response = client.post("/recording-pause")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert called.get("hit") is True


def test_recording_resume_route_calls_handler(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.server as server_mod

    called = {}

    def _fake_handle_resume_recording():
        called["hit"] = True
        return {"success": True, "message": "Reset phase resumed"}

    monkeypatch.setattr(server_mod, "handle_resume_recording", _fake_handle_resume_recording)

    response = client.post("/recording-resume")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert called.get("hit") is True


def test_format_accelerator_flattens_the_hub_object() -> None:
    """huggingface_hub hands us a JobAccelerator OBJECT here, not a string.

    Forwarding it raw put a nested dict on the wire under a field the frontend
    types as `string`, so the hardware picker rendered "[object Object]" for
    every GPU flavor.
    """
    from huggingface_hub._jobs_api import JobAccelerator

    from makermodslab.server import _format_accelerator

    single = JobAccelerator(type="gpu", model="T4", quantity="1", vram="16 GB", manufacturer="Nvidia")
    assert _format_accelerator(single) == "Nvidia T4"

    multi = JobAccelerator(type="gpu", model="A100", quantity="4", vram="320 GB", manufacturer="Nvidia")
    assert _format_accelerator(multi) == "4× Nvidia A100"

    # cpu-* flavors carry no accelerator; the caller falls back to `cpu`.
    assert _format_accelerator(None) is None


def test_format_accelerator_survives_a_renamed_hub_field() -> None:
    """A future hub version could rename the fields out from under us. Any
    string still beats a dict the UI would render as [object Object]."""
    from makermodslab.server import _format_accelerator

    class _Unknown:
        def __str__(self) -> str:
            return "some-future-accelerator"

    assert _format_accelerator(_Unknown()) == "some-future-accelerator"
