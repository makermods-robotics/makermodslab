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
"""Remote teleoperation (remote_host.py + remote_teleoperate.py): the pure
helpers (schema derivation, URDF mapping from published ranges, descriptor,
mismatch reasons), the leader-only readiness scope, the two kinds' start
refusals through /api/v1/sessions (mutex, SFU/extra preconditions, station
handshake against an httpx.MockTransport peer), the status routes, and the
health capability. Portal/hardware happy paths are the manual two-machine
test (tests/ policy)."""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import httpx
import jwt
import pytest

from makermodslab import remote_host, remote_teleoperate, sfu
from makermodslab.nodes import NodeRegistry

# --- remote_host pure helpers -------------------------------------------------


def test_split_features_separates_motors_and_cameras_in_order() -> None:
    features = {
        "shoulder_pan.pos": float,
        "front": (480, 640, 3),
        "gripper.pos": float,
        "wrist": (240, 320, 3),
    }
    motors, cameras = remote_host.split_features(features)
    assert motors == ["shoulder_pan", "gripper"]
    assert cameras == {"front": (480, 640), "wrist": (240, 320)}


def test_split_features_keeps_bimanual_prefixes() -> None:
    motors, _ = remote_host.split_features({"left_elbow_flex.pos": float, "right_elbow_flex.pos": float})
    assert motors == ["left_elbow_flex", "right_elbow_flex"]


def _cal(range_min: int, range_max: int):
    return types.SimpleNamespace(range_min=range_min, range_max=range_max)


def _deg(ticks: int) -> float:
    """Ticks → degrees with the same constant teleoperate._motor_fraction uses."""
    return ticks * 360.0 / remote_host._STS3215_MAX_RES


def test_joint_ranges_deg_matches_the_motor_fraction_formula() -> None:
    cal = {"shoulder_pan": _cal(1024, 3072), "gripper": _cal(0, 4096), "elbow_flex": _cal(100, 100)}
    ranges = remote_host.joint_ranges_deg({"": cal})
    assert ranges == {"shoulder_pan": pytest.approx(_deg(2048))}  # gripper skipped, zero range skipped


def test_joint_ranges_deg_prefixes_bimanual_arms() -> None:
    ranges = remote_host.joint_ranges_deg(
        {"left_": {"wrist_roll": _cal(0, 2048)}, "right_": {"wrist_roll": _cal(0, 4096)}}
    )
    assert ranges == {
        "left_wrist_roll": pytest.approx(_deg(2048)),
        "right_wrist_roll": pytest.approx(_deg(4096)),
    }


def test_observation_to_urdf_joints_maps_with_published_ranges() -> None:
    ranges = {"shoulder_pan": 180.0}
    # 0 deg = midpoint of the calibrated travel = midpoint of the URDF limits.
    joints = remote_host.observation_to_urdf_joints({"shoulder_pan.pos": 0.0, "gripper.pos": 100.0}, ranges)
    assert joints["Rotation"] == pytest.approx(0.0)
    assert joints["Jaw"] == pytest.approx(1.74533)  # gripper 100% = upper limit
    # Full positive travel clamps to the upper URDF limit.
    assert remote_host.observation_to_urdf_joints({"shoulder_pan.pos": 90.0}, ranges)[
        "Rotation"
    ] == pytest.approx(1.91986)
    # No range published: raw degrees rendered as radians (uncalibrated fallback).
    assert remote_host.observation_to_urdf_joints({"shoulder_pan.pos": 45.0}, {})[
        "Rotation"
    ] == pytest.approx(math.pi / 4)
    # Missing joints read 0 so the viewer always gets a full set.
    assert remote_host.observation_to_urdf_joints({}, ranges)["Elbow"] == 0.0


def test_observation_to_urdf_joints_honours_prefix() -> None:
    joints = remote_host.observation_to_urdf_joints({"right_gripper.pos": 0.0}, {}, prefix="right_")
    assert joints["Jaw"] == pytest.approx(-0.174533)


def test_observation_degrees_strips_one_arm_prefix() -> None:
    obs = {"left_j1.pos": 1.0, "right_j1.pos": 2.0, "left_j2.vel": 9.0}
    assert remote_host.observation_degrees(obs, prefix="right_") == {"j1": 2.0}
    assert remote_host.observation_degrees({"j1.pos": 3}) == {"j1": 3.0}
    assert remote_host.observation_degrees(obs) == {}  # single-arm read of a bimanual obs: nothing


def test_build_descriptor_shape() -> None:
    request = remote_host.HostingRequest(
        follower_port="/dev/f",
        follower_config="fc",
        robot_name="bench",
        arm_type="so101",
        fps=25,
        video_codec="MJPEG",
    )
    descriptor = remote_host.build_descriptor(
        request, room="mml-abc", motors=["a", "b"], cameras={"front": (480, 640)}, ranges_deg={"a": 180.0}
    )
    assert descriptor == {
        "robot": "bench",
        "arm_type": "so101",
        "mode": "single",
        "room": "mml-abc",
        "fps": 25,
        "video_codec": "MJPEG",
        "motors": ["a", "b"],
        "cameras": [{"name": "front", "width": 640, "height": 480}],
        "joint_ranges_deg": {"a": 180.0},
    }


# --- remote_teleoperate pure helpers -----------------------------------------


def test_leader_motors_from_action_features() -> None:
    assert remote_teleoperate.leader_motors({"a.pos": float, "b.pos": float, "c.vel": float}) == ["a", "b"]


def test_schema_mismatch_reasons() -> None:
    descriptor = {"arm_type": "so101", "motors": ["a", "b"]}
    assert remote_teleoperate.schema_mismatch(descriptor, "maker", None)
    assert remote_teleoperate.schema_mismatch(descriptor, "so101", None) is None
    assert remote_teleoperate.schema_mismatch(descriptor, "so101", ["a", "b"]) is None
    assert "Motor sets differ" in remote_teleoperate.schema_mismatch(descriptor, "so101", ["b", "a"])


def test_metrics_summary_converts_microseconds() -> None:
    metrics = types.SimpleNamespace(
        rtt=types.SimpleNamespace(rtt_us_last=1050, rtt_us_mean=732, rtt_us_p95=None),
        sync=types.SimpleNamespace(observations_emitted=57, states_dropped=5),
    )
    assert remote_teleoperate.metrics_summary(metrics) == {
        "rtt_ms_last": 1.05,
        "rtt_ms_mean": 0.73,
        "rtt_ms_p95": None,
        "observations": 57,
        "states_dropped": 5,
    }
    assert remote_teleoperate.metrics_summary(None) is None


# --- readiness: leader-only scope ---------------------------------------------


def test_record_readiness_leader_scope(tmp_lerobot_home) -> None:
    from makermodslab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "LC.json").write_text("{}")
    leader_only_record = {
        "mode": "single",
        "leader_port": "/dev/l",
        "leader_config": "LC",
        "follower_port": "",
    }
    assert cfg.is_robot_record_clean(leader_only_record, arms="leader") is True
    assert cfg.is_robot_record_clean(leader_only_record, arms="all") is False
    assert cfg.is_robot_record_clean(leader_only_record, arms="follower") is False


# --- sessions: refusals through the front door ----------------------------------


def _make_robot(name: str, *, leader: bool, follower: bool) -> None:
    from makermodslab.utils import config as cfg

    data: dict = {}
    if follower:
        data |= {"follower_port": "/dev/f", "follower_config": "FC"}
        (Path(cfg.FOLLOWER_CONFIG_PATH) / "FC.json").write_text("{}")
    if leader:
        data |= {"leader_port": "/dev/l", "leader_config": "LC"}
        (Path(cfg.LEADER_CONFIG_PATH) / "LC.json").write_text("{}")
    cfg.save_robot_record(name, data)


@pytest.fixture
def _idle(monkeypatch: pytest.MonkeyPatch):
    """Both remote flags down, and the two heavy preconditions off by default."""
    monkeypatch.setattr(remote_host, "hosting_active", False)
    monkeypatch.setattr(remote_teleoperate, "remote_teleoperation_active", False)
    monkeypatch.delenv(sfu.ENV_KEY_FILE, raising=False)


def test_hosting_needs_a_follower_only_record(client, tmp_lerobot_home, _idle) -> None:
    _make_robot("leaderless", leader=True, follower=False)
    resp = client.post("/api/v1/sessions", json={"kind": "hosting", "robot": "leaderless"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "robot.not_ready"
    assert "follower arm" in resp.json()["detail"]


def test_hosting_refuses_without_the_sfu(client, tmp_lerobot_home, _idle) -> None:
    _make_robot("station", leader=False, follower=True)
    resp = client.post("/api/v1/sessions", json={"kind": "hosting", "robot": "station"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "sfu.disabled"


def test_hosting_refuses_without_the_remote_extra(client, tmp_lerobot_home, _idle, monkeypatch) -> None:
    _make_robot("station", leader=False, follower=True)
    monkeypatch.setenv(sfu.ENV_KEY_FILE, "/nonexistent-but-enabled")
    monkeypatch.setattr(remote_host, "remote_extra_available", lambda: False)
    resp = client.post("/api/v1/sessions", json={"kind": "hosting", "robot": "station"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "system.extra_missing"


def test_hosting_options_validate_codec_and_fps(client, tmp_lerobot_home, _idle) -> None:
    _make_robot("station", leader=False, follower=True)
    for options in ({"video_codec": "VP9"}, {"fps": 1}, {"unknown": 1}):
        resp = client.post(
            "/api/v1/sessions", json={"kind": "hosting", "robot": "station", "options": options}
        )
        assert resp.status_code == 422, options
        assert resp.json()["code"] == "request.validation"


def test_remote_teleoperation_needs_a_leader_only_record(client, tmp_lerobot_home, _idle) -> None:
    _make_robot("followerless", leader=False, follower=True)
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "remote_teleoperation", "robot": "followerless", "options": {"station": "abc"}},
    )
    assert resp.status_code == 400
    assert "leader arm" in resp.json()["detail"]


def test_remote_teleoperation_requires_a_station(client, tmp_lerobot_home, _idle) -> None:
    _make_robot("laptop", leader=True, follower=False)
    resp = client.post("/api/v1/sessions", json={"kind": "remote_teleoperation", "robot": "laptop"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


class _Station:
    """One fake peer behind an httpx.MockTransport: its health doc, its
    hosting status, and its token route."""

    def __init__(self, url: str, instance_id: str, hosting: dict | None) -> None:
        self.url = url
        self.instance_id = instance_id
        self.hosting = hosting
        self.token_requests: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/v1/health":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "message": "",
                        "version": "0.1.0",
                        "instance_id": self.instance_id,
                        "capabilities": {},
                    },
                )
            if path == "/api/v1/hosting":
                return httpx.Response(
                    200, json={"hosting_active": self.hosting is not None, "hosting": self.hosting}
                )
            if path == "/api/v1/sfu/token":
                body = json.loads(request.content)
                self.token_requests.append(body)
                return httpx.Response(
                    200,
                    json={
                        "url": "ws://s:7880",
                        "token": "jwt",
                        "room": body["room"],
                        "identity": body["identity"],
                        "role": body["role"],
                        "expires_at": 1,
                    },
                )
            return httpx.Response(404, json={"detail": "nope"})

        return httpx.MockTransport(handler)


def _registry_with(station: _Station) -> NodeRegistry:
    clock = types.SimpleNamespace(now=1000.0)
    registry = NodeRegistry(clock=lambda: clock.now, transport=station.transport())
    registry.add(station.url)
    return registry


@pytest.fixture
def _operator_ready(tmp_lerobot_home, _idle, monkeypatch):
    _make_robot("laptop", leader=True, follower=False)
    # The operator module binds the probe by name at import; patch it there.
    monkeypatch.setattr(remote_teleoperate, "remote_extra_available", lambda: True)


def test_remote_teleoperation_unknown_station_is_404(client, _operator_ready, monkeypatch) -> None:
    station = _Station("http://s:8000", "a" * 32, hosting=None)
    monkeypatch.setattr(remote_teleoperate, "node_registry", _registry_with(station))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "remote_teleoperation", "robot": "laptop", "options": {"station": "b" * 32}},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_remote_teleoperation_station_not_hosting_is_409(client, _operator_ready, monkeypatch) -> None:
    station = _Station("http://s:8000", "a" * 32, hosting=None)
    monkeypatch.setattr(remote_teleoperate, "node_registry", _registry_with(station))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "remote_teleoperation", "robot": "laptop", "options": {"station": "a" * 32}},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "node.not_hosting"


def test_remote_teleoperation_arm_family_mismatch_is_refused_before_any_hardware(
    client, _operator_ready, monkeypatch
) -> None:
    descriptor = {
        "arm_type": "maker",
        "motors": ["j1"],
        "room": "r",
        "cameras": [],
        "fps": 30,
        "video_codec": "H264",
    }
    station = _Station("http://s:8000", "a" * 32, hosting=descriptor)
    monkeypatch.setattr(remote_teleoperate, "node_registry", _registry_with(station))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "remote_teleoperation", "robot": "laptop", "options": {"station": "a" * 32}},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "robot.schema_mismatch"
    assert station.token_requests == []  # refused before asking for a token


def test_remote_teleoperation_fetches_an_operator_token_from_the_station(
    client, _operator_ready, monkeypatch
) -> None:
    """Up to the hardware step everything is the station handshake: a matching
    descriptor gets an operator token minted BY THE STATION for its room. The
    leader connect that follows is stubbed to fail so no serial port is
    touched — and its failure is reported as a coded refusal, not a 500."""
    descriptor = {
        "arm_type": "so101",
        "motors": ["j1"],
        "room": "mml-x",
        "cameras": [],
        "fps": 30,
        "video_codec": "H264",
    }
    station = _Station("http://s:8000", "a" * 32, hosting=descriptor)
    monkeypatch.setattr(remote_teleoperate, "node_registry", _registry_with(station))

    def boom(request):
        raise RuntimeError("no leader on /dev/l")

    monkeypatch.setattr(remote_teleoperate, "_connect_leader", boom)
    import sys

    monkeypatch.setitem(
        sys.modules,
        "lerobot_robot_livekit",
        types.SimpleNamespace(LiveKitRobot=None, LiveKitRobotConfig=None),
    )
    monkeypatch.setitem(sys.modules, "livekit.portal", types.SimpleNamespace(VideoCodec={}))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "remote_teleoperation", "robot": "laptop", "options": {"station": "a" * 32}},
    )
    assert resp.status_code == 500
    assert resp.json()["code"] == "hardware.connect_failed"
    assert station.token_requests == [
        {"identity": f"operator-{_instance_prefix()}", "room": "mml-x", "role": "operator"}
    ]
    assert remote_teleoperate.remote_teleoperation_active is False


def _instance_prefix() -> str:
    from makermodslab.utils.config import get_instance_id

    return get_instance_id()[:12]


def test_hosting_flag_holds_the_hardware_for_every_kind(client, tmp_lerobot_home, _idle, monkeypatch) -> None:
    """An ENGAGED hosting session holds the hardware like any feature (a
    parked, unseated one yields to a local start instead — see the
    preemption tests below)."""
    _make_robot("bench", leader=True, follower=True)
    monkeypatch.setattr(remote_host, "hosting_active", True)
    monkeypatch.setattr(remote_host, "phase", "engaged")
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "session.held"
    assert resp.json()["details"]["holder"]["kind"] == "hosting"


# --- status routes + health ---------------------------------------------------


def test_hosting_status_idle_shape(client, _idle) -> None:
    body = client.get("/api/v1/hosting").json()
    assert body["hosting_active"] is False
    assert body["hosting"] is None
    assert set(body) >= {"releasing", "last_cleanup_error", "outcome", "error", "hint", "message"}


def test_hosting_status_fills_url_from_the_request_host(client, _idle, monkeypatch) -> None:
    descriptor = remote_host.build_descriptor(
        remote_host.HostingRequest(follower_port="/dev/f", follower_config="fc", robot_name="bench"),
        room="mml-x",
        motors=["a"],
        cameras={},
        ranges_deg={},
    )
    monkeypatch.setattr(remote_host, "hosting_active", True)
    monkeypatch.setattr(remote_host, "current_descriptor", descriptor)
    monkeypatch.setenv(sfu.ENV_PORT, "7880")
    monkeypatch.delenv(sfu.ENV_URL, raising=False)
    body = client.get("/api/v1/hosting", headers={"host": "100.64.0.9:8000"}).json()
    assert body["hosting"]["url"] == "ws://100.64.0.9:7880"
    assert body["hosting"]["active_operator"] is None
    assert body["hosting"]["motors"] == ["a"]
    # ...and health advertises the hosted robot so a laptop's picker can find it.
    caps = client.get("/api/v1/health").json()["capabilities"]
    assert caps["hosting"] == {
        "robot": "bench",
        "arm_type": "so101",
        "phase": "parked",
        "active_operator": None,
    }


def test_health_has_no_hosting_key_when_idle(client, _idle) -> None:
    assert "hosting" not in client.get("/api/v1/health").json()["capabilities"]


def test_remote_teleoperation_status_idle_and_camera_404(client, _idle) -> None:
    body = client.get("/api/v1/remote-teleoperation").json()
    assert body["remote_teleoperation_active"] is False
    assert body["station"] is None and body["cameras"] == [] and body["metrics"] is None
    resp = client.get("/api/v1/remote-teleoperation/camera/front")
    assert resp.status_code == 404


def test_remote_extra_route_shape(client) -> None:
    body = client.get("/api/v1/system/remote-extra").json()
    assert set(body) == {"available", "install_hint"}
    assert isinstance(body["available"], bool)


# --- parked / engaged: the seat policy ------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _monitor() -> tuple[remote_host.SeatMonitor, _Clock]:
    clock = _Clock()
    return remote_host.SeatMonitor(grace_s=15.0, clock=clock), clock


def test_first_operator_takes_the_seat_and_is_engaged() -> None:
    m, _ = _monitor()
    assert m.operator_joined("laptop") == "engage"
    assert m.seat == "laptop"
    # A second operator is ignored (the room cap and token route keep them out anyway).
    assert m.operator_joined("intruder") is None
    assert m.seat == "laptop"


def test_silent_loss_inside_grace_is_tolerated_and_a_rejoin_re_engages() -> None:
    m, clock = _monitor()
    m.operator_joined("laptop")
    m.operator_left("laptop")
    clock.now += 10.0
    assert m.tick(engaged=True) is None  # still within grace: frozen, seat kept
    assert m.operator_joined("laptop") == "engage"  # reconnect resumes with a soft start
    clock.now += 20.0
    m.action_received()
    assert m.tick(engaged=True) is None
    assert m.seat == "laptop"


def test_silent_loss_past_grace_parks_and_frees_the_seat() -> None:
    m, clock = _monitor()
    m.operator_joined("laptop")
    m.operator_left("laptop")
    clock.now += 15.0
    assert m.tick(engaged=True) == "park"
    assert m.seat is None
    assert m.operator_joined("someone-else") == "engage"  # seat is free again


def test_home_parks_and_holds_until_an_explicit_engage() -> None:
    m, clock = _monitor()
    m.operator_joined("laptop")
    assert m.command("home", "laptop") == "park"
    # The leader keeps streaming; that must NOT re-engage a homed arm.
    clock.now += 0.5
    m.action_received()
    assert m.tick(engaged=False) is None
    assert m.command("engage", "stranger") is None  # only the seat holder is heard
    assert m.command("engage", "laptop") == "engage"


def test_release_parks_immediately_and_frees_the_seat() -> None:
    m, _ = _monitor()
    m.operator_joined("laptop")
    assert m.command("release", "laptop") == "park"
    assert m.seat is None


def test_action_stall_parks_but_keeps_the_seat_and_resuming_re_engages() -> None:
    m, clock = _monitor()
    m.operator_joined("laptop")
    clock.now += 15.0
    assert m.tick(engaged=True) == "park"  # no actions for the grace period, operator still present
    assert m.seat == "laptop"
    clock.now += 5.0
    assert m.tick(engaged=False) is None  # still stalled
    m.action_received()
    assert m.tick(engaged=False) == "engage"  # stream is back


def test_a_stall_that_persists_a_second_grace_frees_the_seat() -> None:
    """A hard-crashed laptop is reported late by the SFU; the stall rule
    parks after one grace and must not keep the seat forever."""
    m, clock = _monitor()
    m.operator_joined("laptop")
    clock.now += 15.0
    assert m.tick(engaged=True) == "park"
    assert m.seat == "laptop"
    clock.now += 14.0
    assert m.tick(engaged=False) is None
    assert m.seat == "laptop"
    clock.now += 1.0
    assert m.tick(engaged=False) is None
    assert m.seat is None  # freed: the next operator can take it


def test_soft_start_blend_eases_from_present_to_target() -> None:
    assert remote_host.soft_start_blend(0.0) == 0.0
    assert remote_host.soft_start_blend(0.5) == pytest.approx(0.5)
    assert remote_host.soft_start_blend(1.0) == 1.0
    assert remote_host.soft_start_blend(9.0) == 1.0
    mid = remote_host.blend_action(
        {"a.pos": 0.0, "b.pos": 10.0}, {"a.pos": 100.0, "b.pos": 10.0, "c.pos": 5.0}, 0.25
    )
    assert mid == {"a.pos": 25.0, "b.pos": 10.0, "c.pos": 5.0}


# --- single seat on the token route, station mode, preemption -------------------


@pytest.fixture
def sfu_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """The launcher's --sfu handoff: a key file + the env the app reads."""
    from makermodslab.utils.config import load_or_create_livekit_keys

    path = str(tmp_path / "livekit_keys.yaml")
    key, secret = load_or_create_livekit_keys(path)
    monkeypatch.setenv(sfu.ENV_KEY_FILE, path)
    monkeypatch.setenv(sfu.ENV_PORT, "7880")
    monkeypatch.delenv(sfu.ENV_URL, raising=False)
    sfu._keys_from_file.cache_clear()
    return key, secret


def test_operator_token_refused_while_the_seat_is_held_by_someone_else(client, sfu_env, monkeypatch) -> None:
    monkeypatch.setattr(remote_host, "seat_holder", lambda: "operator-aaaaaaaaaaaa")
    denied = client.post("/api/v1/sfu/token", json={"role": "operator", "identity": "operator-bbbbbbbbbbbb"})
    assert denied.status_code == 409
    assert denied.json()["code"] == "sfu.seat_taken"
    # The holder itself (a reconnect) and non-operator roles are unaffected.
    assert (
        client.post(
            "/api/v1/sfu/token", json={"role": "operator", "identity": "operator-aaaaaaaaaaaa"}
        ).status_code
        == 200
    )
    assert client.post("/api/v1/sfu/token", json={"role": "viewer"}).status_code == 200


def test_robot_token_caps_the_room_at_two_participants() -> None:
    token, _ = sfu.mint_token(
        api_key="k", api_secret="s" * 40, identity="robot", room="r", role="robot", max_participants=2
    )
    assert jwt.decode(token, options={"verify_signature": False})["roomConfig"]["maxParticipants"] == 2


def test_hosting_refuses_can_arms_in_this_release(client, tmp_lerobot_home, _idle, monkeypatch) -> None:
    from makermodslab.utils import config as cfg

    (Path(cfg.follower_config_path_for("maker")) / "FC.json").parent.mkdir(parents=True, exist_ok=True)
    (Path(cfg.follower_config_path_for("maker")) / "FC.json").write_text("{}")
    cfg.save_robot_record(
        "canbot", {"arm_type": "maker", "follower_port": "/dev/can", "follower_config": "FC"}
    )
    monkeypatch.setenv(sfu.ENV_KEY_FILE, "/enabled")
    resp = client.post("/api/v1/sessions", json={"kind": "hosting", "robot": "canbot"})
    assert resp.status_code == 400
    assert "SO-101" in resp.json()["detail"]


def test_local_start_preempts_a_parked_unseated_hosting_session(
    client, tmp_lerobot_home, _idle, monkeypatch
) -> None:
    """Station mode's "local wins when idle": a flow started at the station
    stops a parked, unseated hosting session instead of being refused."""
    from makermodslab import sessions

    _make_robot("bench", leader=True, follower=True)
    monkeypatch.setattr(remote_host, "hosting_active", True)
    monkeypatch.setattr(remote_host, "phase", "parked")
    monkeypatch.setattr(remote_host, "seat", remote_host.SeatMonitor())
    yielded: list[bool] = []

    def fake_yield(timeout_s: float = 10.0) -> bool:
        yielded.append(True)
        monkeypatch.setattr(remote_host, "hosting_active", False)
        return True

    monkeypatch.setattr(remote_host, "yield_for_local", fake_yield)
    # The local start itself is stubbed: only the gate is under test.
    monkeypatch.setattr(
        sessions,
        "_dispatch_start",
        lambda kind, request, ws: {"success": False, "message": "stub", "status_code": 418},
    )
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert yielded == [True]
    assert resp.status_code == 418  # got past the held gate to the (stubbed) start


def test_local_start_is_refused_while_the_seat_is_held(client, tmp_lerobot_home, _idle, monkeypatch) -> None:
    _make_robot("bench", leader=True, follower=True)
    monkeypatch.setattr(remote_host, "hosting_active", True)
    monkeypatch.setattr(remote_host, "phase", "engaged")
    m = remote_host.SeatMonitor()
    m.operator_joined("laptop")
    monkeypatch.setattr(remote_host, "seat", m)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "session.held"
    assert resp.json()["details"]["holder"]["kind"] == "hosting"


def test_hosting_status_reports_phase_seat_and_station_mode(client, _idle, monkeypatch) -> None:
    descriptor = remote_host.build_descriptor(
        remote_host.HostingRequest(follower_port="/dev/f", follower_config="fc", robot_name="bench"),
        room="mml-x",
        motors=["a"],
        cameras={},
        ranges_deg={},
    )
    m = remote_host.SeatMonitor()
    m.operator_joined("operator-xyz")
    monkeypatch.setattr(remote_host, "hosting_active", True)
    monkeypatch.setattr(remote_host, "current_descriptor", descriptor)
    monkeypatch.setattr(remote_host, "seat", m)
    monkeypatch.setattr(remote_host, "phase", "engaged")
    monkeypatch.setattr(remote_host, "station_mode", True)
    body = client.get("/api/v1/hosting").json()["hosting"]
    assert (body["phase"], body["active_operator"], body["station_mode"]) == ("engaged", "operator-xyz", True)
    caps = client.get("/api/v1/health").json()["capabilities"]["hosting"]
    assert (caps["phase"], caps["active_operator"]) == ("engaged", "operator-xyz")


def test_remote_home_and_engage_routes_refuse_when_idle(client, _idle) -> None:
    for verb in ("home", "engage"):
        body = client.post(f"/api/v1/remote-teleoperation/{verb}").json()
        assert body["success"] is False
