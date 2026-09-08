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

"""TB5 — the open arm type and the arms manifest (docs/extensions/plan.md).

What this file pins, in three groups:

* the manifest — `GET /api/v1/arms` is the one document the UI reads arm
  capabilities from, so its shape is pinned entry by entry against the
  built-in families and against the response model (a `response_model`
  silently filters and materializes, so the builder's dicts and the model
  must agree EXACTLY);
* the refusal gates — an `arm_type` the registry does not know is REFUSED
  with `robot.arm_type.unavailable` everywhere a request could otherwise
  reach a device builder, and never normalized to the SO-101 (the locked
  decision the plan records: a silent fallback would open a Feetech serial
  path against whatever hardware the unknown family actually is);
* the unknown-record behaviour — a hand-edited robot record whose arm type
  is not installed is listed, marked `arm_available: false`, never clean,
  and cannot start anything.

House rules: no hardware — every path that would open a port is stubbed so
that a MISSING gate (the RED state) surfaces as a wrong status code rather
than a serial-port error; `tmp_lerobot_home` for every record; no sleeps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from makermodslab import sessions
from makermodslab.api_errors import ApiError, ErrorCode
from makermodslab.arms import registry
from makermodslab.utils import config as cfg
from tests.mocks import make_arm_family, scratch_registry

UNAVAILABLE = "robot.arm_type.unavailable"
NOT_READY = "robot.not_ready"


@pytest.fixture(autouse=True)
def _fresh_tracker():
    """A gate that is missing (RED) lets a start reach a stubbed handler; make
    sure whatever claim that emitted never leaks into the next test."""
    sessions.tracker.reset()
    yield
    sessions.tracker.reset()


# ---------------------------------------------------------------------------
# Records on (redirected) disk
# ---------------------------------------------------------------------------


def _make_ready_record(name: str) -> None:
    """A fully set-up single SO-101 robot: ports, config names, and every
    referenced calibration file present in the SO-101 libraries."""
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FC.json").write_text("{}")
    (Path(cfg.LEADER_CONFIG_PATH) / "LC.json").write_text("{}")
    cfg.save_robot_record(
        name,
        {
            "mode": "single",
            "leader_port": "/dev/l",
            "follower_port": "/dev/f",
            "leader_config": "LC",
            "follower_config": "FC",
        },
        allow_create=True,
    )


def _hand_edit(name: str, **fields) -> None:
    """Rewrite a record's JSON the way a user with a text editor would — the
    exact scenario TB5 is about. A value of None deletes the key."""
    path = Path(cfg.ROBOTS_PATH) / f"{name}.json"
    data = json.loads(path.read_text())
    for key, value in fields.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data))


def _make_unavailable_record(name: str) -> None:
    """A record where ONLY the arm type is wrong: ports and calibration files
    all present, arm_type hand-edited to a family nobody installed."""
    _make_ready_record(name)
    _hand_edit(name, arm_type="nope")


# ---------------------------------------------------------------------------
# utils.config: the open arm type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("so101", True), ("maker", True), ("metal", True), ("nope", False), (None, False), (3, False)],
)
def test_is_known_arm_type_answers_only_for_registered_strings(value: object, expected: bool) -> None:
    """The one predicate every refusal gate consults: True for a registered
    id, False for an unknown string AND for the non-strings a corrupted
    record can carry (those are normalized, never refused)."""
    assert cfg.is_known_arm_type(value) is expected


def test_is_known_arm_type_reads_the_registry_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tuple captured at import is stale the moment an extension registers
    a family (TB6); the predicate must ask the registry every call."""
    assert cfg.is_known_arm_type("nine") is False
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine"))
    assert cfg.is_known_arm_type("nine") is True


def test_normalize_arm_type_defaults_the_absent_and_preserves_the_unknown() -> None:
    """Records written before the Maker arm existed carry no arm_type and ARE
    SO-101s — so None/""/non-strings default. An unknown STRING is a
    different thing: it must survive the read so the record can be shown as
    unavailable and refused, instead of silently becoming an SO-101."""
    assert cfg.normalize_arm_type(None) == "so101"
    assert cfg.normalize_arm_type("") == "so101"
    assert cfg.normalize_arm_type(7) == "so101"
    assert cfg.normalize_arm_type("metal") == "metal"
    assert cfg.normalize_arm_type("nope") == "nope"


def test_a_hand_edited_unknown_arm_type_reads_back_unchanged(tmp_lerobot_home: Path) -> None:
    _make_unavailable_record("wonky")
    assert cfg.get_robot_record("wonky")["arm_type"] == "nope"


def test_a_record_without_an_arm_type_still_reads_as_so101(tmp_lerobot_home: Path) -> None:
    """The pre-Maker on-disk shape keeps its meaning."""
    _make_ready_record("elder")
    _hand_edit("elder", arm_type=None)
    assert cfg.get_robot_record("elder")["arm_type"] == "so101"


def test_an_unknown_arm_type_is_never_clean_even_when_fully_set_up(tmp_lerobot_home: Path) -> None:
    """Readiness is what unlocks every start button. The unknown record has
    every port and every calibration file its twin has — the SO-101 reads
    clean, the unknown one must not, in both scopes, and without a library
    lookup (which would have to raise for a family with no library)."""
    _make_ready_record("twin")
    _make_unavailable_record("wonky")

    assert cfg.is_robot_record_clean(cfg.get_robot_record("twin")) is True
    wonky = cfg.get_robot_record("wonky")
    assert cfg.is_robot_record_clean(wonky) is False
    assert cfg.is_robot_record_clean(wonky, arms="follower") is False


# ---------------------------------------------------------------------------
# HTTP: the robot record surface
# ---------------------------------------------------------------------------


def test_robot_listings_flag_an_unknown_arm_type_as_unavailable(client, tmp_lerobot_home: Path) -> None:
    """The record is LISTED (never hidden, never rewritten) with
    `arm_available: false`, and it is not clean; the SO-101 beside it is
    available and clean. Same flag on the single-record read."""
    _make_ready_record("twin")
    _make_unavailable_record("wonky")

    resp = client.get("/api/v1/robots")
    assert resp.status_code == 200
    by_name = {r["name"]: r for r in resp.json()["robots"]}
    assert by_name["wonky"]["arm_type"] == "nope"
    assert by_name["wonky"]["arm_available"] is False
    assert by_name["wonky"]["is_clean"] is False
    assert by_name["wonky"]["follower_ready"] is False
    assert by_name["twin"]["arm_available"] is True
    assert by_name["twin"]["is_clean"] is True

    single = client.get("/api/v1/robots/wonky").json()["robot"]
    assert single["arm_available"] is False


def test_creating_a_robot_with_an_unknown_arm_type_is_refused(client, tmp_lerobot_home: Path) -> None:
    """The API layer refuses what the disk layer would happily write."""
    resp = client.post("/api/v1/robots/newbot?create=true", json={"arm_type": "nope"})
    assert resp.status_code == 400
    assert resp.json()["code"] == UNAVAILABLE
    assert cfg.get_robot_record("newbot") is None


def test_patching_a_robot_to_an_unknown_arm_type_is_refused_and_writes_nothing(
    client, tmp_lerobot_home: Path
) -> None:
    """The whole body is refused, not just the arm_type key: a partial write
    would leave a record half-switched."""
    cfg.save_robot_record("so", {"leader_port": "/dev/a"}, allow_create=True)

    resp = client.post("/api/v1/robots/so", json={"arm_type": "nope", "leader_port": "/dev/b"})
    assert resp.status_code == 400
    assert resp.json()["code"] == UNAVAILABLE

    record = cfg.get_robot_record("so")
    assert record["arm_type"] == "so101"
    assert record["leader_port"] == "/dev/a"


def test_patching_without_an_arm_type_is_unaffected(client, tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("so", {"leader_port": "/dev/a"}, allow_create=True)
    resp = client.post("/api/v1/robots/so", json={"leader_port": "/dev/b"})
    assert resp.status_code == 200
    assert cfg.get_robot_record("so")["leader_port"] == "/dev/b"


def test_a_null_arm_type_in_the_body_is_unspecified_not_unknown(client, tmp_lerobot_home: Path) -> None:
    """A client that always sends the key with null (the shape before the
    Maker arm existed) still creates an SO-101: null, like absence, means
    "unspecified", and only a STRING nothing registered is refused."""
    resp = client.post("/api/v1/robots/newbot?create=true", json={"arm_type": None, "mode": "single"})
    assert resp.status_code == 200, resp.text
    assert cfg.get_robot_record("newbot")["arm_type"] == "so101"


def test_an_empty_arm_type_query_still_serves_the_default_library(client, tmp_lerobot_home: Path) -> None:
    """`?arm_type=` is the SO-101 library, as it always was — the gate reads
    an empty value the way normalize_arm_type does (unspecified → default),
    so the refusal is reserved for a family that is actually not installed."""
    resp = client.get("/api/v1/calibration-configs/robot?arm_type=")
    assert resp.status_code == 200, resp.text


def test_open_calibration_folder_refuses_an_unknown_arm_type(client, tmp_lerobot_home: Path) -> None:
    """The folder opener resolves the library dir from the arm type like the
    calibration-configs routes do, so it carries the same gate — before any
    directory is created or a file browser is spawned."""
    resp = client.post("/api/v1/open-calibration-folder", json={"device_type": "robot", "arm_type": "nope"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("GET", "/api/v1/calibration-configs/robot?arm_type=nope", None),
        ("DELETE", "/api/v1/calibration-configs/robot/x?arm_type=nope", None),
        ("GET", "/api/v1/calibration-configs/robot/x/download?arm_type=nope", None),
        ("POST", "/api/v1/calibration-configs/robot/upload?arm_type=nope", {"name": "x", "data": {}}),
        ("POST", "/api/v1/calibration-configs/robot/x/rename?arm_type=nope", {"new_name": "y"}),
    ],
)
def test_calibration_config_routes_refuse_an_unknown_arm_type_query(
    client, tmp_lerobot_home: Path, method: str, url: str, body: dict | None
) -> None:
    """`?arm_type=nope` used to resolve to the SO-101 library; it is a 400
    now, so a client can never read, delete or write the wrong family's
    files by naming a family that is not installed."""
    resp = client.request(method, url, json=body)
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE


# ---------------------------------------------------------------------------
# HTTP: POST /api/v1/sessions — every startable kind
# ---------------------------------------------------------------------------

# Options that pass each kind's schema; a guard test below keeps this table in
# step with STARTABLE_KINDS so a new kind cannot slip past the gate untested.
_VALID_OPTIONS: dict[str, dict] = {
    "teleoperation": {},
    "recording": {"dataset_repo_id": "u/d", "single_task": "t"},
    "inference": {"policy_ref": "user/repo@checkpoints/000050"},
    "replay": {"repo_id": "u/d", "episode_index": 0},
    "calibration": {"device_type": "robot", "arm": "left"},
    "auto_calibration": {"arms": [{"device_type": "robot", "arm": "left"}]},
}


def test_every_startable_kind_has_options_here() -> None:
    assert set(_VALID_OPTIONS) == set(sessions.STARTABLE_KINDS)


@pytest.mark.parametrize("kind", sessions.STARTABLE_KINDS)
def test_no_session_kind_can_start_on_an_unknown_arm_type(
    client, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The gate sits right after record resolution and BEFORE readiness, so
    the 400 names the real reason: not "not ready" (the record has every
    port and file) but "this arm type is not installed". Dispatch is stubbed
    to answer with a tell-tale code, so a missing gate fails on the code,
    not on a serial port."""
    _make_unavailable_record("wonky")
    monkeypatch.setattr(
        sessions,
        "_dispatch_start",
        lambda kind, request, ws: {
            "success": False,
            "status_code": 400,
            "message": "dispatch reached: the arm-type gate is missing",
            "code": "test.dispatch_reached",
        },
    )

    resp = client.post(
        "/api/v1/sessions", json={"kind": kind, "robot": "wonky", "options": _VALID_OPTIONS[kind]}
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == UNAVAILABLE
    assert body["code"] != NOT_READY
    assert "nope" in body["detail"]


# ---------------------------------------------------------------------------
# The legacy start handlers (external clients still post to them directly)
# ---------------------------------------------------------------------------


def _assert_unavailable(excinfo: pytest.ExceptionInfo[ApiError]) -> None:
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.ROBOT_ARM_TYPE_UNAVAILABLE
    assert "'nope'" in excinfo.value.detail


def test_legacy_move_arm_refuses_an_unknown_arm_type(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /move-arm is the flat teleoperation start. The refusal is the
    coded 400 (the app-wide ApiError handler renders it), not the handler's
    legacy `{"success": false}` dict at 200."""
    from makermodslab import teleoperate

    def _never(*args, **kwargs):
        raise RuntimeError("the arm-type gate is missing: a device config was built")

    monkeypatch.setattr(teleoperate, "build_single_configs", _never)
    resp = client.post(
        "/api/v1/move-arm",
        json={
            "leader_port": "/dev/null-l",
            "follower_port": "/dev/null-f",
            "leader_config": "lc",
            "follower_config": "fc",
            "arm_type": "nope",
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE
    assert teleoperate.teleoperation_active is False


def test_handle_start_recording_refuses_an_unknown_arm_type(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the device configs is the first thing a gate-less handler reaches
    after its mutex; stubbed to fail, it answers with the legacy refusal dict
    (no thread, no port) — the gate must come first and raise instead."""
    from makermodslab import record

    def _never(*args, **kwargs):
        raise RuntimeError("the arm-type gate is missing: a device config was built")

    monkeypatch.setattr(record, "build_single_configs", _never)
    request = record.RecordingRequest(
        leader_port="/dev/null-l",
        follower_port="/dev/null-f",
        leader_config="lc",
        follower_config="fc",
        dataset_repo_id="u/d",
        single_task="t",
        arm_type="nope",
    )
    with pytest.raises(ApiError) as excinfo:
        record.handle_start_recording(request)
    _assert_unavailable(excinfo)
    assert record.recording_active is False


def test_handle_start_inference_refuses_an_unknown_arm_type() -> None:
    """checkpoint_state_dim=12 on a single robot trips the arm-count guard in
    a gate-less handler (a dict, no thread) — and that guard reads the width
    off the arm type, so an unknown one must be refused before it."""
    from makermodslab import rollout

    request = rollout.InferenceRequest(
        follower_port="/dev/null-f",
        follower_config="fc",
        policy_ref="user/repo@checkpoints/000050",
        mode="single",
        checkpoint_state_dim=12,
        arm_type="nope",
    )
    with pytest.raises(ApiError) as excinfo:
        rollout.handle_start_inference(request)
    _assert_unavailable(excinfo)
    assert rollout.inference_active is False


def test_handle_start_replay_refuses_an_unknown_arm_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "_load_robot_record", lambda name: {"mode": "single"})
    monkeypatch.setattr(replay, "get_episode_action_series", lambda repo, ep: None)
    request = replay.ReplayRequest(
        repo_id="u/d",
        episode_index=0,
        follower_port="/dev/null-f",
        follower_config="fc",
        robot_name="m",
        arm_type="nope",
    )
    with pytest.raises(ApiError) as excinfo:
        replay.handle_start_replay(request)
    _assert_unavailable(excinfo)
    assert replay.replay_active is False


# ---------------------------------------------------------------------------
# The CAN-only requests: validation by the family's FLAGS, not by id
# ---------------------------------------------------------------------------


@pytest.fixture
def _idle_step_calibration(monkeypatch: pytest.MonkeyPatch):
    """Stub the worker so a gate-less start (RED) claims and returns instead
    of opening a bus; release whatever it claimed afterwards."""
    from makermodslab import step_calibrate

    manager = step_calibrate.step_calibration_manager
    monkeypatch.setattr(manager, "_worker", lambda request: None)
    yield manager
    if step_calibrate.step_calibration_is_active():
        manager.stop()


def test_step_calibration_start_refuses_a_range_sweep_family(
    tmp_lerobot_home: Path, _idle_step_calibration
) -> None:
    """An SO-101 has no steps to run — it is calibrated by a range sweep.
    The old schema Literal rejected it as a 422 with no reason; the handler
    refuses with the same coded 400 the sessions surface uses for the mirror
    case (auto-calibration on a CAN arm), naming the family's kind."""
    from makermodslab.step_calibrate import StepCalibrationRequest, step_calibration_is_active

    with pytest.raises(ApiError) as excinfo:
        _idle_step_calibration.start(
            StepCalibrationRequest(
                device_type="robot", port="/dev/null-f", config_file="zc", arm_type="so101"
            )
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.ROBOT_NOT_READY
    assert "range_sweep" in excinfo.value.detail
    assert step_calibration_is_active() is False


def test_step_calibration_start_refuses_an_unknown_arm_type(
    tmp_lerobot_home: Path, _idle_step_calibration
) -> None:
    from makermodslab.step_calibrate import StepCalibrationRequest, step_calibration_is_active

    with pytest.raises(ApiError) as excinfo:
        _idle_step_calibration.start(
            StepCalibrationRequest(device_type="robot", port="/dev/null-f", config_file="zc", arm_type="nope")
        )
    _assert_unavailable(excinfo)
    assert step_calibration_is_active() is False


def test_step_calibration_request_carries_any_arm_type_string() -> None:
    """The request is typed `str`: a family an extension registers must be
    able to ride the same request the built-ins use."""
    from makermodslab.step_calibrate import StepCalibrationRequest

    request = StepCalibrationRequest(
        device_type="robot", port="/dev/x", config_file="c", arm_type="so101_twin"
    )
    assert request.arm_type == "so101_twin"
    assert StepCalibrationRequest(device_type="robot", port="/dev/x", config_file="c").arm_type == "maker"


def test_can_only_request_models_accept_any_arm_type_string() -> None:
    """Opening the Literals: pydantic no longer rejects a family it has not
    heard of — the handler does, by the family's flags."""
    from makermodslab.can_recovery import ReleaseCanTorqueRequest
    from makermodslab.server import MakerIdentifyArmRequest, MakerProbePortsRequest

    assert MakerProbePortsRequest.model_validate({"arm_type": "so101_twin"}).arm_type == "so101_twin"
    assert MakerProbePortsRequest.model_validate({}).arm_type == "maker"
    assert (
        MakerIdentifyArmRequest.model_validate({"device_type": "teleop", "arm_type": "so101_twin"}).arm_type
        == "so101_twin"
    )
    assert (
        ReleaseCanTorqueRequest.model_validate({"arm_type": "so101_twin", "port": "/dev/can0"}).arm_type
        == "so101_twin"
    )


@pytest.fixture
def _no_probe(monkeypatch: pytest.MonkeyPatch):
    from makermodslab import maker_ports

    async def _never(*args, **kwargs):
        raise AssertionError("the probe must not run for a refused request")

    monkeypatch.setattr(maker_ports, "probe_maker_ports", _never)
    monkeypatch.setattr(maker_ports, "identify_maker_arm_by_motion", _never)


def test_probe_ports_refuses_a_family_without_a_protocol_probe(client, _no_probe) -> None:
    """The SO-101's two halves speak one protocol, so `follower_probe_protocol`
    is None — nothing to ask a port. Refused by that FLAG (400 not_ready,
    naming the gesture as the alternative), not by comparing ids."""
    resp = client.post("/api/v1/maker/probe-ports", json={"arm_type": "so101"})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == NOT_READY
    assert "gesture" in body["detail"]


def test_probe_ports_refuses_an_unknown_arm_type(client, _no_probe) -> None:
    resp = client.post("/api/v1/maker/probe-ports", json={"arm_type": "nope"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE


def test_identify_arm_accepts_every_known_family_and_refuses_an_unknown_one(
    client, _no_probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every family implements identify_by_motion (the SO-101 through
    identify.py), so the route takes any KNOWN id — the old Literal shut the
    SO-101 out for no hardware reason."""
    from makermodslab import identify

    async def fake_so(ports=None):
        return {"success": True, "message": "seen", "port": "/dev/y"}

    monkeypatch.setattr(identify, "identify_arm_by_motion", fake_so)

    resp = client.post("/api/v1/maker/identify-arm", json={"device_type": "teleop", "arm_type": "so101"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["port"] == "/dev/y"

    resp = client.post("/api/v1/maker/identify-arm", json={"device_type": "teleop", "arm_type": "nope"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE


@pytest.fixture
def _release_torque_idle(monkeypatch: pytest.MonkeyPatch):
    """No session holds the hardware, and no device may ever be built."""
    from makermodslab import can_recovery, sessions as _sessions

    monkeypatch.setattr(_sessions, "_held_by", lambda: None)

    def _never(arm_type, port):
        raise AssertionError("no device may be built for a refused release")

    monkeypatch.setattr(can_recovery, "_build_follower_device", _never)


def test_release_torque_refuses_a_feetech_family(client, _release_torque_idle) -> None:
    """An SO-101 goes limp on its own when its process dies — there is nothing
    to recover, and a CAN de-energize aimed at a Feetech serial port is
    nonsense. Refused by `uses_feetech_bus`, in the handler, as a coded 400
    (the schema used to say 422 and nothing else)."""
    resp = client.post("/api/v1/arms/release-torque", json={"arm_type": "so101", "port": "/dev/tty0"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == NOT_READY


def test_release_torque_refuses_an_unknown_arm_type(client, _release_torque_idle) -> None:
    resp = client.post("/api/v1/arms/release-torque", json={"arm_type": "nope", "port": "/dev/can0"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == UNAVAILABLE


# ---------------------------------------------------------------------------
# The manifest: GET /api/v1/arms
# ---------------------------------------------------------------------------

# The two ends of the built-in range, pinned whole. Values come from the
# families themselves (arms/so101.py, arms/metal.py, arms/can_common.py):
# a change to any of them is a UI-visible change and must come past this.
SO101_ENTRY = {
    "id": "so101",
    "label": "SO-101",
    "short_label": "SO-101",
    "provided_by": "builtin",
    "calibration_name_suffix": "",
    "joints_per_arm": 6,
    "supports_bimanual": True,
    "image_url": None,
    "calibration": {"kind": "range_sweep", "summary": None, "panel_url": None},
    "telemetry_kind": "urdf",
    "capabilities": {
        "uses_feetech_bus": True,
        "supports_auto_calibration": True,
        "supports_dagger": True,
        "supports_port_probe": False,
        "motion_identify_energizes_follower": False,
    },
    "robot_types": ["so101_follower", "bi_so_follower"],
    "robot_type_markers": ["so100", "so101", "so-100", "so-101", "so_follower", "so_leader"],
}

METAL_ENTRY = {
    "id": "metal",
    "label": "Metal Arm",
    "short_label": "Metal",
    "provided_by": "builtin",
    "calibration_name_suffix": "_metal",
    "joints_per_arm": 7,
    "supports_bimanual": True,
    "image_url": None,
    "calibration": {
        "kind": "steps",
        "summary": {
            "leader": {
                "text": (
                    "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, "
                    "gripper closed — then confirm."
                ),
                "image_url": None,
            },
            "follower": {
                "text": (
                    "Move the arm by hand to its ZERO POSE — standing upright, all "
                    "joints at 0 degrees, gripper closed — then confirm."
                ),
                "image_url": None,
            },
        },
        "panel_url": None,
    },
    "telemetry_kind": "degrees",
    "capabilities": {
        "uses_feetech_bus": False,
        "supports_auto_calibration": False,
        "supports_dagger": False,
        "supports_port_probe": True,
        "motion_identify_energizes_follower": True,
    },
    "robot_types": ["metal_follower", "bi_metal_follower"],
    "robot_type_markers": ["metal"],
}


def test_manifest_lists_the_built_ins_in_registry_order_with_the_documented_shape(client) -> None:
    resp = client.get("/api/v1/arms")
    assert resp.status_code == 200, resp.text
    arms = resp.json()["arms"]
    assert [a["id"] for a in arms] == list(registry.ids())
    by_id = {a["id"]: a for a in arms}
    assert by_id["so101"] == SO101_ENTRY
    assert by_id["metal"] == METAL_ENTRY


def test_manifest_calibration_summary_is_the_families_own(client) -> None:
    """The UI shows the summary BEFORE Start, so it must be the family's own
    per-side answer verbatim — and absent (null, not {}) for a family with
    nothing to summarize; the panel URL is the family's attribute."""
    arms = {a["id"]: a for a in client.get("/api/v1/arms").json()["arms"]}
    for family in registry.families():
        entry = arms[family.id]["calibration"]
        assert entry["kind"] == family.calibration_kind
        assert entry["panel_url"] == family.calibration_panel_url
        if family.calibration_kind == "steps":
            assert entry["summary"] == {
                "leader": family.calibration_summary("teleop"),
                "follower": family.calibration_summary("robot"),
            }
        else:
            assert entry == {"kind": "range_sweep", "summary": None, "panel_url": None}
        assert arms[family.id]["image_url"] == family.image_url is None


def test_manifest_models_name_the_three_kinds_and_forget_zero_pose() -> None:
    """The wire shape TB6a settles on: ``ArmCalibrationInfo.kind`` is one of
    the three registry kinds (``zero_pose`` is gone with ``ArmZeroPose``),
    a summary is two optional sides of ``{text, image_url}``, and the family
    entry carries an optional served image."""
    from pydantic import ValidationError

    from makermodslab.schemas import system as schemas

    assert not hasattr(schemas, "ArmZeroPose")
    side = schemas.ArmCalibrationSide(text="Pose it", image_url=None)
    summary = schemas.ArmCalibrationSummary(leader=side, follower=None)
    info = schemas.ArmCalibrationInfo(kind="steps", summary=summary, panel_url=None)
    assert info.model_dump() == {
        "kind": "steps",
        "summary": {"leader": {"text": "Pose it", "image_url": None}, "follower": None},
        "panel_url": None,
    }
    for kind in ("range_sweep", "steps", "panel"):
        schemas.ArmCalibrationInfo(kind=kind, summary=None, panel_url=None)
    with pytest.raises(ValidationError):
        schemas.ArmCalibrationInfo(kind="zero_pose", summary=None, panel_url=None)
    assert "image_url" in schemas.ArmFamilyInfo.model_fields


def test_manifest_builder_dicts_are_described_exactly_by_the_response_model() -> None:
    """`response_model` filters undeclared fields and materializes absent ones
    silently; the builder's dicts and the model must agree key for key."""
    from makermodslab.arms.manifest import arms_manifest, describe_family
    from makermodslab.schemas.system import ArmFamiliesResponse

    manifest = arms_manifest()
    assert ArmFamiliesResponse.model_validate({"arms": manifest}).model_dump() == {"arms": manifest}
    assert describe_family(registry.get("so101")) == SO101_ENTRY
    assert describe_family(registry.get("metal")) == METAL_ENTRY


def test_manifest_includes_a_later_registered_family_with_its_provenance(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What TB6 relies on: a family registered after import shows up with the
    name of whoever provided it, and the default family stays FIRST."""
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine", joints_per_arm=9), provided_by="ext")

    arms = client.get("/api/v1/arms").json()["arms"]
    assert arms[0]["id"] == registry.DEFAULT_ID
    assert arms[-1]["id"] == "nine"
    nine = arms[-1]
    assert nine["provided_by"] == "ext"
    assert nine["joints_per_arm"] == 9
    assert nine["calibration_name_suffix"] == "_nine"
    assert nine["calibration"]["kind"] == "steps"
    assert nine["image_url"] is None
    assert nine["capabilities"]["supports_port_probe"] is False
    assert all(a["provided_by"] == "builtin" for a in arms[:-1])


def test_manifest_serves_a_panel_family_with_its_url_and_a_served_image(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two TB6a fields an extension fills that no built-in does: the
    panel URL the dialog will mount (TB6b) and the image the create dialog
    shows when it has no bundled photo for the id."""
    from makermodslab.arms.manifest import describe_family
    from makermodslab.schemas.system import ArmFamilyInfo

    scratch_registry(monkeypatch)
    registry.register(
        make_arm_family(
            "paneled",
            calibration_kind="panel",
            calibration_panel_url="/api/v1/ext/p/static/calibrate.html",
            image_url="/api/v1/ext/p/static/arm.png",
        ),
        provided_by="p",
    )

    entry = {a["id"]: a for a in client.get("/api/v1/arms").json()["arms"]}["paneled"]
    assert entry["calibration"] == {
        "kind": "panel",
        "summary": None,
        "panel_url": "/api/v1/ext/p/static/calibrate.html",
    }
    assert entry["image_url"] == "/api/v1/ext/p/static/arm.png"
    assert entry["provided_by"] == "p"
    built = describe_family(registry.get("paneled"))
    assert ArmFamilyInfo.model_validate(built).model_dump() == built == entry


def test_manifest_order_lets_a_client_scan_markers_with_the_default_last(client) -> None:
    """The order contract the frontend's marker scan depends on: registry
    order, default family FIRST. Its markers are the loosest (so_leader), so
    a client scans every other family's markers in manifest order and checks
    the default last — reproducing arm_type_from_robot_type exactly."""
    from makermodslab.arm_capabilities import arm_type_from_robot_type

    arms = client.get("/api/v1/arms").json()["arms"]
    assert arms[0]["id"] == registry.DEFAULT_ID

    def client_scan(robot_type: str) -> str | None:
        text = robot_type.strip().lower()
        for family in arms[1:] + arms[:1]:
            if any(marker in text for marker in family["robot_type_markers"]):
                return family["id"]
        return None

    for robot_type in (
        "so101_follower",
        "bi_so_follower",
        "so_leader",
        "maker_follower",
        "bi_metal_follower",
        "experimental_metal_rig",
        "aloha",
    ):
        assert client_scan(robot_type) == arm_type_from_robot_type(robot_type), robot_type


def test_manifest_route_is_v1_only_in_the_system_namespace(client) -> None:
    """Born versioned: no flat mirror (that surface only shrinks), operation id
    `list_arm_families` (it becomes an SDK method name), tag `system` (beside
    probe/identify/release-torque), typed with ArmFamiliesResponse."""
    from makermodslab.schemas.system import ArmFamiliesResponse
    from makermodslab.server import app

    assert client.get("/arms").status_code == 404

    from tests.test_api_contract import _walk_routes

    # _walk_routes traverses FastAPI >= 0.138's lazy _IncludedRouter entries; a
    # bare scan of app.routes no longer sees the API routes (see test_server.py).
    routes = [r for path, r in _walk_routes(app.routes) if isinstance(r, APIRoute) and path == "/api/v1/arms"]
    assert len(routes) == 1, "GET /api/v1/arms must be mounted exactly once"
    route = routes[0]
    assert route.methods == {"GET"}
    assert route.endpoint.__name__ == "list_arm_families"
    assert route.tags == ["system"]
    assert route.response_model is ArmFamiliesResponse
