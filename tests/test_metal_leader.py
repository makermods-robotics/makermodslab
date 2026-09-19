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
"""Metal-to-Metal teleop: the Metal arm's second, gravity-compensated leader.

A robot record chooses its leader with ``leader_kind`` ("star", the default
Star Arm 102, or "metal", a second Metal arm). What these pin:

* the manifest publishes the option with its live availability (Pinocchio
  installed or not) and the record field round-trips, normalizes, refuses an
  unknown kind, and blanks the leader slots on a switch;
* the family picks lerobot's metal_leader classes for the kind, with the
  fork's hold-on-disconnect pinned OFF, and never for the Star kind;
* every start that opens the leader refuses with robot.leader_kind.unavailable
  while the extra is missing; follower-only kinds and calibration do not;
* the Metal leader calibrates like a follower (bus open, torque off, set zero,
  the FOLLOWER's joint limits) and is released like one afterwards;
* port detection: the probe cannot tell the halves apart and says so; the
  gesture is refused on both sides with ``fallback: "wiggle"``; and the
  gripper wiggle (can_wiggle.py) — the identification of last resort —
  enables only the gripper, stays inside its limits, disables it after even
  when the drive raises partway, and honours the busy matrix;
* the stop path captures, returns (gravity thread stopped first) and
  releases the energized leader like a follower.

No hardware, no sleeps: every bus is a fake and every sleep is injected.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from makermodslab import can_wiggle, maker_ports, maker_rest_pose, sessions, torque
from makermodslab.api_errors import ApiError, ErrorCode
from makermodslab.arm_capabilities import require_leader_available, require_leader_kind
from makermodslab.arms import metal as metal_module, registry
from makermodslab.arms.base import LeaderOption, leader_kwargs
from makermodslab.utils import config as cfg

METAL = registry.get("metal")
MAKER = registry.get("maker")
SO101 = registry.get("so101")


@pytest.fixture(autouse=True)
def _fresh_tracker():
    sessions.tracker.reset()
    yield
    sessions.tracker.reset()


@pytest.fixture
def leader_installed(monkeypatch: pytest.MonkeyPatch):
    """Pretend Pinocchio is importable — the manifest reads availability live."""
    monkeypatch.setattr(metal_module, "_metal_leader_available", lambda: True)


@pytest.fixture
def leader_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metal_module, "_metal_leader_available", lambda: False)


# ---------------------------------------------------------------------------
# The contract: leader options
# ---------------------------------------------------------------------------


def test_only_the_metal_family_offers_a_second_leader() -> None:
    assert [o.id for o in SO101.leader_options()] == ["so101"]
    # Maker's second kind is the Star leader with the trigger grip; Metal's are its
    # own arm and the Star leader with the vertical grip (star_gripper.py).
    assert [o.id for o in MAKER.leader_options()] == ["star", "star_trigger"]
    assert [o.id for o in METAL.leader_options()] == ["star", "metal", "star_vertical"]
    assert METAL.leader_options()[0] == LeaderOption(id="star", label="Star Arm 102 leader")
    assert METAL.leader_option("metal").energized is True
    assert METAL.leader_holds_torque("star") is False
    assert METAL.leader_holds_torque(None) is False
    assert METAL.leader_holds_torque("metal") is True


def test_a_single_leader_family_never_receives_the_keyword() -> None:
    """The rule every leader-side call site follows, so an extension family
    written before leader kinds existed keeps its old signatures."""
    assert leader_kwargs(SO101, "so101") == {}
    assert leader_kwargs(MAKER, None) == {"leader_kind": "star"}
    assert leader_kwargs(METAL, None) == {"leader_kind": "star"}
    assert leader_kwargs(METAL, "metal") == {"leader_kind": "metal"}


def test_an_unknown_leader_kind_raises_a_key_error_naming_the_offered_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        METAL.leader_option("nope")
    assert "star, metal" in str(excinfo.value)


def test_availability_is_read_live_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metal_module, "_metal_leader_available", lambda: False)
    option = METAL.leader_option("metal")
    assert option.available is False
    assert "metal-leader" in option.unavailable_reason
    monkeypatch.setattr(metal_module, "_metal_leader_available", lambda: True)
    assert METAL.leader_option("metal") == LeaderOption(
        id="metal", label="Metal arm leader (gravity-compensated)", energized=True
    )


def test_manifest_publishes_the_options_with_availability(client, leader_missing) -> None:
    arms = {a["id"]: a for a in client.get("/api/v1/arms").json()["arms"]}
    metal = arms["metal"]
    assert metal["default_leader_kind"] == "star"
    assert [o["id"] for o in metal["leader_options"]] == ["star", "metal", "star_vertical"]
    option = metal["leader_options"][1]
    assert option["available"] is False
    assert "pip install" in option["unavailable_reason"]
    assert option["energized"] is True
    # The Metal leader IS a Metal arm: its pre-start summary is the follower's pose.
    assert option["calibration_summary"] == metal["calibration"]["summary"]["follower"]
    assert metal["capabilities"]["supports_gripper_wiggle"] is True
    assert arms["so101"]["leader_options"] == [
        {
            "id": "so101",
            "label": "SO-101 leader",
            "available": True,
            "unavailable_reason": None,
            "energized": False,
            "calibration_summary": None,
        }
    ]


def test_the_metal_leader_keeps_a_library_of_its_own(tmp_lerobot_home: Path) -> None:
    star = METAL.leader_calibration_dir(leader_kind="star")
    own = METAL.leader_calibration_dir(leader_kind="metal")
    assert star == MAKER.leader_calibration_dir() == cfg.MAKER_LEADER_CONFIG_PATH
    assert own == cfg.METAL_LEADER_CONFIG_PATH != star
    assert cfg.leader_config_path_for("metal", "metal") == own
    assert cfg.calibration_dir_for_device("teleop", "metal", "metal") == own
    assert cfg.calibration_dir_for_device("teleop", "metal") == star


# ---------------------------------------------------------------------------
# The robot record
# ---------------------------------------------------------------------------


def test_a_record_reads_its_leader_kind_back_and_defaults_a_missing_one(tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("m", {"arm_type": "metal"}, allow_create=True)
    assert cfg.get_robot_record("m")["leader_kind"] == "star"
    cfg.save_robot_record("m", {"leader_kind": "metal"}, allow_create=False)
    assert cfg.get_robot_record("m")["leader_kind"] == "metal"
    # A record written before leader kinds existed: the key is absent on disk.
    path = Path(cfg.ROBOTS_PATH) / "m.json"
    data = json.loads(path.read_text())
    data.pop("leader_kind")
    path.write_text(json.dumps(data))
    assert cfg.get_robot_record("m")["leader_kind"] == "star"
    # An SO-101 record answers its own family's default.
    cfg.save_robot_record("s", {}, allow_create=True)
    assert cfg.get_robot_record("s")["leader_kind"] == "so101"


def test_a_hand_edited_unknown_leader_kind_is_kept_and_never_clean(tmp_lerobot_home: Path) -> None:
    _make_ready_metal_record("m", leader_kind="metal")
    path = Path(cfg.ROBOTS_PATH) / "m.json"
    data = json.loads(path.read_text())
    data["leader_kind"] = "nope"
    path.write_text(json.dumps(data))
    record = cfg.get_robot_record("m")
    assert record["leader_kind"] == "nope"
    assert cfg.is_robot_record_clean(record) is False
    assert cfg.is_robot_record_clean(record, arms="follower") is True


def test_switching_leader_kind_blanks_the_leader_slots_only(tmp_lerobot_home: Path) -> None:
    _make_ready_metal_record("m", leader_kind="star")
    cfg.save_robot_record("m", {"leader_kind": "metal"}, allow_create=False)
    record = cfg.get_robot_record("m")
    assert record["leader_kind"] == "metal"
    assert (record["leader_port"], record["leader_config"]) == ("", "")
    assert (record["follower_port"], record["follower_config"]) == ("/dev/f", "FC")
    # Fields the same payload sets survive the switch.
    cfg.save_robot_record("m", {"leader_kind": "star", "leader_port": "/dev/star"}, allow_create=False)
    record = cfg.get_robot_record("m")
    assert (record["leader_kind"], record["leader_port"], record["leader_config"]) == (
        "star",
        "/dev/star",
        "",
    )
    # Re-saving the same kind is a no-op on the slots.
    cfg.save_robot_record("m", {"leader_config": "LC2"}, allow_create=False)
    cfg.save_robot_record("m", {"leader_kind": "star"}, allow_create=False)
    assert cfg.get_robot_record("m")["leader_config"] == "LC2"


def test_switching_arm_type_resets_the_leader_kind(tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("m", {"arm_type": "metal", "leader_kind": "metal"}, allow_create=True)
    cfg.save_robot_record("m", {"arm_type": "maker"}, allow_create=False)
    assert cfg.get_robot_record("m")["leader_kind"] == "star"
    cfg.save_robot_record("m", {"arm_type": "so101"}, allow_create=False)
    assert cfg.get_robot_record("m")["leader_kind"] == "so101"


def test_readiness_needs_an_available_leader_unless_follower_only(
    tmp_lerobot_home: Path, leader_missing
) -> None:
    _make_ready_metal_record("m", leader_kind="metal")
    record = cfg.get_robot_record("m")
    assert cfg.is_robot_record_clean(record) is False
    assert cfg.is_robot_record_clean(record, arms="follower") is True
    metal_module._metal_leader_available = lambda: True  # restored by the fixture's monkeypatch
    assert cfg.is_robot_record_clean(cfg.get_robot_record("m")) is True


def test_upsert_refuses_a_leader_kind_the_family_does_not_offer(client, tmp_lerobot_home: Path) -> None:
    resp = client.post("/api/v1/robots/m?create=true", json={"arm_type": "metal", "leader_kind": "nope"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNKNOWN
    assert "star, metal" in resp.json()["detail"]
    assert cfg.get_robot_record("m") is None, "refused whole: nothing was written"

    # The kind is validated against the family the record WILL have.
    client.post("/api/v1/robots/s?create=true", json={"arm_type": "so101"})
    resp = client.post("/api/v1/robots/s", json={"leader_kind": "metal"})
    assert resp.status_code == 400
    assert resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNKNOWN
    resp = client.post("/api/v1/robots/s", json={"arm_type": "metal", "leader_kind": "metal"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["robot"]["leader_kind"] == "metal"


def test_an_unavailable_but_offered_leader_can_still_be_saved(
    client, tmp_lerobot_home: Path, leader_missing
) -> None:
    """The user installs the extra later; the record is theirs to write. It
    lists as not ready and every leader-opening start refuses meanwhile."""
    resp = client.post("/api/v1/robots/m?create=true", json={"arm_type": "metal", "leader_kind": "metal"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["robot"]["leader_kind"] == "metal"


def test_the_gates_carry_their_codes(leader_missing) -> None:
    with pytest.raises(ApiError) as excinfo:
        require_leader_kind("metal", "nope")
    assert (excinfo.value.status_code, excinfo.value.code) == (400, ErrorCode.ROBOT_LEADER_KIND_UNKNOWN)
    require_leader_kind("metal", None)
    require_leader_kind("metal", "")
    with pytest.raises(ApiError) as excinfo:
        require_leader_available("metal", "metal")
    assert (excinfo.value.status_code, excinfo.value.code) == (400, ErrorCode.ROBOT_LEADER_KIND_UNAVAILABLE)
    assert "metal-leader" in excinfo.value.detail
    require_leader_available("metal", "star")
    require_leader_available("so101", None)


# ---------------------------------------------------------------------------
# Sessions: what refuses, what does not
# ---------------------------------------------------------------------------


def _make_ready_metal_record(name: str, leader_kind: str = "star") -> None:
    leader_dir = Path(cfg.leader_config_path_for("metal", leader_kind))
    leader_dir.mkdir(parents=True, exist_ok=True)
    (leader_dir / "LC.json").write_text("{}")
    (Path(cfg.METAL_FOLLOWER_CONFIG_PATH) / "FC.json").write_text("{}")
    cfg.save_robot_record(
        name,
        {
            "arm_type": "metal",
            "leader_kind": leader_kind,
            "mode": "single",
            "leader_port": "/dev/l",
            "follower_port": "/dev/f",
            "leader_config": "LC",
            "follower_config": "FC",
        },
        allow_create=True,
    )


_OPTIONS = {
    "teleoperation": {},
    "recording": {"dataset_repo_id": "u/d", "single_task": "t"},
    "inference": {"policy_ref": "user/repo@checkpoints/000050"},
    "replay": {"repo_id": "u/d", "episode_index": 0},
    "calibration": {"device_type": "teleop", "arm": "left"},
}


@pytest.mark.parametrize("kind", ["teleoperation", "recording"])
def test_starts_that_open_the_leader_refuse_while_the_extra_is_missing(
    client, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, leader_missing, kind: str
) -> None:
    _make_ready_metal_record("m", leader_kind="metal")
    monkeypatch.setattr(
        sessions,
        "_dispatch_start",
        lambda kind, request, ws: {"success": False, "status_code": 400, "code": "test.dispatch_reached"},
    )
    resp = client.post("/api/v1/sessions", json={"kind": kind, "robot": "m", "options": _OPTIONS[kind]})
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNAVAILABLE
    assert "metal-leader" in resp.json()["detail"]


@pytest.mark.parametrize("kind", ["inference", "replay", "calibration"])
def test_follower_only_and_setup_kinds_do_not_need_the_extra(
    client, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, leader_missing, kind: str
) -> None:
    """Inference and replay never open the leader; calibration zeroes it over
    its bus and needs no gravity model. All three reach dispatch."""
    _make_ready_metal_record("m", leader_kind="metal")
    seen: list = []

    def fake_dispatch(kind, request, ws):
        seen.append((kind, request))
        return {"success": False, "status_code": 400, "code": "test.dispatch_reached"}

    monkeypatch.setattr(sessions, "_dispatch_start", fake_dispatch)
    resp = client.post("/api/v1/sessions", json={"kind": kind, "robot": "m", "options": _OPTIONS[kind]})
    assert resp.json()["code"] == "test.dispatch_reached", resp.text
    assert len(seen) == 1
    if kind == "calibration":
        request = seen[0][1]
        assert (request.arm_type, request.leader_kind, request.device_type) == ("metal", "metal", "teleop")


def test_teleop_and_record_requests_carry_the_records_leader_kind(
    tmp_lerobot_home: Path, leader_installed
) -> None:
    _make_ready_metal_record("m", leader_kind="metal")
    record = cfg.get_robot_record("m")
    from makermodslab.sessions import RecordingOptions, TeleoperationOptions

    teleop = sessions._build_teleoperation_request(record, TeleoperationOptions())
    assert (teleop.arm_type, teleop.leader_kind) == ("metal", "metal")
    rec = sessions._build_recording_request(record, RecordingOptions(dataset_repo_id="u/d", single_task="t"))
    assert (rec.arm_type, rec.leader_kind) == ("metal", "metal")


def test_a_hand_edited_unknown_leader_kind_is_refused_for_every_kind(
    client, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_ready_metal_record("m")
    path = Path(cfg.ROBOTS_PATH) / "m.json"
    data = json.loads(path.read_text())
    data["leader_kind"] = "nope"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(
        sessions, "_dispatch_start", lambda *a: {"success": False, "code": "test.dispatch_reached"}
    )
    resp = client.post(
        "/api/v1/sessions", json={"kind": "replay", "robot": "m", "options": _OPTIONS["replay"]}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNKNOWN


# ---------------------------------------------------------------------------
# Device classes and configs
# ---------------------------------------------------------------------------


class _Req(SimpleNamespace):
    def __init__(self, **kw):
        fields = {
            "arm_type": "metal",
            "mode": "single",
            "leader_port": "/dev/leader",
            "follower_port": "/dev/follower",
            "leader_config": "L",
            "follower_config": "F",
            "robot_name": "r",
            "right_leader_port": "/dev/rl",
            "right_follower_port": "/dev/rf",
            "right_leader_config": "RL",
            "right_follower_config": "RF",
            "leader_kind": None,
        }
        fields.update(kw)
        super().__init__(**fields)


@pytest.fixture
def _no_staging(monkeypatch: pytest.MonkeyPatch):
    seen: list = []

    def setup(leader, follower, arm_type="so101", **kw):
        seen.append(("single", arm_type, kw))
        return leader, follower

    def stage(*a, **kw):
        seen.append(("bimanual", a[5], kw))
        return ("/staging/leader", "/staging/follower", "r")

    monkeypatch.setattr("makermodslab.utils.robot_factory.setup_calibration_files", setup)
    monkeypatch.setattr("makermodslab.utils.robot_factory.stage_bimanual_calibrations", stage)
    return seen


def test_the_metal_kind_builds_the_metal_leader_with_hold_pinned_off(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    robot, teleop = build_single_configs(_Req(leader_kind="metal"))
    assert robot.type == "metal_follower"
    assert teleop.type == "metal_leader"
    assert (teleop.port, teleop.id) == ("/dev/leader", "L")
    # The fork's default (50) freezes the arm with torque ON at disconnect;
    # our stop path returns and releases it itself, so the hold is off.
    assert teleop.hold_kp_on_disconnect == 0.0
    # The staging helpers were told which leader library to read.
    assert _no_staging == [("single", "metal", {"leader_kind": "metal"})]


def test_the_star_kind_and_a_kindless_request_build_the_star_preset(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    for kind in (None, "", "star"):
        _, teleop = build_single_configs(_Req(leader_kind=kind))
        assert teleop.type == "rebot_102_leader_metal"
    # A multi-leader family always hears the kind (its default when unset);
    # the SO-101, a single-leader family, never does.
    assert {tuple(sorted(kw)) for _, _, kw in _no_staging} == {("leader_kind",)}
    _no_staging.clear()
    build_single_configs(_Req(arm_type="so101", leader_kind="so101"))
    assert _no_staging == [("single", "so101", {})]


def test_bimanual_metal_kind_builds_bi_metal_leader_with_metal_sub_configs(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, teleop = build_bimanual_configs(_Req(mode="bimanual", leader_kind="metal"))
    assert robot.type == "bi_metal_follower"
    assert teleop.type == "bi_metal_leader"
    assert (teleop.left_arm_config.port, teleop.right_arm_config.port) == ("/dev/leader", "/dev/rl")
    assert teleop.left_arm_config.hold_kp_on_disconnect == 0.0
    assert teleop.right_arm_config.hold_kp_on_disconnect == 0.0
    assert str(teleop.calibration_dir) == "/staging/leader"
    assert _no_staging == [("bimanual", "metal", {"leader_kind": "metal"})]


def test_single_leader_config_for_calibration_follows_the_kind() -> None:
    assert METAL.single_leader_config("/dev/x", "c").type == "rebot_102_leader_metal"
    assert METAL.single_leader_config("/dev/x", "c", leader_kind="star").type == "rebot_102_leader_metal"
    own = METAL.single_leader_config("/dev/x", "c", leader_kind="metal")
    assert (own.type, own.port, own.id, own.hold_kp_on_disconnect) == ("metal_leader", "/dev/x", "c", 0.0)


# ---------------------------------------------------------------------------
# Calibration: the Metal leader zeroes like a follower
# ---------------------------------------------------------------------------


class _FakeMetalLeader:
    """A lerobot MetalLeader as calibration sees it: a Damiao bus built in
    __init__, a config carrying motor_can_ids (and no joint_limits)."""

    name = "metal_leader"

    def __init__(self, config, log: list | None = None) -> None:
        from tests.mocks import FakeCanBus

        self.config = config
        self.log = log if log is not None else []
        self.bus = FakeCanBus(self.log)
        self.calibration = None

    def connect(self, calibrate: bool = True) -> None:
        self.log.append(("device", "connect", calibrate))

    def disconnect(self) -> None:
        self.log.append(("device", "disconnect"))


def _patch_teleop_factory(monkeypatch: pytest.MonkeyPatch, built: list):
    def fake_make_teleop(config):
        built.append(config)
        return _FakeMetalLeader(config)

    monkeypatch.setattr(
        "lerobot.teleoperators.make_teleoperator_from_config", fake_make_teleop, raising=False
    )
    monkeypatch.setattr(
        "lerobot.teleoperators.utils.make_teleoperator_from_config", fake_make_teleop, raising=False
    )


def test_the_metal_leader_opens_bus_only_and_torque_off(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list = []
    _patch_teleop_factory(monkeypatch, built)
    device = METAL.open_for_calibration("teleop", "/dev/can1", "cal", leader_kind="metal")
    assert [c.type for c in built] == ["metal_leader"]
    # Never connect(): that would start the gravity thread and enable torque.
    # The Damiao handshake energizes, so the disable right after is what frees it.
    assert device.log == [("bus", "connect", False), ("bus", "disable_torque")]


def test_a_failed_metal_leader_open_de_energizes_the_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list = []
    _patch_teleop_factory(monkeypatch, built)
    seen: list = []
    monkeypatch.setattr(torque, "de_energize_can_device", lambda device, label: seen.append(label) or [])

    def boom(*a, **k):
        raise ConnectionError("motor 3 silent")

    from tests import mocks

    monkeypatch.setattr(mocks.FakeCanBus, "disable_torque", boom)
    with pytest.raises(ConnectionError):
        METAL.open_for_calibration("teleop", "/dev/can1", "cal", leader_kind="metal")
    assert seen == ["Metal leader arm"]


def test_the_metal_leader_calibrates_with_the_followers_limits_and_a_bus_zero() -> None:
    from lerobot.robots.metal_follower import MetalFollowerConfig

    device = _FakeMetalLeader(METAL.single_leader_config("/dev/can1", "cal", leader_kind="metal"))

    class UI:
        steps: list = []

        def step(self, text, *, image_url=None, live_positions=False):
            self.steps.append(text)

        def message(self, text):
            pass

    ui = UI()
    calibration = METAL.calibrate(device, "teleop", ui)
    # The pose asked for is the FOLLOWER's (this leader is a Metal arm), not the Star leader's.
    assert ui.steps == [METAL.follower_zero_pose]
    assert ("bus", "set_zero_position") in device.log
    limits = MetalFollowerConfig(port="/x").joint_limits
    assert set(calibration) == set(limits)
    for joint, (low, high) in limits.items():
        assert (calibration[joint].range_min, calibration[joint].range_max) == (int(low), int(high))
        assert calibration[joint].homing_offset == 0
    # Damiao ids are (send, recv) tuples; the file stores the SEND id.
    assert calibration["shoulder_pan"].id == 0x01


def test_the_star_leader_still_gets_its_own_pose_and_origin_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mocks import FakeStarLeader

    device = FakeStarLeader(METAL.single_leader_config("/dev/star", "cal", leader_kind="star"))
    monkeypatch.setattr("time.sleep", lambda s: None)

    class UI:
        steps: list = []

        def step(self, text, **kw):
            self.steps.append(text)

        def message(self, text):
            pass

    ui = UI()
    METAL.calibrate(device, "teleop", ui)
    assert "Star Arm 102" in ui.steps[0]
    assert ("bus", "set_zero_position") not in device.log


def test_calibration_summary_per_leader_kind() -> None:
    assert METAL.calibration_summary("teleop", "star")["text"].startswith("Move the Star Arm 102")
    assert METAL.calibration_summary("teleop", "metal")["text"] == METAL.follower_zero_pose
    assert METAL.calibration_summary("teleop")["text"].startswith("Move the Star Arm 102")


def test_calibration_configs_routes_take_the_leader_kind(client, tmp_lerobot_home: Path) -> None:
    (Path(cfg.METAL_LEADER_CONFIG_PATH) / "own.json").write_text("{}")
    (Path(cfg.MAKER_LEADER_CONFIG_PATH) / "star.json").write_text("{}")
    names = lambda resp: sorted(c["name"] for c in resp.json()["configs"])  # noqa: E731
    assert names(client.get("/api/v1/calibration-configs/teleop?arm_type=metal&leader_kind=metal")) == ["own"]
    assert names(client.get("/api/v1/calibration-configs/teleop?arm_type=metal")) == ["star"]
    resp = client.get("/api/v1/calibration-configs/teleop?arm_type=metal&leader_kind=nope")
    assert resp.status_code == 400
    assert resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNKNOWN


def test_deleting_a_star_calibration_leaves_a_metal_leader_record_alone(
    client, tmp_lerobot_home: Path
) -> None:
    """Two leader libraries, one stem: the sweep that unassigns a deleted file
    from robot records must scope by the LEADER library, not the arm type."""
    _make_ready_metal_record("star_bot", leader_kind="star")
    _make_ready_metal_record("own_bot", leader_kind="metal")
    resp = client.delete("/api/v1/calibration-configs/teleop/LC?arm_type=metal")
    assert resp.status_code == 200, resp.text
    assert cfg.get_robot_record("star_bot")["leader_config"] == ""
    assert cfg.get_robot_record("own_bot")["leader_config"] == "LC"


def test_registry_refuses_two_leader_kinds_sharing_one_library(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mocks import make_arm_family, scratch_registry

    scratch_registry(monkeypatch)
    family = make_arm_family("twin")
    family.leader_options = lambda: (LeaderOption(id="a", label="A"), LeaderOption(id="b", label="B"))
    family.leader_calibration_dir = lambda leader_kind=None: "/one/library/for/both"
    with pytest.raises(ValueError, match="same leader calibration dir"):
        registry.register(family)


# ---------------------------------------------------------------------------
# Port detection: the probe, the gesture, and the wiggle fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_with_a_metal_leader_lists_arms_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def opener(port):
        if port == "/dev/none":
            raise RuntimeError("nothing here")
        return object(), 0.0

    monkeypatch.setattr(
        maker_ports, "_openers_for", lambda arm_type: {"robot": (opener, lambda bus: None, lambda bus: 0.0)}
    )
    result = await maker_ports.probe_maker_ports(["/dev/a", "/dev/b", "/dev/none"], "metal", "metal")
    assert result["success"] is False
    assert (result["follower_ports"], result["leader_ports"]) == ([], [])
    assert result["unknown_ports"] == ["/dev/a", "/dev/b", "/dev/none"]
    assert "cannot tell them apart" in result["message"] and "wiggle" in result["message"]


@pytest.mark.asyncio
async def test_probe_with_the_star_leader_is_the_protocol_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    monkeypatch.setattr(
        maker_ports, "_probe_sync", lambda ports, arm_type: seen.append(arm_type) or {"ok": 1}
    )
    monkeypatch.setattr(maker_ports, "_probe_same_protocol_sync", lambda *a: {"ok": 0})
    assert await maker_ports.probe_maker_ports(["/dev/a"], "metal", "star") == {"ok": 1}
    assert await maker_ports.probe_maker_ports(["/dev/a"], "metal") == {"ok": 1}


@pytest.mark.asyncio
async def test_the_gesture_is_refused_for_both_metal_sides_with_the_wiggle_fallback(monkeypatch) -> None:
    monkeypatch.setattr(maker_ports, "find_available_ports", lambda: [])
    follower = await maker_ports.identify_maker_arm_by_motion("robot", ["/dev/a"], "metal", "metal")
    leader = await maker_ports.identify_maker_arm_by_motion("teleop", ["/dev/a"], "metal", "metal")
    assert follower["success"] is False and follower["fallback"] == "wiggle"
    assert leader["success"] is False and leader["fallback"] == "wiggle"
    assert "leader" in leader["message"] and "energize" in leader["message"]
    # The Star leader's gesture is allowed and, when it finds nothing, has no fallback.
    star = await maker_ports.identify_maker_arm_by_motion("teleop", [], "metal", "star")
    assert star["success"] is False and "fallback" not in star


@pytest.mark.asyncio
async def test_a_failed_follower_gesture_on_metal_falls_back_to_the_wiggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) the identification flow: motion first; when it produces no port on
    a Metal device, the wiggle is what the answer names next."""
    monkeypatch.setattr(METAL, "motion_identify_energizes_follower", False)
    monkeypatch.setattr(
        maker_ports,
        "_identify_sync",
        lambda ports, device_type, arm_type, timeout_s=0: {
            "success": False,
            "message": "no motion",
            "skipped": [],
        },
    )
    result = await maker_ports.identify_maker_arm_by_motion("robot", ["/dev/a"], "metal", "star")
    assert result == {"success": False, "message": "no motion", "skipped": [], "fallback": "wiggle"}


def test_choose_identification_is_the_pure_rule() -> None:
    refused = {"success": False, "message": "refused"}
    found = {"success": True, "port": "/dev/a"}
    # Motion found a port: nothing else to do, on any family.
    assert can_wiggle.choose_identification(METAL, "robot", found) is None
    # Metal follower: refused or no match → wiggle.
    assert can_wiggle.choose_identification(METAL, "robot", refused) == "wiggle"
    assert can_wiggle.choose_identification(METAL, "robot", None) == "wiggle"
    # Metal leader: only the ENERGIZED (Metal) leader has a gripper to wiggle.
    assert can_wiggle.choose_identification(METAL, "teleop", refused, "metal") == "wiggle"
    assert can_wiggle.choose_identification(METAL, "teleop", refused, "star") is None
    assert can_wiggle.choose_identification(METAL, "teleop", refused, None) is None
    # A family without the wiggle never falls back to it.
    assert can_wiggle.choose_identification(MAKER, "robot", refused) == "wiggle"
    assert can_wiggle.choose_identification(SO101, "robot", None) is None


# ---------------------------------------------------------------------------
# The gripper wiggle itself
# ---------------------------------------------------------------------------

GRIPPER_LIMITS = (0.0, 137.5)


class _FakeDamiaoBus:
    """A DamiaoMotorsBus as the wiggle drives it: connect() enables every
    motor on the bus (the handshake), MIT writes are logged, disable is
    per motor or whole-bus."""

    def __init__(self, motors: dict, present: float = 30.0, fail_after_writes: int | None = None) -> None:
        self.motors = motors
        self.port = "/dev/can9"
        self.present = present
        self.fail_after_writes = fail_after_writes
        self.enabled: list[str] = []
        self.disabled: list[str | None] = []
        self.writes: list[dict] = []
        self.is_connected = False
        self.closed = 0

    def connect(self, handshake: bool = True) -> None:
        self.is_connected = True
        if handshake:
            self.enabled += list(self.motors)

    def read(self, register: str, motor: str) -> float:
        assert (register, motor) == ("Present_Position", "gripper")
        return self.present

    def enable_torque(self, motor: str) -> None:
        self.enabled.append(motor)

    def write(self, register: str, motor: str, value: float) -> None:
        assert motor == "gripper"
        if register == "Kp":
            self.kp = value
        elif register == "Kd":
            self.kd = value
        else:
            assert register == "Goal_Position"
            if self.fail_after_writes is not None and len(self.writes) >= self.fail_after_writes:
                raise RuntimeError("CAN write failed")
            self.writes.append({motor: (self.kp, self.kd, value, 0.0, 0.0)})
            self.present = value

    def disable_torque(self, motors=None, num_retry: int = 0) -> None:
        self.disabled.append(motors)

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        self.is_connected = False
        self.closed += 1


def test_plan_can_wiggle_preserves_rest_inside_the_limits() -> None:
    high, low, rest = can_wiggle.plan_can_wiggle(30.0, GRIPPER_LIMITS)
    assert (high, low, rest) == (40.0, 20.0, 30.0)
    high, low, rest = can_wiggle.plan_can_wiggle(137.5, GRIPPER_LIMITS)
    assert high == rest == 137.5
    assert low < rest
    assert can_wiggle.plan_can_wiggle(0.0, GRIPPER_LIMITS) == (10.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        can_wiggle.plan_can_wiggle(-5.0, GRIPPER_LIMITS)
    with pytest.raises(ValueError):
        can_wiggle.plan_can_wiggle(10.0, (0.0, 10.0))


def test_the_wiggle_bus_carries_only_the_gripper() -> None:
    """(b) enable ONLY the gripper: the Damiao handshake enables every motor
    on the bus, so the bus is built with exactly one."""
    bus = can_wiggle._open_gripper_bus("metal", "/dev/can9")
    assert list(bus.motors) == ["gripper"]
    assert bus.port == "/dev/can9"
    assert can_wiggle.gripper_limits("metal") == GRIPPER_LIMITS


def test_the_wiggle_drives_inside_the_limits_and_disables_the_gripper_after() -> None:
    bus = _FakeDamiaoBus({"gripper": object()}, present=30.0)
    sleeps: list[float] = []
    can_wiggle.drive_gripper_wiggle(bus, GRIPPER_LIMITS, sleep=sleeps.append)
    assert bus.enabled == ["gripper"]
    assert len(bus.writes) > 2 * can_wiggle.WIGGLE_REPEATS + 1
    for command in bus.writes:
        assert list(command) == ["gripper"], "no other joint is ever commanded"
        kp, kd, position, velocity, tau = command["gripper"]
        assert (kp, kd, velocity, tau) == (can_wiggle.WIGGLE_KP, can_wiggle.WIGGLE_KD, 0.0, 0.0)
        assert GRIPPER_LIMITS[0] <= position
        assert position <= GRIPPER_LIMITS[1]
    assert bus.writes[-1]["gripper"][2] == 30.0, "settles back where it started"
    # Disabled after (the de-energize helper's whole-bus broadcast), then closed without a second disable.
    assert bus.disabled == [None]
    assert bus.closed == 1 and bus.is_connected is False
    assert all(delay == 0.05 for delay in sleeps)


def test_a_drive_that_raises_partway_still_disables_the_gripper() -> None:
    bus = _FakeDamiaoBus({"gripper": object()}, present=30.0, fail_after_writes=2)
    with pytest.raises(RuntimeError, match="CAN write failed"):
        can_wiggle.drive_gripper_wiggle(bus, GRIPPER_LIMITS, sleep=lambda s: None)
    assert bus.enabled == ["gripper"]
    assert bus.disabled == [None], "the handshake energized it; the failure path frees it"
    assert bus.closed == 1


def test_an_enable_failure_is_recovered_without_re_energizing() -> None:
    class HalfBus(_FakeDamiaoBus):
        def enable_torque(self, motor: str) -> None:
            self.enabled.append(motor)
            self.is_connected = False
            raise ConnectionError("enable: no reply")

    bus = HalfBus({"gripper": object()})
    with pytest.raises(ConnectionError):
        can_wiggle.drive_gripper_wiggle(bus, GRIPPER_LIMITS, sleep=lambda s: None)
    assert bus.enabled == ["gripper"]
    assert bus.disabled == [None]


@pytest.mark.asyncio
async def test_the_wiggle_refuses_while_another_feature_holds_the_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) the busy matrix: the wiggle counts as a wiggle and defers to every session kind."""
    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = await can_wiggle.wiggle_can_gripper("metal", "robot", "/dev/can9")
    assert result["success"] is False and result["code"] == ErrorCode.ROBOT_BUSY_TELEOPERATION
    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", False)
    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = await can_wiggle.wiggle_can_gripper("metal", "robot", "/dev/can9")
    assert result["code"] == ErrorCode.ROBOT_BUSY_WIGGLE
    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", False)
    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    result = await can_wiggle.wiggle_can_gripper("metal", "teleop", "/dev/can9", "metal")
    assert result["code"] == ErrorCode.ROBOT_BUSY_INFERENCE


@pytest.mark.asyncio
async def test_the_wiggle_runs_the_drive_and_clears_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.wiggle as wiggle_module

    ran: list = []
    monkeypatch.setattr(can_wiggle, "_open_gripper_bus", lambda arm_type, port: ("bus", arm_type, port))
    monkeypatch.setattr(can_wiggle, "gripper_limits", lambda arm_type: GRIPPER_LIMITS)

    def fake_drive(bus, limits, sleep=None, gains=None):
        ran.append((bus, limits, wiggle_module.wiggle_active))

    monkeypatch.setattr(can_wiggle, "drive_gripper_wiggle", fake_drive)
    result = await can_wiggle.wiggle_can_gripper("metal", "robot", "/dev/can9")
    assert result["success"] is True and "/dev/can9" in result["message"]
    assert ran == [(("bus", "metal", "/dev/can9"), GRIPPER_LIMITS, True)], (
        "the flag was held during the drive"
    )
    assert wiggle_module.wiggle_active is False


@pytest.mark.asyncio
async def test_the_star_leader_has_nothing_to_wiggle() -> None:
    result = await can_wiggle.wiggle_can_gripper("metal", "teleop", "/dev/star", "star")
    assert result["success"] is False and "no motors" in result["message"]
    result = await MAKER.identify_by_gripper_wiggle("teleop", "/dev/star")
    assert result["success"] is False and "no motors" in result["message"]


@pytest.mark.asyncio
async def test_the_wiggle_route_is_typed_and_routes_to_the_family(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list = []

    async def fake(device_type, port, leader_kind=None):
        seen.append((device_type, port, leader_kind))
        return {"success": True, "message": "moved"}

    monkeypatch.setattr(METAL, "identify_by_gripper_wiggle", fake)
    resp = client.post(
        "/api/v1/maker/wiggle-gripper",
        json={"arm_type": "metal", "device_type": "teleop", "port": "/dev/can9", "leader_kind": "metal"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"success": True, "message": "moved"}
    assert seen == [("teleop", "/dev/can9", "metal")]
    resp = client.post(
        "/api/v1/maker/wiggle-gripper",
        json={"arm_type": "metal", "device_type": "teleop", "port": "/dev/can9", "leader_kind": "nope"},
    )
    assert resp.status_code == 400 and resp.json()["code"] == ErrorCode.ROBOT_LEADER_KIND_UNKNOWN


# ---------------------------------------------------------------------------
# The stop path: an energized leader is returned and released like a follower
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self) -> None:
        self.set_calls = 0

    def set(self) -> None:
        self.set_calls += 1


class _FakeThread:
    def __init__(self) -> None:
        self.joined: list[float] = []

    def is_alive(self) -> bool:
        return not self.joined

    def join(self, timeout: float | None = None) -> None:
        self.joined.append(timeout)


class _FakeLeaderBus:
    def __init__(self, pose: dict[str, float]) -> None:
        self.pose = pose
        self.writes: list[dict] = []
        self.disabled = 0
        self.port = "/dev/can1"

    def sync_read(self, register: str) -> dict[str, float]:
        return dict(self.pose)

    def sync_write_metal(self, commands: dict) -> None:
        self.writes.append(dict(commands))

    def disable_torque(self) -> None:
        self.disabled += 1


class _FakeLiveMetalLeader:
    name = "metal_leader"

    def __init__(self, pose: dict[str, float]) -> None:
        self.bus = _FakeLeaderBus(pose)
        self._gravity_stop_event = _FakeEvent()
        self._gravity_thread = _FakeThread()
        import threading

        self._bus_lock = threading.Lock()


POSE = {"shoulder_pan": 1.0, "shoulder_lift": -20.0, "gripper": 40.0}


def test_capture_leader_rest_poses_wraps_an_energized_leader_and_skips_the_star() -> None:
    from tests.mocks import FakeStarLeader

    star = FakeStarLeader(METAL.single_leader_config("/dev/star", "c"))
    assert METAL.capture_leader_rest_poses(star) == []
    leader = _FakeLiveMetalLeader(POSE)
    [(drive, pose)] = METAL.capture_leader_rest_poses(leader)
    assert isinstance(drive, maker_rest_pose.EnergizedLeaderDrive)
    assert pose == {"shoulder_pan": 1.0, "shoulder_lift": -20.0}, "the gripper stays in the operator's hand"
    # Bimanual: one pair per sub-arm.
    bi = SimpleNamespace(left_arm=_FakeLiveMetalLeader(POSE), right_arm=_FakeLiveMetalLeader(POSE))
    assert len(METAL.capture_leader_rest_poses(bi)) == 2


def test_the_drive_stops_gravity_before_the_first_setpoint_and_uses_the_hold_gains() -> None:
    leader = _FakeLiveMetalLeader(POSE)
    drive = maker_rest_pose.EnergizedLeaderDrive(leader)
    assert drive.get_observation() == {f"{m}.pos": v for m, v in POSE.items()}
    assert leader._gravity_stop_event.set_calls == 0
    drive.send_action({"shoulder_pan.pos": 0.5, "shoulder_lift.pos": -10.0})
    assert leader._gravity_stop_event.set_calls == 1
    assert leader._gravity_thread.joined == [maker_rest_pose._GRAVITY_STOP_TIMEOUT_S]
    assert leader.bus.writes == [
        {
            "shoulder_pan": (
                maker_rest_pose.LEADER_RETURN_KP,
                maker_rest_pose.LEADER_RETURN_KD,
                0.5,
                0.0,
                0.0,
            ),
            "shoulder_lift": (
                maker_rest_pose.LEADER_RETURN_KP,
                maker_rest_pose.LEADER_RETURN_KD,
                -10.0,
                0.0,
                0.0,
            ),
        }
    ]
    drive.send_action({"shoulder_pan.pos": 0.0})
    assert leader._gravity_stop_event.set_calls == 1, "stopped once"


def test_the_return_walks_the_leader_through_the_same_loop_as_a_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = _FakeLiveMetalLeader(dict(POSE))
    drive = maker_rest_pose.EnergizedLeaderDrive(leader)
    monkeypatch.setattr(maker_rest_pose.time, "sleep", lambda s: None)
    # The fake bus reports arrival at once: every setpoint the loop writes is the pose.
    leader.bus.pose = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "gripper": 40.0}
    arrived, reason = maker_rest_pose.return_maker_to_pose(drive, {"shoulder_pan": 0.0, "shoulder_lift": 0.0})
    assert (arrived, reason) == (True, "")
    assert leader._gravity_stop_event.set_calls == 0, "already there: nothing was driven"
    # Far from the pose and (a fake bus) never moving: the loop ramps the
    # setpoints, then gives up by convergence — every write went through the
    # drive, and gravity was stopped before the first one.
    leader.bus.pose = {"shoulder_pan": 40.0, "shoulder_lift": 30.0, "gripper": 40.0}
    METAL.return_to_rest([(drive, {"shoulder_pan": 1.0, "shoulder_lift": -20.0})])
    assert leader.bus.writes, "the setpoints went out through the drive"
    assert leader._gravity_stop_event.set_calls == 1
    assert all(set(w) == {"shoulder_pan", "shoulder_lift"} for w in leader.bus.writes)


def test_release_stops_gravity_then_disables_every_leader_bus() -> None:
    leader = _FakeLiveMetalLeader(POSE)
    assert torque.release_maker_torque(leader, "leader") == []
    assert leader._gravity_stop_event.set_calls == 1
    assert leader.bus.disabled == 1
    bi = SimpleNamespace(left_arm=_FakeLiveMetalLeader(POSE), right_arm=_FakeLiveMetalLeader(POSE))
    assert METAL.release_torque(bi, "leaders") == []
    assert (bi.left_arm.bus.disabled, bi.right_arm.bus.disabled) == (1, 1)
    # The Star leader: no motors, nothing released, nothing raised.
    from tests.mocks import FakeStarLeader

    assert METAL.release_torque(FakeStarLeader(METAL.single_leader_config("/dev/s", "c")), "star") == []


def test_stop_gravity_is_a_no_op_on_a_device_without_a_thread() -> None:
    assert maker_rest_pose.stop_gravity_compensation(object()) is False
    leader = _FakeLiveMetalLeader(POSE)
    leader._gravity_thread = None
    assert maker_rest_pose.stop_gravity_compensation(leader) is True


def test_step_calibration_releases_an_energized_leader_but_not_the_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.step_calibrate import StepCalibrationManager, StepCalibrationRequest

    released: list = []
    monkeypatch.setattr(METAL, "release_torque", lambda device, label="": released.append(label) or [])
    for kind, expected in (("metal", ["Metal leader"]), ("star", [])):
        released.clear()
        manager = StepCalibrationManager()
        manager._family = METAL
        manager._current_request = StepCalibrationRequest(
            device_type="teleop", port="/dev/x", config_file="c", arm_type="metal", leader_kind=kind
        )
        manager.status.device_type = "teleop"
        manager.device = _FakeMetalLeader(METAL.single_leader_config("/dev/x", "c", leader_kind=kind))
        manager._release_device()
        assert released == expected, kind
