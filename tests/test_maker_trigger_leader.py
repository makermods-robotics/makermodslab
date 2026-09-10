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
"""The Maker arm's second leader kind: the Star Arm 102 with the trigger gripper.

One physical leader, two gripper mechanisms, one number apart (lerobot's
rebot_102_leader_maker_trigger preset). The kind is never energized, so the
port-detection and stop paths are the Star leader's; what it changes is the
preset every leader config is built from, the calibration library it reads,
and the zero-pose wording.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from makermodslab.arms.base import LeaderOption, leader_kwargs
from makermodslab.arms.maker import MAKER, STAR_TRIGGER_LEADER_KIND
from makermodslab.utils import config as cfg

TRIGGER_GRIPPER_DIRECTION = 0.6424


class _Req(SimpleNamespace):
    def __init__(self, **kw):
        fields = {
            "arm_type": "maker",
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


def _make_ready_maker_record(name: str, leader_kind: str) -> None:
    cfg.save_robot_record(
        name,
        {
            "arm_type": "maker",
            "leader_kind": leader_kind,
            "leader_port": "/dev/l",
            "follower_port": "/dev/f",
            "leader_config": "LC",
            "follower_config": "FC",
        },
        allow_create=True,
    )
    Path(cfg.leader_config_path_for("maker", leader_kind), "LC.json").write_text("{}")
    Path(cfg.follower_config_path_for("maker"), "FC.json").write_text("{}")


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_trigger_kind_is_a_second_unenergized_star_leader() -> None:
    assert MAKER.leader_options() == (
        LeaderOption(id="star", label="Star Arm 102 leader"),
        LeaderOption(id=STAR_TRIGGER_LEADER_KIND, label="Star Arm 102 leader (trigger gripper)"),
    )
    assert MAKER.leader_option(STAR_TRIGGER_LEADER_KIND).available is True
    assert MAKER.leader_holds_torque(STAR_TRIGGER_LEADER_KIND) is False
    assert MAKER.leader_holds_torque(None) is False
    # A multi-leader family always hears the kind, the default when unset.
    assert leader_kwargs(MAKER, None) == {"leader_kind": "star"}
    assert leader_kwargs(MAKER, STAR_TRIGGER_LEADER_KIND) == {"leader_kind": STAR_TRIGGER_LEADER_KIND}


def test_the_trigger_keeps_a_library_of_its_own(tmp_lerobot_home: Path) -> None:
    lever = cfg.leader_config_path_for("maker", "star")
    trigger = cfg.leader_config_path_for("maker", STAR_TRIGGER_LEADER_KIND)
    assert lever == cfg.MAKER_LEADER_CONFIG_PATH
    assert trigger == cfg.MAKER_TRIGGER_LEADER_CONFIG_PATH
    assert lever != trigger
    assert Path(trigger).name == "rebot_102_leader_trigger"
    # The follower library is the kind's business not at all.
    assert cfg.follower_config_path_for("maker") == cfg.MAKER_FOLLOWER_CONFIG_PATH


def test_zero_pose_wording_names_the_trigger_stop() -> None:
    lever = MAKER.zero_pose_instructions("teleop", "star")
    trigger = MAKER.zero_pose_instructions("teleop", STAR_TRIGGER_LEADER_KIND)
    assert lever == MAKER.zero_pose_instructions("teleop")
    assert "gripper fully closed" in lever
    assert "trigger" in trigger and "CLOSED stop" in trigger
    # The follower's pose does not depend on which leader drives it.
    assert MAKER.zero_pose_instructions("robot", STAR_TRIGGER_LEADER_KIND) == MAKER.follower_zero_pose
    assert MAKER.calibration_summary("teleop", STAR_TRIGGER_LEADER_KIND) == {
        "text": trigger,
        "image_url": None,
    }


# ---------------------------------------------------------------------------
# The configs: the trigger preset, and a library the config names outright
# ---------------------------------------------------------------------------


def test_the_trigger_kind_builds_the_trigger_preset(_no_staging, tmp_lerobot_home: Path) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    robot, teleop = build_single_configs(_Req(leader_kind=STAR_TRIGGER_LEADER_KIND))
    assert robot.type == "maker_follower"
    assert teleop.type == "rebot_102_leader_maker_trigger"
    assert (teleop.port, teleop.id) == ("/dev/leader", "L")
    assert teleop.joint_directions["gripper"] == pytest.approx(TRIGGER_GRIPPER_DIRECTION)
    # lerobot would derive the LEVER's directory from the shared class name;
    # the config names the trigger library so reads and writes both land there.
    assert str(teleop.calibration_dir) == cfg.MAKER_TRIGGER_LEADER_CONFIG_PATH
    assert _no_staging == [("single", "maker", {"leader_kind": STAR_TRIGGER_LEADER_KIND})]


def test_the_star_kind_and_a_kindless_request_build_the_lever_preset(
    _no_staging, tmp_lerobot_home: Path
) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    lever_direction = None
    for kind in (None, "", "star"):
        _, teleop = build_single_configs(_Req(leader_kind=kind))
        assert teleop.type == "rebot_102_leader_maker"
        assert str(teleop.calibration_dir) == cfg.MAKER_LEADER_CONFIG_PATH
        lever_direction = teleop.joint_directions["gripper"]
    # The lever pulls the servo one way, the trigger the other.
    assert lever_direction < 0 < TRIGGER_GRIPPER_DIRECTION
    assert {tuple(sorted(kw)) for _, _, kw in _no_staging} == {("leader_kind",)}


def test_bimanual_trigger_kind_builds_trigger_sub_configs(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, teleop = build_bimanual_configs(_Req(mode="bimanual", leader_kind=STAR_TRIGGER_LEADER_KIND))
    assert robot.type == "bi_maker_follower"
    assert teleop.type == "bi_rebot_102_leader_maker_trigger"
    assert (teleop.left_arm_config.port, teleop.right_arm_config.port) == ("/dev/leader", "/dev/rl")
    for arm in (teleop.left_arm_config, teleop.right_arm_config):
        assert arm.joint_directions["gripper"] == pytest.approx(TRIGGER_GRIPPER_DIRECTION)
    assert str(teleop.calibration_dir) == "/staging/leader"
    assert _no_staging == [("bimanual", "maker", {"leader_kind": STAR_TRIGGER_LEADER_KIND})]


def test_single_leader_config_for_calibration_follows_the_kind(tmp_lerobot_home: Path) -> None:
    lever = MAKER.single_leader_config("/dev/x", "c")
    assert (lever.type, str(lever.calibration_dir)) == (
        "rebot_102_leader_maker",
        cfg.MAKER_LEADER_CONFIG_PATH,
    )
    trigger = MAKER.single_leader_config("/dev/x", "c", leader_kind=STAR_TRIGGER_LEADER_KIND)
    assert (trigger.type, trigger.port, trigger.id) == ("rebot_102_leader_maker_trigger", "/dev/x", "c")
    assert str(trigger.calibration_dir) == cfg.MAKER_TRIGGER_LEADER_CONFIG_PATH
    # Everything but the gripper factor is the lever preset's.
    differing = {
        j for j in lever.joint_directions if lever.joint_directions[j] != trigger.joint_directions[j]
    }
    assert differing == {"gripper"}
    assert trigger.joint_ranges == lever.joint_ranges


# ---------------------------------------------------------------------------
# Records, readiness, the manifest
# ---------------------------------------------------------------------------


def test_a_trigger_record_is_ready_only_with_a_file_in_the_trigger_library(tmp_lerobot_home: Path) -> None:
    _make_ready_maker_record("m", leader_kind="star")
    assert cfg.is_robot_record_clean(cfg.get_robot_record("m"))
    # Switching kinds blanks the leader slots (a re-zero is needed for the
    # other mechanism anyway) and the record is no longer ready.
    cfg.save_robot_record("m", {"leader_kind": STAR_TRIGGER_LEADER_KIND}, allow_create=False)
    record = cfg.get_robot_record("m")
    assert record["leader_kind"] == STAR_TRIGGER_LEADER_KIND
    assert (record["leader_port"], record["leader_config"]) == ("", "")
    assert (record["follower_port"], record["follower_config"]) == ("/dev/f", "FC")
    assert not cfg.is_robot_record_clean(record)
    # Assigning the lever's calibration NAME is not enough: the file must be
    # in the trigger library.
    cfg.save_robot_record("m", {"leader_port": "/dev/l", "leader_config": "LC"}, allow_create=False)
    assert not cfg.is_robot_record_clean(cfg.get_robot_record("m"))
    Path(cfg.MAKER_TRIGGER_LEADER_CONFIG_PATH, "LC.json").write_text("{}")
    assert cfg.is_robot_record_clean(cfg.get_robot_record("m"))


def test_manifest_lists_the_trigger_kind_with_its_own_summary(client) -> None:
    arms = {a["id"]: a for a in client.get("/api/v1/arms").json()["arms"]}
    maker = arms["maker"]
    assert maker["default_leader_kind"] == "star"
    options = {o["id"]: o for o in maker["leader_options"]}
    assert list(options) == ["star", STAR_TRIGGER_LEADER_KIND]
    trigger = options[STAR_TRIGGER_LEADER_KIND]
    assert (trigger["available"], trigger["energized"], trigger["unavailable_reason"]) == (True, False, None)
    assert trigger["label"] == "Star Arm 102 leader (trigger gripper)"
    assert "trigger" in trigger["calibration_summary"]["text"]
    assert "trigger" not in options["star"]["calibration_summary"]["text"]


def test_upsert_accepts_the_trigger_kind_and_refuses_an_unknown_one(client, tmp_lerobot_home: Path) -> None:
    resp = client.post(
        "/api/v1/robots/m?create=true", json={"arm_type": "maker", "leader_kind": STAR_TRIGGER_LEADER_KIND}
    )
    assert resp.status_code == 200, resp.text
    assert cfg.get_robot_record("m")["leader_kind"] == STAR_TRIGGER_LEADER_KIND
    resp = client.post("/api/v1/robots/m", json={"leader_kind": "star_lever"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "robot.leader_kind.unknown"
