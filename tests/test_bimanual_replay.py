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
"""Follower-only construction, target splitting and coordinated ease-in contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from makermodslab import replay


def request(arm_type="maker", **updates):
    return replay.ReplayRequest(
        repo_id="u/d",
        episode_index=0,
        mode="bimanual",
        follower_port="left",
        right_follower_port="right",
        follower_config="L",
        right_follower_config="R",
        arm_type=arm_type,
        robot_name="pair",
    ).model_copy(update=updates)


def pair():
    def arm():
        return SimpleNamespace(bus=SimpleNamespace(motors={"joint": None, "gripper": None}))

    return SimpleNamespace(left_arm=arm(), right_arm=arm())


@pytest.fixture(autouse=True)
def reset():
    replay._stop_event.clear()
    yield
    replay._stop_event.clear()


@pytest.mark.parametrize("arm_type", ["so101", "maker", "metal"])
def test_build_actual_pinned_bimanual_follower_without_leaders(arm_type, tmp_lerobot_home):
    from makermodslab.utils import config

    folder = Path(config.follower_config_path_for(arm_type))
    folder.mkdir(parents=True, exist_ok=True)
    cal = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}}
    for name in ("L", "R"):
        (folder / f"{name}.json").write_text(json.dumps(cal))
    robot = replay._build_bimanual_follower(request(arm_type))
    try:
        assert len(robot.action_features) == (12 if arm_type == "so101" else 14)
        assert all(key.startswith(("left_", "right_")) for key in robot.action_features)
        assert robot.left_arm.calibration == robot.right_arm.calibration
        assert not Path(config.MAKERMODSLAB_BISO_STAGING_PATH, "pair", "leader").exists()
        assert robot.left_arm.cameras == robot.right_arm.cameras == {}
        validated = replay._build_bimanual_follower(request(arm_type), list(robot.action_features))
        if hasattr(validated, "_io_pool"):
            validated._io_pool.shutdown()
    finally:
        if hasattr(robot, "_io_pool"):
            robot._io_pool.shutdown()


@pytest.mark.parametrize(
    "updates",
    [
        {"right_follower_port": ""},
        {"right_follower_port": "left"},
        {"right_follower_config": ""},
        {"right_follower_config": "../escape"},
    ],
)
def test_invalid_slots_do_not_stage_or_build(updates, monkeypatch, tmp_lerobot_home):
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.build_follower_config",
        lambda *a: pytest.fail("invalid slots staged"),
    )
    with pytest.raises(ValueError):
        replay._build_bimanual_follower(request(**updates))


def test_split_frame_preserves_sides_and_grippers():
    robot = pair()
    targets = replay._arm_targets(
        robot, {"left_joint.pos": 1, "left_gripper.pos": 2, "right_joint.pos": 3, "right_gripper.pos": 4}
    )
    assert [target for _, _, target in targets] == [{"joint": 1, "gripper": 2}, {"joint": 3, "gripper": 4}]


@pytest.mark.parametrize("feetech", [False, True])
def test_both_sides_must_finish_ease_in(feetech, monkeypatch):
    robot = pair()
    seen = []

    def ease(device, target, **kwargs):
        seen.append((device, target))
        return True, "arrived"

    monkeypatch.setattr(replay, "return_maker_to_pose", ease)
    monkeypatch.setattr(replay._rest_pose, "return_to_rest_pose", ease)
    assert replay._ease_bimanual(robot, {"left_joint.pos": 1, "right_joint.pos": 2}, feetech)[0]
    assert sorted(target["joint"] for _, target in seen) == [1, 2]


def test_one_failure_cancels_peer_and_cannot_start_playback(monkeypatch):
    robot = pair()

    def ease(device, target, **kwargs):
        if device is robot.left_arm:
            return False, "ceiling"
        assert kwargs["abort_event"].is_set()
        return False, "cut-short"

    monkeypatch.setattr(replay, "return_maker_to_pose", ease)
    ok, reason = replay._ease_bimanual(robot, {"left_joint.pos": 1, "right_joint.pos": 2}, False)
    assert not ok and "left follower arm: ceiling" in reason


@pytest.mark.parametrize("arm_type", ["maker", "metal"])
def test_partial_can_connect_deenergizes_entire_pair(arm_type, monkeypatch):
    robot = pair()
    robot.connect = lambda **kw: (_ for _ in ()).throw(RuntimeError("right failed"))
    cleaned = []
    monkeypatch.setattr(
        "makermodslab.torque.de_energize_can_device", lambda device, label: cleaned.append(device)
    )
    with pytest.raises(RuntimeError, match="right failed"):
        replay._connect_bimanual_follower(robot, request(arm_type))
    assert cleaned == [robot]


def test_session_builder_forwards_right_slots():
    from makermodslab.sessions import ReplayOptions, _build_replay_request

    result = _build_replay_request(
        {
            "name": "pair",
            "mode": "bimanual",
            "arm_type": "metal",
            "follower_port": "L",
            "follower_config": "LC",
            "right_follower_port": "R",
            "right_follower_config": "RC",
        },
        ReplayOptions(repo_id="u/d", episode_index=1),
    )
    assert (result.mode, result.right_follower_port, result.right_follower_config) == ("bimanual", "R", "RC")


@pytest.mark.parametrize("bad", ["single", "duplicates", "missing_side", "width", "nan"])
def test_bad_dataset_rejected_before_staging(bad):
    names = [f"{side}_joint{i}.pos" for side in ("left", "right") for i in range(7)]
    series = {"action_names": names, "values": [[0.0] * 14]}
    if bad == "single":
        series["action_names"] = names[:7]
    elif bad == "duplicates":
        names[-1] = names[0]
    elif bad == "missing_side":
        names[-1] = "other_joint.pos"
    elif bad == "width":
        series["values"] = [[0.0] * 13]
    else:
        series["values"][0][-1] = float("nan")
    with pytest.raises(ValueError):
        replay._validate_bimanual_actions(series, "maker")


def test_so_settled_residual_must_cover_every_target(monkeypatch):
    monkeypatch.setattr(replay._rest_pose, "return_to_rest_pose", lambda *a, **k: (False, "settled"))
    monkeypatch.setattr(replay, "_ease_in_residual", lambda *a: {"joint": 1})
    ok, _ = replay._ease_bimanual(
        pair(),
        {"left_joint.pos": 1, "left_gripper.pos": 2, "right_joint.pos": 3, "right_gripper.pos": 4},
        True,
    )
    assert not ok


@pytest.mark.parametrize("feetech", [True, False])
def test_first_close_failure_does_not_strand_other_follower(feetech, monkeypatch):
    robot = pair()
    closed = []

    def left(**kw):
        raise RuntimeError("close failed")

    def right(**kw):
        closed.append("right")

    for arm, fn in ((robot.left_arm, left), (robot.right_arm, right)):
        arm.disconnect = fn
        arm.bus.disconnect = fn
    robot._io_pool = SimpleNamespace(shutdown=lambda **kw: closed.append("pool"))
    monkeypatch.setattr(
        "makermodslab.teleoperate.force_disconnect_partial", lambda *a: closed.append("recovery")
    )
    monkeypatch.setattr("makermodslab.torque.de_energize_can_device", lambda *a: closed.append("recovery"))
    replay._disconnect_replay_followers(robot, feetech)
    assert closed == ["pool", "recovery", "right"]


@pytest.mark.parametrize("arrived", [True, False])
def test_worker_sends_one_full_frame_only_after_both_ease_in_and_returns_both(arrived, monkeypatch):
    robot = pair()
    events = []
    poses = [(robot.left_arm, {"joint": 0}), (robot.right_arm, {"joint": 0})]
    family = SimpleNamespace(
        uses_feetech_bus=False,
        capture_rest_poses=lambda *a, **kw: poses,
        return_to_rest=lambda targets, **kw: events.append(("rest", targets)),
        release_torque=lambda *a: events.append(("release",)),
    )
    monkeypatch.setattr(replay.arm_registry, "get", lambda *a: family)
    monkeypatch.setattr(replay, "replay_active", True)
    monkeypatch.setattr(replay, "_replay_meta", {})
    monkeypatch.setattr(
        replay, "_ease_bimanual", lambda *a: (arrived, "arrived" if arrived else "right: ceiling")
    )
    monkeypatch.setattr(replay, "_disconnect_replay_followers", lambda *a: None)
    monkeypatch.setattr(replay, "notify_session_changed", lambda *a, **kw: None)
    robot.send_action = lambda action: events.append(("send", action))
    replay._replay_worker(
        robot,
        {"action_names": ["left_joint.pos", "right_joint.pos"], "values": [[1, 2]], "timestamps": [0.0]},
        None,
        "maker",
    )
    assert [event for event in events if event[0] == "send"] == (
        [("send", {"left_joint.pos": 1, "right_joint.pos": 2})] if arrived else []
    )
    assert events[-2:] == [("rest", poses), ("release",)]


def test_failed_metal_handshake_reopens_unconnected_side_without_energizing():
    robot = pair()
    events = []
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        arm.bus.is_connected = side == "left"
        arm.bus.connect = lambda *, handshake, side=side: events.append((side, "open", handshake))
        arm.bus.disable_torque = lambda side=side: events.append((side, "disable"))
        arm.bus.disconnect = lambda *, disable_torque, side=side: events.append(
            (side, "close", disable_torque)
        )
    robot.connect = lambda **kw: (_ for _ in ()).throw(RuntimeError("partial right handshake"))
    with pytest.raises(RuntimeError):
        replay._connect_bimanual_follower(robot, request("metal"))
    assert events == [
        ("left", "disable"),
        ("left", "close", False),
        ("right", "open", False),
        ("right", "disable"),
        ("right", "close", False),
    ]


@pytest.mark.parametrize("fails", [False, True])
def test_can_reached_pose_hold_before_waiting_for_peer(fails, monkeypatch):
    held = []
    monkeypatch.setattr(replay, "return_maker_to_pose", lambda *a, **kw: (True, "arrived"))

    def hold(targets):
        held.extend(targets)
        if fails:
            raise RuntimeError("hold failed")

    ok, reason = replay._ease_bimanual(
        pair(), {"left_joint.pos": 1, "right_joint.pos": 2}, False, SimpleNamespace(hold_recording_home=hold)
    )
    assert ok is not fails
    assert len(held) == 2
    if fails:
        assert "hold failed" in reason


def test_exact_names_checked_before_staging(monkeypatch, tmp_lerobot_home):
    from makermodslab.utils import config

    folder = Path(config.follower_config_path_for("maker"))
    folder.mkdir(parents=True, exist_ok=True)
    cal = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}}
    for name in ("L", "R"):
        (folder / f"{name}.json").write_text(json.dumps(cal))
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.build_follower_config",
        lambda *a: pytest.fail("bad names must not stage files"),
    )
    with pytest.raises(ValueError, match="joints"):
        replay._build_bimanual_follower(
            request(), [f"{side}_bogus{i}.pos" for side in ("left", "right") for i in range(7)]
        )
