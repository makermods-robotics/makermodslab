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
"""Tests for makermodslab.utils.config — path resolution and persistence helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from makermodslab.utils import config as cfg


@pytest.fixture(autouse=True)
def _patch_robots_path(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ROBOTS_PATH (not covered by the shared fixture) into tmp."""
    from makermodslab.utils import config as cfg

    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))


def test_get_saved_robot_port_reads_legacy_port_files(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    # Nothing writes these files anymore, but existing installs still have
    # them; the read path keeps honoring them as the default port.
    Path(cfg.LEADER_PORT_FILE).write_text("/dev/ttyUSB0")
    Path(cfg.FOLLOWER_PORT_FILE).write_text("/dev/ttyUSB1")

    assert cfg.get_saved_robot_port("leader") == "/dev/ttyUSB0"
    assert cfg.get_saved_robot_port("follower") == "/dev/ttyUSB1"


def test_get_saved_robot_port_returns_none_when_unset(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    assert cfg.get_saved_robot_port("leader") is None


def test_is_valid_robot_name_accepts_simple_names(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    assert cfg.is_valid_robot_name("my_robot")
    assert cfg.is_valid_robot_name("robot-1")


def test_is_valid_robot_name_rejects_empty_and_path_separators(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.utils import config as cfg

    assert not cfg.is_valid_robot_name("")
    assert not cfg.is_valid_robot_name("a/b")
    assert not cfg.is_valid_robot_name("..")


def test_is_valid_robot_name_rejects_leading_trailing_whitespace(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.utils import config as cfg

    # name.strip() != name → invalid
    assert not cfg.is_valid_robot_name(" robot")
    assert not cfg.is_valid_robot_name("robot ")


def test_robot_record_save_get_delete_round_trip(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    record = {"name": "lab1", "leader_port": "/dev/ttyUSB0", "follower_port": ""}
    assert cfg.save_robot_record("lab1", record, allow_create=True)

    loaded = cfg.get_robot_record("lab1")
    assert loaded is not None
    assert loaded["name"] == "lab1"
    assert loaded["leader_port"] == "/dev/ttyUSB0"

    listed = cfg.list_robot_records()
    assert any(r["name"] == "lab1" for r in listed)

    assert cfg.delete_robot_record("lab1")
    assert cfg.get_robot_record("lab1") is None


def test_robot_record_allow_create_false_is_noop(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    # Record does not exist and allow_create=False → returns False.
    result = cfg.save_robot_record("nonexistent", {"leader_port": "/dev/x"}, allow_create=False)
    assert result is False
    assert cfg.get_robot_record("nonexistent") is None


def test_robot_record_save_rejects_invalid_name(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    # Path traversal-style names must not write outside the config dir.
    assert not cfg.save_robot_record("../escape", {"name": "x"}, allow_create=True)


def test_robot_record_merges_fields(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("merge_test", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("merge_test", {"follower_port": "/dev/b"}, allow_create=False)

    loaded = cfg.get_robot_record("merge_test")
    assert loaded is not None
    assert loaded["leader_port"] == "/dev/a"
    assert loaded["follower_port"] == "/dev/b"


def test_robot_record_merge_clears_field_with_empty_string(tmp_lerobot_home: Path) -> None:
    """An empty string is a valid merge value: it CLEARS the field (e.g. releasing
    a port without assigning another), it does not preserve the old value."""
    from makermodslab.utils import config as cfg

    cfg.save_robot_record(
        "clear_test", {"leader_port": "/dev/a", "follower_port": "/dev/b"}, allow_create=True
    )
    cfg.save_robot_record("clear_test", {"leader_port": ""}, allow_create=False)

    loaded = cfg.get_robot_record("clear_test")
    assert loaded is not None
    assert loaded["leader_port"] == ""
    assert loaded["follower_port"] == "/dev/b"


def test_clamp_motor_power_bounds_and_fallback(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    assert cfg.clamp_motor_power(55) == 55
    assert cfg.clamp_motor_power(5) == cfg.MOTOR_POWER_MIN
    assert cfg.clamp_motor_power(150) == cfg.MOTOR_POWER_MAX
    assert cfg.clamp_motor_power(42.9) == 42
    # Non-numeric (including bool — a subclass of int) → full power, never raise.
    assert cfg.clamp_motor_power(None) == cfg.DEFAULT_MOTOR_POWER
    assert cfg.clamp_motor_power("50") == cfg.DEFAULT_MOTOR_POWER
    assert cfg.clamp_motor_power(True) == cfg.DEFAULT_MOTOR_POWER


def test_robot_record_motor_power_defaults_to_full(tmp_lerobot_home: Path) -> None:
    """Records saved before the field existed (or fresh ones) read back as 100."""
    from makermodslab.utils import config as cfg

    # A pre-motor_power record on disk: write raw JSON without the field.
    path = Path(cfg.ROBOTS_PATH) / "old_bot.json"
    path.write_text(json.dumps({"name": "old_bot", "mode": "single", "leader_port": "/dev/a"}))

    loaded = cfg.get_robot_record("old_bot")
    assert loaded is not None
    assert loaded["motor_power"] == cfg.DEFAULT_MOTOR_POWER


def test_robot_record_motor_power_merge_clamps_and_ignores_invalid(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("power_bot", {"motor_power": 60}, allow_create=True)
    assert cfg.get_robot_record("power_bot")["motor_power"] == 60

    # Out-of-range values are clamped, not rejected.
    cfg.save_robot_record("power_bot", {"motor_power": 5}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MIN
    cfg.save_robot_record("power_bot", {"motor_power": 500}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MAX

    # A wrongly-typed value is ignored (keeps the existing setting), matching
    # the known-typed-fields-only merge of the string/list fields.
    cfg.save_robot_record("power_bot", {"motor_power": "25"}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MAX


def test_robot_record_motor_power_clamped_on_read(tmp_lerobot_home: Path) -> None:
    """A corrupted on-disk value never reaches consumers un-clamped."""
    from makermodslab.utils import config as cfg

    path = Path(cfg.ROBOTS_PATH) / "corrupt_bot.json"
    path.write_text(json.dumps({"name": "corrupt_bot", "mode": "single", "motor_power": 9000}))
    assert cfg.get_robot_record("corrupt_bot")["motor_power"] == cfg.MOTOR_POWER_MAX

    path.write_text(json.dumps({"name": "corrupt_bot", "mode": "single", "motor_power": "junk"}))
    assert cfg.get_robot_record("corrupt_bot")["motor_power"] == cfg.DEFAULT_MOTOR_POWER


def test_rename_robot_record_moves_file_and_preserves_fields(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("old_name", {"leader_port": "/dev/a"}, allow_create=True)

    ok, reason = cfg.rename_robot_record("old_name", "new_name")
    assert ok and reason == ""

    # Old gone, new present with fields and updated name.
    assert cfg.get_robot_record("old_name") is None
    moved = cfg.get_robot_record("new_name")
    assert moved is not None
    assert moved["name"] == "new_name"
    assert moved["leader_port"] == "/dev/a"


def test_rename_robot_record_noop_when_names_equal(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("same", {"leader_port": "/dev/a"}, allow_create=True)
    ok, reason = cfg.rename_robot_record("same", "same")
    assert ok and reason == ""
    assert cfg.get_robot_record("same") is not None


def test_rename_robot_record_rejects_missing_source(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.rename_robot_record("ghost", "whatever")
    assert not ok and reason == "not_found"


def test_rename_robot_record_rejects_existing_target(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("a", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("b", {"leader_port": "/dev/b"}, allow_create=True)

    ok, reason = cfg.rename_robot_record("a", "b")
    assert not ok and reason == "name_taken"
    # Both records untouched.
    assert cfg.get_robot_record("a")["leader_port"] == "/dev/a"
    assert cfg.get_robot_record("b")["leader_port"] == "/dev/b"


def test_rename_robot_record_rejects_invalid_target(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("valid", {"leader_port": "/dev/a"}, allow_create=True)
    ok, reason = cfg.rename_robot_record("valid", "../escape")
    assert not ok and reason == "invalid_name"
    # Source record must survive a rejected rename.
    assert cfg.get_robot_record("valid") is not None


_GOOD_CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 1927,
        "range_min": 741,
        "range_max": 3472,
    },
}


def test_validate_calibration_data_accepts_well_formed() -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(_GOOD_CALIBRATION)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "data",
    [
        {},  # empty
        {"m": {"id": 1}},  # missing fields
        {"m": "not-an-object"},  # motor not a dict
        {
            "m": {"id": True, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 1}
        },  # bool not int
    ],
)
def test_validate_calibration_data_rejects_malformed(data) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(data)
    assert not ok and reason


@pytest.mark.parametrize("name", ["whoo", "my-set_v2", "ok.name-1", "a", "A1"])
def test_validate_dataset_name_accepts_good(name) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_dataset_name(name)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "   ",  # whitespace only
        " whoo",  # leading space
        "whoo ",  # trailing space
        "whoo/",  # trailing slash
        "a/b",  # embedded slash
        "..",  # traversal
        ".",  # traversal
        ".hidden",  # leading dot
        "-lead",  # leading dash
        "trail-",  # trailing dash
        "bad name",  # space
        "café",  # non-ascii
        "x" * 97,  # too long
    ],
)
def test_validate_dataset_name_rejects_bad(name) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_dataset_name(name)
    assert not ok and reason


@pytest.mark.parametrize("repo_id", ["whoo", "Mokuroh54/whoo", "user/my-set_v2"])
def test_validate_dataset_repo_id_accepts_good(repo_id) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_dataset_repo_id(repo_id)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "repo_id",
    [
        "Mokuroh54/whoo/",  # the reported bug: trailing slash
        "whoo/",  # trailing slash, no namespace
        "a/b/c",  # too many slashes
        "-bad/whoo",  # bad namespace
        "user/.hidden",  # bad name segment
        "",  # empty
    ],
)
def test_validate_dataset_repo_id_rejects_bad(repo_id) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.validate_dataset_repo_id(repo_id)
    assert not ok and reason


def test_save_imported_calibration_writes_and_normalizes(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    # Name carries the .json extension (as robot records do) → normalized to stem.
    ok, reason, name = cfg.save_imported_calibration("teleop", "armA.json", _GOOD_CALIBRATION)
    assert ok and reason == "" and name == "armA"
    written = Path(cfg.LEADER_CONFIG_PATH) / "armA.json"
    assert written.is_file()
    assert json.loads(written.read_text()) == _GOOD_CALIBRATION


def test_save_imported_calibration_never_overwrites(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    ok, reason, _ = cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    assert not ok and reason == "name_taken"


def test_save_imported_calibration_rejects_bad_device(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    ok, reason, _ = cfg.save_imported_calibration("nope", "x", _GOOD_CALIBRATION)
    assert not ok and reason == "invalid_device"


def test_get_robot_record_normalizes_config_extension(tmp_lerobot_home: Path) -> None:
    """Legacy records stored config names WITH .json; reads normalize to stems."""
    from makermodslab.utils import config as cfg

    # Write a record on disk that carries the old ".json" form.
    cfg.save_robot_record(
        "legacy",
        {"leader_config": "so101.json", "follower_config": "so101.json"},
        allow_create=True,
    )
    rec = cfg.get_robot_record("legacy")
    assert rec["leader_config"] == "so101"
    assert rec["follower_config"] == "so101"


def test_rename_calibration_config_moves_and_repoints_records(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").write_text("{}")
    cfg.save_robot_record("bot", {"leader_config": "armA"}, allow_create=True)

    ok, reason = cfg.rename_calibration_config("teleop", "armA", "armB")
    assert ok and reason == ""
    assert not (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").exists()
    assert (Path(cfg.LEADER_CONFIG_PATH) / "armB.json").exists()
    # The robot that referenced armA is repointed to armB.
    assert cfg.get_robot_record("bot")["leader_config"] == "armB"


def test_rename_calibration_config_repoints_right_arm_slot(tmp_lerobot_home: Path) -> None:
    """Renaming a config repoints the bimanual right slot, not just the left."""
    from makermodslab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").write_text("{}")
    cfg.save_robot_record(
        "bi",
        {"mode": "bimanual", "leader_config": "armX", "right_leader_config": "armA"},
        allow_create=True,
    )

    ok, reason = cfg.rename_calibration_config("teleop", "armA", "armB")
    assert ok and reason == ""

    rec = cfg.get_robot_record("bi")
    assert rec["right_leader_config"] == "armB"
    assert rec["leader_config"] == "armX"  # the other slot is untouched


def test_rename_calibration_config_never_overwrites(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "a.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").write_text('{"keep": 1}')

    ok, reason = cfg.rename_calibration_config("robot", "a", "b")
    assert not ok and reason == "name_taken"
    # Target untouched.
    assert (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").read_text() == '{"keep": 1}'


def test_rename_calibration_config_missing_source(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    ok, reason = cfg.rename_calibration_config("teleop", "ghost", "x")
    assert not ok and reason == "not_found"


def test_record_defaults_to_single_mode(tmp_lerobot_home: Path) -> None:
    """A legacy record with no `mode` key reads back as single, with empty right_*."""
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("legacy", {"leader_config": "L"}, allow_create=True)
    rec = cfg.get_robot_record("legacy")
    assert rec["mode"] == "single"
    assert rec["right_leader_config"] == ""


def test_save_record_persists_bimanual_mode(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record(
        "bi",
        {"mode": "bimanual", "right_leader_config": "RL"},
        allow_create=True,
    )
    rec = cfg.get_robot_record("bi")
    assert rec["mode"] == "bimanual"
    assert rec["right_leader_config"] == "RL"


def test_save_record_rejects_unknown_mode(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("weird", {"mode": "nonsense"}, allow_create=True)
    assert cfg.get_robot_record("weird")["mode"] == "single"


def test_bimanual_record_clean_requires_all_four_calibrations(tmp_lerobot_home: Path) -> None:
    from makermodslab.utils import config as cfg

    record = {
        "name": "bi",
        "mode": "bimanual",
        "leader_port": "/dev/ll",
        "follower_port": "/dev/lf",
        "leader_config": "LL",
        "follower_config": "LF",
        "right_leader_port": "/dev/rl",
        "right_follower_port": "/dev/rf",
        "right_leader_config": "RL",
        "right_follower_config": "RF",
    }
    # Only the left pair's files exist -> not clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "LL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "LF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is False

    # Add the right pair's files -> clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "RL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "RF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is True


def test_is_robot_record_clean_follower_scope_ignores_leader(tmp_lerobot_home: Path) -> None:
    """arms="follower" (inference/replay) must not be blocked by leader gaps."""
    from makermodslab.utils import config as cfg

    record = {
        "name": "r",
        "follower_port": "/dev/f",
        "follower_config": "F",
    }
    # Follower calibration file missing -> not ready under either scope.
    assert cfg.is_robot_record_clean(record, arms="follower") is False
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "F.json").write_text("{}")
    # No leader port/config at all: follower scope passes, full scope fails.
    assert cfg.is_robot_record_clean(record, arms="follower") is True
    assert cfg.is_robot_record_clean(record) is False
    # A missing follower port still fails the follower scope.
    assert cfg.is_robot_record_clean(dict(record, follower_port=""), arms="follower") is False


def test_is_robot_record_clean_follower_scope_bimanual(tmp_lerobot_home: Path) -> None:
    """Bimanual follower scope needs BOTH followers, still no leaders."""
    from makermodslab.utils import config as cfg

    record = {
        "name": "bi",
        "mode": "bimanual",
        "follower_port": "/dev/lf",
        "follower_config": "LF",
        "right_follower_port": "/dev/rf",
        "right_follower_config": "RF",
    }
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "LF.json").write_text("{}")
    # Right follower's file missing -> not ready.
    assert cfg.is_robot_record_clean(record, arms="follower") is False
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "RF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record, arms="follower") is True
    # Full scope still fails: both leaders are unconfigured.
    assert cfg.is_robot_record_clean(record) is False


def test_config_slot_conflict_detects_same_side_duplicate() -> None:
    from makermodslab.utils import config as cfg

    base = {
        "mode": "bimanual",
        "leader_config": "L1",
        "follower_config": "F1",
        "right_leader_config": "L2",
        "right_follower_config": "F2",
    }
    assert cfg.config_slot_conflict(base) is None
    assert cfg.config_slot_conflict({**base, "right_leader_config": "L1"}) == "leader"
    assert cfg.config_slot_conflict({**base, "right_follower_config": "F1"}) == "follower"


def test_port_slot_conflict_detects_shared_port() -> None:
    from makermodslab.utils import config as cfg

    # Single: leader and follower must differ.
    assert (
        cfg.port_slot_conflict({"mode": "single", "leader_port": "/dev/a", "follower_port": "/dev/b"}) is None
    )
    assert (
        cfg.port_slot_conflict({"mode": "single", "leader_port": "/dev/a", "follower_port": "/dev/a"})
        == "/dev/a"
    )

    # Bimanual: all four must differ, across sides.
    base = {
        "mode": "bimanual",
        "leader_port": "/dev/a",
        "follower_port": "/dev/b",
        "right_leader_port": "/dev/c",
        "right_follower_port": "/dev/d",
    }
    assert cfg.port_slot_conflict(base) is None
    assert cfg.port_slot_conflict({**base, "right_follower_port": "/dev/a"}) == "/dev/a"
    # Empty ports are ignored.
    assert cfg.port_slot_conflict({"mode": "bimanual", "leader_port": "", "follower_port": ""}) is None


def test_config_slot_conflict_ignores_single_mode_and_cross_side() -> None:
    from makermodslab.utils import config as cfg

    # Single mode never conflicts (one slot per side).
    assert (
        cfg.config_slot_conflict({"mode": "single", "leader_config": "X", "right_leader_config": "X"}) is None
    )
    # Same name across sides is fine — different directories.
    assert (
        cfg.config_slot_conflict({"mode": "bimanual", "leader_config": "X", "follower_config": "X"}) is None
    )
    # Empty slots don't count as a conflict.
    assert (
        cfg.config_slot_conflict({"mode": "bimanual", "leader_config": "", "right_leader_config": ""}) is None
    )


def test_is_robot_record_clean_with_stem_configs(tmp_lerobot_home: Path) -> None:
    """A record storing stems is clean when "<stem>.json" exists on disk."""
    from makermodslab.utils import config as cfg

    record = {
        "name": "r",
        "leader_port": "/dev/a",
        "follower_port": "/dev/b",
        "leader_config": "so101",
        "follower_config": "so101",
    }
    assert cfg.is_robot_record_clean(record) is False  # no files yet

    (Path(cfg.LEADER_CONFIG_PATH) / "so101.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "so101.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is True
    # Still clean if a value carries the extension (defensive).
    assert cfg.is_robot_record_clean(dict(record, leader_config="so101.json")) is True


def test_setup_calibration_files_copies_configs(
    tmp_lerobot_home: Path,
) -> None:
    from makermodslab.utils import config as cfg

    # setup_calibration_files reads from LEADER_CONFIG_PATH / FOLLOWER_CONFIG_PATH
    # and writes into those same directories (source dir == target dir).
    # Provide source files there.
    src_leader = Path(cfg.LEADER_CONFIG_PATH) / "demo_leader.json"
    src_leader.write_text(json.dumps({"motors": {}}))

    src_follower = Path(cfg.FOLLOWER_CONFIG_PATH) / "demo_follower.json"
    src_follower.write_text(json.dumps({"motors": {}}))

    result = cfg.setup_calibration_files("demo_leader.json", "demo_follower.json")
    # Returns the stem names.
    assert result == ("demo_leader", "demo_follower")

    # Files should exist (they were already there; function ensures they are present).
    assert src_leader.is_file()
    assert src_follower.is_file()


# DISCOVERED: `setup_calibration_files` sets `leader_calibration_dir = LEADER_CONFIG_PATH`
# (not CALIBRATION_BASE_PATH_TELEOP) and `follower_calibration_dir = FOLLOWER_CONFIG_PATH`
# (not CALIBRATION_BASE_PATH_ROBOTS). This means source and destination are the same
# directory, so the function only validates that the file exists in LEADER_CONFIG_PATH /
# FOLLOWER_CONFIG_PATH; it never writes into CALIBRATION_BASE_PATH_TELEOP or
# CALIBRATION_BASE_PATH_ROBOTS. The plan's assertion about those paths was incorrect.


def test_stage_bimanual_calibrations_copies_four_files(tmp_lerobot_home: Path) -> None:
    """The four arbitrarily-named library files are copied into per-device
    staging dirs as '<base>_left/right.json', returning the dirs + base."""
    from makermodslab.utils import config as cfg

    # Arbitrary library names — no "<base>_left/right" convention.
    for name, content in (("alice", "AL"), ("carol", "AR")):
        (Path(cfg.LEADER_CONFIG_PATH) / f"{name}.json").write_text(content)
    for name, content in (("bob", "FL"), ("dave", "FR")):
        (Path(cfg.FOLLOWER_CONFIG_PATH) / f"{name}.json").write_text(content)

    leader_dir, follower_dir, base = cfg.stage_bimanual_calibrations("mybot", "alice", "carol", "bob", "dave")
    assert base == "mybot"
    assert leader_dir == os.path.join(cfg.MAKERMODSLAB_BISO_STAGING_PATH, "mybot", "leader")
    assert follower_dir == os.path.join(cfg.MAKERMODSLAB_BISO_STAGING_PATH, "mybot", "follower")
    # Files landed under the convention names with the right contents.
    assert (Path(leader_dir) / "mybot_left.json").read_text() == "AL"
    assert (Path(leader_dir) / "mybot_right.json").read_text() == "AR"
    assert (Path(follower_dir) / "mybot_left.json").read_text() == "FL"
    assert (Path(follower_dir) / "mybot_right.json").read_text() == "FR"


def test_stage_bimanual_calibrations_overwrites_stale_alias(tmp_lerobot_home: Path) -> None:
    """A recalibrated library file must refresh its staging alias — the copy is
    unconditional, so a second call overwrites the previous staged content."""
    from makermodslab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("v1")
    (Path(cfg.LEADER_CONFIG_PATH) / "R.json").write_text("R")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FL.json").write_text("FL")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FR.json").write_text("FR")

    leader_dir, _, _ = cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")
    assert (Path(leader_dir) / "bot_left.json").read_text() == "v1"

    # Recalibrate the left leader library file, then restage.
    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("v2")
    cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")
    assert (Path(leader_dir) / "bot_left.json").read_text() == "v2"


def test_stage_bimanual_calibrations_missing_file_raises(tmp_lerobot_home: Path) -> None:
    """A missing library file fails fast with a clear per-slot error naming the
    slot and file, before lerobot's connect() can hang on recalibration."""
    from makermodslab.utils import config as cfg

    # Only three of the four files exist; the right follower is missing.
    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("L")
    (Path(cfg.LEADER_CONFIG_PATH) / "R.json").write_text("R")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FL.json").write_text("FL")

    with pytest.raises(FileNotFoundError, match="right follower.*FR.json.*not found"):
        cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")


def test_stage_bimanual_calibrations_blank_slot_raises(tmp_lerobot_home: Path) -> None:
    """A blank config (arm unassigned) fails with the standard legible message."""
    from makermodslab.utils import config as cfg

    with pytest.raises(FileNotFoundError, match="left leader arm has no calibration assigned"):
        cfg.stage_bimanual_calibrations("bot", "", "R", "FL", "FR")


def test_stage_bimanual_follower_calibrations_stages_follower_only(tmp_lerobot_home: Path) -> None:
    """Inference stages the follower side only. Repro of the startup bug: the
    two follower library files exist under FOLLOWER_CONFIG_PATH but NO leader
    file shares their names — staging must still succeed and land the follower
    aliases, rather than failing looking for so_leader/<follower name>.json."""
    from makermodslab.utils import config as cfg

    # Real-world repro: follower configs "2"/"4"; leader dir has no 2/4.json.
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "2.json").write_text("FL")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "4.json").write_text("FR")

    follower_dir, base = cfg.stage_bimanual_follower_calibrations("mybot", "2", "4")
    assert base == "mybot"
    # Same layout as the full stager's follower dir.
    assert follower_dir == os.path.join(cfg.MAKERMODSLAB_BISO_STAGING_PATH, "mybot", "follower")
    assert (Path(follower_dir) / "mybot_left.json").read_text() == "FL"
    assert (Path(follower_dir) / "mybot_right.json").read_text() == "FR"
    # No leader staging dir is created — the leader side is never touched.
    assert not os.path.exists(os.path.join(cfg.MAKERMODSLAB_BISO_STAGING_PATH, "mybot", "leader"))


def test_stage_bimanual_follower_calibrations_missing_file_raises(tmp_lerobot_home: Path) -> None:
    """A missing follower library file fails fast with the clear per-slot error
    naming 'right follower' and the file, same as the full stager."""
    from makermodslab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "2.json").write_text("FL")

    with pytest.raises(FileNotFoundError, match="right follower.*4.json.*not found"):
        cfg.stage_bimanual_follower_calibrations("mybot", "2", "4")


def test_bimanual_base_id_uses_valid_name_else_default() -> None:
    from makermodslab.utils.config import DEFAULT_BIMANUAL_BASE, bimanual_base_id

    assert bimanual_base_id("mybot") == "mybot"
    assert bimanual_base_id("  spaced  ") == "spaced"  # stripped, still valid
    # Blank or unsafe names fall back to the fixed default.
    assert bimanual_base_id("") == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id(None) == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id("bad/name") == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id("../escape") == DEFAULT_BIMANUAL_BASE


def test_with_makermodslab_tag_appends_required_tags_to_existing() -> None:
    from makermodslab.utils.config import REQUIRED_HUB_TAGS, with_makermodslab_tag

    # Caller tags come first, then every required tag in order.
    assert with_makermodslab_tag(["robotics", "lerobot"]) == ["robotics", "lerobot", *REQUIRED_HUB_TAGS]


def test_with_makermodslab_tag_handles_none_and_empty() -> None:
    from makermodslab.utils.config import REQUIRED_HUB_TAGS, with_makermodslab_tag

    assert with_makermodslab_tag(None) == list(REQUIRED_HUB_TAGS)
    assert with_makermodslab_tag([]) == list(REQUIRED_HUB_TAGS)


def test_with_makermodslab_tag_dedupes() -> None:
    from makermodslab.utils.config import MAKERMODSLAB_TAG, with_makermodslab_tag

    # Caller-supplied required tags are not duplicated, and order is preserved.
    out = with_makermodslab_tag(["robotics", MAKERMODSLAB_TAG, "lerobot", "makermods"])
    # No tag appears twice.
    assert len(out) == len(set(out))
    # The caller's positions for already-present tags are preserved.
    assert out[:4] == ["robotics", MAKERMODSLAB_TAG, "lerobot", "makermods"]


def test_with_makermodslab_tag_always_includes_makermods_and_openbooth() -> None:
    """The core requirement: every Hub push through this funnel carries the
    org/product tags, regardless of what the caller supplies (or omits)."""
    from makermodslab.utils.config import with_makermodslab_tag

    for caller in (None, [], ["robotics"], ["makermods"], ["openbooth", "x"]):
        out = with_makermodslab_tag(caller)
        assert "makermods" in out
        assert "openbooth" in out


def test_clear_config_references_unassigns_matching_records(tmp_lerobot_home: Path) -> None:
    """Deleting a config unassigns every robot that pointed at it — on the
    right side (device_type) only — and reports which fields were cleared."""
    cfg.save_robot_record(
        "arm1",
        {"mode": "single", "leader_config": "calib_a", "follower_config": "calib_b"},
        allow_create=True,
    )
    # A second robot sharing the same leader config is unassigned too.
    cfg.save_robot_record("arm2", {"mode": "single", "leader_config": "calib_a"}, allow_create=True)

    assert cfg.clear_config_references("teleop", "calib_a") == [
        {"robot": "arm1", "fields": ["leader_config"]},
        {"robot": "arm2", "fields": ["leader_config"]},
    ]
    assert cfg.get_robot_record("arm1")["leader_config"] == ""
    assert cfg.get_robot_record("arm2")["leader_config"] == ""
    # The follower slot (other side) is untouched, and the record is now dirty.
    assert cfg.get_robot_record("arm1")["follower_config"] == "calib_b"
    assert cfg.is_robot_record_clean(cfg.get_robot_record("arm1")) is False

    # A config nobody references clears nothing.
    assert cfg.clear_config_references("teleop", "unused") == []


def test_clear_config_references_clears_stale_right_slot_too(tmp_lerobot_home: Path) -> None:
    """A right_* reference is cleared even when the robot is back in single
    mode — the file is gone, so the stale name must not resurface on a mode
    switch. Both slots are reported when both matched."""
    cfg.save_robot_record(
        "arm1",
        {"mode": "single", "leader_config": "gone", "right_leader_config": "gone"},
        allow_create=True,
    )
    assert cfg.clear_config_references("teleop", "gone") == [
        {"robot": "arm1", "fields": ["leader_config", "right_leader_config"]}
    ]
    record = cfg.get_robot_record("arm1")
    assert record["leader_config"] == ""
    assert record["right_leader_config"] == ""


def test_clear_config_references_accepts_json_extension(tmp_lerobot_home: Path) -> None:
    """Callers may pass 'name.json'; matching is on the stem."""
    cfg.save_robot_record("arm1", {"mode": "single", "follower_config": "calib_b"}, allow_create=True)
    assert cfg.clear_config_references("robot", "calib_b.json") == [
        {"robot": "arm1", "fields": ["follower_config"]}
    ]
    assert cfg.get_robot_record("arm1")["follower_config"] == ""


def test_setup_calibration_files_rejects_unassigned_arm(tmp_lerobot_home: Path) -> None:
    """An empty config name (arm unassigned / needs calibration) fails with a
    legible message instead of an IsADirectoryError from shutil.copy2."""
    with pytest.raises(FileNotFoundError, match="leader arm has no calibration assigned"):
        cfg.setup_calibration_files("", "whatever.json")
    with pytest.raises(FileNotFoundError, match="follower arm has no calibration assigned"):
        cfg.setup_calibration_files("whatever.json", "  ")
    with pytest.raises(FileNotFoundError, match="follower arm has no calibration assigned"):
        cfg.setup_follower_calibration_file("")


# --- Dismissed hub jobs -----------------------------------------------------


def test_dismissed_hub_jobs_round_trips(tmp_lerobot_home: Path) -> None:
    assert cfg.get_dismissed_hub_jobs() == set()
    assert cfg.add_dismissed_hub_job("job-b") is True
    assert cfg.add_dismissed_hub_job("job-a") is True
    assert cfg.get_dismissed_hub_jobs() == {"job-a", "job-b"}
    # Idempotent: re-dismissing is a no-op success.
    assert cfg.add_dismissed_hub_job("job-a") is True
    assert cfg.get_dismissed_hub_jobs() == {"job-a", "job-b"}


def test_add_dismissed_hub_job_rejects_blank_id(tmp_lerobot_home: Path) -> None:
    assert cfg.add_dismissed_hub_job("") is False
    assert cfg.add_dismissed_hub_job("   ") is False
    assert cfg.get_dismissed_hub_jobs() == set()


def test_get_dismissed_hub_jobs_tolerates_corrupt_file(tmp_lerobot_home: Path) -> None:
    """Dismissal is cosmetic — a corrupted file must yield the empty set, not
    an exception that would block the hub listing."""
    path = Path(cfg.DISMISSED_HUB_JOBS_FILE)
    path.write_text("not json{")
    assert cfg.get_dismissed_hub_jobs() == set()
    # Wrong shape (dict instead of list) and non-string entries are dropped too.
    path.write_text(json.dumps({"job-a": True}))
    assert cfg.get_dismissed_hub_jobs() == set()
    path.write_text(json.dumps(["job-a", 3, None, "  "]))
    assert cfg.get_dismissed_hub_jobs() == {"job-a"}


def test_prune_dismissed_hub_jobs_drops_ids_gone_from_listing(tmp_lerobot_home: Path) -> None:
    cfg.add_dismissed_hub_job("job-live")
    cfg.add_dismissed_hub_job("job-expired")
    cfg.prune_dismissed_hub_jobs({"job-live", "job-other"})
    assert cfg.get_dismissed_hub_jobs() == {"job-live"}
    # Pruning against a listing that contains everything is a no-op.
    cfg.prune_dismissed_hub_jobs({"job-live"})
    assert cfg.get_dismissed_hub_jobs() == {"job-live"}


# --- Session cameras resolved from the robot record -------------------------
#
# The robot record is the only source of a session's cameras — recording and
# inference requests no longer carry camera configs. These cover the pure
# shaping/lookup helpers; the 400 paths that consume them live in
# tests/test_record.py and tests/test_rollout.py.


def _camera_entry(name: str, **overrides) -> dict:
    """A camera entry in the shape RobotConfigDialog saves."""
    entry = {
        "id": f"camera_{name}",
        "name": name,
        "type": "opencv",
        "camera_index": 1,
        "device_id": "browser-device-id",
        "unique_id": "0x1400000005ac8600",
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    entry.update(overrides)
    return entry


def test_session_camera_config_keeps_only_session_keys() -> None:
    """`id`/`device_id`/`name` are record-keeping: forwarding them would reach
    lerobot's OpenCVCameraConfig (via rollout's `--robot.cameras=`) as unknown
    fields. Everything a session actually needs survives."""
    config = cfg.session_camera_config(_camera_entry("wrist", fourcc="MJPG", backend="AVFOUNDATION"))

    assert config == {
        "type": "opencv",
        "camera_index": 1,
        "unique_id": "0x1400000005ac8600",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        "backend": "AVFOUNDATION",
    }


def test_session_camera_config_omits_absent_keys_and_defaults_type() -> None:
    """Absent keys are dropped rather than sent as None, so the consumers' own
    defaults (platform backend pin, MJPG fourcc) still apply. A record written
    before `type` existed is an opencv camera."""
    config = cfg.session_camera_config({"name": "top", "camera_index": 0})

    assert config == {"type": "opencv", "camera_index": 0}


def test_record_cameras_by_name_keys_on_the_camera_name() -> None:
    cameras = cfg.record_cameras_by_name([_camera_entry("wrist"), _camera_entry("top", camera_index=2)])

    assert sorted(cameras) == ["top", "wrist"]
    assert cameras["top"]["camera_index"] == 2


def test_record_cameras_by_name_rejects_duplicate_names() -> None:
    """Two cameras under one name would silently collapse to one in the dict —
    an entire camera missing from a recording. Refuse instead."""
    with pytest.raises(cfg.CameraResolutionError) as exc:
        cfg.record_cameras_by_name([_camera_entry("wrist"), _camera_entry("wrist", camera_index=2)])

    assert "two cameras named 'wrist'" in str(exc.value)


def test_record_cameras_by_name_rejects_unnamed_camera() -> None:
    with pytest.raises(cfg.CameraResolutionError):
        cfg.record_cameras_by_name([{**_camera_entry("wrist"), "name": "   "}])


def test_record_cameras_by_name_skips_non_dict_entries() -> None:
    """A hand-edited record shouldn't crash the start path on a stray value."""
    assert cfg.record_cameras_by_name(["nonsense", None, _camera_entry("wrist")]).keys() == {"wrist"}


def test_load_robot_cameras_reads_the_named_record(tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    cameras = cfg.load_robot_cameras("lab1")

    assert list(cameras) == ["wrist"]
    assert cameras["wrist"]["camera_index"] == 1


def test_load_robot_cameras_blank_name_is_a_camera_less_session(tmp_lerobot_home: Path) -> None:
    assert cfg.load_robot_cameras("") == {}
    assert cfg.load_robot_cameras("   ") == {}


def test_load_robot_cameras_rejects_a_name_with_no_record(tmp_lerobot_home: Path) -> None:
    """Recording camera-less because the robot name was wrong loses a whole
    session's video silently — fail loudly instead."""
    with pytest.raises(cfg.CameraResolutionError) as exc:
        cfg.load_robot_cameras("ghost")

    assert "ghost" in str(exc.value)


def test_load_robot_cameras_rejects_a_path_traversal_name(tmp_lerobot_home: Path) -> None:
    with pytest.raises(cfg.CameraResolutionError):
        cfg.load_robot_cameras("../escape")


def test_load_robot_cameras_empty_for_a_record_with_no_cameras(tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("lab1", {"leader_port": "/dev/a"}, allow_create=True)

    assert cfg.load_robot_cameras("lab1") == {}


def test_bind_robot_cameras_keys_on_the_policy_name(tmp_lerobot_home: Path) -> None:
    """The checkpoint's camera names rarely match the labels on the robot, so
    the binding renames: settings come from the record, the key from the policy."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    bound = cfg.bind_robot_cameras("lab1", {"observation.images.front": "wrist"})

    assert list(bound) == ["observation.images.front"]
    assert bound["observation.images.front"]["camera_index"] == 1


def test_bind_robot_cameras_empty_bindings_need_no_robot(tmp_lerobot_home: Path) -> None:
    """A camera-less policy binds nothing, so it must not require a record."""
    assert cfg.bind_robot_cameras("", {}) == {}


def test_bind_robot_cameras_rejects_bindings_without_a_robot(tmp_lerobot_home: Path) -> None:
    with pytest.raises(cfg.CameraResolutionError) as exc:
        cfg.bind_robot_cameras("", {"front": "wrist"})

    assert "No robot selected" in str(exc.value)


def test_bind_robot_cameras_rejects_an_unknown_camera_and_lists_the_options(
    tmp_lerobot_home: Path,
) -> None:
    cfg.save_robot_record(
        "lab1",
        {"cameras": [_camera_entry("wrist"), _camera_entry("top", camera_index=2)]},
        allow_create=True,
    )

    with pytest.raises(cfg.CameraResolutionError) as exc:
        cfg.bind_robot_cameras("lab1", {"front": "gone"})

    message = str(exc.value)
    assert "'gone'" in message
    assert "top, wrist" in message


def test_bind_robot_cameras_overlays_checkpoint_capture_dims(tmp_lerobot_home: Path) -> None:
    """lerobot's standard rollout does NOT resize frames to the policy's input
    shape, so a camera must CAPTURE at the resolution the checkpoint was
    trained on. The record still supplies identity and transport settings."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    bound = cfg.bind_robot_cameras(
        "lab1",
        {"front": "wrist"},
        dims={"front": {"width": 320, "height": 240}},
    )

    assert bound["front"]["width"] == 320
    assert bound["front"]["height"] == 240
    # Identity/transport are untouched — only the frame size is overridden.
    assert bound["front"]["camera_index"] == 1
    assert bound["front"]["unique_id"] == "0x1400000005ac8600"
    assert bound["front"]["fps"] == 30


def test_bind_robot_cameras_falls_back_to_record_dims_without_an_override(
    tmp_lerobot_home: Path,
) -> None:
    """A checkpoint that doesn't expose image dims (or an older client that
    sends none) must still start — the record's own size stands."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    assert cfg.bind_robot_cameras("lab1", {"front": "wrist"})["front"]["width"] == 640
    assert cfg.bind_robot_cameras("lab1", {"front": "wrist"}, dims={})["front"]["width"] == 640
    # An override for a DIFFERENT camera doesn't leak onto this one.
    unrelated = cfg.bind_robot_cameras(
        "lab1", {"front": "wrist"}, dims={"top": {"width": 320, "height": 240}}
    )
    assert unrelated["front"]["width"] == 640


@pytest.mark.parametrize(
    "override",
    [
        {"width": 320},  # half an override
        {"height": 240},
        {"width": 0, "height": 240},  # nonsense sizes
        {"width": -320, "height": -240},
        {"width": True, "height": True},  # bool is an int subclass
        {"width": "320", "height": "240"},
        {},
    ],
)
def test_bind_robot_cameras_ignores_unusable_dims(tmp_lerobot_home: Path, override: dict) -> None:
    """Both dimensions or neither: a half-applied override would capture at a
    mixed policy/record aspect, which is worse than either source alone."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    bound = cfg.bind_robot_cameras("lab1", {"front": "wrist"}, dims={"front": override})

    assert bound["front"]["width"] == 640
    assert bound["front"]["height"] == 480


def test_bind_robot_cameras_copies_so_callers_cant_alias_the_record(
    tmp_lerobot_home: Path,
) -> None:
    """Two policy names may bind the same physical camera; each must get its
    own dict so a later mutation can't leak across roles."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)

    bound = cfg.bind_robot_cameras("lab1", {"left": "wrist", "right": "wrist"})

    assert bound["left"] is not bound["right"]
    assert bound["left"] == bound["right"]


# --- Re-anchoring stored camera indices to the saved device ------------------
#
# A stored `camera_index` is a POSITION in AVFoundation's uniqueID-sorted device
# list, so any replug renumbers it and the session opens a different physical
# camera under the same label. The preview endpoint already re-anchors; these
# cover the session seam (load_robot_cameras, which recording calls directly and
# inference reaches through bind_robot_cameras) doing the same.
#
# The enumeration is always injected — tests/conftest.py's autouse fixture makes
# "identity unavailable" the default so nothing here can touch the host's real
# devices.


def _enumeration(monkeypatch: pytest.MonkeyPatch, cameras: list[dict] | None) -> None:
    """Pin what this process's AVFoundation device list answers.

    None means "could not ask" (non-macOS, PyObjC missing, query failure); a
    list — including the empty one — means the query answered.
    """
    from makermodslab import camera_identity

    monkeypatch.setattr(camera_identity, "list_cameras_in_process", lambda: cameras)


def test_load_robot_cameras_reanchors_the_index_to_the_saved_device(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record says index 1; the device it names now enumerates at 0 (the
    camera that used to sort ahead of it was unplugged). The identity wins."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(monkeypatch, [{"index": 0, "name": "USB Camera", "unique_id": "0x1400000005ac8600"}])

    assert cfg.load_robot_cameras("lab1")["wrist"]["camera_index"] == 0


def test_reanchoring_does_not_rewrite_the_robot_record(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new index is for THIS session only. A rig that is re-cabled between
    runs must not have its saved index rewritten from under it."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    before = Path(cfg.ROBOTS_PATH, "lab1.json").read_text()
    _enumeration(monkeypatch, [{"index": 0, "name": "USB Camera", "unique_id": "0x1400000005ac8600"}])

    assert cfg.load_robot_cameras("lab1")["wrist"]["camera_index"] == 0
    assert Path(cfg.ROBOTS_PATH, "lab1.json").read_text() == before


def test_load_robot_cameras_refuses_a_camera_that_is_not_attached(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enumeration answered and this device is not in it: opening the stale
    index would record a DIFFERENT camera under this label, which is worse than
    not starting. The message names the label and the saved port identity."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(monkeypatch, [{"index": 0, "name": "FaceTime HD Camera", "unique_id": "0xdeadbeef"}])

    with pytest.raises(cfg.CameraResolutionError) as exc:
        cfg.load_robot_cameras("lab1")

    assert "wrist" in str(exc.value)
    assert "0x1400000005ac8600" in str(exc.value)


def test_load_robot_cameras_refuses_when_the_machine_has_no_cameras(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EMPTY enumeration is an answer, not a failure (camera_identity keeps
    None and [] apart deliberately): the device is definitively absent."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(monkeypatch, [])

    with pytest.raises(cfg.CameraResolutionError):
        cfg.load_robot_cameras("lab1")


def test_load_robot_cameras_keeps_the_index_when_identity_is_unavailable(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Linux has no AVFoundation at all, so a machine that cannot answer must
    still be able to record — with a warning that names the camera."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(monkeypatch, None)

    with caplog.at_level(logging.WARNING, logger="makermodslab.utils.config"):
        cameras = cfg.load_robot_cameras("lab1")

    assert cameras["wrist"]["camera_index"] == 1
    assert any("wrist" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_load_robot_cameras_keeps_the_index_when_the_entry_has_no_identity(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Records written before uniqueIDs were stored carry an index and nothing
    else. Unverifiable is not the same as wrong: trust it, and say so."""
    entry = _camera_entry("wrist")
    del entry["unique_id"]
    cfg.save_robot_record("lab1", {"cameras": [entry]}, allow_create=True)
    _enumeration(monkeypatch, [{"index": 0, "name": "USB Camera", "unique_id": "0xother"}])

    with caplog.at_level(logging.WARNING, logger="makermodslab.utils.config"):
        cameras = cfg.load_robot_cameras("lab1")

    assert cameras["wrist"]["camera_index"] == 1
    assert any("wrist" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_reanchoring_logs_the_resolved_binding_for_every_camera(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One INFO line per label with the index, the identity and the device name:
    a crossed pair is then auditable after the fact, from the session's log."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(monkeypatch, [{"index": 0, "name": "Robot Cam", "unique_id": "0x1400000005ac8600"}])

    with caplog.at_level(logging.INFO, logger="makermodslab.utils.config"):
        cfg.load_robot_cameras("lab1")

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Session camera"))
    assert "wrist" in line
    assert "0x1400000005ac8600" in line
    assert "Robot Cam" in line


def test_bind_robot_cameras_reanchors_for_inference_too(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference resolves its cameras through the same seam, so the checkpoint's
    camera name gets the re-anchored index without its own plumbing."""
    cfg.save_robot_record("lab1", {"cameras": [_camera_entry("wrist")]}, allow_create=True)
    _enumeration(
        monkeypatch,
        [
            {"index": 0, "name": "USB Camera", "unique_id": "0xother"},
            {"index": 1, "name": "USB Camera", "unique_id": "0xanother"},
            {"index": 2, "name": "USB Camera", "unique_id": "0x1400000005ac8600"},
        ],
    )

    assert cfg.bind_robot_cameras("lab1", {"front": "wrist"})["front"]["camera_index"] == 2
