"""MAKERMODSLAB_HOME resolution and the one-shot migration out of lerobot's cache.

The migration is pure filesystem work with no clock and no network, so it is
tested against real tmp dirs. Every destination is read from the config
module's globals at call time, which is what lets these tests point them into
tmp without touching a developer's real state.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from makermodslab.utils import config as cfg


def test_default_home_is_under_the_makermods_dotdir() -> None:
    assert cfg.resolve_makermodslab_home({}) == os.path.expanduser("~/.makermods/makermodslab")


def test_env_override_wins_and_is_expanded(tmp_path: Path) -> None:
    assert cfg.resolve_makermodslab_home({"MAKERMODSLAB_HOME": str(tmp_path)}) == str(tmp_path)
    assert cfg.resolve_makermodslab_home({"MAKERMODSLAB_HOME": "~/x"}) == os.path.expanduser("~/x")


def test_empty_override_falls_back_to_the_default() -> None:
    assert cfg.resolve_makermodslab_home({"MAKERMODSLAB_HOME": ""}) == cfg.resolve_makermodslab_home({})


def test_every_moved_constant_lives_under_the_home() -> None:
    for _legacy_name, attr in cfg._LEGACY_STATE_ENTRIES:
        assert getattr(cfg, attr).startswith(cfg.MAKERMODSLAB_HOME), attr


def test_calibration_libraries_stay_in_the_lerobot_cache() -> None:
    assert not cfg.CALIBRATION_BASE_PATH_TELEOP.startswith(cfg.MAKERMODSLAB_HOME)
    assert not cfg.CALIBRATION_BASE_PATH_ROBOTS.startswith(cfg.MAKERMODSLAB_HOME)


@pytest.fixture
def split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A legacy root and a home, with every destination constant pointed at the home."""
    legacy = tmp_path / "legacy"
    home = tmp_path / "home"
    legacy.mkdir()
    monkeypatch.setattr(cfg, "MAKERMODSLAB_HOME", str(home))
    for _legacy_name, attr in cfg._LEGACY_STATE_ENTRIES:
        new_name = os.path.basename(getattr(cfg, attr))
        monkeypatch.setattr(cfg, attr, str(home / new_name))
    return legacy, home


def test_migration_moves_files_and_dirs(split: tuple[Path, Path]) -> None:
    legacy, home = split
    (legacy / "robots").mkdir()
    (legacy / "robots" / "bot.json").write_text("{}")
    (legacy / "nodes.json").write_text("[]")
    (legacy / "instance_id.txt").write_text("a" * 32)

    moved = cfg.migrate_legacy_state(str(legacy))

    assert sorted(moved) == sorted([cfg.ROBOTS_PATH, cfg.NODES_FILE, cfg.INSTANCE_ID_FILE])
    assert (home / "robots" / "bot.json").read_text() == "{}"
    assert (home / "nodes.json").read_text() == "[]"
    assert not (legacy / "robots").exists()
    assert not (legacy / "nodes.json").exists()


def test_migration_renames_the_biso_staging_dir(split: tuple[Path, Path]) -> None:
    legacy, home = split
    (legacy / "makermodslab_biso" / "bimanual").mkdir(parents=True)

    cfg.migrate_legacy_state(str(legacy))

    assert (home / "biso_staging" / "bimanual").is_dir()
    assert str(home / "biso_staging") == cfg.MAKERMODSLAB_BISO_STAGING_PATH


def test_existing_destination_wins(split: tuple[Path, Path]) -> None:
    legacy, home = split
    (legacy / "nodes.json").write_text("old")
    home.mkdir()
    (home / "nodes.json").write_text("new")

    assert cfg.migrate_legacy_state(str(legacy)) == []
    assert (home / "nodes.json").read_text() == "new"
    assert (legacy / "nodes.json").read_text() == "old"


def test_migration_is_idempotent_and_tolerates_a_missing_legacy_root(split: tuple[Path, Path]) -> None:
    legacy, _home = split
    (legacy / "ports").mkdir()
    (legacy / "ports" / "leader_port.txt").write_text("/dev/ttyUSB0")

    first = cfg.migrate_legacy_state(str(legacy))
    second = cfg.migrate_legacy_state(str(legacy))
    nowhere = cfg.migrate_legacy_state(str(legacy / "does-not-exist"))

    assert first == [cfg.PORT_CONFIG_PATH]
    assert second == []
    assert nowhere == []


def test_a_failed_move_is_skipped_not_fatal(
    split: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, home = split
    (legacy / "nodes.json").write_text("[]")
    (legacy / "robots").mkdir()

    def boom(src: str, dst: str) -> None:
        if src.endswith("nodes.json"):
            raise OSError("disk says no")
        os.rename(src, dst)

    monkeypatch.setattr(cfg.shutil, "move", boom)

    moved = cfg.migrate_legacy_state(str(legacy))

    assert moved == [cfg.ROBOTS_PATH]
    assert (legacy / "nodes.json").exists()
    assert (home / "robots").is_dir()


def test_the_suite_runs_under_an_override_so_startup_never_migrates_real_state() -> None:
    # conftest pins MAKERMODSLAB_HOME before the package is imported; the
    # server's startup hook keys off this flag to skip the migration.
    assert cfg.HOME_IS_OVERRIDDEN is True
