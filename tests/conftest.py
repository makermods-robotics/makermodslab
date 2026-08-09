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
"""Shared pytest fixtures for the MakerMods Lab test suite."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from makermodslab.utils import config as cfg

# The developer's REAL persisted state. Nothing in the suite may write under
# here, and nothing may read it as the source of truth for an assertion.
REAL_HF_CACHE_ROOT = Path("~/.cache/huggingface").expanduser()
REAL_LEROBOT_CACHE = REAL_HF_CACHE_ROOT / "lerobot"
REAL_CALIBRATION_ROOT = REAL_LEROBOT_CACHE / "calibration"

# Every `from .utils.config import <PATH_CONSTANT>` in the package, as
# (module, attribute). A from-import binds a COPY of the string at import time,
# so `monkeypatch.setattr(cfg, ...)` never reaches these names — each one has to
# be repointed on its own module. Keep this list in sync with:
#
#     grep -rn "from .utils.config import" makermodslab/
#
# A missing entry is not a cosmetic gap. The calibration-config DELETE endpoint
# builds its `os.remove()` target from `server.py`'s own copy of
# FOLLOWER_CONFIG_PATH, so a test exercising that path with only `cfg` patched
# deletes out of the developer's real SO-101 calibration dir — which costs a
# hardware recalibration to get back.
_FROM_IMPORTED_PATH_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("makermodslab.auto_calibrate", "CALIBRATION_BASE_PATH_ROBOTS"),
    ("makermodslab.auto_calibrate", "FOLLOWER_CONFIG_PATH"),
    ("makermodslab.auto_calibrate", "LEADER_CONFIG_PATH"),
    ("makermodslab.arm_identity", "FOLLOWER_CONFIG_PATH"),
    ("makermodslab.arm_identity", "LEADER_CONFIG_PATH"),
    ("makermodslab.server", "FOLLOWER_CONFIG_PATH"),
    ("makermodslab.server", "LEADER_CONFIG_PATH"),
)


def _assert_outside(label: str, value: str | Path, forbidden_root: Path) -> None:
    """Fail loudly if `value` resolves to `forbidden_root` or anything inside it."""
    resolved = Path(value).resolve()
    assert forbidden_root not in [resolved, *resolved.parents], (
        f"REFUSING TO RUN: {label} is {resolved}, inside the real cache {forbidden_root}. "
        "A test writing through this constant would destroy real user data. If this is a newly "
        "from-imported constant, add it to _FROM_IMPORTED_PATH_CONSTANTS in tests/conftest.py."
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """FastAPI TestClient bound to the real `makermodslab.server.app`."""
    from makermodslab.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_lerobot_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every persisted-state path under `~/.cache/huggingface/lerobot/`
    into a tmp directory.

    Patches the module-level constants in `makermodslab.utils.config`, which
    covers every consumer that reads them THROUGH the module (`cfg.ROBOTS_PATH`,
    or a helper inside config.py itself). Also sets the `HF_LEROBOT_HOME` env var
    for any consumer (e.g. `makermodslab.datasets._lerobot_cache_root`) reading it
    directly.

    A `from makermodslab.utils.config import LEADER_CONFIG_PATH` is NOT covered by
    patching `cfg`: the from-import binds a copy of the string into the importing
    module's namespace at import time, and rebinding the name on `cfg` afterwards
    leaves that copy pointing at the developer's real ~/.cache. Every such copy
    therefore has to be repointed on its own module — `_FROM_IMPORTED_PATH_CONSTANTS`
    above lists them and the loop below does it. This is not hypothetical: the
    calibration-config DELETE endpoint `os.remove()`s through `server.py`'s own copy
    of FOLLOWER_CONFIG_PATH, which nothing in the suite repointed before this.

    The assertions below are the load-bearing part: they verify the redirect from
    the perspective of the code under test — including each from-imported copy — so
    a future refactor that reintroduces an unpatched path fails loudly here instead
    of quietly writing to the real cache. The session-scoped
    `_real_calibration_tree_canary` is the backstop for anything this misses.
    """
    cache = tmp_path / "lerobot"
    cache.mkdir()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(cache))

    teleop_dir = cache / "calibration" / "teleoperators" / "so101_leader"
    robot_dir = cache / "calibration" / "robots" / "so101_follower"
    leader_cfg_dir = cache / "configs" / "so_leader"
    follower_cfg_dir = cache / "configs" / "so_follower"
    port_dir = cache / "ports"
    robots_dir = cache / "robots"
    for d in (teleop_dir, robot_dir, leader_cfg_dir, follower_cfg_dir, port_dir, robots_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cfg, "CALIBRATION_BASE_PATH_TELEOP", str(teleop_dir))
    monkeypatch.setattr(cfg, "CALIBRATION_BASE_PATH_ROBOTS", str(robot_dir))
    # Robot records (named robot configs). Without this, every test that
    # exercises /robots writes into the developer's real ~/.cache dir.
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    monkeypatch.setattr(cfg, "LEADER_CONFIG_PATH", str(leader_cfg_dir))
    monkeypatch.setattr(cfg, "FOLLOWER_CONFIG_PATH", str(follower_cfg_dir))
    monkeypatch.setattr(cfg, "PORT_CONFIG_PATH", str(port_dir))
    monkeypatch.setattr(cfg, "LEADER_PORT_FILE", str(port_dir / "leader_port.txt"))
    monkeypatch.setattr(cfg, "FOLLOWER_PORT_FILE", str(port_dir / "follower_port.txt"))
    monkeypatch.setattr(cfg, "DISMISSED_HUB_JOBS_FILE", str(cache / "dismissed_hub_jobs.json"))
    # BiSO staging root — without this, any bimanual staging test writes into the
    # developer's real ~/.cache dir.
    monkeypatch.setattr(cfg, "MAKERMODSLAB_BISO_STAGING_PATH", str(cache / "makermodslab_biso"))

    # Repoint every from-imported COPY of a config path constant (see the
    # module-level list and this fixture's docstring). Patching `cfg` above does
    # not reach these; without this loop `makermodslab.server` still deletes, and
    # `makermodslab.auto_calibrate` still writes, under the developer's real
    # `calibration/robots/so_follower/`.
    for module_name, attr in _FROM_IMPORTED_PATH_CONSTANTS:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attr, getattr(cfg, attr))

    # Verify the redirect from the perspective of the code under test, now that
    # every constant has been patched. An unpatched path is a data-loss bug, so
    # fail here rather than let the test body discover it by writing.
    for name in (
        "CALIBRATION_BASE_PATH_TELEOP",
        "CALIBRATION_BASE_PATH_ROBOTS",
        "LEADER_CONFIG_PATH",
        "FOLLOWER_CONFIG_PATH",
        "ROBOTS_PATH",
        "PORT_CONFIG_PATH",
        "MAKERMODSLAB_BISO_STAGING_PATH",
    ):
        _assert_outside(f"makermodslab.utils.config.{name}", getattr(cfg, name), REAL_LEROBOT_CACHE)
    for module_name, attr in _FROM_IMPORTED_PATH_CONSTANTS:
        value = getattr(importlib.import_module(module_name), attr)
        _assert_outside(f"{module_name}.{attr}", value, REAL_LEROBOT_CACHE)

    return cache


def _snapshot_real_calibration_tree() -> dict[str, tuple[int, int, str]]:
    """Fingerprint every file under the developer's REAL calibration tree.

    Strictly read-only, and tolerant of a missing tree (CI has none) — returns an
    empty mapping rather than creating anything. `.DS_Store` is skipped: Finder
    rewrites it while the suite runs, and no test has any reason to touch it.
    """
    snapshot: dict[str, tuple[int, int, str]] = {}
    if not REAL_CALIBRATION_ROOT.is_dir():
        return snapshot
    for path in sorted(REAL_CALIBRATION_ROOT.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        try:
            stat = path.stat()
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
        except OSError:  # unreadable is not this fixture's problem to report
            continue
        snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size, digest)
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def _real_calibration_tree_canary() -> Iterator[None]:
    """Fail the session, loudly, if the suite modified real calibration data.

    The per-test guards in `tmp_lerobot_home` only protect tests that USE that
    fixture, and only cover constants we already know about. This is the backstop:
    it compares the real tree before and after the whole session, so any escape —
    a new from-imported path constant, a test that forgot the fixture, a code path
    that expands `~` itself — is caught even though the damage is already done. A
    recalibration is expensive; finding out about it days later is worse.
    """
    before = _snapshot_real_calibration_tree()
    yield
    after = _snapshot_real_calibration_tree()
    if before == after:
        return

    problems = [f"  MODIFIED: {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]]
    problems += [f"  DELETED:  {p}" for p in sorted(before.keys() - after.keys())]
    problems += [f"  CREATED:  {p}" for p in sorted(after.keys() - before.keys())]
    pytest.fail(
        f"THE TEST SUITE WROTE TO THE REAL CALIBRATION TREE at {REAL_CALIBRATION_ROOT}:\n"
        + "\n".join(problems)
        + "\n\nThis is real robot calibration data — a modified or deleted file costs a "
        "hardware recalibration. Find the test that escaped `tmp_lerobot_home` (most likely "
        "reading a from-imported path constant that is not listed in "
        "_FROM_IMPORTED_PATH_CONSTANTS) before running the suite again.",
        pytrace=False,
    )


def _reset_module_caches() -> None:
    """Drop every process-lived, module-global cache/singleton state that could
    leak Hub answers (or a real-machine cache read) from one test into the next.

    Covers the short-TTL listing caches (/datasets, /models, /jobs/hub), the
    per-repo Hub-status / Hub-info memo dicts, and the two download-manager
    singletons' public state. The listing caches expose whole-cache invalidation
    functions; the per-repo memo dicts (keyed by repo_id, no whole-clear helper)
    are cleared directly under their locks — the same access pattern the dataset
    tests already use via their local _clear_hub_status_cache helper."""
    import makermodslab.datasets as _ds
    import makermodslab.models as _models
    import makermodslab.server as _srv

    _ds.invalidate_dataset_listing_cache()
    _models.invalidate_model_listing_cache()
    _srv.invalidate_hub_jobs_cache()

    with _ds._HUB_STATUS_LOCK:
        _ds._HUB_STATUS_CACHE.clear()
    with _ds._HUB_DATASET_INFO_LOCK:
        _ds._HUB_DATASET_INFO_CACHE.clear()
    with _models._MODEL_HUB_INFO_LOCK:
        _models._MODEL_HUB_INFO_CACHE.clear()

    # Reset both download-manager singletons to their idle shape so a test that
    # drove one (or hit a /download endpoint) can't leave "running"/"done"/"error"
    # visible to the next test's status poll. (No thread is torn down here: tests
    # join or mock their downloads; the singleton is only ever left dirty by state
    # writes, not live threads.)
    for _mgr in (_ds.download_manager, _models.model_download_manager):
        with _mgr._lock:
            _mgr.state = "idle"
            _mgr.repo_id = None
            _mgr.message = None
            _mgr.error = None


@pytest.fixture(autouse=True)
def _reset_hub_listing_caches() -> Iterator[None]:
    """Clear all process-lived Hub caches + download-manager singleton state
    before AND after each test so cached results (or a real-machine cache read)
    from one test never leak into the next. These caches/singletons are
    module-global and process-lived, so without this a test that populates one
    would make a later test see stale data instead of its own mocked response."""
    _reset_module_caches()
    yield
    _reset_module_caches()


@pytest.fixture
def mock_lerobot_record(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch `lerobot.record.record` so no real recording loop runs.

    Returns the MagicMock; tests can assert on `mock.called` or `mock.call_args`.
    """
    spy = MagicMock(name="lerobot.record.record")
    monkeypatch.setattr("lerobot.record.record", spy)
    return spy


@pytest.fixture
def mock_lerobot_teleoperate(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch `lerobot.teleoperate` so no real teleop loop runs."""
    spy = MagicMock(name="lerobot.teleoperate")
    monkeypatch.setattr("lerobot.teleoperate", spy)
    return spy


@pytest.fixture
def mock_subprocess_popen(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch `subprocess.Popen` (the symbol in makermodslab.jobs) so no real
    subprocess is launched. Returns a MagicMock whose return_value has the
    attributes a `Popen` instance is expected to have."""
    fake_proc = MagicMock(name="Popen()")
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None  # still running
    fake_proc.stdout = iter([])
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0
    fake_proc.kill.return_value = None

    spy = MagicMock(name="subprocess.Popen", return_value=fake_proc)
    monkeypatch.setattr("makermodslab.jobs.subprocess.Popen", spy, raising=False)
    return spy
