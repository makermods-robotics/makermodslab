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

import atexit
import hashlib
import importlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import constants as hf_constants

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
# formerly deleted out of the developer's real SO-101 calibration dir. Server
# now resolves these paths at call time; its retired copies are not listed.
_FROM_IMPORTED_PATH_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("makermodslab.auto_calibrate", "CALIBRATION_BASE_PATH_ROBOTS"),
    ("makermodslab.auto_calibrate", "FOLLOWER_CONFIG_PATH"),
    ("makermodslab.auto_calibrate", "LEADER_CONFIG_PATH"),
    ("makermodslab.arm_identity", "FOLLOWER_CONFIG_PATH"),
    ("makermodslab.arm_identity", "LEADER_CONFIG_PATH"),
)

# The same from-import problem one layer up: these modules bind their own copy
# of `hf_auth.cached_whoami`, which reads the developer's real HF token through
# `get_token()`. Stubbing `hf_auth.cached_whoami` would reach none of them — and
# would stop `tests/test_utils_hf_auth.py` testing the genuine caching
# behaviour — so the copies are stubbed individually instead. Keep in sync with:
#
#     grep -rn "cached_whoami" makermodslab/
_FROM_IMPORTED_WHOAMI: tuple[tuple[str, str], ...] = (
    ("makermodslab.server", "cached_whoami"),
    ("makermodslab.models", "cached_whoami"),
    ("makermodslab.datasets", "cached_whoami"),
    ("makermodslab.jobs", "cached_whoami"),
    ("makermodslab.runners.hf_cloud", "cached_whoami"),
)


def _assert_outside(label: str, value: str | Path, forbidden_root: Path) -> None:
    """Fail loudly if `value` resolves to `forbidden_root` or anything inside it."""
    resolved = Path(value).resolve()
    assert forbidden_root not in [resolved, *resolved.parents], (
        f"REFUSING TO RUN: {label} is {resolved}, inside the real cache {forbidden_root}. "
        "A test writing through this constant would destroy real user data. If this is a newly "
        "from-imported constant, add it to _FROM_IMPORTED_PATH_CONSTANTS in tests/conftest.py."
    )


# Redirect the module-level `job_registry` singleton away from real training
# history. This CANNOT be a fixture: `makermodslab.jobs` resolves
# `_DEFAULT_OUTPUT_ROOT` from this variable at import time and constructs the
# singleton (watchdog and all) at module scope, so by the time any fixture runs
# the root is already bound. conftest is imported before test modules, which is
# the only hook early enough.
#
# Without it the singleton points at ~/.cache/huggingface/lerobot/outputs/train,
# and any test that drives it writes job records into the developer's real
# history — and, because that registry's watchdog is a live thread, can promote
# an injected `queued` record and spawn an ACTUAL lerobot subprocess against it.
# `setdefault` so an explicit root set by the caller still wins.
_TEST_OUTPUT_ROOT = tempfile.mkdtemp(prefix="makermodslab-tests-")
if os.environ.setdefault("MAKERMODSLAB_OUTPUT_ROOT", _TEST_OUTPUT_ROOT) == _TEST_OUTPUT_ROOT:
    atexit.register(shutil.rmtree, _TEST_OUTPUT_ROOT, ignore_errors=True)
else:  # pragma: no cover - only when the caller pinned a root themselves
    shutil.rmtree(_TEST_OUTPUT_ROOT, ignore_errors=True)

# Same mechanism for the app's own state root. `makermodslab.utils.config`
# resolves MAKERMODSLAB_HOME at import, so this too must precede any import
# of the package. Two effects: every state constant (robot records, ports, the
# node list, the instance id, the UI bookkeeping files) points into a tmp dir
# even in a test that forgets the `tmp_lerobot_home` fixture — and, because
# the override is set, the server's startup migration is skipped, so a test
# run can never move a developer's real pre-split state anywhere (least of
# all into a tmp dir that is deleted at exit). Set UNCONDITIONALLY, unlike the
# output root: MAKERMODSLAB_HOME is a production override a station or a
# container exports, and honouring an exported value here would point the
# `client` fixture at that machine's real state.
_TEST_STATE_HOME = tempfile.mkdtemp(prefix="makermodslab-home-")
os.environ["MAKERMODSLAB_HOME"] = _TEST_STATE_HOME
atexit.register(shutil.rmtree, _TEST_STATE_HOME, ignore_errors=True)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """FastAPI TestClient bound to the real `makermodslab.server.app`."""
    from makermodslab.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_lerobot_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every persisted-state path — lerobot's cache AND the app's own
    MAKERMODSLAB_HOME — into per-test tmp directories.

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
    home = tmp_path / "makermodslab"
    home.mkdir()

    from makermodslab.utils import config as cfg

    monkeypatch.setattr(cfg, "MAKERMODSLAB_HOME", str(home))

    teleop_dir = cache / "calibration" / "teleoperators" / "so101_leader"
    robot_dir = cache / "calibration" / "robots" / "so101_follower"
    leader_cfg_dir = cache / "configs" / "so_leader"
    follower_cfg_dir = cache / "configs" / "so_follower"
    # The Maker arm's calibration libraries. Separate directories from the
    # SO-101 pair, so they need their own redirect — without it any test that
    # touches a Maker calibration writes into the developer's real ~/.cache.
    maker_leader_cfg_dir = cache / "configs" / "rebot_102_leader"
    maker_trigger_leader_cfg_dir = cache / "configs" / "rebot_102_leader_trigger"
    maker_follower_cfg_dir = cache / "configs" / "maker_follower"
    # The Metal arm's: its follower library, and the library of its OWN
    # (gravity-compensated) leader — the Star leader's is shared with Maker.
    metal_follower_cfg_dir = cache / "configs" / "metal_follower"
    metal_leader_cfg_dir = cache / "configs" / "metal_leader"
    port_dir = home / "ports"
    robots_dir = home / "robots"
    for d in (
        teleop_dir,
        robot_dir,
        leader_cfg_dir,
        follower_cfg_dir,
        maker_leader_cfg_dir,
        maker_trigger_leader_cfg_dir,
        maker_follower_cfg_dir,
        metal_follower_cfg_dir,
        metal_leader_cfg_dir,
        port_dir,
        robots_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cfg, "CALIBRATION_BASE_PATH_TELEOP", str(teleop_dir))
    monkeypatch.setattr(cfg, "CALIBRATION_BASE_PATH_ROBOTS", str(robot_dir))
    # Robot records (named robot configs). Without this, every test that
    # exercises /robots writes into the developer's real ~/.cache dir.
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    monkeypatch.setattr(cfg, "LEADER_CONFIG_PATH", str(leader_cfg_dir))
    monkeypatch.setattr(cfg, "FOLLOWER_CONFIG_PATH", str(follower_cfg_dir))
    monkeypatch.setattr(cfg, "MAKER_LEADER_CONFIG_PATH", str(maker_leader_cfg_dir))
    monkeypatch.setattr(cfg, "MAKER_TRIGGER_LEADER_CONFIG_PATH", str(maker_trigger_leader_cfg_dir))
    monkeypatch.setattr(cfg, "MAKER_FOLLOWER_CONFIG_PATH", str(maker_follower_cfg_dir))
    monkeypatch.setattr(cfg, "METAL_FOLLOWER_CONFIG_PATH", str(metal_follower_cfg_dir))
    monkeypatch.setattr(cfg, "METAL_LEADER_CONFIG_PATH", str(metal_leader_cfg_dir))
    monkeypatch.setattr(cfg, "PORT_CONFIG_PATH", str(port_dir))
    monkeypatch.setattr(cfg, "LEADER_PORT_FILE", str(port_dir / "leader_port.txt"))
    monkeypatch.setattr(cfg, "FOLLOWER_PORT_FILE", str(port_dir / "follower_port.txt"))
    monkeypatch.setattr(cfg, "DISMISSED_HUB_JOBS_FILE", str(home / "dismissed_hub_jobs.json"))
    # The pinned ("saved custom") and hidden repo-id lists. These leak the
    # HARDEST of the lot: every merged /datasets and /models listing folds them
    # in, so on a developer machine whose real saved_custom_models.json has
    # pinned repos, those repo ids appear in listings the test never seeded and
    # the listing assertions fail — on that machine only, invisibly in CI.
    # `_JsonRepoCollection` takes a `path_of` CALLABLE and re-invokes it on every
    # access precisely so a patched constant is honoured, and it holds no
    # in-memory copy, so redirecting the constant here is sufficient — there is
    # no cache to clear afterwards.
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_DATASETS_FILE", str(home / "saved_custom_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_MODELS_FILE", str(home / "saved_custom_models.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_DATASETS_FILE", str(home / "hidden_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_MODELS_FILE", str(home / "hidden_models.json"))
    monkeypatch.setattr(cfg, "EXCLUDED_EPISODES_FILE", str(home / "excluded_episodes.json"))
    # BiSO staging root — without this, any bimanual staging test writes into the
    # developer's real state dir.
    monkeypatch.setattr(cfg, "MAKERMODSLAB_BISO_STAGING_PATH", str(home / "biso_staging"))
    # Persisted node-registry peer list.
    monkeypatch.setattr(cfg, "NODES_FILE", str(home / "nodes.json"))

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
    # Redirected by `_isolate_real_user_state`, not here, but it is read at call
    # time by the "is this repo already downloaded?" probes and is the one Hub
    # path that would otherwise resolve into the real ~/.cache/huggingface.
    _assert_outside("huggingface_hub.constants.HF_HUB_CACHE", hf_constants.HF_HUB_CACHE, REAL_HF_CACHE_ROOT)

    return cache


@pytest.fixture(autouse=True)
def _isolate_real_user_state(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cut every real-machine path the dataset/model listings read out of the suite.

    `tmp_lerobot_home` redirects the calibration / robots / ports state, but the
    listing tests that don't request it get no redirection at all, and it never
    covered the four saved-repo JSON files the listings fold in. On a clean CI
    runner that is invisible; on a developer's populated machine the real state
    leaks straight into the assertions — a model listing expected to be
    ``["imported_policy"]`` also carries the developer's real pins
    (``lerobot/smolvla_base``), and a real ``hidden_datasets.json`` entry silently
    subtracts a row a test just seeded (observed:
    ``test_list_all_datasets_merges_hub_and_local`` KeyErrors when the real hidden
    set names the merged id).

    So redirect all of them, unconditionally, for every test:

      * the four ``SAVED_CUSTOM_*`` / ``SAVED_HIDDEN_*`` files — the pin lists
        folded in by `list_all_datasets` / `list_all_models`, and the hidden sets
        applied last: the same leak in the additive and subtractive directions.
        `_JsonRepoCollection` (utils/config.py) takes a `path_of` CALLABLE and
        re-invokes it on every access precisely so a patched constant is honoured,
        and it holds no in-memory copy — redirecting the constant is sufficient,
        there is no cache to clear afterwards.
      * ``HF_LEROBOT_HOME`` — read at call time by `datasets._lerobot_cache_root`,
        so it governs both the flat local-dataset scan and `_local_models_root()`.
      * ``huggingface_hub.constants.HF_HUB_CACHE`` — read off the module at call
        time by `_hub_cache_has_repo` and `try_to_load_from_cache`, i.e. the
        default hub cache probed with ``cache_dir=None``. Left alone, tests answer
        "is this repo already downloaded?" from whatever the developer happens to
        have downloaded — machine-dependent, and silently different in CI.

    Each redirect is a plain monkeypatch, so a test that wants its own root
    (`tmp_lerobot_home`, `custom_models_file`, `hidden_datasets_file`, an inline
    `SAVED_CUSTOM_DATASETS_FILE` patch, `_seed_hub_cache`) still overrides this
    one. This is only the floor that keeps the real ``~/.cache`` out of reach by
    default — a floor, not a wall.
    """
    from makermodslab.utils import config as cfg

    # Keep fixture bookkeeping outside the output directory owned by each test.
    isolated = tmp_path_factory.mktemp("isolated-user-state")
    # Saved repo lists are application state and must remain beneath its home.
    # The module-level setup has already isolated MAKERMODSLAB_HOME from the user.
    saved_root = Path(cfg.MAKERMODSLAB_HOME) / isolated.name
    saved_root.mkdir()
    lerobot_home = isolated / "lerobot"
    hub_cache = isolated / "hub"
    for d in (lerobot_home, hub_cache):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HF_LEROBOT_HOME", str(lerobot_home))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_cache))
    # str(), not Path: these config constants are strings everywhere else.
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_DATASETS_FILE", str(saved_root / "saved_custom_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_DATASETS_FILE", str(saved_root / "hidden_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_MODELS_FILE", str(saved_root / "saved_custom_models.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_MODELS_FILE", str(saved_root / "hidden_models.json"))


@pytest.fixture(autouse=True)
def _stub_cached_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop every listing/job code path reading the developer's real HF identity.

    `cached_whoami()` calls `get_token()`, so on a logged-in machine an unmocked
    test resolves the developer's real username and orgs — which decides default
    repo owners and which Hub rows a listing keeps. `None` is the shape the
    existing per-test mocks already use for "not logged in", so it is the
    offline-ish default here too.

    Only the from-imported copies are stubbed; `hf_auth.cached_whoami` itself
    stays real so `tests/test_utils_hf_auth.py` keeps exercising the genuine
    caching behaviour. Plain monkeypatches, so a test's own
    `patch("makermodslab.datasets.cached_whoami", ...)` still overrides.
    """
    for module_name, attr in _FROM_IMPORTED_WHOAMI:
        monkeypatch.setattr(importlib.import_module(module_name), attr, lambda *args, **kwargs: None)


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
        _ds._HUB_HAS_DATA_CACHE.clear()
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
def _no_real_gpu_app_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the GPU launcher's app-id record at tmp_path, for EVERY test.

    That file is what the orphan reaper reads at server startup, and what it
    finds decides whether it shells out to `modal app stop`. The `client`
    fixture runs the real startup event, so without this redirect a suite run
    on a machine that had just launched a GPU could stop the developer's actual
    Modal app. Redirecting it makes the record permanently empty in tests,
    which is the one state in which the reaper does nothing at all.
    """
    from makermodslab import modal_launcher

    monkeypatch.setattr(modal_launcher, "_APP_RECORD_FILE", tmp_path / "drtc_gpu_app.json")


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


@pytest.fixture(autouse=True)
def _reap_job_registry_threads(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop the threads of every JobRegistry a test builds, after the test.

    ~100 tests construct throwaway registries, and each one starts a
    1s-interval "job-registry-watchdog" daemon thread that nothing stops — a
    full run used to end with ~160 of them (plus a few runner tail/poll
    threads) still parked in Event.wait at interpreter exit. Daemon threads
    are normally frozen harmlessly at shutdown, but a teardown that catches
    one inside native code aborts the process AFTER a fully green summary
    (glibc's "FATAL: exception not rethrown", exit 134 — seen on the Linux CI
    runner), which fails the job with zero failing tests.

    Instances are tracked by wrapping __init__ (registries are created inside
    test bodies, so no fixture can hand them out), then stopped the way the
    app's own shutdown hook does: `shutdown()` sets the watchdog's stop event
    and the thread exits within its 1s wait. Runner threads (job-tail-*,
    hf-job-*) get their `_stop_event` set directly — deliberately NOT
    `runner.stop()`, which for a tailing runner SIGTERMs a real process
    group. The module-level `job_registry` singleton (created at import, one
    thread, mirrors production's lifetime) is left alone.
    """
    import makermodslab.jobs as _jobs

    created: list = []
    real_init = _jobs.JobRegistry.__init__

    def _tracking_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        real_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(_jobs.JobRegistry, "__init__", _tracking_init)
    yield
    for reg in created:
        try:
            reg.shutdown()
            for runner in list(getattr(reg, "_runners", {}).values()):
                stop_event = getattr(runner, "_stop_event", None)
                if stop_event is not None:
                    stop_event.set()
        except Exception:
            # Teardown must never fail a test that already passed.
            pass
    for reg in created:
        thread = getattr(reg, "_watchdog_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


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
