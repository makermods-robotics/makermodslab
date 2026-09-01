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
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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

    Patches the module-level constants in `makermodslab.utils.config` so any code
    importing them through `from makermodslab.utils.config import LEADER_CONFIG_PATH`
    sees the redirected path. Also sets `HF_LEROBOT_HOME` env var for any
    consumer (e.g. `makermodslab.datasets._lerobot_cache_root`) reading it directly.
    """
    cache = tmp_path / "lerobot"
    cache.mkdir()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(cache))

    from makermodslab.utils import config as cfg

    teleop_dir = cache / "calibration" / "teleoperators" / "so101_leader"
    robot_dir = cache / "calibration" / "robots" / "so101_follower"
    leader_cfg_dir = cache / "configs" / "so_leader"
    follower_cfg_dir = cache / "configs" / "so_follower"
    # The Maker arm's calibration libraries. Separate directories from the
    # SO-101 pair, so they need their own redirect — without it any test that
    # touches a Maker calibration writes into the developer's real ~/.cache.
    maker_leader_cfg_dir = cache / "configs" / "rebot_102_leader"
    maker_follower_cfg_dir = cache / "configs" / "maker_follower"
    port_dir = cache / "ports"
    robots_dir = cache / "robots"
    for d in (
        teleop_dir,
        robot_dir,
        leader_cfg_dir,
        follower_cfg_dir,
        maker_leader_cfg_dir,
        maker_follower_cfg_dir,
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
    monkeypatch.setattr(cfg, "MAKER_FOLLOWER_CONFIG_PATH", str(maker_follower_cfg_dir))
    monkeypatch.setattr(cfg, "PORT_CONFIG_PATH", str(port_dir))
    monkeypatch.setattr(cfg, "LEADER_PORT_FILE", str(port_dir / "leader_port.txt"))
    monkeypatch.setattr(cfg, "FOLLOWER_PORT_FILE", str(port_dir / "follower_port.txt"))
    monkeypatch.setattr(cfg, "DISMISSED_HUB_JOBS_FILE", str(cache / "dismissed_hub_jobs.json"))
    # The pinned ("saved custom") and hidden repo-id lists. These leak the
    # HARDEST of the lot: every merged /datasets and /models listing folds them
    # in, so on a developer machine whose real saved_custom_models.json has
    # pinned repos, those repo ids appear in listings the test never seeded and
    # the listing assertions fail — on that machine only, invisibly in CI.
    # `_JsonRepoCollection` takes a `path_of` CALLABLE and re-invokes it on every
    # access precisely so a patched constant is honoured, and it holds no
    # in-memory copy, so redirecting the constant here is sufficient — there is
    # no cache to clear afterwards.
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_DATASETS_FILE", str(cache / "saved_custom_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_MODELS_FILE", str(cache / "saved_custom_models.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_DATASETS_FILE", str(cache / "hidden_datasets.json"))
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_MODELS_FILE", str(cache / "hidden_models.json"))
    monkeypatch.setattr(cfg, "EXCLUDED_EPISODES_FILE", str(cache / "excluded_episodes.json"))
    # BiSO staging root — without this, any bimanual staging test writes into the
    # developer's real ~/.cache dir.
    monkeypatch.setattr(cfg, "MAKERMODSLAB_BISO_STAGING_PATH", str(cache / "makermodslab_biso"))
    # Persisted node-registry peer list.
    monkeypatch.setattr(cfg, "NODES_FILE", str(cache / "nodes.json"))

    return cache


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
