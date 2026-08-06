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

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import constants as hf_constants

import lerobot.utils.constants as lr_constants


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

    THE CACHE ROOT IS READ THROUGH THREE INDEPENDENT MECHANISMS and patching one
    does not patch the others — all three are covered here:

      * the ``HF_LEROBOT_HOME`` env var → `datasets._lerobot_cache_root()`, which
        re-reads it on every call;
      * ``lerobot.utils.constants.HF_LEROBOT_HOME`` → a module attribute computed
        at lerobot IMPORT time, so setting the env var afterwards changes
        nothing. `record.handle_delete_dataset` imports it inside the function
        body, so before this was patched here a delete test resolved against the
        developer's REAL ~/.cache/huggingface/lerobot — a `shutil.rmtree` target
        one careless test away from destroying real calibration data;
      * ``huggingface_hub.constants.HF_HUB_CACHE`` → the shared Hub cache, read
        at call time by `models._hub_cache_has_repo` (and anything else asking
        "is this repo already downloaded?"). Left alone, tests answer that
        question from whatever the developer happens to have downloaded, which
        is machine-dependent and silently different in CI.

    The assertion below is the load-bearing part: it verifies the redirect from
    the perspective of the code under test, so a future refactor that reintroduces
    an unpatched path fails loudly here instead of quietly writing to the real
    cache. (`tests/repro/conftest.py` keeps its own copy of this protection —
    those tests call `rmtree` handlers deliberately and must not depend on this
    fixture being used.)
    """
    cache = tmp_path / "lerobot"
    cache.mkdir()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(cache))
    monkeypatch.setattr(lr_constants, "HF_LEROBOT_HOME", cache)

    hub_cache = tmp_path / "hub"
    hub_cache.mkdir(exist_ok=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_cache))

    real_cache = Path("~/.cache/huggingface/lerobot").expanduser().resolve()
    for probe in (Path(lr_constants.HF_LEROBOT_HOME).resolve(), Path(hf_constants.HF_HUB_CACHE).resolve()):
        assert real_cache not in [probe, *probe.parents], (
            f"REFUSING TO RUN: {probe} is inside the real cache {real_cache}"
        )

    from makermodslab.utils import config as cfg

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
    # BiSO staging root — without this, any bimanual staging test writes into the
    # developer's real ~/.cache dir.
    monkeypatch.setattr(cfg, "MAKERMODSLAB_BISO_STAGING_PATH", str(cache / "makermodslab_biso"))

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
