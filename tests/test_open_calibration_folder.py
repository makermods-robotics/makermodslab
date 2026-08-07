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
"""Tests for the open-calibration-folder feature — the cross-platform
open_folder_in_file_browser helper and the POST /open-calibration-folder
endpoint.

The OS file-browser launch is ALWAYS mocked: subprocess.Popen and os.startfile
are monkeypatched so no real Finder/Explorer window is ever spawned. We only
assert the correct platform command is invoked with the resolved path, that the
dir is created if missing, and that an invalid device_type is rejected."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from makermodslab.utils import config as cfg, system

# --- open_folder_in_file_browser helper --------------------------------------


def test_open_folder_macos_uses_open(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(system.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(system.subprocess, "Popen", lambda args, *a, **k: calls.append(args))

    target = str(tmp_path / "so_leader")
    system.open_folder_in_file_browser(target)

    assert calls == [["open", target]]


def test_open_folder_linux_uses_xdg_open(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(system.platform, "system", lambda: "Linux")
    monkeypatch.setattr(system.subprocess, "Popen", lambda args, *a, **k: calls.append(args))

    target = str(tmp_path / "so_follower")
    system.open_folder_in_file_browser(target)

    assert calls == [["xdg-open", target]]


def test_open_folder_windows_uses_startfile(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(system.platform, "system", lambda: "Windows")
    # os.startfile only exists on Windows; add it for the mock so the helper can
    # resolve the attribute regardless of the test host OS.
    monkeypatch.setattr(system.os, "startfile", lambda path: calls.append(path), raising=False)
    # Guard: Popen must NOT be used on the Windows path.
    monkeypatch.setattr(
        system.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Popen must not be called on Windows"),
    )

    target = str(tmp_path / "so_leader")
    system.open_folder_in_file_browser(target)

    assert calls == [target]


def test_open_folder_creates_dir_if_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(system.subprocess, "Popen", lambda *a, **k: None)

    target = tmp_path / "does_not_exist_yet"
    assert not target.exists()
    system.open_folder_in_file_browser(str(target))
    assert target.is_dir()


def test_open_folder_unsupported_platform_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(
        system.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Popen must not run on an unsupported platform"),
    )
    with pytest.raises(OSError):
        system.open_folder_in_file_browser(str(tmp_path / "nope"))


# --- POST /open-calibration-folder endpoint ----------------------------------


def _patch_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Mock the server's open helper to record the path and open nothing."""
    import makermodslab.server as server_mod

    opened: list[str] = []
    monkeypatch.setattr(server_mod, "open_folder_in_file_browser", lambda path: opened.append(path))
    return opened


def test_endpoint_teleop_opens_leader_dir(
    client: TestClient, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _patch_open(monkeypatch)
    res = client.post("/open-calibration-folder", json={"device_type": "teleop"})
    assert res.status_code == 200
    body = res.json()
    assert body["opened"] is True
    assert body["path"] == cfg.LEADER_CONFIG_PATH
    assert opened == [cfg.LEADER_CONFIG_PATH]


def test_endpoint_robot_opens_follower_dir(
    client: TestClient, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _patch_open(monkeypatch)
    res = client.post("/open-calibration-folder", json={"device_type": "robot"})
    assert res.status_code == 200
    body = res.json()
    assert body["opened"] is True
    assert body["path"] == cfg.FOLLOWER_CONFIG_PATH
    assert opened == [cfg.FOLLOWER_CONFIG_PATH]


def test_endpoint_rejects_invalid_device_type(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _patch_open(monkeypatch)
    res = client.post("/open-calibration-folder", json={"device_type": "leader"})
    assert res.status_code == 400
    assert res.json()["opened"] is False
    # Nothing was opened.
    assert opened == []


def test_endpoint_creates_dir_if_missing(
    client: TestClient, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point the leader dir at a path that does not exist yet, and let the REAL
    # helper run with only the actual OS launch mocked — so we exercise the
    # mkdir-parents behavior end to end without opening a window.
    missing = tmp_lerobot_home / "fresh" / "so_leader"
    monkeypatch.setattr(cfg, "LEADER_CONFIG_PATH", str(missing))
    monkeypatch.setattr(system.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(system.subprocess, "Popen", lambda *a, **k: None)

    assert not missing.exists()
    res = client.post("/open-calibration-folder", json={"device_type": "teleop"})
    assert res.status_code == 200
    assert res.json()["path"] == str(missing)
    assert os.path.isdir(missing)


def test_endpoint_reports_error_on_spawn_failure(
    client: TestClient, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import makermodslab.server as server_mod

    def _boom(path: str) -> None:
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(server_mod, "open_folder_in_file_browser", _boom)
    res = client.post("/open-calibration-folder", json={"device_type": "robot"})
    assert res.status_code == 500
    body = res.json()
    assert body["opened"] is False
    assert "spawn failed" in body["message"]
