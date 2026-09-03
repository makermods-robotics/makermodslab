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
"""The three remote-inference routes, exercised idle.

Everything here is offline and touches no hardware: the room probe is
monkeypatched at its one seam (`_probe_room` — livekit-api is aiohttp-based, so
httpx.MockTransport does not apply and no new test dependency is warranted),
and every `DRTC_*` path is redirected into tmp.

Redirecting those paths is NOT something `tmp_lerobot_home` does — it patches
`utils.config`'s robot/calibration constants, while `remote_inference` binds
`DRTC_LOCAL_ENV_PATH` / `DRTC_SFU_CONFIG_PATH` / `DRTC_ENV_PATH` by value at
import time. Patching them on the MODULE is what keeps the clear-override test
from unlinking the developer's real `livekit.local.env`, so the fixture below
does both.

The `[drtc]` extra is optional and CI installs only `.[test]`, so nothing here
may depend on python-dotenv, aiohttp or livekit-api being importable: every one
of the module's extra-provided names is stubbed, including `_dotenv_values`
(with a three-line parser), so the provenance walk under test runs identically
with and without the extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makermodslab import remote_inference as ri
from makermodslab.schemas.sessions import RemoteInferenceStatusResponse

_GOOD_ENV = {
    "LIVEKIT_URL": "wss://x.livekit.cloud",
    "LIVEKIT_ROOM": "portal-lerobot-inference",
    "LIVEKIT_API_KEY": "key",
    "LIVEKIT_API_SECRET": "secret",
}


def _fake_dotenv_values(path):
    """The two-key subset of `dotenv_values` the provenance walk needs."""
    values: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


@pytest.fixture
def drtc_paths(tmp_path, monkeypatch):
    """Redirect the three DRTC file paths and neutralize the working directory.

    `chdir` matters: `read_env`'s chain includes `./.env` and `./.env.local`, so
    a developer running pytest from a checkout that has either would otherwise
    see a different provenance answer than CI."""
    monkeypatch.chdir(tmp_path)
    local_env = tmp_path / "livekit.local.env"
    sfu_config = tmp_path / "livekit.local.yaml"
    saved_env = tmp_path / "livekit.env"
    monkeypatch.setattr(ri, "DRTC_LOCAL_ENV_PATH", str(local_env))
    monkeypatch.setattr(ri, "DRTC_SFU_CONFIG_PATH", str(sfu_config))
    monkeypatch.setattr(ri, "DRTC_ENV_PATH", str(saved_env))
    monkeypatch.setattr(ri, "_dotenv_values", _fake_dotenv_values)
    return {"local_env": local_env, "sfu_config": sfu_config, "saved_env": saved_env}


@pytest.fixture
def no_probe(monkeypatch):
    """`_probe_room` must not be called; calling it fails the test loudly."""

    def _boom(*args, **kwargs):
        raise AssertionError("the room probe ran when it should not have")

    monkeypatch.setattr(ri, "_probe_room", _boom)


@pytest.fixture(autouse=True)
def _idle_session(monkeypatch):
    """Guarantee the module reads idle, whatever an earlier test left behind."""
    monkeypatch.setattr(ri, "remote_inference_active", False)
    monkeypatch.setattr(ri, "_remote_proc", None)
    monkeypatch.setattr(ri, "_remote_meta", {})
    monkeypatch.setattr(ri, "_remote_started_at", None)
    monkeypatch.setattr(ri, "_last_result", None)
    monkeypatch.setattr(ri, "_startup_thread", None)
    monkeypatch.setattr(ri, "_transport", None)
    monkeypatch.setattr(ri, "_stats", None)


# --- GET /api/v1/remote-inference/transport ---------------------------------


def test_transport_reports_nothing_configured(client, tmp_lerobot_home, drtc_paths, no_probe, monkeypatch):
    """No credentials anywhere: report it, and DO NOT probe. Asking an SFU we
    have no url for would either hang out the timeout or fail with a message
    about the wrong problem."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", dict)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["extra_installed"] is True
    assert body["configured"] is False
    assert set(body["missing_vars"]) == {
        "LIVEKIT_URL",
        "LIVEKIT_ROOM",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    }
    assert body["url"] == "" and body["room"] == ""
    assert body["source"] == "none"
    assert body["sfu_config_exists"] is False
    assert body["local_env_exists"] is False
    assert body["local_env_path"] == str(drtc_paths["local_env"])
    # Null, not false: "we never asked" is a third state.
    assert body["endpoint_reachable"] is None
    assert body["operator_present"] is None
    assert body["error_code"] == "transport.not_configured"


def test_transport_reports_the_local_override_as_the_source(
    client, tmp_lerobot_home, drtc_paths, monkeypatch
):
    """`livekit.local.env` is the highest-precedence FILE, and saying so on the
    happy path is the point: a run that works against 127.0.0.1 today breaks
    silently the moment the script behind it is Ctrl-C'd."""
    drtc_paths["local_env"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    drtc_paths["sfu_config"].write_text("port: 7880\n")
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", lambda: {**_GOOD_ENV, "LIVEKIT_URL": "ws://127.0.0.1:7880"})
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["source"] == "local_override"
    assert body["local_env_exists"] is True
    assert body["sfu_config_exists"] is True
    assert body["configured"] is True and body["missing_vars"] == []
    assert body["endpoint_reachable"] is True and body["operator_present"] is True
    assert body["error_code"] is None
    assert "livekit.local.env" in body["message"]


def test_transport_when_the_extra_is_missing_reports_it(
    client, tmp_lerobot_home, drtc_paths, no_probe, monkeypatch
):
    """Reported as a 200, never raised — the panel's job is to say what to
    install, and the command must name the PRIMARY checkout (an editable
    install run from a worktree silently re-points every other session's
    makermodslab)."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: True)

    resp = client.get("/api/v1/remote-inference/transport")
    assert resp.status_code == 200
    body = resp.json()
    assert body["extra_installed"] is False
    assert body["error_code"] == "transport.extra_missing"
    assert "PRIMARY checkout" in body["message"]
    assert body["endpoint_reachable"] is None and body["operator_present"] is None
    # Still echoed, so the panel can name the file even with no extra.
    assert body["local_env_path"] == str(drtc_paths["local_env"])


# --- POST /api/v1/remote-inference/clear-local-override ----------------------


def test_clear_local_override_is_idempotent_when_absent(client, tmp_lerobot_home, drtc_paths):
    """The caller's intent ("stop overriding") is already satisfied — a 404
    would just make every client special-case the happy state."""
    resp = client.post("/api/v1/remote-inference/clear-local-override")
    assert resp.status_code == 200
    assert resp.json() == {
        "success": True,
        "removed": False,
        "path": str(drtc_paths["local_env"]),
    }


def test_clear_local_override_removes_the_file(client, tmp_lerobot_home, drtc_paths):
    """And leaves livekit.local.yaml alone: that file holds the key/secret the
    local SFU minted, so deleting it ROTATES those credentials — a different
    and far more destructive act than un-pointing this machine at the SFU."""
    drtc_paths["local_env"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    drtc_paths["sfu_config"].write_text("port: 7880\n")

    resp = client.post("/api/v1/remote-inference/clear-local-override")
    assert resp.status_code == 200
    assert resp.json()["removed"] is True
    assert not drtc_paths["local_env"].exists()
    assert drtc_paths["sfu_config"].exists()


# --- GET /api/v1/remote-inference-status ------------------------------------


def test_remote_inference_status_when_idle(client, tmp_lerobot_home):
    """Pollable unconditionally, like /inference-status.

    The key-set assertion is the one that earns its keep: `response_model`
    silently FILTERS undeclared fields, so an exclusion mode slipping onto this
    route (or a key added to the handler without the model) fails here rather
    than as a null that never reaches the UI."""
    resp = client.get("/api/v1/remote-inference-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["remote_inference_active"] is False
    assert body["phase"] is None
    assert body["stats"] is None
    assert body["transport"] is None
    assert body["elapsed_s"] == 0
    assert set(body) == set(RemoteInferenceStatusResponse.model_fields)
    # And the model still describes the handler's own dict exactly.
    assert set(body) == set(ri.handle_remote_inference_status())
