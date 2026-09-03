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
"""The two remote-inference routes, exercised idle.

Everything here is offline and touches no hardware: the room probe is
monkeypatched at its one seam (`_probe_room` — livekit-api is aiohttp-based, so
httpx.MockTransport does not apply and no new test dependency is warranted),
`DRTC_ENV_PATH` is redirected into tmp, and the SFU is switched on and off by
patching `sfu_enabled` rather than by starting a server (the launcher owns
`livekit-server`; nothing in the test suite may spawn one).

Redirecting `DRTC_ENV_PATH` is NOT something `tmp_lerobot_home` does — it
patches `utils.config`'s robot/calibration constants, while `remote_inference`
binds the DRTC paths by value at import time, so it is patched on the MODULE.

The `[drtc]` extra is optional and CI installs only `.[test]`, so nothing here
may depend on python-dotenv, aiohttp or livekit-api being importable: every one
of the module's extra-provided names is stubbed, including `_dotenv_values`
(with a three-line parser), so the provenance walk under test runs identically
with and without the extra.

The third route this file used to cover — `POST
/api/v1/remote-inference/clear-local-override` — retired in S3.6 with the shell
SFU scripts whose dotenv override it deleted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makermodslab import remote_inference as ri
from makermodslab.schemas.sessions import (
    RemoteInferenceStatusResponse,
    RemoteInferenceTransportStatusResponse,
)

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
    """Redirect the saved-credentials path, and default the SFU to OFF.

    Off is the fixture default because it is the branch with the interesting
    failure modes (missing credentials, an unreachable Cloud project); the SFU
    tests turn it on explicitly. Nothing here ever runs `livekit-server`, and
    `_tailscale_ipv4` is stubbed so no test shells out either."""
    saved_env = tmp_path / "livekit.env"
    monkeypatch.setattr(ri, "DRTC_ENV_PATH", str(saved_env))
    monkeypatch.setattr(ri, "_dotenv_values", _fake_dotenv_values)
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.setattr(ri.sfu, "external_ip_enabled", lambda *a, **k: False)
    monkeypatch.setattr(ri.sfu, "find_livekit_server", lambda *a, **k: "/usr/bin/livekit-server")
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: None)
    return {"saved_env": saved_env}


@pytest.fixture
def sfu_on(monkeypatch, drtc_paths):
    """Switch the bundled SFU on, with a fixed key pair and tailnet address.

    `_sfu_transport` is left REAL: the point of these tests is that the room,
    the loopback url and the token all come from sfu.py, so only the two things
    that touch the machine (the key file, the tailscale CLI) are stubbed."""
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ri.sfu, "api_keys", lambda *a, **k: ("APIkey123", "s3cret"))
    monkeypatch.setattr(ri.sfu, "local_url", lambda *a, **k: "ws://127.0.0.1:7880")
    monkeypatch.setattr(ri.sfu, "default_room", lambda instance_id: "mml-abcdef012345")
    monkeypatch.setattr(ri.sfu, "mint_token", lambda **kw: ("jwt.for." + kw["identity"], 0))
    monkeypatch.setattr(ri.sfu, "external_ip_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: "100.64.0.7")
    monkeypatch.setenv(ri.sfu.ENV_KEY_FILE, "/tmp/livekit_keys.yaml")  # noqa: S108


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
    """No SFU and no credentials: report it, and DO NOT probe. Asking an SFU we
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
    assert body["sfu_enabled"] is False
    # The whole sfu_* block is null/false when this process runs no SFU, so the
    # panel can render it from the one flag.
    assert body["sfu_url"] is None and body["sfu_modal_url"] is None
    assert body["sfu_key_id"] is None and body["sfu_key_file"] is None
    assert body["sfu_external_ip"] is False
    # Null, not false: "we never asked" is a third state.
    assert body["endpoint_reachable"] is None
    assert body["operator_present"] is None
    assert body["error_code"] == "transport.not_configured"


def test_transport_names_the_saved_file_as_the_source(client, tmp_lerobot_home, drtc_paths, monkeypatch):
    """`livekit.env` is the LiveKit Cloud fallback, and the only credential
    file left after S3.6."""
    drtc_paths["saved_env"].write_text("LIVEKIT_URL=wss://x.livekit.cloud\n")
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", lambda: dict(_GOOD_ENV))
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["source"] == "cloud"
    assert body["configured"] is True and body["missing_vars"] == []
    assert body["endpoint_reachable"] is True and body["operator_present"] is True
    assert body["error_code"] is None
    assert "livekit.env" in body["message"]


def test_transport_names_the_process_environment_when_it_is_what_won(
    client, tmp_lerobot_home, drtc_paths, monkeypatch
):
    """A different remedy from "cloud": telling an operator to edit a file that
    their shell is overriding is the worst of the three answers."""
    drtc_paths["saved_env"].write_text("LIVEKIT_URL=wss://from-file\n")
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", lambda: {**_GOOD_ENV, "LIVEKIT_URL": "wss://from-shell"})
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    assert client.get("/api/v1/remote-inference/transport").json()["source"] == "process_env"


# --- the Lab-owned SFU (makermodslab --sfu) ---------------------------------


def test_transport_under_the_sfu_reports_it_and_never_reads_livekit_env(
    client, tmp_lerobot_home, drtc_paths, sfu_on, monkeypatch
):
    """The SFU wins outright: url, room and credentials are minted in-process,
    and a stale `livekit.env` pointing somewhere else is simply not consulted."""
    drtc_paths["saved_env"].write_text("LIVEKIT_URL=wss://stale.livekit.cloud\nLIVEKIT_ROOM=stale\n")
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

    def _boom():
        raise AssertionError("livekit.env was read while the Lab's own SFU is running")

    monkeypatch.setattr(ri, "_read_env", _boom)
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["source"] == "sfu"
    assert body["sfu_enabled"] is True
    assert body["configured"] is True and body["missing_vars"] == []
    assert body["url"] == "ws://127.0.0.1:7880"
    assert body["sfu_url"] == "ws://127.0.0.1:7880"
    assert body["room"] == "mml-abcdef012345"
    # What a Modal container should dial: the tailnet address, never loopback.
    assert body["sfu_modal_url"] == "ws://100.64.0.7:7880"
    assert body["sfu_external_ip"] is True
    assert body["sfu_key_file"] == "/tmp/livekit_keys.yaml"  # noqa: S108
    assert body["sfu_install_hint"] is None
    assert body["error_code"] is None


def test_transport_returns_the_key_id_but_never_the_secret(
    client, tmp_lerobot_home, drtc_paths, sfu_on, monkeypatch
):
    """The key NAME is the `--livekit-api-key` half of the Modal line and it
    identifies rather than authorizes. The SECRET signs every room token, so a
    status endpoint that returned it would be a credential leak wearing a
    diagnostic hat — the panel names the file and a human reads it."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    raw = client.get("/api/v1/remote-inference/transport").text
    assert "APIkey123" in raw
    assert "s3cret" not in raw


def test_transport_modal_url_is_null_without_tailscale(
    client, tmp_lerobot_home, drtc_paths, sfu_on, monkeypatch
):
    """No tailnet address means there is nothing honest to offer: a container
    has no route to loopback and none to a LAN address either."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: None)
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )

    assert client.get("/api/v1/remote-inference/transport").json()["sfu_modal_url"] is None


def test_transport_offers_the_install_hint_when_the_binary_is_missing(
    client, tmp_lerobot_home, drtc_paths, no_probe, monkeypatch
):
    """Its absence is how the panel knows `--sfu` is a flag the user can
    actually pass; present, it is the one line that makes it passable."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", dict)
    monkeypatch.setattr(ri.sfu, "find_livekit_server", lambda *a, **k: None)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["sfu_install_hint"]


def test_transport_unreachable_sfu_says_so_rather_than_blaming_the_network(
    client, tmp_lerobot_home, drtc_paths, sfu_on, monkeypatch
):
    """The two transports have disjoint remedies — a process on this machine
    versus a file of credentials — so the hint must name which one it is."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_probe_room", lambda *a, **k: ri.RoomProbe(False, False, False, False))

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["error_code"] == "transport.unreachable"
    assert "--sfu" in body["message"]


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


def test_transport_payload_matches_its_response_model_exactly(
    client, tmp_lerobot_home, drtc_paths, no_probe, monkeypatch
):
    """`response_model` silently FILTERS undeclared fields and MATERIALIZES
    declared-but-absent optionals as null, so a key added to the handler alone
    would never reach the UI and one added to the model alone would arrive as a
    lie. This route carries no exclusion mode, so equality is the assertion."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_read_env", dict)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert set(body) == set(RemoteInferenceTransportStatusResponse.model_fields)
    assert set(body) == set(ri.handle_remote_inference_transport())


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
