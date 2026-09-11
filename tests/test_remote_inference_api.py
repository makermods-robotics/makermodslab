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
and the SFU is switched on and off by patching `sfu_enabled` rather than by
starting a server (the launcher owns `livekit-server`; nothing in the test
suite may spawn one).

The SFU is the ONE transport since the combine with remote teleoperation:
there is no credential file to redirect any more. `[remote]` is optional and
CI installs only `.[test]`, so nothing here may depend on `livekit.portal`
being importable — `_extra_missing` is stubbed on every route test.
"""

from __future__ import annotations

import pytest

from makermodslab import remote_inference as ri
from makermodslab.schemas.sessions import (
    RemoteInferenceStatusResponse,
    RemoteInferenceTransportStatusResponse,
)


@pytest.fixture
def sfu_off(monkeypatch):
    """The Lab was started without `--sfu`: nothing to resolve, nothing probed.

    `_tailscale_ipv4` is stubbed so no test shells out."""
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: False)
    monkeypatch.setattr(ri.sfu, "external_ip_enabled", lambda *a, **k: False)
    monkeypatch.setattr(ri.sfu, "find_livekit_server", lambda *a, **k: "/usr/bin/livekit-server")
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: None)


@pytest.fixture
def sfu_on(monkeypatch, sfu_off):
    """Switch the bundled SFU on, with a fixed key pair and tailnet address.

    `_sfu_transport` is left REAL: the point of these tests is that the room,
    the url and BOTH tokens come from sfu.py, so only the two things that
    touch the machine (the key file, the tailscale CLI) are stubbed."""
    monkeypatch.setattr(ri.sfu, "sfu_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ri.sfu, "api_keys", lambda *a, **k: ("APIkey123", "s3cret"))
    monkeypatch.setattr(ri.sfu, "local_url", lambda *a, **k: "ws://100.64.0.7:7880")
    monkeypatch.setattr(ri.sfu, "default_room", lambda instance_id: "mml-abcdef012345")
    monkeypatch.setattr(ri.sfu, "mint_token", lambda **kw: (f"jwt.{kw['role']}.{kw['identity']}", 0))
    monkeypatch.setattr(ri.sfu, "external_ip_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: "100.64.0.7")
    monkeypatch.setenv(ri.sfu.ENV_KEY_FILE, "/tmp/livekit_keys.yaml")  # noqa: S108


@pytest.fixture
def no_probe(monkeypatch):
    """`_probe_room` must not be called; calling it fails the test loudly."""

    def _boom(*args, **kwargs):
        raise AssertionError("the room probe ran when it should not have")

    monkeypatch.setattr(ri, "_probe_room", _boom)


@pytest.fixture
def policy_in_room(monkeypatch):
    monkeypatch.setattr(
        ri, "_probe_room", lambda *a, **k: ri.RoomProbe(True, True, True, operator_present=True)
    )


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


def test_transport_reports_no_sfu_and_does_not_probe(
    client, tmp_lerobot_home, sfu_off, no_probe, monkeypatch
):
    """Started without `--sfu`: say so, name the remedy, and DO NOT probe.
    Asking an SFU we have no url for would either hang out the timeout or fail
    with a message about the wrong problem."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["extra_installed"] is True
    assert body["configured"] is False
    assert body["url"] == "" and body["room"] == ""
    assert body["source"] == "none"
    assert body["sfu_enabled"] is False
    # The whole sfu_* block is null/false when this process runs no SFU, so the
    # panel can render it from the one flag — and there is no token to hand out.
    assert body["sfu_url"] is None and body["sfu_modal_url"] is None
    assert body["sfu_external_ip"] is False
    assert body["policy_token"] is None
    # Null, not false: "we never asked" is a third state.
    assert body["endpoint_reachable"] is None
    assert body["operator_present"] is None
    assert body["error_code"] == "transport.not_configured"
    assert "--sfu" in body["message"]


# --- the Lab-owned SFU (makermodslab --sfu) ---------------------------------


def test_transport_under_the_sfu_mints_the_room_url_and_policy_token(
    client, tmp_lerobot_home, sfu_on, policy_in_room, monkeypatch
):
    """The whole transport comes from sfu.py in-process: the url, the room and
    the OPERATOR-role token the GPU side joins with. Nothing on disk but the
    key file is consulted."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["source"] == "sfu"
    assert body["sfu_enabled"] is True
    assert body["configured"] is True
    assert body["url"] == "ws://100.64.0.7:7880"
    assert body["sfu_url"] == "ws://100.64.0.7:7880"
    assert body["room"] == "mml-abcdef012345"
    # What a Modal container should dial: the tailnet address.
    assert body["sfu_modal_url"] == "ws://100.64.0.7:7880"
    assert body["sfu_external_ip"] is True
    # The GPU side's token: operator role, the exact identity the room probe
    # looks for on the other side.
    assert body["policy_token"] == "jwt.operator.policy"
    assert body["sfu_install_hint"] is None
    assert body["endpoint_reachable"] is True and body["operator_present"] is True
    assert body["error_code"] is None
    assert "--livekit-token" in body["message"]


def test_transport_returns_a_token_but_never_the_key_pair(
    client, tmp_lerobot_home, sfu_on, policy_in_room, monkeypatch
):
    """The SECRET signs every room token for the life of the install, and the
    key NAME is only useful beside it — neither belongs on a status route. The
    token is what `POST /sfu/token` would mint for the same identity, so this
    route exposes nothing that one does not."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

    raw = client.get("/api/v1/remote-inference/transport").text
    assert "s3cret" not in raw
    assert "APIkey123" not in raw
    assert "jwt.operator.policy" in raw


def test_transport_modal_url_is_null_without_tailscale(
    client, tmp_lerobot_home, sfu_on, policy_in_room, monkeypatch
):
    """No tailnet address means there is nothing honest to offer: a container
    has no route to loopback and none to a LAN address either."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_tailscale_ipv4", lambda: None)

    assert client.get("/api/v1/remote-inference/transport").json()["sfu_modal_url"] is None


def test_transport_offers_the_install_hint_when_the_binary_is_missing(
    client, tmp_lerobot_home, sfu_off, no_probe, monkeypatch
):
    """Its absence is how the panel knows `--sfu` is a flag the user can
    actually pass; present, it is the one line that makes it passable."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri.sfu, "find_livekit_server", lambda *a, **k: None)

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["sfu_install_hint"]


def test_transport_unreachable_sfu_names_the_flag(client, tmp_lerobot_home, sfu_on, monkeypatch):
    """The remedy is a process on this machine, never a file of credentials —
    so the hint says `--sfu`, not "check your internet connection"."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)
    monkeypatch.setattr(ri, "_probe_room", lambda *a, **k: ri.RoomProbe(False, False, False, False))

    body = client.get("/api/v1/remote-inference/transport").json()
    assert body["error_code"] == "transport.unreachable"
    assert "--sfu" in body["message"]


def test_transport_an_unreadable_key_file_is_reported_not_raised(
    client, tmp_lerobot_home, sfu_on, no_probe, monkeypatch
):
    """The launcher wrote that file before the app started, so this is a
    broken install — but the panel still has to show it rather than a 500."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

    def _boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(ri.sfu, "api_keys", _boom)

    resp = client.get("/api/v1/remote-inference/transport")
    assert resp.status_code == 200
    body = resp.json()
    assert body["error_code"] == "transport.not_configured"
    assert body["configured"] is False and body["policy_token"] is None
    assert "permission denied" in body["message"]


def test_transport_when_the_extra_is_missing_reports_it(
    client, tmp_lerobot_home, sfu_off, no_probe, monkeypatch
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
    assert body["error_code"] == "system.extra_missing"
    assert "PRIMARY checkout" in body["message"]
    assert body["endpoint_reachable"] is None and body["operator_present"] is None


def test_transport_payload_matches_its_response_model_exactly(
    client, tmp_lerobot_home, sfu_off, no_probe, monkeypatch
):
    """`response_model` silently FILTERS undeclared fields and MATERIALIZES
    declared-but-absent optionals as null, so a key added to the handler alone
    would never reach the UI and one added to the model alone would arrive as a
    lie. This route carries no exclusion mode, so equality is the assertion."""
    monkeypatch.setattr(ri, "_extra_missing", lambda: False)

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
