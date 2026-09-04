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
"""Tests for makermodslab.sfu — the pure helpers (binary lookup, config
rendering, key persistence, token grants) and the /api/v1/sfu/token route +
health capability. Spawning livekit-server is launcher glue and is left to
the manual smoke test (tests/ policy: no subprocess happy paths)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import jwt
import pytest

from makermodslab import sfu
from makermodslab.utils.config import load_or_create_livekit_keys, parse_livekit_keys

# --- binary lookup -----------------------------------------------------------


def test_find_livekit_server_uses_path_by_default() -> None:
    assert sfu.find_livekit_server(env={}, which=lambda name: f"/usr/local/bin/{name}") == (
        "/usr/local/bin/livekit-server"
    )
    assert sfu.find_livekit_server(env={}, which=lambda name: None) is None


def test_find_livekit_server_env_override_wins_when_it_is_a_file(tmp_path: Path) -> None:
    binary = tmp_path / "lk"
    binary.write_text("")
    env = {sfu.ENV_BIN: str(binary)}
    assert sfu.find_livekit_server(env=env, which=lambda name: "/elsewhere/livekit-server") == str(binary)


def test_find_livekit_server_bad_override_does_not_fall_through_to_path(tmp_path: Path) -> None:
    """A typo'd MAKERMODSLAB_LIVEKIT_BIN must fail, not silently run whatever
    build happens to be on PATH."""
    env = {sfu.ENV_BIN: str(tmp_path / "missing")}
    assert sfu.find_livekit_server(env=env, which=lambda name: "/elsewhere/livekit-server") is None


@pytest.mark.parametrize(
    ("system", "needle"),
    [
        ("Darwin", "brew install livekit"),
        ("Linux", "get.livekit.io"),
        ("Windows", "livekit-server.exe"),
        ("FreeBSD", "github.com/livekit/livekit/releases"),
    ],
)
def test_install_hint_is_per_os(system: str, needle: str) -> None:
    assert needle in sfu.install_hint(system)


# --- config rendering --------------------------------------------------------


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_render_config_loopback_binds_loopback_and_leaves_node_ip_auto() -> None:
    text = sfu.render_config(bind_host="127.0.0.1", key_file="/k.yaml")
    lines = _lines(text)
    assert "port: 7880" in lines
    assert "key_file: /k.yaml" in lines
    assert "  tcp_port: 7881" in lines
    assert "  udp_port: 7882" in lines
    assert "  use_external_ip: false" in lines
    assert "bind_addresses:" in lines
    assert "  - 127.0.0.1" in lines
    assert not any(line.strip().startswith("node_ip") for line in lines)


def test_render_config_wildcard_has_no_bind_line_and_no_node_ip() -> None:
    lines = _lines(sfu.render_config(bind_host="0.0.0.0", key_file="/k.yaml"))  # noqa: S104
    assert "bind_addresses:" not in lines
    assert not any("node_ip" in line for line in lines)


def test_render_config_specific_address_pins_node_ip() -> None:
    """The tailnet case: ICE candidates must advertise the address peers
    reach us on, not whatever LAN IP livekit would auto-pick."""
    lines = _lines(sfu.render_config(bind_host="100.64.0.7", key_file="/k.yaml"))
    assert "  node_ip: 100.64.0.7" in lines
    assert "  - 100.64.0.7" in lines


def test_render_config_specific_address_also_binds_loopback() -> None:
    """The launcher's readiness probe and the robot child both dial
    127.0.0.1; binding the tailnet address ALONE made the launcher kill a
    healthy server after 15 s ("never came up")."""
    lines = _lines(sfu.render_config(bind_host="100.64.0.7", key_file="/k.yaml"))
    bind_block = lines[lines.index("bind_addresses:") + 1 :]
    assert bind_block[:2] == ["  - 100.64.0.7", "  - 127.0.0.1"]
    # Loopback and wildcard binds gain nothing and get no second line.
    loop = _lines(sfu.render_config(bind_host="127.0.0.1", key_file="/k.yaml"))
    assert loop.count("  - 127.0.0.1") == 1


def test_render_config_ports_are_overridable() -> None:
    lines = _lines(
        sfu.render_config(bind_host="127.0.0.1", key_file="/k", http_port=1, tcp_port=2, udp_port=3)
    )
    assert "port: 1" in lines
    assert "  tcp_port: 2" in lines
    assert "  udp_port: 3" in lines


def test_render_config_external_ip_turns_the_stun_probe_on_and_drops_the_pin() -> None:
    """`--sfu-external-ip`. A Modal container reaches the signalling URL over
    the tailnet but has to HOLE-PUNCH for media, and it has no route to a
    tailnet address — so the only candidate it can punch to is the public IP
    livekit discovers with STUN. The `node_ip` pin has to go with it: pinning
    the tailnet address is exactly what makes the discovered candidate
    unreachable, so the two are mutually exclusive rather than additive."""
    lines = _lines(sfu.render_config(bind_host="100.64.0.7", key_file="/k.yaml", external_ip=True))
    assert "  use_external_ip: true" in lines
    assert not any(line.strip().startswith("node_ip") for line in lines)
    # The listener still follows the bind: signalling stays on the tailnet.
    assert "  - 100.64.0.7" in lines


def test_render_config_external_ip_skips_hairpin_validation_and_keeps_lan_candidate() -> None:
    """Bench finding (2026-09-03): livekit's post-STUN self-check hairpins
    through the router and times out on a home NAT, after which it drops the
    internal candidates the robot child needs. With validation skipped and
    the internal IP advertised, livekit logged
    `using external IPs ["73.x/192.168.x"]` — one public, one LAN. The
    tailnet is excluded from candidate gathering by CIDR (media never rides
    it, and its probe collides with the LAN socket's on :7882)."""
    text = sfu.render_config(bind_host="100.64.0.7", key_file="/k.yaml", external_ip=True)
    lines = _lines(text)
    assert "  skip_external_ip_validation: true" in lines
    assert "  advertise_internal_ip: true" in lines
    assert "      - 100.64.0.0/10" in lines
    assert "      - fd7a:115c:a1e0::/48" in lines
    # rtc.ips.excludes must sit INSIDE the rtc block, i.e. before the
    # top-level bind_addresses key.
    assert text.index("    excludes:") < text.index("bind_addresses:")
    # None of it leaks into the default (Cloud-free loopback) config.
    off = _lines(sfu.render_config(bind_host="127.0.0.1", key_file="/k.yaml"))
    assert not any("skip_external_ip_validation" in line or "advertise_internal_ip" in line for line in off)


def test_render_config_external_ip_defaults_off() -> None:
    """Off is right for a LAN-only station: the STUN self-probe stalls a
    machine with no internet, and it is only for cloud NAT."""
    assert "  use_external_ip: false" in _lines(sfu.render_config(bind_host="100.64.0.7", key_file="/k.yaml"))


def test_external_ip_enabled_reads_the_launcher_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app REPORTS the flag (on the transport panel) and never acts on it —
    the config it shapes was rendered before the app started."""
    monkeypatch.delenv(sfu.ENV_EXTERNAL_IP, raising=False)
    assert sfu.external_ip_enabled() is False
    monkeypatch.setenv(sfu.ENV_EXTERNAL_IP, "0")
    assert sfu.external_ip_enabled() is False
    monkeypatch.setenv(sfu.ENV_EXTERNAL_IP, "1")
    assert sfu.external_ip_enabled() is True


def test_local_url_is_loopback_and_honours_the_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child this process spawns has no request to derive a host from, and
    loopback is the one address it is always reachable on whatever the SFU
    bound. An external SFU still stays external."""
    monkeypatch.delenv(sfu.ENV_URL, raising=False)
    monkeypatch.setenv(sfu.ENV_PORT, "7880")
    assert sfu.local_url() == "ws://127.0.0.1:7880"
    monkeypatch.setenv(sfu.ENV_URL, "wss://sfu.example.com")
    assert sfu.local_url() == "wss://sfu.example.com"


def test_public_host_names_localhost_for_loopback() -> None:
    assert sfu.public_host("127.0.0.1") == "localhost"
    assert sfu.public_host("100.64.0.7") == "100.64.0.7"


# --- key persistence ---------------------------------------------------------


def test_parse_livekit_keys_reads_key_colon_secret_lines() -> None:
    text = "# comment\n\nmml_ab: s3cret\nbroken line\n: nosecret\nnokey:\n"
    assert parse_livekit_keys(text) == {"mml_ab": "s3cret"}


def test_load_or_create_livekit_keys_mints_once_with_0600(tmp_path: Path) -> None:
    path = str(tmp_path / "nested" / "livekit_keys.yaml")
    key, secret = load_or_create_livekit_keys(path)
    assert key.startswith("mml_") and len(secret) >= 32
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert parse_livekit_keys(Path(path).read_text()) == {key: secret}
    # Idempotent: a second read returns the SAME pair (no rotation).
    assert load_or_create_livekit_keys(path) == (key, secret)


def test_load_or_create_livekit_keys_replaces_a_keyless_file(tmp_path: Path) -> None:
    path = tmp_path / "livekit_keys.yaml"
    path.write_text("# hand-edited to nothing\n")
    key, secret = load_or_create_livekit_keys(str(path))
    assert parse_livekit_keys(path.read_text()) == {key: secret}


# --- app-side settings -------------------------------------------------------


def test_sfu_enabled_follows_the_key_file_env() -> None:
    assert sfu.sfu_enabled(env={}) is False
    assert sfu.sfu_enabled(env={sfu.ENV_KEY_FILE: "/k.yaml"}) is True


def test_sfu_url_is_derived_from_the_request_host_unless_overridden() -> None:
    assert sfu.sfu_url("100.64.0.7", env={}) == "ws://100.64.0.7:7880"
    assert sfu.sfu_url("localhost", env={sfu.ENV_PORT: "7999"}) == "ws://localhost:7999"
    assert sfu.sfu_url("fd7a::1", env={}) == "ws://[fd7a::1]:7880"
    assert sfu.sfu_url("ignored", env={sfu.ENV_URL: "wss://x.livekit.cloud"}) == "wss://x.livekit.cloud"


def test_default_room_and_identity_shapes() -> None:
    assert sfu.default_room("0123456789abcdef0123456789abcdef") == "mml-0123456789ab"
    ident = sfu.default_identity("operator")
    assert ident.startswith("operator-") and len(ident) == len("operator-") + 8


# --- token grants ------------------------------------------------------------


def _claims(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})


@pytest.mark.parametrize(
    ("role", "can_publish"),
    [("robot", True), ("operator", True), ("viewer", False)],
)
def test_mint_token_grants_by_role(role: str, can_publish: bool) -> None:
    token, expires_at = sfu.mint_token(
        api_key="k1", api_secret="s" * 40, identity="who", room="rm", role=role, ttl_seconds=600
    )
    claims = _claims(token)
    assert claims["iss"] == "k1"
    assert claims["sub"] == "who"
    assert claims["exp"] == expires_at
    video = claims["video"]
    assert video["roomJoin"] is True
    assert video["room"] == "rm"
    assert video["canSubscribe"] is True
    assert video["canPublish"] is can_publish
    assert video["canPublishData"] is can_publish
    # Portal sets the lk.portal.role attribute at connect; it fails without this.
    assert video["canUpdateOwnMetadata"] is True
    # Teleop playout: 0–1 ms (min is 0 → omitted by proto3 JSON, max present).
    assert claims["roomConfig"]["name"] == "rm"
    assert claims["roomConfig"]["maxPlayoutDelay"] == 1


def test_mint_token_is_verifiable_with_the_secret() -> None:
    token, _ = sfu.mint_token(api_key="k1", api_secret="s" * 40, identity="a", room="r", role="viewer")
    assert jwt.decode(token, "s" * 40, algorithms=["HS256"], issuer="k1")["sub"] == "a"


def test_mint_token_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        sfu.mint_token(api_key="k", api_secret="s" * 40, identity="a", room="r", role="admin")  # type: ignore[arg-type]


# --- route + health ----------------------------------------------------------


@pytest.fixture
def sfu_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Simulate the launcher's --sfu handoff: a key file + the env the app reads."""
    path = str(tmp_path / "livekit_keys.yaml")
    key, secret = load_or_create_livekit_keys(path)
    monkeypatch.setenv(sfu.ENV_KEY_FILE, path)
    monkeypatch.setenv(sfu.ENV_PORT, "7880")
    monkeypatch.delenv(sfu.ENV_URL, raising=False)
    sfu._keys_from_file.cache_clear()
    return key, secret


def test_token_route_is_409_sfu_disabled_without_the_flag(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sfu.ENV_KEY_FILE, raising=False)
    r = client.post("/api/v1/sfu/token", json={})
    assert r.status_code == 409
    assert r.json()["code"] == "sfu.disabled"


def test_token_route_fills_defaults_and_signs_with_the_install_key(client, sfu_env) -> None:
    key, secret = sfu_env
    r = client.post("/api/v1/sfu/token", json={}, headers={"host": "100.64.0.7:8000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"] == "ws://100.64.0.7:7880"
    assert body["role"] == "operator"
    assert body["identity"].startswith("operator-")
    assert body["room"].startswith("mml-")
    claims = jwt.decode(body["token"], secret, algorithms=["HS256"], issuer=key)
    assert claims["sub"] == body["identity"]
    assert claims["video"]["room"] == body["room"]
    assert claims["exp"] == body["expires_at"]


def test_token_route_honours_explicit_scope(client, sfu_env) -> None:
    r = client.post(
        "/api/v1/sfu/token",
        json={"identity": "jetson-robot", "room": "lab-1", "role": "robot", "ttl_seconds": 120},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["identity"], body["room"], body["role"]) == ("jetson-robot", "lab-1", "robot")
    claims = jwt.decode(body["token"], options={"verify_signature": False})
    assert claims["video"]["canPublish"] is True
    assert claims["exp"] - claims["nbf"] == 120


@pytest.mark.parametrize(
    "body",
    [
        {"identity": "has space"},
        {"room": "-leading-dash"},
        {"role": "admin"},
        {"ttl_seconds": 5},
        {"ttl_seconds": 10**6},
    ],
)
def test_token_route_rejects_malformed_scope_with_a_coded_422(client, sfu_env, body: dict) -> None:
    r = client.post("/api/v1/sfu/token", json=body)
    assert r.status_code == 422
    assert r.json()["code"] == "request.validation"


def test_health_reports_sfu_url_only_when_enabled(
    client, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(sfu.ENV_KEY_FILE, raising=False)
    assert "sfu" not in client.get("/api/v1/health").json()["capabilities"]

    monkeypatch.setenv(sfu.ENV_KEY_FILE, str(tmp_path / "k.yaml"))
    monkeypatch.delenv(sfu.ENV_URL, raising=False)
    monkeypatch.setenv(sfu.ENV_PORT, "7880")
    caps = client.get("/api/v1/health", headers={"host": "192.168.1.20:8000"}).json()["capabilities"]
    assert caps["sfu"] == {"url": "ws://192.168.1.20:7880"}
