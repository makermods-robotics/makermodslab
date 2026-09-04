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

"""The bundled LiveKit SFU (`makermodslab --sfu`): binary lookup, config
rendering, and the room-token broker.

LiveKit is a WebRTC selective-forwarding unit. Remote teleoperation and
remote inference ride it (via LiveKit Portal on the participants), so a
station has to run one somewhere every participant can reach — for us that
is this machine, bound wherever the API is bound (a tailnet interface for
the Modal/laptop case). Nothing in this module speaks WebRTC: the server is
the stock `livekit-server` Go binary, spawned and reaped by the launcher
(scripts/makermodslab.py), never by the app — uvicorn --reload restarts the
app process on every save in dev, and the SFU must outlive that.

Two design rules, both about the API secret:

- The secret lives in a 0600 file (utils/config.LIVEKIT_KEY_FILE), never in
  a command line or an environment variable. The child reads it through
  `--key-file`; the token route reads the same file, lazily, by the path the
  launcher exports.
- The token route is the ONLY signer. Every LiveKit join needs a JWT signed
  with the secret, so without a broker each participant (a laptop, a Modal
  container, a browser) would have to hold the secret itself. Tokens are
  short-lived and scoped to one room and one role.

The binary is resolved from PATH only (or MAKERMODSLAB_LIVEKIT_BIN). LiveKit
publishes no macOS release asset and no PyPI package, so a download-on-demand
would still need a per-OS fallback — the per-OS install hint IS the
fallback, and the launcher fails fast with it before anything else starts.
"""

from __future__ import annotations

import datetime
import functools
import ipaddress
import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from typing import Literal

from makermodslab.utils.config import LIVEKIT_KEY_FILE, parse_livekit_keys

# Signalling (HTTP + WebSocket, also the room API), ICE over TCP, and ONE
# muxed UDP port for media — livekit's own defaults for the first two. A
# single UDP port instead of the sample config's 50000-60000 range keeps a
# station's firewall rule to three lines; a robot room has a handful of
# participants, not a conference.
SFU_HTTP_PORT = 7880
SFU_TCP_PORT = 7881
SFU_UDP_PORT = 7882

# Launcher -> app handoff (must be in the environment before uvicorn imports
# makermodslab.server, like MAKERMODSLAB_NO_UI). KEY_FILE is the path the
# route signs with; PORT is the signalling port the token's URL names; URL,
# when set, overrides the derived ws://<request host>:<port> outright (an
# external SFU — LiveKit Cloud, a VPS — reached by a name the station can't
# infer from the request).
ENV_KEY_FILE = "MAKERMODSLAB_SFU_KEY_FILE"
ENV_PORT = "MAKERMODSLAB_SFU_PORT"
# The host THIS process can reach its own SFU on (the bind host, or loopback
# for the wildcard) — for in-process participants (the hosting worker), which
# have no request to derive a host from.
ENV_HOST = "MAKERMODSLAB_SFU_HOST"
ENV_URL = "MAKERMODSLAB_SFU_URL"
ENV_BIN = "MAKERMODSLAB_LIVEKIT_BIN"

BINARY_NAME = "livekit-server"

Role = Literal["robot", "operator", "viewer"]
ROLES: tuple[Role, ...] = ("robot", "operator", "viewer")

# Token lifetime bounds (seconds). An hour covers a teleop/inference session
# with a reconnect; 12h is the ceiling so a leaked token can't become a
# standing credential. Portal's own examples use 6h.
DEFAULT_TTL_SECONDS = 3600
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 12 * 3600


def find_livekit_server(
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """Path to the livekit-server binary, or None.

    MAKERMODSLAB_LIVEKIT_BIN wins when set (and must point at an existing
    file — a typo'd override must not silently fall through to PATH and run
    a different build than the user asked for); otherwise PATH via `which`
    (which handles the `.exe` suffix on Windows).
    """
    env = os.environ if env is None else env
    override = env.get(ENV_BIN)
    if override:
        return override if os.path.isfile(override) else None
    return which(BINARY_NAME)


def install_hint(system: str) -> str:
    """One line telling the user how to get livekit-server on `system`
    (platform.system(): "Darwin" / "Linux" / "Windows").

    LiveKit ships no macOS release asset, so Homebrew is the only supported
    route there; Linux has their install script (or the GitHub release
    tarball); Windows only the release zip.
    """
    if system == "Darwin":
        return "Install it with Homebrew: `brew install livekit`"
    if system == "Linux":
        return (
            "Install it with LiveKit's script: `curl -sSL https://get.livekit.io | bash` "
            "(or unpack the linux tarball from https://github.com/livekit/livekit/releases "
            "somewhere on your PATH)"
        )
    if system == "Windows":
        return (
            "Download the windows zip from https://github.com/livekit/livekit/releases and put "
            "livekit-server.exe on your PATH"
        )
    return "Get livekit-server from https://github.com/livekit/livekit/releases and put it on your PATH"


def render_config(
    *,
    bind_host: str,
    key_file: str,
    http_port: int = SFU_HTTP_PORT,
    tcp_port: int = SFU_TCP_PORT,
    udp_port: int = SFU_UDP_PORT,
) -> str:
    """The livekit-server YAML for one run.

    `bind_host` is the already-resolved address the API binds (127.0.0.1,
    0.0.0.0, or one interface's IP). The signalling listener follows it. For
    a SPECIFIC address — the tailnet case — `rtc.node_ip` is pinned to it
    too, so ICE candidates advertise the address peers actually reach us on
    rather than whatever LAN IP livekit would auto-pick; for loopback and
    the wildcard, livekit's own auto-detection is left alone.
    `use_external_ip` stays off: the STUN self-probe it triggers stalls a
    station with no internet and is only for cloud NAT.
    Rendered by hand rather than a YAML library so the output is exactly the
    handful of keys we mean to set and nothing else.
    """
    lines = [
        "# Generated by makermodslab --sfu on every start; edits are overwritten.",
        f"port: {http_port}",
        f"key_file: {key_file}",
        "rtc:",
        f"  tcp_port: {tcp_port}",
        f"  udp_port: {udp_port}",
        "  use_external_ip: false",
    ]
    if _is_specific_address(bind_host):
        lines.append(f"  node_ip: {bind_host}")
    if bind_host != "0.0.0.0":  # noqa: S104  # nosec B104 — the wildcard is livekit's default listener, so it needs no bind line
        lines.append("bind_addresses:")
        lines.append(f"  - {bind_host}")
    lines += [
        "room:",
        "  auto_create: true",
        "logging:",
        "  level: warn",
        "  pion_level: error",
    ]
    return "\n".join(lines) + "\n"


def _is_specific_address(host: str) -> bool:
    """True for one concrete, non-loopback, non-wildcard IP."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not (addr.is_loopback or addr.is_unspecified)


def public_host(bind_host: str) -> str:
    """The hostname the launcher logs for the SFU (what a local user would
    type): `localhost` for loopback, the address itself otherwise."""
    return "localhost" if bind_host == "127.0.0.1" else bind_host


# --- App side: settings + token broker ---------------------------------------


def sfu_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when the launcher exported a key file (this process was started
    with --sfu, or an operator wired an external SFU the same way)."""
    env = os.environ if env is None else env
    return bool(env.get(ENV_KEY_FILE))


def sfu_url(request_host: str, env: Mapping[str, str] | None = None) -> str:
    """The signalling URL a participant should connect to.

    Derived from the host the CALLER reached the API on — the one address we
    know is routable from where they sit (a LAN IP, a tailnet IP, localhost)
    — plus the signalling port, unless MAKERMODSLAB_SFU_URL overrides it.
    """
    env = os.environ if env is None else env
    override = env.get(ENV_URL)
    if override:
        return override
    port = env.get(ENV_PORT) or str(SFU_HTTP_PORT)
    host = f"[{request_host}]" if ":" in request_host else request_host
    return f"ws://{host}:{port}"


def local_url(env: Mapping[str, str] | None = None) -> str:
    """The signalling URL for a participant running INSIDE this process
    (remote_host's worker): MAKERMODSLAB_SFU_URL if set, else the bind host
    the launcher exported (loopback for the wildcard bind) plus the port."""
    env = os.environ if env is None else env
    override = env.get(ENV_URL)
    if override:
        return override
    host = env.get(ENV_HOST) or "127.0.0.1"
    port = env.get(ENV_PORT) or str(SFU_HTTP_PORT)
    return f"ws://{host}:{port}"


@functools.lru_cache(maxsize=4)
def _keys_from_file(path: str) -> tuple[str, str]:
    with open(path) as f:
        keys = parse_livekit_keys(f.read())
    if not keys:
        raise RuntimeError(f"no `key: secret` pair in {path}")
    return next(iter(keys.items()))


def api_keys(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(key, secret) from the file the launcher exported. Cached per path —
    the file is written once per install and only rotates with a restart."""
    env = os.environ if env is None else env
    return _keys_from_file(env.get(ENV_KEY_FILE) or LIVEKIT_KEY_FILE)


def default_identity(role: str) -> str:
    """`<role>-<8 hex>`: unique within a room without the caller choosing."""
    return f"{role}-{secrets.token_hex(4)}"


def default_room(instance_id: str) -> str:
    """One room per station by default: `mml-<instance id prefix>`. A station
    hosts one robot, and Portal wants exactly one Robot per room."""
    return f"mml-{instance_id[:12]}"


def mint_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    role: Role,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: datetime.datetime | None = None,
    max_participants: int | None = None,
) -> tuple[str, int]:
    """Sign a LiveKit room token for `identity`. Returns (jwt, expires_at
    epoch seconds).

    Grants by role: `robot` and `operator` publish and subscribe (an operator
    publishes actions over data streams — Portal needs the publish grant for
    them too); `viewer` subscribes only. Every role gets
    `can_update_own_metadata`: Portal sets the `lk.portal.role` participant
    attribute at connect and fails without it. The room config pins playout
    delay to 0–1 ms, LiveKit's own teleop recommendation (smoothness traded
    for latency); the room is created on first join with it. `max_participants`
    (the STATION's own token sets it, since the robot joins first and the
    room is created from its config) makes the SFU itself enforce the single
    operator seat: robot + one operator = 2.
    """
    from livekit import api
    from livekit.protocol.room import RoomConfiguration

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    can_publish = role != "viewer"
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=can_publish,
        can_publish_data=can_publish,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    room_config = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    if max_participants is not None:
        room_config.max_participants = max_participants
    ttl = datetime.timedelta(seconds=ttl_seconds)
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(room_config)
        .with_ttl(ttl)
    )
    issued = now or datetime.datetime.now(datetime.UTC)
    return token.to_jwt(), int((issued + ttl).timestamp())
