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

"""Discovery sources for the node registry (nodes.NodeSource implementations).

TailscaleSource proposes peers from `tailscale status --json`: every ONLINE
tailnet peer's IPv4 (the 100.64.0.0/10 CGNAT range Tailscale assigns) at this
app's backend port. Candidates only — most tailnet machines are not MakerMods
Lab nodes, and the registry's verify handshake is what separates the ones
that are (its short timeout and non-node rejection make the wrong guesses
cheap). Opt-in via the launcher's --discover-tailscale flag.

Every failure mode — no tailscale binary, daemon stopped or logged out,
malformed output — is an EMPTY discovery logged once at INFO per outage,
never an exception: the tailnet being absent must not degrade the server.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable

from .nodes import DiscoveredPeer

logger = logging.getLogger(__name__)

# This app's port, on every node — the registry only ever talks MakerMods Lab
# to MakerMods Lab. Mirrors BACKEND_PORT in scripts/makermodslab.py, which is
# not imported here because that launcher module configures logging at import.
DEFAULT_BACKEND_PORT = 8000

# `tailscale status --json` is a local unix-socket query; a second is
# generous, five means something is wrong with the daemon.
STATUS_TIMEOUT_S = 5.0

# Tailscale hands every tailnet device an IPv4 in the CGNAT range.
_TAILNET_IPV4 = ipaddress.ip_network("100.64.0.0/10")


# The macOS GUI app (App Store or standalone) ships its CLI inside the app
# bundle and installs NO PATH symlink — the number-one reason discovery "works
# on Ubuntu but not macOS". Homebrew's locations are covered by shutil.which
# when brew's bin is on PATH, and listed here for GUI-launched processes whose
# PATH doesn't include it.
_MACOS_APP_BUNDLE_CLI = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
_MACOS_FALLBACK_CLIS = (
    _MACOS_APP_BUNDLE_CLI,
    "/opt/homebrew/bin/tailscale",
    "/usr/local/bin/tailscale",
)


def _resolve_tailscale_binary() -> str | None:
    """The tailscale CLI to run: PATH first, then the macOS fallbacks."""
    found = shutil.which("tailscale")
    if found:
        return found
    if sys.platform == "darwin":
        for candidate in _MACOS_FALLBACK_CLIS:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _run_tailscale_status() -> str:
    """The default runner: the real CLI's stdout (raises on any failure)."""
    binary = _resolve_tailscale_binary()
    if binary is None:
        searched = "PATH" + (", " + ", ".join(_MACOS_FALLBACK_CLIS) if sys.platform == "darwin" else "")
        raise FileNotFoundError(f"tailscale CLI not found (searched: {searched})")
    return subprocess.run(
        [binary, "status", "--json"],
        capture_output=True,
        text=True,
        timeout=STATUS_TIMEOUT_S,
        check=True,
    ).stdout


class TailscaleSource:
    """nodes.NodeSource backed by the tailscale CLI (runner injectable so
    tests feed captured/broken output without a tailnet)."""

    source_id = "tailscale"

    def __init__(
        self,
        *,
        port: int = DEFAULT_BACKEND_PORT,
        runner: Callable[[], str] = _run_tailscale_status,
    ) -> None:
        self._port = port
        self._runner = runner
        self._outage_logged = False

    def discover(self) -> list[DiscoveredPeer]:
        """Current candidates: each online peer's tailnet IPv4 at our port.

        Any failure is an empty discovery; the first failure of an outage logs
        at WARNING (--discover-tailscale is an explicit opt-in, so its silent
        absence is exactly what a user debugs), and a successful run re-arms
        that log for the next outage.
        """
        try:
            doc = json.loads(self._runner())
            state = doc.get("BackendState") if isinstance(doc, dict) else None
            if state != "Running":
                raise RuntimeError(f"tailscale backend state is {state!r}, not 'Running'")
            candidates = self._parse_peers(doc.get("Peer") or {})
        except Exception as exc:
            if not self._outage_logged:
                logger.warning("tailscale discovery unavailable (continuing without it): %s", exc)
                self._outage_logged = True
            return []
        self._outage_logged = False
        return candidates

    def _parse_peers(self, peer_map: dict) -> list[DiscoveredPeer]:
        candidates: list[DiscoveredPeer] = []
        for peer in peer_map.values():
            if not isinstance(peer, dict) or peer.get("Online") is not True:
                continue
            ip = self._first_tailnet_ipv4(peer.get("TailscaleIPs") or [])
            if ip is None:
                continue
            name = peer.get("HostName")
            candidates.append(
                DiscoveredPeer(
                    url=f"http://{ip}:{self._port}",
                    name=name if isinstance(name, str) and name else None,
                )
            )
        return candidates

    @staticmethod
    def _first_tailnet_ipv4(ips: list) -> str | None:
        for raw in ips:
            try:
                address = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if address.version == 4 and address in _TAILNET_IPV4:
                return str(address)
        return None
