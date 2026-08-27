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

"""TailscaleSource: `tailscale status --json` parsing via an injected runner.

The source only produces CANDIDATES — bare urls at this app's backend port on
peers' tailnet IPv4s — and the registry's verify handshake remains the single
trust path. Every failure mode (no binary, logged out, malformed output) is an
EMPTY discovery logged once at INFO, never an exception: tailscale being
absent must not degrade a server that never asked for it beyond a log line.
"""

from __future__ import annotations

import json
import logging
import subprocess

import pytest


def _status_doc(peers: dict, backend_state: str = "Running") -> str:
    """A realistic `tailscale status --json` document (v1.6x shape, trimmed to
    the fields real output carries around the ones we read)."""
    return json.dumps(
        {
            "Version": "1.66.4",
            "TUN": True,
            "BackendState": backend_state,
            "AuthURL": "",
            "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
            "Self": {
                "ID": "n0",
                "PublicKey": "nodekey:self",
                "HostName": "station",
                "DNSName": "station.tail1234.ts.net.",
                "OS": "linux",
                "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
                "Online": True,
            },
            "Health": [],
            "MagicDNSSuffix": "tail1234.ts.net",
            "CertDomains": None,
            "Peer": peers,
        }
    )


def _peer(hostname: str, ips: list[str], online: bool) -> dict:
    return {
        "ID": "n1",
        "PublicKey": f"nodekey:{hostname}",
        "HostName": hostname,
        "DNSName": f"{hostname}.tail1234.ts.net.",
        "OS": "linux",
        "TailscaleIPs": ips,
        "Online": online,
        "ExitNode": False,
        "Active": online,
    }


def _source(output: str | Exception, port: int = 8000):
    from makermodslab.node_sources import TailscaleSource

    def runner() -> str:
        if isinstance(output, Exception):
            raise output
        return output

    return TailscaleSource(port=port, runner=runner)


def test_source_id_is_tailscale():
    assert _source(_status_doc({})).source_id == "tailscale"


def test_online_peers_become_candidates_at_backend_port():
    from makermodslab.nodes import DiscoveredPeer

    doc = _status_doc(
        {
            "nodekey:aaa": _peer("bench-pi", ["100.64.0.7", "fd7a:115c:a1e0::7"], online=True),
            "nodekey:bbb": _peer("laptop", ["100.64.0.8"], online=False),
        }
    )
    assert _source(doc).discover() == [
        DiscoveredPeer(url="http://100.64.0.7:8000", name="bench-pi"),
    ]


def test_candidates_use_the_configured_port():
    doc = _status_doc({"nodekey:aaa": _peer("bench-pi", ["100.64.0.7"], online=True)})
    [candidate] = _source(doc, port=8443).discover()
    assert candidate.url == "http://100.64.0.7:8443"


def test_peer_without_a_tailnet_ipv4_is_skipped():
    """Only 100.64.0.0/10 IPv4s qualify: an IPv6-only peer and one advertising
    a non-CGNAT address (however it got there) produce no candidate."""
    doc = _status_doc(
        {
            "nodekey:aaa": _peer("v6-only", ["fd7a:115c:a1e0::9"], online=True),
            "nodekey:bbb": _peer("weird", ["192.168.1.5"], online=True),
            "nodekey:ccc": _peer("no-ips", [], online=True),
        }
    )
    assert _source(doc).discover() == []


def test_peer_missing_optional_fields_is_tolerated():
    """Real output varies by version; a peer without HostName/Online blocks
    must not crash the parse (absent Online means not online)."""
    doc = _status_doc(
        {
            "nodekey:aaa": {"TailscaleIPs": ["100.64.0.7"], "Online": True},
            "nodekey:bbb": {"TailscaleIPs": ["100.64.0.8"]},  # no Online key
            "nodekey:ccc": "not-a-dict",
        }
    )
    [candidate] = _source(doc).discover()
    assert candidate.url == "http://100.64.0.7:8000"
    assert candidate.name is None


def test_not_logged_in_is_empty_discovery(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert _source(_status_doc({}, backend_state="NeedsLogin")).discover() == []
    assert "NeedsLogin" in caplog.text


def test_stopped_backend_is_empty_discovery():
    assert _source(_status_doc({}, backend_state="Stopped")).discover() == []


def test_missing_or_null_peer_map_is_empty_discovery():
    doc = json.loads(_status_doc({}))
    doc["Peer"] = None
    assert _source(json.dumps(doc)).discover() == []
    del doc["Peer"]
    assert _source(json.dumps(doc)).discover() == []


def test_missing_binary_is_empty_discovery_logged_once_at_info(caplog: pytest.LogCaptureFixture):
    source = _source(FileNotFoundError("No such file or directory: 'tailscale'"))
    with caplog.at_level(logging.INFO):
        assert source.discover() == []
        assert source.discover() == []
    failure_logs = [r for r in caplog.records if "tailscale" in r.getMessage().lower()]
    assert len(failure_logs) == 1
    assert failure_logs[0].levelno == logging.INFO


def test_malformed_output_is_empty_discovery():
    assert _source("flagrant nonsense {").discover() == []


def test_subprocess_failures_are_empty_discovery():
    assert _source(subprocess.CalledProcessError(1, ["tailscale", "status", "--json"])).discover() == []
    assert _source(subprocess.TimeoutExpired(["tailscale", "status", "--json"], 5)).discover() == []


def test_failure_log_rearms_after_a_successful_discovery(caplog: pytest.LogCaptureFixture):
    """One INFO line per outage, not one per process lifetime: a recovery
    resets the once-guard so the NEXT outage is reported too."""
    from makermodslab.node_sources import TailscaleSource

    outputs: list[str | Exception] = [
        FileNotFoundError("no tailscale"),
        _status_doc({}),
        FileNotFoundError("no tailscale"),
    ]

    def runner() -> str:
        output = outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    source = TailscaleSource(runner=runner)
    with caplog.at_level(logging.INFO):
        source.discover()
        source.discover()
        source.discover()
    failure_logs = [r for r in caplog.records if "no tailscale" in r.getMessage()]
    assert len(failure_logs) == 2
