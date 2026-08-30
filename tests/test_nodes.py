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

"""The node registry: static/manual peer source with a verify-on-add handshake.

A peer is another MakerMods Lab server on the LAN/tailnet; its identity
document is /api/v1/health (version, instance_id, capabilities). The registry
is a state machine driven here by an injected clock (no sleeps) and an
injected httpx.MockTransport (no sockets): add/verify, duplicate and self-add
refusals, TTL-driven re-probes on list, unreachable marking (kept, never
evicted), removal, and the url+name persistence round-trip.
"""

from __future__ import annotations

import json
import logging
import re

import httpx
import pytest

LOCAL_ID = "aa" * 16
PEER_A_ID = "bb" * 16
PEER_B_ID = "cc" * 16

PEER_A_URL = "http://peer-a:8000"
PEER_B_URL = "http://peer-b:8000"


def _health_doc(instance_id: str, version: str = "1.2.3", serves_ui: bool = True) -> dict:
    """A /api/v1/health body as server.py health_check produces it."""
    return {
        "status": "ok",
        "message": "FastAPI server is running",
        "version": version,
        "instance_id": instance_id,
        "capabilities": {"serves_ui": serves_ui, "accepts_jobs": True},
    }


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeNetwork:
    """Programmable LAN: base url -> health doc; `down` urls refuse connections.

    Counts probes per base url so tests can assert the TTL actually gates
    re-verification instead of every list() hitting the network. Peers can
    also carry a jobs listing (base url -> the /api/v1/jobs body) for the
    workload-proxy tests, per-job records/logs for the drill-in proxies, and
    programmed (status, body) answers for the forwarded stop/delete calls.
    """

    def __init__(self) -> None:
        self.peers: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.queue: dict[str, dict] = {}
        # Drill-in state, keyed (base url, job id).
        self.job_docs: dict[tuple[str, str], dict] = {}
        self.job_logs: dict[tuple[str, str], list[dict]] = {}
        # Programmed answers for the NON-GET forwards: (status, body); body
        # None ⇒ an empty response (the peer's own DELETE answers 204).
        self.stop_results: dict[tuple[str, str], tuple[int, dict | None]] = {}
        self.delete_results: dict[tuple[str, str], tuple[int, dict | None]] = {}
        self.down: set[str] = set()
        self.probes: dict[str, int] = {}
        # Every request that reached the transport, for asserting what the
        # proxy actually forwarded (method + full url, query string included).
        self.requests: list[tuple[str, str]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            # Computed from the url's parts, not removesuffix on the whole
            # string — the forwarded stop call carries a query string.
            base = f"{request.url.scheme}://{request.url.netloc.decode()}"
            self.requests.append((request.method, str(request.url)))
            if base in self.down:
                raise httpx.ConnectError("connection refused", request=request)
            if path == "/api/v1/jobs/queue":
                if base not in self.queue:
                    return httpx.Response(404, json={"detail": "Not Found"})
                return httpx.Response(200, json=self.queue[base])
            if path == "/api/v1/jobs":
                if base not in self.jobs:
                    return httpx.Response(404, json={"detail": "Not Found"})
                return httpx.Response(200, json=self.jobs[base])
            if path == "/api/v1/health":
                self.probes[base] = self.probes.get(base, 0) + 1
                if base not in self.peers:
                    return httpx.Response(404, json={"detail": "Not Found"})
                return httpx.Response(200, json=self.peers[base])
            # The peer's own per-job surface: /api/v1/jobs/{id}[/logs|/stop].
            match = re.fullmatch(r"/api/v1/jobs/([^/]+)(?:/(logs|stop))?", path)
            assert match, f"unexpected probe path: {request.url}"
            key = (base, match.group(1))
            not_found = {"detail": f"Job {match.group(1)!r} not found"}
            if match.group(2) == "stop":
                assert request.method == "POST", f"stop must be POSTed, got {request.method}"
                status, body = self.stop_results.get(key, (404, not_found))
            elif match.group(2) == "logs":
                assert request.method == "GET"
                if key not in self.job_logs:
                    return httpx.Response(404, json=not_found)
                return httpx.Response(200, json={"logs": self.job_logs[key]})
            elif request.method == "DELETE":
                status, body = self.delete_results.get(key, (404, not_found))
            else:
                assert request.method == "GET"
                if key not in self.job_docs:
                    return httpx.Response(404, json=not_found)
                return httpx.Response(200, json=self.job_docs[key])
            if body is None:
                return httpx.Response(status)
            return httpx.Response(status, json=body)

        return httpx.MockTransport(handler)


@pytest.fixture
def local_identity(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin this process's instance id so self-add tests are deterministic and
    no test ever mints an id into the developer's real cache dir."""
    from makermodslab.utils import config as cfg

    monkeypatch.setattr(cfg, "_instance_id_cache", LOCAL_ID)
    return LOCAL_ID


@pytest.fixture
def nodes_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the persisted peer list (utils/config.NODES_FILE) into tmp."""
    from makermodslab.utils import config as cfg

    path = tmp_path / "nodes.json"
    monkeypatch.setattr(cfg, "NODES_FILE", str(path))
    return path


@pytest.fixture
def network() -> FakeNetwork:
    net = FakeNetwork()
    net.peers[PEER_A_URL] = _health_doc(PEER_A_ID)
    net.peers[PEER_B_URL] = _health_doc(PEER_B_ID, version="1.2.4", serves_ui=False)
    return net


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def wall_clock() -> FakeClock:
    """The wall-clock twin of `clock` (time.time in production). Starts at a
    value far from the monotonic clock's so a test that conflates the two
    clocks fails loudly instead of passing by coincidence."""
    return FakeClock(now=1_700_000_000.0)


@pytest.fixture
def registry(local_identity, nodes_file, network, clock, wall_clock):
    from makermodslab.nodes import NodeRegistry

    return NodeRegistry(clock=clock, transport=network.transport(), wall_clock=wall_clock)


# ---------------------------------------------------------------------------
# Registry state machine
# ---------------------------------------------------------------------------


def test_add_verifies_handshake_and_records_identity(registry, network, clock):
    node = registry.add(PEER_A_URL, name="bench")
    assert node.url == PEER_A_URL
    assert node.instance_id == PEER_A_ID
    assert node.name == "bench"
    assert node.version == "1.2.3"
    assert node.capabilities == {"serves_ui": True, "accepts_jobs": True}
    assert node.status == "ok"
    assert node.last_verified_at == clock.now
    assert network.probes[PEER_A_URL] == 1


def test_add_normalizes_trailing_slash(registry, network):
    node = registry.add(PEER_A_URL + "/")
    assert node.url == PEER_A_URL
    assert node.name is None


def test_add_rejects_url_without_http_scheme(registry):
    with pytest.raises(ValueError):
        registry.add("peer-a:8000")


def test_add_unreachable_is_an_error_not_a_pending_state(registry, network):
    from makermodslab.nodes import NodeUnreachableError

    network.down.add(PEER_A_URL)
    with pytest.raises(NodeUnreachableError):
        registry.add(PEER_A_URL)
    assert registry.list_nodes() == []


def test_add_non_node_answer_is_unreachable(registry, network):
    """A host that answers /api/v1/health with garbage (or a 404) is not a
    MakerMods Lab node — refused the same way as a dead host."""
    from makermodslab.nodes import NodeUnreachableError

    network.peers["http://printer:8000"] = {"hello": "world"}  # no instance_id
    with pytest.raises(NodeUnreachableError):
        registry.add("http://printer:8000")
    with pytest.raises(NodeUnreachableError):
        registry.add("http://nothing-here:8000")  # 404 body
    assert registry.list_nodes() == []


def test_add_self_is_refused(registry, network, local_identity):
    from makermodslab.nodes import SelfAddError

    network.peers["http://me:8000"] = _health_doc(local_identity)
    with pytest.raises(SelfAddError):
        registry.add("http://me:8000")
    assert registry.list_nodes() == []


def test_readd_same_url_is_a_duplicate(registry, network):
    from makermodslab.nodes import DuplicateNodeError

    registry.add(PEER_A_URL)
    with pytest.raises(DuplicateNodeError):
        registry.add(PEER_A_URL)
    assert len(registry.list_nodes()) == 1


def test_readd_same_instance_under_new_url_updates_in_place(registry, network):
    """A machine's address can change (DHCP, tailnet rename); its instance id
    doesn't. Re-adding the same identity under a new URL moves the entry."""
    registry.add(PEER_A_URL, name="bench")
    network.peers["http://peer-a-new:8000"] = _health_doc(PEER_A_ID)
    node = registry.add("http://peer-a-new:8000")
    assert node.instance_id == PEER_A_ID
    assert node.url == "http://peer-a-new:8000"
    assert node.name == "bench"  # name survives unless re-supplied
    [only] = registry.list_nodes()
    assert only.url == "http://peer-a-new:8000"


def test_list_within_ttl_does_not_reprobe(registry, network, clock):
    registry.add(PEER_A_URL)
    registry.list_nodes()
    clock.advance(14.9)
    registry.list_nodes()
    assert network.probes[PEER_A_URL] == 1  # only the verify-on-add probe


def test_list_reprobes_after_ttl(registry, network, clock):
    registry.add(PEER_A_URL)
    verified_at = clock.now
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert network.probes[PEER_A_URL] == 2
    assert node.status == "ok"
    assert node.last_verified_at == verified_at + 15.1


def test_failed_reprobe_marks_unreachable_and_keeps_entry(registry, network, clock):
    registry.add(PEER_A_URL)
    verified_at = clock.now
    network.down.add(PEER_A_URL)
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "unreachable"
    assert node.instance_id == PEER_A_ID  # last known identity retained
    assert node.last_verified_at == verified_at  # last SUCCESSFUL handshake


def test_unreachable_peer_is_not_hammered_within_ttl(registry, network, clock):
    registry.add(PEER_A_URL)
    network.down.add(PEER_A_URL)
    clock.advance(15.1)
    registry.list_nodes()
    probes_after_failure = network.probes[PEER_A_URL]
    registry.list_nodes()  # immediately again — inside the TTL window
    assert network.probes[PEER_A_URL] == probes_after_failure


def test_unreachable_peer_recovers_on_a_later_probe(registry, network, clock):
    registry.add(PEER_A_URL)
    network.down.add(PEER_A_URL)
    clock.advance(15.1)
    registry.list_nodes()
    network.down.discard(PEER_A_URL)
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "ok"
    assert node.last_verified_at == clock.now


def test_remove_deletes_entry_and_persists(registry, network, nodes_file):
    registry.add(PEER_A_URL)
    registry.add(PEER_B_URL)
    registry.remove(PEER_A_ID)
    [only] = registry.list_nodes()
    assert only.instance_id == PEER_B_ID
    assert json.loads(nodes_file.read_text()) == [{"url": PEER_B_URL, "name": None}]


def test_remove_unknown_raises(registry):
    from makermodslab.nodes import NodeNotFoundError

    with pytest.raises(NodeNotFoundError):
        registry.remove(PEER_A_ID)


# ---------------------------------------------------------------------------
# resolve(): the pre-flight lookup for talking to a peer (job offload)
# ---------------------------------------------------------------------------


def test_resolve_trusts_a_fresh_verification_without_reprobing(registry, network, clock):
    registry.add(PEER_A_URL)
    clock.advance(1.0)  # inside the TTL
    node = registry.resolve(PEER_A_ID)
    assert node.url == PEER_A_URL
    assert node.status == "ok"
    assert network.probes[PEER_A_URL] == 1  # add's handshake only


def test_resolve_reprobes_when_stale_and_raises_on_a_dead_peer(registry, network, clock):
    from makermodslab.nodes import NODE_TTL_S, NodeUnreachableError

    registry.add(PEER_A_URL)
    network.down.add(PEER_A_URL)
    clock.advance(NODE_TTL_S + 1)
    with pytest.raises(NodeUnreachableError):
        registry.resolve(PEER_A_ID)
    # The entry is kept (marked unreachable), never evicted.
    [node] = registry.list_nodes()
    assert node.status == "unreachable"


def test_resolve_unknown_instance_raises_not_found(registry, network):
    from makermodslab.nodes import NodeNotFoundError

    registry.add(PEER_A_URL)
    with pytest.raises(NodeNotFoundError):
        registry.resolve("ee" * 16)


def test_resolve_probes_loaded_unverified_entries_first(local_identity, nodes_file, network, clock):
    """A peer loaded from disk has no identity until its first handshake, so
    resolving by instance_id must probe it rather than report not-found."""
    from makermodslab.nodes import NodeRegistry

    NodeRegistry(clock=clock, transport=network.transport()).add(PEER_A_URL)
    reloaded = NodeRegistry(clock=clock, transport=network.transport())
    node = reloaded.resolve(PEER_A_ID)
    assert node.instance_id == PEER_A_ID
    assert node.status == "ok"


# ---------------------------------------------------------------------------
# last_seen_at: wall-clock sibling of the monotonic last_verified_at
# ---------------------------------------------------------------------------


def test_handshake_stamps_wall_clock_last_seen(registry, network, clock, wall_clock):
    """Every successful handshake stamps last_seen_at from the WALL clock
    (time.time in production) — a sibling of the monotonic last_verified_at,
    which keeps its registry-clock semantics untouched."""
    node = registry.add(PEER_A_URL)
    assert node.last_seen_at == wall_clock.now
    assert node.last_verified_at == clock.now  # monotonic field unchanged


def test_last_seen_survives_failure_and_refreshes_on_recovery(registry, network, clock, wall_clock):
    registry.add(PEER_A_URL)
    seen_at = wall_clock.now
    network.down.add(PEER_A_URL)
    clock.advance(15.1)
    wall_clock.advance(240.0)
    [node] = registry.list_nodes()
    assert node.status == "unreachable"
    assert node.last_seen_at == seen_at  # last SUCCESSFUL handshake, like last_verified_at

    network.down.discard(PEER_A_URL)
    clock.advance(15.1)
    wall_clock.advance(240.0)
    [node] = registry.list_nodes()
    assert node.status == "ok"
    assert node.last_seen_at == wall_clock.now


def test_never_verified_peer_has_no_last_seen(local_identity, nodes_file, network, clock, wall_clock):
    from makermodslab.nodes import NodeRegistry

    nodes_file.write_text(json.dumps([{"url": PEER_A_URL, "name": "bench"}]))
    network.down.add(PEER_A_URL)
    registry = NodeRegistry(clock=clock, transport=network.transport(), wall_clock=wall_clock)
    [node] = registry.list_nodes()
    assert node.last_seen_at is None


# ---------------------------------------------------------------------------
# Persistence: url + name only; identity re-verified on load
# ---------------------------------------------------------------------------


def test_persistence_saves_url_and_name_only(registry, network, nodes_file):
    registry.add(PEER_A_URL, name="bench")
    saved = json.loads(nodes_file.read_text())
    assert saved == [{"url": PEER_A_URL, "name": "bench"}]  # no identity on disk


def test_persistence_round_trip_reverifies_identity(local_identity, nodes_file, network, clock):
    from makermodslab.nodes import NodeRegistry

    NodeRegistry(clock=clock, transport=network.transport()).add(PEER_A_URL, name="bench")

    reloaded = NodeRegistry(clock=clock, transport=network.transport())
    [node] = reloaded.list_nodes()  # first list re-runs the handshake
    assert node.instance_id == PEER_A_ID
    assert node.name == "bench"
    assert node.status == "ok"


def test_loaded_peer_that_fails_verification_is_kept_unreachable(local_identity, nodes_file, network, clock):
    from makermodslab.nodes import NodeRegistry

    nodes_file.write_text(json.dumps([{"url": PEER_A_URL, "name": "bench"}]))
    network.down.add(PEER_A_URL)
    [node] = NodeRegistry(clock=clock, transport=network.transport()).list_nodes()
    assert node.status == "unreachable"
    assert node.url == PEER_A_URL
    assert node.name == "bench"
    assert node.instance_id is None  # never verified this run
    assert node.last_verified_at is None


def test_corrupt_nodes_file_is_tolerated(local_identity, nodes_file, network, clock):
    from makermodslab.nodes import NodeRegistry

    nodes_file.write_text("not json {")
    assert NodeRegistry(clock=clock, transport=network.transport()).list_nodes() == []


# ---------------------------------------------------------------------------
# Discovery sources: candidates only — the verify handshake stays the trust path
# ---------------------------------------------------------------------------

TAILNET_A_URL = "http://100.64.0.7:8000"
TAILNET_B_URL = "http://100.64.0.8:8000"


class FakeSource:
    """A programmable NodeSource: mutate `candidates` between passes to model
    peers joining and leaving the tailnet, or set `outage` to model a source
    that cannot answer (raises — the transient path). Counts discover() calls
    so tests can assert the TTL actually gates re-discovery."""

    def __init__(self, source_id: str = "tailscale", candidates: list | None = None) -> None:
        self.source_id = source_id
        self.candidates = candidates or []
        self.discover_calls = 0
        self.outage: Exception | None = None

    def discover(self):
        self.discover_calls += 1
        if self.outage is not None:
            raise self.outage
        return list(self.candidates)


def _tailnet_source(*urls: str):
    from makermodslab.nodes import DiscoveredPeer

    return FakeSource(candidates=[DiscoveredPeer(url=url) for url in urls])


def test_discovered_candidate_is_verified_and_listed_with_its_source(registry, network, clock):
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    [node] = registry.list_nodes()
    assert node.url == TAILNET_A_URL
    assert node.instance_id == PEER_A_ID
    assert node.status == "ok"
    assert node.source == "tailscale"
    assert node.name is None


def test_manual_entries_carry_the_manual_source(registry, network):
    assert registry.add(PEER_A_URL).source == "manual"


def test_discovery_is_ttl_gated_like_the_probes(registry, network, clock):
    source = _tailnet_source()
    registry.register_source(source)
    registry.list_nodes()
    clock.advance(14.9)
    registry.list_nodes()
    assert source.discover_calls == 1
    clock.advance(0.2)  # past the TTL
    registry.list_nodes()
    assert source.discover_calls == 2


def test_unverified_candidates_probed_per_pass_are_capped(local_identity, nodes_file, network, clock):
    """A discovery pass probes at most `discovery_probe_cap` unverified
    candidates; the rest stay `pending` (instance_id unknown) and are picked
    up on later passes — the registry never storms the tailnet."""
    from makermodslab.nodes import NodeRegistry

    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    network.peers[TAILNET_B_URL] = _health_doc(PEER_B_ID)
    registry = NodeRegistry(clock=clock, transport=network.transport(), discovery_probe_cap=1)
    registry.register_source(_tailnet_source(TAILNET_A_URL, TAILNET_B_URL))

    first, second = registry.list_nodes()
    assert (first.status, first.instance_id) == ("ok", PEER_A_ID)
    assert (second.status, second.instance_id) == ("pending", None)
    assert TAILNET_B_URL not in network.probes  # never touched this pass

    clock.advance(15.1)
    statuses = {n.url: n.status for n in registry.list_nodes()}
    assert statuses == {TAILNET_A_URL: "ok", TAILNET_B_URL: "ok"}


def test_discovered_dedupes_against_manual_on_instance_id_and_manual_wins_name(registry, network, clock):
    """The same machine known manually and discovered over the tailnet is ONE
    entry: the manual record (name included) is authoritative."""
    registry.add(PEER_A_URL, name="bench")
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    [node] = registry.list_nodes()
    assert node.instance_id == PEER_A_ID
    assert node.source == "manual"
    assert node.name == "bench"
    assert node.url == PEER_A_URL


def test_discovered_peers_dedupe_against_each_other_on_instance_id(registry, network, clock):
    """One machine, two discovered addresses (e.g. two sources): the first
    verified entry is the record, the duplicate candidate is dropped."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    network.peers[TAILNET_B_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL, TAILNET_B_URL))
    [node] = registry.list_nodes()
    assert node.instance_id == PEER_A_ID
    assert node.url == TAILNET_A_URL


def test_discovered_peer_that_moves_address_converges_to_one_entry(registry, network, clock):
    """DHCP-on-the-tailnet: the peer's old address dies and discovery reports
    a new one. The entry follows the identity to the new url."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    registry.list_nodes()

    del network.peers[TAILNET_A_URL]
    network.down.add(TAILNET_A_URL)
    network.peers[TAILNET_B_URL] = _health_doc(PEER_A_ID)
    source.candidates = _tailnet_source(TAILNET_B_URL).candidates
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.instance_id == PEER_A_ID
    assert node.url == TAILNET_B_URL
    assert node.status == "ok"


def test_discovered_peer_gone_from_source_and_dead_is_evicted(registry, network, clock):
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    assert len(registry.list_nodes()) == 1

    source.candidates = []
    network.down.add(TAILNET_A_URL)
    clock.advance(15.1)
    assert registry.list_nodes() == []


def test_discovered_peer_gone_from_source_but_alive_is_kept(registry, network, clock):
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    registry.list_nodes()

    source.candidates = []  # e.g. tailscale briefly reports the peer offline
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "ok"


def test_discovered_peer_still_in_source_but_down_is_kept_unreachable(registry, network, clock):
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    registry.list_nodes()

    network.down.add(TAILNET_A_URL)
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "unreachable"
    assert node.instance_id == PEER_A_ID


def test_manual_peer_is_never_evicted_even_with_sources_registered(registry, network, clock):
    registry.add(PEER_A_URL)
    registry.register_source(_tailnet_source())
    network.down.add(PEER_A_URL)
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "unreachable"
    assert node.source == "manual"


def test_discovered_peers_are_never_persisted(registry, network, clock, nodes_file):
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    registry.list_nodes()
    assert not nodes_file.exists()  # discovery alone never writes nodes.json

    registry.add(PEER_B_URL, name="bench")
    registry.list_nodes()
    assert json.loads(nodes_file.read_text()) == [{"url": PEER_B_URL, "name": "bench"}]


def test_candidate_answering_as_self_is_dropped(registry, network, clock, local_identity):
    """Tailscale can hand us our own address; the handshake recognises our
    instance_id and the candidate silently disappears (never an error)."""
    network.peers[TAILNET_A_URL] = _health_doc(local_identity)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    assert registry.list_nodes() == []


def test_source_outage_is_never_fatal_and_logged_once_per_outage(registry, network, clock, caplog):
    """A raising source is an OUTAGE, never fatal: the list call succeeds,
    and the failure is logged once per outage (re-armed by a recovery), not
    once per list()."""
    registry.add(PEER_A_URL)
    source = FakeSource()
    source.outage = RuntimeError("boom")
    registry.register_source(source)
    with caplog.at_level(logging.WARNING, logger="makermodslab.nodes"):
        [node] = registry.list_nodes()
        clock.advance(15.1)
        registry.list_nodes()
    assert node.status == "ok"
    assert len([r for r in caplog.records if "boom" in r.getMessage()]) == 1

    source.outage = None  # recovery re-arms the log for the NEXT outage
    clock.advance(15.1)
    registry.list_nodes()
    source.outage = RuntimeError("boom again")
    clock.advance(15.1)
    with caplog.at_level(logging.WARNING, logger="makermodslab.nodes"):
        registry.list_nodes()
    assert any("boom again" in r.getMessage() for r in caplog.records)


def test_source_outage_keeps_a_down_discovered_peer(registry, network, clock):
    """Eviction's gone-from-discovery half needs a DEFINITIVE answer. While
    the source is in outage, a discovered peer that also stops answering is
    kept `unreachable` — never evicted on a guess. (Contrast with
    test_discovered_peer_gone_from_source_and_dead_is_evicted, where the
    source really answered 'no peers'.)"""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    registry.list_nodes()

    source.outage = RuntimeError("tailscaled hiccup")
    network.down.add(TAILNET_A_URL)
    clock.advance(15.1)
    [node] = registry.list_nodes()
    assert node.status == "unreachable"
    assert node.instance_id == PEER_A_ID

    # The outage ends and the source reports the peer gone for real: the
    # definitive answer plus the failing probe now evict it.
    source.outage = None
    source.candidates = []
    clock.advance(15.1)
    assert registry.list_nodes() == []


# ---------------------------------------------------------------------------
# force=True: the manual-refresh contract — this pass bypasses the TTL
# ---------------------------------------------------------------------------


def test_force_bypasses_the_probe_ttl(registry, network, clock):
    registry.add(PEER_A_URL)
    registry.list_nodes()
    clock.advance(1.0)  # well inside the TTL: a plain list would not re-probe
    registry.list_nodes()
    assert network.probes[PEER_A_URL] == 1
    registry.list_nodes(force=True)
    assert network.probes[PEER_A_URL] == 2


def test_force_bypasses_the_discovery_ttl(registry, network, clock):
    source = _tailnet_source()
    registry.register_source(source)
    registry.list_nodes()
    clock.advance(1.0)
    registry.list_nodes(force=True)
    assert source.discover_calls == 2


def test_forced_pass_with_tailnet_down_clears_discovered_entries(registry, network, clock):
    """The user turns tailscale OFF and clicks refresh: the source's clean
    'no peers' answer plus failing probes clear every discovered entry in ONE
    forced pass — no TTL wait, no stale rows."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    assert len(registry.list_nodes()) == 1

    source.candidates = []
    network.down.add(TAILNET_A_URL)
    clock.advance(1.0)  # inside the TTL — only force gets a fresh answer now
    assert registry.list_nodes(force=True) == []


def test_manual_promoted_entry_survives_a_tailnet_down_forced_pass(registry, network, clock):
    """A discovered peer the user adopted by hand is manual — a forced pass
    with the tailnet gone marks it unreachable but never evicts it."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    source = _tailnet_source(TAILNET_A_URL)
    registry.register_source(source)
    registry.list_nodes()
    registry.add(TAILNET_A_URL, name="bench")  # promotion

    source.candidates = []
    network.down.add(TAILNET_A_URL)
    clock.advance(1.0)
    [node] = registry.list_nodes(force=True)
    assert node.source == "manual"
    assert node.status == "unreachable"
    assert node.name == "bench"


def test_force_still_caps_unverified_candidate_probes(local_identity, nodes_file, network, clock):
    """force re-probes every KNOWN entry, but the discovery probe cap still
    bounds unverified candidates — a manual refresh must not storm a big
    tailnet either."""
    from makermodslab.nodes import NodeRegistry

    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    network.peers[TAILNET_B_URL] = _health_doc(PEER_B_ID)
    registry = NodeRegistry(clock=clock, transport=network.transport(), discovery_probe_cap=1)
    registry.register_source(_tailnet_source(TAILNET_A_URL, TAILNET_B_URL))
    first, second = registry.list_nodes(force=True)
    assert first.status == "ok"
    assert second.status == "pending"
    assert TAILNET_B_URL not in network.probes  # capped out of this pass too


def test_manual_add_promotes_a_discovered_entry(registry, network, clock, nodes_file):
    """Adding a discovered peer by hand makes it a manual entry — persisted,
    never evicted — without duplicating it."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL))
    registry.list_nodes()

    node = registry.add(TAILNET_A_URL, name="bench")
    assert node.source == "manual"
    assert node.name == "bench"
    [only] = registry.list_nodes()
    assert only.source == "manual"
    assert json.loads(nodes_file.read_text()) == [{"url": TAILNET_A_URL, "name": "bench"}]


def test_register_sources_from_env_is_opt_in(monkeypatch, local_identity, nodes_file, network, clock):
    """--discover-tailscale sets MAKERMODSLAB_DISCOVER_TAILSCALE=1 before the
    server imports; nothing is registered without it (OFF by default)."""
    from makermodslab import nodes
    from makermodslab.node_sources import TailscaleSource

    registry = nodes.NodeRegistry(clock=clock, transport=network.transport())
    monkeypatch.delenv("MAKERMODSLAB_DISCOVER_TAILSCALE", raising=False)
    nodes.register_sources_from_env(registry)
    assert registry._sources == []

    monkeypatch.setenv("MAKERMODSLAB_DISCOVER_TAILSCALE", "1")
    nodes.register_sources_from_env(registry)
    assert [type(s) for s in registry._sources] == [TailscaleSource]


# ---------------------------------------------------------------------------
# Endpoints (v1-only surface)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_registry(monkeypatch: pytest.MonkeyPatch, local_identity, nodes_file, network, clock, wall_clock):
    """Swap the module singleton for a test-driven registry; handlers look the
    global up at call time, so the routes see this instance."""
    from makermodslab import nodes

    reg = nodes.NodeRegistry(clock=clock, transport=network.transport(), wall_clock=wall_clock)
    monkeypatch.setattr(nodes, "node_registry", reg)
    return reg


def test_get_nodes_lists_self_first(client, api_registry, local_identity):
    from makermodslab.__version__ import __version__

    body = client.get("/api/v1/nodes").json()
    self_entry = body["nodes"][0]
    assert self_entry["is_self"] is True
    assert self_entry["instance_id"] == local_identity
    assert self_entry["version"] == __version__
    assert self_entry["status"] == "ok"
    assert self_entry["url"] is None
    assert isinstance(self_entry["capabilities"]["serves_ui"], bool)
    assert self_entry["capabilities"]["accepts_jobs"] is True


def test_post_nodes_verifies_and_returns_entry(client, api_registry):
    resp = client.post("/api/v1/nodes", json={"url": PEER_A_URL, "name": "bench"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["instance_id"] == PEER_A_ID
    assert body["url"] == PEER_A_URL
    assert body["name"] == "bench"
    assert body["version"] == "1.2.3"
    assert body["status"] == "ok"
    assert body["is_self"] is False

    listed = client.get("/api/v1/nodes").json()["nodes"]
    assert [n["instance_id"] for n in listed] == [LOCAL_ID, PEER_A_ID]


def test_post_nodes_duplicate_409(client, api_registry):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    resp = client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    assert resp.status_code == 409
    assert resp.json()["code"] == "node.duplicate"


def test_post_nodes_self_409(client, api_registry, network, local_identity):
    network.peers["http://me:8000"] = _health_doc(local_identity)
    resp = client.post("/api/v1/nodes", json={"url": "http://me:8000"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "node.self"


def test_post_nodes_unreachable_502(client, api_registry, network):
    network.down.add(PEER_A_URL)
    resp = client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"
    assert isinstance(resp.json()["detail"], str)


def test_post_nodes_bad_scheme_422(client, api_registry):
    resp = client.post("/api/v1/nodes", json={"url": "peer-a:8000"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


def test_delete_node_removes(client, api_registry):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    resp = client.delete(f"/api/v1/nodes/{PEER_A_ID}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "instance_id": PEER_A_ID}
    listed = client.get("/api/v1/nodes").json()["nodes"]
    assert [n["instance_id"] for n in listed] == [LOCAL_ID]


def test_delete_unknown_node_404(client, api_registry):
    resp = client.delete(f"/api/v1/nodes/{PEER_B_ID}")
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_get_nodes_reports_each_entry_source(client, api_registry, network):
    """Additive `source` field: "manual" for the self entry and hand-added
    peers, the source_id for discovered ones."""
    network.peers[TAILNET_A_URL] = _health_doc(PEER_B_ID)
    api_registry.register_source(_tailnet_source(TAILNET_A_URL))
    client.post("/api/v1/nodes", json={"url": PEER_A_URL, "name": "bench"})

    nodes = client.get("/api/v1/nodes").json()["nodes"]
    by_id = {n["instance_id"]: n for n in nodes}
    assert by_id[LOCAL_ID]["source"] == "manual"  # self
    assert by_id[PEER_A_ID]["source"] == "manual"
    assert by_id[PEER_B_ID]["source"] == "tailscale"
    assert by_id[PEER_B_ID]["status"] == "ok"


def test_get_nodes_marks_unverified_candidates_pending(
    client, monkeypatch, local_identity, nodes_file, network, clock
):
    """An unverified candidate is clearly distinguishable in the payload:
    status "pending", null instance_id, its source id — never mistakable for
    a verified peer."""
    from makermodslab import nodes

    registry = nodes.NodeRegistry(clock=clock, transport=network.transport(), discovery_probe_cap=1)
    monkeypatch.setattr(nodes, "node_registry", registry)
    network.peers[TAILNET_A_URL] = _health_doc(PEER_A_ID)
    network.peers[TAILNET_B_URL] = _health_doc(PEER_B_ID)
    registry.register_source(_tailnet_source(TAILNET_A_URL, TAILNET_B_URL))

    nodes_body = client.get("/api/v1/nodes").json()["nodes"]
    pending = next(n for n in nodes_body if n["url"] == TAILNET_B_URL)
    assert pending["status"] == "pending"
    assert pending["instance_id"] is None
    assert pending["source"] == "tailscale"
    assert pending["is_self"] is False
    verified = next(n for n in nodes_body if n["url"] == TAILNET_A_URL)
    assert verified["status"] == "ok"
    assert verified["instance_id"] == PEER_A_ID


def test_get_nodes_reports_registered_source_ids(client, api_registry):
    """Additive `sources`: the registered discovery-source ids, so the UI can
    say whether tailnet discovery is even on. [] when nothing is registered
    (the default — discovery is opt-in)."""
    assert client.get("/api/v1/nodes").json()["sources"] == []
    api_registry.register_source(_tailnet_source())
    assert client.get("/api/v1/nodes").json()["sources"] == ["tailscale"]


def test_get_nodes_force_probes_now(client, api_registry, network, clock):
    """?force=true is the manual-refresh contract: THIS pass runs discovery
    and probes every entry now, TTL notwithstanding. Without it the fresh
    handshake from the add is still trusted."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    clock.advance(1.0)  # inside the TTL
    [_, peer] = client.get("/api/v1/nodes").json()["nodes"]
    assert peer["status"] == "ok"  # un-forced: the add's handshake is fresh
    [_, peer] = client.get("/api/v1/nodes?force=true").json()["nodes"]
    assert peer["status"] == "unreachable"


def test_get_nodes_reports_wall_clock_last_seen(client, api_registry, wall_clock):
    """Additive `last_seen_at`: wall-clock seconds on verified peers, null on
    the self entry (no handshake with ourselves) and on never-verified rows."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    nodes = client.get("/api/v1/nodes").json()["nodes"]
    by_id = {n["instance_id"]: n for n in nodes}
    assert by_id[LOCAL_ID]["last_seen_at"] is None
    assert by_id[PEER_A_ID]["last_seen_at"] == wall_clock.now


# ---------------------------------------------------------------------------
# GET /api/v1/nodes/{instance_id}/jobs — the node-workload proxy
# ---------------------------------------------------------------------------


def _remote_job_doc(job_id: str = "job-1", state: str = "running") -> dict:
    """A JobRecord as the peer's own GET /api/v1/jobs serves it — built from
    the real model, because the peer runs this same code."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id=job_id,
        job_number=7,
        name="act_so101_run",
        state=state,  # type: ignore[arg-type]
        config=TrainingRequest(dataset_repo_id="user/ds"),
        output_dir=f"outputs/train/{job_id}",
        started_at=1_700_000_000.0,
    )
    return json.loads(record.model_dump_json())


def test_node_jobs_proxies_the_peers_typed_listing(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.jobs[PEER_A_URL] = {"jobs": [_remote_job_doc()]}
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs")
    assert resp.status_code == 200
    [job] = resp.json()["jobs"]
    assert job["id"] == "job-1"
    assert job["state"] == "running"
    assert job["name"] == "act_so101_run"


def test_node_jobs_empty_listing(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.jobs[PEER_A_URL] = {"jobs": []}
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs")
    assert resp.status_code == 200
    assert resp.json() == {"jobs": []}


def test_node_queue_proxies_the_peers_queue_listing(client, api_registry, network):
    """The queue proxy answers with the peer's own /api/v1/jobs/queue verbatim,
    so queued counts are exact — the jobs page is limited and can undercount."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.queue[PEER_A_URL] = {
        "jobs": [
            _remote_job_doc(job_id="q-1", state="queued"),
            _remote_job_doc(job_id="q-2", state="queued"),
        ]
    }
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/queue")
    assert resp.status_code == 200
    assert [j["id"] for j in resp.json()["jobs"]] == ["q-1", "q-2"]


def test_node_queue_unknown_node_404(client, api_registry):
    resp = client.get("/api/v1/nodes/deadbeef/jobs/queue")
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_node_queue_dead_peer_502(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/queue")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_jobs_tolerates_a_newer_peers_additive_fields(client, api_registry, network):
    """Version-skew stance: the peer usually runs the same code, but a NEWER
    peer may serve additive fields this build doesn't know. The proxy must
    pass the listing through without failing on them (unknown fields are
    dropped, never an error)."""
    doc = _remote_job_doc()
    doc["shiny_new_field"] = {"from": "the future"}
    network.jobs[PEER_A_URL] = {"jobs": [doc], "next_page": None}
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs")
    assert resp.status_code == 200
    assert resp.json()["jobs"][0]["id"] == "job-1"


def test_node_jobs_unknown_node_404(client, api_registry):
    resp = client.get(f"/api/v1/nodes/{PEER_B_ID}/jobs")
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_node_jobs_dead_peer_502(client, api_registry, network, clock):
    """The peer fails the pre-flight resolve (stale + down): 502 node.unreachable."""
    from makermodslab.nodes import NODE_TTL_S

    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    clock.advance(NODE_TTL_S + 1)
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_jobs_fetch_failure_502(client, api_registry, network):
    """resolve() trusts a fresh handshake, but the jobs request itself can
    still fail — same 502 node.unreachable either way."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)  # freshly verified, so no re-probe; the jobs GET hits the outage
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


# ---------------------------------------------------------------------------
# Peer-job drill-in proxies: GET record/logs, forwarded stop/delete
# ---------------------------------------------------------------------------


def test_node_job_proxies_the_peers_record(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.job_docs[(PEER_A_URL, "job-1")] = _remote_job_doc()
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "job-1"
    assert body["state"] == "running"
    assert body["name"] == "act_so101_run"


def test_node_job_unknown_node_404(client, api_registry):
    resp = client.get(f"/api/v1/nodes/{PEER_B_ID}/jobs/job-1")
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_node_job_dead_peer_502(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_job_missing_on_peer_reads_unreachable(client, api_registry, network):
    """GET proxies keep the workload proxies' stance: ANY HTTP error from the
    peer — its 404 for an unknown job included — counts as 'could not read the
    peer', 502 node.unreachable. Only the forwarded mutations below pass a
    peer's own refusal through."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/no-such-job")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_job_logs_proxies_the_drained_tail(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    lines = [{"timestamp": 1_700_000_100.0, "message": "step 100/1000"}]
    network.job_logs[(PEER_A_URL, "job-1")] = lines
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/logs")
    assert resp.status_code == 200
    assert resp.json() == {"logs": lines}


def test_node_job_logs_dead_peer_502(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    resp = client.get(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/logs")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_job_stop_forwards_and_returns_the_peers_record(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.stop_results[(PEER_A_URL, "job-1")] = (200, _remote_job_doc(state="interrupted"))
    resp = client.post(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "interrupted"
    forwarded = [(m, u) for m, u in network.requests if u.endswith("/stop")]
    assert forwarded == [("POST", f"{PEER_A_URL}/api/v1/jobs/job-1/stop")]


def test_node_job_stop_forwards_expect_state(client, api_registry, network):
    """The optional expect_state precondition rides the forward untouched, so
    the peer's own 409 job.state_changed guard still protects the click."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.stop_results[(PEER_A_URL, "job-1")] = (200, _remote_job_doc(state="queued"))
    resp = client.post(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/stop?expect_state=queued")
    assert resp.status_code == 200
    [(method, url)] = [(m, u) for m, u in network.requests if "/stop" in u]
    assert method == "POST"
    assert url == f"{PEER_A_URL}/api/v1/jobs/job-1/stop?expect_state=queued"


def test_node_job_stop_peer_refusal_passes_through(client, api_registry, network):
    """The peer ANSWERED — with its own coded refusal. That verdict belongs to
    the caller with the PEER's status and body, never re-wrapped as 502."""
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.stop_results[(PEER_A_URL, "job-1")] = (
        409,
        {"detail": "Job 'job-1' is 'running', not 'queued'", "code": "job.state_changed"},
    )
    resp = client.post(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/stop?expect_state=queued")
    assert resp.status_code == 409
    assert resp.json()["code"] == "job.state_changed"
    assert resp.json()["detail"] == "Job 'job-1' is 'running', not 'queued'"


def test_node_job_stop_unknown_node_404(client, api_registry):
    resp = client.post(f"/api/v1/nodes/{PEER_B_ID}/jobs/job-1/stop")
    assert resp.status_code == 404
    assert resp.json()["code"] == "node.not_found"


def test_node_job_stop_dead_peer_502(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    resp = client.post(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1/stop")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_node_job_delete_forwards_and_answers_204(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.delete_results[(PEER_A_URL, "job-1")] = (204, None)
    resp = client.delete(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1")
    assert resp.status_code == 204
    assert resp.content == b""
    forwarded = [(m, u) for m, u in network.requests if m == "DELETE"]
    assert forwarded == [("DELETE", f"{PEER_A_URL}/api/v1/jobs/job-1")]


def test_node_job_delete_peer_refusal_passes_through(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.delete_results[(PEER_A_URL, "job-1")] = (
        409,
        {
            "detail": "Job 'job-1' holds the checkpoint queued run(s) 'q-1' will train from.",
            "code": "job.has_queued_dependents",
        },
    )
    resp = client.delete(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1")
    assert resp.status_code == 409
    assert resp.json()["code"] == "job.has_queued_dependents"


def test_node_job_delete_dead_peer_502(client, api_registry, network):
    client.post("/api/v1/nodes", json={"url": PEER_A_URL})
    network.down.add(PEER_A_URL)
    resp = client.delete(f"/api/v1/nodes/{PEER_A_ID}/jobs/job-1")
    assert resp.status_code == 502
    assert resp.json()["code"] == "node.unreachable"


def test_nodes_surface_is_v1_only(client, api_registry):
    """First v1-only routes: the flat mount is frozen (LEGACY_ROUTES ratchet),
    so /nodes must NOT exist there."""
    assert client.get("/nodes", headers={"accept": "application/json"}).status_code == 404
    # POST falls through to the SPA static mount, which answers 405 for
    # non-GET — either way, no API route exists on the flat surface.
    assert client.post("/nodes", json={"url": PEER_A_URL}).status_code in {404, 405}


def test_peer_clients_never_use_proxy_env(api_registry):
    """Injected MockTransports bypass proxy resolution, so only this attribute
    pin can guard the production path: a user's HTTP_PROXY must never carry
    peer traffic (proxies can't reach tailnet/LAN addresses — field-debugged
    as every node reading 'unreachable' while curl worked)."""
    from makermodslab.nodes import node_registry
    from makermodslab.runners.lan_node import LanNodeJobRunner

    with node_registry._peer_client() as client:
        assert client.trust_env is False
    runner = LanNodeJobRunner.__new__(LanNodeJobRunner)
    runner._transport = None
    with runner._client(1.0) as client:
        assert client.trust_env is False
