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
    re-verification instead of every list() hitting the network.
    """

    def __init__(self) -> None:
        self.peers: dict[str, dict] = {}
        self.down: set[str] = set()
        self.probes: dict[str, int] = {}

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/health", f"unexpected probe path: {request.url}"
            base = str(request.url).removesuffix("/api/v1/health")
            self.probes[base] = self.probes.get(base, 0) + 1
            if base in self.down:
                raise httpx.ConnectError("connection refused", request=request)
            if base not in self.peers:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json=self.peers[base])

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
def registry(local_identity, nodes_file, network, clock):
    from makermodslab.nodes import NodeRegistry

    return NodeRegistry(clock=clock, transport=network.transport())


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
# Endpoints (v1-only surface)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_registry(monkeypatch: pytest.MonkeyPatch, local_identity, nodes_file, network, clock):
    """Swap the module singleton for a test-driven registry; handlers look the
    global up at call time, so the routes see this instance."""
    from makermodslab import nodes

    reg = nodes.NodeRegistry(clock=clock, transport=network.transport())
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


def test_nodes_surface_is_v1_only(client, api_registry):
    """First v1-only routes: the flat mount is frozen (LEGACY_ROUTES ratchet),
    so /nodes must NOT exist there."""
    assert client.get("/nodes", headers={"accept": "application/json"}).status_code == 404
    # POST falls through to the SPA static mount, which answers 405 for
    # non-GET — either way, no API route exists on the flat surface.
    assert client.post("/nodes", json={"url": PEER_A_URL}).status_code in {404, 405}
