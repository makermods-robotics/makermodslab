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

"""The node registry: peers running MakerMods Lab on the same LAN/tailnet.

Phase 1 is deliberately static/manual: peers are added by URL (a Tailscale
source and job offload come later). A peer's identity document is its
/api/v1/health — version, instance_id, capabilities — and the registry never
trusts anything else:

- **Verify-on-add**: adding a peer performs that handshake immediately. An
  unreachable peer is an error, not a pending state; a peer reporting our own
  instance_id is a self-add and refused; a peer whose instance_id we already
  track is updated in place when its URL moved (machines change address,
  identities don't) and refused as a duplicate when nothing changed.
- **Liveness with a TTL, no background thread**: listing re-probes entries
  whose last probe is older than NODE_TTL_S. A failed re-probe marks the
  entry ``unreachable`` but keeps it — peers are only ever removed
  explicitly.
- **Persistence is url + name only** (utils/config.NODES_FILE): identity is
  re-verified live on load/probe, never served stale from disk.

Module-level singleton guarded by a threading.Lock, matching the
feature-module pattern; the clock and the httpx transport are injectable so
tests drive time and the network without sleeps or sockets.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import httpx

from .api_errors import ApiError, ErrorCode
from .utils.config import get_instance_id, load_saved_nodes, save_saved_nodes

# Identity handshake path on a peer (its own /api/v1/health).
HEALTH_PATH = "/api/v1/health"

# A peer verified within this window is trusted without a re-probe; listing
# re-probes anything older. Also the back-off floor for unreachable peers, so
# a down host is retried once per window instead of on every list().
NODE_TTL_S = 15.0

# Handshake budget per probe. LAN/tailnet peers answer /health in
# milliseconds; anything slower is as good as down for scheduling purposes.
PROBE_TIMEOUT_S = 3.0


class NodeError(Exception):
    """Base for registry refusals (each maps to one ErrorCode/status)."""


class NodeUnreachableError(NodeError):
    """The URL could not be verified: dead host, or a live one whose
    /api/v1/health answer is not a node identity document."""


class SelfAddError(NodeError):
    """The URL answered with this install's own instance_id."""


class DuplicateNodeError(NodeError):
    """The peer is already registered under this URL."""


class NodeNotFoundError(NodeError):
    """No registered peer carries the given instance_id."""


@dataclass
class PeerNode:
    """One registered peer. Identity fields are None until the first
    successful handshake of this process's lifetime (a peer loaded from disk
    starts unverified)."""

    url: str
    name: str | None = None
    instance_id: str | None = None
    version: str | None = None
    capabilities: dict[str, Any] | None = None
    status: str = "unreachable"  # "ok" | "unreachable"
    last_verified_at: float | None = None  # registry clock; last SUCCESSFUL handshake
    last_probe_at: float | None = None  # registry clock; last attempt, success or not

    def to_dict(self) -> dict[str, Any]:
        """The wire shape (schemas/nodes.py NodeEntry), minus internal fields."""
        return {
            "url": self.url,
            "instance_id": self.instance_id,
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "status": self.status,
            "last_verified_at": self.last_verified_at,
            "is_self": False,
        }


class NodeRegistry:
    """Peer table + verify/probe state machine. All public methods are
    lock-guarded; probes run with the lock held, which serializes registry
    operations — acceptable at Phase-1 scale (a handful of peers, short
    timeout) and revisited when a discovery source multiplies entries."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.BaseTransport | None = None,
        ttl: float = NODE_TTL_S,
        probe_timeout: float = PROBE_TIMEOUT_S,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._transport = transport
        self._ttl = ttl
        self._probe_timeout = probe_timeout
        self._peers: list[PeerNode] = []
        self._loaded = False

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"node url must start with http:// or https://, got {url!r}")
        return url

    def _probe(self, url: str) -> dict[str, Any]:
        """GET {url}/api/v1/health and return the identity document.

        Any transport failure, non-2xx answer, non-JSON body, or body without
        an instance_id is NodeUnreachableError — "not verifiable as a node"
        is one condition regardless of which layer failed.
        """
        try:
            with httpx.Client(transport=self._transport, timeout=self._probe_timeout) as client:
                response = client.get(url + HEALTH_PATH)
                response.raise_for_status()
                doc = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise NodeUnreachableError(f"could not verify {url}{HEALTH_PATH}: {exc}") from exc
        instance_id = doc.get("instance_id") if isinstance(doc, dict) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise NodeUnreachableError(
                f"{url}{HEALTH_PATH} answered without an instance_id — not a MakerMods Lab node"
            )
        return doc

    def _apply_handshake(self, peer: PeerNode, doc: dict[str, Any]) -> None:
        now = self._clock()
        peer.instance_id = doc["instance_id"]
        peer.version = doc.get("version")
        capabilities = doc.get("capabilities")
        peer.capabilities = capabilities if isinstance(capabilities, dict) else None
        peer.status = "ok"
        peer.last_verified_at = now
        peer.last_probe_at = now

    def _ensure_loaded(self) -> None:
        """Lazy first load of the saved url+name rows (under the lock).

        Loaded peers start unverified/unreachable; the first list() probes
        them (last_probe_at is None), restoring identity from the live peer.
        """
        if self._loaded:
            return
        self._loaded = True
        for row in load_saved_nodes():
            try:
                url = self._normalize_url(row["url"] or "")
            except ValueError:
                continue
            self._peers.append(PeerNode(url=url, name=row["name"]))

    def _save(self) -> None:
        save_saved_nodes([{"url": p.url, "name": p.name} for p in self._peers])

    def add(self, url: str, name: str | None = None) -> PeerNode:
        """Verify-on-add: handshake first, then register (or update) the peer.

        Raises ValueError (bad scheme), NodeUnreachableError, SelfAddError,
        or DuplicateNodeError (same identity already registered at this URL).
        A known identity answering from a NEW url updates that entry's url
        instead of duplicating it.
        """
        url = self._normalize_url(url)
        with self._lock:
            self._ensure_loaded()
            doc = self._probe(url)
            if doc["instance_id"] == get_instance_id():
                raise SelfAddError(f"{url} is this server (instance_id {doc['instance_id']}) — not a peer")

            existing = next((p for p in self._peers if p.instance_id == doc["instance_id"]), None)
            if existing is not None:
                if existing.url == url:
                    raise DuplicateNodeError(f"peer {doc['instance_id']} is already registered at {url}")
                # Same identity, new address: move the entry, drop any stale
                # entry that previously held this URL.
                self._peers = [p for p in self._peers if p is existing or p.url != url]
                existing.url = url
            else:
                # A URL match (e.g. an unverified entry loaded from disk, or a
                # reinstalled machine at the same address) is adopted rather
                # than duplicated.
                existing = next((p for p in self._peers if p.url == url), None)
                if existing is None:
                    existing = PeerNode(url=url)
                    self._peers.append(existing)
            if name is not None:
                existing.name = name
            self._apply_handshake(existing, doc)
            self._save()
            return replace(existing)

    def remove(self, instance_id: str) -> None:
        """Drop the peer with this instance_id; NodeNotFoundError if unknown."""
        with self._lock:
            self._ensure_loaded()
            for index, peer in enumerate(self._peers):
                if peer.instance_id == instance_id:
                    del self._peers[index]
                    self._save()
                    return
        raise NodeNotFoundError(f"no registered peer with instance_id {instance_id!r}")

    def resolve(self, instance_id: str) -> PeerNode:
        """The live peer carrying `instance_id`, for a caller about to TALK to
        it (job offload). A verification within the TTL is trusted as-is;
        anything staler — or a peer currently marked unreachable — is probed
        now, so the answer is at most TTL seconds old.

        Raises NodeNotFoundError when no registered peer carries the id.
        Entries with no identity yet (fresh loads from disk) are probed first,
        since the id being resolved may belong to one of them. Raises
        NodeUnreachableError when the peer is registered but did not answer —
        or answered as a DIFFERENT install (the machine at that URL was
        reinstalled), which for this identity is the same thing.
        """
        with self._lock:
            self._ensure_loaded()
            for candidate in self._peers:
                if candidate.instance_id is not None:
                    continue
                now = self._clock()
                try:
                    doc = self._probe(candidate.url)
                except NodeUnreachableError:
                    candidate.status = "unreachable"
                    candidate.last_probe_at = now
                else:
                    self._apply_handshake(candidate, doc)
            peer = next((p for p in self._peers if p.instance_id == instance_id), None)
            if peer is None:
                raise NodeNotFoundError(f"no registered peer with instance_id {instance_id!r}")
            now = self._clock()
            stale = peer.last_probe_at is None or now - peer.last_probe_at >= self._ttl
            if stale or peer.status != "ok":
                try:
                    doc = self._probe(peer.url)
                except NodeUnreachableError:
                    peer.status = "unreachable"
                    peer.last_probe_at = now
                    raise
                if doc["instance_id"] != instance_id:
                    peer.status = "unreachable"
                    peer.last_probe_at = now
                    raise NodeUnreachableError(
                        f"{peer.url} now answers as a different node "
                        f"({doc['instance_id']!r}, expected {instance_id!r}) — "
                        "re-add the peer to adopt its new identity"
                    )
                self._apply_handshake(peer, doc)
            return replace(peer)

    def list_nodes(self) -> list[PeerNode]:
        """Current peer table, re-probing entries whose last probe is older
        than the TTL (never-probed entries — fresh loads — probe now). A
        failed re-probe marks the entry unreachable and keeps it."""
        with self._lock:
            self._ensure_loaded()
            for peer in self._peers:
                now = self._clock()
                if peer.last_probe_at is not None and now - peer.last_probe_at < self._ttl:
                    continue
                try:
                    doc = self._probe(peer.url)
                except NodeUnreachableError:
                    peer.status = "unreachable"
                    peer.last_probe_at = now
                else:
                    self._apply_handshake(peer, doc)
            return [replace(p) for p in self._peers]


# Module-level singleton, like every feature module's state. Handlers read it
# at call time so tests can swap in a clock/transport-injected instance.
node_registry = NodeRegistry()


def handle_list_nodes() -> list[dict[str, Any]]:
    """Peer entries for GET /api/v1/nodes (the caller prepends the self entry,
    built from the same health fields the handshake reads)."""
    return [peer.to_dict() for peer in node_registry.list_nodes()]


def handle_add_node(url: str, name: str | None = None) -> dict[str, Any]:
    """Verify + add for POST /api/v1/nodes; refusals become coded ApiErrors:
    422 request.validation, 409 node.self / node.duplicate, 502 node.unreachable."""
    try:
        return node_registry.add(url, name=name).to_dict()
    except ValueError as exc:
        raise ApiError(status_code=422, detail=str(exc), code=ErrorCode.REQUEST_VALIDATION) from exc
    except SelfAddError as exc:
        raise ApiError(status_code=409, detail=str(exc), code=ErrorCode.NODE_SELF) from exc
    except DuplicateNodeError as exc:
        raise ApiError(status_code=409, detail=str(exc), code=ErrorCode.NODE_DUPLICATE) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_remove_node(instance_id: str) -> dict[str, Any]:
    """Remove for DELETE /api/v1/nodes/{instance_id}; 404 node.not_found."""
    try:
        node_registry.remove(instance_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    return {"status": "removed", "instance_id": instance_id}
