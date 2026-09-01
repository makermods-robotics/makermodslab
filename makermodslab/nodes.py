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

Peers arrive two ways — added by URL, or proposed by a pluggable discovery
source (node_sources.TailscaleSource; opt-in via --discover-tailscale). A
peer's identity document is its /api/v1/health — version, instance_id,
capabilities — and the registry never trusts anything else:

- **Verify-on-add**: adding a peer performs that handshake immediately. An
  unreachable peer is an error, not a pending state; a peer reporting our own
  instance_id is a self-add and refused; a peer whose instance_id we already
  track is updated in place when its URL moved (machines change address,
  identities don't) and refused as a duplicate when nothing changed.
- **Liveness with a TTL, no background thread**: listing re-probes entries
  whose last probe is older than NODE_TTL_S. A failed re-probe marks the
  entry ``unreachable`` but keeps it — manual peers are only ever removed
  explicitly.
- **Sources produce CANDIDATES, not peers**: discovery rides the same
  TTL-gated list pass as the probes (same injected clock, still no threads)
  and hands back bare urls. A candidate is `pending` until the verify
  handshake — the single trust path — confirms it; at most
  `discovery_probe_cap` unverified candidates are probed per pass. Discovered
  entries dedupe against manual entries and each other on verified
  instance_id (the manual record, name included, wins), and are evicted only
  when they BOTH leave their source's latest DEFINITIVE discovery AND fail
  liveness. A source that cannot answer raises (NodeSourceOutageError by
  convention): an OUTAGE, logged once per outage — the registry keeps that
  source's last known discovery set, so nothing is evicted on a guess.
- **force is the manual-refresh contract**: ``list_nodes(force=True)``
  (GET /api/v1/nodes?force=true) bypasses the TTL for THIS pass — discovery
  runs now and every known entry is probed now (`discovery_probe_cap` still
  bounds unverified candidates), so a refresh button reflects the world as it
  is, not as it was up to TTL seconds ago.
- **Persistence is url + name only, manual entries only**
  (utils/config.NODES_FILE): identity is re-verified live on load/probe,
  never served stale from disk, and sources never touch nodes.json.

Module-level singleton guarded by a threading.Lock, matching the
feature-module pattern; the clock and the httpx transport are injectable so
tests drive time and the network without sleeps or sockets.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from .api_errors import ApiError, ErrorCode
from .utils.config import get_instance_id, load_saved_nodes, save_saved_nodes

# Identity handshake path on a peer (its own /api/v1/health).
HEALTH_PATH = "/api/v1/health"

# The peer's own typed jobs listing, proxied verbatim by
# GET /api/v1/nodes/{instance_id}/jobs (see fetch_peer_jobs).
JOBS_PATH = "/api/v1/jobs"
QUEUE_PATH = "/api/v1/jobs/queue"

# The peer's system group: policy-extra install + self-restart, proxied by the
# /api/v1/nodes/{instance_id}/policy-extra/* and /restart routes.
SYSTEM_PATH = "/api/v1/system"

# A peer verified within this window is trusted without a re-probe; listing
# re-probes anything older. Also the back-off floor for unreachable peers, so
# a down host is retried once per window instead of on every list().
NODE_TTL_S = 15.0

# Handshake budget per probe. LAN/tailnet peers answer /health in
# milliseconds; anything slower is as good as down for scheduling purposes.
PROBE_TIMEOUT_S = 3.0

# The `source` of a hand-added peer (vs. a discovery source's source_id).
MANUAL_SOURCE = "manual"

# Unverified candidates probed per discovery pass, at most. The verify
# handshake's short timeout and non-node rejection already filter non-MakerMods
# machines; the cap keeps a big tailnet from turning one list() into a storm.
DISCOVERY_PROBE_CAP = 8

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredPeer:
    """One candidate a discovery source proposes: an address, nothing more.

    Candidates carry no identity — the registry's verify handshake against the
    url's /api/v1/health is the single trust path.
    """

    url: str
    name: str | None = None


class NodeSource(Protocol):
    """A pluggable peer-discovery source (e.g. node_sources.TailscaleSource).

    `source_id` labels the entries it produces (the wire `source` field);
    `discover()` returns the CURRENT candidate set — the registry calls it on
    the TTL cadence and diffs successive answers, so a source holds no state
    about what it reported before. The return value is a DEFINITIVE answer:
    [] means "there really are no peers" and makes this source's absent
    entries evict-eligible. A source that cannot answer right now must raise
    instead (NodeSourceOutageError by convention; any exception is treated
    the same, never fatal) — the registry logs the outage and keeps the
    source's last known discovery set.
    """

    source_id: str

    def discover(self) -> list[DiscoveredPeer]: ...


class NodeSourceOutageError(RuntimeError):
    """A discovery source could not produce an answer (transient: timeout,
    broken output, daemon not answering). Distinct from a definitive empty
    discovery — on an outage the registry evicts nothing."""


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


class PeerJobRefusalError(NodeError):
    """The peer ANSWERED a forwarded stop/delete — with an HTTP refusal of its
    own (409 job.has_queued_dependents, 404 job.not_found, …). Carries the
    peer's status and parsed body so the proxy can pass the verdict through
    verbatim; transport-level failure stays NodeUnreachableError."""

    def __init__(self, status_code: int, body: Any) -> None:
        super().__init__(f"peer refused with HTTP {status_code}")
        self.status_code = status_code
        self.body = body


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
    # "ok" | "unreachable" | "pending" — pending only for a discovered
    # candidate that has never been probed (manual entries are probed on the
    # very next list(), so they never surface it).
    status: str = "unreachable"
    last_verified_at: float | None = None  # registry clock; last SUCCESSFUL handshake
    last_probe_at: float | None = None  # registry clock; last attempt, success or not
    # Wall-clock (time.time) sibling of last_verified_at, stamped by the same
    # successful handshakes. The monotonic field keeps its registry-clock
    # semantics for TTL bookkeeping; this one exists so clients can render a
    # human "last seen X ago" against their own wall clock.
    last_seen_at: float | None = None
    source: str = MANUAL_SOURCE  # MANUAL_SOURCE, or the discovering source_id

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
            "last_seen_at": self.last_seen_at,
            "is_self": False,
            "source": self.source,
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
        wall_clock: Callable[[], float] = time.time,
        transport: httpx.BaseTransport | None = None,
        ttl: float = NODE_TTL_S,
        probe_timeout: float = PROBE_TIMEOUT_S,
        discovery_probe_cap: int = DISCOVERY_PROBE_CAP,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._wall_clock = wall_clock
        self._transport = transport
        self._ttl = ttl
        self._probe_timeout = probe_timeout
        self._discovery_probe_cap = discovery_probe_cap
        self._peers: list[PeerNode] = []
        self._loaded = False
        self._sources: list[NodeSource] = []
        # Per source_id: when discover() last ran (the TTL gates re-discovery
        # exactly as it gates re-probes) and the urls it last reported (the
        # "still discovered" half of the eviction rule).
        self._last_discovery_at: dict[str, float] = {}
        self._latest_discovery: dict[str, set[str]] = {}
        # Sources currently in outage whose failure has been logged — one
        # line per outage, re-armed by the next successful discover().
        self._source_outage_logged: set[str] = set()
        # Discovered urls that answered with OUR instance_id: our own tailnet
        # address, remembered so re-discovery doesn't re-probe ourselves.
        self._self_urls: set[str] = set()

    def register_source(self, source: NodeSource) -> None:
        """Plug in a discovery source; its discover() runs inside list_nodes()
        on the TTL cadence. One source per source_id."""
        with self._lock:
            if any(s.source_id == source.source_id for s in self._sources):
                raise ValueError(f"a node source with source_id {source.source_id!r} is already registered")
            self._sources.append(source)

    def source_ids(self) -> list[str]:
        """The registered discovery-source ids, in registration order — the
        wire `sources` field, so clients can say whether discovery is even on."""
        with self._lock:
            return [s.source_id for s in self._sources]

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
            with self._peer_client() as client:
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
        peer.last_seen_at = self._wall_clock()

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
        """Persist MANUAL entries only — discovered peers are re-found live by
        their source and must never leak into nodes.json."""
        save_saved_nodes([{"url": p.url, "name": p.name} for p in self._peers if p.source == MANUAL_SOURCE])

    def add(self, url: str, name: str | None = None) -> PeerNode:
        """Verify-on-add: handshake first, then register (or update) the peer.

        Raises ValueError (bad scheme), NodeUnreachableError, SelfAddError,
        or DuplicateNodeError (same identity already MANUALLY registered at
        this URL). A known identity answering from a NEW url updates that
        entry's url instead of duplicating it, and adding a discovered entry
        by hand PROMOTES it to a manual one (persisted, never evicted).
        """
        url = self._normalize_url(url)
        with self._lock:
            self._ensure_loaded()
            doc = self._probe(url)
            if doc["instance_id"] == get_instance_id():
                raise SelfAddError(f"{url} is this server (instance_id {doc['instance_id']}) — not a peer")

            existing = next((p for p in self._peers if p.instance_id == doc["instance_id"]), None)
            if existing is not None:
                if existing.url == url and existing.source == MANUAL_SOURCE:
                    raise DuplicateNodeError(f"peer {doc['instance_id']} is already registered at {url}")
                # Same identity at a new address: move the entry, dropping any
                # stale entry that previously held this URL.
                self._peers = [p for p in self._peers if p is existing or p.url != url]
                existing.url = url
            else:
                # A URL match (e.g. an unverified entry loaded from disk, a
                # discovered candidate, or a reinstalled machine at the same
                # address) is adopted rather than duplicated.
                existing = next((p for p in self._peers if p.url == url), None)
                if existing is None:
                    existing = PeerNode(url=url)
                    self._peers.append(existing)
            existing.source = MANUAL_SOURCE
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
            for candidate in list(self._peers):
                if candidate.instance_id is not None:
                    continue
                if not self._probe_and_integrate(candidate):
                    self._peers = [p for p in self._peers if p is not candidate]
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

    def fetch_peer_jobs(self, instance_id: str) -> Any:
        """The peer's own GET /api/v1/jobs body, for the workload proxy."""
        return self._fetch_peer_path(instance_id, JOBS_PATH)

    def fetch_peer_queue(self, instance_id: str) -> Any:
        """The peer's own GET /api/v1/jobs/queue body — the EXACT queue,
        where the jobs page is limited and could undercount queued runs."""
        return self._fetch_peer_path(instance_id, QUEUE_PATH)

    @staticmethod
    def _job_path(job_id: str) -> str:
        """The peer's own path for one job; the id is data and gets quoted."""
        return f"{JOBS_PATH}/{quote(job_id, safe='')}"

    def fetch_peer_job(self, instance_id: str, job_id: str) -> Any:
        """The peer's own GET /api/v1/jobs/{job_id} body (its JobRecord)."""
        return self._fetch_peer_path(instance_id, self._job_path(job_id))

    def fetch_peer_job_logs(self, instance_id: str, job_id: str) -> Any:
        """The peer's own GET /api/v1/jobs/{job_id}/logs body. The peer drains
        its runner's live queue per call, so this is inherently incremental —
        each read returns only the lines that arrived since the last one."""
        return self._fetch_peer_path(instance_id, self._job_path(job_id) + "/logs")

    def stop_peer_job(self, instance_id: str, job_id: str, expect_state: str | None = None) -> Any:
        """Forward POST /api/v1/jobs/{job_id}/stop to the peer, expect_state
        precondition included, so the peer's own stale-click guard still holds."""
        params = {"expect_state": expect_state} if expect_state is not None else None
        return self._send_peer_request(instance_id, "POST", self._job_path(job_id) + "/stop", params=params)

    def delete_peer_job(self, instance_id: str, job_id: str) -> Any:
        """Forward DELETE /api/v1/jobs/{job_id} to the peer (204, no body)."""
        return self._send_peer_request(instance_id, "DELETE", self._job_path(job_id))

    @staticmethod
    def _policy_extra_path(policy_type: str) -> str:
        """The peer's own path for one policy's extra; the type is data."""
        return f"{SYSTEM_PATH}/policy-extra/{quote(policy_type, safe='')}"

    def fetch_peer_policy_extra(self, instance_id: str, policy_type: str) -> Any:
        """The peer's own GET /api/v1/system/policy-extra/{policy_type} body —
        whether the extra a policy needs is importable in THE PEER's
        environment, which is the one the offloaded run will import from."""
        return self._fetch_peer_path(instance_id, self._policy_extra_path(policy_type))

    def fetch_peer_policy_extra_status(self, instance_id: str, policy_type: str) -> Any:
        """The peer's own GET .../policy-extra/{policy_type}/install-status
        body. The peer drains pending log lines per call, so this is
        incremental, like the job-log proxy."""
        return self._fetch_peer_path(instance_id, self._policy_extra_path(policy_type) + "/install-status")

    def install_peer_policy_extra(self, instance_id: str, policy_type: str) -> Any:
        """Forward POST .../policy-extra/{policy_type}/install to the peer:
        the pip subprocess runs THERE, in the peer's own environment."""
        return self._send_peer_request(instance_id, "POST", self._policy_extra_path(policy_type) + "/install")

    def restart_peer(self, instance_id: str) -> Any:
        """Forward POST /api/v1/system/restart to the peer. The peer answers
        first and re-execs after a grace delay, so a 200 here means the
        restart is SCHEDULED — the registry will see the node flap
        unreachable and recover on its own probes."""
        return self._send_peer_request(instance_id, "POST", SYSTEM_PATH + "/restart")

    def _peer_client(self) -> httpx.Client:
        """An httpx client for PEER traffic: trust_env=False, always.

        Peers live on tailnet CGNAT (100.64/10) or RFC1918 LAN addresses that
        no HTTP proxy can reach — with the default trust_env=True, a user's
        HTTP_PROXY/ALL_PROXY silently reroutes every probe into a proxy that
        cannot deliver it, and the whole registry reads "unreachable" while
        curl (which ignores uppercase HTTP_PROXY for http URLs) works fine.
        Field-debugged on exactly that symptom. Proxies stay honored where
        they belong: Hub traffic (huggingface_hub) is untouched.
        """
        return httpx.Client(transport=self._transport, timeout=self._probe_timeout, trust_env=False)

    def _fetch_peer_path(self, instance_id: str, path: str) -> Any:
        """One read of a peer, passed through verbatim.

        resolve() is the pre-flight (probing if stale, so a dead or
        reincarnated peer refuses before we ask it anything); the GET itself
        runs OUTSIDE the registry lock — it is a read of the peer, not a
        mutation of the table. The body is returned verbatim (parsed JSON,
        not re-modeled): the peer runs this same code, and shaping is the
        route's response_model's job. Raises NodeNotFoundError from resolve,
        NodeUnreachableError when either the resolve probe or the request
        fails.
        """
        peer = self.resolve(instance_id)
        try:
            with self._peer_client() as client:
                response = client.get(peer.url + path)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise NodeUnreachableError(f"could not fetch {peer.url}{path}: {exc}") from exc

    def _send_peer_request(self, instance_id: str, method: str, path: str, params: dict | None = None) -> Any:
        """One MUTATING call to a peer (POST/DELETE), refusals passed through.

        Same pre-flight (resolve) and client discipline as _fetch_peer_path,
        with one deliberate difference in what counts as "unreachable": a GET
        proxy reads state the peer serves unconditionally, so ANY failure —
        transport or HTTP — means the peer couldn't be read and maps to
        node.unreachable. A mutation is a request the peer may REFUSE for its
        own reasons (409 job.has_queued_dependents, 404 job.not_found, …), and
        that verdict belongs to the caller with the peer's status and body:
        only transport-level failure raises NodeUnreachableError; an HTTP
        error status raises PeerJobRefusalError carrying both.
        """
        peer = self.resolve(instance_id)
        try:
            with self._peer_client() as client:
                response = client.request(method, peer.url + path, params=params)
        except httpx.HTTPError as exc:
            raise NodeUnreachableError(f"could not reach {peer.url}{path}: {exc}") from exc
        if response.is_error:
            try:
                body = response.json()
            except ValueError:
                body = {"detail": response.text}
            raise PeerJobRefusalError(response.status_code, body)
        if not response.content:
            return None  # the peer's own DELETE answers 204 No Content
        try:
            return response.json()
        except ValueError as exc:
            raise NodeUnreachableError(f"{peer.url}{path} answered with a non-JSON body: {exc}") from exc

    def _probe_and_integrate(self, peer: PeerNode) -> bool:
        """Probe one entry and fold the answer into the table.

        Returns False when the entry should be DROPPED: a discovered candidate
        that answered as this server (our own tailnet address — remembered so
        it isn't re-adopted next discovery) or as an identity another entry
        already carries (dedupe on verified instance_id; the established
        entry — manual always included — is the record, though a discovered
        record that is currently down follows the identity to this fresher
        address). Manual entries are never dropped: a failed probe marks them
        unreachable, exactly as before sources existed.
        """
        now = self._clock()
        try:
            doc = self._probe(peer.url)
        except NodeUnreachableError:
            peer.status = "unreachable"
            peer.last_probe_at = now
            return True
        if peer.source != MANUAL_SOURCE:
            if doc["instance_id"] == get_instance_id():
                self._self_urls.add(peer.url)
                return False
            twin = next(
                (p for p in self._peers if p is not peer and p.instance_id == doc["instance_id"]), None
            )
            if twin is not None:
                if twin.source != MANUAL_SOURCE and twin.status != "ok":
                    twin.url = peer.url
                    self._apply_handshake(twin, doc)
                return False
        self._apply_handshake(peer, doc)
        return True

    def _run_discovery(self, force: bool = False) -> None:
        """Ask each source (whose last run is older than the TTL; every
        source when `force`) for its current candidates and adopt the new
        urls as `pending` entries.

        Candidates dedupe on url at this stage (identity-level dedupe needs a
        handshake and happens in _probe_and_integrate). A raising source is
        an OUTAGE, logged once per outage, never fatal: its last known
        discovery set stays in place, so the gone-from-discovery half of
        eviction is skipped for its entries until it answers again.
        """
        for source in self._sources:
            now = self._clock()
            last = self._last_discovery_at.get(source.source_id)
            if not force and last is not None and now - last < self._ttl:
                continue
            self._last_discovery_at[source.source_id] = now
            try:
                candidates = source.discover()
            except Exception as exc:
                if source.source_id not in self._source_outage_logged:
                    logger.warning(
                        "node source %r discovery outage (keeping its known peers): %s",
                        source.source_id,
                        exc,
                    )
                    self._source_outage_logged.add(source.source_id)
                continue
            self._source_outage_logged.discard(source.source_id)
            urls: set[str] = set()
            for candidate in candidates:
                try:
                    url = self._normalize_url(candidate.url)
                except ValueError:
                    continue
                urls.add(url)
                if url in self._self_urls:
                    continue
                existing = next((p for p in self._peers if p.url == url), None)
                if existing is not None:
                    if existing.source != MANUAL_SOURCE and existing.name is None:
                        existing.name = candidate.name  # manual names always win
                    continue
                self._peers.append(
                    PeerNode(url=url, name=candidate.name, source=source.source_id, status="pending")
                )
            self._latest_discovery[source.source_id] = urls

    def _probe_due_peers(self, force: bool = False) -> None:
        """Re-probe entries whose last probe is older than the TTL
        (never-probed entries — fresh loads and new candidates — probe now;
        EVERY entry probes now when `force`).

        At most `discovery_probe_cap` UNVERIFIED discovered candidates are
        probed per pass — forced or not; the rest stay `pending` for later
        passes. Verified peers — manual or discovered — always re-probe on
        the TTL, as before.
        """
        budget = self._discovery_probe_cap
        for peer in list(self._peers):
            now = self._clock()
            if not force and peer.last_probe_at is not None and now - peer.last_probe_at < self._ttl:
                continue
            if peer.source != MANUAL_SOURCE and peer.instance_id is None:
                if budget <= 0:
                    continue
                budget -= 1
            if not self._probe_and_integrate(peer):
                self._peers = [p for p in self._peers if p is not peer]

    def _evict_lost_discovered(self) -> None:
        """Drop discovered entries that BOTH left their source's latest
        discovery AND fail liveness. One signal alone never evicts (a peer the
        source blinks on but that still answers stays; a discovered peer that
        merely stops answering stays `unreachable` like a manual one), and
        manual entries are never evicted at all."""
        self._peers = [
            p
            for p in self._peers
            if p.source == MANUAL_SOURCE
            or p.status != "unreachable"
            or p.url in self._latest_discovery.get(p.source, set())
        ]

    def list_nodes(self, force: bool = False) -> list[PeerNode]:
        """Current peer table: run due discovery, probe due entries (TTL-gated,
        unverified-candidate probes capped), evict lost discovered peers. A
        failed re-probe marks an entry unreachable and keeps it.

        `force` is the manual-refresh contract: this ONE pass ignores the TTL
        — discovery runs now, every known entry is probed now (the cap still
        bounds unverified candidates) — so with the local tailnet gone a
        forced pass clears the discovered entries instead of waiting out the
        poll cadence."""
        with self._lock:
            self._ensure_loaded()
            self._run_discovery(force=force)
            self._probe_due_peers(force=force)
            self._evict_lost_discovered()
            return [replace(p) for p in self._peers]


# Module-level singleton, like every feature module's state. Handlers read it
# at call time so tests can swap in a clock/transport-injected instance.
node_registry = NodeRegistry()


def register_sources_from_env(registry: NodeRegistry | None = None) -> None:
    """Register the discovery sources the environment opts into (OFF by
    default). The launcher's --discover-tailscale flag sets
    MAKERMODSLAB_DISCOVER_TAILSCALE=1 before uvicorn imports the server — the
    MAKERMODSLAB_NO_UI precedent — and this module reads it at import."""
    registry = node_registry if registry is None else registry
    if os.environ.get("MAKERMODSLAB_DISCOVER_TAILSCALE") == "1":
        from .node_sources import TailscaleSource

        registry.register_source(TailscaleSource())


register_sources_from_env()


def handle_list_nodes(force: bool = False) -> list[dict[str, Any]]:
    """Peer entries for GET /api/v1/nodes (the caller prepends the self entry,
    built from the same health fields the handshake reads). `force` is the
    route's ?force=true — the manual-refresh pass that bypasses the TTL."""
    return [peer.to_dict() for peer in node_registry.list_nodes(force=force)]


def handle_list_node_sources() -> list[str]:
    """The wire `sources` field: registered discovery-source ids (e.g.
    ["tailscale"] when --discover-tailscale was on; [] otherwise)."""
    return node_registry.source_ids()


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


def handle_get_node_jobs(instance_id: str) -> Any:
    """Workload proxy for GET /api/v1/nodes/{instance_id}/jobs: the peer's own
    typed jobs listing, passed through. 404 node.not_found for an unknown
    instance_id; 502 node.unreachable when the peer doesn't answer (the
    pre-flight resolve or the jobs request itself)."""
    try:
        return node_registry.fetch_peer_jobs(instance_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_get_node_queue(instance_id: str) -> Any:
    """Queue proxy for GET /api/v1/nodes/{instance_id}/jobs/queue; same error
    mapping as the jobs proxy."""
    try:
        return node_registry.fetch_peer_queue(instance_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_get_node_job(instance_id: str, job_id: str) -> Any:
    """Drill-in proxy for GET /api/v1/nodes/{instance_id}/jobs/{job_id}: the
    peer's own JobRecord, passed through. Same error mapping as the jobs
    proxy — 404 node.not_found for an unknown instance_id, 502
    node.unreachable for ANY failure to read the peer (the peer's own 404 for
    an unknown job included)."""
    try:
        return node_registry.fetch_peer_job(instance_id, job_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_get_node_job_logs(instance_id: str, job_id: str) -> Any:
    """Log-tail proxy for GET /api/v1/nodes/{instance_id}/jobs/{job_id}/logs;
    incremental per call (the peer drains its live queue), same error mapping
    as the record proxy above."""
    try:
        return node_registry.fetch_peer_job_logs(instance_id, job_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def _peer_refusal_to_api_error(exc: PeerJobRefusalError) -> ApiError:
    """The peer's refusal, re-raised with ITS status and body fields. The peer
    runs this same code, so its error bodies are {detail, code?, details?} —
    mapped onto ApiError's exact serialization, the body survives verbatim."""
    body = exc.body if isinstance(exc.body, dict) else {"detail": exc.body}
    return ApiError(
        status_code=exc.status_code,
        detail=body.get("detail"),
        code=body.get("code"),
        details=body.get("details"),
    )


def handle_stop_node_job(instance_id: str, job_id: str, expect_state: str | None = None) -> Any:
    """Forwarded stop for POST /api/v1/nodes/{instance_id}/jobs/{job_id}/stop.

    Error stance — subtly DIFFERENT from the GET proxies above: there, any
    HTTP error from the peer counts as "couldn't read the peer" (502
    node.unreachable). A stop is a request the peer may refuse for its own
    reasons (409 job.state_changed / job.has_queued_dependents, 404
    job.not_found, …), and those coded refusals pass through with the PEER's
    status and body, never re-wrapped — only transport-level failure is 502
    node.unreachable. 404 node.not_found still means the NODE is unknown."""
    try:
        return node_registry.stop_peer_job(instance_id, job_id, expect_state=expect_state)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except PeerJobRefusalError as exc:
        raise _peer_refusal_to_api_error(exc) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_delete_node_job(instance_id: str, job_id: str) -> None:
    """Forwarded delete for DELETE /api/v1/nodes/{instance_id}/jobs/{job_id}
    (204 on success, like the peer's own delete). Same passthrough stance as
    the stop above: the peer's coded refusals (409 job.has_queued_dependents
    etc.) keep THEIR status and body; only transport failure is 502
    node.unreachable, and 404 node.not_found names an unknown node."""
    try:
        node_registry.delete_peer_job(instance_id, job_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except PeerJobRefusalError as exc:
        raise _peer_refusal_to_api_error(exc) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_get_node_policy_extra(instance_id: str, policy_type: str) -> Any:
    """Extra-status proxy for GET /api/v1/nodes/{instance_id}/policy-extra/
    {policy_type}: whether the PEER's environment can import what the policy
    needs — the local answer is irrelevant to an offloaded run. Same error
    mapping as the other GET proxies."""
    try:
        return node_registry.fetch_peer_policy_extra(instance_id, policy_type)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_get_node_policy_extra_status(instance_id: str, policy_type: str) -> Any:
    """Install-progress proxy for GET .../policy-extra/{policy_type}/
    install-status; incremental per call (the peer drains its pending log
    lines), same error mapping as the GET proxies."""
    try:
        return node_registry.fetch_peer_policy_extra_status(instance_id, policy_type)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_install_node_policy_extra(instance_id: str, policy_type: str) -> Any:
    """Forwarded install for POST .../policy-extra/{policy_type}/install: the
    pip subprocess runs on the peer, in the environment its jobs import from.
    Mutation stance, like the stop/delete proxies: the peer's own refusals
    pass through with THEIR status and body; only transport failure is 502
    node.unreachable, and 404 node.not_found names an unknown node."""
    try:
        return node_registry.install_peer_policy_extra(instance_id, policy_type)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except PeerJobRefusalError as exc:
        raise _peer_refusal_to_api_error(exc) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_restart_node(instance_id: str) -> Any:
    """Forwarded restart for POST /api/v1/nodes/{instance_id}/restart. Same
    mutation stance: the peer's coded refusals (409 session.held /
    robot.busy.training / system.restart_unsupported — and a plain 404 from
    a peer too old to have the endpoint) keep THEIR status and body; only
    transport failure is 502 node.unreachable."""
    try:
        return node_registry.restart_peer(instance_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    except PeerJobRefusalError as exc:
        raise _peer_refusal_to_api_error(exc) from exc
    except NodeUnreachableError as exc:
        raise ApiError(status_code=502, detail=str(exc), code=ErrorCode.NODE_UNREACHABLE) from exc


def handle_remove_node(instance_id: str) -> dict[str, Any]:
    """Remove for DELETE /api/v1/nodes/{instance_id}; 404 node.not_found."""
    try:
        node_registry.remove(instance_id)
    except NodeNotFoundError as exc:
        raise ApiError(status_code=404, detail=str(exc), code=ErrorCode.NODE_NOT_FOUND) from exc
    return {"status": "removed", "instance_id": instance_id}
