"""The ``nodes`` namespace: the peer-node registry (other MakerMods Lab
servers on the LAN/tailnet, used e.g. to offload training via the
``lan_node`` runner). Peers are verified against their ``/api/v1/health``
identity document on add and re-verified for liveness — a listed peer is a
hint, never a promise.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel
from makermodslab_sdk.resources.jobs import Job, JobList, JobLogs
from makermodslab_sdk.resources.system import (
    InstallStart,
    InstallStatus,
    PolicyExtraStatus,
    RestartResult,
)


class Node(SdkModel):
    """One registry entry (mirrors makermodslab/schemas/nodes.py NodeEntry)."""

    url: str | None = None
    instance_id: str | None = None
    name: str | None = None
    version: str | None = None
    capabilities: dict[str, Any] | None = None
    status: str
    last_verified_at: float | None = None
    is_self: bool = False


class NodeRemoveResult(SdkModel):
    status: str
    instance_id: str


class NodesResource(Resource):
    """``client.nodes`` — manage this server's peer registry.

    Example:
        >>> peers = client.nodes.list()
        >>> [(n.name, n.status) for n in peers]
        [('bench-pi', 'alive')]
    """

    @operation("list_nodes")
    def list(self) -> list[Node]:
        """The registered peers (unwrapped to a plain list for ergonomics).

        Example:
            >>> [n.instance_id for n in client.nodes.list()]
            ['3f2a…']
        """
        body = self._transport.request("GET", "/api/v1/nodes", action="List nodes")
        return [Node.model_validate(entry) for entry in body["nodes"]]

    @operation("add_node")
    def add(self, url: str, name: str | None = None) -> Node:
        """Register a peer by URL. The server verifies it immediately against
        the peer's /api/v1/health identity document — an unreachable or
        non-MakerMods URL refuses with ``node.unreachable``.

        Example:
            >>> client.nodes.add("http://bench-pi:8000", name="bench-pi").status
            'alive'
        """
        payload: dict[str, Any] = {"url": url}
        if name is not None:
            payload["name"] = name
        return Node.model_validate(
            self._transport.request("POST", "/api/v1/nodes", json=payload, action="Add node")
        )

    @operation("remove_node")
    def remove(self, instance_id: str) -> NodeRemoveResult:
        """Remove a peer by its ``instance_id`` (from ``list()``)."""
        return NodeRemoveResult.model_validate(
            self._transport.request("DELETE", f"/api/v1/nodes/{instance_id}", action="Remove node")
        )

    # --- workload proxies (this server relays to the peer's own v1 API) ------
    # Peer-error semantics, by design: MUTATIONS pass the peer's coded
    # refusals through untouched; a GET whose peer cannot be reached flattens
    # to node.unreachable.

    @operation("get_node_jobs")
    def jobs(self, instance_id: str) -> JobList:
        """The peer's recent training jobs (its own /jobs listing, relayed).

        Example:
            >>> [(j.name, j.state) for j in client.nodes.jobs(peer_id).jobs]
            [('act_run', 'running')]
        """
        return JobList.model_validate(
            self._transport.request("GET", self._path(instance_id, "/jobs"), action="List node jobs")
        )

    @operation("get_node_queue")
    def job_queue(self, instance_id: str) -> JobList:
        """The peer's local training queue, in promotion order."""
        return JobList.model_validate(
            self._transport.request(
                "GET", self._path(instance_id, "/jobs/queue"), action="List node job queue"
            )
        )

    @operation("get_node_job")
    def job(self, instance_id: str, job_id: str) -> Job:
        """One job on the peer by id. Known niggle at this snapshot: a
        deleted job reads as node.unreachable here (the GET flattening)."""
        return Job.model_validate(
            self._transport.request(
                "GET", self._path(instance_id, f"/jobs/{quote(job_id, safe='')}"), action="Get node job"
            )
        )

    @operation("get_node_job_logs")
    def job_logs(self, instance_id: str, job_id: str) -> JobLogs:
        """Drain the peer job's live log tail (lines since the last call)."""
        return JobLogs.model_validate(
            self._transport.request(
                "GET",
                self._path(instance_id, f"/jobs/{quote(job_id, safe='')}/logs"),
                action="Get node job logs",
            )
        )

    @operation("stop_node_job")
    def stop_job(self, instance_id: str, job_id: str, *, expect_state: str | None = None) -> Job:
        """Stop (or cancel) a job on the peer. ``expect_state="queued"``
        cancels a queued run and refuses with job.state_changed if it was
        promoted meanwhile — the precondition that makes cancel race-safe."""
        params = {"expect_state": expect_state} if expect_state is not None else None
        return Job.model_validate(
            self._transport.request(
                "POST",
                self._path(instance_id, f"/jobs/{quote(job_id, safe='')}/stop"),
                params=params,
                action="Stop node job",
            )
        )

    @operation("delete_node_job")
    def delete_job(self, instance_id: str, job_id: str) -> None:
        """Delete a finished job's record and outputs on the peer."""
        self._transport.request(
            "DELETE",
            self._path(instance_id, f"/jobs/{quote(job_id, safe='')}"),
            action="Delete node job",
        )

    @operation("get_node_policy_extra")
    def policy_extra(self, instance_id: str, policy_type: str) -> PolicyExtraStatus:
        """Whether the PEER has a policy type's optional dependency installed
        — check before offloading a run that needs it."""
        return PolicyExtraStatus.model_validate(
            self._transport.request(
                "GET",
                self._path(instance_id, f"/policy-extra/{quote(policy_type, safe='')}"),
                action="Get node policy extra",
            )
        )

    @operation("install_node_policy_extra")
    def install_policy_extra(self, instance_id: str, policy_type: str) -> InstallStart:
        """Start installing a policy extra ON THE PEER (async; poll
        ``policy_extra_install_status``)."""
        return InstallStart.model_validate(
            self._transport.request(
                "POST",
                self._path(instance_id, f"/policy-extra/{quote(policy_type, safe='')}/install"),
                action="Install node policy extra",
            )
        )

    @operation("get_node_policy_extra_status")
    def policy_extra_install_status(self, instance_id: str, policy_type: str) -> InstallStatus:
        """Progress of the peer's policy-extra install."""
        return InstallStatus.model_validate(
            self._transport.request(
                "GET",
                self._path(instance_id, f"/policy-extra/{quote(policy_type, safe='')}/install-status"),
                action="Node policy extra install status",
            )
        )

    @operation("restart_node")
    def restart(self, instance_id: str) -> RestartResult:
        """Restart the peer's server process (refused while a live session is
        driving its hardware). The peer drops offline briefly — client.nodes
        listings show it alive again once it's back."""
        return RestartResult.model_validate(
            self._transport.request("POST", self._path(instance_id, "/restart"), action="Restart node")
        )

    @staticmethod
    def _path(instance_id: str, suffix: str = "") -> str:
        return f"/api/v1/nodes/{quote(instance_id, safe='')}{suffix}"
