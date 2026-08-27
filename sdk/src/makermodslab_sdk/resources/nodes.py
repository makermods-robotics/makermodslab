"""The ``nodes`` namespace: the peer-node registry (other MakerMods Lab
servers on the LAN/tailnet, used e.g. to offload training via the
``lan_node`` runner). Peers are verified against their ``/api/v1/health``
identity document on add and re-verified for liveness — a listed peer is a
hint, never a promise.
"""

from __future__ import annotations

from typing import Any

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


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
