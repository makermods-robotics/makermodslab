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

"""Response models for the "nodes" route group (the peer-node registry).

See the package docstring for the fidelity rules; the shape authority is
nodes.py PeerNode.to_dict plus the self entry server.py list_nodes builds
from the health document.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

__all__ = [
    "NodeEntry",
    "NodeListResponse",
    "NodeRemoveResponse",
]


class NodeEntry(BaseModel):
    """One node (nodes.py PeerNode.to_dict; server.py list_nodes for self).

    Nulls are meaningful, so None is never excluded: url is null only on the
    self entry (a server doesn't know its own external address); instance_id,
    version, capabilities and last_verified_at are null on a saved peer that
    hasn't completed a handshake since this process started — and on a
    discovered candidate the verify handshake hasn't confirmed yet, which
    additionally carries status "pending" (never "pending" with a non-null
    instance_id). capabilities is a plain dict, not HealthCapabilities — it's
    the PEER's self-reported block, and a version-skewed peer must not fail
    our response validation. last_verified_at is the server's monotonic
    registry clock (ordering and freshness relative to other entries, not
    wall-clock time). source says how the entry got here: "manual" for the
    self entry and hand-added peers (the only ones persisted), a source id
    for discovery-source candidates.
    """

    url: str | None
    instance_id: str | None
    name: str | None
    version: str | None
    capabilities: dict[str, Any] | None
    status: Literal["ok", "unreachable", "pending"]
    last_verified_at: float | None
    is_self: bool
    source: Literal["manual", "tailscale"]


class NodeListResponse(BaseModel):
    """server.py list_nodes — the self entry first, then registered peers."""

    nodes: list[NodeEntry]


class NodeRemoveResponse(BaseModel):
    """nodes.py handle_remove_node."""

    status: Literal["removed"]
    instance_id: str
