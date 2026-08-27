"""client.nodes — mock-transport behavior plus the safe e2e paths (listing
reads nodes.json; removing a bogus id only touches the registry lookup)."""

from __future__ import annotations

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import NotFoundError
from makermodslab_sdk.resources.nodes import Node

NODE_BODY = {
    "url": "http://bench-pi:8000",
    "instance_id": "cd" * 16,
    "name": "bench-pi",
    "version": "0.1.0",
    "capabilities": {"serves_ui": True, "accepts_jobs": True},
    "status": "alive",
    "last_verified_at": 1756200000.0,
    "is_self": False,
}


def test_list_unwraps_entries():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/nodes"
        return httpx.Response(200, json={"nodes": [NODE_BODY]})

    with mock_client(handler) as client:
        nodes = client.nodes.list()
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0].name == "bench-pi"
    assert nodes[0].capabilities == {"serves_ui": True, "accepts_jobs": True}


def test_add_sends_url_and_optional_name():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json=NODE_BODY)

    with mock_client(handler) as client:
        node = client.nodes.add("http://bench-pi:8000", name="bench-pi")
        assert node.status == "alive"
        assert b"bench-pi" in seen["body"]

        client.nodes.add("http://bench-pi:8000")
        assert b'"name"' not in seen["body"]  # omitted, not null


def test_remove_hits_id_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/api/v1/nodes/{'cd' * 16}"
        return httpx.Response(200, json={"status": "removed", "instance_id": "cd" * 16})

    with mock_client(handler) as client:
        assert client.nodes.remove("cd" * 16).status == "removed"


def test_list_end_to_end(sdk_client):
    nodes = sdk_client.nodes.list()
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, Node)


def test_remove_unknown_end_to_end(sdk_client):
    with pytest.raises(NotFoundError) as excinfo:
        sdk_client.nodes.remove("0" * 32)
    assert excinfo.value.code == "node.not_found"
