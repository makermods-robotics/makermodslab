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


def test_workload_proxy_paths_and_expect_state():
    peer = "cd" * 16
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/jobs") or request.url.path.endswith("/queue"):
            return httpx.Response(200, json={"jobs": []})
        if request.url.path.endswith("/logs"):
            return httpx.Response(200, json={"logs": []})
        if request.url.path.endswith("/stop"):
            return httpx.Response(
                200,
                json={
                    "id": "j1",
                    "name": "run",
                    "state": "interrupted",
                    "config": {"dataset_repo_id": "u/d"},
                    "output_dir": "outputs/train/j1",
                    "started_at": 1.0,
                },
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith("/restart"):
            return httpx.Response(200, json={"restarting": True, "message": "bye"})
        return httpx.Response(200, json={"state": "idle", "logs": []})

    with mock_client(handler) as client:
        client.nodes.jobs(peer)
        client.nodes.job_queue(peer)
        client.nodes.job_logs(peer, "j1")
        stopped = client.nodes.stop_job(peer, "j1", expect_state="queued")
        client.nodes.delete_job(peer, "j1")
        assert client.nodes.restart(peer).restarting is True
    assert stopped.state == "interrupted"
    assert calls[0][1] == f"/api/v1/nodes/{peer}/jobs"
    assert calls[1][1] == f"/api/v1/nodes/{peer}/jobs/queue"
    assert calls[3][2] == {"expect_state": "queued"}  # cancel precondition rides as query
    assert calls[4][0] == "DELETE"


def test_node_policy_extra_proxy_paths():
    peer = "cd" * 16
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/install"):
            return httpx.Response(200, json={"started": True, "message": "installing"})
        if request.url.path.endswith("/install-status"):
            return httpx.Response(200, json={"state": "running"})
        return httpx.Response(
            200,
            json={
                "policy_type": "pi0",
                "needs_extra": True,
                "available": False,
                "package": "lerobot[pi0]",
                "install_target": "lerobot[pi0]",
                "install_hint": "install it",
            },
        )

    with mock_client(handler) as client:
        assert client.nodes.policy_extra(peer, "pi0").needs_extra is True
        assert client.nodes.install_policy_extra(peer, "pi0").started is True
        assert client.nodes.policy_extra_install_status(peer, "pi0").state == "running"
    assert paths == [
        f"/api/v1/nodes/{peer}/policy-extra/pi0",
        f"/api/v1/nodes/{peer}/policy-extra/pi0/install",
        f"/api/v1/nodes/{peer}/policy-extra/pi0/install-status",
    ]
