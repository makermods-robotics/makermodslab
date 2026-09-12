"""client.describe() (the one-call orientation snapshot) and the shipped
cheatsheet (python -m makermodslab_sdk.docs)."""

from __future__ import annotations

import httpx
from helpers import mock_client
from makermodslab_sdk.client import RESOURCE_CLASSES
from makermodslab_sdk.describe import ServerSnapshot
from makermodslab_sdk.docs import cheatsheet

HEALTH = {
    "status": "ok",
    "message": "up",
    "version": "0.1.0",
    "instance_id": "ab" * 16,
    "capabilities": {"serves_ui": True, "accepts_jobs": True},
}
JOB = {
    "id": "job-1",
    "job_number": 4,
    "name": "act_run",
    "state": "running",
    "config": {"dataset_repo_id": "u/d", "policy_type": "act", "steps": 100, "batch_size": 8},
    "output_dir": "outputs/train/job-1",
    "started_at": 1756200000.0,
    "metrics": {"current_step": 10, "total_steps": 100},
}


def scripted(responses):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in responses:
            return responses[path]
        return httpx.Response(500, json={"detail": f"unexpected {path}"})

    return handler


def test_describe_composes_all_sections():
    handler = scripted(
        {
            "/api/v1/health": httpx.Response(200, json=HEALTH),
            "/api/v1/sessions/current": httpx.Response(200, json={"session": None, "last_ended": None}),
            "/api/v1/jobs": httpx.Response(200, json={"jobs": [JOB]}),
            "/api/v1/nodes": httpx.Response(200, json={"nodes": []}),
        }
    )
    with mock_client(handler) as client:
        snap = client.describe()
    assert isinstance(snap, ServerSnapshot)
    assert snap.health is not None and snap.health.version == "0.1.0"
    assert snap.session is None
    assert [job.id for job in snap.running_jobs] == ["job-1"]
    assert snap.errors == {}
    text = snap.summary()
    assert "v0.1.0" in text
    assert "robot is free" in text
    assert "step 10/100" in text


def test_describe_survives_a_failing_section():
    handler = scripted(
        {
            "/api/v1/health": httpx.Response(200, json=HEALTH),
            "/api/v1/sessions/current": httpx.Response(200, json={"session": None, "last_ended": None}),
            "/api/v1/jobs": httpx.Response(500, json={"detail": "registry exploded"}),
            "/api/v1/nodes": httpx.Response(200, json={"nodes": []}),
        }
    )
    with mock_client(handler) as client:
        snap = client.describe()
    assert snap.health is not None
    assert "jobs" in snap.errors and "registry exploded" in snap.errors["jobs"]
    assert "jobs unavailable" in snap.summary()


def test_describe_end_to_end(sdk_client):
    snap = sdk_client.describe()
    assert snap.health is not None and snap.health.status == "ok"
    assert snap.errors == {}
    assert isinstance(snap.summary(), str) and snap.summary()


def test_cheatsheet_covers_every_operation_method():
    text = cheatsheet()
    for tag, cls in RESOURCE_CLASSES.items():
        assert f"### client.{tag}" in text
        for name, member in vars(cls).items():
            if hasattr(member, "_operation_id"):
                assert f"- {name}(" in text, f"cheatsheet missing {tag}.{name}"
    # The load-bearing patterns are present…
    assert "with client.sessions.teleoperate" in text
    assert "Next step" in text
    assert "REFETCH HINTS" in text
    # …and the whole thing stays context-budget sized.
    assert len(text) < 20_000
