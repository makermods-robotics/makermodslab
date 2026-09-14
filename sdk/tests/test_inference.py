"""client.inference — the coaching (DAgger) operator verbs, and the
session-scoped coaching_command twin. MockTransport-only: coaching runs need
live hardware."""

from __future__ import annotations

import json

import httpx
from helpers import mock_client
from makermodslab_sdk.resources.inference import CoachingCommandResult

VERBS = {
    "takeover": "/api/v1/coaching-takeover",
    "handback": "/api/v1/coaching-handback",
    "drop_last": "/api/v1/coaching-drop-last",
    "hold": "/api/v1/coaching-hold",
    "resume": "/api/v1/coaching-resume",
    "reset": "/api/v1/coaching-reset",
    "recovered": "/api/v1/coaching-recovered",
    "cancel": "/api/v1/coaching-cancel",
}


def test_every_verb_hits_its_path_and_parses():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"success": True, "message": "ok"})

    with mock_client(handler) as client:
        for method_name in VERBS:
            result = getattr(client.inference, method_name)()
            assert isinstance(result, CoachingCommandResult)
            assert result.success is True
    assert seen == [("POST", path) for path in VERBS.values()]


def test_no_session_is_a_soft_failure_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "message": "no coaching session running"})

    with mock_client(handler) as client:
        result = client.inference.takeover()
    assert result.success is False
    assert "no coaching session" in result.message


def test_session_scoped_coaching_command_sends_verb_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"result": {"success": True, "message": "took over"}})

    with mock_client(handler) as client:
        result = client.sessions.coaching_command("sess-1", "takeover")
    assert seen["path"] == "/api/v1/sessions/sess-1/coaching"
    assert seen["body"] == {"command": "takeover"}
    assert result.result["success"] is True
