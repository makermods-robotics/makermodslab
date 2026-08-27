"""The sessions namespace: raw surface, the lease-holding ActiveSession, and
the per-kind sugar.

House rules, absolute: the e2e tests (sdk_client fixture, real app) only ever
touch reads and refusals that fire BEFORE any hardware path — current(), a
stop of a bogus id, a start naming a robot record that cannot exist. Every
lease/warning/loss flow runs against MockTransport-scripted sequences with
bodies matching makermodslab/schemas/sessions.py; nothing here sleeps —
heartbeat TICKS are driven directly, and the one thread-lifecycle test
pre-sets the stop event so the loop exits without waiting.
"""

from __future__ import annotations

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import NotFoundError, SessionHeldError
from makermodslab_sdk.resources.sessions import (
    CurrentSession,
    SessionInfo,
    StartedSession,
    StoppedSession,
)

OWNER = "sdk:test-host:1234:abcd"


def session_body(
    *,
    id: str = "sess-1",
    kind: str = "teleoperation",
    robot: str | None = "bench",
    owner: str | None = OWNER,
    phase: str | None = "running",
    lease: bool = True,
    timeout_s: float = 60.0,
    expires_in_s: float = 60.0,
) -> dict:
    return {
        "id": id,
        "kind": kind,
        "robot": robot,
        "owner": owner,
        "started_at": 1000.0,
        "revision": 1,
        "phase": phase,
        "lease": ({"owner": owner, "timeout_s": timeout_s, "expires_in_s": expires_in_s} if lease else None),
    }


class Script:
    """MockTransport handler scripted per (method, path): each call pops the
    next response — a strict sequence, so an unscripted or extra request is a
    loud failure, and every request body is recorded for assertions."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], list] = {}
        self.requests: list[httpx.Request] = []

    def add(self, method: str, path: str, *responses) -> Script:
        self._routes.setdefault((method, path), []).extend(responses)
        return self

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self._routes.get((request.method, request.url.path))
        assert queue, f"unscripted request: {request.method} {request.url.path}"
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)


# --- end to end (real app; reads and pre-hardware refusals ONLY) -------------


def test_current_end_to_end(sdk_client):
    now = sdk_client.sessions.current()
    assert isinstance(now, CurrentSession)
    assert now.session is None  # nothing in this suite ever starts a session
    assert now.last_ended is None


def test_stop_bogus_id_end_to_end(sdk_client):
    """A stop of an id that names no session is the coded already-gone 404."""
    with pytest.raises(NotFoundError) as excinfo:
        sdk_client.sessions.stop("bogus-id")
    err = excinfo.value
    assert err.status == 404
    assert err.code == "session.not_found"
    assert "Next step:" in str(err)


# --- raw surface (scripted) --------------------------------------------------


def test_start_sends_body_and_returns_warnings():
    script = Script().add(
        "POST",
        "/api/v1/sessions",
        (201, {"session": session_body(), "warnings": ["left arm EEPROM disagrees with calibration"]}),
    )
    with mock_client(script) as client:
        started = client.sessions.start(
            "teleoperation",
            "bench",
            owner=OWNER,
            options={"skip_identity_check": True},
            lease_timeout_s=30,
        )
    assert isinstance(started, StartedSession)
    assert started.warnings == ["left arm EEPROM disagrees with calibration"]
    assert started.session.lease is not None and started.session.lease.timeout_s == 60.0
    import json

    body = json.loads(script.requests[0].content)
    assert body == {
        "kind": "teleoperation",
        "robot": "bench",
        "owner": OWNER,
        "options": {"skip_identity_check": True},
        "lease_timeout_s": 30,
    }


def test_start_omits_owner_and_timeout_when_not_given():
    script = Script().add("POST", "/api/v1/sessions", (201, {"session": session_body(lease=False)}))
    with mock_client(script) as client:
        started = client.sessions.start("replay", "bench", options={"repo_id": "u/d", "episode_index": 0})
    assert started.warnings is None
    assert started.session.lease is None
    import json

    body = json.loads(script.requests[0].content)
    assert "owner" not in body and "lease_timeout_s" not in body


def test_start_held_names_the_holder():
    script = Script().add(
        "POST",
        "/api/v1/sessions",
        (
            409,
            {
                "detail": "The robot hardware is held by an active recording session. Stop it first.",
                "code": "session.held",
                "details": {"holder": {"kind": "recording", "session_id": "sess-9"}},
            },
        ),
    )
    with mock_client(script) as client, pytest.raises(SessionHeldError) as excinfo:
        client.sessions.start("teleoperation", "bench", options={})
    err = excinfo.value
    assert err.code == "session.held"
    assert err.holder == {"kind": "recording", "session_id": "sess-9"}
    assert "stop_current" in (err.suggestion or "")


def test_current_parses_live_session_and_last_ended():
    script = Script().add(
        "GET",
        "/api/v1/sessions/current",
        (
            200,
            {
                "session": session_body(kind="recording", phase="recording"),
                "last_ended": {
                    "id": "sess-0",
                    "kind": "teleoperation",
                    "ended_at": 999.0,
                    "phase": "released",
                    "reason": "session.lease_expired",
                },
            },
        ),
    )
    with mock_client(script) as client:
        now = client.sessions.current()
    assert isinstance(now.session, SessionInfo)
    assert now.session.kind == "recording"
    assert now.session.lease is not None and now.session.lease.owner == OWNER
    assert now.last_ended is not None and now.last_ended.reason == "session.lease_expired"


def test_heartbeat_returns_renewed_session():
    script = Script().add(
        "POST",
        "/api/v1/sessions/sess-1/heartbeat",
        (200, {"session": session_body(expires_in_s=60.0)}),
    )
    with mock_client(script) as client:
        session = client.sessions.heartbeat("sess-1", OWNER)
    assert isinstance(session, SessionInfo)
    assert session.lease is not None and session.lease.expires_in_s == 60.0
    import json

    assert json.loads(script.requests[0].content) == {"owner": OWNER}


def test_heartbeat_gone_raises_not_found():
    """The RAW method just raises — classification into "session lost" is the
    ActiveSession layer's job."""
    script = Script().add(
        "POST",
        "/api/v1/sessions/sess-1/heartbeat",
        (404, {"detail": "No active session with id 'sess-1'.", "code": "session.not_found"}),
    )
    with mock_client(script) as client, pytest.raises(NotFoundError) as excinfo:
        client.sessions.heartbeat("sess-1", OWNER)
    assert excinfo.value.code == "session.not_found"


def test_stop_returns_result_verbatim():
    script = Script().add(
        "POST",
        "/api/v1/sessions/sess-1/stop",
        (
            200,
            {
                "session": session_body(phase="releasing"),
                "result": {"success": True, "releasing": True, "warning": None},
            },
        ),
    )
    with mock_client(script) as client:
        stopped = client.sessions.stop("sess-1")
    assert isinstance(stopped, StoppedSession)
    assert stopped.session.phase == "releasing"
    assert stopped.result == {"success": True, "releasing": True, "warning": None}


def test_session_id_is_url_quoted():
    """An id with a path-hostile character travels %-escaped on the wire
    (httpx exposes the decoded .path; the raw request line keeps the escape)."""
    script = Script().add(
        "POST",
        "/api/v1/sessions/a/b/stop",
        (404, {"detail": "No active session with id 'a/b'.", "code": "session.not_found"}),
    )
    with mock_client(script) as client, pytest.raises(NotFoundError):
        client.sessions.stop("a/b")
    assert script.requests[0].url.raw_path == b"/api/v1/sessions/a%2Fb/stop"


def test_stop_current_idle_returns_none_without_stopping():
    script = Script().add("GET", "/api/v1/sessions/current", (200, {"session": None, "last_ended": None}))
    with mock_client(script) as client:
        assert client.sessions.stop_current() is None
    assert len(script.requests) == 1  # the read only — no stop was attempted


def test_stop_current_stops_by_id():
    script = (
        Script()
        .add("GET", "/api/v1/sessions/current", (200, {"session": session_body(), "last_ended": None}))
        .add(
            "POST",
            "/api/v1/sessions/sess-1/stop",
            (200, {"session": session_body(phase="released"), "result": {"success": True}}),
        )
    )
    with mock_client(script) as client:
        stopped = client.sessions.stop_current()
    assert stopped is not None and stopped.result == {"success": True}


def test_stop_current_swallows_the_race_404():
    """The session ending between the read and the stop is already-gone —
    for a stop that is success, reported as None."""
    script = (
        Script()
        .add("GET", "/api/v1/sessions/current", (200, {"session": session_body(), "last_ended": None}))
        .add(
            "POST",
            "/api/v1/sessions/sess-1/stop",
            (404, {"detail": "No active session with id 'sess-1'.", "code": "session.not_found"}),
        )
    )
    with mock_client(script) as client:
        assert client.sessions.stop_current() is None


# --- ActiveSession: tick classification (no threads, no sleeps) --------------


def make_active(script: Script, *, lease: bool = True, timeout_s: float = 60.0, warnings=None):
    """An ActiveSession over a scripted client, heartbeat THREAD disabled —
    tests drive ticks directly (house style: never sleep)."""
    from makermodslab_sdk.resources.sessions import ActiveSession

    client = mock_client(script)
    started = StartedSession.model_validate(
        {"session": session_body(lease=lease, timeout_s=timeout_s), "warnings": warnings}
    )
    return ActiveSession(client.sessions, started, owner=OWNER, auto_heartbeat=False), client


def heartbeat_404():
    return (404, {"detail": "No active session with id 'sess-1'.", "code": "session.not_found"})


def stop_ok(phase="released"):
    return (200, {"session": session_body(phase=phase), "result": {"success": True}})


def test_interval_is_a_third_of_the_lease_floored():
    s, _ = make_active(Script(), timeout_s=60.0)
    assert s.heartbeat_interval_s == 20.0
    s, _ = make_active(Script(), timeout_s=4.0)
    assert s.heartbeat_interval_s == 2.0  # the floor
    s, _ = make_active(Script(), lease=False)
    assert s.heartbeat_interval_s is None


def test_no_lease_means_no_heartbeat_thread_even_when_auto():
    from makermodslab_sdk.resources.sessions import ActiveSession

    client = mock_client(Script())
    started = StartedSession.model_validate({"session": session_body(lease=False)})
    s = ActiveSession(client.sessions, started, owner=OWNER)  # auto_heartbeat default True
    assert s._thread is None
    assert s.alive


def test_tick_renewed_updates_info():
    script = Script().add(
        "POST",
        "/api/v1/sessions/sess-1/heartbeat",
        (200, {"session": session_body(phase="recording", expires_in_s=60.0)}),
    )
    s, _ = make_active(script)
    assert s._tick() == "renewed"
    assert s.alive
    assert s.info.phase == "recording"


def test_tick_gone_records_loss():
    script = Script().add("POST", "/api/v1/sessions/sess-1/heartbeat", heartbeat_404())
    s, _ = make_active(script)
    assert s._tick() == "gone"
    assert not s.alive
    assert s.lost_reason == "session.not_found"


@pytest.mark.parametrize("code", ["session.not_owner", "session.lease_expired"])
def test_tick_unrenewable_409_records_loss(code):
    script = Script().add(
        "POST", "/api/v1/sessions/sess-1/heartbeat", (409, {"detail": "nope", "code": code})
    )
    s, _ = make_active(script)
    assert s._tick() == "lost"
    assert not s.alive
    assert s.lost_reason == code


def test_tick_network_blip_is_transient():
    script = Script().add(
        "POST", "/api/v1/sessions/sess-1/heartbeat", httpx.ConnectError("connection refused")
    )
    s, _ = make_active(script)
    assert s._tick() == "transient"
    assert s.alive  # never kill the session over a network blip
    assert s.lost_reason is None


def test_tick_unexpected_server_error_is_transient():
    script = Script().add(
        "POST", "/api/v1/sessions/sess-1/heartbeat", (500, {"detail": "boom", "code": None})
    )
    s, _ = make_active(script)
    assert s._tick() == "transient"
    assert s.alive


def test_tick_after_stop_is_a_noop():
    """A heartbeat racing our own stop() must not be recorded as a loss —
    its 404 is the stop's success."""
    script = Script().add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    s, _ = make_active(script)
    s.stop()
    assert s._tick() == "stopped"  # no heartbeat request was even sent
    assert s.lost_reason is None
    assert script.calls("POST", "/api/v1/sessions/sess-1/heartbeat") == []


# --- ActiveSession: stop and the context manager -----------------------------


def test_stop_is_idempotent():
    script = Script().add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    s, _ = make_active(script)
    first = s.stop()
    second = s.stop()
    assert first is not None and first.result == {"success": True}
    assert second is first  # cached, no second request
    assert len(script.calls("POST", "/api/v1/sessions/sess-1/stop")) == 1
    assert not s.alive
    assert s.info.phase == "released"


def test_stop_swallows_already_gone():
    script = Script().add("POST", "/api/v1/sessions/sess-1/stop", heartbeat_404())
    s, _ = make_active(script)
    assert s.stop() is None
    assert not s.alive


def test_context_manager_clean_exit_stops_without_raising():
    script = Script().add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    s, _ = make_active(script)
    with s:
        pass
    assert len(script.calls("POST", "/api/v1/sessions/sess-1/stop")) == 1
    assert not s.alive


def test_exit_raises_session_lost_after_loss():
    from makermodslab_sdk import SessionLostError

    script = (
        Script()
        .add("POST", "/api/v1/sessions/sess-1/heartbeat", heartbeat_404())
        .add("POST", "/api/v1/sessions/sess-1/stop", heartbeat_404())
    )
    s, _ = make_active(script)
    with pytest.raises(SessionLostError) as excinfo, s:
        s._tick()  # the heartbeat discovers the session gone
    err = excinfo.value
    assert err.reason == "session.not_found"
    assert err.session_id == "sess-1"
    assert err.kind == "teleoperation"
    assert "Next step:" in str(err)
    # Cleanup still ran: the stop was attempted (and its 404 swallowed).
    assert len(script.calls("POST", "/api/v1/sessions/sess-1/stop")) == 1


def test_exit_never_masks_the_body_exception():
    script = (
        Script()
        .add("POST", "/api/v1/sessions/sess-1/heartbeat", heartbeat_404())
        .add("POST", "/api/v1/sessions/sess-1/stop", heartbeat_404())
    )
    s, _ = make_active(script)
    with pytest.raises(ValueError, match="user error"), s:  # NOT SessionLostError
        s._tick()
        raise ValueError("user error")
    # The stop cleanup still ran.
    assert len(script.calls("POST", "/api/v1/sessions/sess-1/stop")) == 1


def test_deliberate_stop_inside_the_body_does_not_raise_at_exit():
    script = Script().add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    s, _ = make_active(script)
    with s:
        s.stop()
    assert len(script.calls("POST", "/api/v1/sessions/sess-1/stop")) == 1


def test_warnings_surface_verbatim():
    s, _ = make_active(Script(), warnings=["left arm EEPROM disagrees with calibration"])
    assert s.warnings == ["left arm EEPROM disagrees with calibration"]
    s, _ = make_active(Script(), warnings=None)
    assert s.warnings == []


def test_heartbeat_thread_lifecycle_without_waiting():
    """The timing loop honors the stop event immediately: pre-set it, start
    the thread, and the loop exits without a single tick or any real wait."""
    script = Script()  # strict: any request at all would fail the test
    s, _ = make_active(script)
    s._stop_event.set()
    s._start_heartbeat_thread()
    s._thread.join(timeout=5.0)
    assert not s._thread.is_alive()
    assert script.requests == []


# --- per-kind sugar ----------------------------------------------------------


def test_sugar_missing_robot_end_to_end(sdk_client):
    """Sugar with an unknown robot name refuses with robot.not_found BEFORE
    any hardware path — and before any ActiveSession (or thread) exists."""
    with pytest.raises(NotFoundError) as excinfo:
        sdk_client.sessions.teleoperate("__sdk_test_missing__")
    assert excinfo.value.code == "robot.not_found"
    assert "Next step:" in str(excinfo.value)


def start_ok(kind="teleoperation", **session_kwargs):
    return (201, {"session": session_body(kind=kind, **session_kwargs), "warnings": None})


def started_body(script: Script) -> dict:
    import json

    return json.loads(script.calls("POST", "/api/v1/sessions")[0].content)


def test_teleoperate_builds_body_and_defaults_owner():
    from makermodslab_sdk.resources.sessions import ActiveSession

    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok())
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        s = client.sessions.teleoperate("bench")
        assert isinstance(s, ActiveSession)
        assert s.heartbeat_interval_s == 20.0  # 60s lease / 3
        assert s._thread is not None  # sugar always heartbeats
        s.stop()
        assert not s._thread.is_alive()
    body = started_body(script)
    assert body["kind"] == "teleoperation"
    assert body["robot"] == "bench"
    assert body["options"] == {}  # unset knobs dropped — server defaults rule
    assert "lease_timeout_s" not in body
    assert body["owner"].startswith("sdk:")


def test_default_owner_shape():
    import os

    from makermodslab_sdk.resources.sessions import default_session_owner

    one, two = default_session_owner(), default_session_owner()
    assert one != two  # unique per call: the 4-char token
    prefix, host, pid, token = one.split(":")
    assert prefix == "sdk" and host
    assert pid == str(os.getpid())
    assert len(token) == 4
    assert len(one) <= 128  # the server's OWNER_MAX_LENGTH


def test_explicit_owner_and_lease_timeout_pass_through():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok())
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        client.sessions.teleoperate("bench", owner="agent:me", lease_timeout_s=120).stop()
    body = started_body(script)
    assert body["owner"] == "agent:me"
    assert body["lease_timeout_s"] == 120


def test_record_options_mirror_the_contract():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok(kind="recording"))
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        client.sessions.record(
            "bench",
            dataset_repo_id="me/demo",
            single_task="pick the cube",
            num_episodes=3,
            fps=25,
            push_to_hub=False,
            tags=["sdk"],
        ).stop()
    body = started_body(script)
    assert body["kind"] == "recording"
    assert body["options"] == {
        "dataset_repo_id": "me/demo",
        "single_task": "pick the cube",
        "num_episodes": 3,
        "fps": 25,
        "push_to_hub": False,
        "tags": ["sdk"],
    }


def test_infer_options_mirror_the_contract():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok(kind="inference"))
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        client.sessions.infer(
            "bench",
            policy_ref="me/act-pick",
            task="pick",
            camera_bindings={"top": "overhead"},
            camera_dims={"top": {"width": 640, "height": 480}},
            inference_engine="rtc",
            temporal_ensemble_coeff=0.9,
        ).stop()
    body = started_body(script)
    assert body["kind"] == "inference"
    assert body["options"] == {
        "policy_ref": "me/act-pick",
        "task": "pick",
        "camera_bindings": {"top": "overhead"},
        "camera_dims": {"top": {"width": 640, "height": 480}},
        "inference_engine": "rtc",
        "temporal_ensemble_coeff": 0.9,
    }


def test_replay_options_mirror_the_contract():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok(kind="replay"))
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        client.sessions.replay("bench", repo_id="me/demo", episode_index=2).stop()
    body = started_body(script)
    assert body["kind"] == "replay"
    assert body["options"] == {"repo_id": "me/demo", "episode_index": 2}


def test_calibrate_options_mirror_the_contract():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok(kind="calibration"))
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        client.sessions.calibrate(
            "bench", device_type="teleop", arm="right", port="/dev/ttyUSB1", overwrite=True
        ).stop()
    body = started_body(script)
    assert body["kind"] == "calibration"
    assert body["options"] == {
        "device_type": "teleop",
        "arm": "right",
        "port": "/dev/ttyUSB1",
        "overwrite": True,
    }


def test_auto_calibrate_options_mirror_the_contract():
    script = (
        Script()
        .add("POST", "/api/v1/sessions", start_ok(kind="auto_calibration", timeout_s=90.0))
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        s = client.sessions.auto_calibrate(
            "bench",
            arms=[{"device_type": "robot"}, {"device_type": "teleop"}],
            motor_power=50,
        )
        assert s.heartbeat_interval_s == 30.0  # the server's 90s default / 3
        s.stop()
    body = started_body(script)
    assert body["kind"] == "auto_calibration"
    assert "lease_timeout_s" not in body  # the server's per-kind default rules
    assert body["options"] == {
        "arms": [{"device_type": "robot"}, {"device_type": "teleop"}],
        "motor_power": 50,
    }


def test_sugar_relays_start_warnings():
    script = (
        Script()
        .add(
            "POST",
            "/api/v1/sessions",
            (201, {"session": session_body(), "warnings": ["EEPROM disagrees with calibration"]}),
        )
        .add("POST", "/api/v1/sessions/sess-1/stop", stop_ok())
    )
    with mock_client(script) as client:
        s = client.sessions.teleoperate("bench")
        assert s.warnings == ["EEPROM disagrees with calibration"]
        s.stop()


def test_sugar_held_raises_before_any_active_session():
    script = Script().add(
        "POST",
        "/api/v1/sessions",
        (
            409,
            {
                "detail": "The robot hardware is held by an active inference session. Stop it first.",
                "code": "session.held",
                "details": {"holder": {"kind": "inference", "session_id": "sess-7"}},
            },
        ),
    )
    with mock_client(script) as client, pytest.raises(SessionHeldError) as excinfo:
        client.sessions.teleoperate("bench")
    assert excinfo.value.holder == {"kind": "inference", "session_id": "sess-7"}
