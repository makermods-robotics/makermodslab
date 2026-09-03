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

"""Tests for the session lease — server-authoritative ownership with a timeout
fail-safe (makermodslab/sessions.py).

Per CLAUDE.md's testing policy there are no timing-dependent tests: the expiry
state machine is exercised through :func:`sessions.check_expiry` with a fake
clock and explicit ``now`` values, stop dispatch is captured via monkeypatched
stop handlers, and the watchdog thread gets only start/stop lifecycle
assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makermodslab import sessions
from makermodslab.session_events import notify_session_changed


class FakeClock:
    """A controllable stand-in for time.monotonic."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _fresh_tracker():
    """Every test starts and ends with no tracked session and no watchdog."""
    sessions.tracker.reset()
    yield
    sessions.tracker.reset()
    assert sessions._watchdog_thread is None  # reset retires the watchdog


@pytest.fixture
def fake_clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(sessions.tracker, "_clock", clock)
    return clock


def _make_robot(name: str = "bench") -> None:
    """Fabricate a READY single-arm robot record on (redirected) disk."""
    from makermodslab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FC.json").write_text("{}")
    (Path(cfg.LEADER_CONFIG_PATH) / "LC.json").write_text("{}")
    cfg.save_robot_record(
        name,
        {
            "follower_port": "/dev/f",
            "follower_config": "FC",
            "leader_port": "/dev/l",
            "leader_config": "LC",
        },
    )


def _fake_teleop_start(monkeypatch) -> None:
    """handle_start_teleoperation stand-in that emits the claim event."""

    def fake(request, websocket_manager=None):
        notify_session_changed("teleoperation", True)
        return {"success": True}

    monkeypatch.setattr("makermodslab.teleoperate.handle_start_teleoperation", fake)


def _lease_directly(owner: str = "alice", timeout_s: float = 60.0, kind: str = "teleoperation") -> dict:
    """Claim through the seam and attach a lease straight on the tracker —
    thread-free (no watchdog), for check_expiry / renewal state tests."""
    notify_session_changed(kind, True)
    snap = sessions.tracker.attach_lease(kind, owner=owner, timeout_s=timeout_s)
    assert snap is not None
    return snap


def _capture_teleop_stop(monkeypatch, release: bool = True) -> list:
    """Monkeypatch teleoperation's stop handler to a capture; optionally emit
    the release event (release=False models the not-yet-released window)."""
    calls: list = []

    def fake_stop():
        calls.append("stop")
        if release:
            notify_session_changed("teleoperation", False, phase="done")
        return {"success": True, "message": "stopped"}

    monkeypatch.setattr("makermodslab.teleoperate.handle_stop_teleoperation", fake_stop)
    return calls


# --- lease attachment rules ---------------------------------------------------


def test_owner_post_attaches_a_lease(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": "ui-1"})
    assert resp.status_code == 201
    lease = resp.json()["session"]["lease"]
    assert lease["owner"] == "ui-1"
    assert lease["timeout_s"] == 60  # the default
    assert 0 < lease["expires_in_s"] <= 60
    assert "deadline" not in lease  # the monotonic deadline stays internal


def test_auto_calibration_default_timeout_is_longer(client, tmp_lerobot_home, monkeypatch) -> None:
    """auto_calibration defaults to a 90s lease: a real batch run measures
    ~60s+ on hardware, so the plain 60s default leaves no closed-tab survival
    margin for the reopen-and-recover flow. An explicit lease_timeout_s still
    wins (next test asserts the generic override path)."""
    from makermodslab import auto_calibrate
    from tests.test_sessions import _fake_start, _make_robot as _make_sessions_robot

    _make_sessions_robot("bench")
    monkeypatch.setattr(
        auto_calibrate.auto_calibration_batch_manager, "start", _fake_start("auto_calibration", [])
    )
    resp = client.post(
        "/api/v1/sessions",
        json={
            "kind": "auto_calibration",
            "robot": "bench",
            "owner": "ui-1",
            "options": {"arms": [{"device_type": "robot", "arm": "left"}]},
        },
    )
    assert resp.status_code == 201
    lease = resp.json()["session"]["lease"]
    assert lease["timeout_s"] == 90
    assert 0 < lease["expires_in_s"] <= 90


def test_custom_timeout_is_honoured(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "teleoperation", "robot": "bench", "owner": "ui-1", "lease_timeout_s": 120},
    )
    assert resp.status_code == 201
    lease = resp.json()["session"]["lease"]
    assert lease["timeout_s"] == 120
    assert 60 < lease["expires_in_s"] <= 120


def test_ownerless_post_gets_no_lease_and_never_expires(client, tmp_lerobot_home, monkeypatch) -> None:
    """The compatibility linchpin: no owner, no lease, no timeout-stop."""
    _make_robot()
    _fake_teleop_start(monkeypatch)
    calls = _capture_teleop_stop(monkeypatch)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 201
    assert resp.json()["session"]["lease"] is None

    assert sessions.check_expiry(now=1e12) is None  # arbitrarily far future
    assert calls == []
    assert sessions.tracker.current() is not None


def test_legacy_start_gets_no_lease_and_never_expires(client, monkeypatch) -> None:
    """A session started through the legacy endpoints (observed via the seam)
    must never be timeout-stopped — the un-migrated UI heartbeats nothing."""
    notify_session_changed("recording", True, phase="recording")
    calls: list = []
    monkeypatch.setattr(
        "makermodslab.record.handle_stop_recording", lambda: calls.append("stop") or {"success": True}
    )

    assert client.get("/api/v1/sessions/current").json()["session"]["lease"] is None
    assert sessions.check_expiry(now=1e12) is None
    assert calls == []


@pytest.mark.parametrize("timeout_s", [9, 9.99, 601, 0, -5])
def test_lease_timeout_out_of_bounds_422(client, tmp_lerobot_home, monkeypatch, timeout_s) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "teleoperation", "robot": "bench", "owner": "u", "lease_timeout_s": timeout_s},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


@pytest.mark.parametrize("timeout_s", [10, 600])
def test_lease_timeout_bounds_are_inclusive(client, tmp_lerobot_home, monkeypatch, timeout_s) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "teleoperation", "robot": "bench", "owner": "u", "lease_timeout_s": timeout_s},
    )
    assert resp.status_code == 201
    assert resp.json()["session"]["lease"]["timeout_s"] == timeout_s


@pytest.mark.parametrize("owner", ["", "x" * 129])
def test_owner_shape_is_validated_422(client, tmp_lerobot_home, monkeypatch, owner) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": owner})
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


def test_owner_at_the_128_char_limit_is_accepted(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post(
        "/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": "x" * 128}
    )
    assert resp.status_code == 201
    assert resp.json()["session"]["lease"]["owner"] == "x" * 128


# --- GET /sessions/current: exposure without renewal --------------------------


def test_get_current_exposes_expiry_and_does_not_renew(client, fake_clock) -> None:
    _lease_directly(timeout_s=60)

    fake_clock.advance(40)
    lease = client.get("/api/v1/sessions/current").json()["session"]["lease"]
    assert lease == {"owner": "alice", "timeout_s": 60, "expires_in_s": 20}

    # A read is for any observer — it must not push the deadline.
    fake_clock.advance(15)
    lease = client.get("/api/v1/sessions/current").json()["session"]["lease"]
    assert lease["expires_in_s"] == 5


def test_expires_in_s_is_never_negative(client, fake_clock) -> None:
    _lease_directly(timeout_s=60)
    fake_clock.advance(1000)  # long past the deadline; watchdog not involved here
    lease = client.get("/api/v1/sessions/current").json()["session"]["lease"]
    assert lease["expires_in_s"] == 0


# --- POST /sessions/{id}/heartbeat: renewal semantics -------------------------


def test_heartbeat_renews_the_deadline_for_the_owner(client, fake_clock) -> None:
    snap = _lease_directly(owner="alice", timeout_s=60)
    old_deadline = snap["lease"]["deadline"]

    fake_clock.advance(50)
    resp = client.post(f"/api/v1/sessions/{snap['id']}/heartbeat", json={"owner": "alice"})
    assert resp.status_code == 200
    assert resp.json()["session"]["lease"]["expires_in_s"] == 60

    # The renewal pushed the deadline: the old deadline no longer expires it.
    assert sessions.check_expiry(now=old_deadline + 1) is None
    assert sessions.tracker.current()["lease"]["deadline"] == fake_clock.t + 60


def test_heartbeat_owner_mismatch_409_and_no_renewal(client, fake_clock) -> None:
    snap = _lease_directly(owner="alice", timeout_s=60)
    fake_clock.advance(30)
    resp = client.post(f"/api/v1/sessions/{snap['id']}/heartbeat", json={"owner": "mallory"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "session.not_owner"
    assert sessions.tracker.current()["lease"]["deadline"] == snap["lease"]["deadline"]


def test_heartbeat_unknown_or_stale_id_404(client) -> None:
    resp = client.post("/api/v1/sessions/deadbeef/heartbeat", json={"owner": "alice"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "session.not_found"

    _lease_directly(owner="alice")
    resp = client.post("/api/v1/sessions/0000stale0000/heartbeat", json={"owner": "alice"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "session.not_found"


def test_heartbeat_of_an_unleased_session_is_a_noop_200(client) -> None:
    """Documented: heartbeating an unleased (legacy / owner-less) session is
    harmless — eases client rollout while lease attachment is opt-in."""
    notify_session_changed("teleoperation", True)
    live_id = sessions.tracker.current()["id"]
    resp = client.post(f"/api/v1/sessions/{live_id}/heartbeat", json={"owner": "anyone"})
    assert resp.status_code == 200
    assert resp.json()["session"]["lease"] is None
    assert sessions.tracker.current()["lease"] is None  # still no lease


@pytest.mark.parametrize("owner", ["", "x" * 129])
def test_heartbeat_owner_shape_is_validated_422(client, owner) -> None:
    snap = _lease_directly()
    resp = client.post(f"/api/v1/sessions/{snap['id']}/heartbeat", json={"owner": owner})
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


# --- check_expiry: the expiry state machine (fake clock, no threads) ----------


def test_check_expiry_before_the_deadline_is_a_noop(monkeypatch, fake_clock) -> None:
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    assert sessions.check_expiry(now=snap["lease"]["deadline"] - 0.1) is None
    assert calls == []
    assert sessions.tracker.current() is not None


def test_check_expiry_at_the_deadline_stops_the_session(monkeypatch, fake_clock) -> None:
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    stopped = sessions.check_expiry(now=snap["lease"]["deadline"])
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["stop"]
    assert sessions.tracker.current() is None


def test_check_expiry_after_the_deadline_stops_and_records_the_reason(monkeypatch, fake_clock) -> None:
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    stopped = sessions.check_expiry(now=snap["lease"]["deadline"] + 5)
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["stop"]
    ended = sessions.tracker.last_ended()
    assert ended["id"] == snap["id"]
    assert ended["reason"] == "session.lease_expired"  # the reserved code string


def test_normal_endings_carry_no_expiry_reason(monkeypatch, fake_clock) -> None:
    _lease_directly(timeout_s=60)
    notify_session_changed("teleoperation", False, phase="done")
    assert sessions.tracker.last_ended()["reason"] is None


def test_check_expiry_defaults_now_to_the_injected_clock(monkeypatch, fake_clock) -> None:
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    assert sessions.check_expiry() is None  # clock still before the deadline
    fake_clock.advance(61)
    stopped = sessions.check_expiry()
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["stop"]


def test_check_expiry_race_with_a_natural_release_is_a_noop(monkeypatch, fake_clock) -> None:
    """The session is already gone when the check fires — nothing to stop."""
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    notify_session_changed("teleoperation", False, phase="done")  # released naturally

    assert sessions.check_expiry(now=snap["lease"]["deadline"] + 5) is None
    assert calls == []
    assert sessions.tracker.last_ended()["reason"] is None


def test_check_expiry_skips_a_session_already_winding_down(monkeypatch, fake_clock) -> None:
    """Expiry during a release-grace phase: the stop handler already handles
    it — don't double-dispatch a stop into the wind-down."""
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch)
    notify_session_changed("teleoperation", True, phase="releasing")
    assert sessions.check_expiry(now=snap["lease"]["deadline"] + 5) is None
    assert calls == []


def test_check_expiry_never_double_dispatches(monkeypatch, fake_clock) -> None:
    """Once the expiry stop is dispatched, a still-releasing session must not
    be stopped again on the next tick."""
    snap = _lease_directly(timeout_s=60)
    calls = _capture_teleop_stop(monkeypatch, release=False)  # release hasn't landed

    stopped = sessions.check_expiry(now=snap["lease"]["deadline"] + 1)
    assert stopped is not None
    assert sessions.check_expiry(now=snap["lease"]["deadline"] + 2) is None
    assert calls == ["stop"]


def test_heartbeat_in_the_expired_but_unreleased_window_409(client, monkeypatch, fake_clock) -> None:
    """The one place session.lease_expired surfaces as an HTTP error: the stop
    is dispatched but the release event hasn't landed yet. Once it lands the
    session is simply gone — a plain 404, no special code path."""
    snap = _lease_directly(owner="alice", timeout_s=60)
    _capture_teleop_stop(monkeypatch, release=False)
    assert sessions.check_expiry(now=snap["lease"]["deadline"] + 1) is not None

    resp = client.post(f"/api/v1/sessions/{snap['id']}/heartbeat", json={"owner": "alice"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "session.lease_expired"

    notify_session_changed("teleoperation", False, phase="done")  # the release lands
    resp = client.post(f"/api/v1/sessions/{snap['id']}/heartbeat", json={"owner": "alice"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "session.not_found"
    assert sessions.tracker.last_ended()["reason"] == "session.lease_expired"


# --- the calibration kinds: lease attach + expiry dispatch --------------------


def test_calibration_start_with_owner_attaches_a_lease(client, tmp_lerobot_home, monkeypatch) -> None:
    """The new kinds lease exactly like the original four."""
    from makermodslab import calibrate

    _make_robot()

    def fake_start(request):
        notify_session_changed("calibration", True, phase="connecting")
        return {"success": True}

    monkeypatch.setattr(calibrate.calibration_manager, "start_calibration", fake_start)
    resp = client.post(
        "/api/v1/sessions",
        json={
            "kind": "calibration",
            "robot": "bench",
            "owner": "ui-1",
            "options": {"device_type": "teleop"},
        },
    )
    assert resp.status_code == 201
    lease = resp.json()["session"]["lease"]
    assert lease["owner"] == "ui-1"
    assert sessions._watchdog_thread is not None
    notify_session_changed("calibration", False, phase="idle")
    assert sessions._watchdog_thread is None


def test_expiry_of_a_calibration_session_dispatches_its_stop(monkeypatch, fake_clock) -> None:
    """The expiry safety stop must reach calibrate.py's real stop handler —
    the wizard's teardown (torque baseline restore, disconnect), not a
    generic kill."""
    from makermodslab import calibrate

    snap = _lease_directly(kind="calibration")
    calls: list = []

    def fake_stop():
        calls.append("stop")
        notify_session_changed("calibration", False, phase="idle")
        return {"success": True, "message": "Calibration stopped"}

    monkeypatch.setattr(calibrate.calibration_manager, "stop_calibration_process", fake_stop)
    stopped = sessions.check_expiry(now=snap["lease"]["deadline"] + 1)
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["stop"]
    assert sessions.tracker.last_ended()["reason"] == "session.lease_expired"


def test_expiry_of_a_remote_inference_session_dispatches_its_stop(monkeypatch, fake_clock) -> None:
    """The expiry safety stop must reach remote_inference's real stop handler —
    the STOP on the child's stdin that makes it return the arm to its captured
    start pose BEFORE releasing torque, not a generic kill."""
    from makermodslab import remote_inference

    snap = _lease_directly(kind="remote_inference")
    calls: list = []

    def fake_stop():
        calls.append("stop")
        notify_session_changed("remote_inference", False, phase="stopped")
        return {"success": True, "message": "Remote inference stopped"}

    monkeypatch.setattr(remote_inference, "handle_stop_remote_inference", fake_stop)
    stopped = sessions.check_expiry(now=snap["lease"]["deadline"] + 1)
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["stop"]
    assert sessions.tracker.last_ended()["reason"] == "session.lease_expired"


def test_expiry_during_a_remote_return_to_rest_is_not_double_dispatched(monkeypatch, fake_clock) -> None:
    """This is the test that pins remote_inference's "phase stays `stopping`
    through the return-to-rest" decision.

    A `returning` phase of its own would fall OUTSIDE
    sessions._WINDING_DOWN_PHASES, so an expiry tick landing while the child is
    easing the arm home would dispatch a SECOND stop into the live return —
    which the child reads as "cut it short" and drops torque wherever the arm
    happens to be."""
    from makermodslab import remote_inference

    snap = _lease_directly(kind="remote_inference")
    calls: list = []
    monkeypatch.setattr(
        remote_inference,
        "handle_stop_remote_inference",
        lambda: calls.append("stop") or {"success": True},
    )
    notify_session_changed("remote_inference", True, phase=remote_inference.PHASE_STOPPING)

    assert remote_inference.PHASE_STOPPING in sessions._WINDING_DOWN_PHASES
    assert sessions.check_expiry(now=snap["lease"]["deadline"] + 5) is None
    assert calls == []


def test_expiry_of_an_auto_calibration_session_stops_whichever_manager_is_live(
    monkeypatch, fake_clock
) -> None:
    """auto_calibration is an aggregate of the single-arm manager and the
    batch manager: expiry tries the single manager first and falls through to
    the batch when it reports nothing running — the same escalation the stop
    endpoint uses."""
    from makermodslab import auto_calibrate

    snap = _lease_directly(kind="auto_calibration")
    calls: list = []

    monkeypatch.setattr(
        auto_calibrate.auto_calibration_manager,
        "stop",
        lambda: calls.append("single") or {"success": False, "message": "No auto-calibration is running"},
    )

    def fake_batch_stop():
        calls.append("batch")
        notify_session_changed("auto_calibration", False, phase="stopped")
        return {"success": True, "message": "Stopping 2 arm(s)"}

    monkeypatch.setattr(auto_calibrate.auto_calibration_batch_manager, "stop", fake_batch_stop)
    stopped = sessions.check_expiry(now=snap["lease"]["deadline"] + 1)
    assert stopped is not None and stopped["id"] == snap["id"]
    assert calls == ["single", "batch"]
    assert sessions.tracker.last_ended()["reason"] == "session.lease_expired"


# --- the watchdog: start/stop lifecycle only (no timing tests) ----------------


def test_watchdog_starts_on_a_leased_claim_and_retires_on_release(
    client, tmp_lerobot_home, monkeypatch
) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    assert sessions._watchdog_thread is None

    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": "ui-1"})
    assert resp.status_code == 201
    thread = sessions._watchdog_thread
    assert thread is not None and thread.is_alive()

    notify_session_changed("teleoperation", False, phase="done")
    assert sessions._watchdog_thread is None  # retired eagerly with the lease


def test_watchdog_does_not_start_for_ownerless_sessions(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 201
    assert sessions._watchdog_thread is None
    notify_session_changed("teleoperation", False)


# --- stop is never owner-gated ------------------------------------------------


def test_stop_is_never_owner_gated(client, tmp_lerobot_home, monkeypatch) -> None:
    """HARDWARE SAFETY over ownership: a leased session is stoppable by
    whoever can reach the API — the stop endpoint carries no owner at all."""
    _make_robot()
    _fake_teleop_start(monkeypatch)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": "alice"})
    live_id = resp.json()["session"]["id"]
    _capture_teleop_stop(monkeypatch)

    resp = client.post(f"/api/v1/sessions/{live_id}/stop")  # no owner anywhere
    assert resp.status_code == 200
    assert sessions.tracker.current() is None
