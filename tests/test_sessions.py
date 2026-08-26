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

"""Tests for makermodslab/sessions.py — the /api/v1/sessions surface — and the
session_events seam's multi-subscriber extension.

Per CLAUDE.md's testing policy: the tracker (a pure observer, driven entirely
through the seam), the resolution/mutex refusal branches, request-model
assembly (pure — the handle_start_* dispatch is monkeypatched to capture), and
stop-by-id identity checks. No subprocess/thread happy paths.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from makermodslab import session_events, sessions
from makermodslab.api_errors import ErrorCode
from makermodslab.session_events import notify_session_changed


@pytest.fixture(autouse=True)
def _fresh_tracker():
    """Every test starts and ends with no tracked session, whatever earlier
    suites' claim/release events left behind (the tracker is a process
    singleton subscribed to a process-wide seam)."""
    sessions.tracker.reset()
    yield
    sessions.tracker.reset()


@pytest.fixture
def _quiet_notifier():
    """Detach the WS notifier for seam-focused tests, restoring it after."""
    previous = session_events._notifier
    session_events.set_notifier(None)
    yield
    session_events.set_notifier(previous)


# --- the seam's multi-subscriber extension -----------------------------------


def test_seam_delivers_to_subscribers_and_notifier(_quiet_notifier) -> None:
    seen_sub: list[dict] = []
    seen_notifier: list[dict] = []
    session_events.subscribe(seen_sub.append)
    session_events.set_notifier(seen_notifier.append)
    try:
        notify_session_changed("teleoperation", True, phase="x")
    finally:
        session_events.unsubscribe(seen_sub.append)
        session_events.set_notifier(None)
    # Both consumers got the same event; delivery is not either/or.
    assert len(seen_sub) == 1 and len(seen_notifier) == 1
    assert seen_sub[0]["session"] == seen_notifier[0]["session"]


def test_broken_subscriber_starves_neither_notifier_nor_other_subscribers(_quiet_notifier) -> None:
    def broken(_event: dict) -> None:
        raise RuntimeError("subscriber exploded")

    seen_sub: list[dict] = []
    seen_notifier: list[dict] = []
    session_events.subscribe(broken)
    session_events.subscribe(seen_sub.append)
    session_events.set_notifier(seen_notifier.append)
    try:
        notify_session_changed("recording", True)
    finally:
        session_events.unsubscribe(broken)
        session_events.unsubscribe(seen_sub.append)
        session_events.set_notifier(None)
    assert len(seen_sub) == 1
    assert len(seen_notifier) == 1


def test_unsubscribe_and_duplicate_subscribe(_quiet_notifier) -> None:
    seen: list[dict] = []
    session_events.subscribe(seen.append)
    session_events.subscribe(seen.append)  # idempotent: still one entry
    try:
        notify_session_changed("replay", True)
        assert len(seen) == 1
    finally:
        session_events.unsubscribe(seen.append)
    notify_session_changed("replay", False)
    assert len(seen) == 1
    session_events.unsubscribe(seen.append)  # unknown callable: ignored


# --- tracker lifecycle, driven purely through the seam -----------------------


def test_claim_mints_identity() -> None:
    before = time.time()
    notify_session_changed("recording", True, phase="preparing")
    after = time.time()

    current = sessions.tracker.current()
    assert current is not None
    assert current["kind"] == "recording"
    assert current["phase"] == "preparing"
    assert current["revision"] == 1
    assert current["robot"] is None and current["owner"] is None
    assert len(current["id"]) == 32  # uuid4().hex
    assert before <= current["started_at"] <= after


def test_phase_events_bump_revision_and_keep_the_id() -> None:
    notify_session_changed("recording", True, phase="preparing")
    first = sessions.tracker.current()
    notify_session_changed("recording", True, phase="recording")
    notify_session_changed("recording", True, phase="resetting")

    current = sessions.tracker.current()
    assert current["id"] == first["id"]
    assert current["revision"] == 3
    assert current["phase"] == "resetting"


def test_release_clears_current_and_keeps_last_ended() -> None:
    notify_session_changed("replay", True, phase="easing_in")
    live = sessions.tracker.current()
    before = time.time()
    notify_session_changed("replay", False, phase="stopping")
    after = time.time()

    assert sessions.tracker.current() is None
    ended = sessions.tracker.last_ended()
    assert ended == {
        "id": live["id"],
        "kind": "replay",
        "ended_at": ended["ended_at"],
        "phase": "stopping",
        "reason": None,  # a normal ending, not a lease-expiry safety stop
    }
    assert before <= ended["ended_at"] <= after


def test_release_when_idle_or_for_another_kind_is_ignored() -> None:
    notify_session_changed("teleoperation", False)
    assert sessions.tracker.current() is None
    assert sessions.tracker.last_ended() is None

    notify_session_changed("recording", True)
    notify_session_changed("teleoperation", False)  # not the live kind
    assert sessions.tracker.current()["kind"] == "recording"
    assert sessions.tracker.last_ended() is None


def test_attribute_enriches_only_the_matching_kind() -> None:
    notify_session_changed("teleoperation", True)
    assert sessions.tracker.attribute("recording", robot="r1") is None

    snap = sessions.tracker.attribute("teleoperation", robot="r1", owner="ui-1")
    assert snap["robot"] == "r1" and snap["owner"] == "ui-1"
    assert snap["revision"] == 1  # enrichment, not a transition
    assert sessions.tracker.current()["robot"] == "r1"


def test_legacy_started_session_gets_an_identity(monkeypatch) -> None:
    """A start through the un-migrated legacy surface must still mint identity
    — it falls out of seam observation. Reuses test_replay's mocked-connect
    start path: the real legacy handler runs, hardware-free."""
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    monkeypatch.setattr(replay, "replay_thread", None)

    motors = ("shoulder_pan", "gripper")
    monkeypatch.setattr(
        replay,
        "get_episode_action_series",
        lambda repo_id, episode_index: {
            "action_names": [f"{m}.pos" for m in motors],
            "timestamps": [0.0],
            "values": [[1.0, 2.0]],
        },
    )

    class _FakeBus:
        def __init__(self) -> None:
            self.motors = dict.fromkeys(motors)

    class _FakeRobot:
        action_features = {f"{m}.pos": float for m in motors}
        bus = _FakeBus()

    class _NoopThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(replay, "_connect_follower", lambda request: (_FakeRobot(), []))
    monkeypatch.setattr(replay.threading, "Thread", _NoopThread)

    result = replay.handle_start_replay(
        replay.ReplayRequest(
            repo_id="alice/pick", episode_index=0, follower_port="/dev/f", follower_config="fc"
        )
    )
    assert result["success"] is True
    try:
        current = sessions.tracker.current()
        assert current is not None
        assert current["kind"] == "replay"
        assert current["phase"] == "easing_in"
        assert current["robot"] is None  # only the start wrapper knows; never guessed
    finally:
        replay.replay_active = False
        replay.replay_thread = None


# --- endpoint helpers --------------------------------------------------------


def _make_robot(name: str = "bench", mode: str = "single", follower_only: bool = False) -> None:
    """Fabricate a READY robot record on (redirected) disk: ports+configs set
    and every referenced calibration config file present."""
    from makermodslab.utils import config as cfg

    data: dict = {"follower_port": "/dev/f", "follower_config": "FC"}
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FC.json").write_text("{}")
    if not follower_only:
        data |= {"leader_port": "/dev/l", "leader_config": "LC"}
        (Path(cfg.LEADER_CONFIG_PATH) / "LC.json").write_text("{}")
    if mode == "bimanual":
        data |= {
            "mode": "bimanual",
            "right_follower_port": "/dev/rf",
            "right_follower_config": "RFC",
        }
        (Path(cfg.FOLLOWER_CONFIG_PATH) / "RFC.json").write_text("{}")
        if not follower_only:
            data |= {"right_leader_port": "/dev/rl", "right_leader_config": "RLC"}
            (Path(cfg.LEADER_CONFIG_PATH) / "RLC.json").write_text("{}")
    cfg.save_robot_record(name, data)


def _fake_start(kind: str, captured: list, result: dict | None = None):
    """A handle_start_* stand-in: records the constructed request model and
    mimics the contract's claim event on success."""

    def fake(request, websocket_manager=None):
        captured.append(request)
        if result is not None:
            return result
        notify_session_changed(kind, True)
        return {"success": True}

    return fake


_REPLAY_OPTIONS = {"repo_id": "u/d", "episode_index": 0}


# --- POST /api/v1/sessions: resolution failures ------------------------------


def test_start_unknown_robot_404(client, tmp_lerobot_home) -> None:
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "ghost"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "robot.not_found"


def test_start_invalid_robot_name_404(client, tmp_lerobot_home) -> None:
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "../etc"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "robot.not_found"


def test_start_not_ready_400(client, tmp_lerobot_home) -> None:
    from makermodslab.utils import config as cfg

    cfg.save_robot_record("bare", {"leader_port": "/dev/l"})  # no configs, no files
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bare"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "robot.not_ready"


def test_follower_only_kinds_ignore_leader_gaps(client, tmp_lerobot_home, monkeypatch) -> None:
    """Readiness is scoped to the arms the kind drives (the frontend's
    robotSetupGap distinction): a leaderless robot can replay but not
    teleoperate."""
    _make_robot("armless", follower_only=True)

    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "armless"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "robot.not_ready"

    captured: list = []
    monkeypatch.setattr("makermodslab.replay.handle_start_replay", _fake_start("replay", captured))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "replay", "robot": "armless", "options": _REPLAY_OPTIONS},
    )
    assert resp.status_code == 201
    assert len(captured) == 1


# --- POST /api/v1/sessions: per-kind options validation ----------------------


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("recording", {}),  # dataset_repo_id/single_task are required
        ("inference", {}),  # policy_ref is required
        ("replay", {"repo_id": "u/d"}),  # episode_index is required
        ("teleoperation", {"dataset_repo_id": "u/d"}),  # wrong kind's field: extra forbidden
        ("recording", {"dataset_repo_id": "u/d", "single_task": "t", "num_episodes": "lots"}),
    ],
)
def test_options_must_fit_the_kind_422(client, tmp_lerobot_home, kind, options) -> None:
    _make_robot()
    resp = client.post("/api/v1/sessions", json={"kind": kind, "robot": "bench", "options": options})
    assert resp.status_code == 422
    assert resp.json()["code"] == "request.validation"


def test_unknown_kind_is_a_422(client, tmp_lerobot_home) -> None:
    # Not a startable kind this phase (legacy wizard endpoints start it).
    resp = client.post("/api/v1/sessions", json={"kind": "calibration", "robot": "bench"})
    assert resp.status_code == 422


# --- POST /api/v1/sessions: the exclusivity gate -----------------------------


@pytest.mark.parametrize(
    ("patch_target", "holder_kind"),
    [
        ("makermodslab.teleoperate.teleoperation_active", "teleoperation"),
        ("makermodslab.record.recording_active", "recording"),
        ("makermodslab.rollout.inference_active", "inference"),
        ("makermodslab.replay.replay_active", "replay"),
        ("makermodslab.wiggle.wiggle_active", "wiggle"),
    ],
)
def test_start_while_held_409_names_the_holder(client, monkeypatch, patch_target, holder_kind) -> None:
    """The gate outranks robot resolution — even an unknown robot gets the
    held answer while the hardware is claimed (one hardware set per node)."""
    monkeypatch.setattr(patch_target, True)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "ghost"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "session.held"
    assert body["details"]["holder"] == {"kind": holder_kind, "session_id": None}


def test_held_details_carry_the_holder_session_id(client, monkeypatch) -> None:
    notify_session_changed("teleoperation", True)
    holder_id = sessions.tracker.current()["id"]
    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)

    resp = client.post("/api/v1/sessions", json={"kind": "recording", "robot": "ghost"})
    assert resp.status_code == 409
    assert resp.json()["details"]["holder"] == {"kind": "teleoperation", "session_id": holder_id}


def test_busy_refusal_from_a_raced_start_maps_to_held(client, tmp_lerobot_home, monkeypatch) -> None:
    """The feature's own reciprocal check refusing (the gate raced another
    start) surfaces identically to the gate: 409 session.held + holder."""
    _make_robot()
    captured: list = []
    monkeypatch.setattr(
        "makermodslab.teleoperate.handle_start_teleoperation",
        _fake_start(
            "teleoperation",
            captured,
            {"success": False, "message": "Recording is active", "code": ErrorCode.ROBOT_BUSY_RECORDING},
        ),
    )
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "session.held"
    assert body["details"]["holder"]["kind"] == "recording"


def test_non_busy_refusal_passes_through(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    captured: list = []
    refusal = {
        "success": False,
        "status_code": 400,
        "message": "Malformed dataset name",
        "code": ErrorCode.REQUEST_INVALID_NAME,
    }
    monkeypatch.setattr(
        "makermodslab.record.handle_start_recording", _fake_start("recording", captured, refusal)
    )
    resp = client.post(
        "/api/v1/sessions",
        json={
            "kind": "recording",
            "robot": "bench",
            "options": {"dataset_repo_id": "u/whoo/", "single_task": "t"},
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request.invalid_name"
    assert body["detail"] == "Malformed dataset name"


# --- POST /api/v1/sessions: success ------------------------------------------


def test_start_success_201_returns_the_attributed_identity(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    captured: list = []
    monkeypatch.setattr(
        "makermodslab.teleoperate.handle_start_teleoperation", _fake_start("teleoperation", captured)
    )
    resp = client.post(
        "/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench", "owner": "ui-abc"}
    )
    assert resp.status_code == 201
    session = resp.json()["session"]
    assert session["kind"] == "teleoperation"
    assert session["robot"] == "bench"
    assert session["owner"] == "ui-abc"
    assert session["revision"] == 1
    assert session["phase"] is None
    assert session["id"] == sessions.tracker.current()["id"]

    # GET /sessions/current serves the same identity (the lease's
    # expires_in_s is computed at read time, so compare it apart).
    got = client.get("/api/v1/sessions/current").json()["session"]
    assert {k: v for k, v in got.items() if k != "lease"} == {
        k: v for k, v in session.items() if k != "lease"
    }
    assert got["lease"]["owner"] == "ui-abc"


def test_start_success_without_a_claim_event_is_a_500(client, tmp_lerobot_home, monkeypatch) -> None:
    """A feature reporting success without emitting its claim would leave an
    unidentifiable live session — surfaced loudly, never silently."""
    _make_robot()
    monkeypatch.setattr(
        "makermodslab.teleoperate.handle_start_teleoperation",
        lambda request, websocket_manager=None: {"success": True},
    )
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bench"})
    assert resp.status_code == 500
    assert resp.json()["code"] == "internal.unexpected"


# --- request-model construction (pure assembly, per kind) --------------------


def test_teleoperation_request_built_from_the_record(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    captured: list = []
    monkeypatch.setattr(
        "makermodslab.teleoperate.handle_start_teleoperation", _fake_start("teleoperation", captured)
    )
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "teleoperation", "robot": "bench", "options": {"skip_identity_check": True}},
    )
    assert resp.status_code == 201
    req = captured[0]
    assert (req.leader_port, req.follower_port) == ("/dev/l", "/dev/f")
    assert (req.leader_config, req.follower_config) == ("LC", "FC")
    assert req.mode == "single"
    assert req.robot_name == "bench"
    assert req.skip_identity_check is True


def test_bimanual_record_maps_right_arm_fields(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot("bi", mode="bimanual")
    captured: list = []
    monkeypatch.setattr(
        "makermodslab.teleoperate.handle_start_teleoperation", _fake_start("teleoperation", captured)
    )
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "bi"})
    assert resp.status_code == 201
    req = captured[0]
    assert req.mode == "bimanual"
    assert (req.right_leader_port, req.right_follower_port) == ("/dev/rl", "/dev/rf")
    assert (req.right_leader_config, req.right_follower_config) == ("RLC", "RFC")
    assert req.robot_name == "bi"


def test_recording_request_merges_record_and_options(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    captured: list = []
    monkeypatch.setattr("makermodslab.record.handle_start_recording", _fake_start("recording", captured))
    resp = client.post(
        "/api/v1/sessions",
        json={
            "kind": "recording",
            "robot": "bench",
            "options": {
                "dataset_repo_id": "alice/pick",
                "single_task": "pick the cube",
                "num_episodes": 12,
                "fps": 25,
                "push_to_hub": True,
                "tags": ["so101"],
            },
        },
    )
    assert resp.status_code == 201
    req = captured[0]
    assert (req.leader_port, req.follower_port, req.leader_config, req.follower_config) == (
        "/dev/l",
        "/dev/f",
        "LC",
        "FC",
    )
    assert req.robot_name == "bench"  # cameras resolve from this record server-side
    assert (req.dataset_repo_id, req.single_task) == ("alice/pick", "pick the cube")
    assert (req.num_episodes, req.fps) == (12, 25)
    assert req.push_to_hub is True and req.tags == ["so101"]
    assert req.episode_time_s == 30  # untouched defaults stay the feature's own


def test_inference_request_is_follower_only(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot("bi", mode="bimanual", follower_only=True)
    captured: list = []
    monkeypatch.setattr("makermodslab.rollout.handle_start_inference", _fake_start("inference", captured))
    resp = client.post(
        "/api/v1/sessions",
        json={
            "kind": "inference",
            "robot": "bi",
            "options": {
                "policy_ref": "alice/act-pick",
                "task": "pick",
                "camera_bindings": {"top": "workbench"},
                "camera_dims": {"top": {"width": 320, "height": 240}},
                "duration_s": 120,
                "checkpoint_state_dim": 12,
                "eval_episodes": 3,
            },
        },
    )
    assert resp.status_code == 201
    req = captured[0]
    assert (req.follower_port, req.follower_config) == ("/dev/f", "FC")
    assert req.mode == "bimanual"
    assert (req.right_follower_port, req.right_follower_config) == ("/dev/rf", "RFC")
    assert req.robot_name == "bi"
    assert req.policy_ref == "alice/act-pick"
    assert req.camera_bindings == {"top": "workbench"}
    assert req.camera_dims["top"].width == 320 and req.camera_dims["top"].height == 240
    assert (req.duration_s, req.checkpoint_state_dim, req.eval_episodes) == (120, 12, 3)


def test_replay_request_built_from_record_and_options(client, tmp_lerobot_home, monkeypatch) -> None:
    _make_robot()
    captured: list = []
    monkeypatch.setattr("makermodslab.replay.handle_start_replay", _fake_start("replay", captured))
    resp = client.post(
        "/api/v1/sessions",
        json={"kind": "replay", "robot": "bench", "options": {"repo_id": "u/d", "episode_index": 4}},
    )
    assert resp.status_code == 201
    req = captured[0]
    assert (req.repo_id, req.episode_index) == ("u/d", 4)
    assert (req.follower_port, req.follower_config) == ("/dev/f", "FC")
    assert req.robot_name == "bench"


# --- GET /api/v1/sessions/current --------------------------------------------


def test_current_is_null_when_idle(client) -> None:
    assert client.get("/api/v1/sessions/current").json() == {"session": None, "last_ended": None}


def test_current_reports_a_legacy_started_session_and_its_end(client) -> None:
    notify_session_changed("auto_calibration", True, phase="running")
    body = client.get("/api/v1/sessions/current").json()
    assert body["session"]["kind"] == "auto_calibration"
    assert body["session"]["robot"] is None

    notify_session_changed("auto_calibration", False, phase="done")
    body = client.get("/api/v1/sessions/current").json()
    assert body["session"] is None
    assert body["last_ended"]["kind"] == "auto_calibration"
    assert body["last_ended"]["phase"] == "done"


# --- POST /api/v1/sessions/{id}/stop -----------------------------------------


def test_stop_with_no_session_404(client) -> None:
    resp = client.post("/api/v1/sessions/deadbeef/stop")
    assert resp.status_code == 404
    assert resp.json()["code"] == "session.not_found"


def test_stop_with_a_stale_id_404_and_leaves_the_session_alone(client, monkeypatch) -> None:
    notify_session_changed("teleoperation", True)
    live = sessions.tracker.current()

    def must_not_run():
        raise AssertionError("stop dispatched despite an id mismatch")

    monkeypatch.setattr("makermodslab.teleoperate.handle_stop_teleoperation", must_not_run)
    resp = client.post("/api/v1/sessions/0000stale0000/stop")
    assert resp.status_code == 404
    assert resp.json()["code"] == "session.not_found"
    assert sessions.tracker.current() == live


def test_stop_by_id_dispatches_and_returns_final_identity(client, monkeypatch) -> None:
    """Stop works for a legacy-started session (id via the observer), returns
    the kind's stop result verbatim, and reports the ended identity."""
    notify_session_changed("teleoperation", True)
    live_id = sessions.tracker.current()["id"]

    def fake_stop():
        notify_session_changed("teleoperation", False, phase="done")
        return {"success": True, "message": "Teleoperation stopped"}

    monkeypatch.setattr("makermodslab.teleoperate.handle_stop_teleoperation", fake_stop)
    resp = client.post(f"/api/v1/sessions/{live_id}/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"] == {"success": True, "message": "Teleoperation stopped"}
    assert body["session"]["id"] == live_id
    assert body["session"]["phase"] == "done"  # the release event's phase
    assert sessions.tracker.current() is None


def test_stop_of_a_gracefully_releasing_session_reports_it_live(client, monkeypatch) -> None:
    """A stop that only BEGINS the release (teleop's grace) leaves the session
    current, in its releasing phase, still under the same id."""
    notify_session_changed("teleoperation", True)
    live_id = sessions.tracker.current()["id"]

    def fake_stop():
        notify_session_changed("teleoperation", True, phase="releasing")
        return {"success": True, "message": "Releasing"}

    monkeypatch.setattr("makermodslab.teleoperate.handle_stop_teleoperation", fake_stop)
    body = client.post(f"/api/v1/sessions/{live_id}/stop").json()
    assert body["session"]["id"] == live_id
    assert body["session"]["phase"] == "releasing"
    assert body["session"]["revision"] == 2
    assert sessions.tracker.current()["id"] == live_id


def test_stop_of_a_wiggle_is_refused(client) -> None:
    notify_session_changed("wiggle", True)
    live_id = sessions.tracker.current()["id"]
    resp = client.post(f"/api/v1/sessions/{live_id}/stop")
    assert resp.status_code == 409
    assert resp.json()["code"] == "robot.busy.wiggle"
