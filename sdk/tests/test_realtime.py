"""Realtime layer: pure demux, bounded sampling, lazy websockets import.

House rules: no sleeps, nothing that can block. The demux and the sampling
loop are pure (fake frame sources, scripted clocks); the only socket test is
an open/close smoke through fastapi's TestClient that never receives.

Message shapes are copied verbatim from the server's broadcasts:
teleoperate.py (joint_update), replay.py (replay_joint_update), and
server.py's ConnectionManager notify_* methods / session_events.py.
"""

from __future__ import annotations

import json
import sys

import pytest
from makermodslab_sdk import MakerModsError
from makermodslab_sdk.realtime import (
    WS_PATH,
    JobProgress,
    JobsChanged,
    JointData,
    SessionChanged,
    UnknownEvent,
    collect_joint_frames,
    events_from,
    parse_message,
    ws_url,
)

# --- Real message shapes, verbatim from the server's broadcast call sites ---

TELEOP_FRAME = {
    "type": "joint_update",
    "joints": {"shoulder_pan": 12.5, "gripper": 42.0},
    "timestamp": 1700000000.25,
}

BIMANUAL_FRAME = {
    "type": "joint_update",
    "joints": {"shoulder_pan": 12.5},
    "joints_right": {"shoulder_pan": -3.0},
    "follower_currents_ma": {"left_": 210.0, "right_": 190.0},
    "timestamp": 1700000000.5,
}

REPLAY_FRAME = {
    "type": "replay_joint_update",
    "joints": {"elbow_flex": 90.0},
    "timestamp": 1700000001.0,
}

JOBS_CHANGED = {"type": "jobs_changed", "timestamp": 1700000002.0}

JOB_PROGRESS = {
    "type": "job_progress",
    "jobs": [
        {
            "id": "job-1",
            "state": "running",
            "metrics": {"step": 120, "loss": 0.42},
            "wandb_run_url": None,
            "checkpoint_count": 2,
        }
    ],
    "timestamp": 1700000003.0,
}

SESSION_CHANGED = {
    "type": "session_changed",
    "session": {"kind": "teleoperation", "active": True, "phase": "running"},
    "timestamp": 1700000004.0,
}


# --- Demux -----------------------------------------------------------------


def test_parse_teleop_joint_frame():
    ev = parse_message(TELEOP_FRAME)
    assert isinstance(ev, JointData)
    assert ev.type == "joint_update"
    assert ev.joints == {"shoulder_pan": 12.5, "gripper": 42.0}
    assert ev.timestamp == 1700000000.25
    assert ev.joints_right is None
    assert ev.follower_currents_ma is None


def test_parse_bimanual_joint_frame_keeps_right_arm_and_currents():
    ev = parse_message(BIMANUAL_FRAME)
    assert isinstance(ev, JointData)
    assert ev.joints_right == {"shoulder_pan": -3.0}
    assert ev.follower_currents_ma == {"left_": 210.0, "right_": 190.0}


def test_parse_replay_joint_frame_is_joint_data_too():
    ev = parse_message(REPLAY_FRAME)
    assert isinstance(ev, JointData)
    assert ev.type == "replay_joint_update"
    assert ev.joints == {"elbow_flex": 90.0}


def test_parse_jobs_changed():
    ev = parse_message(JOBS_CHANGED)
    assert isinstance(ev, JobsChanged)
    assert ev.timestamp == 1700000002.0


def test_parse_job_progress_carries_snapshots():
    ev = parse_message(JOB_PROGRESS)
    assert isinstance(ev, JobProgress)
    assert len(ev.jobs) == 1
    assert ev.jobs[0]["id"] == "job-1"
    assert ev.jobs[0]["metrics"]["loss"] == 0.42


def test_parse_session_changed():
    ev = parse_message(SESSION_CHANGED)
    assert isinstance(ev, SessionChanged)
    assert ev.session.kind == "teleoperation"
    assert ev.session.active is True
    assert ev.session.phase == "running"


def test_parse_session_changed_null_phase():
    raw = {
        "type": "session_changed",
        "session": {"kind": "recording", "active": False, "phase": None},
        "timestamp": 1.0,
    }
    ev = parse_message(raw)
    assert isinstance(ev, SessionChanged)
    assert ev.session.phase is None


def test_unknown_type_becomes_unknown_event():
    raw = {"type": "brand_new_event", "payload": 1}
    ev = parse_message(raw)
    assert isinstance(ev, UnknownEvent)
    assert ev.raw == raw


def test_missing_type_becomes_unknown_event():
    raw = {"joints": {"a": 1.0}}
    ev = parse_message(raw)
    assert isinstance(ev, UnknownEvent)
    assert ev.raw == raw


def test_malformed_known_type_becomes_unknown_event():
    # Documented choice: a known "type" whose body doesn't validate is
    # downgraded to UnknownEvent (raw preserved), never an exception.
    raw = {"type": "session_changed", "session": "not-a-dict", "timestamp": 1.0}
    ev = parse_message(raw)
    assert isinstance(ev, UnknownEvent)
    assert ev.raw == raw


def test_non_dict_becomes_unknown_event():
    ev = parse_message(["not", "a", "dict"])  # type: ignore[arg-type]
    assert isinstance(ev, UnknownEvent)
    assert ev.raw == ["not", "a", "dict"]


def test_extra_server_fields_are_kept_not_rejected():
    raw = dict(TELEOP_FRAME, future_field="hello")
    ev = parse_message(raw)
    assert isinstance(ev, JointData)
    assert ev.future_field == "hello"  # type: ignore[attr-defined]


# --- events_from + kinds filter --------------------------------------------


def test_events_from_parses_in_order():
    evs = list(events_from([TELEOP_FRAME, JOBS_CHANGED, SESSION_CHANGED]))
    assert [type(e) for e in evs] == [JointData, JobsChanged, SessionChanged]


def test_events_from_kinds_filter():
    msgs = [TELEOP_FRAME, JOBS_CHANGED, REPLAY_FRAME, SESSION_CHANGED, {"type": "??"}]
    evs = list(events_from(msgs, kinds=(JointData,)))
    assert [e.type for e in evs] == ["joint_update", "replay_joint_update"]
    evs = list(events_from(msgs, kinds=(JobsChanged, SessionChanged)))
    assert [type(e) for e in evs] == [JobsChanged, SessionChanged]


def test_events_from_single_kind_not_wrapped_in_tuple():
    evs = list(events_from([TELEOP_FRAME, JOBS_CHANGED], kinds=JointData))
    assert [type(e) for e in evs] == [JointData]


# --- ws_url ----------------------------------------------------------------


def test_ws_url_http_to_ws():
    assert ws_url("http://localhost:8000") == "ws://localhost:8000" + WS_PATH


def test_ws_url_https_to_wss_and_trailing_slash():
    assert ws_url("https://lab.example/") == "wss://lab.example" + WS_PATH


def test_ws_url_rejects_other_schemes():
    with pytest.raises(MakerModsError, match="http"):
        ws_url("ftp://nope")


def test_ws_path_is_the_v1_route():
    assert WS_PATH == "/api/v1/ws/joint-data"


# --- collect_joint_frames: scripted clock, fake frame source ---------------


class ScriptedClock:
    """monotonic() stand-in: returns scripted values, then keeps repeating the
    last one — the test never waits on real time."""

    def __init__(self, *times: float) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        if len(self._times) > 1:
            return self._times.pop(0)
        return self._times[0]


def frame_source(messages):
    """recv(timeout) stand-in feeding scripted messages, TimeoutError after."""
    queue = list(messages)

    def recv(timeout: float):
        assert timeout > 0  # the loop must never ask for a non-positive wait
        if not queue:
            raise TimeoutError
        return queue.pop(0)

    return recv


def test_collect_stops_at_deadline():
    # Deadline = 0 + 2.0. Remaining checked at 0.5, 1.0, 1.5, 2.0 -> 3 recvs.
    clock = ScriptedClock(0.0, 0.5, 1.0, 1.5, 2.0)
    recv = frame_source([TELEOP_FRAME] * 100)
    frames = collect_joint_frames(recv, 2.0, clock=clock)
    assert len(frames) == 3
    assert all(isinstance(f, JointData) for f in frames)


def test_collect_stops_at_max_frames():
    clock = ScriptedClock(0.0)  # time never advances; only max_frames bounds
    recv = frame_source([TELEOP_FRAME] * 100)
    frames = collect_joint_frames(recv, 2.0, max_frames=5, clock=clock)
    assert len(frames) == 5


def test_collect_returns_empty_when_nothing_broadcasts():
    # No hardware flow running -> the source times out immediately.
    clock = ScriptedClock(0.0)
    frames = collect_joint_frames(frame_source([]), 2.0, clock=clock)
    assert frames == []


def test_collect_skips_control_events_without_counting_them():
    clock = ScriptedClock(0.0)
    recv = frame_source([JOBS_CHANGED, TELEOP_FRAME, SESSION_CHANGED, REPLAY_FRAME])
    frames = collect_joint_frames(recv, 2.0, max_frames=10, clock=clock)
    assert [f.type for f in frames] == ["joint_update", "replay_joint_update"]


def test_collect_accepts_json_strings_like_the_wire():
    clock = ScriptedClock(0.0)
    recv = frame_source([json.dumps(TELEOP_FRAME), json.dumps(REPLAY_FRAME)])
    frames = collect_joint_frames(recv, 2.0, clock=clock)
    assert len(frames) == 2


def test_collect_skips_non_json_frames():
    clock = ScriptedClock(0.0)
    recv = frame_source(["not json{{", json.dumps(TELEOP_FRAME)])
    frames = collect_joint_frames(recv, 2.0, clock=clock)
    assert len(frames) == 1


# --- Client integration (fake socket injected, no network) ------------------


class FakeWs:
    """Context-manager + iterator + recv(timeout) triple, like the sync client."""

    def __init__(self, messages) -> None:
        self._messages = [json.dumps(m) if not isinstance(m, str) else m for m in messages]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        yield from self._messages

    def recv(self, timeout: float | None = None):
        if not self._messages:
            raise TimeoutError
        return self._messages.pop(0)


@pytest.fixture()
def offline_client():
    import httpx
    from makermodslab_sdk import Client

    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), base_url="http://mock"
    )
    with Client("http://mock", http_client=http, check_compatibility=False) as client:
        yield client


def test_client_events_yields_typed_events(offline_client, monkeypatch):
    from makermodslab_sdk import realtime

    monkeypatch.setattr(realtime, "_connect", lambda base_url: FakeWs([TELEOP_FRAME, SESSION_CHANGED]))
    evs = list(offline_client.events())
    assert [type(e) for e in evs] == [JointData, SessionChanged]


def test_client_stream_joints_filters_to_joint_data(offline_client, monkeypatch):
    from makermodslab_sdk import realtime

    monkeypatch.setattr(
        realtime, "_connect", lambda base_url: FakeWs([JOBS_CHANGED, TELEOP_FRAME, REPLAY_FRAME])
    )
    evs = list(offline_client.stream_joints())
    assert [e.type for e in evs] == ["joint_update", "replay_joint_update"]


def test_client_sample_joints_returns_bounded_list(offline_client, monkeypatch):
    from makermodslab_sdk import realtime

    monkeypatch.setattr(realtime, "_connect", lambda base_url: FakeWs([TELEOP_FRAME] * 50))
    frames = offline_client.sample_joints(duration_s=9.0, max_frames=4, clock=ScriptedClock(0.0))
    assert isinstance(frames, list)
    assert len(frames) == 4
    assert all(isinstance(f, JointData) for f in frames)


def test_client_sample_joints_empty_when_source_silent(offline_client, monkeypatch):
    from makermodslab_sdk import realtime

    monkeypatch.setattr(realtime, "_connect", lambda base_url: FakeWs([]))
    assert offline_client.sample_joints(duration_s=1.0, clock=ScriptedClock(0.0)) == []


# --- Missing websockets ------------------------------------------------------


def _hide_websockets(monkeypatch):
    """Make `import websockets...` raise ImportError without uninstalling it."""
    for name in list(sys.modules):
        if name == "websockets" or name.startswith("websockets."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "websockets", None)


@pytest.mark.parametrize("method", ["events", "stream_joints", "sample_joints"])
def test_missing_websockets_raises_helpful_error(offline_client, monkeypatch, method):
    _hide_websockets(monkeypatch)
    with pytest.raises(MakerModsError, match=r'pip install "makermodslab-sdk\[realtime\]"'):
        getattr(offline_client, method)()


def test_missing_websockets_error_is_eager_for_generators(offline_client, monkeypatch):
    # client.events() must raise at CALL time, not on first next() — an agent
    # that never iterates still learns the extra is missing.
    _hide_websockets(monkeypatch)
    with pytest.raises(MakerModsError):
        offline_client.events()


def test_demux_needs_no_websockets(monkeypatch):
    _hide_websockets(monkeypatch)
    ev = parse_message(TELEOP_FRAME)
    assert isinstance(ev, JointData)


# --- Optional e2e smoke: open/close only, never receive ---------------------


def test_ws_route_accepts_and_closes(app):
    """The v1 WS path exists and accepts a connection. Nothing broadcasts
    without hardware, so this NEVER calls receive — open/close only."""
    from fastapi.testclient import TestClient

    with TestClient(app) as http, http.websocket_connect(WS_PATH):
        pass
