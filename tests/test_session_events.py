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
"""Tests for makermodslab.session_events — the session_changed event bus seam —
and for the feature modules' transition wiring.

Per CLAUDE.md's testing policy, only the seam itself (a pure helper), the
idle/mutex branches, and claim/release paths existing tests already exercise
with mocked hardware are covered here. Worker-thread and subprocess happy
paths are deliberately NOT tested."""

from __future__ import annotations

import time

import pytest

from makermodslab import session_events


@pytest.fixture(autouse=True)
def _unwire_notifier():
    """Every test starts and ends with no notifier wired, whatever the
    surrounding suite (importing makermodslab.server wires the real one).
    Yields whatever was wired before, for the server-wiring test."""
    previous = session_events._notifier
    session_events.set_notifier(None)
    yield previous
    session_events.set_notifier(previous)


@pytest.fixture
def events():
    """Install a recording fake notifier; yields the list of events it saw."""
    seen: list[dict] = []
    session_events.set_notifier(seen.append)
    return seen


# --- the seam itself ---------------------------------------------------------


def test_notify_builds_the_documented_payload(events) -> None:
    before = time.time()
    session_events.notify_session_changed("teleoperation", True, phase="releasing")
    after = time.time()

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "session_changed"
    assert event["session"] == {"kind": "teleoperation", "active": True, "phase": "releasing"}
    assert before <= event["timestamp"] <= after


def test_notify_phase_defaults_to_none(events) -> None:
    session_events.notify_session_changed("wiggle", False)
    assert events == [
        {
            "type": "session_changed",
            "session": {"kind": "wiggle", "active": False, "phase": None},
            "timestamp": events[0]["timestamp"],
        }
    ]


def test_notify_is_a_noop_when_unwired() -> None:
    # No notifier installed (autouse fixture) — must simply not raise.
    session_events.notify_session_changed("recording", True)


def test_notify_swallows_notifier_exceptions() -> None:
    """A broadcast hiccup must never break the hardware flow that emitted it."""

    def broken(_event: dict) -> None:
        raise RuntimeError("broadcast queue exploded")

    session_events.set_notifier(broken)
    session_events.notify_session_changed("inference", False)


def test_notify_drops_unknown_kinds_without_calling_the_notifier(events) -> None:
    """The wire vocabulary is the mutual-exclusion model's, exactly — a typo'd
    call site is dropped loudly rather than inventing a new kind (or raising
    into a cleanup path)."""
    session_events.notify_session_changed("bogus_feature", True)
    session_events.notify_session_changed("releasing", True)  # a phase, not a kind
    assert events == []


def test_session_kinds_match_the_mutex_features() -> None:
    assert {
        "teleoperation",
        "recording",
        "inference",
        "remote_inference",
        "replay",
        "calibration",
        "auto_calibration",
        "wiggle",
        "hosting",
        "remote_teleoperation",
    } == session_events.SESSION_KINDS


def test_set_notifier_none_unwires(events) -> None:
    session_events.notify_session_changed("replay", True)
    session_events.set_notifier(None)
    session_events.notify_session_changed("replay", False)
    assert len(events) == 1


# --- server wiring -----------------------------------------------------------


def test_server_import_wires_the_manager_notifier(_unwire_notifier) -> None:
    # If the suite already imported server, the autouse fixture stashed the
    # import-time wiring (yielded here); on a standalone run, this import IS
    # what wires it, after the fixture cleared the (None) stash.
    from makermodslab.server import manager

    wired = _unwire_notifier or session_events._notifier
    assert wired == manager.notify_session_changed


def test_manager_notify_session_changed_skips_when_no_clients() -> None:
    """Mirrors notify_jobs_changed: a broadcast with no WS clients is dropped,
    not queued (the event is a refetch hint, and pollers self-heal)."""
    from makermodslab.server import ConnectionManager

    cm = ConnectionManager()
    cm.notify_session_changed({"type": "session_changed"})
    assert cm.broadcast_queue.empty()


def test_manager_notify_session_changed_queues_when_clients_exist() -> None:
    from makermodslab.server import ConnectionManager

    cm = ConnectionManager()
    cm.is_running = True
    cm.active_connections[object()] = None  # any truthy connection map
    event = {"type": "session_changed", "session": {"kind": "wiggle", "active": True, "phase": None}}
    cm.notify_session_changed(event)
    assert cm.broadcast_queue.get_nowait() is event


# --- feature wiring: refused starts and idle stops must NOT notify -----------


def test_refused_teleoperation_start_does_not_notify(events, monkeypatch) -> None:
    from makermodslab.teleoperate import TeleoperateRequest, handle_start_teleoperation

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_teleoperation(
        TeleoperateRequest(
            leader_port="/dev/l", follower_port="/dev/f", leader_config="lc", follower_config="fc"
        )
    )
    assert result["success"] is False
    assert events == []


def test_stop_of_idle_teleoperation_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import teleoperate

    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_thread", None)
    result = teleoperate.handle_stop_teleoperation()
    assert result["success"] is False
    assert events == []


def test_refused_recording_start_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "releasing", False)
    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="/dev/l",
            follower_port="/dev/f",
            leader_config="lc",
            follower_config="fc",
            dataset_repo_id="d",
            single_task="t",
        )
    )
    assert result["success"] is False
    assert events == []


def test_stop_of_idle_recording_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_events", None)
    result = record.handle_stop_recording()
    assert result["success"] is False
    assert events == []


def test_refused_replay_start_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    monkeypatch.setattr(replay, "replay_thread", None)
    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = replay.handle_start_replay(
        replay.ReplayRequest(repo_id="u/d", episode_index=0, follower_port="/dev/f", follower_config="fc")
    )
    assert result["success"] is False
    assert events == []


def test_stop_of_idle_replay_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    result = replay.handle_stop_replay()
    assert result["success"] is False
    assert events == []


def test_refused_calibration_start_does_not_notify(events, monkeypatch) -> None:
    from makermodslab.calibrate import CalibrationRequest, calibration_manager

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = calibration_manager.start_calibration(
        CalibrationRequest(device_type="robot", port="/dev/f", config_file="cfg")
    )
    assert result["success"] is False
    assert events == []


def test_refused_auto_calibration_start_does_not_notify(events, monkeypatch) -> None:
    from makermodslab.auto_calibrate import AutoCalibrationRequest, _AutoCalArmRunner

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    runner = _AutoCalArmRunner()
    result = runner.start(AutoCalibrationRequest(device_type="robot", port="/dev/f", config_file="cfg"))
    assert result["success"] is False
    assert events == []


def test_stop_of_idle_auto_calibration_does_not_notify(events) -> None:
    from makermodslab.auto_calibrate import _AutoCalArmRunner

    result = _AutoCalArmRunner().stop()
    assert result["success"] is False
    assert events == []


def test_refused_wiggle_does_not_notify(events, monkeypatch) -> None:
    import asyncio

    from makermodslab import wiggle

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = asyncio.run(wiggle.wiggle_gripper("/dev/f"))
    assert result["success"] is False
    assert events == []


# --- feature wiring: claim/release on already-mocked paths -------------------


async def test_wiggle_claim_and_release_notify(events, monkeypatch) -> None:
    """The wiggle success path is already exercised hardware-free (see
    tests/test_wiggle.py); the claim and the final release must each emit."""
    from makermodslab import wiggle

    monkeypatch.setattr(wiggle, "_wiggle_gripper_sync", lambda port: None)
    result = await wiggle.wiggle_gripper("/dev/fake")
    assert result["success"] is True
    assert [(e["session"]["kind"], e["session"]["active"]) for e in events] == [
        ("wiggle", True),
        ("wiggle", False),
    ]


async def test_wiggle_release_notifies_even_when_the_drive_fails(events, monkeypatch) -> None:
    from makermodslab import wiggle

    def boom(port: str) -> None:
        raise RuntimeError("no device on port")

    monkeypatch.setattr(wiggle, "_wiggle_gripper_sync", boom)
    result = await wiggle.wiggle_gripper("/dev/fake")
    assert result["success"] is False
    assert [(e["session"]["kind"], e["session"]["active"]) for e in events] == [
        ("wiggle", True),
        ("wiggle", False),
    ]


def test_replay_claim_notifies(events, monkeypatch) -> None:
    """Reuses tests/test_replay.py's mocked-connect start path: a successful
    claim emits before the worker ever runs (the thread here is a no-op)."""
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
    assert [(e["session"]["kind"], e["session"]["active"], e["session"]["phase"]) for e in events] == [
        ("replay", True, "easing_in")
    ]

    # Leave the module idle for the rest of the suite.
    replay.replay_active = False
    replay.replay_thread = None


def test_record_set_phase_records_and_notifies(events, monkeypatch) -> None:
    """record.py's phase helper is the seam every current_phase transition
    goes through — assignment plus hint, no other behavior."""
    from makermodslab import record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")
    record._set_phase("resetting")
    assert record.current_phase == "resetting"
    assert events[-1]["session"] == {"kind": "recording", "active": True, "phase": "resetting"}


def test_rollout_set_phase_notifies_only_while_a_session_is_live(events, monkeypatch) -> None:
    from makermodslab import rollout

    # Idle: no meta — a late stdout line after teardown must not notify.
    monkeypatch.setattr(rollout, "_inference_meta", {})
    rollout._set_phase(rollout.PHASE_CONNECTING)
    assert events == []

    # Live session: the phase lands on the meta AND emits the hint.
    monkeypatch.setattr(rollout, "_inference_meta", {"policy_ref": "u/m", "phase": "starting"})
    rollout._set_phase(rollout.PHASE_CONNECTING)
    assert rollout._inference_meta["phase"] == rollout.PHASE_CONNECTING
    assert events[-1]["session"] == {"kind": "inference", "active": True, "phase": "connecting"}


def test_rollout_fail_startup_when_idle_does_not_notify(events, monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    rollout._fail_startup("boom")
    assert events == []
