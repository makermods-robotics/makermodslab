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
"""Tests for makermodslab.replay — mutex, validation, and idle/status branches.

Per CLAUDE.md's testing policy, the worker thread's happy path (connect,
ease-in, real-time playback loop) is deliberately NOT unit-tested here —
only request schemas, pure helpers, and mutex/idle branches are."""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_replay_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset replay's module-level state around each test, mirroring
    tests/test_rollout.py's _reset_rollout_globals."""
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    monkeypatch.setattr(replay, "replay_thread", None)
    monkeypatch.setattr(replay, "_replay_meta", {})
    monkeypatch.setattr(replay, "_replay_started_at", None)


def _stub_request():
    from makermodslab.replay import ReplayRequest

    return ReplayRequest(
        repo_id="alice/pick",
        episode_index=0,
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
    )


def test_handle_start_replay_blocked_when_teleoperation_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_handle_start_replay_blocked_when_recording_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Recording" in result["message"]


def test_handle_start_replay_blocked_when_inference_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Inference" in result["message"]


def test_handle_start_replay_blocked_when_calibration_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Calibration" in result["message"]


def test_handle_start_replay_blocked_when_auto_calibration_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Auto-calibration" in result["message"]


def test_handle_start_replay_blocked_when_wiggle_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "wiggle" in result["message"].lower()


def test_handle_start_replay_blocked_when_already_active(monkeypatch) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", True)
    result = replay.handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"].lower()


def test_handle_start_replay_blocked_while_previous_worker_still_alive(monkeypatch) -> None:
    """I-series regression shape: replay_active already False but the
    previous worker hasn't actually exited yet must still refuse a new
    start, not race it for the same serial port."""
    from makermodslab import replay

    alive_worker = threading.Thread(target=lambda: None)
    alive_worker._started = threading.Event()  # type: ignore[attr-defined]
    alive_worker.start()
    alive_worker.join()
    # Simulate "still alive" without a real long-running thread: patch is_alive.
    monkeypatch.setattr(alive_worker, "is_alive", lambda: True)
    monkeypatch.setattr(replay, "replay_thread", alive_worker)

    result = replay.handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "still" in result["message"].lower()


def test_handle_start_replay_rejects_bimanual_robot(monkeypatch, tmp_path) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr(
        "makermodslab.replay._load_robot_record",
        lambda name: {"mode": "bimanual", "follower_port": "/dev/x", "follower_config": "c"},
    )
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "bimanual" in result["message"].lower()


def test_handle_start_replay_rejects_action_name_mismatch(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr(
        "makermodslab.replay.get_episode_action_series",
        lambda repo_id, episode_index: {
            "action_names": ["not_a_real_joint.pos"],
            "timestamps": [0.0],
            "values": [[1.0]],
        },
    )

    class _FakeRobot:
        action_features = {"shoulder_pan.pos": float}

    monkeypatch.setattr("makermodslab.replay._connect_follower", lambda request: (_FakeRobot(), []))

    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "match" in result["message"].lower() or "joint" in result["message"].lower()


def test_handle_start_replay_returns_400_when_episode_has_no_action_data(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.replay.get_episode_action_series", lambda repo_id, episode_index: None)

    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400


def test_handle_replay_status_idle() -> None:
    from makermodslab.replay import handle_replay_status

    status = handle_replay_status()
    assert status["replay_active"] is False
    assert status["phase"] == "idle"


def test_handle_stop_replay_when_idle_is_a_noop() -> None:
    from makermodslab.replay import handle_stop_replay

    result = handle_stop_replay()
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "No replay" in result["message"]


# --- ease-in target re-keying -------------------------------------------------
#
# A dataset's action names carry lerobot's robot-level `<motor>.pos` suffix,
# while bus.motors (and rest_pose's target filtering) use bare motor names.
# Passing the unconverted dict through matched zero motors and produced a bare
# "no-pose" — the ease-in silently never wrote a Goal_Position. Verified on real
# hardware before the fix: the worker failed in 7ms with no arm motion.


class _FakeBus:
    """Only the attribute _bus_keyed touches: the motors mapping."""

    def __init__(self, names=("shoulder_pan", "shoulder_lift", "gripper")):
        self.motors = dict.fromkeys(names)


def test_bus_keyed_strips_the_pos_suffix() -> None:
    from makermodslab.replay import _bus_keyed

    bus = _FakeBus()
    frame = {"shoulder_pan.pos": 1.0, "shoulder_lift.pos": -2.0, "gripper.pos": 3.0}

    assert _bus_keyed(frame, bus) == {"shoulder_pan": 1.0, "shoulder_lift": -2.0, "gripper": 3.0}


def test_bus_keyed_accepts_bare_motor_names() -> None:
    """A dataset recorded without the suffix still maps, so the fix can't
    regress an older layout."""
    from makermodslab.replay import _bus_keyed

    bus = _FakeBus()
    frame = {"shoulder_pan": 1.0, "shoulder_lift": -2.0, "gripper": 3.0}

    assert _bus_keyed(frame, bus) == {"shoulder_pan": 1.0, "shoulder_lift": -2.0, "gripper": 3.0}


def test_bus_keyed_ignores_motors_the_frame_has_no_value_for() -> None:
    from makermodslab.replay import _bus_keyed

    assert _bus_keyed({"shoulder_pan.pos": 1.0}, _FakeBus()) == {"shoulder_pan": 1.0}


def test_bus_keyed_output_is_non_empty_for_a_real_lerobot_naming_pair() -> None:
    """The exact pairing that failed on hardware: dataset feature names from
    info.json vs. SO101Follower's bus.motors keys."""
    from makermodslab.replay import _bus_keyed

    bus = _FakeBus(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
    frame = {f"{m}.pos": float(i) for i, m in enumerate(bus.motors)}

    keyed = _bus_keyed(frame, bus)
    assert set(keyed) == set(bus.motors)  # would have been empty() before the fix


def test_handle_start_replay_rejects_joint_names_that_dont_map_onto_the_bus(monkeypatch) -> None:
    """The action_features check can't catch this (it's `.pos`-suffixed on both
    sides), so a target that re-keys onto nothing must be rejected at start
    rather than energizing the arm and failing as "no-pose" in the worker."""
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr(
        "makermodslab.replay.get_episode_action_series",
        lambda repo_id, episode_index: {
            "action_names": ["shoulder_pan.pos"],
            "timestamps": [0.0],
            "values": [[1.0]],
        },
    )

    class _FakeRobot:
        action_features = {"shoulder_pan.pos": float}
        bus = _FakeBus(("a_motor_the_dataset_never_names",))

    cleaned: list[str] = []
    monkeypatch.setattr("makermodslab.replay._connect_follower", lambda request: (_FakeRobot(), []))
    monkeypatch.setattr(
        "makermodslab.replay._cleanup_after_setup_failure",
        lambda *a, **k: cleaned.append("released"),
    )

    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "a_motor_the_dataset_never_names" in result["message"]
    assert cleaned == ["released"]  # the arm must not be left energized


def test_handle_start_replay_accepts_a_correctly_named_dataset(monkeypatch) -> None:
    """Guard against the new check over-rejecting: the normal `.pos` layout
    must still start."""
    from makermodslab import replay
    from makermodslab.replay import handle_start_replay

    motors = ("shoulder_pan", "shoulder_lift", "gripper")
    monkeypatch.setattr(
        "makermodslab.replay.get_episode_action_series",
        lambda repo_id, episode_index: {
            "action_names": [f"{m}.pos" for m in motors],
            "timestamps": [0.0],
            "values": [[1.0, 2.0, 3.0]],
        },
    )

    class _FakeRobot:
        action_features = {f"{m}.pos": float for m in motors}
        bus = _FakeBus(motors)

    monkeypatch.setattr("makermodslab.replay._connect_follower", lambda request: (_FakeRobot(), []))
    monkeypatch.setattr(replay.threading, "Thread", lambda **kw: _NoopThread())

    result = handle_start_replay(_stub_request())
    assert result["success"] is True


class _NoopThread:
    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False


# --- speed cap must not survive into playback --------------------------------
#
# The ease-in stamps RETURN_POS_SPEED into every motor's RAM Goal_Velocity and
# clears it in its own finally — but that restore is best-effort, so one dropped
# serial write leaves a 400 profile cap throttling the WHOLE episode. Observed on
# hardware as "too slow but accurate": a capped servo tracks its setpoints
# cleanly, so it looks like a success in the UI.


class _CapBus:
    def __init__(self, caps, clear_works=True):
        self.motors = dict.fromkeys(caps)
        self.caps = dict(caps)
        self.clear_works = clear_works
        self.clear_calls = 0

    def sync_read(self, reg, normalize=True):
        assert reg == "Goal_Velocity"
        return dict(self.caps)

    def sync_write(self, reg, values, normalize=True):
        if reg == "Goal_Velocity":
            self.clear_calls += 1
            if self.clear_works:
                self.caps = dict.fromkeys(self.caps, 0)


class _CapRobot:
    def __init__(self, bus):
        self.bus = bus


def test_ensure_uncapped_clears_a_surviving_speed_cap(monkeypatch) -> None:
    from makermodslab import replay

    bus = _CapBus({"shoulder_pan": 400, "gripper": 400})
    monkeypatch.setattr(
        replay,
        "clear_goal_velocity",
        lambda robot, side, label: (
            robot.bus.sync_write("Goal_Velocity", dict.fromkeys(robot.bus.motors, 0), normalize=False) or []
        ),
    )

    replay._ensure_uncapped(_CapRobot(bus), "follower arm")

    assert bus.caps == {"shoulder_pan": 0, "gripper": 0}


def test_ensure_uncapped_retries_when_the_cap_survives(monkeypatch) -> None:
    """A cap that survives the first clear must be retried, not silently left to
    throttle the entire replay."""
    from makermodslab import replay

    bus = _CapBus({"shoulder_pan": 400}, clear_works=False)
    monkeypatch.setattr(
        replay,
        "clear_goal_velocity",
        lambda robot, side, label: robot.bus.sync_write("Goal_Velocity", {}, normalize=False) or [],
    )

    replay._ensure_uncapped(_CapRobot(bus), "follower arm")

    assert bus.clear_calls == 2  # initial clear + one retry
    assert bus.caps == {"shoulder_pan": 400}  # still stuck, but now logged loudly


def test_ensure_uncapped_survives_an_unreadable_bus(monkeypatch) -> None:
    """Verification is a nicety — a failed read must not abort the replay."""
    from makermodslab import replay

    class _Dead:
        motors = {"shoulder_pan": None}

        def sync_read(self, reg, normalize=True):
            raise RuntimeError("bus gone")

        def sync_write(self, reg, values, normalize=True):
            pass

    monkeypatch.setattr(replay, "clear_goal_velocity", lambda robot, side, label: [])
    replay._ensure_uncapped(_CapRobot(_Dead()), "follower arm")  # must not raise


# --- stale frames are dropped, not burst-fired -------------------------------
#
# The old loop waited for `now >= target_t`. Once behind, that was already true
# for every overdue frame, so they fired back-to-back with no wait — streaming
# far-apart setpoints at uncapped speed. On hardware the arm lurches through
# them instead of following the recorded path.


def test_max_frame_lag_is_about_one_and_a_half_frames_at_30fps() -> None:
    from makermodslab.replay import _MAX_FRAME_LAG_S

    frame_s = 1 / 30
    assert frame_s < _MAX_FRAME_LAG_S < 3 * frame_s


def _pace(timestamps, clock, lag=None):
    """The worker's frame-admission rule, isolated: which frames get sent when
    the loop is running `clock[i]` seconds into playback at frame i."""
    from makermodslab.replay import _MAX_FRAME_LAG_S

    lag = _MAX_FRAME_LAG_S if lag is None else lag
    last_i = len(timestamps) - 1
    sent = []
    for i, ts in enumerate(timestamps):
        if clock[i] > ts + lag and i != last_i:
            continue
        sent.append(i)
    return sent


def test_pacing_sends_every_frame_when_the_loop_keeps_up() -> None:
    ts = [i / 30 for i in range(10)]
    assert _pace(ts, clock=list(ts)) == list(range(10))


def test_pacing_drops_stale_frames_when_the_loop_falls_behind() -> None:
    """A 1s stall part-way through must not fire the backlog back-to-back."""
    ts = [i / 30 for i in range(10)]
    clock = [t if i < 4 else t + 1.0 for i, t in enumerate(ts)]
    sent = _pace(ts, clock)
    assert sent[:4] == [0, 1, 2, 3]  # before the stall, all sent
    assert set(sent[4:]) == {9}  # backlog dropped, final frame kept


def test_pacing_always_sends_the_final_frame() -> None:
    """Even a badly-behind run must end on the episode's recorded pose."""
    ts = [i / 30 for i in range(5)]
    assert _pace(ts, clock=[t + 10.0 for t in ts]) == [4]


# --- transient serial glitch at connect --------------------------------------


def _connect_stubs(monkeypatch):
    monkeypatch.setattr(
        "makermodslab.replay.setup_follower_calibration_file", lambda cfg, arm_type="so101": "fid"
    )
    monkeypatch.setattr("makermodslab.replay.verify_devices", lambda *a, **k: [])
    monkeypatch.setattr("makermodslab.replay.reset_torque_limit", lambda *a, **k: [])
    monkeypatch.setattr("makermodslab.replay.clear_goal_velocity", lambda *a, **k: [])
    monkeypatch.setattr("makermodslab.replay.time.sleep", lambda s: None)


class _FlakyRobot:
    """configure() fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.configure_calls = 0
        self.calibration = {}
        outer = self

        class _Bus:
            def connect(self):
                pass

            def write_calibration(self, cal):
                pass

        self.bus = _Bus()

        def configure():
            outer.configure_calls += 1
            if outer.configure_calls <= outer.fail_times:
                raise ConnectionError("Failed to write 'Lock' on id_=6 ... no status packet!")

        self.configure = configure


def test_connect_follower_retries_a_transient_configure_failure(monkeypatch) -> None:
    from makermodslab import replay

    _connect_stubs(monkeypatch)
    robot = _FlakyRobot(fail_times=1)
    monkeypatch.setattr(replay, "SO101Follower", lambda cfg: robot)
    monkeypatch.setattr(replay, "SO101FollowerConfig", lambda **kw: None)

    got, warnings = replay._connect_follower(_stub_request())

    assert got is robot
    assert robot.configure_calls == 2  # failed once, retried, succeeded


def test_connect_follower_gives_up_after_the_retry_budget(monkeypatch) -> None:
    """A genuinely dead bus must still fail fast, with an actionable message."""
    from makermodslab import replay

    _connect_stubs(monkeypatch)
    robot = _FlakyRobot(fail_times=99)
    monkeypatch.setattr(replay, "SO101Follower", lambda cfg: robot)
    monkeypatch.setattr(replay, "SO101FollowerConfig", lambda **kw: None)

    with pytest.raises(RuntimeError, match="stopped responding while being configured"):
        replay._connect_follower(_stub_request())
    assert robot.configure_calls == replay._CONNECT_ATTEMPTS


# --- stop during ease-in must abort promptly and still return gently --------
#
# stop_event used to be local to _replay_worker, so handle_stop_replay (which
# only flipped the module-level replay_active flag) had no way to signal it.
# The ease-in's poll loop only checks abort_event.is_set(), so a stop pressed
# during ease-in went unnoticed until the ease-in finished on its own — and
# even then, the early return skipped the graceful return-to-start and fell
# straight into torque release, dropping the arm wherever the ease-in left it.


def test_stop_during_ease_in_signals_the_same_abort_event_and_still_returns_gently(
    monkeypatch,
) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", True)

    calls = []

    class _Bus:
        motors = {"shoulder_pan": None}

        def disconnect(self, disable_torque=False):
            pass

    class _Robot:
        bus = _Bus()

    def fake_capture_rest_pose(bus, normalize=False):
        return {"shoulder_pan": 10}

    def fake_return_to_rest_pose(
        bus, target, abort_event=None, label="arm", normalize=False, tolerance=None, stall_min_progress=None
    ):
        calls.append({"target": dict(target), "normalize": normalize})
        if normalize:
            # This is the ease-in call — simulate a user pressing Stop while
            # it's in flight, the way handle_stop_replay would from another
            # thread/request.
            assert abort_event is not None
            assert not abort_event.is_set()
            replay.handle_stop_replay()
            assert abort_event.is_set(), (
                "handle_stop_replay() must signal the SAME event the ease-in is "
                "polling, or a stop pressed during ease-in goes unnoticed until "
                "the ease-in finishes on its own"
            )
            return False, "cut-short"
        return True, "returned: max delta 0 ticks ()"

    monkeypatch.setattr(replay, "capture_rest_pose", fake_capture_rest_pose)
    monkeypatch.setattr(replay, "return_to_rest_pose", fake_return_to_rest_pose)
    monkeypatch.setattr(replay, "force_disable_torque", lambda robot, label: None)

    action_series = {
        "action_names": ["shoulder_pan.pos"],
        "timestamps": [0.0],
        "values": [[10.0]],
    }

    replay._replay_worker(_Robot(), action_series, None)

    assert len(calls) == 2, "a stop during ease-in must still trigger the graceful stopping return"
    assert calls[1]["normalize"] is False
    assert calls[1]["target"] == {"shoulder_pan": 10}  # returns to the pose captured at session start


# --- shutdown budget must cover two returns, not one ------------------------
#
# Unlike teleoperate.py/record.py, a SIGTERM landing during replay's ease-in
# can owe the RETURN_CEILING_S-bounded ease-in itself AND THEN the
# RETURN_CEILING_S-bounded stopping-phase return (see the ease-in fall-through
# above) — two returns in series, not one. _STOP_AND_WAIT_TIMEOUT_S was copied
# from teleoperate.py's single-return budget and must be sized for replay's
# actual worst case, or shutdown can give up and exit with the arm still
# energized.


def test_stop_and_wait_timeout_budgets_two_returns_not_one() -> None:
    from makermodslab.replay import _STOP_AND_WAIT_TIMEOUT_S
    from makermodslab.rest_pose import RETURN_CEILING_S

    assert _STOP_AND_WAIT_TIMEOUT_S >= 2 * RETURN_CEILING_S + 5.0


def test_replay_status_elapsed_freezes_once_the_session_ends(monkeypatch) -> None:
    """A finished run used to keep ticking upward, so a dead session read live."""
    import time as _time

    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    monkeypatch.setattr(replay, "_replay_started_at", _time.time() - 300.0)
    monkeypatch.setattr(replay, "_replay_meta", {"phase": "done", "played_s": 8.3})

    assert replay.handle_replay_status()["elapsed_s"] == 8.3
