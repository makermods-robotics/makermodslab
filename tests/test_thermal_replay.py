"""Hardware-free tests of the thermal experiment's control and failure paths."""

import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from makermodslab.thermal_replay import ThermalReplayOptions, ThermalTrial, validate_thermal_series


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


class Robot:
    def __init__(self, clock):
        self.clock = clock
        self.pos = {"shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0}
        self.config = SimpleNamespace(joint_limits={"shoulder_lift": (-100, 100), "elbow_flex": (-100, 100)})
        self.bus = SimpleNamespace(
            motors={"shoulder_lift": 1, "elbow_flex": 2},
            _gains={"shoulder_lift": {"kp": 150.0, "kd": 4.5}},
            _last_known_states={},
            last_feedback_time={},
            write=self.write,
        )
        self.commands = []
        self.writes = []
        self.temperature = lambda _: 40
        self.stale = False
        self.stuck = False
        self.on_observation = lambda: None

    def write(self, key, name, val):
        self.writes.append((key, name, val))
        self.bus._gains[name][key.lower()] = val

    def get_observation(self):
        self.on_observation()
        for name in self.bus.motors:
            self.bus._last_known_states[name] = {"temp_mos": self.temperature(self.clock.now), "torque": 3.0}
            self.bus.last_feedback_time[name] = self.clock.now - (1 if self.stale else 0)
        return dict(self.pos)

    def send_action(self, action):
        self.commands.append((self.clock.now, dict(action)))
        if not self.stuck:
            self.pos.update(action)
        return action


def series():
    return {
        "action_names": ["shoulder_lift.pos", "elbow_flex.pos"],
        "timestamps": [i / 10 for i in range(11)],
        "values": [[min(i, 10 - i), 0.0] for i in range(11)],
    }


@pytest.fixture
def trial(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("makermodslab.thermal_replay.time.monotonic", clock.time)
    monkeypatch.setattr("makermodslab.thermal_replay.time.time", clock.time)
    monkeypatch.setattr("makermodslab.thermal_replay.time.sleep", clock.sleep)
    robot = Robot(clock)
    updates = []
    runner = ThermalTrial(
        robot,
        series(),
        ThermalReplayOptions(duration_s=10, rest_pose_confirmed=True),
        threading.Event(),
        threading.Event(),
        lambda p, s: updates.append((p, s)),
        output=tmp_path,
    )
    return runner, robot, clock, updates


def test_repeats_for_requested_duration_returns_captured_pose_and_logs(trial):
    runner, robot, _, _ = trial
    # Rest is deliberately distinct from episode frame 0.
    robot.pos["shoulder_lift.pos"] = 1.0
    result = runner.run()
    assert result["result"] == "completed"
    assert result["elapsed_s"] == 10
    assert result["cycles"] >= 9
    assert result["rest_reached"]
    assert robot.commands[-1][1]["shoulder_lift.pos"] == 1.0
    assert robot.writes == []
    summary = json.loads((runner.root / "summary.json").read_text())
    assert summary["motors"]["follower.shoulder_lift"]["rms_torque_nm"] == 3
    assert (runner.root / "samples.jsonl").stat().st_size > 100


@pytest.mark.parametrize(
    "temperature, expected, critical",
    [
        (65, "completed", False),
        (65.01, "overheated", False),
        (70, "overheated", False),
        (70.01, "overheated", True),
    ],
)
def test_temperature_boundaries_and_return(trial, temperature, expected, critical):
    runner, robot, clock, _ = trial
    origin = clock.now
    robot.temperature = lambda now: temperature if now > origin + 2 else 40
    result = runner.run()
    assert result["result"] == expected
    assert result["critical"] == critical
    assert result["rest_reached"]
    assert robot.commands[-1][1] == runner.rest
    if expected == "overheated":
        assert result["elapsed_s"] < 3


def test_already_hot_never_starts_motion_cycle(trial):
    runner, robot, _, _ = trial
    robot.temperature = lambda _: 66
    result = runner.run()
    assert result["result"] == "overheated"
    assert result["cycles"] == 0
    assert result["rest_reached"]
    assert all(action == runner.rest for _, action in robot.commands)


def test_missing_feedback_does_not_pass_or_claim_return(trial):
    runner, robot, clock, _ = trial
    robot.on_observation = lambda: setattr(robot, "stale", clock.now > 1002)
    result = runner.run()
    assert result["result"] == "return_failed"
    assert result["trigger_result"] == "feedback_fault"
    assert not result["rest_reached"]
    assert "Motors remain energized" in result["message"]


def test_reduced_stiffness_restores_original_gain_before_return(trial):
    runner, robot, clock, _ = trial
    runner.options.experiment = "shoulder_kp_85"
    robot.on_observation = lambda: runner.stop.set() if clock.now > 1002 else None
    result = runner.run()
    assert result["result"] == "stopped"
    assert robot.writes == [("Kp", "shoulder_lift", 127.5), ("Kp", "shoulder_lift", 150.0)]
    assert robot.bus._gains["shoulder_lift"] == {"kp": 150.0, "kd": 4.5}


def test_failed_return_never_becomes_a_pass(trial):
    runner, robot, clock, _ = trial

    def stuck():
        if clock.now > 1002.4:
            robot.stuck = True
            runner.stop.set()

    robot.on_observation = stuck
    result = runner.run()
    assert result["result"] == "return_failed"
    assert not result["rest_reached"]


def test_overheat_during_return_invalidates_completed_run(trial):
    runner, robot, _, _ = trial
    original = runner.move_to

    def move(pose, **kwargs):
        if runner.status["result"] == "completed":
            robot.temperature = lambda _: 66
        original(pose, **kwargs)

    runner.move_to = move
    result = runner.run()
    assert result["result"] == "overheated"
    assert result["rest_reached"]


def test_outside_limits_refused_before_playback(trial):
    runner, robot, _, _ = trial
    runner.series["values"][3][0] = 999
    result = runner.run()
    assert result["result"] == "invalid_setup"
    assert result["rest_reached"]
    assert all(a == runner.rest for _, a in robot.commands)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["timestamps"].__setitem__(0, float("nan")),
        lambda s: s["timestamps"].__setitem__(3, 0.2),
        lambda s: s["values"][2].__setitem__(0, float("inf")),
        lambda s: s["values"][-1].__setitem__(0, 10),
        lambda s: s["values"].pop(),
    ],
)
def test_invalid_trajectory_preflight(mutate):
    data = series()
    mutate(data)
    with pytest.raises(ValueError):
        validate_thermal_series(data, "maker", "single")


def test_requires_supported_rest_and_single_maker():
    validate_thermal_series(series(), "maker", "single")
    for arm, mode in [("metal", "single"), ("maker", "bimanual")]:
        with pytest.raises(ValueError):
            validate_thermal_series(series(), arm, mode)
    for fields in [
        {},
        {"rest_pose_confirmed": False},
        {"rest_pose_confirmed": True, "duration_s": float("nan")},
    ]:
        with pytest.raises(ValidationError):
            ThermalReplayOptions(**fields)


def test_tracking_error_ends_loop_and_does_not_pass(trial):
    runner, robot, _, _ = trial
    robot.stuck = True
    runner.series["values"] = [[0.0, 0.0]] + [[20.0, 0.0]] * 9 + [[0.0, 0.0]]
    result = runner.run()
    assert result["result"] == "tracking_fault"
    assert result["elapsed_s"] < 2
    assert result["rest_reached"]


def test_playback_lateness_invalidates_benchmark(trial):
    runner, robot, clock, _ = trial

    def slow_feedback():
        if runner.started is not None:
            clock.now += 0.15

    robot.on_observation = slow_feedback
    result = runner.run()
    assert result["result"] == "timing_fault"
    assert result["rest_reached"]


def test_lease_stop_during_return_does_not_cut_power(monkeypatch):
    from makermodslab import replay

    stop, release = threading.Event(), threading.Event()
    monkeypatch.setattr(replay, "replay_active", True)
    monkeypatch.setattr(replay, "_stop_event", stop)
    monkeypatch.setattr(replay, "_release_now", release)
    monkeypatch.setattr(replay, "_replay_meta", {"phase": "playing", "thermal_test": {"result": "running"}})
    replay.handle_stop_replay()
    assert stop.is_set()
    assert replay.replay_active
    replay.handle_stop_replay()
    assert not release.is_set()
    replay.handle_stop_replay(release_now=True)
    assert release.is_set()


def test_worker_retains_connection_until_explicit_release_after_failed_return(monkeypatch):
    from makermodslab import replay

    release = threading.Event()
    waiting = threading.Event()
    disconnected = []

    class FakeTrial:
        names = ["shoulder_lift"]
        observation = {"shoulder_lift.pos": 0.0}
        status = {
            "result": "return_failed",
            "message": "fault",
            "rest_reached": False,
            "actuators": [],
            "critical": False,
        }

        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return dict(self.status)

        def sample(self, **kwargs):
            waiting.set()

    monkeypatch.setattr(replay, "ThermalTrial", FakeTrial)
    monkeypatch.setattr(replay, "replay_active", True)
    monkeypatch.setattr(replay, "_release_now", release)
    monkeypatch.setattr(replay, "_replay_meta", {})
    monkeypatch.setattr(replay, "notify_session_changed", lambda *a, **k: None)
    monkeypatch.setattr(
        replay.arm_registry,
        "get",
        lambda _: SimpleNamespace(release_torque=lambda *a: disconnected.append("power")),
    )
    monkeypatch.setattr(replay, "_disconnect_replay_followers", lambda *a: disconnected.append("bus"))
    robot = SimpleNamespace(send_action=lambda _: None)
    worker = threading.Thread(target=replay._thermal_replay_worker, args=(robot, {}, None, "maker", None))
    worker.start()
    try:
        assert waiting.wait(2)
        assert not disconnected
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert disconnected == ["power", "bus"]
    assert replay._replay_meta["phase"] == "error"
    assert not replay.replay_active


def test_session_options_forward_thermal_configuration():
    from makermodslab.schemas.sessions import ReplayOptions
    from makermodslab.sessions import _build_replay_request

    options = ReplayOptions(
        repo_id="owner/episode",
        episode_index=0,
        thermal_test={"rest_pose_confirmed": True, "experiment": "shoulder_kp_85"},
    )
    request = _build_replay_request(
        {"name": "arm", "arm_type": "maker", "follower_port": "can0", "follower_config": "arm"}, options
    )
    assert request.thermal_test.experiment == "shoulder_kp_85"
    assert request.thermal_test.duration_s == 300


def test_loop_boundary_uses_bounded_alignment(trial):
    runner, _, _, _ = trial
    runner.series["values"][-1][0] = 1.5
    moves = []
    original = runner.move_to

    def move(pose, **kwargs):
        moves.append((dict(pose), kwargs.get("returning", True)))
        return original(pose, **kwargs)

    runner.move_to = move
    result = runner.run()
    assert result["result"] == "completed"
    assert sum(not returning for _, returning in moves) > 2
    assert moves[-1] == (runner.rest, True)
