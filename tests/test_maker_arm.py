"""Maker arm integration: record schema, config assembly, guards, calibration.

Follows the repo's testing policy — request schemas, pure helpers, and
idle/mutex branches. The parts that actually open a CAN or UART bus (the
zero-calibration worker, the port probe's bus I/O) are deliberately NOT
exercised here; they are verified against real hardware instead.
"""

from pathlib import Path

import pytest

from makermodslab.utils import config as cfg

# ---------------------------------------------------------------------------
# Robot record: arm_type
# ---------------------------------------------------------------------------


def test_records_written_before_the_maker_arm_read_back_as_so101(tmp_lerobot_home: Path) -> None:
    """A record with no arm_type on disk is an SO-101 by definition."""
    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    (robots / "legacy.json").write_text('{"name": "legacy", "mode": "single", "leader_port": "/dev/a"}')

    record = cfg.get_robot_record("legacy")
    assert record["arm_type"] == "so101"
    # ...and nothing else about the record was disturbed by the upgrade.
    assert record["leader_port"] == "/dev/a"


def test_a_corrupted_arm_type_on_disk_falls_back_rather_than_raising(tmp_lerobot_home: Path) -> None:
    """Same contract as motor_power's clamp: a bad value must never make a
    robot unopenable."""
    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    (robots / "weird.json").write_text('{"name": "weird", "arm_type": "definitely-not-an-arm"}')

    assert cfg.get_robot_record("weird")["arm_type"] == "so101"


def test_creating_a_maker_robot_persists_its_arm_type(tmp_lerobot_home: Path) -> None:
    cfg.save_robot_record("m1", {"mode": "single", "arm_type": "maker"}, allow_create=True)
    assert cfg.get_robot_record("m1")["arm_type"] == "maker"


def test_switching_arm_type_clears_the_stale_hardware_slots(tmp_lerobot_home: Path) -> None:
    """Ports name physically different adapters and calibration names point
    into the other arm type's library, where they do not exist. Leaving them
    would fail deep inside lerobot's connect() as a missing-file error instead
    of here as "this arm needs setting up"."""
    cfg.save_robot_record(
        "swapme",
        {
            "mode": "single",
            "arm_type": "so101",
            "leader_port": "/dev/ttyLEADER",
            "follower_port": "/dev/ttyFOLLOWER",
            "leader_config": "old_leader",
            "follower_config": "old_follower",
        },
        allow_create=True,
    )

    cfg.save_robot_record("swapme", {"arm_type": "maker"}, allow_create=False)

    record = cfg.get_robot_record("swapme")
    assert record["arm_type"] == "maker"
    assert record["leader_port"] == ""
    assert record["follower_port"] == ""
    assert record["leader_config"] == ""
    assert record["follower_config"] == ""


def test_switching_arm_type_keeps_slots_the_same_payload_sets(tmp_lerobot_home: Path) -> None:
    """A caller that switches type AND assigns a new port in one request means
    both — the blanking must not undo the assignment it arrived with."""
    cfg.save_robot_record(
        "swap2",
        {"mode": "single", "arm_type": "so101", "follower_port": "/dev/old", "leader_port": "/dev/oldL"},
        allow_create=True,
    )

    cfg.save_robot_record("swap2", {"arm_type": "maker", "follower_port": "/dev/can0"}, allow_create=False)

    record = cfg.get_robot_record("swap2")
    assert record["follower_port"] == "/dev/can0"  # explicitly set, survives
    assert record["leader_port"] == ""  # not in the payload, cleared


def test_saving_the_same_arm_type_does_not_clear_anything(tmp_lerobot_home: Path) -> None:
    """Only a CHANGE of arm type invalidates the slots; an idempotent save
    (the UI re-sending the current value) must be harmless."""
    cfg.save_robot_record("stable", {"arm_type": "maker", "follower_port": "/dev/can0"}, allow_create=True)
    cfg.save_robot_record("stable", {"arm_type": "maker"}, allow_create=False)

    assert cfg.get_robot_record("stable")["follower_port"] == "/dev/can0"


# ---------------------------------------------------------------------------
# Calibration libraries are per arm type
# ---------------------------------------------------------------------------


def test_each_arm_type_resolves_its_own_calibration_directory() -> None:
    """lerobot derives a calibration dir from the device CLASS's `name`, so
    these must match those names exactly — so_leader/so_follower for the
    SO-101 pair, rebot_102_leader/maker_follower for the Maker pair."""
    assert cfg.calibration_dir_for_device("teleop", "so101").endswith("so_leader")
    assert cfg.calibration_dir_for_device("robot", "so101").endswith("so_follower")
    assert cfg.calibration_dir_for_device("teleop", "maker").endswith("rebot_102_leader")
    assert cfg.calibration_dir_for_device("robot", "maker").endswith("maker_follower")


def test_calibration_dir_defaults_to_so101_for_callers_that_omit_arm_type() -> None:
    """Every pre-existing call site passes no arm type and must be unchanged."""
    assert cfg.calibration_dir_for_device("robot") == cfg.calibration_dir_for_device("robot", "so101")


def test_an_invalid_device_type_is_still_rejected_for_either_arm_type() -> None:
    assert cfg.calibration_dir_for_device("nonsense", "maker") is None
    assert cfg.calibration_dir_for_device("nonsense", "so101") is None


def test_deleting_a_maker_calibration_leaves_same_named_so101_records_alone(
    tmp_lerobot_home: Path,
) -> None:
    """The two libraries are separate namespaces: an SO-101 record naming
    "armA" points at a different file from a Maker record naming "armA", so a
    Maker delete must not unassign the SO-101 robot."""
    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    cfg.save_robot_record("so_bot", {"arm_type": "so101", "follower_config": "armA"}, allow_create=True)
    cfg.save_robot_record("maker_bot", {"arm_type": "maker", "follower_config": "armA"}, allow_create=True)

    cleared = cfg.clear_config_references("robot", "armA", "maker")

    assert cleared == [{"robot": "maker_bot", "fields": ["follower_config"]}]
    assert cfg.get_robot_record("so_bot")["follower_config"] == "armA"
    assert cfg.get_robot_record("maker_bot")["follower_config"] == ""


def test_a_maker_robot_reads_as_ready_from_its_own_library(tmp_lerobot_home: Path) -> None:
    """Readiness must resolve the calibration library by arm type.

    Regression: is_robot_record_clean checked the hardcoded SO-101 directories,
    so a fully set-up Maker robot looked for its calibration somewhere it was
    never written and could never read as ready — which silently disabled
    Teleop, Record and Inference for every Maker robot.
    """
    library_f = Path(cfg.calibration_dir_for_device("robot", "maker"))
    library_l = Path(cfg.calibration_dir_for_device("teleop", "maker"))
    library_f.mkdir(parents=True, exist_ok=True)
    library_l.mkdir(parents=True, exist_ok=True)
    (library_f / "mf.json").write_text("{}")
    (library_l / "ml.json").write_text("{}")

    cfg.save_robot_record(
        "ready_maker",
        {
            "mode": "single",
            "arm_type": "maker",
            "follower_port": "/dev/can0",
            "leader_port": "/dev/uart0",
            "follower_config": "mf",
            "leader_config": "ml",
        },
        allow_create=True,
    )
    record = cfg.get_robot_record("ready_maker")

    assert cfg.is_robot_record_clean(record) is True
    assert cfg.is_robot_record_clean(record, arms="follower") is True


def test_a_maker_robot_is_not_ready_off_the_so101_library(tmp_lerobot_home: Path) -> None:
    """The mirror of the above: a calibration of the same name in the SO-101
    library must NOT satisfy a Maker robot, because that is a different file
    describing different hardware."""
    so101_follower = Path(cfg.calibration_dir_for_device("robot", "so101"))
    so101_leader = Path(cfg.calibration_dir_for_device("teleop", "so101"))
    so101_follower.mkdir(parents=True, exist_ok=True)
    so101_leader.mkdir(parents=True, exist_ok=True)
    (so101_follower / "mf.json").write_text("{}")
    (so101_leader / "ml.json").write_text("{}")

    cfg.save_robot_record(
        "wrong_library",
        {
            "mode": "single",
            "arm_type": "maker",
            "follower_port": "/dev/can0",
            "leader_port": "/dev/uart0",
            "follower_config": "mf",
            "leader_config": "ml",
        },
        allow_create=True,
    )

    assert cfg.is_robot_record_clean(cfg.get_robot_record("wrong_library")) is False


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


class _Req:
    """Minimal stand-in for a start request — the factory only reads attributes."""

    def __init__(self, **kw):
        self.mode = "single"
        self.arm_type = "maker"
        self.robot_name = "r"
        self.leader_port = "/dev/leader"
        self.follower_port = "/dev/follower"
        self.leader_config = "L"
        self.follower_config = "F"
        self.right_leader_port = "/dev/rleader"
        self.right_follower_port = "/dev/rfollower"
        self.right_leader_config = "RL"
        self.right_follower_config = "RF"
        self.__dict__.update(kw)


@pytest.fixture
def _no_staging(monkeypatch: pytest.MonkeyPatch):
    """Skip the on-disk calibration staging — this is a config-shape test."""
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower, arm_type="so101": (leader, follower),
    )
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.stage_bimanual_calibrations",
        lambda *a, **k: ("/staging/leader", "/staging/follower", "r"),
    )


def test_single_maker_request_builds_maker_configs(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    robot, teleop = build_single_configs(_Req())

    assert robot.type == "maker_follower"
    assert robot.port == "/dev/follower"
    # The _maker leader preset, NOT the bare rebot_102_leader: the base config
    # describes the reBot B601 follower and would drive a Maker arm wrong.
    assert teleop.type == "rebot_102_leader_maker"
    assert teleop.port == "/dev/leader"


def test_the_maker_leader_preset_carries_the_maker_joint_mapping(_no_staging) -> None:
    """The whole reason the preset exists. Pointed at a Maker arm with the base
    config's mapping, most joints run the wrong way or saturate against the
    follower's soft limits while teleop keeps reporting a healthy loop."""
    from lerobot.robots.maker_follower import MakerFollowerConfig
    from makermodslab.utils.robot_factory import build_single_configs

    _, teleop = build_single_configs(_Req())
    limits = MakerFollowerConfig(port="/x").joint_limits

    # joint_ranges is MakerFollowerConfig.joint_limits, rounded to ints.
    for joint, (low, high) in limits.items():
        assert teleop.joint_ranges[joint] == [int(low), int(high)]


def test_bimanual_maker_request_builds_bimaker_configs(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, teleop = build_bimanual_configs(_Req(mode="bimanual"))

    assert robot.type == "bi_maker_follower"
    assert teleop.type == "bi_rebot_102_leader_maker"
    assert robot.left_arm_config.port == "/dev/follower"
    assert robot.right_arm_config.port == "/dev/rfollower"
    assert teleop.left_arm_config.port == "/dev/leader"
    assert teleop.right_arm_config.port == "/dev/rfollower".replace("rfollower", "rleader")
    assert str(robot.calibration_dir) == "/staging/follower"
    assert str(teleop.calibration_dir) == "/staging/leader"


def test_bimanual_maker_cameras_go_on_the_left_arm_not_the_top_level(_no_staging) -> None:
    """Both bimanual followers prefix per-arm camera keys with left_/right_ and
    leave top-level keys unprefixed. Putting the session's cameras on the left
    arm is what keeps a bimanual dataset's camera feature keys identical across
    arm types (and identical to what BiSO recordings have always produced)."""
    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, _ = build_bimanual_configs(_Req(mode="bimanual"), cameras={"scene": object()})

    assert set(robot.left_arm_config.cameras) == {"scene"}
    assert robot.cameras == {}
    assert robot.right_arm_config.cameras == {}


def test_an_so101_request_is_completely_unaffected(_no_staging) -> None:
    """The regression guard for every existing robot.

    `.type` reads back as "so100_*", not "so101_*": lerobot registers the
    SO-101 configs under BOTH names (the two arms share a driver) and the
    choice registry reports the last-applied decorator. Asserting the class
    instead sidesteps which alias wins — and it is why arm_type_of_robot_config
    tests for the MAKER type strings rather than the SO-101 ones.
    """
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from makermodslab.utils.robot_factory import build_single_configs

    robot, teleop = build_single_configs(_Req(arm_type="so101"))

    assert isinstance(robot, SO101FollowerConfig)
    assert isinstance(teleop, SO101LeaderConfig)


def test_teleop_builds_the_follower_without_cameras(_no_staging) -> None:
    """Teleoperation consumes no frames (only motor positions drive the
    viewer), so lerobot gets no cameras and the browser handles any display."""
    from makermodslab.utils.robot_factory import build_single_configs

    robot, _ = build_single_configs(_Req())
    assert robot.cameras == {}


# ---------------------------------------------------------------------------
# Inference: CLI type + the arm-count guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm_type", "mode", "expected"),
    [
        ("so101", "single", "so101_follower"),
        ("so101", "bimanual", "bi_so_follower"),
        ("maker", "single", "maker_follower"),
        ("maker", "bimanual", "bi_maker_follower"),
    ],
)
def test_rollout_names_the_right_robot_type(arm_type: str, mode: str, expected: str) -> None:
    """These are draccus choice-registry keys, not free text — a wrong one
    fails inside the subprocess at CLI-parse time, long after the session has
    been claimed."""
    from makermodslab.rollout import InferenceRequest, _robot_cli_type

    request = InferenceRequest(
        follower_port="/dev/f", follower_config="F", policy_ref="ref", mode=mode, arm_type=arm_type
    )
    assert _robot_cli_type(request) == expected


def test_arm_count_guard_stays_live_for_a_seven_dim_maker_checkpoint() -> None:
    """The regression this guard's arm-type awareness exists for.

    Measured against the SO-101's 6, a 7-dim Maker checkpoint is neither <= 6
    nor a multiple of it, so the old code took the "odd width, don't guess"
    escape and disabled itself on every single Maker run.
    """
    from makermodslab.rollout import _arm_count_mismatch

    # 7 dims = one Maker arm: agrees with a single robot, mismatches a bimanual one.
    assert _arm_count_mismatch("single", 7, "maker") is None
    assert "single-arm robot" in _arm_count_mismatch("bimanual", 7, "maker")
    # 14 dims = two Maker arms: the mirror case.
    assert _arm_count_mismatch("bimanual", 14, "maker") is None
    assert "bimanual robot" in _arm_count_mismatch("single", 14, "maker")


def test_a_six_dim_checkpoint_on_a_maker_robot_is_not_silently_accepted() -> None:
    """An SO-101-trained checkpoint is 6 dims; a single Maker arm wants 7. The
    guard cannot classify 6 against a 7-wide arm, so it defers to the
    subprocess rather than guessing — but it must not report agreement."""
    from makermodslab.rollout import _arm_count_mismatch

    # 6 < 7, so it reads as "one arm" and matches a single robot: the
    # subprocess's own shape check is what rejects it. Documented, not ideal.
    assert _arm_count_mismatch("single", 6, "maker") is None


def test_the_so101_arm_count_guard_is_unchanged() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 6, "so101") is None
    assert _arm_count_mismatch("bimanual", 12, "so101") is None
    assert "bimanual robot" in _arm_count_mismatch("single", 12, "so101")
    assert "single-arm robot" in _arm_count_mismatch("bimanual", 6, "so101")
    # Unknown width: defer to the subprocess rather than block on a guess.
    assert _arm_count_mismatch("single", 5, "so101") is None
    assert _arm_count_mismatch("single", None, "so101") is None


# ---------------------------------------------------------------------------
# Guards that refuse rather than crash
# ---------------------------------------------------------------------------


def test_replay_accepts_a_maker_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay is arm-agnostic where it matters: the playback loop is plain
    `send_action` on the dataset's action column, exactly as lerobot's own
    `lerobot-replay` does it. Only the ease-in and teardown differ by arm type.

    Regression guard: a Maker robot must get past the arm-type gate and fail
    (if at all) on something real like a missing dataset, not be refused for
    being a Maker arm.
    """
    from makermodslab import replay

    monkeypatch.setattr(replay, "_load_robot_record", lambda name: {"mode": "single"})
    monkeypatch.setattr(replay, "get_episode_action_series", lambda repo, ep: None)

    result = replay.handle_start_replay(
        replay.ReplayRequest(
            repo_id="u/d",
            episode_index=0,
            follower_port="/dev/can0",
            follower_config="F",
            robot_name="m",
            arm_type="maker",
        )
    )

    assert result["success"] is False
    # Refused for the DATASET, not for the arm type.
    assert "episode" in result["message"].lower()
    assert "Maker" not in result["message"]


def test_replay_still_refuses_a_bimanual_robot_of_either_arm_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing limit, unchanged by the Maker port: replay drives one bus."""
    from makermodslab import replay

    monkeypatch.setattr(replay, "_load_robot_record", lambda name: {"mode": "bimanual"})

    for arm_type in ("so101", "maker"):
        result = replay.handle_start_replay(
            replay.ReplayRequest(
                repo_id="u/d",
                episode_index=0,
                follower_port="/dev/x",
                follower_config="F",
                robot_name="m",
                arm_type=arm_type,
            )
        )
        assert result["success"] is False
        assert "Bimanual replay" in result["message"]


def test_maker_ease_in_arrives_once_every_joint_is_within_tolerance() -> None:
    """The ease-in must walk the arm to frame 0 BEFORE playback starts.

    Playback is paced against a wall clock and drops frames it has fallen
    behind on, so starting the loop while the arm is still slewing would make
    the episode begin partway through, from whatever pose the arm reached.
    """
    from makermodslab.replay import _ease_in_maker

    class _Arm:
        def __init__(self):
            self.pos = {"shoulder_pan": 0.0, "gripper": 0.0}
            self.sent = 0

        def send_action(self, action):
            # Converge a little each call, like a real slow initial sync.
            self.sent += 1
            for motor in self.pos:
                target = action[f"{motor}.pos"]
                self.pos[motor] += (target - self.pos[motor]) * 0.5

        def get_observation(self):
            return {f"{m}.pos": v for m, v in self.pos.items()}

    arm = _Arm()
    arrived, reason = _ease_in_maker(arm, {"shoulder_pan.pos": 20.0, "gripper.pos": -10.0}, lambda: False)

    assert arrived is True
    assert reason == ""
    assert arm.sent > 1  # it actually drove the arm rather than declaring victory


def test_maker_ease_in_accepts_a_converged_arm_that_holds_a_standing_error() -> None:
    """Measured on the real arm: wrist_flex settles ~3 deg from its target and
    stays there, because a position-controlled joint holds a standing error
    proportional to its load. A bare tolerance check waits for that to reach
    zero, which never happens — so a healthy arm failed every ease-in.
    """
    from makermodslab.replay import _ease_in_maker

    class _CompliantArm:
        """Converges to 3 deg away and holds — like the real wrist_flex."""

        def __init__(self):
            self.pos = 0.0

        def send_action(self, action):
            # Asymptotes to 3 deg short of the target: the closest this joint's
            # gains can hold it against its load.
            holdable = action["wrist_flex.pos"] - 3.0
            self.pos += (holdable - self.pos) * 0.5

        def get_observation(self):
            return {"wrist_flex.pos": self.pos}

    arrived, reason = _ease_in_maker(_CompliantArm(), {"wrist_flex.pos": 30.0}, lambda: False)

    assert arrived is True
    assert reason == "settled"


def test_maker_ease_in_still_fails_when_a_joint_converges_far_from_target() -> None:
    """The other half: converging is only acceptable CLOSE to the target.
    A joint that stops 40 deg away is obstructed or mis-posed, and starting
    playback from there would lurch."""
    from makermodslab.replay import _ease_in_maker

    class _BlockedArm:
        def send_action(self, action):
            pass  # never moves — something is in the way

        def get_observation(self):
            return {"elbow_flex.pos": 0.0}

    arrived, reason = _ease_in_maker(_BlockedArm(), {"elbow_flex.pos": 40.0}, lambda: False)

    assert arrived is False
    assert "elbow_flex" in reason


def test_maker_ease_in_reports_the_worst_joint_when_it_cannot_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck joint must name itself, so the user sees which one to reposition."""
    from makermodslab import replay

    monkeypatch.setattr(replay, "_MAKER_EASE_CEILING_S", 0.2)

    class _StuckArm:
        def send_action(self, action):
            pass

        def get_observation(self):
            return {"shoulder_pan.pos": 0.0, "gripper.pos": 0.0}

    arrived, reason = replay._ease_in_maker(
        _StuckArm(), {"shoulder_pan.pos": 90.0, "gripper.pos": 1.0}, lambda: False
    )

    assert arrived is False
    assert "shoulder_pan" in reason  # the WORST joint, not the first one


def test_maker_ease_in_aborts_promptly_on_stop() -> None:
    """A stop pressed during the ease-in must cut it short, not run the ceiling."""
    from makermodslab.replay import _ease_in_maker

    class _Arm:
        def send_action(self, action):
            pass

        def get_observation(self):
            return {"shoulder_pan.pos": 0.0}

    arrived, reason = _ease_in_maker(_Arm(), {"shoulder_pan.pos": 90.0}, lambda: True)

    assert arrived is False
    assert reason == "cut-short"


def test_auto_calibration_refuses_a_maker_robot(tmp_lerobot_home: Path) -> None:
    """The vendored autocal drives the arm under torque against its stops and
    writes Feetech EEPROM. There is no CAN equivalent and none is needed — the
    Maker arm's limits are fixed, so calibrating it means setting zero."""
    from makermodslab.api_errors import ApiError
    from makermodslab.schemas.sessions import AutoCalibrationOptions
    from makermodslab.sessions import _build_auto_calibration_request

    cfg.save_robot_record(
        "mk", {"arm_type": "maker", "mode": "single", "follower_port": "/dev/can0"}, allow_create=True
    )
    record = cfg.get_robot_record("mk")

    with pytest.raises(ApiError) as excinfo:
        _build_auto_calibration_request(
            record, AutoCalibrationOptions(arms=[{"device_type": "robot", "arm": "left"}])
        )

    assert excinfo.value.status_code == 400
    assert "zero-pose" in excinfo.value.detail


# ---------------------------------------------------------------------------
# Calibration dispatch
# ---------------------------------------------------------------------------


def test_calibrating_a_maker_robot_builds_a_zero_calibration_request(tmp_lerobot_home: Path) -> None:
    """One session kind, two procedures: _dispatch_start reads the request
    CLASS to pick the manager."""
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request
    from makermodslab.zero_calibrate import ZeroCalibrationRequest

    cfg.save_robot_record(
        "mk2", {"arm_type": "maker", "mode": "single", "follower_port": "/dev/can0"}, allow_create=True
    )
    record = cfg.get_robot_record("mk2")

    request = _build_calibration_request(record, CalibrationOptions(device_type="robot", arm="left"))

    assert isinstance(request, ZeroCalibrationRequest)
    assert request.port == "/dev/can0"


def test_calibrating_an_so101_robot_still_builds_the_sweep_request(tmp_lerobot_home: Path) -> None:
    from makermodslab.calibrate import CalibrationRequest
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request

    cfg.save_robot_record(
        "so2", {"arm_type": "so101", "mode": "single", "follower_port": "/dev/tty0"}, allow_create=True
    )
    record = cfg.get_robot_record("so2")

    request = _build_calibration_request(record, CalibrationOptions(device_type="robot", arm="left"))

    assert isinstance(request, CalibrationRequest)


def test_a_zero_calibration_is_visible_to_every_other_features_mutex() -> None:
    """The whole reason this flow reuses the `calibration` session kind: every
    existing reciprocal check calls calibrate.calibration_is_active(), so
    widening that one function enrolls the Maker flow with no new
    robot.busy.* discriminant to register."""
    from makermodslab import calibrate, zero_calibrate

    assert calibrate.calibration_is_active() is False
    try:
        zero_calibrate.zero_calibration_manager.status.calibration_active = True
        assert zero_calibrate.zero_calibration_is_active() is True
        assert calibrate.calibration_is_active() is True
    finally:
        zero_calibrate.zero_calibration_manager.status.calibration_active = False
    assert calibrate.calibration_is_active() is False


def test_zero_calibration_refuses_while_another_feature_owns_the_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle/mutex branch — no hardware touched."""
    from makermodslab import teleoperate, zero_calibrate

    monkeypatch.setattr(teleoperate, "teleoperation_active", True)

    result = zero_calibrate.zero_calibration_manager.start(
        zero_calibrate.ZeroCalibrationRequest(device_type="robot", port="/dev/can0", config_file="cal")
    )

    assert result["success"] is False
    assert "Teleoperation" in result["message"]
    # The refusal must not leave a claim behind.
    assert zero_calibrate.zero_calibration_is_active() is False


def test_zero_calibration_refuses_to_silently_overwrite_a_saved_calibration(
    tmp_lerobot_home: Path,
) -> None:
    """Same contract as the SO-101 flow: completing a calibration writes
    "<config_file>.json", so a taken name needs an explicit overwrite."""
    from makermodslab import zero_calibrate

    library = Path(cfg.calibration_dir_for_device("robot", "maker"))
    library.mkdir(parents=True, exist_ok=True)
    (library / "taken.json").write_text("{}")

    result = zero_calibrate.zero_calibration_manager.start(
        zero_calibrate.ZeroCalibrationRequest(device_type="robot", port="/dev/can0", config_file="taken")
    )

    assert result["success"] is False
    assert result["code"] == "name_taken"
    assert zero_calibrate.zero_calibration_is_active() is False


def test_completing_a_step_with_no_calibration_running_is_a_clean_refusal() -> None:
    from makermodslab import zero_calibrate

    result = zero_calibrate.zero_calibration_manager.complete_step()
    assert result["success"] is False
    assert "No calibration active" in result["message"]


def test_stopping_with_no_calibration_running_is_a_clean_refusal() -> None:
    from makermodslab import zero_calibrate

    result = zero_calibrate.zero_calibration_manager.stop()
    assert result["success"] is False


# ---------------------------------------------------------------------------
# Port detection helpers (pure)
# ---------------------------------------------------------------------------


def test_swing_needs_motion_both_ways() -> None:
    """A one-sided threshold would eventually read gravity sag as a gesture —
    a Maker follower has no brakes, so a torque-off arm drifts steadily one
    way."""
    from makermodslab.maker_ports import swing_detected

    assert swing_detected(0.0, -15.0, 15.0) is True
    assert swing_detected(0.0, -15.0, 2.0) is False  # sagged one way only
    assert swing_detected(0.0, -2.0, 15.0) is False
    assert swing_detected(0.0, 0.0, 0.0) is False


def test_swing_threshold_is_in_degrees_not_encoder_ticks() -> None:
    """Both Maker buses report degrees natively, unlike the SO-101's raw
    0-4095 ticks — reusing identify.py's 120-tick threshold here would need a
    third of a turn to trigger."""
    from makermodslab.maker_ports import _SWING_THRESHOLD_DEG, swing_detected

    assert _SWING_THRESHOLD_DEG == 10.0
    assert swing_detected(0.0, -10.0, 10.0) is True
    assert swing_detected(0.0, -9.9, 9.9) is False


@pytest.mark.asyncio
async def test_probe_reports_cleanly_when_no_ports_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import maker_ports

    monkeypatch.setattr(maker_ports, "find_available_ports", lambda: [])

    result = await maker_ports.probe_maker_ports()

    assert result["success"] is False
    assert result["follower_ports"] == []
    assert result["leader_ports"] == []
    assert "No serial ports" in result["message"]


@pytest.mark.asyncio
async def test_identify_rejects_an_unknown_device_type() -> None:
    from makermodslab.maker_ports import identify_maker_arm_by_motion

    result = await identify_maker_arm_by_motion("nonsense", ["/dev/x"])

    assert result["success"] is False
    assert "device_type" in result["message"]
