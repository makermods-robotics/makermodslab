"""Metal arm integration: capabilities, record schema, config assembly, dispatch.

Follows the repo's testing policy — request schemas, pure helpers, and
idle/mutex branches. The parts that actually open a CAN or UART bus are
deliberately NOT exercised here; they are verified against real hardware.

The Metal arm reuses the Maker arm's integration seams (arm_type on the robot
record, arm_capabilities predicates, the zero-pose calibration flow, the CAN
rest-pose return), so most of these tests pin that "metal" flows through the
SAME paths — plus the handful of places the two CAN families genuinely differ:
the Damiao handshake energizes the motors, the leader calibration library is
SHARED with the Maker preset (hence the minted ``<name>_metal`` ids), and the
Metal config's CAN ids are (send, recv) tuples rather than plain ints.
"""

from pathlib import Path

import pytest

from makermodslab.utils import config as cfg

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_metal_is_a_known_arm_type() -> None:
    """"metal" must stop falling back to so101 — the fallback would silently
    re-enable every Feetech-register guard on a bus that has no registers."""
    assert cfg.normalize_arm_type("metal") == "metal"
    assert "metal" in cfg.ARM_TYPES


def test_metal_capabilities_match_its_hardware() -> None:
    """7-DOF Damiao CAN follower driven by the Star Arm 102 (encoders only):
    no Feetech registers, no range sweep (fixed joint_limits), no back-drivable
    leader for a DAgger handover."""
    from makermodslab.arm_capabilities import (
        joints_per_arm,
        supports_auto_calibration,
        supports_dagger,
        uses_feetech_bus,
        uses_zero_calibration,
    )

    assert joints_per_arm("metal") == 7
    assert uses_feetech_bus("metal") is False
    assert supports_auto_calibration("metal") is False
    assert uses_zero_calibration("metal") is True
    assert supports_dagger("metal") is False


def test_arm_type_read_back_off_a_built_metal_config() -> None:
    from lerobot.robots.bi_metal_follower import BiMetalFollowerConfig
    from lerobot.robots.metal_follower import MetalFollowerConfig, MetalFollowerConfigBase

    from makermodslab.arm_capabilities import arm_type_of_robot_config

    assert arm_type_of_robot_config(MetalFollowerConfig(port="/dev/can")) == "metal"
    assert (
        arm_type_of_robot_config(
            BiMetalFollowerConfig(
                left_arm_config=MetalFollowerConfigBase(port="/dev/a"),
                right_arm_config=MetalFollowerConfigBase(port="/dev/b"),
            )
        )
        == "metal"
    )


# ---------------------------------------------------------------------------
# Calibration libraries
# ---------------------------------------------------------------------------


def test_metal_resolves_its_own_follower_library_and_shares_the_leader_one() -> None:
    """The follower dir comes from the device class name (metal_follower), so
    it is naturally separate. The LEADER dir is teleoperators/rebot_102_leader
    for BOTH CAN families — all three Star-leader presets are config-only
    variants of one RebotArm102Leader class, and lerobot derives the dir from
    the class. That sharing is why leader calibration ids must be minted per
    arm type (see the default-name tests below)."""
    assert cfg.calibration_dir_for_device("robot", "metal").endswith("metal_follower")
    assert cfg.calibration_dir_for_device("teleop", "metal").endswith("rebot_102_leader")
    assert cfg.calibration_dir_for_device("teleop", "metal") == cfg.calibration_dir_for_device(
        "teleop", "maker"
    )
    assert cfg.calibration_dir_for_device("nonsense", "metal") is None


def test_a_metal_record_round_trips_and_switching_blanks_stale_slots(tmp_lerobot_home: Path) -> None:
    """Same record semantics as the Maker arm: arm_type persists, and switching
    a record to metal blanks hardware-bound fields not set by the same payload
    (the old names point into another family's library)."""
    cfg.save_robot_record(
        "mt", {"arm_type": "metal", "mode": "single", "follower_port": "/dev/canA"}, allow_create=True
    )
    record = cfg.get_robot_record("mt")
    assert record["arm_type"] == "metal"
    assert record["follower_port"] == "/dev/canA"

    cfg.save_robot_record("mt", {"arm_type": "maker"}, allow_create=False)
    switched = cfg.get_robot_record("mt")
    assert switched["arm_type"] == "maker"
    assert switched["follower_port"] == ""


# ---------------------------------------------------------------------------
# Minted default calibration ids
# ---------------------------------------------------------------------------


def test_default_config_names_are_minted_per_arm_type() -> None:
    """The Star-leader calibration library is shared between the Maker and
    Metal presets, and the presets disagree about what the zero POSE is
    (Maker: folded, gripper open; Metal: upright, gripper closed). An
    unsuffixed name lets a robot of one family silently reuse the other's
    zero. Minting ``<name>_<arm_type>`` into the default makes the collision
    impossible without renaming anything the user already saved."""
    assert cfg.default_slot_config_name("bot", "single", "left", "metal") == "bot_metal"
    assert cfg.default_slot_config_name("bot", "single", "left", "maker") == "bot_maker"
    assert cfg.default_slot_config_name("bot", "bimanual", "left", "metal") == "bot_metal_left"
    assert cfg.default_slot_config_name("bot", "bimanual", "right", "maker") == "bot_maker_right"


def test_so101_default_config_names_are_unchanged() -> None:
    """Every existing SO-101 robot keeps its historical defaults — the two
    SO-101 libraries are per-family already, so there is nothing to collide
    with and nothing to migrate."""
    assert cfg.default_slot_config_name("bot", "single", "left", "so101") == "bot"
    assert cfg.default_slot_config_name("bot", "bimanual", "right", "so101") == "bot_right"
    # The fallback default is so101 (same contract as normalize_arm_type).
    assert cfg.default_slot_config_name("bot", "single", "left", None) == "bot"


def test_calibrating_a_metal_robot_defaults_to_a_minted_config_name(tmp_lerobot_home: Path) -> None:
    """The sessions surface resolves an empty slot to the minted default."""
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request

    cfg.save_robot_record(
        "mt2", {"arm_type": "metal", "mode": "single", "leader_port": "/dev/star"}, allow_create=True
    )
    record = cfg.get_robot_record("mt2")

    request = _build_calibration_request(record, CalibrationOptions(device_type="teleop", arm="left"))
    assert request.config_file == "mt2_metal"


def test_an_explicitly_saved_config_name_beats_the_minted_default(tmp_lerobot_home: Path) -> None:
    """Minting only fills EMPTY slots: a record whose slot already names a
    calibration (hackathon-era Maker robots included) keeps using it."""
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request

    cfg.save_robot_record(
        "mk3",
        {"arm_type": "maker", "mode": "single", "leader_port": "/dev/star", "leader_config": "legacy"},
        allow_create=True,
    )
    record = cfg.get_robot_record("mk3")

    request = _build_calibration_request(record, CalibrationOptions(device_type="teleop", arm="left"))
    assert request.config_file == "legacy"


# ---------------------------------------------------------------------------
# Calibration dispatch
# ---------------------------------------------------------------------------


def test_calibrating_a_metal_robot_builds_a_zero_calibration_request(tmp_lerobot_home: Path) -> None:
    """Metal rides the zero-pose flow, and the request must CARRY its arm type
    — zero_calibrate builds the device configs and resolves the name-collision
    directory from it, and a request that defaulted to maker would connect a
    RobStride config to a Damiao bus."""
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request
    from makermodslab.zero_calibrate import ZeroCalibrationRequest

    cfg.save_robot_record(
        "mt4", {"arm_type": "metal", "mode": "single", "follower_port": "/dev/can0"}, allow_create=True
    )
    record = cfg.get_robot_record("mt4")

    request = _build_calibration_request(record, CalibrationOptions(device_type="robot", arm="left"))
    assert isinstance(request, ZeroCalibrationRequest)
    assert request.arm_type == "metal"
    assert request.port == "/dev/can0"


def test_a_maker_zero_calibration_request_still_says_maker(tmp_lerobot_home: Path) -> None:
    from makermodslab.schemas.sessions import CalibrationOptions
    from makermodslab.sessions import _build_calibration_request

    cfg.save_robot_record(
        "mk5", {"arm_type": "maker", "mode": "single", "follower_port": "/dev/can1"}, allow_create=True
    )
    record = cfg.get_robot_record("mk5")

    request = _build_calibration_request(record, CalibrationOptions(device_type="robot", arm="left"))
    assert request.arm_type == "maker"


def test_auto_calibration_refuses_a_metal_robot(tmp_lerobot_home: Path) -> None:
    from makermodslab.api_errors import ApiError
    from makermodslab.schemas.sessions import AutoCalibrationOptions
    from makermodslab.sessions import _build_auto_calibration_request

    cfg.save_robot_record(
        "mt6", {"arm_type": "metal", "mode": "single", "follower_port": "/dev/can0"}, allow_create=True
    )
    record = cfg.get_robot_record("mt6")

    with pytest.raises(ApiError) as excinfo:
        _build_auto_calibration_request(
            record, AutoCalibrationOptions(arms=[{"device_type": "robot", "arm": "left"}])
        )
    assert excinfo.value.status_code == 400
    assert "zero-pose" in excinfo.value.detail


def test_zero_calibration_builds_metal_ranges_with_the_send_can_id() -> None:
    """The two CAN families disagree about the id field's shape: Maker
    motor_can_ids are plain ints, Metal's are (send_id, recv_id) tuples.
    MotorCalibration.id is an int, and lerobot's own MetalFollower.calibrate()
    stores the SEND id — storing the tuple would produce an unloadable file."""
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig

    from makermodslab.zero_calibrate import ZeroCalibrationRequest, zero_calibration_manager

    manager = zero_calibration_manager
    config = MetalFollowerConfig(port="/dev/x", id="unit")
    device = MetalFollower(config)
    old_device, old_request = manager.device, manager._current_request
    try:
        manager.device = device
        manager._current_request = ZeroCalibrationRequest(
            device_type="robot", port="/dev/x", config_file="unit", arm_type="metal"
        )
        calibration = manager._build_calibration()
    finally:
        manager.device, manager._current_request = old_device, old_request

    for motor, (send_id, _recv_id) in config.motor_can_ids.items():
        assert calibration[motor].id == send_id
        low, high = config.joint_limits[motor]
        assert calibration[motor].range_min == int(low)
        assert calibration[motor].range_max == int(high)


def test_zero_pose_instructions_differ_per_arm_type() -> None:
    """The user is being asked to do something physical, and the two poses are
    OPPOSITES on the gripper (Maker: fully open; Metal: closed). Showing the
    Maker text to a Metal user zeroes the gripper at the wrong end of travel."""
    from makermodslab.zero_calibrate import zero_pose_instructions

    maker_text = zero_pose_instructions("maker")
    metal_text = zero_pose_instructions("metal")
    assert maker_text != metal_text
    assert "open" in maker_text
    assert "upright" in metal_text and "closed" in metal_text


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


class _Req:
    """Minimal stand-in for a start request — the factory only reads attributes."""

    def __init__(self, **kw):
        self.mode = "single"
        self.arm_type = "metal"
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


def test_single_metal_request_builds_metal_configs(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_single_configs

    robot, teleop = build_single_configs(_Req())

    assert robot.type == "metal_follower"
    assert robot.port == "/dev/follower"
    # The _metal preset, NOT the bare rebot_102_leader and NOT the _maker one:
    # each preset carries the joint mapping of ITS follower, and a wrong one
    # runs joints backwards or saturates them while the loop reports healthy.
    assert teleop.type == "rebot_102_leader_metal"
    assert teleop.port == "/dev/leader"


def test_the_metal_leader_preset_carries_the_metal_joint_mapping(_no_staging) -> None:
    """joint_ranges is MetalFollowerConfig.joint_limits verbatim — except the
    gripper, deliberately capped at [0, 115] below the follower's 137.5 limit
    (the leader's own travel), which is why this asserts the exception too
    rather than blanket-equality that would mask a preset regression."""
    from lerobot.robots.metal_follower import MetalFollowerConfig

    from makermodslab.utils.robot_factory import build_single_configs

    _, teleop = build_single_configs(_Req())
    limits = MetalFollowerConfig(port="/x").joint_limits

    for joint, (low, high) in limits.items():
        if joint == "gripper":
            continue
        assert teleop.joint_ranges[joint] == [int(low), int(high)]
    assert teleop.joint_ranges["gripper"] == [0, 115]


def test_bimanual_metal_request_builds_bimetal_configs(_no_staging) -> None:
    """There is no registered bi_rebot_102_leader_metal type — bimanual Metal
    is the GENERIC BiRebot102LeaderConfig with metal-preset sub-configs (the
    fork's documented shape). The sub-config class is the assertion that the
    metal mapping actually made it in."""
    from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderMetalConfig

    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, teleop = build_bimanual_configs(_Req(mode="bimanual"))

    assert robot.type == "bi_metal_follower"
    assert teleop.type == "bi_rebot_102_leader"
    assert isinstance(teleop.left_arm_config, RebotArm102LeaderMetalConfig)
    assert isinstance(teleop.right_arm_config, RebotArm102LeaderMetalConfig)
    assert robot.left_arm_config.port == "/dev/follower"
    assert robot.right_arm_config.port == "/dev/rfollower"
    assert str(robot.calibration_dir) == "/staging/follower"
    assert str(teleop.calibration_dir) == "/staging/leader"


def test_bimanual_metal_cameras_go_on_the_left_arm_not_the_top_level(_no_staging) -> None:
    from makermodslab.utils.robot_factory import build_bimanual_configs

    robot, _ = build_bimanual_configs(_Req(mode="bimanual"), cameras={"scene": object()})

    assert set(robot.left_arm_config.cameras) == {"scene"}
    assert robot.cameras == {}
    assert robot.right_arm_config.cameras == {}


def test_metal_zero_calibration_device_configs() -> None:
    """The single-device helpers zero_calibrate connects through."""
    from makermodslab.utils.robot_factory import metal_follower_config, metal_leader_config

    follower = metal_follower_config("/dev/can0", "cal")
    assert follower.type == "metal_follower"
    assert follower.port == "/dev/can0" and follower.id == "cal"
    assert follower.cameras == {}

    leader = metal_leader_config("/dev/star0", "cal")
    assert leader.type == "rebot_102_leader_metal"
    assert leader.port == "/dev/star0" and leader.id == "cal"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm_type", "mode", "expected"),
    [
        ("metal", "single", "metal_follower"),
        ("metal", "bimanual", "bi_metal_follower"),
    ],
)
def test_rollout_names_the_metal_robot_type(arm_type: str, mode: str, expected: str) -> None:
    from makermodslab.rollout import InferenceRequest, _robot_cli_type

    request = InferenceRequest(
        follower_port="/dev/f", follower_config="F", policy_ref="ref", mode=mode, arm_type=arm_type
    )
    assert _robot_cli_type(request) == expected


def test_arm_count_guard_measures_a_metal_checkpoint_at_seven_dims() -> None:
    """Same 7-dim contract as the Maker arm — pinned separately so a future
    per-family divergence has to come past a test."""
    from makermodslab.rollout import _ARM_STATE_DIMS

    assert _ARM_STATE_DIMS["metal"] == 7
