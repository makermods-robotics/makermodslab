"""The arm-family registry and its contract.

Pure, no I/O: the registry is a dict of stateless singletons, and the
contract is data plus two builders. The point of pinning it is that a family
cannot be half-declared — every seam the core resolves through the registry
must find an answer, and the answer must come from the registry rather than
from a literal somebody re-added to a flow.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import pytest

from makermodslab.arms import ArmFamily, registry
from makermodslab.arms.base import REQUIRED_ATTRIBUTES
from makermodslab.utils import config as cfg

# Concrete types each required attribute must carry. A typed check rather
# than isinstance-on-a-Protocol: the contract is data, and a wrong TYPE (a
# joint count as a string) would pass a name-only check and fail in rollout.
_ATTRIBUTE_TYPES: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "label": str,
    "short_label": str,
    "indefinite_label": str,
    "joints_per_arm": int,
    "supports_bimanual": bool,
    "uses_feetech_bus": bool,
    "supports_auto_calibration": bool,
    "uses_zero_calibration": bool,
    "supports_dagger": bool,
    "single_robot_type": str,
    "bimanual_robot_type": str,
    "robot_type_markers": tuple,
    "leader_library_attr": str,
    "follower_library_attr": str,
    "follower_probe_protocol": (str, type(None)),
    "motion_identify_energizes_follower": bool,
    "telemetry_kind": str,
}


def test_the_three_built_ins_register_in_order() -> None:
    assert registry.ids() == ("so101", "maker", "metal")
    assert registry.default().id == registry.DEFAULT_ID == "so101"


def test_the_typed_attribute_table_covers_the_whole_contract() -> None:
    """Guards the guard: a new required attribute must get a type here too."""
    assert set(_ATTRIBUTE_TYPES) == set(REQUIRED_ATTRIBUTES)


@pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.id)
def test_every_built_in_satisfies_the_contract(family: ArmFamily) -> None:
    assert isinstance(family, ArmFamily)
    for name, expected in _ATTRIBUTE_TYPES.items():
        value = getattr(family, name)
        assert isinstance(value, expected), f"{family.id}.{name} is {type(value).__name__}"
        if expected is int:
            assert not isinstance(value, bool)
    assert family.joints_per_arm > 0
    assert family.robot_type_markers, "a family with no markers can never be read off a dataset"
    assert all(m == m.lower() for m in family.robot_type_markers), "markers are matched lower-cased"
    assert family.single_robot_type != family.bimanual_robot_type
    # The library attrs must name real config constants — resolved at call time.
    assert isinstance(family.leader_calibration_dir(), str)
    assert isinstance(family.follower_calibration_dir(), str)


@pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.id)
def test_single_device_configs_carry_the_port_and_id(family: ArmFamily) -> None:
    """The calibration and recovery flows connect ONE arm through these; the
    follower's registered type must be the family's own (the string that
    reads the family back off a built config)."""
    follower = family.single_follower_config("/dev/f", "cal-f")
    leader = family.single_leader_config("/dev/l", "cal-l")
    assert (follower.port, follower.id) == ("/dev/f", "cal-f")
    assert (leader.port, leader.id) == ("/dev/l", "cal-l")
    # A config the family built reads back as the family (the CAN types by
    # their registered names; the SO-101 by the default-family fallback, since
    # lerobot registers its config under the SO-100 name).
    assert registry.family_for_robot_config_type(follower.type) is family
    assert not getattr(follower, "cameras", None), "single-device configs never open a camera"


@pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.id)
def test_zero_pose_text_exists_exactly_for_zero_calibrated_families(family: ArmFamily) -> None:
    follower_text = family.zero_pose_instructions("robot")
    leader_text = family.zero_pose_instructions("teleop")
    if family.uses_zero_calibration:
        assert "ZERO POSE" in follower_text and "ZERO POSE" in leader_text
        assert "leader" in leader_text and "leader" not in follower_text
    else:
        assert follower_text == "" and leader_text == ""


def test_the_can_followers_zero_poses_are_opposites_on_the_gripper() -> None:
    maker, metal = registry.get("maker"), registry.get("metal")
    assert "gripper fully open" in maker.zero_pose_instructions("robot")
    assert "gripper closed" in metal.zero_pose_instructions("robot")
    assert maker.zero_pose_instructions("teleop") == metal.zero_pose_instructions("teleop")


@pytest.mark.asyncio
async def test_probe_ports_exists_exactly_for_families_with_a_probe_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CAN family's probe is maker_ports' probe spoken in ITS protocol; the
    SO-101 (one Feetech bus for both halves) answers a refusal that names the
    gesture as the alternative, in the same response shape."""
    from makermodslab import maker_ports

    seen: list[tuple[list[str] | None, str]] = []

    async def fake_probe(ports, arm_type="maker"):
        seen.append((ports, arm_type))
        return {"success": True, "follower_ports": ["/dev/f"], "leader_ports": [], "unknown_ports": []}

    monkeypatch.setattr(maker_ports, "probe_maker_ports", fake_probe)
    for family in registry.families():
        result = await family.probe_ports(["/dev/x"])
        assert set(result) >= {"success", "follower_ports", "leader_ports", "unknown_ports"}
        if family.follower_probe_protocol is None:
            assert result["success"] is False and "motion" in result["message"]
            assert result["unknown_ports"] == ["/dev/x"]
        else:
            assert result["success"] is True
    assert seen == [(["/dev/x"], "maker"), (["/dev/x"], "metal")]


@pytest.mark.asyncio
async def test_identify_by_motion_routes_to_the_family_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import identify, maker_ports

    calls: list[tuple] = []

    async def fake_so(ports=None):
        calls.append(("so", ports))
        return {"success": True}

    async def fake_can(device_type, ports=None, arm_type="maker"):
        calls.append(("can", device_type, ports, arm_type))
        return {"success": True}

    monkeypatch.setattr(identify, "identify_arm_by_motion", fake_so)
    monkeypatch.setattr(maker_ports, "identify_maker_arm_by_motion", fake_can)
    for family in registry.families():
        await family.identify_by_motion("teleop", ["/dev/y"])
    assert calls == [
        ("so", ["/dev/y"]),
        ("can", "teleop", ["/dev/y"], "maker"),
        ("can", "teleop", ["/dev/y"], "metal"),
    ]


def test_telemetry_kind_is_one_of_the_two_the_frontend_renders() -> None:
    """`joints` (URDF fractions) drives the 3D viewer; `joints_deg` feeds the
    numeric readout in its slot. A family with a URDF says "urdf"."""
    assert all(f.telemetry_kind in ("urdf", "degrees") for f in registry.families())
    assert [f.telemetry_kind for f in registry.families()] == ["urdf", "degrees", "degrees"]


def test_only_the_damiao_follower_refuses_the_motion_gesture() -> None:
    """The fact maker_ports.identify_maker_arm_by_motion refuses on: the
    Damiao handshake energizes the follower, RobStride and Feetech do not."""
    assert [f.motion_identify_energizes_follower for f in registry.families()] == [False, False, True]
    assert [f.follower_probe_protocol for f in registry.families()] == [None, "robstride", "damiao"]


@pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.id)
def test_preflight_is_feetech_register_work_and_nothing_elsewhere(
    family: ArmFamily, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab import arm_identity, motor_power

    calls: list[tuple] = []
    monkeypatch.setattr(
        arm_identity,
        "verify_devices",
        lambda pairs, skip=False, config_names=None: calls.append(("v", skip)) or ["w"],
    )
    monkeypatch.setattr(
        motor_power, "reset_torque_limit", lambda d, side, label=None: calls.append(("r", side)) or []
    )
    monkeypatch.setattr(
        motor_power, "clear_goal_velocity", lambda d, side, label=None: calls.append(("c", side)) or []
    )

    identity = family.verify_identity((("dev", "follower"),), skip=True)
    registers = family.prepare_follower_registers("dev", "follower arms")
    if family.uses_feetech_bus:
        assert identity == ["w"] and registers == []
        assert calls == [("v", True), ("r", "follower"), ("c", "follower")]
    else:
        assert identity == [] and registers == [] and calls == []


class _Bus:
    def __init__(self, pose: dict) -> None:
        self.pose = pose
        self.port = "/dev/fake"

    def sync_read(self, register: str, normalize: bool = False) -> dict:
        assert register == "Present_Position"
        return dict(self.pose)


class _SoRobot:
    def __init__(self, pose: dict) -> None:
        self.bus = _Bus(pose)


class _CanArm:
    def __init__(self, observation: dict) -> None:
        self._observation = observation

    def get_observation(self) -> dict:
        return dict(self._observation)


def test_so101_stop_path_is_the_feetech_machinery(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import rest_pose, torque

    so = registry.get("so101")
    robot = _SoRobot({"shoulder_pan": 10, "gripper": 5})
    assert so.capture_rest_poses(robot) == [(robot.bus, {"shoulder_pan": 10})]
    assert so.capture_rest_poses(robot, include_gripper=True) == [
        (robot.bus, {"shoulder_pan": 10, "gripper": 5})
    ]

    calls: list[tuple] = []
    monkeypatch.setattr(
        rest_pose, "return_buses_to_rest", lambda poses, abort: calls.append(("return", poses, abort))
    )
    monkeypatch.setattr(
        torque,
        "force_disable_torque",
        lambda device, label: calls.append(("release", device, label)) or ["p"],
    )
    so.return_to_rest([(robot.bus, {})], "abort")
    assert so.release_torque(robot, "follower arm") == ["p"]
    assert calls == [("return", [(robot.bus, {})], "abort"), ("release", robot, "follower arm")]


@pytest.mark.parametrize("family_id", ["maker", "metal"])
def test_can_stop_path_is_the_mit_machinery(family_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import maker_rest_pose, torque

    can = registry.get(family_id)
    arm = _CanArm({"shoulder_pan.pos": 1.5, "gripper.pos": 0.5, "shoulder_pan.vel": 9})
    assert can.capture_rest_poses(arm) == [(arm, {"shoulder_pan": 1.5})]
    assert can.capture_rest_poses(arm, include_gripper=True) == [(arm, {"shoulder_pan": 1.5, "gripper": 0.5})]

    calls: list[tuple] = []
    monkeypatch.setattr(
        maker_rest_pose,
        "return_maker_arms_to_rest",
        lambda poses, abort: calls.append(("return", poses, abort)),
    )
    monkeypatch.setattr(
        torque, "release_maker_torque", lambda device, label: calls.append(("release", device, label)) or []
    )
    can.return_to_rest([(arm, {})], "abort")
    assert can.release_torque(arm, "CAN follower arm") == []
    assert calls == [("return", [(arm, {})], "abort"), ("release", arm, "CAN follower arm")]


def test_library_dirs_are_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test fixtures redirect calibration libraries by monkeypatching the
    config constants; a family that captured the path at import would silently
    write into the developer's real library."""
    monkeypatch.setattr(cfg, "METAL_FOLLOWER_CONFIG_PATH", "/redirected/metal")
    monkeypatch.setattr(cfg, "MAKER_LEADER_CONFIG_PATH", "/redirected/star")
    metal = registry.get("metal")
    assert metal.follower_calibration_dir() == "/redirected/metal"
    assert metal.leader_calibration_dir() == "/redirected/star"
    assert registry.get("maker").leader_calibration_dir() == "/redirected/star"


def test_robot_config_types_are_disjoint_across_families() -> None:
    """A lerobot type string maps to exactly one family, or reading the family
    back off a built config would depend on registration order."""
    seen: dict[str, str] = {}
    for family in registry.families():
        for robot_type in family.robot_config_types():
            assert robot_type not in seen, f"{robot_type} claimed by {seen[robot_type]} and {family.id}"
            seen[robot_type] = family.id


def test_family_for_robot_config_type_falls_back_to_the_default() -> None:
    assert registry.family_for_robot_config_type("bi_metal_follower").id == "metal"
    assert registry.family_for_robot_config_type("maker_follower").id == "maker"
    assert registry.family_for_robot_config_type("so101_follower").id == "so101"
    assert registry.family_for_robot_config_type("aloha").id == "so101"
    assert registry.family_for_robot_config_type(None).id == "so101"


def test_config_arm_types_mirror_the_registry() -> None:
    """utils.config's ArmType literal is still hand-written (TB5 opens it);
    until then it must agree with what is registered."""
    assert registry.ids() == cfg.ARM_TYPES
    assert set(get_args(cfg.ArmType)) == set(registry.ids())
    assert cfg.DEFAULT_ARM_TYPE == registry.DEFAULT_ID


def test_duplicate_ids_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_FAMILIES", dict(registry._FAMILIES))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("maker"))


def test_an_incomplete_family_is_refused_naming_the_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_FAMILIES", dict(registry._FAMILIES))

    class Half(ArmFamily):
        id = "half"
        label = "Half an arm"

        def single_follower_config(self, port, config_id):
            raise NotImplementedError

        def single_leader_config(self, port, config_id):
            raise NotImplementedError

        async def identify_by_motion(self, device_type, ports=None):
            raise NotImplementedError

        def capture_rest_poses(self, robot, *, include_gripper=False):
            raise NotImplementedError

        def return_to_rest(self, rest_poses, abort_event=None):
            raise NotImplementedError

        def release_torque(self, device, label="device"):
            raise NotImplementedError

        def build_single_configs(self, request, cameras, leader_id, follower_id):
            raise NotImplementedError

        def build_bimanual_configs(self, request, cameras, base, leader_staging, follower_staging):
            raise NotImplementedError

    with pytest.raises(TypeError, match="joints_per_arm") as excinfo:
        registry.register(Half())
    assert "indefinite_label" in str(excinfo.value)
    assert "half" not in registry.ids()


def test_a_family_without_builders_cannot_even_be_instantiated() -> None:
    class NoBuilders(ArmFamily):
        pass

    with pytest.raises(TypeError, match="build_single_configs"):
        NoBuilders()  # type: ignore[abstract]


def test_non_families_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_FAMILIES", dict(registry._FAMILIES))
    with pytest.raises(TypeError, match="ArmFamily"):
        registry.register(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Call-site sweep: no arm-type literal comparison outside makermodslab/arms.
#
# A tripwire in the genre of tests/test_motor_power_call_sites.py. The seams
# below resolve arm types through the registry; the thing this protects — that
# nobody re-adds `if arm_type == "maker"` to one of them, which is the branch
# the NEXT family forgets to extend — cannot be reached by a behaviour test.
# Pure AST over the files on disk.
#
# Every module outside the families themselves (and the vendored autocal,
# which is upstream's to re-shape). Computed rather than listed, so a new
# module is swept the day it lands.
# ---------------------------------------------------------------------------

_PACKAGE = Path(__file__).resolve().parents[1] / "makermodslab"
_SWEPT_FILES = tuple(
    sorted(
        str(p.relative_to(_PACKAGE))
        for p in _PACKAGE.rglob("*.py")
        if "vendor" not in p.parts and p.relative_to(_PACKAGE).parts[0] != "arms"
    )
)


def _is_arm_id(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in set(registry.ids())


def _literal_comparisons(path: Path) -> list[str]:
    """Every `x == "<arm id>"` / `x in ("<arm id>", ...)` comparison in the file."""
    offenders = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for comparator in node.comparators:
            elements = (
                comparator.elts if isinstance(comparator, ast.Tuple | ast.List | ast.Set) else [comparator]
            )
            if any(_is_arm_id(e) for e in elements):
                offenders.append(f"{path.name}:{node.lineno}")
                break
    return offenders


def test_the_sweep_reads_the_files_it_claims_to() -> None:
    """Guards the guard: a file list that no longer matches the tree passes
    every assertion below by matching nothing."""
    for name in _SWEPT_FILES:
        assert (_PACKAGE / name).is_file(), name
    assert len(_SWEPT_FILES) >= 30
    # The seams the refactor moved must be in the sweep, or it protects nothing.
    for name in (
        "arm_capabilities.py",
        "can_recovery.py",
        "maker_ports.py",
        "record.py",
        "replay.py",
        "rollout.py",
        "server.py",
        "teleoperate.py",
        "zero_calibrate.py",
        "utils/config.py",
        "utils/robot_factory.py",
    ):
        assert name in _SWEPT_FILES, name
    assert not any(name.startswith("arms/") for name in _SWEPT_FILES)


@pytest.mark.parametrize("name", _SWEPT_FILES)
def test_no_arm_type_literal_comparison_outside_the_families(name: str) -> None:
    offenders = _literal_comparisons(_PACKAGE / name)
    assert not offenders, (
        "arm-type literal comparisons must go through makermodslab.arms.registry "
        "(ask the family for a flag or a builder instead of branching on its id):\n  "
        + "\n  ".join(offenders)
    )


def test_the_sweep_would_catch_a_regression(tmp_path: Path) -> None:
    sample = tmp_path / "flow.py"
    sample.write_text('def f(t):\n    if t == "maker":\n        pass\n    return t in ("maker", "metal")\n')
    assert len(_literal_comparisons(sample)) == 2
