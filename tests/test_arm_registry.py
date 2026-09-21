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

import pytest

from makermodslab.arms import ArmFamily, registry
from makermodslab.arms.base import REQUIRED_ATTRIBUTES
from makermodslab.utils import config as cfg
from tests.mocks import make_arm_family, scratch_registry

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
    "calibration_kind": str,
    "supports_dagger": bool,
    "single_robot_type": str,
    "bimanual_robot_type": str,
    "robot_type_markers": tuple,
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
    # The dir METHODS are the contract (the built-ins answer them through
    # their library-attr constants, resolved at call time).
    assert isinstance(family.leader_calibration_dir(), str) and family.leader_calibration_dir()
    assert isinstance(family.follower_calibration_dir(), str) and family.follower_calibration_dir()
    from makermodslab.arms.base import CALIBRATION_KINDS

    assert family.calibration_kind in CALIBRATION_KINDS


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
def test_calibration_summary_exists_exactly_for_steps_families(family: ArmFamily) -> None:
    """What the config dialog shows BEFORE Start, per device side: the CAN
    families answer their zero-pose text (no image — the frontend keeps its
    bundled photos); a range-sweep family has nothing to summarize and
    answers None, not an empty dict."""
    follower = family.calibration_summary("robot")
    leader = family.calibration_summary("teleop")
    if family.calibration_kind == "steps":
        assert set(follower) == {"text", "image_url"} == set(leader)
        assert follower["image_url"] is None and leader["image_url"] is None
        assert "ZERO POSE" in follower["text"] and "ZERO POSE" in leader["text"]
        assert "leader" in leader["text"] and "leader" not in follower["text"]
    else:
        assert follower is None and leader is None


def test_the_built_in_calibration_kinds_and_panel_urls() -> None:
    """so101 sweeps; the CAN pair are step wizards; nobody built in ships a
    panel or a served image (TB6b fills those for extensions)."""
    from makermodslab.arms.base import CALIBRATION_KINDS

    assert CALIBRATION_KINDS == ("range_sweep", "steps", "panel")
    assert [f.calibration_kind for f in registry.families()] == ["range_sweep", "steps", "steps"]
    assert all(f.calibration_panel_url is None for f in registry.families())
    assert all(f.image_url is None for f in registry.families())
    assert ArmFamily.calibration_panel_url is None and ArmFamily.image_url is None


@pytest.mark.parametrize("family", registry.families(), ids=lambda f: f.id)
def test_default_calibration_name_is_the_record_name_plus_the_family_suffix(family: ArmFamily) -> None:
    """The suffix is published in the manifest so the UI can predict the
    default calibration id a record's empty slot will get (the CAN families
    mint their id in because they share the Star-leader library; the SO-101
    keeps its historical bare name). One property, used by the minting rule
    and served on the wire, so the two can never disagree."""
    expected = "" if family.id == registry.DEFAULT_ID else f"_{family.id}"
    assert family.calibration_name_suffix == expected
    assert family.default_calibration_name("bot") == "bot" + family.calibration_name_suffix


def test_the_can_followers_zero_poses_use_closed_grippers() -> None:
    maker, metal = registry.get("maker"), registry.get("metal")
    assert "gripper fully closed" in maker.calibration_summary("robot")["text"]
    assert "gripper fully closed" in metal.calibration_summary("robot")["text"]
    assert maker.calibration_summary("teleop") == metal.calibration_summary("teleop")


@pytest.mark.asyncio
async def test_probe_ports_exists_exactly_for_families_with_a_probe_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CAN family's probe is maker_ports' probe spoken in ITS protocol; the
    SO-101 (one Feetech bus for both halves) answers a refusal that names the
    gesture as the alternative, in the same response shape."""
    from makermodslab import maker_ports

    seen: list[tuple[list[str] | None, str]] = []

    async def fake_probe(ports, arm_type="maker", leader_kind=None):
        seen.append((ports, arm_type, leader_kind))
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
    # No leader kind reaches maker_ports from a call that named none: the
    # family's default is what the probe then assumes.
    assert seen == [(["/dev/x"], "maker", None), (["/dev/x"], "metal", None)]


@pytest.mark.asyncio
async def test_identify_by_motion_routes_to_the_family_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab import identify, maker_ports

    calls: list[tuple] = []

    async def fake_so(ports=None):
        calls.append(("so", ports))
        return {"success": True}

    async def fake_can(device_type, ports=None, arm_type="maker", leader_kind=None):
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
    """`joints` (URDF joint values) drives the 3D viewer; `joints_deg` feeds the
    numeric readout in its slot. A family with a URDF says "urdf" — the SO-101
    Maker, and Metal arms each ship one."""
    assert all(f.telemetry_kind in ("urdf", "degrees") for f in registry.families())
    assert [f.telemetry_kind for f in registry.families()] == ["urdf", "urdf", "urdf"]


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


def test_config_reads_the_registry_live_instead_of_freezing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """TB5 opened the set: the hand-written Literal and the tuple captured at
    import are gone (both were stale the moment an extension registered a
    family), and the one predicate left asks the registry on every call."""
    assert not hasattr(cfg, "ArmType")
    assert not hasattr(cfg, "ARM_TYPES")
    assert cfg.DEFAULT_ARM_TYPE == registry.DEFAULT_ID
    assert cfg.is_known_arm_type("nine") is False
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine"))
    assert cfg.is_known_arm_type("nine") is True


def test_get_raises_unknown_arm_type_naming_the_id_and_the_registered_ones() -> None:
    """A KeyError subclass (so existing `except KeyError` sites keep working)
    whose message says what was asked for and what exists — the text a log
    line shows when a gate was bypassed."""
    with pytest.raises(registry.UnknownArmType) as excinfo:
        registry.get("nope")
    assert isinstance(excinfo.value, KeyError)
    message = str(excinfo.value)
    assert "nope" in message
    for known in registry.ids():
        assert known in message


def test_provenance_defaults_to_builtin_and_records_an_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest tells the user WHO provides a family; the built-ins say
    "builtin" and an extension's registration (TB6) passes its own name."""
    for family_id in registry.ids():
        assert registry.provided_by(family_id) == "builtin"
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine"), provided_by="ext")
    assert registry.provided_by("nine") == "ext"
    assert registry.get("nine").joints_per_arm == 9


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


def test_library_dirs_are_a_method_contract_not_attributes() -> None:
    """TB6a: an extension family answers ``leader_calibration_dir()`` /
    ``follower_calibration_dir()`` directly (typically from
    utils.config.lerobot_calibration_dir); the ``*_library_attr`` names are
    the built-ins' own way of resolving a monkeypatchable constant and no
    longer part of the required contract."""
    assert "leader_library_attr" not in REQUIRED_ATTRIBUTES
    assert "follower_library_attr" not in REQUIRED_ATTRIBUTES
    assert "uses_zero_calibration" not in REQUIRED_ATTRIBUTES
    assert "calibration_kind" in REQUIRED_ATTRIBUTES


def test_the_base_dir_methods_raise_not_implemented_without_a_library_attr() -> None:
    family = make_arm_family("bare", dirs=False)
    with pytest.raises(NotImplementedError, match="leader_calibration_dir"):
        family.leader_calibration_dir()
    with pytest.raises(NotImplementedError, match="follower_calibration_dir"):
        family.follower_calibration_dir()


def test_a_family_whose_dir_method_raises_is_refused_naming_the_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The library dirs are read by every readiness check and every staging
    copy; a family that cannot answer would fail deep in the first flow that
    asks. Refuse at registration, naming the method that failed."""
    scratch_registry(monkeypatch)
    with pytest.raises((TypeError, ValueError)) as excinfo:
        registry.register(make_arm_family("bare", dirs=False))
    assert "bare" in str(excinfo.value)
    assert "calibration_dir" in str(excinfo.value)
    assert "bare" not in registry.ids()


def test_an_empty_dir_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch_registry(monkeypatch)
    with pytest.raises((TypeError, ValueError), match="follower_calibration_dir"):
        registry.register(make_arm_family("hollow", follower_dir=""))
    assert "hollow" not in registry.ids()


def test_a_follower_dir_collision_with_a_registered_family_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two families writing calibrations into one follower library would
    load each other's files by name; the follower dir is unique per family,
    full stop. Pinned against a built-in's dir AND between two fakes."""
    scratch_registry(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        registry.register(make_arm_family("twin", follower_dir=cfg.MAKER_FOLLOWER_CONFIG_PATH))
    assert "twin" in str(excinfo.value) and "maker" in str(excinfo.value)
    assert "twin" not in registry.ids()

    registry.register(make_arm_family("alpha", follower_dir="/lib/robots/alpha_follower"))
    with pytest.raises(ValueError, match="alpha"):
        registry.register(make_arm_family("beta", follower_dir="/lib/robots/alpha_follower"))
    assert "beta" not in registry.ids()


def test_a_shared_leader_dir_needs_a_calibration_name_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Maker/Metal case, mirrored with fakes: two families whose leaders
    are the same device class share ONE leader library (lerobot derives the
    dir from the class name), and that is legitimate only because
    ``calibration_name_suffix`` mints each family's id into its default
    names so the two never reuse a calibration written for the other. A
    family with an empty suffix sharing a leader dir is refused, and the
    message says so."""
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("alpha", leader_dir="/lib/teleop/star_leader"))

    with pytest.raises(ValueError) as excinfo:
        registry.register(
            make_arm_family("beta", leader_dir="/lib/teleop/star_leader", calibration_name_suffix="")
        )
    message = str(excinfo.value)
    assert "beta" in message and "alpha" in message
    assert "suffix" in message
    assert "beta" not in registry.ids()

    # The same sharing WITH a suffix (the default mints "_beta") is accepted.
    registry.register(make_arm_family("beta", leader_dir="/lib/teleop/star_leader"))
    assert "beta" in registry.ids()
    assert registry.get("beta").leader_calibration_dir() == registry.get("alpha").leader_calibration_dir()


def test_the_built_ins_share_the_star_leader_dir_legitimately() -> None:
    """The real case the rule above encodes: Maker and Metal register, in
    order, with one leader dir and distinct follower dirs and suffixes."""
    maker, metal = registry.get("maker"), registry.get("metal")
    assert maker.leader_calibration_dir() == metal.leader_calibration_dir()
    assert maker.follower_calibration_dir() != metal.follower_calibration_dir()
    assert maker.calibration_name_suffix and metal.calibration_name_suffix
    assert registry.ids() == ("so101", "maker", "metal")


def test_an_unknown_calibration_kind_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch_registry(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        registry.register(make_arm_family("odd", calibration_kind="zero_pose"))
    assert "odd" in str(excinfo.value) and "zero_pose" in str(excinfo.value)
    assert "odd" not in registry.ids()


def test_a_range_sweep_family_must_be_a_feetech_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep manager is Feetech register work (calibrate.py reads and
    writes servo registers by name); a family without that bus cannot ride
    it. A Feetech family may."""
    scratch_registry(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        registry.register(make_arm_family("sweep", calibration_kind="range_sweep", uses_feetech_bus=False))
    assert "sweep" in str(excinfo.value) and "range_sweep" in str(excinfo.value)
    assert "uses_feetech_bus" in str(excinfo.value)
    assert "sweep" not in registry.ids()

    registry.register(make_arm_family("sweep", calibration_kind="range_sweep", uses_feetech_bus=True))
    assert "sweep" in registry.ids()


def test_a_steps_family_must_override_calibrate_and_open_for_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step manager runs ``family.calibrate`` after ``family.open_for_calibration``;
    the base versions raise NotImplementedError, so a family that forgot one
    would fail mid-flow with a bus open. The registry compares the methods
    against the base's and names the missing one."""
    scratch_registry(monkeypatch)
    with pytest.raises(TypeError) as excinfo:
        registry.register(make_arm_family("halfstep", step_methods=False))
    message = str(excinfo.value)
    assert "halfstep" in message
    assert "calibrate" in message and "open_for_calibration" in message
    assert "halfstep" not in registry.ids()

    # One of the two is not enough either.
    only_open = make_arm_family("halfstep", step_methods=False)
    type(only_open).open_for_calibration = lambda self, device_type, port, config_id: object()
    with pytest.raises(TypeError, match="calibrate"):
        registry.register(only_open)
    assert "halfstep" not in registry.ids()


def test_the_base_calibrate_and_open_for_calibration_are_not_implemented() -> None:
    family = make_arm_family("nine", step_methods=False)
    with pytest.raises(NotImplementedError):
        ArmFamily.open_for_calibration(family, "robot", "/dev/x", "cal")
    with pytest.raises(NotImplementedError):
        ArmFamily.calibrate(family, object(), "robot", object())


def test_a_panel_family_must_name_its_panel_url(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch_registry(monkeypatch)
    for bad in (None, ""):
        with pytest.raises((TypeError, ValueError)) as excinfo:
            registry.register(make_arm_family("paneled", calibration_kind="panel", calibration_panel_url=bad))
        assert "paneled" in str(excinfo.value) and "calibration_panel_url" in str(excinfo.value)
        assert "paneled" not in registry.ids()

    registry.register(
        make_arm_family("paneled", calibration_kind="panel", calibration_panel_url="/api/v1/ext/p/static/cal")
    )
    assert registry.get("paneled").calibration_panel_url == "/api/v1/ext/p/static/cal"


def test_a_complete_extension_family_registers_beside_the_built_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive case every refusal above is measured against: the default
    fake (own dirs, steps kind with both overrides) is accepted and the
    built-ins are untouched and still first."""
    scratch_registry(monkeypatch)
    registry.register(make_arm_family("nine"), provided_by="ext")
    assert registry.ids() == ("so101", "maker", "metal", "nine")


def test_read_positions_default_prefers_the_raw_reader_then_the_bus() -> None:
    """The base reader is today's zero-flow heuristic, documented as a read on
    a torque-off bus: a device's private raw reader (the Maker follower and
    the Star leader) wins, the bus's Present_Position sync_read (the Metal
    follower) is the fallback, and a device with neither reads as empty."""

    class _Raw:
        def _read_raw_positions(self):
            return {"j1": 1, "j2": 2.5}

    class _Bus:
        def sync_read(self, register):
            assert register == "Present_Position"
            return {"j1": 3}

    class _WithBus:
        bus = _Bus()

    class _Bare:
        pass

    family = make_arm_family("nine")
    assert family.read_positions(_Raw()) == {"j1": 1.0, "j2": 2.5}
    assert family.read_positions(_WithBus()) == {"j1": 3.0}
    assert family.read_positions(_Bare()) == {}


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
        "step_calibrate.py",
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
