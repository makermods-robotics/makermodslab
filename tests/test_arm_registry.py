"""The arm-family registry and its contract.

Pure, no I/O: the registry is a dict of stateless singletons, and the
contract is data plus two builders. The point of pinning it is that a family
cannot be half-declared — every seam the core resolves through the registry
must find an answer, and the answer must come from the registry rather than
from a literal somebody re-added to a flow.
"""

from __future__ import annotations

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
