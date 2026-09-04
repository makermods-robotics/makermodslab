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


# ---------------------------------------------------------------------------
# Call-site sweep: no arm-type literal comparison outside makermodslab/arms.
#
# A tripwire in the genre of tests/test_motor_power_call_sites.py. The seams
# below resolve arm types through the registry; the thing this protects — that
# nobody re-adds `if arm_type == "maker"` to one of them, which is the branch
# the NEXT family forgets to extend — cannot be reached by a behaviour test.
# Pure AST over the files on disk.
#
# Scoped to the modules step 4a moved (docs/extensions/plan.md); 4b widens
# this list to the calibration, port-detection, stop-path and loop modules.
# ---------------------------------------------------------------------------

_PACKAGE = Path(__file__).resolve().parents[1] / "makermodslab"
_SWEPT_FILES = (
    "arm_capabilities.py",
    "rollout.py",
    "utils/config.py",
    "utils/robot_factory.py",
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
