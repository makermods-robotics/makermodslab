# Copyright 2026 MakerMods. All rights reserved.
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
"""The arms manifest: every registered family described for a client.

``GET /api/v1/arms`` serves ``arms_manifest()``. It is the one document the
UI reads arm capabilities from — which calibration flow to offer, whether a
protocol probe exists or the gesture is needed, how many joints a checkpoint
must have, what to render in the 3D viewer's slot — so a family an extension
registers renders with no frontend change, and the built-ins render from the
same data they always did.

The dicts here are described EXACTLY by schemas.system.ArmFamilyInfo: a
``response_model`` silently filters undeclared keys and materializes absent
ones, so a key added here without the model (or the reverse) is a silent
wire change. tests/test_arms_manifest.py compares the two.

Order is registry order, default family FIRST. The frontend's dataset
``robot_type`` marker scan needs the default family checked LAST (its
markers are the loosest — ``so_leader`` matches almost anything), so a
client scans every other family's markers in manifest order and the default
family's after them, reproducing arm_capabilities.arm_type_from_robot_type.
"""

from __future__ import annotations

from . import registry
from .base import ArmFamily


def describe_family(family: ArmFamily) -> dict:
    """One manifest entry for ``family`` (the shape of ArmFamilyInfo)."""
    # What the config dialog shows BEFORE Start, per side; None (not {}) for
    # a family with nothing to summarize — the range-sweep SO-101 today.
    summary = (
        {
            "leader": family.calibration_summary("teleop"),
            "follower": family.calibration_summary("robot"),
        }
        if family.calibration_kind == "steps"
        else None
    )
    return {
        "id": family.id,
        "label": family.label,
        "short_label": family.short_label,
        "provided_by": registry.provided_by(family.id),
        "joints_per_arm": family.joints_per_arm,
        "supports_bimanual": family.supports_bimanual,
        "image_url": family.image_url,
        "calibration": {
            "kind": family.calibration_kind,
            "summary": summary,
            "panel_url": family.calibration_panel_url,
        },
        "telemetry_kind": family.telemetry_kind,
        "capabilities": {
            "uses_feetech_bus": family.uses_feetech_bus,
            "supports_auto_calibration": family.supports_auto_calibration,
            "supports_dagger": family.supports_dagger,
            "supports_port_probe": family.follower_probe_protocol is not None,
            "motion_identify_energizes_follower": family.motion_identify_energizes_follower,
        },
        "robot_types": [family.single_robot_type, family.bimanual_robot_type],
        "robot_type_markers": list(family.robot_type_markers),
        "calibration_name_suffix": family.calibration_name_suffix,
    }


def arms_manifest() -> list[dict]:
    """Every registered family, in registry order (default family first)."""
    return [describe_family(family) for family in registry.families()]
