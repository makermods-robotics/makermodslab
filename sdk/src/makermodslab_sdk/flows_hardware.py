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
#
"""Passive hardware context composed from read-only SDK operations.

``inspect_hardware`` composes ``system.arms``, ``system.available_ports``,
``system.available_cameras`` and ``robots.list``. It never probes a bus,
opens an arm, moves hardware or mutates a robot record. Active discovery is
returned as explicit, effect-labelled next calls for an agent or user to
choose deliberately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from makermodslab_sdk.resources.system import CameraInfo

if TYPE_CHECKING:
    from collections.abc import Callable

    from makermodslab_sdk.client import Client


PORT_SLOTS = ("leader_port", "follower_port", "right_leader_port", "right_follower_port")
Effect = Literal["read_only", "may_energize", "moves_hardware"]
Section = Literal["arms", "ports", "cameras", "robots"]


@dataclass(frozen=True)
class HardwareSectionError:
    """One unavailable input section; the other sections remain usable."""

    section: Section
    detail: str


@dataclass(frozen=True)
class PortAssignment:
    """One saved robot port slot and whether that port is currently visible."""

    robot: str
    slot: str
    port: str
    present: bool | None


@dataclass(frozen=True)
class CameraAssignment:
    """One saved camera binding and whether its device is currently visible."""

    name: str
    index: int | None
    unique_id: str | None
    present: bool | None


@dataclass(frozen=True)
class SavedRobotHardware:
    """The hardware-facing portion of one saved robot record, with the
    readiness the server computed for it.

    ``is_clean`` gates teleoperation and recording (leader AND follower);
    ``follower_ready`` gates inference, replay and hosting; ``leader_ready``
    gates remote teleoperation. ``arm_available`` is False when no installed
    family claims this record's ``arm_type``. Each is None when the server
    did not report it, so an older server degrades to "unknown" rather than
    to a confident False.
    """

    name: str
    arm_type: str
    mode: str | None
    ports: tuple[PortAssignment, ...]
    cameras: tuple[CameraAssignment, ...]
    is_clean: bool | None = None
    follower_ready: bool | None = None
    leader_ready: bool | None = None
    arm_available: bool | None = None

    def readiness_summary(self) -> str:
        """One phrase naming what this record can start right now."""
        if self.arm_available is False:
            return f"arm type {self.arm_type!r} not installed"
        if self.is_clean:
            return "ready"
        able = [
            label
            for label, ready in (
                ("inference/replay", self.follower_ready),
                ("remote teleop", self.leader_ready),
            )
            if ready
        ]
        if able:
            return "not ready for teleop/recording; can still run " + " and ".join(able)
        return "unknown readiness" if self.is_clean is None else "not ready"


@dataclass(frozen=True)
class ArmDiscoveryCapability:
    """The small manifest subset an agent needs to choose discovery."""

    arm_type: str
    label: str
    joints_per_arm: int | None
    calibration_kind: str | None
    supports_port_probe: bool
    motion_identify_energizes_follower: bool
    supports_gripper_wiggle: bool


@dataclass(frozen=True)
class HardwareObservation:
    """A notable relationship in the assembled context."""

    kind: str
    message: str
    port: str | None = None
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareNextAction:
    """A literal SDK call the caller may choose, with physical effects."""

    call: str
    reason: str
    effects: tuple[Effect, ...]


@dataclass(frozen=True)
class HardwareContext:
    """Passive snapshot of visible devices, records and discovery choices."""

    visible_ports: tuple[str, ...]
    visible_cameras: tuple[CameraInfo, ...]
    robots: tuple[SavedRobotHardware, ...]
    arm_families: tuple[ArmDiscoveryCapability, ...]
    assigned_ports_missing: tuple[PortAssignment, ...]
    unassigned_visible_ports: tuple[str, ...]
    observations: tuple[HardwareObservation, ...]
    next_actions: tuple[HardwareNextAction, ...]
    errors: tuple[HardwareSectionError, ...]

    def summary(self) -> str:
        """Compact agent-readable snapshot; structured fields hold detail."""
        lines = [
            (
                f"Hardware: {len(self.visible_ports)} visible ports; "
                f"cameras: {len(self.visible_cameras)} visible; "
                f"{len(self.robots)} saved robots; {len(self.arm_families)} arm families."
            )
        ]
        for robot in self.robots[:6]:
            slots = ", ".join(
                f"{item.slot}={item.port} "
                f"({'unknown' if item.present is None else 'present' if item.present else 'missing'})"
                for item in robot.ports
            )
            lines.append(
                f"- {robot.name} [{robot.arm_type}/{robot.mode or 'unspecified'}] "
                f"{robot.readiness_summary()}: {slots or 'no ports'}"
            )
        if len(self.robots) > 6:
            lines.append(f"- ... {len(self.robots) - 6} more saved robots")
        if self.unassigned_visible_ports:
            shown = self.unassigned_visible_ports[:8]
            suffix = (
                f" (+{len(self.unassigned_visible_ports) - 8} more ports)"
                if len(self.unassigned_visible_ports) > 8
                else ""
            )
            lines.append("Unassigned visible ports: " + ", ".join(shown) + suffix)
        if self.assigned_ports_missing:
            shown_missing = self.assigned_ports_missing[:8]
            lines.append(
                "Assigned ports missing: "
                + ", ".join(f"{item.robot}.{item.slot}={item.port}" for item in shown_missing)
            )
            if len(self.assigned_ports_missing) > 8:
                lines[-1] += f" (+{len(self.assigned_ports_missing) - 8} more assignments)"
        if self.observations:
            lines.append("Observations: " + " ".join(item.message for item in self.observations[:4]))
            if len(self.observations) > 4:
                lines[-1] += f" (+{len(self.observations) - 4} more observations)"
        if self.next_actions:
            lines.append("Next actions (choose explicitly):")
            lines.extend(f"- {item.call} [{', '.join(item.effects)}]" for item in self.next_actions[:5])
            if len(self.next_actions) > 5:
                lines.append(f"- ... {len(self.next_actions) - 5} more next actions")
        if self.errors:
            lines.append(
                "Partial errors: " + "; ".join(f"{item.section}: {item.detail}" for item in self.errors)
            )
        return "\n".join(lines)


def _flag(record: Any, key: str) -> bool | None:
    """A server-computed readiness flag, or None when it wasn't reported."""
    value = record.get(key)
    return value if isinstance(value, bool) else None


def _detail(exc: Exception) -> str:
    value = getattr(exc, "detail", None)
    return str(value if value else exc)


def _read_section(
    section: Section,
    read: Callable[[], Any],
    errors: list[HardwareSectionError],
) -> Any | None:
    try:
        return read()
    except Exception as exc:  # one malformed/unavailable section must not hide the others
        errors.append(HardwareSectionError(section, _detail(exc)))
        return None


def _camera_assignment(
    entry: Any, visible: tuple[CameraInfo, ...], *, cameras_known: bool
) -> CameraAssignment:
    data = entry if isinstance(entry, dict) else {}
    name = str(data.get("name") or data.get("id") or "unnamed")
    raw_index = data.get("camera_index", data.get("index"))
    index = raw_index if isinstance(raw_index, int) else None
    raw_unique_id = data.get("unique_id")
    unique_id = str(raw_unique_id) if raw_unique_id else None
    present = (
        any(
            (unique_id is not None and camera.unique_id == unique_id)
            or (unique_id is None and index is not None and camera.index == index)
            for camera in visible
        )
        if cameras_known
        else None
    )
    return CameraAssignment(name=name, index=index, unique_id=unique_id, present=present)


def _manifest_entry(entry: dict[str, Any]) -> ArmDiscoveryCapability:
    capabilities = entry.get("capabilities")
    caps = capabilities if isinstance(capabilities, dict) else {}
    calibration = entry.get("calibration")
    calibration_kind = calibration.get("kind") if isinstance(calibration, dict) else None
    joints = entry.get("joints_per_arm")
    return ArmDiscoveryCapability(
        arm_type=str(entry.get("id") or "unknown"),
        label=str(entry.get("label") or entry.get("id") or "Unknown"),
        joints_per_arm=joints if isinstance(joints, int) else None,
        calibration_kind=str(calibration_kind) if calibration_kind else None,
        supports_port_probe=bool(caps.get("supports_port_probe")),
        motion_identify_energizes_follower=bool(caps.get("motion_identify_energizes_follower")),
        supports_gripper_wiggle=bool(caps.get("supports_gripper_wiggle")),
    )


def _literal_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(repr(value) for value in values) + "]"


def _next_actions(
    families: tuple[ArmDiscoveryCapability, ...], free_ports: tuple[str, ...]
) -> tuple[HardwareNextAction, ...]:
    if not free_ports:
        return ()
    ports = _literal_list(free_ports)
    actions: list[HardwareNextAction] = []
    for family in families:
        if family.supports_port_probe:
            probe_effects: tuple[Effect, ...] = (
                ("read_only",) if family.arm_type == "maker" else ("may_energize",)
            )
            actions.append(
                HardwareNextAction(
                    call=(
                        f"client.system.probe_maker_arm_ports(arm_type={family.arm_type!r}, ports={ports})"
                    ),
                    reason=f"Classify unassigned ports using the {family.label} bus protocol.",
                    effects=probe_effects,
                )
            )
        elif not family.motion_identify_energizes_follower:
            actions.append(
                HardwareNextAction(
                    call=(
                        "client.system.identify_maker_arm("
                        f"'robot', arm_type={family.arm_type!r}, ports={ports})"
                    ),
                    reason=f"Ask the user to move the {family.label} follower and identify its port.",
                    effects=("read_only", "moves_hardware"),
                )
            )
        if family.supports_gripper_wiggle:
            for port in free_ports:
                actions.append(
                    HardwareNextAction(
                        call=(f"client.system.wiggle_can_gripper({family.arm_type!r}, 'robot', {port!r})"),
                        reason=f"Visually identify the {family.label} arm on this one port.",
                        effects=("may_energize", "moves_hardware"),
                    )
                )
    return tuple(actions)


class HardwareFlows:
    """Passive hardware inspection bound to a ``Client``."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def inspect_hardware(self) -> HardwareContext:
        """Build an agent-readable hardware snapshot without touching a bus.

        Composes ``system.arms``, ``system.available_ports``,
        ``system.available_cameras`` and ``robots.list``. Each read is
        independent: failures appear in ``result.errors`` while successful
        sections remain available. This flow never probes, energizes, moves,
        creates or updates anything. ``next_actions`` contains literal SDK
        calls with ``read_only``, ``may_energize`` and ``moves_hardware``
        effects; it does not execute them.

        Example:
            >>> hardware = client.flows.inspect_hardware()
            >>> print(hardware.summary())
            Hardware: 4 visible ports; cameras: 1 visible; ...
            >>> hardware.unassigned_visible_ports
            ('/dev/tty.usbmodem3',)
        """
        errors: list[HardwareSectionError] = []
        arms_result = _read_section("arms", self._client.system.arms, errors)
        ports_result = _read_section("ports", self._client.system.available_ports, errors)
        cameras_result = _read_section("cameras", self._client.system.available_cameras, errors)
        robots_result = _read_section("robots", self._client.robots.list, errors)

        arm_families = (
            tuple(_manifest_entry(item) for item in arms_result.arms) if arms_result is not None else ()
        )

        visible_ports: tuple[str, ...] = ()
        ports_known = False
        if ports_result is not None:
            if ports_result.status != "success":
                errors.append(HardwareSectionError("ports", ports_result.message or "port listing failed"))
            else:
                visible_ports = tuple(ports_result.ports or ())
                ports_known = True

        visible_cameras: tuple[CameraInfo, ...] = ()
        cameras_known = False
        if cameras_result is not None:
            if cameras_result.status != "success":
                errors.append(
                    HardwareSectionError("cameras", cameras_result.message or "camera listing failed")
                )
            else:
                visible_cameras = tuple(cameras_result.cameras)
                cameras_known = True

        records: list[dict[str, Any]] = []
        robots_known = False
        if robots_result is not None:
            if robots_result.status != "success":
                errors.append(HardwareSectionError("robots", robots_result.message or "robot listing failed"))
            else:
                records = robots_result.robots
                robots_known = True

        robots: list[SavedRobotHardware] = []
        references: dict[str, list[str]] = defaultdict(list)
        missing: list[PortAssignment] = []
        for record in records:
            name = str(record.get("name") or "unnamed")
            ports: list[PortAssignment] = []
            for slot in PORT_SLOTS:
                value = record.get(slot)
                if not isinstance(value, str) or not value:
                    continue
                assignment = PortAssignment(
                    name, slot, value, value in visible_ports if ports_known else None
                )
                ports.append(assignment)
                references[value].append(f"{name}.{slot}")
                if assignment.present is False:
                    missing.append(assignment)
            raw_cameras = record.get("cameras")
            cameras = tuple(
                _camera_assignment(item, visible_cameras, cameras_known=cameras_known)
                for item in (raw_cameras if isinstance(raw_cameras, list) else [])
            )
            robots.append(
                SavedRobotHardware(
                    name=name,
                    arm_type=str(record.get("arm_type") or "so101"),
                    mode=str(record["mode"]) if record.get("mode") else None,
                    ports=tuple(ports),
                    cameras=cameras,
                    is_clean=_flag(record, "is_clean"),
                    follower_ready=_flag(record, "follower_ready"),
                    leader_ready=_flag(record, "leader_ready"),
                    arm_available=_flag(record, "arm_available"),
                )
            )

        observations = tuple(
            HardwareObservation(
                kind="shared_port_reference",
                port=port,
                references=tuple(refs),
                message=(
                    f"{port} is referenced by {', '.join(refs)}; saved records may intentionally "
                    "alias the same physical hardware."
                ),
            )
            for port, refs in references.items()
            if len(refs) > 1
        )
        assigned = set(references)
        free_ports = (
            tuple(port for port in visible_ports if port not in assigned)
            if ports_known and robots_known
            else ()
        )
        return HardwareContext(
            visible_ports=visible_ports,
            visible_cameras=visible_cameras,
            robots=tuple(robots),
            arm_families=arm_families,
            assigned_ports_missing=tuple(missing),
            unassigned_visible_ports=free_ports,
            observations=observations,
            next_actions=_next_actions(arm_families, free_ports),
            errors=tuple(errors),
        )
