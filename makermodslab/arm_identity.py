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

"""Arm-identity guard: catch swapped leader/follower ports before any motion.

Plugging the leader into the follower's port (and vice versa) still connects
fine — both arms speak the same protocol — but the wrong calibration then gets
applied and teleop runs backwards with the user-held arm energized. This module
runs after each arm's bus connects, strictly before torque enable, the first
action, or any calibration write, and never writes to the servos.

The primary signal is the EEPROM FINGERPRINT: lerobot's write_calibration
stamps every calibration file's homing_offset values into the servos verbatim,
so reading each servo's Homing_Offset and comparing against EVERY saved
calibration file (the "library") identifies which calibration this physical
arm actually carries. Positions are only a fallback — an arm parked folded or
at rest legitimately sits at (or, pushed by gravity, slightly past) its
recorded range extremes, so positions alone can't be trusted to refuse a
verified arm.

Decision table, per arm (first matching row wins):

1. Fingerprint matches the ASSIGNED config      -> PASS (skip position check).
2. Fingerprint matches a config assigned to a
   COUNTERPART slot of this same session        -> HARD BLOCK (ports swapped).
3. Fingerprint matches some OTHER library file  -> START with a named warning.
4. Fingerprint matches NOTHING (factory-reset
   EEPROM, abandoned calibration)               -> positions decide, softened:
   hard block only when >= FALLBACK_MIN_OUT_OF_RANGE_JOINTS joints are each
   more than FALLBACK_MARGIN_FRACTION of their range width outside the
   assigned ranges; otherwise start with a generic can't-verify warning.
"""

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .utils.config import FOLLOWER_CONFIG_PATH, LEADER_CONFIG_PATH

logger = logging.getLogger(__name__)

# lerobot's write_calibration stamps the JSON's homing_offset into EEPROM
# verbatim, so an exact match is expected; the tolerance only absorbs
# sign-magnitude re-encoding quirks.
HOMING_OFFSET_TOLERANCE = 2

# wrist_roll is a full-turn motor: calibration forces its range to (nearly) the
# whole 0-4095 span, so its position carries no identity signal. Its EEPROM
# homing offset DOES count for the fingerprint.
POSITION_EXEMPT_MOTORS = frozenset({"wrist_roll"})

# Position fallback thresholds — used ONLY when the fingerprint matches no
# library file (row 4). Sized against real SO-101 calibrations: non-exempt
# range widths run ~1300-2800 ticks, so 15% is ~200-420 ticks (~18-37 deg) —
# far beyond the tens of ticks gravity pushes a parked joint past its
# hand-swept limit, and far below the hundreds-to-thousands of ticks a
# factory-reset homing offset shifts the readings by. Requiring 3 of the 5
# position-informative joints means a majority must be grossly out; a correct
# arm folded at rest can legitimately sit 1-2 joints at/past its extremes.
FALLBACK_MARGIN_FRACTION = 0.15
FALLBACK_MIN_OUT_OF_RANGE_JOINTS = 3


class ArmIdentityError(RuntimeError):
    """Raised when a connected arm's live readings contradict its assigned
    calibration badly enough that starting would drive it with garbage values.
    The message is user-facing (start handlers return it as-is)."""


@dataclass(frozen=True)
class ArmSlot:
    """One assigned arm slot of the running session.

    `side` is the calibration library dir the config lives in ("leader" or
    "follower"); `label` is the user-facing slot name ("leader",
    "right follower", ...); `config_name` is the assigned config stem.
    """

    label: str
    side: str
    config_name: str


@dataclass
class ArmIdentityFinding:
    """Result of the position fallback against the assigned calibration."""

    out_of_range_joints: list[str] = field(default_factory=list)
    offset_mismatch_joints: list[str] = field(default_factory=list)

    @property
    def positions_mismatch(self) -> bool:
        """Enough joints grossly out of range to rule out a parked arm."""
        return len(self.out_of_range_joints) >= FALLBACK_MIN_OUT_OF_RANGE_JOINTS

    @property
    def offsets_mismatch(self) -> bool:
        """The servos' EEPROM holds different homing offsets than the file."""
        return bool(self.offset_mismatch_joints)


def _cal_value(cal: Any, key: str) -> int:
    """Read a calibration field from either a lerobot MotorCalibration or a
    plain dict (the calibration JSON shape)."""
    if isinstance(cal, dict):
        return cal[key]
    return getattr(cal, key)


def load_calibration_library() -> dict[tuple[str, str], dict[str, Any]]:
    """Every saved calibration on disk, keyed by (side, config_stem).

    Scans both device dirs — so_leader ("leader") and so_follower
    ("follower"). The same stem may exist on both sides with different
    contents, hence the side-qualified key. Unreadable or malformed files are
    skipped (logged): the guard must degrade, not invent failures.
    """
    library: dict[tuple[str, str], dict[str, Any]] = {}
    for side, directory in (("leader", LEADER_CONFIG_PATH), ("follower", FOLLOWER_CONFIG_PATH)):
        try:
            filenames = sorted(os.listdir(directory))
        except OSError:
            continue
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Arm identity: skipping unreadable calibration {path}: {e}")
                continue
            if isinstance(data, dict) and data:
                library[(side, filename[: -len(".json")])] = data
    return library


def fingerprint_match(
    eeprom_offsets: dict[str, int],
    library: dict[tuple[str, str], dict[str, Any]],
    tolerance: int = HOMING_OFFSET_TOLERANCE,
) -> list[tuple[str, str]]:
    """Which library calibration(s) this arm's servos actually carry.

    A file matches when EVERY motor it lists has an EEPROM Homing_Offset
    within `tolerance` ticks of the file's homing_offset. All motors count
    here, including wrist_roll: its RANGE carries no identity signal (full
    turn), but write_calibration stamps its homing offset like any other.
    Returns the matching (side, config_name) keys — usually zero or one, more
    only when identical files exist (e.g. an imported copy).
    """
    matches: list[tuple[str, str]] = []
    for key, cal in library.items():
        try:
            ok = bool(cal) and all(
                motor in eeprom_offsets
                and abs(eeprom_offsets[motor] - _cal_value(spec, "homing_offset")) <= tolerance
                for motor, spec in cal.items()
            )
        except (KeyError, TypeError, AttributeError):
            continue  # malformed entry: not a fingerprint candidate
        if ok:
            matches.append(key)
    return matches


def check_arm_identity(
    present_positions: dict[str, int],
    eeprom_offsets: dict[str, int],
    calibration: dict[str, Any],
) -> ArmIdentityFinding:
    """Position fallback: compare raw readings against the assigned calibration.

    `present_positions` are raw Present_Position ticks (normalize=False),
    `eeprom_offsets` are decoded Homing_Offset values read from the servos, and
    `calibration` maps motor name to a MotorCalibration (or the equivalent
    dict). A joint counts as out of range only when it sits more than
    FALLBACK_MARGIN_FRACTION of its range width outside [range_min, range_max].
    Pure function — the reads happen in `verify_arm`.
    """
    finding = ArmIdentityFinding()
    for motor, cal in calibration.items():
        if motor not in POSITION_EXEMPT_MOTORS and motor in present_positions:
            range_min = _cal_value(cal, "range_min")
            range_max = _cal_value(cal, "range_max")
            margin = FALLBACK_MARGIN_FRACTION * (range_max - range_min)
            position = present_positions[motor]
            if position < range_min - margin or position > range_max + margin:
                finding.out_of_range_joints.append(motor)
        if motor in eeprom_offsets:
            expected = _cal_value(cal, "homing_offset")
            if abs(eeprom_offsets[motor] - expected) > HOMING_OFFSET_TOLERANCE:
                finding.offset_mismatch_joints.append(motor)
    return finding


def swapped_port_refusal(label: str, port: str, matched_name: str, counterpart_label: str) -> str:
    """Row 2: the arm carries a calibration assigned to another slot of this session."""
    return (
        f"The {label} arm on {port} carries calibration '{matched_name}' — assigned to the "
        f"{counterpart_label} arm. Ports appear to be SWAPPED. Swap them back (or reassign) "
        "before starting."
    )


def wrong_calibration_warning(label: str, port: str, matched_names: list[str], assigned_name: str) -> str:
    """Row 3: the arm carries some other saved calibration, not the assigned one."""
    matched = "', '".join(matched_names)
    return (
        f"The {label} arm on {port} carries calibration '{matched}', not the assigned "
        f"'{assigned_name}' — if you just reassigned configs this is expected; otherwise "
        "check your ports."
    )


def unverified_refusal(label: str, port: str, config_name: str, finding: ArmIdentityFinding) -> str:
    """Row 4 hard block: unrecognized EEPROM AND most joints grossly out of range."""
    joints = ", ".join(finding.out_of_range_joints)
    return (
        f"The {label} arm on {port} doesn't match calibration '{config_name}' "
        f"({joints} far out of range, and its servos match no saved calibration) — "
        "the ports may be swapped, or this arm needs recalibration. Unplug one arm "
        "at a time to confirm which port is which."
    )


def unverified_warning(label: str, port: str, config_name: str) -> str:
    """Row 4 warn: unrecognized EEPROM but positions are plausible."""
    return (
        f"Could not verify the {label} arm on {port}: its servos' homing offsets match no "
        f"saved calibration (a factory reset or an interrupted calibration leaves them "
        f"unset). Starting with '{config_name}' anyway — recalibrate this arm if it "
        "behaves oddly."
    )


def verify_arm(
    bus,
    calibration: dict[str, Any],
    label: str,
    config_name: str,
    side: str,
    counterparts: Iterable[ArmSlot] = (),
    library: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[str | None, str | None]:
    """Read-only identity check of one connected arm's bus.

    Runs the decision table from the module docstring. `side` is the library
    dir of the assigned config ("leader"/"follower"), `counterparts` the other
    assigned slots of the same session (row 2), and `library` every saved
    calibration (loaded from disk when None).

    Returns (refusal, warning): `refusal` is a user-facing hard-block message,
    `warning` a proceed-with-caution message; both None when the arm is
    verified. Reads Present_Position and Homing_Offset only — never writes. A
    read failure skips the check (logged) rather than inventing a new failure
    mode: a genuinely dead bus will fail loudly on the first real action anyway.
    """
    if not calibration:
        # Nothing to compare against; the missing-calibration case is handled
        # (and refused) upstream by setup_calibration_files.
        return None, None
    port = getattr(bus, "port", None) or "unknown port"
    try:
        positions = bus.sync_read("Present_Position", normalize=False)
        offsets = {
            motor: bus.read("Homing_Offset", motor, normalize=False)
            for motor in calibration
            if motor in bus.motors
        }
    except Exception as e:
        logger.warning(f"Arm identity check on {port} ({label}) could not read the motors, skipping: {e}")
        return None, None

    if library is None:
        library = load_calibration_library()
    matches = fingerprint_match(offsets, library)

    # Row 1: the servos carry the assigned calibration — this is the right arm.
    # Skip the position check entirely: resting at/past range extremes is normal.
    if (side, config_name) in matches:
        return None, None

    # Row 2: the servos carry a calibration assigned to another slot of this
    # same session — the classic swapped-ports case, even freshly calibrated.
    for slot in counterparts:
        if (slot.side, slot.config_name) in matches:
            return swapped_port_refusal(label, port, slot.config_name, slot.label), None

    # Row 3: the servos carry some other saved calibration. Named warning, and
    # explicitly NO position block — the arm is identified, just reassigned.
    if matches:
        matched_names = sorted({name for _side, name in matches})
        return None, wrong_calibration_warning(label, port, matched_names, config_name)

    # Row 4: fingerprint inconclusive (factory-reset EEPROM, abandoned
    # calibration). Fall back to positions, softened.
    finding = check_arm_identity(positions, offsets, calibration)
    if finding.positions_mismatch:
        return unverified_refusal(label, port, config_name, finding), None
    return None, unverified_warning(label, port, config_name)


def _device_arms(device, side: str) -> list[tuple[Any, str]]:
    """(arm, label) for every arm of a robot/teleop device.

    A single-arm device exposes `.bus`/`.calibration`; a bimanual BiSO device
    exposes `left_arm`/`right_arm` sub-arms which each carry their own.
    """
    return [
        (arm, f"{prefix} {side}")
        for prefix, arm in (
            ("left", getattr(device, "left_arm", None)),
            ("right", getattr(device, "right_arm", None)),
        )
        if arm is not None
    ] or [(device, side)]


def verify_devices(
    pairs: Iterable[tuple[Any, str]],
    skip: bool = False,
    extra_slots: Iterable[ArmSlot] = (),
    library: dict[tuple[str, str], dict[str, Any]] | None = None,
    config_names: Sequence[str] | None = None,
) -> list[str]:
    """Run the identity guard over (device, side) pairs whose buses are connected.

    `side` is "leader" or "follower" — the calibration dir the device's
    config(s) live in. Every arm across the pairs is checked against every
    other arm's assigned slot (row 2 of the decision table); `extra_slots`
    adds counterpart slots for arms NOT present in this session (e.g. the
    robot's leader config during a follower-only inference start).

    By default each arm's assigned config name is read from its device `id`.
    For a bimanual BiSO session the sub-arm ids are the staging aliases
    ("<base>_left"/"<base>_right"), not the library stems the identity library
    is keyed by, so pass `config_names` — the real library stems in the same
    order the arms are iterated (each (device, side) pair yields left then right
    sub-arm, single-arm devices yield one) — to compare against the library.

    Raises ArmIdentityError on a hard mismatch (message covers every failing
    arm); returns the warn-but-allow messages otherwise (also logged). `skip`
    is the explicit user escape hatch — the guard is bypassed entirely.
    """
    if skip:
        logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
        return []

    arms: list[tuple[Any, str, str]] = []  # (arm, label, side)
    for device, side in pairs:
        if device is None:
            continue
        for arm, label in _device_arms(device, side):
            if getattr(arm, "bus", None) is None:
                continue
            arms.append((arm, label, side))
    if not arms:
        return []

    if library is None:
        library = load_calibration_library()

    if config_names is not None and len(config_names) != len(arms):
        raise ValueError(f"config_names has {len(config_names)} entries but {len(arms)} arms were connected")
    slots = [
        ArmSlot(
            label,
            side,
            (config_names[i] if config_names is not None else str(getattr(arm, "id", "") or "unknown")),
        )
        for i, (arm, label, side) in enumerate(arms)
    ]
    all_slots = slots + list(extra_slots)

    refusals: list[str] = []
    warnings: list[str] = []
    for index, (arm, label, side) in enumerate(arms):
        counterparts = [slot for i, slot in enumerate(all_slots) if i != index]
        refusal, warning = verify_arm(
            arm.bus,
            getattr(arm, "calibration", None),
            label,
            slots[index].config_name,
            side,
            counterparts=counterparts,
            library=library,
        )
        if refusal:
            refusals.append(refusal)
        if warning:
            warnings.append(warning)
    for message in warnings:
        logger.warning(message)
    if refusals:
        logger.error("Arm identity check failed: %s", " ".join(refusals))
        raise ArmIdentityError(" ".join(refusals))
    return warnings
