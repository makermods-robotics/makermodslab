# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for makermodslab.arm_identity — the fingerprint-first swapped-port guard.

The fixtures use the real numbers from the 2026-07-02 swapped-port incidents:
the two arms' gripper homing-offset families are -98 (leader) vs -1202
(follower) — over a thousand ticks apart — while their gripper ranges are
nearly identical ([1387, 2707] vs [1407, 2687]). The v1 guard missed a swap
of two FRESHLY CALIBRATED arms (each arm's EEPROM matched its own file and
both arms' similarly-centered ranges passed the position check) and hard-
blocked a CORRECT arm parked past its hand-swept range extremes. v2 decides
by EEPROM fingerprint against the whole calibration library first; positions
are only a softened fallback when the fingerprint matches nothing.
"""

from __future__ import annotations

import json

import pytest

import makermodslab.arm_identity as arm_identity
from makermodslab.arm_identity import (
    ArmIdentityError,
    ArmSlot,
    check_arm_identity,
    fingerprint_match,
    load_calibration_library,
    verify_arm,
    verify_devices,
)

# The leader arm's calibration file (assigned to the leader slot).
LEADER_CAL = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -46, "range_min": 758, "range_max": 3292},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 214, "range_min": 862, "range_max": 3128},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -1103, "range_min": 921, "range_max": 3181},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 68, "range_min": 880, "range_max": 3244},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -371, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -98, "range_min": 1387, "range_max": 2707},
}

# The follower arm's calibration file (assigned to the follower slot). Its
# homing-offset family differs by hundreds-to-thousands of ticks, but its
# ranges are similarly centered — which is exactly why a position check can't
# catch a swap of two freshly calibrated arms.
FOLLOWER_CAL = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -871, "range_min": 664, "range_max": 3430},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 1010, "range_min": 830, "range_max": 3264},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -364, "range_min": 928, "range_max": 3166},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 792, "range_min": 844, "range_max": 3250},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 233, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -1202, "range_min": 1407, "range_max": 2687},
}


def _offsets(cal: dict, shift: int = 0) -> dict[str, int]:
    return {motor: spec["homing_offset"] + shift for motor, spec in cal.items()}


def _shifted_cal(cal: dict, shift: int) -> dict:
    return {motor: dict(spec, homing_offset=spec["homing_offset"] + shift) for motor, spec in cal.items()}


LEADER_OFFSETS = _offsets(LEADER_CAL)
FOLLOWER_OFFSETS = _offsets(FOLLOWER_CAL)
# Abandoned previous calibrations, still stamped in EEPROM after a config was
# assigned without recalibrating.
OLD_LEADER_CAL = _shifted_cal(LEADER_CAL, 40)
OLD_FOLLOWER_CAL = _shifted_cal(FOLLOWER_CAL, 40)
# Factory-reset EEPROM: every homing offset zeroed. Matches no library file.
FACTORY_OFFSETS = dict.fromkeys(LEADER_CAL, 0)

# The calibration library on disk: both assigned files plus abandoned ones.
LIBRARY = {
    ("leader", "leader_a"): LEADER_CAL,
    ("follower", "follower_a"): FOLLOWER_CAL,
    ("leader", "old_leader"): OLD_LEADER_CAL,
    ("follower", "follower_old"): OLD_FOLLOWER_CAL,
}

# An arm at rest mid-range: inside both files' ranges (the swap-blind spot).
RESTING_POSITIONS = {
    "shoulder_pan": 2025,
    "shoulder_lift": 1995,
    "elbow_flex": 2051,
    "wrist_flex": 2062,
    "wrist_roll": 2047,
    "gripper": 1450,
}

# A folded/parked arm pushed by gravity: three joints grossly past the
# LEADER_CAL extremes (each >15% of its range width out).
PARKED_PAST_EXTREMES = {
    "shoulder_pan": 2013,
    "shoulder_lift": 3700,  # > 3128 + 340 (15% of 2266)
    "elbow_flex": 145,  # < 921 - 339 (15% of 2260)
    "wrist_flex": 2100,
    "wrist_roll": 2047,
    "gripper": 283,  # < 1387 - 198 (15% of 1320)
}

LEADER_SLOT = ArmSlot("leader", "leader", "leader_a")
FOLLOWER_SLOT = ArmSlot("follower", "follower", "follower_a")


# ---------------------------------------------------------------------------
# fingerprint_match — the pure fingerprint helper
# ---------------------------------------------------------------------------


def test_fingerprint_exact_match_identifies_the_file() -> None:
    assert fingerprint_match(LEADER_OFFSETS, LIBRARY) == [("leader", "leader_a")]
    assert fingerprint_match(FOLLOWER_OFFSETS, LIBRARY) == [("follower", "follower_a")]


def test_fingerprint_tolerance_is_two_ticks() -> None:
    within = _offsets(LEADER_CAL, 2)
    beyond = _offsets(LEADER_CAL, 3)
    assert ("leader", "leader_a") in fingerprint_match(within, LIBRARY)
    assert fingerprint_match(beyond, LIBRARY) == []


def test_fingerprint_counts_wrist_roll_homing() -> None:
    """wrist_roll is position-exempt (full turn) but its homing offset is
    stamped like any other — a wrist_roll-only difference breaks the match."""
    offsets = dict(LEADER_OFFSETS, wrist_roll=LEADER_OFFSETS["wrist_roll"] + 10)
    assert fingerprint_match(offsets, LIBRARY) == []


def test_fingerprint_requires_every_joint_of_the_file() -> None:
    partial = dict(LEADER_OFFSETS)
    del partial["gripper"]
    assert fingerprint_match(partial, LIBRARY) == []


def test_fingerprint_returns_every_duplicate_match() -> None:
    library = dict(LIBRARY)
    library[("leader", "leader_copy")] = json.loads(json.dumps(LEADER_CAL))
    matches = fingerprint_match(LEADER_OFFSETS, library)
    assert set(matches) == {("leader", "leader_a"), ("leader", "leader_copy")}


def test_load_calibration_library_scans_both_dirs_and_skips_bad_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader_dir = tmp_path / "so_leader"
    follower_dir = tmp_path / "so_follower"
    leader_dir.mkdir()
    follower_dir.mkdir()
    (leader_dir / "leader_a.json").write_text(json.dumps(LEADER_CAL))
    (follower_dir / "follower_a.json").write_text(json.dumps(FOLLOWER_CAL))
    (follower_dir / "broken.json").write_text("{not json")
    (follower_dir / "notes.txt").write_text("ignored")
    monkeypatch.setattr(arm_identity, "LEADER_CONFIG_PATH", str(leader_dir))
    monkeypatch.setattr(arm_identity, "FOLLOWER_CONFIG_PATH", str(follower_dir))

    library = load_calibration_library()

    assert set(library) == {("leader", "leader_a"), ("follower", "follower_a")}
    assert library[("leader", "leader_a")]["gripper"]["homing_offset"] == -98


# ---------------------------------------------------------------------------
# The bus/device doubles
# ---------------------------------------------------------------------------


class _GuardBus:
    """Bus double: serves canned Present_Position / Homing_Offset reads."""

    def __init__(self, positions: dict, offsets: dict, port: str = "COM_TEST") -> None:
        self.port = port
        self.motors = dict.fromkeys(offsets)
        self._positions = positions
        self._offsets = offsets
        self.guard_reads = 0
        self.calibration_written = False

    def connect(self) -> None:
        pass

    def sync_read(self, data_name: str, *, normalize: bool = True) -> dict:
        assert data_name == "Present_Position" and normalize is False
        self.guard_reads += 1
        return dict(self._positions)

    def read(self, data_name: str, motor: str, *, normalize: bool = True) -> int:
        assert data_name == "Homing_Offset" and normalize is False
        self.guard_reads += 1
        return self._offsets[motor]

    def write_calibration(self, calibration) -> None:
        self.calibration_written = True

    def disable_torque(self, motor=None, num_retry: int = 0) -> None:
        pass


class _GuardDevice:
    """Single-arm robot/teleop double carrying a _GuardBus and a calibration."""

    def __init__(self, bus: _GuardBus, calibration: dict, device_id: str = "leader_a") -> None:
        self.bus = bus
        self.calibration = calibration
        self.id = device_id
        self.cameras: dict = {}
        self.disconnected = False

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self.disconnected = True

    def get_action(self) -> dict:
        raise RuntimeError("test double: end the worker loop")

    def send_action(self, action) -> None:
        pass


def _leader_arm_bus(positions: dict = RESTING_POSITIONS, port: str = "COM_TEST") -> _GuardBus:
    """The physical leader arm: its EEPROM carries the leader_a fingerprint."""
    return _GuardBus(positions, LEADER_OFFSETS, port=port)


# ---------------------------------------------------------------------------
# Decision table row 1: fingerprint matches the ASSIGNED config -> PASS
# ---------------------------------------------------------------------------


def test_row1_verified_arm_at_rest_passes() -> None:
    refusal, warning = verify_arm(
        _leader_arm_bus(),
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[FOLLOWER_SLOT],
        library=LIBRARY,
    )
    assert refusal is None and warning is None


def test_row1_verified_arm_parked_past_its_range_extremes_passes_with_no_warning() -> None:
    """The v1 'too strict' incident: the CORRECT arm rests folded, gravity
    pushed three joints past the hand-swept calibration extremes. The
    fingerprint proves identity, so the position check is skipped entirely."""
    refusal, warning = verify_arm(
        _leader_arm_bus(PARKED_PAST_EXTREMES),
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[FOLLOWER_SLOT],
        library=LIBRARY,
    )
    assert refusal is None and warning is None


def test_row1_assigned_match_wins_over_a_duplicate_counterpart_match() -> None:
    """If the assigned and a counterpart file are identical copies, the arm is
    consistent with its assignment — pass, don't cry swap."""
    library = {("leader", "leader_a"): LEADER_CAL, ("follower", "twin"): json.loads(json.dumps(LEADER_CAL))}
    refusal, warning = verify_arm(
        _leader_arm_bus(),
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[ArmSlot("follower", "follower", "twin")],
        library=library,
    )
    assert refusal is None and warning is None


# ---------------------------------------------------------------------------
# Decision table row 2: fingerprint matches a COUNTERPART slot -> HARD BLOCK
# ---------------------------------------------------------------------------


def test_row2_freshly_calibrated_swap_is_blocked_naming_both_configs() -> None:
    """The v1 'missed swap' incident: both arms freshly calibrated, then the
    ports swapped. The leader arm shows up on the follower port with resting
    positions inside the follower file's similarly-centered ranges — the v1
    position signal passed. The fingerprint names the swap."""
    bus = _leader_arm_bus(port="/dev/tty.usbfollower")
    refusal, warning = verify_arm(
        bus,
        FOLLOWER_CAL,
        "follower",
        "follower_a",
        "follower",
        counterparts=[LEADER_SLOT],
        library=LIBRARY,
    )
    assert warning is None
    assert refusal is not None
    assert "/dev/tty.usbfollower" in refusal
    assert "'leader_a'" in refusal  # the calibration the arm actually carries
    assert "leader arm" in refusal  # ... and the slot it belongs to
    assert "SWAPPED" in refusal


def test_row2_verify_devices_blocks_the_swapped_teleop_pair() -> None:
    """Session-level: each device derives its counterpart from the OTHER
    device in the same call — no extra wiring at the teleop/record call sites."""
    follower = _GuardDevice(
        _GuardBus(RESTING_POSITIONS, LEADER_OFFSETS, port="COM_F"), FOLLOWER_CAL, "follower_a"
    )
    leader = _GuardDevice(
        _GuardBus(RESTING_POSITIONS, FOLLOWER_OFFSETS, port="COM_L"), LEADER_CAL, "leader_a"
    )

    with pytest.raises(ArmIdentityError) as excinfo:
        verify_devices(((follower, "follower"), (leader, "leader")), library=LIBRARY)
    message = str(excinfo.value)
    assert "'leader_a'" in message and "'follower_a'" in message
    assert "SWAPPED" in message


def test_row2_bimanual_same_side_counterpart_is_blocked() -> None:
    """Bimanual: the left leader port carries the RIGHT leader's calibration
    (left/right cables crossed) — a same-side counterpart, hard block."""
    bi_library = {
        ("leader", "biman_left"): LEADER_CAL,
        ("leader", "biman_right"): _shifted_cal(LEADER_CAL, 700),
    }

    class _BiDevice:
        def __init__(self) -> None:
            # Left arm's servos carry the right arm's fingerprint.
            self.left_arm = _GuardDevice(
                _GuardBus(RESTING_POSITIONS, _offsets(LEADER_CAL, 700), port="COM_LEFT"),
                LEADER_CAL,
                "biman_left",
            )
            self.right_arm = _GuardDevice(
                _GuardBus(RESTING_POSITIONS, LEADER_OFFSETS, port="COM_RIGHT"),
                _shifted_cal(LEADER_CAL, 700),
                "biman_right",
            )

    with pytest.raises(ArmIdentityError) as excinfo:
        verify_devices(((_BiDevice(), "leader"),), library=bi_library)
    message = str(excinfo.value)
    assert "left leader" in message and "'biman_right'" in message and "SWAPPED" in message


def test_row2_extra_slots_cover_arms_not_in_the_session() -> None:
    """Inference connects only the follower; its leader counterpart arrives as
    an extra slot (looked up from the robot record) and still hard-blocks."""
    follower = _GuardDevice(_leader_arm_bus(), FOLLOWER_CAL, "follower_a")

    with pytest.raises(ArmIdentityError, match="SWAPPED"):
        verify_devices(((follower, "follower"),), extra_slots=[LEADER_SLOT], library=LIBRARY)


# ---------------------------------------------------------------------------
# Decision table row 3: fingerprint matches some OTHER library file -> warn
# ---------------------------------------------------------------------------


def test_row3_other_library_file_starts_with_a_named_warning() -> None:
    """Assign-without-recalibrate: the arm's EEPROM still carries its previous
    calibration ('follower_old'). Start, but name both files."""
    bus = _GuardBus(RESTING_POSITIONS, _offsets(OLD_FOLLOWER_CAL))
    refusal, warning = verify_arm(
        bus,
        FOLLOWER_CAL,
        "follower",
        "follower_a",
        "follower",
        counterparts=[LEADER_SLOT],
        library=LIBRARY,
    )
    assert refusal is None
    assert warning is not None
    assert "'follower_old'" in warning and "'follower_a'" in warning
    assert "check your ports" in warning


def test_row3_never_position_blocks() -> None:
    """A fingerprinted (just not assigned) arm parked past its extremes must
    still start — identity is known, the pose is irrelevant."""
    bus = _GuardBus(PARKED_PAST_EXTREMES, _offsets(OLD_LEADER_CAL))
    refusal, warning = verify_arm(
        bus,
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[FOLLOWER_SLOT],
        library=LIBRARY,
    )
    assert refusal is None
    assert warning is not None and "'old_leader'" in warning


# ---------------------------------------------------------------------------
# Decision table row 4: fingerprint matches NOTHING -> softened position fallback
# ---------------------------------------------------------------------------


def test_row4_factory_reset_arm_in_plausible_pose_warns_but_starts() -> None:
    bus = _GuardBus(RESTING_POSITIONS, FACTORY_OFFSETS)
    refusal, warning = verify_arm(
        bus,
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[FOLLOWER_SLOT],
        library=LIBRARY,
    )
    assert refusal is None
    assert warning is not None
    assert "Could not verify" in warning


def test_row4_three_joints_far_out_of_range_blocks() -> None:
    """Unrecognized EEPROM AND a majority of joints >15% of their range width
    outside the assigned ranges: refuse."""
    bus = _GuardBus(PARKED_PAST_EXTREMES, FACTORY_OFFSETS)
    refusal, warning = verify_arm(
        bus,
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        counterparts=[FOLLOWER_SLOT],
        library=LIBRARY,
    )
    assert warning is None
    assert refusal is not None
    assert "shoulder_lift" in refusal and "elbow_flex" in refusal and "gripper" in refusal
    assert "'leader_a'" in refusal


def test_row4_two_far_out_joints_warn_instead_of_blocking() -> None:
    positions = dict(PARKED_PAST_EXTREMES, gripper=1450)  # back in range: only 2 far out
    finding = check_arm_identity(positions, FACTORY_OFFSETS, LEADER_CAL)
    assert finding.out_of_range_joints == ["shoulder_lift", "elbow_flex"]
    assert not finding.positions_mismatch

    refusal, warning = verify_arm(
        _GuardBus(positions, FACTORY_OFFSETS),
        LEADER_CAL,
        "leader",
        "leader_a",
        "leader",
        library=LIBRARY,
    )
    assert refusal is None and warning is not None


def test_row4_margin_boundary_is_15_percent_of_range_width() -> None:
    """gripper range [1387, 2707] -> width 1320, margin 198: 1189 is the last
    in-margin tick below range_min; 1188 is out."""
    inside = dict(RESTING_POSITIONS, gripper=1189)
    outside = dict(RESTING_POSITIONS, gripper=1188)
    assert check_arm_identity(inside, FACTORY_OFFSETS, LEADER_CAL).out_of_range_joints == []
    assert check_arm_identity(outside, FACTORY_OFFSETS, LEADER_CAL).out_of_range_joints == ["gripper"]


def test_row4_three_joints_out_but_within_margin_do_not_block() -> None:
    """Slightly past the hand-swept extremes (gravity sag) on three joints is
    still a start-with-warning, not a block."""
    positions = dict(
        RESTING_POSITIONS,
        shoulder_lift=3460,  # < 3128 + 339.9
        elbow_flex=590,  # > 921 - 339
        gripper=1200,  # > 1387 - 198
    )
    finding = check_arm_identity(positions, FACTORY_OFFSETS, LEADER_CAL)
    assert finding.out_of_range_joints == []
    assert not finding.positions_mismatch


def test_row4_wrist_roll_is_exempt_from_the_position_fallback() -> None:
    """wrist_roll is full-turn: no position signal even against a synthetic
    narrow range."""
    cal = {
        "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 2000},
        "gripper": LEADER_CAL["gripper"],
    }
    positions = {"wrist_roll": 3900, "gripper": 283}
    finding = check_arm_identity(positions, {"wrist_roll": 500, "gripper": 500}, cal)
    assert finding.out_of_range_joints == ["gripper"]
    assert not finding.positions_mismatch


# ---------------------------------------------------------------------------
# verify_arm / verify_devices — wrappers, escape hatch, fail-open
# ---------------------------------------------------------------------------


def test_verify_arm_fails_open_when_the_bus_cannot_be_read() -> None:
    """A flaky read must not invent a new failure mode; the first real action
    would fail loudly anyway."""

    class _DeadBus(_GuardBus):
        def sync_read(self, data_name, *, normalize=True):
            raise ConnectionError("no response")

    refusal, warning = verify_arm(
        _DeadBus({}, LEADER_OFFSETS), LEADER_CAL, "leader", "leader_a", "leader", library=LIBRARY
    )
    assert refusal is None and warning is None


def test_verify_devices_skip_bypasses_without_reading() -> None:
    device = _GuardDevice(_leader_arm_bus(), FOLLOWER_CAL, "follower_a")
    assert verify_devices(((device, "follower"),), skip=True, library=LIBRARY) == []
    assert device.bus.guard_reads == 0


def test_verify_devices_returns_warnings_for_allowed_starts() -> None:
    device = _GuardDevice(_GuardBus(RESTING_POSITIONS, FACTORY_OFFSETS), LEADER_CAL, "leader_a")
    warnings = verify_devices(((device, "leader"),), library=LIBRARY)
    assert len(warnings) == 1 and "Could not verify" in warnings[0]


def test_verify_devices_config_names_override_staging_alias_ids() -> None:
    """Bimanual staging gives sub-arms alias ids ("<base>_left"), but the library
    is keyed by real stems. Passing config_names lets the guard compare against
    the library and PASS (Row 1) instead of spuriously warning (Row 3)."""
    # Device id is the staging alias; the servos carry the real "leader_a"
    # fingerprint present in LIBRARY under its real stem.
    device = _GuardDevice(_leader_arm_bus(), LEADER_CAL, "biman_left")

    # Without the override, id "biman_left" isn't in the library -> Row 3 warning
    # ("carries calibration 'leader_a', not the assigned 'biman_left'").
    warnings = verify_devices(((device, "leader"),), library=LIBRARY)
    assert len(warnings) == 1
    assert "'leader_a'" in warnings[0] and "'biman_left'" in warnings[0]

    # With config_names=["leader_a"], Row 1 matches -> clean pass, no warning.
    device2 = _GuardDevice(_leader_arm_bus(), LEADER_CAL, "biman_left")
    assert verify_devices(((device2, "leader"),), library=LIBRARY, config_names=["leader_a"]) == []


def test_verify_devices_config_names_length_mismatch_raises() -> None:
    device = _GuardDevice(_leader_arm_bus(), LEADER_CAL, "biman_left")
    with pytest.raises(ValueError, match="config_names has 2 entries but 1 arms"):
        verify_devices(((device, "leader"),), library=LIBRARY, config_names=["a", "b"])


# ---------------------------------------------------------------------------
# Handler-level: teleoperation start
# ---------------------------------------------------------------------------


def _patch_teleop_devices(monkeypatch: pytest.MonkeyPatch, follower: _GuardDevice, leader: _GuardDevice):
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower, arm_type="so101": ("leader_a", "follower_a"),
    )
    monkeypatch.setattr(teleop, "SO101Follower", lambda config: follower)
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: leader)
    # The guard loads the calibration library from disk; pin it to the fixtures.
    monkeypatch.setattr(arm_identity, "load_calibration_library", lambda: LIBRARY)
    return teleop


def _teleop_request(**overrides):
    from makermodslab.teleoperate import TeleoperateRequest

    fields = {
        "leader_port": "COM_LEADER",
        "follower_port": "COM_FOLLOWER",
        "leader_config": "leader_a",
        "follower_config": "follower_a",
    }
    fields.update(overrides)
    return TeleoperateRequest(**fields)


def _join_teleop_worker(teleop) -> None:
    """The worker's first get_action raises, so it exits and cleans up fast."""
    worker = teleop.teleoperation_thread
    if worker is not None:
        worker.join(timeout=5.0)
        assert not worker.is_alive()


def test_start_teleoperation_refuses_a_freshly_calibrated_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The incident v1 missed: both arms freshly calibrated, ports swapped,
    every position inside the (similarly centered) assigned ranges. The start
    must fail BEFORE any calibration write, and both devices must be released."""
    follower = _GuardDevice(
        _GuardBus(RESTING_POSITIONS, LEADER_OFFSETS, port="COM_FOLLOWER"), FOLLOWER_CAL, "follower_a"
    )
    leader = _GuardDevice(
        _GuardBus(RESTING_POSITIONS, FOLLOWER_OFFSETS, port="COM_LEADER"), LEADER_CAL, "leader_a"
    )
    teleop = _patch_teleop_devices(monkeypatch, follower, leader)

    result = teleop.handle_start_teleoperation(_teleop_request())

    assert result["success"] is False
    assert "COM_FOLLOWER" in result["message"]
    assert "SWAPPED" in result["message"]
    assert "'leader_a'" in result["message"] and "'follower_a'" in result["message"]
    # The guard fired before write_calibration could contaminate the EEPROM.
    assert follower.bus.calibration_written is False
    assert leader.bus.calibration_written is False
    assert follower.disconnected is True
    assert leader.disconnected is True
    assert teleop.teleoperation_active is False


def test_start_teleoperation_skip_flag_bypasses_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    follower = _GuardDevice(_GuardBus(RESTING_POSITIONS, LEADER_OFFSETS), FOLLOWER_CAL, "follower_a")
    leader = _GuardDevice(_GuardBus(RESTING_POSITIONS, FOLLOWER_OFFSETS), LEADER_CAL, "leader_a")
    teleop = _patch_teleop_devices(monkeypatch, follower, leader)

    result = teleop.handle_start_teleoperation(_teleop_request(skip_identity_check=True))

    assert result["success"] is True
    # No identity reads happened; calibration was written as usual. The single
    # follower read is NOT the guard: it is the rest-pose capture
    # (makermodslab/rest_pose.py) every start performs so a normal stop can drive the
    # follower back to its starting pose. The leader — never returned to rest,
    # human-held — must stay untouched.
    assert follower.bus.guard_reads == 1
    assert leader.bus.guard_reads == 0
    assert follower.bus.calibration_written is True
    _join_teleop_worker(teleop)


def test_start_teleoperation_surfaces_a_row3_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assign-without-recalibrate starts, with the other file named in the
    response's `warning` (which the frontend now shows as a toast)."""
    follower = _GuardDevice(
        _GuardBus(RESTING_POSITIONS, _offsets(OLD_FOLLOWER_CAL)), FOLLOWER_CAL, "follower_a"
    )
    leader = _GuardDevice(_GuardBus(RESTING_POSITIONS, LEADER_OFFSETS), LEADER_CAL, "leader_a")
    teleop = _patch_teleop_devices(monkeypatch, follower, leader)

    result = teleop.handle_start_teleoperation(_teleop_request())

    assert result["success"] is True
    assert "'follower_old'" in result["warning"]
    assert "check your ports" in result["warning"]
    _join_teleop_worker(teleop)


def test_start_teleoperation_verified_arms_pass_without_identity_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arms verified by fingerprint, parked past their extremes: start."""
    follower = _GuardDevice(_GuardBus(PARKED_PAST_EXTREMES, FOLLOWER_OFFSETS), FOLLOWER_CAL, "follower_a")
    leader = _GuardDevice(_GuardBus(PARKED_PAST_EXTREMES, LEADER_OFFSETS), LEADER_CAL, "leader_a")
    teleop = _patch_teleop_devices(monkeypatch, follower, leader)

    result = teleop.handle_start_teleoperation(_teleop_request())

    assert result["success"] is True
    # Any warning present must be about motor power (the doubles reject the
    # Torque_Limit write), never about arm identity.
    assert "calibration" not in result.get("warning", "")
    _join_teleop_worker(teleop)


# ---------------------------------------------------------------------------
# Handler-level: inference start + request-model defaults
# ---------------------------------------------------------------------------


def test_start_inference_refuses_on_identity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The identity preflight moved into the async startup worker (it opens the
    serial bus, which the instant-start POST must not do). A hard mismatch is
    now finalised by the worker as a `failed` outcome carrying the user-facing
    message — before any subprocess spawns."""
    import threading

    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_STARTING, "policy_ref": "/tmp/model"},
    )
    # Local-path ref: resolve is instant, no download phase — isolates the
    # identity failure as the first thing the worker hits.
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: "/tmp/model")

    def _refuse(request):
        raise ArmIdentityError("The follower arm on /dev/f carries calibration 'leader_a' — SWAPPED")

    monkeypatch.setattr(rollout, "_prepare_robot", _refuse)

    def _no_popen(*a, **k):
        raise AssertionError("no subprocess may spawn after an identity refusal")

    monkeypatch.setattr(rollout.subprocess, "Popen", _no_popen)

    rollout._run_inference_startup(
        rollout.InferenceRequest(
            follower_port="/dev/f", follower_config="follower_a", policy_ref="/tmp/model"
        ),
        threading.Event(),
    )

    assert rollout.inference_active is False
    status = rollout.handle_inference_status()
    assert status["exited"] is True
    assert status["outcome"] == "failed"
    assert "SWAPPED" in (status["error"] or "")


def test_rollout_counterpart_slots_come_from_robot_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference has no leader in the session; the counterpart slot is looked
    up from any robot record pairing this follower config with a leader."""
    from makermodslab import rollout

    records = [
        {
            "name": "bench",
            "mode": "single",
            "leader_config": "leader_a",
            "follower_config": "follower_a",
            "right_leader_config": "",
            "right_follower_config": "",
        },
        {
            "name": "duo",
            "mode": "bimanual",
            "leader_config": "biman_l_left",
            "follower_config": "biman_f_left",
            "right_leader_config": "biman_l_right",
            "right_follower_config": "follower_a",
        },
    ]
    monkeypatch.setattr(rollout, "list_robot_records", lambda: records)

    slots = rollout._counterpart_leader_slots("follower_a")

    assert ArmSlot("leader", "leader", "leader_a") in slots
    assert ArmSlot("right leader", "leader", "biman_l_right") in slots
    assert all(slot.side == "leader" for slot in slots)
    assert rollout._counterpart_leader_slots("unknown_config") == []


def test_request_models_default_to_running_the_guard() -> None:
    from makermodslab.record import RecordingRequest
    from makermodslab.rollout import InferenceRequest
    from makermodslab.teleoperate import TeleoperateRequest

    assert _teleop_request().skip_identity_check is False
    assert (
        RecordingRequest(
            leader_port="/dev/l",
            follower_port="/dev/f",
            leader_config="L",
            follower_config="F",
            dataset_repo_id="d",
            single_task="t",
        ).skip_identity_check
        is False
    )
    assert (
        InferenceRequest(
            follower_port="/dev/f", follower_config="F", policy_ref="user/repo@root"
        ).skip_identity_check
        is False
    )
    assert isinstance(TeleoperateRequest.model_fields["skip_identity_check"].default, bool)
