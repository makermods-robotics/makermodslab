from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from makermodslab.remote_teleop.commissioning import (
    CommissioningError,
    CommissioningRecord,
    CommissioningStore,
    profile_commissioning_digest,
)


def _profile(**changes):
    values = {
        "rig_id": "so101-test",
        "rig_digest": "a" * 64,
        "limits_digest": "b" * 64,
        "device_identity_digest": "e" * 64,
        "joint_names": ("joint_a.pos", "gripper.pos"),
        "units": ("degree", "percent"),
        "follower_calibration": SimpleNamespace(calibration_id="follower-test", digest="c" * 64),
        "leader_calibration": SimpleNamespace(calibration_id="leader-test", digest="d" * 64),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_commissioning_record_is_private_path_free_and_profile_bound(tmp_path) -> None:
    profile = _profile()
    store = CommissioningStore(tmp_path / "private")
    record = CommissioningRecord.from_profile(profile)
    store.save(record)

    assert store.require(profile) == record
    assert store.public()["commissioned"] is True
    raw = store.path.read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    assert "private_key" not in raw
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_changed_calibration_schema_or_limits_invalidates_commissioning(tmp_path) -> None:
    profile = _profile()
    store = CommissioningStore(tmp_path)
    store.save(CommissioningRecord.from_profile(profile))

    changed = (
        _profile(rig_digest="f" * 64),
        _profile(limits_digest="f" * 64),
        _profile(joint_names=("other.pos", "gripper.pos")),
        _profile(leader_calibration=SimpleNamespace(calibration_id="leader-test", digest="f" * 64)),
    )
    for value in changed:
        assert profile_commissioning_digest(value) != profile_commissioning_digest(profile)
        with pytest.raises(CommissioningError, match="does not match"):
            store.require(value)


def test_missing_or_permission_unsafe_commissioning_fails_closed(tmp_path) -> None:
    store = CommissioningStore(tmp_path)
    with pytest.raises(CommissioningError, match="has not passed"):
        store.require(_profile())

    store.save(CommissioningRecord.from_profile(_profile()))
    if os.name != "nt":
        store.path.chmod(0o644)
        with pytest.raises(CommissioningError, match="owner-only"):
            store.load()


def test_robot_configuration_save_can_invalidate_commissioning(tmp_path) -> None:
    store = CommissioningStore(tmp_path)
    store.save(CommissioningRecord.from_profile(_profile()))

    store.invalidate()

    assert store.load() is None
