from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from makermodslab.remote_teleop.fault_journal import (
    FaultJournalError,
    HardwareFaultRecord,
    RemoteFaultJournal,
)


def _profile():
    return SimpleNamespace(
        rig_id="so101-test",
        rig_digest="a" * 64,
        limits_digest="b" * 64,
        device_identity_digest="e" * 64,
        joint_names=("joint_a.pos", "gripper.pos"),
        units=("degree", "percent"),
        follower_calibration=SimpleNamespace(calibration_id="follower-test", digest="c" * 64),
        leader_calibration=SimpleNamespace(calibration_id="leader-test", digest="d" * 64),
    )


def _record() -> HardwareFaultRecord:
    return HardwareFaultRecord.from_profile(
        _profile(),
        reason_code="shutdown_unconfirmed",
        fault_codes=("hardware_stop_unconfirmed", "torque_state_unknown"),
        hardware_stop_completed=False,
        device_closed=True,
        torque_off_confirmed=False,
    )


def test_fault_journal_round_trip_is_owner_private_and_path_free(tmp_path) -> None:
    store = RemoteFaultJournal(tmp_path / "private")
    expected = _record()
    store.save(expected)

    loaded = store.load()
    assert loaded == expected
    assert store.public()["fault_lockout"] is True
    payload = store.path.read_text(encoding="utf-8")
    assert str(tmp_path) not in payload
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600

    store.clear_after_recovery()
    assert store.load() is None


def test_fault_journal_refuses_permission_unsafe_file(tmp_path) -> None:
    store = RemoteFaultJournal(tmp_path)
    store.save(_record())
    if os.name == "nt":
        pytest.skip("POSIX permission check")
    store.path.chmod(0o644)
    with pytest.raises(FaultJournalError, match="owner-only"):
        store.load()


def test_fault_journal_rejects_safe_or_tampered_records(tmp_path) -> None:
    with pytest.raises(FaultJournalError, match="fully safe close"):
        HardwareFaultRecord.from_profile(
            _profile(),
            reason_code="shutdown_unconfirmed",
            fault_codes=("unexpected_safe_record",),
            hardware_stop_completed=True,
            device_closed=True,
            torque_off_confirmed=True,
        )

    store = RemoteFaultJournal(tmp_path)
    store.save(_record())
    body = json.loads(store.path.read_text(encoding="utf-8"))
    body["fault_codes"] = ["../../private/key.pem"]
    store.path.write_text(json.dumps(body), encoding="utf-8")
    if os.name != "nt":
        store.path.chmod(0o600)
    with pytest.raises(FaultJournalError, match="codes"):
        store.load()
