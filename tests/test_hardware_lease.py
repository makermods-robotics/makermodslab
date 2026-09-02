from __future__ import annotations

import itertools
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from makermodslab.hardware_lease import (
    HardwareLeaseHeld,
    HardwareLeaseRegistry,
    HardwareLeaseToken,
    HardwareLeaseTokenError,
    HardwareRecoveryIdentity,
    safe_hardware_receipt,
)

FEATURE_KINDS = (
    "teleoperation",
    "recording",
    "replay",
    "inference",
    "calibration",
    "auto_calibration",
    "zero_calibration",
    "wiggle",
    "diagnostic",
    "remote_teleoperation",
)


def recovery_identity(
    target: str = "/dev/mock-follower",
    *,
    kind: str = "so101_recovery",
    family: str = "so101",
) -> HardwareRecoveryIdentity:
    return HardwareRecoveryIdentity.from_targets(kind, family, target)


def test_claim_is_atomic_across_competing_features() -> None:
    registry = HardwareLeaseRegistry()
    feature_kinds = list(FEATURE_KINDS)
    barrier = threading.Barrier(len(feature_kinds))
    winners = []
    losers = []
    lock = threading.Lock()

    def compete(kind: str) -> None:
        barrier.wait()
        try:
            token = registry.claim(kind, f"{kind}-worker")
        except HardwareLeaseHeld as exc:
            with lock:
                losers.append(exc.snapshot)
        else:
            with lock:
                winners.append(token)

    threads = [threading.Thread(target=compete, args=(kind,)) for kind in feature_kinds]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(losers) == len(feature_kinds) - 1
    assert registry.snapshot().lease_id == winners[0].lease_id


@pytest.mark.parametrize("left,right", tuple(itertools.combinations(FEATURE_KINDS, 2)))
def test_every_feature_pair_opens_only_the_winner_and_releases_after_close(
    left: str,
    right: str,
) -> None:
    registry = HardwareLeaseRegistry()
    barrier = threading.Barrier(2)
    opened: list[tuple[str, HardwareLeaseToken]] = []
    rejected: list[str] = []
    lock = threading.Lock()

    def start(kind: str) -> None:
        barrier.wait()
        try:
            token = registry.claim(kind, f"{kind}-worker")
        except HardwareLeaseHeld:
            with lock:
                rejected.append(kind)
            return
        with lock:
            # This append represents the first hardware constructor call and
            # occurs only after the atomic authority claim.
            opened.append((kind, token))

    threads = [threading.Thread(target=start, args=(kind,)) for kind in (left, right)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(opened) == 1
    assert len(rejected) == 1
    winner, token = opened[0]
    stop = registry.request_stop(token, "test_stop")
    assert stop.accepted is True
    assert registry.snapshot().kind == winner
    with pytest.raises(HardwareLeaseHeld):
        registry.claim(rejected[0], "retry-before-device-close")

    registry.release(token, safe_hardware_receipt("mock device closed after STOP"))
    assert registry.snapshot().held is False


def test_every_arm_hardware_feature_references_the_central_registry() -> None:
    root = Path(__file__).resolve().parents[1] / "makermodslab"
    global_registry_modules = (
        "teleoperate.py",
        "record.py",
        "replay.py",
        "rollout.py",
        "calibrate.py",
        "auto_calibrate.py",
        "zero_calibrate.py",
        "wiggle.py",
        "identify.py",
        "maker_ports.py",
        "motor_power.py",
        "can_recovery.py",
        "so101_recovery.py",
    )
    for relative in global_registry_modules:
        source = (root / relative).read_text(encoding="utf-8")
        assert "hardware_lease_registry" in source, relative
        assert ".claim(" in source or ".begin_recovery(" in source, relative

    for relative in (
        "remote_teleop/robot_service.py",
        "remote_teleop/operator_service.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "HardwareLeaseRegistry" in source, relative
        assert "self.registry.claim" in source, relative


def test_recovery_can_adopt_only_an_unresolved_claim() -> None:
    registry = HardwareLeaseRegistry()
    recovery = recovery_identity()
    token = registry.claim("teleoperation", "lost-worker", recovery=recovery)

    with pytest.raises(HardwareLeaseHeld):
        registry.begin_recovery("technician", expected=recovery)

    registry.mark_unresolved(token, "worker disappeared before finalizer")
    with pytest.raises(HardwareLeaseHeld):
        registry.begin_recovery(
            "wrong-technician",
            expected=recovery_identity("/dev/different-follower"),
        )
    adopted = registry.begin_recovery("technician", expected=recovery)

    assert adopted == token
    assert registry.snapshot().state == "recovering"
    registry.release(adopted, safe_hardware_receipt("recovery disabled torque and closed bus"))
    assert registry.snapshot().held is False


def test_stop_requests_teardown_without_releasing() -> None:
    registry = HardwareLeaseRegistry()
    token = registry.claim("teleoperation", "local")

    first = registry.request_stop(token, "operator_stop")
    second = registry.request_stop(token, "duplicate")

    assert first.first_request is True
    assert second.first_request is False
    assert registry.snapshot().held is True
    assert registry.snapshot().state == "stopping"
    with pytest.raises(HardwareLeaseHeld):
        registry.claim("recording", "local")


def test_only_current_token_can_release_after_safe_close() -> None:
    registry = HardwareLeaseRegistry()
    token = registry.claim("replay", "worker")
    foreign = HardwareLeaseRegistry().claim("replay", "worker")

    with pytest.raises(HardwareLeaseTokenError):
        registry.release(foreign, safe_hardware_receipt("closed"))

    registry.release(token, safe_hardware_receipt("torque disabled and bus closed"))
    assert registry.snapshot().held is False
    assert registry.snapshot().state == "idle"


@pytest.mark.parametrize(
    "receipt",
    [
        {"safe": False, "device_closed": True, "torque_off": True, "evidence": "close failed"},
        {"safe": True, "device_closed": False, "torque_off": True, "evidence": "worker alive"},
        {"safe": True, "device_closed": True, "torque_off": None, "evidence": "unknown torque"},
    ],
)
def test_unconfirmed_release_retains_fault_lockout(receipt: dict) -> None:
    registry = HardwareLeaseRegistry()
    token = registry.claim("remote_teleoperation", "robot-session")

    registry.release(token, receipt)

    snapshot = registry.snapshot()
    assert snapshot.held is True
    assert snapshot.state == "unresolved"
    assert snapshot.unresolved_reason == receipt["evidence"]
    with pytest.raises(HardwareLeaseHeld):
        registry.claim("calibration", "local")


def test_read_only_close_can_mark_torque_not_applicable() -> None:
    registry = HardwareLeaseRegistry()
    token = registry.claim("identify", "diagnostic")

    registry.release(
        token,
        safe_hardware_receipt(
            "read-only buses closed",
            torque_off=None,
            torque_not_applicable=True,
        ),
    )

    assert registry.snapshot().held is False


def test_durable_latch_is_promoted_without_release_to_claim_gap() -> None:
    registry = HardwareLeaseRegistry()
    current = registry.claim("recording", "current-worker")

    installed = registry.install_unresolved_latch(
        kind="remote_recovery",
        owner="durable:robot-fault",
        reason="durable robot fault",
        receipt={
            "safe": False,
            "device_closed": True,
            "torque_off": None,
            "evidence": "durable journal",
        },
        recovery=recovery_identity(kind="remote_recovery"),
    )

    assert installed is None
    assert registry.snapshot().pending_unresolved is True
    with pytest.raises(HardwareLeaseHeld):
        registry.claim("teleoperation", "racer")

    registry.release(current, safe_hardware_receipt("recording closed safely"))

    snapshot = registry.snapshot()
    assert snapshot.state == "unresolved"
    assert snapshot.kind == "remote_recovery"
    assert snapshot.owner == "durable:robot-fault"
    assert snapshot.pending_unresolved is False
    with pytest.raises(HardwareLeaseHeld):
        registry.claim("teleoperation", "racer")


def test_active_claim_is_restored_as_unresolved_after_process_restart(tmp_path: Path) -> None:
    journal = tmp_path / "owner-private" / "hardware-lease.json"
    recovery = recovery_identity("/dev/private-follower")
    first_process = HardwareLeaseRegistry(journal_path=journal)

    first_process.claim("teleoperation", "private-owner", recovery=recovery)

    restarted = HardwareLeaseRegistry(journal_path=journal)
    snapshot = restarted.snapshot()
    assert snapshot.held is True
    assert snapshot.state == "unresolved"
    assert snapshot.kind == "so101_recovery"
    assert snapshot.recovery == recovery.public()
    assert snapshot.owner == "durable:hardware-fault"
    assert restarted.recovery_targets("so101_recovery") == ("/dev/private-follower",)
    assert "private-owner" not in journal.read_text(encoding="utf-8")
    assert "/dev/private-follower" not in journal.read_text(encoding="utf-8")
    target_map = journal.with_name("hardware-lease-targets.json")
    assert "/dev/private-follower" in target_map.read_text(encoding="utf-8")
    if os.name != "nt":
        assert stat.S_IMODE(target_map.stat().st_mode) == 0o600

    adopted = restarted.begin_recovery("technician", expected=recovery)
    restarted.release(adopted, safe_hardware_receipt("torque off and bus closed"))
    assert restarted.snapshot().held is False
    assert journal.exists() is False


def test_unsafe_release_persists_sanitized_receipt_and_safe_recovery_clears_it(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "hardware-lease.json"
    recovery = recovery_identity()
    registry = HardwareLeaseRegistry(journal_path=journal)
    token = registry.claim("remote_teleoperation", "credential:secret-id", recovery=recovery)

    registry.release(
        token,
        {
            "safe": False,
            "device_closed": True,
            "torque_off": None,
            "evidence": "/Users/private/calibration.json failed",
        },
    )

    serialized = journal.read_text(encoding="utf-8")
    assert '"state":"unresolved"' in serialized
    assert '"safe":false' in serialized
    assert "credential:secret-id" not in serialized
    assert "/Users/private" not in serialized

    adopted = registry.begin_recovery("technician", expected=recovery)
    registry.release(adopted, safe_hardware_receipt("verified off"))
    assert journal.exists() is False


def test_invalid_owner_permissions_fail_closed(tmp_path: Path) -> None:
    journal = tmp_path / "hardware-lease.json"
    journal.write_text("{}", encoding="utf-8")
    journal.chmod(0o644)

    registry = HardwareLeaseRegistry(journal_path=journal)

    snapshot = registry.snapshot()
    assert snapshot.state == "unresolved"
    assert snapshot.journal_error == "hardware safety journal must be owner-only"
    with pytest.raises(HardwareLeaseHeld):
        registry.claim("teleoperation", "must-not-open")


def test_restart_rejects_journal_inside_non_private_directory(tmp_path: Path) -> None:
    private_root = tmp_path / "hardware-safety"
    journal = private_root / "hardware-lease.json"
    registry = HardwareLeaseRegistry(journal_path=journal)
    registry.claim("teleoperation", "lost", recovery=recovery_identity())
    private_root.chmod(0o777)

    restarted = HardwareLeaseRegistry(journal_path=journal)

    assert restarted.snapshot().held is True
    assert restarted.snapshot().state == "unresolved"
    assert "owner" in (restarted.snapshot().journal_error or "")


def test_claim_and_clear_fsync_the_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("directory fsync is a POSIX durability boundary")
    real_fsync = os.fsync
    directory_syncs = 0

    def track_fsync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr("makermodslab.hardware_lease.os.fsync", track_fsync)
    registry = HardwareLeaseRegistry(journal_path=tmp_path / "hardware-lease.json")
    token = registry.claim("teleoperation", "worker", recovery=recovery_identity())
    syncs_after_claim = directory_syncs
    registry.release(token, safe_hardware_receipt("verified safe close"))

    assert syncs_after_claim >= 2  # target map and authoritative intent
    assert directory_syncs >= syncs_after_claim + 2  # target removal and intent removal


def test_claim_refuses_authority_when_active_intent_cannot_be_written(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    registry = HardwareLeaseRegistry(journal_path=blocked_parent / "hardware-lease.json")

    with pytest.raises(HardwareLeaseHeld):
        registry.claim(
            "teleoperation",
            "must-not-open",
            recovery=recovery_identity(),
        )

    assert registry.snapshot().held is True
    assert registry.snapshot().state == "unresolved"
    assert registry.snapshot().journal_error is not None


def test_bound_recovery_identity_never_publishes_paths_or_usb_serial() -> None:
    recovery = HardwareRecoveryIdentity.from_bound_targets(
        "so101_recovery",
        "so101",
        {"/dev/private-arm": "0483:5740:SERIAL-SECRET"},
    )

    serialized = json.dumps(recovery.public(), sort_keys=True)
    assert "/dev/private-arm" not in serialized
    assert "SERIAL-SECRET" not in serialized
    assert recovery.private_targets() == ("/dev/private-arm",)


def test_manual_exact_targets_repair_missing_private_map_before_retry(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "hardware-lease.json"
    recovery = recovery_identity("/dev/reentered-arm")
    first = HardwareLeaseRegistry(journal_path=journal)
    first.claim("teleoperation", "lost", recovery=recovery)
    target_map = journal.with_name("hardware-lease-targets.json")
    target_map.unlink()

    restarted = HardwareLeaseRegistry(journal_path=journal)
    assert restarted.recovery_targets("so101_recovery") == ()
    adopted = restarted.begin_recovery("technician", expected=recovery)
    restarted.mark_unresolved(adopted, "readback failed")

    restarted_again = HardwareLeaseRegistry(journal_path=journal)
    assert restarted_again.recovery_targets("so101_recovery") == ("/dev/reentered-arm",)


def test_retargeting_hardware_safety_symlink_cannot_redirect_writes(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX canonical-parent boundary")
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    first_root.mkdir(mode=0o700)
    second_root.mkdir(mode=0o700)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(first_root, target_is_directory=True)
    registry = HardwareLeaseRegistry(journal_path=linked_root / "private" / "hardware-lease.json")
    linked_root.unlink()
    linked_root.symlink_to(second_root, target_is_directory=True)

    registry.claim("teleoperation", "worker", recovery=recovery_identity())

    assert (first_root / "private" / "hardware-lease.json").exists()
    assert not (second_root / "private" / "hardware-lease.json").exists()


def test_posix_child_operations_are_relative_to_validated_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("dirfd child operations are a POSIX boundary")
    real_open = os.open
    real_replace = os.replace
    relative_opens = 0
    relative_replaces = 0

    def track_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal relative_opens
        if dir_fd is not None:
            relative_opens += 1
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def track_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal relative_replaces
        if src_dir_fd is not None and dst_dir_fd is not None:
            relative_replaces += 1
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr("makermodslab.hardware_lease.os.open", track_open)
    monkeypatch.setattr("makermodslab.hardware_lease.os.replace", track_replace)
    registry = HardwareLeaseRegistry(journal_path=tmp_path / "private" / "hardware-lease.json")
    token = registry.claim("teleoperation", "worker", recovery=recovery_identity())
    registry.release(token, safe_hardware_receipt("verified safe"))

    assert relative_opens >= 2
    assert relative_replaces == 2
