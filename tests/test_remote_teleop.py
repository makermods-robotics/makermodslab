from __future__ import annotations

import json
import time

import pytest

from makermodslab.remote_teleop.authority import SessionAuthorityError
from makermodslab.remote_teleop.contracts import (
    MAX_DATAGRAM_BYTES,
    ActionContractError,
    ActionSample,
    SessionSpec,
    decode_action,
    encode_action,
)
from makermodslab.remote_teleop.executor import JointLimit, RemoteExecutor
from makermodslab.remote_teleop.service import RemoteSimulationService
from makermodslab.remote_teleop.simulation import InMemoryRecorder, SimulatedFollower
from makermodslab.remote_teleop.transport import FaultInjectingDatagramTransport

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
JOINTS = ("joint_a", "joint_b")
UNITS = ("rad", "rad")


class FakeClock:
    def __init__(self, now: int = 1_000_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, value: int) -> None:
        self.now += value * 1_000_000


class LiveTorqueFollower(SimulatedFollower):
    @property
    def child_status(self):
        return {
            "stop_receipt": {
                "hardware_stop_completed": True,
                "torque_off_confirmed": True,
            }
        }


def spec() -> SessionSpec:
    return SessionSpec(
        source_id="operator-1",
        rig_id="rig-1",
        rig_digest=DIGEST_A,
        leader_calibration_id="leader-cal-1",
        leader_calibration_digest=DIGEST_B,
        follower_calibration_id="follower-cal-1",
        follower_calibration_digest=DIGEST_C,
        joint_names=JOINTS,
        units=UNITS,
    )


def sample(grant, clock: FakeClock, sequence: int, positions=(0.8, -0.8)) -> ActionSample:
    return ActionSample(
        session_id=grant.session_id,
        source_id=grant.spec.source_id,
        executor_generation=grant.executor_generation,
        rig_id=grant.spec.rig_id,
        rig_digest=grant.spec.rig_digest,
        leader_calibration_id=grant.spec.leader_calibration_id,
        leader_calibration_digest=grant.spec.leader_calibration_digest,
        follower_calibration_id=grant.spec.follower_calibration_id,
        follower_calibration_digest=grant.spec.follower_calibration_digest,
        sequence=sequence,
        source_monotonic_ns=clock.now,
        expires_at_source_monotonic_ns=clock.now + 100_000_000,
        joint_names=grant.spec.joint_names,
        units=grant.spec.units,
        positions=positions,
    )


def executor(clock: FakeClock):
    follower = SimulatedFollower(JOINTS)
    recorder = InMemoryRecorder()
    limits = {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }
    return (
        RemoteExecutor(
            follower,
            limits,
            clock_ns=clock,
            watchdog_ns=200_000_000,
            recorder=recorder,
        ),
        follower,
        recorder,
    )


def encoded(grant, action: ActionSample) -> bytes:
    return encode_action(action, key_id=grant.key_id, key=grant.action_key)


def test_live_torque_status_is_truthful_before_and_after_first_action() -> None:
    clock = FakeClock()
    follower = LiveTorqueFollower(JOINTS)
    limits = {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }
    remote = RemoteExecutor(follower, limits, clock_ns=clock, mode="live")

    grant = remote.open_session(spec())
    assert remote.status()["safety"]["torque_off_confirmed"] is True

    remote.submit_datagram(encoded(grant, sample(grant, clock, 1)))
    clock.advance_ms(20)
    assert remote.tick() is not None
    assert remote.status()["safety"]["torque_off_confirmed"] is False


def test_authenticated_action_round_trip_and_tamper_rejection() -> None:
    clock = FakeClock()
    remote, _follower, _recorder = executor(clock)
    grant = remote.open_session(spec())
    action = sample(grant, clock, 1)
    raw = encoded(grant, action)
    assert decode_action(raw, key_lookup=lambda key_id: grant.action_key).sequence == 1

    body = json.loads(raw)
    body["positions"][0] = 0.9
    tampered = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ActionContractError, match="authentication failed"):
        decode_action(tampered, key_lookup=lambda key_id: grant.action_key)

    with pytest.raises(ActionContractError, match="oversized"):
        decode_action(b"x" * (MAX_DATAGRAM_BYTES + 1), key_lookup=lambda _key_id: None)


def test_future_timestamp_and_maximum_staleness_boundary_are_explicit() -> None:
    clock = FakeClock()
    remote, _follower, _recorder = executor(clock)
    grant = remote.open_session(spec())

    future = sample(grant, clock, 1)
    future = ActionSample(
        **{
            **future.__dict__,
            "source_monotonic_ns": clock.now + 1_000_000,
            "expires_at_source_monotonic_ns": clock.now + 101_000_000,
        }
    )
    with pytest.raises(SessionAuthorityError, match="future_timestamp"):
        remote.submit_datagram(encoded(grant, future))

    at_boundary = sample(grant, clock, 1)
    at_boundary = ActionSample(
        **{
            **at_boundary.__dict__,
            "source_monotonic_ns": clock.now - 150_000_000,
            "expires_at_source_monotonic_ns": clock.now + 100_000_000,
        }
    )
    remote.submit_datagram(encoded(grant, at_boundary))

    too_old = sample(grant, clock, 2)
    too_old = ActionSample(
        **{
            **too_old.__dict__,
            "source_monotonic_ns": clock.now - 150_000_001,
            "expires_at_source_monotonic_ns": clock.now + 99_999_999,
        }
    )
    with pytest.raises(SessionAuthorityError, match="stale"):
        remote.submit_datagram(encoded(grant, too_old))
    assert remote.status()["authority"]["highest_sequence"] == 1


def test_fixed_rate_execution_records_source_executed_observation_and_latency() -> None:
    clock = FakeClock()
    remote, follower, recorder = executor(clock)
    grant = remote.open_session(spec())
    remote.submit_datagram(encoded(grant, sample(grant, clock, 1)))
    clock.advance_ms(20)
    executed = remote.tick()
    assert executed is not None
    assert 0 < executed["joint_a"] < 0.8
    assert -0.8 < executed["joint_b"] < 0
    assert follower.connected

    event = next(event for event in recorder.events if event["event"] == "action.executed")
    assert event["source_positions"] == [0.8, -0.8]
    assert event["executed_positions"] == list(executed.values())
    assert event["observation_positions"] == list(executed.values())
    assert event["network_latency_ns"] == 0
    assert event["command_age_at_execution_ns"] == 20_000_000


def test_duplicate_reorder_stale_and_disconnect_fail_safe() -> None:
    clock = FakeClock()
    remote, follower, _recorder = executor(clock)
    grant = remote.open_session(spec())
    first = encoded(grant, sample(grant, clock, 1))
    remote.submit_datagram(first)
    with pytest.raises(SessionAuthorityError, match="sequence"):
        remote.submit_datagram(first)

    clock.advance_ms(101)
    stale = sample(grant, clock, 2)
    stale = ActionSample(
        **{
            **stale.__dict__,
            "source_monotonic_ns": clock.now - 200_000_000,
            "expires_at_source_monotonic_ns": clock.now - 100_000_000,
        }
    )
    with pytest.raises(SessionAuthorityError, match="stale|expired"):
        remote.submit_datagram(encoded(grant, stale))

    clock.advance_ms(100)
    assert remote.tick() is None
    assert follower.stop_reasons == ["stale_command_watchdog"]
    assert not follower.connected


def test_reconnect_mints_new_session_and_never_accepts_stale_session() -> None:
    clock = FakeClock()
    remote, _follower, _recorder = executor(clock)
    first = remote.open_session(spec())
    old = encoded(first, sample(first, clock, 1))
    remote.stop("disconnect")
    second = remote.open_session(spec())
    assert second.session_id != first.session_id
    assert second.executor_generation > first.executor_generation
    with pytest.raises(ActionContractError, match="unknown action key"):
        remote.submit_datagram(old)


def test_fault_transport_loss_reorder_and_duplicate_never_roll_back_high_water() -> None:
    clock = FakeClock()
    remote, _follower, _recorder = executor(clock)
    grant = remote.open_session(spec())
    rejected: list[str] = []

    def receive(payload: bytes) -> None:
        try:
            remote.submit_datagram(payload)
        except Exception as exc:
            rejected.append(str(exc))

    transport = FaultInjectingDatagramTransport(
        receive,
        drop_numbers={2},
        duplicate_numbers={3},
        delay_numbers={1},
    )
    for sequence in (1, 2, 3):
        transport.send(encoded(grant, sample(grant, clock, sequence)))
        clock.advance_ms(10)
    transport.flush_reordered()
    status = remote.status()
    assert status["authority"]["highest_sequence"] == 3
    assert len(rejected) == 2  # duplicate 3 and delayed/reordered 1


def test_explicit_stop_is_idempotent_and_old_stop_cannot_target_reconnect() -> None:
    clock = FakeClock()
    remote, follower, _recorder = executor(clock)
    first = remote.open_session(spec())
    transition = remote.stop("operator_stop")
    duplicate = remote.stop("operator_stop")
    assert transition["duplicate"] is False
    assert duplicate["duplicate"] is True
    assert follower.stop_reasons == ["operator_stop"]
    second = remote.open_session(spec())
    assert second.session_id != first.session_id


def test_follower_write_failure_stops_locally() -> None:
    clock = FakeClock()
    remote, follower, _recorder = executor(clock)
    grant = remote.open_session(spec())
    remote.submit_datagram(encoded(grant, sample(grant, clock, 1)))
    follower.fail_next_execute = True
    clock.advance_ms(20)
    with pytest.raises(OSError, match="injected"):
        remote.tick()
    assert remote.status()["authority"]["state"] == "idle"
    assert follower.stop_reasons == ["follower_io:OSError"]


def test_simulation_restart_after_watchdog_uses_a_new_executor_thread() -> None:
    service = RemoteSimulationService()
    limits = {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }
    try:
        first = service.start(spec(), limits, watchdog_ms=20)
        deadline = time.monotonic() + 1.0
        while service.status()["state"] != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.status()["state"] == "idle"

        second = service.start(spec(), limits, watchdog_ms=2000)
        assert second["session"]["session_id"] != first["session"]["session_id"]
        assert service.status()["state"] == "active"
        service.stop(second["session"]["session_id"], "test_complete")
    finally:
        service.shutdown()
