from __future__ import annotations

import threading

import pytest

from makermodslab.remote_teleop.contracts import ActionSample, SessionSpec, encode_action
from makermodslab.remote_teleop.executor import JointLimit, RemoteExecutor
from makermodslab.remote_teleop.simulation import SimulatedFollower

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
JOINTS = ("joint_a", "joint_b")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, milliseconds: int) -> None:
        self.now += milliseconds * 1_000_000


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
        units=("rad", "rad"),
    )


def limits() -> dict[str, JointLimit]:
    return {
        joint: JointLimit(-1.0, 1.0, max_velocity_per_s=2.0, max_acceleration_per_s2=20.0) for joint in JOINTS
    }


def action(grant, clock: FakeClock, sequence: int, positions: tuple[float, float]) -> bytes:
    sample = ActionSample(
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
    return encode_action(sample, key_id=grant.key_id, key=grant.action_key)


def test_out_of_bounds_action_does_not_consume_sequence() -> None:
    clock = FakeClock()
    executor = RemoteExecutor(SimulatedFollower(JOINTS), limits(), clock_ns=clock)
    grant = executor.open_session(spec())

    with pytest.raises(ValueError, match="outside"):
        executor.submit_datagram(action(grant, clock, 9, (2.0, 0.0)))
    assert executor.status()["authority"]["highest_sequence"] is None

    executor.submit_datagram(action(grant, clock, 9, (0.5, 0.0)))
    assert executor.status()["authority"]["highest_sequence"] == 9


def test_first_action_deadline_stops_without_waiting_for_udp() -> None:
    clock = FakeClock()
    follower = SimulatedFollower(JOINTS)
    executor = RemoteExecutor(
        follower,
        limits(),
        clock_ns=clock,
        watchdog_ns=100_000_000,
        first_action_deadline_ns=500_000_000,
    )
    executor.open_session(spec())
    clock.advance_ms(499)
    assert executor.tick() is None
    assert follower.connected
    clock.advance_ms(1)
    assert executor.tick() is None
    assert follower.stop_reasons == ["first_action_watchdog"]
    assert executor.status()["safety"]["torque_off_confirmed"] is True


def _first_executed_position(*, first_action_delay_ms: int, tick_delay_ms: int) -> float:
    clock = FakeClock()
    executor = RemoteExecutor(
        SimulatedFollower(JOINTS),
        limits(),
        clock_ns=clock,
        watchdog_ns=200_000_000,
        first_action_deadline_ns=500_000_000,
    )
    grant = executor.open_session(spec())
    clock.advance_ms(first_action_delay_ms)
    executor.submit_datagram(action(grant, clock, 1, (1.0, 0.0)))
    clock.advance_ms(tick_delay_ms)
    executed = executor.tick()
    assert executed is not None
    assert executor.status()["authority"]["state"] == "active"
    return executed["joint_a"]


def test_delayed_first_action_cannot_expand_the_shaped_motion_step() -> None:
    normal = _first_executed_position(first_action_delay_ms=0, tick_delay_ms=20)
    near_deadline = _first_executed_position(first_action_delay_ms=499, tick_delay_ms=1)

    assert near_deadline == pytest.approx(normal)
    assert near_deadline == pytest.approx(0.008)


def _second_executed_position(*, second_tick_delay_ms: int) -> float:
    clock = FakeClock()
    executor = RemoteExecutor(
        SimulatedFollower(JOINTS),
        limits(),
        clock_ns=clock,
        watchdog_ns=200_000_000,
    )
    grant = executor.open_session(spec())
    executor.submit_datagram(action(grant, clock, 1, (1.0, 0.0)))
    clock.advance_ms(20)
    assert executor.tick() is not None
    clock.advance_ms(second_tick_delay_ms)
    executed = executor.tick()
    assert executed is not None
    assert executor.status()["authority"]["state"] == "active"
    return executed["joint_a"]


def test_near_watchdog_scheduler_stall_cannot_expand_the_shaped_motion_step() -> None:
    normal = _second_executed_position(second_tick_delay_ms=20)
    near_watchdog = _second_executed_position(second_tick_delay_ms=179)

    assert near_watchdog == pytest.approx(normal)
    assert near_watchdog == pytest.approx(0.024)


class BlockingFollower(SimulatedFollower):
    def __init__(self) -> None:
        super().__init__(JOINTS)
        self.execute_started = threading.Event()
        self.release_execute = threading.Event()

    def execute(self, positions):
        self.execute_started.set()
        assert self.release_execute.wait(2.0)
        self.positions = dict(positions)
        return dict(self.positions)

    def observe(self):
        return dict(self.positions)


def test_stop_revokes_dispatch_while_vendor_execute_is_blocked() -> None:
    clock = FakeClock()
    follower = BlockingFollower()
    executor = RemoteExecutor(follower, limits(), clock_ns=clock)
    grant = executor.open_session(spec())
    executor.submit_datagram(action(grant, clock, 1, (0.5, -0.5)))
    clock.advance_ms(20)
    tick = threading.Thread(target=executor.tick)
    tick.start()
    assert follower.execute_started.wait(0.2)

    stopped = executor.stop("control_lost")
    assert stopped["safety"]["software_dispatch_halted"] is True
    assert executor.status()["authority"]["state"] == "idle"
    assert tick.is_alive(), "the injected vendor call should still be blocked"

    follower.release_execute.set()
    tick.join(0.5)
    assert not tick.is_alive()
    assert executor.status()["counters"]["action_completed_after_stop"] == 1


class UnverifiedStopFollower(SimulatedFollower):
    def stop(self, reason: str):
        self.stop_reasons.append(reason)
        return {"disable_requested": True, "torque_off_confirmed": None}

    def close(self):
        self.connected = False
        return None


def test_unverified_torque_state_latches_fault_lockout() -> None:
    clock = FakeClock()
    executor = RemoteExecutor(UnverifiedStopFollower(JOINTS), limits(), clock_ns=clock)
    executor.open_session(spec())
    stopped = executor.stop("operator_stop")
    assert stopped["safety"]["disable_requested"] is True
    assert stopped["safety"]["torque_off_confirmed"] is None
    assert stopped["safety"]["fault_lockout"] is True
    with pytest.raises(RuntimeError, match="fault lockout"):
        executor.open_session(spec())
