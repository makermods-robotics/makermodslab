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
"""Control-law and release-decision regressions; no hardware or background workers."""

import threading
from types import SimpleNamespace

import pytest

from makermodslab import teleoperate as teleop


@pytest.mark.parametrize("results", [None, [], [(False, "10.9 deg away")], [(True, ""), (False, "stalled")]])
def test_failed_or_incomplete_landing_blocks_release(results):
    family = SimpleNamespace(requires_verified_rest=True, return_to_rest=lambda *_: results)
    assert "release is blocked" in teleop._return_before_release(family, [(object(), {})], threading.Event())


def test_every_arm_must_confirm_return_before_release():
    family = SimpleNamespace(
        requires_verified_rest=True, return_to_rest=lambda *_: [(True, ""), (True, "settled")]
    )
    assert teleop._return_before_release(family, [(object(), {}), (object(), {})], threading.Event()) is None


def test_return_exception_blocks_release():
    def fail(*_):
        raise OSError("CAN disconnected")

    family = SimpleNamespace(requires_verified_rest=True, return_to_rest=fail)
    assert "CAN disconnected" in teleop._return_before_release(family, [(object(), {})], threading.Event())


@pytest.fixture
def pending_can_return(monkeypatch):
    event = threading.Event()
    worker = SimpleNamespace(is_alive=lambda: True, join=lambda **_: None)
    monkeypatch.setattr(teleop, "_requires_verified_rest", True)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", event)
    monkeypatch.setattr(teleop, "rest_failed", True)
    monkeypatch.setattr(teleop, "last_cleanup_error", "Return failed")
    return event


def test_repeated_stop_and_new_start_do_not_release_failed_return(pending_can_return):
    assert teleop.handle_stop_teleoperation()["rest_failed"]
    assert not teleop.finish_pending_release()
    assert not pending_can_return.is_set()


def test_shutdown_timeout_does_not_command_torque_off(pending_can_return):
    teleop.stop_and_wait(timeout=0)
    assert not pending_can_return.is_set()


def test_only_explicit_release_can_bypass_failed_landing(pending_can_return):
    teleop.handle_stop_teleoperation(release_now=True)
    assert pending_can_return.is_set()


def test_maker_return_rejects_partial_or_invalid_feedback():
    from makermodslab.maker_rest_pose import return_maker_to_pose

    for obs in ({"a.pos": 0.0}, {"a.pos": 0.0, "b.pos": float("nan")}):
        arm = SimpleNamespace(get_observation=lambda obs=obs: obs)
        ok, reason = return_maker_to_pose(arm, {"a": 0.0, "b": 0.0})
        assert not ok
        assert "feedback" in reason


@pytest.fixture
def guarded_arm(monkeypatch):
    from makermodslab import maker_teleop_safety as safety

    clock = [100.0]
    monkeypatch.setattr(safety.time, "time", lambda: clock[0])
    monkeypatch.setattr(safety.time, "monotonic", lambda: clock[0])
    positions = {"a.pos": 0.0, "b.pos": 0.0}
    bus = SimpleNamespace(last_feedback_time={"a": clock[0], "b": clock[0]})
    diagnostics = SimpleNamespace(bus=bus, faults={}, recent=[], snapshot=lambda: {})
    monkeypatch.setattr(safety, "MakerFaultDiagnostics", lambda _: diagnostics)
    sent = []

    def send(action):
        sent.append(dict(action))
        for key in action:
            bus.last_feedback_time[key[:-4]] = clock[0]
        return dict(action)

    arm = SimpleNamespace(bus=bus, get_observation=lambda: positions, send_action=send)
    guard = safety.MakerTeleopSafety(arm)
    return arm, guard, diagnostics, sent, clock


def test_guard_keeps_forwarding_full_targets_when_follower_cannot_keep_up(guarded_arm):
    arm, guard, _diagnostics, sent, clock = guarded_arm
    for _ in range(20):
        arm.send_action({"a.pos": 150, "b.pos": -150})
        clock[0] += 1 / 30
    assert len(sent) == 20
    assert all(action == {"a.pos": 150, "b.pos": -150} for action in sent)
    assert guard.saved_error is None


def test_guard_refuses_faulted_motion_but_keeps_healthy_joint_hold(guarded_arm):
    arm, guard, diagnostics, sent, clock = guarded_arm
    arm.send_action({"a.pos": 50, "b.pos": 50})
    diagnostics.faults["a"] = {"fault_word": 1 << 14}
    clock[0] += 1 / 30
    with pytest.raises(RuntimeError, match="firmware fault"):
        arm.send_action({"a.pos": 50, "b.pos": 50})
    assert len(sent) == 1
    guard.hold()
    assert sent[-1] == {"b.pos": 50.0}
    assert "CAN evidence" in guard.saved_error


def test_guard_refuses_stale_feedback_without_sending_a_new_goal(guarded_arm):
    arm, guard, _diagnostics, sent, clock = guarded_arm
    clock[0] += 1
    with pytest.raises(RuntimeError, match="older than 250 ms"):
        arm.send_action({"a.pos": 50})
    guard.hold()
    assert sent == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_targets_are_refused_without_writing(guarded_arm, bad):
    arm, _guard, _diagnostics, sent, _clock = guarded_arm
    with pytest.raises(RuntimeError, match="Non-finite"):
        arm.send_action({"a.pos": bad})
    assert sent == []


def test_fast_reversal_is_forwarded_immediately_without_extra_reads(guarded_arm):
    arm, _guard, _diagnostics, sent, _clock = guarded_arm

    def no_extra_read():
        raise AssertionError("Safety wrapper added an observation read to tracking")

    arm.get_observation = no_extra_read
    arm.send_action({"a.pos": 150, "b.pos": -150})
    arm.send_action({"a.pos": -150, "b.pos": 150})
    assert sent == [{"a.pos": 150, "b.pos": -150}, {"a.pos": -150, "b.pos": 150}]


def test_hold_retries_last_goals_after_feedback_delay_and_one_joint_failure(guarded_arm):
    arm, guard, _diagnostics, sent, clock = guarded_arm
    arm.send_action({"a.pos": 10, "b.pos": 20})
    clock[0] += 1
    original = guard.send

    def fail_one(action):
        if "a.pos" in action:
            raise ConnectionError("joint a missed its response")
        return original(action)

    guard.send = fail_one
    guard.hold()
    assert sent[-1] == {"b.pos": 20}


def test_partial_landing_failure_never_resumes_old_leader_goal(guarded_arm):
    arm, guard, _diagnostics, _sent, _clock = guarded_arm
    arm.send_action({"a.pos": 150, "b.pos": 150})

    def fail(action):
        raise ConnectionError("failure after first landing write")

    guard.send = fail
    with pytest.raises(ConnectionError):
        guard.recovery_drive().send_action({"a.pos": 5, "b.pos": 6})
    assert guard.previous == {"a.pos": 5, "b.pos": 6}


def test_recovery_still_rejects_fault_in_an_arm_joint(guarded_arm):
    arm, guard, diagnostics, sent, _clock = guarded_arm
    arm.send_action({"a.pos": 10, "b.pos": 20})
    diagnostics.faults["a"] = {"fault_word": 2}
    with pytest.raises(RuntimeError, match="firmware fault"):
        guard.recovery_drive().send_action({"a.pos": 0, "b.pos": 0})
    guard.hold()
    assert sent[-1] == {"b.pos": 20}
