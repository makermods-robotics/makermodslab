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
from types import SimpleNamespace

import pytest

from makermodslab.follower_speed_cap import MAX_STEP_INTERVAL_S, FollowerSpeedCap


class Arm:
    def __init__(self, clip=None):
        self.sent = []
        self.clip = clip

    def send_action(self, action):
        self.sent.append(dict(action))
        if self.clip is None:
            return dict(action)
        return {k: min(v, self.clip) if k.endswith(".pos") else v for k, v in action.items()}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_first_command_passes_then_rate_is_bounded_per_joint():
    arm, clock = Arm(), Clock()
    FollowerSpeedCap(arm, 360.0, clock=clock)
    arm.send_action({"elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0})
    clock.t += 0.01  # 3.6 deg allowed
    arm.send_action({"elbow_flex.pos": 60.0, "wrist_flex.pos": -1.0})
    assert arm.sent[-1] == {"elbow_flex.pos": pytest.approx(3.6), "wrist_flex.pos": -1.0}
    clock.t += 0.01
    arm.send_action({"elbow_flex.pos": 60.0, "wrist_flex.pos": -1.0})
    assert arm.sent[-1]["elbow_flex.pos"] == pytest.approx(7.2)


def test_a_stalled_loop_cannot_bank_a_large_step():
    arm, clock = Arm(), Clock()
    FollowerSpeedCap(arm, 360.0, clock=clock)
    arm.send_action({"elbow_flex.pos": 0.0})
    clock.t += 5.0
    arm.send_action({"elbow_flex.pos": 90.0})
    assert arm.sent[-1]["elbow_flex.pos"] == pytest.approx(360.0 * MAX_STEP_INTERVAL_S)


def test_reference_is_what_the_driver_accepted_and_other_keys_pass_through():
    arm, clock = Arm(clip=10.0), Clock()
    FollowerSpeedCap(arm, 1000.0, clock=clock)
    arm.send_action({"elbow_flex.pos": 50.0, "gripper.vel": 5.0})  # driver clips to 10
    clock.t += 0.01
    arm.send_action({"elbow_flex.pos": -50.0, "gripper.vel": 7.0})
    assert arm.sent[-1] == {"elbow_flex.pos": pytest.approx(0.0, abs=1e-6), "gripper.vel": 7.0}


def _run(cap_arm, clock, goal, ticks, dt=0.01):
    positions = []
    for _ in range(ticks):
        clock.t += dt
        cap_arm.send_action({"shoulder_pan.pos": goal})
        positions.append(cap_arm.sent[-1]["shoulder_pan.pos"])
    return positions


def test_acceleration_ramps_the_target_up_and_brakes_onto_the_goal_without_overshoot():
    arm, clock = Arm(), Clock()
    FollowerSpeedCap(arm, 280.0, 800.0, clock=clock)
    arm.send_action({"shoulder_pan.pos": 0.0})
    positions = _run(arm, clock, 90.0, 300)
    speeds = [(b - a) / 0.01 for a, b in zip([0.0, *positions], positions, strict=False)]
    # Speeds up by at most 800 deg/s^2 (8 deg/s per 10 ms tick) and never past 280 deg/s.
    assert speeds[0] == pytest.approx(8.0)
    assert all(b - a <= 8.0 + 1e-6 for a, b in zip(speeds, speeds[1:], strict=False))
    assert max(speeds) <= 280.0 + 1e-6
    # Slows down before the goal: lands on it and never passes it.
    assert max(positions) <= 90.0 + 1e-9
    assert positions[-1] == pytest.approx(90.0)
    assert all(b - a >= -8.0 - 1e-6 for a, b in zip(speeds, speeds[1:], strict=False) if b > 1.0)


def test_a_leader_that_stops_dead_is_approached_gently_not_slammed():
    arm, clock = Arm(), Clock()
    FollowerSpeedCap(arm, 280.0, 800.0, clock=clock)
    arm.send_action({"shoulder_pan.pos": 0.0})
    _run(arm, clock, 200.0, 40)  # moving fast toward a far goal
    reached = arm.sent[-1]["shoulder_pan.pos"]
    positions = _run(arm, clock, reached + 5.0, 100)  # the leader stops just ahead
    assert max(positions) <= reached + 5.0 + 1e-9
    assert positions[-1] == pytest.approx(reached + 5.0)


def test_exempt_joints_are_forwarded_untouched():
    arm, clock = Arm(), Clock()
    FollowerSpeedCap(arm, 280.0, 800.0, clock=clock, exempt=("gripper",))
    arm.send_action({"elbow_flex.pos": 0.0, "gripper.pos": 0.0})
    clock.t += 0.01
    arm.send_action({"elbow_flex.pos": 60.0, "gripper.pos": 90.0})
    assert arm.sent[-1]["gripper.pos"] == 90.0
    assert arm.sent[-1]["elbow_flex.pos"] < 60.0


def test_invalid_cap_is_refused():
    with pytest.raises(ValueError):
        FollowerSpeedCap(Arm(), 0.0)
    with pytest.raises(ValueError):
        FollowerSpeedCap(Arm(), 280.0, 0.0)


def test_only_the_metal_family_caps_leader_following():
    from makermodslab.arms import registry

    single = SimpleNamespace(send_action=lambda action: action)
    registry.get("metal").prepare_leader_following(single)
    assert single.follower_speed_cap.max_deg_s == registry.get("metal").follower_max_speed_deg_s == 280.0
    assert single.follower_speed_cap.max_deg_s2 == registry.get("metal").follower_max_accel_deg_s2
    assert single.follower_speed_cap._exempt == {"gripper.pos"}

    left, right = SimpleNamespace(send_action=lambda a: a), SimpleNamespace(send_action=lambda a: a)
    registry.get("metal").prepare_leader_following(SimpleNamespace(left_arm=left, right_arm=right))
    assert hasattr(left, "follower_speed_cap") and hasattr(right, "follower_speed_cap")

    for family in ("maker", "so101"):
        untouched = SimpleNamespace(send_action=lambda action: action)
        registry.get(family).prepare_leader_following(untouched)
        assert not hasattr(untouched, "follower_speed_cap")
