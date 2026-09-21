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

from makermodslab import maker_rest_pose, record


class _Arm:
    def __init__(self):
        self.position = 12.0
        self.sent = []

    def get_observation(self):
        return {"shoulder_pan.pos": self.position}

    def send_action(self, action):
        self.position = action["shoulder_pan.pos"]
        self.sent.append(self.position)


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(maker_rest_pose, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))


@pytest.mark.parametrize("arm_type,step", [("maker", 2.0), ("metal", 2.0)])
@pytest.mark.parametrize("arm_count", [1, 2])
def test_recording_realign_uses_family_rate(monkeypatch, clock, arm_type, step, arm_count):
    arms = [_Arm() for _ in range(arm_count)]
    target = {"shoulder_pan": 0.0}
    source = SimpleNamespace(split=lambda: [(arm, target) for arm in arms], for_device=lambda device: target)
    monkeypatch.setattr(record, "_LeaderTargetSource", lambda *args, **kwargs: source)

    assert record._realign_follower_to_leader(arms[0], object(), arm_type, None, None)

    for arm in arms:
        assert 12.0 - arm.sent[0] == pytest.approx(step)
        assert all(abs(b - a) <= step + 1e-9 for a, b in zip(arm.sent, arm.sent[1:], strict=False))
        assert arm.position <= maker_rest_pose.MAKER_RETURN_TOLERANCE_DEG
        # Easing ends with this call; it installs no cap on ordinary sends.
        arm.send_action({"shoulder_pan.pos": 60.0})
        assert arm.sent[-1] == 60.0


def test_stop_return_keeps_default_rate(clock):
    arm = _Arm()
    assert maker_rest_pose.return_maker_arms_to_rest([(arm, {"shoulder_pan": 0.0})]) == [(True, "")]
    assert 12.0 - arm.sent[0] == pytest.approx(1.0)
