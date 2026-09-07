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
"""Hardware-side fakes used across the test suite."""

from __future__ import annotations

from typing import Any


class FakeRobot:
    """Stand-in for a connected SO-101 follower.

    Records every method call on `self.calls` so tests can assert on it.
    Methods are deliberately the minimum surface MakerMods Lab actually invokes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._connected = False
        self.init_args = args
        self.init_kwargs = kwargs

    def connect(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("connect", args, kwargs))
        self._connected = True

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("disconnect", args, kwargs))
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> dict[str, float]:
        self.calls.append(("get_observation", (), {}))
        return {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append(("send_action", (action,), {}))
        return action


class FakeTeleoperator(FakeRobot):
    """Stand-in for a connected SO-101 leader. Same surface as FakeRobot."""

    def get_action(self) -> dict[str, float]:
        self.calls.append(("get_action", (), {}))
        return {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0}


def patch_so101_configs(monkeypatch) -> None:
    """Swap SO101 config constructors so any code path constructing them
    gets a no-op factory that returns a plain dataclass-like stub.

    Use sparingly — most tests prefer to patch the higher-level entry points
    (e.g. `lerobot.record.record`) instead of the config classes.
    """

    class _StubConfig:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    monkeypatch.setattr("lerobot.robots.so101_follower.SO101FollowerConfig", _StubConfig, raising=False)
    monkeypatch.setattr("lerobot.teleoperators.so101_leader.SO101LeaderConfig", _StubConfig, raising=False)


def make_arm_family(
    family_id: str,
    *,
    joints_per_arm: int = 9,
    uses_feetech_bus: bool = False,
    uses_zero_calibration: bool = True,
    follower_probe_protocol: str | None = None,
    telemetry_kind: str = "degrees",
):
    """A complete, registrable ArmFamily the core has never shipped.

    Every REQUIRED_ATTRIBUTE is set and every abstract method exists (each
    one refuses loudly if a test ever reaches it — nothing here opens a bus).
    Tests register the instance into a copied registry dict to prove the
    seams read the registry LIVE rather than a table captured at import, which
    is the property the extension system (docs/extensions/plan.md, TB6)
    depends on.
    """
    from makermodslab.arms import ArmFamily

    class _Fake(ArmFamily):
        id = family_id
        label = f"{family_id.title()} Arm"
        short_label = family_id.title()
        indefinite_label = f"a {family_id.title()} arm"
        supports_bimanual = True
        supports_auto_calibration = not uses_zero_calibration
        supports_dagger = False
        single_robot_type = f"{family_id}_follower"
        bimanual_robot_type = f"bi_{family_id}_follower"
        robot_type_markers = (family_id,)
        leader_library_attr = "MAKER_LEADER_CONFIG_PATH"
        follower_library_attr = "MAKER_FOLLOWER_CONFIG_PATH"
        motion_identify_energizes_follower = False

        def zero_pose_instructions(self, device_type: object | None = None) -> str:
            if not self.uses_zero_calibration:
                return ""
            side = "leader" if device_type == "teleop" else "follower"
            return f"Move the {family_id} {side} to its ZERO POSE — then confirm."

        def single_follower_config(self, port, config_id):
            raise AssertionError("a fake family must never build a device config")

        def single_leader_config(self, port, config_id):
            raise AssertionError("a fake family must never build a device config")

        async def identify_by_motion(self, device_type, ports=None):
            raise AssertionError("a fake family must never touch a port")

        def capture_rest_poses(self, robot, *, include_gripper=False):
            raise AssertionError("a fake family must never touch a robot")

        def return_to_rest(self, rest_poses, abort_event=None):
            raise AssertionError("a fake family must never touch a robot")

        def release_torque(self, device, label="device"):
            raise AssertionError("a fake family must never touch a robot")

        def build_single_configs(self, request, cameras, leader_id, follower_id):
            raise AssertionError("a fake family must never build a device config")

        def build_bimanual_configs(self, request, cameras, base, leader_staging, follower_staging):
            raise AssertionError("a fake family must never build a device config")

    _Fake.joints_per_arm = joints_per_arm
    _Fake.uses_feetech_bus = uses_feetech_bus
    _Fake.uses_zero_calibration = uses_zero_calibration
    _Fake.follower_probe_protocol = follower_probe_protocol
    _Fake.telemetry_kind = telemetry_kind
    return _Fake()


def scratch_registry(monkeypatch) -> None:
    """Give the test its own copy of every table in the arm registry.

    Copies EVERY module-level dict (the family table and whatever bookkeeping
    sits beside it, provenance included) so a family registered inside the
    test vanishes with the monkeypatch — a leftover entry would make the
    next test's duplicate-id refusal fire for the wrong reason.
    """
    from makermodslab.arms import registry

    for name, value in list(vars(registry).items()):
        if isinstance(value, dict) and not name.startswith("__"):
            monkeypatch.setattr(registry, name, dict(value))
