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

import os
from collections.abc import Callable
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


class FakeCalibrationUI:
    """A CalibrationUI that never blocks: records each step and message so a
    family's ``calibrate`` can run to completion on the test thread."""

    def __init__(self, log: list | None = None) -> None:
        self.steps: list[tuple[str, str | None, bool]] = []
        self.messages: list[str] = []
        # A shared log lets a test pin the ORDER of ui calls against bus calls.
        self.log = log if log is not None else []

    def step(self, text: str, *, image_url: str | None = None, live_positions: bool = False) -> None:
        self.steps.append((text, image_url, live_positions))
        self.log.append(("step", text))

    def message(self, text: str) -> None:
        self.messages.append(text)
        self.log.append(("message", text))


class FakeCanBus:
    """A CAN follower's bus (RobStride or Damiao alike): one broadcast zero."""

    def __init__(self, log: list | None = None) -> None:
        self.log = log if log is not None else []

    def connect(self, handshake: bool = True) -> None:
        self.log.append(("bus", "connect", handshake))

    def read(self, register: str, motor: str) -> float:
        assert register == "Present_Position"
        return 1.0

    def disable_torque(self) -> None:
        self.log.append(("bus", "disable_torque"))

    def set_zero_position(self) -> None:
        self.log.append(("bus", "set_zero_position"))

    def write_calibration(self, calibration: dict) -> None:
        self.log.append(("bus", "write_calibration"))


class FakeCanFollower:
    """A Maker/Metal follower with a REAL config (ids, joint_limits) and a fake
    bus; carries the multi-turn bookkeeping a zero must reset."""

    def __init__(self, config: Any, log: list | None = None) -> None:
        self.config = config
        self.log = log if log is not None else []
        self.bus = FakeCanBus(self.log)
        self._joint_motor_names = list(config.motor_can_ids)
        self._turn_offset = dict.fromkeys(self._joint_motor_names, 5.0)
        self._stale_zero = {"shoulder_pan": True}
        self._last_positions = {"shoulder_pan": 33.0}
        self.calibration: dict | None = None

    def _read_raw_positions(self) -> dict[str, float]:
        return dict.fromkeys(self._joint_motor_names, 1.0)

    def connect(self, calibrate: bool = True) -> None:
        self.log.append(("device", "connect", calibrate))

    def disconnect(self) -> None:
        self.log.append(("device", "disconnect"))


class FakeUartBus:
    """The Star Arm 102 leader's FashionStar bus: unlock + origin per servo."""

    def __init__(self, log: list | None = None) -> None:
        self.log = log if log is not None else []

    def read_raw_angle(self, motor_id: int) -> float:
        return 1.0

    def unlock(self, motor_id: int) -> None:
        self.log.append(("bus", "unlock", motor_id))

    def set_origin_point(self, motor_id: int) -> None:
        self.log.append(("bus", "set_origin_point", motor_id))


class FakeStarLeader:
    """A Star Arm 102 leader with a REAL preset config (joint_ids, joint_ranges)."""

    def __init__(self, config: Any, log: list | None = None) -> None:
        self.config = config
        self.log = log if log is not None else []
        self.bus = FakeUartBus(self.log)
        self.calibration: dict | None = None

    def connect(self, calibrate: bool = True) -> None:
        self.log.append(("device", "connect", calibrate))

    def disconnect(self) -> None:
        self.log.append(("device", "disconnect"))


def make_arm_family(
    family_id: str,
    *,
    joints_per_arm: int = 9,
    uses_feetech_bus: bool = False,
    calibration_kind: str = "steps",
    follower_probe_protocol: str | None = None,
    telemetry_kind: str = "degrees",
    leader_dir: str | None = None,
    follower_dir: str | None = None,
    dirs: bool = True,
    calibration_name_suffix: str | None = None,
    calibration_panel_url: str | None = None,
    image_url: str | None = None,
    step_methods: bool = True,
    calibrate: Callable[..., Any] | None = None,
    open_for_calibration: Callable[..., Any] | None = None,
    preflight_ports: Callable[..., Any] | None = None,
    release_torque: Callable[..., Any] | None = None,
):
    """A complete, registrable ArmFamily the core has never shipped.

    Every REQUIRED_ATTRIBUTE is set and every abstract method exists (each
    one refuses loudly if a test ever reaches it — nothing here opens a bus).
    Tests register the instance into a copied registry dict to prove the
    seams read the registry LIVE rather than a table captured at import, which
    is the property the extension system (docs/extensions/plan.md, TB6)
    depends on.

    Calibration libraries: by default the fake owns its OWN pair of dirs,
    ``<CALIBRATION_BASE_PATH_TELEOP>/<id>_leader`` and
    ``<CALIBRATION_BASE_PATH_ROBOTS>/<id>_follower`` (resolved at call time,
    so the ``tmp_lerobot_home`` redirect is honoured) — the same rule
    ``utils.config.lerobot_calibration_dir`` encodes for an extension — so
    registering one beside the built-ins never collides with them.
    ``leader_dir`` / ``follower_dir`` pin a fixed path instead (to stage a
    deliberate collision); ``dirs=False`` overrides NEITHER dir method and
    sets neither library attr, so the base contract's NotImplementedError is
    what the registry meets.

    Calibration kind: ``steps`` (the default) installs ``calibrate`` /
    ``open_for_calibration`` overrides — the hooks passed in, or stubs that
    refuse — because the registry requires both on a steps family;
    ``step_methods=False`` leaves them off to build the half-declared family
    the registry must refuse. ``panel`` needs ``calibration_panel_url``;
    ``range_sweep`` needs ``uses_feetech_bus=True``.

    ``preflight_ports`` / ``release_torque`` hooks, when given, replace the
    refusing stubs so a test can record what the core asked the family for.
    Hooks are called WITHOUT the family instance (plain functions).
    """
    from makermodslab.arms import ArmFamily

    class _Fake(ArmFamily):
        id = family_id
        label = f"{family_id.title()} Arm"
        short_label = family_id.title()
        indefinite_label = f"a {family_id.title()} arm"
        supports_bimanual = True
        supports_auto_calibration = calibration_kind == "range_sweep"
        supports_dagger = False
        single_robot_type = f"{family_id}_follower"
        bimanual_robot_type = f"bi_{family_id}_follower"
        robot_type_markers = (family_id,)
        motion_identify_energizes_follower = False

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
            # In the class body (not assigned after creation): it is abstract
            # on the base, and ABC's abstract set is fixed at class creation.
            if release_torque is None:
                raise AssertionError("a fake family must never touch a robot")
            return release_torque(device, label)

        def build_single_configs(self, request, cameras, leader_id, follower_id):
            raise AssertionError("a fake family must never build a device config")

        def build_bimanual_configs(self, request, cameras, base, leader_staging, follower_staging):
            raise AssertionError("a fake family must never build a device config")

    _Fake.joints_per_arm = joints_per_arm
    _Fake.uses_feetech_bus = uses_feetech_bus
    _Fake.calibration_kind = calibration_kind
    _Fake.follower_probe_protocol = follower_probe_protocol
    _Fake.telemetry_kind = telemetry_kind
    _Fake.calibration_panel_url = calibration_panel_url
    _Fake.image_url = image_url

    if dirs:

        def _leader_dir(self) -> str:
            if leader_dir is not None:
                return leader_dir
            from makermodslab.utils import config as cfg

            return os.path.join(cfg.CALIBRATION_BASE_PATH_TELEOP, f"{family_id}_leader")

        def _follower_dir(self) -> str:
            if follower_dir is not None:
                return follower_dir
            from makermodslab.utils import config as cfg

            return os.path.join(cfg.CALIBRATION_BASE_PATH_ROBOTS, f"{family_id}_follower")

        _Fake.leader_calibration_dir = _leader_dir
        _Fake.follower_calibration_dir = _follower_dir

    if calibration_name_suffix is not None:
        _Fake.calibration_name_suffix = property(lambda self: calibration_name_suffix)

    def _refusing(name: str):
        def method(self, *args, **kwargs):
            raise AssertionError(f"a fake family must never reach {name}")

        return method

    def _hook(fn):
        return lambda self, *args, **kwargs: fn(*args, **kwargs)

    if preflight_ports is not None:
        _Fake.preflight_ports = _hook(preflight_ports)
    if step_methods and (calibration_kind == "steps" or calibrate or open_for_calibration):
        _Fake.calibrate = _hook(calibrate) if calibrate else _refusing("calibrate")
        _Fake.open_for_calibration = (
            _hook(open_for_calibration) if open_for_calibration else _refusing("open_for_calibration")
        )
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
