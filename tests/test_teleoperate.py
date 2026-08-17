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
"""Tests for makermodslab.teleoperate — request schema and status handlers."""

from __future__ import annotations

import inspect
import threading
import time

import pytest

from lerobot.motors import Motor, MotorNormMode


def test_teleoperate_request_rejects_missing_fields() -> None:
    from pydantic import ValidationError

    from makermodslab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest()


def test_teleoperate_request_defaults_to_single_arm() -> None:
    """A single-arm request omits the bimanual fields; they default safely."""
    from makermodslab.teleoperate import TeleoperateRequest

    req = TeleoperateRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="L",
        follower_config="F",
    )
    assert req.mode == "single"
    assert req.right_leader_port == ""
    assert req.right_follower_config == ""


def test_get_joint_positions_from_robot_uses_provided_object() -> None:
    from makermodslab.teleoperate import get_joint_positions_from_robot
    from tests.mocks import FakeRobot

    robot = FakeRobot()
    robot.connect()
    positions = get_joint_positions_from_robot(robot)
    assert isinstance(positions, dict)


def test_start_teleoperation_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that fails to connect must make the start handler return
    success=False (so the UI surfaces the error and doesn't navigate to an
    empty teleop screen) and reset state so a retry isn't blocked. Previously
    the connect ran in a worker thread and the handler always claimed success.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("serial port unavailable")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The message must name the arm that failed (the follower connects first).
    assert "follower" in result["message"].lower()
    assert "COM_FOLLOWER" in result["message"]
    # State must be reset so the next attempt isn't blocked by the mutex.
    assert teleop.teleoperation_active is False


def test_start_teleoperation_disconnects_follower_when_leader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-connect path: if the follower connects but the leader then
    fails, the follower must be disconnected so its serial port is released.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    class _OkBus:
        def connect(self) -> None:
            pass

    class _FailingBus:
        def connect(self) -> None:
            raise RuntimeError("leader offline")

    class _Follower:
        def __init__(self, config) -> None:
            self.bus = _OkBus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class _Leader:
        def __init__(self, config) -> None:
            self.bus = _FailingBus()
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "SO101Follower", lambda config: created.setdefault("follower", _Follower(config))
    )
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: created.setdefault("leader", _Leader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    assert "leader" in result["message"].lower()
    # The already-connected follower must have been cleaned up.
    assert created["follower"].disconnected is True
    assert teleop.teleoperation_active is False


def test_start_teleoperation_force_disables_torque_and_warns_when_setup_fails_after_configure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """robot.configure() is what actually writes Torque_Enable=1 on the real
    follower servos. If a LATER setup step (e.g. the leader's configure())
    raises, the follower is left physically energized — the except block must
    run the same force_disable_torque + _safe_disconnect cleanup as the
    worker's normal path and surface any problem via last_cleanup_error and a
    "warning" key, not silently drop it.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    monkeypatch.setattr(teleop, "verify_devices", lambda *a, **k: [])
    monkeypatch.setattr(teleop, "reset_torque_limit", lambda *a, **k: [])
    monkeypatch.setattr(teleop, "clear_goal_velocity", lambda *a, **k: [])

    class _FollowerBus:
        def __init__(self) -> None:
            self.port = "COM_FOLLOWER"
            self.motors = {"shoulder_pan": 1, "elbow_flex": 3}

        def connect(self) -> None:
            pass

        def write_calibration(self, calibration) -> None:
            pass

        def disable_torque(self, motor: str, num_retry: int = 0) -> None:
            # Once the follower is armed, this motor won't release.
            raise ConnectionError(f"no response from {motor}")

    class _Follower:
        def __init__(self, config) -> None:
            self.bus = _FollowerBus()
            self.cameras: dict = {}
            self.calibration: dict = {}
            self.configured = False
            self.disconnected = False

        def configure(self) -> None:
            # This is the real write that energizes the servos.
            self.configured = True

        def disconnect(self) -> None:
            self.disconnected = True

    class _LeaderBus:
        def connect(self) -> None:
            pass

        def write_calibration(self, calibration) -> None:
            pass

    class _Leader:
        def __init__(self, config) -> None:
            self.bus = _LeaderBus()
            self.calibration: dict = {}
            self.disconnected = False

        def configure(self) -> None:
            # Fails AFTER the follower has already configured (armed).
            raise RuntimeError("leader configure failed")

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "SO101Follower", lambda config: created.setdefault("follower", _Follower(config))
    )
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: created.setdefault("leader", _Leader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The follower really was configured (torque enabled) before the failure.
    assert created["follower"].configured is True
    assert teleop.last_cleanup_error is not None
    assert "TORQUE MAY STILL BE ENABLED" in teleop.last_cleanup_error
    assert "warning" in result
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]
    assert created["follower"].disconnected is True
    assert created["leader"].disconnected is True


def test_start_teleoperation_bimanual_force_disables_torque_and_warns_when_leader_configure_fails_after_followers_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bimanual mirror of the single-arm regression test above.

    _connect_bimanual has its own try/except (it must clean up the four buses
    it opened before handle_start_teleoperation ever sees a robot/teleop_device
    to work with) — robot.configure() there energizes both follower sub-arms
    before teleop_device.configure() (the next setup step) can fail. That
    except block must run the same force_disable_torque + _safe_disconnect
    cleanup as the single-arm path, and the resulting warning must reach the
    handle_start_teleoperation response — not be dropped, and not be
    overwritten by the outer except's own (no-op, since its local
    robot/teleop_device are still None here) cleanup pass.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "build_bimanual_configs", lambda request: ("robot_cfg", "teleop_cfg"))
    monkeypatch.setattr(teleop, "verify_devices", lambda *a, **k: [])
    monkeypatch.setattr(teleop, "reset_torque_limit", lambda *a, **k: [])
    monkeypatch.setattr(teleop, "clear_goal_velocity", lambda *a, **k: [])

    class _SubBus:
        def __init__(self, port: str, motors: dict, fail_disable: bool = False) -> None:
            self.port = port
            self.motors = motors
            self.fail_disable = fail_disable

        def connect(self) -> None:
            pass

        def write_calibration(self, calibration) -> None:
            pass

        def disable_torque(self, motor: str, num_retry: int = 0) -> None:
            if self.fail_disable:
                raise ConnectionError(f"no response from {motor}")

    class _SubArm:
        def __init__(self, bus: _SubBus) -> None:
            self.bus = bus
            self.calibration: dict = {}

    class _BiFollower:
        def __init__(self, config) -> None:
            # Once armed, neither follower sub-arm's motors release.
            self.left_arm = _SubArm(_SubBus("COM_LFOLLOWER", {"shoulder_pan": 1}, fail_disable=True))
            self.right_arm = _SubArm(_SubBus("COM_RFOLLOWER", {"shoulder_pan": 1}, fail_disable=True))
            self.configured = False
            self.disconnected = False

        def configure(self) -> None:
            # This is the real write that energizes the follower servos.
            self.configured = True

        def disconnect(self) -> None:
            self.disconnected = True

    class _BiLeader:
        def __init__(self, config) -> None:
            self.left_arm = _SubArm(_SubBus("COM_LLEADER", {"shoulder_pan": 1}))
            self.right_arm = _SubArm(_SubBus("COM_RLEADER", {"shoulder_pan": 1}))
            self.disconnected = False

        def configure(self) -> None:
            # Fails AFTER the followers have already configured (armed).
            raise RuntimeError("leader configure failed")

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "BiSOFollower", lambda config: created.setdefault("follower", _BiFollower(config))
    )
    monkeypatch.setattr(teleop, "BiSOLeader", lambda config: created.setdefault("leader", _BiLeader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LLEADER",
        follower_port="COM_LFOLLOWER",
        leader_config="leader",
        follower_config="follower",
        mode="bimanual",
        right_leader_port="COM_RLEADER",
        right_follower_port="COM_RFOLLOWER",
        right_leader_config="rleader",
        right_follower_config="rfollower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The followers really were configured (torque enabled) before the failure.
    assert created["follower"].configured is True
    assert teleop.last_cleanup_error is not None
    assert "TORQUE MAY STILL BE ENABLED" in teleop.last_cleanup_error
    assert "warning" in result
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]
    assert created["follower"].disconnected is True
    assert created["leader"].disconnected is True


def _stub_teleop_request():
    from makermodslab.teleoperate import TeleoperateRequest

    return TeleoperateRequest(
        leader_port="/dev/leader",
        follower_port="/dev/follower",
        leader_config="leader",
        follower_config="follower",
    )


def test_start_teleoperation_blocked_when_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teleoperation must refuse to start while manual calibration owns the
    same serial bus, rather than opening a second connection on a live port."""
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)

    result = teleop.handle_start_teleoperation(_stub_teleop_request())
    assert result == {
        "success": False,
        "message": "Calibration is currently active. Stop it first.",
    }


def test_start_teleoperation_blocked_when_auto_calibration_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)

    result = teleop.handle_start_teleoperation(_stub_teleop_request())
    assert result == {
        "success": False,
        "message": "Auto-calibration is currently active. Stop it first.",
    }


def test_start_teleoperation_blocked_when_wiggle_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)

    result = teleop.handle_start_teleoperation(_stub_teleop_request())
    assert result == {
        "success": False,
        "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
    }


def test_start_teleoperation_blocked_when_replay_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay drives the same follower bus open-loop — teleoperation must
    refuse to start while it's active, or both threads race to write goal
    positions to the same servos."""
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr("makermodslab.replay.replay_active", True)

    result = teleop.handle_start_teleoperation(_stub_teleop_request())
    assert result == {
        "success": False,
        "message": "Replay is currently active. Stop it first.",
    }


# ---------------------------------------------------------------------------
# Teleop opens no cameras: it consumes no frames (only motor positions drive the
# URDF viewer). The follower config it builds therefore carries an empty camera
# set in BOTH paths; any camera display is handled by the browser. Recording,
# which DOES consume frames, is unchanged.
# ---------------------------------------------------------------------------


def test_teleop_single_config_carries_no_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-arm follower config teleop builds has no cameras — lerobot
    opens none, so any camera display is handled by the browser."""
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    from makermodslab.teleoperate import TeleoperateRequest
    from makermodslab.utils.robot_factory import build_single_configs

    request = TeleoperateRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="L",
        follower_config="F",
    )
    robot_config, _ = build_single_configs(request)

    assert robot_config.cameras == {}


def test_teleop_bimanual_config_carries_no_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bimanual left-follower config teleop builds has no cameras either —
    the same no-frames-no-cameras rule applies to both arms."""
    monkeypatch.setattr("makermodslab.utils.robot_factory.bimanual_base_id", lambda name: "base")
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.stage_bimanual_calibrations",
        lambda *args: ("leader_staging", "follower_staging", "base"),
    )
    from makermodslab.teleoperate import TeleoperateRequest
    from makermodslab.utils.robot_factory import build_bimanual_configs

    request = TeleoperateRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="L",
        follower_config="F",
        mode="bimanual",
        right_leader_port="/dev/rl",
        right_follower_port="/dev/rf",
        right_leader_config="RL",
        right_follower_config="RF",
    )
    robot_config, _ = build_bimanual_configs(request)

    # Cameras (when present) would be wired onto the LEFT follower arm.
    assert robot_config.left_arm_config.cameras == {}
    assert robot_config.right_arm_config.cameras == {}


class _FakeBus:
    """Motor bus double for the explicit torque-disable cleanup step."""

    def __init__(self, port: str = "COM_FAKE", failing: tuple[str, ...] = ()) -> None:
        self.port = port
        self.motors = {"shoulder_pan": 1, "elbow_flex": 3, "gripper": 6}
        self.failing = set(failing)
        self.disabled: list[tuple[str, int]] = []

    def disable_torque(self, motor: str, num_retry: int = 0) -> None:
        if motor in self.failing:
            raise ConnectionError(f"no response from {motor}")
        self.disabled.append((motor, num_retry))


class _FakePortHandler:
    def __init__(self) -> None:
        self.is_using = True
        self.clear_calls = 0

    def clearPort(self) -> None:  # noqa: N802 — camelCase mimics the real Feetech SDK method this fakes
        self.clear_calls += 1


class _FakeArm:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus


def test_force_disable_torque_disables_every_motor() -> None:
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeBus()
    problems = force_disable_torque(_FakeArm(bus), "follower arm")

    assert problems == []
    # Every motor is disabled individually, with retries.
    assert [motor for motor, _ in bus.disabled] == list(bus.motors)
    assert all(num_retry == 5 for _, num_retry in bus.disabled)


def test_force_disable_torque_clears_busy_port_before_writing() -> None:
    """After a camera/control-loop failure the SDK port handler can still be
    marked in-use; clear that latch before direct torque writes or every motor
    reports '[TxRxResult] Port is in use!'."""
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeBus()
    port_handler = _FakePortHandler()
    bus.port_handler = port_handler  # type: ignore[attr-defined]

    problems = force_disable_torque(_FakeArm(bus), "follower arm")

    assert problems == []
    assert port_handler.clear_calls == 1
    assert port_handler.is_using is False
    assert [motor for motor, _ in bus.disabled] == list(bus.motors)


def test_force_disable_torque_reports_failed_motor_and_port() -> None:
    """One bad motor must not stop the others from being released, and the
    problem message must be unmistakable: it names the port and warns that
    torque may still be enabled (the arm stays rigid until power is pulled).
    """
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeBus(port="COM_FOLLOWER", failing=("elbow_flex",))
    problems = force_disable_torque(_FakeArm(bus), "follower arm")

    assert len(problems) == 1
    assert "TORQUE MAY STILL BE ENABLED" in problems[0]
    assert "COM_FOLLOWER" in problems[0]
    assert "elbow_flex" in problems[0]
    # The remaining motors were still disabled despite the failure.
    assert [motor for motor, _ in bus.disabled] == ["shoulder_pan", "gripper"]


def test_force_disable_torque_handles_bimanual_and_none() -> None:
    from makermodslab.teleoperate import force_disable_torque

    class _BiDevice:
        def __init__(self) -> None:
            self.left_arm = _FakeArm(_FakeBus(port="COM_LEFT"))
            self.right_arm = _FakeArm(_FakeBus(port="COM_RIGHT", failing=("gripper",)))

    device = _BiDevice()
    problems = force_disable_torque(device, "follower arms")

    # Both sub-arm buses are handled; only the right one reports a problem.
    assert len(device.left_arm.bus.disabled) == 3
    assert len(problems) == 1
    assert "COM_RIGHT" in problems[0]

    assert force_disable_torque(None, "nothing") == []


class _FakeCamera:
    """Camera double with lerobot's OpenCVCamera disconnect semantics: raises
    DeviceNotConnectedError when it was never opened, releases otherwise."""

    def __init__(self, name: str, connected: bool = True, failing: bool = False) -> None:
        self.name = name
        self.is_connected = connected
        self.failing = failing
        self.released = False

    def disconnect(self) -> None:
        from lerobot.utils.errors import DeviceNotConnectedError

        if self.failing:
            raise RuntimeError(f"{self.name} wedged")
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} not connected.")
        self.is_connected = False
        self.released = True


class _FakeConnectableBus(_FakeBus):
    """Motor bus double that tracks open/closed state for teardown tests."""

    def __init__(
        self,
        port: str = "COM_FAKE",
        connected: bool = True,
        silent: bool = False,
        ping_dead: bool = False,
    ) -> None:
        super().__init__(port=port)
        self.is_connected = connected
        self.disconnect_calls = 0
        self.disconnect_torque_flags: list[bool] = []
        # silent=True models the real post-handshake-failure state: the serial
        # port is open (is_connected True, because lerobot's is_connected is
        # just port_handler.is_open) but no motor answers — pings AND writes
        # both fail. ping_dead=True models the narrower, more dangerous case:
        # a degraded-but-recoverable bus where the zero-retry ping fails but a
        # retried write would still land.
        self.silent = silent
        self.ping_dead = ping_dead
        self.pings: list[str] = []

    def ping(self, motor: str, num_retry: int = 0):
        self.pings.append(motor)
        return None if (self.silent or self.ping_dead) else 777

    def disable_torque(self, motor: str, num_retry: int = 0) -> None:
        if self.silent:
            raise ConnectionError(f"no response from {motor}")
        super().disable_torque(motor, num_retry)

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls += 1
        self.disconnect_torque_flags.append(disable_torque)
        self.is_connected = False


class _FakePartialRobot:
    def __init__(self, bus: _FakeConnectableBus, cameras: dict[str, _FakeCamera]) -> None:
        self.bus = bus
        self.cameras = cameras


def test_force_disconnect_partial_releases_bus_when_a_later_camera_never_opened() -> None:
    """The regression this helper exists for: connect() opened the bus and the
    first camera, then died on it, leaving the *second* camera never opened.
    lerobot's all-or-nothing is_connected makes robot.disconnect() a no-op
    raise in that state, leaking the bus (next attempt: "FeetechMotorsBus is
    already connected") and the opened camera's read thread.
    """
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    front = _FakeCamera("front", connected=True)
    wrist = _FakeCamera("wrist", connected=False)  # never reached by connect()
    robot = _FakePartialRobot(bus, {"front": front, "wrist": wrist})

    force_disconnect_partial(robot, "robot")

    assert front.released is True  # read thread released, device freed
    assert bus.is_connected is False and bus.disconnect_calls == 1


def test_lerobot_disconnect_cannot_release_a_partially_connected_robot() -> None:
    """Pins *why* force_disconnect_partial exists, against lerobot's own guard.

    Uses the real ``check_if_not_connected`` decorator and the real all-or-
    nothing ``is_connected`` shape from SOFollower. If upstream ever makes
    disconnect() tolerant of partial state, this test fails and the helper can
    collapse back to robot.disconnect().
    """
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.utils.decorators import check_if_not_connected
    from lerobot.utils.errors import DeviceNotConnectedError
    from makermodslab.teleoperate import force_disconnect_partial

    # Watch the real upstream class, not just our copy of its shape: the
    # reconstruction below is only faithful while SO101Follower.disconnect is
    # still guarded and is_connected is still all-or-nothing. Without these
    # two assertions, upstream could drop the decorator entirely and this test
    # would keep passing against its own hand-written stand-in, stranding a
    # workaround that is no longer needed.
    assert getattr(SO101Follower.disconnect, "__wrapped__", None) is not None, (
        "SO101Follower.disconnect is no longer decorated — re-check whether "
        "force_disconnect_partial is still needed."
    )
    assert "all(" in inspect.getsource(SO101Follower.is_connected.fget), (
        "SO101Follower.is_connected is no longer all-or-nothing — re-check "
        "whether force_disconnect_partial is still needed."
    )

    class _LeRobotShapedRobot:
        def __init__(self) -> None:
            self.bus = _FakeConnectableBus(port="COM_FOLLOWER")
            self.cameras = {
                "front": _FakeCamera("front", connected=True),
                "wrist": _FakeCamera("wrist", connected=False),
            }

        @property
        def is_connected(self) -> bool:
            return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())

        @check_if_not_connected
        def disconnect(self) -> None:
            self.bus.disconnect()
            for cam in self.cameras.values():
                cam.disconnect()

    robot = _LeRobotShapedRobot()
    with pytest.raises(DeviceNotConnectedError):
        robot.disconnect()
    # ...and nothing was released — this is the leak that made every later
    # recording attempt fail with "FeetechMotorsBus is already connected".
    assert robot.bus.is_connected is True
    assert robot.cameras["front"].is_connected is True

    force_disconnect_partial(robot, "robot")
    assert robot.bus.is_connected is False
    assert robot.cameras["front"].is_connected is False


def test_force_disconnect_partial_releases_bus_despite_a_wedged_camera() -> None:
    """A camera that fails to release must not strand the serial port."""
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    wedged = _FakeCamera("front", connected=True, failing=True)
    wrist = _FakeCamera("wrist", connected=True)
    robot = _FakePartialRobot(bus, {"front": wedged, "wrist": wrist})

    problems = force_disconnect_partial(robot, "robot")

    assert wedged.released is False
    assert wrist.released is True  # a bad camera doesn't abort the rest
    assert bus.is_connected is False
    # A camera still holding the OS device is exactly what makes the NEXT
    # connect attempt fail, so it must be surfaced, not swallowed.
    assert len(problems) == 1
    assert problems[0] == "Could not release robot camera front: front wedged"


def test_force_disconnect_partial_disables_remaining_motors_despite_one_failing() -> None:
    """force_disconnect_partial must disable torque motor-by-motor, not rely on
    bus.disconnect()'s own internal torque-disable: lerobot's real disconnect()
    disables torque motor-by-motor too, but a single motor's failed write
    aborts that loop and leaves every motor after it energized/rigid — this is
    exactly why force_disable_torque exists as a standalone belt-and-braces
    step elsewhere in this module (see its docstring) and why
    _cleanup_after_setup_failure calls it before disconnecting. This helper
    handles the same "torque may already be enabled after an incomplete
    connect" situation and needs the same protection.
    """
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    bus.failing = {"elbow_flex"}
    robot = _FakePartialRobot(bus, {})

    force_disconnect_partial(robot, "robot")

    # shoulder_pan (before elbow_flex) and gripper (after it) must still get
    # an explicit disable_torque call despite elbow_flex's failure.
    assert [motor for motor, _ in bus.disabled] == ["shoulder_pan", "gripper"]
    assert bus.is_connected is False


def test_force_disconnect_partial_does_not_alarm_on_a_bus_that_never_opened() -> None:
    """The torque-disable pass used to run unconditionally, writing to a
    closed port for every motor and then printing the single most alarming
    string the system can emit — "TORQUE MAY STILL BE ENABLED ... unplug its
    power to release it" — for an arm that was never connected.

    Scope note: `is_connected` is `port_handler.is_open`, so this guard covers
    only the case where `openPort()` itself failed (device node missing or
    busy). The far more common unpowered/wrong-baud arm keeps `is_connected`
    True — that one is caught by the ping probe, not here. See
    test_force_disable_torque_does_not_alarm_when_no_motor_answers.
    """
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER", connected=False)
    robot = _FakePartialRobot(bus, {})

    problems = force_disconnect_partial(robot, "robot")

    assert bus.disabled == []  # no motor writes against an unopened port
    assert bus.pings == []  # not even probed — the port was never open
    assert bus.disconnect_calls == 0
    assert problems == []


def test_force_disable_torque_still_writes_when_the_probe_fails_but_the_bus_is_alive() -> None:
    """The exact case the probe must not be allowed to veto: a
    degraded-but-recoverable bus where the zero-retry ping used for liveness
    fails on every motor, but the retried (num_retry=5) disable_torque write
    still lands. Gating the write on the probe result would leave a genuinely
    energized arm rigid — the probe may only be consulted after a write
    failure, never before one.
    """
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeConnectableBus(port="COM_FOLLOWER", ping_dead=True)

    problems = force_disable_torque(_FakeArm(bus), "robot")

    # Every motor's write was attempted and landed, despite the dead probe.
    assert [motor for motor, _ in bus.disabled] == list(bus.motors)
    assert problems == []


def test_force_disable_torque_does_not_alarm_when_no_motor_answers() -> None:
    """The real "arm unplugged / wrong port" shape, and the one the
    is_connected guard cannot catch.

    lerobot's MotorsBus._connect calls openPort() BEFORE _handshake(), and
    does not close the port when the handshake fails. So for an unpowered arm,
    browned-out servos, a wrong baud rate, or a valid-but-wrong serial device,
    is_connected is still True on a bus no motor is listening on. The
    torque-disable write is still attempted on every motor (a
    degraded-but-recoverable bus could have taken it), and only once every
    motor's write has failed does the probe get consulted to say what was
    actually observed, instead of reporting "TORQUE MAY STILL BE ENABLED ...
    unplug its power" on an arm that has no power to unplug.
    """
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeConnectableBus(port="COM_FOLLOWER", silent=True)

    problems = force_disable_torque(_FakeArm(bus), "robot")

    assert bus.pings == list(bus.motors)  # probed every motor after every write failed
    assert bus.disabled == []  # every write was attempted and failed, none landed
    assert len(problems) == 1
    assert "No motor answered on COM_FOLLOWER" in problems[0]
    # The alarm is the thing under test: it must NOT be asserted as fact.
    assert "TORQUE MAY STILL BE ENABLED" not in problems[0]
    # ...but the rigid-arm advice survives as a conditional, so a genuinely
    # energized arm still tells the operator what to do.
    assert "If the arm is rigid" in problems[0]


def test_force_disable_torque_still_alarms_when_a_live_bus_has_a_bad_motor() -> None:
    """The probe must not become a blanket excuse: when the bus answers, a
    motor that won't take the disable is a real "this joint may stay rigid"
    condition and keeps the loud alarm."""
    from makermodslab.teleoperate import force_disable_torque

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    bus.failing = {"elbow_flex"}

    problems = force_disable_torque(_FakeArm(bus), "robot")

    assert bus.pings == ["shoulder_pan"]  # short-circuits on the first answer
    assert [motor for motor, _ in bus.disabled] == ["shoulder_pan", "gripper"]
    assert len(problems) == 1
    assert "TORQUE MAY STILL BE ENABLED" in problems[0]
    assert "elbow_flex" in problems[0]


def test_force_disconnect_partial_does_not_re_disable_torque_on_disconnect() -> None:
    """force_disable_torque already disabled every motor independently.
    lerobot's default disconnect(disable_torque=True) would re-run that pass
    with num_retry=5, where the first unresponsive motor raises BEFORE
    closePort() — leaking the port and appending a second, misleading "could
    not release the bus" problem. Same call as rollout.py, motor_power.py,
    identify.py and auto_calibrate.py already use."""
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    robot = _FakePartialRobot(bus, {})

    problems = force_disconnect_partial(robot, "robot")

    assert bus.disconnect_torque_flags == [False]
    assert problems == []


class _FakeFailingPortHandler(_FakePortHandler):
    def __init__(self, fail_close: bool = False) -> None:
        super().__init__()
        self.fail_close = fail_close
        self.close_calls = 0

    def closePort(self) -> None:  # noqa: N802 — camelCase mimics the real Feetech SDK method this fakes
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("port already gone")


class _FakeUnreleasableBus(_FakeConnectableBus):
    """Bus whose disconnect() always raises, to exercise the force-close fallback."""

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls += 1
        raise RuntimeError("bus wedged")


def test_force_disconnect_partial_force_closes_port_when_disconnect_raises() -> None:
    """If bus.disconnect() itself fails, the port handle must still be forced
    closed — same last-resort fallback utils/devices.py's
    _force_close_device_resources uses elsewhere — instead of leaking the COM
    handle for the rest of the process.
    """
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeUnreleasableBus(port="COM_FOLLOWER")
    port_handler = _FakeFailingPortHandler()
    bus.port_handler = port_handler  # type: ignore[attr-defined]
    robot = _FakePartialRobot(bus, {})

    problems = force_disconnect_partial(robot, "robot")

    # force_disable_torque's own pre-write port clear already ran (the bus is
    # connected, so it isn't skipped) and already set is_using False — so
    # asserting those two here would pass even with the fallback's own
    # clearPort()/is_using deleted. Pin the exact count instead: 2 means the
    # fallback did its own defensive clear rather than relying on the earlier
    # one.
    assert port_handler.clear_calls == 2
    assert port_handler.is_using is False
    assert port_handler.close_calls == 1
    # Pin *which* problem was reported, not merely that the port appears in
    # one of them — the force-close message also contains the port.
    assert len(problems) == 1
    assert problems[0].startswith("Could not release robot bus on COM_FOLLOWER")


def test_force_disconnect_partial_reports_a_failed_force_close() -> None:
    """The last-resort branch itself: when closePort() ALSO fails, the port is
    genuinely wedged for the rest of the process and the operator must be told
    — this is the one state the fallback cannot rescue, so it must not fail
    silently."""
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeUnreleasableBus(port="COM_FOLLOWER")
    bus.port_handler = _FakeFailingPortHandler(fail_close=True)  # type: ignore[attr-defined]
    robot = _FakePartialRobot(bus, {})

    problems = force_disconnect_partial(robot, "robot")

    assert len(problems) == 2
    assert problems[0].startswith("Could not release robot bus on COM_FOLLOWER")
    assert problems[1].startswith("Failed to force-close robot bus port on COM_FOLLOWER")
    assert "port already gone" in problems[1]


def test_force_disconnect_partial_returns_problems_instead_of_none() -> None:
    """Sibling _cleanup_after_setup_failure returns joined problem text so
    callers can surface teardown failures to the operator; force_disconnect_partial
    silently returned None despite discarding force_disable_torque's problems.
    """
    from makermodslab.teleoperate import force_disconnect_partial

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    bus.failing = {"elbow_flex"}
    robot = _FakePartialRobot(bus, {})

    problems = force_disconnect_partial(robot, "robot")

    assert isinstance(problems, list)
    assert any("TORQUE MAY STILL BE ENABLED" in p and "elbow_flex" in p for p in problems)


def test_force_disconnect_partial_releases_vanished_cameras_capture_handle() -> None:
    """A camera whose DEVICE disappeared mid-connect (unplugged between the
    cv2 open and the read thread starting) reports not-connected with no
    thread — but still owns the OS capture session via ``videocapture``.
    lerobot's disconnect() raises without releasing it (its guard is
    ``if not self.is_connected and self.thread is None: raise``), and left in
    place the stale in-process session poisons every later open of the
    replugged camera until MakerMods Lab restarts. The helper must release the raw
    handle on exactly this path.
    """
    from lerobot.utils.errors import DeviceNotConnectedError
    from makermodslab.teleoperate import force_disconnect_partial

    class _FakeVideoCapture:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    class _VanishedCamera:
        """lerobot OpenCVCamera's shape at the vanished-device point."""

        def __init__(self) -> None:
            self.is_connected = False  # isOpened() is False on the dead device
            self.thread = None  # unplug hit before _start_read_thread
            self.videocapture = _FakeVideoCapture()

        def disconnect(self) -> None:
            # Real OpenCVCamera.disconnect() guard, verbatim semantics.
            if not self.is_connected and self.thread is None:
                raise DeviceNotConnectedError("OpenCVCamera(0) not connected.")

    bus = _FakeConnectableBus(port="COM_FOLLOWER")
    cam = _VanishedCamera()
    capture = cam.videocapture
    robot = _FakePartialRobot(bus, {"wrist": cam})

    force_disconnect_partial(robot, "robot")

    assert capture.released is True
    assert cam.videocapture is None
    assert bus.is_connected is False


def test_force_disconnect_partial_is_idempotent_and_handles_bimanual_and_none() -> None:
    from makermodslab.teleoperate import force_disconnect_partial

    # Already fully disconnected: no raise, no bus disconnect call.
    bus = _FakeConnectableBus(connected=False)
    robot = _FakePartialRobot(bus, {"front": _FakeCamera("front", connected=False)})
    force_disconnect_partial(robot, "robot")
    force_disconnect_partial(robot, "robot")
    assert bus.disconnect_calls == 0

    # Bimanual: buses live on the sub-arms, cameras are merged at the top level.
    class _BiRobot:
        def __init__(self) -> None:
            self.left_arm = _FakeArm(_FakeConnectableBus(port="COM_LEFT"))
            self.right_arm = _FakeArm(_FakeConnectableBus(port="COM_RIGHT"))
            self.cameras = {"left_front": _FakeCamera("left_front")}

    bi = _BiRobot()
    force_disconnect_partial(bi, "robot")
    assert bi.left_arm.bus.is_connected is False
    assert bi.right_arm.bus.is_connected is False
    assert bi.cameras["left_front"].released is True

    # Bimanual, PARTIALLY connected: connect() opened the left arm's bus and
    # died before the right one. The guard must skip exactly one of the two,
    # and the left arm must still be released — a mixed state is the whole
    # reason the teardown is scoped per-bus rather than per-device.
    bi = _BiRobot()
    bi.right_arm.bus.is_connected = False
    force_disconnect_partial(bi, "robot")
    assert bi.left_arm.bus.disconnect_calls == 1
    assert bi.right_arm.bus.disconnect_calls == 0  # never opened, never touched
    assert bi.left_arm.bus.disabled != []  # torque released on the live arm
    assert bi.right_arm.bus.disabled == []

    # A device with no cameras attribute at all, and None.
    assert force_disconnect_partial(_FakeArm(_FakeConnectableBus()), "teleop") == []
    # None must be a clean no-op, not an AttributeError on a cleanup path.
    assert force_disconnect_partial(None, "nothing") == []


def test_stop_teleoperation_surfaces_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the worker's cleanup could not release an arm, the stop response
    must carry a warning instead of claiming a clean disconnect.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(
        teleop, "last_cleanup_error", "TORQUE MAY STILL BE ENABLED on COM_FOLLOWER (follower arm)."
    )

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]
    assert teleop.teleoperation_active is False


class _FakeWorker:
    """Thread double: reports alive until joined."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.joined = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
        self._alive = False


def test_hold_torque_release_grace_cut_short_by_release_request() -> None:
    """A set release event must end the hold immediately (no 5s sleep)."""
    import threading
    import time

    from makermodslab.teleoperate import hold_torque_release_grace

    release_now = threading.Event()
    release_now.set()
    start = time.monotonic()
    assert hold_torque_release_grace(release_now, grace_s=30.0) is True
    assert time.monotonic() - start < 1.0


def test_hold_torque_release_grace_elapses_without_release_request() -> None:
    import threading

    from makermodslab.teleoperate import hold_torque_release_grace

    assert hold_torque_release_grace(threading.Event(), grace_s=0.01) is False


def test_stop_teleoperation_enters_release_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first stop of a live session must return immediately (not block
    through the rest-pose return), report `releasing`, and tell the user the
    arm goes back to its starting position and that a second Stop releases it
    now. There is no timed hold anymore — same behavior as the auto-cal stop.
    """
    import makermodslab.teleoperate as teleop

    worker = _FakeWorker()
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert result["releasing"] is True
    assert "returns to its starting position" in result["message"]
    assert "holds its pose" not in result["message"]  # the hold phase is gone
    assert "Stop again" in result["message"]
    # The response must not join through the return.
    assert worker.joined is False
    assert teleop.teleoperation_active is False


def test_second_stop_during_grace_releases_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing Stop again during the grace hold is the 'skip the wait'
    gesture: it must set the release event and wait for the worker's cleanup.
    """
    import threading

    import makermodslab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert release_now.is_set()
    assert worker.joined is True
    assert teleop.teleoperation_thread is None


def test_second_stop_during_grace_surfaces_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", _FakeWorker())
    monkeypatch.setattr(teleop, "_release_now", threading.Event())
    monkeypatch.setattr(
        teleop, "last_cleanup_error", "TORQUE MAY STILL BE ENABLED on COM_FOLLOWER (follower arm)."
    )

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]


class _StuckWorker:
    """Thread double: never actually exits — join() times out, is_alive() stays True.

    Simulates a worker still mid rest-pose-return/cleanup after a second stop's
    `join(timeout=5.0)` elapses, so the code must NOT abandon the reference.
    """

    def __init__(self) -> None:
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


def test_second_stop_timeout_keeps_thread_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: when the second-stop join() times out (the worker is still
    genuinely alive), `teleoperation_thread` must keep pointing at it instead
    of being nulled out — nulling it would let a later start's mutex/guard
    checks wrongly treat the still-running, still-hardware-holding worker as
    gone (see finish_pending_release, which follows this same discipline).
    """
    import threading

    import makermodslab.teleoperate as teleop

    worker = _StuckWorker()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", threading.Event())
    monkeypatch.setattr(teleop, "last_cleanup_error", None)

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert "has not shut down yet" in result["message"]
    assert "did not shut down within 5s" in result["warning"]
    # The key regression assertion: the module must still hold the SAME live
    # thread object, not None.
    assert teleop.teleoperation_thread is worker

    # A subsequent finish_pending_release() must still recognize the worker as
    # alive (not treat it as already gone) and attempt its own join.
    assert teleop.finish_pending_release() is False
    assert worker.join_calls  # finish_pending_release actually tried to join it

    # A third manual stop press must re-enter the same second-stop branch
    # (not fall through to "No teleoperation session is active").
    result2 = teleop.handle_stop_teleoperation()
    assert result2["message"] != "No teleoperation session is active"
    assert teleop.teleoperation_thread is worker


def test_finish_pending_release_cuts_grace_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start arriving during the grace hold must release the arms and free
    the ports instead of failing port-busy for the rest of the grace.
    """
    import threading

    import makermodslab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)

    assert teleop.finish_pending_release() is True
    assert release_now.is_set()
    assert worker.joined is True
    assert teleop.teleoperation_thread is None


def test_finish_pending_release_leaves_live_session_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live session (active flag still set) is not a pending release: the
    caller's mutex check reports it, and torque must stay untouched.
    """
    import threading

    import makermodslab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)

    assert teleop.finish_pending_release() is False
    assert not release_now.is_set()
    assert worker.joined is False


def test_finish_pending_release_noop_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    assert teleop.finish_pending_release() is True


def test_teleoperation_status_reports_releasing(monkeypatch: pytest.MonkeyPatch) -> None:
    """During the post-stop return the status must say the arm is still
    energized and going home (releasing) rather than pretending the session
    is fully over.
    """
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "releasing", True)

    status = teleop.handle_teleoperation_status()

    assert status["teleoperation_active"] is False
    assert status["releasing"] is True
    assert "returning the arm" in status["message"].lower()


# ---------------------------------------------------------------------------
# Rest-pose return (makermodslab.rest_pose) and its stop-path integration
# ---------------------------------------------------------------------------


class _RestBus:
    """Bus double for rest-pose capture/return (makermodslab.rest_pose)."""

    _MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    # Real SO-101 layout: every joint is +/-100 range, only the gripper is 0..100.
    _NORM_MODES = dict.fromkeys(_MOTORS, MotorNormMode.RANGE_M100_100) | {
        "gripper": MotorNormMode.RANGE_0_100
    }

    def __init__(self, positions=None, moving: int = 1, port: str = "COM_FOLLOWER") -> None:
        self.port = port
        self.motors = {m: Motor(i + 1, "sts3215", self._NORM_MODES[m]) for i, m in enumerate(self._MOTORS)}
        self.positions = dict.fromkeys(self._MOTORS, 1000) if positions is None else dict(positions)
        self.moving = moving
        self.fail_reads = False
        self.writes: list[tuple] = []
        self.sync_writes: list[tuple] = []

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        if self.fail_reads:
            raise ConnectionError("bus gone")
        if reg == "Present_Position":
            return dict(self.positions)
        if reg == "Moving":
            return dict.fromkeys(self.positions, self.moving)
        raise KeyError(reg)

    def write(self, reg: str, motor: str, value: int, normalize: bool = True) -> None:
        self.writes.append((reg, motor, value))

    def sync_write(self, reg: str, values: dict, normalize: bool = True) -> None:
        self.sync_writes.append((reg, dict(values)))


class _RestClock:
    """Simulated time: sleep() advances monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def rest_clock(monkeypatch: pytest.MonkeyPatch) -> _RestClock:
    """Drive makermodslab.rest_pose's time off a simulated clock (no real sleeps)."""
    import makermodslab.rest_pose as rest_pose

    clock = _RestClock()
    monkeypatch.setattr(rest_pose.time, "sleep", clock.sleep)
    monkeypatch.setattr(rest_pose.time, "monotonic", clock.monotonic)
    return clock


def test_capture_rest_pose_reads_raw_ticks() -> None:
    """Raw Present_Position is captured as-is: teleoperation never rewrites
    Homing_Offset mid-session, so raw ticks are directly replayable later."""
    from makermodslab.rest_pose import capture_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 123, "gripper": 90})
    assert capture_rest_pose(bus) == {"shoulder_pan": 123, "gripper": 90}

    bus.fail_reads = True
    assert capture_rest_pose(bus) == {}  # never raises — the session must still start


def test_capture_rest_pose_normalized_reads_floats() -> None:
    """normalize=True reads the same normalized units robot.send_action()
    uses, for a caller (replay's ease-in) whose target is an action dict
    rather than raw ticks."""
    from makermodslab.rest_pose import capture_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 12.5, "gripper": 90.0})
    assert capture_rest_pose(bus, normalize=True) == {"shoulder_pan": 12.5, "gripper": 90.0}


def test_return_to_rest_pose_arrives_and_writes_gentle_goals(rest_clock: _RestClock) -> None:
    """The return writes a gentle profile speed then the captured goals, and
    reports 'returned' once every motor is within tolerance."""
    import makermodslab.rest_pose as rest_pose

    targets = {"shoulder_pan": 1000, "shoulder_lift": 1005}
    bus = _RestBus(positions={"shoulder_pan": 1000, "shoulder_lift": 1000})

    arrived, reason = rest_pose.return_to_rest_pose(bus, targets, label="follower arm")

    assert arrived is True
    # The completion report carries per-motor |final - target| deltas so a
    # subtly-off landing is diagnosable from the log.
    assert reason.startswith("returned: max delta 5 ticks")
    assert "shoulder_lift=5" in reason
    # Goals land via one sync_write; the finally-restore then zeroes the gentle
    # speed cap (Goal_Velocity is RAM-persistent — a leftover 400 would
    # throttle the next session's follower until a power cycle).
    assert bus.sync_writes == [
        ("Goal_Position", targets),
        ("Goal_Velocity", dict.fromkeys(targets, 0)),
    ]
    speed_writes = [w for w in bus.writes if w[0] == "Goal_Velocity"]
    assert {w[2] for w in speed_writes} == {rest_pose.RETURN_POS_SPEED}
    assert {w[1] for w in speed_writes} == set(targets)


def test_return_to_rest_pose_arrives_with_normalized_targets(rest_clock: _RestClock) -> None:
    """normalize=True writes/reads/compares in the same normalized units as
    robot.send_action() — no raw-tick conversion needed for a target that's
    already an action dict — and a caller-supplied tolerance is honored
    instead of the raw-ticks default."""
    import makermodslab.rest_pose as rest_pose

    targets = {"shoulder_pan": 10.0, "gripper": 50.0}
    bus = _RestBus(positions={"shoulder_pan": 10.5, "gripper": 49.0})

    arrived, reason = rest_pose.return_to_rest_pose(
        bus, targets, label="follower arm", normalize=True, tolerance=2.0
    )

    assert arrived is True
    assert reason.startswith("returned: max delta 1.0")
    # Written via the SAME sync_write call shape, only normalize flips.
    assert bus.sync_writes[0] == ("Goal_Position", targets)


def test_return_to_rest_pose_normalized_default_tolerance_is_raw_ticks_constant(
    rest_clock: _RestClock,
) -> None:
    """Omitting `tolerance` with normalize=True still falls back to
    RETURN_ARRIVE_TOLERANCE (20) — documented so a caller isn't surprised by
    an inherited raw-ticks-sized tolerance in normalized-unit space."""
    import makermodslab.rest_pose as rest_pose

    bus = _RestBus(positions={"shoulder_pan": 10.0})
    arrived, _ = rest_pose.return_to_rest_pose(bus, {"shoulder_pan": 10.0}, normalize=True)
    assert arrived is True


def test_return_to_rest_pose_clamps_out_of_range_normalized_target(rest_clock: _RestClock) -> None:
    """A recorded action can carry a normalized target outside the bus's
    representable range (lerobot's own calibration-aware conversion clamps
    both the Goal_Position write and the Present_Position read-back to
    [-100, 100] / [0, 100]). The arrival check must compare against the same
    clamped value that was actually written, or a saturated joint can never
    be reported as arrived — it stalls no matter how the arm is posed."""
    import makermodslab.rest_pose as rest_pose

    # shoulder_lift is pinned at its clamp boundary (-100.0) and cannot move
    # any further — exactly what lerobot's read-back reports for a target
    # below -100.
    bus = _RestBus(positions={"shoulder_lift": -100.0})

    arrived, reason = rest_pose.return_to_rest_pose(
        bus, {"shoulder_lift": -103.56}, label="follower arm", normalize=True, tolerance=2.0
    )

    assert arrived is True
    assert reason.startswith("returned: max delta 0")
    # The goal actually written is the clamped, reachable value — matches
    # what the comparison used, and what lerobot would have written anyway.
    assert bus.sync_writes[0] == ("Goal_Position", {"shoulder_lift": -100.0})


def test_return_to_rest_pose_clamps_out_of_range_gripper_target(rest_clock: _RestClock) -> None:
    """Same clamp behavior on the gripper's [0, 100] range, not just the
    +/-100 joints."""
    import makermodslab.rest_pose as rest_pose

    bus = _RestBus(positions={"gripper": 100.0})

    arrived, reason = rest_pose.return_to_rest_pose(
        bus, {"gripper": 104.0}, label="follower arm", normalize=True, tolerance=2.0
    )

    assert arrived is True
    assert reason.startswith("returned: max delta 0")


class _ConvergingBus:
    """Bus double for one motor whose Present_Position closes toward the
    Goal_Position target at a fixed rate per elapsed second, driven off the
    same simulated clock the rest_pose loop's own time.sleep/monotonic calls
    advance — reproduces a physically-converging (never stuck) motor without
    a real sleep, per the handoff's reproduction recipe for the stall-window
    unit-space defect."""

    def __init__(
        self, clock: _RestClock, motor: str, norm_mode: MotorNormMode, start: float, rate_per_s: float
    ):
        self.clock = clock
        self.motors = {motor: Motor(1, "sts3215", norm_mode)}
        self._motor = motor
        self._start = start
        self._rate = rate_per_s
        self._target: float | None = None
        self._t0: float | None = None

    def write(self, *a, **k) -> None:
        pass

    def sync_write(self, reg: str, values: dict, normalize: bool = True) -> None:
        if reg == "Goal_Position":
            self._target = float(values[self._motor])
            self._t0 = self.clock.now

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        if reg != "Present_Position":
            return {}
        elapsed = max(0.0, self.clock.now - (self._t0 or 0.0))
        direction = 1.0 if self._target >= self._start else -1.0
        pos = self._start + direction * self._rate * elapsed
        pos = min(pos, self._target) if direction > 0 else max(pos, self._target)
        return {self._motor: pos}


def test_return_to_rest_pose_normalized_default_stall_progress_matches_raw_ticks_constant(
    rest_clock: _RestClock,
) -> None:
    """Omitting `stall_min_progress` preserves today's behavior (the
    raw-ticks RETURN_STALL_MIN_PROGRESS, unconverted) — same
    opt-in-to-change shape as `tolerance`'s existing default. A motor
    converging at only 3 units/s never clears that raw-ticks-sized 10-unit
    bar within one stall window, so it is reported as stalled even though it
    was steadily, genuinely moving toward the target."""
    import makermodslab.rest_pose as rest_pose

    bus = _ConvergingBus(rest_clock, "wrist_roll", MotorNormMode.RANGE_M100_100, start=-20.0, rate_per_s=3.0)

    arrived, reason = rest_pose.return_to_rest_pose(
        bus, {"wrist_roll": 0.0}, label="follower arm", normalize=True, tolerance=2.0
    )

    assert arrived is False
    assert reason.startswith("stalled")


def test_return_to_rest_pose_custom_stall_min_progress_lets_slow_convergence_arrive(
    rest_clock: _RestClock,
) -> None:
    """A caller in normalized-unit space (replay's ease-in) can supply a
    stall-progress threshold sized for its own unit space instead of
    inheriting the raw-ticks constant — the SAME physical convergence as
    above must now be recognized as real progress and arrive."""
    import makermodslab.rest_pose as rest_pose

    bus = _ConvergingBus(rest_clock, "wrist_roll", MotorNormMode.RANGE_M100_100, start=-20.0, rate_per_s=3.0)

    arrived, reason = rest_pose.return_to_rest_pose(
        bus,
        {"wrist_roll": 0.0},
        label="follower arm",
        normalize=True,
        tolerance=2.0,
        stall_min_progress=1.0,
    )

    assert arrived is True
    assert reason.startswith("returned")


def test_return_to_rest_pose_stalls_without_progress(rest_clock: _RestClock) -> None:
    """Positions that never move toward the target must end in a stall (and
    fall through to the release) instead of looping to the ceiling."""
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("stalled")
    assert "shoulder_pan=1000" in reason  # the culprit and its distance are named
    import makermodslab.rest_pose as rest_pose

    assert rest_clock.now < rest_pose.RETURN_CEILING_S  # stall beat the ceiling


def test_return_to_rest_pose_reports_settled_short_motor(rest_clock: _RestClock) -> None:
    """A motor that stops moving (Moving == 0) while still far from target is
    NOT a successful return — bench symptom: 'the starting position was not
    right'. It must be reported as its own 'settled' outcome with the deltas."""
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000}, moving=0)
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("settled short of target")
    assert "shoulder_pan=1000" in reason


def test_return_to_rest_pose_cut_short_by_abort(rest_clock: _RestClock) -> None:
    """A set abort event (second stop, or a new session start) must end the
    return immediately so the release can run right away."""
    import threading

    from makermodslab.rest_pose import return_to_rest_pose

    abort = threading.Event()
    abort.set()
    bus = _RestBus(positions={"shoulder_pan": 1000})

    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000}, abort_event=abort)

    assert (arrived, reason) == (False, "cut-short")


def test_return_to_rest_pose_without_pose_is_a_noop() -> None:
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus()
    assert return_to_rest_pose(bus, {}) == (False, "no-pose")
    assert bus.sync_writes == []  # nothing written — straight to the release


def _assert_speed_cap_restored(bus: _RestBus, targets: dict[str, int]) -> None:
    """The last sync_write must zero the gentle Goal_Velocity cap on exactly
    the motors the return drove (RAM-persistent: a leftover cap would throttle
    the next session's follower until a power cycle)."""
    assert bus.sync_writes[-1] == ("Goal_Velocity", dict.fromkeys(targets, 0))


def test_return_restores_speed_cap_on_stall(rest_clock: _RestClock) -> None:
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert (arrived, reason[:7]) == (False, "stalled")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_settled(rest_clock: _RestClock) -> None:
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000}, moving=0)
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("settled")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_cut_short(rest_clock: _RestClock) -> None:
    import threading

    from makermodslab.rest_pose import return_to_rest_pose

    abort = threading.Event()
    abort.set()
    bus = _RestBus(positions={"shoulder_pan": 1000})

    assert return_to_rest_pose(bus, {"shoulder_pan": 2000}, abort_event=abort) == (
        False,
        "cut-short",
    )
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_ceiling(
    rest_clock: _RestClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the pathological ceiling exit (positions creep just enough to
    never stall) and check the cap is still zeroed on the way out."""
    import makermodslab.rest_pose as rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    original_sync_read = bus.sync_read

    def _creeping_read(reg: str, normalize: bool = True) -> dict:
        # Enough progress per poll to keep resetting the stall window.
        bus.positions["shoulder_pan"] += rest_pose.RETURN_STALL_MIN_PROGRESS + 1
        return original_sync_read(reg, normalize)

    monkeypatch.setattr(bus, "sync_read", _creeping_read)
    arrived, reason = rest_pose.return_to_rest_pose(bus, {"shoulder_pan": 10**6})

    assert arrived is False
    assert reason.startswith("ceiling")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 10**6})


def test_return_restores_speed_cap_on_failed_start(rest_clock: _RestClock) -> None:
    """A comm-error while writing the goals may have already stamped the
    gentle cap on some motors — the best-effort zeroing must still run."""
    from makermodslab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})

    def _failing_write(reg: str, motor: str, value: int, normalize: bool = True) -> None:
        raise ConnectionError("bus gone")

    bus.write = _failing_write
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("comm-error")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_speed_cap_restore_failure_never_raises(rest_clock: _RestClock) -> None:
    """The zeroing is best-effort: a dead bus at restore time must not raise —
    the caller's torque release has to run no matter what."""
    from makermodslab.rest_pose import return_to_rest_pose

    targets = {"shoulder_pan": 1000}
    bus = _RestBus(positions={"shoulder_pan": 1000})
    original_sync_write = bus.sync_write

    def _failing_sync_write(reg: str, values: dict, normalize: bool = True) -> None:
        if reg == "Goal_Velocity":
            raise ConnectionError("bus gone")
        original_sync_write(reg, values, normalize)

    bus.sync_write = _failing_sync_write
    arrived, reason = return_to_rest_pose(bus, targets)

    assert arrived is True  # the return itself still completed and reported
    assert reason.startswith("returned")


def test_return_followers_to_rest_covers_every_follower_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bimanual: both follower buses get their return (the leader is never in
    the list — it is human-held with torque off)."""
    import threading

    import makermodslab.teleoperate as teleop

    calls: list[tuple] = []
    lock = threading.Lock()

    def _spy(bus, pose, abort_event=None, label=""):
        with lock:  # runs on per-arm threads now — guard the shared list
            calls.append((bus, pose, abort_event))
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    abort = threading.Event()
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], abort)

    # Order is no longer deterministic (arms run concurrently), so assert on the
    # set of (bus, pose) covered rather than the sequence.
    assert {(c[0], tuple(sorted(c[1].items()))) for c in calls} == {
        ("busL", (("m", 1),)),
        ("busR", (("m", 2),)),
    }
    # The worker's abort event is passed through so a second stop cuts the return.
    assert all(c[2] is abort for c in calls)


def test_return_followers_run_concurrently_not_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both followers' returns overlap in time: a slow first arm must NOT delay
    the second arm starting. Proven with a barrier — if the returns were
    sequential, the second arm would never enter while the first is still in
    its return, and the barrier would time out."""
    import threading

    import makermodslab.teleoperate as teleop

    started = threading.Barrier(2, timeout=5.0)
    both_started = threading.Event()

    def _spy(bus, pose, abort_event=None, label=""):
        # Every arm's return must have *entered* before any is allowed to
        # finish. A sequential loop can never satisfy this (arm 2 hasn't
        # started while arm 1 blocks here) — the barrier would raise BrokenBarrier.
        started.wait()
        both_started.set()
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    abort = threading.Event()
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], abort)

    assert both_started.is_set()  # both entered before either returned


def test_return_followers_wrapper_waits_for_all_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper returns only after every per-arm return has finished — the
    downstream torque release ordering depends on it. A slow arm must be joined,
    not left running."""
    import threading

    import makermodslab.teleoperate as teleop

    finished = {"busL": False, "busR": False}
    fast_arm_done = threading.Event()
    release = threading.Event()

    def _spy(bus, pose, abort_event=None, label=""):
        # The slow arm (busL) blocks until released; the wrapper must not
        # return until it too has finished. The fast arm signals when it's done
        # so the test can then release the slow one — no real sleeps.
        if bus == "busL":
            release.wait(timeout=5.0)
        else:
            fast_arm_done.set()
        finished[bus] = True
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)

    def _release_after_fast_arm():
        # Once the fast arm has finished, let the slow arm complete. If the
        # wrapper joined all threads it will still be blocked in join() here.
        fast_arm_done.wait(timeout=5.0)
        release.set()

    releaser = threading.Thread(target=_release_after_fast_arm)
    releaser.start()
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], threading.Event())
    releaser.join()

    # If the wrapper returned before joining busL, this would still be False.
    assert finished == {"busL": True, "busR": True}


def test_return_followers_one_arm_failing_does_not_block_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One arm's return raising (despite return_to_rest_pose's never-raise
    contract) must not propagate out of the wrapper, nor prevent the other
    arm's return from completing."""
    import threading

    import makermodslab.teleoperate as teleop

    completed: set = set()
    lock = threading.Lock()

    def _spy(bus, pose, abort_event=None, label=""):
        if bus == "busL":
            raise RuntimeError("bus L exploded")
        with lock:
            completed.add(bus)
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    # Must not raise even though busL's return raised.
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], threading.Event())

    assert "busR" in completed  # the healthy arm still finished


def test_return_followers_abort_stops_every_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set abort event (second stop / release-now) reaches every arm's return
    — each sees the same event set and bails out promptly."""
    import threading

    import makermodslab.teleoperate as teleop

    seen_set: list[bool] = []
    lock = threading.Lock()

    def _spy(bus, pose, abort_event=None, label=""):
        with lock:
            seen_set.append(abort_event is not None and abort_event.is_set())
        return False, "cut-short"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    abort = threading.Event()
    abort.set()
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], abort)

    assert seen_set == [True, True]  # both arms saw the abort already set


def test_return_followers_single_arm_still_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common single-arm case still drives its one bus and returns cleanly
    (one thread, joined) — same observable outcome as before."""
    import threading

    import makermodslab.teleoperate as teleop

    calls: list[tuple] = []

    def _spy(bus, pose, abort_event=None, label=""):
        calls.append((bus, pose))
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    teleop._return_followers_to_rest([("busSolo", {"m": 7})], threading.Event())

    assert calls == [("busSolo", {"m": 7})]


def test_start_clears_stale_release_state_from_previous_double_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-session leak regression: session 1's double-stop sets
    _release_now; if session 2's start didn't clear it (under the state lock),
    every later grace hold AND rest-pose return would be cut short instantly
    until the server restarts."""
    import threading

    import makermodslab.teleoperate as teleop

    stale = threading.Event()
    stale.set()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(teleop, "_release_now", stale)
    monkeypatch.setattr(teleop, "releasing", True)
    monkeypatch.setattr(
        "makermodslab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("port busy")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    result = teleop.handle_start_teleoperation(
        teleop.TeleoperateRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
        )
    )

    # The connect fails, but the per-session reset already ran under the lock.
    assert result["success"] is False
    assert not stale.is_set()
    assert teleop.releasing is False


# ---------------------------------------------------------------------------
# Follower power telemetry (Present_Current, ~1 Hz)
# ---------------------------------------------------------------------------


class _CurrentBus:
    """Bus double serving Present_Current sync_reads."""

    def __init__(self, readings: list[dict], port: str = "COM_FOLLOWER") -> None:
        self._readings = list(readings)
        self.port = port

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        assert reg == "Present_Current" and normalize is False
        if not self._readings:
            raise ConnectionError("bus gone")
        return self._readings.pop(0)


def test_power_telemetry_tracks_peak_and_mean() -> None:
    """Peaks/means in mA (6.5 mA per register LSB), one INFO line per session."""
    import makermodslab.teleoperate as teleop

    telemetry = teleop.PowerTelemetry()
    bus = _CurrentBus([{"shoulder_pan": 100, "gripper": 20}, {"shoulder_pan": 40, "gripper": 60}])
    telemetry.sample(bus)
    telemetry.sample(bus)

    assert telemetry.peak_ma["shoulder_pan"] == 100 * 6.5
    assert telemetry.latest_ma["shoulder_pan"] == 40 * 6.5
    summary = telemetry.summary()
    assert summary is not None
    assert summary.startswith("power telemetry:")
    assert f"shoulder_pan peak {100 * 6.5:.0f}mA / mean {70 * 6.5:.0f}mA" in summary


def test_power_telemetry_prefixes_bimanual_and_survives_bus_errors() -> None:
    import makermodslab.teleoperate as teleop

    telemetry = teleop.PowerTelemetry()
    telemetry.sample(_CurrentBus([{"gripper": 10}]), prefix="left_")
    telemetry.sample(_CurrentBus([]))  # dead bus: sample must not raise

    assert set(telemetry.peak_ma) == {"left_gripper"}


def test_power_telemetry_summary_none_without_samples() -> None:
    import makermodslab.teleoperate as teleop

    assert teleop.PowerTelemetry().summary() is None


# ---------------------------------------------------------------------------
# Session error taxonomy — outcome / error / hint in the status payload (the
# in-process twin of rollout's exited payload; the pure classifier itself is
# covered in tests/test_record.py). A mid-loop death is "failed"; a user stop
# whose cleanup alone complained is "ran_with_warning"; a clean stop is "ok".
# ---------------------------------------------------------------------------


def test_teleoperation_status_carries_failed_outcome_with_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that died mid-loop surfaces outcome/error/hint through the
    status payload, with the hint mapped from the error text."""
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "releasing", False)
    monkeypatch.setattr(teleop, "last_session_outcome", "failed")
    monkeypatch.setattr(
        teleop,
        "last_session_error",
        "DeviceNotConnectedError: could not connect to the follower arm",
    )

    status = teleop.handle_teleoperation_status()

    assert status["outcome"] == "failed"
    assert "could not connect" in status["error"]
    assert "plugged in" in status["hint"]


def test_teleoperation_status_carries_cleanup_warning_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user stop whose cleanup tripped (gripper overload on torque disable)
    is ran_with_warning — the session itself ran fine."""
    import makermodslab.teleoperate as teleop

    cleanup_text = "TORQUE MAY STILL BE ENABLED on COM_FOLLOWER (follower arm; gripper: Overload)."
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "releasing", False)
    monkeypatch.setattr(teleop, "last_cleanup_error", cleanup_text)
    monkeypatch.setattr(teleop, "last_session_outcome", "ran_with_warning")
    monkeypatch.setattr(teleop, "last_session_error", cleanup_text)

    status = teleop.handle_teleoperation_status()

    assert status["outcome"] == "ran_with_warning"
    assert "TORQUE MAY STILL BE ENABLED" in status["error"]
    assert "motor overloaded" in status["hint"].lower()
    # The existing raw safety field is not regressed by the new taxonomy.
    assert status["last_cleanup_error"] == cleanup_text


def test_teleoperation_status_outcome_none_before_any_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before any session ends (and after a start clears the fields) the
    taxonomy keys are present but null — the frontend treats that as no-op."""
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "releasing", False)
    monkeypatch.setattr(teleop, "last_session_outcome", None)
    monkeypatch.setattr(teleop, "last_session_error", None)

    status = teleop.handle_teleoperation_status()

    assert status["outcome"] is None
    assert status["error"] is None
    assert status["hint"] is None


# ---------------------------------------------------------------------------
# I8: shutdown_event() has no UI to poll and no "press Stop again" gesture
# available, but handle_stop_teleoperation()'s first call is deliberately
# fire-and-forget — it flips teleoperation_active and returns immediately so
# the request thread isn't blocked on the return-to-rest motion, relying on a
# status poll or a second Stop press to observe or force completion. Calling
# it alone from shutdown would let the process exit while the worker is still
# mid-return, with no return-to-rest and no torque release. stop_and_wait()
# composes the stop with a bounded join so a caller with no UI can get the
# same graceful-then-forced guarantee synchronously.
# ---------------------------------------------------------------------------


def test_stop_and_wait_is_a_noop_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)

    teleop.stop_and_wait(timeout=1.0)  # must return promptly, no exception


def test_stop_and_wait_blocks_until_worker_finishes_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real worker that responds to teleoperation_active going False (the
    normal graceful path: return-to-rest, then release) must be allowed to
    finish on its own within the timeout -- stop_and_wait must not return
    before that, and must not force an early release when it didn't need to."""
    import makermodslab.teleoperate as teleop

    released = threading.Event()

    def _worker() -> None:
        while teleop.teleoperation_active:
            time.sleep(0.01)
        released.set()

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)
    teleop._release_now.clear()
    worker.start()

    teleop.stop_and_wait(timeout=2.0)

    assert released.is_set(), "stop_and_wait returned before the worker finished releasing"
    assert teleop.teleoperation_active is False
    assert not teleop._release_now.is_set(), "a worker that finished gracefully must not be force-released"
    worker.join(timeout=2.0)


def test_stop_and_wait_forces_release_if_worker_does_not_finish_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker stuck past the graceful-release window (e.g. a stalled
    return-to-rest) must be force-released via the same "second stop"
    mechanism the UI's second Stop press uses -- shutdown has no operator to
    press it, so stop_and_wait must do it automatically once the bound
    elapses."""
    import makermodslab.teleoperate as teleop

    def _worker() -> None:
        # Ignores teleoperation_active; only responds to a forced release,
        # standing in for a stalled/wedged return-to-rest.
        teleop._release_now.wait(timeout=5.0)

    worker = threading.Thread(target=_worker, daemon=True)
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)
    teleop._release_now.clear()
    worker.start()

    teleop.stop_and_wait(timeout=0.2)

    assert teleop._release_now.is_set(), "a worker that outlasts the timeout must be force-released"
    worker.join(timeout=5.0)
    assert not worker.is_alive()
