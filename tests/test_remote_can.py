"""Pure CAN remote-control guards and hardware transition helpers; no Portal threads."""

from types import SimpleNamespace

import pytest

from makermodslab import remote_can, remote_host, remote_teleoperate


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "10"])
def test_remote_pose_refuses_missing_or_invalid_joints(bad):
    with pytest.raises(ValueError, match="gripper.pos"):
        remote_can.checked_pose({"elbow.pos": 2, "gripper.pos": bad}, ["elbow.pos", "gripper.pos"])


def test_remote_rest_preserves_both_arm_prefixes_and_excludes_both_grippers():
    assert remote_can.rest_pose(
        {
            "left_elbow.pos": 20,
            "right_elbow.pos": -30,
            "left_gripper.pos": 80,
            "right_gripper.pos": 10,
        }
    ) == {"left_elbow": 20, "right_elbow": -30}


@pytest.mark.parametrize("elapsed,step", [(1 / 30, 1), (1 / 60, 0.5), (15, 3), (-1, 0)])
def test_remote_targets_have_a_rate_limit_even_after_network_gaps(elapsed, step):
    result = remote_can.limit_action(
        {"left_elbow.pos": 10.0, "right_elbow.pos": -10.0, "left_gripper.pos": 0.0},
        {"left_elbow.pos": 180.0, "right_elbow.pos": -180.0, "left_gripper.pos": 0.1},
        elapsed,
    )
    assert result == pytest.approx(
        {
            "left_elbow.pos": 10 + step,
            "right_elbow.pos": -10 - step,
            "left_gripper.pos": min(0.1, step),
        }
    )


@pytest.mark.parametrize("family", ["maker", "metal"])
@pytest.mark.parametrize("bimanual", [False, True])
def test_station_publishes_every_can_joint_in_degrees(family, bimanual):
    motors = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ]
    prefixes = ["left_", "right_"] if bimanual else [""]
    observation = {f"{prefix}{motor}.pos": i + 0.5 for prefix in prefixes for i, motor in enumerate(motors)}
    expected = {motor: i + 0.5 for i, motor in enumerate(motors)}
    result = remote_host.host_joint_data(observation, {}, family, bimanual)
    assert result == (
        {"joints_deg": expected, "joints_deg_right": expected} if bimanual else {"joints_deg": expected}
    )
    descriptor = {"arm_type": family, "motors": remote_teleoperate.leader_motors(observation)}
    assert remote_teleoperate.schema_mismatch(descriptor, family, descriptor["motors"]) is None
    assert remote_teleoperate.schema_mismatch(descriptor, "metal" if family == "maker" else "maker", None)
    assert remote_teleoperate.schema_mismatch(descriptor, family, descriptor["motors"][:-1])


def test_so101_station_keeps_urdf_units():
    observation = {"shoulder_pan.pos": 45}
    assert remote_host.host_joint_data(observation, {}, "so101", False) == {
        "joints": remote_host.observation_to_urdf_joints(observation, {})
    }


class Bus:
    def __init__(self, events, connected=True):
        self.events = events
        self.is_connected = connected
        self.motors = {"elbow": object(), "gripper": object()}
        self.gains = {"Kp": {"elbow": 20.0, "gripper": 5.0}, "Kd": {"elbow": 1.0, "gripper": 0.2}}
        self.position = {"elbow": 15.0, "gripper": 30.0}

    def sync_write(self, name, values):
        if name == "Goal_Position":
            self.events.append(("mit", dict(self.gains["Kp"]), dict(self.gains["Kd"])))
        else:
            self.gains[name] = dict(values)

    def read(self, name, motor):
        assert name == "Present_Position"
        self.events.append(("read", motor))
        return self.position[motor]

    def enable_torque(self):
        self.events.append("enable")

    def disable_torque(self):
        self.events.append("disable")

    def connect(self, handshake):
        self.events.append(("connect", handshake))
        self.is_connected = True

    def disconnect(self, disable_torque):
        self.events.append(("disconnect", disable_torque))
        self.is_connected = False


def robot_double(events, connected=True):
    bus = Bus(events, connected)
    return SimpleNamespace(
        bus=bus,
        cameras={},
        action_features={"elbow.pos": float, "gripper.pos": float},
        _resolved_gains={"elbow": (20.0, 1.0), "gripper": (5.0, 0.2)},
        config=SimpleNamespace(
            joint_limits={"elbow": (-90, 90), "gripper": (0, 100)}, velocity_feedforward=True
        ),
        get_observation=lambda: {f"{k}.pos": v for k, v in bus.position.items()},
        send_action=lambda action: events.append(("hold", action)),
    )


def test_can_engage_clears_old_effort_and_reads_again_after_enable():
    events = []
    robot = robot_double(events)
    remote_can.prepare(robot)
    assert not robot.config.velocity_feedforward
    pose = remote_can.engage(robot)
    zero = {"elbow": 0.0, "gripper": 0.0}
    assert events == [
        ("read", "elbow"),
        ("read", "gripper"),
        ("mit", zero, zero),
        "enable",
        ("read", "elbow"),
        ("read", "gripper"),
        ("hold", {"elbow.pos": 15.0, "gripper.pos": 30.0}),
    ]
    assert pose == {"elbow.pos": 15.0, "gripper.pos": 30.0}
    assert robot.bus.gains == {"Kp": {"elbow": 20.0, "gripper": 5.0}, "Kd": {"elbow": 1.0, "gripper": 0.2}}


def test_can_refuses_to_enable_if_present_pose_would_be_clamped():
    events = []
    robot = robot_double(events)
    robot.bus.position["elbow"] = 100
    with pytest.raises(ValueError, match="soft limits"):
        remote_can.engage(robot)
    assert "enable" not in events


def test_can_refuses_to_enable_when_motor_feedback_is_missing():
    events = []
    robot = robot_double(events)

    def no_feedback(*args):
        raise ConnectionError("elbow did not answer")

    robot.bus.read = no_feedback
    with pytest.raises(ConnectionError, match="did not answer"):
        remote_can.engage(robot)
    assert "enable" not in events


def test_partial_can_connect_is_released_without_another_handshake():
    events = []
    robot = robot_double(events, connected=False)
    assert remote_can.release(robot) is None
    assert events == [("connect", False), "disable", ("disconnect", False)]


def test_bimanual_release_closes_other_bus_and_cameras_after_failure():
    events = []
    left, right = robot_double(events), robot_double(events)

    def failed_disable():
        events.append("failed-disable")
        raise OSError("CAN cable lost")

    left.bus.disable_torque = failed_disable
    robot = SimpleNamespace(
        left_arm=left,
        right_arm=right,
        cameras={
            "wrist": SimpleNamespace(is_connected=True, disconnect=lambda: events.append("camera-close"))
        },
    )
    assert "CAN cable lost" in remote_can.release(robot)
    assert events[-3:] == ["disable", ("disconnect", False), "camera-close"]


def test_long_can_return_times_out_instead_of_accelerating(monkeypatch):
    from makermodslab import maker_rest_pose

    now = [0.0]
    monkeypatch.setattr(
        maker_rest_pose,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    sent = []
    device = SimpleNamespace(
        get_observation=lambda: {"wrist_roll.pos": 360.0},
        send_action=lambda action: sent.append(action["wrist_roll.pos"]),
    )
    arrived, _reason = maker_rest_pose.return_maker_to_pose(device, {"wrist_roll": 0.0}, ceiling_s=2.0)
    assert not arrived
    assert sent[-1] >= 298  # 30 deg/s, never a faster ramp to fit the deadline
    assert all(abs(a - b) <= 1.01 for a, b in zip([360.0] + sent, sent, strict=False))


@pytest.mark.parametrize("family", ["maker", "metal"])
@pytest.mark.parametrize("mode", ["single", "bimanual"])
def test_native_remote_devices_have_the_same_wire_schema(monkeypatch, tmp_path, family, mode):
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from makermodslab.utils import robot_factory

    monkeypatch.setattr(robot_factory, "setup_follower_calibration_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(robot_factory, "setup_leader_calibration_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        robot_factory,
        "stage_bimanual_follower_calibrations",
        lambda *args, **kwargs: (tmp_path, None),
    )
    monkeypatch.setattr(
        robot_factory,
        "stage_bimanual_leader_calibrations",
        lambda *args, **kwargs: (tmp_path, None),
    )
    request = SimpleNamespace(
        arm_type=family,
        mode=mode,
        robot_name="wire-preview",
        follower_port="/not-opened",
        leader_port="/not-opened",
        follower_config="f",
        leader_config="l",
        right_follower_port="/not-opened-right",
        right_leader_port="/not-opened-right",
        right_follower_config="rf",
        right_leader_config="rl",
    )
    # Constructors only: no connect or bus traffic, no Portal or worker loop.
    follower = make_robot_from_config(robot_factory.build_follower_config(request))
    leader = make_teleoperator_from_config(robot_factory.build_leader_config(request))
    try:
        remote_can.prepare(follower)
        motors, _ = remote_host.split_features(follower.observation_features)
        assert len(motors) == (14 if mode == "bimanual" else 7)
        assert (
            remote_teleoperate.schema_mismatch(
                {"arm_type": family, "motors": motors},
                family,
                remote_teleoperate.leader_motors(leader.action_features),
            )
            is None
        )
    finally:
        for device in (follower, leader):
            pool = getattr(device, "_io_pool", None)
            if pool is not None:
                pool.shutdown(wait=True)


@pytest.mark.parametrize("family", ["maker", "metal"])
def test_failed_can_host_connect_uses_can_cleanup(monkeypatch, family):
    import lerobot.robots

    events = []
    robot = robot_double(events, connected=False)

    def failed_connect(calibrate):
        assert calibrate is False
        raise OSError("partial handshake")

    robot.connect = failed_connect
    monkeypatch.setattr(remote_host, "build_follower_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(lerobot.robots, "make_robot_from_config", lambda config: robot)

    def feetech_is_wrong(*args, **kwargs):
        raise AssertionError("CAN setup must not use Feetech preflight or cleanup")

    for name in ("force_disconnect_partial", "verify_devices", "reset_torque_limit", "clear_goal_velocity"):
        monkeypatch.setattr(remote_host, name, feetech_is_wrong)
    with pytest.raises(RuntimeError, match="partial handshake") as caught:
        remote_host._connect_follower(
            remote_host.HostingRequest(arm_type=family, follower_port="/not-opened", follower_config="f")
        )
    assert caught.value.cleanup_error is None
    assert events == [("connect", False), "disable", ("disconnect", False)]


def test_robstride_feedback_allows_usb_latency_and_updates_each_motor():
    events = []
    robot = robot_double(events)

    def query(motor, *, timeout):
        # A valid reply arrives after the native driver's 3 ms deadline.
        assert timeout >= 0.020
        events.append(("query", motor))
        return False, SimpleNamespace(data=motor.encode())

    robot.bus._query_status_via_clear_fault = query
    robot.bus._decode_motor_state = lambda data: events.append(("decode", data))
    assert remote_can.get_observation(robot) == {"elbow.pos": 15.0, "gripper.pos": 30.0}
    assert events == [("query", "elbow"), ("decode", b"elbow"), ("query", "gripper"), ("decode", b"gripper")]


@pytest.mark.parametrize(
    "fault,message,error",
    [(False, None, ConnectionError), (True, SimpleNamespace(data=b"fault"), RuntimeError)],
)
def test_robstride_feedback_still_rejects_silence_and_motor_faults(fault, message, error):
    robot = robot_double([])
    robot.bus._query_status_via_clear_fault = lambda motor, timeout: (fault, message)

    def stale_observation():
        raise AssertionError("must not use a cached observation after a failed refresh")

    robot.get_observation = stale_observation
    with pytest.raises(error, match="elbow"):
        remote_can.get_observation(robot)


def test_maker_folded_zero_is_accepted_and_alignment_does_not_snap_to_soft_limit():
    robot = robot_double([])
    robot.bus.position = {"elbow": 0.0, "gripper": 0.0}
    robot.config.joint_limits = {"elbow": (2.8, 236.1), "gripper": (-120.1, -2.5)}
    original = robot.config.joint_limits
    pose = remote_can.read_pose(robot)
    goal = remote_can.clamp_action(robot, {"elbow.pos": 0.0, "gripper.pos": 0.0})
    assert goal == {"elbow.pos": 2.8, "gripper.pos": -2.5}
    sent = []

    def native_clamp(action):
        sent.append(
            {
                key: min(
                    robot.config.joint_limits[key[:-4]][1], max(robot.config.joint_limits[key[:-4]][0], value)
                )
                for key, value in action.items()
            }
        )

    robot.send_action = native_clamp
    # Holding the actual resting zero must not jump 2.8 degrees inward.
    remote_can.send_action(robot, pose, pose)
    assert sent[-1] == pose
    for _ in range(3):
        action = remote_can.limit_action(pose, goal, 1 / 30)
        remote_can.send_action(robot, action, pose)
        assert all(abs(sent[-1][key] - pose[key]) <= 1.001 for key in pose)
        assert robot.config.joint_limits is original
        pose = action
    assert sent[-1] == goal


def test_transitional_limits_are_restored_when_a_write_fails():
    robot = robot_double([])
    robot.config.joint_limits = {"elbow": (2.8, 236.1), "gripper": (-120.1, -2.5)}
    original = robot.config.joint_limits

    def failed_write(action):
        raise OSError("CAN cable lost")

    robot.send_action = failed_write
    with pytest.raises(OSError, match="CAN cable lost"):
        remote_can.send_action(
            robot, {"elbow.pos": 1.0, "gripper.pos": -1.0}, {"elbow.pos": 0.0, "gripper.pos": 0.0}
        )
    assert robot.config.joint_limits is original
