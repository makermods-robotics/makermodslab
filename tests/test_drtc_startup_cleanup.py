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
"""Execute each entrypoint's startup/teardown with fake devices, without Portal.

The coroutine is compiled from its real source to avoid importing optional FFI
and policy packages. These are behavior tests, not assertions about AST shape.
"""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from makermodslab.drtc import _session_glue as glue


@pytest.mark.parametrize("engine", ["robot_sync", "robot_rtc"])
@pytest.mark.parametrize("failure", ["partial_connect", "post_connect_print", "after_capture"])
def test_startup_failure_always_releases_robot(engine, failure):
    calls = []

    class Bus:
        is_connected = True
        motors = {"elbow": object(), "gripper": object()}

        def disable_torque(self, motor, num_retry):
            calls.append(f"disable:{motor}")

        def disconnect(self, disable_torque):
            assert disable_torque is False
            calls.append("bus_closed")
            self.is_connected = False

    class Robot:
        bus = Bus()
        cameras = {
            "opened": SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("camera_closed")),
            "failed": SimpleNamespace(is_connected=False, disconnect=lambda: pytest.fail("never connected")),
        }

        def connect(self):
            calls.append("connect")
            if failure == "partial_connect":
                raise RuntimeError("configuration failed after opening bus")

        def disconnect(self):
            calls.append("disconnect")

    def startup_print(*args, **kwargs):
        if failure == "post_connect_print":
            raise BrokenPipeError("parent exited")

    robot = Robot()
    path = Path(__file__).parents[1] / "makermodslab" / "drtc" / f"{engine}.py"
    run = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
    )
    run.args.args[0].annotation = None
    run.returns = None
    ns = {
        "LoopControl": lambda: SimpleNamespace(
            start_command_pump=lambda: None,
            quit_event=SimpleNamespace(is_set=lambda: False),
            abort_event=None,
        ),
        "load_env": lambda: None,
        "emit": lambda *args: None,
        "format_ready": lambda *args: "",
        "make_robot_from_config": lambda _: robot,
        "feetech_buses": lambda _: [],
        "robot_wire_schema": lambda _: (SimpleNamespace(cameras=[]), [], []),
        "print": startup_print,
        "capture_start_poses_or_warn": lambda *args: ["captured"],
        "VideoCodec": SimpleNamespace(),  # unknown codec fails AFTER pose capture
        "CameraTimingMonitor": lambda *args: SimpleNamespace(
            start=lambda: None, stop=lambda: calls.append("timing_stopped")
        ),
        "shielded": glue.shielded,
        "disconnect_robot": glue.disconnect_robot,
        "say": lambda *args: None,
        "return_step": lambda *args: lambda: calls.append("return"),
    }
    ns.update(
        {
            name: name
            for name in ("EVENT_READY", "EVENT_ERROR", "EVENT_STOPPING", "EVENT_RETURNING", "EVENT_BYE")
        }
    )
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), "exec"), ns)
    cfg = SimpleNamespace(
        livekit_url="unused",
        livekit_room="unused",
        livekit_token="unused",
        robot=None,
        return_to_rest=True,
        video_codec="bad",
    )
    with pytest.raises((RuntimeError, BrokenPipeError, AttributeError)):
        asyncio.run(ns["run"](cfg))
    if failure == "partial_connect":
        assert calls == ["connect", "disable:elbow", "disable:gripper", "camera_closed", "bus_closed"]
    else:
        assert calls[-1] == "disconnect"
        assert ("return" in calls) == (failure == "after_capture")
        if failure == "after_capture":
            assert calls.index("return") < calls.index("disconnect")


def test_partial_disconnect_continues_after_camera_failure():
    calls = []

    def failed_camera():
        calls.append("camera")
        raise RuntimeError("camera release failed")

    bus = SimpleNamespace(
        is_connected=True,
        motors={"elbow": object()},
        disable_torque=lambda *args, **kwargs: calls.append("torque_off"),
        disconnect=lambda **kwargs: calls.append("bus_closed"),
    )
    robot = SimpleNamespace(
        bus=bus,
        cameras={"bad": SimpleNamespace(is_connected=True, disconnect=failed_camera)},
    )
    with pytest.raises(RuntimeError, match="camera release failed"):
        glue.disconnect_robot(robot, False)
    assert calls == ["torque_off", "camera", "bus_closed"]
