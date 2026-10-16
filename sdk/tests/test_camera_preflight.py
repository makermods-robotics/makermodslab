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
#
"""The client-side camera-coverage preflight for inference starts.

An empty ``camera_bindings`` is a camera-less run server-side, so a vision
policy started without them energizes the arm and dies on its first action.
The SDK reads the policy's config (``GET /api/v1/policy-config``) first and
refuses an uncovered start before anything is sent. The matrix under test:
uncovered fails CLOSED, an unknown config (unreadable, older server) fails
OPEN with a warning, and ``camera_bindings="auto"`` fails CLOSED on anything
it cannot resolve. Every case runs on MockTransport; nothing energizes.
"""

from __future__ import annotations

import json
import warnings

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import (
    ApiError,
    CameraBindingError,
    ServerTooOldError,
    UnverifiedCamerasWarning,
)
from makermodslab_sdk.resources.jobs import CheckpointPolicyConfig

REF = "org/repo@checkpoints/000100"
ROBOT = "maker_sdk"
TOP_WRIST = {"top": {"height": 480, "width": 640}, "wrist": {"height": 240, "width": 320}}


def policy_config(image_features: dict | None = None) -> dict:
    return {
        "policy_type": "act",
        "image_features": dict(image_features or {}),
        "requires_task": False,
        "state_dim": 7,
        "action_dim": 7,
    }


def robot_record(*camera_names: str) -> dict:
    return {
        "status": "success",
        "robot": {
            "name": ROBOT,
            "arm_type": "maker",
            "cameras": [
                {"name": name, "type": "opencv", "index_or_path": i} for i, name in enumerate(camera_names)
            ],
        },
    }


def session_body(kind: str) -> dict:
    return {
        "id": "sess-1",
        "kind": kind,
        "robot": ROBOT,
        "owner": "sdk:test",
        "started_at": 1.0,
        "revision": 1,
        "phase": "running",
        "lease": None,
    }


class Server:
    """A scripted server: fixed answers per (method, path), every request kept."""

    def __init__(
        self,
        *,
        config: tuple[int, dict] | None = None,
        robot: tuple[int, dict] | None = None,
        kind: str = "inference",
    ) -> None:
        self.config = config if config is not None else (200, policy_config(TOP_WRIST))
        self.robot = robot if robot is not None else (200, robot_record("top", "wrist"))
        self.kind = kind
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/policy-config":
            return httpx.Response(self.config[0], json=self.config[1])
        if request.method == "GET" and path == f"/api/v1/robots/{ROBOT}":
            return httpx.Response(self.robot[0], json=self.robot[1])
        if request.method == "POST" and path == "/api/v1/sessions":
            return httpx.Response(201, json={"session": session_body(self.kind), "warnings": None})
        if request.method == "POST" and path == "/api/v1/sessions/sess-1/stop":
            return httpx.Response(200, json={"session": session_body(self.kind), "result": {}})
        raise AssertionError(f"unscripted request: {request.method} {path}")

    def paths(self, method: str | None = None) -> list[str]:
        return [r.url.path for r in self.requests if method is None or r.method == method]

    def started(self) -> bool:
        return "/api/v1/sessions" in self.paths("POST")

    def start_options(self) -> dict:
        starts = [r for r in self.requests if r.method == "POST" and r.url.path == "/api/v1/sessions"]
        assert len(starts) == 1
        return json.loads(starts[0].content)["options"]

    def config_queries(self) -> list[str]:
        return [r.url.params["policy_ref"] for r in self.requests if r.url.path == "/api/v1/policy-config"]


UNREADABLE = (
    404,
    {"detail": f"Could not read the policy config for {REF!r}", "code": "checkpoint.config_unreadable"},
)
BARE_404 = (404, {"detail": "Not Found"})
INVALID_REF = (400, {"detail": "Unrecognised policy ref: 'org/repo'", "code": "checkpoint.invalid_ref"})


# --- jobs.policy_config ----------------------------------------------------------


def test_policy_config_reads_by_ref():
    server = Server()
    with mock_client(server) as client:
        cfg = client.jobs.policy_config(REF)
    assert isinstance(cfg, CheckpointPolicyConfig)
    assert sorted(cfg.image_features) == ["top", "wrist"]
    assert cfg.image_features["top"].width == 640
    assert server.config_queries() == [REF]


def test_policy_config_on_server_predating_route_says_server_too_old():
    with mock_client(Server(config=BARE_404)) as client, pytest.raises(ServerTooOldError) as excinfo:
        client.jobs.policy_config(REF)
    assert "checkpoint_policy_config" in str(excinfo.value)


def test_policy_config_invalid_ref_end_to_end(sdk_client):
    # A bare repo id is not a ref inference accepts; the route refuses it
    # before reading anything — no network, no hardware.
    with pytest.raises(ApiError) as excinfo:
        sdk_client.jobs.policy_config("__sdk_test__/not-a-ref")
    assert excinfo.value.status == 400
    assert excinfo.value.code == "checkpoint.invalid_ref"
    assert "Next step:" in str(excinfo.value)


# --- infer: the coverage check -----------------------------------------------------


def test_uncovered_bindings_refuse_before_any_start():
    server = Server()
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref=REF, task="pick")
    assert not server.started()
    error = excinfo.value
    assert error.expected == ("top", "wrist")
    assert error.missing == ("top", "wrist")
    assert error.record_cameras == ("top", "wrist")
    text = str(error)
    assert "'top'" in text and "'wrist'" in text
    assert "Nothing was started" in text
    assert "Next step:" in text
    assert f'client.sessions.infer("{ROBOT}", policy_ref="{REF}", camera_bindings="auto"' in text


def test_partially_covered_bindings_name_only_the_missing_camera():
    server = Server(robot=(200, robot_record("overhead", "gripper_cam")))
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings={"top": "overhead"})
    assert not server.started()
    assert excinfo.value.missing == ("wrist",)
    text = str(excinfo.value)
    # Names don't match, so "auto" is not the suggestion; the literal call
    # keeps the binding already given and names the record's cameras.
    assert 'camera_bindings="auto"' not in text
    assert '"top": "overhead"' in text
    assert "'gripper_cam'" in text


def test_robot_lookup_failure_is_omitted_not_fatal():
    server = Server(robot=(500, {"detail": "boom", "code": "internal.unexpected"}))
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref=REF)
    assert not server.started()
    assert excinfo.value.record_cameras is None
    assert f'client.robots.get("{ROBOT}").cameras' in str(excinfo.value)


def test_coaching_run_goes_through_the_same_check():
    server = Server()
    with mock_client(server) as client, pytest.raises(CameraBindingError):
        client.sessions.infer(ROBOT, policy_ref=REF, coaching=True, target_corrections=3)
    assert not server.started()


def test_covered_bindings_start_unchanged():
    server = Server()
    bindings = {"top": "overhead", "wrist": "gripper"}
    with mock_client(server) as client, warnings.catch_warnings():
        warnings.simplefilter("error")
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings=bindings).stop()
    options = server.start_options()
    assert options["camera_bindings"] == bindings
    assert "camera_dims" not in options  # explicit bindings never grow dims
    assert server.config_queries() == [REF]
    assert f"/api/v1/robots/{ROBOT}" not in server.paths()  # only fetched when needed


def test_camera_less_policy_without_bindings_starts():
    server = Server(config=(200, policy_config({})))
    with mock_client(server) as client, warnings.catch_warnings():
        warnings.simplefilter("error")
        client.sessions.infer(ROBOT, policy_ref=REF).stop()
    assert "camera_bindings" not in server.start_options()


def test_unreadable_config_warns_and_starts():
    server = Server(config=UNREADABLE)
    with mock_client(server) as client, pytest.warns(UnverifiedCamerasWarning, match="could not be verified"):
        client.sessions.infer(ROBOT, policy_ref=REF).stop()
    assert server.started()
    assert "camera_bindings" not in server.start_options()


def test_server_predating_the_route_warns_and_starts():
    server = Server(config=BARE_404)
    with mock_client(server) as client, pytest.warns(UnverifiedCamerasWarning, match="predates"):
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings={"top": "top"}).stop()
    assert server.started()
    assert server.start_options()["camera_bindings"] == {"top": "top"}


def test_invalid_ref_raises_early_for_inference():
    # The inference start refuses the same ref shapes, so the coded error is
    # raised while nothing has started.
    server = Server(config=INVALID_REF)
    with mock_client(server) as client, pytest.raises(ApiError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref="org/repo")
    assert excinfo.value.code == "checkpoint.invalid_ref"
    assert "Next step:" in str(excinfo.value)
    assert not server.started()


def test_verify_cameras_false_skips_the_lookup_entirely():
    server = Server()
    with mock_client(server) as client:
        client.sessions.infer(ROBOT, policy_ref=REF, verify_cameras=False).stop()
    assert server.config_queries() == []
    assert server.started()


# --- infer: camera_bindings="auto" -----------------------------------------------


def test_auto_binds_identically_named_cameras_and_fills_dims():
    server = Server()
    with mock_client(server) as client:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto").stop()
    options = server.start_options()
    assert options["camera_bindings"] == {"top": "top", "wrist": "wrist"}
    assert options["camera_dims"] == {
        "top": {"width": 640, "height": 480},
        "wrist": {"width": 320, "height": 240},
    }


def test_auto_keeps_explicit_dims():
    server = Server()
    dims = {"top": {"width": 1280, "height": 720}}
    with mock_client(server) as client:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto", camera_dims=dims).stop()
    assert server.start_options()["camera_dims"] == dims


def test_auto_for_a_camera_less_policy_sends_no_bindings():
    server = Server(config=(200, policy_config({})))
    with mock_client(server) as client:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto").stop()
    options = server.start_options()
    assert "camera_bindings" not in options and "camera_dims" not in options


def test_auto_is_exact_and_case_sensitive_never_partial():
    server = Server(robot=(200, robot_record("Top", "wrist")))
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto")
    assert not server.started()
    assert excinfo.value.missing == ("top",)
    text = str(excinfo.value)
    assert "'top'" in text and "'Top'" in text
    assert "auto" in text


@pytest.mark.parametrize("config", [UNREADABLE, BARE_404], ids=["unreadable", "old-server"])
def test_auto_fails_closed_when_the_config_is_unknown(config):
    server = Server(config=config)
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto")
    assert not server.started()
    assert "camera_bindings={" in str(excinfo.value)


def test_auto_with_verify_cameras_false_is_refused_client_side():
    server = Server()
    with mock_client(server) as client, pytest.raises(ValueError, match="verify_cameras"):
        client.sessions.infer(ROBOT, policy_ref=REF, camera_bindings="auto", verify_cameras=False)
    assert server.requests == []


# --- remote_infer ------------------------------------------------------------------


def test_remote_infer_refuses_uncovered_bindings():
    server = Server(kind="remote_inference")
    with mock_client(server) as client, pytest.raises(CameraBindingError) as excinfo:
        client.sessions.remote_infer(ROBOT, policy_ref=REF)
    assert not server.started()
    assert f'client.sessions.remote_infer("{ROBOT}"' in str(excinfo.value)


def test_remote_infer_reads_a_bare_repo_id_at_its_root():
    # Remote inference accepts a bare "<owner>/<repo>" (the GPU loads it); its
    # config lives at the repo root, which the route addresses as "@root".
    server = Server(kind="remote_inference")
    with mock_client(server) as client:
        client.sessions.remote_infer(ROBOT, policy_ref="org/repo", camera_bindings="auto").stop()
    assert server.config_queries() == ["org/repo@root"]
    assert server.start_options()["policy_ref"] == "org/repo"
    assert server.start_options()["camera_bindings"] == {"top": "top", "wrist": "wrist"}


def test_remote_infer_invalid_ref_is_unknown_not_fatal():
    # The remote start does not validate the ref's shape, so an early refusal
    # would block a run the server allows.
    server = Server(config=INVALID_REF, kind="remote_inference")
    with mock_client(server) as client, pytest.warns(UnverifiedCamerasWarning):
        client.sessions.remote_infer(ROBOT, policy_ref="/some/relative-ish/ref").stop()
    assert server.started()
