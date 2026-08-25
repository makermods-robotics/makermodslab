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

"""The machine-readable error-code taxonomy: grammar, coverage, and wiring.

Codes follow `<domain>.<condition>[.<detail>]` — dots separate levels,
underscores separate words within a level, so `code.split(".")` is always the
hierarchy. Level 1 comes from a closed domain set; the grammar test keeps the
namespace from drifting back to ad-hoc strings.
"""

from __future__ import annotations

import re

import pytest

# Dots for levels, underscores for words within a level; 2 or 3 levels.
CODE_GRAMMAR = re.compile(r"^[a-z]+(_[a-z]+)*(\.[a-z]+(_[a-z]+)*){1,2}$")

# Level-1 namespace. Closed set: extending it is a taxonomy decision, made
# here first.
DOMAINS = frozenset(
    ["request", "robot", "hardware", "hub", "job", "dataset", "model", "checkpoint", "session", "internal"]
)

# The mutual-exclusion matrix (CLAUDE.md "State model & mutual exclusion"),
# plus the post-session release grace. A new robot-driving feature must add
# its discriminant here AND reciprocal-check refusals in every peer.
BUSY_DISCRIMINANTS = frozenset(
    [
        "recording",
        "teleoperation",
        "inference",
        "replay",
        "calibration",
        "auto_calibration",
        "wiggle",
        "releasing",
    ]
)


def test_error_codes_follow_grammar():
    from makermodslab.api_errors import ErrorCode

    for code in ErrorCode:
        assert CODE_GRAMMAR.match(code.value), f"malformed code: {code.value!r}"
        assert code.value.split(".")[0] in DOMAINS, f"unknown domain: {code.value!r}"


def test_busy_discriminants_cover_mutex_matrix():
    from makermodslab.api_errors import ErrorCode

    busy = {c.value.split(".", 2)[2] for c in ErrorCode if c.value.startswith("robot.busy.")}
    assert busy == BUSY_DISCRIMINANTS


def _teleop_request():
    from makermodslab.teleoperate import TeleoperateRequest

    return TeleoperateRequest(
        leader_port="/dev/null-l", follower_port="/dev/null-f", leader_config="lc", follower_config="fc"
    )


@pytest.mark.parametrize(
    ("patch_target", "expected_code"),
    [
        ("makermodslab.teleoperate.teleoperation_active", "robot.busy.teleoperation"),
        ("makermodslab.record.recording_active", "robot.busy.recording"),
        ("makermodslab.rollout.inference_active", "robot.busy.inference"),
        ("makermodslab.replay.replay_active", "robot.busy.replay"),
        ("makermodslab.wiggle.wiggle_active", "robot.busy.wiggle"),
    ],
)
def test_teleop_start_refusals_carry_codes(monkeypatch, patch_target, expected_code):
    """Every reciprocal-check refusal names WHAT holds the robot, as a code."""
    from makermodslab.teleoperate import handle_start_teleoperation

    monkeypatch.setattr(patch_target, True)
    result = handle_start_teleoperation(_teleop_request())
    assert result["success"] is False
    assert result["code"] == expected_code


@pytest.mark.parametrize(
    ("manager_attr", "expected_code"),
    [
        ("calibration_manager", "robot.busy.calibration"),
        ("auto_calibration_manager", "robot.busy.auto_calibration"),
    ],
)
def test_teleop_start_refusals_carry_codes_manager_features(monkeypatch, manager_attr, expected_code):
    from makermodslab import auto_calibrate, calibrate
    from makermodslab.teleoperate import handle_start_teleoperation

    if manager_attr == "calibration_manager":
        monkeypatch.setattr(calibrate, "calibration_is_active", lambda: True)
    else:
        monkeypatch.setattr(auto_calibrate, "auto_calibration_is_active", lambda: True)
    result = handle_start_teleoperation(_teleop_request())
    assert result["success"] is False
    assert result["code"] == expected_code


def test_replay_start_refusal_carries_code(monkeypatch):
    from makermodslab.replay import ReplayRequest, handle_start_replay

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_replay(
        ReplayRequest(repo_id="u/d", episode_index=0, follower_port="/dev/null-f", follower_config="fc")
    )
    assert result["success"] is False
    assert result["code"] == "robot.busy.recording"


def test_endpoint_propagates_code(client, monkeypatch):
    """The dict→HTTPException conversion in server.py must not drop the code:
    the HTTP body carries `code` beside the legacy string `detail`."""
    monkeypatch.setattr("makermodslab.record.recording_active", True)
    resp = client.post(
        "/start-replay",
        json={"repo_id": "u/d", "episode_index": 0, "follower_port": "/dev/null-f", "follower_config": "fc"},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "robot.busy.recording"
    assert isinstance(body["detail"], str)  # legacy shape untouched


@pytest.mark.xfail(
    strict=True,
    reason="session.* codes are reserved for the Phase-2 /api/v1/sessions lease; "
    "this flips when that surface lands — remove the marker then",
)
def test_sessions_surface_uses_reserved_codes(client, monkeypatch):
    monkeypatch.setattr("makermodslab.record.recording_active", True)
    resp = client.post("/api/v1/sessions", json={"kind": "teleoperation", "robot": "nope"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "session.held"
