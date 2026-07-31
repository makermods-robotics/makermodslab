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
"""Tests for makermodslab.focus_tune — the uvc-util table parser, the
AVFoundation-uniqueID→uvc-index mapping, binary discovery, and the
refusal branches of handle_start_focus_tune. The sweep itself (cv2 capture
+ live UVC writes) is hardware and deliberately untested, like the other
subprocess happy paths."""

from __future__ import annotations

import pytest

import makermodslab
from makermodslab import focus_tune, record, rollout, teleoperate  # noqa: F401

# Verbatim `uvc-util -d` output from the SO-101 rig's two cameras.
UVC_DEVICE_TABLE = """\
------------ -------------- ------------ ------------ ------------------------------------------------
Index        Vend:Prod      LocationID   UVC Version  Device name
------------ -------------- ------------ ------------ ------------------------------------------------
0            0x1e45:0x0209  0x01113000   1.00         USB Camera
1            0x1e45:0x0209  0x01114000   1.00         USB Camera
------------ -------------- ------------ ------------ ------------------------------------------------
"""


class TestParseUvcDevices:
    def test_parses_rig_table(self):
        devices = focus_tune.parse_uvc_devices(UVC_DEVICE_TABLE)
        assert devices == [
            {"uvc_index": 0, "vid": 0x1E45, "pid": 0x0209, "location_id": 0x01113000},
            {"uvc_index": 1, "vid": 0x1E45, "pid": 0x0209, "location_id": 0x01114000},
        ]

    def test_ignores_headers_and_rules(self):
        assert focus_tune.parse_uvc_devices("Index Vend:Prod\n----\n") == []


class TestMatchUvcIndex:
    DEVICES = focus_tune.parse_uvc_devices(UVC_DEVICE_TABLE)

    def test_matches_usb_camera_unique_id(self):
        # AVFoundation composes uniqueID = locationID + VID + PID; cv2 strips
        # nothing, but leading zeros differ between the two representations.
        assert focus_tune.match_uvc_index("0x11130001e450209", self.DEVICES) == 0
        assert focus_tune.match_uvc_index("0x11140001e450209", self.DEVICES) == 1

    def test_builtin_camera_uuid_matches_nothing(self):
        builtin = "6C707041-05AC-0011-0003-000000000001"
        assert focus_tune.match_uvc_index(builtin, self.DEVICES) is None

    def test_empty_unique_id_matches_nothing(self):
        assert focus_tune.match_uvc_index("", self.DEVICES) is None


class TestRoiFor:
    def test_wrist_cameras_score_the_gripper_window(self):
        assert focus_tune._roi_for("wrist") == focus_tune.WRIST_ROI
        assert focus_tune._roi_for("left_WRIST") == focus_tune.WRIST_ROI

    def test_other_cameras_score_the_central_crop(self):
        assert focus_tune._roi_for("front") == focus_tune.DEFAULT_ROI
        assert focus_tune._roi_for("") == focus_tune.DEFAULT_ROI


class TestFindUvcUtil:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        binary = tmp_path / "uvc-util"
        binary.touch()
        monkeypatch.setenv(focus_tune.UVC_UTIL_ENV, str(binary))
        assert focus_tune.find_uvc_util() == str(binary)

    def test_missing_everywhere_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv(focus_tune.UVC_UTIL_ENV, raising=False)
        monkeypatch.setattr(focus_tune.shutil, "which", lambda _: None)
        monkeypatch.setattr(focus_tune, "_UVC_UTIL_CANDIDATES", (tmp_path / "absent",))
        assert focus_tune.find_uvc_util() is None


@pytest.fixture
def darwin_with_uvc_util(monkeypatch):
    """Pretend we're on macOS with a discoverable uvc-util binary."""
    monkeypatch.setattr(focus_tune.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(focus_tune, "find_uvc_util", lambda: "/fake/uvc-util")
    return None


CAMERAS = [{"camera_index": 0, "name": "front"}]
AVF_CAMERAS = [{"index": 0, "name": "USB Camera", "unique_id": "0x11130001e450209"}]


class TestHandleStartRefusals:
    def test_refuses_off_macos(self, monkeypatch):
        monkeypatch.setattr(focus_tune.platform, "system", lambda: "Linux")
        result = focus_tune.handle_start_focus_tune(CAMERAS, AVF_CAMERAS)
        assert result["success"] is False
        assert "macOS" in result["message"]

    def test_refuses_without_uvc_util(self, monkeypatch):
        monkeypatch.setattr(focus_tune.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(focus_tune, "find_uvc_util", lambda: None)
        result = focus_tune.handle_start_focus_tune(CAMERAS, AVF_CAMERAS)
        assert result["success"] is False
        assert focus_tune.UVC_UTIL_ENV in result["message"]

    def test_refuses_empty_camera_list(self, darwin_with_uvc_util):
        result = focus_tune.handle_start_focus_tune([], AVF_CAMERAS)
        assert result["success"] is False

    @pytest.mark.parametrize(
        ("module", "flag", "label"),
        [
            ("record", "recording_active", "recording"),
            ("teleoperate", "teleoperation_active", "teleoperation"),
            ("rollout", "inference_active", "inference"),
        ],
    )
    def test_refuses_while_robot_flow_active(self, darwin_with_uvc_util, monkeypatch, module, flag, label):
        monkeypatch.setattr(getattr(makermodslab, module), flag, True)
        result = focus_tune.handle_start_focus_tune(CAMERAS, AVF_CAMERAS)
        assert result["success"] is False
        assert label in result["message"]

    def test_refuses_when_already_tuning(self, darwin_with_uvc_util, monkeypatch):
        monkeypatch.setattr(
            focus_tune,
            "_uvc_run",
            lambda *a: UVC_DEVICE_TABLE if a[1] == ["-d"] else "auto-focus\nfocus-abs\n",
        )
        monkeypatch.setattr(focus_tune, "focus_tune_active", True)
        result = focus_tune.handle_start_focus_tune(CAMERAS, AVF_CAMERAS)
        assert result["success"] is False
        assert "already running" in result["message"]

    def test_refuses_when_no_camera_is_tunable(self, darwin_with_uvc_util, monkeypatch):
        # uvc-util sees no devices at all → every requested camera unmappable.
        monkeypatch.setattr(focus_tune, "_uvc_run", lambda *a: "")
        result = focus_tune.handle_start_focus_tune(CAMERAS, AVF_CAMERAS)
        assert result["success"] is False
        assert "focus-tunable" in result["message"]


class TestStatus:
    def test_idle_status_shape(self):
        status = focus_tune.get_focus_tune_status()
        assert status["active"] is False
        assert isinstance(status["cameras"], list)

    def test_status_returns_copies(self):
        status = focus_tune.get_focus_tune_status()
        status["cameras"].append({"tampered": True})
        assert focus_tune.get_focus_tune_status()["cameras"] == []
