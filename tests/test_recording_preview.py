"""Recording previews use captured observations, never another hardware read."""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from makermodslab import recording_preview as previews, server
from makermodslab.arms import registry


def test_preview_sample_is_bounded_owned_rgb_and_preserves_aspect(monkeypatch):
    preview = previews.RecordingPreview()
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    assert not preview.publish({"wrist": image})
    preview.start()
    assert preview.publish({"wrist": image, "shoulder.pos": 3.0})
    image[:] = 0  # source buffer reuse must not change the captured snapshot
    assert not preview.publish({"other": image})
    decoded = cv2.imdecode(np.frombuffer(preview.jpeg("wrist"), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)
    assert decoded[0, 0, 2] > 240 and decoded[0, 0, 0] < 10
    assert preview.jpeg("shoulder.pos") is None
    now[0] += 0.2
    assert preview.publish({"other": image})
    assert preview.jpeg("wrist") is None
    preview.stop()
    assert preview.jpeg("other") is None
    preview.start()
    assert preview.jpeg("other") is None


def test_tap_broadcasts_bimanual_can_without_reading_robot(monkeypatch):
    preview = previews.RecordingPreview()
    monkeypatch.setattr(previews, "recording_preview", preview)
    preview.joint_notifier = Mock()
    preview.start()
    robot = SimpleNamespace(
        left_arm=object(), right_arm=object(), get_observation=Mock(side_effect=AssertionError("extra read"))
    )
    family = registry.get("metal")
    tap = previews.observation_tap(robot, family)
    observation = {
        "left_shoulder_pan.pos": 20.0,
        "right_shoulder_pan.pos": -10.0,
        "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    tap(observation)
    robot.get_observation.assert_not_called()
    data = preview.joint_notifier.call_args.args[0]
    assert data["joints_deg"] == {"shoulder_pan": 20.0}
    assert data["joints_deg_right"] == {"shoulder_pan": -10.0}
    assert data["joints"] == family.urdf_joint_positions({"shoulder_pan": 20.0})
    assert preview.jpeg("left_wrist") is not None


def test_tap_failure_cannot_interrupt_recording(monkeypatch):
    preview = previews.RecordingPreview()
    monkeypatch.setattr(previews, "recording_preview", preview)
    monkeypatch.setattr(preview, "publish", Mock(side_effect=RuntimeError("preview failed")))
    previews.observation_tap(object(), registry.get("metal"))({})


def test_preview_endpoint_only_serves_session_snapshots(monkeypatch):
    preview = previews.RecordingPreview()
    monkeypatch.setattr(server, "recording_preview", preview)
    monkeypatch.setattr(cv2, "VideoCapture", Mock(side_effect=AssertionError("must not open hardware")))
    with TestClient(server.app) as client:
        assert client.get("/api/v1/recording-preview/wrist").status_code == 503
        preview.start()
        preview.publish({"wrist": np.zeros((8, 8, 3), dtype=np.uint8)})
        response = client.get("/api/v1/recording-preview/wrist")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert response.content.startswith(b"\xff\xd8")
        assert client.get("/api/v1/recording-preview/absent").status_code == 503
        preview.stop()
        assert client.get("/api/v1/recording-preview/wrist").status_code == 503
