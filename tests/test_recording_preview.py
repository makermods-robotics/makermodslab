"""Recording previews use captured observations, never another hardware read."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from makermodslab import recording_preview as previews, server
from makermodslab.arms import registry


@pytest.fixture(autouse=True)
def fake_jpeg_encoder(monkeypatch):
    # Exercise resize/color conversion, but never run a real codec.
    encode = Mock(return_value=(True, np.frombuffer(b"\xff\xd8preview", dtype=np.uint8)))
    monkeypatch.setattr(cv2, "imencode", encode)
    return encode


def test_preview_sample_is_bounded_owned_rgb_and_preserves_aspect(monkeypatch, fake_jpeg_encoder):
    preview = previews.RecordingPreview()
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    assert not preview.publish({"wrist": image})
    preview.start()
    assert preview.jpeg("wrist") is None  # First request activates sampling.
    assert preview.publish({"wrist": image, "shoulder.pos": 3.0})
    image[:] = 0  # source buffer reuse must not change the captured snapshot
    assert not preview.publish({"other": image})
    assert preview.jpeg("wrist") == b"\xff\xd8preview"
    bgr = fake_jpeg_encoder.call_args.args[1]
    assert bgr.shape == (360, 640, 3)
    assert bgr[0, 0, 2] == 255 and bgr[0, 0, 0] == 0
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
    assert not preview._cameras  # Joint telemetry does not create camera demand.


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
        assert client.get("/api/v1/recording-preview/wrist").status_code == 503
        preview.publish({"wrist": np.zeros((8, 8, 3), dtype=np.uint8)})
        response = client.get("/api/v1/recording-preview/wrist")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert response.content.startswith(b"\xff\xd8")
        assert client.get("/api/v1/recording-preview/absent").status_code == 503
        preview.stop()
        assert client.get("/api/v1/recording-preview/wrist").status_code == 503


def test_preview_demand_expires_per_camera_and_resumes(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    preview = previews.RecordingPreview()
    preview.start()

    class TrackedFrame(np.ndarray):
        def copy(self):
            copies.append(self)
            return super().copy()

    copies = []
    frame = np.zeros((8, 8, 3), np.uint8).view(TrackedFrame)
    observation = {"left": frame, "right": frame}
    preview.publish(observation)
    assert copies == []
    preview.jpeg("left")
    now[0] += 0.2
    preview.publish(observation)
    assert len(copies) == 1
    preview.jpeg("absent")
    now[0] += 2.1
    preview.publish(observation)
    assert len(copies) == 1
    assert preview._cameras == {}
    preview.jpeg("right")
    now[0] += 0.2
    preview.publish(observation)
    assert len(copies) == 2
    assert preview.jpeg("right") is not None


def test_concurrent_viewers_share_encode_without_blocking_publish(monkeypatch):
    preview = previews.RecordingPreview()
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    preview.start()
    preview.jpeg("wrist")
    preview.publish({"wrist": np.zeros((8, 8, 3), np.uint8)})
    entered, release = threading.Event(), threading.Event()

    def encode(frame):
        entered.set()
        assert release.wait(2)
        return b"shared"

    encoder = Mock(side_effect=encode)
    monkeypatch.setattr(preview, "_encode", encoder)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(preview.jpeg, "wrist")
        assert entered.wait(2)
        second = pool.submit(preview.jpeg, "wrist")
        # Control publication can proceed while HTTP encoding is in flight.
        now[0] += 0.2
        pending_frame = preview._cameras["wrist"]["frame"]
        assert pool.submit(preview.publish, {"wrist": np.ones((8, 8, 3), np.uint8)}).result(timeout=2)
        assert preview._cameras["wrist"]["frame"] is pending_frame
        release.set()
        assert first.result(timeout=2) == second.result(timeout=2) == b"shared"
    assert preview.jpeg("wrist") == b"shared"
    assert encoder.call_count == 1
    now[0] += 0.2
    preview.publish({"wrist": np.ones((8, 8, 3), np.uint8)})
    assert preview.jpeg("wrist") == b"shared"
    assert encoder.call_count == 2


def test_stop_restart_discards_inflight_jpeg(monkeypatch):
    preview = previews.RecordingPreview()
    preview.start()
    preview.jpeg("wrist")
    preview.publish({"wrist": np.zeros((8, 8, 3), np.uint8)})
    entered, release = threading.Event(), threading.Event()

    def encode(frame):
        entered.set()
        assert release.wait(2)
        return b"old session"

    monkeypatch.setattr(preview, "_encode", encode)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(preview.jpeg, "wrist")
        assert entered.wait(2)
        preview.stop()
        preview.start()
        release.set()
        assert pending.result(timeout=2) is None
    assert preview.jpeg("wrist") is None


@pytest.mark.parametrize("failure", [None, RuntimeError("codec unavailable")])
def test_failed_encode_releases_single_flight_lock(monkeypatch, failure):
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    preview = previews.RecordingPreview()
    preview.start()
    preview.jpeg("wrist")
    preview.publish({"wrist": np.zeros((8, 8, 3), np.uint8)})
    encode = Mock(side_effect=[failure, b"recovered"])
    monkeypatch.setattr(preview, "_encode", encode)
    if failure is None:
        assert preview.jpeg("wrist") is None
    else:
        with pytest.raises(RuntimeError, match="codec unavailable"):
            preview.jpeg("wrist")
    assert preview.jpeg("wrist") is None  # Failed raw snapshot was discarded.
    now[0] += 0.2
    preview.publish({"wrist": np.ones((8, 8, 3), np.uint8)})
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(preview.jpeg, "wrist").result(timeout=2) == b"recovered"
    assert encode.call_count == 2


def test_pending_snapshot_is_copied_once_until_consumed_per_camera(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(previews.time, "monotonic", lambda: now[0])
    preview = previews.RecordingPreview()
    preview.start()
    copies = []

    class TrackedFrame(np.ndarray):
        def copy(self):
            copies.append(int(self[0, 0, 0]))
            return super().copy()

    source = np.full((8, 8, 3), 7, np.uint8).view(TrackedFrame)
    preview.jpeg("left")
    preview.jpeg("right")
    preview.publish({"left": source, "right": source})
    source[:] = 19  # Producer reuse must not alter either pending snapshot.
    for _ in range(4):
        now[0] += 0.2
        assert preview.publish({"left": source, "right": source})  # Joint ticks continue.
    assert copies == [7, 7]
    pixels = []

    def encode(frame):
        pixels.append(int(frame[0, 0, 0]))
        return bytes([pixels[-1]])

    monkeypatch.setattr(preview, "_encode", encode)
    assert preview.jpeg("left") == bytes([7])
    assert preview.jpeg("left") == bytes([7])
    assert pixels == [7]  # Shared cached result, no second encode.
    now[0] += 0.2
    preview.publish({"left": source, "right": source})
    assert copies == [7, 7, 19]  # Right's unconsumed frame is still retained.
    assert preview.jpeg("right") == bytes([7])
    assert preview.jpeg("left") == bytes([19])
    assert pixels == [7, 7, 19]
