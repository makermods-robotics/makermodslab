"""Timing claims must distinguish observed buffers from sensor exposure."""

import threading
from types import SimpleNamespace

import pytest

from makermodslab.drtc._diagnostics import CameraTimingMonitor, parameter_summary, startup_stage


def test_camera_monitor_counts_updates_not_repeated_polls():
    camera = SimpleNamespace(
        frame_lock=threading.Lock(),
        latest_timestamp=10.0,
        latest_frame=SimpleNamespace(shape=(480, 640, 3)),
    )
    monitor = CameraTimingMonitor({"front": camera})
    monitor.sample()
    monitor.sample()
    camera.latest_timestamp = 10.1
    monitor.sample()
    assert len(monitor.samples["front"]) == 2
    assert "observed_hz=10.0" in monitor.summary(now=10.15)
    assert "buffer_age_ms=50.0" in monitor.summary(now=10.15)
    assert "size=640x480" in monitor.summary(now=10.15)
    assert "observed_hz=0.0" in monitor.summary(now=20.0)


def test_camera_monitor_skips_busy_and_unsupported_cameras():
    lock = threading.Lock()
    camera = SimpleNamespace(frame_lock=lock, latest_timestamp=1.0, latest_frame=None)
    monitor = CameraTimingMonitor({"busy": camera, "unsupported": object()})
    with lock:
        monitor.sample()  # Must not block the camera writer or control loop.
    assert monitor.summary(now=1.1) == "busy:unavailable | unsupported:unavailable"


def test_parameter_summary_reports_mixed_loaded_precision():
    parameters = [
        SimpleNamespace(dtype="bfloat16", device="cuda:0", numel=lambda: 12),
        SimpleNamespace(dtype="float32", device="cpu", numel=lambda: 3),
    ]
    policy = SimpleNamespace(parameters=lambda: iter(parameters))
    assert parameter_summary(policy) == "bfloat16@cuda:0=12, float32@cpu=3"


def test_failed_startup_stage_does_not_report_success(capsys):
    with pytest.raises(ValueError), startup_stage("load"):
        raise ValueError("bad checkpoint")
    output = capsys.readouterr().out
    assert "load: failed" in output
    assert "load: done" not in output
