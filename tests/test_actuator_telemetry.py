"""Monitor boundaries, missing feedback, and no-I/O integration with real preview paths."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from makermodslab.actuator_telemetry import cached_maker_telemetry, temperature_status
from tools.actuator_monitor import Monitor


def arm(temperature=101, torque=7, stamp=100):
    bus = SimpleNamespace(
        motors={"shoulder_lift": object()},
        _last_known_states={"shoulder_lift": {"temp_mos": temperature, "torque": torque}},
        last_feedback_time={"shoulder_lift": stamp},
        read=Mock(side_effect=AssertionError("must not read hardware")),
        connect=Mock(side_effect=AssertionError("must not connect")),
    )
    return SimpleNamespace(bus=bus)


@pytest.mark.parametrize(
    "temperature,status",
    [
        (None, "NO DATA"),
        (65, "OK"),
        (70, "OK"),
        (99.99, "OK"),
        (100, "OVERHEATING"),
        (134.99, "OVERHEATING"),
        (135, "CRITICAL"),
    ],
)
def test_inclusive_user_thresholds(temperature, status):
    assert temperature_status(temperature) == status


def test_cache_snapshot_handles_both_arms_and_holding_wrapper_without_bus_io():
    left, right = arm(), arm(30)
    original = left.bus
    left.bus = SimpleNamespace(_base=original)
    rows = cached_maker_telemetry(SimpleNamespace(left_arm=left, right_arm=right), now=100.2)["actuators"]
    assert [(r["actuator"], r["status"]) for r in rows] == [
        ("left.shoulder_lift", "OVERHEATING"),
        ("right.shoulder_lift", "OK"),
    ]
    assert rows[0]["torque_nm"] == 7
    original.read.assert_not_called()
    original.connect.assert_not_called()


@pytest.mark.parametrize(
    "temperature,stamp,status",
    [
        (0, None, "NO DATA"),
        (35, 98, "STALE"),
        (float("nan"), 100, "INVALID"),
        (3342, 100, "INVALID"),
        (35, 101, "INVALID"),
    ],
)
def test_missing_stale_and_bad_decoder_data_never_look_safe(temperature, stamp, status):
    row = cached_maker_telemetry(arm(temperature=temperature, stamp=stamp), now=100)["actuators"][0]
    assert row["status"] == status
    if stamp is None:
        assert row["temperature_c"] is None and row["torque_nm"] is None
    json.dumps(row, allow_nan=False)


def test_teleop_and_recording_share_the_cache_only_payload(monkeypatch):
    from makermodslab import recording_preview as previews
    from makermodslab.arms import registry
    from makermodslab.teleoperate import get_can_joint_data

    robot = arm(stamp=time.time())
    robot.get_observation = Mock(side_effect=AssertionError("must reuse supplied observation"))
    family = registry.get("maker")
    observation = {"shoulder_lift.pos": 20.0}
    data = get_can_joint_data(robot, family, False, time.time(), observation=observation)
    assert data["actuator_telemetry"]["actuators"][0]["status"] == "OVERHEATING"
    preview = previews.RecordingPreview()
    preview.start()
    preview.joint_notifier = Mock()
    monkeypatch.setattr(previews, "recording_preview", preview)
    previews.observation_tap(robot, family)(observation)
    assert preview.joint_notifier.call_args.args[0]["actuator_telemetry"]["actuators"][0]["torque_nm"] == 7
    robot.get_observation.assert_not_called()
    robot.bus.read.assert_not_called()


def packet(stamp, torque=3, temp=101):
    return {
        "actuators": [
            {
                "actuator": "left.shoulder_lift",
                "feedback_ts": stamp,
                "torque_nm": torque,
                "temperature_c": temp,
                "status": "OK",
            }
        ]
    }


def test_monitor_deduplicates_recomputes_thresholds_and_expires_disconnected_data():
    monitor = Monitor()
    assert len(monitor.ingest(packet(100), now=100)) == 1
    assert monitor.ingest(packet(100), now=100.1) == []
    monitor.ingest(packet(100.5, torque=4, temp=135), now=100.5)
    stats = monitor.summary()["left.shoulder_lift"]
    assert stats["samples"] == 2
    assert stats["rms_torque_nm"] == pytest.approx(math.sqrt(12.5))
    assert stats["samples_at_or_above_100"] == 2 and stats["samples_at_or_above_135"] == 1
    assert monitor.rows(now=100.6)[0]["status"] == "CRITICAL"
    assert monitor.rows(now=102)[0]["status"] == "STALE"
    assert monitor.rows(now=131)[0]["rms_30s_nm"] is None


def test_monitor_reports_gaps_and_rejects_backward_clock():
    monitor = Monitor()
    monitor.ingest(packet(100), now=100)
    monitor.ingest(packet(105), now=105)
    assert monitor.summary()["left.shoulder_lift"]["feedback_gap_seconds"] == 5
    assert monitor.ingest(packet(104.5), now=105) == []
    assert monitor.rows(now=105)[0]["status"] == "INVALID"
    assert monitor.summary()["left.shoulder_lift"]["samples"] == 2


def test_demo_cli_exports_seven_actuators_and_explicit_simulation_metadata(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/actuator_monitor.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo",
            "--duration",
            "0.15",
            "--output",
            str(tmp_path),
            "--no-color",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["source"] == "SIMULATED"
    assert summary["requested_duration_completed"] is True
    assert len(summary["actuators"]) == 7
    assert "SIMULATED DEMO" in result.stdout


def test_original_66c_reading_is_below_the_new_software_limit():
    # Regression fixture: measured shoulder values in the episode reviewed with the user.
    temperatures = [56, 57, 61, 61, 62, 64, 61, 66]
    monitor = Monitor()
    for i, temp in enumerate(temperatures):
        monitor.ingest(packet(100 + i * 10, temp=temp), now=100 + i * 10)
    stats = monitor.summary()["left.shoulder_lift"]
    assert stats["max_temperature_c"] == 66
    assert stats["samples_at_or_above_100"] == 0 and stats["samples_at_or_above_135"] == 0
