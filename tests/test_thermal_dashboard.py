import copy
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tools.thermal_dashboard import Dashboard, handler_for, prepare_loop


def recording():
    return {
        "action_names": ["shoulder_lift.pos", "gripper.pos"],
        "timestamps": [0.0, 0.1, 0.2],
        "values": [[-3.0, -4.0], [-20.0, -8.0], [-3.0, -10.5]],
    }


def test_preparation_preserves_recording_and_closes_loop_at_bounded_speed():
    original = recording()
    before = copy.deepcopy(original)
    data, report = prepare_loop(original, {"shoulder_lift": (-175.3, -3.2), "gripper": (-120, -2.5)})
    assert original == before
    assert data["values"][0] == data["values"][-1]
    assert report["target_adjustments"]["shoulder_lift.pos"]["max_adjustment_deg"] == pytest.approx(0.2)
    for i in range(3, len(data["values"])):
        dt = data["timestamps"][i] - data["timestamps"][i - 1]
        assert (
            max(abs(a - b) / dt for a, b in zip(data["values"][i], data["values"][i - 1], strict=True)) <= 20
        )


def test_large_adjustment_or_arm_endpoint_mismatch_refused():
    limits = {"shoulder_lift": (-175.3, -3.2), "gripper": (-120, -2.5)}
    s = recording()
    s["values"][1][0] = 0
    with pytest.raises(ValueError, match="exceed configured limits"):
        prepare_loop(s, limits)
    s = recording()
    s["values"][-1][0] = -15
    with pytest.raises(ValueError, match="end/start mismatch"):
        prepare_loop(s, limits)


def test_dashboard_controls_require_same_origin_and_stop_does_not_release():
    dashboard = Dashboard(1800, "shoulder_kp_85")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(dashboard, 8092))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/status") as response:
            assert response.status == 200
        assert dashboard.seen.is_set()
        with urllib.request.urlopen(base + "/") as response:
            html = response.read().decode()
        assert "110°C stop / 135°C reference" in html
        assert "0:00 / 30:00" in html
        assert "__STOP_AT_C__" not in html
        assert dashboard.snapshot(0)["status"]["stop_at_c"] == 110
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(base + "/api/stop", method="POST"))
        assert exc.value.code == 403
        assert not dashboard.stop.is_set()
        with urllib.request.urlopen(
            urllib.request.Request(
                base + "/api/stop", method="POST", headers={"Origin": "http://127.0.0.1:8092"}
            )
        ) as response:
            assert response.status == 200
        assert dashboard.stop.is_set()
        assert not dashboard.release.is_set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_explicit_test_copy_clamps_gripper_and_smoothly_bridges_arm():
    source = recording()
    source["values"][1][1] = -120
    source["values"][-1][0] = -10.9
    before = copy.deepcopy(source)
    limits = {"shoulder_lift": (-175.3, -3.2), "gripper": (-96, -2.5)}
    data, report = prepare_loop(source, limits, prepare_recorded_loop=True)
    assert source == before
    assert min(row[1] for row in data["values"]) == -96
    assert report["target_adjustments"]["gripper.pos"]["max_adjustment_deg"] == 24
    assert report["reset_duration_s"] >= 2
    assert data["values"][0] == data["values"][-1]
    for i in range(3, len(data["values"])):
        dt = data["timestamps"][i] - data["timestamps"][i - 1]
        assert (
            max(abs(a - b) / dt for a, b in zip(data["values"][i], data["values"][i - 1], strict=True)) <= 10
        )
    source["values"][-1][0] = -30
    with pytest.raises(ValueError, match="end/start mismatch"):
        prepare_loop(source, limits, prepare_recorded_loop=True)
