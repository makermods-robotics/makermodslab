#!/usr/bin/env python3
"""Standalone supervised Maker endurance test and local dashboard.

No MakerModsLab replay session or web panel. The existing driver, saved calibration,
and tested thermal guard are reused. Hardware moves only with --run and --ready.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from makermodslab.thermal_limits import (  # noqa: E402
    CRITICAL_AT_C,
    DEFAULT_TEST_DURATION_S,
    MAX_TEST_DURATION_S,
    STOP_AT_C,
    thermal_policy,
)


def prepare_loop(series, limits, *, prepare_recorded_loop=False):
    """Make a logged, limited copy; bridge endpoints without modifying the dataset."""
    from makermodslab.thermal_replay import validate_thermal_series

    data = copy.deepcopy(series)
    changes = {}
    for index, name in enumerate(data["action_names"]):
        low, high = limits[name.removesuffix(".pos")]
        count, worst = 0, 0.0
        for row in data["values"]:
            value = row[index]
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite recorded target: {name}")
            clipped = max(low, min(high, value))
            if clipped != value:
                worst = max(worst, abs(clipped - value))
                count += 1
                row[index] = clipped
        allowed_clip = 30 if prepare_recorded_loop and name == "gripper.pos" else 1
        if worst > allowed_clip:
            raise ValueError(f"{name}: targets exceed configured limits by {worst:.2f} degrees")
        if count:
            changes[name] = {"clipped_frames": count, "max_adjustment_deg": worst}
    first, last = data["values"][0], data["values"][-1]
    # A large mismatch deserves a new recording, not an invented return route.
    for name, a, b in zip(data["action_names"], first, last, strict=True):
        bound = 10 if prepare_recorded_loop or name == "gripper.pos" else 2
        if abs(a - b) > bound:
            raise ValueError(f"{name}: end/start mismatch {abs(a - b):.2f} degrees exceeds {bound}")
    # Smoothstep closes the loop with zero slope at both endpoints. Its maximum
    # slope is 1.5, included in the duration calculation to stay <=20 deg/s.
    speed = 10.0 if prepare_recorded_loop else 20.0
    duration = max(
        2.0 if prepare_recorded_loop else 1.0,
        1.5 * max(abs(a - b) for a, b in zip(first, last, strict=True)) / speed,
    )
    steps = math.ceil(duration * 30)
    end = data["timestamps"][-1]
    for step in range(1, steps + 1):
        u = step / steps
        blend = u * u * (3 - 2 * u)
        data["timestamps"].append(end + duration * u)
        data["values"].append([b + (a - b) * blend for a, b in zip(first, last, strict=True)])
    validate_thermal_series(data, "maker", "single")
    return data, {
        "target_adjustments": changes,
        "reset_duration_s": duration,
        "original_duration_s": end,
        "loop_duration_s": end + duration,
        "preparation_explicitly_requested": prepare_recorded_loop,
        "return_max_speed_deg_s": speed,
        "endpoint_delta_deg": dict(
            zip(data["action_names"], [a - b for a, b in zip(first, last, strict=True)], strict=True)
        ),
    }


class Dashboard:
    def __init__(self, duration, experiment):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.release = threading.Event()
        self.seen = threading.Event()
        self.last_seen = time.monotonic()
        self.sequence = 0
        self.history = deque(maxlen=20000)
        self.status = {
            "phase": "waiting",
            "result": "waiting",
            "message": "Waiting for dashboard",
            "elapsed_s": 0,
            "duration_s": duration,
            "cycles": 0,
            "experiment": experiment,
            "actuators": [],
            "critical": False,
            "rest_reached": False,
            "motor_connection_closed": True,
            **thermal_policy(),
        }
        self.trial = None

    def update(self, phase, status):
        with self.lock:
            self.status = copy.deepcopy({**status, "phase": phase})
            if self.trial:
                self.status["motors"] = {
                    name: {
                        "max_temperature_c": s["max_temperature_c"],
                        "peak_abs_torque_nm": s["peak_abs_torque_nm"],
                        "rms_torque_nm": math.sqrt(s["sum_torque_squared"] / s["samples"]),
                    }
                    for name, s in self.trial.stats.items()
                    if s["samples"]
                }
            self.sequence += 1
            self.history.append(
                {
                    "sequence": self.sequence,
                    "time": time.time(),
                    "elapsed_s": status.get("elapsed_s", 0),
                    "actuators": copy.deepcopy(status.get("actuators", [])),
                }
            )

    def snapshot(self, since):
        with self.lock:
            self.last_seen = time.monotonic()
            self.seen.set()
            return {
                "status": copy.deepcopy(self.status),
                "sequence": self.sequence,
                "points": [p for p in self.history if p["sequence"] > since],
            }


def handler_for(dashboard, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(
                    200,
                    Path(__file__)
                    .with_suffix(".html")
                    .read_text()
                    .replace("__STOP_AT_C__", str(STOP_AT_C))
                    .replace("__CRITICAL_AT_C__", str(CRITICAL_AT_C))
                    .replace("__TEST_SECONDS__", str(int(dashboard.status["duration_s"])))
                    .replace("__TEST_MINUTES__", str(int(dashboard.status["duration_s"] / 60)))
                    .encode(),
                    "text/html; charset=utf-8",
                )
            elif self.path.startswith("/api/status"):
                from urllib.parse import parse_qs, urlparse

                try:
                    since = int(parse_qs(urlparse(self.path).query).get("since", [0])[0])
                except ValueError:
                    since = 0
                self.respond(
                    200, json.dumps(dashboard.snapshot(since), allow_nan=False).encode(), "application/json"
                )
            else:
                self.respond(404, b"Not found", "text/plain")

        def do_POST(self):
            # Browser-only controls, same origin. No cross-origin control requests.
            if self.headers.get("Origin") not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                self.respond(403, b"Same-origin dashboard required", "text/plain")
                return
            if self.path == "/api/stop":
                dashboard.stop.set()
            elif self.path == "/api/release":
                dashboard.stop.set()
                dashboard.release.set()
            else:
                self.respond(404, b"Not found", "text/plain")
                return
            self.respond(200, b'{"ok":true}', "application/json")

    return Handler


def load_test(args):
    from makermodslab.maker_can import install

    install()
    from makermodslab.arms import registry
    from makermodslab.datasets import get_episode_action_series, read_dataset_robot_type
    from makermodslab.utils.config import get_robot_record, setup_follower_calibration_file

    record = get_robot_record(args.robot)
    if not record or record["arm_type"] != "maker" or record.get("mode") != "single":
        raise ValueError("Select a saved single-Maker robot")
    if read_dataset_robot_type(args.dataset) != "maker_follower":
        raise ValueError("Dataset must explicitly identify a Maker follower")
    config = registry.get("maker").single_follower_config(
        record["follower_port"], setup_follower_calibration_file(record["follower_config"], "maker")
    )
    original = get_episode_action_series(args.dataset, args.episode)
    if not original or not original["values"]:
        raise ValueError("Episode not found")
    series, preparation = prepare_loop(
        original, config.joint_limits, prepare_recorded_loop=args.prepare_recorded_loop
    )
    return config, series, preparation


def run_hardware(args, dashboard, config, series, preparation):
    from lerobot.robots import make_robot_from_config
    from makermodslab.thermal_replay import ThermalReplayOptions, ThermalTrial, mark_connection_closed

    robot = None
    trial = None
    diagnostics = None
    try:
        # The standalone runner must not race the web app for the same bus.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("Close the MakerModsLab backend before this standalone hardware test")
        if not dashboard.seen.wait(60):
            raise RuntimeError("Dashboard was not opened; no motor connection made")
        dashboard.update(
            "countdown", {**dashboard.status, "message": "Starting in 3 seconds — keep the return path clear"}
        )
        if dashboard.stop.wait(3):
            raise RuntimeError("Cancelled before motor connection")
        robot = make_robot_from_config(config)
        from makermodslab.thermal_diagnostics import ThermalDiagnostics

        diagnostics = ThermalDiagnostics(robot.bus)
        robot.connect(calibrate=False)
        trial = ThermalTrial(
            robot,
            series,
            ThermalReplayOptions(
                duration_s=args.duration, experiment=args.experiment, rest_pose_confirmed=True
            ),
            dashboard.stop,
            dashboard.release,
            dashboard.update,
            output=args.output,
            source={
                "repo_id": args.dataset,
                "episode_index": args.episode,
                "robot_name": args.robot,
                "preparation": preparation,
                "runner": "standalone-dashboard",
            },
        )
        dashboard.trial = trial
        result = trial.run()
        if not result["rest_reached"] and not dashboard.release.is_set():
            try:
                trial.sample(returning=True)
                robot.send_action({f"{n}.pos": trial.observation[f"{n}.pos"] for n in trial.names})
            except Exception:
                pass
            trial.log = (trial.root / "samples.jsonl").open("a", buffering=1)
            dashboard.update("return_failed", result)
            while not dashboard.release.wait(0.1):
                with suppress(Exception):
                    trial.sample(returning=True)
                dashboard.update("return_failed", trial.status)
        if trial.log:
            trial.log.close()
            trial.log = None
        result = dict(trial.status)
        if result["rest_reached"] and not dashboard.release.is_set():
            diagnostics.read_fault_status()
        robot.disconnect()
        robot = None
        result = mark_connection_closed(result)
        trial.status.update(result)
        trial.write_summary()
        dashboard.update("done", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        dashboard.update("error", {**dashboard.status, "result": "error", "message": str(exc)})
        print(f"Test error: {exc}", file=sys.stderr, flush=True)
    finally:
        # A normal failed return is held above until explicit release. This path
        # covers connection failure or a driver exception on teardown.
        if robot is not None:
            with suppress(Exception):
                robot.disconnect()
        if diagnostics is not None:
            if trial is not None:
                (trial.root / "diagnostics.json").write_text(json.dumps(diagnostics.snapshot(), indent=2))
            diagnostics.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="metal_dlabs_thermal")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Show the new settings without loading an episode or connecting motors",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--prepare-recorded-loop",
        action="store_true",
        help="Explicitly allow gripper clipping up to 30 degrees and endpoint bridging up to 10 degrees at <=10 deg/s",
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_TEST_DURATION_S)
    parser.add_argument("--experiment", choices=["baseline", "shoulder_kp_85"], default="shoulder_kp_85")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--output", default="output/thermal-replay")
    parser.add_argument("--run", action="store_true", help="Connect and execute the physical test")
    parser.add_argument(
        "--ready", action="store_true", help="Supported rest, empty gripper, clear path, operator present"
    )
    args = parser.parse_args()
    if not 10 <= args.duration <= MAX_TEST_DURATION_S:
        parser.error(f"Duration must be between 10 and {MAX_TEST_DURATION_S} seconds")
    if args.dashboard_only:
        if args.run:
            parser.error("--dashboard-only cannot be combined with --run")
        dashboard = Dashboard(args.duration, args.experiment)
        dashboard.status["message"] = (
            f"Awaiting your new recording. {args.duration / 60:g}-minute test; return and alert at {STOP_AT_C}°C. No motors connected."
        )
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dashboard, args.port))
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Dashboard only: {url}", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
        return
    if not args.dataset:
        parser.error("--dataset is required unless --dashboard-only is used")
    config, series, preparation = load_test(args)
    print(json.dumps(preparation, indent=2), flush=True)
    if not args.run:
        print("Preflight only. No motors connected. Use --run --ready for hardware.")
        return
    if not args.ready:
        parser.error("--run requires --ready")
    dashboard = Dashboard(args.duration, args.experiment)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dashboard, args.port))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def stop_signal(_sig, _frame):
        if dashboard.stop.is_set():
            dashboard.release.set()
        else:
            dashboard.stop.set()

    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    worker = threading.Thread(
        target=run_hardware, args=(args, dashboard, config, series, preparation), daemon=False
    )
    worker.start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard: {url}", flush=True)
    webbrowser.open(url)
    while worker.is_alive():
        if dashboard.seen.is_set() and time.monotonic() - dashboard.last_seen > 10:
            dashboard.stop.set()
        worker.join(0.25)
    print("Test ended. Dashboard remains available; Ctrl+C exits.", flush=True)
    # Restore normal signal behavior after the motor connection is closed.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
