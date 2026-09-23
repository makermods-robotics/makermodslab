#!/usr/bin/env python3
"""Live MakerLab temperature/torque monitor. Subscribes to feedback; never opens CAN.

Run with the MakerLab environment:
  .venv/bin/python tools/actuator_monitor.py --duration 600 --label baseline
  .venv/bin/python tools/actuator_monitor.py --demo --duration 15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

# Permit direct execution from any current directory, without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from makermodslab.actuator_telemetry import STALE_AFTER_S, finite_number, temperature_status  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
VALID = {"OK", "OVERHEATING", "CRITICAL"}


class Monitor:
    """Track distinct fresh motor samples, not repeated reads of cached feedback."""

    def __init__(self):
        self.latest = {}
        self.history = {}
        self.stats = {}
        self.last_stamp = {}
        self.first_received = None

    @staticmethod
    def status(sample, now):
        stamp = finite_number(sample.get("feedback_ts"))
        temperature = finite_number(sample.get("temperature_c"))
        torque = finite_number(sample.get("torque_nm"))
        if stamp is None or stamp <= 0:
            return "NO DATA"
        if stamp > now or sample.get("status") == "INVALID":
            return "INVALID"
        if temperature is None or not 0 <= temperature <= 409.5 or torque is None:
            return "INVALID"
        if now - stamp > STALE_AFTER_S:
            return "STALE"
        return temperature_status(temperature)

    def ingest(self, packet, now=None):
        now = time.time() if now is None else now
        recorded = []
        for source in packet.get("actuators", []):
            if not isinstance(source, dict) or not isinstance(source.get("actuator"), str):
                continue
            sample = dict(source)
            key = sample["actuator"]
            state = self.status(sample, now)
            sample["status"] = state
            self.latest[key] = sample
            stamp = finite_number(sample.get("feedback_ts"))
            if state not in VALID or self.last_stamp.get(key) == stamp:
                continue
            temperature = float(sample["temperature_c"])
            torque = float(sample["torque_nm"])
            previous = self.last_stamp.get(key)
            if previous is not None and stamp < previous:
                # A clock reset is not a valid continuation of this measurement.
                sample["status"] = "INVALID"
                sample["reason"] = "Feedback clock moved backwards; start a new monitor run"
                continue
            self.last_stamp[key] = stamp
            if self.first_received is None:
                self.first_received = now
            hist = self.history.setdefault(key, deque())
            hist.append((stamp, torque))
            while hist and hist[0][0] < stamp - 30:
                hist.popleft()
            stats = self.stats.setdefault(
                key,
                {
                    "samples": 0,
                    "sum_torque_squared": 0.0,
                    "max_abs_torque_nm": 0.0,
                    "start_temperature_c": temperature,
                    "max_temperature_c": temperature,
                    "samples_above_65": 0,
                    "samples_above_70": 0,
                    "first_feedback_ts": stamp,
                    "last_feedback_ts": stamp,
                    "first_overheat_ts": None,
                    "first_critical_ts": None,
                    "feedback_gap_count": 0,
                    "feedback_gap_seconds": 0.0,
                },
            )
            stats["samples"] += 1
            stats["sum_torque_squared"] += torque**2
            stats["max_abs_torque_nm"] = max(stats["max_abs_torque_nm"], abs(torque))
            stats["max_temperature_c"] = max(stats["max_temperature_c"], temperature)
            stats["last_feedback_ts"] = stamp
            for threshold, label in ((65, "overheat"), (70, "critical")):
                if temperature > threshold:
                    stats[f"samples_above_{threshold}"] += 1
                    if stats[f"first_{label}_ts"] is None:
                        stats[f"first_{label}_ts"] = stamp
            if previous is not None and stamp - previous > STALE_AFTER_S:
                stats["feedback_gap_count"] += 1
                stats["feedback_gap_seconds"] += stamp - previous
            recorded.append({"received_ts": now, **sample})
        return recorded

    def rows(self, now=None):
        now = time.time() if now is None else now
        result = []
        for name, sample in self.latest.items():
            recent = [q for ts, q in self.history.get(name, ()) if ts >= now - 30]
            result.append(
                {
                    **sample,
                    "status": self.status(sample, now),
                    "rms_30s_nm": math.sqrt(sum(q * q for q in recent) / len(recent)) if recent else None,
                    "peak_temperature_c": self.stats.get(name, {}).get("max_temperature_c"),
                }
            )
        return result

    def summary(self):
        result = {}
        for key, original in self.stats.items():
            stats = dict(original)
            stats["rms_torque_nm"] = math.sqrt(stats.pop("sum_torque_squared") / stats["samples"])
            result[key] = stats
        return result


def draw(monitor, message, *, color, elapsed, bell, previous):
    def number(value, width=8):
        return f"{value:{width}.2f}" if value is not None else f"{'—':>{width}}"

    lines = [
        "MAKER ARM — LIVE TEMPERATURE + TORQUE",
        message,
        "Above 65°C: OVERHEATING  |  Above 70°C: CRITICAL  |  Alerts only; does not stop motors.",
        f"Run elapsed: {elapsed:.1f}s  |  RMS: distinct feedback samples in the last 30s",
        "",
        f"{'Actuator':<29} {'Temp °C':>8} {'Torque Nm':>10} {'RMS Nm':>8} {'Peak °C':>8}  State",
    ]
    rows = monitor.rows()
    for row in rows:
        state = row["status"]
        text = (
            f"{row['actuator']:<29} {number(row.get('temperature_c'))} "
            f"{number(row.get('torque_nm'), 10)} {number(row['rms_30s_nm'])} "
            f"{number(row['peak_temperature_c'])}  {state}"
        )
        if state in {"STALE", "INVALID", "NO DATA"}:
            text += " — readings unavailable or not current"
        if color:
            code = {
                "OVERHEATING": "31",
                "CRITICAL": "1;97;41",
                "STALE": "33",
                "INVALID": "33",
                "NO DATA": "33",
            }.get(state, "0")
            text = f"\033[{code}m{text}\033[0m"
        if state in {"OVERHEATING", "CRITICAL"} and previous.get(row["actuator"]) != state and bell:
            sys.stdout.write("\a")
        previous[row["actuator"]] = state
        lines.append(text)
    if not rows:
        lines.append(
            "Waiting for MakerLab teleop/recording feedback. No motors have been read by this monitor."
        )
    if color:
        sys.stdout.write("\033[H\033[2J")
    print("\n".join(lines), flush=True)


def demo_packet(elapsed, now):
    """Explicitly synthetic input for testing the UI, never hardware telemetry."""
    return {
        "timestamp": now,
        "actuators": [
            {
                "actuator": f"follower.{name}",
                "feedback_ts": now,
                "temperature_c": min(73, 60 + elapsed) if name == "shoulder_lift" else 30 + i,
                "torque_nm": 6 + math.sin(elapsed * 3) if name == "shoulder_lift" else 0.15 * (i + 1),
                "status": "OK",
            }
            for i, name in enumerate(JOINTS)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/api/v1/ws/joint-data")
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Stop monitoring after this many seconds from first fresh data; 0 is unlimited. Does NOT stop the arm.",
    )
    parser.add_argument("--label", default="baseline", help="Describe this test/change in the run metadata")
    parser.add_argument("--output", type=Path, default=Path("output/actuator-monitor"))
    parser.add_argument(
        "--bell",
        action="store_true",
        help="Terminal bell when an actuator enters an overheating/critical state",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--demo", action="store_true", help="Synthetic readings; no connection to MakerLab or motors"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be a finite nonnegative number")
    run_dir = args.output / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "label": args.label,
        "source": "SIMULATED" if args.demo else args.url,
        "started_at": datetime.now().astimezone().isoformat(),
        "requested_duration_s": args.duration,
        "overheat_above_c": 65,
        "critical_above_c": 70,
        "stale_after_s": STALE_AFTER_S,
        "note": "Observer only. Records the MakerLab preview sampling rate (~20 Hz teleop/~10 Hz recording), not every CAN frame. No trajectory playback or automatic motor stop.",
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    monitor = Monitor()
    connection = None
    next_connect = 0.0
    last_draw = float("-inf")
    start = time.monotonic()
    first_fresh_mono = None
    previous = {}
    state_transitions = {}
    message = "SIMULATED DEMO — NOT CONNECTED TO MOTORS" if args.demo else f"Connecting to {args.url}"
    color = sys.stdout.isatty() and not args.no_color
    error = None
    completed = False
    fields = ["received_ts", "actuator", "feedback_ts", "temperature_c", "torque_nm", "status"]
    with (
        (run_dir / "samples.csv").open("w", newline="") as samples,
        (run_dir / "events.jsonl").open("w") as events,
    ):
        writer = csv.DictWriter(samples, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        try:
            while True:
                now_mono = time.monotonic()
                packet = None
                if args.demo:
                    packet = demo_packet(now_mono - start, time.time())
                    time.sleep(0.05)
                else:
                    if connection is None and now_mono >= next_connect:
                        try:
                            from websockets.sync.client import connect

                            connection = connect(
                                args.url,
                                open_timeout=1,
                                close_timeout=1,
                                max_size=2**20,
                                max_queue=16,
                                proxy=None,
                            )
                            message = "Connected. Waiting for a MakerLab teleop/recording session."
                        except ImportError as exc:
                            raise RuntimeError(
                                "Run this script with MakerLab's .venv/bin/python (websockets is required)."
                            ) from exc
                        except (OSError, TimeoutError, ValueError) as exc:
                            message = f"Connection unavailable: {exc} — retrying"
                            next_connect = time.monotonic() + 2
                    if connection is not None:
                        try:
                            envelope = json.loads(connection.recv(timeout=0.1))
                            packet = (
                                envelope.get("actuator_telemetry") if isinstance(envelope, dict) else None
                            )
                            if packet is not None:
                                message = "Live MakerLab cached feedback — no CAN commands from this monitor"
                        except TimeoutError:
                            pass
                        except Exception as exc:
                            message = f"Feedback stream interrupted: {exc} — retrying"
                            connection.close()
                            connection = None
                            next_connect = time.monotonic() + 2
                    else:
                        time.sleep(0.1)
                if isinstance(packet, dict) and isinstance(packet.get("actuators"), list):
                    records = monitor.ingest(packet)
                    writer.writerows(records)
                    if records and first_fresh_mono is None:
                        first_fresh_mono = time.monotonic()
                now_mono = time.monotonic()
                elapsed = 0.0 if first_fresh_mono is None else now_mono - first_fresh_mono
                for row in monitor.rows():
                    key, state = row["actuator"], row["status"]
                    if state_transitions.get(key) != state:
                        events.write(
                            json.dumps(
                                {
                                    "ts": time.time(),
                                    "actuator": key,
                                    "status": state,
                                    "temperature_c": row.get("temperature_c"),
                                }
                            )
                            + "\n"
                        )
                        state_transitions[key] = state
                if now_mono - last_draw >= (0.25 if color else 1):
                    draw(monitor, message, color=color, elapsed=elapsed, bell=args.bell, previous=previous)
                    samples.flush()
                    events.flush()
                    last_draw = now_mono
                if args.duration and first_fresh_mono is not None and elapsed >= args.duration:
                    completed = True
                    break
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            error = str(exc)
        finally:
            if connection is not None:
                connection.close()
            stats = monitor.summary()
            summary = {
                **metadata,
                "monitor_elapsed_s": time.monotonic() - start,
                "measurement_elapsed_s": 0
                if first_fresh_mono is None
                else time.monotonic() - first_fresh_mono,
                "requested_duration_completed": completed,
                "error": error,
                "actuators": stats,
                "final_states": {r["actuator"]: r["status"] for r in monitor.rows()},
                "overheated_actuators": [k for k, v in stats.items() if v["max_temperature_c"] > 65],
                "critical_actuators": [k for k, v in stats.items() if v["max_temperature_c"] > 70],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"\nMonitor stopped. Motor control was not changed. Results: {run_dir.resolve()}")
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
