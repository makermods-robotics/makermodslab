"""Pair RTC timing logs without assuming agreement between host clocks.

Run with --gpu-log PATH --robot-log PATH. This only reads saved text files.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def read_records(text, tag):
    records = defaultdict(list)
    for line in text.splitlines():
        if f"[{tag}]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line.split(f"[{tag}]", 1)[1]))
        try:
            stamp = int(fields.pop("obs_ts"))
            parsed = {key: float(value) for key, value in fields.items() if key.endswith("_ms")}
        except (KeyError, ValueError):
            continue  # Interrupted/truncated line is not a valid measurement.
        if not all(math.isfinite(value) and value >= 0 for value in parsed.values()):
            continue
        parsed["rtc_used"] = fields.get("rtc_used")
        records[stamp].append(parsed)
    return records


def percentiles(values):
    values = sorted(values)

    def at(fraction):
        index = (len(values) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return round(values[low] + (values[high] - values[low]) * (index - low), 2)

    return {"p50": at(0.5), "p95": at(0.95)}


def summarize(gpu_text, robot_text, skip_seconds=10, window_seconds=30):
    if skip_seconds < 0 or window_seconds <= 0:
        raise ValueError("skip_seconds must be nonnegative and window_seconds positive")
    gpu = read_records(gpu_text, "policy-timing")
    robot = read_records(robot_text, "robot-timing")
    # The ID is the robot's observation stamp, not a cross-host clock reading.
    start = min(robot, default=0) + skip_seconds * 1_000_000
    end = start + window_seconds * 1_000_000
    pairs = []
    ambiguous = 0
    for stamp in sorted(gpu.keys() & robot.keys()):
        if not start <= stamp < end:
            continue
        if len(gpu[stamp]) != 1 or len(robot[stamp]) != 1:
            ambiguous += 1
            continue  # Never guess which repeated prediction a reply belongs to.
        policy, reply = gpu[stamp][0], robot[stamp][0]
        if not {"queue_ms", "service_ms", "infer_ms"} <= policy.keys() or "rtt_ms" not in reply:
            continue
        pairs.append(
            {
                "rtt_ms": reply["rtt_ms"],
                "queue_ms": policy["queue_ms"],
                "service_ms": policy["service_ms"],
                "infer_ms": policy["infer_ms"],
                "residual_ms": reply["rtt_ms"] - policy["queue_ms"] - policy["service_ms"],
                "rtc_used": policy["rtc_used"],
            }
        )
    result = {
        "policy_records": sum(map(len, gpu.values())),
        "robot_records": sum(map(len, robot.values())),
        "paired_in_window": len(pairs),
        "ambiguous_ids_excluded": ambiguous,
        "skip_seconds": skip_seconds,
        "window_seconds": window_seconds,
        "warnings": [],
    }
    if not pairs:
        result["warnings"].append("No paired samples: use fresh instrumented logs from the same session.")
        return result
    result["milliseconds"] = {
        key: percentiles([pair[key] for pair in pairs])
        for key in ("rtt_ms", "queue_ms", "service_ms", "infer_ms", "residual_ms")
    }
    result["rtc_guided_pairs"] = sum(pair["rtc_used"] == "True" for pair in pairs)
    result["negative_residual_pairs"] = sum(pair["residual_ms"] < -5 for pair in pairs)
    if result["negative_residual_pairs"]:
        result["warnings"].append("Residual below -5 ms: check log pairing and timing boundaries.")
    if len(pairs) < 30:
        result["warnings"].append("Fewer than 30 pairs; collect a longer steady window before comparing p95.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--robot-log", type=Path, required=True)
    parser.add_argument("--skip-seconds", type=float, default=10)
    parser.add_argument("--window-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.skip_seconds < 0 or args.window_seconds <= 0:
        parser.error("--skip-seconds must be nonnegative and --window-seconds positive")
    result = summarize(
        args.gpu_log.read_text(errors="replace"),
        args.robot_log.read_text(errors="replace"),
        args.skip_seconds,
        args.window_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["paired_in_window"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
