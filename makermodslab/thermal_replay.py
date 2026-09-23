"""Supervised, single-Maker thermal experiment. Never owns or opens a CAN bus.

The replay session owns connection, lease, and release. This runner uses only that
session's robot. Temperature feedback is a software guard, not a hardware interlock.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .actuator_telemetry import cached_maker_telemetry, finite_number
from .thermal_limits import (
    CRITICAL_AT_C,
    DEFAULT_TEST_DURATION_S,
    MAX_TEST_DURATION_S,
    STOP_AT_C,
    thermal_policy,
)


class ThermalReplayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_s: float = Field(
        default=DEFAULT_TEST_DURATION_S, ge=10, le=MAX_TEST_DURATION_S, allow_inf_nan=False
    )
    experiment: Literal["baseline", "shoulder_kp_85"] = "baseline"
    # Capturing an arbitrary pose does not make it safe to de-energize there.
    rest_pose_confirmed: Literal[True]


def validate_thermal_series(series: dict, arm_type: str, mode: str) -> None:
    """Run before connecting/enabling any motors."""
    if arm_type != "maker" or mode != "single":
        raise ValueError("Thermal tests currently support one Maker follower arm only.")
    names, stamps, frames = (series.get(k, []) for k in ("action_names", "timestamps", "values"))
    if not names or len(set(names)) != len(names) or any(not n.endswith(".pos") for n in names):
        raise ValueError("Thermal replay requires unique joint position actions.")
    if len(stamps) != len(frames) or len(frames) < 2:
        raise ValueError("Thermal replay requires at least two timestamped frames.")
    if any(finite_number(t) is None for t in stamps) or abs(stamps[0]) > 0.001:
        raise ValueError("Thermal replay timestamps must be finite and start at zero.")
    if any(not 0 < b - a <= 0.2 for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError("Thermal replay requires increasing timestamps with no gap above 200 ms.")
    if any(len(f) != len(names) or any(finite_number(v) is None for v in f) for f in frames):
        raise ValueError("Thermal replay contains missing or invalid joint positions.")
    if any(abs(a - b) > 2 for a, b in zip(frames[0], frames[-1], strict=True)):
        raise ValueError("Record a loop that starts and ends at the same resting pose (within 2 degrees).")


class TrialEndedError(Exception):
    def __init__(self, result: str, message: str):
        self.result = result
        super().__init__(message)


def mark_connection_closed(result):
    """Call only after disconnect returns; do not leave an energized warning latched."""
    result = dict(result, motor_connection_closed=True)
    if not result.get("rest_reached"):
        detail = result.get("message", "").split("Support the arm, then Release now.")[0].strip()
        result["message"] = detail + " Motor connection closed after release; return was not confirmed."
    return result


class ThermalTrial:
    """Run on the replay worker thread; all I/O remains serialized on that thread."""

    def __init__(
        self, robot, series, options, stop, release, update, broadcast=None, output=None, source=None
    ):
        self.robot, self.series, self.options = robot, series, options
        self.stop, self.release, self.update, self.broadcast = stop, release, update, broadcast
        self.root = Path(output or "output/thermal-replay") / (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        self.source = source or {}
        self.names = list(robot.bus.motors)
        self.rest = {}
        self.observation = {}
        self.last_action = {}
        self.motion_commanded = False
        self.original_kp = None
        self.started = None
        self.returning = False
        self.last_publish = 0.0
        self.bad_tracking_since = None
        self.hot = None
        self.stats = {}
        self.log = None
        self.status = {
            "result": "running",
            "message": "Checking feedback and resting pose",
            "elapsed_s": 0.0,
            "duration_s": options.duration_s,
            "cycles": 0,
            "experiment": options.experiment,
            "actuators": [],
            "rest_reached": False,
            "log_dir": str(self.root.resolve()),
            "critical": False,
            "motor_connection_closed": False,
            **thermal_policy(),
        }

    def publish(self, phase):
        self.update(phase, dict(self.status))

    def sample(self, *, returning=False):
        # Read all motors even with no browser/monitor attached. The driver can
        # fall back to cached positions, so independently check feedback age.
        self.observation = self.robot.get_observation()
        telemetry = cached_maker_telemetry(self.robot)
        rows = telemetry["actuators"]
        self.status["actuators"] = rows
        if self.started is not None and not returning:
            self.status["elapsed_s"] = min(time.monotonic() - self.started, self.options.duration_s)
        fault = None
        if len(rows) != len(self.names) or not rows:
            fault = "Missing actuator feedback"
        for row in rows:
            name = row["actuator"]
            if row["status"] in {"INVALID", "NO DATA", "STALE"} or row["feedback_age_s"] > 0.25:
                fault = f"{name}: missing, invalid, or older-than-250-ms feedback"
                continue
            temp, torque = row["temperature_c"], row["torque_nm"]
            stat = self.stats.setdefault(
                name,
                {
                    "max_temperature_c": temp,
                    "sum_torque_squared": 0.0,
                    "samples": 0,
                    "peak_abs_torque_nm": 0.0,
                    "last_ts": None,
                },
            )
            if row["feedback_ts"] != stat["last_ts"]:
                stat["max_temperature_c"] = max(stat["max_temperature_c"], temp)
                stat["peak_abs_torque_nm"] = max(stat["peak_abs_torque_nm"], abs(torque))
                stat["sum_torque_squared"] += torque * torque
                stat["samples"] += 1
                stat["last_ts"] = row["feedback_ts"]
                if self.log:
                    self.log.write(
                        json.dumps(
                            {
                                "type": "sample",
                                **row,
                                "position_deg": self.observation.get(
                                    row["actuator"].split(".", 1)[1] + ".pos"
                                ),
                                "returning": self.returning,
                                "phase": "return"
                                if self.returning
                                else "playback"
                                if self.started is not None
                                else "alignment",
                            }
                        )
                        + "\n"
                    )
            if temp >= STOP_AT_C and self.hot is None:
                self.hot = f"{name} reached {temp:.1f}°C (at or above {STOP_AT_C}°C)"
            if temp >= CRITICAL_AT_C:
                self.status["critical"] = True
        for name in self.names:
            if finite_number(self.observation.get(f"{name}.pos")) is None:
                fault = f"{name}: invalid position feedback"
        now = time.monotonic()
        if now - self.last_publish >= 0.1:
            self.last_publish = now
            self.publish(
                "stopping" if self.returning else "playing" if self.started is not None else "easing_in"
            )
            if self.broadcast:
                self.broadcast(
                    {
                        "type": "replay_joint_update",
                        "joints": self.observation,
                        "timestamp": time.time(),
                        "actuator_telemetry": telemetry,
                    }
                )
        if fault:
            raise TrialEndedError("feedback_fault", fault)
        if not returning:
            if self.hot:
                raise TrialEndedError("overheated", self.hot)
            if self.stop.is_set():
                raise TrialEndedError("stopped", "Stopped by operator or session lease")
            if self.last_action:
                error = max(abs(self.observation[k] - v) for k, v in self.last_action.items())
                self.status["max_tracking_error_deg"] = max(
                    self.status.get("max_tracking_error_deg", 0), error
                )
                self.bad_tracking_since = (self.bad_tracking_since or now) if error > 12 else None
                if self.bad_tracking_since is not None and now - self.bad_tracking_since > 0.5:
                    raise TrialEndedError("tracking_fault", "Tracking error exceeded 12° for 0.5 seconds")
            if self.started is not None and now - self.started >= self.options.duration_s:
                raise TrialEndedError("completed", "Test duration completed below the temperature threshold")

    def send(self, action):
        self.motion_commanded = True
        applied = self.robot.send_action(action)
        self.last_action = dict(action)
        # Clipping or startup synchronization changes the benchmark trajectory.
        if not isinstance(applied, dict) or any(
            finite_number(applied.get(k)) is None or abs(applied[k] - v) > 0.1 for k, v in action.items()
        ):
            raise TrialEndedError(
                "tracking_fault", "Driver changed a requested target; test cannot be compared"
            )
        if self.log:
            self.log.write(
                json.dumps({"type": "action", "timestamp": time.time(), "positions": action}) + "\n"
            )

    def move_to(self, pose, *, returning=True):
        """Controlled return at <=20 deg/s. Require fresh feedback and <=2° arrival."""
        self.sample(returning=returning)
        start = {k: self.observation[k] for k in pose}
        duration = max(abs(start[k] - v) for k, v in pose.items()) / 20.0
        t0 = time.monotonic()
        deadline = t0 + max(10.0, duration + 3.0)
        while time.monotonic() < deadline:
            if self.release.is_set():
                raise TrialEndedError("released", "Immediate release requested")
            self.sample(returning=returning)
            if returning:
                self.status["return_error_deg"] = {k: abs(self.observation[k] - v) for k, v in pose.items()}
            fraction = min(1.0, (time.monotonic() - t0 + 1 / 30) / max(duration, 1 / 30))
            action = {k: start[k] + (v - start[k]) * fraction for k, v in pose.items()}
            # Feedback can sit just outside a configured command limit (rounding
            # or a supported mechanical rest). Keep approach/return setpoints
            # within limits; still judge arrival against the captured pose below.
            # Recorded playback targets remain strict and are never altered here.
            for key, value in action.items():
                low, high = self.robot.config.joint_limits[key.removesuffix(".pos")]
                action[key] = max(low, min(high, value))
            self.send(action)
            if fraction == 1 and all(abs(self.observation[k] - v) <= 2 for k, v in pose.items()):
                # Confirm after the final command; never infer arrival from goals.
                self.sample(returning=returning)
                if all(abs(self.observation[k] - v) <= 2 for k, v in pose.items()):
                    return
            time.sleep(1 / 30)
        errors = {k: abs(self.observation[k] - v) for k, v in pose.items()}
        self.status["return_error_deg"] = errors
        detail = ", ".join(f"{k}: {error:.2f}°" for k, error in errors.items() if error > 2)
        raise TrialEndedError(
            "return_failed", f"Arm did not reach the captured rest pose within 2° ({detail})"
        )

    def restore_gains(self):
        if self.original_kp is not None:
            self.robot.bus.write("Kp", "shoulder_lift", self.original_kp)
            self.original_kp = None

    def run(self):
        """Return a result. Caller must retain bus ownership if rest_reached is false."""
        try:
            self.root.mkdir(parents=True, exist_ok=False)
            self.log = (self.root / "samples.jsonl").open("w", buffering=1)
            self.sample(returning=True)
            self.rest = {f"{n}.pos": self.observation[f"{n}.pos"] for n in self.names}
            if set(self.rest) != set(self.series["action_names"]):
                raise TrialEndedError("invalid_setup", "Dataset does not cover every actuator")
            # Reject an uncommandable home before any trajectory command. This
            # caught the previous trial's gripper home: -0.16° versus a -2.5°
            # command limit, farther apart than the 2° arrival tolerance.
            for key, position in self.rest.items():
                bounds = self.robot.config.joint_limits.get(key.removesuffix(".pos"))
                if bounds is None or not bounds[0] <= position <= bounds[1]:
                    raise TrialEndedError(
                        "invalid_setup",
                        f"Captured rest {key}={position:.2f}° is outside configured limits {bounds}. "
                        "Choose a supported, commandable home before starting the test.",
                    )
            for frame in self.series["values"]:
                for key, v in zip(self.series["action_names"], frame, strict=True):
                    bounds = self.robot.config.joint_limits.get(key.removesuffix(".pos"))
                    if bounds is None or not bounds[0] <= v <= bounds[1]:
                        raise TrialEndedError(
                            "invalid_setup", f"Recorded {key} target is outside calibrated limits"
                        )
            self.sample()
            first = dict(zip(self.series["action_names"], self.series["values"][0], strict=True))
            if any(abs(first[k] - v) > 5 for k, v in self.rest.items()):
                raise TrialEndedError(
                    "invalid_setup", "Place the supported arm within 5° of the recorded starting pose"
                )
            (self.root / "run.json").write_text(
                json.dumps(
                    {
                        "options": self.options.model_dump(),
                        "source": self.source,
                        "initial_gains": getattr(self.robot.bus, "_gains", {}),
                        "rest_pose": self.rest,
                        "series": self.series,
                        **thermal_policy(),
                        "feedback_max_age_s": 0.25,
                    },
                    indent=2,
                )
            )
            # Use nominal gains for the initial alignment and return. No claimed
            # torque limit: this only changes the driver's local MIT Kp value.
            self.move_to(first, returning=False)
            self.sample()
            if self.options.experiment == "shoulder_kp_85":
                kp = finite_number(getattr(self.robot.bus, "_gains", {}).get("shoulder_lift", {}).get("kp"))
                if kp is None or kp <= 0:
                    raise TrialEndedError("invalid_setup", "Driver does not expose verified shoulder Kp")
                self.original_kp = kp
                self.robot.bus.write("Kp", "shoulder_lift", kp * 0.85)
            self.started = time.monotonic()
            self.status["message"] = f"Repeating recorded motion; return threshold ≥{STOP_AT_C}°C"
            while True:
                cycle_start = time.monotonic()
                for stamp, frame in zip(self.series["timestamps"], self.series["values"], strict=True):
                    target = cycle_start + stamp
                    while True:
                        self.sample()
                        now = time.monotonic()
                        if now - self.started >= self.options.duration_s:
                            raise TrialEndedError(
                                "completed", "Test duration completed below the temperature threshold"
                            )
                        if now >= target:
                            break
                        time.sleep(min(1 / 30, target - now))
                    if now - target > 0.1:
                        raise TrialEndedError(
                            "timing_fault", "Playback fell over 100 ms behind; benchmark invalid"
                        )
                    self.send(dict(zip(self.series["action_names"], frame, strict=True)))
                self.status["cycles"] += 1
                # Even a valid 2-degree endpoint mismatch must not become an
                # instantaneous target jump at the loop boundary. This short
                # alignment is monitored and included in the test duration.
                self.move_to(first, returning=False)
        except TrialEndedError as exc:
            self.status.update(result=exc.result, message=str(exc))
        except Exception as exc:
            self.status.update(result="error", message=str(exc))
        finally:
            self.returning = True
            self.publish("stopping")
            try:
                self.restore_gains()
                if not self.rest:
                    # No motion was requested; still cannot assert a measured return.
                    raise TrialEndedError("return_failed", "No valid rest pose was captured")
                if not self.motion_commanded:
                    self.sample(returning=True)
                    if any(abs(self.observation[k] - v) > 2 for k, v in self.rest.items()):
                        raise TrialEndedError(
                            "return_failed", "Arm moved from the supported pose during setup"
                        )
                else:
                    self.move_to(self.rest)
                self.status["rest_reached"] = True
                self.status["final_pose"] = dict(self.observation)
                if self.hot:
                    self.status.update(result="overheated", message=self.hot)
            except Exception as exc:
                self.status["trigger_result"] = self.status["result"]
                self.status.update(
                    result="return_failed",
                    message=f"Return failed: {exc}. Support the arm, then Release now. Motors remain energized.",
                )
            if self.log:
                self.log.close()
                self.log = None
            self.write_summary()
        return dict(self.status)

    def write_summary(self):
        summary = {
            **self.status,
            "statistics_scope": "All fresh samples, including alignment and return",
            "motors": {
                name: {
                    "max_temperature_c": s["max_temperature_c"],
                    "peak_abs_torque_nm": s["peak_abs_torque_nm"],
                    "rms_torque_nm": math.sqrt(s["sum_torque_squared"] / s["samples"]),
                    "samples": s["samples"],
                }
                for name, s in self.stats.items()
                if s["samples"]
            },
        }
        if self.root.is_dir():
            (self.root / "summary.json").write_text(json.dumps(summary, indent=2))
