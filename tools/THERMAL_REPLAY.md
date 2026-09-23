# Maker thermal replay experiment

Branch: `experiment/maker-thermal-replay`. Experimental software guard, tested with
simulated feedback. A physical trial stopped at the former 65°C threshold; a gripper home outside command limits prevented verified return. The new preflight rejects that setup before playback.
This version supports one Maker follower (all seven motors), through MakerModsLab's
existing replay session and CAN connection. Ordinary replay is unchanged.

## Record and run

1. Record teleoperation normally in MakerModsLab. Start **and finish** in the same
   supported resting pose, with an empty gripper. The endpoints must match within
   2 degrees on every joint. Keep the motion and direct return path unobstructed.
2. Stop recording. Put the follower at that supported rest pose, within 5 degrees
   of the recording's first frame. The pose measured before replay is the return
   destination. Every measured home joint must lie inside its configured command limits. An invalid home is rejected before motion; the runner confirms the arm stayed at its supported start before releasing. A pose is not automatically safe merely because it is called rest.
3. Open that episode's **Replay on hardware** panel. Select **Repeat as a 30-minute
   thermal test**, choose **Baseline**, and confirm the physical setup.
4. Start and supervise. The UI shows each motor's temperature, signed torque, and
   status. At 100°C is red; 135°C is a manufacturer-reported critical reference. A persistent message and toast
   identify the result. Keep the page open so the session heartbeat continues.
5. Let the arm cool to a comparable initial temperature. Repeat the same episode
   with **Shoulder stiffness −15%**. Keep load, environment, and starting pose the
   same. Compare heat, torque, and tracking; a cooler but inaccurate run is not an
   equivalent result.

The experiment changes only shoulder_lift Kp to 85% of its original value during
playback, and restores it before returning. Kd and recorded timing are unchanged.
**This is not a hard torque cap.** It can increase sag/tracking error and may not
reduce heating under a sustained gravity load. No firmware torque-limit register
has been verified or written. Baseline is the default.

## Stop and result behavior

- Any fresh actuator temperature **≥100°C** ends repeated motion and requests a
  controlled return to the captured rest pose at up to 20 degrees/second. Thermal
  readings continue during return; crossing **≥135°C** also raises the critical flag.
- A run completes after 1800 seconds of repeated playback, excluding initial
  alignment and final return. A bounded alignment between loops bridges any
  small endpoint mismatch and counts toward those 1800 seconds. Arrival must be measured within 2 degrees before
  torque is released. An overheat during return invalidates completion.
- Missing/invalid feedback, feedback older than 250 ms, >12° tracking error for
  more than 0.5 seconds, target clipping, or more than 100 ms playback lateness invalidates the run
  and requests a return. No silent frame dropping or speed reduction is used.
- **Stop** requests that same controlled return. **Release now** explicitly cuts
  torque and can let an unsupported arm fall. A lease timeout requests a return;
  repeated lease-stop requests do not force a release during that return.
- If fresh feedback or motion control fails, software cannot guarantee a return.
  The session stays occupied and reports **Return failed — support the arm, then
  Release now**. It attempts to hold the measured position when feedback permits;
  motors remain energized and can continue heating. Operator intervention is
  required. Backend shutdown retains the existing timed forced-release fallback;
  power loss and motor protection are outside this software guard.

This is a short endurance comparison, not a certified thermal protection system.
Verify the clear return route with a short, unloaded baseline before longer runs.

## Artifacts and API

Each attempt writes `output/thermal-replay/<timestamp-id>/`:

- `run.json`: exact trajectory, captured rest pose, options, thresholds.
- `samples.jsonl`: fresh temperature/torque samples with feedback timestamps and
  phase labels and return flags, plus timestamped commanded positions.
- `summary.json`: result, completed loops, elapsed playback time, critical/return
  status, observed maximum temperature, peak torque and sample RMS torque per motor.
  Summary torque statistics cover alignment, playback and return; use sample logs
  to compare only the playback interval. They are sampled feedback, not all CAN frames.

Session options include:

```json
{
  "repo_id": "owner/dataset",
  "episode_index": 0,
  "thermal_test": {
    "duration_s": 1800,
    "experiment": "baseline",
    "rest_pose_confirmed": true
  }
}
```

API durations can be 10–1800 seconds for supervised commissioning. The UI uses 1800.
`experiment` is `baseline` or `shoulder_kp_85`. Invalid/nonclosed trajectories and
unsupported arm types are rejected before connection. The selected robot's saved
ports and calibration are resolved by the usual session API. Run the separate
[actuator monitor](ACTUATOR_MONITOR.md) for terminal telemetry during recording or replay.
