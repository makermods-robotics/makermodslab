# Standalone Maker thermal dashboard

This runs independently of the MakerModsLab web test panel. It reads a saved robot
configuration and episode, then owns the CAN adapter directly. Close the MakerModsLab
backend first; the runner refuses to connect while port 8000 is occupied.

From the repository root, inspect the prepared trajectory without connecting motors:

```sh
.venv/bin/python tools/thermal_dashboard.py \
  --dataset OWNER/NEW_DATASET
```

With the arm at its supported starting/resting pose, empty gripper, clear recorded
motion and direct return path, and an operator supervising:

```sh
.venv/bin/python tools/thermal_dashboard.py \
  --dataset OWNER/NEW_DATASET \
  --robot metal_dlabs_thermal --duration 1800 \
  --experiment shoulder_kp_85 --run --ready
```

The default and maximum duration are 1800 seconds (30 minutes). To view the current
settings while waiting for a new episode, without loading data or connecting motors:

```sh
.venv/bin/python tools/thermal_dashboard.py --dashboard-only
```

The dashboard opens at <http://127.0.0.1:8092>. Hardware starts after the page connects
and a three-second countdown. It shows seven charts with temperature on the left axis
and signed torque on the right, live readings, maximum temperature, peak torque, RMS
torque, elapsed test time, completed loops, and the test outcome. Choose the last two
minutes or the whole test. Curves use sampled dashboard feedback; log statistics use
all fresh samples collected by the runner, including alignment and return.

The first experiment uses shoulder_lift Kp = 85% of its original value. It is a
stiffness experiment, not a hard torque cap. Kd and recorded playback timing are
unchanged. Original Kp is restored before returning to rest.

The recording is not modified. The test copy clamps at most one degree of recorded
rounding/limit mismatch to existing configured joint limits and appends a smooth
return segment of at least one second. Non-gripper endpoints must differ by no more
than two degrees, and the empty gripper by no more than ten. Larger differences are
refused. All adjustments and the exact tested trajectory are stored in `run.json`.

Reaching 100°C on any motor ends repetition and requests a bounded return to captured
rest. Reaching 135°C raises the critical-reference flag. Stale/invalid feedback, sustained tracking error and
playback timing faults also stop the test. See [guard details](THERMAL_REPLAY.md).

**Stop and return** requests a controlled return. **Release now** cuts motor power
and requires supporting the arm first. If the dashboard loses contact for ten
seconds, the runner requests a return. A failed return keeps the connection occupied
and reports that operator support and explicit release are needed. It must not be
left unattended: energized motors can continue heating even after motion stops.

Ctrl+C while running first requests a return; a second Ctrl+C requests immediate
release. After the test finishes and the motor connection closes, the dashboard
stays open for review; Ctrl+C then exits the dashboard server.

Run artifacts are saved under `output/thermal-replay/<timestamp-id>/`. A successful
short trial establishes only the observed behavior under that trajectory, load,
initial temperature and environment. Compare a matched baseline before attributing
an improvement to reduced stiffness.

## Home and temperature checks

The captured supported home must be inside every joint's configured command limits.
An uncommandable home is rejected before any trajectory command. This prevents the
previous trial's gripper problem (-0.16° captured versus a -2.5° command limit).
The 2° measured-arrival tolerance is unchanged. Return errors now identify the joints
that missed home, and logs include measured positions. A released connection is labeled
closed rather than continuing to say the motors are energized.

All live tools use `makermodslab/thermal_limits.py`: return/alert at **100°C or higher**,
critical reference at **135°C or higher**, and **1800 seconds** by default. Historical
run files retain their original thresholds. Temperatures can rise during return;
this cutoff cannot promise an absolute peak below 100°C.

The 135°C number is user-reported manufacturer guidance, not a firmware readback.
RobStride's [published RS02 manual](https://www.robstride.com/assets/product_manual_robStride02-e7f9f7c4.pdf)
distinguishes a 135°C winding limit from an 80°C control-board maximum. The user has
not confirmed which sensor/hardware revision the guidance refers to and requested
keeping the 100°C software limit. The driver labels the reported field `temp_mos`;
its physical sensor location is not inferred. No firmware protection value is changed.
