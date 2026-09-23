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

The 135°C reference is not a firmware threshold readback. The newer official
[RS02 July 2026 manual](https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS02/RS02User%20Manual260713.pdf)
identifies MIT feedback bytes 6–7 as winding temperature, with mode/fault/warning
bits in the upper nibble. Our installed driver calls that field `temp_mos`; this
name does not identify it as a board sensor. The standalone diagnostic adapter
preserves the raw word and separates these flags before the legacy decoder.
The current MIT stream exposes no separate board temperature. Manuals differ by
revision, so neither a board limit nor the actual firmware cutoff is read back here.
No firmware protection value is changed.

For a recording whose gripper requests were already clamped by the driver during
teleop, `--prepare-recorded-loop` explicitly allows up to 30° of gripper clipping
and up to 10° endpoint mismatch per joint. It preserves the source and logs every
adjustment; other joints retain the 1° clipping allowance. A smooth return of at
least two seconds is appended, capped at 10°/s. This requires a clear return path.

Each standalone trial also saves `can_frames.jsonl` (raw RX/TX after the bus's initial
handshake), `stop_event.json` (trigger, all latest measurements, positions, gains and
fault evidence **before** return), and `diagnostics.json` (final latched evidence).
The normal `samples.jsonl` includes all sampled temperatures, torques and positions
through return. Separate board temperature and firmware threshold readback are
explicitly unavailable, not zero. Firmware fault flags stop repetition; a latched
fault blocks automatic fault-clear requests. A failed return still requires support
and explicit release. Fault-status-shaped frames retain raw bytes and the heuristic
classification, because firmware versions can vary.

After a verified return (unless immediate release was requested), the runner also
requests each motor's fault status using MIT command `FF FF FF FF FF FF 00 FB`.
The `00` requests status; `FF` in that byte would clear faults and is not used by
this read. Replies or timeouts are saved in `diagnostics.json`; the read is bounded
to 40 ms per motor and never runs during playback. It does not expose board
temperature or read back thermal threshold settings.
