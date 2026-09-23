# Maker Arm live temperature and torque monitor

Run MakerLab from this checkout, then use a second terminal:

```bash
cd /Users/isaac/Documents/GitHub/MakerLab
.venv/bin/python tools/actuator_monitor.py --duration 1800 --label baseline --bell
```

Start teleop or recording in MakerLab as usual. If MakerLab was already running before
these changes, restart it **after ending any active session normally** so it loads the
new telemetry field. The monitor waits for feedback; it never starts teleop itself.
The 1800-second timer begins at the first fresh motor sample. At 1800 seconds only the
monitor stops; the robot remains under MakerLab's control.

The monitor uses `ws://127.0.0.1:8000/api/v1/ws/joint-data`. Set `--url` if your backend
uses another port. It displays all configured follower actuators (seven for a single
arm, fourteen for bimanual). The encoder-only Star leader has no actuator temperatures.

Each row shows current temperature, signed motor-reported torque in N·m, rolling
30-second sample RMS torque, maximum observed temperature, and state:

| Reading                                                        | State                  |
| -------------------------------------------------------------- | ---------------------- |
| Temperature <110°C                                             | OK                     |
| Temperature ≥110°C and <135°C                                  | OVERHEATING, red       |
| Temperature ≥135°C                                             | CRITICAL, white on red |
| Feedback more than 1 second old                                | STALE, yellow          |
| No feedback timestamp                                          | NO DATA, yellow        |
| Nonfinite/invalid feedback or unsupported temperature decoding | INVALID, yellow        |

Colors are used in an interactive terminal; redirected output retains explicit state
labels. `--bell` rings the terminal bell on entry to overheating or critical (audibility
depends on terminal settings). The maximum and run summary preserve earlier hot readings
after a motor cools. Current alerts clear on a fresh reading below their threshold.

This is an **observer and recorder**, not a thermal interlock. It sends no CAN commands,
does not open the adapter, and does not alter torque, gains, motor power, or trajectory.
It copies the driver's cache on existing MakerLab preview ticks (~20 Hz teleop, ~10 Hz
recording). It therefore does not capture every CAN sample or guarantee capture of brief
torque peaks. Paused/stopped control or a disconnected link becomes stale, never a fresh
zero. The temperature is the driver's `temp_mos` feedback field. The July 2026
RobStride MIT manual labels this field winding temperature; no separate board reading
is exposed here. An incompatible packed-status decoder is flagged rather than
silently masking bits and claiming a safe reading. This integration covers local
MakerLab teleop/recording; separate inference subprocesses need the same telemetry tap.

## Saved measurements

Every invocation creates a unique folder under `output/actuator-monitor/` containing:

- `run.json`: test label, source, thresholds, and sampling limitations.
- `samples.csv`: one row per distinct fresh feedback sample received by the monitor.
  Temperature and torque share the motor feedback timestamp, avoiding the earlier
  episode/wall-clock alignment ambiguity. Each motor has its own timestamp.
- `events.jsonl`: state transitions, including stale/missing data and hot alerts.
- `summary.json`: per-actuator starting/maximum temperature, observed peak and RMS
  torque, hot-sample counts, first threshold crossings, gaps, final states, and whether
  the requested monitoring duration completed. Missing data never becomes a passing
  thermal test. The tool reports observations and completion separately; it does not
  automatically declare a safety pass.

Repeated cached feedback is excluded from RMS/counts. RMS is sample based over received
fresh samples, not a continuous-time thermal model. Gap seconds are the spans between
successive accepted feedback samples more than one second apart, not inferred time at
any particular torque or temperature. Inspect final stale states and events as well as
inter-sample gaps, since a run can end while the stream is unavailable.

Try the display without any motor connection:

```bash
.venv/bin/python tools/actuator_monitor.py --demo --duration 15 --label display-test --bell
```

All demo outputs are explicitly labeled SIMULATED. The demo crosses both temperature
thresholds. It never connects to MakerLab or the CAN adapter.

## Repeatable thermal test

1. Record the trajectory with MakerLab. Keep its dataset/episode identifier, payload,
   mount, cooling configuration, room temperature, starting motor temperatures, and
   software revision with the experiment notes. Use a new monitor `--label` per change.
2. Compare the same trajectory, pace, number of cycles and starting thermal condition.
   Target thirty minutes, but stop/unload using the established operator procedure if an
   actuator reaches 110°C; record time to crossing. The 135°C reference is not a test target.
3. Change one thing at a time: cooling, payload/reach, rest pose, then controller tuning.
   Compare temperature rise, first threshold crossing, RMS torque, and task/tracking
   quality. A cooler run that drops the payload or no longer tracks is not a successful fix.
4. Retain the trajectory recording alongside these telemetry files. This monitor does
   not record or play back a commanded trajectory and does not launch repeated motion.

## Torque limits with minimal speed loss

Prioritize cooling and reducing static shoulder load before globally lowering speed.
An extended arm consumes current even when motion stops. Consider counterbalancing,
less distal weight, or retracted/supported rest poses. Check mechanical drag and tune
oscillation only after recording tracking error and velocity as well as effort.

A verified firmware current/total-torque limit can preserve short acceleration peaks
while a separate sustained-current budget manages heat. In MIT impedance control, the
feedforward torque field is **not** a total torque limit: position and velocity error
also produce torque. Clamp/derate the complete control effort only after checking the
actual driver, firmware support, and minimum support torque. Reducing Kp alone changes
stiffness and can cause sag; gravity feedforward improves tracking but does not remove
the current needed to support weight. A limit below the required gravity torque may
lose position even if the requested speed is unchanged.

This observer does not change gains or torque limits. The separate test runner can reduce shoulder-lift Kp by 15%; that is not a torque cap. The downloaded experiment's
left shoulder was about 5.94 N·m sample RMS, with a 16.89 N·m observed peak. Do not copy
the 6–7 N·m catalog rating directly into a sustained holding limit: published ratings
depend on cooling, speed and product revision.

References: [RobStride specifications and heat-sink conditions](https://github.com/RobStride/Product_Information/blob/main/README.md),
[maxon on current-related motor heating and RMS load](https://support.maxongroup.com/hc/en-us/articles/360004427413-On-the-heating-of-motors-in-hand-held-tools).
