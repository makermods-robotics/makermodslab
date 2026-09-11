# Metal gripper holding torque

Metal followers use an MIT holding-effort controller during local teleoperation
and recording. Free movement keeps the normal position tracking and Kp/Kd gains.
When closing effort rises while movement stops, the controller adjusts the
position target to regulate sustained squeezing. Opening resumes normal tracking.
Other arm joints retain their existing control.

## Configuration

In Metal Robot Config, expand **Advanced parameters** and adjust **Holding torque**.
The range is **0.1–2.0 N·m**, with a **0.5 N·m** default. Moving the slider edits a
draft; **Save** applies the value to an active holding controller and persists it
for future sessions. Bimanual robots apply the same setting to both followers.
Stop the session before enabling, disabling, or switching control modes.

The saved field is `gripper_hold_torque_nm`. New Metal records, old records missing
the field, and unnamed Metal teleoperation/recording requests use the default.
Explicit values, including `null` to opt out, are preserved. An explicitly selected
`gripper_current_limit_a` selects the alternative mode-4 experiment; both settings
cannot be enabled together. Other robot families do not receive this default.

`POST /api/v1/robots/{name}` accepts the setting. To switch an idle robot from
current limiting to holding torque, send:

```json
{
  "gripper_current_limit_a": null,
  "gripper_hold_torque_nm": 0.5
}
```

`GET /api/v1/robots/{name}/gripper-status` reports the holding target, whether
holding is engaged, measured motor torque, temperature, and faults.

## Control behavior

The controller uses measured closing effort and velocity to detect contact.
During holding, for Metal's decreasing-angle closing direction, its target
correction is:

```text
q_raw = q_last - (holding_torque - abs(measured_torque)) / Kp
```

Angles in this equation are radians. The correction is smoothed over time and
bounded by joint limits and the leader's requested closure. Kp/Kd stay fixed;
desired velocity and feedforward torque are zero during holding. A brief low-effort
reading does not immediately release the hold, preventing repeated full-close
commands while the effort settles. This approach is inspired by
[YAM's holding controller](https://github.com/i2rt-robotics/i2rt/blob/5b72c47239bd056d0fa6c1a39edeb0537c89443c/i2rt/robots/utils.py#L702-L782).

A serialized background worker continues regulation during recording pauses.
Setup verifies MIT scaling before enabling and temporarily arms a 500 ms motor
watchdog. Confirmed shutdown disables the gripper and restores mode/watchdog
settings without writing flash. Communication and motor faults latch a stop.
Either reported motor temperature reaching 55°C stops and releases the gripper;
restart requires both sensors below 45°C. There is no temperature-based derating.

This regulates **sustained motor-output effort**, not fingertip force or a hard
instantaneous torque ceiling. Contact transients can exceed the target. The
software range is not a continuous-current/torque rating or a guarantee against
overheating; continuous holding requires validation for the actual mechanism,
load and environment.
