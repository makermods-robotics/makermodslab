# Maker gripper holding torque

Maker followers use the same sustained-effort holding controller as Metal during
local teleoperation and recording, including single and bimanual robots. Normal
movement follows the leader with the configured Kp/Kd. Once closing effort rises
and motion stops, the controller adjusts the target near the object to regulate
squeeze. Opening releases the hold. A serialized 50 Hz worker continues regulation
during recording pauses. Other joints keep their existing control.

In Maker Robot Config, expand **Advanced parameters** and adjust **Holding torque**.
The default is **0.5 N·m**, adjustable from **0.1 to 2.0 N·m**. **Save** applies the
setting to an active controller and persists it. Both follower grippers use the
same value on bimanual robots. This is motor-output effort, not fingertip force
or a hard instantaneous torque ceiling; contact transients may exceed the target.

The saved/API field is `gripper_hold_torque_nm`. New records, legacy records missing
this field, and unnamed sessions default to 0.5 N·m. Explicit `null` opts out.
Stop the session before enabling or disabling holding. The alternative Metal
`gripper_current_limit_a` mode is not supported on Maker.

`GET /api/v1/robots/{name}/gripper-status` reports target torque, holding state,
measured torque, winding temperature, and latched errors. Save failures after a
live update stop the active grippers, including both sides of a bimanual robot.

## Hardware and firmware

The adapter targets the **RobStride 00** gripper, identified as `O0` in the pinned
LeRobot driver, on classic CAN using the **standard-frame MIT protocol**, default
host ID `0xFD`, and MIT operation mode. It requires firmware supporting MIT
parameter reads/writes (**0.0.3.27 or later**) and packed operating-mode/fault flags
in status feedback. Unsupported parameter access or unconfirmed motor enable
fails setup rather than running without the protections below.

Wire behavior follows the manufacturer's [RS00 manual, sections 6.1, 6.17 and
6.18](https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS00/RS00User%20Manual260713.pdf):

- RS00 scaling is ±12.57 rad, ±33 rad/s, ±14 N·m. Temperature is the low 12 bits
  of bytes 6–7, in tenths of a degree; upper bits report operating mode and faults.
- Setup confirms stop and reads MIT operation mode and the existing CAN timeout.
  Before enabling, it writes and verifies parameter `0x7028` at 10,000 ticks
  (500 ms). These are RobStride parameters, not Damiao register operations.
- Winding temperature reaching 55°C, warning/fault flags, unexpected disable,
  or missing fresh feedback latch a stop. Restart requires temperature below 45°C.
  RobStride MIT reports one temperature, not Metal's two sensors.
- Shutdown always releases the gripper, even if other joints are configured to
  remain energized. The original watchdog is restored only after confirmed stop.
  No motor flash writes, fault clears, zero changes, or protocol switches occur.

Maker closes toward **increasing** angles, opposite Metal. Its correction is:

```text
q_raw = q_last + (holding_torque - abs(measured_torque)) / Kp
```

Angles are radians. Smoothed commands stay within joint bounds and never close
past the leader's request. The adapter shifts its limits by the follower's
session-specific full-turn offset so raw motor targets and limits agree.

Validation includes simulated contact convergence and fake-CAN tests against the
real pinned Maker follower. Physical RS00 firmware, grasp stability, actual
sustained torque, thermal behavior, and power-loss shutdown still need bench
validation on the target mechanism before relying on continuous holding.
