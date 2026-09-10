# Metal gripper limits

Robot Settings provides three independent, opt-in controls for local Metal teleoperation and recording. All are disabled by default and take effect on the next session. A bimanual robot uses the same settings for both grippers, except the leader gripper hold, which is single-arm only for now. None of them apply to policy inference, replay, remote sessions, or standalone vendor tools.

## Soft squeeze limit

Set a positive closing error in motor degrees. The driver limits how far the closing target can move beyond the measured jaws, and disables gripper velocity and torque feedforward. Opening remains responsive. Smaller errors reduce the position controller's nominal sustained squeezing effort; this is not a motor-current or jaw-force guarantee. A fresh gripper response is required for every action, adding an unmeasured round trip to the control loop.

## Leader gripper hold

Set a positive engage gap in follower motor degrees. While the operator commands the gripper at least this many degrees more-closed than the follower physically reaches — i.e. an object is in the follower's jaws and the operator keeps squeezing — the driver energizes the **leader** gripper servo rigid, so the operator feels a wall instead of driving the follower into a stall. The hold is not steady: it re-asserts a few times a second and briefly releases every ~0.4 s to sense the operator's hand; pull the handle open during one of those windows to release it. It also releases on its own once the follower catches up (the object is gone) and on every session stop.

Requires a **motorized** Star Arm 102 leader. There is no way to detect a motorless leader in software, so this is opt-in and does nothing (beyond a logged warning) on a leader whose gripper has no servo. Single-arm only for now. Layers on top of the soft squeeze limit but does not require it — the engage signal is the leader-command vs follower-position gap, not the soft limiter's clamp.

## Experimental motor current limit

Set both a current percentage and a maximum gripper speed. The percentage is relative to the motor's maximum current (`Imax`), not its rated torque, protocol `TMAX`, or jaw force. There is no assumed safe preset. Current resolution is 0.01 percentage points; velocity resolution is 0.01 rad/s. Commands round these quantities down to the protocol's resolution.

The implementation uses the manufacturer's force-position mode. It checks the Metal J7 IDs, firmware bytes `7018`, and protocol scales before enabling the gripper. It verifies mode changes and routes gripper position commands through mode 4. It does not flash firmware, change protocol scaling, or save parameters to flash. Shutdown disables the gripper before restoring a mode changed by this session. If a process is killed while mode 4 is active, ordinary Metal startup refuses that leftover mode rather than silently sending incompatible MIT commands.

This feature has been tested with simulated CAN responses, not physical hardware. Manufacturer documentation requires enabling before sending control commands; current behavior during the enable-to-first-command interval is not specified. The first command holds the sampled gripper position with the configured current and speed. This is not a guaranteed startup current ceiling, an emergency stop, or a calibrated jaw-force limit. Verify startup, sustained grip, release, stop, and communication failure on an appropriate test fixture before relying on it.

Soft and current limits may be enabled together. Recording stores the gripper position command actually sent after soft limiting; it does not infer measured jaw force from that command.

## Protocol sources

- [Manufacturer SDK](https://github.com/dmBots/motor-sdk/blob/main/Python%E4%BE%8B%E7%A8%8B/u2can/DM_CAN.py): force-position payload, register transactions, enable/disable commands.
- [Firmware version rules](https://github.com/dmBots/motor-firmware/blob/master/%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E/%E5%9B%BA%E4%BB%B6%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E.md) and [changelog](https://github.com/dmBots/motor-firmware/blob/master/%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E/%E5%9B%BA%E4%BB%B6%E6%9B%B4%E6%96%B0%E6%97%A5%E5%BF%97-3.1.docx): revision support and current-command saturation.
