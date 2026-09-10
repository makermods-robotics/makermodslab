# Star arm vertical grip

In Robot settings for a Metal robot, set **Leader arm** to **Star arm vertical grip**.
Assign the leader port and run the normal leader zero calibration with the arm in its
zero pose and the gripper fully closed. A leader change clears the old leader setup;
the follower setup is preserved. The selection is saved with the robot and applies
to teleoperation, recording, and remote leader sessions, including bimanual rigs.

The vertical gripper measured on September 10, 2026 had these encoder readings:

| Position | Encoder angle |
| --- | --- |
| Fully closed | -17.2° |
| Fully open | 24.0° |
| Closed-to-open travel | 41.2° |

Each endpoint was stable across ten readings. The readings measure travel, not a
permanent offset: zero calibration makes the closed position 0°. The profile maps
0° to the Metal gripper's closed target and 41.2° to its existing 115° open target,
with a scale of 115 / 41.2 (about 2.79126). Output stays within the existing 0–115°
bounds. Other joints and the regular Star profile keep their existing mappings.

The preset lives in `makermodslab/star_gripper.py`, so changing this hardware profile
does not require changing LeRobot, the combined branch, or its pinned commit.
