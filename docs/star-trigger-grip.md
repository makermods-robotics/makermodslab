# Star arm trigger grip

In Robot settings for a Maker robot, set **Leader arm** to **Star arm trigger grip**.
Assign the leader port and run the normal leader zero calibration with the arm in its
zero pose and the trigger at its closed stop. A leader change clears the old leader
setup; the follower setup is preserved. The selection is saved with the robot and
applies to teleoperation, recording, and remote leader sessions, including bimanual rigs.

The trigger gripper measured on September 10, 2026 had these encoder readings:

| Position             | Encoder angle |
| -------------------- | ------------- |
| Closed stop          | 0.0°          |
| Far stop             | -186.8°       |
| Closed-to-far travel | -186.8°       |

Each endpoint was stable across two runs (0.1° spread). The readings measure travel, not
a permanent offset: zero calibration makes the closed stop 0°. The trigger turns the
servo the opposite way to the stock lever, which is why the lever preset barely moved
the jaw: the pull sat outside the lever's mapped band and the jaw only snapped between
closed and open where the driver's 360° unwrap window flipped branches, near -150°.

Only the first half of the pull is used. The profile maps 0° to the Maker gripper's
closed target (-2°) and -93.4° to its open target (-120°), with a scale of
120 / 93.4 (about 1.2848). The rest of the pull, down to the far stop, clamps at fully
open, because pulling the trigger all the way is physically awkward. Change
`TRIGGER_USABLE_TRAVEL_DEG` in `makermodslab/star_gripper.py` to use more or less of
the pull. Other joints and the regular Star profile keep their existing mappings.

The preset lives in `makermodslab/star_gripper.py` beside the Metal arm's vertical grip,
so changing this hardware profile does not require changing LeRobot, the combined
branch, or its pinned commit.
