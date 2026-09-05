/**
 * What each arm type can and cannot do — the client mirror of the backend's
 * makermodslab/arm_capabilities.py. Import these instead of writing
 * `arm_type === "maker"` comparisons inline: a scattered check is a check
 * somebody forgets to extend when the next arm type arrives.
 *
 * Lives in lib/ (not useRobots.ts, ArmType's public home) so it stays a pure,
 * independently testable module — useRobots pulls in the API context on
 * import. useRobots re-exports everything here.
 */

/**
 * Which hardware family a robot is.
 *
 * - "so101" — SO-101 leader/follower, Feetech servos on USB serial. 6 joints
 *   per arm, range-sweep calibration (manual or automatic).
 * - "maker" — Maker Arm v1: a 7-DOF RobStride CAN follower driven by a Star
 *   Arm 102 leader on UART servos. Zero-pose calibration only, no automatic
 *   calibration. Ships a 3D URDF, so teleop drives the model (six revolute
 *   joints; the gripper reads out numerically).
 * - "metal" — Metal arm: a 7-DOF Damiao CAN follower driven by the same Star
 *   Arm 102 leader with a Metal joint-mapping preset. Same UI seams as the
 *   Maker arm (zero-pose calibration, probe detection); no Metal URDF ships
 *   yet, so it keeps the numeric readout. Its zero POSE differs (upright,
 *   gripper closed vs folded, gripper open).
 *
 * Records created before the Maker arm existed have no arm_type on disk; the
 * backend reads those back as "so101", so this is never undefined in practice.
 */
export type ArmType = "so101" | "maker" | "metal";

/**
 * True for the CAN families (Maker, Metal) — the UI seams they share:
 * zero-pose calibration in place of the range sweep, no auto-calibration, no
 * wiggle/Feetech identity, and protocol-probe port detection.
 *
 * NOT the right check for the 3D viewer any more — the Maker arm is a CAN arm
 * that DOES ship a URDF. Use `armHasUrdf` for that.
 */
export function isCanArmType(armType: ArmType | undefined): boolean {
  return armType === "maker" || armType === "metal";
}

/**
 * True when a 3D URDF for this arm type ships in `frontend/public/`, so the
 * teleop panel shows `UrdfViewer` rather than `JointAngleReadout`. Mirrors
 * `ships_urdf` in makermodslab/arm_capabilities.py — change them together.
 *
 * `undefined` → true: a record with no arm_type is an SO-101, which ships one.
 */
export function armHasUrdf(armType: ArmType | undefined): boolean {
  return armType === undefined || armType === "so101" || armType === "maker";
}

/**
 * Flat proprioceptive width of ONE follower arm — one dim per joint. The
 * SO-101 has 6; the CAN arms have 7 (six joints plus a permanent gripper).
 * Mirrors `_JOINTS_PER_ARM` in makermodslab/arm_capabilities.py (and the
 * server's `_ARM_STATE_DIMS` in rollout.py) — change them together.
 */
export function jointsPerArm(armType: ArmType | undefined): number {
  return isCanArmType(armType) ? 7 : 6;
}

/**
 * Best-effort arm family for a dataset's / checkpoint's raw `robot_type` string
 * (lerobot writes the robot's `.name`: "so101_follower", "bi_maker_follower",
 * …; datasets recorded elsewhere carry anything). The client mirror of
 * `arm_capabilities.arm_type_from_robot_type`.
 *
 * Returns null — NOT a default — when the string is missing or unrecognized:
 * the cross-arm warnings that call this must stay silent when the arm can't be
 * established rather than raise a false alarm.
 */
export function armTypeFromRobotType(
  robotType: string | null | undefined,
): ArmType | null {
  if (!robotType) return null;
  const text = robotType.trim().toLowerCase();
  if (!text) return null;
  if (text.includes("maker")) return "maker";
  if (text.includes("metal")) return "metal";
  if (
    text.includes("so100") ||
    text.includes("so101") ||
    text.includes("so-100") ||
    text.includes("so-101") ||
    text.includes("so_follower") ||
    text.includes("so_leader")
  )
    return "so101";
  return null;
}

/** Human-readable name per arm type, for the cross-arm warning prose. */
export const ARM_TYPE_LABEL: Record<ArmType, string> = {
  so101: "SO-101",
  maker: "Maker",
  metal: "Metal",
};
