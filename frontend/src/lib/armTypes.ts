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
 *   calibration, and no 3D viewer (no Maker URDF ships yet).
 * - "metal" — Metal arm: a 7-DOF Damiao CAN follower driven by the same Star
 *   Arm 102 leader with a Metal joint-mapping preset. Same UI seams as the
 *   Maker arm (zero-pose calibration, probe detection, numeric readout); its
 *   zero POSE differs (upright, gripper closed vs folded, gripper open).
 *
 * Records created before the Maker arm existed have no arm_type on disk; the
 * backend reads those back as "so101", so this is never undefined in practice.
 */
export type ArmType = "so101" | "maker" | "metal";

/**
 * True for the CAN families (Maker, Metal) — every UI seam they share:
 * zero-pose calibration in place of the range sweep, no auto-calibration, no
 * wiggle/Feetech identity, protocol-probe port detection, and the numeric
 * joint readout in place of the SO-101 URDF viewer.
 */
export function isCanArmType(armType: ArmType | undefined): boolean {
  return armType === "maker" || armType === "metal";
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
