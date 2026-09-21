// Hardware replies can arrive in a different order on every sample. Display
// joints from base to gripper instead of inheriting the response object's order.
const JOINT_ORDER: readonly string[] = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_yaw",
  "wrist_roll",
  "gripper",
];

export function orderedJointEntries<T>(
  joints: Record<string, T>,
): [string, T][] {
  const rank = (name: string) => {
    const index = JOINT_ORDER.indexOf(name);
    return index === -1 ? JOINT_ORDER.length : index;
  };
  return Object.entries(joints).sort(
    ([a], [b]) => rank(a) - rank(b) || a.localeCompare(b),
  );
}
