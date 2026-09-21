import { describe, expect, it } from "vitest";
import { orderedJointEntries } from "./jointOrder";

describe("joint display order", () => {
  it("keeps each joint in the same slot when hardware response order changes", () => {
    const sample = {
      wrist_roll: 5.7,
      wrist_yaw: -1.1,
      gripper: -7.2,
      wrist_flex: -0.2,
      elbow_flex: 0,
      shoulder_lift: 0.1,
      shoulder_pan: -0.3,
    };
    const reversed = Object.fromEntries(Object.entries(sample).reverse());
    expect(orderedJointEntries(reversed)).toEqual(orderedJointEntries(sample));
    expect(orderedJointEntries(sample).map(([name]) => name)).toEqual([
      "shoulder_pan",
      "shoulder_lift",
      "elbow_flex",
      "wrist_flex",
      "wrist_yaw",
      "wrist_roll",
      "gripper",
    ]);
    expect(Object.keys(sample)[0]).toBe("wrist_roll");
  });

  it("also orders six-joint arms and preserves unknown joints and their values", () => {
    expect(
      orderedJointEntries({
        extra_z: 9,
        gripper: 2,
        shoulder_pan: 1,
        extra_a: 8,
      }),
    ).toEqual([
      ["shoulder_pan", 1],
      ["gripper", 2],
      ["extra_a", 8],
      ["extra_z", 9],
    ]);
  });
});
