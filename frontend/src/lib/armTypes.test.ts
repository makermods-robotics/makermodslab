import { describe, expect, it } from "vitest";
import {
  armHasUrdf,
  armTypeFromRobotType,
  isCanArmType,
  jointsPerArm,
} from "./armTypes";

// The client mirror of the backend's arm_capabilities.py predicates. These
// pins are what keeps a new arm type from silently inheriting SO-101
// behavior in the UI: adding a value to ArmType forces a decision here.

describe("isCanArmType", () => {
  it("is false for the SO-101 (Feetech serial, full range-sweep flows)", () => {
    expect(isCanArmType("so101")).toBe(false);
  });

  it("is true for both CAN families (zero-pose calibration, probe detection, numeric readout)", () => {
    expect(isCanArmType("maker")).toBe(true);
    expect(isCanArmType("metal")).toBe(true);
  });

  it("treats a missing arm_type as SO-101, like the backend's normalize_arm_type", () => {
    expect(isCanArmType(undefined)).toBe(false);
  });
});

describe("jointsPerArm", () => {
  // Mirrors _JOINTS_PER_ARM in makermodslab/arm_capabilities.py — change
  // both together.
  it("gives the SO-101 6 dims and the CAN arms 7 (six joints + permanent gripper)", () => {
    expect(jointsPerArm("so101")).toBe(6);
    expect(jointsPerArm("maker")).toBe(7);
    expect(jointsPerArm("metal")).toBe(7);
  });

  it("defaults a missing arm_type to the SO-101 width", () => {
    expect(jointsPerArm(undefined)).toBe(6);
  });
});

describe("armHasUrdf", () => {
  // Mirrors ships_urdf in makermodslab/arm_capabilities.py. Decides the teleop
  // panel shows the 3D UrdfViewer rather than the numeric JointAngleReadout.
  it("is true for the SO-101 and the Maker arm (each ships a URDF)", () => {
    expect(armHasUrdf("so101")).toBe(true);
    expect(armHasUrdf("maker")).toBe(true);
  });

  it("is false for the Metal arm (no Metal URDF ships yet)", () => {
    expect(armHasUrdf("metal")).toBe(false);
  });

  it("treats a missing arm_type as SO-101, which keeps the 3D viewer", () => {
    expect(armHasUrdf(undefined)).toBe(true);
  });
});

describe("armTypeFromRobotType", () => {
  // Mirrors arm_capabilities.arm_type_from_robot_type — a robot_type STRING
  // (from a dataset's meta/info.json), not a built config.
  it("maps the names this app records", () => {
    expect(armTypeFromRobotType("so101_follower")).toBe("so101");
    expect(armTypeFromRobotType("bi_so_follower")).toBe("so101");
    expect(armTypeFromRobotType("maker_follower")).toBe("maker");
    expect(armTypeFromRobotType("bi_metal_follower")).toBe("metal");
  });

  it("maps legacy / differently-cased strings", () => {
    expect(armTypeFromRobotType("so100_follower")).toBe("so101");
    expect(armTypeFromRobotType("  Maker_Follower ")).toBe("maker");
  });

  it("returns null — not a default — when the arm can't be established", () => {
    expect(armTypeFromRobotType(null)).toBeNull();
    expect(armTypeFromRobotType(undefined)).toBeNull();
    expect(armTypeFromRobotType("")).toBeNull();
    expect(armTypeFromRobotType("aloha")).toBeNull();
  });
});
