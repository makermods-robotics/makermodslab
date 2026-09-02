import { describe, expect, it } from "vitest";
import { isCanArmType, jointsPerArm } from "./armTypes";

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
