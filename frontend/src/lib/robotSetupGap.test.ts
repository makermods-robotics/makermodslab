import { describe, expect, it } from "vitest";
// Imported from lib/ rather than hooks/useRobots: the hook module touches
// localStorage at import time, which some Node versions break on.
import {
  robotLayoutReady,
  robotSetupGap,
  setupScopeForArms,
  type RobotArmFields,
} from "@/lib/robotSetupGap";

/**
 * Locks the ENGLISH output of robotSetupGap byte-for-byte.
 *
 * The function was restructured so a translated variant could render the same
 * diagnosis (it used to build one English sentence from fragments, which is
 * untranslatable). That restructure must not change a single character of what
 * an English user sees — these assertions are copied from the original
 * implementation's output.
 */
const base: RobotArmFields & { name: string } = {
  name: "arm-1",
  mode: "single",
  leader_port: "/dev/tty.leader",
  follower_port: "/dev/tty.follower",
  leader_config: "teleop",
  follower_config: "robot",
  right_leader_port: "",
  right_follower_port: "",
  right_leader_config: "",
  right_follower_config: "",
};

const bimanual: RobotArmFields & { name: string } = {
  ...base,
  mode: "bimanual",
  right_leader_port: "/dev/tty.rleader",
  right_follower_port: "/dev/tty.rfollower",
  right_leader_config: "teleop-r",
  right_follower_config: "robot-r",
};

describe("robotSetupGap (English output is frozen)", () => {
  it("single arm, one missing calibration", () => {
    expect(robotSetupGap({ ...base, leader_config: "" })).toBe(
      "is missing a calibration for the leader arm",
    );
  });

  it("single arm, both calibrations missing — plural + ' and ' join", () => {
    expect(
      robotSetupGap({ ...base, leader_config: "", follower_config: "" }),
    ).toBe("is missing a calibration for the leader and follower arms");
  });

  it("single arm, one missing port", () => {
    expect(robotSetupGap({ ...base, follower_port: "" })).toBe(
      "has no port assigned for the follower arm",
    );
  });

  it("combines both clauses with ' and '", () => {
    expect(
      robotSetupGap({ ...base, leader_config: "", follower_port: "" }),
    ).toBe(
      "is missing a calibration for the leader arm and has no port assigned for the follower arm",
    );
  });

  it("falls back to the stale-file message when nothing is empty", () => {
    expect(robotSetupGap(base)).toBe(
      "references a calibration file that no longer exists — reassign or recalibrate",
    );
  });

  it("bimanual arm labels", () => {
    expect(
      robotSetupGap({ ...bimanual, right_follower_config: "" }),
    ).toBe("is missing a calibration for the right follower arm");
  });

  it("bimanual, two arms missing calibration", () => {
    expect(
      robotSetupGap({ ...bimanual, leader_config: "", right_leader_config: "" }),
    ).toBe("is missing a calibration for the left leader and right leader arms");
  });

  it("scope 'follower' ignores leader-side gaps", () => {
    expect(
      robotSetupGap({ ...base, leader_config: "", follower_port: "" }, "follower"),
    ).toBe("has no port assigned for the follower arm");
  });

  it("scope 'follower' on a bimanual robot lists both follower arms", () => {
    expect(
      robotSetupGap(
        { ...bimanual, follower_config: "", right_follower_config: "" },
        "follower",
      ),
    ).toBe(
      "is missing a calibration for the left follower and right follower arms",
    );
  });

  it("scope 'leader' ignores follower-side gaps", () => {
    expect(
      robotSetupGap({ ...base, leader_config: "", follower_port: "" }, "leader"),
    ).toBe("is missing a calibration for the leader arm");
  });

  it("scope 'leader' with a clean leader falls back to the stale message", () => {
    // A follower gap is invisible in this scope, so the leader side reads as
    // fully populated — the same fallback scope 'all' gives a clean record.
    expect(robotSetupGap({ ...base, follower_port: "" }, "leader")).toBe(
      "references a calibration file that no longer exists — reassign or recalibrate",
    );
  });

  it("scope 'leader' on a bimanual robot lists both leader arms", () => {
    expect(
      robotSetupGap(
        { ...bimanual, leader_port: "", right_leader_port: "" },
        "leader",
      ),
    ).toBe("has no port assigned for the left leader and right leader arms");
  });
});

describe("setupScopeForArms", () => {
  it("maps each layout to the arm(s) it owns", () => {
    expect(setupScopeForArms("both")).toBe("all");
    expect(setupScopeForArms("follower")).toBe("follower");
    expect(setupScopeForArms("leader")).toBe("leader");
  });

  it("treats a record without a layout as a pair", () => {
    expect(setupScopeForArms(undefined)).toBe("all");
  });
});

describe("robotLayoutReady", () => {
  const flags = { is_clean: false, follower_ready: true, leader_ready: false };

  it("a station is ready once its follower side is", () => {
    expect(robotLayoutReady({ ...flags, arms: "follower" })).toBe(true);
  });

  it("a controller is ready once its leader side is", () => {
    expect(robotLayoutReady({ ...flags, arms: "leader" })).toBe(false);
    expect(
      robotLayoutReady({ ...flags, arms: "leader", leader_ready: true }),
    ).toBe(true);
  });

  it("a pair (or a record without a layout) needs every arm", () => {
    expect(robotLayoutReady({ ...flags, arms: "both" })).toBe(false);
    expect(robotLayoutReady(flags)).toBe(false);
    expect(robotLayoutReady({ ...flags, is_clean: true })).toBe(true);
  });
});
