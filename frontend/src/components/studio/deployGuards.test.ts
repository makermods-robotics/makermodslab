import { describe, expect, it } from "vitest";

import { deployBlockedReason } from "./deployGuards";
import type { DeployGuardContext, DeployRunMode } from "./deployGuards";

const ok: DeployGuardContext = {
  hasRobot: true,
  followerReady: true,
  hasCheckpoint: true,
  armMismatch: false,
  allCamerasBound: true,
  temporalEnsembleInvalid: false,
  inferenceActive: false,
  leaderMissing: false,
  requiresTask: false,
  task: "pick up the red block",
};

const MODES: DeployRunMode[] = ["single", "eval", "coach"];

describe("a language-conditioned policy cannot launch without its task", () => {
  // The check used to live only under `mode === "coach"`, so a plain run or a
  // scored eval could start with the field blank. It does not fail loudly — it
  // just steers the policy with an empty string, which looks like the policy
  // being bad, and any success rate measured that way is measuring the wrong
  // thing.
  it.each(MODES)("blocks %s when the task is empty", (mode) => {
    const reason = deployBlockedReason(mode, { ...ok, requiresTask: true, task: "" });
    expect(reason).toMatch(/blocked\.taskRequired/);
  });

  it.each(MODES)("blocks %s when the task is only whitespace", (mode) => {
    expect(deployBlockedReason(mode, { ...ok, requiresTask: true, task: "   " })).not.toBeNull();
  });

  it.each(MODES)("allows %s once a task is typed", (mode) => {
    expect(deployBlockedReason(mode, { ...ok, requiresTask: true })).toBeNull();
  });

  it("does not demand a task from a policy that reads none", () => {
    expect(deployBlockedReason("single", { ...ok, requiresTask: false, task: "" })).toBeNull();
    expect(deployBlockedReason("eval", { ...ok, requiresTask: false, task: "" })).toBeNull();
  });
});

describe("coaching keeps its own task requirement", () => {
  it("still blocks coach with an empty task even when the policy reads none", () => {
    const reason = deployBlockedReason("coach", { ...ok, requiresTask: false, task: "" });
    expect(reason).toMatch(/blocked\.coachTaskRequired/);
  });

  it("blocks coach without a leader arm", () => {
    expect(deployBlockedReason("coach", { ...ok, leaderMissing: true })).toMatch(/blocked\.leaderMissing/);
  });

  it("does not demand a leader arm for the other modes", () => {
    expect(deployBlockedReason("single", { ...ok, leaderMissing: true })).toBeNull();
    expect(deployBlockedReason("eval", { ...ok, leaderMissing: true })).toBeNull();
  });
});

describe("preflight order puts the most fundamental gap first", () => {
  it.each([
    ["no robot", { hasRobot: false }, /blocked\.noRobot/],
    ["follower not ready", { followerReady: false }, /blocked\.followerNotReady/],
    ["no checkpoint", { hasCheckpoint: false }, /blocked\.noCheckpoint/],
    ["arm mismatch", { armMismatch: true }, /blocked\.armMismatch/],
    ["cameras unbound", { allCamerasBound: false }, /blocked\.camerasUnbound/],
    ["bad ensemble", { temporalEnsembleInvalid: true }, /blocked\.temporalEnsemble/],
    ["already running", { inferenceActive: true }, /blocked\.runInProgress/],
  ])("reports %s", (_label, patch, expected) => {
    expect(deployBlockedReason("coach", { ...ok, ...patch })).toMatch(expected);
  });

  it("reports the missing robot before anything else it could also complain about", () => {
    const reason = deployBlockedReason("coach", {
      ...ok,
      hasRobot: false,
      followerReady: false,
      hasCheckpoint: false,
      leaderMissing: true,
      task: "",
    });
    expect(reason).toMatch(/blocked\.noRobot/);
  });
});

describe("a fully configured panel launches every mode", () => {
  it.each(MODES)("permits %s", (mode) => {
    expect(deployBlockedReason(mode, ok)).toBeNull();
  });
});

describe("several tasks are refused rather than guessed between", () => {
  // A merged dataset carries every task string of its inputs, and nothing
  // distinguishes them well enough to pick one: measured on real merged
  // datasets, two near-identical sentences sat 99 vs 100 episodes apart. The
  // panel used to send whichever had one more episode, and on a coaching run
  // that string is then stamped into every recorded frame.
  it("blocks every mode for a language-conditioned policy", () => {
    for (const mode of MODES) {
      const reason = deployBlockedReason(mode, {
        ...ok,
        requiresTask: true,
        task: "",
        taskAmbiguous: true,
      });
      expect(reason).toMatch(/blocked\.taskAmbiguous/);
    }
  });

  it("blocks coaching even for a policy that never reads the task", () => {
    const reason = deployBlockedReason("coach", {
      ...ok,
      requiresTask: false,
      task: "",
      taskAmbiguous: true,
    });
    expect(reason).toMatch(/blocked\.taskAmbiguous/);
  });

  it("does not block a plain run whose policy ignores the task", () => {
    // The field is not even on screen there, so there is nothing to resolve and
    // refusing would be a dead end.
    for (const mode of ["single", "eval"] as DeployRunMode[]) {
      expect(
        deployBlockedReason(mode, {
          ...ok,
          requiresTask: false,
          task: "",
          taskAmbiguous: true,
        }),
      ).toBeNull();
    }
  });

  it("clears once the operator picks one", () => {
    for (const mode of MODES) {
      expect(
        deployBlockedReason(mode, {
          ...ok,
          requiresTask: true,
          task: "the chosen one",
          taskAmbiguous: false,
        }),
      ).toBeNull();
    }
  });

  it("is absent by default, so existing callers are unaffected", () => {
    expect(deployBlockedReason("coach", { ...ok })).toBeNull();
  });
});
