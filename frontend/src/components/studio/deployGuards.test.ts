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
  durationValid: true,
  inferenceActive: false,
  leaderMissing: false,
  requiresTask: false,
  task: "pick up the red block",
  transportReady: true,
  armSupportsRemote: true,
  remoteEngineSupported: true,
};

const MODES: DeployRunMode[] = ["single", "eval", "coach", "remote"];

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
    // Remote inference least of all: the Star Arm 102 leader holds encoders
    // and no motors, and the remote run never touches a leader anyway.
    expect(deployBlockedReason("remote", { ...ok, leaderMissing: true })).toBeNull();
  });
});

describe("the shared max-duration field means different things per mode", () => {
  // One field, two contracts: 0 is "run until I stop you" for a remote run and
  // a run that ends the instant it starts for a local one. The panel cannot
  // express that with a min= on the input, because the same input serves both.
  it.each(["single", "eval", "coach"] as DeployRunMode[])(
    "blocks %s on a duration this mode cannot use",
    (mode) => {
      expect(deployBlockedReason(mode, { ...ok, durationValid: false })).toMatch(
        /blocked\.durationRequired/,
      );
    },
  );

  it("lets a remote run keep an unbounded duration", () => {
    expect(deployBlockedReason("remote", { ...ok, durationValid: false })).toBeNull();
  });
});

describe("remote inference has three guards of its own", () => {
  it("blocks remote when the transport is not ready", () => {
    // Also the pre-probe state. Launching into an unverified transport
    // energizes the arm for a run that nothing may ever drive.
    expect(deployBlockedReason("remote", { ...ok, transportReady: false })).toMatch(
      /blocked\.transportNotReady/,
    );
  });

  it("blocks remote on an arm family the ease-in does not support", () => {
    expect(deployBlockedReason("remote", { ...ok, armSupportsRemote: false })).toMatch(
      /blocked\.remoteArmUnsupported/,
    );
  });

  it("reports the unsupported arm before the transport", () => {
    // The arm is a fact about the robot that no transport fix changes; naming
    // the transport first sends the operator to the wrong problem.
    expect(
      deployBlockedReason("remote", {
        ...ok,
        armSupportsRemote: false,
        transportReady: false,
      }),
    ).toMatch(/blocked\.remoteArmUnsupported/);
  });

  it("blocks remote when the engine does not suit the checkpoint", () => {
    // The engine guard has NO backend twin and cannot have one: the server
    // never loads the checkpoint, so it cannot tell a flow policy from an ACT
    // one and accepts whichever engine it is handed. This is the only gate.
    expect(
      deployBlockedReason("remote", { ...ok, remoteEngineSupported: false }),
    ).toMatch(/blocked\.remoteEngineUnsupported/);
  });

  it("reports the unsuitable engine before the transport", () => {
    // A fact about the CHECKPOINT, with a one-click remedy (switch back to
    // Adaptive sync) that "the transport isn't ready" would send them past.
    expect(
      deployBlockedReason("remote", {
        ...ok,
        remoteEngineSupported: false,
        transportReady: false,
      }),
    ).toMatch(/blocked\.remoteEngineUnsupported/);
  });

  it("still reports the unsupported arm before the engine", () => {
    expect(
      deployBlockedReason("remote", {
        ...ok,
        armSupportsRemote: false,
        remoteEngineSupported: false,
      }),
    ).toMatch(/blocked\.remoteArmUnsupported/);
  });

  it("imposes none of the three on the local modes", () => {
    for (const mode of ["single", "eval", "coach"] as DeployRunMode[]) {
      expect(
        deployBlockedReason(mode, {
          ...ok,
          transportReady: false,
          armSupportsRemote: false,
          remoteEngineSupported: false,
        }),
      ).toBeNull();
    }
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
