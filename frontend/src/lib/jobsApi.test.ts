import { describe, expect, it } from "vitest";
import { JobState, isTerminalJobState } from "./jobsApi";

describe("isTerminalJobState", () => {
  it("marks the three states a run can never leave", () => {
    for (const state of ["done", "failed", "interrupted"] as JobState[]) {
      expect(isTerminalJobState(state)).toBe(true);
    }
  });

  it("keeps queued and running non-terminal — they take Stop/Cancel, not Delete", () => {
    for (const state of ["queued", "running"] as JobState[]) {
      expect(isTerminalJobState(state)).toBe(false);
    }
  });
});
