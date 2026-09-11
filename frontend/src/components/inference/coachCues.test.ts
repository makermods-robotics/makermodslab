import { describe, expect, it } from "vitest";

import { phaseCue } from "./coachCues";

describe("a discard is not a hand-back", () => {
  // THE regression. Both transitions are correcting -> paused, so the handback
  // tone (falling 660->440, "the policy has it back, all is well") fired first
  // and the discard thud followed. The destructive event's audio opened with
  // the reassuring one.
  it("stays silent on the phase change when a discard landed with it", () => {
    expect(
      phaseCue({ previous: "correcting", next: "paused", discardPending: true }),
    ).toBeNull();
  });

  it("still cues a hand-back when no discard landed", () => {
    expect(phaseCue({ previous: "correcting", next: "paused", discardPending: false })).toBe(
      "handback",
    );
  });
});

describe("taking control is always announced", () => {
  it.each([
    ["from autonomous", "autonomous"],
    ["from paused", "paused"],
    ["from a reset", "resetting"],
    ["with no prior phase", null],
  ])("cues granted %s", (_label, previous) => {
    expect(phaseCue({ previous, next: "correcting", discardPending: false })).toBe("granted");
  });

  it("cues granted even if a discard notice is somehow pending", () => {
    // Entering a correction is never the discard's transition; suppressing it
    // here would silence the single most important cue in the set.
    expect(
      phaseCue({ previous: "paused", next: "correcting", discardPending: true }),
    ).toBe("granted");
  });
});

describe("everything else is silent", () => {
  it("says nothing when the phase has not changed", () => {
    expect(phaseCue({ previous: "correcting", next: "correcting", discardPending: false })).toBeNull();
    expect(phaseCue({ previous: "paused", next: "paused", discardPending: true })).toBeNull();
  });

  it.each([
    ["autonomous -> paused", "autonomous", "paused"],
    ["paused -> autonomous", "paused", "autonomous"],
    ["autonomous -> handing_over", "autonomous", "handing_over"],
    ["saving -> autonomous", "saving", "autonomous"],
    ["resetting -> paused", "resetting", "paused"],
  ])("says nothing for %s", (_label, previous, next) => {
    expect(phaseCue({ previous, next, discardPending: false })).toBeNull();
  });

  it("does not cue a hand-back when the phase goes null (session ending)", () => {
    expect(phaseCue({ previous: "correcting", next: null, discardPending: false })).toBeNull();
  });

  it("does not cue on the first phase ever seen, unless it is a correction", () => {
    expect(phaseCue({ previous: null, next: "autonomous", discardPending: false })).toBeNull();
  });
});
