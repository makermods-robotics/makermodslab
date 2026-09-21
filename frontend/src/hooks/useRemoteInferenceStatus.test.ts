import { describe, expect, it } from "vitest";

import { perSecondRate } from "./useRemoteInferenceStatus";

/**
 * `holds` is the one DRTC counter whose cumulative value actively misleads: a
 * healthy run climbs it during warm-up and then FREEZES it, so the number on
 * screen (41 at t=1s) looks like a problem forever afterwards while the thing
 * that matters — is it still growing? — is invisible. The panel therefore
 * renders the derivative, and this pins it.
 */
describe("perSecondRate", () => {
  it("has no rate to report from a single sample", () => {
    expect(perSecondRate({ t: 1, value: 41 }, null)).toBeNull();
  });

  it("reports zero for a frozen counter — the healthy state", () => {
    expect(perSecondRate({ t: 5, value: 41 }, { t: 4, value: 41 })).toBe(0);
  });

  it("differences over the elapsed seconds, not per sample", () => {
    // A dropped sample must not double the apparent rate.
    expect(perSecondRate({ t: 6, value: 61 }, { t: 4, value: 41 })).toBe(10);
  });

  it("refuses a non-advancing clock rather than dividing by zero", () => {
    expect(perSecondRate({ t: 4, value: 50 }, { t: 4, value: 41 })).toBeNull();
    expect(perSecondRate({ t: 3, value: 50 }, { t: 4, value: 41 })).toBeNull();
  });
});
