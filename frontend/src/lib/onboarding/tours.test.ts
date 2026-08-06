import { describe, expect, it } from "vitest";
import { launchpadTour } from "@/lib/onboarding/tours";

describe("launchpadTour", () => {
  it("has at least one step", () => {
    expect(launchpadTour.steps.length).toBeGreaterThan(0);
  });

  it("every step has a well-formed [data-tour=...] selector, title, and description", () => {
    for (const step of launchpadTour.steps) {
      expect(step.target).toMatch(/^\[data-tour=[\w-]+\]$/);
      expect(() => document.querySelector(step.target)).not.toThrow();
      expect(step.title.length).toBeGreaterThan(0);
      expect(step.description.length).toBeGreaterThan(0);
    }
  });

  it("has no duplicate target selectors", () => {
    const targets = launchpadTour.steps.map((s) => s.target);
    expect(new Set(targets).size).toBe(targets.length);
  });
});
