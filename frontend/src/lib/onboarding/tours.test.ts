import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import { launchpadTour, studioTour } from "@/lib/onboarding/tours";
import { resources } from "@/i18n";

const ALL_TOURS = [launchpadTour, studioTour];
const LANGUAGES = Object.keys(resources) as (keyof typeof resources)[];

describe.each(ALL_TOURS)("$id tour", (tour) => {
  it("has at least one step", () => {
    expect(tour.steps.length).toBeGreaterThan(0);
  });

  it("every step has a well-formed [data-tour=...] selector", () => {
    for (const step of tour.steps) {
      expect(step.target).toMatch(/^\[data-tour=[\w-]+\]$/);
      expect(() => document.querySelector(step.target)).not.toThrow();
    }
  });

  // Replaces the old `step.title.length > 0` check: now that steps hold keys,
  // the useful invariant is that each key actually resolves in every shipped
  // language, rather than rendering as the raw key.
  it.each(LANGUAGES)("resolves every step's copy in %s", (lng) => {
    const fixed = i18n.getFixedT(lng);
    for (const step of tour.steps) {
      for (const key of [step.titleKey, step.descriptionKey]) {
        const value = fixed(key);
        expect(value).toBeTruthy();
        expect(value).not.toBe(key);
      }
    }
  });

  it("has no duplicate target selectors", () => {
    const targets = tour.steps.map((s) => s.target);
    expect(new Set(targets).size).toBe(targets.length);
  });
});
