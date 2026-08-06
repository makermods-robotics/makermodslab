import { useEffect } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { OnboardingProvider, useOnboarding } from "@/contexts/OnboardingContext";
import type { Tour } from "@/lib/onboarding/types";
import Spotlight from "./Spotlight";

describe("Spotlight", () => {
  it("renders nothing when no tour is active", () => {
    const { container } = render(
      <OnboardingProvider>
        <Spotlight />
      </OnboardingProvider>,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the first step immediately when its target is measurable on mount (regression: step 1 must not be auto-skipped)", () => {
    // Regression test for a bug where the FIRST step of every tour was
    // silently auto-skipped (or, for a one-step tour, the tour never
    // rendered at all) even when its target was perfectly visible and
    // measurable. Root cause: useSpotlightTarget's measurement effect ran as
    // a passive useEffect, so on first mount its setRect(measured) call only
    // scheduled a re-render — it didn't retroactively fix the `rect` value
    // that Spotlight's own auto-skip passive effect had already closed over
    // in the same commit. That auto-skip effect ran right after, still
    // seeing the stale initial `rect = null`, and called advance()
    // unconditionally.
    //
    // The fix makes useSpotlightTarget's measurement effect a
    // useLayoutEffect: its setRect call now forces a synchronous re-render
    // (and re-commit of layout effects) before any passive effect — like
    // Spotlight's auto-skip — runs for the first time, so the auto-skip
    // effect only ever observes the real, current measurement.
    const targetA = document.createElement("div");
    targetA.setAttribute("data-tour", "step-a");
    targetA.getBoundingClientRect = () =>
      ({
        top: 10,
        left: 10,
        width: 100,
        height: 40,
        right: 110,
        bottom: 50,
        x: 10,
        y: 10,
        toJSON() {
          return this;
        },
      }) as DOMRect;
    document.body.appendChild(targetA);

    const targetB = document.createElement("div");
    targetB.setAttribute("data-tour", "step-b");
    targetB.getBoundingClientRect = () =>
      ({
        top: 200,
        left: 10,
        width: 100,
        height: 40,
        right: 110,
        bottom: 240,
        x: 10,
        y: 200,
        toJSON() {
          return this;
        },
      }) as DOMRect;
    document.body.appendChild(targetB);

    const tour: Tour = {
      id: "regression-tour",
      steps: [
        { target: "[data-tour=step-a]", title: "Step A", description: "desc a" },
        { target: "[data-tour=step-b]", title: "Step B", description: "desc b" },
      ],
    };

    function TourStarter() {
      const { start } = useOnboarding();
      useEffect(() => {
        start(tour, () => {});
        // Start exactly once, on mount — mirrors how a real page would kick
        // off a tour after its own data/target is ready.
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return null;
    }

    try {
      render(
        <OnboardingProvider>
          <TourStarter />
          <Spotlight />
        </OnboardingProvider>,
      );

      expect(screen.getByText("Step 1 of 2")).toBeInTheDocument();
      expect(screen.getByText("Step A")).toBeInTheDocument();
    } finally {
      document.body.removeChild(targetA);
      document.body.removeChild(targetB);
    }
  });
});
