import { describe, expect, it, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { OnboardingProvider, useOnboarding } from "./OnboardingContext";
import type { Tour } from "@/lib/onboarding/types";

const tour: Tour = {
  id: "test-tour",
  steps: [
    { target: "[data-tour=a]", title: "A", description: "a" },
    { target: "[data-tour=b]", title: "B", description: "b" },
  ],
};

function setup() {
  return renderHook(() => useOnboarding(), {
    wrapper: ({ children }) => (
      <OnboardingProvider>{children}</OnboardingProvider>
    ),
  });
}

describe("OnboardingContext", () => {
  it("starts a tour at step 0", () => {
    const { result } = setup();
    const onDone = vi.fn();
    act(() => result.current.start(tour, onDone));
    expect(result.current.activeTour?.id).toBe("test-tour");
    expect(result.current.stepIndex).toBe(0);
    expect(onDone).not.toHaveBeenCalled();
  });

  it("advance() moves to the next step", () => {
    const { result } = setup();
    act(() => result.current.start(tour, vi.fn()));
    act(() => result.current.advance());
    expect(result.current.stepIndex).toBe(1);
  });

  it("advance() past the last step calls onDone and clears the tour", () => {
    const { result } = setup();
    const onDone = vi.fn();
    act(() => result.current.start(tour, onDone));
    act(() => result.current.advance()); // -> step 1
    act(() => result.current.advance()); // -> past last step
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(result.current.activeTour).toBeNull();
  });

  it("back() does not go below step 0", () => {
    const { result } = setup();
    act(() => result.current.start(tour, vi.fn()));
    act(() => result.current.back());
    expect(result.current.stepIndex).toBe(0);
  });

  it("skip() ends the tour immediately and calls onDone once", () => {
    const { result } = setup();
    const onDone = vi.fn();
    act(() => result.current.start(tour, onDone));
    act(() => result.current.advance());
    act(() => result.current.skip());
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(result.current.activeTour).toBeNull();
    expect(result.current.stepIndex).toBe(0);
  });

  it("useOnboarding throws outside a provider", () => {
    expect(() => renderHook(() => useOnboarding())).toThrow(
      /useOnboarding must be used within OnboardingProvider/,
    );
  });

  it("skip() called twice only invokes onDone once", () => {
    const { result } = setup();
    const onDone = vi.fn();
    act(() => result.current.start(tour, onDone));
    act(() => result.current.skip());
    act(() => result.current.skip());
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(result.current.activeTour).toBeNull();
  });

  it("starting a new tour while one is active calls the outgoing tour's onDone", () => {
    const { result } = setup();
    const firstDone = vi.fn();
    const secondDone = vi.fn();
    const secondTour: Tour = {
      id: "second-tour",
      steps: [{ target: "[data-tour=c]", title: "C", description: "c" }],
    };

    act(() => result.current.start(tour, firstDone));
    act(() => result.current.advance()); // mid-flight on the first tour
    act(() => result.current.start(secondTour, secondDone));

    expect(firstDone).toHaveBeenCalledTimes(1);
    expect(secondDone).not.toHaveBeenCalled();
    expect(result.current.activeTour?.id).toBe("second-tour");
    expect(result.current.stepIndex).toBe(0);

    act(() => result.current.skip());
    expect(secondDone).toHaveBeenCalledTimes(1);
    expect(firstDone).toHaveBeenCalledTimes(1);
  });
});
