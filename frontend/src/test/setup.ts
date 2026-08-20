import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; Task 3's useSpotlightTarget needs one.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub;
}
