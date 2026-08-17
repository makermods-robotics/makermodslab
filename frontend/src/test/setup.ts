import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; Task 3's useSpotlightTarget needs one.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  // @ts-expect-error jsdom has no ResizeObserver
  globalThis.ResizeObserver = ResizeObserverStub;
}
