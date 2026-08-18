import "@testing-library/jest-dom/vitest";
// Boots i18next so components calling t() render real English copy in tests
// (assertions stay written in English regardless of the browser locale).
import "@/i18n";

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
