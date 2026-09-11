import "@testing-library/jest-dom/vitest";
// Boots i18next so components calling t() render real English copy in tests
// (assertions stay written in English regardless of the browser locale).
import "@/i18n";

// jsdom 30 under Node 26 gives us a `window` with NO `localStorage` — Node's
// own experimental implementation is inert without --localstorage-file, and
// jsdom no longer fills the gap. Without this, `localStorage` is `undefined`
// rather than empty, so any module that touches storage at IMPORT time (e.g.
// src/lib/mockHub.ts) throws while a test is merely importing the component
// under test, and the failure looks like a component bug rather than a
// missing browser API.
const memoryStorage = (): Storage => {
  let store: Record<string, string> = {};
  return {
    get length() {
      return Object.keys(store).length;
    },
    key: (i: number) => Object.keys(store)[i] ?? null,
    getItem: (k: string) => (k in store ? store[k] : null),
    setItem: (k: string, v: string) => {
      store[k] = String(v);
    },
    removeItem: (k: string) => {
      delete store[k];
    },
    clear: () => {
      store = {};
    },
  } as Storage;
};

for (const name of ["localStorage", "sessionStorage"] as const) {
  if (typeof globalThis !== "undefined" && !(globalThis as Record<string, unknown>)[name]) {
    Object.defineProperty(globalThis, name, {
      value: memoryStorage(),
      configurable: true,
      writable: true,
    });
  }
}

// jsdom has no ResizeObserver; Task 3's useSpotlightTarget needs one.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub;
}
