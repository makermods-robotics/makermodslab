import { describe, expect, it, vi, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useSpotlightTarget } from "./useSpotlightTarget";

afterEach(() => {
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

function mockRect(el: Element, rect: Partial<DOMRect>) {
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    top: 0,
    left: 0,
    width: 0,
    height: 0,
    right: 0,
    bottom: 0,
    x: 0,
    y: 0,
    toJSON() {
      return this;
    },
    ...rect,
  });
}

describe("useSpotlightTarget", () => {
  it("returns null when the selector matches nothing", () => {
    const { result } = renderHook(() =>
      useSpotlightTarget("[data-tour=missing]"),
    );
    expect(result.current).toBeNull();
  });

  it("returns null when the matched element has zero size", () => {
    const el = document.createElement("div");
    el.setAttribute("data-tour", "zero");
    document.body.appendChild(el);
    mockRect(el, { width: 0, height: 0 });
    const { result } = renderHook(() => useSpotlightTarget("[data-tour=zero]"));
    expect(result.current).toBeNull();
  });

  it("returns the measured rect when the element has real size", () => {
    const el = document.createElement("div");
    el.setAttribute("data-tour", "real");
    document.body.appendChild(el);
    mockRect(el, { top: 10, left: 20, width: 100, height: 40 });
    const { result } = renderHook(() => useSpotlightTarget("[data-tour=real]"));
    expect(result.current).toEqual({ top: 10, left: 20, width: 100, height: 40 });
  });
});
