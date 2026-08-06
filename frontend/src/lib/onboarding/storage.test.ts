import { describe, expect, it, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useOnceFlag } from "./storage";

describe("useOnceFlag", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("reads false when the key has never been set", () => {
    const { result } = renderHook(() => useOnceFlag("makerlab:test-flag"));
    expect(result.current.seen).toBe(false);
  });

  it("reads true when the key is already '1'", () => {
    localStorage.setItem("makerlab:test-flag", "1");
    const { result } = renderHook(() => useOnceFlag("makerlab:test-flag"));
    expect(result.current.seen).toBe(true);
  });

  it("markSeen persists the flag and flips seen to true", () => {
    const { result } = renderHook(() => useOnceFlag("makerlab:test-flag"));
    act(() => result.current.markSeen());
    expect(result.current.seen).toBe(true);
    expect(localStorage.getItem("makerlab:test-flag")).toBe("1");
  });

  it("does not throw when localStorage.getItem throws", () => {
    const spy = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });
    const { result } = renderHook(() => useOnceFlag("makerlab:test-flag"));
    expect(result.current.seen).toBe(false);
    spy.mockRestore();
  });

  it("does not throw when localStorage.setItem throws", () => {
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });
    const { result } = renderHook(() => useOnceFlag("makerlab:test-flag"));
    act(() => result.current.markSeen());
    expect(result.current.seen).toBe(true);
    spy.mockRestore();
  });
});