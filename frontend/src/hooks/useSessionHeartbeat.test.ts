import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HEARTBEAT_INTERVAL_MS, useSessionHeartbeat } from "./useSessionHeartbeat";
import { ApiError } from "@/lib/apiClient";
import { heartbeatSession } from "@/lib/sessionApi";

vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "", fetchWithHeaders: vi.fn() }),
}));
vi.mock("@/lib/sessionApi", () => ({ heartbeatSession: vi.fn() }));

const tick = () => vi.advanceTimersByTimeAsync(HEARTBEAT_INTERVAL_MS);

describe("useSessionHeartbeat", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(heartbeatSession).mockReset();
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("keeps beating through transient failures", async () => {
    vi.mocked(heartbeatSession).mockRejectedValue(new Error("Offline"));
    renderHook(() => useSessionHeartbeat("s1", "tab", true));
    await tick();
    await tick();
    await tick();
    expect(heartbeatSession).toHaveBeenCalledTimes(3);
  });

  it.each(["session.not_found", "session.lease_expired", "session.not_owner"])(
    "stops beating once the server answers %s",
    async (code) => {
      vi.mocked(heartbeatSession).mockRejectedValue(
        new ApiError("Heartbeat session failed", code === "session.not_found" ? 404 : 409, null, code)
      );
      renderHook(() => useSessionHeartbeat("s1", "tab", true));
      await tick();
      await tick();
      await tick();
      expect(heartbeatSession).toHaveBeenCalledTimes(1);
    }
  );
});
