import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useHostingStatus } from "./useHostingStatus";
import { getHostingStatus } from "@/lib/remoteApi";

const fetchWithHeaders = vi.fn();
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "", fetchWithHeaders }),
}));
vi.mock("@/hooks/useActiveSession", () => ({ useSessionEvent: () => null }));
vi.mock("@/lib/remoteApi", () => ({ getHostingStatus: vi.fn() }));

const idleStatus = {
  hosting_active: false,
  hosting: null,
  releasing: false,
  last_cleanup_error: null,
  outcome: null,
  error: null,
  hint: null,
  message: "Hosting status retrieved successfully",
};

describe("hosting status freshness", () => {
  beforeEach(() => {
    vi.mocked(getHostingStatus).mockReset();
  });

  it("marks retained status as unverified after a failed check and clears the error on recovery", async () => {
    vi.mocked(getHostingStatus).mockResolvedValue(idleStatus);
    const { result } = renderHook(() => useHostingStatus({ intervalMs: 60_000 }));
    await waitFor(() => expect(result.current.status).toEqual(idleStatus));
    expect(result.current.isError).toBe(false);

    vi.mocked(getHostingStatus).mockRejectedValueOnce(new Error("Offline"));
    await act(() => result.current.refresh());
    expect(result.current.status).toEqual(idleStatus);
    expect(result.current.isError).toBe(true);

    await act(() => result.current.refresh());
    expect(result.current.isError).toBe(false);
  });

  it("reports an unavailable initial check without inventing a hosting status", async () => {
    vi.mocked(getHostingStatus).mockRejectedValue(new Error("Offline"));
    const { result } = renderHook(() => useHostingStatus());
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.status).toBeNull();
  });
});
