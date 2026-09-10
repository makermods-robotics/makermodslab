import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import type { SessionChangedEvent } from "@/hooks/useActiveSession";

// Hoisted so the (hoisted) vi.mock factories can close over them.
// `api` is a stable reference on purpose — useDatasets' `refresh` is a
// useCallback keyed on it, and a fresh object per render would re-fire its
// mount effect forever.
const h = vi.hoisted(() => ({
  api: { baseUrl: "http://test", fetchWithHeaders: vi.fn() },
  listDatasets: vi.fn(async () => []),
  state: { sessionEvent: null as SessionChangedEvent | null },
}));
const { listDatasets } = h;

vi.mock("@/contexts/ApiContext", () => ({ useApi: () => h.api }));
vi.mock("@/lib/replayApi", () => ({ listDatasets: h.listDatasets }));
vi.mock("@/hooks/useActiveSession", () => ({
  useSessionEvent: () => h.state.sessionEvent,
}));

import { useDatasets } from "./useDatasets";

const evt = (over: Partial<SessionChangedEvent>): SessionChangedEvent => ({
  kind: "recording",
  active: false,
  phase: null,
  receivedAt: Date.now(),
  ...over,
});

beforeEach(() => {
  vi.clearAllMocks();
  h.state.sessionEvent = null;
});

describe("useDatasets session-event refetch", () => {
  it("refetches when a recording session ends", async () => {
    const { rerender } = renderHook(() => useDatasets());
    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(1));

    h.state.sessionEvent = evt({ kind: "recording", active: false });
    rerender();

    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(2));
  });

  it("refetches when an inference (coaching) session ends", async () => {
    const { rerender } = renderHook(() => useDatasets());
    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(1));

    h.state.sessionEvent = evt({ kind: "inference", active: false });
    rerender();

    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(2));
  });

  it("ignores the event already present at mount", async () => {
    h.state.sessionEvent = evt({ kind: "recording", active: false, receivedAt: 1000 });
    const { rerender } = renderHook(() => useDatasets());
    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(1));

    rerender();
    await new Promise((r) => setTimeout(r, 20));
    expect(listDatasets).toHaveBeenCalledTimes(1);
  });

  it("ignores an active:true (session started) transition", async () => {
    const { rerender } = renderHook(() => useDatasets());
    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(1));

    h.state.sessionEvent = evt({ kind: "recording", active: true });
    rerender();

    await new Promise((r) => setTimeout(r, 20));
    expect(listDatasets).toHaveBeenCalledTimes(1);
  });

  it("ignores session kinds that never touch datasets", async () => {
    const { rerender } = renderHook(() => useDatasets());
    await waitFor(() => expect(listDatasets).toHaveBeenCalledTimes(1));

    h.state.sessionEvent = evt({ kind: "teleoperation", active: false });
    rerender();

    await new Promise((r) => setTimeout(r, 20));
    expect(listDatasets).toHaveBeenCalledTimes(1);
  });
});
