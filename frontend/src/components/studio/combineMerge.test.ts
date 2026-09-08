import { describe, it, expect, vi } from "vitest";

import { runTemporaryMerge } from "./combineMerge";
import type { MergeStatus, MergeStartResult } from "@/lib/replayApi";

const noSleep = () => Promise.resolve();

const status = (over: Partial<MergeStatus>): MergeStatus => ({
  state: "running",
  error: null,
  output_repo_id: null,
  logs: [],
  ...over,
});

describe("runTemporaryMerge", () => {
  it("polls until done and returns the minted output repo id", async () => {
    const startMerge = vi.fn<() => Promise<MergeStartResult>>(async () => ({
      started: true,
      message: "",
    }));
    const getStatus = vi
      .fn<() => Promise<MergeStatus>>()
      .mockResolvedValueOnce(status({ state: "running" }))
      .mockResolvedValueOnce(
        status({ state: "done", output_repo_id: "ns/mix-ab12" }),
      );

    const seen: MergeStatus[] = [];
    const out = await runTemporaryMerge({
      startMerge,
      getStatus,
      onStatus: (s) => seen.push(s),
      sleep: noSleep,
    });

    expect(out).toBe("ns/mix-ab12");
    expect(startMerge).toHaveBeenCalledOnce();
    expect(seen.at(-1)?.state).toBe("done");
  });

  it("throws when the merge is refused up front", async () => {
    await expect(
      runTemporaryMerge({
        startMerge: async () => ({
          started: false,
          message: "sources differ by a column",
        }),
        getStatus: async () => status({}),
        sleep: noSleep,
      }),
    ).rejects.toThrow(/differ by a column/);
  });

  it("throws when the merge job ends in error", async () => {
    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus: async () => status({ state: "error", error: "disk full" }),
        sleep: noSleep,
      }),
    ).rejects.toThrow(/disk full/);
  });

  it("gives up when the status endpoint stays unreachable", async () => {
    const getStatus = vi.fn<() => Promise<MergeStatus>>(async () => {
      throw new Error("still down");
    });

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus,
        sleep: noSleep,
      }),
    ).rejects.toThrow(/Lost contact/);
    // Bounded, not infinite.
    expect(getStatus.mock.calls.length).toBeLessThanOrEqual(20);
  });

  it("rides out a transient status failure and keeps polling", async () => {
    const getStatus = vi
      .fn<() => Promise<MergeStatus>>()
      .mockRejectedValueOnce(new Error("network blip"))
      .mockResolvedValueOnce(
        status({ state: "done", output_repo_id: "ns/mix-cd34" }),
      );

    const out = await runTemporaryMerge({
      startMerge: async () => ({ started: true, message: "" }),
      getStatus,
      sleep: noSleep,
    });

    expect(out).toBe("ns/mix-cd34");
  });
});
