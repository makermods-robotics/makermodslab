import { describe, it, expect, vi } from "vitest";

import { runTemporaryMerge, MergeAborted } from "./combineMerge";
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

  it("throws MergeAborted when the signal is already aborted", async () => {
    const startMerge = vi.fn<() => Promise<MergeStartResult>>(async () => ({
      started: true,
      message: "",
    }));
    const ac = new AbortController();
    ac.abort();

    await expect(
      runTemporaryMerge({
        startMerge,
        getStatus: async () => status({}),
        sleep: noSleep,
        signal: ac.signal,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
    // Aborted before it even asked the backend to start.
    expect(startMerge).not.toHaveBeenCalled();
  });

  it("stops polling once the signal aborts mid-merge", async () => {
    const ac = new AbortController();
    let n = 0;
    const getStatus = vi.fn<() => Promise<MergeStatus>>(async () => {
      n += 1;
      if (n === 2) ac.abort();
      return status({ state: "running" });
    });

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus,
        sleep: noSleep,
        signal: ac.signal,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
    expect(getStatus.mock.calls.length).toBe(2);
  });

  it("does not return a completed merge's id once aborted on the same tick", async () => {
    const ac = new AbortController();
    const getStatus = vi.fn<() => Promise<MergeStatus>>(async () => {
      ac.abort();
      return status({ state: "done", output_repo_id: "ns/mix-race" });
    });

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus,
        sleep: noSleep,
        signal: ac.signal,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
  });

  it("asks the backend to cancel the merge when aborted mid-merge", async () => {
    const cancelMerge = vi.fn(async () => ({ cancelled: true, message: "" }));
    const ac = new AbortController();
    let n = 0;
    const getStatus = vi.fn<() => Promise<MergeStatus>>(async () => {
      n += 1;
      if (n === 2) ac.abort();
      return status({ state: "running" });
    });

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus,
        sleep: noSleep,
        signal: ac.signal,
        cancelMerge,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
    expect(cancelMerge).toHaveBeenCalledOnce();
  });

  it("asks the backend to cancel when aborted on the same tick as done", async () => {
    const cancelMerge = vi.fn(async () => ({ cancelled: true, message: "" }));
    const ac = new AbortController();
    const getStatus = vi.fn<() => Promise<MergeStatus>>(async () => {
      ac.abort();
      return status({ state: "done", output_repo_id: "ns/mix-race" });
    });

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus,
        sleep: noSleep,
        signal: ac.signal,
        cancelMerge,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
    expect(cancelMerge).toHaveBeenCalledOnce();
  });

  it("does not cancel a merge that never started (aborted up front)", async () => {
    const cancelMerge = vi.fn(async () => ({ cancelled: false, message: "" }));
    const startMerge = vi.fn<() => Promise<MergeStartResult>>(async () => ({
      started: true,
      message: "",
    }));
    const ac = new AbortController();
    ac.abort();

    await expect(
      runTemporaryMerge({
        startMerge,
        getStatus: async () => status({}),
        sleep: noSleep,
        signal: ac.signal,
        cancelMerge,
      }),
    ).rejects.toBeInstanceOf(MergeAborted);
    expect(startMerge).not.toHaveBeenCalled();
    expect(cancelMerge).not.toHaveBeenCalled();
  });

  it("does not cancel the backend on normal completion or error", async () => {
    const cancelMerge = vi.fn();

    const out = await runTemporaryMerge({
      startMerge: async () => ({ started: true, message: "" }),
      getStatus: async () =>
        status({ state: "done", output_repo_id: "ns/mix-ok" }),
      sleep: noSleep,
      cancelMerge,
    });
    expect(out).toBe("ns/mix-ok");

    await expect(
      runTemporaryMerge({
        startMerge: async () => ({ started: true, message: "" }),
        getStatus: async () => status({ state: "error", error: "boom" }),
        sleep: noSleep,
        cancelMerge,
      }),
    ).rejects.toThrow(/boom/);

    expect(cancelMerge).not.toHaveBeenCalled();
  });
});
