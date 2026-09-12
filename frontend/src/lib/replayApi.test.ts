import { describe, expect, it } from "vitest";

import { cancelDatasetMerge } from "./replayApi";
import type { Fetcher } from "./apiClient";

/** A Fetcher that records the (url, method) it was called with and answers
 * with `body`. */
function fetcherReturning(
  status: number,
  body: unknown,
): { fetcher: Fetcher; calls: { url: string; method: string }[] } {
  const calls: { url: string; method: string }[] = [];
  const fetcher: Fetcher = async (url, init) => {
    calls.push({ url, method: (init?.method ?? "GET").toUpperCase() });
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
  return { fetcher, calls };
}

describe("cancelDatasetMerge", () => {
  it("POSTs the v1 cancel route and returns the parsed body", async () => {
    const { fetcher, calls } = fetcherReturning(200, {
      cancelled: true,
      message: "Merge cancelled.",
    });

    await expect(
      cancelDatasetMerge("http://bench.local:8000", fetcher),
    ).resolves.toEqual({ cancelled: true, message: "Merge cancelled." });
    expect(calls).toEqual([
      { url: "http://bench.local:8000/api/v1/datasets/merge/cancel", method: "POST" },
    ]);
  });

  it("passes through the no-merge-running answer", async () => {
    const { fetcher } = fetcherReturning(200, {
      cancelled: false,
      message: "No merge is running.",
    });

    await expect(
      cancelDatasetMerge("http://localhost:8000", fetcher),
    ).resolves.toEqual({ cancelled: false, message: "No merge is running." });
  });
});
