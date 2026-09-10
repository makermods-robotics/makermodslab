import { describe, it, expect, vi } from "vitest";
import type { Fetcher } from "@/lib/apiClient";
import {
  cancelDatasetMerge,
  cleanupTemporaryMerges,
  startDatasetMerge,
} from "@/lib/replayApi";

/** Capture the request body apiRequest hands the fetcher. */
function captureFetcher() {
  const calls: { url: string; body: unknown }[] = [];
  const fetcher: Fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({
      url,
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    });
    return new Response(JSON.stringify({ started: true, message: "" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  return { fetcher, calls };
}

/** Fetcher that answers with a fixed merge-cleanup result body. */
function cleanupFetcher() {
  const calls: { url: string; method?: string; body: unknown }[] = [];
  const result = {
    deleted: ["ns/tmp_a"],
    skipped: [{ repo_id: "ns/tmp_b", reason: "in use" }],
    hub_deleted: [],
    hub_failed: [],
  };
  const fetcher: Fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({
      url,
      method: init?.method,
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    });
    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  return { fetcher, calls, result };
}

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

describe("cleanupTemporaryMerges", () => {
  it("POSTs the given repo ids to the cleanup route", async () => {
    const { fetcher, calls } = cleanupFetcher();

    await cleanupTemporaryMerges("http://x", fetcher, ["ns/tmp_a", "ns/tmp_b"]);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://x/api/v1/datasets/merge/cleanup");
    expect(calls[0].method).toBe("POST");
    expect(calls[0].body).toEqual({ repo_ids: ["ns/tmp_a", "ns/tmp_b"] });
  });

  it("sends an empty body to clean every temporary merge", async () => {
    const { fetcher, calls } = cleanupFetcher();

    await cleanupTemporaryMerges("http://x", fetcher);

    expect(calls[0].body).toEqual({});
  });

  it("returns the parsed cleanup result", async () => {
    const { fetcher, result } = cleanupFetcher();

    const res = await cleanupTemporaryMerges("http://x", fetcher);

    expect(res).toEqual(result);
  });
});

describe("startDatasetMerge", () => {
  it("sends temporary:true with a blank output name for a combine-and-train merge", async () => {
    const { fetcher, calls } = captureFetcher();

    await startDatasetMerge(
      "http://x",
      fetcher,
      ["ns/a", "ns/b"],
      "",
      [1, 3],
      [],
      true /* acknowledgeWarnings */,
      true /* temporary */,
    );

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://x/api/v1/datasets/merge");
    expect(calls[0].body).toEqual({
      source_repo_ids: ["ns/a", "ns/b"],
      output_repo_id: "",
      source_weights: [1, 3],
      drop_features: [],
      acknowledge_warnings: true,
      temporary: true,
    });
  });

  it("omits temporary (and weights) for an ordinary dialog merge — unchanged body", async () => {
    const { fetcher, calls } = captureFetcher();

    await startDatasetMerge("http://x", fetcher, ["ns/a", "ns/b"], "ns/merged");

    expect(calls[0].body).toEqual({
      source_repo_ids: ["ns/a", "ns/b"],
      output_repo_id: "ns/merged",
      drop_features: [],
    });
  });
});

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
      {
        url: "http://bench.local:8000/api/v1/datasets/merge/cancel",
        method: "POST",
      },
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
