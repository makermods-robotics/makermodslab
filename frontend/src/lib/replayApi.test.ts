import { describe, it, expect, vi } from "vitest";
import type { Fetcher } from "@/lib/apiClient";
import { startDatasetMerge } from "@/lib/replayApi";

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
